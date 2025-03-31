from __future__ import annotations
from ..compiler.ir import (
    builtin_d,
    InsertionPoint,
    Location,
    Operation,
    transform_d,
    UnitAttr,
    Value,
)
from typing import Optional, Callable, Any, List, Tuple, Sequence
from .._support.tracing import CapturedTrace
from .._support.indexing import (
    IndexExpr,
    IndexingContext,
    IndexSymbol,
    IndexSequence,
    xor,
)
from ..lang.global_symbols import *
from ..ops.wave_ops import (
    get_custom,
    Output,
    Read,
    Write,
    MMA,
    CustomOp,
    Reduction,
    GetResult,
    ExtractSlice,
    IterArg,
    Reshape,
)
from ..lang.wave_types import IndexMapping
from .constraints import (
    Constraint,
    WorkgroupConstraint,
    HardwareConstraint,
    TilingConstraint,
    MMAType,
    MMAOperand,
)
from .assumptions import Assumption
from .utils import print_trace
import torch.fx as fx
import iree.turbine.kernel.lang as tkl
from pathlib import Path


import tempfile
from ...support.conversions import TORCH_DTYPE_TO_SIGNED_MLIR_TYPE_ASM
from iree.compiler.dialects.transform import (
    interpreter as transform_interpreter,
    any_op_t,
)
from iree.compiler.dialects import (
    _structured_transform_ops_gen as structured_transform_ops,
)

import sympy
import torch
from iree.compiler import compile_str
import iree.runtime as rt
import iree.runtime.benchmark as bench

import numpy
import ml_dtypes

K = tkl.sym.K
M = tkl.sym.M
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_K = tkl.sym.BLOCK_K
ARGK = tkl.IndexSymbol("$ARGK", integer=True, nonnegative=True)


def partition_reads_by_memory(reads: list[CustomOp]) -> dict[CustomOp, list[CustomOp]]:
    """
    Partitions reads by their source memory location.
    Returns a dict mapping memory nodes to lists of read operations from that memory.
    """
    memory_to_reads: dict[CustomOp, list[CustomOp]] = {}

    for read_node in reads:
        memory_node = get_custom(read_node.memory)

        if memory_node not in memory_to_reads:
            memory_to_reads[memory_node] = []

        memory_to_reads[memory_node].append(read_node)

    return memory_to_reads


def analyze_buffer_access_pattern(
    read_nodes: list[CustomOp], write_nodes: list[CustomOp]
) -> bool:
    """
    Analyzes if a buffer's access pattern is suitable for double buffering.
    Returns True if double buffering can be applied.
    """
    # Check if we have both reads and writes
    if not read_nodes or not write_nodes:
        return False

    # Verify the access pattern:
    # - Writes should happen before reads in each iteration
    # - There should be a clear producer-consumer relationship
    # TODO: Implement pattern analysis

    return True


def multi_buffering(trace: CapturedTrace):
    # Find all reductions
    reductions = trace.walk(lambda node: isinstance(get_custom(node), Reduction))

    # Get reduction dimension from first reduction
    # (Assuming all reductions use same dimension)
    if not reductions:
        return
    reduction_axis = get_custom(reductions[0]).axis

    # Find reads that index using the reduction dim and are from shared memory

    # reads = [
    #     get_custom(node)
    #     for node in trace.walk(
    #         lambda node: isinstance((get_custom(node)), Read)
    #         and reduction_axis in node.tkw_op.indexing_dims
    #         and get_custom(node).memory_type.address_space == SHARED_ADDRESS_SPACE
    #     )
    # ]
    reads = []
    for node in trace.get_subgraph(get_custom(reductions[0]).subgraph_name).nodes:
        if (
            isinstance((get_custom(node)), Read)
            and reduction_axis in get_custom(node).indexing_dims
            and get_custom(node).memory_type.address_space == SHARED_ADDRESS_SPACE
        ):
            reads.append(get_custom(node))

    # writes = [
    #     get_custom(node)
    #     for node in trace.walk(
    #         lambda node: isinstance((get_custom(node)), Write)
    #         and get_custom(node).memory_type.address_space == SHARED_ADDRESS_SPACE
    #     )
    # ]
    writes = []
    for node in trace.get_subgraph(get_custom(reductions[0]).subgraph_name).nodes:
        if (
            isinstance((get_custom(node)), Write)
            and reduction_axis in get_custom(node).indexing_dims
            and get_custom(node).memory_type.address_space == SHARED_ADDRESS_SPACE
        ):
            writes.append(get_custom(node))

    # Partition reads and writes by memory location
    memory_to_reads = partition_reads_by_memory(reads)
    memory_to_writes = partition_reads_by_memory(writes)

    for memory_location in set(memory_to_reads.keys()) | set(memory_to_writes.keys()):
        read_nodes = memory_to_reads.get(memory_location, [])
        write_nodes = memory_to_writes.get(memory_location, [])

        if not analyze_buffer_access_pattern(read_nodes, write_nodes):
            continue

        implement_double_buffering(
            trace, memory_location, read_nodes, write_nodes, reduction_axis
        )


def implement_double_buffering(
    trace: CapturedTrace,
    original_buffer: CustomOp,
    read_nodes: list[Read],
    write_nodes: list[Write],
    axis: IndexSymbol,
):
    """
    Implements double buffering for a shared memory buffer.
    """
    # For now onlydouble buffering, so we are doubling the
    # size of the original buffer.

    assert len(original_buffer.shape) == 2

    # double the memory in the non-reduction dimension
    reduction_dim_index = original_buffer.shape.index(axis)
    original_dim = original_buffer.shape[1 - reduction_dim_index]

    block_size = original_buffer.distributed_shape[1 - reduction_dim_index]
    new_shape = tuple(
        dim * 2 if i != reduction_dim_index else dim
        for i, dim in enumerate(original_buffer.shape)
    )
    new_distributed_shape = tuple(
        dim * 2 if i != reduction_dim_index else dim
        for i, dim in enumerate(original_buffer.distributed_shape)
    )
    original_buffer.update_arg(0, new_shape)
    original_buffer.update_arg(1, new_distributed_shape)

    # 2. The indexing for the reads should depend on the loop index.
    # i.e. if even access lower part of the buffer, else the higher part.
    for read in read_nodes:
        pass

    # Get the block size (size in the non-reduction dimension)

    # For each read/write operation, modify its index to include buffer offset
    stage_mapping: dict[int, list[CustomOp]] = {}

    for node in read_nodes + write_nodes:
        custom_op = get_custom(node)
        current_index = custom_op.index

        # print(custom_op)
        # print(custom_op.fx_node.scheduling_parameters["cycle"])
        # Get cycle from scheduling parameters
        cycle = custom_op.fx_node.scheduling_parameters["cycle"]

        # Group nodes by their cycle
        if cycle not in stage_mapping:
            stage_mapping[cycle] = []
        stage_mapping[cycle].append(custom_op)

    # breakpoint()
    for stage in stage_mapping.keys():
        # if stage % 2 != 0 or stage == 2:
        print(f"modifying for {stage}")
        offset = 0
        for op in stage_mapping[stage]:
            buffer_offset = (ARGK % 2) * block_size
            # if stage % 2 == 0:
            # TODO: For now we use Piecewise, but actually we want xor
            if stage < 2:
                offset = buffer_offset
            elif stage >= 2:
                offset = xor(buffer_offset, block_size)
                # buffer_offset += block_size
                # op.index[original_dim].start = op.index[original_dim].start + 77
            # op.index[original_dim].start = op.index[original_dim].start + buffer_offset
            else:
                raise CodegenError(
                    f"Stage > 4 not supported in multibuffering currently"
                )

            op.index[original_dim].start = op.index[original_dim].start + offset
            print(f"from mb: {op.name}: {op.index[original_dim].start}")

            # apply_expr
        # custom_op.index[original_size].start + (axis * original_size)
        # & original_size

        # Create buffer offset based on iteration: (iteration * block_size) & block_size
        # This will alternate between 0 and block_size
        # buffer_offset = (axis * block_size) & block_size

        # TODO: axis is not correct here! it is substituted with the actual size and this this offset is always 0
        # buffer_offset = (ARGK % 2) * block_size
        # buffer_offset = (ARGK % 2) * block_size

        # TODO: it's weird to call this block_size here!
        # TODO: move this up
        # custom_op.index[original_dim].start = (
        #     custom_op.index[original_dim].start + buffer_offset
        # )

    # Note: This does not have the same indexing like later. The indexing is changed in wave_ops.py with align_indexing.
    # That should not be an issue as it only substitutes stuff. So let's still add this here.
    # Question: Do we have the induction var already available?

    # 3. Update memory accesses:
    # - Even iterations: write to buffer 0, read from buffer 1
    # - Odd iterations: write to buffer 1, read from buffer 0

    # 4. Add appropriate synchronization barriers

    # TODO: Implement the transformation
