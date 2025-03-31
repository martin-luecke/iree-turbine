# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import defaultdict
from dataclasses import dataclass
from ..constraints import Constraint
from ..._support.tracing import CapturedTrace
from ...ops.wave_ops import Reduction, IterArg, get_custom, CustomOp, Read, Write
from typing import Optional


@dataclass
class BufferExpansion:
    """Represents how a buffer is expanded for multi-buffering"""

    original_memory: CustomOp  # The original memory node
    num_partitions: int  # Total number of partitions (2 for double buffering)


@dataclass
class MultibufferGroup:
    """Represents a group of operations that share a multi-buffered memory"""

    buffer_expansion: BufferExpansion
    reads: list[CustomOp]
    writes: list[CustomOp]


from ..._support.indexing import xor, IndexSymbol
from .modulo_scheduling import ModuloScheduler
from .graph_utils import create_scheduling_edges, Edge
from .resources import get_available_resources, annotate_resource_usage
from ..visualization import visualize_edges, visualize_graph, visualize_schedule
from .loop_reconstruction import construct_pipelined_loop
from ..utils import (
    graph_copy,
    erase_graph,
    get_tiling_constraint,
    subs_idxc,
    get_assumptions,
    evaluate_with_assumptions,
)
import torch.fx as fx
from ....support.logging import get_logger
import iree.turbine.kernel.lang as tkl
from ...lang.global_symbols import SHARED_ADDRESS_SPACE

logger = get_logger("turbine.wave.scheduling.schedule")

ARGK = tkl.IndexSymbol("$ARGK", integer=True, nonnegative=True)


def visualize_scheduling_graph(edges: list[Edge]):
    visualize_edges(edges, "reduction_graph.png")


def prepare_multibuffering(
    graph: fx.Graph,
    reduction_axis: IndexSymbol,
    num_partitions: int = 2,
) -> dict[CustomOp, MultibufferGroup]:
    """
    Returns dict mapping memory nodes to their MultibufferGroup, which contains:
    - buffer_expansion: How this memory is expanded for multi-buffering
    - reads: Read operations from this memory
    - writes: Write operations to this memory
    """
    buffer_groups: dict[CustomOp, MultibufferGroup] = {}

    # Group reads and writes by memory location
    for node in graph.nodes:
        custom_op = get_custom(node)
        memory = get_custom(custom_op.memory) if hasattr(custom_op, "memory") else None

        if not memory:
            continue

        if memory not in buffer_groups:
            buffer_groups[memory] = MultibufferGroup(
                buffer_expansion=BufferExpansion(
                    original_memory=memory,
                    num_partitions=num_partitions,
                ),
                reads=[],
                writes=[],
            )

        if isinstance(custom_op, Read):
            if (
                reduction_axis in custom_op.indexing_dims
                and custom_op.memory_type.address_space == SHARED_ADDRESS_SPACE
            ):
                buffer_groups[memory].reads.append(custom_op)
        elif isinstance(custom_op, Write):
            if (
                reduction_axis in custom_op.indexing_dims
                and custom_op.memory_type.address_space == SHARED_ADDRESS_SPACE
            ):
                buffer_groups[memory].writes.append(custom_op)

    # Filter and expand buffers
    result = {}
    for memory_node, group in buffer_groups.items():
        if analyze_buffer_access_pattern(group.reads, group.writes):

            # Update memory size for actual code generation
            reduction_dim_index = memory_node.shape.index(reduction_axis)
            new_shape = tuple(
                dim * num_partitions if i != reduction_dim_index else dim
                for i, dim in enumerate(memory_node.shape)
            )
            new_distributed_shape = tuple(
                dim * num_partitions if i != reduction_dim_index else dim
                for i, dim in enumerate(memory_node.distributed_shape)
            )
            memory_node.update_arg(0, new_shape)
            memory_node.update_arg(1, new_distributed_shape)

            result[memory_node] = group

    return result


def analyze_buffer_dependencies(
    edges: list[Edge], multibuffer_groups: dict[CustomOp, MultibufferGroup]
) -> dict[CustomOp, str]:
    """
    Analyzes scheduling edges to determine ping/pong assignments.
    Returns dict mapping operations to 'ping' or 'pong'.
    """
    assignments = {}

    # Build dependency chains for each memory
    for memory_node, group in multibuffer_groups.items():
        # Find all write->read chains for this memory
        write_to_reads = defaultdict(set)
        read_to_writes = defaultdict(set)

        for edge in edges:
            from_op = get_custom(edge._from)
            to_op = get_custom(edge._to)

            # Track write->read dependencies
            if isinstance(from_op, Write) and isinstance(to_op, Read):
                if to_op in group.reads:
                    write_to_reads[from_op].add(to_op)

            # Track read->write dependencies
            if isinstance(from_op, Read) and isinstance(to_op, Write):
                if from_op in group.reads:
                    read_to_writes[from_op].add(to_op)

        # Group operations that must be in different phases
        # Operations connected by edges must be in opposite phases
        current_phase = "ping"
        assigned = set()

        def assign_phase(op: CustomOp, phase: str):
            if op not in assigned:
                assignments[op] = phase
                assigned.add(op)
                # Assign opposite phase to dependent ops
                next_phase = "pong" if phase == "ping" else "ping"
                for read in write_to_reads.get(op, set()):
                    assign_phase(read, next_phase)
                for write in read_to_writes.get(op, set()):
                    assign_phase(write, next_phase)

        # Start with writes as anchors
        for write in group.writes:
            assign_phase(write, current_phase)
            current_phase = "pong" if current_phase == "ping" else "ping"

    return assignments


def analyze_buffer_access_pattern(
    read_nodes: list[CustomOp], write_nodes: list[CustomOp]
) -> bool:
    """
    Analyzes if a buffer's access pattern is suitable for double buffering.
    Returns True if double buffering can be applied.
    """
    # Must have both reads and writes
    if not read_nodes or not write_nodes:
        return False

    # For now, accept all patterns that have both reads and writes
    # TODO: Add more sophisticated pattern analysis
    return True


def adjust_multibuffer_indexing(
    custom: CustomOp,
    buffer_assignments: dict[CustomOp, str],
    multibuffered_ops: dict[CustomOp, MultibufferGroup],
    reduction_axis: IndexSymbol,
) -> None:
    """
    Adjusts the indexing for multibuffered operations based on their ping/pong assignments.

    Args:
        custom: The operation to adjust indexing for
        buffer_assignments: Dict mapping operations to their ping/pong assignment
        multibuffered_ops: Dict mapping memory nodes to their MultibufferGroup
        reduction_axis: The reduction axis being used
    """
    if custom in buffer_assignments:
        for memory_node, group in multibuffered_ops.items():
            if custom in (group.reads + group.writes):
                block_size = memory_node.distributed_shape[
                    1 - memory_node.shape.index(reduction_axis)
                ]

                # Use ping/pong assignment to determine offset
                if buffer_assignments[custom] == "ping":
                    offset = (ARGK % 2) * block_size
                else:  # pong
                    offset = xor((ARGK % 2) * block_size, block_size)

                original_dim = memory_node.shape[
                    1 - memory_node.shape.index(reduction_axis)
                ]
                custom.index[original_dim].start = (
                    custom.index[original_dim].start + offset
                )


def schedule_reduction(
    reduction: Reduction,
    trace: CapturedTrace,
    constraints: list[Constraint],
    use_scheduling_barriers: bool = False,
):
    """
    Clones the reduction graph and does the following:
    1. Analyzes and prepares buffers for multibuffering
    2. Annotates resource usage for each node
    3. Creates edges between outputs and return args for scheduling
       and assigns weights to all edges
    Does scheduling on the cloned graph and applies the schedule
    to the original graph. Finally, erases the cloned graph.
    """
    reduction_graph = trace.get_subgraph(reduction.subgraph_name)
    graph, node_map = graph_copy(reduction_graph)

    # Analyze and prepare buffers for multibuffering
    multibuffered_ops = prepare_multibuffering(graph, reduction.axis)

    ignore_nodes, iter_args, output = annotate_resource_usage(graph)
    edges = create_scheduling_edges(
        graph, ignore_nodes, iter_args, output, multibuffered_ops
    )

    # Analyze edges for buffer assignments before scheduling
    buffer_assignments = analyze_buffer_dependencies(edges, multibuffered_ops)

    visualize = True
    if visualize:
        visualize_scheduling_graph(edges)
        visualize_graph(graph, "scheduling_fx_graph.png")

    scheduler = ModuloScheduler(graph, edges, get_available_resources())
    schedule, success = scheduler.schedule_graph()
    if not success:
        raise ValueError("Scheduling failed.")
    if visualize:
        visualize_schedule(schedule, scheduler.initiation_interval, "schedule.html")

    # Apply schedule and handle multibuffering indexing
    inverse_node_map = {v: k for k, v in node_map.items()}
    iter_args: list[CustomOp] = []
    for node, cycle in schedule.items():
        if node not in inverse_node_map:
            continue

        custom = get_custom(inverse_node_map[node])
        stage = cycle // scheduler.initiation_interval

        # Update scheduling parameters
        custom.scheduling_parameters = {
            "absolute_cycle": cycle,
            "cycle": cycle % scheduler.initiation_interval,
            "stage": stage,
            "initiation_interval": scheduler.initiation_interval,
        }

        # Adjust indexing for multibuffered operations
        adjust_multibuffer_indexing(
            custom, buffer_assignments, multibuffered_ops, reduction.axis
        )

        # Erase edges between outputs and iter args
        if isinstance(get_custom(node), IterArg):
            node.args = ()
            iter_args.append(custom)

    for custom in iter_args:
        cycle = min([x.scheduling_parameters["absolute_cycle"] for x in custom.users])
        custom.scheduling_parameters = {
            "absolute_cycle": cycle,
            "cycle": cycle % scheduler.initiation_interval,
            "stage": cycle // scheduler.initiation_interval,
            "initiation_interval": scheduler.initiation_interval,
        }

    erase_graph(graph)

    # After scheduling has completed, we have enough information to decide
    # whether to pipeline the loop. For pipelining to be possible, we need
    # to have atleast N iterations of the loop where N > num_stages - 1 (because
    # we will be peeling off num_stages iterations from the loop).
    tiling_constraint = get_tiling_constraint(reduction, constraints)
    max_induction_variable = subs_idxc(tiling_constraint.count)

    if max_induction_variable.is_number:
        # We can only do a compile-time check if the induction variable
        # is not dynamic.
        max_induction_variable = int(max_induction_variable)
        if max_induction_variable <= scheduler.num_stages - 1:
            logger.warning(
                "Not enough iterations to pipeline the loop. Skipping pipelining."
            )
            return {}
    else:
        # Otherwise, we need to rely on assumptions provided by the author.
        assumptions = get_assumptions(constraints)
        if not assumptions:
            logger.warning(
                "No assumptions provided to determine if the loop can be pipelined. Skipping pipelining."
            )
            return {}

        result = evaluate_with_assumptions(
            constraints, max_induction_variable > scheduler.num_stages - 1
        )
        if not result:
            logger.warning(
                "Not enough iterations to pipeline the loop. Skipping pipelining."
            )
            return {}

    new_reduction = construct_pipelined_loop(
        trace,
        reduction,
        reduction_graph,
        constraints,
        scheduler,
        node_map,
        max_induction_variable,
        visualize,
        use_scheduling_barriers,
    )

    # Update new reduction count.
    new_reduction.count = max_induction_variable - (scheduler.num_stages - 1)


def schedule_graph(
    trace: CapturedTrace,
    constraints: list[Constraint],
    use_scheduling_barriers: bool = False,
):
    """
    Given a graph, pipelines the reductions in the graph.
    """

    def is_reduction(node: fx.Node) -> bool:
        return isinstance(get_custom(node), Reduction)

    reduction_nodes = trace.walk(is_reduction)
    if not reduction_nodes:
        return

    for reduction_node in reduction_nodes:
        schedule_reduction(
            get_custom(reduction_node), trace, constraints, use_scheduling_barriers
        )
