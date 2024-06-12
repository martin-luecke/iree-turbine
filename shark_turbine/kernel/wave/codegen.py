from typing import Any, Callable, ClassVar, Type, Optional, Sequence, Union, List
from dataclasses import dataclass, field
import sympy
import torch.fx as fx
import torch.utils._pytree as pytree

from .._support.indexing import (
    IndexExpr,
    IndexingContext,
    IndexSymbol,
    SymIndex,
    index_expr,
)

# from .._support.nodes import write
from .._support.tracing import CapturedTrace
from ..ops.wave_ops import register, mma, read, reduction, write
from ..compiler.builder import (
    IRProxyValue,
    ScalarBuilder,
)

from ..compiler.base import (
    CodegenError,
    NDEBUG,
    ValidationError,
)

from ..compiler.ir import (
    AffineMap,
    Attribute,
    AffineExpr,
    AffineMapAttr,
    ArrayAttr,
    FunctionType,
    VectorType,
    DenseElementsAttr,
    F32Type,
    IndexType,
    FloatAttr,
    InsertionPoint,
    IntegerType,
    IntegerAttr,
    IrType,
    Location,
    MemRefType,
    OpResult,
    ShapedType,
    Value,
    VectorType,
    arith_d,
    func_d,
    math_d,
    memref_d,
    vector_d,
    scf_d,
    stream_d,
)


from ..compiler.kernel_codegen import (
    BoundKernelSignature,
)

from ..compiler.vector_codegen import (
    cast_py_literal,
    cast_py_value,
    cast_kernel_buffer,
    cast_slice_spec,
    cast_vector,
    extract_slice_starts,
)
import operator as py_operator


@dataclass
class NodeAttrs:
    # By default, integers are assumed signed. We propagate unsigned as graph
    # node attrs.
    unsigned: bool = False

    @staticmethod
    def load(py_value) -> "NodeAttrs":
        if isinstance(py_value, fx.Node):
            return NodeAttrs(unsigned=bool(py_value.meta.get("unsigned")))
        return NodeAttrs()

    def store(self, node: fx.Node):
        node.meta["unsigned"] = self.unsigned


@dataclass
class WaveEmitter:
    """Emits a warp function as a `func` with a signature derived from the gm."""

    root_sig: BoundKernelSignature
    trace: CapturedTrace
    ip: InsertionPoint = None
    OP_HANDLERS: ClassVar[dict[Any, Callable[["WaveEmitter", fx.Node], None]]] = {}

    def __post_init__(self):
        self.ip = InsertionPoint(self.root_sig.entry_block)

    def emit(self, graph: Optional[fx.Graph] = None):
        with self.ip, Location.unknown():
            self._emit_graph(
                graph if graph is not None else self.trace.get_root_graph()
            )

    def _emit_graph(self, graph: fx.Graph):
        """Emits the given graph at the current insertion point."""
        for node in graph.nodes:
            if node.op == "call_function" or node.op == "call_method":
                self._emit_function_call_node(node)

    def _emit_function_call_node(self, node: fx.Node):
        target_op = node.target
        try:
            handler = self.OP_HANDLERS[target_op]
        except KeyError:
            raise CodegenError(f"No handler registered for op {target_op}")
        try:
            handler(self, node)
        except NotImplementedError as e:
            print(f"Error emitting node {node}: {e}")


def handle_op(op):
    def decorator(
        f: Callable[[WaveEmitter, fx.Node], None]
    ) -> Callable[[WaveEmitter, fx.Node], None]:
        WaveEmitter.OP_HANDLERS[op] = f
        return f

    return decorator


###############################################################################
# Memory Ops
###############################################################################


@handle_op(register)
def handle_register(emitter: WaveEmitter, node: fx.Node):
    raise NotImplementedError("Currently only stub implementation")


@handle_op(read)
def handle_read(emitter: WaveEmitter, node: fx.Node):
    raise NotImplementedError("Currently only stub implementation")


@handle_op(write)
def handle_write(emitter: WaveEmitter, node: fx.Node):
    raise NotImplementedError("Currently only stub implementation")


###############################################################################
# Math Ops
###############################################################################


@handle_op(mma)
def handle_mma(emitter: WaveEmitter, node: fx.Node):
    raise NotImplementedError("Currently only stub implementation")


###############################################################################
# Control Flow ops
###############################################################################


@handle_op(reduction)
def handle_reduction(emitter: WaveEmitter, node: fx.Node):
    raise NotImplementedError("Currently only stub implementation")
