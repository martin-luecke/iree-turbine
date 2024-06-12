from typing import Any, Callable, Optional, Type
import inspect

from ..compiler import builder, dispatch_codegen, kernel_codegen
from ..compiler.ir import Context, Operation
from .codegen import WaveEmitter
from ..lang import Grid
from .._support.tracing import (
    CapturedTrace,
    CompiledContext,
    KernelRegionGraph,
    Launchable,
)

# from .._support.nodes import (
#     NewRegister,
#     CustomOp,
#     MMA,
#     Read,
#     Write,
#     Reduction,
#     Placeholder,
# )
from .._support.nodes import *

__all__ = ["wave"]


def wave():
    def decorator(f: Callable[[Any], Any]) -> "LaunchableWave":
        return LaunchableWave(f.__name__, f)

    return decorator


class LaunchableWave(Launchable):
    def __init__(
        self,
        name: str,
        eager_function: Callable[[Any], Any],
        debug: bool = True,
    ):
        super().__init__(eager_function)

        self.grid_type = Grid[None, None]
        self._name = name
        self._f = eager_function
        self._sig = inspect.signature(eager_function)
        self.debug = debug

    def _trace(self) -> CapturedTrace:
        region_graph = KernelRegionGraph()
        with CompiledContext(region_graph, grid_type=self.grid_type) as context:
            custom_ops: dict[str, Type[CustomOp]] = {
                "register": NewRegister,
                "mma": MMA,
                "read": Read,
                "write": Write,
                "reduction": Reduction,
                "placeholder": Placeholder,
            }

            # Register custom ops
            for name, op in custom_ops.items():
                context.register_custom_op(name, op)

            with region_graph.subtracer() as subtracer:
                root_name, _ = subtracer.trace(self._f)
                trace = CapturedTrace(region_graph, root_name)

            if self.debug:
                print(trace.get_root_graph())
                for node in trace.get_root_graph().nodes:
                    print(context.node(node))
        return trace

    def _trace_and_get_kernel_signature(
        self,
        args,
        kwargs,
        context: Optional[Context] = None,
        module_op: Optional[Operation] = None,
    ) -> CapturedTrace:
        # Trace the function.
        trace = self._trace()

        kernel_sig = kernel_codegen.KernelSignature()
        # Fixed for now, will be determined through constraints
        self.grid_type.dims = [32, 32]  # Will be determined by constraints
        grid = self.grid_type

        mb = builder.ModuleBuilder(context=context, module_op=module_op)
        entrypoint_name = self._name
        exe = dispatch_codegen.StreamExecutable(mb, name=entrypoint_name)
        dispatch_entrypoint = exe.define_entrypoint(entrypoint_name, kernel_sig, grid)

        emitter = WaveEmitter(dispatch_entrypoint, trace)
        emitter.emit(trace.get_root_graph())

        return trace

    def test_execute(self, args, kwargs):
        # For now only tracing
        self._trace_and_get_kernel_signature(args, kwargs)

    def aot_execute(self, args, kwargs):
        raise NotImplementedError("AOT execution for wave not implemented yet.")

    def eager_execute(self, args, kwargs):
        raise NotImplementedError("Eager execution for wave not implemented yet.")

    def __repr__(self):
        return f"tk.wave @{self._name}[{self.grid_type}]"
