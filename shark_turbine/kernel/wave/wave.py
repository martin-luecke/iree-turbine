from typing import Callable, Optional
import inspect

from shark_turbine.kernel.wave.constraints import (
    Constraint,
    WorkgroupConstraint,
    get_grid_shape,
)

from ..compiler.ir import Context, Operation
from ..lang import Grid
from .._support.tracing import (
    CapturedTrace,
    CompiledContext,
    KernelRegionGraph,
    Launchable,
)
from .._support.nodes import *


__all__ = ["wave"]


def wave(constraints: list[Constraint]):
    def decorator(f: Callable[[Any], Any]) -> "LaunchableWave":
        return LaunchableWave(constraints, f.__name__, f)

    return decorator


class LaunchableWave(Launchable):
    def __init__(
        self,
        constraints: list[Constraint],
        name: str,
        eager_function: Callable[[Any], Any],
    ):
        super().__init__(eager_function)

        self.constraints = constraints
        self._name = name
        self._f = eager_function
        self._sig = inspect.signature(eager_function)

        self.grid_type = Grid[tuple(get_grid_shape(self.workgroup_constraints))]

    @property
    def workgroup_constraints(self) -> list[WorkgroupConstraint]:
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, WorkgroupConstraint)
        ]

    def _trace(self, dump: bool = False) -> CapturedTrace:
        region_graph = KernelRegionGraph()
        with CompiledContext(region_graph, grid_type=self.grid_type) as context:
            custom_ops: dict[str, Type[CustomNode]] = {
                "construct_register_from_metadata": ConstructRegisterFromMetadataNode,
                "mma": MmaNode,
                "read": ReadNode,
                "write": WriteNode,
                "reduction": ReductionNode,
                "placeholder": PlaceholderNode,
            }

            # Register custom ops
            for name, op in custom_ops.items():
                context.register_custom_op(name, op)

            with region_graph.subtracer() as subtracer:
                root_name, _ = subtracer.trace(self._f)
                trace = CapturedTrace(region_graph, root_name)

            if dump:
                print(trace.get_root_graph())
                for node in trace.get_root_graph().nodes:
                    print(context.node(node))
        return trace

    def _trace_and_get_kernel_signature(
        self,
        args,
        kwargs,
        dump: bool = False,
        context: Optional[Context] = None,
        module_op: Optional[Operation] = None,
    ) -> CapturedTrace:
        # Trace the function.
        trace = self._trace(dump=dump)

        # TODO: Get kernel signature from the trace.
        #       We want to reuse the existing kernel_codegen for this which
        #       requires making it aware of tkw.Memory
        return trace

    def test_execute(self, args, kwargs):
        # For now only tracing
        self._trace_and_get_kernel_signature(args, kwargs, dump=True)

    def aot_execute(self, args, kwargs):
        raise NotImplementedError("AOT execution for wave not implemented yet.")

    def eager_execute(self, args, kwargs):
        raise NotImplementedError("Eager execution for wave not implemented yet.")

    def __repr__(self):
        return f"tk.wave @{self._name}[{self.grid_type}]"
