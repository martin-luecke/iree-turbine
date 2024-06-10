from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional
import shark_turbine.kernel.lang as tkl
from sympy import Expr, Symbol
from .._support.indexing import IndexExpr, IndexSymbol


@dataclass
class Constraint(ABC):
    """
    Base class for constraints. Every constraint reduces to
    the following form:
        Variables: [x0, x1, ...., xN]
        Bounds: [lb0 <= x0 <= ub0, ..., lbN <= xN <= ubN]
        Equality Constraints: [f0(x0, ..., xN) = 0, f1(x0, ..., xN) = 0, ...]
        Inequality Constraints: [g0(x0, ..., xN) <= 0, g1(x0, ..., xN) <= 0, ...]
    """

    @abstractmethod
    def apply(self) -> Expr:
        pass


@dataclass
class WorkgroupConstraint(Constraint):
    """
    A constraint of the form `tkw.WorkgroupConstraint(M, BLOCK_M, 0)`
    specifies that we want to distribute dimension M along workgroup dim 0
    with a tile size of BLOCK_M resulting in M // BLOCK_M workgroups along that
    dimension. This translates to an index constraint for all tensors of the
    shape [M, ?] -> index += (workgroup_id_0 * BLOCK_M, 0)
    """

    dim: Symbol
    tile_size: Symbol
    workgroup_dim: int

    def apply(self) -> Expr:
        match self.workgroup_dim:
            case 0:
                wg_dim = tkl.sym.WG0
            case 1:
                wg_dim = tkl.sym.WG1
            case _:
                raise ValueError("Invalid workgroup index. Expected 0 or 1")
        return wg_dim * self.tile_size


def get_grid_shape(wg_constraints: list[WorkgroupConstraint]) -> list[Expr]:
    grid: list[Expr] = [Expr(1) for _ in range(len(wg_constraints))]
    for constraint in wg_constraints:
        grid[constraint.workgroup_dim] = constraint.dim // constraint.tile_size
    return grid
