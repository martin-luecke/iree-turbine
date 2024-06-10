from ..ops.base import (
    define_op,
)

__all__ = [
    "construct_register_from_metadata",
    "read",
    "write",
    "mma",
    "reduction",
]


@define_op
def construct_register_from_metadata(shape, dtype, value) -> None:
    ...


@define_op
def read(memory: "Memory", elements_pre_thread) -> "Register":
    ...


@define_op
def write(register: "Register", memory: "Memory", elements_pre_thread) -> None:
    ...


@define_op
def mma(lhs: "Register", rhs: "Register", acc: "Register") -> "Register":
    ...


@define_op
def reduction(axis: "IndexExpr", init_args):
    ...
