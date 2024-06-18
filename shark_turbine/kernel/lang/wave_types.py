from typing import Optional, Type, TypeVar, cast, ClassVar
from enum import Enum

from .kernel_buffer import KernelBufferUsage
from ..ops.wave_ops import register
from .._support.shaped_type import ShapedDataType
from .._support.dtype import DataType
from .._support.indexing import IndexExpr

__all__ = [
    "Memory",
    "Register",
    "AddressSpace",
]

MemoryTypeT = TypeVar("MemoryTypeT")


class AddressSpace(Enum):
    REGISTER = 0
    SHARED_MEMORY = 1
    GLOBAL_MEMORY = 2


class _MemoryStorage(ShapedDataType):
    def new_subtype(
        cls: Type[MemoryTypeT],
        *,
        symbolic_shape: tuple[IndexExpr, ...],
        address_space: AddressSpace,
        dtype: DataType,
        usage: Optional[KernelBufferUsage] = None,
    ) -> Type[MemoryTypeT]:
        init_symbolic_shape = symbolic_shape
        init_dtype = dtype
        init_address_space = (
            address_space if address_space else AddressSpace.GLOBAL_MEMORY
        )
        init_usage = usage

        class MemoryType(cls):
            symbolic_shape = init_symbolic_shape
            rank = len(symbolic_shape)
            address_space = init_address_space
            dtype = init_dtype
            usage = init_usage

        return cast(Type[MemoryTypeT], MemoryType)


class Memory(metaclass=_MemoryStorage):
    """
    Represents storage anywhere in the memory hierarchy except registers.
    Parameterized by a shape, address space and element type. The allocated
    memory is traversed by an iterator that specifies the offset, stride
    and size along each dimension.
    """

    symbolic_shape: ClassVar[tuple[IndexExpr, ...]]
    address_space: ClassVar[int]
    rank: ClassVar[int]
    dtype: ClassVar[DataType]
    usage: ClassVar[Optional[KernelBufferUsage]]

    def __init__(self) -> None:
        raise NotImplementedError("Memory types are not directly instantiated.")

    def __class_getitem__(
        cls, shape_and_dtype: tuple[IndexExpr | DataType, ...]
    ) -> Type["Memory"]:
        """Syntax: `Memory[shape1, ...., shapeN, addressSpace, dtype, Optional[usage]]"""
        if len(shape_and_dtype) < 3:
            raise TypeError(f"Expected at least 3 arguments, got: {shape_and_dtype}")

        shift = 0
        usage = None
        if isinstance(shape_and_dtype[-1], KernelBufferUsage):
            shift = 1
            usage = shape_and_dtype[-1]
        shape = shape_and_dtype[: -2 - shift]
        addressSpace = shape_and_dtype[-2 - shift]
        dtype = shape_and_dtype[-1 - shift]

        # Allow constant int expressions in shape
        shape = tuple(IndexExpr(s) if isinstance(s, int) else s for s in shape)
        if not all(isinstance(s, IndexExpr) for s in shape) or len(shape) == 0:
            raise TypeError(f"Expected shape to be a tuple of IndexExpr, got {shape}")
        if not isinstance(dtype, DataType):
            raise TypeError(f"Expected dtype to be a DataType, got {dtype}")
        if not (
            isinstance(addressSpace, IndexExpr)
            or isinstance(addressSpace, AddressSpace)
        ):
            raise TypeError(
                f"Expected addressSpace to be a AddressSpace, got {addressSpace}"
            )
        if addressSpace == AddressSpace.REGISTER:
            raise TypeError(
                f"Memory does not support address space register, use Register instead."
            )

        return cls.new_subtype(
            symbolic_shape=shape, address_space=addressSpace, dtype=dtype, usage=usage
        )


class Register(metaclass=_MemoryStorage):
    """
    Represents virtual registers. Parameterized by a shape and element type.
    Instantiating this class emits a new `register` operation.
    """

    symbolic_shape: ClassVar[tuple[IndexExpr, ...]]
    rank: ClassVar[int]
    dtype: ClassVar[DataType]
    value: float

    def __new__(cls, value: float) -> "Register":
        return register(cls.symbolic_shape, cls.dtype, value)

    def __class_getitem__(
        cls, shape_and_dtype: tuple[IndexExpr | DataType, ...]
    ) -> Type["Register"]:

        if len(shape_and_dtype) < 2:
            raise TypeError(f"Expected at least 2 arguments, got: {shape_and_dtype}")

        shape = shape_and_dtype[:-1]
        dtype = shape_and_dtype[-1]

        # Allow constant int expressions in shape
        shape = tuple(IndexExpr(s) if isinstance(s, int) else s for s in shape)

        if not isinstance(dtype, DataType):
            raise TypeError(f"Expected dtype to be a DataType, got {dtype}")

        return cls.new_subtype(
            symbolic_shape=shape, dtype=dtype, address_space=AddressSpace.REGISTER.value
        )


def is_memory_meta_derived(t: type) -> bool:
    return isinstance(t, _MemoryStorage)
