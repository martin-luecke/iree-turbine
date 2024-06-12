from abc import ABC
from dataclasses import dataclass, field
from functools import wraps
import sys
from typing import Any, Optional, Sequence, Type, TypeVar, final
import torch.fx as fx

from ..lang.functional_types import Memory, Register
from .._support.indexing import IndexExpr
from .._support.dtype import DataType


def get_node_name(string: str, skip_first: bool = True):
    snakeString = ""
    if skip_first:
        snakeString += string[0].lower()
        string = string[1:]
    for i in string:
        if i.isupper():
            snakeString += "_" + i.lower()
        else:
            snakeString += i
    return snakeString


T = TypeVar("T")


def define_op(op_name: str):
    def decorator(cls):
        def new_function(*args: Any, **kwargs):
            # TODO: This function represents the new operation, so it needs
            # the proper fields from the dataclass as arguments.
            # For the actual body look at the `define_op` in base.py
            # TODO: Make sure this is picked up properly by the tracing
            print("This is the new function added by the decorator.")

        current_module = sys.modules[cls.__module__]
        setattr(current_module, op_name, new_function)

        original_init = cls.__init__

        @wraps(original_init)
        def new_init(self, *args, **kwargs):
            # Call the original __init__ method
            original_init(self, *args, **kwargs)

            # Mock only, I don't think we will have to modify the init.
            # This is for prototyping purposes.
            # Iterate through the fields and modify them
            for field_name, field_def in cls.__dataclass_fields__.items():
                value = getattr(self, field_name)
                if isinstance(
                    value, int
                ):  # Example modification: double integer values
                    setattr(self, field_name, value * 2)
            print(f"invoked new op:{op_name}")

        # cls.__init__ = new_init
        return cls

    return decorator


CustomNodeT = TypeVar("CustomNodeT", bound="CustomNode")
PlaceholderNodeT = TypeVar("PlaceholderNodeT", bound="PlaceholderNode")


@dataclass
class CustomOp(ABC):
    """
    Base class for all custom fx nodes.
    """

    graph: fx.Graph
    fx_op: Any
    fx_node: Optional[fx.Node] = field(default=False, init=False)

    @classmethod
    def from_fx_node(cls: Type[CustomNodeT], node: fx.Node) -> CustomNodeT:
        instance = cls(node.graph, node.op, *node.args)
        instance.fx_node = node
        return instance

    def __str__(self) -> str:
        name = get_node_name(self.__class__.__name__)
        # print all variables of the node apart from graph and op
        vars_list = [f"{key}={value}" for key, value in vars(self).items()][2:]
        vars_str = ", ".join(vars_list)
        return f"{name}({vars_str})"

    def custom_string(self, value_map: dict[str, str]) -> str:
        # If a subclass does not define custom printing we revert to the default
        return str(self)

    def add_to_graph(self, region_graph):
        arg_list = tuple([value for _, value in vars(self).items()][2:])
        self.fx_node = region_graph.create_proxy(
            "call_function",
            target=self.fx_op,
            args=arg_list,
            kwargs={},
        )

    @classmethod
    def handle(cls, graph, *args, **kwargs) -> fx.Node:
        node = cls(graph, *args, **kwargs)
        node.add_to_graph(graph)
        node.fx_node.node.tkw_op = cls
        return node.fx_node

    @property
    def name(self) -> str:
        if hasattr(self, "_name"):
            return self._name
        return self.fx_node.name


@final
@dataclass
class Unknown(CustomOp):
    """
    Represents an fx.Node that has no corresponding CustomNode class.
    """

    args: Sequence[Any]
    kwargs: dict[Any, Any]

    @classmethod
    def from_fx_node(cls, node: fx.Node) -> "Unknown":
        instance = cls(node.graph, node.op, node.args, node.kwargs)
        instance.fx_node = node
        return instance

    def __str__(self) -> str:
        # print all variables of the node apart from graph and op
        vars_list = [f"{key}={value}" for key, value in vars(self).items()][2:]
        vars_str = ", ".join(vars_list)
        return f"unkown: {self.fx_node.name}({vars_str})"


@dataclass
class Placeholder(CustomOp):
    """
    Represents a placeholder node in the graph, i.e. an input to a function.
    """

    _name: str
    type: Optional[DataType]

    @classmethod
    def from_fx_node(cls: Type[PlaceholderNodeT], node: fx.Node) -> PlaceholderNodeT:
        return cls(node.graph, node.op, node.name, node.type)


# Ops modeling TKW operations in the kernel language


@dataclass
class NewRegister(CustomOp):
    shape: tuple[IndexExpr, ...]
    dtype: DataType
    value: float


@dataclass
class MMA(CustomOp):
    lhs: fx.Node
    rhs: fx.Node
    acc: fx.Node


@dataclass
class Read(CustomOp):
    memory: fx.Proxy
    elements_per_thread: Optional[Any] = None
    type: Optional[Type[Register]] = None


@dataclass
class Reduction(CustomOp):
    axis: IndexExpr
    init_args: Sequence[Any]
    subgraph_name: str
    implicit_captures: Sequence[fx.Proxy]

    @classmethod
    def handle(cls, graph, *args, **kwargs):
        def wrapper(f):
            with graph.subtracer() as subtracer:
                subgraph_name, implicit_captures = subtracer.trace(f)
            node = Reduction(
                graph,
                *args,
                **kwargs,
                subgraph_name=subgraph_name,
                implicit_captures=implicit_captures,
            )
            node.add_to_graph(graph)
            node.fx_node.node.tkw_op = cls
            return node.fx_node

        return wrapper


@define_op("write2")
@dataclass
class Write(CustomOp):
    register_: fx.Proxy
    memory: fx.Proxy
    elements_per_thread: Optional[Any]
