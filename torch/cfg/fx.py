from __future__ import annotations

from typing import Any

import torch
from torch.fx import Graph as FxGraph, GraphModule, Node
from torch.fx.passes.shape_prop import TensorMetadata

from .ir import (
    Block,
    Graph,
    Instruction,
    Literal,
    Location,
    ObjectSpec,
    Return,
    ScalarSpec,
    Spec,
    TensorSpec,
    Value,
)


__all__ = ["from_fx"]


def from_fx(graph: FxGraph | GraphModule, *, name: str | None = None) -> Graph:
    """
    Lower a straight-line FX graph into a single-block CFG.

    The adapter normalizes FX's metadata conventions into a single ``Value.spec``
    field, so downstream code never needs to branch on ``node.meta["val"]``
    versus ``node.meta["example_value"]``. It intentionally preserves FX
    operations verbatim inside one block; higher-order control-flow operators
    remain opaque instructions until a dedicated structured lowering is added.
    """

    fx_graph = graph.graph if isinstance(graph, GraphModule) else graph
    graph_name = name
    if graph_name is None:
        graph_name = getattr(graph, "__class__", type(graph)).__name__

    values: dict[Node, Value] = {}
    parameters: list[Value] = []
    instructions: list[Instruction] = []
    terminator: Return | None = None

    for node in fx_graph.nodes:
        if node.op == "placeholder":
            value = Value(node.name, spec=_node_spec(node))
            values[node] = value
            parameters.append(value)
            continue

        if node.op == "output":
            terminator = Return(_convert_argument(node.args[0], values))
            continue

        value = Value(node.name, spec=_node_spec(node))
        values[node] = value
        instructions.append(
            Instruction(
                name=node.name,
                opcode=node.op,
                target=node.target,
                inputs=tuple(_convert_argument(arg, values) for arg in node.args),
                attributes={
                    key: _convert_argument(arg, values)
                    for key, arg in node.kwargs.items()
                },
                outputs=(value,),
                location=_node_location(node),
            )
        )

    if terminator is None:
        raise ValueError("FX graph is missing an output node")

    return Graph(
        name=graph_name,
        entry="entry",
        blocks=(
            Block(
                name="entry",
                parameters=tuple(parameters),
                instructions=tuple(instructions),
                terminator=terminator,
            ),
        ),
    )


def _convert_argument(argument: Any, values: dict[Node, Value]) -> Any:
    if isinstance(argument, Node):
        return values[argument]
    if isinstance(argument, tuple):
        return tuple(_convert_argument(elem, values) for elem in argument)
    if isinstance(argument, list):
        return [_convert_argument(elem, values) for elem in argument]
    if isinstance(argument, dict):
        return {
            str(key): _convert_argument(value, values)
            for key, value in argument.items()
        }
    if isinstance(argument, slice):
        return slice(
            _convert_argument(argument.start, values),
            _convert_argument(argument.stop, values),
            _convert_argument(argument.step, values),
        )
    return Literal(argument)


def _node_location(node: Node) -> Location | None:
    stack = node.meta.get("stack_trace")
    if stack is None:
        return None
    return Location(stack=stack)


def _node_spec(node: Node) -> Spec | None:
    if "val" in node.meta:
        return Spec.from_value(node.meta["val"])
    if "example_value" in node.meta:
        return Spec.from_value(node.meta["example_value"])
    tensor_meta = node.meta.get("tensor_meta")
    if isinstance(tensor_meta, TensorMetadata):
        return TensorSpec(
            shape=tuple(tensor_meta.shape),
            dtype=tensor_meta.dtype,
            device=torch.device("meta"),
            stride=tuple(tensor_meta.stride),
            requires_grad=tensor_meta.requires_grad,
        )
    if isinstance(node.type, type):
        if issubclass(node.type, (bool, int, float, complex, str, bytes)):
            return ScalarSpec(node.type)
        return ObjectSpec(node.type)
    return None
