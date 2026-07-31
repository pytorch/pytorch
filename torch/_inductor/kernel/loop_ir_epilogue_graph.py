# mypy: allow-untyped-defs
"""Trace Inductor loop IR into a restricted GEMM epilogue graph."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import sympy

from torch._inductor.ops_handler import DefaultHandler
from torch._inductor.virtualized import V


@dataclasses.dataclass(frozen=True)
class LoopIREpilogueValue:
    index: int


@dataclasses.dataclass(frozen=True)
class LoopIREpilogueNode:
    op: str
    args: tuple[Any, ...]
    kwargs: tuple[tuple[str, Any], ...] = ()


@dataclasses.dataclass(frozen=True)
class LoopIREpilogueStore:
    index: sympy.Expr
    value: LoopIREpilogueValue


@dataclasses.dataclass(frozen=True)
class LoopIREpilogueGraph:
    nodes: tuple[LoopIREpilogueNode, ...]
    stores: dict[str, LoopIREpilogueStore]

    def reachable_values(self, output: str) -> frozenset[int]:
        seen: set[int] = set()

        def visit(value: Any) -> None:
            if not isinstance(value, LoopIREpilogueValue) or value.index in seen:
                return
            seen.add(value.index)
            for arg in self.nodes[value.index].args:
                visit(arg)

        visit(self.stores[output].value)
        return frozenset(seen)

    def reduction_boundary(self, output: str, accumulator: str) -> LoopIREpilogueValue:
        reachable = self.reachable_values(output)
        reductions = [
            LoopIREpilogueValue(index)
            for index in reachable
            if self.nodes[index].op == "reduction"
        ]
        if len(reductions) != 1:
            raise NotImplementedError(
                "GEMM epilogue output must have one reduction boundary"
            )
        reduction = reductions[0]
        reduction_inputs: set[str] = set()

        def collect_loads(value: Any) -> None:
            if not isinstance(value, LoopIREpilogueValue):
                return
            node = self.nodes[value.index]
            if node.op == "load":
                reduction_inputs.add(node.args[0])
            for arg in node.args:
                collect_loads(arg)

        collect_loads(self.nodes[reduction.index].args[3])
        if reduction_inputs != {accumulator}:
            raise NotImplementedError(
                f"reduction must consume only {accumulator}: {reduction_inputs}"
            )
        external_loads = {
            self.nodes[index].args[0]
            for index in reachable
            if self.nodes[index].op == "load" and self.nodes[index].args[2] is None
        }
        if external_loads != {accumulator}:
            captures = external_loads - {accumulator}
            raise NotImplementedError(
                f"GEMM epilogue captures unsupported tensors: {captures}"
            )
        return reduction


class LoopIREpilogueGraphHandler(DefaultHandler):
    """Record supported loop IR operations and link loads to producer stores."""

    SUPPORTED_OPS = frozenset(
        {
            "abs",
            "add",
            "constant",
            "exp",
            "ge",
            "gt",
            "index_expr",
            "log",
            "maximum",
            "minimum",
            "mul",
            "neg",
            "reciprocal",
            "relu",
            "sqrt",
            "sub",
            "to_dtype",
            "truediv",
            "where",
        }
    )
    SUPPORTED_REDUCTIONS = frozenset(
        {"sum", "max", "min", "prod", "online_softmax_reduce", "welford_reduce"}
    )

    def __init__(self) -> None:
        self.nodes: list[LoopIREpilogueNode] = []
        self.stores: dict[str, LoopIREpilogueStore] = {}
        self._interned: dict[LoopIREpilogueNode, LoopIREpilogueValue] = {}

    def emit(self, op: str, *args: Any, **kwargs: Any) -> LoopIREpilogueValue:
        node = LoopIREpilogueNode(op, args, tuple(sorted(kwargs.items())))
        if node not in self._interned:
            self._interned[node] = LoopIREpilogueValue(len(self.nodes))
            self.nodes.append(node)
        return self._interned[node]

    def _default(
        self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> LoopIREpilogueValue:
        if name not in self.SUPPORTED_OPS:
            raise NotImplementedError(f"unsupported GEMM epilogue operation: {name}")
        return self.emit(name, *args, **kwargs)

    def load(self, name: str, index: sympy.Expr) -> LoopIREpilogueValue:
        producer = self.stores.get(name)
        return self.emit("load", name, index, producer.value if producer else None)

    def reduction(self, dtype, src_dtype, reduction_type, value):
        if reduction_type not in self.SUPPORTED_REDUCTIONS:
            raise NotImplementedError(
                f"unsupported GEMM epilogue reduction: {reduction_type}"
            )
        reduction = self.emit("reduction", dtype, src_dtype, reduction_type, value)
        if reduction_type in {"online_softmax_reduce", "welford_reduce"}:
            count = 2 if reduction_type == "online_softmax_reduce" else 3
            return tuple(
                self.emit("getitem", reduction, index) for index in range(count)
            )
        return reduction

    def store(
        self,
        name: str,
        index: sympy.Expr,
        value: LoopIREpilogueValue,
        mode=None,
    ) -> None:
        if mode is not None:
            raise NotImplementedError("mutating GEMM epilogue stores are unsupported")
        self.stores[name] = LoopIREpilogueStore(index, value)

    def store_reduction(
        self, name: str, index: sympy.Expr, value: LoopIREpilogueValue
    ) -> None:
        self.stores[name] = LoopIREpilogueStore(index, value)

    def graph(self) -> LoopIREpilogueGraph:
        return LoopIREpilogueGraph(tuple(self.nodes), dict(self.stores))


def trace_loop_ir_epilogue(
    bodies: Sequence[Callable[[], None]],
) -> LoopIREpilogueGraph:
    handler = LoopIREpilogueGraphHandler()
    with V.set_ops_handler(handler):
        for body in bodies:
            body()
    return handler.graph()


def trace_loop_ir_epilogue_buffers(buffers: Sequence[Any]) -> LoopIREpilogueGraph:
    handler = LoopIREpilogueGraphHandler()
    with V.set_ops_handler(handler):
        for buffer in buffers:
            buffer.get_store_function()(*buffer.data.inner_fn_args())
    return handler.graph()
