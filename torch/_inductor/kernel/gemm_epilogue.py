# mypy: allow-untyped-defs
"""Backend-neutral FX graph helpers for GEMM epilogues."""

import dataclasses
from collections.abc import Iterator, Sequence
from typing import Any

import sympy

import torch
from torch._inductor.virtualized import V
from torch.fx.experimental.symbolic_shapes import (
    statically_known_true as fx_statically_known_true,
)
from torch.utils._ordered_set import OrderedSet


def statically_known(expr: Any) -> bool:
    """Return whether a symbolic predicate is known true without adding guards."""
    if isinstance(expr, bool):
        return expr
    if isinstance(expr, sympy.Basic):
        return V.graph.sizevars.statically_known_true(expr)
    return fx_statically_known_true(expr)


def statically_known_equal(lhs: Any, rhs: Any) -> bool:
    """Return whether symbolic shape values are known equal without adding guards."""
    return statically_known(lhs == rhs)


def statically_known_shape_equal(
    actual_shape: Sequence[Any], expected_shape: Sequence[Any]
) -> bool:
    """Compare possibly symbolic shape tuples without adding guards."""
    return len(actual_shape) == len(expected_shape) and all(
        statically_known_equal(actual, expected)
        for actual, expected in zip(actual_shape, expected_shape)
    )


@dataclasses.dataclass(frozen=True)
class GemmReductionGeometry:
    """Describe the grouped output axis shared by GEMM reduction consumers."""

    group: int
    axis: int

    def __post_init__(self) -> None:
        if self.group <= 0:
            raise RuntimeError("local_reduce_group must be positive")
        if self.axis not in (0, 1):
            raise RuntimeError("local_reduce_axis must be 0 or 1")

    @property
    def needs_physical_callbacks(self) -> bool:
        return self.axis == 0 or self.group > 32


@dataclasses.dataclass(frozen=True)
class GemmReductionConfig:
    """Describe a grouped reduction recognized during scheduler analysis."""

    output_name: str
    group: int
    axis: int
    reduction_type: str
    source_type: str

    @property
    def contract(self) -> tuple[int, int, str, str]:
        return self.group, self.axis, self.reduction_type, self.source_type

    def replace(
        self,
        *,
        output_name: str | None = None,
        reduction_type: str | None = None,
        source_type: str | None = None,
    ) -> "GemmReductionConfig":
        return dataclasses.replace(
            self,
            output_name=self.output_name if output_name is None else output_name,
            reduction_type=(
                self.reduction_type if reduction_type is None else reduction_type
            ),
            source_type=self.source_type if source_type is None else source_type,
        )


@dataclasses.dataclass(frozen=True)
class GemmReductionPlan:
    """Describe grouped reduction outputs passed from lowering to codegen."""

    reduction_output: str | None
    group: int
    axis: int
    reduction_type: str
    source_type: str
    primary_output: str
    feeds_main: bool = False
    feed_output: str | None = None
    secondary_feed_output: str | None = None
    secondary_feed_type: str | None = None

    @property
    def geometry(self) -> tuple[int, int]:
        return self.group, self.axis

    @property
    def auxiliary_outputs(self) -> tuple[str, ...]:
        return tuple(
            output
            for output in (
                self.reduction_output,
                self.feed_output,
                self.secondary_feed_output,
            )
            if output is not None
        )

    def with_primary_output(self, output: str) -> "GemmReductionPlan":
        return dataclasses.replace(self, primary_output=output)


@dataclasses.dataclass(frozen=True)
class GroupedReductionLayout:
    """Describe a grouped view over one axis of a two-dimensional GEMM output."""

    axis: int
    group_size: int

    @property
    def reduce_dims(self) -> tuple[int, ...]:
        return (-1, 2) if self.axis == 1 else (-2, 1)

    def matches_reduction_dim(self, dim: Any) -> bool:
        dims = tuple(dim) if isinstance(dim, (list, tuple)) else (dim,)
        return len(dims) == 1 and dims[0] in self.reduce_dims

    def matches_output_shape(
        self, output_shape: Sequence[Any], gemm_shape: Sequence[Any]
    ) -> bool:
        if len(gemm_shape) != 2:
            return False
        m, n = gemm_shape
        grouped = (
            (m, n // self.group_size, self.group_size)
            if self.axis == 1
            else (m // self.group_size, self.group_size, n)
        )
        return statically_known_shape_equal(
            output_shape, (m, n)
        ) or statically_known_shape_equal(output_shape, grouped)


def iter_fx_node_inputs(value: Any) -> Iterator[torch.fx.Node]:
    """Yield FX node inputs nested in args/kwargs-style containers."""
    result: list[torch.fx.Node] = []
    torch.fx.map_arg(value, lambda node: result.append(node))
    yield from result


@dataclasses.dataclass(frozen=True)
class GemmEpilogueGraph:
    """Index transitive dependencies between nodes in an epilogue FX graph."""

    dependencies: dict[torch.fx.Node, frozenset[torch.fx.Node]]

    @classmethod
    def from_nodes(cls, nodes: Sequence[torch.fx.Node]) -> "GemmEpilogueGraph":
        dependencies: dict[torch.fx.Node, frozenset[torch.fx.Node]] = {}
        for node in nodes:
            node_dependencies: OrderedSet[torch.fx.Node] = OrderedSet()
            for input_node in iter_fx_node_inputs((node.args, node.kwargs)):
                node_dependencies.add(input_node)
                node_dependencies.update(dependencies.get(input_node, ()))
            dependencies[node] = frozenset(node_dependencies)
        return cls(dependencies)

    @classmethod
    def from_graph_module(
        cls, graph_module: torch.fx.GraphModule
    ) -> "GemmEpilogueGraph":
        return cls.from_nodes(tuple(graph_module.graph.nodes))

    def depends_on(self, value: Any, target: torch.fx.Node) -> bool:
        return any(
            node is target or target in self.dependencies.get(node, ())
            for node in iter_fx_node_inputs(value)
        )
