# mypy: allow-untyped-defs
"""Backend-neutral FX graph and runtime argument helpers for GEMM epilogues."""

import dataclasses
from collections.abc import Iterator, Sequence
from typing import Any, ClassVar

import torch
from torch._inductor.kernel.gemm_epilogue_ir import GemmReductionDescriptor
from torch.utils._ordered_set import OrderedSet


@dataclasses.dataclass(frozen=True)
class GemmReductionArguments:
    """Typed view of grouped-reduction fields attached to GEMM arguments."""

    output: Any | None
    feed_output: Any | None
    secondary_feed_output: Any | None
    secondary_feed_type: str | None
    group: int
    axis: int
    reduction_type: str
    source_type: str
    feeds_main: bool

    SPECIALIZATION_KEYS: ClassVar[tuple[str, ...]] = (
        "local_reduce_group",
        "local_reduce_axis",
        "local_reduce_type",
        "local_reduce_source",
        "local_reduce_feeds_main",
        "local_reduce_secondary_feed_type",
    )

    @classmethod
    def from_operator_args(cls, args: Any) -> "GemmReductionArguments":
        return cls(
            getattr(args, "local_reduce_out", None),
            getattr(args, "local_reduce_feed_out", None),
            getattr(args, "local_reduce_secondary_feed_out", None),
            getattr(args, "local_reduce_secondary_feed_type", None),
            getattr(args, "local_reduce_group", 0),
            getattr(args, "local_reduce_axis", 1),
            getattr(args, "local_reduce_type", "sum"),
            getattr(args, "local_reduce_source", "identity"),
            getattr(args, "local_reduce_feeds_main", False),
        )

    @property
    def enabled(self) -> bool:
        return (
            any(
                value is not None
                for value in (self.output, self.feed_output, self.secondary_feed_output)
            )
            or self.feeds_main
        )

    @property
    def primary_enabled(self) -> bool:
        return (
            self.output is not None or self.feed_output is not None or self.feeds_main
        )

    @property
    def descriptor(self) -> GemmReductionDescriptor:
        return GemmReductionDescriptor.parse(self.reduction_type)

    def tensors(self, attr: str) -> tuple[Any | None, Any | None, Any | None]:
        def tensor(value: Any | None) -> Any | None:
            return getattr(value, attr) if value is not None else None

        return (
            tensor(self.output),
            tensor(self.feed_output),
            tensor(self.secondary_feed_output),
        )


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

    def depends_on(self, value: Any, target: torch.fx.Node) -> bool:
        return any(
            node is target or target in self.dependencies.get(node, ())
            for node in iter_fx_node_inputs(value)
        )
