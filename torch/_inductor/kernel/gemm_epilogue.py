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
    """Runtime tensors and compile-time parameters for a grouped GEMM reduction.

    Attributes:
        output: Optional tensor receiving the compressed reduction.
        feed_output: Optional full-shape tensor receiving the reduction consumer.
        secondary_feed_output: Optional second full-shape reduction consumer.
        secondary_feed_type: Expression implemented by ``secondary_feed_output``.
        group: Number of adjacent GEMM output elements in each reduction group.
        axis: GEMM output axis grouped by the reduction, either M (0) or N (1).
        reduction_type: Reduction or normalized consumer expression to compute.
        source_type: Transformation applied to GEMM accumulator values.
        feeds_main: Whether the reduction also produces the primary GEMM output.
    """

    output: Any | None = None
    feed_output: Any | None = None
    secondary_feed_output: Any | None = None
    secondary_feed_type: str | None = None
    group: int = 0
    axis: int = 1
    reduction_type: str = "sum"
    source_type: str = "identity"
    feeds_main: bool = False

    SPECIALIZATION_FIELDS: ClassVar[tuple[str, ...]] = (
        "group",
        "axis",
        "reduction_type",
        "source_type",
        "feeds_main",
        "secondary_feed_type",
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
