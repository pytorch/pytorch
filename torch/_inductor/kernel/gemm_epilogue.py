# mypy: allow-untyped-defs
"""Backend-neutral FX graph and runtime argument helpers for GEMM epilogues."""

import dataclasses
from collections.abc import Iterator, Sequence
from typing import Any, ClassVar

import torch
from torch._inductor.virtualized import V
from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode
from torch.utils._ordered_set import OrderedSet

from .gemm_epilogue_utils import statically_known_shape_equal


@dataclasses.dataclass(frozen=True)
class GemmReductionGeometry:
    """Grouped M/N reduction geometry shared by frontend and backend plans.

    Attributes:
        group: Number of adjacent GEMM output elements in each reduction group.
        axis: GEMM output axis grouped by the reduction, either M (0) or N (1).
    """

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

    @property
    def group_size(self) -> int:
        return self.group

    @classmethod
    def from_output_shape(
        cls, output_shape: Sequence[Any], gemm_shape: Sequence[Any]
    ) -> "GemmReductionGeometry | None":
        if len(output_shape) != 3 or len(gemm_shape) != 2:
            return None
        for axis, group_dim in ((0, 1), (1, 2)):
            try:
                group = V.graph.sizevars.optimization_hint(output_shape[group_dim])
            except (GuardOnDataDependentSymNode, TypeError, ValueError):
                continue
            geometry = cls(group=group, axis=axis)
            if geometry.matches_output_shape(output_shape, gemm_shape):
                return geometry
        return None

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
            (m, n // self.group, self.group)
            if self.axis == 1
            else (m // self.group, self.group, n)
        )
        return statically_known_shape_equal(
            output_shape, (m, n)
        ) or statically_known_shape_equal(output_shape, grouped)


@dataclasses.dataclass(frozen=True)
class GemmReductionDescriptor:
    """Backend lowering descriptor for a recognized reduction expression.

    Attributes:
        kind: Canonical reduction or normalized-consumer expression name.
        parameters: Compile-time scalar parameters encoded by that expression.
    """

    kind: str
    parameters: tuple[float, ...] = ()

    @classmethod
    def parse(cls, value: str) -> "GemmReductionDescriptor":
        kind, *parameters = value.split(":")
        return cls(kind, tuple(float(parameter) for parameter in parameters))

    def serialize(self) -> str:
        if not self.parameters:
            return self.kind
        return (
            self.kind
            + ":"
            + ":".join(format(parameter, ".17g") for parameter in self.parameters)
        )


@dataclasses.dataclass(frozen=True)
class GemmReductionConfig:
    """Reduction recognized from frontend graph or scheduler loop IR.

    This is an analysis result, before output ownership and feed-main behavior
    are finalized into a :class:`GemmReductionPlan`.

    Attributes:
        output_name: Buffer produced by the recognized reduction.
        group: Number of adjacent GEMM output elements in each reduction group.
        axis: GEMM output axis grouped by the reduction, either M (0) or N (1).
        reduction_type: Reduction or normalized consumer expression to compute.
        source_type: Transformation applied to GEMM accumulator values.
    """

    output_name: str
    group: int
    axis: int
    reduction_type: str
    source_type: str

    @property
    def geometry(self) -> GemmReductionGeometry:
        return GemmReductionGeometry(self.group, self.axis)

    @property
    def contract(self) -> tuple[int, int, str, str]:
        return self.group, self.axis, self.reduction_type, self.source_type


@dataclasses.dataclass(frozen=True)
class GemmReductionPlan:
    """Backend-neutral reduction outputs passed from analysis to codegen.

    Attributes:
        reduction_output: Optional compressed reduction output buffer.
        group: Number of adjacent GEMM output elements in each reduction group.
        axis: GEMM output axis grouped by the reduction, either M (0) or N (1).
        reduction_type: Reduction or normalized consumer expression to compute.
        source_type: Transformation applied to GEMM accumulator values.
        primary_output: Buffer receiving the primary GEMM result.
        feeds_main: Whether the reduction participates in the primary output.
        feed_output: Optional full-shape output consuming the reduction.
        secondary_feed_output: Optional second full-shape reduction consumer.
        secondary_feed_type: Expression implemented by the secondary consumer.
    """

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
    def geometry(self) -> GemmReductionGeometry:
        return GemmReductionGeometry(self.group, self.axis)

    @property
    def auxiliary_outputs(self) -> tuple[str, ...]:
        return tuple(
            OrderedSet(
                output
                for output in (
                    self.reduction_output,
                    self.feed_output,
                    self.secondary_feed_output,
                )
                if output is not None and output != self.primary_output
            )
        )


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
