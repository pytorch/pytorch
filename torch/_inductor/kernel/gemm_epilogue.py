# mypy: allow-untyped-defs
"""Backend-neutral FX graph and runtime argument helpers for GEMM epilogues."""

import dataclasses
import operator
from collections.abc import Iterator, Sequence
from typing import Any, ClassVar

import torch
from torch._inductor import inductor_prims
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


@dataclasses.dataclass(frozen=True)
class NormalizedView:
    """Canonical source and shape for an FX view or reshape."""

    source: torch.fx.Node
    shape: tuple[Any, ...]


@dataclasses.dataclass(frozen=True)
class NormalizedDtypeView:
    """Canonical source and target dtype for a storage reinterpretation."""

    source: torch.fx.Node
    dtype: torch.dtype


@dataclasses.dataclass(frozen=True)
class NormalizedReduction:
    """Canonical arguments for a supported FX reduction."""

    source: torch.fx.Node
    dim: Any
    keepdim: Any
    dtype: Any
    reduction_type: str


@dataclasses.dataclass(frozen=True)
class NormalizedPrepareSoftmax:
    """Canonical source and dimension for online softmax preparation."""

    source: torch.fx.Node
    dim: Any


@dataclasses.dataclass(frozen=True)
class NormalizedSqueeze:
    """Canonical source for an FX squeeze alias."""

    source: torch.fx.Node


@dataclasses.dataclass(frozen=True)
class NormalizedGetItem:
    """Canonical aggregate source and literal FX getitem index."""

    source: torch.fx.Node
    index: int


@dataclasses.dataclass(frozen=True)
class NormalizedSplit:
    """Canonical source, width, and dimension for an FX tensor split."""

    source: torch.fx.Node
    split_size: Any
    dim: int


@dataclasses.dataclass(frozen=True)
class NormalizedSelect:
    """Canonical source, dimension, and index for an FX tensor select."""

    source: torch.fx.Node
    dim: int
    index: Any


@dataclasses.dataclass(frozen=True)
class NormalizedUnsupportedReduction:
    """Canonical source and target for an unsupported FX reduction."""

    source: torch.fx.Node
    target: str


NormalizedNode = (
    NormalizedView
    | NormalizedDtypeView
    | NormalizedReduction
    | NormalizedPrepareSoftmax
    | NormalizedSqueeze
    | NormalizedGetItem
    | NormalizedSplit
    | NormalizedSelect
    | NormalizedUnsupportedReduction
)


FUNCTION_REDUCTION_TYPES = {
    torch.ops.aten.sum.dim_IntList: ("sum", True),
    torch.ops.aten.mean.dim: ("mean", True),
    torch.ops.aten.prod.dim_int: ("prod", True),
    torch.ops.aten.amax.default: ("max", False),
    torch.ops.aten.amin.default: ("min", False),
}

FUNCTION_UNSUPPORTED_REDUCTIONS = frozenset(
    (
        torch.ops.aten.all.dim,
        torch.ops.aten.all.dims,
        torch.ops.aten.all.default,
        torch.ops.aten.any.dim,
        torch.ops.aten.any.dims,
        torch.ops.aten.any.default,
        torch.ops.aten.argmax.default,
        torch.ops.aten.argmin.default,
        torch.ops.aten.std.correction,
        torch.ops.aten.std.dim,
        torch.ops.aten.var.correction,
        torch.ops.aten.var.dim,
    )
)


def normalize_gemm_epilogue_fx_node(node: torch.fx.Node) -> NormalizedNode | None:
    """Return canonical arguments for a selected epilogue FX node."""
    if node.op != "call_function":
        return None
    if node.target is torch.ops.aten.view.dtype:
        source, dtype = node.args
        if not isinstance(source, torch.fx.Node) or not isinstance(dtype, torch.dtype):
            raise AssertionError(
                f"malformed GEMM epilogue dtype view: {node.format_node()}"
            )
        return NormalizedDtypeView(source, dtype)
    if node.target in (
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
    ):
        source = node.args[0]
        shape = node.args[1]
        if not isinstance(source, torch.fx.Node) or not isinstance(
            shape, (tuple, list, torch.Size)
        ):
            raise AssertionError(f"malformed GEMM epilogue view: {node.format_node()}")
        return NormalizedView(
            source,
            tuple(
                arg.meta.get("val", arg) if isinstance(arg, torch.fx.Node) else arg
                for arg in shape
            ),
        )
    if node.target in FUNCTION_REDUCTION_TYPES:
        source = node.args[0]
        if not isinstance(source, torch.fx.Node):
            raise AssertionError(
                f"malformed GEMM epilogue reduction: {node.format_node()}"
            )
        reduction_type, has_dtype = FUNCTION_REDUCTION_TYPES[node.target]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        keepdim = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        )
        dtype = node.args[3] if len(node.args) > 3 else node.kwargs.get("dtype")
        return NormalizedReduction(
            source,
            dim,
            keepdim,
            dtype if has_dtype else None,
            reduction_type,
        )
    if node.target is inductor_prims.prepare_softmax_online:
        source = node.args[0]
        if not isinstance(source, torch.fx.Node):
            raise AssertionError(
                f"malformed GEMM epilogue softmax: {node.format_node()}"
            )
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        return NormalizedPrepareSoftmax(source, dim)
    if node.target is torch.ops.aten.split.Tensor:
        source = node.args[0]
        dim = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim", 0)
        if not isinstance(source, torch.fx.Node) or not isinstance(dim, int):
            raise AssertionError(f"malformed GEMM epilogue split: {node.format_node()}")
        return NormalizedSplit(source, node.args[1], dim)
    if node.target is torch.ops.aten.select.int:
        source = node.args[0]
        dim = node.args[1]
        if not isinstance(source, torch.fx.Node) or not isinstance(dim, int):
            raise AssertionError(
                f"malformed GEMM epilogue select: {node.format_node()}"
            )
        return NormalizedSelect(source, dim, node.args[2])
    if node.target in (
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.dims,
        torch.ops.aten.squeeze.default,
    ):
        source = node.args[0]
        if not isinstance(source, torch.fx.Node):
            raise AssertionError(
                f"malformed GEMM epilogue squeeze: {node.format_node()}"
            )
        return NormalizedSqueeze(source)
    if node.target is operator.getitem:
        source, index = node.args
        if isinstance(source, torch.fx.Node) and isinstance(index, int):
            return NormalizedGetItem(source, index)
        return None
    if node.target in FUNCTION_UNSUPPORTED_REDUCTIONS:
        source = node.args[0]
        if not isinstance(source, torch.fx.Node):
            raise AssertionError(
                f"malformed GEMM epilogue reduction: {node.format_node()}"
            )
        return NormalizedUnsupportedReduction(source, str(node.target))
    return None


def iter_fx_node_inputs(value: Any) -> Iterator[torch.fx.Node]:
    """Yield FX node inputs nested in args/kwargs-style containers."""
    result: list[torch.fx.Node] = []
    torch.fx.map_arg(value, lambda node: result.append(node))
    yield from result


@dataclasses.dataclass(frozen=True)
class GemmEpilogueGraph:
    """Index dependencies and canonical interpretations of epilogue FX nodes."""

    dependencies: dict[torch.fx.Node, frozenset[torch.fx.Node]]
    normalized_nodes: dict[torch.fx.Node, NormalizedNode]

    @classmethod
    def from_nodes(cls, nodes: Sequence[torch.fx.Node]) -> "GemmEpilogueGraph":
        """Build the FX dependency and normalization indexes in one graph walk."""
        dependencies: dict[torch.fx.Node, frozenset[torch.fx.Node]] = {}
        normalized_nodes: dict[torch.fx.Node, NormalizedNode] = {}
        for node in nodes:
            node_dependencies: OrderedSet[torch.fx.Node] = OrderedSet()
            for input_node in iter_fx_node_inputs((node.args, node.kwargs)):
                node_dependencies.add(input_node)
                node_dependencies.update(dependencies.get(input_node, ()))
            dependencies[node] = frozenset(node_dependencies)
            if (normalized := normalize_gemm_epilogue_fx_node(node)) is not None:
                normalized_nodes[node] = normalized
        return cls(dependencies, normalized_nodes)

    def depends_on(self, value: Any, target: torch.fx.Node) -> bool:
        """Return whether a value is or transitively depends on the target node."""
        return any(
            node is target or target in self.dependencies.get(node, ())
            for node in iter_fx_node_inputs(value)
        )
