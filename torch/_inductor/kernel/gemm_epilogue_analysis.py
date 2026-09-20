# mypy: allow-untyped-defs
"""Shared FX analysis and output planning for grouped GEMM epilogues."""

import dataclasses
import math
from typing import Any

import torch
from torch._inductor.kernel.flex_gemm.constraints import (
    FLEX_GEMM_OUTPUT_CONTRACTION_SHAPE_ERROR,
    FLEX_GEMM_OUTPUT_PLAN_NODE_ERROR,
    FlexGemmOutputContraction,
    local_reduce_compressed_shape,
    LOCAL_REDUCE_EXPLICIT_DTYPE_ERROR,
    LOCAL_REDUCE_FEED_MAIN_AXIS1_FRAGMENT_ERROR,
    LOCAL_REDUCE_FEED_MAIN_MIXED_MATCH_ERROR,
    LOCAL_REDUCE_FEED_MAIN_SAME_WARP_ERROR,
    LOCAL_REDUCE_FRAGMENT_WIDTH,
    LOCAL_REDUCE_GROUPED_RESHAPE_ERROR,
    LOCAL_REDUCE_INNERMOST_GROUPED_DIM_ERROR,
    LOCAL_REDUCE_MATCH_NODE_ERROR,
    LOCAL_REDUCE_MIXED_GROUPED_LAYOUT_ERROR,
    LOCAL_REDUCE_MIXED_MATCH_ERROR,
    LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR,
    LOCAL_REDUCE_OUTPUT_PLAN_NODE_ERROR,
    LOCAL_REDUCE_SOURCE_EXPRESSION_ERROR,
    ungrouped_reduction_error,
    unsupported_reduction_op_error,
    validate_local_reduce_tensorssa_group_size,
)
from torch._inductor.kernel.flex_gemm.output_layout import (
    BLOCKED_128X4,
    FlexGemmOutputStorageLayout,
    TRANSPOSED,
)
from torch._inductor.kernel.flex_gemm.quack_reductions import (
    FlexGemmStructuralInt,
    is_shape_preserving_pointwise_node,
    tensor_meta_shape,
)
from torch._inductor.kernel.gemm_epilogue import (
    GemmEpilogueGraph,
    GemmReductionGeometry,
    iter_fx_node_inputs,
    NormalizedGetItem,
    NormalizedReduction,
    NormalizedSelect,
    NormalizedSplit,
    NormalizedSqueeze,
    NormalizedUnsupportedReduction,
    NormalizedView,
)
from torch._inductor.kernel.gemm_epilogue_utils import (
    normalize_shape,
    statically_known_equal,
    statically_known_shape_equal,
)
from torch.utils._ordered_set import OrderedSet


def _syntactic_grouped_tensor_layout(
    shape: tuple[Any, ...],
) -> GemmReductionGeometry | None:
    """Match grouped-reshape syntax before validating source geometry."""
    if len(shape) not in (3, 4):
        return None
    last = FlexGemmStructuralInt.from_value(shape[-1])
    penultimate = FlexGemmStructuralInt.from_value(shape[-2])
    if (
        last is not None
        and last.value > 0
        and type(shape[-2]) is int
        and shape[-2] == -1
    ):
        return GemmReductionGeometry(group=last.value, axis=1)
    if (
        type(shape[-3]) is int
        and shape[-3] == -1
        and penultimate is not None
        and penultimate.value > 0
    ):
        return GemmReductionGeometry(group=penultimate.value, axis=0)
    return None


def _group_count_matches_selected_dim(
    group_count: Any, selected_size: Any, group: int
) -> bool:
    if type(group_count) is int and group_count == -1:
        return True
    return statically_known_equal(
        group_count * group, selected_size
    ) or statically_known_equal(group_count, selected_size // group)


def _grouped_layout_matches_source_shape(
    shape: tuple[Any, ...],
    source_shape: tuple[Any, ...],
    layout: GemmReductionGeometry,
) -> bool:
    """Require a 2-D GEMM output reshape to split exactly M or N."""
    if len(shape) != 3:
        return False

    m, n = source_shape
    match layout.axis, shape:
        case 1, (kept_m, group_count, group):
            structural_group = FlexGemmStructuralInt.from_value(group)
            return (
                structural_group is not None
                and structural_group.value == layout.group
                and statically_known_equal(kept_m, m)
                and _group_count_matches_selected_dim(group_count, n, layout.group)
            )
        case 0, (group_count, group, kept_n):
            structural_group = FlexGemmStructuralInt.from_value(group)
            return (
                structural_group is not None
                and structural_group.value == layout.group
                and statically_known_equal(kept_n, n)
                and _group_count_matches_selected_dim(group_count, m, layout.group)
            )
        case _:
            return False


def grouped_tensor_layout(
    shape: Any, source_shape: Any | None = None
) -> GemmReductionGeometry | None:
    """Recognize grouped M/N geometry, specializing backed group dimensions."""
    shape = normalize_shape(shape)
    if not isinstance(shape, tuple):
        return None
    if len(shape) == 1 and isinstance(shape[0], (list, tuple, torch.Size)):
        shape = tuple(shape[0])
    if source_shape is not None:
        source_shape = normalize_shape(source_shape)
        if isinstance(source_shape, tuple) and len(source_shape) == 2:
            candidates = []
            if shape:
                group = FlexGemmStructuralInt.from_value(shape[-1])
                if group is not None and group.value > 0:
                    candidates.append(GemmReductionGeometry(group=group.value, axis=1))
            if len(shape) >= 2:
                group = FlexGemmStructuralInt.from_value(shape[-2])
                if group is not None and group.value > 0:
                    candidates.append(GemmReductionGeometry(group=group.value, axis=0))
            for layout in candidates:
                if _grouped_layout_matches_source_shape(shape, source_shape, layout):
                    return layout
            if _syntactic_grouped_tensor_layout(shape) is not None:
                raise NotImplementedError(LOCAL_REDUCE_GROUPED_RESHAPE_ERROR)
            return None
    return _syntactic_grouped_tensor_layout(shape)


FEED_MAIN_BINARY_FUNCTIONS = frozenset(
    (
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add.Scalar,
        torch.ops.aten.div.Tensor,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.mul.Scalar,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.sub.Scalar,
    )
)


@dataclasses.dataclass(frozen=True)
class GemmLocalReduceMatch:
    """Describe a supported grouped local-reduction value found in the FX graph.

    Attributes:
        value_node: FX node that produces the matched local-reduction value.
        geometry: Group size and GEMM output axis reduced by the value.
    """

    value_node: torch.fx.Node
    geometry: GemmReductionGeometry

    def __post_init__(self) -> None:
        if not isinstance(self.value_node, torch.fx.Node):
            raise RuntimeError(LOCAL_REDUCE_MATCH_NODE_ERROR)

    def to_plan(
        self,
        *,
        store: "GemmLocalReduceStore | None",
        feeds_main: bool,
    ) -> "GemmOutputLocalReducePlan":
        """Bind this matched value to its output consumers."""
        return GemmOutputLocalReducePlan(self, store=store, feeds_main=feeds_main)

    @classmethod
    def common(
        cls,
        matches: list["GemmLocalReduceMatch"],
        mixed_match_error: str,
    ) -> "GemmLocalReduceMatch | None":
        """Return the common match when all values use one reduction geometry."""
        if not matches:
            return None
        match = matches[0]
        if any(item.geometry != match.geometry for item in matches):
            raise NotImplementedError(mixed_match_error)
        return match

    @classmethod
    def common_value(
        cls,
        matches: list["GemmLocalReduceMatch"],
        mixed_match_error: str,
    ) -> "GemmLocalReduceMatch | None":
        """Return the common match when all consumers use one physical value."""
        match = cls.common(matches, mixed_match_error)
        if match is None:
            return None
        if any(item.value_node is not match.value_node for item in matches):
            raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
        return match


@dataclasses.dataclass(frozen=True)
class FlexGemmLocalReduceOutputStorage:
    """Describe the physical storage selected for a returned local reduction."""

    source: torch.fx.Node
    layout: FlexGemmOutputStorageLayout
    nodes: tuple[torch.fx.Node, ...]


def match_flex_gemm_local_reduce_output_storage(
    node: torch.fx.Node,
) -> FlexGemmLocalReduceOutputStorage | None:
    """Recognize a supported terminal storage transform for a local reduction."""
    if node.target is torch.ops.flex_gemm.to_blocked.default:
        source = node.args[0] if node.args else None
        if not isinstance(source, torch.fx.Node):
            raise AssertionError(
                f"malformed FlexGEMM output transform: {node.format_node()}"
            )
        return FlexGemmLocalReduceOutputStorage(source, BLOCKED_128X4, (node,))

    if node.target is torch.ops.aten.clone.default:
        if node.kwargs.get("memory_format") not in (None, torch.contiguous_format):
            return None
        transpose = node.args[0]
        if not isinstance(transpose, torch.fx.Node) or tuple(transpose.users) != (
            node,
        ):
            return None
        nodes = (transpose, node)
    else:
        transpose = node
        nodes = (node,)

    if transpose.target not in (
        torch.ops.aten.t.default,
        torch.ops.aten.transpose.int,
        torch.ops.aten.permute.default,
    ):
        return None
    source = transpose.args[0]
    if not isinstance(source, torch.fx.Node):
        return None

    source_meta = source.meta.get("val")
    transpose_meta = transpose.meta.get("val")
    output_meta = node.meta.get("val")
    if (
        not isinstance(source_meta, torch.Tensor)
        or not isinstance(transpose_meta, torch.Tensor)
        or not isinstance(output_meta, torch.Tensor)
        or source_meta.ndim != 2
        or not statically_known_shape_equal(output_meta.shape, source_meta.shape[::-1])
        or not statically_known_shape_equal(
            transpose_meta.stride(), source_meta.stride()[::-1]
        )
        or not output_meta.is_contiguous()
        or not node.users
        or any(user.op != "output" for user in node.users)
    ):
        return None
    return FlexGemmLocalReduceOutputStorage(source, TRANSPOSED, nodes)


@dataclasses.dataclass(frozen=True)
class GemmLocalReduceStore:
    """Describe a logical reduction value and its returned physical carrier."""

    node: torch.fx.Node
    output_storage: FlexGemmLocalReduceOutputStorage | None = None

    def __post_init__(self) -> None:
        storage = self.output_storage
        if not isinstance(self.node, torch.fx.Node) or (
            storage is not None
            and (
                not isinstance(storage, FlexGemmLocalReduceOutputStorage)
                or not storage.nodes
                or storage.nodes[-1] is not self.node
            )
        ):
            raise RuntimeError(LOCAL_REDUCE_OUTPUT_PLAN_NODE_ERROR)

    @property
    def value_node(self) -> torch.fx.Node:
        """Return the logical value written through the optional storage transform."""
        return self.node if self.output_storage is None else self.output_storage.source

    @property
    def output_layout(self) -> FlexGemmOutputStorageLayout | None:
        """Return the selected physical storage layout."""
        return None if self.output_storage is None else self.output_storage.layout


@dataclasses.dataclass(frozen=True)
class GemmOutputLocalReducePlan:
    """Bind a matched local reduction to store and/or main-output consumers.

    Attributes:
        match: Supported local-reduction value identified during FX analysis.
        store: Compressed auxiliary output receiving the value, when requested.
        feeds_main: Whether the reduced value is also consumed by the main output.
    """

    match: GemmLocalReduceMatch
    store: GemmLocalReduceStore | None = None
    feeds_main: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.match, GemmLocalReduceMatch) or (
            self.store is None and not self.feeds_main
        ):
            raise RuntimeError(LOCAL_REDUCE_OUTPUT_PLAN_NODE_ERROR)


@dataclasses.dataclass(frozen=True)
class GemmOutputPlan:
    """Classify the values returned by a FlexGEMM body.

    Attributes:
        output: FX node returned as the main GEMM result.
        returned_aux_outputs: Auxiliary FX outputs in the user-visible tuple order.
        local_reduce: Compressed or feed-main local-reduction output behavior.
    """

    output: torch.fx.Node
    returned_aux_outputs: tuple[torch.fx.Node, ...] = ()
    local_reduce: GemmOutputLocalReducePlan | None = None
    output_contraction: FlexGemmOutputContraction | None = None
    output_storage: torch.fx.Node | None = None
    output_storage_nodes: tuple[torch.fx.Node, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.output, torch.fx.Node)
            or not all(
                isinstance(aux_output, torch.fx.Node)
                for aux_output in self.returned_aux_outputs
            )
            or (
                self.output_storage is not None
                and not isinstance(self.output_storage, torch.fx.Node)
            )
            or not all(
                isinstance(node, torch.fx.Node) for node in self.output_storage_nodes
            )
        ):
            raise RuntimeError(FLEX_GEMM_OUTPUT_PLAN_NODE_ERROR)

    @property
    def aux_outputs(self) -> tuple[torch.fx.Node, ...]:
        """Return ordinary auxiliary values emitted by the generated callback."""
        store = None if self.local_reduce is None else self.local_reduce.store
        return tuple(
            output
            for output in self.returned_aux_outputs
            if store is None or output is not store.node
        )


@dataclasses.dataclass
class GemmLocalReduceAnalysis:
    """Collect grouped TensorSSA layouts and supported local-reduction matches.

    ``from_graph_module`` visits the FX graph in topological order. See
    ``GemmReductionGeometry`` for the grouped layout attached to reshape and
    pointwise nodes, and ``GemmLocalReduceMatch`` for each supported reduced
    value found from those layouts.

    Attributes:
        graph: Dependency index used by recursive feed-main matching.
        grouped_tensors: FX nodes whose values carry a grouped TensorSSA layout.
        matches: FX values matched to a supported grouped local reduction.
    """

    graph: GemmEpilogueGraph
    grouped_tensors: dict[torch.fx.Node, GemmReductionGeometry] = dataclasses.field(
        default_factory=dict
    )
    grouped_structural_values: dict[
        torch.fx.Node, tuple[FlexGemmStructuralInt, ...]
    ] = dataclasses.field(default_factory=dict)
    matches: dict[torch.fx.Node, GemmLocalReduceMatch] = dataclasses.field(
        default_factory=dict
    )
    gemm: torch.fx.Node | None = None
    gemm_shape: tuple[Any, ...] | None = None

    @classmethod
    def from_graph_module(
        cls,
        graph_module: torch.fx.GraphModule,
        gemm: torch.fx.Node | None = None,
    ) -> "GemmLocalReduceAnalysis":
        """Build shared dependency and reduction state in one topological pass."""
        gemm_shape = tensor_meta_shape(gemm) if gemm is not None else None
        analysis = cls(
            GemmEpilogueGraph.from_nodes(tuple(graph_module.graph.nodes)),
            gemm=gemm,
            gemm_shape=gemm_shape,
        )
        for node in graph_module.graph.nodes:
            if node.op == "output":
                break
            analysis.visit_node(node)
        return analysis

    def physical_reduction_nodes(
        self, match: GemmLocalReduceMatch
    ) -> tuple[torch.fx.Node, ...]:
        """Return grouped reductions contributing to a propagated match."""
        dependencies = self.graph.dependencies.get(match.value_node, ())
        return tuple(
            node
            for node in self.graph.dependencies
            if (node is match.value_node or node in dependencies)
            and isinstance(self.graph.normalized_nodes.get(node), NormalizedReduction)
            and node in self.matches
        )

    def visit_node(self, node: torch.fx.Node) -> None:
        """Record grouped layouts and local-reduction matches for one FX node."""
        if node.op != "call_function":
            return
        normalized = self.graph.normalized_nodes.get(node)
        if isinstance(normalized, NormalizedView):
            propagated = self.propagate_local_reduce_match(node, normalized.source)
            grouped = self.bind_grouped_layout(
                node, normalized.shape, normalized.source
            )
            if propagated or grouped:
                return
        if isinstance(normalized, NormalizedReduction):
            if self.bind_grouped_reduction(node, normalized):
                return
            if (
                normalized.source not in self.grouped_tensors
                and self.gemm is not None
                and self.graph.depends_on(normalized.source, self.gemm)
            ):
                op_name = str(getattr(node.target, "overloadpacket", node.target))
                raise ungrouped_reduction_error(op_name)
        elif isinstance(normalized, NormalizedUnsupportedReduction):
            raise unsupported_reduction_op_error(normalized.target)
        if isinstance(
            normalized, (NormalizedSqueeze, NormalizedGetItem)
        ) and self.propagate_local_reduce_match(node, normalized.source):
            return
        if is_shape_preserving_pointwise_node(node):
            self.propagate_pointwise_match(node, LOCAL_REDUCE_MIXED_MATCH_ERROR)

    def bind_grouped_layout(self, node: torch.fx.Node, shape: Any, source: Any) -> bool:
        """Attach a grouped TensorSSA layout introduced by a reshape."""
        source_shape = (
            tensor_meta_shape(source) if isinstance(source, torch.fx.Node) else None
        )
        node_shape = tensor_meta_shape(node)
        if (
            isinstance(source, torch.fx.Node)
            and source in self.matches
            and self.gemm_shape is not None
            and (
                node_shape is None
                or not statically_known_equal(
                    math.prod(node_shape), math.prod(self.gemm_shape)
                )
            )
        ):
            return False
        layout = grouped_tensor_layout(shape, source_shape)
        if layout is None or not isinstance(source, torch.fx.Node):
            return False
        self.grouped_tensors[node] = layout
        shape_values = shape
        if (
            isinstance(shape_values, (list, tuple, torch.Size))
            and len(shape_values) == 1
            and isinstance(shape_values[0], (list, tuple, torch.Size))
        ):
            shape_values = shape_values[0]
        if isinstance(shape_values, (list, tuple, torch.Size)):
            group_index = -1 if layout.axis == 1 else -2
            structural = FlexGemmStructuralInt.from_value(shape_values[group_index])
            if structural is not None and structural.symbolic is not None:
                self.grouped_structural_values[node] = (structural,)
        return True

    def propagate_local_reduce_match(self, node: torch.fx.Node, source: Any) -> bool:
        """Copy a matched local-reduction value through an FX wrapper."""
        if not isinstance(source, torch.fx.Node):
            return False
        match = self.matches.get(source)
        if match is None:
            return False
        self.matches[node] = match
        return True

    def bind_grouped_reduction(
        self,
        node: torch.fx.Node,
        reduction: NormalizedReduction,
    ) -> bool:
        """Match and record a reduction over a grouped TensorSSA layout."""
        layout = self.grouped_tensors.get(reduction.source)
        if layout is None:
            return False
        if reduction.dtype is not None:
            raise NotImplementedError(LOCAL_REDUCE_EXPLICIT_DTYPE_ERROR)
        validate_local_reduce_tensorssa_group_size(layout.axis, layout.group)
        if not layout.matches_reduction_dim(reduction.dim):
            raise NotImplementedError(LOCAL_REDUCE_INNERMOST_GROUPED_DIM_ERROR)
        self.matches[node] = GemmLocalReduceMatch(node, layout)
        return True

    def has_physical_grouped_input(self, value: torch.fx.node.Argument) -> bool:
        """Return whether a value needs a cross-fragment grouped combine."""
        active_geometries = OrderedSet(
            match.geometry for match in self.matches.values()
        )
        physical_grouped_nodes = OrderedSet(
            node
            for node, layout in self.grouped_tensors.items()
            if layout.needs_physical_callbacks and layout in active_geometries
        )
        return any(
            node in physical_grouped_nodes
            or any(
                dependency in physical_grouped_nodes
                for dependency in self.graph.dependencies.get(node, ())
            )
            for node in iter_fx_node_inputs(value)
        )

    def propagate_pointwise_match(
        self, node: torch.fx.Node, mixed_match_error: str
    ) -> bool:
        """Propagate grouped layouts and local-reduction matches through pointwise ops."""
        grouped_layouts = [
            self.grouped_tensors[arg]
            for arg in iter_fx_node_inputs((node.args, node.kwargs))
            if arg in self.grouped_tensors
        ]
        if grouped_layouts:
            grouped_layout = grouped_layouts[0]
            if any(layout != grouped_layout for layout in grouped_layouts):
                raise NotImplementedError(LOCAL_REDUCE_MIXED_GROUPED_LAYOUT_ERROR)
            self.grouped_tensors[node] = grouped_layout
            structural_values = tuple(
                structural
                for arg in iter_fx_node_inputs((node.args, node.kwargs))
                for structural in self.grouped_structural_values.get(arg, ())
            )
            if structural_values:
                self.grouped_structural_values[node] = structural_values
        match = GemmLocalReduceMatch.common(
            [
                self.matches[arg]
                for arg in iter_fx_node_inputs((node.args, node.kwargs))
                if arg in self.matches
            ],
            mixed_match_error,
        )
        if match is None:
            return False
        self.matches[node] = dataclasses.replace(match, value_node=node)
        return True

    def commit_output_guards(self, outputs: GemmOutputPlan) -> None:
        """Commit backed grouped-shape hints only when they affect accepted outputs."""
        output_values = [outputs.output, *outputs.aux_outputs]
        if outputs.local_reduce is not None and outputs.local_reduce.store is not None:
            output_values.append(outputs.local_reduce.store.value_node)
        active_geometries = OrderedSet(
            match.geometry for match in self.matches.values()
        )
        if outputs.local_reduce is not None:
            active_geometries.add(outputs.local_reduce.match.geometry)
        seen: OrderedSet[tuple[int, int]] = OrderedSet()
        for node, values in self.grouped_structural_values.items():
            layout = self.grouped_tensors[node]
            geometry = layout
            if geometry not in active_geometries or not any(
                self.graph.depends_on(output, node) for output in output_values
            ):
                continue
            for structural in values:
                key = (id(structural.symbolic), structural.value)
                if key not in seen:
                    seen.add(key)
                    structural.guard()

    def match_feed_value(
        self,
        value: torch.fx.node.Argument,
        grouped_source: torch.fx.Node,
        layout: GemmReductionGeometry,
    ) -> GemmLocalReduceMatch | None:
        """Find the grouped normalized that produces a broadcast value."""
        if not isinstance(value, torch.fx.Node):
            return None
        normalized = self.graph.normalized_nodes.get(value)
        if isinstance(normalized, NormalizedReduction):
            if not self.same_grouped_view(normalized.source, grouped_source):
                if self.graph.depends_on(normalized.source, grouped_source):
                    if not (
                        layout.axis == 1
                        and layout.group <= LOCAL_REDUCE_FRAGMENT_WIDTH
                        and is_shape_preserving_pointwise_node(normalized.source)
                    ):
                        raise NotImplementedError(LOCAL_REDUCE_SOURCE_EXPRESSION_ERROR)
                else:
                    raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
            if (
                normalized.dtype is not None
                or not normalized.keepdim
                or not layout.matches_reduction_dim(normalized.dim)
            ):
                raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
            return GemmLocalReduceMatch(value, layout)
        if not is_shape_preserving_pointwise_node(value):
            return None
        matches = [
            match
            for arg in iter_fx_node_inputs((value.args, value.kwargs))
            if (match := self.match_feed_value(arg, grouped_source, layout)) is not None
        ]
        return GemmLocalReduceMatch.common_value(
            matches, LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR
        )

    def same_grouped_view(self, node: Any, grouped_source: torch.fx.Node) -> bool:
        """True for the grouped view itself or a sibling view of the same source and shape."""
        if node is grouped_source:
            return True
        if not isinstance(node, torch.fx.Node):
            return False
        this = self.graph.normalized_nodes.get(node)
        other = self.graph.normalized_nodes.get(grouped_source)
        return (
            isinstance(this, NormalizedView)
            and isinstance(other, NormalizedView)
            and this.source is other.source
            and self.grouped_tensors.get(node) is not None
            and self.grouped_tensors.get(node)
            == self.grouped_tensors.get(grouped_source)
        )

    def validate_hidden_feed_main_reduction_input(
        self,
        input_node: torch.fx.node.Argument,
        grouped_source: torch.fx.Node,
    ) -> None:
        """Reject reduction inputs that would need another physical feed-main value."""
        if self.same_grouped_view(input_node, grouped_source):
            raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
        if not isinstance(input_node, torch.fx.Node):
            return
        if self.graph.depends_on(input_node, grouped_source):
            raise NotImplementedError(LOCAL_REDUCE_SOURCE_EXPRESSION_ERROR)
        if self.has_physical_grouped_input(input_node):
            raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)

    def validate_feed_main_source_reductions(
        self,
        value: torch.fx.node.Argument,
        grouped_source: torch.fx.Node,
        selected_reduction: torch.fx.Node,
        seen: OrderedSet[torch.fx.Node] | None = None,
    ) -> None:
        """Reject hidden physical reductions outside the selected feed-main value."""
        if not isinstance(value, torch.fx.Node):
            for arg in iter_fx_node_inputs(value):
                self.validate_feed_main_source_reductions(
                    arg, grouped_source, selected_reduction, seen
                )
            return
        if value is selected_reduction:
            return
        if seen is None:
            seen = OrderedSet()
        if value in seen:
            return
        seen.add(value)
        normalized = self.graph.normalized_nodes.get(value)
        if isinstance(normalized, NormalizedReduction):
            self.validate_hidden_feed_main_reduction_input(
                normalized.source, grouped_source
            )
        for arg in iter_fx_node_inputs((value.args, value.kwargs)):
            self.validate_feed_main_source_reductions(
                arg, grouped_source, selected_reduction, seen
            )

    def validate_feed_main_source_match(
        self,
        source: torch.fx.Node,
        match: GemmLocalReduceMatch | None,
    ) -> GemmLocalReduceMatch | None:
        """Preserve the one-physical-value ABI across recursive source matching."""
        if match is None:
            return None
        normalized = self.graph.normalized_nodes.get(match.value_node)
        if isinstance(normalized, NormalizedReduction):
            self.validate_feed_main_source_reductions(
                source, normalized.source, match.value_node
            )
        return match

    @staticmethod
    def feed_main_binary_candidates(
        source: torch.fx.Node,
    ) -> tuple[tuple[Any, Any], ...]:
        """Return operand orderings for supported binary feed-main expressions."""
        if (
            len(source.args) < 2
            or source.op != "call_function"
            or source.target not in FEED_MAIN_BINARY_FUNCTIONS
        ):
            return ()
        lhs, rhs = source.args[:2]
        return ((lhs, rhs), (rhs, lhs))

    def feed_main_grouped_reduction(
        self,
        value: Any,
        grouped_source: torch.fx.Node,
        layout: GemmReductionGeometry,
    ) -> bool:
        """Return whether a candidate contains a grouped feed-main normalized."""
        if not isinstance(value, torch.fx.Node):
            return False
        normalized = self.graph.normalized_nodes.get(value)
        if isinstance(normalized, NormalizedReduction):
            return (
                normalized.dtype is None
                and bool(normalized.keepdim)
                and layout.matches_reduction_dim(normalized.dim)
                and (
                    self.same_grouped_view(normalized.source, grouped_source)
                    or self.graph.depends_on(normalized.source, grouped_source)
                )
            )
        if not is_shape_preserving_pointwise_node(value):
            return False
        return any(
            self.feed_main_grouped_reduction(arg, grouped_source, layout)
            for arg in iter_fx_node_inputs((value.args, value.kwargs))
        )

    def match_feed_main_candidate(
        self,
        grouped_source: Any,
        value: Any,
        output_meta: Any,
    ) -> GemmLocalReduceMatch | None:
        """Match one grouped-source and reduced-value operand ordering."""
        if not isinstance(grouped_source, torch.fx.Node) or not isinstance(
            value, torch.fx.Node
        ):
            return None
        normalized = self.graph.normalized_nodes.get(grouped_source)
        if not isinstance(normalized, NormalizedView):
            return None
        source_node = normalized.source
        layout = grouped_tensor_layout(normalized.shape, tensor_meta_shape(source_node))
        if layout is None:
            return None
        if layout.axis != 0:
            if not self.feed_main_grouped_reduction(value, grouped_source, layout):
                return None
            if layout.group <= LOCAL_REDUCE_FRAGMENT_WIDTH:
                return self.match_feed_value(value, grouped_source, layout)
            raise NotImplementedError(LOCAL_REDUCE_FEED_MAIN_AXIS1_FRAGMENT_ERROR)
        if layout.group > LOCAL_REDUCE_FRAGMENT_WIDTH:
            raise NotImplementedError(LOCAL_REDUCE_FEED_MAIN_SAME_WARP_ERROR)
        source_meta = source_node.meta.get("val")
        if (
            output_meta is not None
            and source_meta is not None
            and not statically_known_shape_equal(output_meta.shape, source_meta.shape)
        ):
            return None
        return self.match_feed_value(value, grouped_source, layout)

    def match_feed_main_source(
        self,
        source: torch.fx.Node,
        output_meta: Any,
    ) -> GemmLocalReduceMatch | None:
        """Find one physical feed-main value inside a pointwise expression."""
        matches = [
            match
            for grouped_source, value in self.feed_main_binary_candidates(source)
            if (
                match := self.match_feed_main_candidate(
                    grouped_source, value, output_meta
                )
            )
            is not None
        ]
        if matches:
            return self.validate_feed_main_source_match(
                source,
                GemmLocalReduceMatch.common_value(
                    matches, LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR
                ),
            )
        if not is_shape_preserving_pointwise_node(source):
            return None
        matches = [
            match
            for arg in iter_fx_node_inputs((source.args, source.kwargs))
            if isinstance(arg, torch.fx.Node)
            if (match := self.match_feed_main_source(arg, output_meta)) is not None
        ]
        return self.validate_feed_main_source_match(
            source,
            GemmLocalReduceMatch.common_value(
                matches, LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR
            ),
        )

    def feed_main_plan(
        self,
        output: torch.fx.Node,
    ) -> GemmLocalReduceMatch | None:
        """Match feed-main reductions through trailing pointwise nodes."""
        normalized = self.graph.normalized_nodes.get(output)
        if isinstance(normalized, NormalizedView):
            return self.match_feed_main_source(
                normalized.source, output.meta.get("val")
            )
        if not is_shape_preserving_pointwise_node(output):
            return None
        matches = [
            match
            for arg in iter_fx_node_inputs((output.args, output.kwargs))
            if isinstance(arg, torch.fx.Node)
            if (match := self.feed_main_plan(arg)) is not None
        ]
        return self.validate_feed_main_source_match(
            output,
            GemmLocalReduceMatch.common_value(
                matches, LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR
            ),
        )

    def common_feed_main_match(
        self,
        candidates: tuple[Any, ...],
    ) -> GemmLocalReduceMatch | None:
        """Find the physical reduction value shared by feed-main consumers."""
        matches = [
            match
            for candidate in candidates
            if isinstance(candidate, torch.fx.Node)
            if (match := self.feed_main_plan(candidate)) is not None
        ]
        return GemmLocalReduceMatch.common_value(
            matches, LOCAL_REDUCE_FEED_MAIN_MIXED_MATCH_ERROR
        )

    def compressed_aux_plan(
        self,
        output: Any,
        aux: torch.fx.Node,
    ) -> GemmOutputLocalReducePlan | None:
        """Plan a matched reduction returned through one physical layout."""
        output_storage = match_flex_gemm_local_reduce_output_storage(aux)
        value_node = aux if output_storage is None else output_storage.source
        match = self.matches.get(value_node) or self.feed_main_plan(value_node)
        output_meta = (
            output.meta.get("val") if isinstance(output, torch.fx.Node) else None
        )
        value_meta = value_node.meta.get("val")
        aux_meta = aux.meta.get("val")
        if (
            match is None
            or value_meta is None
            or aux_meta is None
            or output_meta is None
        ):
            return None
        expected_aux_shape = local_reduce_compressed_shape(
            self.gemm_shape or output_meta.shape,
            match.geometry.group,
            match.geometry.axis,
        )
        if not statically_known_shape_equal(expected_aux_shape, value_meta.shape):
            return None
        if output_storage is not None:
            output_storage.layout.validate_geometry(match.geometry)
        return match.to_plan(
            store=GemmLocalReduceStore(aux, output_storage),
            feeds_main=False,
        )

    def feed_main_output_plan(
        self,
        output: torch.fx.Node,
        aux_outputs: tuple[torch.fx.Node, ...] = (),
    ) -> GemmOutputPlan | None:
        """Plan one physical reduction value consumed by the main output."""
        match = self.common_feed_main_match((output, *aux_outputs))
        if match is None:
            return None
        return GemmOutputPlan(
            output,
            aux_outputs,
            match.to_plan(store=None, feeds_main=True),
        )


@dataclasses.dataclass(frozen=True)
class OutputContractionUse:
    """Describe one grouped-main lane before complete-output validation."""

    source: torch.fx.Node
    group: int
    chunked: bool
    index: int
    layout_node: torch.fx.Node
    structural_values: tuple[FlexGemmStructuralInt, ...] = ()


@dataclasses.dataclass(frozen=True)
class OutputContractionPlan:
    """Describe one complete grouped-main output and its lowering metadata."""

    transform: FlexGemmOutputContraction
    select_indices: dict[torch.fx.Node, int]
    layouts: dict[torch.fx.Node, GemmReductionGeometry]
    structural_values: tuple[FlexGemmStructuralInt, ...] = ()

    def commit_guards(self) -> None:
        """Specialize backed structural values after complete validation."""
        for structural in self.structural_values:
            structural.guard()


def canonical_output_contraction_source(
    node: torch.fx.Node,
    gemm: torch.fx.Node,
    local_reduce: GemmLocalReduceAnalysis,
) -> torch.fx.Node:
    """Strip shape-preserving pointwise wrappers from a grouped lane source."""
    while node is not gemm and is_shape_preserving_pointwise_node(node):
        inputs = [
            arg
            for arg in iter_fx_node_inputs((node.args, node.kwargs))
            if local_reduce.graph.depends_on(arg, gemm)
        ]
        if len(inputs) != 1:
            break
        node = inputs[0]
    return node


def match_output_contraction_use(
    node: torch.fx.Node,
    gemm: torch.fx.Node,
    local_reduce: GemmLocalReduceAnalysis,
) -> OutputContractionUse | None:
    """Match one interleaved select or contiguous split lane."""
    normalized = local_reduce.graph.normalized_nodes.get(node)
    if isinstance(normalized, NormalizedSelect):
        view = normalized.source
        view_normalized = local_reduce.graph.normalized_nodes.get(view)
        dim = FlexGemmStructuralInt.from_value(normalized.dim)
        index = FlexGemmStructuralInt.from_value(normalized.index)
        shape = tensor_meta_shape(view)
        if (
            not isinstance(view_normalized, NormalizedView)
            or shape is None
            or len(shape) != 3
            or dim is None
            or index is None
            or not local_reduce.graph.depends_on(view_normalized.source, gemm)
        ):
            return None
        selected_dim = dim.value % len(shape)
        structural_values = [dim, index]
        if selected_dim == len(shape) - 1:
            layout = local_reduce.grouped_tensors.get(view)
            if layout is None or layout.axis != 1:
                return None
            group, chunked = layout.group, False
        elif selected_dim == 1:
            structural_group = FlexGemmStructuralInt.from_value(shape[1])
            source_shape = tensor_meta_shape(view_normalized.source)
            if (
                structural_group is None
                or structural_group.value <= 1
                or source_shape is None
                or not statically_known_shape_equal(
                    (shape[0], structural_group.value * shape[2]), source_shape
                )
            ):
                return None
            group = structural_group.value
            structural_values.append(structural_group)
            chunked = True
        else:
            return None
        return OutputContractionUse(
            view_normalized.source,
            group,
            chunked,
            index.value,
            view,
            tuple(structural_values),
        )

    if not isinstance(normalized, NormalizedGetItem):
        return None
    split = normalized.source
    split_normalized = local_reduce.graph.normalized_nodes.get(split)
    if not isinstance(split_normalized, NormalizedSplit):
        return None
    index = FlexGemmStructuralInt.from_value(normalized.index)
    source = split_normalized.source
    split_size = FlexGemmStructuralInt.from_value(split_normalized.split_size)
    shape = tensor_meta_shape(source)
    if (
        index is None
        or shape is None
        or len(shape) != 2
        or not isinstance(shape[-1], int)
        or split_size is None
        or split_size.value <= 0
        or shape[-1] % split_size.value
        or split_normalized.dim not in (-1, 1)
        or not local_reduce.graph.depends_on(source, gemm)
    ):
        return None
    group = shape[-1] // split_size.value
    if group <= 1:
        return None
    return OutputContractionUse(
        source,
        group,
        True,
        index.value,
        split,
        (index, split_size),
    )


def build_output_contraction_plan(
    output: torch.fx.Node,
    gemm: torch.fx.Node,
    local_reduce: GemmLocalReduceAnalysis,
) -> OutputContractionPlan | None:
    """Recognize a complete adjacent-N grouped main-output expression."""
    lanes: list[tuple[torch.fx.Node, OutputContractionUse]] = []
    seen: OrderedSet[torch.fx.Node] = OrderedSet()
    pending: list[Any] = [output]
    while pending:
        node = pending.pop()
        if not isinstance(node, torch.fx.Node) or node in seen:
            continue
        seen.add(node)
        match = match_output_contraction_use(node, gemm, local_reduce)
        if match is not None:
            lanes.append((node, match))
            continue
        if node is gemm or (
            node in local_reduce.grouped_tensors
            and local_reduce.graph.depends_on(node, gemm)
        ):
            return None
        pending.extend(reversed(tuple(iter_fx_node_inputs((node.args, node.kwargs)))))
    if not lanes:
        return None

    first = lanes[0][1]
    canonical_source = canonical_output_contraction_source(
        first.source, gemm, local_reduce
    )
    indices: OrderedSet[int] = OrderedSet()
    select_indices: dict[torch.fx.Node, int] = {}
    layouts: dict[torch.fx.Node, GemmReductionGeometry] = {}
    structural_values: list[FlexGemmStructuralInt] = []
    for node, match in lanes:
        if (
            canonical_output_contraction_source(match.source, gemm, local_reduce)
            is not canonical_source
            or match.group != first.group
            or match.chunked != first.chunked
            or not -first.group <= match.index < first.group
        ):
            return None
        index = match.index % first.group
        indices.add(index)
        select_indices[node] = index
        layouts[match.layout_node] = GemmReductionGeometry(first.group, 1)
        structural_values.extend(match.structural_values)
    if indices != OrderedSet(range(first.group)):
        return None

    gemm_meta = gemm.meta.get("val")
    output_meta = output.meta.get("val")
    if not isinstance(gemm_meta, torch.Tensor) or not isinstance(
        output_meta, torch.Tensor
    ):
        return None
    expected_shape = (gemm_meta.shape[0], gemm_meta.shape[1] // first.group)
    if not statically_known_shape_equal(output_meta.shape, expected_shape):
        raise NotImplementedError(FLEX_GEMM_OUTPUT_CONTRACTION_SHAPE_ERROR)
    return OutputContractionPlan(
        FlexGemmOutputContraction(first.group, first.chunked),
        select_indices,
        layouts,
        tuple(structural_values),
    )
