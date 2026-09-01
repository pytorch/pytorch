# mypy: allow-untyped-defs
"""Shared FX analysis and output planning for grouped GEMM epilogues."""

import dataclasses
from collections.abc import Sequence
from typing import Any

import torch
from torch._inductor.kernel.flex_gemm.constraints import (
    FLEX_GEMM_OUTPUT_PLAN_NODE_ERROR,
    LOCAL_REDUCE_EXPLICIT_DTYPE_ERROR,
    LOCAL_REDUCE_FEED_MAIN_AXIS1_FRAGMENT_ERROR,
    LOCAL_REDUCE_FEED_MAIN_MIXED_MATCH_ERROR,
    LOCAL_REDUCE_GROUPED_RESHAPE_ERROR,
    LOCAL_REDUCE_INNERMOST_GROUPED_DIM_ERROR,
    LOCAL_REDUCE_MATCH_NODE_ERROR,
    LOCAL_REDUCE_MIXED_GROUPED_LAYOUT_ERROR,
    LOCAL_REDUCE_MIXED_MATCH_ERROR,
    LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR,
    LOCAL_REDUCE_OUTPUT_PLAN_NODE_ERROR,
    LOCAL_REDUCE_SOURCE_EXPRESSION_ERROR,
    local_reduce_unsupported_tensorssa_error,
    validate_local_reduce_feed_main_capability,
    validate_local_reduce_tensorssa_group_size,
)
from torch._inductor.kernel.flex_gemm.quack_reductions import (
    is_shape_preserving_pointwise_node,
    tensor_meta_shape,
)
from torch._inductor.kernel.gemm_epilogue import (
    GEMM_REDUCTION_IDENTITY_SOURCE,
    GemmEpilogueGraph,
    GemmReductionGeometry,
    GemmReductionPlan,
    GemmReductionType,
    iter_fx_node_inputs,
    NormalizedGetItem,
    NormalizedPrepareSoftmax,
    NormalizedReduction,
    NormalizedSqueeze,
    NormalizedUnsupportedReduction,
    NormalizedView,
)
from torch._inductor.kernel.gemm_epilogue_utils import (
    guarded_int,
    normalize_shape,
    statically_known_equal,
    statically_known_shape_equal,
)
from torch.utils._ordered_set import OrderedSet


def _is_inferred_reshape_dim(value: object) -> bool:
    """Return whether a reshape dimension is the literal inferred-size marker."""
    return isinstance(value, int) and value == -1


def _kept_dim_matches_source(kept_size: object, source_size: object) -> bool:
    return _is_inferred_reshape_dim(kept_size) or statically_known_equal(
        kept_size, source_size
    )


def _guard_grouped_reshape_group(
    shape: tuple[Any, ...], source_shape: tuple[Any, ...]
) -> tuple[Any, ...]:
    """Specialize a backed group dimension used to recognize a grouped reshape."""
    if len(shape) != 3:
        return shape
    for group_index, kept_index in ((-1, 0), (-2, -1)):
        if not _kept_dim_matches_source(shape[kept_index], source_shape[kept_index]):
            continue
        group_value = shape[group_index]
        symbolic = (
            group_value.meta.get("val")
            if isinstance(group_value, torch.fx.Node)
            else group_value
        )
        if not isinstance(symbolic, torch.SymInt):
            continue
        group = guarded_int(group_value)
        if group is not None:
            result = list(shape)
            result[group_index] = group
            return tuple(result)
    return shape


def _syntactic_grouped_tensor_layout(
    shape: tuple[Any, ...],
) -> GemmReductionGeometry | None:
    """Match grouped-reshape syntax before validating source geometry."""
    if len(shape) not in (3, 4):
        return None
    if (
        isinstance(shape[-1], int)
        and shape[-1] > 0
        and _is_inferred_reshape_dim(shape[-2])
    ):
        return GemmReductionGeometry(group=shape[-1], axis=1)
    if (
        _is_inferred_reshape_dim(shape[-3])
        and isinstance(shape[-2], int)
        and shape[-2] > 0
    ):
        return GemmReductionGeometry(group=shape[-2], axis=0)
    return None


def _group_count_matches_selected_dim(
    group_count: Any,
    selected_size: Any,
    group: int,
    kept_size: Any,
) -> bool:
    """Match a group count, allowing -1 to infer the selected source dimension."""
    if _is_inferred_reshape_dim(group_count):
        return True
    return statically_known_equal(group_count * group, selected_size) or (
        not _is_inferred_reshape_dim(kept_size)
        and statically_known_equal(group_count, selected_size // group)
    )


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
        case 1, (kept_m, group_count, group) if group == layout.group:
            return _kept_dim_matches_source(
                kept_m, m
            ) and _group_count_matches_selected_dim(group_count, n, group, kept_m)
        case 0, (group_count, group, kept_n) if group == layout.group:
            return _kept_dim_matches_source(
                kept_n, n
            ) and _group_count_matches_selected_dim(group_count, m, group, kept_n)
        case _:
            return False


def grouped_tensor_layout(
    shape: object, source_shape: object | None = None
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
            shape = _guard_grouped_reshape_group(shape, source_shape)
            candidates = []
            match shape:
                case (*_, int(group)) if group > 0:
                    candidates.append(GemmReductionGeometry(group=group, axis=1))
            match shape:
                case (*_, int(group), _) if group > 0:
                    candidates.append(GemmReductionGeometry(group=group, axis=0))
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


@dataclasses.dataclass(frozen=True, kw_only=True)
class GemmLocalReduceMatch:
    """Describe a supported grouped local-reduction value found in the FX graph.

    Attributes:
        value_node: FX node that produces the matched local-reduction value.
        geometry: Group size and GEMM output axis reduced by the value.
        reduction_type: Primitive reduction represented by the generated epilogue.
    """

    value_node: torch.fx.Node
    geometry: GemmReductionGeometry
    reduction_node: torch.fx.Node | None = None
    reduction_type: GemmReductionType = "sum"

    def __post_init__(self) -> None:
        if not isinstance(self.value_node, torch.fx.Node):
            raise RuntimeError(LOCAL_REDUCE_MATCH_NODE_ERROR)
        if self.reduction_node is None:
            object.__setattr__(self, "reduction_node", self.value_node)

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
class GemmLocalReduceStore:
    """Describe where a compressed local reduction appears in graph outputs.

    Attributes:
        node: FX node returned as the compressed local-reduction output.
        aux_index: Position of that node among the graph's auxiliary outputs.
    """

    node: torch.fx.Node
    aux_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.node, torch.fx.Node) or self.aux_index < 0:
            raise RuntimeError(LOCAL_REDUCE_OUTPUT_PLAN_NODE_ERROR)


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

    @property
    def needs_physical_callbacks(self) -> bool:
        return self.match.geometry.needs_physical_callbacks


@dataclasses.dataclass(frozen=True)
class GemmOutputPlan:
    """Classify the values returned by a FlexGEMM body.

    Attributes:
        output: FX node returned as the main GEMM result.
        aux_outputs: Same-shape auxiliary FX outputs returned after the main result.
        local_reduce: Compressed or feed-main local-reduction output behavior.
    """

    output: torch.fx.Node
    aux_outputs: tuple[torch.fx.Node, ...] = ()
    local_reduce: GemmOutputLocalReducePlan | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.output, torch.fx.Node) or not all(
            isinstance(aux_output, torch.fx.Node) for aux_output in self.aux_outputs
        ):
            raise RuntimeError(FLEX_GEMM_OUTPUT_PLAN_NODE_ERROR)

    @property
    def reduction_plan(self) -> GemmReductionPlan | None:
        """Finalize FX ownership metadata into the shared reduction contract."""
        local_reduce = self.local_reduce
        if local_reduce is None:
            return None
        match = local_reduce.match
        reduction_output = (
            local_reduce.store.node.name if local_reduce.store is not None else None
        )
        return GemmReductionPlan(
            reduction_output=reduction_output,
            group=match.geometry.group,
            axis=match.geometry.axis,
            reduction_type=match.reduction_type,
            source_type="identity",
            source_fn=GEMM_REDUCTION_IDENTITY_SOURCE,
            primary_output=self.output.name,
            feeds_main=local_reduce.feeds_main,
            feed_output=self.output.name if local_reduce.feeds_main else None,
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
    matches: dict[torch.fx.Node, GemmLocalReduceMatch] = dataclasses.field(
        default_factory=dict
    )

    @classmethod
    def from_graph_module(
        cls, graph_module: torch.fx.GraphModule
    ) -> "GemmLocalReduceAnalysis":
        """Build shared dependency and reduction state in one topological pass."""
        nodes = tuple(graph_module.graph.nodes)
        analysis = cls(GemmEpilogueGraph.from_nodes(nodes))
        for node in nodes:
            if node.op == "output":
                break
            analysis.visit_node(node)
        return analysis

    def visit_node(self, node: torch.fx.Node) -> None:
        """Record grouped layouts and local-reduction matches for one FX node."""
        if node.op != "call_function":
            return
        normalized = self.graph.normalized_nodes.get(node)
        if isinstance(normalized, NormalizedView):
            if self.propagate_local_reduce_match(node, normalized.source):
                return
            if self.bind_grouped_layout(node, normalized.shape, normalized.source):
                return
        elif isinstance(normalized, NormalizedReduction):
            if self.bind_grouped_reduction(
                node,
                normalized.source,
                normalized.dim,
                normalized.dtype,
                reduction_type=normalized.reduction_type,
            ):
                return
        elif isinstance(normalized, NormalizedPrepareSoftmax):
            if self.bind_grouped_reduction(
                node,
                normalized.source,
                normalized.dim,
                raise_invalid_dims=False,
            ):
                return
        elif isinstance(normalized, NormalizedUnsupportedReduction):
            if normalized.source in self.grouped_tensors:
                raise local_reduce_unsupported_tensorssa_error(normalized.target)
        elif isinstance(normalized, (NormalizedSqueeze, NormalizedGetItem)):
            if self.propagate_local_reduce_match(node, normalized.source):
                return
        if is_shape_preserving_pointwise_node(node):
            self.propagate_pointwise_match(node, LOCAL_REDUCE_MIXED_MATCH_ERROR)

    def bind_grouped_layout(self, node: torch.fx.Node, shape: Any, source: Any) -> bool:
        """Attach a grouped TensorSSA layout introduced by a reshape."""
        source_shape = (
            tensor_meta_shape(source) if isinstance(source, torch.fx.Node) else None
        )
        grouped_layout = grouped_tensor_layout(shape, source_shape)
        if grouped_layout is None or not isinstance(source, torch.fx.Node):
            return False
        self.grouped_tensors[node] = grouped_layout
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
        input_node: Any,
        dim: Any,
        dtype: Any = None,
        *,
        reduction_type: GemmReductionType = "sum",
        raise_invalid_dims: bool = True,
    ) -> bool:
        """Match and record a reduction over a grouped TensorSSA layout."""
        if not isinstance(input_node, torch.fx.Node):
            return False
        layout = self.grouped_tensors.get(input_node)
        if layout is None:
            return False
        if dtype is not None:
            raise NotImplementedError(LOCAL_REDUCE_EXPLICIT_DTYPE_ERROR)
        validate_local_reduce_tensorssa_group_size(layout.axis, layout.group)
        if not layout.matches_reduction_dim(dim):
            if not raise_invalid_dims:
                return False
            raise NotImplementedError(LOCAL_REDUCE_INNERMOST_GROUPED_DIM_ERROR)
        self.matches[node] = GemmLocalReduceMatch(
            value_node=node,
            geometry=layout,
            reduction_node=node,
            reduction_type=reduction_type,
        )
        return True

    def has_physical_grouped_input(self, value: Any) -> bool:
        """Return whether a value depends on a grouped layout needing callbacks."""
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

    def match_feed_value(
        self,
        value: Any,
        grouped_source: torch.fx.Node,
        layout: GemmReductionGeometry,
    ) -> GemmLocalReduceMatch | None:
        """Find the grouped reduction that produces a broadcast value."""
        if not isinstance(value, torch.fx.Node):
            return None
        normalized = self.graph.normalized_nodes.get(value)
        if isinstance(normalized, NormalizedReduction):
            if normalized.source is not grouped_source:
                if self.graph.depends_on(normalized.source, grouped_source):
                    raise NotImplementedError(LOCAL_REDUCE_SOURCE_EXPRESSION_ERROR)
                raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
            if (
                normalized.dtype is not None
                or not normalized.keepdim
                or not layout.matches_reduction_dim(normalized.dim)
            ):
                raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
            return GemmLocalReduceMatch(
                value_node=value,
                geometry=layout,
                reduction_node=value,
                reduction_type=normalized.reduction_type,
            )
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

    def validate_hidden_feed_main_reduction_input(
        self,
        input_node: Any,
        grouped_source: torch.fx.Node,
    ) -> None:
        """Reject reduction inputs that would need another physical feed-main value."""
        if input_node is grouped_source:
            raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
        if not isinstance(input_node, torch.fx.Node):
            return
        if self.graph.depends_on(input_node, grouped_source):
            raise NotImplementedError(LOCAL_REDUCE_SOURCE_EXPRESSION_ERROR)
        if self.has_physical_grouped_input(input_node):
            raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)

    def validate_feed_main_source_reductions(
        self,
        value: Any,
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
        """Return whether a candidate contains a grouped feed-main reduction."""
        if not isinstance(value, torch.fx.Node):
            return False
        normalized = self.graph.normalized_nodes.get(value)
        if isinstance(normalized, NormalizedReduction):
            return (
                normalized.dtype is None
                and bool(normalized.keepdim)
                and layout.matches_reduction_dim(normalized.dim)
                and (
                    normalized.source is grouped_source
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
        layout = self.grouped_tensors.get(grouped_source)
        if layout is None:
            return None
        if layout.axis != 0:
            if not self.feed_main_grouped_reduction(value, grouped_source, layout):
                return None
            if not layout.needs_physical_callbacks:
                # Intentional fallthrough: axis-1 feeds within one TensorSSA
                # fragment lower as plain generated TensorSSA without a feed plan.
                return None
            raise NotImplementedError(LOCAL_REDUCE_FEED_MAIN_AXIS1_FRAGMENT_ERROR)
        validate_local_reduce_feed_main_capability(layout.axis, layout.group)
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

    def common_reduction_dependency_match(
        self, outputs: Sequence[torch.fx.Node]
    ) -> GemmLocalReduceMatch | None:
        """Return one physical reduction transitively consumed by the outputs."""
        reachable = OrderedSet(outputs)
        for output in outputs:
            reachable.update(self.graph.dependencies.get(output, ()))
        matches = [self.matches[node] for node in reachable if node in self.matches]
        if not matches:
            return None
        reduction_node = matches[0].reduction_node
        if reduction_node is None or any(
            match.reduction_node is not reduction_node for match in matches
        ):
            return None
        return matches[0]

    def feed_main_output_plan(
        self,
        output: torch.fx.Node,
        aux_outputs: tuple[torch.fx.Node, ...] = (),
        *,
        allow_dependency_match: bool = False,
    ) -> GemmOutputPlan | None:
        """Plan one physical reduction value consumed by the main output."""
        match = self.common_feed_main_match((output, *aux_outputs))
        if match is None and allow_dependency_match:
            match = self.common_reduction_dependency_match((output, *aux_outputs))
        if match is None:
            return None
        return GemmOutputPlan(
            output,
            aux_outputs,
            match.to_plan(store=None, feeds_main=True),
        )
