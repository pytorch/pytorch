# mypy: allow-untyped-defs
"""Analyze FlexGEMM epilogue FX graphs and materialize CuTeDSL source.

``analyze_flex_gemm_epilogue`` indexes FX dependencies, identifies nodes that
carry grouped TensorSSA layouts, matches supported local reductions, and plans
the main, auxiliary, and local-reduction consumers.

``materialize_flex_gemm_epilogue`` uses that analysis to generate the CuTeDSL
epilogue and physical reduction callbacks. Selected FX nodes are normalized once
during analysis; materialization consumes the same canonical arguments while
ordinary pointwise nodes remain on the open-ended ops-handler path.
"""

import dataclasses
import hashlib
from typing import Any

import torch
from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
    CuteDSLCSEVariable,
    CuteDSLOpOverrides,
    upcast_compute_type,
    use_cutedsl_fast_math,
)
from torch._inductor.kernel.flex_gemm.constraints import (
    FLEX_GEMM_GROUPED_MAIN_COMPOSITION_ERROR,
    FLEX_GEMM_GROUPED_MAIN_SHAPE_ERROR,
    FLEX_GEMM_MAIN_OUTPUT_SHAPE_ERROR,
    FLEX_GEMM_OUTPUT_PLAN_NODE_ERROR,
    FLEX_GEMM_OUTPUT_TENSOR_ERROR,
    FlexGemmGroupedMainOutputTransform,
    FlexGemmLocalReduceGeometry,
    LOCAL_REDUCE_AUX_TENSORSSA_ERROR,
    LOCAL_REDUCE_COMBINE_FN_SUFFIX,
    local_reduce_compressed_shape,
    LOCAL_REDUCE_EXPLICIT_DTYPE_ERROR,
    LOCAL_REDUCE_FEED_MAIN_ARG_NAME,
    LOCAL_REDUCE_FEED_MAIN_AXIS1_FRAGMENT_ERROR,
    LOCAL_REDUCE_FEED_MAIN_MIXED_MATCH_ERROR,
    LOCAL_REDUCE_FINALIZE_FN_SUFFIX,
    LOCAL_REDUCE_FINALIZE_SCALAR_ONLY_ERROR,
    LOCAL_REDUCE_FRAGMENT_WIDTH,
    LOCAL_REDUCE_INNERMOST_GROUPED_DIM_ERROR,
    LOCAL_REDUCE_MATCH_NODE_ERROR,
    LOCAL_REDUCE_MIXED_GROUPED_LAYOUT_ERROR,
    LOCAL_REDUCE_MIXED_MATCH_ERROR,
    LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR,
    LOCAL_REDUCE_OUTPUT_PLAN_NODE_ERROR,
    LOCAL_REDUCE_POST_POINTWISE_FINALIZE_ERROR,
    LOCAL_REDUCE_SINGLE_PHYSICAL_FINALIZE_ERROR,
    LOCAL_REDUCE_SOURCE_EXPRESSION_ERROR,
    local_reduce_unsupported_tensorssa_error,
    statically_known_shape_equal,
    validate_local_reduce_feed_main_capability,
    validate_local_reduce_tensorssa_group_size,
)
from torch._inductor.kernel.flex_gemm.epilogue_nodes import (
    normalize_flex_gemm_epilogue_fx_node,
    NormalizedGetItem,
    NormalizedNode,
    NormalizedPrepareSoftmax,
    NormalizedReduction,
    NormalizedSelect,
    NormalizedSplit,
    NormalizedSqueeze,
    NormalizedUnsupportedReduction,
    NormalizedView,
)
from torch._inductor.kernel.flex_gemm.quack_reductions import (
    _cute_arg,
    _cute_call,
    _local_reduce_store_arg,
    FlexGemmGroupedLayoutMatch,
    FlexGemmPhysicalReduction,
    FlexGemmStructuralInt,
    grouped_tensor_layout,
    is_shape_preserving_pointwise_node,
    iter_fx_node_inputs,
    lower_full_scalar,
    lower_getitem,
    lower_grouped_n_select,
    lower_grouped_n_split,
    lower_prepare_softmax_online,
    lower_squeeze,
    lower_tensorssa_reduce,
    lower_view_or_reshape,
    tensor_meta_shape,
)
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.value_ranges import ValueRanges


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


class FlexGemmCuteDSLBody:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def writeline(self, line: str) -> None:
        self.lines.append(line)


class FlexGemmCuteDSLCSE:
    def __init__(self) -> None:
        self.index = 0

    def generate(self, body, expr, *, bounds=None, dtype=None, shape=None):
        name = f"tmp{self.index}"
        self.index += 1
        body.writeline(f"{name} = {expr}")
        return CuteDSLCSEVariable(
            name,
            ValueRanges.unknown() if bounds is None else bounds,
            dtype=dtype,
            shape=shape,
        )


class FlexGemmCuteDSLKernel:
    def __init__(self) -> None:
        self.body = FlexGemmCuteDSLBody()
        self.cse = FlexGemmCuteDSLCSE()


class FlexGemmCuteDSLOpOverrides(CuteDSLOpOverrides):
    # Aten add/sub carry alpha as schema sugar; CuTeDSL only needs the scaled RHS.
    @staticmethod
    def add(a: Any, b: Any, *, alpha: Any = 1) -> Any:
        rhs = b if alpha == 1 else CuteDSLOpOverrides.mul(b, alpha)
        return CuteDSLOpOverrides.add(a, rhs)

    @staticmethod
    def sub(a: Any, b: Any, *, alpha: Any = 1) -> Any:
        rhs = b if alpha == 1 else CuteDSLOpOverrides.mul(b, alpha)
        return CuteDSLOpOverrides.sub(a, rhs)

    @staticmethod
    def _to_copy(x: Any, *, dtype: torch.dtype, **kwargs: Any) -> Any:
        unsupported_kwargs = {
            key: value
            for key, value in kwargs.items()
            if value not in (None, False, torch.preserve_format)
        }
        if unsupported_kwargs:
            raise NotImplementedError(
                "unsupported kwargs for FlexGEMM epilogue op _to_copy: "
                f"{unsupported_kwargs}"
            )
        return CuteDSLOpOverrides.to_dtype(x, dtype)

    @staticmethod
    def clamp(x: Any, min: Any = None, max: Any = None) -> Any:
        result = x
        if min is not None:
            result = CuteDSLOpOverrides.maximum(result, min)
        if max is not None:
            result = CuteDSLOpOverrides.minimum(result, max)
        return result

    @staticmethod
    def clamp_min(x: Any, min: Any) -> Any:
        return CuteDSLOpOverrides.maximum(x, min)

    @staticmethod
    def clamp_max(x: Any, max: Any) -> Any:
        return CuteDSLOpOverrides.minimum(x, max)

    @staticmethod
    def convert_element_type(x: Any, dtype: torch.dtype) -> Any:
        return CuteDSLOpOverrides.to_dtype(x, dtype)


@dataclasses.dataclass(frozen=True)
class FlexGemmLocalReduceMatch:
    """Describe a supported grouped local-reduction value found in the FX graph.

    Attributes:
        value_node: FX node that produces the matched local-reduction value.
        geometry: Group size and GEMM output axis reduced by the value.
        structural_values: Backed shape values guarded after analysis accepts the graph.
    """

    value_node: torch.fx.Node
    geometry: FlexGemmLocalReduceGeometry
    structural_values: tuple[FlexGemmStructuralInt, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.value_node, torch.fx.Node):
            raise RuntimeError(LOCAL_REDUCE_MATCH_NODE_ERROR)

    def commit_guards(self) -> None:
        """Install structural guards after the epilogue analysis is accepted."""
        for structural in self.structural_values:
            structural.guard()

    def to_plan(
        self,
        *,
        store: "FlexGemmLocalReduceStore | None",
        feeds_main: bool,
    ) -> "FlexGemmOutputLocalReducePlan":
        """Bind this matched value to its output consumers."""
        return FlexGemmOutputLocalReducePlan(self, store=store, feeds_main=feeds_main)

    @classmethod
    def common(
        cls,
        matches: list["FlexGemmLocalReduceMatch"],
        mixed_match_error: str,
    ) -> "FlexGemmLocalReduceMatch | None":
        """Return the common match when all values use one reduction geometry."""
        if not matches:
            return None
        match = matches[0]
        if any(item.geometry != match.geometry for item in matches):
            raise NotImplementedError(mixed_match_error)
        return dataclasses.replace(
            match,
            structural_values=tuple(
                value for item in matches for value in item.structural_values
            ),
        )

    @classmethod
    def common_value(
        cls,
        matches: list["FlexGemmLocalReduceMatch"],
        mixed_match_error: str,
    ) -> "FlexGemmLocalReduceMatch | None":
        """Return the common match when all consumers use one physical value."""
        match = cls.common(matches, mixed_match_error)
        if match is None:
            return None
        if any(item.value_node is not match.value_node for item in matches):
            raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
        return match


@dataclasses.dataclass(frozen=True)
class FlexGemmLocalReduceStore:
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
class FlexGemmOutputLocalReducePlan:
    """Bind a matched local reduction to store and/or main-output consumers.

    Attributes:
        match: Supported local-reduction value identified during FX analysis.
        store: Compressed auxiliary output receiving the value, when requested.
        feeds_main: Whether the reduced value is also consumed by the main output.
    """

    match: FlexGemmLocalReduceMatch
    store: FlexGemmLocalReduceStore | None = None
    feeds_main: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.match, FlexGemmLocalReduceMatch) or (
            self.store is None and not self.feeds_main
        ):
            raise RuntimeError(LOCAL_REDUCE_OUTPUT_PLAN_NODE_ERROR)

    @property
    def needs_physical_callbacks(self) -> bool:
        return self.match.geometry.needs_physical_callbacks


@dataclasses.dataclass(frozen=True)
class FlexGemmOutputPlan:
    """Classify the values returned by a FlexGEMM body."""

    main: torch.fx.Node
    aux_outputs: tuple[torch.fx.Node, ...] = ()
    local_reduce: FlexGemmOutputLocalReducePlan | None = None
    main_transform: FlexGemmGroupedMainOutputTransform | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.main, torch.fx.Node) or not all(
            isinstance(aux_output, torch.fx.Node) for aux_output in self.aux_outputs
        ):
            raise RuntimeError(FLEX_GEMM_OUTPUT_PLAN_NODE_ERROR)


@dataclasses.dataclass(frozen=True)
class FlexGemmEpilogueGraph:
    """Index dependencies and normalized nodes alongside the original FX graph.

    Attributes:
        dependencies: Every FX node mapped to all of its direct and transitive
            input nodes.
        normalized_nodes: Selected FX nodes mapped to canonical arguments shared
            by semantic analysis and emission.
    """

    dependencies: dict[torch.fx.Node, frozenset[torch.fx.Node]]
    normalized_nodes: dict[torch.fx.Node, NormalizedNode]

    @classmethod
    def from_graph_module(
        cls, graph_module: torch.fx.GraphModule
    ) -> "FlexGemmEpilogueGraph":
        """Build transitive dependencies in the graph's topological order."""
        dependencies: dict[torch.fx.Node, frozenset[torch.fx.Node]] = {}
        normalized_nodes: dict[torch.fx.Node, NormalizedNode] = {}
        for node in graph_module.graph.nodes:
            node_dependencies: OrderedSet[torch.fx.Node] = OrderedSet()
            for input_node in iter_fx_node_inputs((node.args, node.kwargs)):
                node_dependencies.add(input_node)
                node_dependencies.update(dependencies.get(input_node, ()))
            dependencies[node] = frozenset(node_dependencies)
            if (normalized := normalize_flex_gemm_epilogue_fx_node(node)) is not None:
                normalized_nodes[node] = normalized
        return cls(dependencies, normalized_nodes)

    def depends_on(self, value: Any, target: torch.fx.Node) -> bool:
        """Return whether a value is or transitively depends on the target node."""
        return any(
            node is target or target in self.dependencies.get(node, ())
            for node in iter_fx_node_inputs(value)
        )


@dataclasses.dataclass
class FlexGemmLocalReduceAnalysis:
    """Collect grouped TensorSSA layouts and supported local-reduction matches.

    ``from_graph_module`` visits the FX graph in topological order. See
    ``FlexGemmLocalReduceGeometry`` for the grouped layout attached to reshape and
    pointwise nodes, and ``FlexGemmLocalReduceMatch`` for each supported reduced
    value found from those layouts.

    Attributes:
        graph: Dependency index used by recursive feed-main matching.
        grouped_layouts: Grouped TensorSSA layouts propagated through FX.
        matches: FX values matched to a supported grouped local reduction.
    """

    graph: FlexGemmEpilogueGraph
    grouped_layouts: dict[torch.fx.Node, FlexGemmGroupedLayoutMatch] = (
        dataclasses.field(default_factory=dict)
    )
    matches: dict[torch.fx.Node, FlexGemmLocalReduceMatch] = dataclasses.field(
        default_factory=dict
    )

    @classmethod
    def from_graph_module(
        cls, graph_module: torch.fx.GraphModule
    ) -> "FlexGemmLocalReduceAnalysis":
        """Build shared dependency and reduction state in one topological pass."""
        analysis = cls(FlexGemmEpilogueGraph.from_graph_module(graph_module))
        for node in graph_module.graph.nodes:
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
            if self.bind_grouped_layout(node, normalized):
                return
        elif isinstance(normalized, NormalizedReduction):
            if self.bind_grouped_reduction(
                node, normalized.source, normalized.dim, normalized.dtype
            ):
                return
        elif isinstance(normalized, NormalizedPrepareSoftmax):
            if self.bind_grouped_reduction(
                node, normalized.source, normalized.dim, raise_invalid_dims=False
            ):
                return
        elif isinstance(normalized, NormalizedUnsupportedReduction):
            if normalized.source in self.grouped_layouts:
                raise local_reduce_unsupported_tensorssa_error(normalized.target)
        elif isinstance(normalized, (NormalizedSqueeze, NormalizedGetItem)):
            if self.propagate_local_reduce_match(node, normalized.source):
                return
        if is_shape_preserving_pointwise_node(node):
            self.propagate_pointwise_match(node, LOCAL_REDUCE_MIXED_MATCH_ERROR)

    def bind_grouped_layout(
        self, node: torch.fx.Node, normalized: NormalizedView
    ) -> bool:
        """Record the TensorSSA layout when a view exposes grouped GEMM values."""
        grouped_layout = grouped_tensor_layout(
            normalized.shape, tensor_meta_shape(normalized.source)
        )
        if grouped_layout is None:
            return False
        self.grouped_layouts[node] = grouped_layout
        return True

    def propagate_local_reduce_match(
        self, node: torch.fx.Node, source: torch.fx.Node
    ) -> bool:
        """Copy a matched local-reduction value through an FX wrapper."""
        match = self.matches.get(source)
        if match is None:
            return False
        self.matches[node] = match
        return True

    def bind_grouped_reduction(
        self,
        node: torch.fx.Node,
        input_node: torch.fx.Node,
        dim: Any,
        dtype: Any = None,
        *,
        raise_invalid_dims: bool = True,
    ) -> bool:
        """Match and record a reduction over a grouped TensorSSA layout."""
        grouped_layout = self.grouped_layouts.get(input_node)
        if grouped_layout is None:
            return False
        layout = grouped_layout.layout
        if dtype is not None:
            raise NotImplementedError(LOCAL_REDUCE_EXPLICIT_DTYPE_ERROR)
        validate_local_reduce_tensorssa_group_size(layout.axis, layout.group)
        if not layout.matches_reduction_dim(dim):
            if not raise_invalid_dims:
                return False
            raise NotImplementedError(LOCAL_REDUCE_INNERMOST_GROUPED_DIM_ERROR)
        self.matches[node] = FlexGemmLocalReduceMatch(
            node, layout, grouped_layout.structural_values
        )
        return True

    def has_physical_grouped_input(self, value: Any) -> bool:
        """Return whether a value depends on a grouped layout needing callbacks."""
        active_geometries = OrderedSet(
            match.geometry for match in self.matches.values()
        )
        physical_grouped_nodes = OrderedSet(
            node
            for node, grouped_layout in self.grouped_layouts.items()
            if grouped_layout.layout.needs_physical_callbacks
            and grouped_layout.layout in active_geometries
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
            self.grouped_layouts[arg]
            for arg in iter_fx_node_inputs((node.args, node.kwargs))
            if arg in self.grouped_layouts
        ]
        if grouped_layouts:
            grouped_layout = grouped_layouts[0]
            if any(layout != grouped_layout for layout in grouped_layouts):
                raise NotImplementedError(LOCAL_REDUCE_MIXED_GROUPED_LAYOUT_ERROR)
            self.grouped_layouts[node] = grouped_layout
        match = FlexGemmLocalReduceMatch.common(
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
        layout: FlexGemmLocalReduceGeometry,
    ) -> FlexGemmLocalReduceMatch | None:
        """Find the grouped reduction that produces a broadcast value."""
        if not isinstance(value, torch.fx.Node):
            return None
        normalized = self.graph.normalized_nodes.get(value)
        if isinstance(normalized, NormalizedReduction):
            if normalized.source is not grouped_source:
                input_node = normalized.source
                if self.graph.depends_on(input_node, grouped_source):
                    raise NotImplementedError(LOCAL_REDUCE_SOURCE_EXPRESSION_ERROR)
                raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
            if (
                normalized.dtype is not None
                or not normalized.keepdim
                or not layout.matches_reduction_dim(normalized.dim)
            ):
                raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
            return FlexGemmLocalReduceMatch(
                value,
                layout,
                self.grouped_layouts[grouped_source].structural_values,
            )
        if not is_shape_preserving_pointwise_node(value):
            return None
        matches = [
            match
            for arg in iter_fx_node_inputs((value.args, value.kwargs))
            if (match := self.match_feed_value(arg, grouped_source, layout)) is not None
        ]
        return FlexGemmLocalReduceMatch.common_value(
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
        match: FlexGemmLocalReduceMatch | None,
    ) -> FlexGemmLocalReduceMatch | None:
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
        layout: FlexGemmLocalReduceGeometry,
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
    ) -> FlexGemmLocalReduceMatch | None:
        """Match one grouped-source and reduced-value operand ordering."""
        if not isinstance(grouped_source, torch.fx.Node) or not isinstance(
            value, torch.fx.Node
        ):
            return None
        normalized = self.graph.normalized_nodes.get(grouped_source)
        if not isinstance(normalized, NormalizedView):
            return None
        source_node = normalized.source
        grouped_layout = self.grouped_layouts.get(grouped_source)
        if grouped_layout is None:
            return None
        layout = grouped_layout.layout
        if layout.axis != 0:
            if not self.feed_main_grouped_reduction(value, grouped_source, layout):
                return None
            if layout.group <= LOCAL_REDUCE_FRAGMENT_WIDTH:
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
    ) -> FlexGemmLocalReduceMatch | None:
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
                FlexGemmLocalReduceMatch.common_value(
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
            FlexGemmLocalReduceMatch.common_value(
                matches, LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR
            ),
        )

    def feed_main_plan(
        self,
        output: torch.fx.Node,
    ) -> FlexGemmLocalReduceMatch | None:
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
            FlexGemmLocalReduceMatch.common_value(
                matches, LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR
            ),
        )

    def common_feed_main_match(
        self,
        candidates: tuple[Any, ...],
    ) -> FlexGemmLocalReduceMatch | None:
        """Find the physical reduction value shared by feed-main consumers."""
        matches = [
            match
            for candidate in candidates
            if isinstance(candidate, torch.fx.Node)
            if (match := self.feed_main_plan(candidate)) is not None
        ]
        return FlexGemmLocalReduceMatch.common_value(
            matches, LOCAL_REDUCE_FEED_MAIN_MIXED_MATCH_ERROR
        )

    def compressed_aux_plan(
        self,
        output: Any,
        aux: torch.fx.Node,
        aux_index: int,
    ) -> FlexGemmOutputLocalReducePlan | None:
        """Plan a matched local reduction returned in compressed output shape."""
        match = self.matches.get(aux)
        output_meta = (
            output.meta.get("val") if isinstance(output, torch.fx.Node) else None
        )
        aux_meta = aux.meta.get("val")
        if match is None or aux_meta is None or output_meta is None:
            return None
        expected_aux_shape = local_reduce_compressed_shape(
            output_meta.shape, match.geometry.group, match.geometry.axis
        )
        if not statically_known_shape_equal(expected_aux_shape, aux_meta.shape):
            return None
        return match.to_plan(
            store=FlexGemmLocalReduceStore(aux, aux_index), feeds_main=False
        )

    def feed_main_output_plan(
        self,
        output: torch.fx.Node,
        aux_outputs: tuple[torch.fx.Node, ...] = (),
    ) -> FlexGemmOutputPlan | None:
        """Plan one physical reduction value consumed by the main output."""
        match = self.common_feed_main_match((output, *aux_outputs))
        if match is None:
            return None
        return FlexGemmOutputPlan(
            output,
            aux_outputs,
            match.to_plan(store=None, feeds_main=True),
        )


def tuple_output_plan(
    output: Any,
    aux_outputs: tuple[Any, ...],
    analysis: FlexGemmLocalReduceAnalysis,
) -> FlexGemmOutputPlan:
    """Classify multi-output epilogues after checking local-reduce consumers."""
    if not isinstance(output, torch.fx.Node) or not all(
        isinstance(aux_output, torch.fx.Node) for aux_output in aux_outputs
    ):
        raise NotImplementedError(FLEX_GEMM_OUTPUT_TENSOR_ERROR)
    feed_match = analysis.common_feed_main_match((output, *aux_outputs))
    compressed_aux_plans = tuple(
        (index, match, plan)
        for index, aux_output in enumerate(aux_outputs)
        if (match := analysis.matches.get(aux_output)) is not None
        if (plan := analysis.compressed_aux_plan(output, aux_output, index)) is not None
    )
    if len(compressed_aux_plans) > 1:
        raise NotImplementedError(LOCAL_REDUCE_MIXED_MATCH_ERROR)
    if compressed_aux_plans:
        local_reduce_index, compressed_match, compressed_aux_plan = (
            compressed_aux_plans[0]
        )
        if feed_match is not None:
            if feed_match.value_node is not compressed_match.value_node:
                raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
            compressed_aux_plan = feed_match.to_plan(
                store=FlexGemmLocalReduceStore(
                    aux_outputs[local_reduce_index], local_reduce_index
                ),
                feeds_main=True,
            )
        return FlexGemmOutputPlan(
            output,
            tuple(
                aux_output
                for index, aux_output in enumerate(aux_outputs)
                if index != local_reduce_index
            ),
            local_reduce=compressed_aux_plan,
        )
    feed_main_plan = analysis.feed_main_output_plan(output, aux_outputs)
    if feed_main_plan is not None:
        return feed_main_plan
    return FlexGemmOutputPlan(output, aux_outputs)


def output_plan(
    graph_module: torch.fx.GraphModule,
    local_reduce: FlexGemmLocalReduceAnalysis,
) -> FlexGemmOutputPlan:
    """Classify output consumers from one shared local-reduce analysis."""
    output_nodes = [node for node in graph_module.graph.nodes if node.op == "output"]
    if len(output_nodes) != 1:
        raise NotImplementedError("FlexGEMM expects one output node")
    output_value = output_nodes[0].args[0]
    if isinstance(output_value, (tuple, list)):
        if len(output_value) == 1:
            output_value = output_value[0]
        else:
            output, *aux_outputs = output_value
            return tuple_output_plan(output, tuple(aux_outputs), local_reduce)
    if not isinstance(output_value, torch.fx.Node):
        raise NotImplementedError("FlexGEMM expects one tensor output")
    feed_main_plan = local_reduce.feed_main_output_plan(output_value)
    if feed_main_plan is not None:
        return feed_main_plan
    return FlexGemmOutputPlan(output_value)


@dataclasses.dataclass(frozen=True)
class GroupedMainLaneMatch:
    """Describe one grouped-main spelling normalized for semantic validation.

    Attributes:
        source: GEMM-derived tensor before the grouped spelling.
        group: Number of physical N values contracted into one logical output.
        chunked: Whether lanes occupy contiguous physical N chunks.
        indices: Lanes covered by this spelling before modulo normalization.
        layout_node: Split/view node registered after complete validation.
        structural_values: Symbolic values guarded only after acceptance.
    """

    source: torch.fx.Node
    group: int
    chunked: bool
    indices: tuple[int, ...]
    layout_node: torch.fx.Node | None = None
    structural_values: tuple[FlexGemmStructuralInt, ...] = ()


@dataclasses.dataclass(frozen=True)
class GroupedMainOutputMatch:
    """Hold one complete grouped-main match until analysis commits it."""

    transform: FlexGemmGroupedMainOutputTransform
    select_indices: dict[torch.fx.Node, int]
    grouped_layouts: dict[torch.fx.Node, FlexGemmGroupedLayoutMatch]
    structural_values: tuple[FlexGemmStructuralInt, ...]

    def commit_guards(self) -> None:
        """Install symbolic specializations after all composition checks pass."""
        for structural in self.structural_values:
            structural.guard()


def canonical_grouped_main_source(
    node: torch.fx.Node,
    gemm: torch.fx.Node,
    local_reduce: FlexGemmLocalReduceAnalysis,
) -> torch.fx.Node:
    """Strip shape-preserving pointwise wrappers from one grouped lane source."""
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


def match_grouped_main_lane(
    node: torch.fx.Node,
    gemm: torch.fx.Node,
    gemm_shape: tuple[Any, ...] | None,
    local_reduce: FlexGemmLocalReduceAnalysis,
) -> GroupedMainLaneMatch | None:
    """Normalize one supported grouped-N output spelling.

    The accepted leaves are ``split(...)[i]`` for chunked lanes,
    ``view(...).select(..., i)`` for interleaved or chunked lanes, and
    ``nvfp4_pack(view(...))`` for the two interleaved packed lanes.
    """
    if gemm_shape is None:
        return None
    normalized = local_reduce.graph.normalized_nodes.get(node)

    if isinstance(normalized, NormalizedGetItem):
        split = normalized.source
        split_normalized = local_reduce.graph.normalized_nodes.get(split)
        if not isinstance(split_normalized, NormalizedSplit):
            return None
        shape = tensor_meta_shape(split_normalized.source)
        if (
            shape is None
            or len(shape) != 2
            or not statically_known_shape_equal(shape, gemm_shape)
            or not local_reduce.graph.depends_on(split_normalized.source, gemm)
            or not isinstance(shape[-1], int)
            or split_normalized.dim not in (-1, 1)
        ):
            return None
        split_size = FlexGemmStructuralInt.from_value(split_normalized.split_size)
        if (
            split_size is None
            or split_size.value <= 0
            or shape[-1] % split_size.value != 0
        ):
            return None
        group = shape[-1] // split_size.value
        if group <= 1:
            return None
        return GroupedMainLaneMatch(
            source=split_normalized.source,
            group=group,
            chunked=True,
            indices=(normalized.index,),
            layout_node=split,
            structural_values=(split_size,),
        )

    if not isinstance(normalized, NormalizedSelect):
        return None
    view = normalized.source
    view_normalized = local_reduce.graph.normalized_nodes.get(view)
    shape = tensor_meta_shape(view)
    if (
        not isinstance(view_normalized, NormalizedView)
        or not local_reduce.graph.depends_on(view_normalized.source, gemm)
        or shape is None
        or len(shape) != 3
        or not statically_known_shape_equal((shape[0], shape[1] * shape[2]), gemm_shape)
        or not -len(shape) <= normalized.dim < len(shape)
    ):
        return None
    index = FlexGemmStructuralInt.from_value(normalized.index)
    if index is None:
        return None
    dim = normalized.dim % len(shape)
    structural_values = [index]
    if dim == 1:
        structural_group = FlexGemmStructuralInt.from_value(shape[1])
        if structural_group is None or structural_group.value <= 1:
            return None
        group = structural_group.value
        structural_values.append(structural_group)
        chunked = True
        layout_node = view
    elif dim == len(shape) - 1:
        grouped_layout = local_reduce.grouped_layouts.get(view)
        if grouped_layout is None or grouped_layout.layout.axis != 1:
            return None
        group = grouped_layout.layout.group
        structural_values.extend(grouped_layout.structural_values)
        chunked = False
        layout_node = None
    else:
        return None
    return GroupedMainLaneMatch(
        source=view_normalized.source,
        group=group,
        chunked=chunked,
        indices=(index.value,),
        layout_node=layout_node,
        structural_values=tuple(structural_values),
    )


def collect_grouped_main_lanes(
    output: torch.fx.Node,
    gemm: torch.fx.Node,
    local_reduce: FlexGemmLocalReduceAnalysis,
) -> list[tuple[torch.fx.Node, GroupedMainLaneMatch]] | None:
    """Collect grouped lane leaves without mutating analysis or installing guards."""
    gemm_shape = tensor_meta_shape(gemm)
    lanes: list[tuple[torch.fx.Node, GroupedMainLaneMatch]] = []
    seen: OrderedSet[torch.fx.Node] = OrderedSet()
    stack: list[Any] = [output]
    while stack:
        node = stack.pop()
        if not isinstance(node, torch.fx.Node) or node in seen:
            continue
        seen.add(node)
        match = match_grouped_main_lane(node, gemm, gemm_shape, local_reduce)
        if match is not None:
            lanes.append((node, match))
            continue
        if node is gemm or (
            node in local_reduce.grouped_layouts
            and local_reduce.graph.depends_on(node, gemm)
        ):
            return None
        stack.extend(reversed(tuple(iter_fx_node_inputs((node.args, node.kwargs)))))
    return lanes or None


def grouped_main_output_match(
    output: torch.fx.Node,
    gemm: torch.fx.Node,
    local_reduce: FlexGemmLocalReduceAnalysis,
) -> GroupedMainOutputMatch | None:
    """Validate a complete grouped main output and stage its node plans.

    Each lane spelling has already been checked against the physical GEMM shape,
    so aggregate agreement requires only one canonical source, group, and layout.
    """
    collected = collect_grouped_main_lanes(output, gemm, local_reduce)
    if collected is None:
        return None
    first = collected[0][1]
    source = canonical_grouped_main_source(first.source, gemm, local_reduce)
    indices: OrderedSet[int] = OrderedSet()
    select_indices: dict[torch.fx.Node, int] = {}
    grouped_layouts: dict[torch.fx.Node, FlexGemmGroupedLayoutMatch] = {}
    structural_values: list[FlexGemmStructuralInt] = []
    for node, match in collected:
        if (
            canonical_grouped_main_source(match.source, gemm, local_reduce)
            is not source
            or match.group != first.group
            or match.chunked != first.chunked
        ):
            return None
        for index in match.indices:
            if not -first.group <= index < first.group:
                return None
            indices.add(index % first.group)
        if isinstance(local_reduce.graph.normalized_nodes.get(node), NormalizedSelect):
            select_indices[node] = match.indices[0] % first.group
        if match.layout_node is not None:
            grouped_layouts[match.layout_node] = FlexGemmGroupedLayoutMatch(
                FlexGemmLocalReduceGeometry(group=match.group, axis=1)
            )
        structural_values.extend(match.structural_values)
    if indices != OrderedSet(range(first.group)):
        return None
    gemm_meta = gemm.meta.get("val")
    output_meta = output.meta.get("val")
    if gemm_meta is None or output_meta is None or len(gemm_meta.shape) != 2:
        return None
    expected_shape = (gemm_meta.shape[0], gemm_meta.shape[1] // first.group)
    if not statically_known_shape_equal(output_meta.shape, expected_shape):
        raise NotImplementedError(FLEX_GEMM_GROUPED_MAIN_SHAPE_ERROR)
    return GroupedMainOutputMatch(
        FlexGemmGroupedMainOutputTransform(group=first.group, chunked=first.chunked),
        select_indices,
        grouped_layouts,
        tuple(structural_values),
    )


@dataclasses.dataclass(frozen=True)
class FlexGemmEpilogueAnalysis:
    """Bundle the immutable analysis consumed by FlexGEMM lowering and emission.

    Attributes:
        gemm: The validated GEMM node shared by lowering and emission.
        outputs: Classification of main, auxiliary, and local-reduction outputs.
        local_reduce: Grouped layouts and local-reduction matches from the FX graph.
        grouped_select_indices: Select indices committed after grouped validation.
    """

    gemm: torch.fx.Node
    outputs: FlexGemmOutputPlan
    local_reduce: FlexGemmLocalReduceAnalysis
    grouped_select_indices: dict[torch.fx.Node, int] = dataclasses.field(
        default_factory=dict
    )

    @classmethod
    def from_graph_module(
        cls, graph_module: torch.fx.GraphModule, gemm: torch.fx.Node
    ) -> "FlexGemmEpilogueAnalysis":
        """Analyze grouped values and classify logical output consumers."""
        local_reduce = FlexGemmLocalReduceAnalysis.from_graph_module(graph_module)
        outputs = output_plan(graph_module, local_reduce)
        grouped_main = grouped_main_output_match(outputs.main, gemm, local_reduce)
        grouped_select_indices: dict[torch.fx.Node, int] = {}
        if grouped_main is not None:
            if outputs.aux_outputs or outputs.local_reduce is not None:
                raise NotImplementedError(FLEX_GEMM_GROUPED_MAIN_COMPOSITION_ERROR)
            grouped_main.commit_guards()
            local_reduce.grouped_layouts.update(grouped_main.grouped_layouts)
            grouped_select_indices = grouped_main.select_indices
            outputs = dataclasses.replace(
                outputs, main_transform=grouped_main.transform
            )
        else:
            if any(
                isinstance(normalized, NormalizedSelect)
                and normalized.source in local_reduce.grouped_layouts
                and local_reduce.grouped_layouts[normalized.source].layout.axis == 1
                for normalized in local_reduce.graph.normalized_nodes.values()
            ):
                raise NotImplementedError(
                    "FlexGEMM grouped selects must form a complete grouped main output"
                )
            main_shape = tensor_meta_shape(outputs.main)
            gemm_shape = tensor_meta_shape(gemm)
            if (
                main_shape is None
                or gemm_shape is None
                or not statically_known_shape_equal(main_shape, gemm_shape)
            ):
                raise NotImplementedError(FLEX_GEMM_MAIN_OUTPUT_SHAPE_ERROR)
        for match in local_reduce.matches.values():
            match.commit_guards()
        if outputs.local_reduce is not None:
            outputs.local_reduce.match.commit_guards()
        analysis = cls(gemm, outputs, local_reduce, grouped_select_indices)
        active_geometries = OrderedSet(analysis.required_geometries)
        for grouped_layout in local_reduce.grouped_layouts.values():
            if grouped_layout.layout in active_geometries:
                grouped_layout.commit_guards()
        return analysis

    @property
    def required_geometries(self) -> tuple[FlexGemmLocalReduceGeometry, ...]:
        """Return every grouped geometry that constrains kernel configuration."""
        geometries = OrderedSet(
            match.geometry for match in self.local_reduce.matches.values()
        )
        if self.outputs.local_reduce is not None:
            geometries.add(self.outputs.local_reduce.match.geometry)
        if self.outputs.main_transform is not None:
            geometries.add(
                FlexGemmLocalReduceGeometry(
                    group=self.outputs.main_transform.group,
                    axis=1,
                )
            )
        return tuple(geometries)


def analyze_flex_gemm_epilogue(
    graph_module: torch.fx.GraphModule,
    gemm: torch.fx.Node,
) -> FlexGemmEpilogueAnalysis:
    """Analyze FlexGEMM body for output planning and epilogue code generation.

    This is the analysis entry point called by FlexGEMM lowering. It builds a
    dependency index, performs topological local-reduction analysis, and
    returns the shared immutable plan consumed by config selection and
    ``materialize_flex_gemm_epilogue``.

    Args:
        graph_module: FlexGEMM body graph containing GEMM and epilogue nodes.
        gemm: The validated GEMM node within ``graph_module``.

    Returns:
        Output and local-reduction analysis shared by later lowering phases.
    """
    return FlexGemmEpilogueAnalysis.from_graph_module(graph_module, gemm)


def gemm_node(
    graph_module: torch.fx.GraphModule, gemm_op: torch._ops.OpOverload
) -> torch.fx.Node:
    gemm_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target == gemm_op
    ]
    if len(gemm_nodes) != 1:
        raise NotImplementedError("FlexGEMM expects one GEMM body")
    return gemm_nodes[0]


class FlexGemmEpilogueEmitter:
    """Visit an analyzed FlexGEMM FX graph and emit its CuTeDSL epilogue.

    The analysis dataclasses flow into each other as follows:

    ::

        FlexGemmEpilogueGraph
          `--> FlexGemmLocalReduceAnalysis
                 +--> grouped_layouts
                 `--> matches
                        `--> FlexGemmLocalReduceMatch
                               `--> FlexGemmOutputLocalReducePlan
                                      `--> optional FlexGemmLocalReduceStore

        FlexGemmLocalReduceAnalysis
          `--> output_plan()
                 `--> FlexGemmOutputPlan

        FlexGemmLocalReduceAnalysis + FlexGemmOutputPlan
          `--> FlexGemmEpilogueAnalysis
                 `--> FlexGemmEpilogueEmitter

    At emitter construction, ``analysis.outputs`` becomes ``self.outputs``;
    its local-reduce match and optional store initialize ``self.feed_main`` and
    ``self.aux``. ``analysis.local_reduce.grouped_layouts`` is copied into
    mutable emission state, while ``analysis.required_geometries`` determines
    the active grouped layouts.

    The emitter owns all mutable code-generation state: FX values lowered so far,
    grouped TensorSSA layouts, compressed-store expressions, and physical
    reduction callbacks. ``lower_graph`` performs a topological traversal and
    delegates each ``call_function`` node to ordered handlers; ``render`` turns
    the resulting state into the generated epilogue and callback source.
    """

    def __init__(
        self,
        graph_module: torch.fx.GraphModule,
        analysis: FlexGemmEpilogueAnalysis,
        epilogue_arg_placeholders: tuple[torch.fx.Node, ...] = (),
        *,
        fast_math: bool = False,
    ) -> None:
        self.graph_module = graph_module
        self.epilogue_arg_placeholders = epilogue_arg_placeholders
        self.fast_math = fast_math
        self.gemm = analysis.gemm
        self.outputs = analysis.outputs
        self.normalized_nodes = analysis.local_reduce.graph.normalized_nodes
        self.grouped_select_indices = analysis.grouped_select_indices
        self.kernel = FlexGemmCuteDSLKernel()
        self.env: dict[torch.fx.Node, Any] = {
            self.gemm: CuteDSLCSEVariable(
                "acc", ValueRanges.unknown(), dtype=torch.float32, shape=(1,)
            )
        }
        self.grouped_tensors = {
            node: grouped.layout
            for node, grouped in analysis.local_reduce.grouped_layouts.items()
        }
        self.active_grouped_layouts = OrderedSet(analysis.required_geometries)
        self.store_sources: dict[torch.fx.Node, Any] = {}
        self.physical_reductions: dict[torch.fx.Node, FlexGemmPhysicalReduction] = {}
        self.local_reduce = self.outputs.local_reduce
        self.feed_main: torch.fx.Node | None = None
        self.aux: torch.fx.Node | None = None
        self.feed_main_input: torch.fx.Node | None = None
        match self.local_reduce:
            case FlexGemmOutputLocalReducePlan(
                match=local_reduce_match, store=store, feeds_main=True
            ):
                self.feed_main = local_reduce_match.value_node
                normalized = self.normalized_nodes.get(local_reduce_match.value_node)
                if not isinstance(normalized, NormalizedReduction):
                    raise AssertionError("feed-main plans require a matched reduction")
                self.feed_main_input = normalized.source
                self.aux = None if store is None else store.node
            case FlexGemmOutputLocalReducePlan(
                store=FlexGemmLocalReduceStore(node=store_node)
            ):
                self.aux = store_node
            case None:
                pass

    def bind_epilogue_args(self) -> None:
        """Bind captured tensor placeholders to generated CuTeDSL parameters."""
        for index, node in enumerate(self.epilogue_arg_placeholders):
            epilogue_arg_meta = node.meta["val"]
            physical_dtype = (
                torch.uint8
                if epilogue_arg_meta.dtype is torch.bool
                else epilogue_arg_meta.dtype
            )
            logical_dtype = upcast_compute_type(epilogue_arg_meta.dtype)
            self.env[node] = CuteDSLCSEVariable(
                f"aux{index}",
                ValueRanges.unknown(),
                dtype=physical_dtype,
                shape=(1,),
            )
            if logical_dtype != physical_dtype:
                self.env[node] = FlexGemmCuteDSLOpOverrides.to_dtype(
                    self.env[node], logical_dtype, use_compute_types=False
                )

    def bind_reduction(self, node: torch.fx.Node, lowered_reduce: Any) -> None:
        """Bind a generated reduction or replace it with the feed-main parameter."""
        if self.feed_main is not None and node is self.feed_main:
            self.env[node] = CuteDSLCSEVariable(
                LOCAL_REDUCE_FEED_MAIN_ARG_NAME,
                ValueRanges.unknown(),
                dtype=lowered_reduce.dtype,
                shape=lowered_reduce.shape,
            )
            if self.feed_main_input in self.grouped_tensors:
                self.grouped_tensors[node] = self.grouped_tensors[self.feed_main_input]
            return
        self.env[node] = lowered_reduce

    def lower_pointwise_store(self, node: torch.fx.Node) -> bool:
        """Lower pointwise expressions that consume a compressed store value."""
        if (
            self.feed_main is not None
            or not is_shape_preserving_pointwise_node(node)
            or not any(
                arg in self.store_sources
                for arg in iter_fx_node_inputs((node.args, tuple(node.kwargs.values())))
            )
        ):
            return False
        store_args = tuple(
            _local_reduce_store_arg(arg, self.env, self.store_sources)
            for arg in node.args
        )
        store_kwargs = {
            key: _local_reduce_store_arg(value, self.env, self.store_sources)
            for key, value in node.kwargs.items()
        }
        self.env[node] = _cute_call(node.target, store_args, store_kwargs)
        self.store_sources[node] = self.env[node]
        return True

    def propagate_physical_reduction(self, node: torch.fx.Node, source: Any) -> None:
        """Preserve physical callback provenance through shape-only wrappers."""
        if isinstance(source, torch.fx.Node) and source in self.physical_reductions:
            self.physical_reductions[node] = self.physical_reductions[source]

    def physical_finalize_arg(self, value: Any) -> Any:
        """Replace physical reduction inputs with their generated value expression."""
        if isinstance(value, torch.fx.Node) and value in self.physical_reductions:
            return self.physical_reductions[value].finalize_expr
        if isinstance(value, (tuple, list)):
            return type(value)(self.physical_finalize_arg(item) for item in value)
        return _cute_arg(value, self.env)

    def compose_physical_finalize(self, node: torch.fx.Node) -> Any | None:
        """Fold a pointwise consumer into one generated physical finalizer."""
        physical_inputs = list(
            OrderedSet(
                arg
                for arg in iter_fx_node_inputs((node.args, node.kwargs))
                if arg in self.physical_reductions
            )
        )
        if not physical_inputs:
            return None
        if len(physical_inputs) > 1:
            raise NotImplementedError(LOCAL_REDUCE_SINGLE_PHYSICAL_FINALIZE_ERROR)
        base = physical_inputs[0]
        args = tuple(self.physical_finalize_arg(arg) for arg in node.args)
        kwargs = {
            key: self.physical_finalize_arg(value) for key, value in node.kwargs.items()
        }
        finalize_expr = _cute_call(node.target, args, kwargs)
        if not isinstance(finalize_expr, str):
            raise NotImplementedError(LOCAL_REDUCE_FINALIZE_SCALAR_ONLY_ERROR)
        self.store_sources[node] = self.store_sources[base]
        self.physical_reductions[node] = dataclasses.replace(
            self.physical_reductions[base], finalize_expr=finalize_expr
        )
        return finalize_expr

    def lower_call_function(self, node: torch.fx.Node) -> None:
        """Lower one call_function node using the ordered FlexGEMM handlers."""
        normalized = self.normalized_nodes.get(node)
        lowered = lower_full_scalar(node)
        if lowered is not None:
            self.env[node] = lowered
            return
        match normalized:
            case NormalizedSqueeze():
                lowered = lower_squeeze(node, normalized, self.env, self.store_sources)
                if lowered is not None:
                    self.env[node] = lowered
                    self.propagate_physical_reduction(node, normalized.source)
                    return
            case NormalizedSplit():
                lowered = lower_grouped_n_split(
                    node, normalized, self.env, self.kernel, self.grouped_tensors
                )
                if lowered is not None:
                    self.env[node] = lowered
                    return
            case NormalizedSelect():
                index = self.grouped_select_indices.get(node)
                if index is not None:
                    self.env[node] = lower_grouped_n_select(
                        normalized, index, self.env, self.kernel
                    )
                    return
            case NormalizedGetItem():
                lowered = lower_getitem(node, normalized, self.env, self.store_sources)
                if lowered is not None:
                    self.env[node] = lowered
                    self.propagate_physical_reduction(node, normalized.source)
                    return
            case NormalizedPrepareSoftmax():
                self.env[node] = lower_prepare_softmax_online(
                    node,
                    normalized,
                    self.env,
                    self.kernel,
                    self.grouped_tensors,
                    self.store_sources,
                )
                return
            case NormalizedView():
                lowered = lower_view_or_reshape(
                    node,
                    normalized,
                    self.env,
                    self.kernel,
                    self.grouped_tensors,
                    self.active_grouped_layouts,
                    self.store_sources,
                    node is self.feed_main_input,
                )
                if lowered is not None:
                    self.env[node] = lowered
                    self.propagate_physical_reduction(node, normalized.source)
                    return
            case NormalizedReduction():
                self.bind_reduction(
                    node,
                    lower_tensorssa_reduce(
                        node,
                        normalized,
                        self.env,
                        self.kernel,
                        self.grouped_tensors,
                        self.store_sources,
                        self.physical_reductions,
                    ),
                )
                return
            case NormalizedUnsupportedReduction():
                raise local_reduce_unsupported_tensorssa_error(
                    normalized.target, value_only=True
                )
        is_shape_preserving = is_shape_preserving_pointwise_node(node)
        if is_shape_preserving and self.feed_main is None:
            if self.aux is None and any(
                arg in self.physical_reductions
                for arg in iter_fx_node_inputs((node.args, node.kwargs))
            ):
                raise NotImplementedError(LOCAL_REDUCE_POST_POINTWISE_FINALIZE_ERROR)
            physical_finalize = self.compose_physical_finalize(node)
            if physical_finalize is not None:
                self.env[node] = physical_finalize
                return
        if self.lower_pointwise_store(node):
            return
        node_args = tuple(_cute_arg(arg, self.env) for arg in node.args)
        node_kwargs = {
            key: _cute_arg(value, self.env) for key, value in node.kwargs.items()
        }
        self.env[node] = _cute_call(node.target, node_args, node_kwargs)

    def lower_graph(self) -> None:
        """Lower body nodes in FX topological order."""
        for node in self.graph_module.graph.nodes:
            if node is self.gemm or node.op in ("placeholder", "output"):
                continue
            if isinstance(node.meta.get("val"), (int, torch.SymInt)):
                continue
            with V.set_current_node(node):
                if node.op != "call_function":
                    raise NotImplementedError(
                        f"unsupported FlexGEMM epilogue node: {node.format_node()}"
                    )
                self.lower_call_function(node)

    @staticmethod
    def aux_result(
        aux: torch.fx.Node | None, store_sources: dict[torch.fx.Node, Any]
    ) -> Any | None:
        """Return the compressed-aux expression or reject missing TensorSSA."""
        if aux is None:
            return None
        result = store_sources.get(aux)
        if result is None:
            raise NotImplementedError(LOCAL_REDUCE_AUX_TENSORSSA_ERROR)
        return result

    def render(self) -> tuple[str, str]:
        """Render the generated epilogue and physical callback source."""
        body = "\n".join(f"    {line}" for line in self.kernel.body.lines)
        if body:
            body += "\n"
        aux_args = [
            f"aux{index}" for index in range(len(self.epilogue_arg_placeholders))
        ]
        feed_main_args = (
            [LOCAL_REDUCE_FEED_MAIN_ARG_NAME] if self.feed_main is not None else []
        )
        epilogue_params = ", ".join(["acc", *aux_args, *feed_main_args])
        result = _cute_arg(self.outputs.main, self.env)
        aux_result = self.aux_result(self.aux, self.store_sources)
        if self.outputs.aux_outputs or aux_result is not None:
            tuple_items = [result]
            tuple_items.extend(
                _cute_arg(aux_output, self.env)
                for aux_output in self.outputs.aux_outputs
            )
            if aux_result is not None:
                tuple_items.append(aux_result)
            result = f"({', '.join(str(item) for item in tuple_items)})"
        physical_reduction = (
            None
            if self.local_reduce is None
            else self.physical_reductions.get(self.local_reduce.match.value_node)
        )
        physical_reduction_payload = (
            ""
            if physical_reduction is None
            else (
                f"\ncombine {physical_reduction.combine_expr}"
                f"\nfinalize {physical_reduction.finalize_expr}"
            )
        )
        key_payload = (
            f"fast_math={self.fast_math}\n{self.graph_module.code}\n"
            f"{body}\nreturn {result}{physical_reduction_payload}"
        )
        key = hashlib.sha256(key_payload.encode()).hexdigest()[:16]
        name = f"flex_gemm_epilogue_{key}"
        local_reduce_source = ""
        if physical_reduction is not None:
            combine_name = f"{name}{LOCAL_REDUCE_COMBINE_FN_SUFFIX}"
            finalize_name = f"{name}{LOCAL_REDUCE_FINALIZE_FN_SUFFIX}"
            local_reduce_source = (
                f"@cute.jit\ndef {combine_name}(lhs, rhs):\n"
                f"    return {physical_reduction.combine_expr}\n"
                f"{combine_name}.__cache_key__ = lambda: {combine_name!r}\n\n"
                f"@cute.jit\ndef {finalize_name}(value):\n"
                f"    return {physical_reduction.finalize_expr}\n"
                f"{finalize_name}.__cache_key__ = lambda: {finalize_name!r}\n\n"
            )
        return (
            name,
            "import cutlass\n"
            "import cutlass.cute as cute\n"
            "import operator\n"
            "from cutlass._mlir.dialects import math as mlir_math\n\n"
            f"{local_reduce_source}"
            f"@cute.jit\ndef {name}({epilogue_params}):\n"
            f"{body}    return {result}\n",
        )

    def materialize(self) -> tuple[str, str]:
        """Lower and render this epilogue under the CuTeDSL virtualized handlers."""
        with (
            V.set_kernel_handler(self.kernel),
            V.set_ops_handler(FlexGemmCuteDSLOpOverrides()),
            use_cutedsl_fast_math(self.fast_math),
        ):
            self.bind_epilogue_args()
            self.lower_graph()
        return self.render()


def materialize_flex_gemm_epilogue(
    graph_module: torch.fx.GraphModule,
    analysis: FlexGemmEpilogueAnalysis,
    epilogue_arg_placeholders: tuple[torch.fx.Node, ...] = (),
    *,
    fast_math: bool = False,
) -> tuple[str, str]:
    """Materialize an analyzed FlexGEMM body as generated CuTeDSL source.

    This is the code-generation entry point called by FlexGEMM lowering after
    ``analyze_flex_gemm_epilogue`` has classified outputs and local-reduction
    matches. The emitter visits the FX graph once in topological order while
    owning the environment and reduction state needed across nodes.

    Args:
        graph_module: FlexGEMM body graph containing the GEMM and epilogue nodes.
        analysis: Shared GEMM, output, and local-reduction analysis for the graph.
        epilogue_arg_placeholders: Captured tensor placeholders exposed as
            generated epilogue parameters.
        fast_math: Whether supported CuTeDSL math operations may use approximate
            fast-math lowering.
        swap_ab: Whether generated local-reduction expressions use QuACK's
            transposed physical accumulator coordinates.

    Returns:
        The generated epilogue function name and complete CuTeDSL source.
    """
    return FlexGemmEpilogueEmitter(
        graph_module,
        analysis,
        epilogue_arg_placeholders,
        fast_math=fast_math,
    ).materialize()
