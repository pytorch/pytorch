# mypy: allow-untyped-defs
"""Analyze FlexGEMM epilogue FX graphs and materialize CuTeDSL source.

``analyze_flex_gemm_epilogue`` indexes FX dependencies, identifies nodes that
carry grouped TensorSSA layouts, matches supported local reductions, and plans
the main, auxiliary, and local-reduction consumers.

``materialize_flex_gemm_epilogue`` uses that analysis to generate the CuTeDSL
epilogue and physical reduction callbacks.
"""

import dataclasses
import hashlib
import operator
from typing import Any

import torch
from torch._inductor import inductor_prims
from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
    CuteDSLCSEVariable,
    CuteDSLOpOverrides,
    upcast_compute_type,
)
from torch._inductor.kernel.flex_gemm.constraints import (
    FLEX_GEMM_OUTPUT_PLAN_NODE_ERROR,
    FLEX_GEMM_OUTPUT_TENSOR_ERROR,
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
from torch._inductor.kernel.flex_gemm.quack_reductions import (
    _cute_arg,
    _cute_call,
    _local_reduce_store_arg,
    FlexGemmPhysicalReduction,
    grouped_tensor_layout,
    GroupedTensorSSALayout,
    is_shape_preserving_pointwise_node,
    iter_fx_node_inputs,
    lower_full_scalar,
    lower_getitem,
    lower_prepare_softmax_online,
    lower_squeeze,
    lower_tensorssa_reduce,
    lower_view_or_reshape,
    reduction_from_node,
    squeeze_source_node,
    tensor_meta_shape,
    unsupported_reduction_from_node,
    view_or_reshape_args,
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
    """

    value_node: torch.fx.Node
    geometry: FlexGemmLocalReduceGeometry

    def __post_init__(self) -> None:
        if not isinstance(self.value_node, torch.fx.Node):
            raise RuntimeError(LOCAL_REDUCE_MATCH_NODE_ERROR)

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
        return match

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
    """Classify the values returned by a FlexGEMM body.

    Attributes:
        output: FX node returned as the main GEMM result.
        aux_outputs: Same-shape auxiliary FX outputs returned after the main result.
        local_reduce: Compressed or feed-main local-reduction output behavior.
    """

    output: torch.fx.Node
    aux_outputs: tuple[torch.fx.Node, ...] = ()
    local_reduce: FlexGemmOutputLocalReducePlan | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.output, torch.fx.Node) or not all(
            isinstance(aux_output, torch.fx.Node) for aux_output in self.aux_outputs
        ):
            raise RuntimeError(FLEX_GEMM_OUTPUT_PLAN_NODE_ERROR)


@dataclasses.dataclass(frozen=True)
class FlexGemmEpilogueGraph:
    """Index transitive dependencies between nodes in an epilogue FX graph.

    Attributes:
        dependencies: Every FX node mapped to all of its direct and transitive
            input nodes.
    """

    dependencies: dict[torch.fx.Node, frozenset[torch.fx.Node]]

    @classmethod
    def from_graph_module(
        cls, graph_module: torch.fx.GraphModule
    ) -> "FlexGemmEpilogueGraph":
        """Build transitive dependencies in the graph's topological order."""
        dependencies: dict[torch.fx.Node, frozenset[torch.fx.Node]] = {}
        for node in graph_module.graph.nodes:
            node_dependencies: OrderedSet[torch.fx.Node] = OrderedSet()
            for input_node in iter_fx_node_inputs((node.args, node.kwargs)):
                node_dependencies.add(input_node)
                node_dependencies.update(dependencies.get(input_node, ()))
            dependencies[node] = frozenset(node_dependencies)
        return cls(dependencies)

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
    ``GroupedTensorSSALayout`` for the grouped layout attached to reshape and
    pointwise nodes, and ``FlexGemmLocalReduceMatch`` for each supported reduced
    value found from those layouts.

    Attributes:
        graph: Dependency index used by recursive feed-main matching.
        grouped_tensors: FX nodes whose values carry a grouped TensorSSA layout.
        matches: FX values matched to a supported grouped local reduction.
    """

    graph: FlexGemmEpilogueGraph
    grouped_tensors: dict[torch.fx.Node, GroupedTensorSSALayout] = dataclasses.field(
        default_factory=dict
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
        view_args = view_or_reshape_args(node)
        if view_args is not None:
            source_node, shape = view_args
            if self.propagate_local_reduce_match(node, source_node):
                return
            if self.bind_grouped_layout(node, shape, source_node):
                return
        reduction = reduction_from_node(node)
        if reduction is not None:
            input_node, dim, _, dtype, _ = reduction
            if self.bind_grouped_reduction(node, input_node, dim, dtype):
                return
        if node.target is inductor_prims.prepare_softmax_online:
            input_node = node.args[0]
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
            if self.bind_grouped_reduction(
                node, input_node, dim, raise_invalid_dims=False
            ):
                return
        unsupported_reduction = unsupported_reduction_from_node(node)
        if unsupported_reduction is not None:
            input_node = node.args[0]
            if (
                isinstance(input_node, torch.fx.Node)
                and input_node in self.grouped_tensors
            ):
                raise local_reduce_unsupported_tensorssa_error(unsupported_reduction)
        if self.propagate_local_reduce_match(node, squeeze_source_node(node)):
            return
        if node.target is operator.getitem and self.propagate_local_reduce_match(
            node, node.args[0]
        ):
            return
        if is_shape_preserving_pointwise_node(node):
            self.propagate_pointwise_match(node, LOCAL_REDUCE_MIXED_MATCH_ERROR)

    def bind_grouped_layout(self, node: torch.fx.Node, shape: Any, source: Any) -> bool:
        """Attach a grouped TensorSSA layout introduced by a reshape."""
        source_shape = (
            tensor_meta_shape(source) if isinstance(source, torch.fx.Node) else None
        )
        layout = grouped_tensor_layout(shape, source_shape)
        if layout is None or not isinstance(source, torch.fx.Node):
            return False
        self.grouped_tensors[node] = layout
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
        validate_local_reduce_tensorssa_group_size(layout.axis, layout.group_size)
        if not layout.matches_reduction_dim(dim):
            if not raise_invalid_dims:
                return False
            raise NotImplementedError(LOCAL_REDUCE_INNERMOST_GROUPED_DIM_ERROR)
        self.matches[node] = FlexGemmLocalReduceMatch(
            node, FlexGemmLocalReduceGeometry(layout.group_size, layout.axis)
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
            if layout.needs_physical_combine
            and FlexGemmLocalReduceGeometry(layout.group_size, layout.axis)
            in active_geometries
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
        layout: GroupedTensorSSALayout,
    ) -> FlexGemmLocalReduceMatch | None:
        """Find the grouped reduction that produces a broadcast value."""
        if not isinstance(value, torch.fx.Node):
            return None
        reduction = reduction_from_node(value)
        if reduction is not None:
            input_node, dim, keepdim, dtype, _ = reduction
            if input_node is not grouped_source:
                if self.graph.depends_on(input_node, grouped_source):
                    raise NotImplementedError(LOCAL_REDUCE_SOURCE_EXPRESSION_ERROR)
                raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
            if (
                dtype is not None
                or not keepdim
                or not layout.matches_reduction_dim(dim)
            ):
                raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
            return FlexGemmLocalReduceMatch(
                value, FlexGemmLocalReduceGeometry(layout.group_size, layout.axis)
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
        reduction = reduction_from_node(value)
        if reduction is not None:
            self.validate_hidden_feed_main_reduction_input(reduction[0], grouped_source)
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
        reduction = reduction_from_node(match.value_node)
        if reduction is not None and isinstance(reduction[0], torch.fx.Node):
            self.validate_feed_main_source_reductions(
                source, reduction[0], match.value_node
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
        layout: GroupedTensorSSALayout,
    ) -> bool:
        """Return whether a candidate contains a grouped feed-main reduction."""
        if not isinstance(value, torch.fx.Node):
            return False
        reduction = reduction_from_node(value)
        if reduction is not None:
            input_node, dim, keepdim, dtype, _ = reduction
            return (
                dtype is None
                and bool(keepdim)
                and layout.matches_reduction_dim(dim)
                and (
                    input_node is grouped_source
                    or self.graph.depends_on(input_node, grouped_source)
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
        view_args = view_or_reshape_args(grouped_source)
        if view_args is None:
            return None
        source_node, input_shape = view_args
        if not isinstance(source_node, torch.fx.Node):
            return None
        layout = grouped_tensor_layout(input_shape, tensor_meta_shape(source_node))
        if layout is None:
            return None
        if layout.axis != 0:
            if not self.feed_main_grouped_reduction(value, grouped_source, layout):
                return None
            if layout.group_size <= LOCAL_REDUCE_FRAGMENT_WIDTH:
                # Intentional fallthrough: axis-1 feeds within one TensorSSA
                # fragment lower as plain generated TensorSSA without a feed plan.
                return None
            raise NotImplementedError(LOCAL_REDUCE_FEED_MAIN_AXIS1_FRAGMENT_ERROR)
        validate_local_reduce_feed_main_capability(layout.axis, layout.group_size)
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
        view_args = view_or_reshape_args(output)
        if view_args is not None:
            source, _ = view_args
            if not isinstance(source, torch.fx.Node):
                return None
            return self.match_feed_main_source(source, output.meta.get("val"))
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
    return (
        FlexGemmOutputPlan(output_value) if feed_main_plan is None else feed_main_plan
    )


@dataclasses.dataclass(frozen=True)
class FlexGemmEpilogueAnalysis:
    """Bundle the immutable analysis consumed by FlexGEMM lowering and emission.

    Attributes:
        outputs: Classification of main, auxiliary, and local-reduction outputs.
        local_reduce: Grouped layouts and local-reduction matches from the FX graph.
    """

    outputs: FlexGemmOutputPlan
    local_reduce: FlexGemmLocalReduceAnalysis

    @classmethod
    def from_graph_module(
        cls, graph_module: torch.fx.GraphModule
    ) -> "FlexGemmEpilogueAnalysis":
        """Run the one-pass local-reduction analysis and classify graph outputs."""
        local_reduce = FlexGemmLocalReduceAnalysis.from_graph_module(graph_module)
        return cls(output_plan(graph_module, local_reduce), local_reduce)

    @property
    def required_geometries(self) -> tuple[FlexGemmLocalReduceGeometry, ...]:
        """Return every grouped geometry that constrains kernel configuration."""
        geometries = OrderedSet(
            match.geometry for match in self.local_reduce.matches.values()
        )
        if self.outputs.local_reduce is not None:
            geometries.add(self.outputs.local_reduce.match.geometry)
        return tuple(geometries)


def analyze_flex_gemm_epilogue(
    graph_module: torch.fx.GraphModule,
) -> FlexGemmEpilogueAnalysis:
    """Analyze FlexGEMM body for output planning and epilogue code generation.

    This is the analysis entry point called by FlexGEMM lowering. It builds a
    dependency index, performs topological local-reduction analysis, and
    returns the shared immutable plan consumed by config selection and
    ``materialize_flex_gemm_epilogue``.

    Args:
        graph_module: FlexGEMM body graph containing GEMM and epilogue nodes.

    Returns:
        Output and local-reduction analysis shared by later lowering phases.
    """
    return FlexGemmEpilogueAnalysis.from_graph_module(graph_module)


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
                 +--> grouped_tensors
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
    ``self.aux``. ``analysis.local_reduce.grouped_tensors`` is copied into
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
        gemm_op: torch._ops.OpOverload,
        analysis: FlexGemmEpilogueAnalysis,
        epilogue_arg_placeholders: tuple[torch.fx.Node, ...] = (),
    ) -> None:
        self.graph_module = graph_module
        self.epilogue_arg_placeholders = epilogue_arg_placeholders
        self.gemm = gemm_node(graph_module, gemm_op)
        self.outputs = analysis.outputs
        self.kernel = FlexGemmCuteDSLKernel()
        self.env: dict[torch.fx.Node, Any] = {
            self.gemm: CuteDSLCSEVariable(
                "acc", ValueRanges.unknown(), dtype=torch.float32, shape=(1,)
            )
        }
        self.grouped_tensors = dict(analysis.local_reduce.grouped_tensors)
        self.active_grouped_layouts = OrderedSet(
            GroupedTensorSSALayout(geometry.axis, geometry.group)
            for geometry in analysis.required_geometries
        )
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
                reduction = reduction_from_node(local_reduce_match.value_node)
                if reduction is None or not isinstance(reduction[0], torch.fx.Node):
                    raise AssertionError("feed-main plans require a matched reduction")
                self.feed_main_input = reduction[0]
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
        lowered = lower_full_scalar(node)
        if lowered is not None:
            self.env[node] = lowered
            return
        lowered = lower_squeeze(node, self.env, self.store_sources)
        if lowered is not None:
            self.env[node] = lowered
            self.propagate_physical_reduction(node, node.args[0])
            return
        lowered = lower_getitem(node, self.env, self.store_sources)
        if lowered is not None:
            self.env[node] = lowered
            self.propagate_physical_reduction(node, node.args[0])
            return
        lowered = lower_prepare_softmax_online(
            node,
            self.env,
            self.kernel,
            self.grouped_tensors,
            self.store_sources,
        )
        if lowered is not None:
            self.env[node] = lowered
            return
        lowered = lower_view_or_reshape(
            node,
            self.env,
            self.kernel,
            self.grouped_tensors,
            self.active_grouped_layouts,
            self.store_sources,
            node is self.feed_main_input,
        )
        if lowered is not None:
            self.env[node] = lowered
            self.propagate_physical_reduction(node, node.args[0])
            return
        lowered = lower_tensorssa_reduce(
            node,
            self.env,
            self.kernel,
            self.grouped_tensors,
            self.store_sources,
            self.physical_reductions,
        )
        if lowered is not None:
            self.bind_reduction(node, lowered)
            return
        unsupported_reduction = unsupported_reduction_from_node(node)
        if unsupported_reduction is not None:
            raise local_reduce_unsupported_tensorssa_error(
                unsupported_reduction, value_only=True
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
        result = _cute_arg(self.outputs.output, self.env)
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
            f"{self.graph_module.code}\n{body}\nreturn {result}"
            f"{physical_reduction_payload}"
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
        ):
            self.bind_epilogue_args()
            self.lower_graph()
        return self.render()


def materialize_flex_gemm_epilogue(
    graph_module: torch.fx.GraphModule,
    gemm_op: torch._ops.OpOverload,
    analysis: FlexGemmEpilogueAnalysis,
    epilogue_arg_placeholders: tuple[torch.fx.Node, ...] = (),
) -> tuple[str, str]:
    """Materialize an analyzed FlexGEMM body as generated CuTeDSL source.

    This is the code-generation entry point called by FlexGEMM lowering after
    ``analyze_flex_gemm_epilogue`` has classified outputs and local-reduction
    matches. The emitter visits the FX graph once in topological order while
    owning the environment and reduction state needed across nodes.

    Args:
        graph_module: FlexGEMM body graph containing the GEMM and epilogue nodes.
        gemm_op: GEMM overload expected to occur exactly once in the body.
        analysis: Shared output and local-reduction analysis for the graph.
        epilogue_arg_placeholders: Captured tensor placeholders exposed as
            generated epilogue parameters.

    Returns:
        The generated epilogue function name and complete CuTeDSL source.
    """
    return FlexGemmEpilogueEmitter(
        graph_module, gemm_op, analysis, epilogue_arg_placeholders
    ).materialize()
