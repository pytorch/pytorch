# mypy: allow-untyped-defs
"""Analyze FlexGEMM FX graphs and emit QuACK EpiMod functions.

The shared analysis identifies GEMM inputs, captures, outputs, grouped layouts,
and reduction dependencies. ``materialize_flex_gemm_epimod`` then changes only
the emission boundary, preserving the analysis as the semantic source of truth.
"""

import dataclasses
import hashlib
import math
import operator
from typing import Any

from sympy import Max, Min

import torch
from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
    CuteDSLCSEVariable,
    CuteDSLOpOverrides,
    tensorssa_reduction,
    use_cutedsl_fast_math,
)
from torch._inductor.kernel.flex_gemm.constraints import (
    FLEX_GEMM_GROUPED_MAIN_COMPOSITION_ERROR,
    FLEX_GEMM_GROUPED_MAIN_SHAPE_ERROR,
    FLEX_GEMM_INDEXED_OUTPUT_SOURCE_ERROR,
    FLEX_GEMM_MAIN_OUTPUT_SHAPE_ERROR,
    FLEX_GEMM_NESTED_TENSORSSA_CAPTURE_ERROR,
    FLEX_GEMM_NESTED_TENSORSSA_LANES_ERROR,
    FLEX_GEMM_OUTPUT_PLAN_NODE_ERROR,
    FLEX_GEMM_OUTPUT_TENSOR_ERROR,
    FlexGemmGroupedMainOutputTransform,
    FlexGemmLocalReduceGeometry,
    INDEXED_OUTPUT_STORE_ARG_NAME,
    local_reduce_compressed_shape,
    LOCAL_REDUCE_EXPLICIT_DTYPE_ERROR,
    LOCAL_REDUCE_FEED_MAIN_ARG_NAME,
    LOCAL_REDUCE_FEED_MAIN_AXIS1_FRAGMENT_ERROR,
    LOCAL_REDUCE_FEED_MAIN_MIXED_MATCH_ERROR,
    LOCAL_REDUCE_FRAGMENT_WIDTH,
    LOCAL_REDUCE_INNERMOST_GROUPED_DIM_ERROR,
    LOCAL_REDUCE_MATCH_NODE_ERROR,
    LOCAL_REDUCE_MIXED_GROUPED_LAYOUT_ERROR,
    LOCAL_REDUCE_MIXED_MATCH_ERROR,
    LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR,
    LOCAL_REDUCE_OUTPUT_PLAN_NODE_ERROR,
    LOCAL_REDUCE_PREPASS_FN_SUFFIX,
    LOCAL_REDUCE_SOURCE_EXPRESSION_ERROR,
    LOCAL_REDUCE_STORE_ARG_NAME,
    NESTED_TENSORSSA_PACKED_STORAGE_SPAN,
    NESTED_TENSORSSA_PHYSICAL_SPAN,
    ungrouped_reduction_error,
    unsupported_reduction_op_error,
    validate_local_reduce_feed_main_capability,
    validate_local_reduce_tensorssa_group_size,
)
from torch._inductor.kernel.flex_gemm.output_layout import (
    BLOCKED_128X4,
    FlexGemmOutputLayout,
    TRANSPOSED,
)
from torch._inductor.kernel.flex_gemm.quack_reductions import (
    FlexGemmStructuralInt,
    FlexGemmTensorSSAFact,
    grouped_tensor_layout,
    GroupedTensorSSALayout,
    is_shape_preserving_pointwise_node,
    squeeze_source_node,
    tensor_meta_shape,
    view_or_reshape_args,
)
from torch._inductor.kernel.gemm_epilogue import (
    GemmEpilogueGraph,
    GemmReductionType,
    iter_fx_node_inputs,
    NormalizedGemmReduction,
    NormalizedPrepareSoftmax,
    NormalizedReduction,
    NormalizedUnsupportedReduction,
)
from torch._inductor.kernel.gemm_epilogue_codegen import (
    gemm_epilogue_arg,
    gemm_epilogue_source_expr,
    GemmEpilogueCuteDSLKernel,
    GemmEpilogueCuteDSLOpOverrides,
    lower_gemm_epilogue_fx_node,
)
from torch._inductor.kernel.gemm_epilogue_utils import (
    statically_known_equal,
    statically_known_shape_equal,
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


@dataclasses.dataclass(frozen=True)
class FlexGemmLocalReduceMatch:
    """Describe a supported grouped local-reduction value found in the FX graph.

    Attributes:
        value_node: FX node that produces the matched local-reduction value.
        geometry: Group size and GEMM output axis reduced by the value.
    """

    value_node: torch.fx.Node
    geometry: FlexGemmLocalReduceGeometry
    physical_span: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.value_node, torch.fx.Node):
            raise RuntimeError(LOCAL_REDUCE_MATCH_NODE_ERROR)

    @property
    def physical_geometry(self) -> FlexGemmLocalReduceGeometry:
        """Geometry in physical accumulator columns: paired lanes folded into the group."""
        return FlexGemmLocalReduceGeometry(
            self.geometry.group * self.physical_span, self.geometry.axis
        )

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
        """Return the common match when all values share one geometry."""
        if not matches:
            return None
        match = matches[0]
        if any(
            item.geometry != match.geometry or item.physical_span != match.physical_span
            for item in matches
        ):
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
class FlexGemmLocalReduceOutputStorage:
    """Describe the physical storage selected for a returned local reduction."""

    source: torch.fx.Node
    layout: FlexGemmOutputLayout
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
class FlexGemmLocalReduceStore:
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
    def output_layout(self) -> FlexGemmOutputLayout | None:
        """Return the selected physical storage layout."""
        return None if self.output_storage is None else self.output_storage.layout


@dataclasses.dataclass(frozen=True)
class FlexGemmIndexedOutputStore:
    """Describe one terminal row-indexed output stored from the main result."""

    node: torch.fx.Node
    indices: torch.fx.Node
    owned_nodes: tuple[torch.fx.Node, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.node, torch.fx.Node)
            or not isinstance(self.indices, torch.fx.Node)
            or not all(isinstance(node, torch.fx.Node) for node in self.owned_nodes)
        ):
            raise RuntimeError("indexed output plans require tensor nodes")


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


@dataclasses.dataclass(frozen=True)
class FlexGemmOutputPlan:
    """Classify returned values and backend-owned terminal stores."""

    output: torch.fx.Node
    returned_aux_outputs: tuple[torch.fx.Node, ...] = ()
    local_reduce: FlexGemmOutputLocalReducePlan | None = None
    indexed_output: FlexGemmIndexedOutputStore | None = None
    main_transform: FlexGemmGroupedMainOutputTransform | None = None
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
                self.local_reduce is not None
                and not isinstance(self.local_reduce, FlexGemmOutputLocalReducePlan)
            )
            or (
                self.indexed_output is not None
                and not isinstance(self.indexed_output, FlexGemmIndexedOutputStore)
            )
            or (
                self.output_storage is not None
                and not isinstance(self.output_storage, torch.fx.Node)
            )
            or not all(
                isinstance(node, torch.fx.Node) for node in self.output_storage_nodes
            )
            or bool(self.output_storage_nodes) != (self.output_storage is not None)
            or any(
                node not in self.returned_aux_outputs
                for node in self.structural_outputs
            )
        ):
            raise RuntimeError(FLEX_GEMM_OUTPUT_PLAN_NODE_ERROR)

    @property
    def structural_outputs(self) -> tuple[torch.fx.Node, ...]:
        """Return auxiliary values stored by backend-owned EpiOps."""
        store = None if self.local_reduce is None else self.local_reduce.store
        return (
            *(() if self.indexed_output is None else (self.indexed_output.node,)),
            *(() if store is None else (store.node,)),
        )

    @property
    def aux_outputs(self) -> tuple[torch.fx.Node, ...]:
        """Return ordinary auxiliary values emitted by the generated callback."""
        return tuple(
            output
            for output in self.returned_aux_outputs
            if output not in self.structural_outputs
        )

    @property
    def terminal_rewrites(self) -> dict[torch.fx.Node, torch.fx.Node | None]:
        """Map terminal wrappers to aliases or backend-owned omissions."""
        rewrites = dict.fromkeys(self.output_storage_nodes, self.output_storage)
        store = None if self.local_reduce is None else self.local_reduce.store
        if store is not None and store.output_storage is not None:
            rewrites.update(dict.fromkeys(store.output_storage.nodes))
        if self.indexed_output is not None:
            rewrites.update(dict.fromkeys(self.indexed_output.owned_nodes))
        return rewrites


FlexGemmEpilogueGraph = GemmEpilogueGraph


def bind_terminal_output_storage(
    outputs: FlexGemmOutputPlan,
) -> FlexGemmOutputPlan:
    """Record the physical value beneath a terminal same-width dtype view."""
    dtype_view = outputs.output
    owned_nodes = []
    while dtype_view.target is torch.ops.aten.alias.default:
        owned_nodes.append(dtype_view)
        source = dtype_view.args[0]
        if not isinstance(source, torch.fx.Node):
            return outputs
        dtype_view = source
    if dtype_view.target is not torch.ops.aten.view.dtype:
        return outputs
    source, dtype = dtype_view.args
    if not isinstance(source, torch.fx.Node):
        raise NotImplementedError(
            "FlexGEMM terminal dtype views must preserve shape, stride, and element size"
        )
    source_meta = source.meta.get("val")
    output_meta = outputs.output.meta.get("val")
    if (
        not isinstance(source_meta, torch.Tensor)
        or not isinstance(output_meta, torch.Tensor)
        or not isinstance(dtype, torch.dtype)
        or dtype is not output_meta.dtype
        or source_meta.dtype.itemsize != output_meta.dtype.itemsize
        or not statically_known_shape_equal(source_meta.shape, output_meta.shape)
        or not statically_known_shape_equal(source_meta.stride(), output_meta.stride())
    ):
        raise NotImplementedError(
            "FlexGEMM terminal dtype views must preserve shape, stride, and element size"
        )
    owned_nodes.append(dtype_view)
    return dataclasses.replace(
        outputs,
        output_storage=source,
        output_storage_nodes=tuple(owned_nodes),
    )


def terminal_dtype_conversion_source(node: torch.fx.Node) -> torch.fx.Node:
    """Peel one terminal element-type conversion from an output."""
    if node.target not in (
        torch.ops.aten._to_copy.default,
        torch.ops.prims.convert_element_type.default,
    ):
        return node
    source = node.args[0]
    return source if isinstance(source, torch.fx.Node) else node


def flex_gemm_indexed_output_store(
    main_output: torch.fx.Node,
    aux: torch.fx.Node,
) -> FlexGemmIndexedOutputStore | None:
    """Match one terminal row gather.

    Return ``None`` when the graph is not this topology. Raise when a matched
    gather violates a FlexGEMM legality requirement.
    """
    main_meta = main_output.meta.get("val")
    aux_meta = aux.meta.get("val")
    if (
        not isinstance(main_meta, torch.Tensor)
        or not isinstance(aux_meta, torch.Tensor)
        or main_meta.ndim != 2
        or not statically_known_shape_equal(aux_meta.shape, (main_meta.shape[0],))
    ):
        return None

    target = terminal_dtype_conversion_source(aux)
    target_conversion = () if target is aux else (aux,)
    gather_node = squeeze_source_node(target)
    if not isinstance(gather_node, torch.fx.Node):
        return None
    squeeze_dim = target.args[1] if len(target.args) > 1 else None
    if squeeze_dim is not None:
        squeeze_dims = (
            tuple(squeeze_dim)
            if isinstance(squeeze_dim, (tuple, list))
            else (squeeze_dim,)
        )
        if len(squeeze_dims) != 1 or squeeze_dims[0] not in (-1, 1):
            return None
    if gather_node.target is not torch.ops.aten.gather.default:
        return None
    source, dim, unsqueeze_node, *gather_options = gather_node.args
    sparse_grad = (
        gather_options[0]
        if gather_options
        else gather_node.kwargs.get("sparse_grad", False)
    )
    if (
        len(gather_options) > 1
        or not isinstance(source, torch.fx.Node)
        or dim not in (-1, 1)
        or not isinstance(unsqueeze_node, torch.fx.Node)
        or sparse_grad is not False
        or unsqueeze_node.target is not torch.ops.aten.unsqueeze.default
    ):
        return None
    if (
        source not in (main_output, terminal_dtype_conversion_source(main_output))
        or aux_meta.dtype is not main_meta.dtype
    ):
        raise NotImplementedError(FLEX_GEMM_INDEXED_OUTPUT_SOURCE_ERROR)
    indices, unsqueeze_dim = unsqueeze_node.args
    indices_meta = (
        indices.meta.get("val") if isinstance(indices, torch.fx.Node) else None
    )
    gather_meta = gather_node.meta.get("val")
    if (
        unsqueeze_dim not in (-1, 1)
        or not isinstance(indices, torch.fx.Node)
        or indices.op != "placeholder"
        or tuple(indices.users) != (unsqueeze_node,)
        or tuple(unsqueeze_node.users) != (gather_node,)
        or tuple(gather_node.users) != (target,)
        or (target is not aux and tuple(target.users) != (aux,))
        or any(user.op != "output" for user in aux.users)
        or not isinstance(indices_meta, torch.Tensor)
        or indices_meta.dtype not in (torch.int32, torch.int64)
        or not statically_known_shape_equal(indices_meta.shape, aux_meta.shape)
        or not isinstance(gather_meta, torch.Tensor)
        or not statically_known_shape_equal(gather_meta.shape, (main_meta.shape[0], 1))
    ):
        return None
    if not statically_known_equal(indices_meta.stride(0), 1):
        raise NotImplementedError("FlexGEMM indexed output indices must be contiguous")
    return FlexGemmIndexedOutputStore(
        aux,
        indices,
        (unsqueeze_node, gather_node, target, *target_conversion),
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
    grouped_structural_values: dict[
        torch.fx.Node, tuple[FlexGemmStructuralInt, ...]
    ] = dataclasses.field(default_factory=dict)
    matches: dict[torch.fx.Node, FlexGemmLocalReduceMatch] = dataclasses.field(
        default_factory=dict
    )
    tensorssa_facts: dict[torch.fx.Node, FlexGemmTensorSSAFact] = dataclasses.field(
        default_factory=dict
    )
    grouped_main_lanes: dict[torch.fx.Node, "GroupedMainLaneMatch"] = dataclasses.field(
        default_factory=dict
    )
    gemm: torch.fx.Node | None = None
    gemm_shape: tuple[Any, ...] | None = None

    @classmethod
    def from_graph_module(
        cls,
        graph_module: torch.fx.GraphModule,
        gemm: torch.fx.Node | None = None,
    ) -> "FlexGemmLocalReduceAnalysis":
        """Build shared dependency and reduction state in one topological pass."""
        gemm_shape = tensor_meta_shape(gemm) if gemm is not None else None
        analysis = cls(
            FlexGemmEpilogueGraph.from_nodes(tuple(graph_module.graph.nodes)),
            gemm=gemm,
            gemm_shape=gemm_shape,
        )
        for node in graph_module.graph.nodes:
            if node.op == "output":
                break
            analysis.visit_node(node)
        return analysis

    def physical_reduction_nodes(
        self, match: FlexGemmLocalReduceMatch
    ) -> tuple[torch.fx.Node, ...]:
        """Return grouped reductions contributing to a propagated match."""
        dependencies = self.graph.dependencies.get(match.value_node, ())
        return tuple(
            node
            for node in self.graph.dependencies
            if (node is match.value_node or node in dependencies)
            and isinstance(
                self.graph.normalized_nodes.get(node), NormalizedGemmReduction
            )
            and node in self.matches
        )

    def visit_node(self, node: torch.fx.Node) -> None:
        """Record grouped layouts and local-reduction matches for one FX node."""
        if node.op != "call_function":
            return
        view_args = view_or_reshape_args(node)
        if view_args is not None:
            source_node, shape = view_args
            propagated = self.propagate_local_reduce_match(node, source_node)
            fact = self.propagate_tensorssa_view(node, source_node)
            grouped = self.bind_grouped_layout(node, shape, source_node)
            if propagated or fact or grouped:
                return
        normalized = self.graph.normalized_nodes.get(node)
        if isinstance(normalized, NormalizedGemmReduction):
            if self.bind_grouped_reduction(node, normalized):
                return
            if (
                normalized.source not in self.grouped_tensors
                and self.gemm is not None
                and self.graph.depends_on(normalized.source, self.gemm)
            ):
                op_name = (
                    "softmax/logsumexp"
                    if isinstance(normalized, NormalizedPrepareSoftmax)
                    else str(getattr(node.target, "overloadpacket", node.target))
                )
                raise ungrouped_reduction_error(op_name)
        elif isinstance(normalized, NormalizedUnsupportedReduction):
            raise unsupported_reduction_op_error(normalized.target)
        if self.propagate_tensorssa_storage_select(node):
            return
        lane_fact = self.bind_grouped_main_lane_fact(node)
        squeeze_source = squeeze_source_node(node)
        propagated_match = self.propagate_local_reduce_match(node, squeeze_source)
        propagated_fact = self.propagate_tensorssa_view(node, squeeze_source)
        if propagated_match or propagated_fact:
            return
        if node.target is operator.getitem and self.propagate_local_reduce_match(
            node, node.args[0]
        ):
            return
        if lane_fact:
            return
        if is_shape_preserving_pointwise_node(node):
            self.propagate_pointwise_match(node, LOCAL_REDUCE_MIXED_MATCH_ERROR)
            self.propagate_tensorssa_pointwise(node)

    def propagate_tensorssa_view(self, node: torch.fx.Node, source: Any) -> bool:
        """Propagate a logical TensorSSA fact through a numel-preserving view."""
        if not isinstance(source, torch.fx.Node):
            return False
        fact = self.tensorssa_facts.get(source)
        source_shape = tensor_meta_shape(source)
        output_shape = tensor_meta_shape(node)
        if fact is None or source_shape is None or output_shape is None:
            return False
        if not statically_known_equal(math.prod(source_shape), math.prod(output_shape)):
            return False
        self.tensorssa_facts[node] = fact
        return True

    def propagate_tensorssa_storage_select(self, node: torch.fx.Node) -> bool:
        """Track one logical slot selected for a packed main-output element."""
        if node.target is not torch.ops.aten.select.int:
            return False
        source, dim, index = node.args
        if not isinstance(source, torch.fx.Node):
            return False
        fact = self.tensorssa_facts.get(source)
        source_shape = tensor_meta_shape(source)
        output_shape = tensor_meta_shape(node)
        structural_dim = FlexGemmStructuralInt.from_value(dim)
        structural_index = FlexGemmStructuralInt.from_value(index)
        if (
            fact is None
            or not fact.complete
            or fact.storage_span != 1
            or source_shape is None
            or output_shape is None
            or structural_dim is None
            or structural_dim.symbolic is not None
            or structural_index is None
            or structural_index.symbolic is not None
            or structural_dim.value % len(source_shape) != len(source_shape) - 1
        ):
            return False
        storage_span = FlexGemmStructuralInt.from_value(source_shape[-1])
        if (
            storage_span is None
            or storage_span.symbolic is not None
            or storage_span.value != NESTED_TENSORSSA_PACKED_STORAGE_SPAN
            or fact.physical_span != NESTED_TENSORSSA_PHYSICAL_SPAN
            or not -storage_span.value <= structural_index.value < storage_span.value
        ):
            return False
        self.tensorssa_facts[node] = dataclasses.replace(
            fact,
            storage_span=storage_span.value,
            storage_offsets=frozenset((structural_index.value % storage_span.value,)),
        )
        return True

    def bind_grouped_main_lane_fact(self, node: torch.fx.Node) -> bool:
        """Record one selected physical lane as a logical TensorSSA value."""
        if self.gemm is None:
            return False
        match = grouped_main_lane_match(node, self.gemm, self)
        if match is None:
            return False
        self.grouped_main_lanes[node] = match
        self.tensorssa_facts[node] = FlexGemmTensorSSAFact(
            canonical_grouped_main_source(match.source, self.gemm, self),
            match.group,
            match.chunked,
            frozenset((match.index % match.group,)),
        )
        return True

    def propagate_tensorssa_pointwise(self, node: torch.fx.Node) -> bool:
        """Merge compatible logical lane facts through one enumerated pointwise op."""
        inputs = tuple(iter_fx_node_inputs((node.args, node.kwargs)))
        facts = [
            self.tensorssa_facts[input_node]
            for input_node in inputs
            if input_node in self.tensorssa_facts
        ]
        if not facts:
            return False
        first = facts[0]
        if any(
            fact.root is not first.root
            or fact.physical_span != first.physical_span
            or fact.chunked != first.chunked
            or fact.storage_span != first.storage_span
            for fact in facts[1:]
        ):
            raise NotImplementedError(FLEX_GEMM_NESTED_TENSORSSA_LANES_ERROR)
        external = frozenset(
            input_node
            for input_node in inputs
            if input_node not in self.tensorssa_facts
            and tensor_meta_shape(input_node) is not None
        )
        self.tensorssa_facts[node] = FlexGemmTensorSSAFact(
            root=first.root,
            physical_span=first.physical_span,
            chunked=first.chunked,
            lane_offsets=frozenset().union(*(fact.lane_offsets for fact in facts)),
            storage_span=first.storage_span,
            storage_offsets=frozenset().union(
                *(fact.storage_offsets for fact in facts)
            ),
            reduced=any(fact.reduced for fact in facts),
            external_tensor_inputs=external.union(
                *(fact.external_tensor_inputs for fact in facts)
            ),
        )
        return True

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

    def tensorssa_reduction_physical_span(
        self, source: torch.fx.Node, axis: int
    ) -> int:
        """Return the supported physical span for a logical grouped reduction."""
        fact = self.tensorssa_facts.get(source)
        if fact is None:
            return 1
        if (
            not fact.complete
            or fact.storage_span != 1
            or fact.physical_span != NESTED_TENSORSSA_PHYSICAL_SPAN
        ):
            raise NotImplementedError(FLEX_GEMM_NESTED_TENSORSSA_LANES_ERROR)
        if axis != 1:
            raise NotImplementedError(
                "nested TensorSSA physical spans support logical axis N only"
            )
        return fact.physical_span

    def bind_grouped_reduction(
        self,
        node: torch.fx.Node,
        reduction: NormalizedGemmReduction,
    ) -> bool:
        """Match and record a reduction over a grouped TensorSSA layout."""
        layout = self.grouped_tensors.get(reduction.source)
        if layout is None:
            return False
        if isinstance(reduction, NormalizedReduction) and reduction.dtype is not None:
            raise NotImplementedError(LOCAL_REDUCE_EXPLICIT_DTYPE_ERROR)
        validate_local_reduce_tensorssa_group_size(layout.axis, layout.group_size)
        if not layout.matches_reduction_dim(reduction.dim):
            if isinstance(reduction, NormalizedPrepareSoftmax):
                return False
            raise NotImplementedError(LOCAL_REDUCE_INNERMOST_GROUPED_DIM_ERROR)
        source_fact = self.tensorssa_facts.get(reduction.source)
        self.matches[node] = FlexGemmLocalReduceMatch(
            node,
            FlexGemmLocalReduceGeometry(layout.group_size, layout.axis),
            self.tensorssa_reduction_physical_span(reduction.source, layout.axis),
        )
        if source_fact is not None:
            self.tensorssa_facts[node] = dataclasses.replace(source_fact, reduced=True)
        return True

    def has_physical_grouped_input(self, value: Any) -> bool:
        """Return whether a value needs a cross-fragment grouped combine."""
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
            structural_values = tuple(
                structural
                for arg in iter_fx_node_inputs((node.args, node.kwargs))
                for structural in self.grouped_structural_values.get(arg, ())
            )
            if structural_values:
                self.grouped_structural_values[node] = structural_values
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

    def commit_output_guards(self, outputs: FlexGemmOutputPlan) -> None:
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
            geometry = FlexGemmLocalReduceGeometry(layout.group_size, layout.axis)
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
        value: Any,
        grouped_source: torch.fx.Node,
        layout: GroupedTensorSSALayout,
    ) -> FlexGemmLocalReduceMatch | None:
        """Find the grouped reduction that produces a broadcast value."""
        if not isinstance(value, torch.fx.Node):
            return None
        reduction = self.graph.normalized_nodes.get(value)
        if isinstance(reduction, NormalizedReduction):
            if reduction.source is not grouped_source:
                if self.graph.depends_on(reduction.source, grouped_source):
                    if not (
                        layout.axis == 1
                        and layout.group_size <= LOCAL_REDUCE_FRAGMENT_WIDTH
                        and is_shape_preserving_pointwise_node(reduction.source)
                    ):
                        raise NotImplementedError(LOCAL_REDUCE_SOURCE_EXPRESSION_ERROR)
                else:
                    raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
            if (
                reduction.dtype is not None
                or not reduction.keepdim
                or not layout.matches_reduction_dim(reduction.dim)
            ):
                raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
            bound_match = self.matches.get(value)
            if bound_match is not None:
                return bound_match
            return FlexGemmLocalReduceMatch(
                value,
                FlexGemmLocalReduceGeometry(layout.group_size, layout.axis),
                self.tensorssa_reduction_physical_span(reduction.source, layout.axis),
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
        reduction = self.graph.normalized_nodes.get(value)
        if isinstance(reduction, NormalizedReduction):
            self.validate_hidden_feed_main_reduction_input(
                reduction.source, grouped_source
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
        reduction = self.graph.normalized_nodes.get(match.value_node)
        if isinstance(reduction, NormalizedReduction):
            self.validate_feed_main_source_reductions(
                source, reduction.source, match.value_node
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
        reduction = self.graph.normalized_nodes.get(value)
        if isinstance(reduction, NormalizedReduction):
            return (
                reduction.dtype is None
                and bool(reduction.keepdim)
                and layout.matches_reduction_dim(reduction.dim)
                and (
                    reduction.source is grouped_source
                    or self.graph.depends_on(reduction.source, grouped_source)
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
                return self.match_feed_value(value, grouped_source, layout)
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
    ) -> FlexGemmOutputLocalReducePlan | None:
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
            match.physical_geometry.group,
            match.geometry.axis,
        )
        if not statically_known_shape_equal(expected_aux_shape, value_meta.shape):
            return None
        if output_storage is not None:
            output_storage.layout.validate_geometry(match.geometry)
        return match.to_plan(
            store=FlexGemmLocalReduceStore(aux, output_storage),
            feeds_main=False,
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
            local_reduce=match.to_plan(store=None, feeds_main=True),
        )


def flex_gemm_indexed_output_plan(
    output: Any,
    aux_outputs: tuple[Any, ...],
) -> FlexGemmIndexedOutputStore | None:
    """Return the unique indexed auxiliary output store, if any."""
    if not isinstance(output, torch.fx.Node):
        return None
    indexed_plans = tuple(
        plan
        for aux_output in aux_outputs
        if isinstance(aux_output, torch.fx.Node)
        if (plan := flex_gemm_indexed_output_store(output, aux_output)) is not None
    )
    if len(indexed_plans) > 1:
        raise NotImplementedError("FlexGEMM supports one indexed row output")
    return indexed_plans[0] if indexed_plans else None


def tuple_output_plan(
    output: Any,
    aux_outputs: tuple[Any, ...],
    analysis: FlexGemmLocalReduceAnalysis,
) -> FlexGemmOutputPlan:
    """Classify ordinary and backend-owned auxiliary outputs."""
    if not isinstance(output, torch.fx.Node) or not all(
        isinstance(aux_output, torch.fx.Node) for aux_output in aux_outputs
    ):
        raise NotImplementedError(FLEX_GEMM_OUTPUT_TENSOR_ERROR)

    indexed_output = flex_gemm_indexed_output_plan(output, aux_outputs)
    indexed_node = None if indexed_output is None else indexed_output.node
    non_indexed_aux_outputs = tuple(
        aux_output for aux_output in aux_outputs if aux_output is not indexed_node
    )

    feed_match = analysis.common_feed_main_match((output, *non_indexed_aux_outputs))
    compressed_aux_plans = tuple(
        plan
        for aux_output in non_indexed_aux_outputs
        if (plan := analysis.compressed_aux_plan(output, aux_output)) is not None
    )
    if len(compressed_aux_plans) > 1:
        raise NotImplementedError(LOCAL_REDUCE_MIXED_MATCH_ERROR)
    if compressed_aux_plans:
        compressed_aux_plan = compressed_aux_plans[0]
        compressed_match = compressed_aux_plan.match
        compressed_reductions = analysis.physical_reduction_nodes(compressed_match)
        if feed_match is None and any(
            analysis.graph.depends_on(output, reduction)
            for reduction in compressed_reductions
        ):
            feed_match = compressed_match
        if compressed_match.physical_span > 1 and feed_match is None:
            raise NotImplementedError(
                "nested TensorSSA reductions must feed the main output"
            )
        if feed_match is not None:
            if OrderedSet(analysis.physical_reduction_nodes(feed_match)) != OrderedSet(
                compressed_reductions
            ):
                raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
            store = compressed_aux_plan.store
            if store is None:
                raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
            compressed_aux_plan = feed_match.to_plan(store=store, feeds_main=True)
        return FlexGemmOutputPlan(
            output,
            aux_outputs,
            local_reduce=compressed_aux_plan,
            indexed_output=indexed_output,
        )
    feed_main_plan = analysis.feed_main_output_plan(output, non_indexed_aux_outputs)
    if feed_main_plan is not None:
        return dataclasses.replace(
            feed_main_plan,
            returned_aux_outputs=aux_outputs,
            indexed_output=indexed_output,
        )
    return FlexGemmOutputPlan(
        output,
        aux_outputs,
        indexed_output=indexed_output,
    )


def flex_gemm_output_values(
    graph_module: torch.fx.GraphModule,
) -> tuple[Any, tuple[Any, ...]]:
    """Return the main value and ordered auxiliary values from the FX output."""
    output_nodes = [node for node in graph_module.graph.nodes if node.op == "output"]
    if len(output_nodes) != 1:
        raise NotImplementedError("FlexGEMM expects one output node")
    output_value = output_nodes[0].args[0]
    if isinstance(output_value, (tuple, list)):
        if not output_value:
            raise NotImplementedError("FlexGEMM expects one tensor output")
        return output_value[0], tuple(output_value[1:])
    return output_value, ()


def output_plan(
    graph_module: torch.fx.GraphModule,
    local_reduce: FlexGemmLocalReduceAnalysis,
) -> FlexGemmOutputPlan:
    """Classify output consumers from one shared local-reduce analysis."""
    output_value, aux_outputs = flex_gemm_output_values(graph_module)
    if aux_outputs:
        return tuple_output_plan(output_value, aux_outputs, local_reduce)
    if not isinstance(output_value, torch.fx.Node):
        raise NotImplementedError("FlexGEMM expects one tensor output")
    feed_main_plan = local_reduce.feed_main_output_plan(output_value)
    return (
        FlexGemmOutputPlan(output_value) if feed_main_plan is None else feed_main_plan
    )


def validate_output_storage_transforms(
    graph: FlexGemmEpilogueGraph,
    outputs: FlexGemmOutputPlan,
) -> None:
    """Require every storage transform to belong to the output plan."""
    store = None if outputs.local_reduce is None else outputs.local_reduce.store
    selected_node = (
        store.node if store is not None and store.output_storage is not None else None
    )
    if any(
        match_flex_gemm_local_reduce_output_storage(node) is not None
        and node is not selected_node
        for node in graph.dependencies
    ):
        raise NotImplementedError("output layout transforms must be returned directly")


@dataclasses.dataclass(frozen=True)
class GroupedMainLaneMatch:
    """Describe one grouped-main lane before complete-output validation."""

    source: torch.fx.Node
    group: int
    chunked: bool
    index: int
    layout_node: torch.fx.Node
    structural_values: tuple[FlexGemmStructuralInt, ...] = ()


@dataclasses.dataclass(frozen=True)
class GroupedMainOutputMatch:
    """Describe one complete grouped-main output and its lowering metadata."""

    transform: FlexGemmGroupedMainOutputTransform
    select_indices: dict[torch.fx.Node, int]
    layouts: dict[torch.fx.Node, GroupedTensorSSALayout]
    structural_values: tuple[FlexGemmStructuralInt, ...] = ()

    def commit_guards(self) -> None:
        """Specialize backed structural values after complete validation."""
        for structural in self.structural_values:
            structural.guard()


def canonical_grouped_main_source(
    node: torch.fx.Node,
    gemm: torch.fx.Node,
    local_reduce: FlexGemmLocalReduceAnalysis,
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


def grouped_main_lane_match(
    node: torch.fx.Node,
    gemm: torch.fx.Node,
    local_reduce: FlexGemmLocalReduceAnalysis,
) -> GroupedMainLaneMatch | None:
    """Match one interleaved select or contiguous split lane."""
    if node.op == "call_function" and node.target is torch.ops.aten.select.int:
        view = node.args[0]
        if not isinstance(view, torch.fx.Node):
            return None
        view_args = view_or_reshape_args(view)
        dim = FlexGemmStructuralInt.from_value(node.args[1])
        index = FlexGemmStructuralInt.from_value(node.args[2])
        shape = tensor_meta_shape(view)
        if (
            view_args is None
            or shape is None
            or len(shape) != 3
            or dim is None
            or index is None
            or not local_reduce.graph.depends_on(view_args[0], gemm)
        ):
            return None
        selected_dim = dim.value % len(shape)
        structural_values = [dim, index]
        if selected_dim == len(shape) - 1:
            layout = local_reduce.grouped_tensors.get(view)
            if layout is None or layout.axis != 1:
                return None
            group, chunked = layout.group_size, False
        elif selected_dim == 1:
            structural_group = FlexGemmStructuralInt.from_value(shape[1])
            source_shape = tensor_meta_shape(view_args[0])
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
        return GroupedMainLaneMatch(
            view_args[0],
            group,
            chunked,
            index.value,
            view,
            tuple(structural_values),
        )

    if node.op != "call_function" or node.target is not operator.getitem:
        return None
    split = node.args[0]
    index = FlexGemmStructuralInt.from_value(node.args[1])
    if (
        not isinstance(split, torch.fx.Node)
        or split.op != "call_function"
        or split.target is not torch.ops.aten.split.Tensor
        or index is None
    ):
        return None
    source = split.args[0]
    split_size = FlexGemmStructuralInt.from_value(split.args[1])
    dim = split.args[2] if len(split.args) > 2 else split.kwargs.get("dim", 0)
    shape = tensor_meta_shape(source) if isinstance(source, torch.fx.Node) else None
    if (
        not isinstance(source, torch.fx.Node)
        or shape is None
        or len(shape) != 2
        or not isinstance(shape[-1], int)
        or split_size is None
        or split_size.value <= 0
        or shape[-1] % split_size.value
        or dim not in (-1, 1)
        or not local_reduce.graph.depends_on(source, gemm)
    ):
        return None
    group = shape[-1] // split_size.value
    if group <= 1:
        return None
    return GroupedMainLaneMatch(
        source,
        group,
        True,
        index.value,
        split,
        (index, split_size),
    )


def nested_grouped_main_output_match(
    output: torch.fx.Node,
    gemm: torch.fx.Node,
    local_reduce: FlexGemmLocalReduceAnalysis,
) -> GroupedMainOutputMatch | None:
    """Build a grouped-main match from accepted forward TensorSSA facts."""
    fact = local_reduce.tensorssa_facts.get(output)
    if fact is None or fact.physical_span == 1 or not fact.reduced:
        return None
    if fact.external_tensor_inputs:
        raise NotImplementedError(FLEX_GEMM_NESTED_TENSORSSA_CAPTURE_ERROR)
    if not fact.complete:
        raise NotImplementedError(FLEX_GEMM_NESTED_TENSORSSA_LANES_ERROR)
    lane_nodes = (
        output,
        *local_reduce.graph.dependencies.get(output, ()),
    )
    lanes = tuple(
        (node, local_reduce.grouped_main_lanes[node])
        for node in lane_nodes
        if node in local_reduce.grouped_main_lanes
    )
    if not lanes:
        return None
    if any(match.group != fact.physical_span for _, match in lanes):
        raise NotImplementedError(FLEX_GEMM_NESTED_TENSORSSA_LANES_ERROR)
    select_indices = {node: match.index % fact.physical_span for node, match in lanes}
    layouts = {
        match.layout_node: GroupedTensorSSALayout(1, fact.physical_span)
        for _, match in lanes
    }
    if fact.storage_span > 1:
        storage_selects = tuple(
            (node, local_reduce.tensorssa_facts[node])
            for node in lane_nodes
            if node.target is torch.ops.aten.select.int
            and node in local_reduce.tensorssa_facts
            and local_reduce.tensorssa_facts[node].storage_span == fact.storage_span
        )
        storage_sources = OrderedSet(
            node.args[0]
            for node, _ in storage_selects
            if isinstance(node.args[0], torch.fx.Node)
        )
        storage_offsets = frozenset(
            next(iter(selected.storage_offsets))
            for _, selected in storage_selects
            if len(selected.storage_offsets) == 1
        )
        if (
            len(storage_sources) != 1
            or len(storage_offsets) != len(storage_selects)
            or storage_offsets != frozenset(range(fact.storage_span))
        ):
            raise NotImplementedError(FLEX_GEMM_NESTED_TENSORSSA_LANES_ERROR)
        storage_source = next(iter(storage_sources))
        layouts[storage_source] = GroupedTensorSSALayout(1, fact.storage_span)
        select_indices.update(
            {
                node: next(iter(selected.storage_offsets))
                for node, selected in storage_selects
            }
        )
    gemm_meta = gemm.meta.get("val")
    output_meta = output.meta.get("val")
    if not isinstance(gemm_meta, torch.Tensor) or not isinstance(
        output_meta, torch.Tensor
    ):
        return None
    expected_shape = (gemm_meta.shape[0], gemm_meta.shape[1] // fact.output_span)
    if not statically_known_shape_equal(output_meta.shape, expected_shape):
        raise NotImplementedError(FLEX_GEMM_GROUPED_MAIN_SHAPE_ERROR)
    return GroupedMainOutputMatch(
        FlexGemmGroupedMainOutputTransform(fact.output_span, fact.chunked),
        select_indices,
        layouts,
        tuple(
            structural for _, match in lanes for structural in match.structural_values
        ),
    )


def grouped_main_output_match(
    output: torch.fx.Node,
    gemm: torch.fx.Node,
    local_reduce: FlexGemmLocalReduceAnalysis,
) -> GroupedMainOutputMatch | None:
    """Recognize a complete adjacent-N grouped main-output expression."""
    nested = nested_grouped_main_output_match(output, gemm, local_reduce)
    if nested is not None:
        return nested
    lanes: list[tuple[torch.fx.Node, GroupedMainLaneMatch]] = []
    seen: OrderedSet[torch.fx.Node] = OrderedSet()
    pending: list[Any] = [output]
    while pending:
        node = pending.pop()
        if not isinstance(node, torch.fx.Node) or node in seen:
            continue
        seen.add(node)
        match = local_reduce.grouped_main_lanes.get(node)
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
    canonical_source = canonical_grouped_main_source(first.source, gemm, local_reduce)
    indices: OrderedSet[int] = OrderedSet()
    select_indices: dict[torch.fx.Node, int] = {}
    layouts: dict[torch.fx.Node, GroupedTensorSSALayout] = {}
    structural_values: list[FlexGemmStructuralInt] = []
    for node, match in lanes:
        if (
            canonical_grouped_main_source(match.source, gemm, local_reduce)
            is not canonical_source
            or match.group != first.group
            or match.chunked != first.chunked
            or not -first.group <= match.index < first.group
        ):
            return None
        index = match.index % first.group
        indices.add(index)
        select_indices[node] = index
        layouts[match.layout_node] = GroupedTensorSSALayout(1, first.group)
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
        raise NotImplementedError(FLEX_GEMM_GROUPED_MAIN_SHAPE_ERROR)
    return GroupedMainOutputMatch(
        FlexGemmGroupedMainOutputTransform(first.group, first.chunked),
        select_indices,
        layouts,
        tuple(structural_values),
    )


@dataclasses.dataclass(frozen=True)
class FlexGemmEpilogueAnalysis:
    """Bundle the immutable analysis consumed by FlexGEMM lowering and emission.

    Attributes:
        outputs: Classification of main, auxiliary, and local-reduction outputs.
        local_reduce: Grouped layouts and local-reduction matches from the FX graph.
    """

    gemm: torch.fx.Node
    outputs: FlexGemmOutputPlan
    local_reduce: FlexGemmLocalReduceAnalysis
    grouped_select_indices: dict[torch.fx.Node, int] = dataclasses.field(
        default_factory=dict
    )
    grouped_main_layouts: dict[torch.fx.Node, GroupedTensorSSALayout] = (
        dataclasses.field(default_factory=dict)
    )

    @classmethod
    def from_graph_module(
        cls, graph_module: torch.fx.GraphModule, gemm: torch.fx.Node
    ) -> "FlexGemmEpilogueAnalysis":
        """Analyze reductions and an optional grouped main-output transform."""
        local_reduce = FlexGemmLocalReduceAnalysis.from_graph_module(graph_module, gemm)
        outputs = bind_terminal_output_storage(output_plan(graph_module, local_reduce))
        validate_output_storage_transforms(local_reduce.graph, outputs)
        if outputs.indexed_output is not None and outputs.output_storage_nodes:
            raise NotImplementedError(
                "FlexGEMM indexed outputs do not compose with terminal dtype views"
            )
        grouped_main = grouped_main_output_match(
            outputs.output_storage or outputs.output,
            gemm,
            local_reduce,
        )
        if grouped_main is None:
            gemm_meta = gemm.meta.get("val")
            output_meta = outputs.output.meta.get("val")
            if (
                isinstance(gemm_meta, torch.Tensor)
                and isinstance(output_meta, torch.Tensor)
                and not statically_known_shape_equal(output_meta.shape, gemm_meta.shape)
            ):
                raise NotImplementedError(FLEX_GEMM_MAIN_OUTPUT_SHAPE_ERROR)
            local_reduce.commit_output_guards(outputs)
            return cls(gemm, outputs, local_reduce)
        if outputs.aux_outputs or outputs.indexed_output is not None:
            raise NotImplementedError(FLEX_GEMM_GROUPED_MAIN_COMPOSITION_ERROR)
        if (
            grouped_main.transform.chunked
            and outputs.local_reduce is not None
            and outputs.local_reduce.match.physical_span == 1
        ):
            raise NotImplementedError(
                "chunked grouped main outputs do not compose with grouped reductions"
            )
        grouped_main.commit_guards()
        local_reduce.commit_output_guards(outputs)
        return cls(
            gemm,
            dataclasses.replace(outputs, main_transform=grouped_main.transform),
            local_reduce,
            grouped_main.select_indices,
            grouped_main.layouts,
        )

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
                FlexGemmLocalReduceGeometry(self.outputs.main_transform.group, 1)
            )
        return tuple(geometries)


def analyze_flex_gemm_epilogue(
    graph_module: torch.fx.GraphModule,
    gemm: torch.fx.Node,
) -> FlexGemmEpilogueAnalysis:
    """Analyze FlexGEMM body for output planning and epilogue code generation.

    This is the analysis entry point called by FlexGEMM lowering. It builds a
    dependency index, performs topological local-reduction analysis, and
    returns the shared immutable plan consumed by the QuACK EpiMod emitter.

    Args:
        graph_module: FlexGEMM body graph containing GEMM and epilogue nodes.

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


def flex_gemm_epilogue_arg(value: Any, env: dict[torch.fx.Node, Any]) -> Any:
    """Adapt one FlexGEMM FX value to the shared CuTeDSL expression frontend."""
    return gemm_epilogue_arg(value, env, "FlexGEMM")


class FlexGemmTensorSSAOpOverrides(GemmEpilogueCuteDSLOpOverrides):
    """Add PyTorch NaN propagation to the shared TensorSSA operation lowering."""

    @staticmethod
    def nan_propagating_minmax(a: Any, b: Any, op: str) -> Any:
        """Apply an IEEE min or max that propagates NaN in one operation."""
        match op:
            case "min":
                op_name, index_expr_fn = "min", Min
            case "max":
                op_name, index_expr_fn = "max", Max
            case _:
                raise AssertionError(f"unexpected minmax op: {op}")
        return CuteDSLOpOverrides._apply_binary_op(
            a,
            b,
            f"cutlass.{op_name}({{a}}, {{b}})",
            index_expr_fn,
        )

    @staticmethod
    def minimum(a: Any, b: Any) -> Any:
        return FlexGemmTensorSSAOpOverrides.nan_propagating_minmax(a, b, "min")

    @staticmethod
    def maximum(a: Any, b: Any) -> Any:
        return FlexGemmTensorSSAOpOverrides.nan_propagating_minmax(a, b, "max")

    @staticmethod
    def clamp(x: Any, min: Any = None, max: Any = None) -> Any:
        result = x
        if min is not None:
            result = FlexGemmTensorSSAOpOverrides.maximum(result, min)
        if max is not None:
            result = FlexGemmTensorSSAOpOverrides.minimum(result, max)
        return result

    @staticmethod
    def clamp_min(x: Any, min: Any) -> Any:
        return FlexGemmTensorSSAOpOverrides.maximum(x, min)

    @staticmethod
    def clamp_max(x: Any, max: Any) -> Any:
        return FlexGemmTensorSSAOpOverrides.minimum(x, max)


class FlexGemmScalarCallbackOpOverrides(CuteDSLOpOverrides):
    """Emit per-element QuACK ``epi_math`` expressions for scalar callbacks.

    The generated epilogue body is TensorSSA-only; this emitter serves QuACK's
    scalar callback ABI, which invokes grouped-reduction finalizers and prepass
    functions once per element.
    """

    def __init__(self, fast_math: bool) -> None:
        self.fast_math = fast_math

    @staticmethod
    def _expr(value: Any) -> str:
        return CuteDSLOpOverrides._as_expr(value)

    @classmethod
    def _binary_exprs(cls, a: Any, b: Any) -> tuple[str, str]:
        a_expr, b_expr = cls._expr(a), cls._expr(b)
        a_var = CuteDSLOpOverrides._get_cse_var(a)
        b_var = CuteDSLOpOverrides._get_cse_var(b)
        if (
            a_var is not None
            and a_var.dtype is not None
            and a_var.dtype.is_floating_point
        ):
            if b_var is None and isinstance(b, (int, float)):
                b_expr = repr(float(b))
        if (
            b_var is not None
            and b_var.dtype is not None
            and b_var.dtype.is_floating_point
        ):
            if a_var is None and isinstance(a, (int, float)):
                a_expr = repr(float(a))
        return a_expr, b_expr

    @classmethod
    def add(cls, a: Any, b: Any, *, alpha: Any = 1) -> str:
        a_expr, b_expr = cls._binary_exprs(a, b)
        rhs = b_expr if alpha == 1 else f"({b_expr} * {alpha})"
        return f"({a_expr} + {rhs})"

    @classmethod
    def sub(cls, a: Any, b: Any, *, alpha: Any = 1) -> str:
        a_expr, b_expr = cls._binary_exprs(a, b)
        rhs = b_expr if alpha == 1 else f"({b_expr} * {alpha})"
        return f"({a_expr} - {rhs})"

    @classmethod
    def mul(cls, a: Any, b: Any) -> str:
        a_expr, b_expr = cls._binary_exprs(a, b)
        return f"({a_expr} * {b_expr})"

    def truediv(self, a: Any, b: Any) -> str:
        a_expr, b_expr = self._binary_exprs(a, b)
        return f"epi_math.divide({a_expr}, {b_expr}, fast={self.fast_math!r})"

    @classmethod
    def neg(cls, x: Any) -> str:
        return f"(-{cls._expr(x)})"

    def _unary_math(self, name: str, x: Any) -> str:
        return f"epi_math.{name}({self._expr(x)}, fast={self.fast_math!r})"

    def abs(self, x: Any) -> str:
        return self._unary_math("abs", x)

    def exp(self, x: Any) -> str:
        return self._unary_math("exp", x)

    def sqrt(self, x: Any) -> str:
        return self._unary_math("sqrt", x)

    def rsqrt(self, x: Any) -> str:
        return self._unary_math("rsqrt", x)

    def log(self, x: Any) -> str:
        return self._unary_math("log", x)

    def erf(self, x: Any) -> str:
        return self._unary_math("erf", x)

    # pyrefly: ignore [bad-override]
    def tanh(self, x: Any) -> str:
        return self._unary_math("tanh", x)

    def reciprocal(self, x: Any) -> str:
        return self._unary_math("reciprocal", x)

    def log1p(self, x: Any) -> str:
        return self._unary_math("log1p", x)

    def sigmoid(self, x: Any) -> str:
        return self._unary_math("sigmoid", x)

    def relu(self, x: Any) -> str:
        return self._unary_math("relu", x)

    @classmethod
    def _binary_math(cls, name: str, a: Any, b: Any) -> str:
        a_expr, b_expr = cls._binary_exprs(a, b)
        return f"epi_math.{name}({a_expr}, {b_expr})"

    @classmethod
    def maximum(cls, a: Any, b: Any) -> str:
        return cls._binary_math("maximum", a, b)

    @classmethod
    def minimum(cls, a: Any, b: Any) -> str:
        return cls._binary_math("minimum", a, b)

    @classmethod
    def where(cls, condition: Any, a: Any, b: Any) -> str:
        a_expr, b_expr = cls._binary_exprs(a, b)
        return f"epi_math.where({cls._expr(condition)}, {a_expr}, {b_expr})"

    @classmethod
    # pyrefly: ignore [bad-override]
    def logical_not(cls, x: Any) -> str:
        return f"epi_math.logical_not({cls._expr(x)})"

    @classmethod
    # pyrefly: ignore [bad-override]
    def bitwise_and(cls, a: Any, b: Any) -> str:
        return f"({cls._expr(a)} & {cls._expr(b)})"

    @classmethod
    def eq(cls, a: Any, b: Any) -> str:
        return cls._binary_math("eq", a, b)

    @classmethod
    def ne(cls, a: Any, b: Any) -> str:
        return cls._binary_math("ne", a, b)

    @classmethod
    def lt(cls, a: Any, b: Any) -> str:
        return cls._binary_math("lt", a, b)

    @classmethod
    def le(cls, a: Any, b: Any) -> str:
        return cls._binary_math("le", a, b)

    @classmethod
    def gt(cls, a: Any, b: Any) -> str:
        return cls._binary_math("gt", a, b)

    @classmethod
    def ge(cls, a: Any, b: Any) -> str:
        return cls._binary_math("ge", a, b)

    @classmethod
    def pow(cls, a: Any, b: Any) -> str:
        if cls._expr(b) not in ("2", "2.0"):
            raise NotImplementedError(
                "FlexGEMM EpiMod currently supports only square operations"
            )
        a_expr = cls._expr(a)
        return f"({a_expr} * {a_expr})"

    @classmethod
    # pyrefly: ignore [bad-override]
    def to_dtype(cls, x: Any, dtype: Any, **kwargs: Any) -> str:
        unsupported_kwargs = {
            key: value
            for key, value in kwargs.items()
            if value not in (None, False, torch.preserve_format)
        }
        if unsupported_kwargs:
            raise NotImplementedError(
                f"unsupported FlexGEMM EpiMod cast options: {unsupported_kwargs}"
            )
        x_expr = cls._expr(x)
        if dtype is torch.bool:
            return f"epi_math.ne({x_expr}, 0)"
        return f"epi_math.to_dtype({x_expr}, {dtype})"

    @classmethod
    def _to_copy(cls, x: Any, *, dtype: Any, **kwargs: Any) -> str:
        return cls.to_dtype(x, dtype, **kwargs)

    @classmethod
    def convert_element_type(cls, x: Any, dtype: Any) -> str:
        return cls.to_dtype(x, dtype)

    @classmethod
    def clamp(cls, x: Any, min: Any = None, max: Any = None) -> str:
        x_expr = cls._expr(x)
        options = []
        if min is not None:
            options.append(f"min={cls._binary_exprs(x, min)[1]}")
        if max is not None:
            options.append(f"max={cls._binary_exprs(x, max)[1]}")
        suffix = f", {', '.join(options)}" if options else ""
        return f"epi_math.clamp({x_expr}{suffix})"

    @classmethod
    def clamp_min(cls, x: Any, min: Any) -> str:
        return cls._binary_math("clamp_min", x, min)

    @classmethod
    def clamp_max(cls, x: Any, max: Any) -> str:
        return cls._binary_math("clamp_max", x, max)


@dataclasses.dataclass(frozen=True)
class FlexGemmEpiModSource:
    """Generated QuACK function plus optional grouped-reduction semantics."""

    name: str
    source: str
    local_reduce_combine: str | None = None
    local_reduce_finalize: str | None = None
    local_reduce_store_finalize: str | None = None
    local_reduce_prepass_combine: str | None = None
    local_reduce_prepass_finalize: str | None = None
    local_reduce_planes: int = 1
    local_reduce_fragment_reduced: bool = False


def online_softmax_combine_body(fast_math: bool) -> tuple[str, ...]:
    """Cross-fragment combine of (running max, rescaled sum) state, NaN-propagating."""
    return (
        "maximum = cute.arch.fmax(lhs[0], rhs[0], nan=True)",
        "one = cutlass.Float32(1.0)",
        "lhs_scale = cutlass.select_(",
        "    lhs[0] == maximum,",
        "    one,",
        f"    epi_math.exp(lhs[0] - maximum, fast={fast_math!r}),",
        ")",
        "rhs_scale = cutlass.select_(",
        "    rhs[0] == maximum,",
        "    one,",
        f"    epi_math.exp(rhs[0] - maximum, fast={fast_math!r}),",
        ")",
        "return maximum, lhs[1] * lhs_scale + rhs[1] * rhs_scale",
    )


@dataclasses.dataclass(frozen=True)
class FlexGemmEpiModReductionSpec:
    """Describe one normalized grouped reduction lowered into a QuACK EpiOp."""

    node: torch.fx.Node
    aliases: tuple[torch.fx.Node, ...]
    reduction: NormalizedGemmReduction
    component_aliases: tuple[tuple[torch.fx.Node, ...], ...] = ()

    def __post_init__(self) -> None:
        if self.component_aliases and len(self.component_aliases) != self.reduce_planes:
            raise RuntimeError(
                "FlexGEMM state component aliases must match reduction arity"
            )

    @property
    def source(self) -> torch.fx.Node:
        return self.reduction.source

    @property
    def is_online_softmax(self) -> bool:
        return isinstance(self.reduction, NormalizedPrepareSoftmax)

    @property
    def combine(self) -> str | None:
        if isinstance(self.reduction, NormalizedPrepareSoftmax):
            return None
        return {
            "sum": "add",
            "mean": "add",
            "prod": "mul",
            "max": "max",
            "min": "min",
        }[self.reduction.reduction_type]

    @property
    def finalize(self) -> str | None:
        return (
            "mean"
            if isinstance(self.reduction, NormalizedReduction)
            and self.reduction.reduction_type == "mean"
            else None
        )

    @property
    def reduce_planes(self) -> int:
        """Return the number of independently transported state planes."""
        return self.reduction.associative_state.planes

    @property
    def boundary_aliases(self) -> tuple[torch.fx.Node, ...]:
        """Return aggregate and projected reduction nodes that bound finalization."""
        return (
            *self.aliases,
            *(alias for aliases in self.component_aliases for alias in aliases),
        )

    def lift_value(self, source: Any) -> Any:
        """Lift one source value into this reduction's logical state."""
        if self.is_online_softmax:
            return f"({source}, cute.full_like({source}, 1.0))"
        return source

    def generated_combine_body(self, fast_math: bool) -> tuple[str, ...]:
        """Return a generated tuple combine, or no body for a built-in combine."""
        return online_softmax_combine_body(fast_math) if self.is_online_softmax else ()


@dataclasses.dataclass(frozen=True)
class FlexGemmEpiModLocalReduceSpec:
    """Describe the sink reduction and its optional accumulator prepass."""

    sink: FlexGemmEpiModReductionSpec
    prepass: FlexGemmEpiModReductionSpec | None = None


def epimod_reduction_alias_key(
    reduction: NormalizedGemmReduction,
) -> tuple[Any, ...]:
    """Return the physical identity shared by equivalent FX reductions."""
    dims = (
        tuple(reduction.dim)
        if isinstance(reduction.dim, (tuple, list))
        else (reduction.dim,)
    )
    details = (
        (bool(reduction.keepdim), reduction.dtype, reduction.reduction_type)
        if isinstance(reduction, NormalizedReduction)
        else (True, None, type(reduction))
    )
    return reduction.source, dims, *details


def epimod_projected_reduction(
    node: torch.fx.Node,
    reduction: NormalizedGemmReduction,
) -> tuple[NormalizedGemmReduction, tuple[torch.fx.Node, ...]]:
    """Collapse an aggregate used only through one scalar-state projection."""
    state = reduction.associative_state
    if state.planes == 1 or not node.users:
        return reduction, ()
    projection: GemmReductionType | None = None
    aliases = []
    for user in node.users:
        index = user.args[1] if user.target is operator.getitem else None
        if not isinstance(index, int) or not -state.planes <= index < state.planes:
            return reduction, ()
        candidate = state.reduction_projections[index % state.planes]
        if candidate is None or (projection is not None and candidate != projection):
            return reduction, ()
        projection = candidate
        aliases.append(user)
    if projection is None:
        return reduction, ()
    return (
        NormalizedReduction(reduction.source, reduction.dim, True, None, projection),
        tuple(aliases),
    )


def epimod_reduction_spec(
    node: torch.fx.Node,
    aliases: tuple[torch.fx.Node, ...],
    reduction: NormalizedGemmReduction,
) -> FlexGemmEpiModReductionSpec:
    """Translate one normalized FX reduction into QuACK semantics."""
    if isinstance(reduction, NormalizedReduction) and reduction.dtype is not None:
        raise NotImplementedError(LOCAL_REDUCE_EXPLICIT_DTYPE_ERROR)
    return FlexGemmEpiModReductionSpec(node, aliases, reduction)


def epimod_state_projection(
    state: FlexGemmEpiModReductionSpec,
    candidate: FlexGemmEpiModReductionSpec,
    matches: dict[torch.fx.Node, FlexGemmLocalReduceMatch],
) -> int | None:
    """Return the state component that makes a scalar reduction redundant."""
    if (
        state.source is not candidate.source
        or matches[state.node].geometry != matches[candidate.node].geometry
        or not isinstance(candidate.reduction, NormalizedReduction)
    ):
        return None
    projections = state.reduction.associative_state.reduction_projections
    reduction_type = candidate.reduction.reduction_type
    return projections.index(reduction_type) if reduction_type in projections else None


def epimod_dependency_slice(
    result: torch.fx.Node, boundaries: frozenset[torch.fx.Node]
) -> frozenset[torch.fx.Node]:
    """Collect the backward FX slice, stopping at supplied scalar boundaries."""
    nodes = OrderedSet[torch.fx.Node]()
    pending = [result]
    while pending:
        node = pending.pop()
        if node in nodes:
            continue
        nodes.add(node)
        if node not in boundaries:
            pending.extend(iter_fx_node_inputs((node.args, node.kwargs)))
    return frozenset(nodes)


def epimod_local_reduce_spec(
    analysis: FlexGemmEpilogueAnalysis,
    local_reduce: FlexGemmOutputLocalReducePlan,
) -> FlexGemmEpiModLocalReduceSpec:
    """Map an analyzed grouped reduction DAG onto QuACK EpiOps."""
    physical_nodes = analysis.local_reduce.physical_reduction_nodes(local_reduce.match)
    graph = analysis.local_reduce.graph
    alias_groups: dict[
        tuple[Any, ...], tuple[NormalizedGemmReduction, list[torch.fx.Node]]
    ] = {}
    for node in physical_nodes:
        reduction = graph.normalized_nodes.get(node)
        if not isinstance(reduction, NormalizedGemmReduction):
            raise AssertionError(
                "analyzed grouped reduction requires reduction metadata"
            )
        reduction, projection_aliases = epimod_projected_reduction(node, reduction)
        key = epimod_reduction_alias_key(reduction)
        group = alias_groups.get(key)
        if group is None:
            alias_groups[key] = (reduction, [node, *projection_aliases])
        else:
            group[1].extend((node, *projection_aliases))
    reduction_specs = tuple(
        epimod_reduction_spec(nodes[0], tuple(nodes), reduction)
        for reduction, nodes in alias_groups.values()
    )
    matches = analysis.local_reduce.matches
    state_specs = tuple(spec for spec in reduction_specs if spec.reduce_planes > 1)
    if len(state_specs) == 1:
        state = state_specs[0]
        component_aliases = [[] for _ in range(state.reduce_planes)]
        remaining = []
        for candidate in reduction_specs:
            if candidate is state:
                continue
            projection = epimod_state_projection(state, candidate, matches)
            if projection is None:
                remaining.append(candidate)
            else:
                component_aliases[projection].extend(candidate.aliases)
        if any(component_aliases):
            state = dataclasses.replace(
                state,
                component_aliases=tuple(
                    tuple(aliases) for aliases in component_aliases
                ),
            )
        reduction_specs = (state, *remaining)

    geometry = local_reduce.match.geometry
    if len(reduction_specs) == 1:
        sink = reduction_specs[0]
        if analysis.local_reduce.matches[sink.node].geometry != geometry:
            raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
        spec = FlexGemmEpiModLocalReduceSpec(sink)
    elif len(reduction_specs) == 2:
        inner_spec, outer_spec = reduction_specs
        inner_node, outer_node = inner_spec.node, outer_spec.node
        inner = (
            inner_spec.reduction
            if isinstance(inner_spec.reduction, NormalizedReduction)
            else None
        )
        outer = (
            outer_spec.reduction
            if isinstance(outer_spec.reduction, NormalizedReduction)
            else None
        )
        outer_source = None if outer is None else outer.source
        reduction_nodes = (inner_node, outer_node)
        if (
            local_reduce.feeds_main
            or geometry.axis != 1
            or geometry.group > LOCAL_REDUCE_FRAGMENT_WIDTH
            or geometry.group & (geometry.group - 1)
            or any(matches[node].geometry != geometry for node in reduction_nodes)
            or any(spec.reduce_planes != 1 for spec in reduction_specs)
            or inner is None
            or not inner.keepdim
            or not isinstance(outer_source, torch.fx.Node)
            or not is_shape_preserving_pointwise_node(outer_source)
            or not analysis.local_reduce.graph.depends_on(outer_source, inner_node)
        ):
            raise NotImplementedError(
                "FlexGEMM EpiMod supports two grouped reductions only when they "
                "have scalar state, the same axis-1 geometry, and the first "
                "keepdim reduction feeds the pointwise source of the second"
            )
        spec = FlexGemmEpiModLocalReduceSpec(outer_spec, inner_spec)
    else:
        raise NotImplementedError(
            "FlexGEMM EpiMod supports one grouped reduction, or exactly two "
            "same-geometry axis-1 reductions in an inner-to-outer chain"
        )
    if spec.sink.reduce_planes > 1 and local_reduce.feeds_main:
        raise NotImplementedError(
            "FlexGEMM multi-plane grouped reductions currently support returned "
            "outputs, not feed-main consumers"
        )
    if local_reduce.match.physical_span > 1 and spec.sink.finalize == "mean":
        raise NotImplementedError("nested TensorSSA reductions do not support mean")
    reduction_node = spec.sink.node
    if (
        local_reduce.feeds_main
        and geometry.axis == 0
        and local_reduce.store is not None
        and local_reduce.store.value_node is not reduction_node
    ):
        store = local_reduce.store.value_node
        chain = [
            node
            for node in (
                *analysis.local_reduce.graph.dependencies.get(store, ()),
                store,
            )
            if node is not reduction_node
            and analysis.local_reduce.graph.depends_on(node, reduction_node)
        ]
        if any(
            view_or_reshape_args(node) is None
            and squeeze_source_node(node) is None
            and node.target is not operator.getitem
            for node in chain
        ):
            raise NotImplementedError(
                "FlexGEMM EpiMod cannot apply a store-only post-reduction "
                "transform to a value that also feeds the main output"
            )
    return spec


class FlexGemmEpiModEmitter:
    """Emit QuACK EpiMod source from shared FlexGEMM analysis."""

    def __init__(
        self,
        graph_module: torch.fx.GraphModule,
        analysis: FlexGemmEpilogueAnalysis,
        epilogue_arg_placeholders: tuple[torch.fx.Node, ...],
        alpha: float,
        beta: float,
        epilogue_arg_kinds: tuple[str, ...],
        *,
        fast_math: bool,
        swap_ab: bool,
        mainloop_scale_count: int,
    ) -> None:
        self.graph_module = graph_module
        self.gemm = analysis.gemm
        self.analysis = analysis
        self.outputs = analysis.outputs
        self.local_reduce = self.outputs.local_reduce
        self.grouped_select_indices = analysis.grouped_select_indices
        self.terminal_rewrites = self.outputs.terminal_rewrites
        self.local_reduce_spec: FlexGemmEpiModLocalReduceSpec | None = None
        self.local_reduce_prepass: FlexGemmEpiModReductionSpec | None = None
        self.local_reduce_source_nodes: frozenset[torch.fx.Node] = frozenset()
        self.local_reduce_finalize_nodes: frozenset[torch.fx.Node] = frozenset()
        self.local_reduce_finalize_uses_prepass = False
        self.local_reduce_finalize_body: tuple[str, ...] = ()
        self.local_reduce_finalize_result: Any | None = None
        self.local_reduce_prepass_body: tuple[str, ...] = ()
        self.local_reduce_prepass_result: Any | None = None
        self.local_reduce_fragment_reduced = False
        self.local_reduce_sink_value: CuteDSLCSEVariable | None = None
        if self.local_reduce is not None:
            spec = epimod_local_reduce_spec(analysis, self.local_reduce)
            self.local_reduce_spec = spec
            sink = spec.sink
            match = self.local_reduce.match
            paired = match.physical_span > 1
            if paired and swap_ab:
                raise NotImplementedError(
                    "nested TensorSSA reductions do not support swap_ab=True"
                )
            self.local_reduce_source_nodes = frozenset(
                (
                    *analysis.local_reduce.graph.dependencies.get(sink.source, ()),
                    sink.source,
                )
            )
            prepass = spec.prepass
            if (
                prepass is None
                and self.local_reduce.feeds_main
                and match.geometry.axis == 1
                and not paired
            ):
                prepass = sink
            self.local_reduce_prepass = prepass
            # Paired lanes complete their logical group inside one fragment
            # (GroupedMainStore min_fragment_n); axis-N multi-plane state returns
            # fragment partials. Both skip only QuACK's in-fragment fold.
            self.local_reduce_fragment_reduced = paired or (
                sink.reduce_planes > 1 and match.geometry.axis == 1 and not swap_ab
            )
            if (
                (not self.local_reduce.feeds_main or prepass is not None or paired)
                and self.local_reduce.store is not None
                and self.local_reduce.store.value_node is not sink.node
            ):
                sink_aliases = frozenset(sink.boundary_aliases)
                prepass_aliases = frozenset(
                    () if prepass is None else prepass.boundary_aliases
                )
                self.local_reduce_finalize_nodes = epimod_dependency_slice(
                    self.local_reduce.store.value_node,
                    sink_aliases | prepass_aliases,
                )
                self.local_reduce_finalize_uses_prepass = bool(
                    self.local_reduce_finalize_nodes & (prepass_aliases - sink_aliases)
                )
        grouped_tensors = analysis.grouped_main_layouts | (
            analysis.local_reduce.grouped_tensors
            if self.local_reduce_fragment_reduced
            else {}
        )
        self.grouped_layouts = {
            node: layout
            for node, layout in grouped_tensors.items()
            if view_or_reshape_args(node) is not None
            or node.target is torch.ops.aten.split.Tensor
        }
        self.epilogue_arg_placeholders = epilogue_arg_placeholders
        self.alpha = alpha
        self.beta = beta
        self.epilogue_arg_kinds = epilogue_arg_kinds
        self.fast_math = fast_math
        self.operand_names = tuple(
            f"operand{index}" for index in range(len(epilogue_arg_placeholders))
        )
        if not 0 <= mainloop_scale_count <= len(self.operand_names):
            raise RuntimeError("invalid FlexGEMM main-loop scale operand count")
        self.mainloop_scale_count = mainloop_scale_count
        self.kernel = GemmEpilogueCuteDSLKernel()
        self.params = ["acc"]
        self.base_env = self.initial_env_for_params(self.params)
        if self.local_reduce_prepass is not None or (
            self.local_reduce is not None
            and self.local_reduce.feeds_main
            and self.local_reduce.match.physical_span == 1
        ):
            self.params.append(LOCAL_REDUCE_FEED_MAIN_ARG_NAME)
        self.local_reduce_prepass_value: CuteDSLCSEVariable | None = None
        self.env = dict(self.base_env)

    @staticmethod
    def value(name: str, dtype: torch.dtype) -> CuteDSLCSEVariable:
        """Represent one generated EpiMod scalar/F2 value with dtype metadata."""
        return CuteDSLCSEVariable(
            name,
            ValueRanges.unknown(),
            dtype=torch.float32 if dtype.is_floating_point else dtype,
            shape=(1,),
        )

    def initial_env_for_params(self, params: list[str]) -> dict[torch.fx.Node, Any]:
        """Bind accumulator and captures while recording one function signature."""
        gemm_value = "acc"
        if self.alpha != 1:
            params.append("alpha")
            gemm_value = "(acc * alpha)"
        for name in self.operand_names[: self.mainloop_scale_count]:
            gemm_value = f"({gemm_value} * {name})"
        if (
            self.gemm.target
            in (
                torch.ops.aten.addmm.default,
                torch.ops.aten.baddbmm.default,
            )
            and self.beta != 0
        ):
            params.append("c")
            bias_value = "c"
            if self.beta != 1:
                params.append("beta")
                bias_value = "(c * beta)"
            gemm_value = f"({gemm_value} + {bias_value})"
        params.extend(self.operand_names)
        captures = {}
        for node, name in zip(
            self.epilogue_arg_placeholders,
            self.operand_names,
            strict=True,
        ):
            dtype = node.meta["val"].dtype
            if dtype is torch.bool:
                expr = f"operator.ne({name}, cute.full_like({name}, 0))"
            else:
                expr = name
            captures[node] = self.value(expr, dtype)
        return {
            self.gemm: self.value(gemm_value, torch.float32),
            **captures,
        }

    def lower_local_reduce_finalize(self) -> None:
        """Lower a compressed-output transform as a scalar QuACK finalizer."""
        spec = self.local_reduce_spec
        local_reduce = self.local_reduce
        if (
            not self.local_reduce_finalize_nodes
            or local_reduce is None
            or local_reduce.store is None
            or spec is None
        ):
            return
        sink = spec.sink
        kernel = GemmEpilogueCuteDSLKernel()
        if sink.reduce_planes == 1:
            reduction_meta = sink.node.meta.get("val")
            dtype = (
                reduction_meta.dtype
                if isinstance(reduction_meta, torch.Tensor)
                else torch.float32
            )
            value = "value"
            if sink.finalize == "mean":
                value = f"(value / {float(local_reduce.match.geometry.group)!r})"
            reduced: Any = self.value(value, dtype)
        else:
            reduced = tuple(
                self.value(f"state[{index}]", torch.float32)
                for index in range(sink.reduce_planes)
            )
        env: dict[torch.fx.Node, Any] = dict.fromkeys(sink.aliases, reduced)
        if sink.component_aliases:
            if not isinstance(reduced, tuple):
                raise AssertionError("state projections require multi-plane reduction")
            for index, aliases in enumerate(sink.component_aliases):
                env.update((alias, reduced[index]) for alias in aliases)
        if self.local_reduce_finalize_uses_prepass:
            prepass = self.local_reduce_prepass
            if prepass is None:
                raise AssertionError("prepass finalizer requires a prepass")
            prepass_meta = prepass.node.meta.get("val")
            prepass_dtype = (
                prepass_meta.dtype
                if isinstance(prepass_meta, torch.Tensor)
                else torch.float32
            )
            prepass_value = self.value("prepass_value", prepass_dtype)
            env.update((alias, prepass_value) for alias in prepass.aliases)
        with (
            V.set_kernel_handler(kernel),
            V.set_ops_handler(FlexGemmScalarCallbackOpOverrides(self.fast_math)),
            use_cutedsl_fast_math(self.fast_math),
        ):
            for node in self.graph_module.graph.nodes:
                if node in env or node not in self.local_reduce_finalize_nodes:
                    continue
                env[node] = lower_gemm_epilogue_fx_node(
                    kernel, env, node, context="FlexGEMM"
                )
        self.local_reduce_finalize_body = tuple(kernel.body.lines)
        self.local_reduce_finalize_result = flex_gemm_epilogue_arg(
            local_reduce.store.value_node, env
        )

    def lower_local_reduce_prepass(self) -> None:
        """Lower the grouped source expression for an axis-1 accumulator prepass."""
        if self.local_reduce_prepass is None:
            return
        source = self.local_reduce_prepass.source
        dependencies = frozenset(
            (*self.analysis.local_reduce.graph.dependencies.get(source, ()), source)
        )
        bool_captures = [
            node
            for node in self.epilogue_arg_placeholders
            if node in dependencies and node.meta["val"].dtype is torch.bool
        ]
        if bool_captures:
            raise NotImplementedError(
                "FlexGEMM accumulator prepasses do not support captured bool tensors"
            )
        kernel = GemmEpilogueCuteDSLKernel()
        env = dict(self.base_env)
        with (
            V.set_kernel_handler(kernel),
            V.set_ops_handler(FlexGemmScalarCallbackOpOverrides(self.fast_math)),
            use_cutedsl_fast_math(self.fast_math),
        ):
            for node in self.graph_module.graph.nodes:
                if (
                    node is self.gemm
                    or node.op in ("placeholder", "output")
                    or node not in dependencies
                ):
                    continue
                if isinstance(node.meta.get("val"), (int, torch.SymInt)):
                    continue
                if node.op not in ("call_function", "call_method"):
                    raise NotImplementedError(
                        f"unsupported FlexGEMM EpiMod prepass node: {node.format_node()}"
                    )
                env[node] = lower_gemm_epilogue_fx_node(
                    kernel, env, node, context="FlexGEMM"
                )
        self.local_reduce_prepass_body = tuple(kernel.body.lines)
        self.local_reduce_prepass_result = flex_gemm_epilogue_arg(source, env)

    def lower_grouped_layout(
        self, node: torch.fx.Node, layout: GroupedTensorSSALayout
    ) -> None:
        """Reshape one physical TensorSSA fragment into grouped lanes."""
        if node.target is torch.ops.aten.split.Tensor:
            source_node = node.args[0]
        else:
            view_args = view_or_reshape_args(node)
            if view_args is None:
                raise AssertionError("grouped main layout requires a view or split")
            source_node = view_args[0]
        source = flex_gemm_epilogue_arg(source_node, self.env)
        grouped = self.kernel.cse.generate(
            self.kernel.body,
            f"{source}.reshape({layout.tensorssa_shape(source)})",
            dtype=torch.float32,
            shape=(1,),
        )
        if node.target is torch.ops.aten.split.Tensor:
            self.env[node] = tuple(
                self.kernel.cse.generate(
                    self.kernel.body,
                    f"{grouped}[((0, {index}, None), None, None)]",
                    dtype=torch.float32,
                    shape=(1,),
                )
                for index in range(layout.group_size)
            )
        else:
            self.env[node] = grouped

    def lower_grouped_main_select(self, node: torch.fx.Node, index: int) -> None:
        """Select one analysis-validated lane from a grouped TensorSSA value."""
        source = flex_gemm_epilogue_arg(node.args[0], self.env)
        expression = (
            source[index]
            if isinstance(source, tuple)
            else f"{source}[((0, {index}, None), None, None)]"
        )
        meta = node.meta.get("val")
        dtype = meta.dtype if isinstance(meta, torch.Tensor) else torch.float32
        self.env[node] = self.kernel.cse.generate(
            self.kernel.body, expression, dtype=dtype, shape=(1,)
        )

    def generate_like(
        self, expression: str, reference: Any, *, shape_reference: Any | None = None
    ) -> CuteDSLCSEVariable:
        """Emit one expression while preserving reference dtype and shape metadata."""
        shape_reference = reference if shape_reference is None else shape_reference
        return self.kernel.cse.generate(
            self.kernel.body,
            expression,
            dtype=getattr(reference, "dtype", None),
            shape=getattr(shape_reference, "shape", None),
        )

    def broadcast_fragment_partial(
        self, reduced: Any, layout: GroupedTensorSSALayout, source: Any
    ) -> CuteDSLCSEVariable:
        """Broadcast one fragment partial back to its grouped TensorSSA shape."""
        return self.generate_like(
            f"{reduced}.reshape({layout.keepdim_shape(source)}).broadcast_to({source}.shape)",
            reduced,
            shape_reference=source,
        )

    def lower_online_softmax_fragment_partial(
        self, source: Any, layout: GroupedTensorSSALayout
    ) -> tuple[CuteDSLCSEVariable, CuteDSLCSEVariable]:
        """Reduce one TensorSSA fragment into online maximum and safe-exp sum planes."""
        maximum = self.generate_like(
            f'{source}.reduce(cute.ReductionOp.MAX, init_val=float("-inf"), '
            f"reduction_profile={layout.reduction_profile})",
            source,
        )
        maximum_broadcast = self.broadcast_fragment_partial(maximum, layout, source)
        centered = self.generate_like(f"({source} - {maximum_broadcast})", source)
        exp_centered = CuteDSLOpOverrides.exp(centered)
        is_maximum = self.generate_like(
            f"operator.eq({source}, cute.full_like({source}, {maximum_broadcast}))",
            source,
        )
        safe_exp = self.generate_like(
            f"cute.where({is_maximum}, cute.full_like({exp_centered}, 1.0), "
            f"{exp_centered})",
            exp_centered,
        )
        total = self.generate_like(
            f"{safe_exp}.reduce(cute.ReductionOp.ADD, init_val=0.0, "
            f"reduction_profile={layout.reduction_profile})",
            safe_exp,
        )
        return maximum_broadcast, self.broadcast_fragment_partial(total, layout, source)

    def lower_fragment_partial_state(
        self, sink: FlexGemmEpiModReductionSpec, source: Any
    ) -> Any:
        """Reduce one TensorSSA fragment before the generic physical combine."""
        if self.local_reduce is None:
            raise AssertionError(
                "TensorSSA grouped reduction requires a reduction plan"
            )
        match = self.local_reduce.match
        geometry = match.geometry
        layout = GroupedTensorSSALayout(geometry.axis, geometry.group)
        if isinstance(sink.reduction, NormalizedPrepareSoftmax):
            return self.lower_online_softmax_fragment_partial(source, layout)
        kind = sink.reduction.reduction_type
        desc = tensorssa_reduction("sum" if kind == "mean" else kind)
        reduced = self.generate_like(
            f"{source}.reduce({desc.cute_op}, init_val={desc.init_val}, "
            f"reduction_profile={layout.reduction_profile})",
            source,
        )
        if match.physical_span > 1:
            # QuACK collects this sink at physical fragment width; broadcast the
            # logical group value across both paired lanes.
            physical = GroupedTensorSSALayout(
                geometry.axis, match.physical_geometry.group
            )
            self.local_reduce_sink_value = self.generate_like(
                f"{reduced}.reshape({physical.keepdim_shape('acc')})"
                f".broadcast_to({physical.tensorssa_shape('acc')})",
                reduced,
            )
        return self.broadcast_fragment_partial(reduced, layout, source)

    def lower_graph(self) -> None:
        """Lower FX nodes through Inductor's standard operation-dispatch API."""
        spec = self.local_reduce_spec
        local_reduce = self.local_reduce
        sink = None if spec is None else spec.sink
        prepass_aliases = (
            frozenset()
            if self.local_reduce_prepass is None
            else frozenset(self.local_reduce_prepass.aliases)
        )
        with (
            V.set_kernel_handler(self.kernel),
            V.set_ops_handler(FlexGemmTensorSSAOpOverrides()),
            use_cutedsl_fast_math(self.fast_math),
        ):
            for node in self.graph_module.graph.nodes:
                if node is self.gemm or node.op in ("placeholder", "output"):
                    continue
                if (
                    (local_reduce is None or local_reduce.match.physical_span == 1)
                    and (
                        self.local_reduce_prepass is None
                        or (spec is not None and spec.prepass is not None)
                    )
                    and node in self.local_reduce_finalize_nodes
                    and node not in self.local_reduce_source_nodes
                    and sink is not None
                    and node is not sink.node
                ):
                    continue
                if isinstance(node.meta.get("val"), (int, torch.SymInt)):
                    continue
                if node.op not in ("call_function", "call_method"):
                    raise NotImplementedError(
                        f"unsupported FlexGEMM EpiMod node: {node.format_node()}"
                    )
                if node in self.grouped_layouts:
                    self.lower_grouped_layout(node, self.grouped_layouts[node])
                    continue
                if node in self.grouped_select_indices:
                    self.lower_grouped_main_select(
                        node, self.grouped_select_indices[node]
                    )
                    continue
                if node in self.terminal_rewrites:
                    source = self.terminal_rewrites[node]
                    if source is not None:
                        self.env[node] = flex_gemm_epilogue_arg(source, self.env)
                    continue
                if self.local_reduce_prepass is not None and node in prepass_aliases:
                    if self.local_reduce_prepass_value is None:
                        meta = node.meta.get("val")
                        dtype = (
                            meta.dtype
                            if isinstance(meta, torch.Tensor)
                            else torch.float32
                        )
                        self.local_reduce_prepass_value = self.value(
                            LOCAL_REDUCE_FEED_MAIN_ARG_NAME, dtype
                        )
                    self.env[node] = self.local_reduce_prepass_value
                    continue
                if sink is not None and local_reduce is not None and node is sink.node:
                    source = flex_gemm_epilogue_arg(sink.source, self.env)
                    if self.local_reduce_fragment_reduced:
                        self.env[node] = self.lower_fragment_partial_state(sink, source)
                    elif local_reduce.feeds_main:
                        meta = node.meta.get("val")
                        dtype = (
                            meta.dtype
                            if isinstance(meta, torch.Tensor)
                            else torch.float32
                        )
                        self.env[node] = self.kernel.cse.generate(
                            self.kernel.body,
                            f"{LOCAL_REDUCE_FEED_MAIN_ARG_NAME}({source})",
                            dtype=dtype,
                            shape=(1,),
                        )
                    else:
                        self.env[node] = sink.lift_value(source)
                    continue
                self.env[node] = lower_gemm_epilogue_fx_node(
                    self.kernel, self.env, node, context="FlexGEMM"
                )

    def render(self) -> FlexGemmEpiModSource:
        """Render a deterministic generated EpiMod definition."""
        from torch._inductor.codegen.cutedsl._inline_asm import inline_asm_cache_key

        spec = self.local_reduce_spec
        sink = None if spec is None else spec.sink
        main_result = flex_gemm_epilogue_arg(self.outputs.output, self.env)
        aux_names = tuple(
            f"output{index}" for index in range(len(self.outputs.aux_outputs))
        )
        aux_results = tuple(
            flex_gemm_epilogue_arg(output, self.env)
            for output in self.outputs.aux_outputs
        )
        main_name = "main" if self.outputs.main_transform is not None else "D"
        result_items = [
            (main_name, main_result),
            *zip(aux_names, aux_results, strict=True),
        ]
        if self.outputs.indexed_output is not None:
            result_items.append((INDEXED_OUTPUT_STORE_ARG_NAME, main_result))
        if (
            self.local_reduce is not None
            and sink is not None
            and self.local_reduce.store is not None
            and (
                not self.local_reduce.feeds_main
                or self.local_reduce_prepass is not None
                or self.local_reduce.match.physical_span > 1
            )
        ):
            store_value = (
                flex_gemm_epilogue_arg(sink.node, self.env)
                if self.local_reduce_sink_value is None
                else self.local_reduce_sink_value
            )
            if self.local_reduce_finalize_uses_prepass:
                store_value = f"({store_value}, {LOCAL_REDUCE_FEED_MAIN_ARG_NAME})"
            result_items.append(
                (
                    LOCAL_REDUCE_STORE_ARG_NAME
                    if self.local_reduce_prepass is not None
                    else LOCAL_REDUCE_FEED_MAIN_ARG_NAME,
                    store_value,
                )
            )
        return_source = ", ".join(
            f"{name!r}: {gemm_epilogue_source_expr(result)}"
            for name, result in result_items
        )
        body = "\n".join(f"    {line}" for line in self.kernel.body.lines)
        if body:
            body += "\n"
        combine_lines = (
            () if sink is None else sink.generated_combine_body(self.fast_math)
        )
        combine_body = "\n".join(f"    {line}" for line in combine_lines)
        if combine_body:
            combine_body += "\n"
        finalize_body = "\n".join(
            f"    {line}" for line in self.local_reduce_finalize_body
        )
        if finalize_body:
            finalize_body += "\n"
        finalize_payload = (
            ""
            if self.local_reduce_finalize_result is None
            else f"{finalize_body}return {self.local_reduce_finalize_result}\n"
        )
        prepass_body = "\n".join(
            f"    {line}" for line in self.local_reduce_prepass_body
        )
        if prepass_body:
            prepass_body += "\n"
        prepass_payload = (
            ""
            if self.local_reduce_prepass_result is None
            else f"{prepass_body}return {self.local_reduce_prepass_result}\n"
        )
        key_payload = (
            f"inline_asm={inline_asm_cache_key()}\n"
            f"reduce_planes={1 if sink is None else sink.reduce_planes}\n"
            f"fragment_reduced={self.local_reduce_fragment_reduced}\n"
            f"{self.graph_module.code}\n{body}return {{{return_source}}}\n"
            f"{combine_body}{finalize_payload}{prepass_payload}"
            f"{self.epilogue_arg_kinds!r}"
        )
        key = hashlib.sha256(key_payload.encode()).hexdigest()[:16]
        name = f"flex_gemm_epimod_{key}"
        combine_name = None
        combine_source = ""
        if combine_lines:
            combine_name = f"{name}_local_reduce_combine"
            combine_source = f"def {combine_name}(lhs, rhs):\n{combine_body}\n"
        finalize_name = None
        finalize_source = ""
        if self.local_reduce_finalize_result is not None:
            finalize_name = f"{name}_local_reduce_finalize"
            finalize_params = (
                "value, prepass_value"
                if self.local_reduce_finalize_uses_prepass
                else "state"
                if sink is not None and sink.reduce_planes > 1
                else "value"
            )
            finalize_source = (
                f"def {finalize_name}({finalize_params}):\n"
                f"{finalize_body}    return {self.local_reduce_finalize_result}\n\n"
            )
        prepass_name = None
        prepass_source = ""
        if self.local_reduce_prepass_result is not None:
            prepass_name = f"{name}{LOCAL_REDUCE_PREPASS_FN_SUFFIX}"
            prepass_params = [
                param
                for param in self.params
                if param != LOCAL_REDUCE_FEED_MAIN_ARG_NAME
            ]
            prepass_source = (
                f"def {prepass_name}({', '.join(prepass_params)}):\n"
                f"{prepass_body}    return "
                f"{{{LOCAL_REDUCE_FEED_MAIN_ARG_NAME!r}: "
                f"{self.local_reduce_prepass_result}}}\n\n"
            )
        generated_imports = (
            "import cutlass\n"
            "import cutlass.cute as cute\n"
            "import operator\n"
            "from cutlass._mlir.dialects import math as mlir_math\n"
            "from torch._inductor.codegen.cutedsl._inline_asm import (\n"
            "    inline_asm_elementwise_intrinsic,\n"
            ")\n\n"
        )
        return FlexGemmEpiModSource(
            name=name,
            source=(
                f"{generated_imports}{combine_source}{finalize_source}{prepass_source}"
                f"@cute.jit\ndef {name}({', '.join(self.params)}):\n"
                f"{body}    return {{{return_source}}}\n"
            ),
            local_reduce_combine=(
                None if sink is None else combine_name or sink.combine
            ),
            local_reduce_finalize=(
                None
                if sink is None
                else sink.finalize
                if self.local_reduce_prepass is not None
                else finalize_name or sink.finalize
            ),
            local_reduce_store_finalize=(
                finalize_name if self.local_reduce_prepass is not None else None
            ),
            local_reduce_prepass_combine=(
                None
                if self.local_reduce_prepass is None
                else self.local_reduce_prepass.combine
            ),
            local_reduce_prepass_finalize=(
                None
                if self.local_reduce_prepass is None
                else self.local_reduce_prepass.finalize
            ),
            local_reduce_planes=1 if sink is None else sink.reduce_planes,
            local_reduce_fragment_reduced=self.local_reduce_fragment_reduced,
        )

    def materialize(self) -> FlexGemmEpiModSource:
        """Lower the shared FX graph and return a generated QuACK function."""
        self.lower_local_reduce_finalize()
        self.lower_local_reduce_prepass()
        self.lower_graph()
        return self.render()


def materialize_flex_gemm_epimod(
    graph_module: torch.fx.GraphModule,
    analysis: FlexGemmEpilogueAnalysis,
    epilogue_arg_placeholders: tuple[torch.fx.Node, ...],
    alpha: float,
    beta: float,
    epilogue_arg_kinds: tuple[str, ...],
    *,
    fast_math: bool = False,
    swap_ab: bool = False,
    mainloop_scale_count: int = 0,
) -> FlexGemmEpiModSource:
    """Materialize an analyzed FlexGEMM body as QuACK EpiMod source."""
    return FlexGemmEpiModEmitter(
        graph_module,
        analysis,
        epilogue_arg_placeholders,
        alpha,
        beta,
        epilogue_arg_kinds,
        fast_math=fast_math,
        swap_ab=swap_ab,
        mainloop_scale_count=mainloop_scale_count,
    ).materialize()
