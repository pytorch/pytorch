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

from sympy import Max, Min

import torch
from torch._inductor import inductor_prims
from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
    CuteDSLCSEVariable,
    CuteDSLOpOverrides,
    use_cutedsl_fast_math,
)
from torch._inductor.kernel.flex_gemm.constraints import (
    FLEX_GEMM_MAIN_OUTPUT_SHAPE_ERROR,
    FLEX_GEMM_OUTPUT_CONTRACTION_COMPOSITION_ERROR,
    FLEX_GEMM_OUTPUT_TENSOR_ERROR,
    FlexGemmLocalReduceGeometry,
    LOCAL_REDUCE_EXPLICIT_DTYPE_ERROR,
    LOCAL_REDUCE_FEED_MAIN_ARG_NAME,
    LOCAL_REDUCE_FINALIZE_CAPTURE_ERROR,
    LOCAL_REDUCE_FRAGMENT_WIDTH,
    LOCAL_REDUCE_MIXED_MATCH_ERROR,
    LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR,
    LOCAL_REDUCE_PREPASS_FN_SUFFIX,
    LOCAL_REDUCE_STORE_ARG_NAME,
    LOCAL_REDUCE_UNPLANNED_ERROR,
)
from torch._inductor.kernel.flex_gemm.quack_reductions import (
    is_shape_preserving_pointwise_node,
    squeeze_source_node,
    view_or_reshape_args,
)
from torch._inductor.kernel.gemm_epilogue import (
    GemmEpilogueGraph,
    iter_fx_node_inputs,
    NormalizedReduction,
)
from torch._inductor.kernel.gemm_epilogue_analysis import (
    build_output_contraction_plan,
    GemmLocalReduceAnalysis,
    GemmOutputLocalReducePlan,
    GemmOutputPlan,
    match_flex_gemm_local_reduce_output_storage,
)
from torch._inductor.kernel.gemm_epilogue_codegen import (
    _cute_arg,
    gemm_epilogue_source_expr,
    GemmEpilogueCuteDSLKernel,
    GemmEpilogueCuteDSLOpOverrides,
    lower_gemm_epilogue_fx_node,
)
from torch._inductor.kernel.gemm_epilogue_utils import statically_known_shape_equal
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.value_ranges import ValueRanges


def bind_terminal_output_storage(
    outputs: GemmOutputPlan,
) -> GemmOutputPlan:
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


def tuple_output_plan(
    output: Any,
    aux_outputs: tuple[Any, ...],
    analysis: GemmLocalReduceAnalysis,
) -> GemmOutputPlan:
    """Classify multi-output epilogues after checking local-reduce consumers."""
    if not isinstance(output, torch.fx.Node) or not all(
        isinstance(aux_output, torch.fx.Node) for aux_output in aux_outputs
    ):
        raise NotImplementedError(FLEX_GEMM_OUTPUT_TENSOR_ERROR)
    feed_match = analysis.common_feed_main_match((output, *aux_outputs))
    compressed_aux_plans = tuple(
        plan
        for aux_output in aux_outputs
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
        if feed_match is not None:
            if OrderedSet(analysis.physical_reduction_nodes(feed_match)) != OrderedSet(
                compressed_reductions
            ):
                raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
            store = compressed_aux_plan.store
            if store is None:
                raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
            compressed_aux_plan = feed_match.to_plan(store=store, feeds_main=True)
        return GemmOutputPlan(
            output,
            aux_outputs,
            local_reduce=compressed_aux_plan,
        )
    feed_main_plan = analysis.feed_main_output_plan(output, aux_outputs)
    if feed_main_plan is not None:
        return feed_main_plan
    return GemmOutputPlan(output, aux_outputs)


def output_plan(
    graph_module: torch.fx.GraphModule,
    local_reduce: GemmLocalReduceAnalysis,
) -> GemmOutputPlan:
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
    return GemmOutputPlan(output_value) if feed_main_plan is None else feed_main_plan


def reject_unplanned_reductions(
    local_reduce: GemmLocalReduceAnalysis, outputs: GemmOutputPlan
) -> None:
    """Every matched grouped reduction must belong to the planned local reduce."""
    planned = (
        OrderedSet()
        if outputs.local_reduce is None
        else OrderedSet(
            local_reduce.physical_reduction_nodes(outputs.local_reduce.match)
        )
    )
    for node in local_reduce.matches:
        if (
            isinstance(
                local_reduce.graph.normalized_nodes.get(node), NormalizedReduction
            )
            and node not in planned
        ):
            raise NotImplementedError(LOCAL_REDUCE_UNPLANNED_ERROR)


def validate_output_layout_transforms(
    graph: GemmEpilogueGraph,
    outputs: GemmOutputPlan,
) -> None:
    """Require every layout transform to be the output validated by the plan."""
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
class FlexGemmEpilogueAnalysis:
    """Bundle the immutable analysis consumed by FlexGEMM lowering and emission.

    Attributes:
        outputs: Classification of main, auxiliary, and local-reduction outputs.
        local_reduce: Grouped layouts and local-reduction matches from the FX graph.
    """

    gemm: torch.fx.Node
    outputs: GemmOutputPlan
    local_reduce: GemmLocalReduceAnalysis
    output_contraction_select_indices: dict[torch.fx.Node, int] = dataclasses.field(
        default_factory=dict
    )
    output_contraction_layouts: dict[torch.fx.Node, FlexGemmLocalReduceGeometry] = (
        dataclasses.field(default_factory=dict)
    )

    @classmethod
    def from_graph_module(
        cls, graph_module: torch.fx.GraphModule, gemm: torch.fx.Node
    ) -> "FlexGemmEpilogueAnalysis":
        """Analyze reductions and an optional grouped main-output transform."""
        local_reduce = GemmLocalReduceAnalysis.from_graph_module(graph_module, gemm)
        outputs = bind_terminal_output_storage(output_plan(graph_module, local_reduce))
        validate_output_layout_transforms(local_reduce.graph, outputs)
        reject_unplanned_reductions(local_reduce, outputs)
        contraction_plan = build_output_contraction_plan(
            outputs.output_storage or outputs.output,
            gemm,
            local_reduce,
        )
        if contraction_plan is None:
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
        if outputs.aux_outputs:
            raise NotImplementedError(FLEX_GEMM_OUTPUT_CONTRACTION_COMPOSITION_ERROR)
        if contraction_plan.transform.chunked and outputs.local_reduce is not None:
            raise NotImplementedError(
                "chunked grouped main outputs do not compose with grouped reductions"
            )
        contraction_plan.commit_guards()
        local_reduce.commit_output_guards(outputs)
        return cls(
            gemm,
            dataclasses.replace(outputs, output_contraction=contraction_plan.transform),
            local_reduce,
            contraction_plan.select_indices,
            contraction_plan.layouts,
        )

    @property
    def required_geometries(self) -> tuple[FlexGemmLocalReduceGeometry, ...]:
        """Return the backend-neutral reduction plan produced by FX analysis."""
        geometries = OrderedSet(
            match.geometry for match in self.local_reduce.matches.values()
        )
        if self.outputs.local_reduce is not None:
            geometries.add(self.outputs.local_reduce.match.geometry)
        if self.outputs.output_contraction is not None:
            geometries.add(
                FlexGemmLocalReduceGeometry(self.outputs.output_contraction.group, 1)
            )
        return tuple(geometries)


def expand_epimod_prepare_softmax_online(
    graph_module: torch.fx.GraphModule,
) -> None:
    """Expand online-softmax tuples into the max and sum EpiMod can schedule."""
    graph = graph_module.graph
    changed = False
    for node in tuple(graph.nodes):
        if (
            node.op != "call_function"
            or node.target is not inductor_prims.prepare_softmax_online
        ):
            continue
        source = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        values = node.meta.get("val")
        if (
            not isinstance(source, torch.fx.Node)
            or not isinstance(dim, int)
            or not isinstance(values, (tuple, list))
            or len(values) != 2
        ):
            raise NotImplementedError(
                "FlexGEMM EpiMod requires a static prepare_softmax_online dimension"
            )
        with graph.inserting_before(node):
            maximum = graph.call_function(
                torch.ops.aten.amax.default, (source, [dim], True)
            )
            centered = graph.call_function(torch.ops.aten.sub.Tensor, (source, maximum))
            exponential = graph.call_function(torch.ops.aten.exp.default, (centered,))
            summed = graph.call_function(
                torch.ops.aten.sum.dim_IntList, (exponential, [dim], True)
            )
        for expanded, value in (
            (maximum, values[0]),
            (centered, source.meta.get("val")),
            (exponential, source.meta.get("val")),
            (summed, values[1]),
        ):
            expanded.meta.update(node.meta)
            expanded.meta["val"] = value
        for user in tuple(node.users):
            if (
                user.op != "call_function"
                or user.target is not operator.getitem
                or not isinstance(user.args[1], int)
                or user.args[1] not in (0, 1)
            ):
                raise NotImplementedError(
                    "FlexGEMM EpiMod requires direct prepare_softmax_online tuple uses"
                )
            user.replace_all_uses_with((maximum, summed)[user.args[1]])
            graph.erase_node(user)
        graph.erase_node(node)
        changed = True
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()


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
    return _cute_arg(value, env, "FlexGEMM")


class FlexGemmCuteDSLOpOverrides(GemmEpilogueCuteDSLOpOverrides):
    """Add PyTorch NaN propagation to the shared TensorSSA operation lowering."""

    @staticmethod
    def nan_propagating_minmax(a: Any, b: Any, op: str) -> Any:
        """Add FlexGEMM-specific NaN-propagating clamp semantics."""
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
        return FlexGemmCuteDSLOpOverrides.nan_propagating_minmax(a, b, "min")

    @staticmethod
    def maximum(a: Any, b: Any) -> Any:
        return FlexGemmCuteDSLOpOverrides.nan_propagating_minmax(a, b, "max")

    @staticmethod
    def clamp(x: Any, min: Any = None, max: Any = None) -> Any:
        result = x
        if min is not None:
            result = FlexGemmCuteDSLOpOverrides.maximum(result, min)
        if max is not None:
            result = FlexGemmCuteDSLOpOverrides.minimum(result, max)
        return result

    @staticmethod
    def clamp_min(x: Any, min: Any) -> Any:
        return FlexGemmCuteDSLOpOverrides.maximum(x, min)

    @staticmethod
    def clamp_max(x: Any, max: Any) -> Any:
        return FlexGemmCuteDSLOpOverrides.minimum(x, max)


@dataclasses.dataclass(frozen=True)
class FlexGemmEpiModSource:
    """Generated QuACK function plus optional grouped-reduction semantics."""

    name: str
    source: str
    local_reduce_combine: str | None = None
    local_reduce_finalize: str | None = None
    local_reduce_finalize_operands: tuple[str, ...] = ()
    local_reduce_store_finalize: str | None = None
    local_reduce_prepass_combine: str | None = None
    local_reduce_prepass_finalize: str | None = None


@dataclasses.dataclass(frozen=True)
class FlexGemmEpiModReductionSpec:
    """Describe one grouped reduction lowered into an external QuACK EpiOp."""

    node: torch.fx.Node
    aliases: tuple[torch.fx.Node, ...]
    source: torch.fx.Node
    combine: str
    finalize: str | None


@dataclasses.dataclass(frozen=True)
class FlexGemmEpiModLocalReduceSpec:
    """Describe the sink reduction and its optional accumulator prepass."""

    sink: FlexGemmEpiModReductionSpec
    prepass: FlexGemmEpiModReductionSpec | None = None


def epimod_reduction_alias_key(reduction: NormalizedReduction) -> tuple[Any, ...]:
    """Return the physical identity shared by equivalent FX reductions."""
    dims = (
        tuple(reduction.dim)
        if isinstance(reduction.dim, (tuple, list))
        else (reduction.dim,)
    )
    return (
        reduction.source,
        dims,
        bool(reduction.keepdim),
        reduction.dtype,
        reduction.reduction_type,
    )


def epimod_reduction_spec(
    node: torch.fx.Node,
    aliases: tuple[torch.fx.Node, ...],
    reduction: NormalizedReduction,
) -> FlexGemmEpiModReductionSpec:
    """Translate one normalized FX reduction into QuACK semantics."""
    if reduction.dtype is not None:
        raise NotImplementedError(LOCAL_REDUCE_EXPLICIT_DTYPE_ERROR)
    return FlexGemmEpiModReductionSpec(
        node,
        aliases,
        reduction.source,
        {
            "sum": "add",
            "mean": "add",
            "prod": "mul",
            "max": "max",
            "min": "min",
        }[reduction.reduction_type],
        "mean" if reduction.reduction_type == "mean" else None,
    )


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
    local_reduce: GemmOutputLocalReducePlan,
) -> FlexGemmEpiModLocalReduceSpec:
    """Map an analyzed grouped reduction DAG onto QuACK EpiOps."""
    physical_nodes = analysis.local_reduce.physical_reduction_nodes(local_reduce.match)
    graph = analysis.local_reduce.graph
    alias_groups: dict[
        tuple[Any, ...], tuple[NormalizedReduction, list[torch.fx.Node]]
    ] = {}
    for node in physical_nodes:
        reduction = graph.normalized_nodes.get(node)
        if not isinstance(reduction, NormalizedReduction):
            raise AssertionError(
                "analyzed grouped reduction requires reduction metadata"
            )
        key = epimod_reduction_alias_key(reduction)
        group = alias_groups.get(key)
        if group is None:
            alias_groups[key] = (reduction, [node])
        else:
            group[1].append(node)
    reductions = {nodes[0]: reduction for reduction, nodes in alias_groups.values()}
    reduction_nodes = tuple(reductions)
    aliases = {nodes[0]: tuple(nodes) for _, nodes in alias_groups.values()}
    geometry = local_reduce.match.geometry
    if len(reduction_nodes) == 1:
        reduction_node = reduction_nodes[0]
        if analysis.local_reduce.matches[reduction_node].geometry != geometry:
            raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
        spec = FlexGemmEpiModLocalReduceSpec(
            epimod_reduction_spec(
                reduction_node, aliases[reduction_node], reductions[reduction_node]
            )
        )
    elif len(reduction_nodes) == 2:
        inner_node, outer_node = reduction_nodes
        inner = reductions[inner_node]
        outer_source = reductions[outer_node].source
        matches = analysis.local_reduce.matches
        if (
            local_reduce.feeds_main
            or geometry.axis != 1
            or geometry.group > LOCAL_REDUCE_FRAGMENT_WIDTH
            or geometry.group & (geometry.group - 1)
            or any(matches[node].geometry != geometry for node in reduction_nodes)
            or not inner.keepdim
            or not is_shape_preserving_pointwise_node(outer_source)
            or not analysis.local_reduce.graph.depends_on(outer_source, inner_node)
        ):
            raise NotImplementedError(
                "FlexGEMM EpiMod supports two grouped reductions only when they "
                "have the same axis-1 geometry and the first keepdim reduction "
                "feeds the pointwise source of the second"
            )
        spec = FlexGemmEpiModLocalReduceSpec(
            epimod_reduction_spec(
                outer_node, aliases[outer_node], reductions[outer_node]
            ),
            epimod_reduction_spec(inner_node, aliases[inner_node], inner),
        )
    else:
        raise NotImplementedError(
            "FlexGEMM EpiMod supports one grouped reduction, or exactly two "
            "same-geometry axis-1 reductions in an inner-to-outer chain"
        )
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


class FlexGemmEpilogueEmitter:
    """Visit an analyzed FlexGEMM FX graph and emit its QuACK EpiMod source.

    The analysis dataclasses flow into each other as follows:

    ::

        GemmEpilogueGraph
          `--> GemmLocalReduceAnalysis
                 +--> grouped_tensors
                 `--> matches
                        `--> GemmLocalReduceMatch
                               `--> GemmOutputLocalReducePlan
                                      `--> optional GemmLocalReduceStore

        GemmLocalReduceAnalysis
          `--> output_plan()
                 `--> GemmOutputPlan

        GemmLocalReduceAnalysis + GemmOutputPlan
          `--> FlexGemmEpilogueAnalysis
                 `--> FlexGemmEpilogueEmitter

    At emitter construction, ``analysis.outputs`` becomes ``self.outputs``;
    its local-reduce plan and optional store select the QuACK reduction op and
    the compressed-store finalizer. ``analysis.required_geometries`` determines
    the active grouped layouts.

    The emitter owns all mutable code-generation state: FX values lowered so far,
    grouped TensorSSA layouts, the local-reduce spec (sink, prepass, finalizer),
    and the callback source slices. ``lower_graph`` performs a topological
    traversal and delegates each ``call_function`` node to ordered handlers;
    ``render`` turns the resulting state into the generated EpiMod and callback
    source.
    """

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
    ) -> None:
        self.graph_module = graph_module
        self.gemm = analysis.gemm
        self.analysis = analysis
        self.outputs = analysis.outputs
        self.local_reduce = self.outputs.local_reduce
        self.output_contraction_select_indices = (
            analysis.output_contraction_select_indices
        )
        self.output_contraction_layouts = analysis.output_contraction_layouts
        local_reduce_store = (
            None if self.local_reduce is None else self.local_reduce.store
        )
        self.local_reduce_output_storage = (
            None if local_reduce_store is None else local_reduce_store.output_storage
        )
        self.output_storage = self.outputs.output_storage
        self.output_storage_nodes = frozenset(self.outputs.output_storage_nodes)
        self.local_reduce_spec: FlexGemmEpiModLocalReduceSpec | None = None
        self.local_reduce_prepass: FlexGemmEpiModReductionSpec | None = None
        self.local_reduce_source_nodes: frozenset[torch.fx.Node] = frozenset()
        self.local_reduce_finalize_nodes: frozenset[torch.fx.Node] = frozenset()
        self.local_reduce_finalize_uses_prepass = False
        self.local_reduce_finalize_captures: tuple[torch.fx.Node, ...] = ()
        self.local_reduce_finalize_body: tuple[str, ...] = ()
        self.local_reduce_finalize_result: Any | None = None
        self.local_reduce_prepass_body: tuple[str, ...] = ()
        self.local_reduce_prepass_result: Any | None = None
        self.epilogue_arg_placeholders = epilogue_arg_placeholders
        self.epilogue_arg_kinds = epilogue_arg_kinds
        self.operand_names = tuple(
            f"operand{index}" for index in range(len(epilogue_arg_placeholders))
        )
        if self.local_reduce is not None:
            spec = epimod_local_reduce_spec(analysis, self.local_reduce)
            self.local_reduce_spec = spec
            sink = spec.sink
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
                and self.local_reduce.match.geometry.axis == 1
            ):
                prepass = sink
            self.local_reduce_prepass = prepass
            if (
                (not self.local_reduce.feeds_main or prepass is not None)
                and self.local_reduce.store is not None
                and self.local_reduce.store.value_node is not sink.node
            ):
                sink_aliases = frozenset(sink.aliases)
                prepass_aliases = frozenset(() if prepass is None else prepass.aliases)
                self.local_reduce_finalize_nodes = epimod_dependency_slice(
                    self.local_reduce.store.value_node,
                    sink_aliases | prepass_aliases,
                )
                self.local_reduce_finalize_captures = tuple(
                    node
                    for node in epilogue_arg_placeholders
                    if node in self.local_reduce_finalize_nodes
                )
                # Only floating scalar captures travel to the sink's finalizer;
                # row/col/tile captures have no per-group value.
                if any(
                    kind != "scalar" or not node.meta["val"].dtype.is_floating_point
                    for node, kind in zip(
                        epilogue_arg_placeholders, epilogue_arg_kinds, strict=True
                    )
                    if node in self.local_reduce_finalize_captures
                ):
                    raise NotImplementedError(LOCAL_REDUCE_FINALIZE_CAPTURE_ERROR)
                self.local_reduce_finalize_uses_prepass = bool(
                    self.local_reduce_finalize_nodes & (prepass_aliases - sink_aliases)
                )
        self.alpha = alpha
        self.beta = beta
        self.fast_math = fast_math
        self.kernel = GemmEpilogueCuteDSLKernel()
        self.params = ["acc"]
        self.base_env = self.initial_env_for_params(self.params)
        if self.local_reduce is not None and (
            self.local_reduce.feeds_main or self.local_reduce_prepass is not None
        ):
            self.params.append(LOCAL_REDUCE_FEED_MAIN_ARG_NAME)
        self.local_reduce_prepass_value: CuteDSLCSEVariable | None = None
        self.env = dict(self.base_env)

    @property
    def local_reduce_finalize_operands(self) -> tuple[str, ...]:
        """Return the scalar operand names the compressed-store finalizer reads."""
        return tuple(
            name
            for node, name in zip(
                self.epilogue_arg_placeholders, self.operand_names, strict=True
            )
            if node in self.local_reduce_finalize_captures
        )

    @staticmethod
    def value(name: str, dtype: torch.dtype) -> CuteDSLCSEVariable:
        """Represent one generated EpiMod TensorSSA value with dtype metadata."""
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
        """Lower a compressed-output transform as a fragment QuACK finalizer."""
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
        reduction_meta = sink.node.meta.get("val")
        dtype = (
            reduction_meta.dtype
            if isinstance(reduction_meta, torch.Tensor)
            else torch.float32
        )
        value = "value"
        if sink.finalize == "mean":
            value = f"(value / {float(local_reduce.match.geometry.group)!r})"
        kernel = GemmEpilogueCuteDSLKernel()
        reduced = self.value(value, dtype)
        env: dict[torch.fx.Node, Any] = dict.fromkeys(sink.aliases, reduced)
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
        env.update(
            (node, self.base_env[node]) for node in self.local_reduce_finalize_captures
        )
        with (
            V.set_kernel_handler(kernel),
            V.set_ops_handler(FlexGemmCuteDSLOpOverrides()),
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
            V.set_ops_handler(FlexGemmCuteDSLOpOverrides()),
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

    def lower_output_contraction_layout(
        self, node: torch.fx.Node, layout: FlexGemmLocalReduceGeometry
    ) -> None:
        """Reshape one physical TensorSSA fragment into adjacent-N lane groups."""
        if node.target is torch.ops.aten.split.Tensor:
            source_node = node.args[0]
        else:
            view_args = view_or_reshape_args(node)
            if view_args is None:
                raise AssertionError("grouped main layout requires a view or split")
            source_node = view_args[0]
        source = flex_gemm_epilogue_arg(source_node, self.env)
        fragment_group = (
            f"cutlass.const_expr(min({layout.group}, "
            f"cute.size({source}.shape, mode=[0])))"
        )
        repeats = (
            f"cutlass.const_expr(cute.size({source}.shape, mode=[0]) "
            f"// min({layout.group}, cute.size({source}.shape, mode=[0])))"
        )
        grouped = self.kernel.cse.generate(
            self.kernel.body,
            f"{source}.reshape(((1, {fragment_group}, {repeats}), 1, 1))",
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
                for index in range(layout.group)
            )
        else:
            self.env[node] = grouped

    def lower_output_contraction_select(self, node: torch.fx.Node, index: int) -> None:
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
            V.set_ops_handler(FlexGemmCuteDSLOpOverrides()),
            use_cutedsl_fast_math(self.fast_math),
        ):
            for node in self.graph_module.graph.nodes:
                if node is self.gemm or node.op in ("placeholder", "output"):
                    continue
                if (
                    (
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
                if node in self.output_contraction_layouts:
                    self.lower_output_contraction_layout(
                        node, self.output_contraction_layouts[node]
                    )
                    continue
                if node in self.output_contraction_select_indices:
                    self.lower_output_contraction_select(
                        node, self.output_contraction_select_indices[node]
                    )
                    continue
                if node in self.output_storage_nodes:
                    if self.output_storage is None:
                        raise AssertionError("output storage wrapper requires a source")
                    self.env[node] = flex_gemm_epilogue_arg(
                        self.output_storage, self.env
                    )
                    continue
                if (
                    self.local_reduce_output_storage is not None
                    and node in self.local_reduce_output_storage.nodes
                ):
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
                    if local_reduce.feeds_main:
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
                        self.env[node] = source
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
        main_name = "main" if self.outputs.output_contraction is not None else "D"
        result_items = [
            (main_name, main_result),
            *zip(aux_names, aux_results, strict=True),
        ]
        if (
            self.local_reduce is not None
            and sink is not None
            and self.local_reduce.store is not None
            and (
                not self.local_reduce.feeds_main
                or self.local_reduce_prepass is not None
            )
        ):
            store_value = flex_gemm_epilogue_arg(sink.source, self.env)
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
            f"inline_asm={inline_asm_cache_key()}\n{self.graph_module.code}\n"
            f"{body}return {{{return_source}}}\n"
            f"{finalize_payload}{prepass_payload}{self.epilogue_arg_kinds!r}"
        )
        key = hashlib.sha256(key_payload.encode()).hexdigest()[:16]
        name = f"flex_gemm_epilogue_{key}"
        finalize_name = None
        finalize_source = ""
        if self.local_reduce_finalize_result is not None:
            finalize_name = f"{name}_local_reduce_finalize"
            finalize_params = ", ".join(
                (
                    "value",
                    *(
                        ("prepass_value",)
                        if self.local_reduce_finalize_uses_prepass
                        else ()
                    ),
                    *self.local_reduce_finalize_operands,
                )
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
                f"{generated_imports}{finalize_source}{prepass_source}"
                f"@cute.jit\ndef {name}({', '.join(self.params)}):\n"
                f"{body}    return {{{return_source}}}\n"
            ),
            local_reduce_combine=None if sink is None else sink.combine,
            local_reduce_finalize=(
                None
                if sink is None
                else sink.finalize
                if self.local_reduce_prepass is not None
                else finalize_name or sink.finalize
            ),
            local_reduce_finalize_operands=(
                () if finalize_name is None else self.local_reduce_finalize_operands
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
        )

    def materialize(self) -> FlexGemmEpiModSource:
        """Lower and render this epilogue under the CuTeDSL virtualized handlers."""
        self.lower_local_reduce_finalize()
        self.lower_local_reduce_prepass()
        self.lower_graph()
        return self.render()


def materialize_flex_gemm_epilogue(
    graph_module: torch.fx.GraphModule,
    analysis: FlexGemmEpilogueAnalysis,
    epilogue_arg_placeholders: tuple[torch.fx.Node, ...],
    alpha: float,
    beta: float,
    epilogue_arg_kinds: tuple[str, ...],
    *,
    fast_math: bool = False,
) -> FlexGemmEpiModSource:
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
        fast_math: Whether supported CuTeDSL math operations may use approximate
            fast-math lowering.

    Returns:
        The generated epilogue function name and complete CuTeDSL source.
    """
    return FlexGemmEpilogueEmitter(
        graph_module,
        analysis,
        epilogue_arg_placeholders,
        alpha,
        beta,
        epilogue_arg_kinds,
        fast_math=fast_math,
    ).materialize()
