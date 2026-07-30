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
from typing import Any

import torch
from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
    CuteDSLCSEVariable,
    upcast_compute_type,
    use_cutedsl_fast_math,
)
from torch._inductor.kernel import gemm_epilogue_analysis as _epilogue_analysis
from torch._inductor.kernel.flex_gemm.constraints import (
    FLEX_GEMM_OUTPUT_TENSOR_ERROR,
    FlexGemmLocalReduceGeometry,
    LOCAL_REDUCE_AUX_TENSORSSA_ERROR,
    LOCAL_REDUCE_COMBINE_FN_SUFFIX,
    LOCAL_REDUCE_FEED_MAIN_ARG_NAME,
    LOCAL_REDUCE_FINALIZE_FN_SUFFIX,
    LOCAL_REDUCE_FINALIZE_SCALAR_ONLY_ERROR,
    LOCAL_REDUCE_MIXED_MATCH_ERROR,
    LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR,
    LOCAL_REDUCE_POST_POINTWISE_FINALIZE_ERROR,
    LOCAL_REDUCE_SINGLE_PHYSICAL_FINALIZE_ERROR,
    local_reduce_unsupported_tensorssa_error,
)
from torch._inductor.kernel.flex_gemm.quack_reductions import (
    _cute_arg,
    _cute_call,
    _local_reduce_store_arg,
    FlexGemmPhysicalReduction,
    GroupedTensorSSALayout,
    is_shape_preserving_pointwise_node,
    lower_full_scalar,
    lower_getitem,
    lower_prepare_softmax_online,
    lower_squeeze,
    lower_tensorssa_reduce,
    lower_view_or_reshape,
    reduction_from_node,
    unsupported_reduction_from_node,
)
from torch._inductor.kernel.gemm_epilogue import iter_fx_node_inputs
from torch._inductor.kernel.gemm_epilogue_codegen import (
    GemmEpilogueCuteDSLKernel,
    GemmEpilogueCuteDSLOpOverrides,
)
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.value_ranges import ValueRanges


FlexGemmEpilogueGraph = _epilogue_analysis.GemmEpilogueGraph
FlexGemmLocalReduceAnalysis = _epilogue_analysis.GemmLocalReduceAnalysis
FlexGemmLocalReduceMatch = _epilogue_analysis.GemmLocalReduceMatch
FlexGemmLocalReduceStore = _epilogue_analysis.GemmLocalReduceStore
FlexGemmOutputLocalReducePlan = _epilogue_analysis.GemmOutputLocalReducePlan
FlexGemmOutputPlan = _epilogue_analysis.GemmOutputPlan


FlexGemmCuteDSLKernel = GemmEpilogueCuteDSLKernel
FlexGemmCuteDSLOpOverrides = GemmEpilogueCuteDSLOpOverrides


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
        *,
        fast_math: bool = False,
    ) -> None:
        self.graph_module = graph_module
        self.epilogue_arg_placeholders = epilogue_arg_placeholders
        self.fast_math = fast_math
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
            GroupedTensorSSALayout(group=geometry.group, axis=geometry.axis)
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
    gemm_op: torch._ops.OpOverload,
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
        gemm_op,
        analysis,
        epilogue_arg_placeholders,
        fast_math=fast_math,
    ).materialize()
