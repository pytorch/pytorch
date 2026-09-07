# mypy: allow-untyped-defs
"""Lower FlexGEMM HOP bodies through ordinary Inductor or QuACK EpiMod.

``flex_gemm_lowering`` dispatches non-QUACK requests through ordinary subgraph
lowering and routes QUACK requests through shared analysis and one EpiMod choice.
"""

from __future__ import annotations

import dataclasses
import importlib.util
from typing import Any, TYPE_CHECKING

import torch
import torch.utils._pytree as pytree
from torch._higher_order_ops.flex_gemm import (
    _SUPPORTED_FLEX_GEMM_OP_NAMES,
    flex_gemm_body_gemm_op,
    flex_gemm_hop,
    FLEX_GEMM_OP_SPECS,
)
from torch.fx.operator_schemas import normalize_function
from torch.utils._ordered_set import OrderedSet

from ... import config, ir
from ...ir import IRNode, TensorBox
from ...lowering import empty_strided, process_subgraph_nodes, register_lowering
from ...utils import _IntLike, ceildiv
from ..gemm_epilogue_utils import statically_known_shape_equal
from .configs import flex_gemm_search_space
from .constraints import aux_output_shape_error, LOCAL_REDUCE_DENSE_MM_SCOPE_ERROR
from .debug import (
    format_flex_gemm_analysis,
    format_flex_gemm_analysis_details,
    format_flex_gemm_config_candidates,
    format_flex_gemm_lowering_plan,
    format_flex_gemm_problem,
    log_flex_gemm_artifact,
)


if TYPE_CHECKING:
    from collections.abc import Sequence


def decompose_nvgemm_additive_gemm(graph_module: torch.fx.GraphModule) -> None:
    """Rewrite additive GEMMs so NVGEMM can fuse their pointwise epilogues."""
    graph = graph_module.graph
    changed = False
    for node in list(graph.nodes):
        if node.target not in (
            torch.ops.aten.addmm.default,
            torch.ops.aten.baddbmm.default,
        ):
            continue
        bias, mat1, mat2 = node.args[:3]
        alpha = node.kwargs.get("alpha", 1.0)
        beta = node.kwargs.get("beta", 1.0)
        gemm_target = (
            torch.ops.aten.mm.default
            if node.target is torch.ops.aten.addmm.default
            else torch.ops.aten.bmm.default
        )
        with graph.inserting_before(node):
            result = graph.call_function(gemm_target, (mat1, mat2))
            if alpha != 1:
                result = graph.call_function(torch.ops.aten.mul.Tensor, (result, alpha))
            if beta != 0:
                if beta != 1:
                    bias = graph.call_function(torch.ops.aten.mul.Tensor, (bias, beta))
                result = graph.call_function(torch.ops.aten.add.Tensor, (result, bias))
        result.meta = node.meta
        node.replace_all_uses_with(result)
        graph.erase_node(node)
        changed = True
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()


class QuackScaledMmUnsupported(NotImplementedError):
    """Request ordinary lowering before FlexGEMM mutates the graph or realizes IR."""


def has_flex_gemm_quack() -> bool:
    """Whether the vendored QuACK backend can import its CuTeDSL dependency."""
    return importlib.util.find_spec("cutlass") is not None


# QuACK epilogue captures, aux outputs and local reductions are validated for
# these 2-D, bias-free GEMMs only.
QUACK_EPILOGUE_FEATURE_OPS = frozenset(
    (torch.ops.aten.mm.default, torch.ops.aten._scaled_mm_v2.default)
)

_BLOCKWISE_1X16 = torch.nn.functional.ScalingType.BlockWise1x16.value
_BLOCKWISE_1X32 = torch.nn.functional.ScalingType.BlockWise1x32.value
_TENSORWISE = torch.nn.functional.ScalingType.TensorWise.value
_SWIZZLE_32_4_4 = torch.nn.functional.SwizzleType.SWIZZLE_32_4_4.value
_NO_SWIZZLE = torch.nn.functional.SwizzleType.NO_SWIZZLE.value
# Per-operand scale recipe -> (QuACK format, required swizzles, data dtype,
# block-scale dtype). Both operands must use the same entry.
QUACK_BLOCKSCALED_RECIPES: dict[
    tuple[int, ...], tuple[str, tuple[int, ...], torch.dtype, torch.dtype]
] = {
    (_BLOCKWISE_1X32,): (
        "mxfp8_e4m3",
        (_SWIZZLE_32_4_4,),
        torch.float8_e4m3fn,
        torch.float8_e8m0fnu,
    ),
    (_BLOCKWISE_1X16,): (
        "nvfp4",
        (_SWIZZLE_32_4_4,),
        torch.float4_e2m1fn_x2,
        torch.float8_e4m3fn,
    ),
    (_BLOCKWISE_1X16, _TENSORWISE): (
        "nvfp4",
        (_SWIZZLE_32_4_4, _NO_SWIZZLE),
        torch.float4_e2m1fn_x2,
        torch.float8_e4m3fn,
    ),
}


@dataclasses.dataclass(frozen=True)
class QuackBlockScaledContract:
    """One traced aten._scaled_mm_v2 call resolved to QuACK's block-scaled main loop.

    ``gemm_inputs`` is (A, B, SFA, SFB) in template-input order;
    ``tensorwise_scales`` are the optional NVFP4 global scales folded into the
    epilogue as scalar operands.
    """

    format: str
    gemm_inputs: tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node, torch.fx.Node]
    tensorwise_scales: tuple[torch.fx.Node, ...]


def quack_blockscaled_contract(gemm_fx_node: torch.fx.Node) -> QuackBlockScaledContract:
    """Resolve the contract or raise QuackScaledMmUnsupported to request ordinary lowering."""
    normalized = normalize_function(
        torch.ops.aten._scaled_mm_v2.default,
        gemm_fx_node.args,
        gemm_fx_node.kwargs,
        normalize_to_only_use_kwargs=True,
    )
    if normalized is None:
        raise AssertionError("aten._scaled_mm_v2 arguments must bind to its schema")
    call = normalized.kwargs
    recipe = tuple(call["recipe_a"])
    contract = QUACK_BLOCKSCALED_RECIPES.get(recipe)
    if contract is None or tuple(call["recipe_b"]) != recipe:
        raise QuackScaledMmUnsupported(
            "FlexGEMM QUACK scaled-mm currently supports matching "
            "BlockWise1x32 MXFP8 or BlockWise1x16 NVFP4 recipes, with "
            "optional NVFP4 TensorWise global scales"
        )
    format_name, swizzles, data_dtype, scale_dtype = contract
    scale_a, scale_b = tuple(call["scale_a"]), tuple(call["scale_b"])
    if (
        tuple(call["swizzle_a"]) != swizzles
        or tuple(call["swizzle_b"]) != swizzles
        or len(scale_a) != len(recipe)
        or len(scale_b) != len(recipe)
        or call["bias"] is not None
        or call["contraction_dim"]
        or call["use_fast_accum"]
    ):
        raise QuackScaledMmUnsupported(
            "FlexGEMM QUACK scaled-mm requires one SWIZZLE_32_4_4 block "
            "scale per operand, optional unswizzled NVFP4 TensorWise scales, "
            "and no bias, custom contraction, or fast accumulation"
        )
    gemm_inputs = (call["input"], call["mat2"], scale_a[0], scale_b[0])
    if tuple(node.meta["val"].dtype for node in gemm_inputs) != (
        data_dtype,
        data_dtype,
        scale_dtype,
        scale_dtype,
    ):
        raise QuackScaledMmUnsupported(
            f"FlexGEMM QUACK {format_name} scaled-mm requires "
            f"{data_dtype} data and {scale_dtype} scales"
        )
    tensorwise_scales = (*scale_a[1:], *scale_b[1:])
    for node in tensorwise_scales:
        meta = node.meta["val"]
        if meta.dtype is not torch.float32 or not any(
            statically_known_shape_equal(
                ir.convert_shape_to_inductor(meta.shape), shape
            )
            for shape in ([], [1], [1, 1])
        ):
            raise QuackScaledMmUnsupported(
                "FlexGEMM NVFP4 TensorWise scales must be scalar Float32 tensors"
            )
    return QuackBlockScaledContract(format_name, gemm_inputs, tensorwise_scales)


def flex_gemm_tensor_placeholders(
    graph_module: torch.fx.GraphModule,
) -> list[torch.fx.Node]:
    """Return placeholders QuACK can bind as tensor epilogue arguments.

    FlexGEMM identifies the GEMM A/B inputs from the mm node, then treats the
    remaining tensor-valued placeholders as closed-over epilogue tensors. Scalar
    SymInt placeholders are shape values, not tensor arguments; the current QuACK
    FlexGEMM entrypoint has no scalar epilogue-argument slots for them.
    """
    return [
        node
        for node in graph_module.graph.nodes
        if node.op == "placeholder" and isinstance(node.meta.get("val"), torch.Tensor)
    ]


def flex_gemm_epilogue_arg_placeholders(
    graph_module: torch.fx.GraphModule, gemm_fx_node: torch.fx.Node
) -> tuple[torch.fx.Node, ...]:
    """Find tensor inputs captured by epilogue loads, excluding GEMM operands."""
    gemm_placeholders = OrderedSet(
        arg
        for arg in pytree.tree_leaves((gemm_fx_node.args, gemm_fx_node.kwargs))
        if isinstance(arg, torch.fx.Node)
    )
    return tuple(
        node
        for node in flex_gemm_tensor_placeholders(graph_module)
        if node not in gemm_placeholders
    )


def infer_flex_gemm_epilogue_arg_kinds(
    gemm_op: torch._ops.OpOverload,
    epilogue_args: list[IRNode],
    output_size: list[Any],
) -> tuple[str, ...]:
    """Classify realized captured epilogue tensors for static wrapper kwargs."""
    if not epilogue_args:
        return ()
    if gemm_op not in QUACK_EPILOGUE_FEATURE_OPS:
        raise NotImplementedError(
            "FlexGEMM generated epilogues with captured tensor reads currently "
            "support only aten.mm and aten._scaled_mm_v2"
        )
    m, n = output_size[-2], output_size[-1]
    epilogue_arg_kinds = []
    for epilogue_arg in epilogue_args:
        epilogue_arg_size = epilogue_arg.get_size()
        if statically_known_shape_equal(epilogue_arg_size, [1, 1]):
            epilogue_arg_kinds.append("scalar")
        elif statically_known_shape_equal(epilogue_arg_size, output_size):
            epilogue_arg_kinds.append("tile")
        elif statically_known_shape_equal(epilogue_arg_size, [1, n]):
            epilogue_arg_kinds.append("row")
        elif statically_known_shape_equal(epilogue_arg_size, [m, 1]):
            epilogue_arg_kinds.append("col")
        else:
            raise NotImplementedError(
                "FlexGEMM captured tensor epilogue args currently must match "
                "the GEMM output shape or broadcast as [1, N] / [M, 1] / [1, 1]"
            )
    return tuple(epilogue_arg_kinds)


def validate_flex_gemm_aux_outputs(
    gemm_op: torch._ops.OpOverload,
    aux_outputs: tuple[torch.fx.Node, ...],
    output_size: Sequence[_IntLike],
) -> tuple[Any, ...]:
    """Validate QUACK aux-output support and return fake tensor metadata."""
    if not aux_outputs:
        return ()
    if gemm_op not in QUACK_EPILOGUE_FEATURE_OPS:
        raise NotImplementedError(
            "FlexGEMM generic aux tuple epilogues currently support only "
            "aten.mm and aten._scaled_mm_v2"
        )
    aux_metas = []
    for aux_output in aux_outputs:
        aux_meta = aux_output.meta.get("val")
        if aux_meta is None:
            raise NotImplementedError(
                "FlexGEMM generic aux tuple epilogues require aux output metadata"
            )
        aux_size = ir.convert_shape_to_inductor(aux_meta.shape)
        if not statically_known_shape_equal(aux_size, output_size):
            raise aux_output_shape_error(aux_size, output_size)
        aux_metas.append(aux_meta)
    return tuple(aux_metas)


def allocate_flex_gemm_aux_outs(
    aux_metas: tuple[Any, ...], mat1: TensorBox
) -> tuple[TensorBox, ...]:
    """Allocate same-shape aux output buffers beside the main GEMM output."""
    return tuple(
        empty_strided(
            ir.convert_shape_to_inductor(aux_meta.shape),
            ir.convert_shape_to_inductor(aux_meta.stride()),
            dtype=aux_meta.dtype,
            device=mat1.get_device_or_error(),
        )
        for aux_meta in aux_metas
    )


def append_flex_gemm_template_inputs(
    input_nodes: list[IRNode], nodes: list[IRNode]
) -> tuple[int, ...]:
    """Append template inputs and return their assigned positions."""
    start = len(input_nodes)
    input_nodes.extend(nodes)
    return tuple(range(start, len(input_nodes)))


def flex_gemm_local_reduce_metas(local_reduce) -> tuple[Any, ...]:
    """Return metadata for the optional compressed local-reduce output."""
    if local_reduce is None or local_reduce.store is None:
        return ()
    return (local_reduce.store.node.meta["val"],)


def flex_gemm_quack_configs(
    template: Any, template_kwargs: dict[str, Any], config: Any
) -> tuple[tuple[tuple[str, Any], ...], ...]:
    """Ask QuACK which GemmConfigs this generated call may pin, default first."""
    from torch._inductor.codecache import PyCodeCache
    from torch._inductor.codegen.cutedsl.cutedsl_kernel import MAIN_SUFFIX
    from torch._inductor.kernel.flex_gemm.runtime import select_flex_gemm_configs
    from torch._subclasses.fake_tensor import FakeTensorMode

    probe: list[Any] = []
    error = template.maybe_append_choice(probe, config=config, **template_kwargs)
    if error is not None:
        raise error
    bmreq = probe[0].bmreq
    module = PyCodeCache.load_by_key_path(
        bmreq.module_cache_key, bmreq.module_path, set_sys_modules=False
    )
    main = getattr(module, f"{bmreq.kernel_name}_{MAIN_SUFFIX}")
    with FakeTensorMode():
        inputs = [meta.to_tensor() for meta in bmreq.input_tensor_meta]
        output = bmreq.output_tensor_meta.to_tensor()
    with select_flex_gemm_configs() as legal_configs:
        main(*inputs, output, *bmreq.extra_args, stream=None)
    if not legal_configs:
        raise AssertionError("FlexGEMM config probe did not reach gemm_epimod")
    return tuple(
        tuple(sorted(dataclasses.asdict(quack_config).items()))
        for quack_config in legal_configs
    )


def flex_gemm_autotune_view_input(node: ir.ReinterpretView) -> torch.Tensor:
    """Rebuild a logical view for Python-backed template benchmarks."""
    from torch._inductor.select_algorithm import (
        AlgorithmSelectorCache,
        get_strides_with_layout_constraints,
    )
    from torch._inductor.virtualized import V

    value = AlgorithmSelectorCache.benchmark_example_value(node)
    base = value if value._base is None else value._base
    sizevars = V.graph.sizevars
    sizes = sizevars.optimization_hints(node.get_size())
    strides = sizevars.optimization_hints(get_strides_with_layout_constraints(node))
    offset = sizevars.optimization_hint(node.get_layout().offset)
    return torch.as_strided(base, sizes, strides, offset)


def lower_quack_flex_gemm(gemm_op, subgraph, args, gemm_kwargs, kernel_options):
    """Lower FlexGEMM analysis into one generated QuACK EpiMod choice."""
    if gemm_op not in FLEX_GEMM_OP_SPECS:
        raise NotImplementedError(
            f"FlexGEMM QUACK backend currently supports only aten.{_SUPPORTED_FLEX_GEMM_OP_NAMES}"
        )
    tuned = kernel_options.get("tuned", False)
    fast_math = kernel_options.get("fast_math", False)
    explicit_config = kernel_options.get("config")
    supported_options = OrderedSet(["backend", "tuned", "fast_math", "config"])
    unsupported_options = OrderedSet(kernel_options) - supported_options
    if unsupported_options:
        raise NotImplementedError(
            f"unsupported FlexGEMM kernel options: {sorted(unsupported_options)}"
        )
    if not isinstance(fast_math, bool):
        raise NotImplementedError("FlexGEMM fast_math kernel option must be bool")
    if "config" in kernel_options and not isinstance(explicit_config, dict):
        raise NotImplementedError("FlexGEMM config kernel option must be a dict")

    from torch._inductor.kernel.flex_gemm.epilogue import (
        analyze_flex_gemm_epilogue,
        expand_epimod_prepare_softmax_online,
        flex_gemm_indexed_output_plan,
        flex_gemm_output_values,
        gemm_node as flex_gemm_node,
        materialize_flex_gemm_epimod,
    )
    from torch._inductor.kernel.flex_gemm.template import (
        flex_gemm_epilogue_template,
        FlexGemmEpilogueBlockScaledConfig,
        FlexGemmEpilogueConfig,
        FlexGemmEpilogueIndexedOutputConfig,
        FlexGemmEpilogueLocalReduceConfig,
    )
    from torch._inductor.select_algorithm import autotune_select_algorithm

    op_spec = FLEX_GEMM_OP_SPECS[gemm_op]
    mat1_index = op_spec.mat1_index
    gemm_fx_node = flex_gemm_node(subgraph.graph_module, gemm_op)
    if gemm_op is torch.ops.aten._scaled_mm_v2.default:
        try:
            indexed_store = flex_gemm_indexed_output_plan(
                *flex_gemm_output_values(subgraph.graph_module)
            )
        except NotImplementedError as exc:
            raise QuackScaledMmUnsupported(
                "FlexGEMM QUACK scaled-mm does not yet support indexed outputs"
            ) from exc
        if indexed_store is not None:
            raise QuackScaledMmUnsupported(
                "FlexGEMM QUACK scaled-mm does not yet support indexed outputs"
            )
    placeholders = [
        node for node in subgraph.graph_module.graph.nodes if node.op == "placeholder"
    ]
    placeholder_args = dict(zip(placeholders, args, strict=True))
    if gemm_op is torch.ops.aten._scaled_mm_v2.default:
        blockscaled = quack_blockscaled_contract(gemm_fx_node)
        gemm_nodes = blockscaled.gemm_inputs
        mainloop_scale_nodes = blockscaled.tensorwise_scales
        alpha, beta = 1.0, 0.0
    else:
        blockscaled = None
        mainloop_scale_nodes = ()
        unsupported_gemm_kwargs = OrderedSet(gemm_kwargs) - OrderedSet(
            ["alpha", "beta"]
        )
        if unsupported_gemm_kwargs:
            raise NotImplementedError(
                f"unsupported FlexGEMM GEMM kwargs: {sorted(unsupported_gemm_kwargs)}"
            )
        gemm_nodes = gemm_fx_node.args
        alpha = gemm_fx_node.kwargs.get("alpha", gemm_kwargs.get("alpha", 1.0))
        beta = gemm_fx_node.kwargs.get("beta", gemm_kwargs.get("beta", 1.0))
        if not isinstance(alpha, (int, float)) or not isinstance(beta, (int, float)):
            raise NotImplementedError("FlexGEMM alpha/beta must be static scalars")

    gemm_args: list[TensorBox] = []
    for arg in gemm_nodes:
        gemm_arg = placeholder_args[arg] if isinstance(arg, torch.fx.Node) else arg
        if not isinstance(gemm_arg, TensorBox):
            raise NotImplementedError("FlexGEMM lowering expects tensor GEMM operands")
        gemm_args.append(gemm_arg)
    epilogue_arg_placeholders = (
        *mainloop_scale_nodes,
        *flex_gemm_epilogue_arg_placeholders(subgraph.graph_module, gemm_fx_node),
    )
    epilogue_args: list[TensorBox] = []
    for arg in epilogue_arg_placeholders:
        epilogue_arg = placeholder_args[arg]
        if not isinstance(epilogue_arg, TensorBox):
            raise NotImplementedError(
                "FlexGEMM lowering expects tensor epilogue operands"
            )
        epilogue_args.append(epilogue_arg)
    gemm_input_names = tuple(
        arg.name if isinstance(arg, torch.fx.Node) else f"gemm_arg{index}"
        for index, arg in enumerate(gemm_nodes)
    )
    log_flex_gemm_artifact(
        "problem",
        lambda: format_flex_gemm_problem(
            subgraph.graph_module,
            gemm_op,
            tuple(zip(gemm_input_names, gemm_args, strict=True)),
            tuple(
                zip(
                    (node.name for node in epilogue_arg_placeholders),
                    epilogue_args,
                    strict=True,
                )
            ),
            tuned=tuned,
            fast_math=fast_math,
            explicit_config=explicit_config,
        ),
        lowering_name=subgraph.name,
    )
    # Normalize supported online-softmax forms before shared analysis.
    expand_epimod_prepare_softmax_online(subgraph.graph_module)
    epilogue_analysis = analyze_flex_gemm_epilogue(subgraph.graph_module, gemm_fx_node)
    log_flex_gemm_artifact(
        "analysis",
        lambda: format_flex_gemm_analysis(epilogue_analysis),
        lowering_name=subgraph.name,
    )
    log_flex_gemm_artifact(
        "analysis_details",
        lambda: format_flex_gemm_analysis_details(epilogue_analysis),
        lowering_name=subgraph.name,
        verbose=True,
    )
    if (
        epilogue_analysis.required_geometries
        and gemm_op not in QUACK_EPILOGUE_FEATURE_OPS
    ):
        raise NotImplementedError(LOCAL_REDUCE_DENSE_MM_SCOPE_ERROR)
    outputs = epilogue_analysis.outputs
    indexed_output = outputs.indexed_output
    indexed_input = None
    if indexed_output is not None:
        if gemm_op is not torch.ops.aten.mm.default:
            raise NotImplementedError(
                "FlexGEMM indexed outputs currently support only aten.mm"
            )
        indexed_input = placeholder_args[indexed_output.indices]
        if not isinstance(indexed_input, TensorBox):
            raise NotImplementedError("FlexGEMM indexed outputs require tensor indices")
        epilogue_pairs = tuple(
            (placeholder, arg)
            for placeholder, arg in zip(
                epilogue_arg_placeholders, epilogue_args, strict=True
            )
            if placeholder is not indexed_output.indices
        )
        epilogue_arg_placeholders = tuple(
            placeholder for placeholder, _ in epilogue_pairs
        )
        epilogue_args = [arg for _, arg in epilogue_pairs]

    main_transform = outputs.main_transform
    if main_transform is not None and epilogue_args[len(mainloop_scale_nodes) :]:
        raise NotImplementedError(
            "FlexGEMM grouped main outputs do not yet support captured tensors"
        )
    local_reduce_store = (
        None if outputs.local_reduce is None else outputs.local_reduce.store
    )
    output_meta = outputs.output.meta.get("val")
    if output_meta is None:
        raise NotImplementedError(
            "FlexGEMM generated epilogues require output metadata"
        )
    output_size = ir.convert_shape_to_inductor(output_meta.shape)
    aux_metas = validate_flex_gemm_aux_outputs(
        gemm_op, outputs.aux_outputs, output_size
    )
    indexed_metas = () if indexed_output is None else (indexed_output.node.meta["val"],)
    if not has_flex_gemm_quack():
        raise NotImplementedError("FlexGEMM QUACK backend requires CuTeDSL")
    packed_uint8_main = main_transform is not None and output_meta.dtype is torch.uint8
    if (
        not output_meta.dtype.is_floating_point
        and output_meta.dtype is not torch.bool
        and not packed_uint8_main
    ):
        raise NotImplementedError(
            "FlexGEMM generic main outputs support only floating-point and bool dtypes"
        )
    local_reduce_metas = flex_gemm_local_reduce_metas(outputs.local_reduce)
    output_stride = ir.convert_shape_to_inductor(output_meta.stride())
    if main_transform is not None:
        # Grouped main outputs use TMA stores, whose outer stride must preserve
        # 16-byte alignment even when the contracted N extent is not aligned.
        output_alignment = max(16 // output_meta.dtype.itemsize, 1)
        output_stride[-2] = (
            ceildiv(output_size[-1], output_alignment) * output_alignment
        )
    layout = ir.FixedLayout(
        gemm_args[mat1_index].get_device_or_error(),
        output_meta.dtype,
        output_size,
        output_stride,
    )
    gemm_input_nodes = [
        ir.TemplateBuffer.realize_template_input(arg) for arg in gemm_args
    ]
    epilogue_input_nodes = [
        ir.TemplateBuffer.realize_template_input(arg) for arg in epilogue_args
    ]
    indexed_index_input_nodes = (
        []
        if indexed_input is None
        else [ir.TemplateBuffer.realize_template_input(indexed_input)]
    )
    aux_outs = allocate_flex_gemm_aux_outs(aux_metas, gemm_args[mat1_index])
    indexed_outs = allocate_flex_gemm_aux_outs(indexed_metas, gemm_args[mat1_index])
    local_reduce_outs = allocate_flex_gemm_aux_outs(
        local_reduce_metas, gemm_args[mat1_index]
    )
    aux_input_nodes = [
        ir.TemplateBuffer.realize_template_input(aux_out) for aux_out in aux_outs
    ]
    indexed_out_input_nodes = [
        ir.TemplateBuffer.realize_template_input(indexed_out)
        for indexed_out in indexed_outs
    ]
    local_reduce_input_nodes = [
        ir.TemplateBuffer.realize_template_input(local_reduce_out)
        for local_reduce_out in local_reduce_outs
    ]
    input_nodes: list[IRNode] = []
    gemm_input_indices = append_flex_gemm_template_inputs(input_nodes, gemm_input_nodes)
    epilogue_arg_indices = append_flex_gemm_template_inputs(
        input_nodes, epilogue_input_nodes
    )
    indexed_index_input_indices = append_flex_gemm_template_inputs(
        input_nodes, indexed_index_input_nodes
    )
    aux_out_indices = append_flex_gemm_template_inputs(input_nodes, aux_input_nodes)
    indexed_out_indices = append_flex_gemm_template_inputs(
        input_nodes, indexed_out_input_nodes
    )
    local_reduce_out_indices = append_flex_gemm_template_inputs(
        input_nodes, local_reduce_input_nodes
    )
    mutated_input_nodes = (
        aux_input_nodes + indexed_out_input_nodes + local_reduce_input_nodes
    )
    local_reduce_out_index = (
        local_reduce_out_indices[0] if local_reduce_out_indices else None
    )
    mainloop_scale_count = len(mainloop_scale_nodes)
    epilogue_arg_kinds = (
        *("scalar" for _ in range(mainloop_scale_count)),
        *infer_flex_gemm_epilogue_arg_kinds(
            gemm_op,
            epilogue_input_nodes[mainloop_scale_count:],
            output_size,
        ),
    )
    epimod_source = materialize_flex_gemm_epimod(
        subgraph.graph_module,
        epilogue_analysis,
        epilogue_arg_placeholders,
        float(alpha),
        float(beta),
        epilogue_arg_kinds,
        fast_math=fast_math,
        mainloop_scale_count=mainloop_scale_count,
    )
    log_flex_gemm_artifact(
        "lowering_plan",
        lambda: format_flex_gemm_lowering_plan(
            output_size,
            output_meta.dtype,
            tuple(
                zip(
                    (node.name for node in epilogue_arg_placeholders),
                    epilogue_arg_kinds,
                    strict=True,
                )
            ),
            aux_metas,
            indexed_metas,
            local_reduce_metas,
            local_reduce_layout=(
                None if local_reduce_store is None else local_reduce_store.output_layout
            ),
        ),
        lowering_name=subgraph.name,
    )
    log_flex_gemm_artifact(
        "generated_epilogue",
        lambda: epimod_source.source.strip(),
        lowering_name=subgraph.name,
        verbose=True,
    )
    template_indexed_output = None
    if indexed_output is not None:
        template_indexed_output = FlexGemmEpilogueIndexedOutputConfig(
            out_index=indexed_out_indices[0],
            indices_index=indexed_index_input_indices[0],
        )
    template_local_reduce = FlexGemmEpilogueLocalReduceConfig.from_output_plan(
        outputs.local_reduce,
        local_reduce_out_index,
        combine=epimod_source.local_reduce_combine,
        finalize=epimod_source.local_reduce_finalize,
        store_finalize=epimod_source.local_reduce_store_finalize,
        prepass_combine=epimod_source.local_reduce_prepass_combine,
        prepass_finalize=epimod_source.local_reduce_prepass_finalize,
    )
    template_config = FlexGemmEpilogueConfig(
        epilogue_name=epimod_source.name,
        epilogue_source=epimod_source.source,
        gemm_op=op_spec,
        alpha=float(alpha),
        beta=float(beta),
        blockscaled=(
            None
            if blockscaled is None
            else FlexGemmEpilogueBlockScaledConfig(
                blockscaled.format, *gemm_input_indices[2:]
            )
        ),
        quack_config_constraints=(
            tuple(sorted(explicit_config.items()))
            if explicit_config is not None
            else ()
        ),
        quack_config=None,
        epilogue_arg_indices=epilogue_arg_indices,
        epilogue_arg_kinds=epilogue_arg_kinds,
        aux_out_indices=aux_out_indices,
        indexed_output=template_indexed_output,
        local_reduce=template_local_reduce,
        main_transform=main_transform,
    )
    template_kwargs = dict(
        input_nodes=input_nodes,
        layout=layout,
        mutated_inputs=mutated_input_nodes or None,
    )
    legal_configs = flex_gemm_quack_configs(
        flex_gemm_epilogue_template, template_kwargs, template_config
    )
    quack_configs = (
        flex_gemm_search_space(legal_configs) if tuned else legal_configs[:1]
    )
    log_flex_gemm_artifact(
        "config_candidates",
        lambda: format_flex_gemm_config_candidates(quack_configs, tuned=tuned),
        lowering_name=subgraph.name,
    )
    choices: list[Any] = []
    for quack_config in quack_configs:
        error = flex_gemm_epilogue_template.maybe_append_choice(
            choices,
            config=dataclasses.replace(
                template_config,
                quack_config=quack_config,
                quack_config_constraints=(),
            ),
            **template_kwargs,
        )
        if error is not None:
            raise error
    input_gen_fns = {
        index: flex_gemm_autotune_view_input
        for index, input_node in enumerate(input_nodes)
        if isinstance(input_node, ir.ReinterpretView)
    }
    result, _ = autotune_select_algorithm(
        "flex_gemm_epilogue",
        choices,
        input_nodes,
        layout,
        input_gen_fns=input_gen_fns or None,
        **({"return_multi_template": False} if mutated_input_nodes else {}),
    )
    if len(choices) == 1:
        # A single choice skips autotuning, so overlap its kernel compile with
        # the rest of Inductor's compilation instead of paying it at first call.
        choices[0].precompile(wait=False)
    structural_outs = {}
    if indexed_output is not None:
        structural_outs[indexed_output.node] = indexed_outs[0]
    if local_reduce_store is not None:
        structural_outs[local_reduce_store.node] = local_reduce_outs[0]
    aux_iter = iter(aux_outs)
    ordered_aux_outs = [
        structural_outs[node] if node in structural_outs else next(aux_iter)
        for node in outputs.returned_aux_outputs
    ]
    return (result, *ordered_aux_outs)


@register_lowering(flex_gemm_hop, type_promotion_kind=None)
def flex_gemm_lowering(gemm_op, subgraph, args, gemm_kwargs, kernel_options):
    """Dispatch FlexGEMM to ordinary Inductor lowering or a backend template."""
    backend = kernel_options.get("backend", "TRITON")
    if backend == "NVGEMM":
        decompose_nvgemm_additive_gemm(subgraph.graph_module)
        with config.patch(
            max_autotune=True,
            max_autotune_gemm_backends="NVGEMM",
        ):
            return process_subgraph_nodes(subgraph.graph_module, list(args))
    body_gemm_op = flex_gemm_body_gemm_op(gemm_op, gemm_kwargs)
    if backend == "QUACK":
        try:
            return lower_quack_flex_gemm(
                body_gemm_op, subgraph, args, gemm_kwargs, kernel_options
            )
        except QuackScaledMmUnsupported as error:
            fallback_reason = str(error)
            log_flex_gemm_artifact(
                "fallback",
                lambda: fallback_reason,
                lowering_name=subgraph.name,
            )
            return process_subgraph_nodes(subgraph.graph_module, list(args))
    return process_subgraph_nodes(subgraph.graph_module, list(args))
