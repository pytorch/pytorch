# mypy: allow-untyped-defs
"""Lower FlexGEMM HOP bodies and connect epilogue analysis to backend templates.

``flex_gemm_lowering`` is the main entry point. Non-QUACK
requests execute the captured body through ordinary Inductor lowering. Although this
is stale and will fix up later on. See ``lower_quack_flex_gemm`` for the flow.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import importlib.util
import logging
from typing import Any, TYPE_CHECKING

import torch
import torch.utils._pytree as pytree
from torch._higher_order_ops.flex_gemm import (
    _SUPPORTED_FLEX_GEMM_OP_NAMES,
    flex_gemm_body_gemm_op,
    flex_gemm_hop,
    FLEX_GEMM_OP_SPECS,
)
from torch._inductor import config
from torch._logging import warning_once
from torch.fx.operator_schemas import normalize_function
from torch.utils._ordered_set import OrderedSet

from ... import ir
from ...heuristics.template.flex_gemm import (
    flex_gemm_default_config,
    flex_gemm_search_space,
    QuackConfigKey,
)
from ...ir import IRNode, TensorBox
from ...lowering import (
    constant_pad_nd,
    empty_strided,
    process_subgraph_nodes,
    register_lowering,
    view,
)
from ...utils import _IntLike, ceildiv, is_bf16x9_matmul
from ..gemm_epilogue_utils import statically_known_equal, statically_known_shape_equal
from .constraints import (
    aux_output_shape_error,
    FLEX_GEMM_CAPTURE_SHAPE_ERROR,
    LOCAL_REDUCE_DENSE_MM_SCOPE_ERROR,
)
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

    from .template import FlexGemmEpilogueConfig


log = logging.getLogger(__name__)


def decompose_nvgemm_additive_gemm(graph_module: torch.fx.GraphModule) -> None:
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


class QuackFallbackUnsupported(NotImplementedError):
    """Request ordinary lowering before FlexGEMM mutates the graph or realizes IR.

    Raised only for compositions QuACK cannot run (scaled-mm recipes, varlen gaps,
    fp32 without TF32); a pinned ``config`` turns it into a hard error instead of a
    silent fallback.
    """


def check_quack_fp32_operand(gemm_arg: TensorBox) -> None:
    """Reject CUDA float32 GEMMs unless the matmul policy permits TF32."""
    precision = torch.backends.cuda.matmul.fp32_precision
    if (
        gemm_arg.get_dtype() is not torch.float32
        or gemm_arg.get_device_or_error().type != "cuda"
        or precision == "tf32"
    ):
        return
    raise QuackFallbackUnsupported(
        "FlexGEMM QUACK computes float32 GEMM operands in TF32, but "
        f"torch.backends.cuda.matmul.fp32_precision is {precision!r}; "
        "opt in with torch.set_float32_matmul_precision('high') or use bfloat16/float16 operands"
    )


def has_flex_gemm_quack() -> bool:
    """Whether the vendored QuACK backend can import its CuTeDSL dependency."""
    return importlib.util.find_spec("cutlass") is not None


# QuACK epilogue captures, aux outputs and local reductions are validated for
# these 2-D, bias-free GEMMs only; grouped_mm additionally rejects the
# features QuACK's varlen-M path lacks.
QUACK_EPILOGUE_FEATURE_OPS = frozenset(
    (
        torch.ops.aten.mm.default,
        torch.ops.aten._scaled_mm_v2.default,
        torch.ops.aten._grouped_mm.default,
    )
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
    """Resolve the contract or raise QuackFallbackUnsupported to request ordinary lowering."""
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
        raise QuackFallbackUnsupported(
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
        raise QuackFallbackUnsupported(
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
        raise QuackFallbackUnsupported(
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
            raise QuackFallbackUnsupported(
                "FlexGEMM NVFP4 TensorWise scales must be scalar Float32 tensors"
            )
    return QuackBlockScaledContract(format_name, gemm_inputs, tensorwise_scales)


def quack_grouped_mm_contract(
    gemm_fx_node: torch.fx.Node,
) -> tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node]:
    """Resolve ``(mat_a, mat_b, offs)`` for varlen-M or request ordinary lowering.

    QuACK's varlen path is the MoE forward form: k-major bf16/fp16 A
    ``[total_m, K]``, per-group B ``[E, K, N]`` and int32 ``offs[E]`` end offsets,
    which the runtime turns into ``cu_seqlens_m = [0, *offs]``.
    """
    normalized = normalize_function(
        torch.ops.aten._grouped_mm.default,
        gemm_fx_node.args,
        gemm_fx_node.kwargs,
        normalize_to_only_use_kwargs=True,
    )
    if normalized is None:
        raise AssertionError("aten._grouped_mm arguments must bind to its schema")
    call = normalized.kwargs
    mat_a, mat_b, offs = call["input"], call["mat2"], call["offs"]
    if offs is None or call["bias"] is not None or call["out_dtype"] is not None:
        raise QuackFallbackUnsupported(
            "FlexGEMM QUACK grouped_mm requires offs and no bias or out_dtype"
        )
    a_meta, b_meta, offs_meta = (node.meta["val"] for node in (mat_a, mat_b, offs))
    if (
        a_meta.ndim != 2
        or a_meta.stride(-1) != 1
        or b_meta.ndim != 3
        or a_meta.dtype not in (torch.bfloat16, torch.float16)
        or b_meta.dtype is not a_meta.dtype
        or offs_meta.dtype is not torch.int32
    ):
        raise QuackFallbackUnsupported(
            "FlexGEMM QUACK grouped_mm supports only bf16/fp16 k-major 2-D A "
            "[total_m, K], 3-D B [E, K, N] and int32 offs"
        )
    return mat_a, mat_b, offs


def flex_gemm_cu_seqlens_benchmark_input(
    node: IRNode, total_m: _IntLike
) -> torch.Tensor:
    """Build evenly spaced ``[0, ..., total_m]`` boundaries for autotune benchmarks."""
    from torch._inductor.virtualized import V

    sizevars = V.graph.sizevars
    return torch.linspace(
        0,
        sizevars.optimization_hint(total_m),
        sizevars.optimization_hint(node.get_size()[0]),
        dtype=torch.int32,
        device=node.get_device_or_error(),
    )


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
            "support only aten.mm, aten._scaled_mm_v2 and aten._grouped_mm"
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
                f"{FLEX_GEMM_CAPTURE_SHAPE_ERROR}; got {list(epilogue_arg_size)} "
                f"against GEMM output {list(output_size)}"
            )
    return tuple(epilogue_arg_kinds)


CAPTURE_RESHAPE_TARGETS = frozenset(
    (
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
    )
)


def capture_reshape_shape(node: torch.fx.Node) -> tuple[Any, ...] | None:
    """Return the shape a body reshape gives a captured tensor, else None."""
    meta = node.meta.get("val")
    if node.target not in CAPTURE_RESHAPE_TARGETS or not isinstance(meta, torch.Tensor):
        return None
    return tuple(meta.shape)


def flex_gemm_1d_capture_shape(
    node: torch.fx.Node, m: Any, n: Any
) -> tuple[Any, Any] | None:
    """Return the [1, N] / [M, 1] / [1, 1] reading of one 0-D or 1-D capture.

    ``w[:, None]`` reads a length-M capture as a column; direct broadcasting and
    ``w[None, :]`` read a length-N capture as a row, so an [N] capture with
    M == N follows ``acc + w`` semantics. Return ``None`` when the length fits
    neither reading, leaving the capture for the later shape check.
    """
    meta = node.meta["val"]
    if meta.ndim > 1:
        return None
    length = meta.numel()
    if statically_known_equal(length, 1):
        return (1, 1)
    column_uses = [
        (shape := capture_reshape_shape(user)) is not None
        and statically_known_shape_equal(shape, (length, 1))
        for user in node.users
    ]
    if all(column_uses):
        return (m, 1) if statically_known_equal(length, m) else None
    if any(column_uses):
        raise NotImplementedError(
            f"FlexGEMM capture {node.name} is used as both a row and a column"
        )
    return (1, n) if statically_known_equal(length, n) else None


def normalize_flex_gemm_1d_captures(
    graph_module: torch.fx.GraphModule,
    placeholders: Sequence[torch.fx.Node],
    epilogue_args: list[TensorBox],
    gemm_shape: Sequence[Any],
    *,
    indices: torch.fx.Node | None,
) -> list[TensorBox]:
    """Rewrite 0-D/1-D captures to the 2-D broadcast views QuACK binds.

    The placeholder metadata and the realized template input become the 2-D
    view, and body reshapes producing that view fold into the placeholder so
    ``w[None, :]`` inside the epilogue lowers like a hoisted ``[1, N]`` capture.
    Indexed-output ``indices`` stay 1-D for the gather store.
    """
    m, n = gemm_shape[-2:]
    normalized = list(epilogue_args)
    changed = False
    for index, (node, arg) in enumerate(zip(placeholders, epilogue_args, strict=True)):
        shape = None if node is indices else flex_gemm_1d_capture_shape(node, m, n)
        if shape is None:
            continue
        for user in tuple(node.users):
            user_shape = capture_reshape_shape(user)
            if user_shape is not None and statically_known_shape_equal(
                user_shape, shape
            ):
                user.replace_all_uses_with(node)
                graph_module.graph.erase_node(user)
        node.meta["val"] = node.meta["val"].view(shape)
        normalized[index] = view(arg, ir.convert_shape_to_inductor(shape))
        changed = True
    if changed:
        graph_module.graph.lint()
        graph_module.recompile()
    return normalized


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
            "aten.mm, aten._scaled_mm_v2 and aten._grouped_mm"
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
    """Allocate auxiliary buffers with their requested dense strides."""
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
    template_config: FlexGemmEpilogueConfig, input_nodes: list[IRNode]
) -> tuple[QuackConfigKey, ...]:
    """Ask QuACK which GemmConfigs this call may pin, default first.

    Legality is decided by the EpiOps, so the runtime's EpiMod is built here
    from metadata with a stub epilogue and pruned against the GEMM shape hints;
    ``FlexGemmEpilogueCaller.output_node`` guards the shape-dependent rules.
    """
    from torch._inductor.kernel.flex_gemm.runtime import (
        flex_gemm_preferred_config,
        flex_gemm_problem,
        selection_callback,
    )
    from torch._inductor.virtualized import V
    from torch._vendor.quack.gemm_runtime.autotune import legal_mod_configs

    output_contraction = template_config.output_contraction
    epimod = template_config.epimod(
        selection_callback,
        [node.get_dtype() for node in input_nodes],
        lambda name: selection_callback,
    )
    sizevars = V.graph.sizevars
    mat1 = input_nodes[template_config.gemm_op.mat1_index]
    mat2 = input_nodes[template_config.gemm_op.mat2_index]
    device = mat1.get_device_or_error()
    problem = flex_gemm_problem(
        device,
        sizevars.optimization_hint(mat1.get_size()[-2]),
        sizevars.optimization_hint(mat2.get_size()[-1]),
        None if output_contraction is None else output_contraction.concat_layout,
        blockscaled=template_config.blockscaled is not None,
        varlen_m=template_config.cu_seqlens_index is not None,
    )
    legal = legal_mod_configs(
        epimod, device, problem, preferred_config=flex_gemm_preferred_config(problem)
    )
    return tuple(
        tuple(sorted(dataclasses.asdict(quack_config).items()))
        for quack_config in legal
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
    """Lower FlexGEMM through the generated QUACK CuTeDSL template.

    The current pipeline is:

    ::

        FlexGEMM HOP body (FX GraphModule)
                         |
                         v
          find GEMM operands and captured epilogue args
                         |
                         v
             analyze_flex_gemm_epilogue()
                         |
                         +--> output plan + buffer ABI
                         |        `--> derive layout + allocate aux
                         |
                         +--> grouped/reduction geometry
                         |        `--> flex_gemm_quack_configs()
                         |
                         `--> materialize_flex_gemm_epilogue()
                                  `--> CuTeDSL epilogue + callbacks
                         |
                         v
             combine into template choices -> autotune
                         |
                         v
              restore captured output order
    """
    if gemm_op not in FLEX_GEMM_OP_SPECS:
        raise NotImplementedError(
            f"FlexGEMM QUACK backend currently supports only aten.{_SUPPORTED_FLEX_GEMM_OP_NAMES}"
        )
    tuned = kernel_options.get("tuned", False)
    fast_math = kernel_options.get("fast_math", False)
    explicit_config = kernel_options.get("config")
    unsupported_options = OrderedSet(kernel_options) - OrderedSet(
        ["backend", "tuned", "fast_math", "config"]
    )
    if unsupported_options:
        raise NotImplementedError(
            f"unsupported FlexGEMM kernel options: {sorted(unsupported_options)}"
        )
    if not isinstance(fast_math, bool):
        raise NotImplementedError("FlexGEMM fast_math kernel option must be bool")
    if "config" in kernel_options and not isinstance(explicit_config, dict):
        raise NotImplementedError("FlexGEMM config kernel option must be a dict")
    config_constraints = {} if explicit_config is None else dict(explicit_config)
    if config_constraints:
        from torch._vendor.quack.gemm_config import GemmConfig

        config_fields = OrderedSet(
            field.name for field in dataclasses.fields(GemmConfig)
        )
        unknown_fields = OrderedSet(config_constraints) - config_fields
        if unknown_fields:
            raise NotImplementedError(
                f"unknown GemmConfig constraint {sorted(unknown_fields)}; "
                f"choose one of {', '.join(config_fields)}"
            )
    explicit_swap_ab = config_constraints.get("swap_ab") is True

    from torch._inductor.kernel.flex_gemm.fx_cutedsl_codegen import (
        analyze_flex_gemm_epilogue,
        flex_gemm_indexed_output_plan,
        flex_gemm_output_values,
        gemm_node as flex_gemm_node,
        materialize_flex_gemm_epilogue,
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
    scaled_mm = gemm_op is torch.ops.aten._scaled_mm_v2.default
    grouped_mm = gemm_op is torch.ops.aten._grouped_mm.default
    indexed_output_error = (
        f"FlexGEMM QUACK {op_spec.name} does not yet support indexed outputs"
    )
    try:
        indexed_store = flex_gemm_indexed_output_plan(
            *flex_gemm_output_values(subgraph.graph_module)
        )
    except NotImplementedError as exc:
        if scaled_mm or grouped_mm:
            raise QuackFallbackUnsupported(indexed_output_error) from exc
        raise
    if (scaled_mm or grouped_mm) and indexed_store is not None:
        raise QuackFallbackUnsupported(indexed_output_error)
    placeholders = [
        node for node in subgraph.graph_module.graph.nodes if node.op == "placeholder"
    ]
    placeholder_args = dict(zip(placeholders, args, strict=True))
    blockscaled = None
    mainloop_scale_nodes: tuple[torch.fx.Node, ...] = ()
    if scaled_mm:
        blockscaled = quack_blockscaled_contract(gemm_fx_node)
        gemm_fx_node.args = blockscaled.gemm_inputs
        mainloop_scale_nodes = blockscaled.tensorwise_scales
        alpha, beta = 1.0, 0.0
    elif grouped_mm:
        gemm_fx_node.args = quack_grouped_mm_contract(gemm_fx_node)
        alpha, beta = 1.0, 0.0
    else:
        unsupported_gemm_kwargs = OrderedSet(gemm_kwargs) - OrderedSet(
            ["alpha", "beta"]
        )
        if unsupported_gemm_kwargs:
            raise NotImplementedError(
                f"unsupported FlexGEMM GEMM kwargs: {sorted(unsupported_gemm_kwargs)}"
            )
        alpha = gemm_fx_node.kwargs.get("alpha", gemm_kwargs.get("alpha", 1.0))
        beta = gemm_fx_node.kwargs.get("beta", gemm_kwargs.get("beta", 1.0))
        if not isinstance(alpha, (int, float)) or not isinstance(beta, (int, float)):
            raise NotImplementedError("FlexGEMM alpha/beta must be static scalars")

    gemm_args: list[TensorBox] = []
    for arg in gemm_fx_node.args:
        gemm_arg = placeholder_args[arg] if isinstance(arg, torch.fx.Node) else arg
        if not isinstance(gemm_arg, TensorBox):
            raise NotImplementedError("FlexGEMM lowering expects tensor GEMM operands")
        gemm_args.append(gemm_arg)
    check_quack_fp32_operand(gemm_args[mat1_index])
    if grouped_mm:
        # QuACK's varlen path reads [0, *offs]; pad in-graph so Inductor owns
        # the buffer instead of the runtime allocating during CUDA graph capture.
        gemm_args[2] = constant_pad_nd(gemm_args[2], [1, 0])
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
    capture_start = len(mainloop_scale_nodes)
    epilogue_args[capture_start:] = normalize_flex_gemm_1d_captures(
        subgraph.graph_module,
        epilogue_arg_placeholders[capture_start:],
        epilogue_args[capture_start:],
        gemm_fx_node.meta["val"].shape,
        indices=None if indexed_store is None else indexed_store.indices,
    )
    gemm_input_names = tuple(
        arg.name if isinstance(arg, torch.fx.Node) else f"gemm_arg{index}"
        for index, arg in enumerate(gemm_fx_node.args)
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
            alpha=float(alpha),
            beta=float(beta),
            tuned=tuned,
            fast_math=fast_math,
            explicit_config=explicit_config,
        ),
        lowering_name=subgraph.name,
    )
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
    if grouped_mm and outputs.local_reduce is not None:
        raise QuackFallbackUnsupported(
            "FlexGEMM QUACK grouped_mm (varlen) does not yet support grouped reductions"
        )
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

    output_contraction = outputs.output_contraction
    if output_contraction is not None and epilogue_args[len(mainloop_scale_nodes) :]:
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
    logical_output_size = ir.convert_shape_to_inductor(output_meta.shape)
    if output_contraction is not None and outputs.aux_outputs:
        aux_metas = (gemm_fx_node.meta["val"],)
    else:
        aux_metas = validate_flex_gemm_aux_outputs(
            gemm_op, outputs.aux_outputs, logical_output_size
        )
    indexed_metas = () if indexed_output is None else (indexed_output.node.meta["val"],)
    if not has_flex_gemm_quack():
        raise NotImplementedError("FlexGEMM QUACK backend requires CuTeDSL")
    # A terminal ``view(dtype)`` reinterprets bits: the kernel stores the source
    # dtype and the result is re-viewed below.
    output_storage = outputs.output_storage
    output_storage_dtype = (
        output_meta.dtype
        if output_storage is None
        else output_storage.meta["val"].dtype
    )
    packed_uint8_main = (
        output_contraction is not None and output_storage_dtype is torch.uint8
    )
    if (
        not output_storage_dtype.is_floating_point
        and output_storage_dtype is not torch.bool
        and not packed_uint8_main
    ):
        raise NotImplementedError(
            "FlexGEMM generic main outputs support only floating-point and bool dtypes"
        )
    local_reduce_metas = flex_gemm_local_reduce_metas(outputs.local_reduce)
    output_stride = ir.convert_shape_to_inductor(output_meta.stride())
    if output_contraction is not None:
        # Grouped main outputs use TMA stores, whose outer stride must preserve
        # 16-byte alignment even when the contracted N extent is not aligned.
        output_alignment = max(16 // output_storage_dtype.itemsize, 1)
        output_stride[-2] = (
            ceildiv(logical_output_size[-1], output_alignment) * output_alignment
        )
    layout = ir.FixedLayout(
        gemm_args[mat1_index].get_device_or_error(),
        output_storage_dtype,
        logical_output_size,
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
            logical_output_size,
        ),
    )
    if grouped_mm and "tile" in epilogue_arg_kinds:
        raise QuackFallbackUnsupported(
            "FlexGEMM QUACK grouped_mm (varlen) does not yet support captured "
            "tensors of the full [total_m, N] output shape"
        )
    if gemm_args[mat1_index].get_device_or_error().type != "cuda":
        raise NotImplementedError("FlexGEMM QUACK backend requires CUDA tensors")
    epimod_source = materialize_flex_gemm_epilogue(
        subgraph.graph_module,
        epilogue_analysis,
        epilogue_arg_placeholders,
        float(alpha),
        float(beta),
        epilogue_arg_kinds,
        fast_math=fast_math,
        swap_ab=explicit_swap_ab,
        mainloop_scale_count=mainloop_scale_count,
    )
    log_flex_gemm_artifact(
        "lowering_plan",
        lambda: format_flex_gemm_lowering_plan(
            logical_output_size,
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
    template_local_reduce = FlexGemmEpilogueLocalReduceConfig.from_plan(
        outputs.local_reduce, local_reduce_out_index, epimod_source
    )
    if epimod_source.local_reduce_fragment_reduced:
        # Fragment partials are lowered for the unswapped accumulator geometry.
        config_constraints["swap_ab"] = False
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
        quack_config=None,
        cu_seqlens_index=gemm_input_indices[2] if grouped_mm else None,
        epilogue_arg_indices=epilogue_arg_indices,
        epilogue_arg_kinds=epilogue_arg_kinds,
        aux_out_indices=aux_out_indices,
        indexed_output=template_indexed_output,
        local_reduce=template_local_reduce,
        output_contraction=output_contraction,
    )
    template_kwargs = dict(
        input_nodes=input_nodes,
        layout=layout,
        mutated_inputs=mutated_input_nodes or None,
    )
    legal_configs = flex_gemm_quack_configs(template_config, input_nodes)
    if config_constraints:
        legal_configs = tuple(
            config
            for config in legal_configs
            if all(
                type(field) is type(value) and field == value
                for name, value in config_constraints.items()
                for field in (dict(config)[name],)
            )
        )
        if not legal_configs:
            raise NotImplementedError(
                "no supported GemmConfig matches "
                f"config_constraints={config_constraints!r} for this call"
            )
    if tuned:
        quack_configs = flex_gemm_search_space(legal_configs, varlen=grouped_mm)
    else:
        from torch._inductor.virtualized import V

        sizevars = V.graph.sizevars
        mat1, mat2 = (
            gemm_input_nodes[i] for i in (op_spec.mat1_index, op_spec.mat2_index)
        )
        m_hint = sizevars.optimization_hint(mat1.get_size()[-2])
        n_hint = sizevars.optimization_hint(mat2.get_size()[-1])
        # Block-scaled calls keep QuACK's shape-aware blockscaled default.
        dense_shape = None if blockscaled is not None else (m_hint, n_hint)
        default = flex_gemm_default_config(
            legal_configs, varlen=grouped_mm, dense_shape=dense_shape
        )
        quack_configs = (default,)
    log_flex_gemm_artifact(
        "config_candidates",
        lambda: format_flex_gemm_config_candidates(quack_configs, tuned=tuned),
        lowering_name=subgraph.name,
    )
    choices: list[Any] = []
    for quack_config in quack_configs:
        error = flex_gemm_epilogue_template.maybe_append_choice(
            choices,
            config=dataclasses.replace(template_config, quack_config=quack_config),
            **template_kwargs,
        )
        if error is not None:
            raise error
    input_gen_fns = {
        index: flex_gemm_autotune_view_input
        for index, input_node in enumerate(input_nodes)
        if isinstance(input_node, ir.ReinterpretView)
    }
    if grouped_mm:
        input_gen_fns[gemm_input_indices[2]] = functools.partial(
            flex_gemm_cu_seqlens_benchmark_input, total_m=logical_output_size[0]
        )
    result, _ = autotune_select_algorithm(
        "flex_gemm_epilogue",
        choices,
        input_nodes,
        layout,
        input_gen_fns=input_gen_fns or None,
        **({"return_multi_template": False} if mutated_input_nodes else {}),
    )
    if tuned and len(choices) == 1:
        # autotune_select_algorithm skips a lone tuned candidate; compile it now so
        # tuned=True never defers compilation to the first call. Untuned calls keep
        # lazy first-call compilation and allocate no example tensors.
        choices[0].precompile(use_workers=False)
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
    if output_storage_dtype is not output_meta.dtype:
        result = TensorBox(ir.DtypeView.create(result, output_meta.dtype))
    return (result, *ordered_aux_outs)


@register_lowering(flex_gemm_hop, type_promotion_kind=None)
def flex_gemm_lowering(gemm_op, subgraph, args, gemm_kwargs, kernel_options):
    """Dispatch FlexGEMM to ordinary Inductor lowering or the QUACK template."""
    backend = kernel_options.get("backend", "TRITON")
    if backend in ("NVGEMM", "QUACK") and gemm_op in FLEX_GEMM_OP_SPECS:
        mat1 = args[FLEX_GEMM_OP_SPECS[gemm_op].mat1_index]
        if isinstance(mat1, TensorBox) and is_bf16x9_matmul(
            mat1.get_device_or_error().type, mat1.get_dtype()
        ):
            # See Note [BF16x9 precision] in torch/_inductor/utils.py.
            warning_once(
                log,
                f"FlexGEMM {backend} does not support bfx9 precision; using ATen/cuBLAS instead.",
            )
            return process_subgraph_nodes(subgraph.graph_module, list(args))
    if backend == "NVGEMM":
        unsupported_options = OrderedSet(kernel_options) - OrderedSet(
            ("backend", "tuned")
        )
        if unsupported_options:
            raise NotImplementedError(
                f"Unsupported NVGEMM FlexGEMM options: {unsupported_options}"
            )
        tuned = kernel_options.get("tuned", False)
        if not isinstance(tuned, bool):
            raise NotImplementedError("NVGEMM FlexGEMM tuned must be a bool")
        decompose_nvgemm_additive_gemm(subgraph.graph_module)
        nvgemm_config: dict[str, Any] = {
            "max_autotune": True,
            "max_autotune_gemm_backends": "NVGEMM",
        }
        if tuned:
            nvgemm_config.update(
                nvgemm_max_profiling_configs=None,
                nvgemm_supplement_configs=True,
                nvgemm_swap_ab=True,
            )
        with config.patch(nvgemm_config):
            return process_subgraph_nodes(subgraph.graph_module, list(args))
    body_gemm_op = flex_gemm_body_gemm_op(gemm_op, gemm_kwargs)
    if backend == "QUACK":
        # The QUACK path rewrites the body in place (1-D capture folding); the
        # fallback below must re-lower the untouched original.
        quack_subgraph = dataclasses.replace(
            subgraph, graph_module=copy.deepcopy(subgraph.graph_module)
        )
        try:
            return lower_quack_flex_gemm(
                body_gemm_op, quack_subgraph, args, gemm_kwargs, kernel_options
            )
        except QuackFallbackUnsupported as error:
            if "config" in kernel_options:
                raise
            fallback_reason = str(error)
            log_flex_gemm_artifact(
                "fallback",
                lambda: fallback_reason,
                lowering_name=subgraph.name,
            )
            return process_subgraph_nodes(subgraph.graph_module, list(args))
    return process_subgraph_nodes(subgraph.graph_module, list(args))
