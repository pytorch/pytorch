# mypy: allow-untyped-defs
"""Lower FlexGEMM HOP bodies and connect epilogue analysis to backend templates.

``flex_gemm_lowering`` is the main entry point. Non-QUACK
requests execute the captured body through ordinary Inductor lowering. Although this
is stale and will fix up later on. See ``lower_quack_flex_gemm`` for the flow.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any, TYPE_CHECKING

import torch
import torch.utils._pytree as pytree
from torch._higher_order_ops.flex_gemm import (
    _SUPPORTED_FLEX_GEMM_OP_NAMES,
    flex_gemm_hop,
    FLEX_GEMM_OP_SPECS,
)
from torch._inductor import config
from torch._logging import warning_once
from torch.utils._ordered_set import OrderedSet

from ... import ir
from ...ir import IRNode, TensorBox
from ...lowering import empty_strided, process_subgraph_nodes, register_lowering
from ...utils import _IntLike, ceildiv, is_bf16x9_matmul
from ..gemm_epilogue_utils import statically_known_shape_equal
from .constraints import aux_output_shape_error, LOCAL_REDUCE_DENSE_MM_SCOPE_ERROR
from .debug import (
    format_flex_gemm_analysis,
    format_flex_gemm_analysis_details,
    format_flex_gemm_lowering_plan,
    format_flex_gemm_problem,
    format_flex_gemm_selection,
    log_flex_gemm_artifact,
)


if TYPE_CHECKING:
    from collections.abc import Sequence


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


def has_flex_gemm_quack() -> bool:
    """Whether the vendored QuACK backend can import its CuTeDSL dependency."""
    return importlib.util.find_spec("cutlass") is not None


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
    if gemm_op is not torch.ops.aten.mm.default:
        raise NotImplementedError(
            "FlexGEMM generated epilogues with captured tensor reads currently support only aten.mm"
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
    if gemm_op is not torch.ops.aten.mm.default:
        raise NotImplementedError(
            "FlexGEMM generic aux tuple epilogues currently support only aten.mm"
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


def flex_gemm_local_reduce_metas(local_reduce) -> tuple[Any, ...]:
    """Return metadata for the optional compressed local-reduce output."""
    if local_reduce is None or local_reduce.store is None:
        return ()
    return (local_reduce.store.node.meta["val"],)


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
                         |        `--> filter QuACK configurations
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

    from torch._inductor.kernel.flex_gemm.fx_cutedsl_codegen import (
        analyze_flex_gemm_epilogue,
        expand_epimod_prepare_softmax_online,
        gemm_node as flex_gemm_node,
        materialize_flex_gemm_epilogue,
    )
    from torch._inductor.kernel.flex_gemm.template import (
        flex_gemm_epilogue_template,
        FlexGemmEpilogueConfig,
        FlexGemmEpilogueLocalReduceConfig,
    )
    from torch._inductor.select_algorithm import autotune_select_algorithm

    op_spec = FLEX_GEMM_OP_SPECS[gemm_op]
    mat1_index = op_spec.mat1_index
    gemm_fx_node = flex_gemm_node(subgraph.graph_module, gemm_op)
    placeholders = [
        node for node in subgraph.graph_module.graph.nodes if node.op == "placeholder"
    ]
    placeholder_args = dict(zip(placeholders, args, strict=True))
    unsupported_gemm_kwargs = OrderedSet(gemm_kwargs) - OrderedSet(["alpha", "beta"])
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
    epilogue_arg_placeholders = flex_gemm_epilogue_arg_placeholders(
        subgraph.graph_module, gemm_fx_node
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
        and gemm_op is not torch.ops.aten.mm.default
    ):
        raise NotImplementedError(LOCAL_REDUCE_DENSE_MM_SCOPE_ERROR)
    outputs = epilogue_analysis.outputs
    output_contraction = outputs.output_contraction
    if output_contraction is not None and epilogue_args:
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
    aux_metas = validate_flex_gemm_aux_outputs(
        gemm_op, outputs.aux_outputs, logical_output_size
    )
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
    aux_outs = allocate_flex_gemm_aux_outs(aux_metas, gemm_args[mat1_index])
    local_reduce_outs = allocate_flex_gemm_aux_outs(
        local_reduce_metas, gemm_args[mat1_index]
    )
    aux_input_nodes = [
        ir.TemplateBuffer.realize_template_input(aux_out) for aux_out in aux_outs
    ]
    local_reduce_input_nodes = [
        ir.TemplateBuffer.realize_template_input(local_reduce_out)
        for local_reduce_out in local_reduce_outs
    ]
    input_nodes = [
        *gemm_input_nodes,
        *epilogue_input_nodes,
        *aux_input_nodes,
        *local_reduce_input_nodes,
    ]
    mutated_input_nodes = aux_input_nodes + local_reduce_input_nodes
    aux_out_start = len(gemm_input_nodes) + len(epilogue_input_nodes)
    aux_out_indices = tuple(range(aux_out_start, aux_out_start + len(aux_input_nodes)))
    local_reduce_out_index = (
        aux_out_start + len(aux_input_nodes) if local_reduce_input_nodes else None
    )
    epilogue_arg_kinds = infer_flex_gemm_epilogue_arg_kinds(
        gemm_op, epilogue_input_nodes, logical_output_size
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
    template_local_reduce = FlexGemmEpilogueLocalReduceConfig.from_plan(
        outputs.local_reduce, local_reduce_out_index, epimod_source
    )
    epilogue_arg_indices = tuple(
        range(
            len(gemm_input_nodes),
            len(gemm_input_nodes) + len(epilogue_input_nodes),
        )
    )
    choices: list[Any] = []
    error = flex_gemm_epilogue_template.maybe_append_choice(
        choices,
        input_nodes=input_nodes,
        layout=layout,
        mutated_inputs=mutated_input_nodes or None,
        config=FlexGemmEpilogueConfig(
            epilogue_name=epimod_source.name,
            epilogue_source=epimod_source.source,
            gemm_op=op_spec,
            alpha=float(alpha),
            beta=float(beta),
            quack_config_constraints=(
                tuple(sorted(explicit_config.items()))
                if explicit_config is not None
                else ()
            ),
            epilogue_arg_indices=epilogue_arg_indices,
            epilogue_arg_kinds=epilogue_arg_kinds,
            aux_out_indices=aux_out_indices,
            local_reduce=template_local_reduce,
            output_contraction=output_contraction,
            tuned=tuned,
        ),
    )
    if error is not None:
        raise error
    input_gen_fns = {
        index: flex_gemm_autotune_view_input
        for index, input_node in enumerate(input_nodes)
        if isinstance(input_node, ir.ReinterpretView)
    }
    result, selected_choice = autotune_select_algorithm(
        "flex_gemm_epilogue",
        choices,
        input_nodes,
        layout,
        input_gen_fns=input_gen_fns or None,
        **({"return_multi_template": False} if mutated_input_nodes else {}),
    )
    log_flex_gemm_artifact(
        "selection",
        lambda: format_flex_gemm_selection(selected_choice, tuned=tuned),
        lowering_name=subgraph.name,
    )
    structural_outs = {}
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
    if backend == "QUACK":
        return lower_quack_flex_gemm(
            gemm_op, subgraph, args, gemm_kwargs, kernel_options
        )
    return process_subgraph_nodes(subgraph.graph_module, list(args))
