# mypy: allow-untyped-defs
"""Lower FlexGEMM HOP bodies and connect epilogue analysis to backend templates.

``flex_gemm_lowering`` is the main entry point. Non-QUACK
requests execute the captured body through ordinary Inductor lowering. Although this
is stale and will fix up later on. See ``lower_quack_flex_gemm`` for the flow.
"""

from __future__ import annotations

from functools import partial
from typing import Any, TYPE_CHECKING

import sympy

import torch
import torch.utils._pytree as pytree
from torch._higher_order_ops.flex_gemm import (
    _SUPPORTED_FLEX_GEMM_OP_NAMES,
    flex_gemm_hop,
    FLEX_GEMM_OP_SPECS,
)
from torch._inductor import config
from torch.utils._ordered_set import OrderedSet

from ... import ir
from ...ir import IRNode, TensorBox
from ...lowering import empty_strided, process_subgraph_nodes, register_lowering
from ...utils import ceildiv, has_free_symbols
from ...virtualized import V
from .constraints import (
    FLEX_GEMM_CHUNKED_CONTIGUOUS_B_ERROR,
    flex_gemm_local_reduce_config_error,
    flex_gemm_output_config_supported,
    FLEX_GEMM_OUTPUT_CONTRACTION_CAPTURE_ERROR,
    FLEX_GEMM_OUTPUT_CONTRACTION_COMPOSITION_ERROR,
    FlexGemmOutputContraction,
    is_flex_gemm_partial_reduction_shape,
    LOCAL_REDUCE_AUX_OUTPUT_CONTRACT_ERROR,
    LOCAL_REDUCE_DENSE_MM_SCOPE_ERROR,
    LOCAL_REDUCE_PARTIAL_OUTPUT_CONTRACT_ERROR,
    output_contraction_capture_supported,
    output_contraction_config_supported,
    statically_known_multiple,
    statically_known_shape_equal,
    validate_flex_gemm_local_reduce_config,
)
from .debug import (
    format_flex_gemm_analysis,
    format_flex_gemm_analysis_details,
    format_flex_gemm_config_candidates,
    format_flex_gemm_lowering_plan,
    format_flex_gemm_problem,
    format_flex_gemm_selection,
    log_flex_gemm_artifact,
)
from .output_layout import FlexGemmOutputStorageLayout, output_layout_supports_config


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


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
    output_size: list[Any],
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
            if is_flex_gemm_partial_reduction_shape(aux_size, output_size):
                raise NotImplementedError(LOCAL_REDUCE_PARTIAL_OUTPUT_CONTRACT_ERROR)
            raise NotImplementedError(LOCAL_REDUCE_AUX_OUTPUT_CONTRACT_ERROR)
        aux_metas.append(aux_meta)
    return tuple(aux_metas)


def allocate_flex_gemm_aux_outs(
    aux_metas: tuple[Any, ...],
    mat1: TensorBox,
    *,
    column_major: bool = False,
) -> tuple[TensorBox, ...]:
    """Allocate auxiliary buffers with their requested dense strides."""
    outs = []
    for aux_meta in aux_metas:
        size = ir.convert_shape_to_inductor(aux_meta.shape)
        stride = (
            [1, size[-2]]
            if column_major
            else ir.convert_shape_to_inductor(aux_meta.stride())
        )
        outs.append(
            empty_strided(
                size,
                stride,
                dtype=aux_meta.dtype,
                device=mat1.get_device_or_error(),
            )
        )
    return tuple(outs)


def flex_gemm_ordered_outputs(result, aux_outs, local_reduce_outs, local_reduce_index):
    """Return generated outputs in the user-visible tuple order."""
    match local_reduce_outs, local_reduce_index:
        case (), _:
            return (result, *aux_outs)
        case (local_reduce_out,), None:
            return (result, *aux_outs, local_reduce_out)
        case (local_reduce_out,), index:
            return (
                result,
                *aux_outs[:index],
                local_reduce_out,
                *aux_outs[index:],
            )
        case _:
            raise AssertionError("FlexGEMM expects at most one local-reduce output")


def flex_gemm_local_reduce_metas(store) -> tuple[Any, ...]:
    """Return metadata for the optional compressed local-reduce output."""
    return () if store is None else (store.node.meta["val"],)


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


def explicit_config_swaps_ab(explicit_config: dict[str, Any] | None) -> bool:
    """Return whether every explicitly constrained candidate uses swap-ab."""
    return explicit_config is not None and explicit_config.get("swap_ab") is True


def filter_gemm_configs(
    configs: Sequence[Any],
    predicate: Callable[[Any], bool],
    *,
    explicit_config: dict[str, Any] | None,
    explicit_error: str,
    inferred_error: str,
) -> tuple[Any, ...]:
    """Filter configs while preserving explicit and inferred failure diagnostics."""
    supported = tuple(config for config in configs if predicate(config))
    if supported:
        return supported
    raise NotImplementedError(
        explicit_error if explicit_config is not None else inferred_error
    )


def flex_gemm_config_keys(
    device,
    m: int,
    n: int,
    local_reduce_geometries: tuple[Any, ...],
    tuned: bool,
    output_contraction: FlexGemmOutputContraction | None = None,
    explicit_config: dict[str, Any] | None = None,
    swap_ab_alignment: int = 1,
    local_reduce_output_layout: FlexGemmOutputStorageLayout | None = None,
    local_reduce_output_geometry: Any | None = None,
    local_reduce_feeds_main: bool = False,
) -> tuple[tuple[Any, ...], ...]:
    """Select QuACK config keys after applying grouped-layout config constraints.

    Every grouped geometry constrains the config the same way, whether it backs
    a runtime local-reduce plan or a plan-less grouped TensorSSA fragment
    reshape in the generated epilogue: swap_ab reorients the accumulator
    fragment and non-divisible tiles split groups across fragments. Output
    contractions additionally require tile_n not to oversubscribe physical N and
    a single CTA along N until QuACK predicates those layouts correctly. Explicit
    config fields constrain the candidate set; untuned selection fills omitted
    fields from the normal default, while tuned selection benchmarks the matches.
    """
    if output_contraction is not None:
        output_contraction.validate_quack(torch.cuda.get_device_capability(device)[0])
        if has_free_symbols((n,)):
            raise NotImplementedError(
                "FlexGEMM output contractions require statically known physical N"
            )

    from torch._inductor.heuristics.template.flex_gemm import (
        candidate_gemm_configs_for_device,
        default_gemm_config_key,
        explicit_gemm_configs_for_device,
        gemm_config_from_key,
        gemm_config_key,
    )

    candidate_configs = (
        candidate_gemm_configs_for_device(device)
        if explicit_config is None
        else explicit_gemm_configs_for_device(explicit_config, device)
    )
    swap_ab_aligned = statically_known_multiple(n, swap_ab_alignment)
    requires_swap_ab = all(config.swap_ab for config in candidate_configs)
    if not swap_ab_aligned and requires_swap_ab and has_free_symbols((n,)):
        swap_ab_aligned = V.graph.sizevars.guard_or_false(
            sympy.Eq(n % swap_ab_alignment, 0)
        )
    candidate_configs = tuple(
        config for config in candidate_configs if not config.swap_ab or swap_ab_aligned
    )
    if not candidate_configs:
        raise NotImplementedError(
            "FlexGEMM QUACK swap_ab configs require output N "
            f"divisible by {swap_ab_alignment}"
        )
    if output_contraction is not None:
        candidate_configs = tuple(
            config for config in candidate_configs if not config.swap_ab
        )
        if not candidate_configs:
            raise NotImplementedError(
                "FlexGEMM output contractions do not support swap_ab configs"
            )
    candidate_configs = tuple(
        config
        for config in candidate_configs
        if output_layout_supports_config(
            local_reduce_output_layout, config, local_reduce_output_geometry
        )
    )
    if not candidate_configs:
        raise NotImplementedError(
            "FlexGEMM QUACK config constraints are incompatible with "
            f"output layout {local_reduce_output_layout!r}"
        )
    if local_reduce_feeds_main:
        candidate_configs = tuple(
            config for config in candidate_configs if not config.swap_ab
        )
        if not candidate_configs:
            raise NotImplementedError(
                "FlexGEMM feed-main local reductions do not support swap_ab configs"
            )
    allow_local_reduce_swap_ab = explicit_config_swaps_ab(explicit_config)
    if not tuned:
        default_key = default_gemm_config_key(device, m, n, candidate_configs)
        default_config = gemm_config_from_key(default_key)
        if flex_gemm_output_config_supported(
            default_config,
            n,
            local_reduce_geometries,
            output_contraction,
            local_reduce_output_layout,
            local_reduce_output_geometry,
            allow_local_reduce_swap_ab=allow_local_reduce_swap_ab,
        ):
            return (default_key,)

    configs = candidate_configs
    if output_contraction is not None:
        configs = filter_gemm_configs(
            configs,
            lambda config: output_contraction_config_supported(config, n),
            explicit_config=explicit_config,
            explicit_error=(
                "FlexGEMM explicit QUACK config constraints are incompatible with "
                "the output contraction"
            ),
            inferred_error=(
                "FlexGEMM output-contraction physical N is smaller than every "
                "validated QuACK tile_n"
            ),
        )
    for geometry in local_reduce_geometries:
        configs = filter_gemm_configs(
            configs,
            lambda config: validate_flex_gemm_local_reduce_config(
                config,
                geometry.group,
                geometry.axis,
                allow_swap_ab=allow_local_reduce_swap_ab,
            ),
            explicit_config=explicit_config,
            explicit_error=(
                "FlexGEMM explicit QUACK config constraints are incompatible with "
                f"local reduction group={geometry.group}, axis={geometry.axis}"
            ),
            inferred_error=flex_gemm_local_reduce_config_error(
                candidate_configs,
                geometry.group,
                geometry.axis,
            ),
        )
    if tuned:
        return tuple(gemm_config_key(config) for config in configs)
    return (gemm_config_key(configs[0]),)


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
        gemm_node as flex_gemm_node,
        materialize_flex_gemm_epilogue,
    )
    from torch._inductor.kernel.flex_gemm.template import (
        flex_gemm_epilogue_template,
        FlexGemmEpilogueConfig,
        FlexGemmEpilogueLocalReduceConfig,
        FlexGemmEpilogueOutputConfig,
    )
    from torch._inductor.select_algorithm import autotune_select_algorithm

    op_spec = FLEX_GEMM_OP_SPECS[gemm_op]
    mat1_index, mat2_index = op_spec.mat1_index, op_spec.mat2_index
    unsupported_gemm_kwargs = OrderedSet(gemm_kwargs) - OrderedSet(["alpha", "beta"])
    if unsupported_gemm_kwargs:
        raise NotImplementedError(
            f"unsupported FlexGEMM GEMM kwargs: {sorted(unsupported_gemm_kwargs)}"
        )
    gemm_fx_node = flex_gemm_node(subgraph.graph_module, gemm_op)
    placeholders = [
        node for node in subgraph.graph_module.graph.nodes if node.op == "placeholder"
    ]
    placeholder_args = dict(zip(placeholders, args, strict=True))
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
    alpha = gemm_fx_node.kwargs.get("alpha", gemm_kwargs.get("alpha", 1.0))
    beta = gemm_fx_node.kwargs.get("beta", gemm_kwargs.get("beta", 1.0))
    if not isinstance(alpha, (int, float)) or not isinstance(beta, (int, float)):
        raise NotImplementedError("FlexGEMM alpha/beta must be static scalars")
    gemm_input_names = tuple(
        arg.name if isinstance(arg, torch.fx.Node) else f"gemm_arg{index}"
        for index, arg in enumerate(gemm_fx_node.args)
    )
    problem_report = partial(
        format_flex_gemm_problem,
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
    )
    log_flex_gemm_artifact(
        "problem",
        problem_report,
        lowering_name=subgraph.name,
    )
    explicit_swap_ab = explicit_config_swaps_ab(explicit_config)
    # This is where we figure out what the fx-graph body is doing
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
    outputs = epilogue_analysis.outputs
    if (
        epilogue_analysis.required_geometries
        and gemm_op is not torch.ops.aten.mm.default
    ):
        error = (
            FLEX_GEMM_OUTPUT_CONTRACTION_COMPOSITION_ERROR
            if outputs.output_contraction is not None
            else LOCAL_REDUCE_DENSE_MM_SCOPE_ERROR
        )
        raise NotImplementedError(error)
    local_reduce_store = outputs.local_reduce_store
    local_reduce_layout = (
        None if local_reduce_store is None else local_reduce_store.output_layout
    )
    output_meta = outputs.output.meta.get("val")
    if output_meta is None:
        raise NotImplementedError(
            "FlexGEMM generated epilogues require output metadata"
        )
    output_storage = outputs.output_storage
    output_storage_dtype = (
        output_meta.dtype
        if output_storage is None
        else output_storage.meta["val"].dtype
    )
    logical_output_size = ir.convert_shape_to_inductor(output_meta.shape)
    physical_output_size = [
        *gemm_args[mat1_index].get_size()[:-1],
        gemm_args[mat2_index].get_size()[-1],
    ]
    output_contraction = outputs.output_contraction
    if (
        output_contraction is not None
        and output_contraction.chunked
        and gemm_args[mat2_index].get_stride()[-1] == 1
    ):
        raise NotImplementedError(FLEX_GEMM_CHUNKED_CONTIGUOUS_B_ERROR)
    aux_metas = validate_flex_gemm_aux_outputs(
        gemm_op, outputs.aux_outputs, physical_output_size
    )
    local_reduce_metas = flex_gemm_local_reduce_metas(local_reduce_store)
    output_stride = ir.convert_shape_to_inductor(output_meta.stride())
    if output_contraction is not None:
        # Output contractions use TMA stores with 16-byte-aligned outer strides.
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
    swap_ab_alignment = max(
        16 // dtype.itemsize
        for dtype in (
            output_meta.dtype,
            *(arg.get_dtype() for arg in gemm_args),
            *(arg.get_dtype() for arg in epilogue_args),
            *(aux_meta.dtype for aux_meta in aux_metas),
        )
    )
    gemm_input_nodes = [
        ir.TemplateBuffer.realize_template_input(arg) for arg in gemm_args
    ]
    epilogue_input_nodes = [
        ir.TemplateBuffer.realize_template_input(arg) for arg in epilogue_args
    ]
    aux_outs = allocate_flex_gemm_aux_outs(aux_metas, gemm_args[mat1_index])
    local_reduce_outs = allocate_flex_gemm_aux_outs(
        local_reduce_metas,
        gemm_args[mat1_index],
        column_major=explicit_swap_ab and local_reduce_layout is None,
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
        gemm_op, epilogue_input_nodes, physical_output_size
    )
    if output_contraction is not None and any(
        not output_contraction_capture_supported(kind, arg.get_dtype() is torch.bool)
        for arg, kind in zip(epilogue_input_nodes, epilogue_arg_kinds, strict=True)
    ):
        raise NotImplementedError(FLEX_GEMM_OUTPUT_CONTRACTION_CAPTURE_ERROR)
    log_flex_gemm_artifact(
        "lowering_plan",
        lambda: format_flex_gemm_lowering_plan(
            logical_output_size,
            physical_output_size,
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
            local_reduce_layout=local_reduce_layout,
            swap_ab_alignment=swap_ab_alignment,
        ),
        lowering_name=subgraph.name,
    )
    template_local_reduce = FlexGemmEpilogueLocalReduceConfig.from_output_plan(
        outputs.local_reduce,
        local_reduce_out_index,
        output_layout=local_reduce_layout,
        swap_ab=explicit_swap_ab,
    )
    epilogue_name, epilogue_source = materialize_flex_gemm_epilogue(
        subgraph.graph_module,
        epilogue_analysis,
        epilogue_arg_placeholders,
        fast_math=fast_math,
        swap_ab=explicit_swap_ab,
    )
    log_flex_gemm_artifact(
        "generated_epilogue",
        lambda: epilogue_source.strip(),
        lowering_name=subgraph.name,
        verbose=True,
    )
    quack_config_keys = flex_gemm_config_keys(
        layout.device,
        gemm_args[mat1_index].get_size()[-2],
        gemm_args[mat2_index].get_size()[-1],
        epilogue_analysis.required_geometries,
        tuned,
        output_contraction,
        explicit_config,
        swap_ab_alignment=swap_ab_alignment,
        local_reduce_output_layout=local_reduce_layout,
        local_reduce_output_geometry=(
            None
            if outputs.local_reduce is None
            else outputs.local_reduce.match.geometry
        ),
        local_reduce_feeds_main=(
            outputs.local_reduce is not None and outputs.local_reduce.feeds_main
        ),
    )
    log_flex_gemm_artifact(
        "config_candidates",
        lambda: format_flex_gemm_config_candidates(quack_config_keys),
        lowering_name=subgraph.name,
        verbose=True,
    )
    epilogue_arg_indices = tuple(
        range(
            len(gemm_input_nodes),
            len(gemm_input_nodes) + len(epilogue_input_nodes),
        )
    )
    choices: list[Any] = []
    choice_config_keys = {}
    for quack_config_key in quack_config_keys:
        choice_start = len(choices)
        error = flex_gemm_epilogue_template.maybe_append_choice(
            choices,
            input_nodes=input_nodes,
            layout=layout,
            mutated_inputs=mutated_input_nodes or None,
            config=FlexGemmEpilogueConfig(
                epilogue_name=epilogue_name,
                epilogue_source=epilogue_source,
                gemm_op=op_spec,
                alpha=float(alpha),
                beta=float(beta),
                quack_config_key=quack_config_key,
                epilogue_arg_indices=epilogue_arg_indices,
                epilogue_arg_kinds=epilogue_arg_kinds,
                outputs=FlexGemmEpilogueOutputConfig(
                    aux_out_indices=aux_out_indices,
                    local_reduce=template_local_reduce,
                    output_contraction=output_contraction,
                ),
            ),
        )
        if error is not None:
            raise error
        choice_config_keys.update(
            (choice.name, quack_config_key) for choice in choices[choice_start:]
        )
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
    if output_storage is not None and output_storage_dtype is not output_meta.dtype:
        result = TensorBox(ir.DtypeView.create(result, output_meta.dtype))
    selected_config_key = (
        None
        if selected_choice is None
        else choice_config_keys.get(selected_choice.name)
    )
    log_flex_gemm_artifact(
        "selection",
        lambda: format_flex_gemm_selection(
            selected_choice,
            selected_config_key,
            candidate_count=len(quack_config_keys),
            tuned=tuned,
        ),
        lowering_name=subgraph.name,
    )
    return flex_gemm_ordered_outputs(
        result,
        aux_outs,
        local_reduce_outs,
        None if local_reduce_store is None else local_reduce_store.aux_index,
    )


@register_lowering(flex_gemm_hop, type_promotion_kind=None)
def flex_gemm_lowering(gemm_op, subgraph, args, gemm_kwargs, kernel_options):
    """Dispatch FlexGEMM to ordinary Inductor lowering or the QUACK template."""
    backend = kernel_options.get("backend", "TRITON")
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
