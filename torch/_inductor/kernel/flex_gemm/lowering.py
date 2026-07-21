# mypy: allow-untyped-defs
"""Lower FlexGEMM HOP bodies and connect epilogue analysis to backend templates.

``flex_gemm_lowering`` is the main entry point. Non-QUACK
requests execute the captured body through ordinary Inductor lowering. Although this
is stale and will fix up later on. See ``lower_quack_flex_gemm`` for the flow.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.utils._pytree as pytree
from torch._higher_order_ops.flex_gemm import (
    _SUPPORTED_FLEX_GEMM_OP_NAMES,
    flex_gemm_hop,
    FLEX_GEMM_OP_SPECS,
)
from torch.utils._ordered_set import OrderedSet

from ... import ir
from ...ir import IRNode, TensorBox
from ...lowering import empty_strided, process_subgraph_nodes, register_lowering
from .constraints import (
    flex_gemm_local_reduce_config_error,
    is_flex_gemm_partial_reduction_shape,
    LOCAL_REDUCE_AUX_OUTPUT_CONTRACT_ERROR,
    LOCAL_REDUCE_DENSE_MM_SCOPE_ERROR,
    LOCAL_REDUCE_PARTIAL_OUTPUT_CONTRACT_ERROR,
    statically_known_shape_equal,
    validate_flex_gemm_local_reduce_config,
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


def flex_gemm_config_keys_for_local_reduce(
    device,
    m: int,
    n: int,
    local_reduce_geometries: tuple[Any, ...],
    tuned: bool,
) -> tuple[tuple[Any, ...], ...]:
    """Select QuACK config keys after applying grouped-layout config constraints.

    Every grouped geometry constrains the config the same way, whether it backs
    a runtime local-reduce plan or a plan-less grouped TensorSSA fragment
    reshape in the generated epilogue: swap_ab reorients the accumulator
    fragment and non-divisible tiles split groups across fragments, so either
    would silently regroup the wrong elements.
    """
    from torch._inductor.heuristics.template.flex_gemm import (
        candidate_gemm_configs_for_device,
        default_gemm_config_key,
        gemm_config_from_key,
        gemm_config_key,
    )

    if not tuned:
        default_key = default_gemm_config_key(device, m, n)
        if all(
            validate_flex_gemm_local_reduce_config(
                gemm_config_from_key(default_key), geometry.group, geometry.axis
            )
            for geometry in local_reduce_geometries
        ):
            return (default_key,)

    candidate_configs = candidate_gemm_configs_for_device(device)
    configs = candidate_configs
    for geometry in local_reduce_geometries:
        configs = tuple(
            config
            for config in configs
            if validate_flex_gemm_local_reduce_config(
                config, geometry.group, geometry.axis
            )
        )
        if not configs:
            raise NotImplementedError(
                flex_gemm_local_reduce_config_error(
                    candidate_configs,
                    geometry.group,
                    geometry.axis,
                )
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
    unsupported_options = OrderedSet(kernel_options) - OrderedSet(["backend", "tuned"])
    if unsupported_options:
        raise NotImplementedError(
            f"unsupported FlexGEMM kernel options: {sorted(unsupported_options)}"
        )

    from torch._inductor.kernel.flex_gemm.epilogue import (
        analyze_flex_gemm_epilogue,
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
    # This is where we figure out what the fx-graph body is doing
    epilogue_analysis = analyze_flex_gemm_epilogue(subgraph.graph_module)
    if (
        epilogue_analysis.required_geometries
        and gemm_op is not torch.ops.aten.mm.default
    ):
        raise NotImplementedError(LOCAL_REDUCE_DENSE_MM_SCOPE_ERROR)
    outputs = epilogue_analysis.outputs
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
    local_reduce_metas = flex_gemm_local_reduce_metas(outputs.local_reduce)
    layout = ir.FixedLayout(
        gemm_args[mat1_index].get_device_or_error(),
        output_meta.dtype,
        output_size,
        ir.convert_shape_to_inductor(output_meta.stride()),
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
        gemm_op, epilogue_input_nodes, output_size
    )
    template_local_reduce = FlexGemmEpilogueLocalReduceConfig.from_output_plan(
        outputs.local_reduce, local_reduce_out_index
    )
    epilogue_name, epilogue_source = materialize_flex_gemm_epilogue(
        subgraph.graph_module, gemm_op, epilogue_analysis, epilogue_arg_placeholders
    )
    quack_config_keys = flex_gemm_config_keys_for_local_reduce(
        layout.device,
        gemm_args[mat1_index].get_size()[-2],
        gemm_args[mat2_index].get_size()[-1],
        epilogue_analysis.required_geometries,
        tuned,
    )
    epilogue_arg_indices = tuple(
        range(
            len(gemm_input_nodes),
            len(gemm_input_nodes) + len(epilogue_input_nodes),
        )
    )
    choices: list[Any] = []
    for quack_config_key in quack_config_keys:
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
                aux_out_indices=aux_out_indices,
                local_reduce=template_local_reduce,
            ),
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
    if kernel_options.get("backend", "TRITON") == "QUACK":
        return lower_quack_flex_gemm(
            gemm_op, subgraph, args, gemm_kwargs, kernel_options
        )
    return process_subgraph_nodes(subgraph.graph_module, list(args))
