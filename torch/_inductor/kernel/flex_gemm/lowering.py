# mypy: allow-untyped-defs
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
        if epilogue_arg_size == output_size:
            epilogue_arg_kinds.append("tile")
        elif epilogue_arg_size == [1, n]:
            epilogue_arg_kinds.append("row")
        elif epilogue_arg_size == [m, 1]:
            epilogue_arg_kinds.append("col")
        else:
            raise NotImplementedError(
                "FlexGEMM captured tensor epilogue args currently must match "
                "the GEMM output shape or broadcast as [1, N] / [M, 1]"
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
    if len(aux_outputs) > 1:
        raise NotImplementedError(
            "FlexGEMM QUACK backend currently supports at most one aux output"
        )
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
        if aux_size != output_size:
            raise NotImplementedError(
                "FlexGEMM generic aux tuple epilogues currently require aux "
                "output shapes to match the GEMM output shape"
            )
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


@register_lowering(flex_gemm_hop, type_promotion_kind=None)
def flex_gemm_lowering(gemm_op, subgraph, args, gemm_kwargs, kernel_options):
    """Lower FlexGEMM to the regular subgraph path or the QUACK template."""
    if kernel_options.get("backend", "TRITON") != "QUACK":
        return process_subgraph_nodes(subgraph.graph_module, list(args))
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
        gemm_node as flex_gemm_node,
        materialize_flex_gemm_epilogue,
        output_plan as flex_gemm_output_plan,
    )
    from torch._inductor.kernel.flex_gemm.template import (
        flex_gemm_epilogue_template,
        FlexGemmEpilogueConfig,
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
    outputs = flex_gemm_output_plan(subgraph.graph_module)
    output_meta = outputs.output.meta.get("val")
    if output_meta is None:
        raise NotImplementedError(
            "FlexGEMM generated epilogues require output metadata"
        )
    output_size = ir.convert_shape_to_inductor(output_meta.shape)
    aux_metas = validate_flex_gemm_aux_outputs(
        gemm_op, outputs.aux_outputs, output_size
    )
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
    aux_input_nodes = [
        ir.TemplateBuffer.realize_template_input(aux_out) for aux_out in aux_outs
    ]
    input_nodes = [*gemm_input_nodes, *epilogue_input_nodes, *aux_input_nodes]
    aux_out_index = (
        len(gemm_input_nodes) + len(epilogue_input_nodes) if aux_input_nodes else None
    )
    epilogue_arg_kinds = infer_flex_gemm_epilogue_arg_kinds(
        gemm_op, epilogue_input_nodes, output_size
    )
    epilogue_name, epilogue_source = materialize_flex_gemm_epilogue(
        subgraph.graph_module, gemm_op, epilogue_arg_placeholders
    )
    if tuned:
        from torch._inductor.heuristics.template.flex_gemm import (
            candidate_gemm_configs_for_device,
            gemm_config_key,
        )

        quack_config_keys = tuple(
            gemm_config_key(config)
            for config in candidate_gemm_configs_for_device(layout.device)
        )
    else:
        from torch._inductor.heuristics.template.flex_gemm import (
            default_gemm_config_key,
        )

        quack_config_keys = (
            default_gemm_config_key(
                layout.device,
                gemm_args[mat1_index].get_size()[-2],
                gemm_args[mat2_index].get_size()[-1],
            ),
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
            mutated_inputs=aux_input_nodes or None,
            config=FlexGemmEpilogueConfig(
                epilogue_name=epilogue_name,
                epilogue_source=epilogue_source,
                gemm_op=op_spec,
                alpha=float(alpha),
                beta=float(beta),
                out_dtype=output_meta.dtype,
                quack_config_key=quack_config_key,
                epilogue_arg_indices=epilogue_arg_indices,
                epilogue_arg_kinds=epilogue_arg_kinds,
                aux_out_index=aux_out_index,
            ),
        )
        if error is not None:
            raise error
    result, _ = autotune_select_algorithm(
        "flex_gemm_epilogue", choices, input_nodes, layout
    )
    if aux_outs:
        return (result, *aux_outs)
    return (result,)
