# mypy: allow-untyped-defs
from __future__ import annotations

from typing import Any

import torch
from torch._higher_order_ops.flex_gemm import (
    _SUPPORTED_FLEX_GEMM_OP_NAMES,
    flex_gemm_hop,
    FLEX_GEMM_OP_SPECS,
)
from torch.utils._ordered_set import OrderedSet

from ... import ir
from ...ir import TensorBox
from ...lowering import process_subgraph_nodes, register_lowering


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
        output_node as flex_gemm_output_node,
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
    alpha = gemm_fx_node.kwargs.get("alpha", gemm_kwargs.get("alpha", 1.0))
    beta = gemm_fx_node.kwargs.get("beta", gemm_kwargs.get("beta", 1.0))
    if not isinstance(alpha, (int, float)) or not isinstance(beta, (int, float)):
        raise NotImplementedError("FlexGEMM alpha/beta must be static scalars")
    output_meta = flex_gemm_output_node(subgraph.graph_module).meta.get("val")
    if output_meta is None:
        raise NotImplementedError(
            "FlexGEMM generated epilogues require output metadata"
        )
    layout = ir.FixedLayout(
        gemm_args[mat1_index].get_device_or_error(),
        output_meta.dtype,
        ir.convert_shape_to_inductor(output_meta.shape),
        ir.convert_shape_to_inductor(output_meta.stride()),
    )
    epilogue_name, epilogue_source = materialize_flex_gemm_epilogue(
        subgraph.graph_module, gemm_op
    )
    input_nodes = [ir.TemplateBuffer.realize_template_input(arg) for arg in gemm_args]
    if tuned:
        from torch._inductor.template_heuristics.flex_gemm import (
            candidate_gemm_configs_for_device,
            gemm_config_key,
        )

        quack_config_keys = tuple(
            gemm_config_key(config)
            for config in candidate_gemm_configs_for_device(layout.device)
        )
    else:
        from torch._inductor.template_heuristics.flex_gemm import (
            default_gemm_config_key,
        )

        quack_config_keys = (
            default_gemm_config_key(
                layout.device,
                gemm_args[mat1_index].get_size()[-2],
                gemm_args[mat2_index].get_size()[-1],
            ),
        )
    choices: list[Any] = []
    for quack_config_key in quack_config_keys:
        error = flex_gemm_epilogue_template.maybe_append_choice(
            choices,
            input_nodes=input_nodes,
            layout=layout,
            config=FlexGemmEpilogueConfig(
                epilogue_name=epilogue_name,
                epilogue_source=epilogue_source,
                gemm_op=op_spec,
                alpha=float(alpha),
                beta=float(beta),
                out_dtype=output_meta.dtype,
                quack_config_key=quack_config_key,
            ),
        )
        if error is not None:
            raise error
    result, _ = autotune_select_algorithm(
        "flex_gemm_epilogue", choices, input_nodes, layout
    )
    return (result,)
