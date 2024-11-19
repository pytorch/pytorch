"""Implementation for higher-order operators."""

from __future__ import annotations

from typing import Sequence

import torch
from torch.onnx._internal._lazy_import import onnxscript, onnxscript_ir as ir
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


@onnx_impl(torch.ops.higher_order.cond)
def higher_order_cond(
    cond: ir.Value,
    true_func: ir.Function,
    false_func: ir.Function,
    inputs: Sequence[ir.Value],
) -> Sequence[ir.Value]:
    op = onnxscript.opset18
    then_node = ir.Node(
        true_func.domain, true_func.name, inputs, num_outputs=len(true_func.outputs)
    )
    else_node = ir.Node(
        false_func.domain, false_func.name, inputs, num_outputs=len(false_func.outputs)
    )

    # FIXME(justinchuby): Set the output number of the If node and make it traceable by onnxscript
    return (
            op.If(
            cond,
            then_branch=ir.Graph(
                (), then_node.outputs, nodes=[then_node], name=true_func.name
            ),
            else_branch=ir.Graph(
                (), else_node.outputs, nodes=[else_node], name=false_func.name
            ),
        ),
    )
