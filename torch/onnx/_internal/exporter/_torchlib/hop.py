"""Implementation for higher-order operators."""
from __future__ import annotations

from typing import Sequence

import torch

from onnxscript import opset18 as op
from onnxscript import ir


@register(torch.ops.higher_order.cond)
def higher_order_cond(
    cond: ir.Value,
    then_func: ir.Function,
    else_func: ir.Function,
    inputs: Sequence[ir.Value],
):
    then_node = ir.Node(
        then_func.domain, then_func.name, inputs, num_outputs=len(then_func.outputs)
    )
    else_node = ir.Node(
        else_func.domain, else_func.name, inputs, num_outputs=len(else_func.outputs)
    )

    return op.If(
        cond,
        then_branch=ir.Graph((), then_node.outputs, nodes=[then_node]),
        else_branch=ir.Graph((), else_node.outputs, nodes=[else_node]),
    )
