"""Implementation for higher-order operators."""

from __future__ import annotations

import typing
from typing import Sequence

import torch
from torch.onnx._internal._lazy_import import onnxscript_ir as ir
from torch.onnx._internal.exporter import _building, _core
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


def call_op(
    op_type: str,
    *args: ir.Value,
    _num_outputs: int = 1,
    **kwargs: int | float | str | bool | ir.Graph | ir.TensorProtocol,
) -> Sequence[ir.Value]:
    """Call an operator with the given arguments and keyword arguments.

    Arguments are always inputs, while keyword arguments are attributes.
    """
    from onnxscript.ir import convenience as ir_convenience

    assert _core.current_tracer is not None
    tracer = typing.cast(_building.OpRecorder, _core.current_tracer)

    inputs = list(args)

    # If final inputs are None, strip them from the node inputs
    for input in reversed(inputs):
        if input is not None:
            break
        inputs.pop()

    # Construct and filter out None attributes
    attributes = [
        attr
        for attr in ir_convenience.convert_attributes(kwargs)
        if attr.value is not None
    ]
    tracer.nodes.append(
        node := ir.Node(
            "",
            op_type,
            inputs=inputs,
            attributes=attributes,
            num_outputs=_num_outputs,
            version=tracer.opset.version,
        )
    )
    return node.outputs


@onnx_impl(torch.ops.higher_order.cond)
def higher_order_cond(
    cond: ir.Value,
    true_func: ir.Function,
    false_func: ir.Function,
    inputs: Sequence[ir.Value],
):
    then_node = ir.Node(
        true_func.domain, true_func.name, inputs, num_outputs=len(true_func.outputs)
    )
    else_node = ir.Node(
        false_func.domain, false_func.name, inputs, num_outputs=len(false_func.outputs)
    )

    return call_op(
        "If",
        cond,
        _num_outputs=len(true_func.outputs),
        then_branch=ir.Graph(
            (), then_node.outputs, nodes=[then_node], name=true_func.name
        ),
        else_branch=ir.Graph(
            (), else_node.outputs, nodes=[else_node], name=false_func.name
        ),
    )
