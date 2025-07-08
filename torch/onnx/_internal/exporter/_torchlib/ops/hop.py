"""Implementation for higher-order operators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.onnx._internal._lazy_import import onnxscript_ir as ir
from torch.onnx._internal.exporter import _core
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


if TYPE_CHECKING:
    from collections.abc import Sequence


def call_op(
    op_type: str,
    *args: ir.Value,
    _num_outputs: int = 1,
    _domain: str = "",
    **kwargs: int | float | str | bool | ir.Graph | ir.TensorProtocol | Sequence[int],
) -> Sequence[ir.Value]:
    """Call an operator with the given arguments and keyword arguments.

    Arguments are always inputs, while keyword arguments are attributes.
    """
    # This is a wrapper around the IR node creation that hooks into the _builder.OpRecorder
    # tracer so that all nodes created are recorded the same way as if we were to use
    # onnxscript ops directly.
    from onnxscript.ir import convenience as ir_convenience

    assert _core.current_tracer is not None
    tracer = _core.current_tracer

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
        if attr.value is not None  # type: ignore[union-attr]
    ]
    tracer.nodes.append(
        node := ir.Node(
            _domain,
            op_type,
            inputs=inputs,
            attributes=attributes,
            num_outputs=_num_outputs,
            version=tracer.opset.version,
        )
    )
    return node.outputs


@onnx_impl(torch.ops.higher_order.cond, no_compile=True)
def higher_order_cond(
    cond: ir.Value,
    true_func: ir.Function,
    false_func: ir.Function,
    inputs: Sequence[ir.Value],
) -> Sequence[ir.Value]:
    then_node = ir.Node(
        true_func.domain, true_func.name, inputs, num_outputs=len(true_func.outputs)
    )
    else_node = ir.Node(
        false_func.domain, false_func.name, inputs, num_outputs=len(false_func.outputs)
    )

    # ONNX Runtime complains about duplicate output names if we don't rename them.
    # But the doesn't seem to be an actual violation of SSA form without renaming.
    for func_out, out in zip(true_func.outputs, then_node.outputs):
        out.name = f"{func_out.name}_{true_func.name}"
    for func_out, out in zip(false_func.outputs, else_node.outputs):
        out.name = f"{func_out.name}_{false_func.name}"

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


@onnx_impl(torch.ops.higher_order.scan, no_compile=True)
def higher_order_scan(
    body_func: ir.Function,
    scan_inits: Sequence[ir.Value],
    scan_inputs: Sequence[ir.Value],
    additional_inputs: Sequence[ir.Value] | None,
    reverse: bool = False,
) -> Sequence[ir.Value]:
    """https://github.com/pytorch/pytorch/blob/66ac724b56e6c37a534f3e066423ef2f41d7477f/torch/_higher_order_ops/scan.py#L109"""
    subgraph_inputs = [
        *[
            ir.Value(
                name=f"{inp.name}_{body_func.name}__subgraph_in",
                shape=inp.shape,
                type=ir.TensorType(inp.dtype),  # type: ignore[arg-type]
            )
            for inp in scan_inits
        ],
        *[
            ir.Value(
                name=f"{inp.name}_{body_func.name}__subgraph_in",
                # The iterated element passed to the body subgraph does not have a sequence axis.
                # It will have a rank one less than the rank of the corresponding scan_input.
                shape=ir.Shape(inp.shape[1:]),  # type: ignore[index]
                type=ir.TensorType(inp.dtype),  # type: ignore[arg-type]
            )
            for inp in scan_inputs
        ],
    ]
    # The one and only node in the Scan subgraph that calls the body_func
    body_node = ir.Node(
        body_func.domain,
        body_func.name,
        [
            *subgraph_inputs,
            *(additional_inputs or []),
        ],
        num_outputs=len(body_func.outputs),
    )

    # ONNX Runtime complains about duplicate output names if we don't rename them.
    # But the doesn't seem to be an actual violation of SSA form without renaming.
    for func_out, out in zip(body_func.outputs, body_node.outputs):
        out.name = f"{func_out.name}_{body_func.name}"

    n_outputs = len(body_func.outputs) - len(scan_inits)
    return call_op(
        "Scan",
        *scan_inits,
        *scan_inputs,
        _num_outputs=len(body_func.outputs),
        body=ir.Graph(
            subgraph_inputs,
            body_node.outputs,
            nodes=[body_node],
            name=body_func.name,
        ),
        num_scan_inputs=len(scan_inputs),
        scan_input_directions=[(1 if reverse else 0) for _ in scan_inputs],
        scan_output_directions=[(1 if reverse else 0) for _ in range(n_outputs)],
    )
