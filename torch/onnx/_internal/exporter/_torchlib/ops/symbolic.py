"""Implementation for higher-order operators."""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript.ir import convenience as ir_convenience

import torch
from torch.onnx._internal._lazy_import import onnx_ir as ir
from torch.onnx._internal.exporter import _core
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl
from torch.onnx.ops import _symbolic_impl


if TYPE_CHECKING:
    from collections.abc import Sequence


def _call_symbolic_op(
    op_type: str,
    domain: str,
    args: Sequence[ir.Value | None],
    kwargs: dict[str, int | float | str | bool | list[int] | list[float] | list[str]],
    dtypes: Sequence[int],
    version: int | None,
    metadata_props: dict[str, str] | None,
) -> Sequence[ir.Value]:
    """Call an operator with the given arguments and keyword arguments.

    Arguments are always inputs, while keyword arguments are attributes.
    """
    # This is a wrapper around the IR node creation that hooks into the _builder.OpRecorder
    # tracer so that all nodes created are recorded the same way as if we were to use
    # onnxscript ops directly.

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
        for attr in ir_convenience.convert_attributes(kwargs)  # type: ignore[arg-type]
        if attr.value is not None  # type: ignore[union-attr]
    ]
    tracer.nodes.append(
        node := ir.Node(
            domain,
            op_type,
            inputs=inputs,
            attributes=attributes,
            num_outputs=len(dtypes),
            version=version,
            metadata_props=metadata_props,
        )
    )
    # Set the dtypes for the outputs. We set them here because the graph builder
    # Uses PyTorch types which are sometimes inaccurate when they are ONNX only
    # types like float4e2m1.
    for value, dtype in zip(node.outputs, dtypes):
        value.dtype = ir.DataType(dtype)
        # The shape is set by the graph builder. We don't need to set it here.
    return node.outputs


@onnx_impl(torch.ops.onnx_symbolic._symbolic.default, no_compile=True)
def onnx_symbolic_symbolic(
    inputs: Sequence[ir.Value | None],
    op_type: str,
    onnx_dtype: int,
    *,
    shape: Sequence[int | ir.Value],
    attr_keys: Sequence[str],
    attr_types: Sequence[str],
    attr_pos: Sequence[tuple[int, int]],
    attr_ints: Sequence[int],
    attr_floats: Sequence[float],
    attr_strs: Sequence[str],
    metadata_props_keys: Sequence[str] = (),
    metadata_props_values: Sequence[str] = (),
    domain: str = "",
    version: int | None = None,
) -> ir.Value:
    del shape  # Unused. The shapes are set by the graph builder
    encoded = _symbolic_impl.EncodedAttrs(
        attr_keys=list(attr_keys),
        attr_types=list(attr_types),
        attr_pos=list(attr_pos),
        attr_ints=list(attr_ints),
        attr_floats=list(attr_floats),
        attr_strs=list(attr_strs),
    )
    attrs = encoded.to_dict()
    return _call_symbolic_op(
        op_type,
        domain,
        inputs,
        attrs,
        dtypes=[onnx_dtype],
        version=version,
        metadata_props=dict(zip(metadata_props_keys, metadata_props_values)),
    )[0]


@onnx_impl(torch.ops.onnx_symbolic._symbolic_multi_out.default, no_compile=True)
def onnx_symbolic_symbolic_multi_out(
    inputs: Sequence[ir.Value | None],
    op_type: str,
    onnx_dtypes: Sequence[int],
    *,
    shapes: Sequence[Sequence[int | ir.Value]],
    attr_keys: Sequence[str],
    attr_types: Sequence[str],
    attr_pos: Sequence[tuple[int, int]],
    attr_ints: Sequence[int],
    attr_floats: Sequence[float],
    attr_strs: Sequence[str],
    metadata_props_keys: Sequence[str] = (),
    metadata_props_values: Sequence[str] = (),
    domain: str = "",
    version: int | None = None,
) -> Sequence[ir.Value]:
    del shapes  # Unused. The shapes are set by the graph builder
    encoded = _symbolic_impl.EncodedAttrs(
        attr_keys=list(attr_keys),
        attr_types=list(attr_types),
        attr_pos=list(attr_pos),
        attr_ints=list(attr_ints),
        attr_floats=list(attr_floats),
        attr_strs=list(attr_strs),
    )
    attrs = encoded.to_dict()
    return _call_symbolic_op(
        op_type,
        domain,
        inputs,
        attrs,
        dtypes=onnx_dtypes,
        version=version,
        metadata_props=dict(zip(metadata_props_keys, metadata_props_values)),
    )
