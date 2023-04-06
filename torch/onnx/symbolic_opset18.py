"""This file exports ONNX ops for opset 18.

Note [ONNX Operators that are added/updated in opset 18]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-18-of-the-default-onnx-operator-set
New operators:
    CenterCropPad
    Col2Im
    Mish
    OptionalGetElement
    OptionalHasElement
    Pad
    Resize
    ScatterElements
    ScatterND
"""

import functools
from typing import Sequence

from torch import _C
from torch.onnx import symbolic_helper
from torch.onnx._internal import _beartype, registration

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

__all__ = ["col2im"]

_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=18)


@_onnx_symbolic("aten::col2im")
@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is")
@_beartype.beartype
def col2im(
    g,
    input: _C.Value,
    output_size: _C.Value,
    kernel_size: _C.Value,
    dilation: Sequence[int],
    padding: Sequence[int],
    stride: Sequence[int],
):
    # convert [i0, i1, ..., in] into [i0, i0, i1, i1, ..., in, in]
    adjusted_padding = []
    for pad in padding:
        for _ in range(2):
            adjusted_padding.append(pad)

    num_dimensional_axis = symbolic_helper._get_tensor_sizes(output_size)[0]
    if not adjusted_padding:
        adjusted_padding = [0, 0] * num_dimensional_axis

    if not dilation:
        dilation = [1] * num_dimensional_axis

    if not stride:
        stride = [1] * num_dimensional_axis

    return g.op(
        "Col2Im",
        input,
        output_size,
        kernel_size,
        dilations_i=dilation,
        pads_i=adjusted_padding,
        strides_i=stride,
    )
