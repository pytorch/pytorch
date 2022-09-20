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

from typing import List, Optional, Sequence, Tuple, Union

from torch import _C
from torch.onnx import symbolic_helper
from torch.onnx._internal import _beartype


# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py


@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is")
@_beartype.beartype
def col2im(
    g,
    input: _C.Value,
    output_size: _C.Value,
    kernel_size: _C.Value,
    dilation: List[int],
    padding: List[int],
    stride: List[int],
):
    # Padding for onnx::col2im has separate beginning and ending values for each dimension
    adjusted_padding = []
    for i in range(2):
        for pad in padding:
            adjusted_padding.append(pad)
    return g.op(
        "Col2Im",
        input,
        output_size,
        kernel_size,
        dilations_i=dilation,
        padding_i=adjusted_padding,
        strides_i=stride,
    )
