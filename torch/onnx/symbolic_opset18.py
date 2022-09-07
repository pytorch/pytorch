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

import torch
from torch.onnx import symbolic_helper


# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is")
def col2im(g, input, output_size, kernel_size, dilation, padding, stride):
    adjusted_padding = [pad for pad in padding for _ in (0, 1)]
    return g.op("Col2Im",
        input,
        output_size,
        kernel_size,
        dilations_i=dilation,
        padding_i=adjusted_padding,
        strides_i=stride)
