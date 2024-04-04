"""This file exports ONNX ops for opset 19.

Note [ONNX Operators that are added/updated in opset 19]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-19-of-the-default-onnx-operator-set
New operators:
AveragePool
Cast
CastLike
Constant
DeformConv
DequantizeLinear
Equal
Identity
If
Loop
Pad
QuantizeLinear
Reshape
Resize
Scan
Shape
Size
"""

from typing import List

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

__all__: List[str] = []
