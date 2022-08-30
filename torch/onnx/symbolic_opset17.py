"""This file exports ONNX ops for opset 17.

Note [ONNX Operators that are added/updated in opset 17]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-17-of-the-default-onnx-operator-set
New operators:
    BlackmanWindow
    DFT
    HammingWindow
    HannWindow
    LayerNormalization
    MelWeightMatrix
    STFT
    SequenceMap
"""

from torch.onnx import symbolic_helper

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py


@symbolic_helper.parse_args("v", "is", "v", "v", "f", "i")
def layer_norm(g, input, normalized_shape, weight, bias, eps, cudnn_enable):
    return g.op(
        "LayerNormalization",
        input,
        weight,
        bias,
        epsilon_f=eps,
        axis_i=len(normalized_shape),
    )
