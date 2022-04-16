# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

# This file exports ONNX ops for opset 16

# Note [ONNX Operators that are added/updated in opset 16]
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-16-of-the-default-onnx-operator-set
# New operators:
#   GridSample https://github.com/onnx/onnx/pull/3557
#
# Updated operators:
#    Identity
#    If
#    LeakyRelu
#    Loop
#    PRelu
#    RoiAlign
#    Scan
#    ScatterElemenets
#    ScatterND
#    Where
#    GreaterOrEqual
#    LessOrEqual
#    SequenceMap

from torch.onnx.symbolic_helper import parse_args


@parse_args("v", "v", "s", "s", "b")
def gridsample2d(g, input, grid, mode, padding_mode, align_corners):
    return g.op(
        "GridSample",
        input,
        grid,
        align_corners_i=int(align_corners),
        mode_s=mode,
        padding_mode_s=padding_mode,
    )
