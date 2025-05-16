# mypy: allow-untyped-defs
"""This file exports ONNX ops for opset 21.

Note [ONNX Operators that are added/updated in opset 21]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-21-of-the-default-onnx-operator-set
New operators:
    - Gelu
"""

import functools

import torch.nn.functional as F
from torch import _C
from torch.onnx import symbolic_helper
from torch.onnx._internal import jit_utils, registration

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

__all__ = [
    "gelu",
]

def convert_grid_sample_mode(mode_s):
    return (
        "linear" if mode_s == "bilinear" else "cubic" if mode_s == "bicubic" else mode_s
    )

_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=21)


@_onnx_symbolic("aten::gelu")
@symbolic_helper.parse_args("v", "s")
def gelu(g: jit_utils.GraphContext, self: _C.Value, approximate: str = "none"):
    return g.op("Gelu", self, approximate_s=approximate)
