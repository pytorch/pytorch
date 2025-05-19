# mypy: allow-untyped-defs
"""This file exports ONNX ops for opset 22.

Note [ONNX Operators that are added/updated in opset 22]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-22-of-the-default-onnx-operator-set
New operators:
    Selu
"""

import functools
import torch
from torch import _C
from torch.onnx import symbolic_helper
from torch.onnx._internal import jit_utils, registration

__all__ = [
    "selu",
]

_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=22)

@_onnx_symbolic("aten::selu")
@symbolic_helper.parse_args("v")
def selu(g: jit_utils.GraphContext, self: _C.Value):
    # Use default alpha and gamma values as per ONNX Selu-22 spec
    alpha = 1.6732632
    gamma = 1.0507
    return g.op("Selu", self, alpha_f=alpha, gamma_f=gamma)

