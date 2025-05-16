# mypy: allow-untyped-defs
"""This file exports ONNX ops for opset 23.

Note [ONNX Operators that are added/updated in opset 23]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-23-of-the-default-onnx-operator-set
New operators:
    - Attention
"""

import functools
from torch import _C
from torch.onnx import symbolic_helper
from torch.onnx._internal import jit_utils, registration

__all__ = ["attention"]

_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=23)

@_onnx_symbolic("aten::attention")
@symbolic_helper.parse_args("v", "v", "v", "v", "v", "v", "v", "v", "v", "v", "v", "v", "v")
def attention(
    g: jit_utils.GraphContext,
    query: _C.Value,
    key: _C.Value,
    value: _C.Value,
    bias: _C.Value,
    mask_index: _C.Value,
    past_key: _C.Value,
    past_value: _C.Value,
    static_kv: _C.Value,
    use_past: _C.Value,
    unidirectional: _C.Value,
    num_heads: _C.Value,
    scale: _C.Value,
    dropout: _C.Value,
):
    return g.op(
        "Attention",
        query,
        key,
        value,
        bias,
        mask_index,
        past_key,
        past_value,
        static_kv,
        use_past,
        unidirectional,
        num_heads,
        scale,
        dropout,
    )
