# mypy: allow-untyped-defs
"""This file exports ONNX ops for opset 23.

Note [ONNX Operators that are added/updated in opset 23]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-23-of-the-default-onnx-operator-set
New operators:
    Attention
    RMSNormalization
    Reshape
    RotaryEmbedding
"""

import functools
from torch import _C
from torch.onnx import symbolic_helper
from torch.onnx._internal import jit_utils, registration

__all__ = ["attention", "rms_normalization", "reshape", "rotary_embedding"]

_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=23)


@_onnx_symbolic("aten::attention")
@symbolic_helper.parse_args("v", "v", "v", "v", "v", "v", "i", "i", "i", "f", "f", "i")
def attention(
    g: jit_utils.GraphContext,
    q: _C.Value,
    k: _C.Value,
    v: _C.Value,
    attn_mask: _C.Value,
    past_key: _C.Value,
    past_value: _C.Value,
    q_num_heads: int,
    kv_num_heads: int,
    qk_matmul_output_mode: int,
    scale: float,
    softcap: float,
    softmax_precision: int,
):
    inputs = [q, k, v]
    if attn_mask.node().kind() != "prim::Constant" or attn_mask.type().kind() != "NoneType":
        inputs.append(attn_mask)
    if past_key.node().kind() != "prim::Constant" or past_key.type().kind() != "NoneType":
        inputs.append(past_key)
    if past_value.node().kind() != "prim::Constant" or past_value.type().kind() != "NoneType":
        inputs.append(past_value)

    return g.op(
        "Attention",
        *inputs,
        q_num_heads_i=q_num_heads,
        kv_num_heads_i=kv_num_heads,
        qk_matmul_output_mode_i=qk_matmul_output_mode,
        scale_f=scale,
        softcap_f=softcap,
        softmax_precision_i=softmax_precision,
        outputs=4,
    )


@_onnx_symbolic("aten::rms_norm")
@symbolic_helper.parse_args("v", "v", "i", "f")
def rms_normalization(g, input, scale, axis, epsilon):
    squared = g.op("Mul", input, input)
    rank = symbolic_helper._get_tensor_rank(input)
    axes = list(range(axis if axis >= 0 else rank + axis, rank)) # type: ignore
    mean = g.op("ReduceMean", squared, axes_i=axes, keepdims_i=1)
    mean_eps = g.op("Add", mean, g.op("Constant", value_t=symbolic_helper._scalar(epsilon)))
    rms = g.op("Sqrt", mean_eps)
    normalized = g.op("Div", input, rms)
    return g.op("Mul", normalized, scale)


@_onnx_symbolic("aten::reshape")
@symbolic_helper.parse_args("v", "v")
def reshape(g, input, shape):
    return g.op("Reshape", input, shape)


@_onnx_symbolic("aten::rotary_embedding")
@symbolic_helper.parse_args("v", "v", "v", "v", "i", "i", "i")
def rotary_embedding(g, input, position_ids, sin_cache, cos_cache, interleaved, rotary_embedding_dim, num_heads):
    return g.op(
        "RotaryEmbedding",
        input,
        position_ids,
        sin_cache,
        cos_cache,
        interleaved_i=interleaved,
        rotary_embedding_dim_i=rotary_embedding_dim,
        num_heads_i=num_heads
    )
