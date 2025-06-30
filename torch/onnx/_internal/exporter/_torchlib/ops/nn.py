"""torch.ops.aten operators under the `core` module."""
# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value,type-var,operator,no-untyped-def,index"
# ruff: noqa: TCH001,TCH002
# flake8: noqa: B950

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from onnxscript.onnx_opset import (  # type: ignore[attr-defined]
    opset20 as op20,
    opset21 as op21,
    opset23 as op23,
)

import torch
from torch.onnx._internal._lazy_import import onnxscript_ir as ir
from torch.onnx._internal.exporter._torchlib._tensor_typing import TFloat, TReal
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


if TYPE_CHECKING:
    from onnxscript.values import Opset

aten = torch.ops.aten

_INT64_MAX = 9223372036854775807
_INT64_MIN = -9223372036854775808


@onnx_impl(aten.gelu.default, trace_only=True, opset_introduced=20)
def aten_gelu_opset20(
    self: TReal,
    approximate: str = "none",
) -> TReal:
    """gelu(Tensor self, *, bool approximate=False) -> Tensor"""
    return op20.Gelu(self, approximate=approximate)


@onnx_impl(aten.group_norm.default, trace_only=True, opset_introduced=21)
def aten_group_norm(
    input: TFloat,
    num_groups: int,
    weight: Optional[TFloat] = None,
    bias: Optional[TFloat] = None,
    eps: float = 1e-05,
    cudnn_enabled: bool = True,
) -> TFloat:
    """group_norm(Tensor input, int num_groups, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enabled=True) -> Tensor"""

    c = op21.Shape(input, start=1, end=2)
    if weight is None:
        weight = op21.ConstantOfShape(c, value=ir.tensor(1.0, dtype=input.dtype))
    if bias is None:
        bias = op21.ConstantOfShape(c, value=ir.tensor(0.0, dtype=input.dtype))
    return op21.GroupNormalization(
        input, weight, bias, epsilon=eps, num_groups=num_groups
    )


@onnx_impl(
    aten.scaled_dot_product_attention.default, trace_only=True, opset_introduced=23
)
def aten_scaled_dot_product_attention_23(
    query: TFloat,
    key: TFloat,
    value: TFloat,
    attn_mask: Optional[TFloat] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> TFloat:
    """scaled_dot_product_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float dropout_p=0.0, bool is_causal=False, *, float? scale=None, bool enable_gqa=False) -> Tensor

    Reference:
        1. https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        2. https://onnx.ai/onnx/operators/onnx__Attention.html

    Attempts to convert SDPA to Attention onnx op and fallbacks to an onnx graph equivivalent to the following PyTorch code::
        scale_factor = 1 / math.sqrt(Q.size(-1)) if scale is None else scale
        attn_mask = (
            torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            if is_causal
            else attn_mask
        )
        attn_mask = (
            attn_mask.masked_fill(not attn_mask, -float("inf"))
            if attn_mask.dtype == torch.bool
            else attn_mask
        )
        attn_weight = torch.softmax(
            (Q @ K.transpose(-2, -1) *  attn_mask, dim=-1
        )
        attn_weight = torch.dropout(attn_weight, dropout_p)
        return attn_weight @ V

    where Q, K, V are the query, key, and value tensors, respectively.
    L is the target sequence length, S is the source sequence length, and E is the embedding size.
    """
    assert (not is_causal) or (is_causal and attn_mask is None), (
        "is_causal and attn_mask cannot be set at the same time"
    )
    assert len(query.shape) == 4 and len(key.shape) == 4 and len(value.shape) == 4, (
        "only 4D query, key, and value are supported"
    )

    # Attention onnx op can only handle non-training scenarios where dropout is disabled.
    if dropout_p == 0:
        if enable_gqa:
            assert (
                query.shape[1] > key.shape[1] == value.shape[1]
                and query.shape[1] % key.shape[1] == 0
            ), (
                "SDPA (GQA or MQA) requires q_num_heads > kv_num_heads & q_num_heads % kv_num_heads == 0"
            )
        else:
            assert query.shape[1] == key.shape[1] == value.shape[1], (
                "SDPA (MHA) requires q_num_heads = kv_num_heads"
            )

        # NOTE: num_heads attributes (q_num_heads/kv_num_heads) should not be specified for 4D.
        # They are not populated with 4D inputs because this information directy comes from input shapes:
        # `q_num_heads=query.shape[1]` and `kv_num_heads=key.shape[1]`.
        # This dimension is usually static but it could not be dynamic if also given as an attribute.
        # num_heads attributes are needed for 3D attention inputs:
        # (shape: [B, S, N*H]), 4D shape is ([B, N, S, H]).

        Y, _, _, _ = op23.Attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            scale=scale,
            is_causal=is_causal,
        )
        return Y

    if scale is None:
        scale = _attention_scale(query, op23)
    scale = op23.CastLike(scale, query)

    if is_causal:
        attn_mask = _causal_attention_mask(query, key, op23)

    if attn_mask is None:
        return _aten_scaled_dot_product_attention_no_mask_onnx(
            query, key, value, scale, dropout_p, op23
        )

    return _aten_scaled_dot_product_attention_float_mask_onnx(
        query, key, value, attn_mask, scale, dropout_p, op23
    )


def _attention_scale(query: TFloat, op: Opset) -> TFloat:
    """Calculate the scale factor for the attention result.

    Args:
        query: Tensor of shape [..., L, E]

    Returns:
        Scalar scale factor := 1 / math.sqrt(query.size(-1))
    """
    q_shape = op.Shape(query)
    q_last_dim = op.Gather(q_shape, op.Constant(value_ints=[-1]))
    embedding_size = op.CastLike(q_last_dim, query)
    one = op.Constant(value_float=1.0)
    cast_one = op.CastLike(one, query)
    scale = op.Div(cast_one, op.Sqrt(embedding_size))
    return scale


def _causal_attention_mask(query: TFloat, key: TFloat, op: Opset) -> TFloat:
    """Create a causal mask for the given query and key tensors.

    Equivalent to::
        mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_mask = torch.zeros(L, S, dtype=torch.float)
        attn_mask = attn_mask.masked_fill(not mask, -float("inf"))

    Args:
        query: Tensor of shape [..., L, E]
        key: Tensor of shape [..., S, E]

    Returns:
        Tensor of shape [L, S]
    """
    q_shape = op.Shape(query)
    k_shape = op.Shape(key)

    target_length = op.Slice(
        q_shape, op.Constant(value_ints=[-2]), op.Constant(value_ints=[-1])
    )
    source_length = op.Slice(
        k_shape, op.Constant(value_ints=[-2]), op.Constant(value_ints=[-1])
    )
    # attn_mask = torch.ones(L, S) := {
    size = op.Concat(target_length, source_length, axis=0)
    attn_mask = op.Expand(op.Constant(value_float=1.0), size)
    # }
    attn_mask = op.Trilu(attn_mask, upper=0)
    # The causal mask has 0s in the lower triangle and -inf in the upper triangle.
    attn_mask = op.Where(
        op.Equal(attn_mask, op.Constant(value_float=0.0)),
        op.Constant(value_float=-float("inf")),
        op.Constant(value_float=0.0),
    )
    attn_mask = op.CastLike(attn_mask, query)
    return attn_mask


def _aten_scaled_dot_product_attention_no_mask_onnx(
    query: TFloat,
    key: TFloat,
    value: TFloat,
    scale: TFloat,
    dropout_p: float,
    op: Opset,
) -> TFloat:
    # Swap the last two axes of key
    key_last_dim = op.Shape(key, start=-1)
    key_second_last_dim = op.Shape(key, start=-2, end=-1)
    key_first_dims = op.Shape(key, end=-2)
    # Contract the dimensions that are not the last two so we can transpose
    # with a static permutation.
    key_squeezed_shape = op.Concat(
        op.Constant(value_ints=[-1]), key_second_last_dim, key_last_dim, axis=0
    )
    key_squeezed = op.Reshape(key, key_squeezed_shape)
    key_squeezed_transposed = op.Transpose(key_squeezed, perm=[0, 2, 1])
    key_transposed_shape = op.Concat(
        key_first_dims, key_last_dim, key_second_last_dim, axis=0
    )
    key_transposed = op.Reshape(key_squeezed_transposed, key_transposed_shape)

    # https://github.com/pytorch/pytorch/blob/12da0c70378b5be9135c6fda62a9863bce4a4818/aten/src/ATen/native/transformers/attention.cpp#L653
    # Scale q, k before matmul for stability see https://tinyurl.com/sudb9s96 for math
    query_scaled = op.Mul(query, op.Sqrt(scale))
    key_transposed_scaled = op.Mul(
        key_transposed, op.CastLike(op.Sqrt(scale), key_transposed)
    )
    attn_weight = op.Softmax(
        op.MatMul(query_scaled, key_transposed_scaled),
        axis=-1,
    )
    attn_weight, _ = op.Dropout(attn_weight, dropout_p)
    return op.MatMul(attn_weight, value)


def _aten_scaled_dot_product_attention_float_mask_onnx(
    query: TFloat,
    key: TFloat,
    value: TFloat,
    attn_mask: TFloat,
    scale: TFloat,
    dropout_p: float,
    op: Opset,
) -> TFloat:
    # Swap the last two axes of key
    key_last_dim = op.Shape(key, start=-1)
    key_second_last_dim = op.Shape(key, start=-2, end=-1)
    key_first_dims = op.Shape(key, end=-2)
    # Contract the dimensions that are not the last two so we can transpose
    # with a static permutation.
    key_squeezed_shape = op.Concat(
        op.Constant(value_ints=[-1]), key_second_last_dim, key_last_dim, axis=0
    )
    key_squeezed = op.Reshape(key, key_squeezed_shape)
    key_squeezed_transposed = op.Transpose(key_squeezed, perm=[0, 2, 1])
    key_transposed_shape = op.Concat(
        key_first_dims, key_last_dim, key_second_last_dim, axis=0
    )
    key_transposed = op.Reshape(key_squeezed_transposed, key_transposed_shape)

    # https://github.com/pytorch/pytorch/blob/12da0c70378b5be9135c6fda62a9863bce4a4818/aten/src/ATen/native/transformers/attention.cpp#L653
    # Scale q, k before matmul for stability see https://tinyurl.com/sudb9s96 for math
    query_scaled = op.Mul(query, op.Sqrt(scale))
    key_transposed_scaled = op.Mul(key_transposed, op.Sqrt(scale))
    attn_weight = op.Softmax(
        op.Add(op.MatMul(query_scaled, key_transposed_scaled), attn_mask),
        axis=-1,
    )
    attn_weight, _ = op.Dropout(attn_weight, dropout_p)
    return op.MatMul(attn_weight, value)
