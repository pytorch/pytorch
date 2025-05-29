"""torch.ops.aten operators under the `core` module."""
# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value,type-var,operator,no-untyped-def,index"
# ruff: noqa: TCH001,TCH002
# flake8: noqa: B950

from __future__ import annotations

from typing import Optional

import onnxscript.function_libs.torch_lib.ops.nn as nn
from onnxscript.onnx_opset import opset18 as op18, opset20 as op20, opset21 as op21, opset23 as op23

import torch
from torch.onnx._internal._lazy_import import onnxscript_ir as ir
from torch.onnx._internal.exporter._torchlib._tensor_typing import TFloat, TReal
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


aten = torch.ops.aten


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

@onnx_impl(aten.scaled_dot_product_attention.default, trace_only=True, opset_introduced=23)
def scaled_dot_product_attention(
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
        attn_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0) if is_causal else attn_mask
        attn_mask = attn_mask.masked_fill(not attn_mask, -float('inf')) if attn_mask.dtype==torch.bool else attn_mask
        attn_weight = torch.softmax((Q @ K.transpose(-2, -1) * scale_factor) + attn_mask, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p)
        return attn_weight @ V

    where Q, K, V are the query, key, and value tensors, respectively.
    L is the target sequence length, S is the source sequence length, and E is the embedding size.
    """
    # Use trace_only to handle optional inputs
    assert (not is_causal) or (is_causal and attn_mask is None), (
        "is_causal and attn_mask cannot be set at the same time"
    )

    # Attention onnx op can only handle non-training scenarios where dropout is disabled.
    if dropout_p == 0:
        if enable_gqa:
            assert (query.shape[1] > (key.shape[1] == value.shape[1]) and query.shape[1] % key.shape[1] == 0), (
                "SDPA (GQA or MQA) requires q_num_heads > kv_num_heads & q_num_heads % kv_num_heads == 0"
            )
        else:
            assert (query.shape[1] == key.shape[1] == value.shape[1]), (
                "SDPA (MHA) requires q_num_heads = kv_num_heads"
            )
        Y, _, _, _ = op23.Attention(
            query, key, value, attn_mask=attn_mask, scale=scale, q_num_heads=query.shape[3], kv_num_heads=key.shape[3],
            is_causal=(1 if is_causal == True else 0)
        )
        return Y

    if scale is None:
        scale = nn._attention_scale(query)
    scale = op18.CastLike(scale, query)

    if is_causal:
        attn_mask = nn._causal_attention_mask(query, key)

    if attn_mask is None:
        return nn._aten_scaled_dot_product_attention_no_mask_onnx(
            query, key, value, scale, dropout_p
        )

    return nn._aten_scaled_dot_product_attention_float_mask_onnx(
        query, key, value, attn_mask, scale, dropout_p
    )
