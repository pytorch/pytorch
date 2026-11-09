"""torch.ops.aten operators under the `core` module."""
# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value,type-var,operator,no-untyped-def,index"
# pyrefly: ignore-errors

from __future__ import annotations

from typing import Sequence  # noqa: UP035

import torch
from torch.onnx._internal._lazy_import import onnx_ir as ir
from torch.onnx._internal.exporter._ir_builder import current_builder
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


aten = torch.ops.aten


@onnx_impl(aten.gelu.default, opset_introduced=20)
def aten_gelu_opset20(
    self: ir.Value,
    approximate: str = "none",
) -> ir.Value:
    """gelu(Tensor self, *, str approximate="none") -> Tensor"""
    builder = current_builder()
    return builder.call("Gelu", self, approximate=approximate)[0]


@onnx_impl(aten.group_norm.default, opset_introduced=21)
def aten_group_norm(
    input: ir.Value,
    num_groups: int,
    weight: ir.Value | None = None,
    bias: ir.Value | None = None,
    eps: float = 1e-05,
    cudnn_enabled: bool = True,
) -> ir.Value:
    """group_norm(Tensor input, int num_groups, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enabled=True) -> Tensor"""
    builder = current_builder()

    c = builder.call("Shape", input, start=1, end=2)[0]
    if weight is None:
        weight = builder.call(
            "ConstantOfShape", c, value=ir.tensor([1.0], dtype=input.dtype)
        )[0]
    if bias is None:
        bias = builder.call(
            "ConstantOfShape", c, value=ir.tensor([0.0], dtype=input.dtype)
        )[0]
    return builder.call(
        "GroupNormalization", input, weight, bias, epsilon=eps, num_groups=num_groups
    )[0]


@onnx_impl(aten.rms_norm.default, opset_introduced=23)
def aten_rms_norm(
    input: ir.Value,
    normalized_shape: Sequence[int],
    weight: ir.Value | None = None,
    eps: float | None = None,
) -> ir.Value:
    """rms_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight=None, float? eps=None) -> Tensor"""
    builder = current_builder()

    # Default eps value if not provided
    if eps is None:
        eps = torch.finfo(torch.float).eps  # Observed from decomp

    # Calculate axis: the first normalization dimension
    # For normalized_shape with D dimensions, normalize over last D dimensions
    # Since ONNX RMSNormalization supports negative axis values, we use -len(normalized_shape)
    # which correctly maps to the first axis of the normalized dimensions
    normalized_dims = len(normalized_shape)
    axis = -normalized_dims

    # Create weight tensor if not provided
    if weight is None:
        weight = builder.call(
            "ConstantOfShape",
            builder.call("Shape", input)[0],
            value=ir.tensor([1], dtype=input.dtype),
        )[0]

    return builder.call("RMSNormalization", input, weight, axis=axis, epsilon=eps)[0]


@onnx_impl(aten.scaled_dot_product_attention.default, opset_introduced=23)
def aten_scaled_dot_product_attention_23(
    query: ir.Value,
    key: ir.Value,
    value: ir.Value,
    attn_mask: ir.Value | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> ir.Value:
    """scaled_dot_product_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float dropout_p=0.0, bool is_causal=False, *, float? scale=None, bool enable_gqa=False) -> Tensor

    Reference:
        1. https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        2. https://onnx.ai/onnx/operators/onnx__Attention.html

    Attempts to convert SDPA to Attention onnx op and fallbacks to an onnx graph equivalent to the following PyTorch code::
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
            (Q @ K.transpose(-2, -1) * scale_factor) + attn_mask, dim=-1
        )
        attn_weight = torch.dropout(attn_weight, dropout_p)
        return attn_weight @ V

    where Q, K, V are the query, key, and value tensors, respectively.
    L is the target sequence length, S is the source sequence length, and E is the embedding size.
    """
    builder = current_builder()
    if is_causal and attn_mask is not None:
        raise AssertionError("is_causal and attn_mask cannot be set at the same time")
    if not (len(query.shape) == 4 and len(key.shape) == 4 and len(value.shape) == 4):
        raise AssertionError("only 4D query, key, and value are supported")

    # Attention onnx op can only handle non-training scenarios where dropout is disabled.
    if dropout_p == 0:
        if enable_gqa:
            if not (
                query.shape[1] > key.shape[1] == value.shape[1]
                and query.shape[1] % key.shape[1] == 0
            ):
                raise AssertionError(
                    "SDPA (GQA or MQA) requires q_num_heads > kv_num_heads & "
                    "q_num_heads % kv_num_heads == 0"
                )
        else:
            if not (query.shape[1] == key.shape[1] == value.shape[1]):
                raise AssertionError("SDPA (MHA) requires q_num_heads = kv_num_heads")

        # NOTE: num_heads attributes (q_num_heads/kv_num_heads) should not be specified for 4D.
        # They are not populated with 4D inputs because this information directly comes from input shapes:
        # `q_num_heads=query.shape[1]` and `kv_num_heads=key.shape[1]`.
        # This dimension is usually static but it could not be dynamic if also given as an attribute.
        # num_heads attributes are needed for 3D attention inputs:
        # (shape: [B, S, N*H]), 4D shape is ([B, N, S, H]).

        Y, _, _, _ = builder.call(
            "Attention",
            query,
            key,
            value,
            attn_mask,
            _num_outputs=4,
            scale=scale,
            is_causal=is_causal,
        )
        return Y

    if scale is None:
        scale = _attention_scale(query)
    scale = builder.call("CastLike", scale, query)[0]

    if is_causal:
        attn_mask = _causal_attention_mask(query, key)

    if enable_gqa:
        key, value = _attention_repeat_kv_for_group_query(query, key, value)

    if attn_mask is None:
        return _aten_scaled_dot_product_attention_no_mask_onnx(
            query, key, value, scale, dropout_p
        )

    return _aten_scaled_dot_product_attention_float_mask_onnx(
        query, key, value, attn_mask, scale, dropout_p
    )


def _attention_repeat_kv_for_group_query(
    query: ir.Value, key: ir.Value, value: ir.Value
) -> tuple[ir.Value, ir.Value]:
    """Expand key and value for group query attention.

    repeat_interleave is applied on key and value to match the number of heads in query.

    Args:
        query: Tensor of shape [B, q_num_heads, q_S, E]
        key: Tensor of shape [B, k_num_heads, kv_S, E]
        value: Tensor of shape [B, v_num_heads, kv_S, E]

    Returns:
        Tuple of (expanded_key, expanded_value) where:
            - expanded_key: Tensor of shape [B, q_num_heads, kv_S, E]
            - expanded_value: Tensor of shape [B, q_num_heads, kv_S, E]
    """
    builder = current_builder()

    if not (
        query.shape[1] > key.shape[1] == value.shape[1]
        and query.shape[1] % key.shape[1] == 0
    ):
        raise AssertionError(
            "SDPA (GQA or MQA) requires q_num_heads > kv_num_heads & "
            "q_num_heads % kv_num_heads == 0"
        )

    # NOTE: QKV are expected to be 4D tensors

    batch_size = builder.call("Shape", query, start=0, end=1)[0]  # [B]
    q_num_heads = builder.call("Shape", query, start=1, end=2)[0]  # [Hq]
    kv_num_heads = builder.call("Shape", key, start=1, end=2)[0]  # [Hk]
    qk_head_size = builder.call("Shape", key, start=3, end=4)[0]  # [Dk]
    v_head_size = builder.call("Shape", value, start=3, end=4)[0]  # [Dv]
    new_kv_seq_len = builder.call("Shape", key, start=2, end=3)[0]  # [T]

    interleave_dim = builder.call("Div", q_num_heads, kv_num_heads)[0]  # Hq / Hk
    two = builder.call("Constant", value_int=2)[0]
    k_unsqueezed = builder.call("Unsqueeze", key, two)[0]  # [B, Hk, 1, T, Dk]
    v_unsqueezed = builder.call("Unsqueeze", value, two)[0]  # [B, Hv, 1, T, Dv]

    k_expand_shape = builder.call(
        "Concat",
        batch_size,
        kv_num_heads,
        interleave_dim,
        new_kv_seq_len,
        qk_head_size,
        axis=0,
    )[0]
    k_expand = builder.call("Expand", k_unsqueezed, k_expand_shape)[0]
    v_expand_shape = builder.call(
        "Concat",
        batch_size,
        kv_num_heads,
        interleave_dim,
        new_kv_seq_len,
        v_head_size,
        axis=0,
    )[0]
    v_expand = builder.call("Expand", v_unsqueezed, v_expand_shape)[0]

    k_attention_shape = builder.call(
        "Concat", batch_size, q_num_heads, new_kv_seq_len, qk_head_size, axis=0
    )[0]
    v_attention_shape = builder.call(
        "Concat", batch_size, q_num_heads, new_kv_seq_len, v_head_size, axis=0
    )[0]

    expanded_key = builder.call("Reshape", k_expand, k_attention_shape)[0]
    expanded_value = builder.call("Reshape", v_expand, v_attention_shape)[0]

    return expanded_key, expanded_value


def _attention_scale(query: ir.Value) -> ir.Value:
    """Calculate the scale factor for the attention result.

    Args:
        query: Tensor of shape [..., L, E]

    Returns:
        Scalar scale factor := 1 / math.sqrt(query.size(-1))
    """
    builder = current_builder()
    q_shape = builder.call("Shape", query)[0]
    q_last_dim = builder.call(
        "Gather", q_shape, builder.call("Constant", value_ints=[-1])[0]
    )[0]
    embedding_size = builder.call("CastLike", q_last_dim, query)[0]
    one = builder.call("Constant", value_float=1.0)[0]
    cast_one = builder.call("CastLike", one, query)[0]
    scale = builder.call("Div", cast_one, builder.call("Sqrt", embedding_size)[0])[0]
    return scale


def _causal_attention_mask(query: ir.Value, key: ir.Value) -> ir.Value:
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
    builder = current_builder()
    q_shape = builder.call("Shape", query)[0]
    k_shape = builder.call("Shape", key)[0]

    target_length = builder.call(
        "Slice",
        q_shape,
        builder.call("Constant", value_ints=[-2])[0],
        builder.call("Constant", value_ints=[-1])[0],
    )[0]
    source_length = builder.call(
        "Slice",
        k_shape,
        builder.call("Constant", value_ints=[-2])[0],
        builder.call("Constant", value_ints=[-1])[0],
    )[0]
    # attn_mask = torch.ones(L, S) := {
    size = builder.call("Concat", target_length, source_length, axis=0)[0]
    attn_mask = builder.call(
        "Expand", builder.call("Constant", value_float=1.0)[0], size
    )[0]
    # }
    attn_mask = builder.call("Trilu", attn_mask, upper=0)[0]
    # The causal mask has 0s in the lower triangle and -inf in the upper triangle.
    attn_mask = builder.call(
        "Where",
        builder.call("Equal", attn_mask, builder.call("Constant", value_float=0.0)[0])[
            0
        ],
        builder.call("Constant", value_float=-float("inf"))[0],
        builder.call("Constant", value_float=0.0)[0],
    )[0]
    attn_mask = builder.call("CastLike", attn_mask, query)[0]
    return attn_mask


def _aten_scaled_dot_product_attention_no_mask_onnx(
    query: ir.Value,
    key: ir.Value,
    value: ir.Value,
    scale: ir.Value,
    dropout_p: float,
) -> ir.Value:
    # Swap the last two axes of key
    builder = current_builder()
    key_last_dim = builder.call("Shape", key, start=-1)[0]
    key_second_last_dim = builder.call("Shape", key, start=-2, end=-1)[0]
    key_first_dims = builder.call("Shape", key, end=-2)[0]
    # Contract the dimensions that are not the last two so we can transpose
    # with a static permutation.
    key_squeezed_shape = builder.call(
        "Concat",
        builder.call("Constant", value_ints=[-1])[0],
        key_second_last_dim,
        key_last_dim,
        axis=0,
    )[0]
    key_squeezed = builder.call("Reshape", key, key_squeezed_shape)[0]
    key_squeezed_transposed = builder.call("Transpose", key_squeezed, perm=[0, 2, 1])[0]
    key_transposed_shape = builder.call(
        "Concat", key_first_dims, key_last_dim, key_second_last_dim, axis=0
    )[0]
    key_transposed = builder.call(
        "Reshape", key_squeezed_transposed, key_transposed_shape
    )[0]

    # https://github.com/pytorch/pytorch/blob/12da0c70378b5be9135c6fda62a9863bce4a4818/aten/src/ATen/native/transformers/attention.cpp#L653
    # Scale q, k before matmul for stability see https://tinyurl.com/sudb9s96 for math
    query_scaled = builder.call("Mul", query, builder.call("Sqrt", scale)[0])[0]
    key_transposed_scaled = builder.call(
        "Mul",
        key_transposed,
        builder.call("CastLike", builder.call("Sqrt", scale)[0], key_transposed)[0],
    )[0]
    attn_weight = builder.call(
        "Softmax",
        builder.call("MatMul", query_scaled, key_transposed_scaled)[0],
        axis=-1,
    )[0]
    attn_weight, _ = builder.call("Dropout", attn_weight, dropout_p, _num_outputs=2)
    return builder.call("MatMul", attn_weight, value)[0]


def _aten_scaled_dot_product_attention_float_mask_onnx(
    query: ir.Value,
    key: ir.Value,
    value: ir.Value,
    attn_mask: ir.Value,
    scale: ir.Value,
    dropout_p: float,
) -> ir.Value:
    # Swap the last two axes of key
    builder = current_builder()
    key_last_dim = builder.call("Shape", key, start=-1)[0]
    key_second_last_dim = builder.call("Shape", key, start=-2, end=-1)[0]
    key_first_dims = builder.call("Shape", key, end=-2)[0]
    # Contract the dimensions that are not the last two so we can transpose
    # with a static permutation.
    key_squeezed_shape = builder.call(
        "Concat",
        builder.call("Constant", value_ints=[-1])[0],
        key_second_last_dim,
        key_last_dim,
        axis=0,
    )[0]
    key_squeezed = builder.call("Reshape", key, key_squeezed_shape)[0]
    key_squeezed_transposed = builder.call("Transpose", key_squeezed, perm=[0, 2, 1])[0]
    key_transposed_shape = builder.call(
        "Concat", key_first_dims, key_last_dim, key_second_last_dim, axis=0
    )[0]
    key_transposed = builder.call(
        "Reshape", key_squeezed_transposed, key_transposed_shape
    )[0]

    # https://github.com/pytorch/pytorch/blob/12da0c70378b5be9135c6fda62a9863bce4a4818/aten/src/ATen/native/transformers/attention.cpp#L653
    # Scale q, k before matmul for stability see https://tinyurl.com/sudb9s96 for math
    query_scaled = builder.call("Mul", query, builder.call("Sqrt", scale)[0])[0]
    key_transposed_scaled = builder.call(
        "Mul", key_transposed, builder.call("Sqrt", scale)[0]
    )[0]
    attn_weight = builder.call(
        "Softmax",
        builder.call(
            "Add",
            builder.call("MatMul", query_scaled, key_transposed_scaled)[0],
            attn_mask,
        )[0],
        axis=-1,
    )[0]
    attn_weight, _ = builder.call("Dropout", attn_weight, dropout_p, _num_outputs=2)
    return builder.call("MatMul", attn_weight, value)[0]
