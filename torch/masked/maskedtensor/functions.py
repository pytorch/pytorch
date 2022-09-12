# Copyright (c) Meta Platforms, Inc. and affiliates

import math

import torch
from torch.nn.functional import dropout, linear

from .matmul import masked_bmm

__all__ = ['multi_head_attention_forward']

def _in_projection_packed(q, k, v, w, b):
    w_q, w_k, w_v = w.chunk(3)
    if b is not None:
        raise ValueError("b must be None")
    b_q = b_k = b_v = None
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0):
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = masked_bmm(q, k.transpose(-2, -1), attn_mask)
    attn = torch.nn.functional.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn


def multi_head_attention_forward(
    query,
    key,
    value,
    embed_dim_to_check,
    num_heads,
    in_proj_weight,
    in_proj_bias,
    bias_k,
    bias_v,
    add_zero_attn,
    dropout_p,
    out_proj_weight,
    out_proj_bias,
    training,
    key_padding_mask,
    need_weights,
    attn_mask,
    use_separate_proj_weight,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    static_k,
    static_v,
    average_attn_weights,
):
    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    if embed_dim != embed_dim_to_check:
        raise ValueError(f"Was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}")

    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode="trunc")
    else:
        head_dim = embed_dim // num_heads

    if head_dim * num_heads != embed_dim:
        raise ValueError(f"embed_dim {embed_dim} not divisible by num_heads {num_heads}")

    if use_separate_proj_weight:
        raise ValueError("`use_separate_proj_weight` must be False")
    if q_proj_weight is not None:
        raise ValueError("`q_proj_weight` must be None")
    if k_proj_weight is not None:
        raise ValueError("`k_proj_weight` must be None")
    if v_proj_weight is not None:
        raise ValueError("`v_proj_weight` must be None")
    if in_proj_bias is not None:
        raise ValueError("`in_proj_bias` must be None")
    if bias_k is not None:
        raise ValueError("`bias_k` must be None")
    if bias_v is not None:
        raise ValueError("`bias_v` must be None")
    if static_k is not None:
        raise ValueError("`static_k` must be None")
    if static_v is not None:
        raise ValueError("`static_v` must be None")
    if add_zero_attn:
        raise ValueError("`add_zero_attn` must be False")
    if key_padding_mask is not None:
        raise ValueError("`key_padding_mask` must be None")

    q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)

    # prep attention mask
    if attn_mask is not None:
        if not (attn_mask.is_floating_point() or attn_mask.dtype == torch.bool):
            raise TypeError(f"Only float and bool types are supported for attn_mask, not {attn_mask.dtype}")
        # ensure attn_mask's dim is 3
        if attn_mask.dim() != 2:
            raise ValueError("attn_mask must have dim 3")
        correct_2d_size = (tgt_len, src_len)
        if attn_mask.shape != correct_2d_size:
            raise RuntimeError(
                f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
            )
        attn_mask = attn_mask.unsqueeze(0)

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

    # update source sequence length after adjustments
    src_len = k.size(1)

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    attn_output, attn_output_weights = _scaled_dot_product_attention(
        q, k, v, attn_mask, dropout_p
    )
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.sum(dim=1) / num_heads
        return attn_output, attn_output_weights
    else:
        return attn_output, None
