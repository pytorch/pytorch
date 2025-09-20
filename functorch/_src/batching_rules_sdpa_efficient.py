# SPDX-License-Identifier: BSD-3-Clause
"""
Batching rule for aten::_scaled_dot_product_efficient_attention (+ backward).

Strategy:
- If vmap is over the batch dimension, move that dim into the heads axis (-3),
  so we can call the efficient attention primitive once.
- Tell vmap that the output (and grads) carry their batch along -3 as well.
"""
from __future__ import annotations
from typing import Optional, Tuple

import torch
from torch import Tensor

# functorch C-extension exposes register_batch_rule; in the monorepo it's here:
from functorch import _C as functorchC  # type: ignore[attr-defined]


def _move_bdim(x: Tensor, bdim: Optional[int], to: int) -> Tuple[Tensor, Optional[int]]:
    """Move a vmapped (batch) dim to position `to`; no-op if bdim is None."""
    if bdim is None:
        return x, None
    if bdim < 0:
        bdim = x.dim() + bdim
    if to < 0:
        to = x.dim() + to
    return x.movedim(bdim, to), to


def _sdpa_efficient_batching_rule(
    q: Tensor, q_bdim: Optional[int],
    k: Tensor, k_bdim: Optional[int],
    v: Tensor, v_bdim: Optional[int],
    attn_mask: Optional[Tensor], attn_bdim: Optional[int],
    dropout_p: float,
    is_causal: bool,
    scale: Optional[float],
    enable_gqa: bool = False,
):
    """
    Batching rule for:
      aten::_scaled_dot_product_efficient_attention(q, k, v, attn_mask=None,
        dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False)
    """
    TARGET = -3  # heads dim (common layout: [B?, H, L, D])

    q, q_bdim = _move_bdim(q, q_bdim, TARGET)
    k, k_bdim = _move_bdim(k, k_bdim, TARGET)
    v, v_bdim = _move_bdim(v, v_bdim, TARGET)
    if attn_mask is not None:
        attn_mask, attn_bdim = _move_bdim(attn_mask, attn_bdim, TARGET)

    out = torch._scaled_dot_product_efficient_attention(  # type: ignore[attr-defined]
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )
    # We folded batch into heads; return output + its bdim.
    return out, TARGET


def _sdpa_efficient_backward_batching_rule(
    grad: Tensor, grad_bdim: Optional[int],
    q: Tensor, q_bdim: Optional[int],
    k: Tensor, k_bdim: Optional[int],
    v: Tensor, v_bdim: Optional[int],
    out: Tensor, out_bdim: Optional[int],
    attn_mask: Optional[Tensor], attn_bdim: Optional[int],
    dropout_p: float,
    is_causal: bool,
    scale: Optional[float],
    enable_gqa: bool = False,
):
    """
    Batching rule for:
      aten::_scaled_dot_product_efficient_attention_backward(grad, q, k, v, out,
         attn_mask, dropout_p, is_causal, scale, enable_gqa)
    """
    TARGET = -3

    grad, grad_bdim = _move_bdim(grad, grad_bdim, TARGET)
    q, q_bdim = _move_bdim(q, q_bdim, TARGET)
    k, k_bdim = _move_bdim(k, k_bdim, TARGET)
    v, v_bdim = _move_bdim(v, v_bdim, TARGET)
    out, out_bdim = _move_bdim(out, out_bdim, TARGET)
    if attn_mask is not None:
        attn_mask, attn_bdim = _move_bdim(attn_mask, attn_bdim, TARGET)

    gq, gk, gv = torch._scaled_dot_product_efficient_attention_backward(  # type: ignore[attr-defined]
        grad, q, k, v, out, attn_mask, dropout_p, is_causal, scale, enable_gqa
    )
    # Return ((tensor, bdim), ...) per output for vmap plumbing.
    return (gq, TARGET), (gk, TARGET), (gv, TARGET)


# ---- Register with functorch ----
# Names must match the dispatcher schema exactly.
functorchC.register_batch_rule(
    "aten::_scaled_dot_product_efficient_attention",
    _sdpa_efficient_batching_rule,
)
functorchC.register_batch_rule(
    "aten::_scaled_dot_product_efficient_attention_backward",
    _sdpa_efficient_backward_batching_rule,
)
