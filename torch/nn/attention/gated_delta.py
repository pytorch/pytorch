# Copyright (c) Meta Platforms, Inc. and affiliates
# Gated DeltaNet (GDN) for PyTorch.
#
# The recurrence matches Megatron-LM / FLA ``chunk_gated_delta_rule``:
# decay the state, then a Householder-style delta write, then read with q.
# Context parallelism is *not* Ring-Attention: GDN is sequential in time, so
# CP follows Megatron — all-to-all sequence-sharded activations onto a head
# shard, run the recurrence on the full sequence, all-to-all back.
from __future__ import annotations

import math

import torch
from torch import Tensor


__all__: list[str] = []


def _l2norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)


def _gated_delta_rule_impl(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    decay: Tensor,
    beta: Tensor,
    *,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
) -> Tensor:
    """Gated delta-rule core.

    Args:
        query, key: ``[B, T, H, K]``
        value: ``[B, T, H, V]``
        decay: ``[B, T, H]`` log-space gate; state is multiplied by ``exp(decay)``.
        beta: ``[B, T, H]`` write strength, typically in ``(0, 1)``.
        scale: optional multiplier on ``query``. Default ``1/sqrt(K)``.
        use_qk_l2norm: L2-normalize q/k along the head dim (FLA default).

    Returns:
        output of shape ``[B, T, H, V]``.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query/key/value must be [B, T, H, dim]")
    if decay.shape != query.shape[:3] or beta.shape != query.shape[:3]:
        raise ValueError("decay/beta must be [B, T, H]")
    if query.shape != key.shape:
        raise ValueError("query and key must share shape")
    if value.shape[:3] != query.shape[:3]:
        raise ValueError("value batch/time/heads must match query")

    q = query
    k = key
    v = value
    if use_qk_l2norm:
        q = _l2norm(q)
        k = _l2norm(k)
    if scale is None:
        scale = 1.0 / math.sqrt(q.size(-1))
    q = q * scale
    return _gated_delta_rule_python(q, k, v, decay, beta)


def gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    decay: Tensor,
    beta: Tensor,
    *,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
) -> Tensor:
    """Apply the gated delta rule to query, key, and value.

    Recurrence matches Megatron-LM / FLA ``chunk_gated_delta_rule``:
    decay the state, apply a Householder-style delta write, then read with
    ``query``. Context parallel patches this name, not
    ``_gated_delta_rule_impl``.

    Args:
        query, key: ``[B, T, H, K]``
        value: ``[B, T, H, V]``
        decay: ``[B, T, H]`` log-space gate; state is multiplied by ``exp(decay)``.
        beta: ``[B, T, H]`` write strength, typically in ``(0, 1)``.
        scale: optional multiplier on ``query``. Default ``1/sqrt(K)``.
        use_qk_l2norm: L2-normalize q/k along the head dim (FLA default).

    Returns:
        output of shape ``[B, T, H, V]``.
    """
    return _gated_delta_rule_impl(
        query,
        key,
        value,
        decay,
        beta,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
    )


def _gated_delta_rule_python(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    decay: Tensor,
    beta: Tensor,
) -> Tensor:
    """Autograd-friendly recurrence. Fine for smoke / short sequences."""
    bsz, seq_len, n_heads, k_dim = q.shape
    v_dim = v.size(-1)
    state = q.new_zeros(bsz, n_heads, k_dim, v_dim)
    outs = []
    for t in range(seq_len):
        state = state * decay[:, t].exp().unsqueeze(-1).unsqueeze(-1)
        k_t = k[:, t]
        v_t = v[:, t]
        beta_t = beta[:, t].unsqueeze(-1)
        k_beta = k_t * beta_t
        kv_state = torch.einsum("bhkv,bhk->bhv", state, k_t)
        state = state - torch.einsum("bhk,bhv->bhkv", k_beta, kv_state)
        state = state + torch.einsum("bhk,bhv->bhkv", k_beta, v_t)
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, t], state))
    return torch.stack(outs, dim=1)
