# mypy: allow-untyped-defs
"""Helion kernels of flex_attention's BACKEND="HELION".

The forward and backward follow the envelope Helion's CuTe backend lowers to its
fused tcgen05 flash kernels (``helion/_compiler/cute/flex_match.py`` and
``flex_bwd_match.py``): a ``[B, H, M]`` (forward) or ``[B, Hkv, N]`` (backward)
tile grid walking the BlockMask's full and partial block lists, with
``score_mod`` / ``mask_mod`` (and the backward's ``score_mod_bwd``) called on
broadcast coordinate vectors. Inductor passes the mods as FX GraphModules, which
Helion inlines at trace time; the tensors they capture arrive in
``score_mod_buffers`` / ``mask_mod_buffers`` and are forwarded to them as
trailing arguments. The forward stores the logsumexp in base 2, like Inductor's
Triton flex_attention kernels.

This module imports helion; only BACKEND="HELION" imports it.
"""

from __future__ import annotations

import functools
import json
from typing import Any, TYPE_CHECKING

import helion  # type: ignore[import-not-found]
import helion.language as hl  # type: ignore[import-not-found]

import torch


if TYPE_CHECKING:
    from collections.abc import Callable


def flex_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    block_mask_kv_num_blocks: torch.Tensor,
    block_mask_kv_indices: torch.Tensor,
    block_mask_full_kv_num_blocks: torch.Tensor,
    block_mask_full_kv_indices: torch.Tensor,
    block_mask_mask_mod: Callable,
    block_mask_m: int,
    block_mask_n: int,
    scale: float,
    score_mod_buffers: tuple[torch.Tensor, ...],
    mask_mod_buffers: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    B, H, M, D = query.size()
    D = hl.specialize(D)
    # The V/O head dim may differ from the Q/K head dim.
    Dv = hl.specialize(value.size(-1))
    _, Hk, N, _ = key.size()
    num_groups = H // Hk
    # The BlockMask block sizes are compile-time constants of the walk.
    block_mask_m = hl.specialize(block_mask_m)
    block_mask_n = hl.specialize(block_mask_n)
    # BlockMask tables may broadcast over batch and/or heads (size-1 dims).
    table_batch, table_heads, _ = block_mask_kv_num_blocks.size()
    if Dv == D:
        out = torch.empty_like(query)
    else:
        out = torch.empty((B, H, M, Dv), dtype=query.dtype, device=query.device)
    lse = torch.empty((B, H, M), dtype=torch.float32, device=out.device)
    log_2_e = 1.44269504
    block_m = hl.register_block_size(min(256, block_mask_m))
    block_n = hl.register_block_size(min(256, block_mask_n))
    for tile_b, tile_h, tile_m in hl.tile([B, H, M], block_size=[1, 1, block_m]):
        m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_m, Dv], dtype=torch.float32)
        q_i = query[tile_b.begin, tile_h.begin, tile_m, :]

        sparse_row = tile_m.begin // block_mask_m
        b_idx = tile_b.begin
        h_idx = tile_h.begin
        h_kv_idx = h_idx // num_groups
        tb_idx = b_idx % table_batch
        th_idx = h_idx % table_heads
        # One-element batch / head vectors broadcast against every score tile.
        bcast_b = (tile_b.begin + hl.arange(1))[:, None]
        bcast_h = (tile_h.begin + hl.arange(1))[:, None]
        bcast_m = (tile_m.begin + hl.arange(tile_m.block_size))[:, None]

        # Full blocks: no mask_mod.
        sparse_num_blocks = block_mask_full_kv_num_blocks[tb_idx, th_idx, sparse_row]
        for block_idx in hl.tile(sparse_num_blocks, block_size=1):
            start_n = (
                block_mask_full_kv_indices[tb_idx, th_idx, sparse_row, block_idx.id]
                * block_mask_n
            )
            end_n = start_n + block_mask_n
            end_N = end_n.new_full([], N)
            end_n = torch.minimum(end_n, end_N)

            for tile_n in hl.tile(start_n, end_n, block_size=block_n):
                k = key[b_idx, h_kv_idx, tile_n, :]
                bcast_n = (tile_n.begin + hl.arange(tile_n.block_size))[None, :]
                qk = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                qk = hl.dot(q_i, k.T, acc=qk)
                qk = qk * scale
                qk = score_mod(
                    qk, bcast_b, bcast_h, bcast_m, bcast_n, *score_mod_buffers
                )
                qk *= log_2_e
                m_ij = torch.maximum(m_i, torch.amax(qk, -1))
                # Rescale a row whose scores are all masked so far against 0,
                # so -inf - -inf never produces NaN.
                m_safe = torch.where(m_ij == float("-inf"), 0.0, m_ij)
                qk = qk - m_safe[:, None]
                p = torch.exp2(qk)
                l_ij = torch.sum(p, -1)
                alpha = torch.exp2(m_i - m_safe)
                m_i = m_ij
                l_i = l_i * alpha + l_ij
                acc = acc * alpha[:, None]
                v = value[b_idx, h_kv_idx, tile_n, :]
                p = p.to(v.dtype)
                acc = hl.dot(p, v, acc=acc)

        # Partial blocks: mask_mod applied.
        sparse_num_blocks = block_mask_kv_num_blocks[tb_idx, th_idx, sparse_row]
        for block_idx in hl.tile(sparse_num_blocks, block_size=1):
            start_n = (
                block_mask_kv_indices[tb_idx, th_idx, sparse_row, block_idx.id]
                * block_mask_n
            )
            end_n = start_n + block_mask_n
            end_N = end_n.new_full([], N)
            end_n = torch.minimum(end_n, end_N)

            for tile_n in hl.tile(start_n, end_n, block_size=block_n):
                k = key[b_idx, h_kv_idx, tile_n, :]
                bcast_n = (tile_n.begin + hl.arange(tile_n.block_size))[None, :]
                qk = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                qk = hl.dot(q_i, k.T, acc=qk)
                qk = qk * scale
                score = score_mod(
                    qk, bcast_b, bcast_h, bcast_m, bcast_n, *score_mod_buffers
                )
                mask = block_mask_mask_mod(
                    bcast_b, bcast_h, bcast_m, bcast_n, *mask_mod_buffers
                )
                qk = torch.where(mask, score, -float("inf"))
                qk *= log_2_e
                m_ij = torch.maximum(m_i, torch.amax(qk, -1))
                m_safe = torch.where(m_ij == float("-inf"), 0.0, m_ij)
                qk = qk - m_safe[:, None]
                p = torch.exp2(qk)
                l_ij = torch.sum(p, -1)
                alpha = torch.exp2(m_i - m_safe)
                m_i = m_ij
                l_i = l_i * alpha + l_ij
                acc = acc * alpha[:, None]
                v = value[b_idx, h_kv_idx, tile_n, :]
                p = p.to(v.dtype)
                acc = hl.dot(p, v, acc=acc)

        # Rows with no visible key: output 0 and logsumexp -inf.
        l_i = torch.where(l_i == 0.0, 1.0, l_i)
        lse_row = m_i + torch.log2(l_i)
        acc = acc / l_i[:, None]
        out[tile_b, tile_h, tile_m, :] = acc[None, None, :, :].to(out.dtype)
        lse[tile_b, tile_h, tile_m] = lse_row[None, None, :]
    return out, lse


def flex_attention_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    grad_out: torch.Tensor,
    lse2: torch.Tensor,
    delta: torch.Tensor,
    score_mod: Callable,
    score_mod_bwd: Callable,
    block_mask_q_num_blocks: torch.Tensor,
    block_mask_q_indices: torch.Tensor,
    block_mask_full_q_num_blocks: torch.Tensor,
    block_mask_full_q_indices: torch.Tensor,
    block_mask_mask_mod: Callable,
    block_mask_m: int,
    block_mask_n: int,
    scale: float,
    score_mod_buffers: tuple[torch.Tensor, ...],
    mask_mod_buffers: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """dq (fp32), dk and dv. ``lse2`` is the forward's base-2 logsumexp with
    no-visible-key rows mapped to 0, ``delta`` the ``rowsum(out * grad_out)``
    terms. One work item per KV tile of one (batch, kv head): dK / dV
    accumulate over the Q blocks the BlockMask lists for that KV block, for
    every q head of the GQA group; dQ accumulates with atomics.
    ``score_mod_bwd(grad, score, b, h, q_idx, kv_idx, *buffers)`` maps the
    gradient w.r.t. the modified score back to the raw one."""
    B, H, M, D = query.size()
    D = hl.specialize(D)
    _, Hk, N, _ = key.size()
    Dv = hl.specialize(value.size(-1))
    num_groups = H // Hk
    block_mask_m = hl.specialize(block_mask_m)
    block_mask_n = hl.specialize(block_mask_n)
    table_batch, table_heads, _ = block_mask_q_num_blocks.size()
    dq = torch.zeros((B, H, M, D), dtype=torch.float32, device=query.device)
    dk = torch.empty_like(key)
    dv = torch.empty_like(value)
    log_2_e = 1.44269504
    block_m = hl.register_block_size(min(256, block_mask_m))
    block_n = hl.register_block_size(min(256, block_mask_n))
    for tile_b, tile_hk, tile_n in hl.tile([B, Hk, N], block_size=[1, 1, block_n]):
        k_j = key[tile_b.begin, tile_hk.begin, tile_n, :]
        v_j = value[tile_b.begin, tile_hk.begin, tile_n, :]
        dk_acc = hl.zeros([tile_n, D], dtype=torch.float32)
        dv_acc = hl.zeros([tile_n, Dv], dtype=torch.float32)
        sparse_col = tile_n.begin // block_mask_n
        b_idx = tile_b.begin
        hk_idx = tile_hk.begin
        tb_idx = b_idx % table_batch
        bcast_b = (tile_b.begin + hl.arange(1))[:, None]
        bcast_n = (tile_n.begin + hl.arange(tile_n.block_size))[:, None]
        # The q heads sharing this KV head (one for MHA).
        for tile_h in hl.tile(
            hk_idx * num_groups, hk_idx * num_groups + num_groups, block_size=1
        ):
            h_idx = tile_h.begin
            th_idx = h_idx % table_heads
            bcast_h = (tile_h.begin + hl.arange(1))[:, None]

            # Q blocks that need no mask.
            full_num_blocks = block_mask_full_q_num_blocks[tb_idx, th_idx, sparse_col]
            for block_idx in hl.tile(full_num_blocks, block_size=1):
                start_m = (
                    block_mask_full_q_indices[tb_idx, th_idx, sparse_col, block_idx.id]
                    * block_mask_m
                )
                end_m = start_m + block_mask_m
                end_M = end_m.new_full([], M)
                end_m = torch.minimum(end_m, end_M)
                for tile_m in hl.tile(start_m, end_m, block_size=block_m):
                    q_i = query[b_idx, h_idx, tile_m, :]
                    do_i = grad_out[b_idx, h_idx, tile_m, :]
                    lse2_i = lse2[b_idx, h_idx, tile_m]
                    delta_i = delta[b_idx, h_idx, tile_m]
                    bcast_m = (tile_m.begin + hl.arange(tile_m.block_size))[None, :]
                    s_t = hl.dot(k_j, q_i.T, out_dtype=torch.float32)
                    s_t = s_t * scale
                    score = score_mod(
                        s_t, bcast_b, bcast_h, bcast_m, bcast_n, *score_mod_buffers
                    )
                    p_t = torch.exp2(score * log_2_e - lse2_i[None, :])
                    dp_t = hl.dot(v_j, do_i.T, out_dtype=torch.float32)
                    dv_acc = hl.dot(p_t.to(value.dtype), do_i, acc=dv_acc)
                    ds_t = p_t * (dp_t - delta_i[None, :])
                    ds_t = score_mod_bwd(
                        ds_t,
                        s_t,
                        bcast_b,
                        bcast_h,
                        bcast_m,
                        bcast_n,
                        *score_mod_buffers,
                    )
                    ds_c = ds_t.to(query.dtype)
                    dq_i = hl.dot(ds_c.T, k_j, out_dtype=torch.float32)
                    hl.atomic_add(dq, [b_idx, h_idx, tile_m, slice(None)], dq_i * scale)
                    dk_acc = hl.dot(ds_c, q_i, acc=dk_acc)

            # Q blocks with the mask applied.
            num_blocks = block_mask_q_num_blocks[tb_idx, th_idx, sparse_col]
            for block_idx in hl.tile(num_blocks, block_size=1):
                start_m = (
                    block_mask_q_indices[tb_idx, th_idx, sparse_col, block_idx.id]
                    * block_mask_m
                )
                end_m = start_m + block_mask_m
                end_M = end_m.new_full([], M)
                end_m = torch.minimum(end_m, end_M)
                for tile_m in hl.tile(start_m, end_m, block_size=block_m):
                    q_i = query[b_idx, h_idx, tile_m, :]
                    do_i = grad_out[b_idx, h_idx, tile_m, :]
                    lse2_i = lse2[b_idx, h_idx, tile_m]
                    delta_i = delta[b_idx, h_idx, tile_m]
                    bcast_m = (tile_m.begin + hl.arange(tile_m.block_size))[None, :]
                    s_t = hl.dot(k_j, q_i.T, out_dtype=torch.float32)
                    s_t = s_t * scale
                    score = score_mod(
                        s_t, bcast_b, bcast_h, bcast_m, bcast_n, *score_mod_buffers
                    )
                    mask = block_mask_mask_mod(
                        bcast_b, bcast_h, bcast_m, bcast_n, *mask_mod_buffers
                    )
                    score = torch.where(mask, score, -float("inf"))
                    p_t = torch.exp2(score * log_2_e - lse2_i[None, :])
                    dp_t = hl.dot(v_j, do_i.T, out_dtype=torch.float32)
                    dv_acc = hl.dot(p_t.to(value.dtype), do_i, acc=dv_acc)
                    ds_t = p_t * (dp_t - delta_i[None, :])
                    ds_t = score_mod_bwd(
                        ds_t,
                        s_t,
                        bcast_b,
                        bcast_h,
                        bcast_m,
                        bcast_n,
                        *score_mod_buffers,
                    )
                    ds_c = ds_t.to(query.dtype)
                    dq_i = hl.dot(ds_c.T, k_j, out_dtype=torch.float32)
                    hl.atomic_add(dq, [b_idx, h_idx, tile_m, slice(None)], dq_i * scale)
                    dk_acc = hl.dot(ds_c, q_i, acc=dk_acc)

        dv[tile_b, tile_hk, tile_n, :] = dv_acc[None, None, :, :].to(dv.dtype)
        dk[tile_b, tile_hk, tile_n, :] = (dk_acc * scale)[None, None, :, :].to(dk.dtype)
    return dq, dk, dv


def _accuracy_check(actual: object, expected: object) -> None:
    """Autotuning accuracy check: every element's difference is bounded by 2%
    of the output's peak (bf16 outputs carry rounding error proportional to
    each output's peak, and gradient tensors are heavy-tailed, so the default
    absolute tolerance rejects correct configs)."""
    for got, ref in zip(actual, expected, strict=True):  # type: ignore[call-overload]
        ref = ref.float()
        peak = ref.abs().max().item()
        torch.testing.assert_close(got.float(), ref, atol=2e-2 * peak, rtol=0.0)


_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "forward": flex_attention_forward,
    "backward": flex_attention_backward,
}


@functools.cache
def _kernel(direction: str, autotune_effort: str, config_json: str | None):
    config = None if config_json is None else helion.Config(**json.loads(config_json))
    return helion.kernel(
        _FUNCTIONS[direction],
        backend="cute",
        static_shapes=True,
        # fast_math: the CuTe flash kernels' polynomial exp2 packets and
        # tanh.approx (the same approximation class as FlashAttention-4).
        fast_math=True,
        autotune_effort=autotune_effort,
        config=config,
        autotune_baseline_accuracy_check_fn=_accuracy_check,
    )


def get_kernel(
    direction: str,
    *,
    autotune_effort: str = "none",
    config: dict[str, Any] | None = None,
):
    """The ``direction`` ("forward" / "backward") kernel: with ``config``, a
    kernel fixed to it; otherwise one that autotunes at ``autotune_effort``
    ("none" takes Helion's default config)."""
    config_json = None if config is None else json.dumps(config, sort_keys=True)
    return _kernel(direction, autotune_effort, config_json)
