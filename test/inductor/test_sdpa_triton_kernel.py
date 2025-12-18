# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Triton SDPA Kernel for ExecuTorch CUDA Backend.

This module provides a Triton-optimized implementation of scaled dot-product attention
that can replace the default ATen/Edge SDPA operator during graph transformation to allow
us export the model without decomposing the SDPA operator under libtorch free environment
and have better performance.
"""

import math
from typing import Optional

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


def _next_power_of_2(n: int) -> int:
    """Round up to the next power of 2."""
    if n <= 0:
        return 1
    if n & (n - 1) == 0:
        return n

    power = 1
    while power < n:
        power <<= 1
    return power


def _validate_qkv_shapes(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[int, int, int, int, int, int]:
    """
    Validate dimensions and return shape info.
    Args:
        query: Query tensor [B, H, L_q, D]
        key: Key tensor [B, H, L_kv, D]
        value: Value tensor [B, H, L_kv, D]
    Returns:
        Tuple of (B, H, L_q, L_kv, D_q, D_kv)
    Raises:
        RuntimeError: If dimensions are incompatible
    """
    B_q, H_q, L_q, D_q = query.shape
    B_k, H_k, L_kv_k, D_k = key.shape
    B_v, H_v, L_kv_v, D_v = value.shape
    # Validate batch and head dimensions
    if not (B_q == B_k == B_v):
        raise RuntimeError(
            f"Batch dimension must match; got B_q={B_q}, B_k={B_k}, B_v={B_v}."
        )

    if not (H_q == H_k == H_v):
        raise RuntimeError(
            f"Head dimension must match; got H_q={H_q}, H_k={H_k}, H_v={H_v}."
        )
    # Head dimension must match
    if not (D_q == D_k == D_v):
        raise RuntimeError(
            f"Head dimension must match across Q, K, V; got D_q={D_q}, D_k={D_k}, D_v={D_v}."
        )
    # Key and Value sequence lengths must match
    if L_kv_k != L_kv_v:
        raise RuntimeError(
            f"Key and Value must have the same sequence length; got L_k={L_kv_k}, L_v={L_kv_v}."
        )
    return B_q, H_q, L_q, L_kv_k, D_q, D_k


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_stages=1, num_warps=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_stages=1, num_warps=2),
    ],
    key=["L_Q", "L_KV", "HEAD_DIM"],
)
@triton.jit
def _sdpa_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    mask_ptr,
    o_ptr,
    B,
    H,
    L_Q,  # Query sequence length
    L_KV,  # Key/Value sequence length
    HEAD_DIM,  # Actual head dimension (may not be power of 2)
    stride_qb,
    stride_qh,
    stride_ql,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kl,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vl,
    stride_vd,
    stride_mb,
    stride_mh,
    stride_ml,
    stride_mn,
    stride_ob,
    stride_oh,
    stride_ol,
    stride_od,
    sm_scale,
    IS_CAUSAL: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM_CE: tl.constexpr,  # Rounded up for tl.arange
):
    """
    Fused SDPA kernel that handles different sequence lengths for Q and K/V.

    Q shape: [B, H, L_Q, D]
    K/V shape: [B, H, L_KV, D]
    Output shape: [B, H, L_Q, D]
    """
    # Program IDs
    pid_m = tl.program_id(axis=0)  # along query length
    pid_hz = tl.program_id(axis=1)  # flattened batch*head
    off_b = pid_hz // H
    off_h = pid_hz % H
    # Compute ranges for queries
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM_CE)
    mask_m = offs_m < L_Q  # Mask based on query length
    # Base pointers for this (b, h)
    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    k_base = k_ptr + off_b * stride_kb + off_h * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_h * stride_vh
    o_base = o_ptr + off_b * stride_ob + off_h * stride_oh
    # Mask base pointer (if provided)
    if HAS_MASK:
        mask_base = mask_ptr + off_b * stride_mb + off_h * stride_mh
    # Mask for actual head dimension (HEAD_DIM may not be power of 2)
    mask_d = offs_d < HEAD_DIM
    # Make head-dim addresses compiler-friendly
    offs_d_ctg = tl.max_contiguous(tl.multiple_of(offs_d, 16), HEAD_DIM_CE)
    # Load Q tile [BLOCK_M, HEAD_DIM] - coalesced along HEAD_DIM
    q_ptrs = q_base + (offs_m[:, None] * stride_ql + offs_d_ctg[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    q = q.to(tl.bfloat16)
    # Initialize accumulators and softmax stats
    acc = tl.zeros((BLOCK_M, HEAD_DIM_CE), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    # Convert to base-2 scale for exp2
    qk_scale = sm_scale * 1.4426950408889634
    # Loop over keys/values along L_KV dimension (not L_Q!)
    for start_n in tl.range(0, L_KV, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < L_KV  # Mask based on key/value length
        # Load K tile [BLOCK_N, HEAD_DIM] (contiguous along HEAD_DIM)
        k_ptrs = k_base + (
            offs_n[:, None] * stride_kl + offs_d_ctg[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        k = k.to(tl.bfloat16)
        # Compute attention logits [BLOCK_M, BLOCK_N] = Q[BM,D] @ K[BN,D]^T
        qk = tl.dot(q, tl.trans(k)).to(tl.float32)
        qk = qk * qk_scale
        # Apply causal mask if needed
        # For causal masking with different lengths: position i can attend to position j if i >= j
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(causal_mask, qk, -float("inf"))
        # Apply attention mask if provided
        if HAS_MASK:
            # Load mask tile [BLOCK_M, BLOCK_N]
            # Mask shape should be [B, H, L_Q, L_KV]
            mask_ptrs = mask_base + (
                offs_m[:, None] * stride_ml + offs_n[None, :] * stride_mn
            )
            attn_mask = tl.load(
                mask_ptrs,
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            )
            # Convert boolean mask to additive mask (-inf for False, 0 for True)
            qk = tl.where(attn_mask, qk, -float("inf"))
        # Apply OOB masks for both rows and cols
        qk = tl.where(mask_n[None, :], qk, -float("inf"))
        qk = tl.where(mask_m[:, None], qk, -float("inf"))
        # Online softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        # Load V tile [BLOCK_N, HEAD_DIM] (contiguous along HEAD_DIM)
        v_ptrs = v_base + (
            offs_n[:, None] * stride_vl + offs_d_ctg[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        v = v.to(tl.bfloat16)
        # Update accumulator
        acc = acc * alpha[:, None]
        p_bf16 = p.to(tl.bfloat16)
        acc = tl.dot(p_bf16, v, acc)
        # Update softmax stats
        l_i = l_i * alpha + l_ij
        m_i = m_ij
    # Normalize accumulator by softmax denominator
    acc = acc / l_i[:, None]
    # Store output [BLOCK_M, HEAD_DIM] - shape matches query
    o_ptrs = o_base + (offs_m[:, None] * stride_ol + offs_d_ctg[None, :] * stride_od)
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_d[None, :])


@triton_op("triton::sdpa", mutates_args={})
def sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float = 0.0,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """
    Triton fused Scaled Dot-Product Attention with support for different sequence lengths.

    Args:
        query: Query tensor with szie [B, H, L_q, D] and dtype torch.bfloat16
        key: Key tensor [B, H, L_kv, D] and dtype torch.bfloat16
        value: Value tensor [B, H, L_kv, D] and dtype torch.bfloat16
        attn_mask: Optional attention mask [B, H, L_q, L_kv] or
            broadcastable shape (2D: [L_q, L_kv] or 3D: [B, L_q, L_kv])
        dropout_p: must be 0.0 (others are not supported)
        is_causal: whether to apply causal masking
        scale: attention scale (default: 1/sqrt(D))
        enable_gqa: must be False (True is not supported)
    Returns:
        Output tensor [B, H, L_q, D] with dtype torch.bfloat16
    """
    # Validate inputs
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        raise RuntimeError("Q, K, V must be CUDA tensors.")
    if (
        query.dtype != torch.bfloat16
        or key.dtype != torch.bfloat16
        or value.dtype != torch.bfloat16
    ):
        raise RuntimeError("Expected bfloat16 inputs")
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise RuntimeError(
            f"Expected 4D tensors shaped [B, H, L, D]; got "
            f"query.dim()={query.dim()}, key.dim()={key.dim()}, "
            f"value.dim()={value.dim()}."
        )
    # Enforce unsupported features
    if dropout_p != 0.0:
        raise RuntimeError(
            "dropout_p must be 0.0 (not supported in this implementation)."
        )
    if enable_gqa is not False:
        raise RuntimeError(
            "enable_gqa must be False (not supported in this implementation)."
        )
    # Validate and get dimensions
    B, H, L_q, L_kv, D_q, D_kv = _validate_qkv_shapes(query, key, value)
    D = D_q  # Head dimension
    # Allocate output with query shape
    out = torch.empty_like(query)
    # Element-wise strides
    sqb, sqh, sql, sqd = query.stride()
    skb, skh, skl, skd = key.stride()
    svb, svh, svl, svd = value.stride()
    sob, soh, sol, sod = out.stride()

    # Grid: tile queries (M) and batch*heads axis
    def grid(META):
        return (
            triton.cdiv(L_q, META["BLOCK_M"]),  # Based on query length
            B * H,
        )

    # Scale factor for SDPA
    sm_scale = 1.0 / math.sqrt(D) if scale == 0.0 else scale
    # Handle attention mask
    has_mask = attn_mask is not None
    if has_mask:
        # Expand mask to [B, H, L_q, L_kv] if needed
        if attn_mask.dim() == 2:
            # [L_q, L_kv] -> [B, H, L_q, L_kv]
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
        elif attn_mask.dim() == 3:
            # [B, L_q, L_kv] -> [B, H, L_q, L_kv]
            attn_mask = attn_mask.unsqueeze(1).expand(-1, H, -1, -1)

        # Validate mask shape
        if attn_mask.shape != (B, H, L_q, L_kv):
            # Try to expand if broadcastable
            attn_mask = attn_mask.expand(B, H, L_q, L_kv)

        smb, smh, sml, smn = attn_mask.stride()
    else:
        # Dummy strides and mask
        smb, smh, sml, smn = 0, 0, 0, 0
        attn_mask = torch.empty(0, dtype=torch.bool, device=query.device)
    # Round up head dimension to next power of 2 for tile.arange in Triton kernel
    HEAD_DIM_CE = _next_power_of_2(D)
    # Launch kernel
    wrap_triton(_sdpa_fwd_kernel)[grid](
        query,
        key,
        value,
        attn_mask,
        out,
        B,
        H,
        L_q,  # Query sequence length
        L_kv,  # Key/Value sequence length
        D,  # Actual head dimension
        sqb,
        sqh,
        sql,
        sqd,
        skb,
        skh,
        skl,
        skd,
        svb,
        svh,
        svl,
        svd,
        smb,
        smh,
        sml,
        smn,
        sob,
        soh,
        sol,
        sod,
        sm_scale,
        IS_CAUSAL=is_causal,
        HAS_MASK=has_mask,
        HEAD_DIM_CE=HEAD_DIM_CE,  # Rounded to power of 2
    )
    return out


# Register the abstract/fake implementation for torch.export
# This is critical to avoid accessing real tensor data during export
@sdpa.register_fake
def _sdpa_abstract(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float = 0.0,
    enable_gq: bool = False,
) -> torch.Tensor:
    """
    Abstract/fake implementation for torch.export.
    This just returns an empty tensor with the correct shape/dtype/device.
    """
    # Validate dtypes match
    assert query.dtype == key.dtype == value.dtype, "Q, K, V must have the same dtype"
    # Validate kqv's shape and get the output shape
    B, H, L_q, _, D_q, _ = _validate_qkv_shapes(query, key, value)

    return torch.empty(B, H, L_q, D_q, dtype=query.dtype, device=query.device)
