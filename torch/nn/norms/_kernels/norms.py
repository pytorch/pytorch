"""CuteDSL norm kernels for RMSNorm/LayerNorm.

These functions adapt the CuTE DSL kernel interface to match the ATen op signatures
for ``_fused_rms_norm``, ``_fused_rms_norm_backward``, ``native_layer_norm``,
and ``native_layer_norm_backward``.
"""

from __future__ import annotations

import math

import torch

from ._rmsnorm_kernels import _get_sm_count, _rmsnorm_bwd, _rmsnorm_fwd


def _stat_shape(input: torch.Tensor, normalized_shape: list[int]) -> list[int]:
    """Compute the shape for mean/rstd stat tensors, matching the C++ convention."""
    axis = input.dim() - len(normalized_shape)
    shape: list[int] = []
    for i in range(input.dim()):
        shape.append(input.shape[i] if i < axis else 1)
    return shape


# ---------------------------------------------------------------------------
# RMSNorm forward
# ---------------------------------------------------------------------------


def cutedsl_rmsnorm_fwd(
    input: torch.Tensor,
    weight: torch.Tensor | None,
    normalized_shape: list[int],
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_shape = input.shape
    N = math.prod(normalized_shape)
    M = input.numel() // N
    x = input.reshape(M, N)

    out = torch.empty_like(x)
    rstd = torch.empty(M, device=x.device, dtype=torch.float32)

    _rmsnorm_fwd(x, weight, out, None, rstd, None, None, None, eps, False)

    out = out.reshape(input_shape)
    rstd = rstd.view(_stat_shape(input, normalized_shape))
    return out, rstd


# ---------------------------------------------------------------------------
# RMSNorm backward
# ---------------------------------------------------------------------------


def cutedsl_rmsnorm_bwd(
    grad_out: torch.Tensor,
    input: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor | None,
    normalized_shape: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    N = math.prod(normalized_shape)
    M = input.numel() // N
    x = input.reshape(M, N).contiguous()
    dout = grad_out.reshape(M, N).contiguous()
    rstd_flat = rstd.reshape(M).contiguous()

    dx = torch.empty_like(x)
    sm_count = _get_sm_count(N, x.device)
    dw_partial: torch.Tensor | None = None
    if weight is not None:
        dw_partial = torch.empty(sm_count, N, device=x.device, dtype=torch.float32)

    _rmsnorm_bwd(x, weight, dout, rstd_flat, dx, dw_partial, None, None, None, sm_count)

    dx = dx.reshape(input.shape)
    dw = (
        dw_partial.sum(dim=0).to(weight.dtype)  # pyrefly: ignore[missing-attribute]
        if weight is not None
        else torch.Tensor()
    )
    return dx, dw


# ---------------------------------------------------------------------------
# LayerNorm forward
# ---------------------------------------------------------------------------


def cutedsl_layernorm_fwd(
    input: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    normalized_shape: list[int],
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_shape = input.shape
    N = math.prod(normalized_shape)
    M = input.numel() // N
    x = input.reshape(M, N)

    out = torch.empty_like(x)
    rstd = torch.empty(M, device=x.device, dtype=torch.float32)
    mean = torch.empty(M, device=x.device, dtype=torch.float32)

    _rmsnorm_fwd(x, weight, out, bias, rstd, mean, None, None, eps, True)

    stat = _stat_shape(input, normalized_shape)
    out = out.reshape(input_shape)
    mean = mean.view(stat)
    rstd = rstd.view(stat)
    return out, mean, rstd


# ---------------------------------------------------------------------------
# LayerNorm backward
#
# The quack backward kernel implements RMSNorm backward only (no mean
# subtraction). For LayerNorm we fall back to a composite implementation.
# ---------------------------------------------------------------------------


def cutedsl_layernorm_bwd(
    grad_out: torch.Tensor,
    input: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    normalized_shape: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Composite backward: recompute x_hat, then derive gradients.
    N = math.prod(normalized_shape)
    M = input.numel() // N
    x = input.reshape(M, N).float()
    dout = grad_out.reshape(M, N).float()
    mean_flat = mean.reshape(M, 1)
    rstd_flat = rstd.reshape(M, 1)

    x_hat = (x - mean_flat) * rstd_flat

    if weight is not None:
        wdy = dout * weight.float()
    else:
        wdy = dout

    c1 = (x_hat * wdy).mean(dim=-1, keepdim=True)
    c2 = wdy.mean(dim=-1, keepdim=True)
    dx = (wdy - x_hat * c1 - c2) * rstd_flat
    dx = dx.to(input.dtype).reshape(input.shape)

    dw = (
        (dout * x_hat).sum(dim=0).to(weight.dtype)
        if weight is not None
        else torch.Tensor()
    )
    db = dout.sum(dim=0).to(bias.dtype) if bias is not None else torch.Tensor()
    return dx, dw, db
