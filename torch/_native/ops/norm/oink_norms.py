"""Adaptor for oink's 2-D RMSNorm kernel interface to match ATen op signatures.

These functions handle tensor reshaping, memory allocation, and call oink's
compiled kernels through the public ``rmsnorm_forward`` / ``rmsnorm_backward``
entry points.
"""

from __future__ import annotations

import math

import torch


def _reshape_2d(t: torch.Tensor, M: int, N: int) -> torch.Tensor:
    if t.ndim == 2 and t.shape[0] == M and t.shape[1] == N and t.is_contiguous():
        return t
    return t.reshape(M, N).contiguous()


def _flatten_rstd(t: torch.Tensor, M: int) -> torch.Tensor:
    if t.ndim == 1 and t.shape[0] == M:
        return t
    if t.is_contiguous() and t.numel() == M:
        return t.detach().view(M)
    return t.reshape(M).contiguous()


def oink_rmsnorm_fwd(
    input: torch.Tensor,
    weight: torch.Tensor | None,
    normalized_shape: list[int],
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # oink imports cutlass at module load; keep lazy so this file stays
    # importable on builds without the cuteDSL deps installed.
    from torch._vendor.oink.rmsnorm import rmsnorm_forward

    input_shape = input.shape
    N = math.prod(normalized_shape)
    M = input.numel() // N

    x = _reshape_2d(input, M, N)

    if weight is not None and weight.ndim != 1:
        weight = weight.reshape(N).contiguous()

    y, rstd, _ = rmsnorm_forward(
        x,
        weight=weight,
        bias=None,
        residual=None,
        eps=eps,
        store_rstd=True,
    )

    y = y.reshape(input_shape)
    stat_shape = list(input_shape[: -len(normalized_shape)]) + [1] * len(
        normalized_shape
    )
    rstd = rstd.view(stat_shape)
    return y, rstd


def oink_rmsnorm_bwd(
    grad_out: torch.Tensor,
    input: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor | None,
    normalized_shape: list[int],
    dw_mask: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    from torch._vendor.oink.rmsnorm import rmsnorm_backward

    N = math.prod(normalized_shape)
    M = input.numel() // N

    x = _reshape_2d(input, M, N)
    dout = _reshape_2d(grad_out, M, N)
    rstd_flat = _flatten_rstd(rstd, M)

    # oink folds dW gating into the `weight is not None` branch; pass None
    # for weight when the caller doesn't want dW so the kernel skips the
    # dw_partial allocation/reduction.
    w = weight if dw_mask else None

    dx, dw, _db, _dres = rmsnorm_backward(
        x,
        w,
        dout,
        rstd_flat,
        dresidual_out=None,
        has_bias=False,
        has_residual=False,
    )

    dx = dx.reshape(input.shape)
    if dw is not None:
        dw = dw.reshape(normalized_shape)
    return dx, dw
