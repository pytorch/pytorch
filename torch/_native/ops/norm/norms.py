"""Adaptor for quack's 2-D RMSNorm kernel interface to match ATen op signatures.

These functions handle tensor reshaping, memory allocation, and call quack's
compiled kernels directly (bypassing the ``@torch.library.custom_op`` wrapper
to avoid dispatcher overhead).
"""

from __future__ import annotations

import importlib
import math
from functools import cache

import torch


@cache
def _quack_rmsnorm():  # type: ignore[no-untyped-def]
    return importlib.import_module("quack.rmsnorm")


@cache
def _quack_cute_dsl_utils():  # type: ignore[no-untyped-def]
    return importlib.import_module("quack.cute_dsl_utils")


def _torch2cute(t: torch.Tensor | None):  # type: ignore[no-untyped-def]
    if t is None:
        return None
    return _quack_cute_dsl_utils().torch2cute_dtype_map[t.dtype]


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


def quack_rmsnorm_fwd(
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

    if weight is not None and weight.ndim != 1:
        weight = weight.view(N)

    dtype = _torch2cute(x)
    out_dtype = _torch2cute(out)
    weight_dtype = _torch2cute(weight)

    kernel = _quack_rmsnorm()._compile_rmsnorm_fwd(
        dtype,
        out_dtype,
        None,
        weight_dtype,
        None,
        None,
        N,
        True,
        False,
        False,
    )
    # compile order: (x, weight, bias, res, out, res_out, rstd, mean, eps)
    kernel(x, weight, None, None, out, None, rstd, None, eps)

    out = out.reshape(input_shape)
    stat_shape = list(input_shape[: -len(normalized_shape)]) + [1] * len(
        normalized_shape
    )
    rstd = rstd.view(stat_shape)
    return out, rstd


def quack_rmsnorm_bwd(
    grad_out: torch.Tensor,
    input: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor | None,
    normalized_shape: list[int],
    dw_mask: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    mod = _quack_rmsnorm()

    N = math.prod(normalized_shape)
    M = input.numel() // N
    x = _reshape_2d(input, M, N)
    dout = _reshape_2d(grad_out, M, N)
    rstd_flat = _flatten_rstd(rstd, M)

    dx = torch.empty_like(x)
    sm_count = mod._get_sm_count(N, x.device)
    dw_partial: torch.Tensor | None = None
    if weight is not None and dw_mask:
        dw_partial = torch.empty(sm_count, N, device=x.device, dtype=torch.float32)

    dtype = _torch2cute(x)
    dout_dtype = _torch2cute(dout)
    dx_dtype = _torch2cute(dx)
    weight_dtype = _torch2cute(weight) if dw_mask else None

    kernel = mod._compile_rmsnorm_bwd(
        N,
        dtype,
        dout_dtype,
        dx_dtype,
        weight_dtype,
        False,
        None,
        None,
        dw_partial is not None,
    )
    # compile order: (x, weight, dout, dres_out, rstd, dx, dw_partial, dres, db_partial, sm_count)
    w = weight if dw_mask else None
    kernel(x, w, dout, None, rstd_flat, dx, dw_partial, None, None, sm_count)

    dx = dx.reshape(input.shape)
    dw = (
        dw_partial.sum(dim=0, dtype=weight.dtype)  # pyrefly: ignore[missing-attribute]
        if dw_partial is not None
        else None
    )
    return dx, dw
