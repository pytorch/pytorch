"""CuteDSL norm kernels for RMSNorm.

These functions adapt the CuTE DSL kernel interface to match the ATen op signatures
for ``_fused_rms_norm`` and ``_fused_rms_norm_backward``.
"""

from __future__ import annotations

import functools
import math

import torch


def _stat_shape(input: torch.Tensor, normalized_shape: list[int]) -> list[int]:
    """Compute the shape for mean/rstd stat tensors, matching the C++ convention."""
    axis = input.dim() - len(normalized_shape)
    shape: list[int] = []
    for i in range(input.dim()):
        shape.append(input.shape[i] if i < axis else 1)
    return shape


def cutedsl_rmsnorm_fwd(
    input: torch.Tensor,
    weight: torch.Tensor | None,
    normalized_shape: list[int],
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from ._rmsnorm_kernels import _rmsnorm_fwd

    input_shape = input.shape
    N = math.prod(normalized_shape)
    M = input.numel() // N
    x = input.reshape(M, N)

    out = torch.empty_like(x)
    rstd = torch.empty(M, device=x.device, dtype=torch.float32)

    _rmsnorm_fwd(x, weight, out, None, rstd, None, None, eps)

    out = out.reshape(input_shape)
    # Return rstd flat — the backward only needs the raw data, and both
    # _fused_rms_norm_backward and the higher-order-grad path adapt to
    # any rstd shape. Avoiding the stat_shape view here saves a reshape
    # dispatch in the backward.
    return out, rstd


@functools.cache
def _get_semaphore(device: torch.device) -> torch.Tensor:
    return torch.zeros(1, device=device, dtype=torch.int32)


def _reshape_2d(t: torch.Tensor, M: int, N: int) -> torch.Tensor:
    if t.ndim == 2 and t.shape[0] == M and t.shape[1] == N and t.is_contiguous():
        return t
    return t.reshape(M, N).contiguous()


def _flatten_rstd(t: torch.Tensor, M: int) -> torch.Tensor:
    if t.ndim == 1 and t.shape[0] == M:
        return t
    # rstd arrives as stat_shape, e.g. (M, 1) with stride (1, 1).
    # The underlying storage is already M contiguous float32 values,
    # so we can reinterpret it as 1-D without any C++ reshape dispatch.
    if t.is_contiguous() and t.numel() == M:
        return t.detach().view(M)
    return t.reshape(M).contiguous()


def cutedsl_rmsnorm_bwd(
    grad_out: torch.Tensor,
    input: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor | None,
    normalized_shape: list[int],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    from ._rmsnorm_kernels import _get_sm_count, _rmsnorm_bwd

    N = math.prod(normalized_shape)
    M = input.numel() // N
    x = _reshape_2d(input, M, N)
    dout = _reshape_2d(grad_out, M, N)
    rstd_flat = _flatten_rstd(rstd, M)

    dx = torch.empty_like(x)
    sm_count = _get_sm_count(N, x.device)
    dw_partial: torch.Tensor | None = None
    dw: torch.Tensor | None = None
    semaphore: torch.Tensor | None = None
    # In-kernel cross-CTA dw reduction is only supported for cluster_n == 1
    # (N <= 8192). For larger N the kernel ignores dw/semaphore and we fall
    # back to a host-side reduction of dw_partial.
    use_in_kernel_dw_reduction = N <= 8192
    if weight is not None:
        dw_partial = torch.empty(sm_count, N, device=x.device, dtype=torch.float32)
        if use_in_kernel_dw_reduction:
            dw = torch.empty(N, device=x.device, dtype=weight.dtype)
            semaphore = _get_semaphore(x.device)

    _rmsnorm_bwd(
        x, weight, dout, rstd_flat, dx, dw_partial,
        None, None, None, sm_count, dw, semaphore,
    )

    dx = dx.reshape(input.shape)
    if weight is not None and not use_in_kernel_dw_reduction:
        dw = dw_partial.sum(dim=0, dtype=weight.dtype)  # pyrefly: ignore[missing-attribute]
    return dx, dw
