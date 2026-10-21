"""Adaptor for quack's 2-D RMSNorm kernel interface to match ATen op signatures.

These functions handle tensor reshaping, memory allocation, and call quack's
compiled kernels directly (bypassing the ``@torch.library.custom_op`` wrapper
to avoid dispatcher overhead).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch._native.cutedsl.hw_caps import caps
from torch._native.utils.lazy import LazyModule
from torch._native.utils.tensor import reshape_contiguous, row_alignment

from .rmsnorm_launch import backward_launch, NORMALIZED_SIZES


if TYPE_CHECKING:
    from torch._native.cutedsl import launch

    from . import rmsnorm_kernels as kernels
else:
    launch = LazyModule("torch._native.cutedsl.launch")
    kernels = LazyModule("torch._native.ops.norm.rmsnorm_kernels")


def quack_rmsnorm_fwd(
    input: torch.Tensor,
    weight: torch.Tensor | None,
    normalized_shape: list[int],
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_shape = input.shape
    N = math.prod(normalized_shape)
    M = input.numel() // N
    x = reshape_contiguous(
        input, (M, N), alignment=row_alignment(N, input.element_size())
    )
    out = torch.empty_like(x)
    if weight is not None:
        weight = reshape_contiguous(
            weight, (N,), alignment=row_alignment(N, weight.element_size())
        )
    rstd = torch.empty((M, 1), device=x.device, dtype=torch.float32)
    hw = caps(x.device.index)
    arch = hw.cc
    kernel = kernels.compile_rmsnorm("forward", x.dtype, N, weight is not None, arch)
    kernel(
        launch.read_only(x),
        launch.read_only(weight),
        out,
        rstd,
        M,
        eps,
        launch.stream(x.device.index),
    )
    stat_shape = list(input_shape[: -len(normalized_shape)]) + [1] * len(
        normalized_shape
    )
    return (
        out if out.shape == input_shape else out.view(input_shape),
        rstd if rstd.shape == tuple(stat_shape) else rstd.view(stat_shape),
    )


def quack_rmsnorm_bwd(
    grad_out: torch.Tensor,
    input: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor | None,
    normalized_shape: list[int],
    dw_mask: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    N = math.prod(normalized_shape)
    M = input.numel() // N
    x = reshape_contiguous(
        input, (M, N), alignment=row_alignment(N, input.element_size())
    )
    dout = reshape_contiguous(
        grad_out, (M, N), alignment=row_alignment(N, grad_out.element_size())
    )
    if weight is not None:
        weight = reshape_contiguous(
            weight, (N,), alignment=row_alignment(N, weight.element_size())
        )
    dx = torch.empty_like(x)
    rstd_2d = reshape_contiguous(rstd, (M, 1))
    compute_dw = weight is not None and dw_mask
    hw = caps(x.device.index)
    arch = hw.cc
    if N in NORMALIZED_SIZES and arch in ((9, 0), (10, 0)) and dout.dtype == x.dtype:
        blocks = backward_launch(N, compute_dw, x.element_size()).blocks(M, hw.sm_count)
    else:
        from torch._vendor.quack.rmsnorm_config import get_sm_count

        blocks = min(M, get_sm_count(N, x.device))
    partial = (
        torch.empty(blocks, N, device=x.device, dtype=torch.float32)
        if compute_dw
        else None
    )
    dw = torch.empty(N, device=x.device, dtype=x.dtype) if compute_dw else None
    kernel = kernels.compile_rmsnorm(
        "backward",
        x.dtype,
        N,
        weight is not None,
        arch,
        compute_dw,
        dout.dtype,
    )
    kernel(
        launch.read_only(x),
        launch.read_only(weight),
        launch.read_only(dout),
        launch.read_only(rstd_2d),
        dx,
        partial,
        dw,
        M,
        blocks,
        launch.stream(x.device.index),
    )
    if dx.shape != input.shape:
        dx = dx.view(input.shape)
    if dw is not None and len(normalized_shape) != 1:
        dw = dw.view(normalized_shape)
    return dx, dw
