"""Adaptor for quack's 2-D RMSNorm kernel interface to match ATen op signatures.

These functions handle tensor reshaping, memory allocation, and call quack's
compiled kernels directly (bypassing the ``@torch.library.custom_op`` wrapper
to avoid dispatcher overhead).
"""

from __future__ import annotations

import math

import torch


# quack imports `cutlass`, which is only installed on CUDA-enabled x86 Linux
# builds. Keep the import lazy so `torch._native.ops.norm.norms` stays
# importable on CPU-only builds (test_public_bindings /
# test_circular_dependencies walk every module under `torch.`).
def _torch2cute(t: torch.Tensor | None):  # type: ignore[no-untyped-def]
    from torch._vendor.quack import cute_dsl_utils as _cute_dsl_utils

    if t is None:
        return None
    return _cute_dsl_utils.torch2cute_dtype_map[t.dtype]


def _required_align_bytes(t: torch.Tensor, N: int) -> int:
    # quack compiles the kernel assuming the input base pointer is aligned to
    # the vectorized cp.async width: gcd(N, 128 // dtype_bits) elements. A
    # contiguous tensor with a non-zero storage offset (e.g. buf[1:].view(M, N))
    # has fine strides but a misaligned base, which trips CuTe's runtime
    # alignment check ("Misaligned Tensor data on argument #0").
    itemsize = t.element_size()
    return math.gcd(N, 128 // (itemsize * 8)) * itemsize


def _reshape_2d(t: torch.Tensor, M: int, N: int) -> torch.Tensor:
    align = _required_align_bytes(t, N)
    if t.ndim == 2 and t.shape[0] == M and t.shape[1] == N and t.is_contiguous():
        # .contiguous() is a no-op on an already-contiguous tensor and would
        # preserve a misaligned base; clone to force a fresh (aligned) buffer.
        return t if t.data_ptr() % align == 0 else t.clone()
    out = t.reshape(M, N).contiguous()
    # reshape can return a view that keeps the original (misaligned) storage
    # offset, and .contiguous() then leaves it untouched.
    return out if out.data_ptr() % align == 0 else out.clone()


def _aligned_weight(w: torch.Tensor, N: int) -> torch.Tensor:
    # Same trap as _reshape_2d: reshape(N).contiguous() is a no-op for a
    # contiguous-but-offset weight, leaving a misaligned base. Weight is only
    # N elements, so always clone rather than perf-gating like the input.
    w = w.reshape(N).contiguous()
    if w.data_ptr() % _required_align_bytes(w, N) != 0:
        w = w.clone()
    return w


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
    # quack's kernel requires a contiguous 2-D input with trailing stride 1;
    # a plain `.reshape(M, N)` can return a non-contiguous view.
    x = _reshape_2d(input, M, N)

    out = torch.empty_like(x)
    rstd = torch.empty(M, device=x.device, dtype=torch.float32)

    if weight is not None:
        weight = _aligned_weight(weight, N)

    dtype = _torch2cute(x)
    out_dtype = _torch2cute(out)
    weight_dtype = _torch2cute(weight)

    from torch._vendor.quack.rmsnorm import _compile_rmsnorm_fwd

    kernel = _compile_rmsnorm_fwd(
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
        per_head=False,
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
    N = math.prod(normalized_shape)
    M = input.numel() // N
    x = _reshape_2d(input, M, N)
    dout = _reshape_2d(grad_out, M, N)
    rstd_flat = _flatten_rstd(rstd, M)

    dx = torch.empty_like(x)
    from torch._vendor.quack.rmsnorm import _compile_rmsnorm_bwd
    from torch._vendor.quack.rmsnorm_config import get_sm_count

    sm_count = get_sm_count(N, x.device)

    # quack's kernel requires a contiguous 1-D weight with matching dtype.
    if weight is not None:
        weight = _aligned_weight(weight, N)

    # quack treats `weight_dtype` (whether weight is present in the kernel)
    # and `has_dw_partial` (whether to accumulate dw) as independent flags.
    # dx's formula depends on weight, so we always hand weight to the kernel
    # when the user supplied one; dw_mask only gates the dw_partial buffer.
    dw_partial: torch.Tensor | None = None
    if weight is not None and dw_mask:
        dw_partial = torch.empty(sm_count, N, device=x.device, dtype=torch.float32)

    dtype = _torch2cute(x)
    dout_dtype = _torch2cute(dout)
    dx_dtype = _torch2cute(dx)
    weight_dtype = _torch2cute(weight)

    kernel = _compile_rmsnorm_bwd(
        N,
        dtype,
        dout_dtype,
        dx_dtype,
        weight_dtype,
        False,
        None,
        None,
        dw_partial is not None,
        per_head=False,
    )
    # compile order: (x, weight, dout, dres_out, rstd, dx, dw_partial, dres, db_partial, sm_count)
    kernel(x, weight, dout, None, rstd_flat, dx, dw_partial, None, None, sm_count)

    dx = dx.reshape(input.shape)
    # `dw_partial is not None` implies `weight is not None` (invariant from
    # allocation above), but pyrefly can't see the dependency.
    if dw_partial is not None:
        weight_dtype_for_sum = weight.dtype  # pyrefly: ignore[missing-attribute]
        dw = dw_partial.sum(dim=0).reshape(normalized_shape).to(weight_dtype_for_sum)
    else:
        dw = None
    return dx, dw
