"""CuTeDSL RMSNorm overrides for aten fused RMSNorm operators."""
# mypy: allow-untyped-defs

from __future__ import annotations

import functools
import logging
from collections.abc import Callable

import torch

from ... import cutedsl_utils as cu


log = logging.getLogger(__name__)

_RMSNormFwdFallback = Callable[
    [torch.DispatchKeySet, torch.Tensor, list[int], torch.Tensor | None, float | None],
    tuple[torch.Tensor, torch.Tensor],
]
_RMSNormBwdFallback = Callable[
    [
        torch.DispatchKeySet,
        torch.Tensor,
        torch.Tensor,
        list[int],
        torch.Tensor,
        torch.Tensor | None,
        list[bool],
    ],
    tuple[torch.Tensor | None, torch.Tensor | None],
]


@functools.cache
def _get_device_major(device: torch.device) -> int:
    major, _ = torch.cuda.get_device_capability(device)
    return major


@functools.cache
def _get_rmsnorm_kernels():
    from .norms import cutedsl_rmsnorm_bwd, cutedsl_rmsnorm_fwd

    return cutedsl_rmsnorm_fwd, cutedsl_rmsnorm_bwd


def _collect_tensors(*tensors: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
    return tuple(t for t in tensors if t is not None)


def _support_error(
    input: torch.Tensor,
    tensors: tuple[torch.Tensor, ...],
    name: str,
) -> str | None:
    if not all(t.is_cuda for t in tensors):
        return "inputs must be CUDA tensors"
    if len({t.device for t in tensors}) != 1:
        return "inputs must share device"
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return "input dtype must be float16, bfloat16, or float32"
    if not torch.cuda.is_available():
        return "CUDA not available"
    if _get_device_major(input.device) not in (9, 10):
        return f"CuTeDSL {name} requires compute capability 9.0 or 10.0"
    return None


def _cutedsl_fused_rms_norm_impl(
    dispatch_keys: torch.DispatchKeySet,
    input: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor | None,
    eps: float | None,
    *,
    fallback_kernel: _RMSNormFwdFallback,
) -> tuple[torch.Tensor, torch.Tensor]:
    error = _support_error(input, _collect_tensors(input, weight), "RMSNorm")
    if error is not None:
        return fallback_kernel(dispatch_keys, input, normalized_shape, weight, eps)

    if eps is None:
        eps = torch.finfo(input.dtype).eps

    cutedsl_rmsnorm_fwd, _ = _get_rmsnorm_kernels()
    return cutedsl_rmsnorm_fwd(input, weight, normalized_shape, eps)


def _cutedsl_fused_rms_norm_backward_impl(
    dispatch_keys: torch.DispatchKeySet,
    grad_out: torch.Tensor,
    input: torch.Tensor,
    normalized_shape: list[int],
    rstd: torch.Tensor,
    weight: torch.Tensor | None,
    output_mask: list[bool],
    *,
    fallback_kernel: _RMSNormBwdFallback,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    error = _support_error(
        input, _collect_tensors(grad_out, input, rstd, weight), "RMSNorm"
    )
    if error is not None:
        return fallback_kernel(
            dispatch_keys,
            grad_out,
            input,
            normalized_shape,
            rstd,
            weight,
            output_mask,
        )

    _, cutedsl_rmsnorm_bwd = _get_rmsnorm_kernels()
    grad_input, grad_weight = cutedsl_rmsnorm_bwd(
        grad_out, input, rstd, weight, normalized_shape
    )

    if not output_mask[0]:
        grad_input = None
    if not output_mask[1]:
        grad_weight = None
    return grad_input, grad_weight


def register_cutedsl_rmsnorm_overrides() -> None:
    if not cu.runtime_available():
        return

    try:
        fwd_fallback = torch.library.get_kernel("aten::_fused_rms_norm", "CUDA")
        bwd_fallback = torch.library.get_kernel(
            "aten::_fused_rms_norm_backward", "CUDA"
        )
    except RuntimeError:
        log.info(
            "Skipping CuTeDSL RMSNorm override registration: CUDA kernels unavailable"
        )
        return

    fwd_impl = functools.partial(
        _cutedsl_fused_rms_norm_impl,
        fallback_kernel=fwd_fallback,
    )
    bwd_impl = functools.partial(
        _cutedsl_fused_rms_norm_backward_impl,
        fallback_kernel=bwd_fallback,
    )

    cu.register_op_override("aten", "_fused_rms_norm", "CUDA", fwd_impl)
    cu.register_op_override("aten", "_fused_rms_norm_backward", "CUDA", bwd_impl)


register_cutedsl_rmsnorm_overrides()
