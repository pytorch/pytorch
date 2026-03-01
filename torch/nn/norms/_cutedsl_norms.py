"""CuteDSL norm dispatcher overrides (RMSNorm)."""
# mypy: allow-untyped-defs

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import torch
from torch.library import Library

from . import _registry
from ._kernels.norms import cutedsl_rmsnorm_bwd, cutedsl_rmsnorm_fwd


__all__ = [
    "register_cutedsl_rmsnorm",
]


@dataclass
class _CuteDSLNormHandle:
    library: Library | None

    def remove(self) -> None:
        self.library = None


@cache
def _get_device_major(device: torch.device) -> int:
    major, _ = torch.cuda.get_device_capability(device)
    return major


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
        return f"CuteDSL {name} requires compute capability 9.0 or 10.0"
    return None


def _collect_tensors(*tensors: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
    return tuple(t for t in tensors if t is not None)


# RMSNorm
def register_cutedsl_rmsnorm() -> _CuteDSLNormHandle:
    """Register CuteDSL RMSNorm kernels with the PyTorch dispatcher."""
    lib = Library("aten", "IMPL", "CUDA")  # noqa: TOR901
    lib.impl("_fused_rms_norm", _cutedsl_fused_rms_norm_impl, "CUDA")
    lib.impl("_fused_rms_norm_backward", _cutedsl_fused_rms_norm_backward_impl, "CUDA")
    return _CuteDSLNormHandle(lib)


def _cutedsl_fused_rms_norm_impl(
    input: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor | None,
    eps: float | None,
):
    error = _support_error(input, _collect_tensors(input, weight), "RMSNorm")
    if error is not None:
        raise RuntimeError(f"CuteDSL RMSNorm forward unsupported: {error}")

    if eps is None:
        eps = torch.finfo(input.dtype).eps

    return cutedsl_rmsnorm_fwd(input, weight, normalized_shape, eps)


def _cutedsl_fused_rms_norm_backward_impl(
    grad_out: torch.Tensor,
    input: torch.Tensor,
    normalized_shape: list[int],
    rstd: torch.Tensor,
    weight: torch.Tensor | None,
    output_mask: list[bool],
):
    error = _support_error(
        input, _collect_tensors(grad_out, input, rstd, weight), "RMSNorm"
    )
    if error is not None:
        raise RuntimeError(f"CuteDSL RMSNorm backward unsupported: {error}")

    grad_input, grad_weight = cutedsl_rmsnorm_bwd(
        grad_out, input, rstd, weight, normalized_shape
    )

    if not output_mask[0]:
        grad_input = None
    if not output_mask[1]:
        grad_weight = None
    return grad_input, grad_weight


# Self-registration
_registry.register_norm_impl("cutedsl_rmsnorm", register_fn=register_cutedsl_rmsnorm)
