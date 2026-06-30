"""FlyDSL-backed RMSNorm override for ``aten::rms_norm``."""

# mypy: allow-untyped-defs

from __future__ import annotations

import torch

from ... import flydsl_utils as fu


_SUPPORTED_HIDDEN_SIZES = frozenset({128, 256, 512, 1024, 2000, 2048, 4096, 8192})
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SUPPORTED_EPS = 1e-5


def _normalized_shape_1d(normalized_shape) -> int | None:
    try:
        shape = tuple(int(x) for x in normalized_shape)
    except TypeError:
        try:
            shape = (int(normalized_shape),)
        except (TypeError, ValueError):
            return None
    except ValueError:
        return None

    if len(shape) != 1:
        return None
    return shape[0]


def _eps_supported(eps: float | None) -> bool:
    return eps is not None and float(eps) == _SUPPORTED_EPS


def _rms_norm_cond(
    input: torch.Tensor,
    normalized_shape,
    weight: torch.Tensor | None = None,
    eps: float | None = None,
) -> bool:
    n = _normalized_shape_1d(normalized_shape)
    if n is None or n not in _SUPPORTED_HIDDEN_SIZES:
        return False
    if not fu.runtime_available():
        return False
    if torch.version.hip is None:
        return False
    if input.device.type != "cuda" or input.dtype not in _SUPPORTED_DTYPES:
        return False
    if input.requires_grad or (weight is not None and weight.requires_grad):
        return False
    if input.ndim < 1 or input.shape[-1] != n or input.numel() == 0:
        return False
    if not input.is_contiguous():
        return False
    if weight is None:
        return False
    if weight.shape != (n,) or weight.dtype != input.dtype or weight.device != input.device:
        return False
    if not weight.is_contiguous():
        return False
    if not _eps_supported(eps):
        return False
    return True


def _rms_norm_impl(
    input: torch.Tensor,
    normalized_shape,
    weight: torch.Tensor | None = None,
    eps: float | None = None,
) -> torch.Tensor:
    from .flydsl_kernels import rmsnorm

    return rmsnorm(input, normalized_shape, weight, eps)


def register_flydsl_rmsnorm_overrides() -> None:
    fu.register_op_override(
        "aten",
        "rms_norm",
        "CUDA",
        cond=_rms_norm_cond,
        impl=_rms_norm_impl,
    )
