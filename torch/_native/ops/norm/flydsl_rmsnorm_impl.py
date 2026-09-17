"""FlyDSL override for ATen's internal fused RMSNorm forward operator."""

# mypy: allow-untyped-defs

from __future__ import annotations

import torch

from ... import flydsl_utils as fu
from .flydsl_rmsnorm_utils import normalized_shape_1d, SUPPORTED_DTYPES


_SUPPORTED_ARCHES = ("gfx950",)


def _common_supported(
    input: torch.Tensor,
    n: int,
    weight: torch.Tensor | None,
) -> bool:
    """Cheap dispatcher predicate for supported forward inputs."""
    if input.device.type != "cuda":
        return False
    device_index = input.device.index
    if not fu._is_supported_arch(device_index, _SUPPORTED_ARCHES):
        return False
    if input.dtype not in SUPPORTED_DTYPES:
        return False
    if input.ndim < 1 or input.shape[-1] != n or input.numel() == 0:
        return False
    if not input.is_contiguous():
        return False
    if weight is None:
        return False
    return (
        weight.shape == (n,)
        and weight.dtype == input.dtype
        and weight.device == input.device
        and weight.is_contiguous()
    )


def _fused_rms_norm_fwd_perf_wins(rows_m: int, n: int) -> bool:
    # Tuned on gfx950 (MI355) at rows_m=2048. 114688 is the last N where all
    # three dtypes still beat aten (1.15x-1.19x)
    return (
        (4096 <= n < 8192 and rows_m >= 8192)
        or (8192 <= n < 16384 and rows_m >= 4096)
        or (16384 <= n <= 114688 and rows_m >= 2048)
    )


def _fused_rms_norm_cond(
    input: torch.Tensor,
    normalized_shape,
    weight: torch.Tensor | None,
    eps: float | None,
) -> bool:
    n = normalized_shape_1d(normalized_shape)
    if n is None:
        return False
    if eps is not None and not eps >= 0.0:
        return False
    if not _common_supported(input, n, weight):
        return False
    rows_m = input.numel() // n  # n has been validated to be non-Zero
    if not fu._fits_int32_buffer_span(rows_m, n, input.element_size()):
        return False
    return _fused_rms_norm_fwd_perf_wins(rows_m, n)


def _fused_rms_norm_impl(
    input: torch.Tensor,
    normalized_shape,
    weight: torch.Tensor | None,
    eps: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # The predicate guarantees weight is present. Keeping the check here makes
    # failures clear if the implementation is ever called directly.
    if weight is None:
        raise RuntimeError("FlyDSL RMSNorm requires an explicit weight")
    if eps is None:
        # Match aten/src/ATen/native/cuda/layer_norm_kernel.cu
        eps = torch.finfo(torch.float32).eps

    # Imported on first dispatch, not at module scope: the kernel module pulls
    # in flydsl, which must stay out of `import torch`.
    from .flydsl_rmsnorm_fwd import rmsnorm_fwd

    return rmsnorm_fwd(input, normalized_shape, weight, float(eps))


def register_flydsl_rmsnorm_overrides() -> None:
    fu.register_op_override(
        "aten",
        "_fused_rms_norm",
        "CUDA",
        cond=_fused_rms_norm_cond,
        impl=_fused_rms_norm_impl,
    )
