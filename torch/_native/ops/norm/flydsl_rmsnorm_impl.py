"""FlyDSL overrides for ATen's internal fused RMSNorm FWD/BWD operators."""

# mypy: allow-untyped-defs

from __future__ import annotations

import torch

from ... import flydsl_utils as fu

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SUPPORTED_EPS = 1e-5
_HIP_AVAILABLE = torch.version.hip is not None
_is_cow_tensor = torch._C._is_cow_tensor  # pyrefly: ignore[missing-attribute]
_rmsnorm_fwd = None
_rmsnorm_bwd = None


def _normalized_shape_1d(normalized_shape) -> int | None:
    """Return N for the one-dimensional normalization supported by the kernel."""

    if isinstance(normalized_shape, int):
        return normalized_shape
    if isinstance(normalized_shape, (tuple, list, torch.Size)):
        if len(normalized_shape) != 1:
            return None
        try:
            return int(normalized_shape[0])
        except (TypeError, ValueError):
            return None

    try:
        shape = tuple(int(x) for x in normalized_shape)
    except TypeError:
        try:
            shape = (int(normalized_shape),)
        except (TypeError, ValueError):
            return None
    except ValueError:
        return None
    return shape[0] if len(shape) == 1 else None


def _common_supported(
    input: torch.Tensor,
    n: int,
    weight: torch.Tensor | None,
) -> bool:
    """Cheap dispatcher predicate shared by forward and backward."""
    if not _HIP_AVAILABLE or input.device.type != "cuda":
        return False
    if input.dtype not in _SUPPORTED_DTYPES:
        return False
    if input.ndim < 1 or input.shape[-1] != n or input.numel() == 0:
        return False
    if not input.is_contiguous():
        return False
    if weight is None:
        return False
    if (
        weight.shape != (n,)
        or weight.dtype != input.dtype
        or weight.device != input.device
        or not weight.is_contiguous()
    ):
        return False
    # The N>2048 fp16/bf16 forward path issues 128-bit vector copies. A
    # contiguous offset view can still have a misaligned base address.
    if input.dtype in (torch.float16, torch.bfloat16) and n > 2048:
        if input.data_ptr() % 16 != 0 or weight.data_ptr() % 16 != 0:
            return False
    # Reshaping a copy-on-write tensor would materialize it. Let ATen preserve
    # its normal semantics for these uncommon inputs.
    if _is_cow_tensor(input) or _is_cow_tensor(weight):
        return False
    return True


def _fused_rms_norm_fwd_perf_wins(input: torch.Tensor, n: int) -> bool:
    rows_m = input.numel() // n
    # Tuned on MI355.
    return (
        (4096 <= n < 8192 and rows_m >= 8192)
        or (8192 <= n < 16384 and rows_m >= 4096)
        or (n >= 16384 and rows_m >= 2048)
    )


def _fused_rms_norm_bwd_perf_wins(input: torch.Tensor, n: int) -> bool:
    # Tuned on MI355. Keep only regions where benchmarks show at least a
    # 10% win over ATen.  Borderline fp16/bf16 large-N and fp32 cases fall back.
    rows_m = input.numel() // n
    if input.dtype in (torch.float16, torch.bfloat16):
        return 8192 <= n < 32768 and rows_m >= 32768
    return False


def _fused_rms_norm_cond(
    input: torch.Tensor,
    normalized_shape,
    weight: torch.Tensor | None,
    eps: float | None,
) -> bool:
    n = _normalized_shape_1d(normalized_shape)
    if n is None:
        return False
    if not _common_supported(input, n, weight):
        return False
    if eps is None or float(eps) != _SUPPORTED_EPS:
        return False
    return _fused_rms_norm_fwd_perf_wins(input, n)


def _get_rmsnorm_fwd(input: torch.Tensor):
    global _rmsnorm_fwd
    if _rmsnorm_fwd is None:
        # Import under the input device guard because the vendored builders query
        # the active ROCm architecture while initializing wave-size constants.
        with torch.cuda.device(input.device):
            from .flydsl_kernels import rmsnorm_fwd

        _rmsnorm_fwd = rmsnorm_fwd
    return _rmsnorm_fwd


def _get_rmsnorm_bwd(input: torch.Tensor):
    global _rmsnorm_bwd
    if _rmsnorm_bwd is None:
        # See _get_rmsnorm_fwd: importing the kernel module can query the active
        # device while initializing vendored FlyDSL constants.
        with torch.cuda.device(input.device):
            from .flydsl_kernels import rmsnorm_bwd

        _rmsnorm_bwd = rmsnorm_bwd
    return _rmsnorm_bwd


def _fused_rms_norm_impl(
    input: torch.Tensor,
    normalized_shape,
    weight: torch.Tensor | None,
    eps: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # The predicate guarantees both values are present. Keeping the checks here
    # makes failures clear if the implementation is ever called directly.
    if weight is None or eps is None:
        raise RuntimeError("FlyDSL RMSNorm requires explicit weight and eps")

    rmsnorm_fwd = _get_rmsnorm_fwd(input)
    return rmsnorm_fwd(input, normalized_shape, weight, float(eps))


def _fused_rms_norm_backward_cond(
    grad_out: torch.Tensor,
    input: torch.Tensor,
    normalized_shape,
    rstd: torch.Tensor,
    weight: torch.Tensor | None,
    output_mask,
) -> bool:
    # Backward receives the already-computed rstd, so its formula is independent
    # of the eps value used by forward.
    n = _normalized_shape_1d(normalized_shape)
    if n is None:
        return False
    if not _common_supported(input, n, weight):
        return False
    rows_m = input.numel() // n
    if (
        grad_out.shape != input.shape
        or grad_out.dtype != input.dtype
        or grad_out.device != input.device
        or not grad_out.is_contiguous()
    ):
        return False
    if (
        rstd.device != input.device
        or rstd.dtype != torch.float32
        or rstd.numel() != rows_m
        or not rstd.is_contiguous()
    ):
        return False
    if len(output_mask) != 2 or not any(bool(x) for x in output_mask):
        return False

    if _is_cow_tensor(grad_out) or _is_cow_tensor(rstd):
        return False
    return _fused_rms_norm_bwd_perf_wins(input, n)


def _fused_rms_norm_backward_impl(
    grad_out: torch.Tensor,
    input: torch.Tensor,
    normalized_shape,
    rstd: torch.Tensor,
    weight: torch.Tensor | None,
    output_mask,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if weight is None:
        raise RuntimeError("FlyDSL RMSNorm backward requires an explicit weight")

    rmsnorm_bwd = _get_rmsnorm_bwd(input)
    grad_input, grad_weight = rmsnorm_bwd(grad_out, input, normalized_shape, rstd, weight)
    return (
        grad_input if bool(output_mask[0]) else None,
        grad_weight if bool(output_mask[1]) else None,
    )


def register_flydsl_rmsnorm_overrides() -> None:
    """Register both training operators while retaining transparent fallback."""

    # QuACK also overrides these symbols for NVIDIA. Multiple registrations are
    # intentional: the predicates are mutually exclusive (ROCm versus NVIDIA).
    fu.register_op_override(
        "aten",
        "_fused_rms_norm",
        "CUDA",
        cond=_fused_rms_norm_cond,
        impl=_fused_rms_norm_impl,
        allow_multiple_override=True,
    )
    fu.register_op_override(
        "aten",
        "_fused_rms_norm_backward",
        "CUDA",
        cond=_fused_rms_norm_backward_cond,
        impl=_fused_rms_norm_backward_impl,
        allow_multiple_override=True,
    )