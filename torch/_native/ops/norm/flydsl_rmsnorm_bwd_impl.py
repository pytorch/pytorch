"""FlyDSL override for ATen's internal fused RMSNorm backward operator."""

# mypy: allow-untyped-defs

from __future__ import annotations

import torch

from ... import flydsl_utils as fu


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# CDNA raw-buffer copies use 32-bit byte offsets; exactly 4 GiB wraps to zero.
_RMSNORM_BWD_BUFFER_ADDRESS_LIMIT_BYTES = 1 << 32

_HIP_AVAILABLE = torch.version.hip is not None
_is_cow_tensor = torch._C._is_cow_tensor  # pyrefly: ignore[missing-attribute]
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
    """Cheap dispatcher predicate for supported inputs."""
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
    # Reshaping a copy-on-write tensor would materialize it. Let ATen preserve
    # its normal semantics for these uncommon inputs.
    if _is_cow_tensor(input) or _is_cow_tensor(weight):
        return False
    return True


def _fused_rms_norm_bwd_perf_wins(input: torch.Tensor, n: int) -> bool:
    rows_m = input.numel() // n
    # Tuned on MI355X from the complete power-of-two M/N grid using
    # synchronized wall-to-sync measurements. Near-crossover points stay on
    # ATen, and every boundary is a power of two.
    if input.dtype in (torch.float16, torch.bfloat16):
        return (
            (16 <= n < 64 and rows_m >= 8192)
            or (1024 <= n < 2048 and rows_m >= 65536)
            or (2048 <= n < 4096 and rows_m >= 32768)
            or (4096 <= n < 8192 and rows_m >= 8192)
            or (8192 <= n < 16384 and rows_m >= 2048)
            or (16384 <= n < 32768 and rows_m >= 512)
            or (32768 <= n <= 65536 and rows_m >= 16)
        )
    if input.dtype == torch.float32:
        return (
            (16 <= n < 64 and rows_m >= 16384)
            or (256 <= n < 512 and rows_m >= 65536)
            or (512 <= n < 2048 and rows_m >= 16384)
            or (2048 <= n < 4096 and rows_m >= 8192)
            or (4096 <= n < 8192 and rows_m >= 4096)
            or (8192 <= n < 16384 and rows_m >= 2048)
            or (16384 <= n < 32768 and rows_m >= 64)
            or (32768 <= n <= 65536 and rows_m >= 16)
        )
    return False


def _fused_rms_norm_bwd_buffer_addressable(input: torch.Tensor) -> bool:
    """Return whether BWD raw-buffer copies can address the full matrix."""

    return (
        input.numel() * input.element_size() < _RMSNORM_BWD_BUFFER_ADDRESS_LIMIT_BYTES
    )


def _get_rmsnorm_bwd(input: torch.Tensor):
    global _rmsnorm_bwd
    if _rmsnorm_bwd is None:
        # Import under the input device guard because the vendored builders query
        # the active ROCm architecture while initializing wave-size constants.
        with torch.cuda.device(input.device):
            from .flydsl_rmsnorm_bwd import rmsnorm_bwd

        _rmsnorm_bwd = rmsnorm_bwd
    return _rmsnorm_bwd


def _fused_rms_norm_backward_cond(
    grad_out: torch.Tensor,
    input: torch.Tensor,
    normalized_shape,
    rstd: torch.Tensor,
    weight: torch.Tensor | None,
    output_mask,
) -> bool:
    n = _normalized_shape_1d(normalized_shape)
    if n is None:
        return False
    if not _common_supported(input, n, weight):
        return False
    if not _fused_rms_norm_bwd_buffer_addressable(input):
        return False
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
        or rstd.shape != (*input.shape[:-1], 1)
        or not rstd.is_contiguous()
    ):
        return False
    try:
        mask = tuple(bool(value) for value in output_mask)
    except TypeError:
        return False
    # V2 always executes K1/K2/K3 and computes both outputs. Let ATen handle
    # single-output calls so the override does not do avoidable work.
    if mask != (True, True):
        return False
    assert weight is not None
    if any(tensor.data_ptr() % 16 != 0 for tensor in (input, weight, grad_out)):
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
    del normalized_shape
    if weight is None:
        raise RuntimeError("FlyDSL RMSNorm backward requires an explicit weight")

    need_grad_input = bool(output_mask[0])
    need_grad_weight = bool(output_mask[1])
    with torch.cuda.device(input.device):
        grad_input, grad_weight = _get_rmsnorm_bwd(input)(
            grad_out,
            input,
            rstd,
            weight,
            need_grad_weight=need_grad_weight,
        )
    return grad_input if need_grad_input else None, grad_weight


def register_op_override() -> None:
    fu.register_op_override(
        "aten",
        "_fused_rms_norm_backward",
        "CUDA",
        cond=_fused_rms_norm_backward_cond,
        impl=_fused_rms_norm_backward_impl,
        allow_multiple_override=True,
    )
