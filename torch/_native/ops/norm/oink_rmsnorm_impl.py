"""Oink-backed RMSNorm overrides for aten fused RMSNorm operators.

Uses the vendored kernelagent_oink subset at ``torch._vendor.oink``. Targets
Blackwell (SM10.x); falls through to aten on Hopper or older.
"""

# mypy: allow-untyped-defs

from __future__ import annotations

import torch

from ... import cutedsl_utils as cu


# oink kernels require N (the inner dim) >= 128; below that the kernel
# allocates an empty smem partition and crashes. See oink's aten_override.py
# for the same gate.
_OINK_MIN_N = 128


def _is_supported(input: torch.Tensor) -> bool:
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if input.shape[-1] < _OINK_MIN_N:
        return False
    major, _ = torch.cuda.get_device_capability(input.device)
    # oink is Blackwell-only; Hopper is left to other DSL backends or aten.
    return major >= 10


def _shape_is_valid(
    input: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor | None,
) -> bool:
    # Mirror aten's _check_layer_norm_inputs: on any mismatch the aten path
    # raises, so we fall through and let it produce the proper error rather
    # than silently running the override on ill-shaped inputs.
    n = len(normalized_shape)
    if n < 1 or input.ndim < n:
        return False
    if list(input.shape[-n:]) != list(normalized_shape):
        return False
    if weight is not None and list(weight.shape) != list(normalized_shape):
        return False
    return True


def _fused_rms_norm_cond(
    input: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor | None,
    eps: float | None,
) -> bool:
    if not _is_supported(input):
        return False
    if not _shape_is_valid(input, normalized_shape, weight):
        return False
    # Weight must match input dtype and device for the oink kernel; mismatches
    # are legal under aten (it casts), so fall through.
    if weight is not None and (
        weight.dtype != input.dtype or weight.device != input.device
    ):
        return False
    # Empty inputs would feed a degenerate grid into the cuteDSL kernel.
    if input.numel() == 0:
        return False
    # The override reshapes + makes contiguous, which materializes a COW input.
    # Match bmm_outer_product's cond and fall through to aten so
    # composite-compliance tests don't flag spurious materialization.
    is_cow = torch._C._is_cow_tensor  # pyrefly: ignore[missing-attribute]
    if is_cow(input):
        return False
    if weight is not None and is_cow(weight):
        return False
    return True


def _fused_rms_norm_impl(
    input: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor | None,
    eps: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if eps is None:
        # Match aten/src/ATen/native/cuda/layer_norm_kernel.cu:1841-1847:
        # aten picks eps from the accumulator dtype, which is float32 for
        # fp16/bf16/fp32 inputs (the only dtypes our cond accepts).
        eps = torch.finfo(torch.float32).eps

    from .oink_norms import oink_rmsnorm_fwd

    return oink_rmsnorm_fwd(input, weight, normalized_shape, eps)


def _fused_rms_norm_backward_cond(
    grad_out: torch.Tensor,
    input: torch.Tensor,
    normalized_shape: list[int],
    rstd: torch.Tensor,
    weight: torch.Tensor | None,
    output_mask: list[bool],
) -> bool:
    if not _is_supported(input):
        return False
    if not _shape_is_valid(input, normalized_shape, weight):
        return False
    if weight is not None and (
        weight.dtype != input.dtype or weight.device != input.device
    ):
        return False
    if input.numel() == 0:
        return False
    is_cow = torch._C._is_cow_tensor  # pyrefly: ignore[missing-attribute]
    for t in (grad_out, input, rstd, weight):
        if t is not None and is_cow(t):
            return False
    return True


def _fused_rms_norm_backward_impl(
    grad_out: torch.Tensor,
    input: torch.Tensor,
    normalized_shape: list[int],
    rstd: torch.Tensor,
    weight: torch.Tensor | None,
    output_mask: list[bool],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    from .oink_norms import oink_rmsnorm_bwd

    grad_input, grad_weight = oink_rmsnorm_bwd(
        grad_out,
        input,
        rstd,
        weight,
        normalized_shape,
        dw_mask=output_mask[1],
    )

    if not output_mask[0]:
        grad_input = None
    return grad_input, grad_weight


def register_rmsnorm_overrides() -> None:
    if not torch.cuda.is_available():
        return
    cu.register_op_override(
        "aten",
        "_fused_rms_norm",
        "CUDA",
        cond=_fused_rms_norm_cond,
        impl=_fused_rms_norm_impl,
    )
    cu.register_op_override(
        "aten",
        "_fused_rms_norm_backward",
        "CUDA",
        cond=_fused_rms_norm_backward_cond,
        impl=_fused_rms_norm_backward_impl,
    )
