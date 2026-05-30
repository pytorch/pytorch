"""Quack-backed RMSNorm overrides for aten fused RMSNorm operators.

Uses the vendored quack subset at ``torch._vendor.quack``.
"""
# mypy: allow-untyped-defs

from __future__ import annotations

import torch

from ... import cutedsl_utils as cu


def _is_supported(input: torch.Tensor) -> bool:
    if input.device.type != "cuda":
        return False
    if torch.version.hip is not None:
        return False
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    major, _ = torch.cuda.get_device_capability(input.device)
    return major in (9, 10)


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
    # Weight must match input dtype and device for quack's kernel; mismatches
    # are legal under aten (it casts), so fall through.
    if weight is not None and (
        weight.dtype != input.dtype or weight.device != input.device
    ):
        return False
    # Empty inputs crash quack with cudaErrorInvalidConfiguration -- the bad
    # launch config poisons the CUDA context for subsequent calls. Quack's own
    # rmsnorm_bwd guards against this with `if x.numel() > 0` (rmsnorm.py:1111)
    # but the fwd path doesn't.
    if input.numel() == 0:
        return False
    # Non-contiguous weight would require a reshape+copy that we haven't
    # measured; fall through to aten until we do.
    if weight is not None and not weight.is_contiguous():
        return False
    # The override reshapes + makes contiguous, which materializes a COW input.
    # Match the bmm_outer_product cond (triton_impl.py:46) and fall through to
    # aten so composite-compliance tests don't flag spurious materialization.
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
        # aten picks eps from the *accumulator* dtype, which is float32 for
        # fp16/bf16/fp32 inputs (the only dtypes our cond accepts).
        eps = torch.finfo(torch.float32).eps

    from .norms import quack_rmsnorm_fwd

    return quack_rmsnorm_fwd(input, weight, normalized_shape, eps)


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
    if weight is not None and not weight.is_contiguous():
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
    from .norms import quack_rmsnorm_bwd

    grad_input, grad_weight = quack_rmsnorm_bwd(
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
    # Don't gate on torch.cuda.is_available() here: it calls cuInit and
    # poisons fork (vLLM EngineCore relies on fork). cu.register_op_override
    # already short-circuits on _cuda.is_built() via _check_runtime_available.
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
