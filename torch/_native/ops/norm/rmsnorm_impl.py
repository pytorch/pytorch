"""Quack-backed RMSNorm overrides for aten fused RMSNorm operators.

Uses the vendored quack subset at ``torch._vendor.quack``.
"""
# mypy: allow-untyped-defs

from __future__ import annotations

import torch

from ... import cutedsl_utils as cu


def _is_supported(input: torch.Tensor) -> bool:
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    major, _ = torch.cuda.get_device_capability(input.device)
    return major in (9, 10)


def _fused_rms_norm_cond(
    input: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor | None,
    eps: float | None,
) -> bool:
    if not _is_supported(input):
        return False
    # Empty inputs crash quack with cudaErrorInvalidConfiguration -- the bad
    # launch config poisons the CUDA context for subsequent calls. Quack's own
    # rmsnorm_bwd guards against this with `if x.numel() > 0` (rmsnorm.py:1111)
    # but the fwd path doesn't.
    if input.numel() == 0:
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
        eps = torch.finfo(input.dtype).eps

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
