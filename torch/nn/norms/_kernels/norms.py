"""CuteDSL norm kernel placeholders.

Implement the following functions with CuteDSL kernels:

RMSNorm:
- ``cutedsl_rmsnorm_fwd``: Forward pass — compute RMSNorm and reciprocal std.
- ``cutedsl_rmsnorm_bwd``: Backward pass — compute gradients for input and weight.

LayerNorm:
- ``cutedsl_layernorm_fwd``: Forward pass — compute LayerNorm, mean, and reciprocal std.
- ``cutedsl_layernorm_bwd``: Backward pass — compute gradients for input, weight, and bias.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


def cutedsl_rmsnorm_fwd(
    input: torch.Tensor,
    weight: torch.Tensor | None,
    normalized_shape: list[int],
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass for CuteDSL RMSNorm.

    Args:
        input: Input tensor (CUDA, fp16/bf16/fp32).
        weight: Optional learnable weight (same shape as ``normalized_shape``).
        normalized_shape: Dimensions to normalize over (trailing dims of input).
        eps: Epsilon for numerical stability.

    Returns:
        Tuple of (output, rstd) where rstd is the reciprocal standard deviation.
    """
    raise NotImplementedError("CuteDSL RMSNorm forward kernel not yet implemented")


def cutedsl_rmsnorm_bwd(
    grad_out: torch.Tensor,
    input: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor | None,
    normalized_shape: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass for CuteDSL RMSNorm.

    Args:
        grad_out: Gradient of the loss w.r.t. the output.
        input: Original input tensor from the forward pass.
        rstd: Reciprocal standard deviation from the forward pass.
        weight: Optional learnable weight.
        normalized_shape: Dimensions that were normalized over.

    Returns:
        Tuple of (grad_input, grad_weight).
    """
    raise NotImplementedError("CuteDSL RMSNorm backward kernel not yet implemented")


# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------


def cutedsl_layernorm_fwd(
    input: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    normalized_shape: list[int],
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass for CuteDSL LayerNorm.

    Args:
        input: Input tensor (CUDA, fp16/bf16/fp32).
        weight: Optional learnable weight (same shape as ``normalized_shape``).
        bias: Optional learnable bias (same shape as ``normalized_shape``).
        normalized_shape: Dimensions to normalize over (trailing dims of input).
        eps: Epsilon for numerical stability.

    Returns:
        Tuple of (output, mean, rstd).
    """
    raise NotImplementedError("CuteDSL LayerNorm forward kernel not yet implemented")


def cutedsl_layernorm_bwd(
    grad_out: torch.Tensor,
    input: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    normalized_shape: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward pass for CuteDSL LayerNorm.

    Args:
        grad_out: Gradient of the loss w.r.t. the output.
        input: Original input tensor from the forward pass.
        mean: Mean from the forward pass.
        rstd: Reciprocal standard deviation from the forward pass.
        weight: Optional learnable weight.
        bias: Optional learnable bias.
        normalized_shape: Dimensions that were normalized over.

    Returns:
        Tuple of (grad_input, grad_weight, grad_bias).
    """
    raise NotImplementedError("CuteDSL LayerNorm backward kernel not yet implemented")
