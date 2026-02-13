"""CuteDSL RMSNorm kernel placeholder.

Implement the following two functions with CuteDSL kernels:

- ``cutedsl_rmsnorm_fwd``: Forward pass — compute RMSNorm and reciprocal std.
- ``cutedsl_rmsnorm_bwd``: Backward pass — compute gradients for input and weight.
"""

from __future__ import annotations

import torch


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
