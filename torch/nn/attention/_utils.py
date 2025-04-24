# mypy: allow-untyped-defs
"""Defines utilities for interacting with scaled_dot_product_attention"""
import math
from typing import Optional, Union

import torch


__all__: list[str] = []


def _input_requires_grad(*tensors: torch.Tensor) -> bool:
    """Returns True if any of the tensors requires grad"""
    return any(t.requires_grad for t in tensors)


def _postprocess_flash_output(inpt_tensor: torch.Tensor, og_size: int) -> torch.Tensor:
    """Handles the unpad of the last dimension"""
    if inpt_tensor.size(-1) != og_size:
        return inpt_tensor[..., :og_size]
    return inpt_tensor


def _calculate_scale(head_dim_size: int, scale: Optional[float]) -> float:
    """
    For FlashAttention we pad the head dimension to be a multiple of 8 so we need to scale the output
    by the original head size and not the padded.
    """
    if scale is not None:
        return scale
    return 1.0 / math.sqrt(head_dim_size)


_SUPPORTED_HEAD_DIMS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def _supported_head_dim(n: Union[int, torch.SymInt]) -> bool:
    """Returns true if the head dim is supported by FlexAttention"""
    return n in _SUPPORTED_HEAD_DIMS


def _validate_sdpa_input(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
):
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError(
            f"Expected query, key, and value to have the same dtype, "
            f"but got query.dtype: {query.dtype}, key.dtype: {key.dtype}, "
            f"and value.dtype: {value.dtype} instead."
        )
    if query.device != key.device or query.device != value.device:
        raise ValueError(
            f"Expected query, key, and value to have the same device type, "
            f"but got query.device: {query.device}, key.device: {key.device}, "
            f"and value.device: {value.device} instead."
        )
    if query.dim() < 2 or key.dim() < 2 or value.dim() < 2:
        raise ValueError(
            f"Expected query, key, and value to all be  at least 2 dimensional, but got query.dim: "
            f"{query.dim()}, key.dim: {key.dim()} and value.dim: {value.dim()} instead."
        )
