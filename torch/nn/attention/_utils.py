"""Defines utilities for interacting with scaled_dot_product_attention"""
import math
from typing import List, Optional, Tuple

import torch

__all__: List[str] = []


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


def _preprocess_flash(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float]:
    """Preprocesses the input for _scaled_dot_product_flash_attention
    Args:
        query: torch.Tensor: The query tensor
        key: torch.Tensor: The key tensor
        value: torch.Tensor: The value tensor
        scale: Optional[float]: The scale to use for the attention
    """
    needs_padding = query.size(-1) % 8 != 0
    og_head_size = query.size(-1)
    og_scale = _calculate_scale(og_head_size, scale)
    if needs_padding:
        query = torch.nn.functional.pad(query, (0, 8 - query.size(-1) % 8))
        key = torch.nn.functional.pad(key, (0, 8 - key.size(-1) % 8))
        value = torch.nn.functional.pad(value, (0, 8 - value.size(-1) % 8))
    return query, key, value, og_head_size, og_scale
