# mypy: allow-untyped-defs
"""Defines utilities for interacting with scaled_dot_product_attention"""

import math
from enum import IntEnum

import torch


__all__: list[str] = []

# fp8 dtypes that flash attention runs through its `.quantized` ops, with a bf16 output.
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _empty_with_matching_layout(
    query: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Allocate an output with the query's dimension order, in ``dtype`` (default: the query's)."""
    dtype = query.dtype if dtype is None else dtype
    if tuple(query.shape) == shape:
        return torch.empty_like(query, dtype=dtype)

    fill_order = sorted(
        range(query.dim()),
        key=lambda idx: query.stride()[idx] if query.stride()[idx] else math.inf,
    )
    strides = [0] * len(fill_order)
    stride = 1
    for idx in fill_order:
        strides[idx] = stride
        stride *= shape[idx]
    return torch.empty_strided(shape, strides, dtype=dtype, device=query.device)


def _input_requires_grad(*tensors: torch.Tensor) -> bool:
    """Returns True if any of the tensors requires grad"""
    return any(t.requires_grad for t in tensors)


def _postprocess_flash_output(inpt_tensor: torch.Tensor, og_size: int) -> torch.Tensor:
    """Handles the unpad of the last dimension"""
    if inpt_tensor.size(-1) != og_size:
        return inpt_tensor[..., :og_size]
    return inpt_tensor


def _calculate_scale(head_dim_size: int, scale: float | None) -> float:
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
    attn_mask: torch.Tensor | None = None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    allow_lowp_kv=False,
) -> None:
    if not allow_lowp_kv:
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


class DescaleType(IntEnum):
    """Describes the scaling granularity for FP8 descale tensors.

    Used by the fp8 attention APIs (_scaled_dot_product_attention_quantized, varlen_attn)
    to specify how the descale factors are applied to the quantized inputs.

    .. warning::
        This enum is experimental and subject to change.
    """

    PER_HEAD = 0
    """Per-head descaling. Descale tensor shape: (batch_size, num_kv_heads)."""


def _validate_descale(
    descale: torch.Tensor | None,
    name: str,
    batch_size: int,
    num_kv_heads: int,
    descale_type: DescaleType,
) -> None:
    """Validate descale tensor for the specified scaling type.

    Args:
        descale: The descale tensor to validate (may be None)
        name: Name of the descale tensor ("q", "k", or "v") for error messages
        batch_size: Number of sequences in the batch
        num_kv_heads: Number of key/value heads
        descale_type: The scaling granularity being used

    Raises:
        ValueError: If the descale tensor has invalid dtype, device, or shape

    Note:
        All descale tensors (q, k, v) use num_kv_heads for the head dimension.
        For GQA/MQA where num_query_heads > num_kv_heads, q_descale is broadcast
        from (B, H_kv) to match the query heads internally.
    """
    if descale is None:
        return

    # Check dtype
    if descale.dtype != torch.float32:
        raise ValueError(f"{name}_descale must have dtype float32, got {descale.dtype}")

    # Check device
    if not descale.is_cuda:
        raise ValueError(f"{name}_descale must be a CUDA tensor")

    # Check shape based on descale type
    if descale_type == DescaleType.PER_HEAD:
        # All descale tensors use num_kv_heads, even q_descale (broadcast internally)
        if descale.dim() != 2:
            raise ValueError(
                f"{name}_descale must be a 2D tensor with shape (batch_size, num_kv_heads) "
                f"for PER_HEAD descaling, got {descale.dim()}D tensor"
            )

        if descale.size(0) != batch_size:
            raise ValueError(
                f"{name}_descale batch dimension must match the number of sequences, "
                f"expected {batch_size}, got {descale.size(0)}"
            )

        if descale.size(1) != num_kv_heads:
            raise ValueError(
                f"{name}_descale head dimension must match num_kv_heads, "
                f"expected {num_kv_heads}, got {descale.size(1)}"
            )
