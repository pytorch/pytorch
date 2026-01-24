# mypy: allow-untyped-defs
"""
This operator implements FP8 scaled dot product attention using Flash Attention 3.
This operator is experimental and subject to change.
"""

import warnings
from enum import IntEnum
from typing import Optional

import torch
from torch import Tensor


class DescaleType(IntEnum):
    """Describes the scaling granularity for FP8 descale tensors.

    Used with _scaled_dot_product_attention_quantized to explicitly specify
    how the descale factors are applied to the quantized inputs.

    .. warning::
        This enum is experimental and subject to change.
    """

    PER_HEAD = 0
    """Per-head descaling. Descale tensor shape: (batch_size, num_kv_heads)."""


def _validate_descale(
    descale: Optional[Tensor],
    name: str,
    query: Tensor,
    key: Tensor,
    descale_type: DescaleType,
) -> None:
    """Validate descale tensor for the specified scaling type.

    Args:
        descale: The descale tensor to validate (may be None)
        name: Name of the descale tensor ("q", "k", or "v") for error messages
        query: Query tensor to get batch size
        key: Key tensor to get num_kv_heads
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
        batch_size = query.size(0)
        # All descale tensors use num_kv_heads, even q_descale (broadcast internally)
        # For BHSD layout, num_kv_heads is at dim 1 of key
        num_kv_heads = key.size(1)

        if descale.dim() != 2:
            raise ValueError(
                f"{name}_descale must be a 2D tensor with shape (batch_size, num_kv_heads) "
                f"for PER_HEAD descaling, got {descale.dim()}D tensor"
            )

        if descale.size(0) != batch_size:
            raise ValueError(
                f"{name}_descale batch dimension must match query batch size, "
                f"expected {batch_size}, got {descale.size(0)}"
            )

        if descale.size(1) != num_kv_heads:
            raise ValueError(
                f"{name}_descale head dimension must match num_kv_heads, "
                f"expected {num_kv_heads}, got {descale.size(1)}"
            )


def _scaled_dot_product_attention_quantized(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None,
    q_descale: Optional[Tensor] = None,
    k_descale: Optional[Tensor] = None,
    v_descale: Optional[Tensor] = None,
    q_descale_type: DescaleType = DescaleType.PER_HEAD,
    k_descale_type: DescaleType = DescaleType.PER_HEAD,
    v_descale_type: DescaleType = DescaleType.PER_HEAD,
) -> Tensor:
    r"""Scaled dot product attention for FP8 inputs.

    This is a specialized version of scaled_dot_product_attention that supports
    FP8 quantized inputs (float8_e4m3fn) with per-head descaling. Requires the
    Flash Attention 3 backend to be activated.

    .. warning::
        This function is experimental and only supports forward pass.

    Args:
        query (Tensor): Query tensor; shape :math:`(N, H_q, L, E)` dtype float8_e4m3fn
        key (Tensor): Key tensor; shape :math:`(N, H, S, E)` dtype float8_e4m3fn
        value (Tensor): Value tensor; shape :math:`(N, H, S, E_v)` dtype float8_e4m3fn
        is_causal (bool): Apply causal attention mask
        scale (float, optional): Scaling factor for attention weights
        q_descale (Tensor, optional): Query descale tensor; shape :math:`(N, H)` for PER_HEAD
        k_descale (Tensor, optional): Key descale tensor; shape :math:`(N, H)` for PER_HEAD
        v_descale (Tensor, optional): Value descale tensor; shape :math:`(N, H)` for PER_HEAD
        q_descale_type (DescaleType): Specifies the descaling granularity for query. Default: PER_HEAD
        k_descale_type (DescaleType): Specifies the descaling granularity for key. Default: PER_HEAD
        v_descale_type (DescaleType): Specifies the descaling granularity for value. Default: PER_HEAD

    Returns:
        Tensor: Attention output; shape :math:`(N, H_q, L, E_v)` dtype bfloat16
    """
    # Validate descale tensors
    _validate_descale(q_descale, "q", query, key, q_descale_type)
    _validate_descale(k_descale, "k", query, key, k_descale_type)
    _validate_descale(v_descale, "v", query, key, v_descale_type)

    if torch.is_grad_enabled() and (
        query.requires_grad or key.requires_grad or value.requires_grad
    ):
        warnings.warn(
            "_scaled_dot_product_attention_quantized does not support backward pass. "
            "Gradients will not be computed for query, key, or value.",
            UserWarning,
        )
    # Directly call the internal flash attention operator which has descale support
    # NOTE: This should be torch._scaled_dot_product_flash_attention, but it does not work with torch.compile
    result = torch.ops.aten._scaled_dot_product_flash_attention.quantized(
        query,
        key,
        value,
        q_descale,
        k_descale,
        v_descale,
        0.0,
        is_causal,
        False,
        scale=scale,
    )
    return result[0]  # Return the output tensor, mirroring scaled_dot_product_attention
