# mypy: allow-untyped-defs
"""
FP8 scaled dot product attention with per-head (FA3) and per-tensor (cuDNN) descaling.
This operator is experimental and subject to change.
"""

import warnings
from enum import IntEnum

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
    """Per-head descaling (FA3 backend). Descale tensor shape: (batch_size, num_kv_heads)."""

    PER_TENSOR = 1
    """Per-tensor descaling (cuDNN backend). Descale tensor shape: (1, 1, 1, 1)."""


def _validate_descale(
    descale: Tensor | None,
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

    elif descale_type == DescaleType.PER_TENSOR:
        if descale.dim() != 4 or descale.shape != (1, 1, 1, 1):
            raise ValueError(
                f"{name}_descale must have shape (1, 1, 1, 1) "
                f"for PER_TENSOR descaling, got shape {tuple(descale.shape)}"
            )


def _scaled_dot_product_attention_quantized(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool = False,
    scale: float | None = None,
    q_descale: Tensor | None = None,
    k_descale: Tensor | None = None,
    v_descale: Tensor | None = None,
    q_descale_type: DescaleType = DescaleType.PER_HEAD,
    k_descale_type: DescaleType = DescaleType.PER_HEAD,
    v_descale_type: DescaleType = DescaleType.PER_HEAD,
    scale_s: float | None = None,
) -> Tensor:
    r"""Scaled dot product attention for FP8 inputs.

    This is a specialized version of scaled_dot_product_attention that supports
    FP8 quantized inputs (float8_e4m3fn). Two backends are available:

    - **PER_HEAD** (FA3): Per-head descaling, forward-only. Requires Flash Attention 3.
    - **PER_TENSOR** (cuDNN): Per-tensor descaling, forward + backward. Requires cuDNN 9.1+.

    .. warning::
        This function is experimental and subject to change.

    Args:
        query (Tensor): Query tensor; shape :math:`(N, H_q, L, E)` dtype float8_e4m3fn
        key (Tensor): Key tensor; shape :math:`(N, H, S, E)` dtype float8_e4m3fn
        value (Tensor): Value tensor; shape :math:`(N, H, S, E_v)` dtype float8_e4m3fn
        is_causal (bool): Apply causal attention mask
        scale (float, optional): Scaling factor for attention weights
        q_descale (Tensor, optional): Query descale tensor
        k_descale (Tensor, optional): Key descale tensor
        v_descale (Tensor, optional): Value descale tensor
        q_descale_type (DescaleType): Descaling granularity for query. Default: PER_HEAD
        k_descale_type (DescaleType): Descaling granularity for key. Default: PER_HEAD
        v_descale_type (DescaleType): Descaling granularity for value. Default: PER_HEAD
        scale_s (float, optional): Scale factor for the softmax output before FP8
            requantization in BMM2. Only used with PER_TENSOR. Default: 256.0
            (maps softmax output [0,1] to [0,256] to fill FP8 E4M3 range).

    Returns:
        Tensor: Attention output; shape :math:`(N, H_q, L, E_v)` dtype bfloat16
    """
    # Separate descale type parameters allow future backends to support mixed
    # granularity (e.g., per-tensor Q/K with per-head V). For now, all
    # quantized attention variants require the same type across Q, K, and V.
    if not (q_descale_type == k_descale_type == v_descale_type):
        raise ValueError(
            "All descale types must match. Got "
            f"q_descale_type={q_descale_type.name}, "
            f"k_descale_type={k_descale_type.name}, "
            f"v_descale_type={v_descale_type.name}"
        )
    descale_type = q_descale_type

    # Validate descale tensors
    _validate_descale(q_descale, "q", query, key, descale_type)
    _validate_descale(k_descale, "k", query, key, descale_type)
    _validate_descale(v_descale, "v", query, key, descale_type)

    # Route to per-tensor cuDNN backend (forward-only for now)
    if descale_type == DescaleType.PER_TENSOR:
        if torch.is_grad_enabled() and (
            query.requires_grad or key.requires_grad or value.requires_grad
        ):
            warnings.warn(
                "_scaled_dot_product_attention_quantized with PER_TENSOR descaling "
                "does not yet support backward pass. "
                "Gradients will not be computed for query, key, or value.",
                UserWarning,
            )
        scale_s = scale_s if scale_s is not None else 256.0
        _ones = torch.ones(1, 1, 1, 1, dtype=torch.float32, device=query.device)
        with torch.no_grad():
            result = (
                torch.ops.aten._scaled_dot_product_cudnn_attention_quantized_per_tensor(
                    query,
                    key,
                    value,
                    q_descale if q_descale is not None else _ones,
                    k_descale if k_descale is not None else _ones,
                    v_descale if v_descale is not None else _ones,
                    scale_s,
                    is_causal,
                    scale=scale,
                )
            )
        return result[0]

    # PER_HEAD: route to FA3 backend (forward-only)
    if torch.is_grad_enabled() and (
        query.requires_grad or key.requires_grad or value.requires_grad
    ):
        warnings.warn(
            "_scaled_dot_product_attention_quantized with PER_HEAD descaling "
            "does not support backward pass. "
            "Gradients will not be computed for query, key, or value.",
            UserWarning,
        )
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
    return result[0]
