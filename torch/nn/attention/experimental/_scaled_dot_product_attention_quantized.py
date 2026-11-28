# mypy: allow-untyped-defs
"""
This operator implements FP8 scaled dot product attention using Flash Attention 3 or 4.
This operator is experimental and subject to change.
"""

import warnings

import torch
from torch import Tensor
from torch.nn.attention._utils import _validate_descale, DescaleType


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
) -> Tensor:
    r"""Scaled dot product attention for FP8 inputs.

    This is a specialized version of scaled_dot_product_attention that supports
    FP8 quantized inputs with per-head descaling. Requires the Flash Attention 3
    (SM90, float8_e4m3fn) or Flash Attention 4 (SM100, float8_e4m3fn or float8_e5m2)
    implementation to be activated.

    .. warning::
        This function is experimental and only supports forward pass.

    Args:
        query (Tensor): Query tensor; shape :math:`(N, H_q, L, E)` dtype float8_e4m3fn or float8_e5m2
        key (Tensor): Key tensor; shape :math:`(N, H, S, E)`, same dtype as ``query``
        value (Tensor): Value tensor; shape :math:`(N, H, S, E_v)`, same dtype as ``query``
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
    # Validate descale tensors. In the BHSD layout, num_kv_heads is key's dim 1.
    batch_size, num_kv_heads = query.size(0), key.size(1)
    _validate_descale(q_descale, "q", batch_size, num_kv_heads, q_descale_type)
    _validate_descale(k_descale, "k", batch_size, num_kv_heads, k_descale_type)
    _validate_descale(v_descale, "v", batch_size, num_kv_heads, v_descale_type)

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
