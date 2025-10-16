"""
Variable-length attention implementation using Flash Attention.

This module provides a high-level Python interface for variable-length attention
that calls into the optimized Flash Attention kernels.
"""

import logging
from functools import lru_cache
from typing import NamedTuple, Optional, Union

import torch


log = logging.getLogger(__name__)

__all__ = ["varlen_attn", "AuxRequest"]


@lru_cache(maxsize=8)
def _should_use_cudnn(device_index: int) -> bool:
    """Cache device capability check to avoid repeated CUDA calls."""
    return False


class AuxRequest(NamedTuple):
    """
    Request which auxiliary outputs to compute from varlen_attn.

    Each field is a boolean indicating whether that auxiliary output should be computed.
    """

    lse: bool = False


# import failures when I try to register as custom op
# @torch.library.custom_op("torch_nn_attention::_varlen_attn", mutates_args={})
def _varlen_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Private custom op for variable-length attention.

    This is the internal implementation. Users should use the public varlen_attn function instead.
    """

    use_cudnn = query.is_cuda and _should_use_cudnn(query.device.index)

    if use_cudnn:
        log.info("Using cuDNN backend for varlen_attn")
        result = torch.ops.aten._cudnn_attention_forward(
            query,
            key,
            value,
            None,  # attn_bias
            cu_seq_q,
            cu_seq_k,
            max_q,
            max_k,
            True,  # compute_log_sumexp
            0.0,  # dropout_p hardcoded to 0.0
            is_causal,
            False,  # return_debug_mask
        )
        # cuDNN returns: (output, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, debug_attn_mask)
        output, softmax_lse = result[0], result[1]
    else:
        log.info("Using Flash Attention backend for varlen_attn")
        output, softmax_lse, rng_state, _, _ = torch.ops.aten._flash_attention_forward(
            query,
            key,
            value,
            cu_seq_q,
            cu_seq_k,
            max_q,
            max_k,
            0.0,  # dropout_p hardcoded to 0.0
            is_causal,
            return_debug_mask=False,
        )

    return output, softmax_lse


# @_varlen_attn.register_fake
def _varlen_attn_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fake implementation for meta tensor computation and tracing.

    Based on the 3D varlen path from meta__flash_attention_forward:
    - query shape: (total, num_heads, head_dim)
    - logsumexp shape: (num_heads, total_q)
    """
    # Output has same shape as query
    output = torch.empty_like(query)

    # For varlen path: logsumexp shape is (num_heads, total_q)
    total_q = query.size(0)
    num_heads = query.size(1)
    logsumexp = torch.empty(
        (num_heads, total_q), dtype=torch.float, device=query.device
    )

    return output, logsumexp


def varlen_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    is_causal: bool = False,
    return_aux: Optional[AuxRequest] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute variable-length attention using Flash Attention.
    This function is similar to scaled_dot_product_attention but optimized for
    variable-length sequences using cumulative sequence position tensors.
    Args:
    - query (Tensor): Query tensor; shape :math:`(T_q, H, D)`
    - key (Tensor): Key tensor; shape :math:`(T_k, H, D)`
    - value (Tensor): Value tensor; shape :math:`(T_k, H, D)`
    - cu_seq_q (Tensor): Cumulative sequence positions for queries; shape :math:`(N+1,)`
    - cu_seq_k (Tensor): Cumulative sequence positions for keys/values; shape :math:`(N+1,)`
    - max_q (int): Maximum query sequence length in the batch.
    - max_k (int): Maximum key/value sequence length in the batch.
    - is_causal (bool, optional): If set to True, applies causal masking (default: False).
    - return_aux (Optional[AuxRequest]): If not None and ``return_aux.lse`` is True, also returns the logsumexp tensor.

    Shape legend:
    - :math:`N`: Batch size
    - :math:`T_q`: Total number of query tokens in the batch (sum of all query sequence lengths)
    - :math:`T_k`: Total number of key/value tokens in the batch (sum of all key/value sequence lengths)
    - :math:`H`: Number of attention heads
    - :math:`D`: Head dimension

    Returns:
    - Tensor: Output tensor from attention computation
    - If ``return_aux`` is not None and ``return_aux.lse`` is True, returns a tuple of Tensors:
    (output, lse), where lse is the logsumexp

    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> batch_size, max_seq_len, embed_dim, num_heads = 2, 512, 1024, 16
        >>> head_dim = embed_dim // num_heads
        >>> seq_lengths = []
        >>> for _ in range(batch_size):
        ...     length = torch.randint(1, max_seq_len // 64 + 1, (1,)).item() * 64
        ...     seq_lengths.append(min(length, max_seq_len))
        >>> seq_lengths = torch.tensor(seq_lengths, device="cuda")
        >>> total_tokens = seq_lengths.sum().item()
        >>>
        >>> # Create packed query, key, value tensors
        >>> query = torch.randn(
        ...     total_tokens, num_heads, head_dim, dtype=torch.float16, device="cuda"
        ... )
        >>> key = torch.randn(
        ...     total_tokens, num_heads, head_dim, dtype=torch.float16, device="cuda"
        ... )
        >>> value = torch.randn(
        ...     total_tokens, num_heads, head_dim, dtype=torch.float16, device="cuda"
        ... )
        >>>
        >>> # Build cumulative sequence tensor
        >>> cu_seq = torch.zeros(batch_size + 1, device="cuda", dtype=torch.int32)
        >>> cu_seq[1:] = seq_lengths.cumsum(0)
        >>> max_len = seq_lengths.max().item()
        >>>
        >>> # Call varlen_attn
        >>> output = varlen_attn(
        ...     query, key, value, cu_seq, cu_seq, max_len, max_len, is_causal=False
        ... )
    """
    out, lse = _varlen_attn(
        query, key, value, cu_seq_q, cu_seq_k, max_q, max_k, is_causal
    )
    if return_aux is not None and return_aux.lse:
        return out, lse
    return out
