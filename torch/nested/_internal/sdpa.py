from typing import Optional
import logging
import torch
from torch.backends.cuda import (
    can_use_efficient_attention,
    can_use_flash_attention,
    flash_sdp_enabled,
    math_sdp_enabled,
    mem_efficient_sdp_enabled,
    SDPAParams,
    SDPBackend,
)

from .nested_tensor import NestedTensor

log = logging.getLogger(__name__)

def _validate_sdpa_input(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
):
    if (
        not isinstance(query, NestedTensor)
        or not isinstance(key, NestedTensor)
        or not isinstance(value, NestedTensor)
    ):
        raise ValueError(
            f"Expected query, key, and value to be nested tensors, "
            f"but got query.is_nested: {query.is_nested}, key.is_nested: {key.is_nested}, "
            f"and value.is_nested: {value.is_nested} instead."
        )
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
    if query._ragged_idx != 2 or key._ragged_idx != 2 or value._ragged_idx != 2:
        raise ValueError(
            f"Expected query, key, and value to all be be jagged at dimension 2, but got query._ragged_idx: "
            f"{query._ragged_idx}, key._ragged_idx: {key._ragged_idx} and value._ragged_idx: {value._ragged_idx} instead."
        )
    if attn_mask is not None:
        # TODO: Figure out whether masks are actually supported for this layout or not
        if attn_mask.dtype != torch.bool and attn_mask.dtype != query.dtype:
            raise ValueError(
                f"Expected attn_mask dtype to be bool or to match query dtype, but got attn_mask.dtype: "
                f"{attn_mask.dtype}, and query.dtype: {query.dtype} instead."
            )


def _can_use_math_sdpa_jagged(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
):
    if (
        not query.is_contiguous()
        or not key.is_contiguous()
        or not value.is_contiguous()
    ):
        raise ValueError("If inputs are nested tensors they must be contiguous.")
    if is_causal:
        raise ValueError(
            "Nested tensors for query / key are not supported when is_causal=True."
        )


def _select_sdp_backend(query, key, value, attn_mask, dropout, is_causal):
    if (
        not flash_sdp_enabled()
        and not mem_efficient_sdp_enabled()
        and not math_sdp_enabled()
    ):
        return SDPBackend.ERROR

    ordering = (
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    )

    params = SDPAParams(query, key, value, attn_mask, dropout, is_causal)

    for backend in ordering:
        if backend == SDPBackend.FLASH_ATTENTION:
            if can_use_flash_attention(params):
                return SDPBackend.FLASH_ATTENTION
        if backend == SDPBackend.EFFICIENT_ATTENTION:
            if can_use_efficient_attention(params):
                return SDPBackend.EFFICIENT_ATTENTION
        if backend == SDPBackend.EFFICIENT_ATTENTION:
            if math_sdp_enabled():
                return SDPBackend.MATH

    log.warning("Memory efficient kernel not used because:")
    can_use_efficient_attention(params, True)
    log.warning("Flash attention kernel not used because:")
    can_use_flash_attention(params, True)
    raise ValueError("No available kernel. Aborting execution.")


def _cumulative_and_max_seq_len_nnz(qkv):
    """
    This function is used to calculate two pieces of metadata that are needed
    for use with flash-attention and efficient_attention kernels. They are the
    cumulative sequence_length over a batch of sequences and the maximum
    sequence length.

    @return A tuple of cumulative sequence lengths and the maximum sequence
    length, and the last element in the cumulative_sequence_lengths
    """
    if not qkv.is_nested:
        raise ValueError("QKV must be nested for flash cumulative_seq_len calculation.")

    if qkv.lengths() is None:
        cumulative_seqlen = qkv.offsets().to(dtype=torch.int32, device=qkv.device)
        max_seqlen = torch.max(qkv.offsets().diff()).item()
    else:
        cumulative_seqlen = (
            qkv.lengths().cumsum(0).to(dtype=torch.int32, device=qkv.device)
        )
        batch_size = qkv.size(0)
        max_seqlen = qkv.values().size(0) / batch_size
    return cumulative_seqlen, max_seqlen, cumulative_seqlen[-1]


def _is_safe_to_get_storage_as_tensor(tensor):
    """
    This function checks if a nested tensor is valid for
    use with the flash-attention and efficient_attention kernels without
    needing to call contiguous on the nested tensor input.
    It checks that the storage offsets' adjacent_differences are a constant
    mutiple of the previous tensor in the nested tensor and that the strides
    are monitonically decreasing. This check is done after calling transpose on
    the nested tensor. Resulting in a Nt of shape [bsz, {seq_len}, num_heads, dim]

    @return A boolean indicating if contiguous needs to be called for input
    """
    offsets = tensor.offsets()
    strides = tensor._stride

    n_tensors = offsets.size(0) - 1
    if n_tensors <= 1:
        return True

    # Check initially that the tensor strides are in strictly descending order
    prev_stride = strides[1]
    for stride in strides[2:]:
        if prev_stride <= stride:
            # This would mean that the last stride is greater than the seq_len
            # stride
            return False
        prev_stride = stride

    # Check the offsets are a constant multiple from the previous numels
    offset_constant = offsets[1] - offsets[0]
    for i in range(2, len(offsets)):
        current_offset_constant = offsets[i] - offsets[i - 1]
        if current_offset_constant != offset_constant:
            return False

    # Congrats you made it!
    return True


def _view_as_dense(tensor, Nnz, num_heads, head_dim):
    if tensor.is_nested:
        return tensor.values().reshape(Nnz, num_heads, head_dim)
    return tensor.reshape(Nnz, num_heads, head_dim)


def _sdpa_nested_preprocessing_with_broadcast(query, key, value):
    # Query (Batch x Num_heads x {Q_seq_len}  x Dim_per_head)
    # Key   (Batch x Num_heads x {KV_seq_len} x Dim_per_head)
    # Value (Batch x Num_heads x {KV_seq_len} x Dim_per_head)
    q_batch_size = query.size(0)
    k_batch_size = key.size(0)
    v_batch_size = value.size(0)

    output_batch_size = max(q_batch_size, k_batch_size, v_batch_size)

    q_num_heads = query.size(1)
    k_num_heads = key.size(1)
    v_num_heads = value.size(1)

    output_num_heads = max(q_num_heads, k_num_heads, v_num_heads)

    head_dim_qk = query.size(3)
    head_dim_v = value.size(3)

    q_t = query.transpose(1, 2)
    k_t = key.transpose(1, 2)
    v_t = value.transpose(1, 2)

    # Checks in sdp_utils ensure that if {*}_batch_size/{*}_num_heads !=
    # output_batch_size/num_heads then they are 1
    q_batch_size_needs_broadcast = q_batch_size != output_batch_size
    k_batch_size_needs_broadcast = k_batch_size != output_batch_size
    v_batch_size_needs_broadcast = v_batch_size != output_batch_size

    # If {*}_batch_size_needs_broadcast, then
    # (1) max_seqlen_batch_{*} is given by {*}_t.size(1)
    #     this is because needs_broadcast indicates that the batch_size is 1
    #     and hence there is only 1 value for seq_len
    # (2) The cum_seq_lens are given by [0, {*}_t.size(1), 2 * {*}_t.size(1),
    # ..., outut_batch_size * {*}_t.size(1)]
    # (3) Nnz_{*} is given by output_batch_size * {*}_t.size(1)

    if q_batch_size_needs_broadcast or not q_t.is_nested:
        max_seqlen_batch_q = q_t.size(1)
        cumulative_sequence_length_q = torch.arange(
            0,
            (output_batch_size + 1) * max_seqlen_batch_q,
            max_seqlen_batch_q,
            device=q_t.device,
            dtype=torch.int32,
        )
        Nnz_q = output_batch_size * max_seqlen_batch_q
    else:
        (
            cumulative_sequence_length_q,
            max_seqlen_batch_q,
            Nnz_q,
        ) = _cumulative_and_max_seq_len_nnz(q_t)

    if k_batch_size_needs_broadcast and v_batch_size_needs_broadcast:
        assert k_t.size(1) == v_t.size(1)
        max_seqlen_batch_kv = k_t.size(1)
        cumulative_sequence_length_kv = torch.arange(
            0,
            (output_batch_size + 1) * max_seqlen_batch_kv,
            max_seqlen_batch_kv,
            device=k_t.device,
            dtype=torch.int32,
        )
        Nnz_kv = output_batch_size * max_seqlen_batch_kv
    else:
        cumulative_sequence_length_kv, max_seqlen_batch_kv, Nnz_kv = (
            _cumulative_and_max_seq_len_nnz(v_t)
            if k_batch_size_needs_broadcast
            else _cumulative_and_max_seq_len_nnz(k_t)
        )

    q_num_heads_needs_broadcast = q_num_heads != output_num_heads
    k_num_heads_needs_broadcast = k_num_heads != output_num_heads
    v_num_heads_needs_broadcast = v_num_heads != output_num_heads

    if not q_t.is_nested:
        query_buffer_reshaped = q_t.expand(
            output_batch_size, q_t.size(1), output_num_heads, head_dim_qk
        )
        query_buffer_reshaped = query_buffer_reshaped.reshape(
            Nnz_q, output_num_heads, head_dim_qk
        )
    else:
        if not q_t.is_contiguous() and not _is_safe_to_get_storage_as_tensor(q_t):
            q_t = q_t.contiguous()
        # If we are broadcasting then Nnz_q will be the output_batch_size since
        # seq_len is 1
        effective_batch_size_q = (
            output_batch_size if q_batch_size_needs_broadcast else Nnz_q
        )
        query_buffer_reshaped = _view_as_dense(
            q_t, effective_batch_size_q, output_num_heads, head_dim_qk
        )

    # If the physical layout of the NestedTensor's storage
    # is not: batch, {seq_len}, num_heads, head_dim then we need
    # to call contiguous
    if not k_t.is_contiguous() and not _is_safe_to_get_storage_as_tensor(k_t):
        k_t = k_t.contiguous()
    if not v_t.is_contiguous() and not _is_safe_to_get_storage_as_tensor(v_t):
        v_t = v_t.contiguous()

    effective_batch_size_k = (
        output_batch_size if k_batch_size_needs_broadcast else Nnz_kv
    )
    key_buffer_reshaped = _view_as_dense(
        k_t, effective_batch_size_k, output_num_heads, head_dim_qk
    )

    effective_batch_size_v = (
        output_batch_size if v_batch_size_needs_broadcast else Nnz_kv
    )
    value_buffer_reshaped = _view_as_dense(
        v_t, effective_batch_size_v, output_num_heads, head_dim_v
    )

    if not q_batch_size_needs_broadcast:
        output_shape = q_t._size
        if head_dim_v != head_dim_qk:
            output_shape[-1] = head_dim_v
        if q_num_heads_needs_broadcast:
            output_shape[1] = output_num_heads
    else:
        output_shape = torch.empty(3, dtype=torch.int64, device=torch.device("cpu"))
        output_shape[0] = q_t.size(1)
        output_shape[1] = output_num_heads
        output_shape[2] = head_dim_v

    return (
        query_buffer_reshaped,
        key_buffer_reshaped,
        value_buffer_reshaped,
        cumulative_sequence_length_q,
        cumulative_sequence_length_kv,
        max_seqlen_batch_q,
        max_seqlen_batch_kv,
        output_shape,
    )


def _sdpa_nested_preprocessing(query, key, value):
    # Query (Batch x Num_heads x {Q_seq_len}  x Dim_per_head)
    # Key   (Batch x Num_heads x {KV_seq_len} x Dim_per_head)
    # Value (Batch x Num_heads x {KV_seq_len} x Dim_per_head)
    q_batch_size = query.size(0)
    k_batch_size = key.size(0)
    v_batch_size = value.size(0)

    q_num_heads = query.size(1)
    k_num_heads = key.size(1)
    v_num_heads = value.size(1)

    if not (q_batch_size == k_batch_size and q_batch_size == v_batch_size) or not (
        q_num_heads == k_num_heads and k_num_heads == v_num_heads
    ):
        return _sdpa_nested_preprocessing_with_broadcast(query, key, value)

    num_heads = query.size(1)
    head_dim_qk = query.size(3)
    head_dim_v = value.size(3)

    q_t = query.transpose(1, 2)
    k_t = key.transpose(1, 2)
    v_t = value.transpose(1, 2)

    (
        cumulative_sequence_length_q,
        max_seqlen_batch_q,
        Nnz_q,
    ) = _cumulative_and_max_seq_len_nnz(q_t)
    (
        cumulative_sequence_length_kv,
        max_seqlen_batch_kv,
        Nnz_kv,
    ) = _cumulative_and_max_seq_len_nnz(k_t)

    # [TODO] K and V have to have the same Nnz, should probably torch_check
    # assume in order to not iterate over v

    # If the physical layout of the NestedTensor's storage
    # is not: batch, {seq_len}, num_heads, head_dim then we need
    # to call contiguous
    if not q_t.is_contiguous() and not _is_safe_to_get_storage_as_tensor(q_t):
        q_t = q_t.contiguous()
    if not k_t.is_contiguous() and not _is_safe_to_get_storage_as_tensor(k_t):
        k_t = k_t.contiguous()
    if not v_t.is_contiguous() and not _is_safe_to_get_storage_as_tensor(v_t):
        v_t = v_t.contiguous()

    query_buffer_reshaped = _view_as_dense(q_t, Nnz_q, num_heads, head_dim_qk)
    key_buffer_reshaped = _view_as_dense(k_t, Nnz_kv, num_heads, head_dim_qk)
    value_buffer_reshaped = _view_as_dense(v_t, Nnz_kv, num_heads, head_dim_v)

    output_shape = q_t._size
    if head_dim_v != head_dim_qk:
        output_shape[-1] = head_dim_v

    return (
        query_buffer_reshaped,
        key_buffer_reshaped,
        value_buffer_reshaped,
        cumulative_sequence_length_q,
        cumulative_sequence_length_kv,
        max_seqlen_batch_q,
        max_seqlen_batch_kv,
        output_shape,
    )


def jagged_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
):
    _validate_sdpa_input(query, key, value, attn_mask, dropout_p, is_causal, scale)
    compute_logsumexp = query.requires_grad or key.requires_grad or value.requires_grad

    backend_choice = _select_sdp_backend(
        query, key, value, attn_mask, dropout_p, is_causal
    )

    if backend_choice == SDPBackend.FLASH_ATTENTION:
        raise RuntimeError(
            "Dispatcher for nested tensors with jagged layout cannot run Flash Attention v2 just yet."
        )
    elif backend_choice == SDPBackend.EFFICIENT_ATTENTION:
        (
            query_reshaped,
            key_reshaped,
            value_reshaped,
            cumulative_sequence_length_q,
            cumulative_sequence_length_kv,
            max_seqlen_batch_q,
            _,
            output_shape,
        ) = _sdpa_nested_preprocessing(query, key, value)
        (
            attention,
            log_sumexp,
            seed,
            offset,
            max_seqlen_q,
            max_seqlen_batch_kv,
        ) = torch.ops.aten._efficient_attention_forward(
            query_reshaped.unsqueeze(0),
            key_reshaped.unsqueeze(0),
            value_reshaped.unsqueeze(0),
            None,
            cumulative_sequence_length_q,
            cumulative_sequence_length_kv,
            max_seqlen_batch_q,
            dropout_p,
            int(is_causal),
            compute_logsumexp,
            scale,
        )

        # Reshape output to convert nnz to batch_size and seq_len
        return attention.view(-1).reshape(*output_shape).transpose(1, 2)
    elif backend_choice == SDPBackend.MATH:
        _can_use_math_sdpa_jagged(query, key, value)
        return torch._scaled_dot_product_attention_math(
            query, key, value, attn_mask, dropout_p, is_causal, scale=scale
        )[0]
    else:
        raise RuntimeError(
            "No viable backend for scaled_dot_product_attention was found."
        )
