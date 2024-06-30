# mypy: allow-untyped-defs
import logging
from typing import Optional, Tuple

import torch
import torch.nn
import torch.nn.functional as F
from torch.backends.cuda import (
    can_use_efficient_attention,
    can_use_flash_attention,
    flash_sdp_enabled,
    math_sdp_enabled,
    mem_efficient_sdp_enabled,
    SDPAParams,
)
from torch.nn.attention import SDPBackend

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
    if query.dim() < 3 or key.dim() < 3 or value.dim() < 3:
        raise ValueError(
            f"Expected query, key, and value to all be  at least 3 dimensional, but got query.dim: "
            f"{query.dim()}, key.dim: {key.dim()} and value.dim: {value.dim()} instead."
        )
    if query._ragged_idx != key._ragged_idx or query._ragged_idx != value._ragged_idx:
        raise ValueError(
            f"Expected query, key, and value to all be ragged on the same dimension, but got ragged "
            f"dims {query._ragged_idx}, {key._ragged_idx}, and {value._ragged_idx}, respectively."
        )
    if attn_mask is not None:
        # TODO: Figure out whether masks are actually supported for this layout or not
        raise ValueError("Masks are not yet supported!")
        if attn_mask.dtype != torch.bool and attn_mask.dtype != query.dtype:
            raise ValueError(
                f"Expected attn_mask dtype to be bool or to match query dtype, but got attn_mask.dtype: "
                f"{attn_mask.dtype}, and query.dtype: {query.dtype} instead."
            )


def _check_batch_size_nested(params: SDPAParams, debug=False) -> bool:
    # This is expected to be called after check_tensor_shapes ensuring that the
    # size() calls won't error since the inputs are all 4 dimensional
    q_batch_size = params.query.size(0)
    k_batch_size = params.key.size(0)
    v_batch_size = params.value.size(0)

    # num_heads logic for nested input is checked in
    # check_for_seq_len_0_nested_tensor as there is handling there to make sure
    # num_heads is not ragged
    return q_batch_size == k_batch_size and q_batch_size == v_batch_size


def _check_head_dim_size_flash_nested(params: SDPAParams, debug=False) -> bool:
    max_size = 256
    query_size_last = params.query.size(-1)
    key_size_last = params.key.size(-1)
    value_size_last = params.value.size(-1)
    same_head_dim_size = (
        query_size_last == key_size_last and query_size_last == value_size_last
    )
    if not (
        same_head_dim_size
        and (query_size_last % 8 == 0)
        and (query_size_last <= max_size)
    ):
        if debug:
            log.warning(
                "For NestedTensor inputs, Flash attention requires q,k,v to have the same "
                "last dimension and to be a multiple of 8 and less than or equal to 256. "
                "Got Query.size(-1): %d, Key.size(-1): %d, Value.size(-1): %d instead.",
                query_size_last,
                key_size_last,
                value_size_last,
            )
        return False
    return True


def _check_for_seq_len_0_and_consistent_head_dim_nested_helper(
    param: torch.Tensor, param_name: str, debug=False
) -> bool:
    assert isinstance(param, NestedTensor), "param should be a jagged NT"

    if param._ragged_idx == 1:
        # num_head_dims is ragged
        if debug:
            log.warning(
                "Fused kernels do not support ragged num_head_dims, %s has a ragged num_heads.",
                param_name,
            )
        return False

    # This is being called inside sdp with shape [batch, heads, {seq_len}, dim]
    if param._get_min_seqlen() == 0:
        if debug:
            log.warning(
                "Fused kernels do not support seq_len == 0, %s has a seq len of 0.",
                param_name,
            )
        return False

    return True


def _try_broadcast_param_size(q_size, k_size, v_size, param_name, debug=False) -> bool:
    max_size = max(q_size, k_size, v_size)
    if (
        (q_size != max_size and q_size != 1)
        or (k_size != max_size and k_size != 1)
        or (v_size != max_size and v_size != 1)
    ):
        if debug:
            log.warning(
                "Both fused kernels require query, key and value to have broadcastable %s, "
                "got Query %s %d, Key %s %d, Value %s %d instead.",
                param_name,
                param_name,
                q_size,
                param_name,
                k_size,
                param_name,
                v_size,
            )
        return False
    return True


def _check_for_seq_len_0_nested(params: SDPAParams, debug=False) -> bool:
    # When this function is called we are assured that the nt is dim==4
    q_is_safe = (
        _check_for_seq_len_0_and_consistent_head_dim_nested_helper(
            params.query, "query", debug
        )
        if params.query.is_nested
        else True
    )
    # short circuit if any is unsafe
    if not q_is_safe:
        return False

    k_is_safe = (
        _check_for_seq_len_0_and_consistent_head_dim_nested_helper(
            params.key, "key", debug
        )
        if params.key.is_nested
        else True
    )
    # short circuit if any is unsafe
    if not k_is_safe:
        return False

    v_is_safe = (
        _check_for_seq_len_0_and_consistent_head_dim_nested_helper(
            params.value, "value", debug
        )
        if params.value.is_nested
        else True
    )
    # short circuit if any is unsafe
    if not v_is_safe:
        return False

    # We now know none of the inputs have ragged num_heads, so we can safely
    # access .size(1)
    q_num_heads = params.query.size(1)
    k_num_heads = params.key.size(1)
    v_num_heads = params.value.size(1)
    same_num_heads = q_num_heads == k_num_heads and q_num_heads == v_num_heads

    if not same_num_heads:
        if (
            params.query.requires_grad
            or params.key.requires_grad
            or params.value.requires_grad
        ):
            if debug:
                log.warning(
                    "Both fused kernels do not support training with broadcasted NT inputs."
                )
            return False
        return _try_broadcast_param_size(
            q_num_heads, k_num_heads, v_num_heads, "num heads", debug
        )
    return True


def _can_use_flash_sdpa_jagged(params: SDPAParams, debug=False) -> bool:
    constraints = (
        _check_batch_size_nested,
        _check_head_dim_size_flash_nested,
        _check_for_seq_len_0_nested,
    )
    for constraint in constraints:
        if not constraint(params, debug):
            return False
    return True


def _can_use_efficient_sdpa_jagged(params: SDPAParams, debug=False) -> bool:
    constraints = (
        _check_batch_size_nested,
        _check_for_seq_len_0_nested,
    )
    for constraint in constraints:
        if not constraint(params, debug):
            return False
    return True


def _can_use_math_sdpa_jagged(params: SDPAParams, debug=False) -> bool:
    if (
        not params.query.transpose(1, 2).is_contiguous()
        or not params.key.transpose(1, 2).is_contiguous()
        or not params.value.transpose(1, 2).is_contiguous()
    ):
        if debug:
            log.warning(
                "If inputs are nested tensors they must be contiguous after transposing."
            )
        return False
    if params.is_causal:
        if debug:
            log.warning(
                "Nested tensors for query / key are not supported when is_causal=True."
            )
        return False
    return True


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
            if can_use_flash_attention(params) and _can_use_flash_sdpa_jagged(params):
                return SDPBackend.FLASH_ATTENTION
        if backend == SDPBackend.EFFICIENT_ATTENTION:
            if can_use_efficient_attention(params) and _can_use_efficient_sdpa_jagged(
                params
            ):
                return SDPBackend.EFFICIENT_ATTENTION
        if backend == SDPBackend.MATH:
            if math_sdp_enabled() and _can_use_math_sdpa_jagged(params):
                return SDPBackend.MATH

    log.warning("Memory efficient kernel not used because:")
    can_use_efficient_attention(params, debug=True)
    _can_use_efficient_sdpa_jagged(params, debug=True)
    log.warning("Flash attention kernel not used because:")
    can_use_flash_attention(params, debug=True)
    _can_use_flash_sdpa_jagged(params, debug=True)
    log.warning("Math attention kernel not used because:")
    _can_use_math_sdpa_jagged(params, debug=True)
    return SDPBackend.ERROR


def _cumulative_and_max_seq_len_nnz(qkv: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    # This function is used to calculate two pieces of metadata that are needed
    # for use with flash-attention and efficient_attention kernels. They are the
    # cumulative sequence_length over a batch of sequences and the maximum
    # sequence length.

    # It returns a tuple of cumulative sequence lengths and the maximum sequence
    # length, and the last element in the cumulative_sequence_lengths
    if not isinstance(qkv, NestedTensor):
        raise ValueError("QKV must be nested for flash cumulative_seq_len calculation.")

    if qkv.lengths() is None:
        # TODO: Explore performance impact of copying
        cumulative_seqlen = qkv.offsets().to(dtype=torch.int32, device=qkv.device)
        max_seqlen = qkv._get_max_seqlen()
        n_elem = qkv.values().shape[0]
    else:
        # TODO: Explore performance impact of copying
        cumulative_seqlen = (
            qkv.lengths().cumsum(0).to(dtype=torch.int32, device=qkv.device)
        )
        batch_size = qkv.size(0)
        max_seqlen = qkv._get_max_seqlen()
        # TODO: Explore performance impact when compiling
        n_elem = int(cumulative_seqlen[-1].item())
    return cumulative_seqlen, max_seqlen, n_elem


def _is_safe_to_get_storage_as_tensor(tensor: torch.Tensor):
    # This function checks if a nested tensor is valid for
    # use with the flash-attention and efficient_attention kernels without
    # needing to call contiguous on the nested tensor input.
    # It checks that the storage offsets' adjacent_differences are a constant
    # mutiple of the previous tensor in the nested tensor and that the strides
    # are monitonically decreasing. This check is done after calling transpose on
    # the nested tensor resulting in a Nt of shape [bsz, {seq_len}, num_heads, dim]

    # Returns a boolean indicating if contiguous needs to be called for input
    assert isinstance(tensor, NestedTensor)
    offsets = tensor.offsets()
    strides = tensor._strides

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

    # Congrats you made it!
    return True


def _view_as_dense(
    tensor: torch.Tensor, Nnz: int, num_heads: int, head_dim: int
) -> torch.Tensor:
    if tensor.is_nested:
        return tensor.values()
    return tensor.view(Nnz, num_heads, head_dim)


# TODO: Next iteration should add test cases and check it works
# def _sdpa_nested_preprocessing_with_broadcast(query, key, value):
#     # Query (Batch x Num_heads x {Q_seq_len}  x Dim_per_head)
#     # Key   (Batch x Num_heads x {KV_seq_len} x Dim_per_head)
#     # Value (Batch x Num_heads x {KV_seq_len} x Dim_per_head)
#     q_batch_size = query.size(0)
#     k_batch_size = key.size(0)
#     v_batch_size = value.size(0)

#     output_batch_size = max(q_batch_size, k_batch_size, v_batch_size)

#     q_num_heads = query.size(1)
#     k_num_heads = key.size(1)
#     v_num_heads = value.size(1)

#     output_num_heads = max(q_num_heads, k_num_heads, v_num_heads)

#     head_dim_qk = query.size(3)
#     head_dim_v = value.size(3)

#     q_t = query.transpose(1, 2)
#     k_t = key.transpose(1, 2)
#     v_t = value.transpose(1, 2)

#     # Checks in sdp_utils ensure that if {*}_batch_size/{*}_num_heads !=
#     # output_batch_size/num_heads then they are 1
#     q_batch_size_needs_broadcast = q_batch_size != output_batch_size
#     k_batch_size_needs_broadcast = k_batch_size != output_batch_size
#     v_batch_size_needs_broadcast = v_batch_size != output_batch_size

#     # If {*}_batch_size_needs_broadcast, then
#     # (1) max_seqlen_batch_{*} is given by {*}_t.size(1)
#     #     this is because needs_broadcast indicates that the batch_size is 1
#     #     and hence there is only 1 value for seq_len
#     # (2) The cum_seq_lens are given by [0, {*}_t.size(1), 2 * {*}_t.size(1),
#     # ..., outut_batch_size * {*}_t.size(1)]
#     # (3) Nnz_{*} is given by output_batch_size * {*}_t.size(1)

#     if q_batch_size_needs_broadcast or not q_t.is_nested:
#         max_seqlen_batch_q = q_t.size(1)
#         cumulative_sequence_length_q = torch.arange(
#             0,
#             (output_batch_size + 1) * max_seqlen_batch_q,
#             max_seqlen_batch_q,
#             device=q_t.device,
#             dtype=torch.int32,
#         )
#         Nnz_q = output_batch_size * max_seqlen_batch_q
#     else:
#         (
#             cumulative_sequence_length_q,
#             max_seqlen_batch_q,
#             Nnz_q,
#         ) = _cumulative_and_max_seq_len_nnz(q_t)

#     if k_batch_size_needs_broadcast and v_batch_size_needs_broadcast:
#         assert k_t.size(1) == v_t.size(1)
#         max_seqlen_batch_kv = k_t.size(1)
#         cumulative_sequence_length_kv = torch.arange(
#             0,
#             (output_batch_size + 1) * max_seqlen_batch_kv,
#             max_seqlen_batch_kv,
#             device=k_t.device,
#             dtype=torch.int32,
#         )
#         Nnz_kv = output_batch_size * max_seqlen_batch_kv
#     else:
#         cumulative_sequence_length_kv, max_seqlen_batch_kv, Nnz_kv = (
#             _cumulative_and_max_seq_len_nnz(v_t)
#             if k_batch_size_needs_broadcast
#             else _cumulative_and_max_seq_len_nnz(k_t)
#         )

#     q_num_heads_needs_broadcast = q_num_heads != output_num_heads
#     k_num_heads_needs_broadcast = k_num_heads != output_num_heads
#     v_num_heads_needs_broadcast = v_num_heads != output_num_heads

#     if not q_t.is_nested:
#         query_buffer_reshaped = q_t.expand(
#             output_batch_size, q_t.size(1), output_num_heads, head_dim_qk
#         )
#         query_buffer_reshaped = query_buffer_reshaped.reshape(
#             Nnz_q, output_num_heads, head_dim_qk
#         )
#     else:
#         if not q_t.is_contiguous() and not _is_safe_to_get_storage_as_tensor(q_t):
#             q_t = q_t.contiguous()
#         # If we are broadcasting then Nnz_q will be the output_batch_size since
#         # seq_len is 1
#         effective_batch_size_q = (
#             output_batch_size if q_batch_size_needs_broadcast else Nnz_q
#         )
#         query_buffer_reshaped = _view_as_dense(
#             q_t, effective_batch_size_q, output_num_heads, head_dim_qk
#         )

#     # If the physical layout of the NestedTensor's storage
#     # is not: batch, {seq_len}, num_heads, head_dim then we need
#     # to call contiguous
#     if not k_t.is_contiguous() and not _is_safe_to_get_storage_as_tensor(k_t):
#         k_t = k_t.contiguous()
#     if not v_t.is_contiguous() and not _is_safe_to_get_storage_as_tensor(v_t):
#         v_t = v_t.contiguous()

#     effective_batch_size_k = (
#         output_batch_size if k_batch_size_needs_broadcast else Nnz_kv
#     )
#     key_buffer_reshaped = _view_as_dense(
#         k_t, effective_batch_size_k, output_num_heads, head_dim_qk
#     )

#     effective_batch_size_v = (
#         output_batch_size if v_batch_size_needs_broadcast else Nnz_kv
#     )
#     value_buffer_reshaped = _view_as_dense(
#         v_t, effective_batch_size_v, output_num_heads, head_dim_v
#     )

#     if not q_batch_size_needs_broadcast:
#         output_shape = q_t._size
#         if head_dim_v != head_dim_qk:
#             output_shape[-1] = head_dim_v
#         if q_num_heads_needs_broadcast:
#             output_shape[1] = output_num_heads
#     else:
#         output_shape = torch.empty(3, dtype=torch.int64, device=torch.device("cpu"))
#         output_shape[0] = q_t.size(1)
#         output_shape[1] = output_num_heads
#         output_shape[2] = head_dim_v

#     return (
#         query_buffer_reshaped,
#         key_buffer_reshaped,
#         value_buffer_reshaped,
#         cumulative_sequence_length_q,
#         cumulative_sequence_length_kv,
#         max_seqlen_batch_q,
#         max_seqlen_batch_kv,
#         output_shape,
#     )


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
        raise RuntimeError(
            "This path is currently not implemented for jagged layout NT."
        )
        # return _sdpa_nested_preprocessing_with_broadcast(query, key, value)

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

    output_nt_info = {
        "offsets": q_t.offsets(),
        "_max_seqlen": q_t._get_max_seqlen(),
        "_min_seqlen": q_t._get_min_seqlen(),
    }

    return (
        query_buffer_reshaped,
        key_buffer_reshaped,
        value_buffer_reshaped,
        cumulative_sequence_length_q,
        cumulative_sequence_length_kv,
        max_seqlen_batch_q,
        max_seqlen_batch_kv,
        output_nt_info,
    )


def _pad_last_dim(
    tensor: torch.Tensor, alignment_size: int, slice: bool
) -> torch.Tensor:
    # FlashAttentionV2 requires that head dimension be a multiple of 8
    # This was previously done within the kernel, however
    # This causes the kernel to maybe alias query, key, value
    # So instead we pad the head_dimensions to be a multiple of 8
    # in the composite region
    last_dim_size = tensor.size(-1)
    if last_dim_size % alignment_size == 0:
        return tensor
    pad_count = alignment_size - (last_dim_size % alignment_size)
    tensor = torch.nn.functional.pad(tensor, [0, pad_count])
    if slice:
        return tensor[..., 0:last_dim_size]
    return tensor


# TODO: coalesce with torch/nn/utils/attention.py
def _calculate_scale(query, scale):
    # TODO: Investigate why math.sqrt() isn't properly handled by Dynamo?
    softmax_scale = scale if scale is not None else torch.sym_sqrt(1.0 / query.size(-1))
    return softmax_scale


def _post_process_flash_output(out: torch.Tensor, og_size):
    if not out.is_nested and out.size(-1) != og_size:
        out = out[..., 0:og_size]
    return out


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
    # for mypy, ugh
    assert (
        isinstance(query, NestedTensor)
        and isinstance(key, NestedTensor)
        and isinstance(value, NestedTensor)
    )
    from torch.nested._internal.nested_tensor import nested_view_from_values_offsets

    # Special path for non-ragged sequence length (e.g. for SAM where we have a ragged
    # second batch dim instead). For this case, we can just send the dense buffers through
    # vanilla SDPA.
    if query.dim() > 3 and key.dim() > 3 and value.dim() > 3 and query._ragged_idx == 1:
        output = F.scaled_dot_product_attention(
            query.values(),
            key.values(),
            value.values(),
            attn_mask=(
                attn_mask.values() if isinstance(attn_mask, NestedTensor) else attn_mask
            ),
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )
        return nested_view_from_values_offsets(output, query.offsets())

    compute_logsumexp = query.requires_grad or key.requires_grad or value.requires_grad

    backend_choice = _select_sdp_backend(
        query, key, value, attn_mask, dropout_p, is_causal
    )

    if backend_choice == SDPBackend.FLASH_ATTENTION:
        og_size = query.size(-1)
        query_padded = _pad_last_dim(query, 8, False)
        key_padded = _pad_last_dim(key, 8, False)
        value_padded = _pad_last_dim(value, 8, False)
        # We need to calculate the scale based off the OG head dim size
        og_scale = _calculate_scale(query, scale)
        (
            query_buffer_reshaped,
            key_buffer_reshaped,
            value_buffer_reshaped,
            cumulative_sequence_length_q,
            cumulative_sequence_length_kv,
            max_seqlen_batch_q,
            max_seqlen_batch_kv,
            output_nt_info,
        ) = _sdpa_nested_preprocessing(query_padded, key_padded, value_padded)

        (
            attention,
            logsumexp,
            philox_seed,
            philox_offset,
            debug_attn_mask,
        ) = torch.ops.aten._flash_attention_forward(
            query_buffer_reshaped,
            key_buffer_reshaped,
            value_buffer_reshaped,
            cumulative_sequence_length_q,
            cumulative_sequence_length_kv,
            max_seqlen_batch_q,
            max_seqlen_batch_kv,
            dropout_p,
            is_causal,
            False,
            scale=og_scale,
        )

        # Reshape output to convert nnz to batch_size and seq_len
        attention = nested_view_from_values_offsets(
            attention.squeeze(0),
            output_nt_info["offsets"],
            min_seqlen=output_nt_info["_min_seqlen"],
            max_seqlen=output_nt_info["_max_seqlen"],
        ).transpose(1, 2)
        return _post_process_flash_output(attention, og_size)
    elif backend_choice == SDPBackend.EFFICIENT_ATTENTION:
        (
            query_reshaped,
            key_reshaped,
            value_reshaped,
            cumulative_sequence_length_q,
            cumulative_sequence_length_kv,
            max_seqlen_batch_q,
            max_seqlen_batch_kv,
            output_nt_info,
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
            max_seqlen_batch_kv,
            dropout_p,
            int(is_causal),
            compute_logsumexp,
            scale=scale,
        )

        # Reshape output to convert nnz to batch_size and seq_len
        return nested_view_from_values_offsets(
            attention.squeeze(0),
            output_nt_info["offsets"],
            min_seqlen=output_nt_info["_min_seqlen"],
            max_seqlen=output_nt_info["_max_seqlen"],
        ).transpose(1, 2)
    elif backend_choice == SDPBackend.MATH:
        # save the offsets and shape of the inputs, so we can reshape the final output
        # query @ key = attn: [B, D1, j0, D'] @ [B, D1, D' j1] = [B, D1, j0, j1]
        # attn @ value = out: [B, D1, j0, j1] @ [B, D1, j1, D2] = [B, D1, j0, D2]
        offsets = query.offsets()
        d1 = query._size[1]
        d2 = value._size[-1]

        min_seqlen_tensor = query._metadata_cache.get(
            "min_seqlen", None
        )  # type: ignore[attr-defined]
        max_seqlen_tensor = query._metadata_cache.get(
            "max_seqlen", None
        )  # type: ignore[attr-defined]

        # convert jagged layout Nested Tensor to strided layout Nested Tensor
        # which support the math implementation of SDPA
        def get_strided_layout_nested_tensor(jagged_layout_nt):
            lengths = jagged_layout_nt._offsets[1:] - jagged_layout_nt._offsets[:-1]
            transpose = torch.transpose(jagged_layout_nt, 1, 2)
            tensor_list = transpose.values().split(list(lengths), dim=0)
            strided_nt = torch.nested.as_nested_tensor(list(tensor_list))
            strided_nt = strided_nt.transpose(1, 2).contiguous()
            return strided_nt

        query = get_strided_layout_nested_tensor(query)
        key = get_strided_layout_nested_tensor(key)
        value = get_strided_layout_nested_tensor(value)

        attn_out = torch._scaled_dot_product_attention_math(
            query, key, value, attn_mask, dropout_p, is_causal, scale=scale
        )[0]

        from torch.nested._internal.nested_tensor import _load_val_from_tensor

        # convert strided layout Nested Tensor back to jagged layout Nested Tensor
        attn_out = attn_out.transpose(1, 2).contiguous().values()
        attn_out = attn_out.view(-1, d1, d2)
        attn_out = nested_view_from_values_offsets(
            attn_out,
            offsets,
            min_seqlen=(
                None
                if min_seqlen_tensor is None
                else _load_val_from_tensor(min_seqlen_tensor)
            ),
            max_seqlen=(
                None
                if max_seqlen_tensor is None
                else _load_val_from_tensor(max_seqlen_tensor)
            ),
        ).transpose(1, 2)

        return attn_out
    else:
        raise RuntimeError(
            "No viable backend for scaled_dot_product_attention was found."
        )
