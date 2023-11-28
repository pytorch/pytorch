import functools
import logging
import math

import torch

from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.backends.cuda import (
    can_use_efficient_attention,
    can_use_flash_attention,
    flash_sdp_enabled,
    math_sdp_enabled,
    mem_efficient_sdp_enabled,
    SDPAParams,
    SDPBackend,
)
from torch.fx.operator_schemas import normalize_function

log = logging.getLogger(__name__)

__all__: List[Any] = []

JAGGED_OPS_TABLE: Dict[Any, Any] = {}


# Simplifying assumption: we assume that the batch dim is always the left-most
# dim, and the ragged dim is always the second dim.
def _outer_to_inner_dim(ndim, dim):
    assert dim >= 0 and dim < ndim
    return 0 if dim < 2 else dim - 1


def _wrap_jagged_dim(ndim, dim, op_name):
    from torch._prims_common import canonicalize_dims

    wrapped = canonicalize_dims(ndim, dim)
    if wrapped < 2:
        raise RuntimeError(
            f"{op_name}(): not supported for NestedTensor on dim=0 or dim=1"
        )
    return _outer_to_inner_dim(ndim, wrapped)


def _wrap_jagged_dims(ndim, dims, op_name):
    # ex: (2, 3, 4) -> (1, 2, 3)
    # ex: (0, 1, 4) -> (0, 3)
    from torch._prims_common import canonicalize_dims

    wrapped_dims = [canonicalize_dims(ndim, d) for d in dims]
    # This logic needs to be done after we canonicalize dims but before we
    # map to inner dims so we can print a nicer error message.
    zero_in_dims = 0 in wrapped_dims
    one_in_dims = 1 in wrapped_dims
    if zero_in_dims ^ one_in_dims:
        apply, not_apply = ("batch", "ragged") if zero_in_dims else ("ragged", "batch")
        raise RuntimeError(
            f"{op_name}(): applying over the {apply} dimension, but not the {not_apply}"
            " dimension is not supported for NestedTensor"
        )
    return (
        tuple(_outer_to_inner_dim(ndim, d) for d in dims if d != 0),
        zero_in_dims,
    )


def check_schema(schema_str: str, func, *args, **kwargs) -> None:
    named_arg_types = schema_str.split(", ")
    num_optional_args = sum([x.endswith("?") for x in named_arg_types])
    min_args = len(named_arg_types) - num_optional_args

    if not (len(args) >= min_args and len(args) <= len(named_arg_types)):
        raise ValueError(
            f"NestedTensor {func.__name__}({schema_str}): expected at least {min_args} "
            f"arguments and at most {len(named_arg_types)} arguments, but got: "
            f"{len(args)} arguments"
        )

    arg_type_check_fns = {
        "t": lambda x: isinstance(x, torch.Tensor) and not isinstance(x, NestedTensor),
        "jt": lambda x: isinstance(x, NestedTensor)
        and x._lengths is None,  # ops with "jt" require contiguous JT only
        "jt_all": lambda x: isinstance(
            x, NestedTensor
        ),  # ops with "jt_all" can accept all kinds of JT
        "any": lambda x: True,
    }
    for i, named_arg_type in enumerate(named_arg_types):
        name, arg_type = named_arg_type.split(": ")
        is_optional = arg_type.endswith("?")
        normalized_arg_type = arg_type[:-1] if is_optional else arg_type
        if normalized_arg_type not in arg_type_check_fns.keys():
            raise AssertionError(f"Unknown arg type: {normalized_arg_type}")

        if i >= len(args):
            if not is_optional:
                raise ValueError(
                    f"NestedTensor {func.__name__}({schema_str}) "
                    f"missing required argument: {name}"
                )
            continue

        if not arg_type_check_fns[normalized_arg_type](args[i]):
            raise ValueError(
                f"NestedTensor {func.__name__}({schema_str}): {name} should be of "
                f"type {arg_type}, but got: {type(args[i])}"
            )


def check_ragged_dim_same(
    func, a: NestedTensor, a_name: str, b: NestedTensor, b_name: str
) -> None:
    # Calling into .shape here
    if a._size[a._ragged_idx] != b._size[b._ragged_idx]:
        raise RuntimeError(
            f"NestedTensor {func.__name__}: expected {a_name} and {b_name} to have the "
            "same exact offsets tensor."
        )


# returns True if the raggedness-relevant portions of the NT shape
# match those of the specified size
def raggedness_matches(nt, size):
    end = nt._ragged_idx + 1
    return list(nt._size[:end]) == list(size[:end])


def squeeze_leading_ones(t):
    # Note: [ Squeezing leading ones ]
    #
    # Squeeze leading ones from t.
    #
    # We want:
    #   (B, j0, ?, ?) + (1, 1, ?, ?) -> (B, j0, ?, ?)
    #   (B, j0, ?, ?) + (1, 1, 1, ?, ?) -> (1, B, j0, ?, ?)  (not yet supported)
    #
    # 1) Squeeze extra ones and grab values from NT
    #   (1, 1, ?, ?) -> (?, ?)   and   (sum(*), ?, ?) -> (B, j0, ?, ?)
    # 2) Do dense broadcasting:
    #   (sum(*), ?, ?) + (?, ?) -> (sum(*), ?, ?)
    # 3) Construct nested tensor
    #   (sum(*), ?, ?) -> (B, j0, ?, ?)
    #
    # If unsqueezing on the 0th dim becomes supported, we would unsqueeze
    # at step (4) and we would need to update this function to record how
    # many ones we unsqueezed.
    while t.shape[0] == 1:
        t = t.squeeze(0)
    return t


def register_func(tables, aten_ops, schema_str):
    if not isinstance(aten_ops, list):
        aten_ops = [aten_ops]
    if not isinstance(tables, list):
        tables = [tables]

    def wrapper(func):
        for aten_op in aten_ops:

            def get_inner(aten_op):
                def inner(*args, **kwargs):
                    check_schema(schema_str, func, *args, **kwargs)
                    return func(aten_op, *args, **kwargs)

                return inner

            for table in tables:
                table[aten_op] = get_inner(aten_op)
        return func

    return wrapper


register_jagged_func = functools.partial(register_func, JAGGED_OPS_TABLE)


def lookup_jagged(func, *args, **kwargs) -> Optional[Callable]:
    dispatch_func = JAGGED_OPS_TABLE.get(func, None)
    if dispatch_func is not None:
        return dispatch_func

    # Handle pointwise fallbacks
    if torch.Tag.pointwise in func.tags:
        # Assume there aren't additional tensors that aren't the "unary/binary" args
        num_tensor_args = sum([isinstance(x, torch.Tensor) for x in args])
        if num_tensor_args == 1:
            return functools.partial(jagged_unary_pointwise, func)
        elif num_tensor_args == 2:
            check_schema("lhs: any, rhs: any", func, *args, **kwargs)
            return functools.partial(jagged_binary_pointwise, func)

    return None


def extract_kwargs(arg):
    kwargs = {
        "offsets": arg.offsets(),
    }
    return kwargs


def jagged_unary_pointwise(func, *args, **kwargs):
    return NestedTensor(
        func(args[0]._values, *args[1:], **kwargs), **extract_kwargs(args[0])
    )


def jagged_binary_pointwise(func, *args, **kwargs):
    a, b = args[0], args[1]
    assert isinstance(a, NestedTensor) or isinstance(b, NestedTensor)

    mismatch_error_msg = (
        f"cannot call binary pointwise function {func.__name__} with inputs of shapes "
        f"{a.shape} and {b.shape}"
    )

    # a is NT, b is NT
    if isinstance(a, NestedTensor) and isinstance(b, NestedTensor):
        # ex: (B, j0, D) + (B, j0, D)
        # ex: (B, j0, D) + (B, j0, 1)
        if raggedness_matches(a, b.shape):
            return NestedTensor(
                func(a._values, b._values, *args[2:], **kwargs), **extract_kwargs(a)
            )
        raise RuntimeError(mismatch_error_msg)

    # either a is NT or b is NT at this point
    a_is_nt = isinstance(a, NestedTensor)
    extracted_kwargs = extract_kwargs(a) if a_is_nt else extract_kwargs(b)

    # === Handle broadcasting across the batch / ragged dims ===

    # Easy case: take advantage of pre-existing broadcasting logic
    # ex: (B, j0, ?, ?) + (?) -> (B, j0, ?, ?)
    # ex: (B, j0, ?, ?) + (?, ?) -> (B, j0, ?, ?)
    # ex: (B, j0, ?, ?) + (1, 1, ?, ?) -> (B, j0, ?, ?)
    nt, t = (a, b) if a_is_nt else (b, a)
    # See Note: [ Squeezing leading ones ]
    if t.dim() > nt.dim():
        raise NotImplementedError("NYI: broadcasting NT with T with larger dim")
    t_squeezed = squeeze_leading_ones(t)
    if nt.dim() >= t_squeezed.dim() + 2:
        lhs, rhs = (nt._values, t_squeezed) if a_is_nt else (t_squeezed, nt._values)
        return NestedTensor(func(lhs, rhs, *args[2:], **kwargs), **extracted_kwargs)

    # Harder case: do manual broadcasting over unbound components
    # when NT dim == non-NT dim
    # ex: (B, j0, D_0, D_1) + (B, 1, D_0, D_1) -> (B, j0, D_0, D_1)
    if a.dim() == b.dim():
        # ex: (B, j0, D_0, D_1) + (1, 1, D_0, D_1) -> should
        # be (B, j0, D_0, D_1) but not yet supported
        if a.shape[0] != b.shape[0]:
            raise RuntimeError(mismatch_error_msg)

        # need to use offsets to broadcast across ragged dim properly
        # NB: inefficient fallback here; Triton codegen can help this
        # TODO: Make this work with autograd
        outputs = []
        for a_comp, b_comp in zip(a.unbind(), b.unbind()):
            outputs.append(func(a_comp, b_comp, *args[2:], **kwargs))
        new_values = torch.cat(outputs, dim=0)
        return NestedTensor(new_values, **extracted_kwargs)

    # ex: (B, j0, D_0, D_1) + (A, B, 1, D_0, D_1) -> error because this breaks the invariant
    # that ragged dim is wrt left-most batch dim
    raise RuntimeError(mismatch_error_msg)


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
        ) = torch._efficient_attention_forward(  # type: ignore[attr-defined]
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


def jagged_torch_function(func, *args, **kwargs):
    # SDPA has special kernels that handle nested tensors.
    # Dispatch to the correct implementation here
    if func is torch._C._nn.scaled_dot_product_attention:
        return jagged_scaled_dot_product_attention(*args, **kwargs)

    # Handle flatten() here because it's CompositeImplicit.
    if func.__name__ == "flatten":

        def _flatten_sig(input, start_dim=0, end_dim=-1):
            pass

        _, new_kwargs = normalize_function(
            _flatten_sig, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
        )

        inp = new_kwargs.pop("input")
        new_kwargs["start_dim"] = _wrap_jagged_dim(
            inp.dim(), new_kwargs["start_dim"], "flatten"
        )
        new_kwargs["end_dim"] = _wrap_jagged_dim(
            inp.dim(), new_kwargs["end_dim"], "flatten"
        )

        return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))

    raise NotImplementedError(func)


@register_jagged_func(
    [
        torch.ops.aten.is_non_overlapping_and_dense.default,
        torch.ops.aten.sym_size.default,
        torch.ops.aten.dim.default,
        torch.ops.aten.sym_numel.default,
        torch.ops.aten.sym_stride.default,
        torch.ops.aten.sym_storage_offset.default,
    ],
    "self: jt_all",
)
def tensor_attr_supported_getter(func, *args, **kwargs):
    if func == torch.ops.aten.is_non_overlapping_and_dense.default:
        return False

    if func == torch.ops.aten.sym_size.default:
        return args[0]._size

    if func == torch.ops.aten.dim.default:
        return len(args[0]._size)

    if func == torch.ops.aten.sym_numel.default:
        if args[0]._lengths is not None:
            return int(sum(args[0]._lengths) * math.prod(args[0]._size[2:]))
        return args[0]._values.numel()

    if func == torch.ops.aten.sym_stride.default:
        return args[0]._strides

    if func == torch.ops.aten.sym_storage_offset.default:
        return args[0]._values.storage_offset()


@register_jagged_func(torch.ops.prim.layout.default, "self: jt_all")
def prim_layout_default(func, *args, **kwargs):
    return torch.jagged


@register_jagged_func(
    [torch.ops.aten.size.default],
    "self: jt_all",
)
def tensor_attr_unsupported_getter(func, *args, **kwargs):
    if func == torch.ops.aten.size.default:
        raise RuntimeError(
            "NestedTensors does not support directly calling torch.ops.aten.size "
            "please use `nested_tensor.size()` instead."
        )


@register_jagged_func(torch.ops.aten.is_contiguous.default, "self: jt_all")
def is_contiguous_general(func, *args, **kwargs):
    from torch._prims_common import is_contiguous_for_memory_format

    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    inp = new_kwargs.pop("input")

    # If created from narrow() check for lengths
    if inp.lengths() is not None:
        return False

    # If jagged dim is not 1 it's not contiguous
    if inp._ragged_idx != 1:
        return False

    new_kwargs["memory_format"] = new_kwargs.get(
        "memory_format", torch.contiguous_format
    )
    if new_kwargs["memory_format"] == torch.preserve_format:
        return True
    return is_contiguous_for_memory_format(inp.values(), **new_kwargs)


register_jagged_func(
    torch.ops.aten.is_contiguous.memory_format, "self: jt_all, memory_format: any?"
)(is_contiguous_general)


@register_jagged_func(torch.ops.aten.linear.default, "input: jt, weight: t, bias: t?")
def linear_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    weight = new_kwargs["weight"]
    bias = new_kwargs["bias"]

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(
    torch.ops.aten.linear_backward.default,
    "self: jt, grad_output: jt, weight: t, output_mask: any",
)
def linear_backward_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    grad_output = new_kwargs.pop("grad_output")
    weight = new_kwargs.pop("weight")

    check_ragged_dim_same(func, inp, "self", grad_output, "grad_output")
    ds = NestedTensor(
        torch.mm(grad_output._values, weight), **extract_kwargs(grad_output)
    )
    dw = torch.mm(grad_output._values.T, inp._values)
    db = None  # NYI: gradient for bias, need to reduce over ragged dim
    return (ds, dw, db)


@register_jagged_func(torch.ops.aten._to_copy.default, "self: jt")
def to_copy_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    # don't change layout
    new_kwargs.pop("layout")

    new_values = func(inp._values, **new_kwargs)
    # NB: Purposefully keep offsets on the old device.
    return NestedTensor(new_values, **extract_kwargs(inp))


register_jagged_func(
    [
        torch.ops.aten.ones_like.default,
        torch.ops.aten.zeros_like.default,
        torch.ops.aten.randn_like.default,
        torch.ops.aten.detach.default,
    ],
    "self: jt",
)(jagged_unary_pointwise)


register_jagged_func(
    torch.ops.aten._softmax.default, "self: jt, dim: any, half_to_float: any"
)(jagged_unary_pointwise)


@register_jagged_func(
    torch.ops.aten.native_dropout.default, "self: jt, float: any, train: any?"
)
def native_dropout_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    out1, out2 = func(inp._values, **new_kwargs)
    return (
        NestedTensor(out1, **extract_kwargs(inp)),
        NestedTensor(out2, **extract_kwargs(inp)),
    )


@register_jagged_func(
    torch.ops.aten.native_dropout_backward.default,
    "grad_output: jt, mask: jt, scale: any",
)
def native_dropout_backward_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    grad_output = new_kwargs.pop("grad_output")
    mask = new_kwargs.pop("mask")
    return NestedTensor(
        func(grad_output._values, mask._values, **new_kwargs),
        **extract_kwargs(grad_output),
    )


@register_jagged_func(torch.ops.aten.prod.dim_int, "self: jt, dim: any, keepdim: any?")
def prod_dim_int(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    # TODO: Figure out how to handle this better
    # keep_dim is required to keep it in jagged format
    if not new_kwargs["keepdim"]:
        raise RuntimeError("prod(): keepdim=True must be set for NestedTensor")
    dim = new_kwargs["dim"]
    new_kwargs["dim"] = _wrap_jagged_dim(len(inp.shape), dim, "prod")

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(args[0]))


@register_jagged_func(
    torch.ops.aten.split.Tensor, "self: jt, split_size: any, dim: any"
)
def split_tensor(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    new_kwargs["dim"] = _wrap_jagged_dim(inp.dim(), new_kwargs["dim"], "split")

    return tuple(
        NestedTensor(values=x, **extract_kwargs(inp))
        for x in func(inp._values, **new_kwargs)
    )


@register_jagged_func(
    torch.ops.aten.split_with_sizes.default, "self: jt, split_sizes: any, dim: any"
)
def split_with_sizes_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    new_kwargs["dim"] = _wrap_jagged_dim(
        inp.dim(), new_kwargs["dim"], "split_with_sizes"
    )

    return [
        NestedTensor(values=x, **extract_kwargs(inp))
        for x in func(inp._values, **new_kwargs)
    ]


@register_jagged_func(torch.ops.aten.unbind.int, "self: jt_all, dim: any?")
def unbind_int(func, *args, **kwargs):
    # Note that this specializes on the length of the offsets
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    dim = new_kwargs["dim"]
    if dim != 0:
        raise RuntimeError("unbind(): only supported for NestedTensor on dim=0")

    inp = new_kwargs.pop("input")
    values = inp.values()
    offsets = inp.offsets()
    lengths = inp.lengths()

    if inp._ragged_idx != 1:
        raise RuntimeError(
            "unbind(): only supported for NestedTensor when jagged dimension is 1"
        )

    if lengths is None:
        return torch.split(values, offsets.diff().tolist())
    return [
        values[offsets[i] : (offsets[i] + lengths[i])] for i in range(lengths.shape[0])
    ]


@register_jagged_func(torch.ops.aten.unsqueeze.default, "self: jt, dim: any")
def unsqueeze_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    values = inp._values
    offsets = inp.offsets

    # Account for collapsed jagged dim
    dim = new_kwargs["dim"]
    new_kwargs["dim"] = _wrap_jagged_dim(len(inp.shape) + 1, dim, "unsqueeze")
    return NestedTensor(func(values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(torch.ops.aten.cat.default, "tensors: any, dim: any")
def cat_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    tensors = new_kwargs.pop("tensors")

    # Convert any non-nested to nested
    nested = [t for t in tensors if t.is_nested]
    assert len(nested) > 0
    first = nested[0]
    tensors = [t if t.is_nested else t.expand_as(first) for t in tensors]

    # Account for collapsed jagged dim
    dim = new_kwargs["dim"]
    new_kwargs["dim"] = _wrap_jagged_dim(len(first.shape), dim, "cat")

    return NestedTensor(
        func([t._values for t in tensors], **new_kwargs), **extract_kwargs(tensors[0])
    )


@register_jagged_func(torch.ops.aten.matmul.default, "self: jt, other: any")
def matmul_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    other = new_kwargs.pop("other")

    if inp.is_nested and not other.is_nested:
        return NestedTensor(
            func(inp._values, other, **new_kwargs), **extract_kwargs(inp)
        )
    elif inp.is_nested and other.is_nested:
        # BMM with equivalent ragged dims between the two inputs
        if inp.dim() > 3 and other.dim() > 3 and raggedness_matches(inp, other.shape):
            return NestedTensor(func(inp._values, other._values), **extract_kwargs(inp))

    raise RuntimeError(
        f"matmul(): not supported between inputs of shapes {inp.shape} and {other.shape}"
    )


@register_jagged_func(
    torch.ops.aten.expand.default, "self: jt, size: any, implicit: any?"
)
def expand_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    size = new_kwargs["size"]

    assert ("implicit" not in new_kwargs) or (not new_kwargs.pop("implicit"))
    if not raggedness_matches(inp, size):
        raise RuntimeError(f"expand(): cannot expand shape {inp.shape} -> {size}")

    expand_arg = [-1, *size[2:]]
    return NestedTensor(func(inp._values, expand_arg), **extract_kwargs(inp))


@register_jagged_func(torch.ops.aten.expand_as.default, "self: t, other: jt")
def expand_as_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    other = new_kwargs.pop("other")

    return NestedTensor(func(inp, other._values), **extract_kwargs(other))


@register_jagged_func(torch.ops.aten.where.self, "condition: jt, self: jt, other: jt")
def where_self(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    condition = new_kwargs.pop("condition")
    inp = new_kwargs.pop("input")
    other = new_kwargs.pop("other")

    assert condition.shape == other.shape == inp.shape

    return NestedTensor(
        func(condition._values, inp._values, other._values, **new_kwargs),
        **extract_kwargs(condition),
    )


@register_jagged_func(torch.ops.aten._pin_memory.default, "self: jt, device: any?")
def _pin_memory_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(torch.ops.aten.is_pinned.default, "self: jt, device: any?")
def is_pinned_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    return func(inp._values, **new_kwargs)


@register_jagged_func(torch.ops.aten.is_same_size.default, "self: jt, other: jt")
def is_same_size_default(func, *args, **kwargs):
    return args[0]._size == args[1]._size


@register_jagged_func(
    torch.ops.aten.sum.dim_IntList, "self: jt, dim: any?, keepdim: any?, dtype: any?"
)
def sum_dim_IntList(func, *args, **kwargs):
    # sum_dim_IntList can produce a NT or a T depending on whether the ragged dims
    # are reduced away.
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    inp = new_kwargs.pop("input")
    assert inp._ragged_idx == 1
    new_kwargs["dim"], ragged_reduced_away = _wrap_jagged_dims(
        inp.dim(), new_kwargs["dim"], "sum"
    )

    if not ragged_reduced_away:
        return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))
    else:
        # Don't wrap because we reduced away the raggedness
        out = func(inp._values, **new_kwargs)
        if new_kwargs["keepdim"]:
            out = out.unsqueeze(0)
        return out


@register_jagged_func(torch.ops.aten.transpose.int, "self: jt, dim0: any, dim1: any")
def transpose_int(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    from torch._prims_common import canonicalize_dims

    inp = new_kwargs.pop("input")
    dim0, dim1 = canonicalize_dims(inp.dim(), (new_kwargs["dim0"], new_kwargs["dim1"]))

    # To support the SDPA API, inputs need to have the ragged idx transposed to dim 2
    # instead of 1, although the internal Flash and mem-effn implementations will
    # use the inputs with raggedness in dim 1. Therefore, when requesting this transpose,
    # move things around but don't change anything in _values
    if dim0 == inp._ragged_idx or dim1 == inp._ragged_idx:
        if dim0 == inp._ragged_idx:
            to_dim = dim1
        else:
            to_dim = dim0
        inp._ragged_idx = to_dim
        new_size = list(inp._size)
        new_size[dim0], new_size[dim1] = new_size[dim1], new_size[dim0]
        inp._size = tuple(new_size)

    new_kwargs["dim0"] = _wrap_jagged_dim(inp.dim(), new_kwargs["dim0"], "transpose")
    new_kwargs["dim1"] = _wrap_jagged_dim(inp.dim(), new_kwargs["dim1"], "transpose")

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(torch.ops.aten.view.default, "self: jt, size: any")
def view_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    size = new_kwargs.pop("size")

    # Ensure specified size still includes batch and ragged dims
    if len(size) < 3 or not raggedness_matches(inp, size):
        raise RuntimeError(f"view(): cannot view shape {inp.shape} as {size}")

    jagged_size = [inp._values.shape[0]] + size[2:]
    return NestedTensor(func(inp._values, jagged_size), **extract_kwargs(inp))


@register_jagged_func(
    torch.ops.aten.native_layer_norm.default,
    "input: jt, normalized_shape: any, weight: any?, bias: any?, eps: any",
)
def native_layer_norm_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    normalized_shape = new_kwargs["normalized_shape"]

    # Ensure we're not trying to normalize over the ragged dim
    if inp.dim() < 3 or (inp.dim() - len(normalized_shape)) < 2:
        raise RuntimeError(
            "layer_norm(): normalizing over ragged dim not supported for nested tensors"
        )

    output, mean, std = func(inp._values, **new_kwargs)
    return (NestedTensor(output, **extract_kwargs(inp)), mean, std)


@register_jagged_func(
    torch.ops.aten.native_layer_norm_backward.default,
    "grad_out: jt, input: jt, normalized_shape: any, mean: any, rstd: any, weight: any?, bias: any?, output_mask: any",
)
def native_layer_norm_backward_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    grad_out = new_kwargs.pop("grad_out")
    inp = new_kwargs.pop("input")
    d_input, d_gamma, d_beta = func(grad_out._values, inp._values, **new_kwargs)
    if d_input is None:
        return (None, d_gamma, d_beta)

    return (NestedTensor(d_input, **extract_kwargs(inp)), d_gamma, d_beta)


@register_jagged_func(torch.ops.aten.select.int, "self: jt, dim: any, index: any")
def select_int(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    new_kwargs["dim"] = _wrap_jagged_dim(inp.dim(), new_kwargs["dim"], "select")

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(
    torch.ops.aten.slice.Tensor,
    "self: jt, dim: any?, start: any?, end: any?, step: any?",
)
def slice_tensor(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    new_kwargs["dim"] = _wrap_jagged_dim(inp.dim(), new_kwargs["dim"], "slice")

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(
    torch.ops.aten.convolution.default,
    "input: jt, weight: t, bias: t?, stride: any, padding: any, "
    "dilation: any, transposed: any, output_padding: any, groups: any",
)
def convolution_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(
    torch.ops.aten.mean.dim, "self: jt, dim: any?, keepdim: any, dtype: any?"
)
def mean_dim(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    # NB: mean expects dim as a single item list of ints for some reason
    new_kwargs["dim"] = [_wrap_jagged_dim(inp.dim(), new_kwargs["dim"][0], "mean")]

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(torch.ops.aten.stack.default, "tensors: any, dim: any")
def stack_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # guaranteed this is non-empty if we got here
    tensors = new_kwargs.pop("tensors")
    for t in tensors:
        if not isinstance(t, NestedTensor):
            raise RuntimeError("stack(): expected all nested tensors inputs")

        if t.dim() != tensors[0].dim():
            raise RuntimeError(
                "stack(): expected all nested tensors to have the same dim"
            )

        if not raggedness_matches(t, tensors[0].shape):
            raise RuntimeError(
                "stack(): expected all nested tensors to have the same nested structure"
            )

    new_kwargs["dim"] = _wrap_jagged_dim(
        tensors[0].dim() + 1, new_kwargs["dim"], "stack"
    )

    return NestedTensor(
        func([t._values for t in tensors], **new_kwargs), **extract_kwargs(tensors[0])
    )


@register_jagged_func(
    torch.ops.aten.embedding.default,
    "weight: t, indices: jt, padding_idx: any?, scale_grad_by_freq: any?, sparse: any?",
)
def embedding_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # guaranteed this is non-empty if we got here
    indices = new_kwargs.pop("indices")
    weight = new_kwargs.pop("weight")

    return NestedTensor(
        func(weight, indices._values, **new_kwargs), **extract_kwargs(indices)
    )
