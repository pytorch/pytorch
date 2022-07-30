import copy
import math

import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import (
    ShardedTensor,
)
from torch.distributed._shard.sharding_spec._internals import (
    get_chunk_sharding_params,
)
from torch.distributed.nn.functional import (
    all_reduce,
)

from ._common import (
    _chunk_sharding_spec_check,
    _register_sharded_op_on_local_tensor,
)


def transpose_same_dim(*args, **kwargs):
    """
    When the dim0 and dim1 of transpose are the same, return the original ShardedTensor.

    Args: same as ``torch.Tensor.transpose``.

    Return (bool): Whether to return early or not.
    """
    return args[1] == args[2]


def sharded_transpose_check(*args, **kwargs):
    """
    Perform extra checks for the sharded_transpose op such as the input needs to
    be at least 2 and the sharding spec needs to be a ChunkShardingSpec.

    Args: same as ``torch.Tensor.type_as``.

    Return: None
    """
    if len(args) < 3:
        raise ValueError("Needs at least two dimensions for transpose op!")
    _chunk_sharding_spec_check(args[0].sharding_spec(), torch.Tensor.transpose)


def sharded_transpose(args, kwargs, pg):
    """
    Handles ``__torch_function__`` dispatch for the ``torch.Tensor.transpose`` op.

    Returns a new sharded tensor with the given dimensions transposed.
    During the transpose, we keep the original shading dim, if the sharding
    dim is not neither dim0 nor dim1. Otherwise, we will swap the sharding
    dim with the other input of transpose.

    Args: (same as ``torch.Tensor.transpose``.)
        dim0 (Int): the first dimension to be transposed.
        dim1 (Int): the second dimension to be transposed.

    Returns:
        new_local_shards (List[Shard]): Local shards for the new sharded tensor.
        st_meta (ShardedTensorMetadata): Metadata of the new sharded tensor.
    """

    def _swap_meta_data(data, idx0, idx1):
        """
        Swap the item at idx0 and idx1 in the data list.
        """
        data[idx0], data[idx1] = data[idx1], data[idx0]

    st = args[0]
    dim0 = args[1]
    dim1 = args[2]

    sharding_spec = copy.deepcopy(st.sharding_spec())
    if sharding_spec.dim == dim0:
        sharding_spec.dim = dim1
    elif sharding_spec.dim == dim1:
        sharding_spec.dim = dim0

    st_size = list(st.size())
    _swap_meta_data(st_size, dim0, dim1)
    local_tensor = st.local_tensor().transpose(dim0, dim1).contiguous()
    return local_tensor, sharding_spec, tuple(st_size)


_register_sharded_op_on_local_tensor(
    torch.transpose,
    early_stop_func=transpose_same_dim,
    extra_check=sharded_transpose_check,
    customized_func=sharded_transpose,
)
_register_sharded_op_on_local_tensor(
    torch.Tensor.transpose,
    early_stop_func=transpose_same_dim,
    extra_check=sharded_transpose_check,
    customized_func=sharded_transpose,
)


def sharded_masked_fill_check(*args, **kwargs):
    """
    Perform extra checks for the ``torch.Tensor.masked_fill`` op.
    Ensure the mask size is broadcastable with the size of
    the sharded tensor.

    Args: same as ``torch.Tensor.masked_fill``.

    Return: None
    """
    st = args[0]
    mask = args[1]
    if st.dim() < mask.dim():
        raise ValueError(
            "mask dim must not greater than the dim of the sharded tensor."
        )
    for idx in range(-1, -mask.dim() - 1, -1):
        if mask.size(idx) != st.size(idx) and mask.size(idx) != 1:
            raise ValueError(
                f"The size of mask {mask.dim() + idx} must match the size of "
                f"sharded tensor {st.dim() + idx} at non-singleton dimension {mask.dim() + idx}"
            )


def sharded_masked_fill(args, kwargs, pg):
    """
    Handles ``__torch_function__`` dispatch for the ``torch.Tensor.masked_fill`` op.
    We first narrow down the mask to the size of local tensor if the mask
    contains the sharding dim and then apply the mask to the local tensor.

    Args: same as ``torch.Tensor.masked_fill``.

    Return:
        local_tensor (Tensor): New local tensor to build the sharded tensor.
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`):
            sharding spec of the new sharded tensor.
        new_st_size (torch.Size): Size of the new sharded tensor.
    """
    st = args[0]
    mask = args[1]
    value = args[2]
    current_rank = dist.get_rank(pg)  # type: ignore[attr-defined]
    sharding_dim = st.sharding_spec().dim  # type: ignore[attr-defined]
    narrow_idx = None
    for idx in range(-1, -mask.dim() - 1, -1):
        if st.dim() + idx == sharding_dim and mask.size(idx) != 1:
            narrow_idx = idx
    if narrow_idx is not None:
        rank_idx = None
        for idx, placement in enumerate(st._sharding_spec.placements):  # type: ignore[attr-defined]
            if placement.rank() == current_rank:  # type: ignore[index]
                rank_idx = idx  # type: ignore[attr-defined]
        shard_metadata = st.metadata().shards_metadata[rank_idx]  # type: ignore[index]
        mask = mask.narrow(
            narrow_idx,
            shard_metadata.shard_offsets[sharding_dim],
            shard_metadata.shard_sizes[sharding_dim],
        )
    local_tensor = st.local_tensor().masked_fill(mask, value)
    return local_tensor, st.sharding_spec(), st.size()


_register_sharded_op_on_local_tensor(
    torch.Tensor.masked_fill,
    extra_check=sharded_masked_fill_check,
    customized_func=sharded_masked_fill,
)


def sharded_view_check(*args, **kwargs):
    """
    Perform extra checks for the ``torch.Tensor.view`` op.

    Args: same as ``torch.Tensor.view``.

    Return: None
    """
    st = args[0]
    shape = args[1:]
    if len(shape) == 0:
        raise ValueError("Missing *shape for sharded view op.")
    if len(shape) <= st.sharding_spec().dim:
        raise NotImplementedError(
            f"Shape having dim {len(shape)} is not supported "
            f"for sharded tensor sharded on dim {st.sharding_spec().dim}."
        )
    st_size = math.prod(st.size())  # type: ignore[attr-defined]
    shape_size = math.prod(shape)  # type: ignore[attr-defined]
    neg_sum = sum(i for i in shape if i < 0)
    if shape_size > st_size or st_size % shape_size:
        raise ValueError(
            f"Shape '{list(shape)}' is invalid for sharded tensor size {st_size}."
        )
    if neg_sum < -1:
        raise ValueError("Only one dimension can be inferred for sharded view op.")


def sharded_view(args, kwargs, pg):
    """
    Handles ``__torch_function__`` dispatch for the ``torch.Tensor.view`` op.
    For now we always keep the sharding dim after view. For example, if
    a sharded tensor with size [16, 5] and sharded by 0. If we now view
    it as [4, 2, 2, 5], it will still be sharded by dim 0.

    Args: same as ``torch.Tensor.view``.

    Return:
        local_tensor (Tensor): New local tensor to build the sharded tensor.
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`):
            sharding spec of the new sharded tensor.
        new_st_size (torch.Size): Size of the new sharded tensor.
    """
    st = args[0]
    shape = args[1:]
    try:
        infer_idx = shape.index(-1)
    except ValueError:
        infer_idx = None

    # Infer the dim which is specified with -1.
    if infer_idx is not None:
        st_size = math.prod(st.size())  # type: ignore[attr-defined]
        shape_size = -1 * math.prod(shape)  # type: ignore[attr-defined]
        shape = (*shape[:infer_idx], st_size // shape_size, *shape[infer_idx + 1 :])
    if st.size() == shape:
        return st.local_tensor(), st.sharding_spec(), shape

    sharding_dim = st.sharding_spec().dim
    sharding_spec = st.sharding_spec()
    # When the sharding dim is negative, we need to ensure the new
    # sharded tensor is still sharded by the original dimension.
    if sharding_dim < 0:
        sharding_spec = copy.deepcopy(sharding_spec)
        sharding_dim = st.dim() + sharding_dim
        sharding_spec.dim = sharding_dim

    world_size = dist.get_world_size(pg)
    if shape[sharding_dim] % world_size:
        raise NotImplementedError(
            f"Case when dim '({shape[sharding_dim]})' is not divisible "
            "by world_size is not supported."
        )
    new_local_tensor_size = (
        *shape[:sharding_dim],
        shape[sharding_dim] // world_size,
        *shape[sharding_dim + 1 :],
    )
    new_local_tensor = st.local_tensor().view(*new_local_tensor_size)
    return new_local_tensor, sharding_spec, shape


_register_sharded_op_on_local_tensor(
    torch.Tensor.view,
    extra_check=sharded_view_check,
    customized_func=sharded_view,
)


def sharded_bmm_check(*args, **kwargs):
    """
    Perform extra checks for the sharded_bmm op, for example, st2 needs to
    be a sharded tensor and both tensors need to sharded by dim 0, etc.

    Args: same as ``torch.bmm``.

    Return: None
    """
    if len(args) < 2:
        raise TypeError("Needs two tensors to perform torch.bmm.")
    st = args[0]
    st2 = args[1]
    # Validate types
    if not isinstance(st2, ShardedTensor):
        raise TypeError("st2 needs to be a ShardedTensor for torch.bmm.")
    _chunk_sharding_spec_check(st2.sharding_spec(), torch.bmm)
    if st.dim() != 3 or st2.dim() != 3:
        raise TypeError("both st and st2 need to be a 3D ShardedTensor")
    if (
        st.sharding_spec().dim != st2.sharding_spec().dim  # type: ignore[attr-defined]
        or st.sharding_spec().dim != 0
    ):
        raise NotImplementedError(
            "Only support performing bmm on tensors sharded on dim 0 now."
        )
    if st.sharding_spec().placements != st2.sharding_spec().placements:  # type: ignore[attr-defined]
        raise NotImplementedError(
            "Both st and st2 need to have same placements for bmm."
        )


def sharded_bmm(args, kwargs, pg):
    """
    Handles ``__torch_function__`` dispatch for the sharded_bmm op.

    Warning: For now we only supports the case when both tensors are sharded
             by dim 0 so that no local communication.

    Args: same as ``torch.bmm``.

    Return:
        local_tensor (Tensor): New local tensor to build the sharded tensor.
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`):
            sharding spec of the new sharded tensor.
        new_st_size (torch.Size): Size of the new sharded tensor.
    """
    st = args[0]
    st2 = args[1]
    local_tensor = torch.bmm(st.local_tensor(), st2.local_tensor())
    new_st_size = (*st.size()[:-1], st2.size(-1))
    return local_tensor, st.sharding_spec(), new_st_size


_register_sharded_op_on_local_tensor(
    torch.Tensor.bmm,
    extra_check=sharded_bmm_check,
    customized_func=sharded_bmm,
)

_register_sharded_op_on_local_tensor(
    torch.bmm,
    extra_check=sharded_bmm_check,
    customized_func=sharded_bmm,
)


def sharded_layer_norm_check(*args, **kwargs):
    """
    Perform extra checks for the ``nn.LayerNorm`` op.
    Ensure the normalized shape is compatible with
    the size of the sharded tensor.

    Args: same as ``torch.nn.LayerNorm``.

    Return: None
    """
    st = args[0]
    normalized_shape = args[1]
    if st.dim() < len(normalized_shape):
        raise ValueError(
            "normalized_shape dim must not be greater than "
            "the dim of the sharded tensor."
        )
    for idx in range(-1, -len(normalized_shape) - 1, -1):
        if normalized_shape[idx] != st.size(idx):
            raise ValueError(
                f"Given normalized_shape=[{normalized_shape[idx]}], expected input with shape "
                f"[*, {normalized_shape[idx]}], but got input of size {list(st.size())}."
            )


def sharded_layer_norm(args, kwargs, pg):
    """
    Handles ``__torch_function__`` dispatch for the ``torch.nn.LayerNorm`` op.
    We gather all shards from local shards and perform a global normalization.
    We then scatter the result back to each rank.

    Args: same as ``torch.nn.LayerNorm``.

    Return:
        local_tensor (Tensor): New local tensor to build the sharded tensor.
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`):
            sharding spec of the new sharded tensor.
        new_st_size (torch.Size): Size of the new sharded tensor.
    """
    st = args[0]
    normalized_shape = args[1]
    sharding_dim = st.sharding_spec().dim  # type: ignore[attr-defined]
    sharding_dim = sharding_dim if sharding_dim >= 0 else st.dim() + sharding_dim
    local_tensor = st.local_tensor()
    # If sharding dim is smaller than shape start, we just perform a local norm.
    shape_start = st.dim() - len(normalized_shape)
    if shape_start > sharding_dim:
        args = (local_tensor, *args[1:])
        local_tensor = torch.nn.functional.layer_norm(*args, **kwargs)
        return local_tensor, st.sharding_spec(), st.size()

    elementwise_affine = kwargs.get("elementwise_affine", False)
    eps = kwargs.get("eps", 1e-05)

    norm_dims = tuple(i for i in range(-1, -len(normalized_shape) - 1, -1))
    local_size = math.prod(local_tensor.size()[shape_start:])  # type: ignore[attr-defined]
    st_size = math.prod(st.size()[shape_start:])  # type: ignore[attr-defined]
    local_mean = torch.mul(local_tensor.mean(norm_dims, keepdim=True), local_size)
    global_mean = torch.div(all_reduce(local_mean), st_size)
    local_variant_sq = torch.square(local_tensor - global_mean).sum(
        norm_dims, keepdim=True
    )
    global_variant = torch.div(all_reduce(local_variant_sq), st_size)

    denom = torch.rsqrt(global_variant + eps)
    local_tensor = torch.mul(local_tensor - global_mean, denom)

    if elementwise_affine:
        weight = kwargs["weight"]
        bias = kwargs["bias"]
        current_rank = dist.get_rank(pg)  # type: ignore[attr-defined]
        world_size = dist.get_world_size(pg)
        (start_pos, chunk_size) = get_chunk_sharding_params(
            bias.size(0), world_size, st.sharding_spec(), current_rank
        )
        local_tensor = torch.addmm(
            torch.narrow(bias, 0, start_pos, chunk_size),
            local_tensor,
            torch.narrow(weight, sharding_dim - shape_start, start_pos, chunk_size),
        )

    return local_tensor, st.sharding_spec(), st.size()


_register_sharded_op_on_local_tensor(
    torch.nn.LayerNorm,
    extra_check=sharded_layer_norm_check,
    customized_func=sharded_layer_norm,
)

_register_sharded_op_on_local_tensor(
    torch.nn.functional.layer_norm,
    extra_check=sharded_layer_norm_check,
    customized_func=sharded_layer_norm,
)
