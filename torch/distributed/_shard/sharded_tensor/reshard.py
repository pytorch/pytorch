import copy
from typing import List, Tuple

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import (
    ProcessGroup,
)
from torch.distributed._shard.sharding_spec import (
    ShardingSpec,
    ShardMetadata,
)
from torch.distributed._shard.sharding_spec._internals import (
    get_split_size,
    get_chunked_dim_size,
)
from torch.distributed.nn.functional import (
    all_to_all,
    all_to_all_single,
)

from .shard import Shard


def get_idx_from_placements(placements, current_rank) -> int:
    """
    Return the position of the current rank in the given placements.

    Args:
        placements(List[Union[_remote_device, str]]):
            Specifies the placement of each shard of the Tensor. The size of
            the list represents the number of shards to be created. This could
            be a list of
            :class:`torch.distributed._remote_device`'s. This list
            could also contain a string which represents remote
            device as accepted by
            :class:`torch.distributed._remote_device`
        current_rank (int): number of current device.

    Returns:
        A int which contains the position of current device in the placement list.
    """
    for idx, placement in enumerate(placements):  # type: ignore[attr-defined]
        if current_rank == placement.rank():  # type: ignore[union-attr]
            return idx
    raise RuntimeError('current_rank not in the placement.')


def build_reshard_metadata(
    st_size: torch.Size,
    sharding_spec: ShardingSpec,
    world_size: int,
) -> Tuple[List[ShardMetadata], List[int]]:
    """
    Based the given sharding spec, we calculate the offset and local shard size.
    We then build a ShardMetadata on top of the calculation result.

    Args:
        st_size (torch.Size): The size of the sharded tensor.
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The
            specification describing how the tensor is sharded.
        world_size (int): number of ranks.

    Returns:
        A Tuple of the followings:
            A List[`ShardMetadata`] which contains the metadata for the shard, including
                offsets, lengths and device placement.
            A List[int] which contains the ranks in the order of placement.
    """
    shard_dim = int(sharding_spec.dim)  # type: ignore[attr-defined]
    shards_metadata = [None] * world_size
    ranks = []
    offsets = [0] * len(st_size)
    split_size = get_split_size(st_size[shard_dim], world_size)
    for idx, placement in enumerate(sharding_spec.placements):  # type: ignore[attr-defined]
        ranks.append(placement.rank())
        sharded_dim_size = get_chunked_dim_size(st_size[shard_dim], split_size, idx)
        local_tensor_size = list(st_size)
        local_tensor_size[shard_dim] = sharded_dim_size
        shards_metadata[placement.rank()] = ShardMetadata(  # type: ignore[call-overload]
            shard_offsets=copy.deepcopy(offsets),
            shard_sizes=local_tensor_size,
            placement=placement,
        )
        offsets[shard_dim] += sharded_dim_size
    return shards_metadata, ranks  # type: ignore[return-value]


def reshuffle_local_shard(
    local_shard: torch.Tensor,
    st_size: torch.Size,
    sharding_spec: ShardingSpec,
    resharding_spec: ShardingSpec,
    pg: ProcessGroup,
) -> Tuple[List[Shard], List[ShardMetadata]]:
    """
    Reshuffle the local shard directly when the reshard dim is same as the original
    sharding dim. Logically we do this in two step:
    1. To collect all shards based on original sharding spec.
    2. Reshard the tensor based on the given resharding spec.

    In reality, we consolidate the two steps into one by sending the local tensor to
    the new shard directly based on the resharding spec.

    Args:
        local_tensor (Tensor): Local tensor stored in the current rank.
        st_size (torch.Size): The size of the sharded tensor.
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The
            specification describing how the tensor is sharded originally.
        resharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The
            specification describing how the tensor will be resharded.
        pg (ProcessGroup): The process group to aggregate on.

    Returns:
        A Tuple of the followings:
            A List[`Shard`] which contains the local tensor and its metadata.
            A List[`ShardMetadata`] which contains the metadata for the shard, including
                offsets, lengths and device placement.
    """
    current_rank = dist.get_rank(pg)
    world_size = dist.get_world_size(pg)
    # Build shards_metadata first.
    shards_metadata, ranks = build_reshard_metadata(
        st_size, resharding_spec, world_size
    )
    # Get input split size for all2all.
    reshard_dim = int(resharding_spec.dim)  # type: ignore[attr-defined]
    split_size = get_split_size(st_size[reshard_dim], world_size)
    input_split_sizes = [0] * world_size
    idx = get_idx_from_placements(sharding_spec.placements, current_rank)  # type: ignore[attr-defined]
    new_rank = resharding_spec.placements[idx].rank()  # type: ignore[union-attr, attr-defined]
    input_split_sizes[new_rank] = local_shard.size(reshard_dim)
    # Get output split size for all2all.
    output_split_sizes = [0] * world_size
    new_idx = ranks.index(current_rank)
    sharded_dim_size = get_chunked_dim_size(st_size[reshard_dim], split_size, new_idx)
    output_split_sizes[new_rank] = sharded_dim_size
    # Get gathered_input for all2all.
    local_shard = local_shard.transpose(0, reshard_dim).contiguous()
    gathered_input_size = list(local_shard.size())
    gathered_input_size[0] = sharded_dim_size
    gathered_input = torch.empty(gathered_input_size, device=local_shard.device)
    # all2all.
    local_shard = all_to_all_single(
        gathered_input,
        local_shard,
        input_split_sizes=input_split_sizes,
        output_split_sizes=output_split_sizes,
        group=pg,
    )
    local_tensor = local_shard.transpose(0, reshard_dim).contiguous()
    local_shards = [Shard(local_tensor, shards_metadata[current_rank])]
    return local_shards, shards_metadata


def reshard_local_shard(
    local_tensor: torch.Tensor,
    st_size: torch.Size,
    sharding_spec: ShardingSpec,
    resharding_spec: ShardingSpec,
    pg: ProcessGroup,
) -> Tuple[List[Shard], List[ShardMetadata]]:
    """
    Reshard a sharded tensor given the ``resharding_spec``. When the reshard dim is
    different from the original sharding dim, we need to do two steps logically:
    1. To collect all shards based on original sharding spec.
    2. Reshard the tensor based on the given resharding spec.

    In reality, we consolidate the two steps into one by sending each rank the new
    shard based on the resharding spec.

    Args:
        local_tensor (Tensor): Local tensor stored in the current rank.
        st_size (torch.Size): The size of the sharded tensor.
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The
            specification describing how the tensor is sharded originally.
        resharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The
            specification describing how the tensor will be resharded.
        pg (ProcessGroup): The process group to aggregate on.

    Returns:
        A Tuple of the followings:
            A List[`Shard`] which contains the local tensor and its metadata.
            A List[`ShardMetadata`] which contains the metadata for the shard, including
                offsets, lengths and device placement.
    """
    current_rank = dist.get_rank(pg)
    world_size = dist.get_world_size(pg)
    current_sharding_dim = int(sharding_spec.dim)  # type: ignore[attr-defined]
    reshard_dim = int(resharding_spec.dim)  # type: ignore[attr-defined]

    # Build shards_metadata first.
    shards_metadata, ranks = build_reshard_metadata(
        st_size, resharding_spec, world_size
    )

    # Compute expected size
    input_split_sizes = []
    for metadata in shards_metadata:
        input_split_sizes.append(metadata.shard_sizes[reshard_dim])
    rearrange_input = any(ranks[i] > ranks[i + 1] for i in range(len(ranks) - 1))

    if rearrange_input:
        # Need to re-arrange reshard_dim of local_tensor before all2all.
        indices: List[int] = []
        for metadata in shards_metadata:
            offset_start_idx = metadata.shard_offsets[reshard_dim]
            split_size = metadata.shard_sizes[reshard_dim]
            indices += range(offset_start_idx, offset_start_idx + split_size)
        local_tensor = local_tensor.index_select(
            reshard_dim, torch.tensor(indices, device=local_tensor.device)
        )

    # Because reshard_dim != original shard_dim. We need to compute the
    # size of tensor from each rank.
    output_tensor_list = [torch.tensor(1)] * world_size
    split_size = get_split_size(st_size[current_sharding_dim], world_size)
    rearrange_output_list = False
    indices = []
    for idx, placement in enumerate(sharding_spec.placements):  # type: ignore[attr-defined]
        sharded_dim_size = get_chunked_dim_size(
            st_size[current_sharding_dim], split_size, idx
        )
        output_tensor_size = list(st_size)
        output_tensor_size[current_sharding_dim] = sharded_dim_size
        output_tensor_size[reshard_dim] = input_split_sizes[current_rank]
        output_tensor_list[
            placement.rank()
        ] = torch.empty(  # type: ignore[union-attr, index]
            output_tensor_size, device=local_tensor.device
        )
        indices.append(placement.rank())  # type: ignore[union-attr, index, arg-type]
        if idx != placement.rank():  # type: ignore[union-attr]
            rearrange_output_list = True

    # Perform autograd enabled all2all.
    input_tensor_list = torch.split(local_tensor, input_split_sizes, dim=reshard_dim)
    input_tensor_list = [tensor.contiguous() for tensor in input_tensor_list]
    output_tensor_list = all_to_all(
        output_tensor_list,
        input_tensor_list,
        group=pg,
    )

    if rearrange_output_list:
        # Need to re-arrange original shard_dim of output_tensor_list.
        output_tensor_list = [output_tensor_list[idx] for idx in indices]  # type: ignore[call-overload]
    local_tensor = torch.cat(output_tensor_list, dim=current_sharding_dim)
    local_shards = [Shard(local_tensor, shards_metadata[current_rank])]
    return local_shards, shards_metadata
