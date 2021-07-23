from typing import List

import torch
from torch.distributed.utils import _parse_remote_device
from torch.distributed._sharding_spec.api import ShardMetadata


def is_valid_device(device):
    """
    Checks if this is a valid local/remote device.
    """
    # Check for torch.device
    try:
        torch.device(device)
        return True
    except Exception:
        pass

    # Check for remote device.
    try:
        _parse_remote_device(device)
        return True
    except Exception:
        pass

    return False

def _check_shard_metadata_pair_overlap(shard1: ShardMetadata, shard2: ShardMetadata):
    """
    Checks if two shards overlap.
    """

    # For each dim of each shard, check if one shard resides on the other
    # end of second shard with respect to that dim. As an example for a 2D
    # shard, we would check if one shard is above or on the left of the
    # other shard.
    ndims = len(shard1.shard_offsets)
    for i in range(ndims):
        if shard1.shard_offsets[i] >= shard2.shard_offsets[i] + shard2.shard_lengths[i]:
            return False
        if shard2.shard_offsets[i] >= shard1.shard_offsets[i] + shard1.shard_lengths[i]:
            return False

    return True

def validate_non_overlapping_shards_metadata(shards: List[ShardMetadata]):
    """
    Ensures none of the shards overlap with each other.
    """
    # TODO: evaluate optimizing this if needed.
    for i in range(len(shards)):
        for j in range(i + 1, len(shards)):
            if _check_shard_metadata_pair_overlap(shards[i], shards[j]):
                raise ValueError(f'Shards {shards[i]} and {shards[j]} overlap')


def check_tensor(shards_metadata, tensor_dims) -> None:
    """
    Checks if the shards_metadata is compatible with the provided tensor dims.

    Args:
        tensor_dims(Sequence of int): Dimensions of tensor to verify
    Raises:
        ``ValueError`` if not compatible.
    """

    # If the tensor's volume matches the total volume of all shards and
    # all shard boundaries are within tensor dims, we have a compatible
    # sharding spec for this tensor. Note that we have already verified
    # we don't have overlapping shards.
    tensor_rank = len(tensor_dims)
    shards_rank = len(shards_metadata[0].shard_offsets)
    if tensor_rank != shards_rank:
        raise ValueError(f'Rank of tensor is {tensor_rank}, but shards rank is {shards_rank}')

    total_shard_volume = 0
    for shard in shards_metadata:
        shard_volume = 1
        for i, shard_length in enumerate(shard.shard_lengths):
            shard_volume *= shard_length
            if shard.shard_offsets[i] + shard.shard_lengths[i] > tensor_dims[i]:
                raise ValueError(
                    f'Shard offset {shard.shard_offsets[i]} and length '
                    f'{shard.shard_lengths[i]} exceeds tensor dim: {tensor_dims[i]} for shard {shard}')
        total_shard_volume += shard_volume

    tensor_volume = 1
    for size in tensor_dims:
        tensor_volume *= size

    if total_shard_volume != tensor_volume:
        # TODO: Can we improve this error message to point out the gaps?
        raise ValueError(
            f'Total volume of shards: {total_shard_volume}'
            f'does not match tensor volume: {tensor_volume}, in other words'
            f' all the individual shards do not cover the entire tensor')
