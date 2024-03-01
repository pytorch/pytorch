from typing import List, Optional, Tuple

from torch.distributed._shard.metadata import ShardMetadata


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
        if shard1.shard_offsets[i] >= shard2.shard_offsets[i] + shard2.shard_sizes[i]:
            return False
        if shard2.shard_offsets[i] >= shard1.shard_offsets[i] + shard1.shard_sizes[i]:
            return False

    return True


def _find_nd_overlapping_shards(
    shards: List[ShardMetadata], sharded_dims: List[int]
) -> Optional[Tuple[int, int]]:
    # Each rank has len(sharded_dims) tuples. Each tuple represent the
    # [begin, end] (inclusive) pair of that dimension.
    shard_intervals = [
        [
            (s.shard_offsets[dim], s.shard_offsets[dim] + s.shard_sizes[dim] - 1)
            for dim in sharded_dims
        ]
        for s in shards
    ]

    for i in range(len(shards)):
        shard_i = shard_intervals[i]
        for j in range(i + 1, len(shards)):
            shard_j = shard_intervals[j]
            # For each dim of each shard, check if one shard resides on the other
            # end of second shard with respect to that dim. As an example for a 2D
            # shard, we would check if one shard is above or on the left of the
            # other shard.
            overlap = True
            for interval_i, interval_j in zip(shard_i, shard_j):
                if interval_i[0] > interval_j[1] or interval_j[0] > interval_i[1]:
                    overlap = False
                    break
            if overlap:
                return (i, j)
    return None


def _find_1d_overlapping_shards(
    shards: List[ShardMetadata], dim: int
) -> Optional[Tuple[int, int]]:
    # (begin, end, index_in_shards). Begin and end are inclusive.
    intervals = [
        (s.shard_offsets[dim], s.shard_offsets[dim] + s.shard_sizes[dim] - 1, i)
        for i, s in enumerate(shards)
    ]
    intervals.sort()
    for i in range(len(shards) - 1):
        if intervals[i][1] >= intervals[i + 1][0]:
            return (intervals[i][2], intervals[i + 1][2])
    return None


def validate_non_overlapping_shards_metadata(shards: List[ShardMetadata]):
    """
    Ensures none of the shards overlap with each other.

    Args:
        shards(List[ShardMetadata]): List of :class:`ShardMetadata` objects representing
            each shard.
    Raises:
        ``ValueError`` if there's overlap in any two shards.
    """
    if not shards or len(shards) == 1:
        return

    sharded_dims: List[int] = []
    for dim in range(len(shards[0].shard_offsets)):
        for i in range(1, len(shards)):
            if (
                shards[i].shard_offsets[dim] != shards[0].shard_offsets[dim] or
                shards[i].shard_sizes[dim] != shards[0].shard_sizes[dim]
            ):
                sharded_dims.append(dim)
                break

    pair: Optional[Tuple[int, int]] = None
    if len(sharded_dims) == 0:
        # All shards are the same, all dims are not partitioned. Choose any 2.
        pair = (0, 1)
    elif len(sharded_dims) == 1:
        # Shards are partitioned over only one dimension. Overlap can be found
        # using a O(nlogn) overlapping interval algorithm.
        pair = _find_1d_overlapping_shards(shards, sharded_dims[0])
    else:
        # Shards are partitioned over more than one dimension. Fall back to
        # pair-wise check. Even though O(nlogn) algorithms (line sweep) exist
        # for 2D overlap, the implementation is not trivial and may not justify
        # the time saving in most cases.
        pair = _find_nd_overlapping_shards(shards, sharded_dims)

    if pair:
        raise ValueError(f'Shards {shards[pair[0]]} and {shards[pair[1]]} overlap')


def check_tensor(shards_metadata, tensor_dims) -> None:
    """
    Checks if the shards_metadata is compatible with the provided tensor dims.

    Args:
        shards_metadata(List[ShardMetadata]): List of :class:`ShardMetadata`
            objects representing each shard of the tensor.
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
        for i, shard_length in enumerate(shard.shard_sizes):
            shard_volume *= shard_length
            if shard.shard_offsets[i] + shard.shard_sizes[i] > tensor_dims[i]:
                raise ValueError(
                    f'Shard offset {shard.shard_offsets[i]} and length '
                    f'{shard.shard_sizes[i]} exceeds tensor dim: {tensor_dims[i]} for shard {shard}')
        total_shard_volume += shard_volume

    tensor_volume = 1
    for size in tensor_dims:
        tensor_volume *= size

    if total_shard_volume != tensor_volume:
        # TODO: Can we improve this error message to point out the gaps?
        raise ValueError(
            f'Total volume of shards: {total_shard_volume} '
            f'does not match tensor volume: {tensor_volume}, in other words '
            f'all the individual shards do not cover the entire tensor')

def get_split_size(dim_size, chunks):
    """
    Computes the split size inline with ``torch.chunk``

    Args:
        dim_size(int): Size of the dimension being chunked.
        chunks(int): Number of chunks to create for ``dim_size``.

    Returns:
        An int indicating the split size to use.
    """
    return (dim_size + chunks - 1) // chunks

def get_chunked_dim_size(dim_size, split_size, idx):
    """
    Computes the dim size of the chunk for provided ``idx`` given ``dim_size``
    and ``split_size``.

    Args:
        dim_size(int): Size of the dimension being chunked.
        split_size(int): The chunk size for each chunk of ``dim_size``.
        idx(int): The index of chunk whose dim size is being requested.

    Returns:
        An int indicating the dim size of the chunk.
    """
    return max(min(dim_size, split_size * (idx + 1)) - split_size * idx, 0)

def get_chunk_sharding_params(sharding_dim_size, world_size, spec, rank):
    """
    Generate the start pos and offset length for the current rank for
    chunk sharding.

    Args:
        sharding_dim_size(int): The dimension length which we shard on.
        world_size(int): number of ranks.
        spec (:class:`torch.distributed._shard.sharding_spec.ChunkShardingSpec`):
            sharding spec.
        rank(int): # of cuda process.

    Returns:
        start_pos(int): start position of sharded tensor on the given rank.
        chunk_size(int): chunk size of sharded tensor on the given rank.
    """
    split_size = get_split_size(sharding_dim_size, world_size)
    current_offsets = 0
    start_pos = current_offsets
    for idx, placement in enumerate(spec.placements):
        chunk_size = get_chunked_dim_size(sharding_dim_size, split_size, idx)
        if rank == placement.rank():
            start_pos = current_offsets
            break
        current_offsets += chunk_size
    return start_pos, chunk_size  # type: ignore[possibly-undefined]
