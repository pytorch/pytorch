import copy

import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import (
    sharded_op_impl,
    ShardedTensor,
)
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.sharding_spec._internals import (
    get_chunked_dim_size,
    get_split_size,
)
from torch.distributed.nn.functional import (
    all_to_all_single,
)


def _generate_chunk_indices(
    chunks,
    shards_metadata,
    chunk_size,
    world_size,
    idx_metadata,
    placement_ranks,
    dim,
):
    """
    Select out indices of sharded tensor which is sharded on the current rank
    and distribute them onto corresponding ranks after the sharding of each
    chunks.

    Args:
        chunks (List[Tensor]): Chunks of tensor indices by the sharding dim.
        shards_metadata (List[ShardMetadata]): Metadata of st's each shard.
        chunk_size (Int): Length of each chunk. If non-evenly chunked, it's the
            length of non-last chunk.
        world_size (Int): Number of ranks.
        idx_metadata (Int): Index in the shards_metadata for the current rank.
        placement_ranks (List[Int]): Rank sequence from sharding spec placements.
        dim (Int): The dimension which the tensor is chunked by and sharded on.

    Args:
        index (List[List[Int]]): A list of indices list for each rank. All indices
            are from the current shard.
    """
    shard_size = shards_metadata[idx_metadata].shard_sizes[dim]
    start_idx = shards_metadata[idx_metadata].shard_offsets[dim]
    chunk_begin = start_idx // chunk_size
    chunk_end = (start_idx + shard_size) // chunk_size + 1
    chunks = chunks[chunk_begin:chunk_end]
    index = [[] for _ in range(world_size)]
    ranges = set(range(start_idx, start_idx + shard_size))
    for chunk in chunks:
        indices = chunk.tolist()
        split_size = get_split_size(len(indices), world_size)
        for i in indices:
            if i not in ranges:
                continue
            shard_id = (i % chunk_size) // split_size
            shard_rank = placement_ranks[shard_id]
            index[shard_rank].append(i - start_idx)
    return index


def _shuffle_local_tensor_for_chunk(local_tensor, st, chunks, chunk_num, pg):
    """
    Shuffle local tensor to achieve the combination of gather all local
    shards together, then perform the chunk and shard the tensor for each
    chunk separately. We calculate the part to send to each rank based on
    the length of each chunk.

    Args:
        local_tensor (Tensor): Local tensor to apply Chunk to.
        st (ShardedTensor): original shardedtensor.
        chunks (List[Tensor]): Chunks of tensor indices by the sharding dim.
        chunk_num (Int): Number of chunks.
        pg (ProcessGroup, optional): process group.

    Args:
        func (Callable): Torch function for which we want to provide a sharded
            implementation (ex: torch.transpose)
    """
    dim = st.sharding_spec().dim
    local_tensor = local_tensor.transpose(0, dim)
    shards_metadata = st.metadata().shards_metadata
    cur_rank = dist.get_rank(pg)
    world_size = dist.get_world_size(pg)
    spec = st.sharding_spec()
    placement_ranks = []
    meta_ranks = []
    rank_idx = -1
    idx_metadata = -1
    output_rearrange = False
    for idx, shard_mt in enumerate(shards_metadata):
        placement_ranks.append(spec.placements[idx].rank())
        meta_ranks.append(shard_mt.placement.rank())
        if spec.placements[idx].rank() == cur_rank:
            rank_idx = idx
        if spec.placements[idx].rank() != idx:
            output_rearrange = True
        if shard_mt.placement.rank() == cur_rank:
            idx_metadata = idx
    assert rank_idx >= 0
    assert idx_metadata >= 0
    chunk_size = get_split_size(st.size(dim), chunk_num)
    index = _generate_chunk_indices(
        chunks,
        shards_metadata,
        chunk_size,
        world_size,
        idx_metadata,
        placement_ranks,
        dim,
    )
    chunk_local_split = []
    for chunk in chunks:
        split_size = get_split_size(len(chunk), world_size)
        chunk_local_split.append(get_chunked_dim_size(len(chunk), split_size, rank_idx))

    input_split_sizes = [len(indices) for indices in index]
    indices_flatten = list(idx for indices in index for idx in indices)
    output_split_sizes = None
    output_split_sizes = [0] * world_size
    output_split_sizes[cur_rank] = input_split_sizes[cur_rank]
    for rank in range(world_size):
        if rank == cur_rank:
            continue
        idx_metadata = meta_ranks.index(rank)
        index = _generate_chunk_indices(
            chunks,
            shards_metadata,
            chunk_size,
            world_size,
            idx_metadata,
            placement_ranks,
            dim,
        )
        output_split_sizes[rank] = len(index[cur_rank])

    local_tensor = local_tensor.index_select(
        0, torch.tensor(indices_flatten, device=local_tensor.device)
    )
    local_tensor_size = local_tensor.size()
    gathered_input_size = [sum(output_split_sizes)] + list(local_tensor_size[1:])
    gathered_input = torch.empty(gathered_input_size, device=local_tensor.device)
    all_to_all_single(
        gathered_input,
        local_tensor,
        input_split_sizes=input_split_sizes,
        output_split_sizes=output_split_sizes,
        group=pg,
    )
    if output_rearrange:
        rearrage_list = []
        gathered_list = torch.split(gathered_input, output_split_sizes)
        for rank in placement_ranks:
            rearrage_list.append(gathered_list[rank])
        gathered_input = torch.cat(rearrage_list).contiguous()

    return gathered_input.transpose(0, dim), chunk_local_split


def register_chunk_op(op):
    @sharded_op_impl(op)
    def sharded_chunk(types, args=(), kwargs=None, pg=None):
        """
        Handles ``__torch_function__`` dispatch for the chunk op.
        If we chunk by the non-sharding dim, we just directly chunk the
        local tensor and create a list of sharded tensor based on them.

        If we chunk by the sharding dim, we first gather all local shards
        together, then perform the chunk and shard the tensor for each
        chunk separately. In reality, we combine all three steps as one
        communication to save communication cost.

        Args: same as ``torch.chunk``.

        Return:
            List[ShardedTensor]: Chunk results as a list of ShardedTensor.
        """
        st = args[0]
        chunk_num = args[1]
        dim = kwargs.get("dim")
        dim = dim if dim else 0

        # Validate types
        if not isinstance(st, ShardedTensor):
            raise TypeError(
                f"torch function '{op.__name__}', with args: {args} and "
                f"kwargs: {kwargs} are called for non ShardedTensor!"
            )
        spec = st.sharding_spec()
        if not isinstance(spec, ChunkShardingSpec):
            raise NotImplementedError("Only ChunkShardingSpec is supported for chunk.")
        st_size = list(st.size())  # type: ignore[arg-type]
        sharding_dim = spec.dim
        local_tensor = st.local_tensor()
        chunks = torch.arange(st_size[dim]).chunk(chunk_num)
        chunk_split_size = [len(chunk) for chunk in chunks]
        local_tensor_chunks = torch.chunk(local_tensor, chunk_num, dim)

        # When performing chunk on the dim of sharding, we need to shuffle the tensor.
        if (
            sharding_dim == dim
            or st.dim() + sharding_dim == dim
            or st.dim() + dim == sharding_dim
        ):
            local_tensor, chunk_local_split = _shuffle_local_tensor_for_chunk(
                local_tensor, st, chunks, chunk_num, pg
            )
            local_tensor_chunks = torch.split(local_tensor, chunk_local_split, dim=dim)

        results = []
        for idx, split_size in enumerate(chunk_split_size):
            local_tensor = local_tensor_chunks[idx]
            new_st_size = copy.deepcopy(st_size)
            new_st_size[dim] = split_size
            results.append(
                ShardedTensor._init_from_local_tensor(
                    local_tensor.contiguous(),
                    st.sharding_spec(),
                    tuple(new_st_size),
                    process_group=pg,
                )
            )
        return results


chunk_ops = [
    torch.chunk,
    torch.Tensor.chunk,
]
for op in chunk_ops:
    register_chunk_op(op)
