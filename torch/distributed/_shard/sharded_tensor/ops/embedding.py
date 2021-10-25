from typing import List, cast

import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor.ops._common import (
    _handle_col_wise_sharding_common,
)
from torch.distributed._shard.sharding_spec._internals import (
    get_split_size,
    get_chunked_dim_size,
)


def sharded_embedding(types, args, kwargs, pg):
    """
    Handles ``__torch_function__`` dispatch for ``torch.nn.functional.embedding``.
    This method computes a sharded embedding lookup and has the following limitations:

    1. Supports only sharding of ``weight``.
    2. Supports only ``ChunkShardingSpec``.
    3. Supports only a single local shard per rank.
    4. Supports only simple look-up and including specs like padding_idx, max_norm, etc.

    Based on the dimension that the weight is sharded on, there are two
    algorithms:

    ROWWISE SHARDING
    ================
    For row-wise sharding the weight is sharded on dimension 0.

    The overall algorithm can be best explained with an example. Let's assume
    the dims for input are (4 x 6) and W are (10 x 17) and W is sharded across
    4 GPUs creating 3 shard of (3 x 17) and 1 shard of (1 x 17).
    The algorithm is as follows:

    1. First the input is flattened to 1D and gets sorted so that we can distribute
       them to the corresponding rank. For example if the given input is
       tensor([[6, 5, 2, 9, 6, 3],
               [3, 1, 2, 4, 7, 6],
               [4, 0, 4, 9, 8, 9],
               [8, 6, 6, 4, 6, 1]])
       Then we have the 1D array like:
       tensor([6, 5, 2, 9, 6, 3, 3, 1, 2, 4, 7, 6, 4, 0, 4, 9, 8, 9, 8, 6, 6, 4, 6, 1])
       And sort it:
       tensor([0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 7, 8, 8, 9, 9, 9])
       We also record the indices so that we can recover back.
    2. Next we perform the split by search the index of the chunking
       boundary. So the above array will be split into 4 parts:
       tensor([[0, 1, 1, 2, 2], [3, 3, 4, 4, 4, 4, 5],
              [6, 6, 6, 6, 6, 6, 7, 8, 8], [9, 9, 9])
       Rearragement may be needed if the rank order is different from
       its index in the placement.
    3. Next, we communicate the length of each part to each rank via all2all
       so that each rank now knows what input it will get from all other ranks.
    5. Before we send out the array to other ranks, we need to do the modular operation
       so that each rank do use that for embedding look-up.
       The above tensor will look like the below after performing the moduler of 3:
       tensor([[0, 1, 1, 2, 2], [0, 0, 1, 1, 1, 1, 2],
              [0, 0, 0, 0, 0, 0, 1, 2, 2], [0, 0, 0])
    4. Now, each rank receives a matrix (size may vary) and do the look-up. We then use
       all2all to send the result back to each rank.
    5. We use the recorded indices to recover the sorted positions and reshape the
       matrix to (4 x 6 x 17), which is what we need.

    COLWISE SHARDING
    ================
    For col-wise sharding the weight is sharded on dimension 1.

    The overall algorithm can be best explained with an example. Let's assume
    the dims for input are (4 x 6) and W are (16 x 17) and W is sharded across
    4 GPUs creating 3 shards of (16 x 5) and 1 shard of (16 x 2).
    The algorithm is as follows:

    1. First the input is broadcasted to all ranks, since this is SPMD we
       actually do an all_gather for all the inputs resulting in 4 (4 x 6)
       inputs on each rank.
    2. Next we perform local embedding look-up operation by apply each
       input (4 x 6) with the local shard (16 x 5) ((16 x 2) for the last).
       This results in 4 (5 x 6 x 4) ((2 x 6 x 4) for the last) matrices
       on each rank. We transpose dim 0 and dim 2.
    3. Next, we concat these 4 matrices and perform an all2all to share the
       appropriate (5 x 6 x 4) or (2 x 6 x 4) matrices to each rank.
    4. Now, each rank receives a (17 x 6 x 4) matrix which is basically the
       size of the result we need.
    5. If placements are not in order any appropriate rearrangement of columns
       are done for the (17 x 6 x 4) matrix and finally we transponse the
       dim 0 and dim 2 again.
    """
    from torch.distributed._shard import ShardedTensor
    from torch.distributed._shard import ChunkShardingSpec

    input = args[0]
    weight = args[1]

    # Validate types
    if not isinstance(input, torch.Tensor):
        raise TypeError("input need to be torch.Tensor")
    if not isinstance(weight, ShardedTensor):
        raise TypeError("weight needs to be ShardedTensor")
    weight_size = cast(torch.Size, weight.size())
    if len(weight_size) != 2:
        raise ValueError("Weight needs to have exactly 2 dims")
    if int(torch.min(input).item()) < 0:
        raise ValueError(
            "Index out of range in Input %d %d",
            int(torch.min(input).item()),
            weight_size[1],
        )
    if int(torch.max(input).item()) >= weight_size[0]:
        raise ValueError(
            "Index out of range in Input %d %d",
            int(torch.max(input).item()),
            weight_size[1],
        )

    if not isinstance(weight._sharding_spec, ChunkShardingSpec):
        raise ValueError("Only ChunkShardingSpec supported for ShardedTensor ops!")
    if len(weight.local_shards()) != 1:
        raise ValueError("Only one local shard supported!")

    local_shard = weight.local_shards()[0].tensor.contiguous()
    sharding_dim = weight._sharding_spec.dim
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)

    if sharding_dim == 1:
        return _handle_col_wise_sharding(input, world_size, weight, local_shard, pg)
    elif sharding_dim == 0:
        return _handle_row_wise_sharding(
            input, world_size, weight, rank, local_shard, pg
        )
    else:
        raise RuntimeError(
            f"nn.Embedding weight sharded on dim {sharding_dim} not supported!"
        )


def _handle_col_wise_sharding(input, world_size, weight, local_shard, pg):
    return _handle_col_wise_sharding_common(
        torch.nn.functional.embedding,
        weight.size(1),
        len(input.size()),
        input,
        world_size,
        weight,
        local_shard,
        pg,
    )


def _handle_row_wise_sharding(input, world_size, weight, rank, local_shard, pg):
    # flatten the ids across all input and sort
    input_size = input.size()
    input_1d = torch.reshape(input, (-1,)).contiguous()
    input_sorted, indices_1d = torch.sort(input_1d)
    rearrange_indices_1d = torch.argsort(indices_1d)
    input_sorted.contiguous()

    # Decide which rank the input goes to by check the sharding range.
    split_size = get_split_size(weight.size(0), world_size)
    rearrange_rows = False
    input_split_sizes: List[int] = [0] * world_size
    input_split_start_indices: List[int] = [0] * world_size
    # When we do the chunk split, we always ensure the first N - 1 chunks get max out
    # and then the Nth chunk gets the rest. So input_split_sizes like [3, 3, 3, 4]
    # are not possible. The expected split size will be [4, 4, 4, 1].
    sharded_dim_size_max = get_chunked_dim_size(weight.size(0), split_size, 0)
    for idx, placement in enumerate(weight._sharding_spec.placements):
        sharded_dim_size = get_chunked_dim_size(weight.size(0), split_size, idx)
        start_row_idx = idx * sharded_dim_size_max
        end_row_idx = start_row_idx + sharded_dim_size
        start_idx = torch.searchsorted(input_sorted, start_row_idx).item()
        end_idx = torch.searchsorted(input_sorted, end_row_idx).item()
        input_split_sizes[placement.rank()] = int(end_idx - start_idx)
        input_split_start_indices[placement.rank()] = int(start_idx)
        if placement.rank() != idx:
            rearrange_rows = True

    rearrange_indices_1d_second_order = None
    if rearrange_rows:
        # Need to re-arrange the 1D tensor to be sent via all2all.
        indices: List[List[int]] = [[0]] * world_size
        for placement in weight._sharding_spec.placements:
            split_length = input_split_sizes[placement.rank()]
            offset_idx = input_split_start_indices[placement.rank()]
            indices[placement.rank()] = list(
                range(offset_idx, offset_idx + split_length)
            )
        indices_flatten = list(idx for indice in indices for idx in indice)

        input_sorted = input_sorted.index_select(
            0, torch.tensor(indices_flatten, device=input.device)
        )
        rearrange_indices_1d_second_order = torch.argsort(torch.Tensor(indices_flatten))

    # Get the input split size to be sent from each rank to the current rank.
    # We can then infer the output split size.
    input_split_sizes_tensor = (
        torch.Tensor(input_split_sizes).type("torch.IntTensor").cuda(rank)
    )
    output_split_sizes_tensor = torch.empty(
        world_size, dtype=torch.int32, device=input.device
    )
    dist.all_to_all_single(
        output_split_sizes_tensor,
        input_split_sizes_tensor,
        group=pg,
    )
    output_split_sizes = output_split_sizes_tensor.tolist()

    # Input sent from each rank to the current rank may have different sizes.
    gathered_input = torch.empty(
        sum(output_split_sizes), dtype=torch.int64, device=input.device
    )

    # Perform the modular operation of the 1D tensor to be sent to each rank.
    input_sorted = torch.remainder(input_sorted, sharded_dim_size_max)

    # Perform alltoall
    dist.all_to_all_single(
        gathered_input,
        input_sorted,
        input_split_sizes=input_split_sizes,
        output_split_sizes=output_split_sizes,
        group=pg,
    )

    # Perform local embedding look up.
    gathered_input_embeddings = torch.nn.functional.embedding(
        gathered_input, local_shard
    )

    # Gather all lookup result appropriately by performing alltoall again
    gathered_output = torch.empty(
        input_sorted.size(0), weight.size(1), device=input.device
    )
    dist.all_to_all_single(
        gathered_output,
        gathered_input_embeddings,
        input_split_sizes=output_split_sizes,
        output_split_sizes=input_split_sizes,
        group=pg,
    )

    # Rearrange the results to its original shape.
    if rearrange_indices_1d_second_order is not None:
        gathered_output = gathered_output[rearrange_indices_1d_second_order]
    gathered_output = gathered_output[rearrange_indices_1d]

    # Return the appropriate local result.
    return torch.reshape(gathered_output, (*input_size, weight.size(1)))
