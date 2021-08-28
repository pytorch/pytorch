import torch
import torch.distributed as dist
from torch.distributed._sharding_spec import ChunkShardingSpec
from torch.distributed._sharding_spec._internals import (
    get_split_size,
    get_chunked_dim_size,
)


def sharded_linear(types, args, kwargs, pg):
    from torch.distributed._sharded_tensor import ShardedTensor

    input = args[0]
    weight = args[1]
    bias = kwargs["bias"]

    # Validate types
    if not isinstance(input, torch.Tensor) or not isinstance(bias, torch.Tensor):
        raise TypeError("input and bias need to be torch.Tensor")
    if not isinstance(weight, ShardedTensor):
        raise TypeError("weight needs to be ShardedTensor")

    if input.size()[1] != weight.size()[1]:
        raise ValueError(
            f'Input dim: {input.size()[1]} does not match '
            f'appropriate weight dim: {weight.size()[1]}')
    if not isinstance(weight._sharding_spec, ChunkShardingSpec):
        raise ValueError("Only ChunkShardingSpec supported for ShardedTensor ops!")
    if len(weight.local_shards()) != 1:
        raise ValueError("Only one local shard supported!")

    local_shard = weight.local_shards()[0].tensor
    local_shard_t = local_shard.t().contiguous()
    sharding_dim = weight._sharding_spec.dim
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)

    if sharding_dim == 1:
        return _handle_row_wise_sharding(input, world_size, weight, rank, local_shard_t, bias, pg)
    elif sharding_dim == 0:
        return _handle_col_wise_sharding(input, world_size, weight, local_shard_t, bias, pg)
    else:
        raise RuntimeError(f'nn.Linear weight sharded on dim {sharding_dim} not supported!')

def _handle_col_wise_sharding(input, world_size, weight, local_shard_t, bias, pg):
    # allgather the inputs first.
    gathered_inputs = [torch.zeros_like(input) for _ in range(world_size)]
    dist.all_gather(gathered_inputs, input, group=pg)

    # matmul all the inputs.
    results = []
    for i, inp in enumerate(gathered_inputs):
        results.append(inp.matmul(local_shard_t).t())

    # Process inputs and outputs for all2all.
    sharding_dim_size = weight.size()[0]
    output = torch.empty((sharding_dim_size, input.size(0)), device=input.device)
    combined_results = torch.cat(results)

    # Compute output splits
    split_size = get_split_size(sharding_dim_size, world_size)
    output_split_sizes = [
        get_chunked_dim_size(sharding_dim_size, split_size, placement.rank())
        for placement in weight._sharding_spec.placements
    ]

    # distribute the outputs using all2all.
    dist.all_to_all_single(output, combined_results, output_split_sizes=output_split_sizes, group=pg)

    # Check if we need to rearrange rows appropriately for output.
    rearrange_rows = any([idx != placement.rank() for idx, placement in enumerate(weight._sharding_spec.placements)])
    if rearrange_rows:
        indices = []
        for placement in weight._sharding_spec.placements:
            dim_size = output_split_sizes[placement.rank()]
            start = sum([split_size if i < placement.rank() else 0 for i, split_size in enumerate(output_split_sizes)])
            indices += list(range(start, start + dim_size))

        output = output.index_select(0, torch.tensor(indices, device=output.device))

    # add bias and return result.
    return output.t() + bias

def _handle_row_wise_sharding(input, world_size, weight, rank, local_shard_t, bias, pg):
    # alltoall to gather all the appropriate inputs.
    input_t = input.t().contiguous()
    input_t_size = input_t.size()

    # Compute expected size
    split_size = get_split_size(input_t_size[0], world_size)
    input_split_sizes = [0] * world_size
    rearrange_rows = False

    for idx, placement in enumerate(weight._sharding_spec.placements):
        sharded_dim_size = get_chunked_dim_size(input_t_size[0], split_size, idx)
        input_split_sizes[placement.rank()] = sharded_dim_size
        if placement.rank() != idx:
            rearrange_rows = True

    if rearrange_rows:
        # Need to re-arrange rows of input_t for all2all.
        indices = []
        for placement in weight._sharding_spec.placements:
            sharded_dim_size = get_chunked_dim_size(input_t_size[0], split_size, placement.rank())
            input_idx = placement.rank() * split_size
            indices += range(input_idx, input_idx + sharded_dim_size)

        input_t = input_t.index_select(0, torch.tensor(indices, device=input_t.device))

    gathered_input = torch.empty(input_split_sizes[rank] * world_size, input_t_size[1], device=input_t.device)

    # Perform alltoall
    dist.all_to_all_single(gathered_input, input_t, input_split_sizes=input_split_sizes, group=pg)
    gathered_input = gathered_input.t()

    # Perform local matmuls for all shards
    shard_size = local_shard_t.size()[0]
    results = []
    for r in range(world_size):
        inp = torch.narrow(gathered_input, 1, r * shard_size, shard_size)
        results.append(inp.matmul(local_shard_t))

    # Gather all the results appropriately.
    local_result = torch.empty_like(results[rank])
    dist.reduce_scatter(local_result, results, group=pg)

    # Return the appropriate local result.
    return local_result + bias
