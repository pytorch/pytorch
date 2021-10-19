import torch
import torch.distributed as dist
from torch.distributed._sharding_spec._internals import (
    get_split_size,
    get_chunked_dim_size,
)


def _handle_col_wise_sharding_common(
    op_func, sharding_dim_size, col_dim, input, world_size, weight, local_shard, pg
):
    # allgather the inputs first.
    gathered_inputs = [torch.zeros_like(input) for _ in range(world_size)]
    dist.all_gather(gathered_inputs, input, group=pg)

    # run the operator's function for all the inputs.
    results = []
    for i, inp in enumerate(gathered_inputs):
        result = op_func(inp, local_shard)
        results.append(torch.transpose(result, 0, col_dim))

    # Distribute results to each rank with col rearrangement.
    output = _result_distribute_with_col_rearrange(
        results, input, sharding_dim_size, world_size, weight, pg
    )

    # transpose the output and return result.
    return torch.transpose(output, 0, col_dim)


def _result_distribute_with_col_rearrange(
    results, input, sharding_dim_size, world_size, weight, pg
):
    # Process results and outputs for all2all.
    dims = list(results[0].size())
    dims[0] = sharding_dim_size
    output = torch.empty(*dims, device=input.device)
    combined_results = torch.cat(results)

    # Compute output splits
    split_size = get_split_size(sharding_dim_size, world_size)
    output_split_sizes = [0] * world_size
    for idx, placement in enumerate(weight._sharding_spec.placements):
        output_split_sizes[placement.rank()] = get_chunked_dim_size(
            sharding_dim_size, split_size, idx
        )

    # distribute the outputs using all2all.
    dist.all_to_all_single(
        output, combined_results, output_split_sizes=output_split_sizes, group=pg
    )

    # Check if we need to rearrange columns appropriately for output.
    rearrange_columns = any(
        [
            idx != placement.rank()
            for idx, placement in enumerate(weight._sharding_spec.placements)
        ]
    )
    if not rearrange_columns:
        return output

    indices = []
    for placement in weight._sharding_spec.placements:
        dim_size = output_split_sizes[placement.rank()]
        start = sum(
            [
                split_size if i < placement.rank() else 0
                for i, split_size in enumerate(output_split_sizes)
            ]
        )
        indices += list(range(start, start + dim_size))

    return output.index_select(0, torch.tensor(indices, device=output.device))
