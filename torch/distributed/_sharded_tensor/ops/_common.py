from typing import List

import torch
import torch.distributed as dist
from torch.distributed._sharding_spec._internals import (
    get_split_size,
    get_chunked_dim_size,
)


def _handle_col_wise_sharding_common(
    op_func,
    sharding_dim_size,
    col_dim,
    input,
    world_size,
    weight,
    local_shard,
    pg,
    mode=None,
    gathered_per_sample_weights=None,
    gathered_offsets=None,
):
    # allgather the inputs first.
    gathered_inputs = [torch.zeros_like(input) for _ in range(world_size)]
    dist.all_gather(gathered_inputs, input, group=pg)

    # run the operator's function for all the inputs.
    results = []
    for i, inp in enumerate(gathered_inputs):
        if op_func == torch.nn.functional.embedding_bag:
            result = op_func(
                inp,
                local_shard,
                offsets=gathered_offsets[i] if gathered_offsets is not None else None,
                mode=mode,
                per_sample_weights=gathered_per_sample_weights[i]
                if gathered_per_sample_weights is not None
                else None,
            )
        else:
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


def _handle_lookup_distribute_common(input_sorted, input, world_size, weight):
    # Decide which rank the input goes to by check the sharding range.
    split_size = get_split_size(weight.size(0), world_size)
    rearrange_rows = False
    indices_flatten = None
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

    return (
        input_sorted,
        input_split_sizes,
        sharded_dim_size_max,
        torch.tensor(indices_flatten, device=input.device) if rearrange_rows else None,
        rearrange_indices_1d_second_order,
    )


def _communicate_size_to_each_rank_common(
    input_size_list, world_size, input, pg, tensor_type=torch.int
):
    input_size_list_tensor = torch.tensor(
        input_size_list, dtype=tensor_type, device=input.device
    )
    output_size_list_tensor = torch.empty(
        world_size, dtype=tensor_type, device=input.device
    )
    dist.all_to_all_single(
        output_size_list_tensor,
        input_size_list_tensor,
        group=pg,
    )
    return output_size_list_tensor.tolist()


def _communicate_list_to_each_rank_common(
    input_lists, output_lists, world_size, input, pg, tensor_type=torch.int64
):
    input_tensor_list = []
    for input_list in input_lists:
        if isinstance(input_list, torch.Tensor):
            input_tensor_list = input_lists
            break
        else:
            input_tensor_list.append(
                torch.tensor(input_list, dtype=tensor_type, device=input.device)
            )
    output_tensor_list = []
    for output_list in output_lists:
        output_tensor_list.append(
            torch.empty(output_list, dtype=tensor_type, device=input.device)
        )
    dist.all_to_all(
        output_tensor_list,
        input_tensor_list,
        group=pg,
    )
    return output_tensor_list
