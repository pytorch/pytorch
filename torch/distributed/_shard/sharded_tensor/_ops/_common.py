# coding=utf-8

from typing import List

import torch
import torch.distributed as dist
from torch.distributed._shard.sharding_spec._internals import (
    get_split_size,
    get_chunked_dim_size,
)
from torch.distributed.nn.functional import (
    all_gather,
    all_to_all_single,
)


def _handle_col_wise_sharding_base(
    op_func,
    col_dim,
    input,
    world_size,
    weight,
    local_shard,
    pg,
    gathered_inputs=None,
    mode=None,
    gathered_per_sample_weights=None,
    gathered_offsets=None,
    padding_idx=None,
):
    """
    For col-wise sharding of weight, lots of logic are common.
    So we extract the common logic and put in this function:
    Step 1. To get input from each rank and
    Step 2. To perform the op on the concatenated tensor.
    Step 3. To distribute results to each rank with col rearrangement.
    Step 4. To concatenate all results from all ranks.

    Args:
        op_func: operator which is applied to the input tensor.
        col_dim: dim of result tensor after the operation.
        input: tensor to be applied op on.
        world_size: number of ranks.
        weight: shareded weight tensor.
        local_shard: col-wise sharded weight tensor.
        pg: process group.
        gathered_inputs: list of inputs from all ranks. If specified, we
            don't need to communicate with each rank any more.
        mode: aggregation mode of EmbeddingBag.
        gathered_per_sample_weights: per_sample_weights across all ranks.
        gathered_offsets: offsets across all ranks.
        padding_idx: If specified, the entries at padding_idx do
            not contribute to the gradient; therefore, the embedding
            vector at padding_idx is not updated during training,
            i.e. it remains as a fixed “pad”.
            Note that the embedding vector at padding_idx is
            excluded from the reduction.

    Return: final result of input being applied with the op.
    """
    if gathered_inputs is None:
        # allgather the inputs first.
        gathered_inputs = all_gather(input, group=pg)

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
                padding_idx=padding_idx,
            )
        elif op_func == torch.nn.functional.embedding:
            result = op_func(
                inp,
                local_shard,
                padding_idx=padding_idx,
            )
        else:
            result = op_func(inp, local_shard)
        results.append(torch.transpose(result, 0, col_dim))

    # Distribute results to each rank with col rearrangement.
    output = _result_distribute_with_col_rearrange(
        results, input, world_size, weight, pg
    )

    # transpose the output and return result.
    return torch.transpose(output, 0, col_dim)


def _result_distribute_with_col_rearrange(
    results, input, world_size, weight, pg
):
    """
    For col-wise sharding of weight, we need to distribute
    results to each rank. We do them in this function.
    Note that, if the index in the Sharding Spec is not equal to
    the rank number, we need to do the rearrangement based on the
    order given by the Sharding Spec (placement).

    Args:
        results: results from ops applied to inputs from all ranks.
            We need to distribute them back to their original ranks.
        input: tensor to be applied op to.
        world_size: number of ranks.
        weight: shareded weight tensor.
        pg: process group.

    Return: column rearranged result.
    """
    # Process results and outputs for all2all.
    sharding_dim = weight._sharding_spec.dim
    sharding_dim_size = weight.size(sharding_dim)
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
    output = all_to_all_single(
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


def _handle_row_wise_lookup_distribute(
    input_sorted, input, world_size, weight, rank, padding_idx
):
    """
    In the circumstance of row-wise sharding of weight, we need to distribute
    the sorted lookup IDs of embedding/embeddingBag to each rank.
    If the index in the placement is not equal to the rank number, we need to
    do the rearrangement based on the order given by the Sharding Spec (placement).

    In addition, we do two things for padding_idx. The first thing is to only
    set it if it's within the range of the current rank and the other thing
    is to do the modularization of it by sharded_dim_size_max.

    Args:
        input_sorted: sorted lookup IDs of embedding/embeddingBag.
        input: tensor to be applied op on.
        world_size: number of ranks.
        weight: shareded weight tensor.
        rank: # of cuda process.
        padding_idx: If specified, the entries at padding_idx do
            not contribute to the gradient and reduction.

    Return:
        input_sorted: sorted lookup IDs of embedding/embeddingBag
            Rearrangement performed if it is needed.
        input_split_sizes: size of IDs to be assigned to each rank.
        sharded_dim_size_max: the max size of the row each rank gets.
        input_split_rearrange_indices: indices of row rearrangement.
        rearrange_indices_1d_second_order: reverse indices of row
            rearrangement, which will be used to restore the original
            order.
        padding_idx: Same as input if padding_idx is within the range
            of the given rank; otherwise, None is returned. It is
            also modularized by sharded_dim_size_max.
    """
    # Decide which rank the input goes to by check the sharding range.
    split_size = get_split_size(weight.size(0), world_size)
    rearrange_rows = False
    indices_flatten = None
    input_split_sizes: List[int] = [0] * world_size
    input_split_start_indices: List[int] = [0] * world_size
    start_row_idx_rank = None
    end_row_idx_rank = None
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
        # Store the range of the current rank.
        if placement.rank() == rank:
            start_row_idx_rank = start_row_idx
            end_row_idx_rank = end_row_idx

    # Perform the modular if padding_idx is within the range.
    if padding_idx is not None:
        if padding_idx < start_row_idx_rank or padding_idx >= end_row_idx_rank:
            padding_idx = None
        else:
            padding_idx = padding_idx % sharded_dim_size_max

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
        padding_idx,
    )


def _communicate_size_to_each_rank(
    input_size_list, output_size, input, pg, tensor_type=torch.int
):
    """
    In the circumstance of row-wise sharding of weight, we need to first
    communicate the input length to each rank because each rank gets a
    different one.

    Args:
        input_size_list: list of sizes to be sent to each rank.
        output_size: length of the output tensor.
        input: tensor to be applied op on.
        pg: process group.
        tensor_type: dtype of tensor.

    Return: A list of communication results (int).
    """
    input_size_list_tensor = torch.tensor(
        input_size_list, dtype=tensor_type, device=input.device
    )
    output_size_list_tensor = torch.empty(
        output_size, dtype=tensor_type, device=input.device
    )
    dist.all_to_all_single(
        output_size_list_tensor,
        input_size_list_tensor,
        group=pg,
    )
    return output_size_list_tensor.tolist()


def _communicate_list_to_each_rank(
    input_tensor_list, output_lists, input, pg, tensor_type=torch.int64
):
    """
    In the circumstance of row-wise sharding of weight, we need to
    communicate a list of input tensors to each rank. Because the
    input could be a list of list, we need to first convert the list
    to a tensor.

    Args:
        input_tensor_list: list of tensors to be sent to each rank.
        output_lists: list of sizes to be obtained from each rank.
        input: tensor to be applied op on.
        pg: process group.
        tensor_type: dtype of tensor.

    Return: A list of communication results (tensors).
    """
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


def _handle_max_norm_col_wise(
    max_norm,
    norm_type,
    local_shard,
    input,
    world_size,
    pg,
):
    """
    For col-wise sharding of weight, we need to aggregate the
    norm across all ranks before we can perform the proper re-norm.
    Note that, the max_norm logic is only applied to the embedding
    indices that are looked up and not the whole shard.

    Args:
        max_norm: If given, each embedding vector with norm larger
            than max_norm is renormalized to have norm max_norm.
            Note: this will modify weight in-place.
        norm_type: The p in the p-norm to compute for the max_norm option.
        local_shard: col-wise shared local weight used for lookup.
        input: tensor to be applied op to.
        world_size: number of ranks.
        pg: process group.

    Return:
        local_shard_norm_renormed: local_shard re-normed to max_norm if the norm is larger
            than it.
        gathered_inputs: list of inputs from all ranks.
    """
    norm_type = norm_type if norm_type is not None else 2.0
    # allgather the inputs first.
    gathered_inputs = [torch.zeros_like(input) for _ in range(world_size)]
    dist.all_gather(gathered_inputs, input, group=pg)
    unique_inp = torch.unique(torch.cat(gathered_inputs))
    local_shard_sum = torch.sum(
        torch.pow(torch.abs(local_shard), norm_type), dim=1, dtype=local_shard.dtype
    )
    # For col-wise sharding, we need to first aggregate the powered sum
    # from each rank first and then calculate the norm.
    dist.all_reduce(local_shard_sum, group=pg)
    local_shard_norm = torch.pow(local_shard_sum, 1.0 / norm_type)
    max_norm_tensor = torch.full(
        (local_shard.size(0),),
        float("inf"),
        dtype=local_shard.dtype,
        device=input.device,
    )
    max_norm_tensor[unique_inp] = max_norm
    local_shard_t = local_shard.t().contiguous()
    normalized_tensor = torch.where(
        local_shard_norm > max_norm_tensor, max_norm_tensor, local_shard_norm
    )
    # Make sure divisor is not zero.
    local_shard_norm[local_shard_norm == 0.0] = 1.0
    local_shard_norm_renormed = (
        torch.div(torch.mul(local_shard_t, normalized_tensor), local_shard_norm)
        .t()
        .contiguous()
    )
    return local_shard_norm_renormed, gathered_inputs
