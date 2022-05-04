# coding=utf-8

from typing import List, cast

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import (
    ReduceOp,
)
from ._common import (
    _communicate_list_to_each_rank,
    _communicate_size_to_each_rank,
    _handle_col_wise_sharding_base,
    _handle_row_wise_lookup_distribute,
    _handle_max_norm_col_wise,
)
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.sharded_tensor import (
    sharded_op_impl,
    ShardedTensor
)


@sharded_op_impl(torch.nn.functional.embedding_bag)
def sharded_embedding_bag(types, args, kwargs, pg):
    """
    Handles ``__torch_function__`` dispatch for ``torch.nn.functional.embedding_bag``.
    This method computes a sharded embedding bag aggregation and has the following limitations:

    1. Supports only sharding of ``weight``.
    2. Supports only ``ChunkShardingSpec``.
    3. Supports only a single local shard per rank.
    4. Supports all specs except for scale_grad_by_freq, sparse, etc.

    Based on the dimension that the weight is sharded on, there are two
    algorithms:

    ROWWISE SHARDING
    ================
    For row-wise sharding the weight is sharded on dimension 0.

    The overall algorithm can be best explained with an example. Let's assume
    the dims for input are (4 x 6) and W are (16 x 17) and W is sharded across
    4 GPUs creating 4 shard of (4 x 17).
    The algorithm is as follows:

    1. First if the input is a 2D tensor, we sort by row. (If it's a 1D tensor, we sort
       the tensor per interval defined by offset.
       For example if the given input is generated within [1, 9] like
       tensor([[ 3,  7,  7,  9,  2,  1],
               [ 0,  0, 14,  5,  3, 12],
               [ 4,  5,  5,  9,  5, 13],
               [10,  3,  0,  7, 13,  9]])
       Then we have the sorted 2D tensor like:
       tensor([[ 1,  2,  3,  7,  7,  9],
               [ 0,  0,  3,  5, 12, 14],
               [ 4,  5,  5,  5,  9, 13],
               [ 0,  3,  7,  9, 10, 13]])
       Note if placement not equal to rank we will rearrange accordingly.
    2. Based on sorted result, we now have the offset like the following:
       [tensor([0, 3, 5, 6]), tensor([0, 3, 4, 4]),
        tensor([0, 0, 4, 5]), tensor([0, 2, 3, 5])]
       Note that embedding bag does allow the offset idx equal to length of
       input or repetitive. For these cases, it return a zero tensor.
    3. Next, we rearrange the sorted tensor into different ranks by first
       flattening it and grouping by ranks. Finally, we get a list of 1D tensors.
       So the given tensor now becomes:
       [tensor([1, 2, 3, 0, 0, 3, 0, 3]), tensor([7, 7, 5, 4, 5, 5, 5, 7]),
        tensor([9, 9, 9, 10]), tensor([12, 14, 13, 13])]
       We sync offsets with IDs. Offset now becomes:
       [tensor([0, 3, 6, 6]), tensor([0, 2, 3, 7]),
        tensor([0, 1, 1, 2]), tensor([0, 0, 2, 3])]
    5. Before we send out the array to other ranks, we need to do the modular operation
       so that each rank do use that for embedding look-up.
       The above ID tensor list will look like the below after performing the moduler of 4:
       [tensor([1, 2, 3, 0, 0, 3, 0, 3]), tensor([3, 3, 1, 0, 1, 1, 1, 3]),
        tensor([1, 1, 1, 2]), tensor([0, 2, 1, 1])]
    4. The example above only happens in one rank and each rank does a very similar thing
       with different rearranged IDs and offsets list. We then send IDs and offsets to the
       corresponding rank. Each rank do the look-up and aggregation on its local shard.
       We then use reduce_scatter to send the result back to each rank and perform the
       aggregation simultaneously.
    5. For "Mean" mode we need to divide by either column size (2D) or the interval length
       defined by the offset. We also need to mask the unexisting row to neg Inf so that
       negative value does not gets wiped out in the "Max" mode.

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
    2. Next we perform local embedding bag operation under the given mode by
       apply each input (4 x 6) with the local shard (16 x 5) ((16 x 2) for the last).
       This results in 4 (5 x 4) ((2 x 4) for the last) matrices on each rank.
       We transpose the aggregation result.
    3. Next, we concatenate these 4 matrices and perform an all2all to share the
       appropriate (5 x 4) or (2 x 4) matrices to each rank.
    4. Now, each rank receives a (17 x 4) matrix which is basically the
       size of the result we need.
    5. If placements are not in order any appropriate rearrangement of columns
       are done for the (17 x 4) matrix and finally we transpose the output again.
    6. If max_norm is specified, we manually sum up the norm and renorm. Because
       the renorm must be in place, we need to override the local_shard to mimic
       this behavior.
    """
    # Validate input params
    _validate_embedding_bag_param(args, kwargs)

    input = args[0]
    weight = args[1]
    offsets = kwargs.get("offsets")
    per_sample_weights = kwargs.get("per_sample_weights")
    mode = kwargs.get("mode")
    max_norm = kwargs.get("max_norm")
    norm_type = kwargs.get("norm_type")
    include_last_offset = kwargs.get("include_last_offset")
    padding_idx = kwargs.get("padding_idx")

    local_shard = weight.local_tensor().contiguous()
    sharding_dim = weight._sharding_spec.dim
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)
    if include_last_offset:
        offsets = offsets[:-1]

    if sharding_dim == 1:
        output, local_shard = _handle_col_wise_sharding(
            input,
            world_size,
            weight,
            local_shard,
            offsets,
            per_sample_weights,
            mode,
            max_norm,
            norm_type,
            padding_idx,
            pg,
        )
        weight.local_shards()[0].tensor = local_shard
        return output
    elif sharding_dim == 0:
        return _handle_row_wise_sharding(
            input,
            world_size,
            weight,
            local_shard,
            offsets,
            per_sample_weights,
            mode,
            max_norm,
            norm_type,
            padding_idx,
            rank,
            pg,
        )
    else:
        raise RuntimeError(
            f"nn.EmbeddingBag weight sharded on dim {sharding_dim} not supported!"
        )


def _validate_embedding_bag_param(args, kwargs):
    """
    Validate input params of sharded embeddingBag op.

    Args:
        input: list of ID used for lookup and aggregation.
        weight: shareded weight tensor.
        kwargs: same as normal EmbeddingBag.

    Return: None.
    """

    input = args[0]
    weight = args[1]
    offsets = kwargs.get("offsets")
    per_sample_weights = kwargs.get("per_sample_weights")
    mode = kwargs.get("mode")
    max_norm = kwargs.get("max_norm")
    norm_type = kwargs.get("norm_type")
    scale_grad_by_freq = kwargs.get("scale_grad_by_freq")
    sparse = kwargs.get("sparse")
    include_last_offset = kwargs.get("include_last_offset")
    padding_idx = kwargs.get("padding_idx")

    # Validate types
    if not isinstance(input, torch.Tensor):
        raise TypeError("input need to be torch.Tensor")
    if offsets is not None and not isinstance(offsets, torch.Tensor):
        raise TypeError("offsets need to be torch.Tensor")
    if per_sample_weights is not None and not isinstance(
        per_sample_weights, torch.Tensor
    ):
        raise TypeError("per_sample_weights need to be torch.Tensor")
    if not isinstance(weight, ShardedTensor):
        raise TypeError("weight needs to be ShardedTensor")
    if len(input.size()) > 2:
        raise ValueError("Input more than 2 dims not supported")
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
    if offsets is not None and len(input.size()) != 1:
        raise ValueError("Input dimension needs to be exactly 1 dim")
    if len(input.size()) == 1 and offsets is None:
        raise ValueError("offsets is required for 1D input")
    if per_sample_weights is not None and per_sample_weights.size() != input.size():
        raise ValueError(
            f"per_sample_weights size {per_sample_weights.size()} not equal to input size {input.size()}"
        )
    if mode is None:
        mode = "mean"
    if mode not in ["sum", "mean", "max"]:
        raise ValueError(f"mode '{mode}' is not supported")
    if scale_grad_by_freq:
        raise RuntimeError(
            'nn.Embedding weight sharded with flag on "scale_grad_by_freq" not supported!'
        )
    if sparse:
        raise RuntimeError(
            'nn.Embedding weight sharded with flag on "sparse" not supported!'
        )
    if include_last_offset and offsets is None:
        raise ValueError('offsets is required for flag "include_last_offset"!')
    if include_last_offset and cast(List[int], offsets)[-1] != input.size(0):
        raise ValueError(
            'offsets need to have the input size in the end when the flag "include_last_offset" is on!'
        )

    if max_norm and max_norm <= 0.0:
        raise ValueError('"max_norm" must be larger than zero!')

    if not isinstance(weight._sharding_spec, ChunkShardingSpec):
        raise ValueError("Only ChunkShardingSpec supported for ShardedTensor ops!")
    if len(weight.local_shards()) != 1:
        raise ValueError("Only one local shard supported!")


def _handle_col_wise_sharding(
    input,
    world_size,
    weight,
    local_shard,
    offsets,
    per_sample_weights,
    mode,
    max_norm,
    norm_type,
    padding_idx,
    pg,
):
    """
    Entry-point function to handle the logic of col-wise sharding of weight
    for embeddingBag. (Detailed explanations of the logic can be found in
    the comment for sharded_embedding_bag.)

    Args:
        input: list of ID used for lookup and aggregation.
        world_size: number of ranks.
        weight: shareded weight tensor.
        local_shard: col-wise shared local weight used for lookup.
        offsets: list of start positions of each bag for 1D input.
        per_sample_weights: weights for weighted sum mode.
        mode: aggregation method of each bag.
        max_norm: If given, each embedding vector with norm larger
            than max_norm is renormalized to have norm max_norm.
            Note: this will modify weight in-place.
        norm_type: The p in the p-norm to compute for the max_norm option.
        padding_idx: If specified, the entries at padding_idx do
            not contribute to the gradient; therefore, the embedding
            vector at padding_idx is not updated during training,
            i.e. it remains as a fixed “pad”.
            Note that the embedding vector at padding_idx is
            excluded from the reduction.
        pg: process group.

    Return:
        output: final result of lookup and aggregation.
        local_shard: col-wise shared local weight used for lookup.
            If max_norm, this will be the renormed weight.
    """
    # allgather the special input of embedding bag first.
    gathered_per_sample_weights = None
    if per_sample_weights is not None:
        gathered_per_sample_weights = [
            torch.zeros_like(per_sample_weights) for _ in range(world_size)
        ]
        dist.all_gather(gathered_per_sample_weights, per_sample_weights, group=pg)
    gathered_offsets = None
    if offsets is not None:
        gathered_offsets = [torch.zeros_like(offsets) for _ in range(world_size)]
        dist.all_gather(gathered_offsets, offsets, group=pg)

    gathered_inputs = None
    if max_norm is not None:
        # max_norm changes the weight in-place
        local_shard, gathered_inputs = _handle_max_norm_col_wise(
            max_norm, norm_type, local_shard, input, world_size, pg
        )

    output = _handle_col_wise_sharding_base(
        torch.nn.functional.embedding_bag,
        1,
        input,
        world_size,
        weight,
        local_shard,
        pg,
        mode=mode,
        gathered_per_sample_weights=gathered_per_sample_weights,
        gathered_offsets=gathered_offsets,
        padding_idx=padding_idx,
        gathered_inputs=gathered_inputs,
    )
    return (output, local_shard)


def _handle_row_wise_sharding(
    input,
    world_size,
    weight,
    local_shard,
    offsets,
    per_sample_weights,
    mode,
    max_norm,
    norm_type,
    padding_idx,
    rank,
    pg,
):
    """
    Entry-point function to handle the logic of row-wise sharding of weight
    for embeddingBag. (Detailed explanations of the logic can be found in
    the comment for sharded_embedding_bag.)

    Args:
        input: list of ID used for lookup and aggregation.
        world_size: number of ranks.
        weight: shareded weight tensor.
        local_shard: row-wise shared local weight used for lookup.
        offsets: list of start positions of each bag for 1D input.
        per_sample_weights: weights for weighted sum mode.
        mode: aggregation method of each bag.
        max_norm: If given, each embedding vector with norm larger
            than max_norm is renormalized to have norm max_norm.
            Note: this will modify weight in-place.
        norm_type: The p in the p-norm to compute for the max_norm option.
        padding_idx: If specified, the entries at padding_idx do
            not contribute to the gradient; therefore, the embedding
            vector at padding_idx is not updated during training,
            i.e. it remains as a fixed “pad”.
            Note that the embedding vector at padding_idx is
            excluded from the reduction.
        rank: # of cuda process.
        pg: process group.

    Returns:
        gathered_output: final result of lookup and aggregation.
    """
    # We sort each interval defined by offset. If 2D, each interval is a row.
    input_size = input.size()
    (
        input_split_sorted_list,
        input_split_sorted_indices,
        split_sizes_1d,
        split_sizes_1d_with_padding,
    ) = _input_split_sort(input, offsets, padding_idx)

    # Within each interval of the sorted list, we first need to distribute
    # each ID to different bucket(rank) and also ensure the rearrangement
    # has been done in case the placement idx not equal to rank.
    # We then perform some simple stats on each interval for the next step
    # If user specifies per_sample_weights we need to rearrange them
    # to be sync with IDs and then distribute them to each rank
    (
        input_combined,
        input_combined_split_sizes,
        offsets_rearrange_list,
        offsets_rearrange_sizes,
        per_sample_weights,
        sharded_dim_size_max,
        padding_idx,
    ) = _sorted_input_distribute_prepare(
        input_split_sorted_list,
        input_split_sorted_indices,
        world_size,
        input,
        weight,
        per_sample_weights,
        rank,
        padding_idx,
    )

    # Send ID/offsets/per_sample_weights to different bucket(rank).
    (
        gathered_input,
        output_offsets_tensor_list,
        output_split_sizes,
        gathered_per_sample_weights,
    ) = _distribute_input(
        input_combined,
        input_combined_split_sizes,
        offsets_rearrange_list,
        offsets_rearrange_sizes,
        sharded_dim_size_max,
        world_size,
        input,
        per_sample_weights,
        pg,
    )

    # Perform the embedding bag look-up and aggregation
    results = []
    for i, inp in enumerate(gathered_input):
        per_sample_weights = (
            gathered_per_sample_weights[i]
            if gathered_per_sample_weights is not None
            else None
        )
        # If input is None, passing in max_norm causes
        # errors in CUDA.
        if max_norm is not None and inp.size(0) == 0:
            max_norm = None

        # Perform local embedding look up and aggregation.
        result = torch.nn.functional.embedding_bag(
            inp,
            local_shard,
            offsets=output_offsets_tensor_list[i],
            mode=mode if mode != "mean" else "sum",
            per_sample_weights=per_sample_weights,
            max_norm=max_norm,
            norm_type=norm_type,
            padding_idx=padding_idx,
        )
        if mode != "max":
            results.append(result)
        # For max case, it there is no look-up from some ranks
        # it will return all zero for that. For that case, we need
        # to set the row to neg inf; otherwise, in the final
        # aggregation negative values will be rounded up to zero.
        elif inp.size(0) == 0:
            result[:] = -float("Inf")
            results.append(result)
        else:
            for idx, current_offset in enumerate(output_offsets_tensor_list[i]):
                next_offset = current_offset
                if idx == len(output_offsets_tensor_list[i]) - 1:
                    next_offset = output_split_sizes[i]
                else:
                    next_offset = output_offsets_tensor_list[i][idx + 1]
                # When there is no interval in the current rank or all IDs
                # are equal to padding_idx, we then need to ensure they
                # don't contribute to the final result.
                if (current_offset == next_offset) or (
                    padding_idx is not None
                    and not torch.any(
                        torch.ne(inp[current_offset:next_offset], padding_idx)
                    )
                ):
                    result[idx] = -float("Inf")
            results.append(result)

    # Gather all the aggregated results appropriately by using reduce_scatter.
    row_size = input.size(0) if len(input_size) > 1 else len(split_sizes_1d)
    gathered_output = torch.empty(row_size, weight.size(1), device=input.device)
    op = ReduceOp.SUM if mode != "max" else ReduceOp.MAX
    dist.reduce_scatter(gathered_output, results, op=op, group=pg)

    # For Mean, we cannot do the division until very end because the sum of means
    # not equal to the mean of sum. (Divisor is different)
    if mode == "mean":
        split_sizes_1d_tensor = torch.tensor(
            split_sizes_1d_with_padding, dtype=torch.float, device=input.device
        )
        # Make sure divisor is not zero.
        split_sizes_1d_tensor[split_sizes_1d_tensor == 0.0] = 1.0
        return (
            torch.div(gathered_output.t().contiguous(), split_sizes_1d_tensor)
            .t()
            .contiguous()
        )

    # Return the appropriate local result.
    return gathered_output


def _input_split_sort(input, offsets, padding_idx):
    """
    In the circumstance of row-wise sharding of weight, we need to distribute
    the sorted lookup IDs of embeddingBag to each rank by range. The constraint
    here is that we can not directly sort the whole input because we have to
    differentiate between each interval because the result is aggregated.

    If the index in the placement is not equal to the rank number, we need to
    do the rearrangement based on the order given by the Sharding Spec (placement).

    We also calculate the split_size with padding_idx excluded per interval
    so that we can use it as the divisor to calculate the mean correctly.

    Args:
        input: tensor to be applied op on.
        offsets: start index of each interval in the 1D case.
        padding_idx: the embedding vector at padding_idx is
            excluded from the reduction.

    Return:
        input_split_sorted_list: list of ID positions sorted per interval.
        input_split_sorted_indices: sorted indices for per_sample_weights
            rearrangments.
        split_sizes_1d: size of each split for 1D input because it can be
            different in such scenario.
        split_sizes_1d_with_padding: size of each split for 1D input with
            padding_idx excluded. This is for the divisor of `mean` mode.
    """
    input_size = input.size()
    input_split_sorted_list = []
    split_sizes_1d = []
    split_sizes_1d_with_padding = []
    padding_idx = padding_idx if padding_idx is not None else -1

    # For 2D tensor, we just first sort and then append row by row into a list.
    if len(input_size) > 1:
        indice_offset = 0
        sorted_input, input_split_sorted_indices = torch.sort(input)
        for i in range(0, sorted_input.size(0)):
            input_split_sorted_list.append(sorted_input[i])
            input_split_sorted_indices[i] += indice_offset
            indice_offset += input.size(1)
            split_sizes_1d_with_padding.append(
                torch.sum(torch.ne(sorted_input[i], padding_idx)).item()
            )
        input_split_sorted_indices = torch.reshape(input_split_sorted_indices, (-1,))
    # Split 1D input tensor based on the given offsets.
    else:
        input_split_sorted_indices_list = []
        offset_len = len(offsets)
        split_size = offsets[1:offset_len] - offsets[0:-1]
        split_sizes_1d = split_size.tolist()
        if torch.sum(split_size) < input.size(0):
            split_sizes_1d.append(input.size(0) - offsets[-1].item())
        indice_offset = 0
        for idx, split_result in enumerate(torch.split(input, split_sizes_1d)):
            split_result_sorted, indices = torch.sort(split_result)
            input_split_sorted_list.append(split_result_sorted)
            split_sizes_1d_with_padding.append(
                torch.sum(torch.ne(split_result_sorted, padding_idx)).item()
            )
            input_split_sorted_indices_list.append(indices + indice_offset)
            indice_offset += split_sizes_1d[idx]
        input_split_sorted_indices = torch.cat(input_split_sorted_indices_list)

    return (
        input_split_sorted_list,
        input_split_sorted_indices,
        split_sizes_1d,
        split_sizes_1d_with_padding,
    )


def _sorted_input_distribute_prepare(
    input_split_sorted_list,
    input_split_sorted_indices,
    world_size,
    input,
    weight,
    per_sample_weights,
    rank,
    padding_idx,
):
    """
    In the circumstance of row-wise sharding of weight, we need to distribute
    the sorted lookup IDs of embeddingBag to each rank by range. After sorting
    per interval, we need to distribute each position to the corresponding
    rank and we need to sync this change to offsets and per_sample_weights.
    Also, we perform rearrangements, if the order in Sharding Spec is not
    same as the rank sequence.

    In addition, in the row-wise sharding, we need to do two things for
    padding_idx. The first thing is only to set it if it's within the range
    of the current rank and the other thing is to do the modularization of
    it by sharded_dim_size_max.

    Args:
        input_split_sorted_list: list of ID positions sorted per interval.
        input_split_sorted_indices: sorted indices for per_sample_weights
            rearrangments.
        input: tensor to be applied op on.
        world_size: number of ranks.
        weight: shareded weight tensor.
        per_sample_weights: weights for weighted sum mode.
        rank: # of cuda process.
        padding_idx: If specified, the entries at padding_idx do
            not contribute to the gradient and reduction.

    Returns:
        input_combined: list of ID to be sent to each rank.
        input_combined_split_sizes: # of bags sent to each rank.
        offsets_rearrange_list: list of starting position of each bag.
        offsets_rearrange_sizes: # of bag offsets sent to each rank.
        per_sample_weights: weights for weighted sum mode.
        sharded_dim_size_max: the max size of the row each rank gets.
        padding_idx: Modularized padding_idx if it is within the range,
            otherwise, None is returned.
    """
    input_sorted_list = []
    input_split_sizes_list = []
    input_split_sizes_rolling_sum = []
    rearrange_indices_list = []
    input_split_rearrange_indices_combined = None
    split_sizes_rolling_sum = 0
    for idx, split_result_sorted in enumerate(input_split_sorted_list):
        split_result_sorted.contiguous()
        (
            input_sorted,
            input_split_sizes,
            sharded_dim_size_max,
            input_split_rearrange_indices,
            _,
            padding_idx_modular,
        ) = _handle_row_wise_lookup_distribute(
            split_result_sorted, input, world_size, weight, rank, padding_idx
        )
        rearrange_indices_list.append(
            input_split_rearrange_indices + split_sizes_rolling_sum
            if input_split_rearrange_indices is not None
            else None
        )
        input_sorted_list.append(input_sorted)
        input_split_sizes_list.append(input_split_sizes)
        input_split_sizes_rolling_sum.append(split_sizes_rolling_sum)
        split_sizes_rolling_sum += sum(input_split_sizes)

    # padding_idx cannot be directly overridden in the for loop because the
    # later iteration will wipe out the modularized padding_idx.
    padding_idx = padding_idx_modular
    if not (any(x is None for x in rearrange_indices_list)):
        input_split_rearrange_indices_combined = torch.cat(rearrange_indices_list)

    # Flatten each interval into a big 1D tensor.
    input_combined = torch.cat(input_sorted_list)

    # Rearrange the 1D tensor to move the IDs of look-up within each
    # interval to the corresponding sharding rank. We also rearrange
    # the offsets to be in sync with IDs.
    input_combined_rearrange_indices = []
    offsets_rearrange_list = []
    offsets_rearrange_sizes = []
    input_combined_split_sizes = []
    # Calculate the indices for rearrangements
    for rank in range(0, world_size):
        offsets_rearrange = []
        offset = 0
        for idx, input_split_sizes in enumerate(input_split_sizes_list):
            offsets_rearrange.append(offset)
            split_length = input_split_sizes[rank]
            offset_idx = input_split_sizes_rolling_sum[idx] + sum(
                [
                    split_size if i < rank else 0
                    for i, split_size in enumerate(input_split_sizes)
                ]
            )
            input_combined_rearrange_indices += list(
                range(offset_idx, offset_idx + split_length)
            )
            offset += split_length
        offsets_rearrange_list.append(offsets_rearrange)
        offsets_rearrange_sizes.append(len(offsets_rearrange))
        input_combined_split_sizes.append(offset)

    # Perform the actual rearrangements of IDs
    input_combined = input_combined.index_select(
        0, torch.tensor(input_combined_rearrange_indices, device=input.device)
    )

    # If per_sample_weights exists, we need to sync the shift which
    # we applied to the position IDs for look-up.
    if per_sample_weights is not None:
        # Rearrange per interval.
        per_sample_weights = torch.reshape(per_sample_weights, (-1,))
        per_sample_weights = per_sample_weights[input_split_sorted_indices]
        if input_split_rearrange_indices_combined is not None:
            per_sample_weights = per_sample_weights[
                input_split_rearrange_indices_combined
            ]
        # Rearrange across different ranks.
        per_sample_weights = per_sample_weights.index_select(
            0,
            torch.tensor(input_combined_rearrange_indices, device=input.device),
        )

    return (
        input_combined,
        input_combined_split_sizes,
        offsets_rearrange_list,
        offsets_rearrange_sizes,
        per_sample_weights,
        sharded_dim_size_max,
        padding_idx,
    )


def _distribute_input(
    input_combined,
    input_combined_split_sizes,
    offsets_rearrange_list,
    offsets_rearrange_sizes,
    sharded_dim_size_max,
    world_size,
    input,
    per_sample_weights,
    pg,
):
    """
    In the circumstance of row-wise sharding of weight, we need to distribute
    the sorted lookup IDs of embeddingBag, offsets and per_sample_weights to
    each rank by range. To save the # of communication, we consolidate the
    communication of tensors which shares the same dtype.

    Args:
        input_combined: list of ID to be sent to each rank.
        input_combined_split_sizes: # of bags sent to each rank.
        offsets_rearrange_list: list of starting position of each bag.
        offsets_rearrange_sizes: # of bag offsets sent to each rank.
        sharded_dim_size_max: the max size of the row each rank gets.
        world_size: number of ranks.
        input: tensor to be applied op on.
        per_sample_weights: weights for weighted sum mode.
        pg: process group.

    Returns:
        gathered_input: list of tensors of IDs for lookup and aggregation.
        output_offsets_tensor_list: list of tensors of offsets which specifies the
            boundary of each bag.
        output_split_sizes: list of size of IDs sent from each rank.
        gathered_per_sample_weights: per_sample_weights from each rank.
    """
    # Communicate the length of offset and ID split size to each rank
    # To save the # of communications, we interleave the sizes into one list.
    input_size_list = offsets_rearrange_sizes + input_combined_split_sizes
    input_size_list[::2] = offsets_rearrange_sizes
    input_size_list[1::2] = input_combined_split_sizes
    output_size_list = _communicate_size_to_each_rank(
        input_size_list, world_size * 2, input, pg
    )

    # Perform the modular operation of the 1D tensor to be sent to each rank.
    input_combined = torch.remainder(input_combined, sharded_dim_size_max)
    input_combined_list = list(torch.split(input_combined, input_combined_split_sizes))

    # Covert each offset list to a tensor and combine with the input
    # so we only perform one communication to each rank.
    input_tensor_list = []
    output_tensor_size_list = []
    for idx, input_list in enumerate(offsets_rearrange_list):
        input_tensor_list.append(
            torch.cat(
                (
                    torch.tensor(input_list, dtype=torch.int64, device=input.device),
                    input_combined_list[idx],
                )
            )
        )
        output_tensor_size_list.append(
            output_size_list[2 * idx] + output_size_list[2 * idx + 1]
        )

    output_tensor_list = _communicate_list_to_each_rank(
        input_tensor_list, output_tensor_size_list, input, pg
    )
    output_tensor_list = list(
        torch.split(torch.cat(output_tensor_list), output_size_list)
    )
    output_offsets_tensor_list = output_tensor_list[::2]
    gathered_input = output_tensor_list[1::2]
    output_split_sizes = output_size_list[1::2]

    # If user specifies per_sample_weights we need to communicate
    # them to the corresponding rank.
    gathered_per_sample_weights = None
    if per_sample_weights is not None:
        # Split the 1D tensor per_sample_weights to be sent to each rank.
        per_sample_weights_list = list(
            torch.split(per_sample_weights, input_combined_split_sizes)
        )
        gathered_per_sample_weights = _communicate_list_to_each_rank(
            per_sample_weights_list,
            output_split_sizes,
            input,
            pg,
            tensor_type=per_sample_weights.dtype,
        )

    return (
        gathered_input,
        output_offsets_tensor_list,
        output_split_sizes,
        gathered_per_sample_weights,
    )
