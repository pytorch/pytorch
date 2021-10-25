from typing import cast

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import (
    ReduceOp,
)
from torch.distributed._sharded_tensor.ops._common import (
    _communicate_list_to_each_rank_common,
    _communicate_size_to_each_rank_common,
    _handle_col_wise_sharding_common,
    _handle_lookup_distribute_common,
)
from torch.distributed._sharding_spec import ChunkShardingSpec


def sharded_embedding_bag(types, args, kwargs, pg):
    """
    Handles ``__torch_function__`` dispatch for ``torch.nn.functional.embedding_bag``.
    This method computes a sharded embedding bag aggregation and has the following limitations:

    1. Supports only sharding of ``weight``.
    2. Supports only ``ChunkShardingSpec``.
    3. Supports only a single local shard per rank.
    4. Supports only limited specs like offsets, per_sample_weights, and excluding specs
       like padding_idx, max_norm, etc.

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
       Note if placement not equal to rank we will rearrage accordingly.
    2. Based on sorted result, we now have the offset like the following:
       [tensor([0, 3, 5, 6]), tensor([0, 3, 4, 4]),
        tensor([0, 0, 4, 5]), tensor([0, 2, 3, 5])]
       Note that embedding bag does allow the offset idx equal to length of
       input or repeative. For these cases, it return a zero tensor.
    3. Next, we rearranges the sorted tensor into different ranks by first
       flattening it and rearranging and then split to a list of 1D tensors.
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
       corresponding rank. Each rank do the look-up and aggergation on its local shard.
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
    3. Next, we concat these 4 matrices and perform an all2all to share the
       appropriate (5 x 4) or (2 x 4) matrices to each rank.
    4. Now, each rank receives a (17 x 4) matrix which is basically the
       size of the result we need.
    5. If placements are not in order any appropriate rearrangement of columns
       are done for the (17 x 4) matrix and finally we transponse the output again.
    """
    from torch.distributed._sharded_tensor import ShardedTensor

    input = args[0]
    weight = args[1]
    offsets = kwargs["offsets"]
    per_sample_weights = kwargs["per_sample_weights"]
    mode = kwargs["mode"]

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

    if not isinstance(weight._sharding_spec, ChunkShardingSpec):
        raise ValueError("Only ChunkShardingSpec supported for ShardedTensor ops!")
    if len(weight.local_shards()) != 1:
        raise ValueError("Only one local shard supported!")

    local_shard = weight.local_shards()[0].tensor.contiguous()
    sharding_dim = weight._sharding_spec.dim
    world_size = dist.get_world_size(pg)

    if sharding_dim == 1:
        return _handle_col_wise_sharding(
            input,
            world_size,
            weight,
            local_shard,
            offsets,
            per_sample_weights,
            mode,
            pg,
        )
    elif sharding_dim == 0:
        return _handle_row_wise_sharding(
            input,
            world_size,
            weight,
            local_shard,
            offsets,
            per_sample_weights,
            mode,
            pg,
        )
    else:
        raise RuntimeError(
            f"nn.EmbeddingBag weight sharded on dim {sharding_dim} not supported!"
        )


def _handle_col_wise_sharding(
    input, world_size, weight, local_shard, offsets, per_sample_weights, mode, pg
):
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

    return _handle_col_wise_sharding_common(
        torch.nn.functional.embedding_bag,
        weight.size(1),
        len(input.size()),
        input,
        world_size,
        weight,
        local_shard,
        pg,
        mode=mode,
        gathered_per_sample_weights=gathered_per_sample_weights,
        gathered_offsets=gathered_offsets,
    )


def _handle_row_wise_sharding(
    input, world_size, weight, local_shard, offsets, per_sample_weights, mode, pg
):
    # We sort each interval defined by offset. If 2D, each interval is a row.
    input_size = input.size()
    input_split_sorted_list = []

    # For 2D tensor, we just first sort and then append row by row into a list.
    if len(input_size) > 1:
        indice_offset = 0
        sorted_input, input_split_sorted_indices = torch.sort(input)
        for i in range(0, sorted_input.size(0)):
            input_split_sorted_list.append(sorted_input[i])
            input_split_sorted_indices[i] += indice_offset
            indice_offset += input.size(1)
        input_split_sorted_indices = torch.reshape(input_split_sorted_indices, (-1,))
    # Split 1D input tensor based on the given offsets
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
            input_split_sorted_indices_list.append(indices + indice_offset)
            indice_offset += split_sizes_1d[idx]
        input_split_sorted_indices = torch.cat(input_split_sorted_indices_list)

    # Within each interval of sorted list, we first need to distribute
    # each ID to different bucket(rank) and also ensure the rearrangement
    # has been done in case the placement idx not equal to rank
    # We then perform some simple stats on each interval for the next step
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
        ) = _handle_lookup_distribute_common(
            split_result_sorted, input, world_size, weight
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

    # Communicate the length of offset to each rank
    output_offsets_sizes = _communicate_size_to_each_rank_common(
        offsets_rearrange_sizes, world_size, input, pg
    )

    # Covert each offset from a list to a list of tensors to send to each rank
    output_offsets_tensor_list = _communicate_list_to_each_rank_common(
        offsets_rearrange_list,
        output_offsets_sizes,
        world_size,
        input,
        pg,
    )

    # Get the input split size to be sent from each rank to the current rank.
    # We can then infer the output split size.
    output_split_sizes = _communicate_size_to_each_rank_common(
        input_combined_split_sizes, world_size, input, pg
    )

    # Perform the modular operation of the 1D tensor to be sent to each rank.
    input_combined = torch.remainder(input_combined, sharded_dim_size_max)
    input_combined_list = list(torch.split(input_combined, input_combined_split_sizes))

    # Input sent from each rank to the current rank may have different sizes.
    gathered_input = _communicate_list_to_each_rank_common(
        input_combined_list,
        output_split_sizes,
        world_size,
        input,
        pg,
    )

    # If user specify per_sample_weights we need to first rearrange them
    # to be sync with IDs and then distribute them to each rank
    gathered_per_sample_weights = None
    if per_sample_weights is not None:
        # Rearrange per interval.
        weight_type = per_sample_weights.dtype
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
        # Perform the modular operation of the 1D tensor to be sent to each rank.
        per_sample_weights_list = list(
            torch.split(per_sample_weights, input_combined_split_sizes)
        )
        gathered_per_sample_weights = _communicate_list_to_each_rank_common(
            per_sample_weights_list,
            output_split_sizes,
            world_size,
            input,
            pg,
            tensor_type=weight_type,
        )

    # Perform the embedding bag look-up and aggregation
    results = []
    for i, inp in enumerate(gathered_input):
        per_sample_weights = (
            gathered_per_sample_weights[i]
            if gathered_per_sample_weights is not None
            else None
        )
        result = torch.nn.functional.embedding_bag(
            inp,
            local_shard,
            offsets=output_offsets_tensor_list[i],
            mode=mode if mode != "mean" else "sum",
            per_sample_weights=per_sample_weights,
        )
        if mode != "max":
            results.append(result)
        # For max case, it there is no look-up from some ranks
        # it will return all zero for that. For that case, we need
        # to set the row to neg inf, otherwise, in the final
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
                if current_offset == next_offset:
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
        # For a 2D tensor, we just average by col number
        if len(input_size) > 1:
            return torch.div(gathered_output, float(input.size(1)))
        # For a 1D tensor, we need to average by each offset
        else:
            split_sizes_1d_tensor = torch.tensor(
                split_sizes_1d, dtype=torch.float, device=input.device
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
