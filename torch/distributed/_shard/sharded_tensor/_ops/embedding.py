# coding=utf-8

from typing import cast

import torch
import torch.distributed as dist
from ._common import (
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

@sharded_op_impl(torch.nn.functional.embedding)
def sharded_embedding(types, args, kwargs, pg):
    """
    Handles ``__torch_function__`` dispatch for ``torch.nn.functional.embedding``.
    This method computes a sharded embedding lookup and has the following limitations:

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
       Rearrangement may be needed if the rank order is different from
       its index in the placement.
    3. Next, we communicate the length of each part to each rank via all2all
       so that each rank now knows what input it will get from all other ranks.
    4. Before we send out the array to other ranks, we need to do the modular operation
       so that each rank do use that for embedding lookup.
       The above tensor will look like the below after performing the moduler of 3:
       tensor([[0, 1, 1, 2, 2], [0, 0, 1, 1, 1, 1, 2],
              [0, 0, 0, 0, 0, 0, 1, 2, 2], [0, 0, 0])
    5. Now, each rank receives a matrix (size may vary) and do the lookup. We then use
       all2all to send the result back to each rank.
    6. We use the recorded indices to recover the sorted positions and reshape the
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
    2. Next we perform local embedding lookup operation by apply each
       input (4 x 6) with the local shard (16 x 5) ((16 x 2) for the last).
       This results in 4 (5 x 6 x 4) ((2 x 6 x 4) for the last) matrices
       on each rank. We transpose dim 0 and dim 2.
    3. Next, we concat these 4 matrices and perform an all2all to share the
       appropriate (5 x 6 x 4) or (2 x 6 x 4) matrices to each rank.
    4. Now, each rank receives a (17 x 6 x 4) matrix which is basically the
       size of the result we need.
    5. If placements are not in order any appropriate rearrangement of columns
       are done for the (17 x 6 x 4) matrix and finally we transpose the
       dim 0 and dim 2 again.
    6. If max_norm is specified, we manually sum up the norm and renorm. Because
       the renorm must be in place, we need to override the local_shard to mimic
       this behavior.
    """
    # Validate input params
    _validate_embedding_param(args, kwargs)

    input = args[0]
    weight = args[1]
    max_norm = kwargs.get("max_norm")
    norm_type = kwargs.get("norm_type")
    padding_idx = kwargs.get("padding_idx")

    local_shard = weight.local_tensor().contiguous()
    sharding_dim = weight._sharding_spec.dim
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)

    if sharding_dim == 1:
        output, local_shard = _handle_col_wise_sharding(
            input, world_size, weight, local_shard, max_norm, norm_type, padding_idx, pg
        )
        weight.local_shards()[0].tensor = local_shard
        return output
    elif sharding_dim == 0:
        return _handle_row_wise_sharding(
            input,
            world_size,
            weight,
            local_shard,
            max_norm,
            norm_type,
            padding_idx,
            rank,
            pg,
        )
    else:
        raise RuntimeError(
            f"nn.Embedding weight sharded on dim {sharding_dim} not supported!"
        )


def _validate_embedding_param(args, kwargs):
    """
    Validate input params of sharded embedding op.

    Args:
        input: list of ID used for lookup.
        weight: shareded weight tensor.
        kwargs: same as normal Embedding.

    Return: None.
    """

    input = args[0]
    weight = args[1]
    max_norm = kwargs.get("max_norm")
    norm_type = kwargs.get("norm_type")
    scale_grad_by_freq = kwargs.get("scale_grad_by_freq")
    sparse = kwargs.get("sparse")
    padding_idx = kwargs.get("padding_idx")

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
    if scale_grad_by_freq:
        raise RuntimeError(
            'nn.Embedding weight sharded with flag on "scale_grad_by_freq" not supported!'
        )
    if sparse:
        raise RuntimeError(
            'nn.Embedding weight sharded with flag on "sparse" not supported!'
        )
    if max_norm and max_norm <= 0.0:
        raise ValueError('"max_norm" must be larger than zero!')

    if not isinstance(weight._sharding_spec, ChunkShardingSpec):
        raise ValueError("Only ChunkShardingSpec supported for ShardedTensor ops!")
    if len(weight.local_shards()) != 1:
        raise ValueError("Only one local shard supported!")


def _handle_col_wise_sharding(
    input, world_size, weight, local_shard, max_norm, norm_type, padding_idx, pg
):
    """
    Entry-point function to handle the logic of col-wise sharding of weight
    for embedding. (Detailed explanations of the logic can be found in
    the comment for sharded_embedding.)

    Args:
        input: list of ID used for lookup and aggregation.
        world_size: number of ranks.
        weight: shareded weight tensor.
        local_shard: col-wise shared local weight used for lookup.
        max_norm: If given, each embedding vector with norm larger
            than max_norm is renormalized to have norm max_norm.
            Note: this will modify weight in-place.
        norm_type: The p in the p-norm to compute for the max_norm option.
        padding_idx: If specified, the entries at padding_idx do
            not contribute to the gradient; therefore, the embedding
            vector at padding_idx is not updated during training,
            i.e. it remains as a fixed “pad”.
        pg: process group.

    Returns: final result of lookup.
    """
    gathered_inputs = None
    if max_norm is not None:
        # max_norm changes the weight in-place
        local_shard, gathered_inputs = _handle_max_norm_col_wise(
            max_norm, norm_type, local_shard, input, world_size, pg
        )

    output = _handle_col_wise_sharding_base(
        torch.nn.functional.embedding,
        len(input.size()),
        input,
        world_size,
        weight,
        local_shard,
        pg,
        padding_idx=padding_idx,
        gathered_inputs=gathered_inputs,
    )
    return (output, local_shard)


def _handle_row_wise_sharding(
    input, world_size, weight, local_shard, max_norm, norm_type, padding_idx, rank, pg
):
    """
    Entry-point function to handle the logic of row-wise sharding of weight
    for embedding. (Detailed explanations of the logic can be found in
    the comment for sharded_embedding.)

    Args:
        input: list of ID used for lookup and aggregation.
        world_size: number of ranks.
        weight: shareded weight tensor.
        local_shard: row-wise shared local weight used for lookup.
        max_norm: If given, each embedding vector with norm larger
            than max_norm is renormalized to have norm max_norm.
            Note: this will modify weight in-place.
        norm_type: The p in the p-norm to compute for the max_norm option.
        padding_idx: If specified, the entries at padding_idx do
            not contribute to the gradient; therefore, the embedding
            vector at padding_idx is not updated during training,
            i.e. it remains as a fixed “pad”.
        rank: # of cuda process.
        pg: process group.

    Returns: final result of lookup.
    """
    # flatten the ids across all input and sort
    input_size = input.size()
    input_1d = torch.reshape(input, (-1,)).contiguous()
    input_sorted, indices_1d = torch.sort(input_1d)
    rearrange_indices_1d = torch.argsort(indices_1d)
    input_sorted.contiguous()

    (
        input_sorted,
        input_split_sizes,
        sharded_dim_size_max,
        _,
        rearrange_indices_1d_second_order,
        padding_idx,
    ) = _handle_row_wise_lookup_distribute(
        input_sorted, input, world_size, weight, rank, padding_idx
    )

    # Get the input split size to be sent from each rank to the current rank.
    # We can then infer the output split size.
    output_split_sizes = _communicate_size_to_each_rank(
        input_split_sizes, world_size, input, pg
    )

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

    # If input is None, passing in max_norm causes
    # errors in CUDA.
    if max_norm is not None and gathered_input.size(0) == 0:
        max_norm = None

    # Perform local embedding look up.
    gathered_input_embeddings = torch.nn.functional.embedding(
        gathered_input,
        local_shard,
        padding_idx=padding_idx,
        max_norm=max_norm,
        norm_type=norm_type,
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
