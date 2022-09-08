# coding=utf-8

import torch
import torch.distributed as dist
from torch.distributed._shard.replicated_tensor import ReplicatedTensor
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.sharding_spec.api import custom_sharding_spec_op
from torch.distributed.nn.functional import all_gather, all_reduce, reduce_scatter

from ._common import (
    _all_gather_base_input,
    _handle_col_wise_sharding_base,
    _handle_max_norm_col_wise,
    _handle_row_wise_mask,
)


@custom_sharding_spec_op(ChunkShardingSpec, torch.nn.functional.embedding)
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

    1. First the input is all gathered to all ranks, since this is SPMD and
       input is actually sharded across all ranks. The inputs then become a
       4 (4 x 6) tensor on each rank. For example if the given input is
       tensor([[6, 5, 2, 9, 6, 3],
               [3, 1, 2, 4, 7, 6],
               [4, 0, 4, 9, 8, 9],
               [8, 6, 6, 4, 6, 1]])
       on rank 0.
       Then on every rank, we will have this tensor.
       If input itself is already replicated, no all-gather will be done.
    2. Next, we mask the ID which are not stored on that rank.
       For example on rank 0, we store ID [0, 1, 2]. We only keep the ID
       inside the set of numbers. The rest of them will be masked to an extra row.
       The masked matrix will be used for embedding look up and is like:
       tensor([[4, 4, 2, 4, 4, 4],
               [4, 1, 2, 4, 4, 4],
               [4, 0, 4, 4, 4, 4],
               [4, 4, 4, 4, 4, 1]])
       The reason of having an extra row (aka, number 4 in the example) is
       because when max_norm is specified only weight which has looked will
       be re-normed so mask IDs whose embeddings are not stored in current
       rank will to an extra row will ensure max_norm still works as expected.
    3. If max_norm is specified, the extra row gurantee that the mask ID will
       not affect the behavior of weigh re-norm.

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
    scale_grad_by_freq = kwargs.get("scale_grad_by_freq")
    sparse = kwargs.get("sparse")

    # Validate types
    if not isinstance(input, torch.Tensor):
        raise TypeError("input need to be torch.Tensor")
    if not isinstance(weight, ShardedTensor):
        raise TypeError("weight needs to be ShardedTensor")
    weight_size = weight.size()
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
    if not isinstance(input, ReplicatedTensor):
        # allgather the inputs first for non Replicated Tensor.
        gathered_inputs = all_gather(input, group=pg)
    else:
        gathered_inputs = input

    if max_norm is not None:
        # max_norm changes the weight in-place
        local_shard = _handle_max_norm_col_wise(
            max_norm, norm_type, local_shard, input, world_size, gathered_inputs, pg
        )

    output = _handle_col_wise_sharding_base(
        torch.nn.functional.embedding,
        len(input.size()),
        input,
        world_size,
        weight,
        local_shard,
        pg,
        gathered_inputs,
        padding_idx=padding_idx,
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
    if not isinstance(input, ReplicatedTensor):
        # allgather the inputs first for non Replicated Tensor.
        gather_inp = _all_gather_base_input(input, pg)
    else:
        gather_inp = input

    # Mask the input according to sharding spec.
    lookup_input, padding_idx, padding_row = _handle_row_wise_mask(
        gather_inp, padding_idx, weight, world_size, rank
    )

    # When input is a large tensor, the value of weight is changed.
    # This is a walk-around for now. GH issue: #81717
    if max_norm is not None:
        torch.nn.functional.embedding(
            torch.unique(lookup_input)[:-1],
            local_shard,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
        )
        max_norm = None

    local_input_embeddings = torch.nn.functional.embedding(
        lookup_input,
        torch.cat([local_shard, padding_row]),
        padding_idx=padding_idx,
        max_norm=max_norm,
        norm_type=norm_type,
    )

    # TODO: Make the result a PartialTensor.
    if isinstance(input, ReplicatedTensor):
        return all_reduce(local_input_embeddings, group=pg)
    else:
        local_shards = local_input_embeddings.chunk(pg.size())
        return reduce_scatter(
            torch.empty_like(local_shards[0]),
            list(local_shards),
            group=pg,
        )
