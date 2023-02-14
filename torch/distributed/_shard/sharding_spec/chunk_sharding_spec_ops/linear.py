from typing import List

import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.distributed.nn.functional import (
    _all_gather_base,
    all_to_all_single,
)
from torch.distributed._shard.partial_tensor import _PartialTensor
from torch.distributed._shard.sharded_tensor import (
    ShardedTensor,
)
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.sharding_spec.api import custom_sharding_spec_op
from torch.distributed._shard.sharding_spec._internals import (
    get_split_size,
    get_chunked_dim_size,
    get_chunk_sharding_params,
)

from ._common import (
    _result_distribute_with_col_rearrange,
)


@custom_sharding_spec_op(ChunkShardingSpec, torch.nn.functional.linear)
def sharded_linear(types, args, kwargs, pg):
    """
    Handles ``__torch_function__`` dispatch for ``torch.nn.functional.linear``.
    This method computes a sharded linear and has the following limitations:

    1. Supports only sharding of ``weight``.
    2. Supports only ``ChunkShardingSpec``.
    3. Supports only a single local shard per rank.
    4. Tailored for Megatron-LM style model(tensor) parallelism. Further API
       calls are needed if a fully synced local tensor is needed.
       Megatron-LM paper link: https://arxiv.org/abs/1909.08053

    Based on the dimension that the weight is sharded on, there are two
    algorithms:

    ROWWISE SHARDING
    ================
    For row-wise sharding the weight is sharded on dimension 1, but this is
    row-wise since the actual computation for the linear layer involves
    transposing the weight: :math:`y = xA^T + b`

    The overall algorithm can be best explained with an example. Let's assume
    the dims for x are (13 x 16) and A are (17 x 16) and A is sharded across
    4 GPUs creating shards of (17 x 4). The algorithm is as follows:

    1. First the input is split on the column dimension to create shards of
       (13 x 4) and communicated to all other ranks. Since we are running in
       an SPMD mode with each rank having distinct input, this is done via
       an all2all run on all ranks.
    2. Now each (13 x 4) shard on each GPU is multiplied with the local shard
       (4 x 17) (transposed) resulting in a (13 x 17) matrix which is the same
       size that we need for the global result which would be (13 x 16)
       multiplied by (16 x 17). But the final result needs to be aggregated
       across the rest of the ranks.
    3. Here we just return the partial result here. One can call API
       aggregate_partial_tensor_list to get the aggregated final result.
       The API uses a reduce_scatter operation ensuring each rank
       aggregates its own result. This is essentially a sum operation across
       all the (13 x 17) local computations we did for each rank.
    4. For partial result, we only add 1 / n of the bias term to the partial
       result. n is # of all GPUs.

    COLWISE SHARDING
    ================
    For col-wise sharding the weight is sharded on dimension 0, but this is
    col-wise since the actual computation for the linear layer involves
    transposing the weight: :math:`y = xA^T + b`

    The overall algorithm can be best explained with an example. Let's assume
    the dims for x are (13 x 17) and A are (16 x 17) and A is sharded across
    4 GPUs creating shards of (4 x 17). The algorithm is as follows:

    1. First the input is broadcasted to all ranks, since this is SPMD we
       actually do an all_gather for all the inputs resulting in 4 (13 x 17)
       inputs on each rank.
    2. Next we perform local matmuls by multiplying each input (13 x 17)
       with the local shard (17 x 4) (transposed). This results in 4 (13 x 4)
       matrices on each rank.
    3. Next, we stack them into a (4 x 13 x 4) tensor and build a sharded
       tensor across 4 ranks.
    4. To merge them into a fully-sync local tensor, one can call API
       merge_sharded_local_results.
       This API concat these 4 matrices and perform an all2all to share the
       appropriate (13 x 4) matrices to each rank. Specifically, each rank
       receives a (13 x 16) matrix which is basically the size of the result.
    5. If placements are not in order any appropriate rearrangement of rows
       are done for the (13 x 16) matrix and finally the bias term is added.
    """
    # Validate input params
    _validate_linear_op_param(args, kwargs)
    input = args[0]
    weight = args[1]
    bias = args[2]

    local_shard = weight.local_tensor()
    local_shard_t = local_shard.t()
    sharding_dim = weight._sharding_spec.dim
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)

    if sharding_dim == 1 and isinstance(input, ShardedTensor):
        return _handle_row_wise_sharding_sharded_tensor(
            input, world_size, weight, local_shard_t, bias, pg
        )
    elif sharding_dim == 1 and isinstance(input, torch.Tensor):
        return _handle_row_wise_sharding_tensor(
            input, world_size, weight, rank, local_shard_t, bias, pg
        )
    elif sharding_dim == 0:
        return _handle_col_wise_sharding(
            input, world_size, weight, rank, local_shard_t, bias, pg
        )
    else:
        raise RuntimeError(
            f"nn.Linear weight sharded on dim {sharding_dim} not supported!"
        )


def _validate_linear_op_param(args, kwargs):
    """
    Validate input params of sharded linear op.

    Args:
        input: input of the linear layer.
        weight: sharded weight tensor.
        kwargs: same as normal Linear.

    Return: None.
    """
    input = args[0]
    weight = args[1]
    bias = args[2]

    # Validate types
    if not isinstance(input, torch.Tensor) and not isinstance(input, ShardedTensor):
        raise TypeError("input needs to be either torch.Tensor or ShardedTensor")
    if type(bias) != torch.Tensor and type(bias) != torch.nn.Parameter:
        raise TypeError("bias needs to be torch.Tensor")
    if not isinstance(weight, ShardedTensor):
        raise TypeError("weight needs to be ShardedTensor")
    if len(input.size()) < 1:  # type: ignore[arg-type]
        raise ValueError("Input needs to have at least 1 dim")
    weight_size = weight.size()
    if len(weight_size) != 2:
        raise ValueError("Weight needs to have exactly 2 dims")
    if len(bias.size()) != 1:
        raise ValueError("Bias needs to have exactly 1 dim")
    if input.size()[-1] != weight_size[1]:  # type: ignore[index]
        raise ValueError(
            f"Input dim: {input.size()[-1]} does not match "  # type: ignore[index]
            f"appropriate weight dim: {weight_size[1]}"
        )
    if not isinstance(weight._sharding_spec, ChunkShardingSpec):
        raise ValueError("Only ChunkShardingSpec supported for ShardedTensor ops!")
    if len(weight.local_shards()) != 1:
        raise ValueError("Only one local shard supported!")


def _handle_col_wise_sharding(input, world_size, weight, rank, local_shard_t, bias, pg):
    """
    Entry-point function to handle the logic of col-wise sharding of weight
    for Linear. (Detailed explanations of the logic can be found in the
    comment for sharded_linear.)

    When the local tensor only has one dimension, we increase one more dimension
    for reshard. We need to do squeeze manually to reduce the dimension later-on.

    For example, if we have:
    input: size[15]
    weight: size[15, 16]
    world_size: 4

    In each rank, we will have 4 * [4] tensors. We then stack them into a [4, 4]
    tensor and generate a sharded tenor sharded by dim 1.

    For the rest situations, we just simply concatenate local tensors. No more actions
    are needed afterward.

    Args:
        input: matrix to be multiplied with the sharded weight.
        world_size: number of ranks.
        weight: sharded weight tensor.
        rank: # of cuda process.
        local_shard_t: row-wise shared local weight used for lookup.
        bias: bias term of linear op.
        pg: process group.

    Returns:
        A :class:`ShardedTensor` object which filled with local intermediate results.
    """
    # allgather the inputs first.
    out_size = list(input.size())
    out_size[0] = input.size(0) * dist.get_world_size(pg)
    output = torch.empty(out_size, device=input.device, dtype=input.dtype)
    output = _all_gather_base(output, input, group=pg)

    # Adjust bias and perform local matmul.
    (start_pos, chunk_size) = get_chunk_sharding_params(
        bias.size(0), world_size, weight._sharding_spec, rank
    )
    local_bias = _BiasTensorNarrow.apply(
        world_size, start_pos, chunk_size, weight, pg, bias
    )

    if output.dim() == 1:
        output = output.view(dist.get_world_size(pg), -1)

    if output.dim() <= 2:
        # Use fused version if possible.
        result = torch.addmm(local_bias, output, local_shard_t)
    else:
        result = output.matmul(local_shard_t) + local_bias

    # Build ShardedTensor as result.
    st_size = list(result.size())
    st_size[-1] = weight.size(0)
    new_sharding_spec = ChunkShardingSpec(
        dim=-1,
        placements=weight.sharding_spec().placements
    )
    return ShardedTensor._init_from_local_tensor(
        result,
        new_sharding_spec,
        *st_size,  # type: ignore[arg-type]
        process_group=pg,
    )


def _handle_row_wise_sharding_tensor(
    input, world_size, weight, rank, local_shard_t, bias, pg
):
    """
    Entry-point function to handle the logic of row-wise sharding of weight
    for Linear. (Detailed explanations of the logic can be found in the
    comment for sharded_linear.)

    Args:
        input: matrix to be multiplied with the sharded weight.
        world_size: number of ranks.
        weight: sharded weight tensor.
        rank: # of cuda process.
        local_shard_t: row-wise shared local weight used for lookup.
        bias: bias term of linear op.
        pg: process group.

    Returns:
        A :class:`_PartialTensor` object which stores the partial local result.
    """
    # alltoall to gather all the appropriate inputs.
    input_t = input.transpose(0, -1).contiguous()
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
        indices: List[List[int]] = [[0]] * world_size
        # When we do the chunk split, we always ensure the first N - 1 chunks get max out
        # and then the Nth chunk gets the rest. So input_split_sizes like [3, 3, 3, 4]
        # are not possible. The expected split size will be [4, 4, 4, 1].
        sharded_dim_size_max = max(input_split_sizes)
        for idx, placement in enumerate(weight._sharding_spec.placements):
            split_size = input_split_sizes[placement.rank()]
            offset_start_idx = idx * sharded_dim_size_max
            indices[placement.rank()] = list(
                range(offset_start_idx, offset_start_idx + split_size)
            )
        indices_flatten = [idx for indice in indices for idx in indice]

        input_t = input_t.index_select(
            0, torch.tensor(indices_flatten, device=input_t.device)
        )

    gathered_input_size = [input_split_sizes[rank] * world_size] + list(
        input_t_size[1:]
    )
    gathered_input = torch.empty(gathered_input_size, device=input_t.device, dtype=input_t.dtype)

    # Perform autograd enabled alltoall
    all_to_all_single(
        gathered_input, input_t, input_split_sizes=input_split_sizes, group=pg
    )

    # Reshape gathered_input appropriately for matmul
    shard_size = local_shard_t.size()[0]
    reshaped_inputs = [
        torch.narrow(gathered_input, 0, r * shard_size, shard_size).transpose(0, -1)
        for r in range(world_size)
    ]
    reshaped_input = torch.cat(reshaped_inputs)
    if reshaped_input.dim() == 1:
        reshaped_input = reshaped_input.view(-1, local_shard_t.size(0))

    # Perform appropriate local matmul
    if reshaped_input.dim() <= 2:
        result = torch.addmm(_BiasTensorPartial.apply(world_size, bias), reshaped_input, local_shard_t)
    else:
        result = reshaped_input.matmul(local_shard_t) + _BiasTensorPartial.apply(world_size, bias)

    # Return the partial local result.
    return _PartialTensor(result, pg)


def _handle_row_wise_sharding_sharded_tensor(
    input, world_size, weight, local_shard_t, bias, pg
):
    """
    Entry-point function to handle the logic of row-wise sharding of weight
    for Linear when the input is a sharded tensor. (Detailed explanations
    of the logic can be found in the comment for sharded_linear.)

    Args:
        input: matrix to be multiplied with the sharded weight.
        world_size: number of ranks.
        weight: sharded weight tensor.
        local_shard_t: row-wise shared local weight used for lookup.
        bias: bias term of linear op.
        pg: process group.

    Returns:
        A :class:`_PartialTensor` object which stores the partial local result.
    """
    local_input = input.local_tensor()
    if input.sharding_spec().dim not in (-1, len(input.size()) - 1):
        raise NotImplementedError(
            "The case when the input does not come from col-wise sharded "
            "linear is not supported for row-wise sharded linear."
        )

    # Use fused version if possible.
    if local_input.dim() <= 2:
        result = torch.addmm(_BiasTensorPartial.apply(world_size, bias), local_input, local_shard_t)
    else:
        result = local_input.matmul(local_shard_t) + _BiasTensorPartial.apply(world_size, bias)

    # Return the partial local result.
    return _PartialTensor(result, pg)


class _BiasTensorNarrow(Function):
    """
    Since we now return the intermediate results in a col-wise sharding. We
    need to narrow the bias term in the forward while doing backward, we need
    to gather all gradients of narrowed bias across all ranks.
    """

    @staticmethod
    def forward(ctx, world_size, start_pos, chunk_size, weight, pg, bias):
        ctx.weight = weight
        ctx.pg = pg
        ctx.world_size = world_size
        return torch.narrow(bias, 0, start_pos, chunk_size)

    @staticmethod
    def backward(ctx, grad_output):
        results = [grad_output.clone()] * ctx.world_size
        return (None, None, None, None, None) + (
            _result_distribute_with_col_rearrange(
                results, grad_output, ctx.world_size, ctx.weight, ctx.pg
            ),
        )


class _BiasTensorPartial(Function):
    """
    Since we now only return partial results in a row-wise sharding. We need to
    divide the bias term by the world size in the forward while doing backward,
    we need to skip this division op.
    """

    @staticmethod
    def forward(ctx, world_size, bias):
        ctx.world_size = world_size
        return torch.div(bias, world_size)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, grad_output)
