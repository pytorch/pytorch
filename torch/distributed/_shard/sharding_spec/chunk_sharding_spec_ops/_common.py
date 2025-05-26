# mypy: allow-untyped-defs

import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharded_tensor._ops._common import _sharded_op_common
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.sharding_spec._internals import (
    get_chunk_sharding_params,
    get_chunked_dim_size,
    get_split_size,
)
from torch.distributed._shard.sharding_spec.api import custom_sharding_spec_op
from torch.distributed.nn.functional import (
    _all_gather_base,
    all_reduce,
    all_to_all_single,
)


def _chunk_sharding_spec_check(spec, op):
    """
    For the given op implementation check if the sharding spec is ChunkShardingSpec.
    """
    if not isinstance(spec, ChunkShardingSpec):
        raise NotImplementedError(
            f"Only ChunkShardingSpec supported for '{op.__name__}'."
        )


def _register_sharded_op_on_local_tensor(
    op, early_stop_func=None, extra_check=None, customized_func=None
):
    """
    Handles ``__torch_function__`` dispatch for ops which are performed on
    the single local tensor of the sharded tensor such as op like
    ``torch.nn.functional.softmax`` or ``torch.Tensor.view``.

    For more complicated ops, a customized func can be used to generate
    the new local tensor, sharding spec and sharded tensor size.

    Args:
        op: The op to be registered and applied to all shards of the st.
        early_stop_func (Callable, optional): the func for early stop.
            Default: if ``None``, no early stop.
        extra_check (Callable, optional): the func for extra condition check.
            Default: if ``None``, no extra check.
        customized_func (Callable, optional): the func for customized logic
            to generate the new local tensor, sharding spec and sharded tensor size.
            Default: if ``None``, we simply lower to the real op call with
                the single local tensor of the st.

    Return:
        func (Callable): registered implementation for sharded op for
        ``__torch_function__`` dispatch.
    """

    @custom_sharding_spec_op(ChunkShardingSpec, op)
    @_sharded_op_common(op, early_stop_func, extra_check)
    def sharded_tensor_op_on_local_tensor(types, args=(), kwargs=None, pg=None):
        st = args[0]
        sharding_spec = st.sharding_spec()
        if len(st.local_shards()) != 1:
            raise TypeError(
                f"torch function '{op.__name__}', with args: {args} and "
                f"kwargs: {kwargs} only supported for single local tensor!"
            )
        st_size = st.size()
        if customized_func:
            local_tensor, sharding_spec, st_size = customized_func(args, kwargs, pg)
        else:
            args = (st.local_tensor(), *args[1:])
            local_tensor = op(*args, **kwargs)
        return ShardedTensor._init_from_local_tensor(
            local_tensor.contiguous(),
            sharding_spec,
            st_size,  # type: ignore[arg-type]
            process_group=pg,
            init_rrefs=st._init_rrefs,
        )


def _handle_col_wise_sharding_base(
    op_func,
    col_dim,
    input,
    world_size,
    weight,
    local_shard,
    pg,
    gathered_inputs,
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
        weight: sharded weight tensor.
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
            i.e. it remains as a fixed "pad".
            Note that the embedding vector at padding_idx is
            excluded from the reduction.

    Return: final result of input being applied with the op.
    """
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


def _result_distribute_with_col_rearrange(results, input, world_size, weight, pg):
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
        weight: sharded weight tensor.
        pg: process group.

    Return: column rearranged result.
    """
    # Process results and outputs for all2all.
    sharding_dim = weight._sharding_spec.dim
    sharding_dim_size = weight.size(sharding_dim)
    dims = list(results[0].size())
    dims[0] = sharding_dim_size
    combined_results = torch.cat(results)
    output = torch.empty(
        *dims, device=combined_results.device, dtype=combined_results.dtype
    )

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
        idx != placement.rank()
        for idx, placement in enumerate(weight._sharding_spec.placements)
    )
    if not rearrange_columns:
        return output

    indices = []
    for placement in weight._sharding_spec.placements:
        dim_size = output_split_sizes[placement.rank()]
        start = sum(
            split_size if i < placement.rank() else 0
            for i, split_size in enumerate(output_split_sizes)
        )
        indices += list(range(start, start + dim_size))

    return output.index_select(0, torch.tensor(indices, device=output.device))


def _handle_max_norm_col_wise(
    max_norm,
    norm_type,
    local_shard,
    input,
    world_size,
    gathered_inputs,
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
        gathered_inputs: list of inputs from all ranks.
        pg: process group.

    Return:
        local_shard_norm_renormed: local_shard re-normed to max_norm if the norm is larger
            than it.

    """
    norm_type = norm_type if norm_type is not None else 2.0
    unique_inp = torch.unique(torch.cat(gathered_inputs))
    local_shard_sum = torch.sum(
        torch.pow(torch.abs(local_shard), norm_type), dim=1, dtype=local_shard.dtype
    )
    # For col-wise sharding, we need to first aggregate the powered sum
    # from each rank first and then calculate the norm.
    local_shard_sum = all_reduce(local_shard_sum, group=pg)
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
    return local_shard_norm_renormed


def _all_gather_base_input(input, pg):
    """
    Use _all_gather_base to get a concatenated input from each rank.

    Args:
        input: tensor to be applied op on.
        pg: process group.

    Returns:
        gathered_inputs: input gathered from each rank and concat by dim 0.
    """
    # allgather the inputs first.
    gather_inp_size = list(input.size())
    gather_inp_size[0] = input.size(0) * dist.get_world_size(pg)
    gather_inp = torch.empty(gather_inp_size, device=input.device, dtype=input.dtype)
    return _all_gather_base(gather_inp, input, group=pg)


def _handle_row_wise_mask(gather_inp, padding_idx, weight, world_size, rank):
    """
    Mask the input for embedding look-up for IDs which are not stored
    on the current rank. This function also adjust the ``padding_idx``
    so that it is only used on the rank where the corresponding row is
    stored.

    Note that, with ``max_norm`` flag on, only weights of rows being
    looked up will be re-normed. So we need an extra row for masked ID
    so that it does not affect the final result and ``max_norm``.

    Args:
        gather_inp: tensor to be applied op on gathered from all ranks.
        padding_idx: If specified, the entries at padding_idx do
            not contribute to the gradient; therefore, the embedding
            vector at padding_idx is not updated during training,
            i.e. it remains as a fixed "pad".
            Note that the embedding vector at padding_idx is
            excluded from the reduction.
        weight: weight tensor of Embedding look-up table.
        world_size: number of ranks.
        rank: # of cuda process.

    Returns:
        lookup_input: Tensor of masked input.
        padding_idx: adjusted padding_idx.
        padding_row: The extra row we used during lookup so that
            looking up does not affect ``max_norm``.
    """
    (start_pos, chunk_size) = get_chunk_sharding_params(
        weight.size(0), world_size, weight._sharding_spec, rank
    )
    mask = (gather_inp < start_pos) | (gather_inp >= start_pos + chunk_size)
    lookup_input = gather_inp.clone() - start_pos
    lookup_input[mask] = chunk_size
    if (
        padding_idx is not None
        and padding_idx >= start_pos
        and padding_idx < (start_pos + chunk_size)
    ):
        padding_idx = padding_idx - start_pos
    else:
        padding_idx = None

    # When max_norm is set, it will only re-norm the row being looked up.
    padding_row = torch.zeros(
        1, weight.size(1), device=gather_inp.device, dtype=weight.dtype
    )
    return lookup_input, padding_idx, padding_row
