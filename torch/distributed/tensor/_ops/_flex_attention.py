# Copyright (c) Meta Platforms, Inc. and affiliates
from collections.abc import Callable
from typing import Any

import torch
from torch._guards import detect_fake_mode
from torch._higher_order_ops.flex_attention import (
    _permute_strides,
    flex_attention as flex_attention_hop,
    flex_attention_backward as flex_attention_backward_hop,
)
from torch._subclasses import FakeTensorMode
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import (
    OpInfo,
    OpSchema,
    OpStrategy,
    OutputSharding,
    OutputSpecType,
    PlacementList,
    RuntimeSchemaInfo,
)
from torch.distributed.tensor._ops.registration import register_op_strategy
from torch.distributed.tensor._ops.utils import expand_to_full_mesh_op_strategy
from torch.distributed.tensor.placement_types import _Partial, Replicate, Shard
from torch.fx.graph_module import GraphModule
from torch.utils import _cxx_pytree as pytree


@register_op_strategy(flex_attention_hop, schema_info=RuntimeSchemaInfo())  # type: ignore[arg-type]
def flex_attention_strategy(op_schema: OpSchema) -> OpStrategy:
    mesh = op_schema.get_mesh_from_args(validate=True)

    # Replicate everything.
    replicate_sharding: PlacementList = [
        Replicate(),  # output,
        Replicate(),  # logsumexp,
        Replicate(),  # max_logits
        Replicate(),  # q,
        Replicate(),  # k,
        Replicate(),  # v,
        Replicate(),  # kv_num_blocks,
        Replicate(),  # kv_indices,
    ]

    # When sharding on the batch dimension but mask's batch size is larger than 1,
    # the mask has to be sharded along the batch dimension as well.
    batch_dim_sharding: PlacementList = [
        Shard(0),  # output,
        Shard(0),  # logsumexp,
        Shard(0),  # max_logits
        Shard(0),  # q,
        Shard(0),  # k,
        Shard(0),  # v,
        Shard(0),  # kv_num_blocks,
        Shard(0),  # kv_indices,
    ]

    # When sharding on the batch dimension but mask's batch size is 1, the mask
    # is orthogonal to the batch dimension and can be replicated.
    batch_dim_sharding_mask_replicate: PlacementList = [
        Shard(0),  # output,
        Shard(0),  # logsumexp,
        Shard(0),  # max_logits
        Shard(0),  # q,
        Shard(0),  # k,
        Shard(0),  # v,
        Replicate(),  # kv_num_blocks,
        Replicate(),  # kv_indices,
    ]

    # When sharding on the num_heads dimension, the mask has to be sharded along
    # the num_heads dimension as well if masks' num_heads size is larger than 1.
    # TODO: is this legal? Can the mask shard on the head dimension?
    num_heads_dim_sharding: PlacementList = [
        Shard(1),  # output,
        Shard(1),  # logsumexp,
        Shard(1),  # max_logits
        Shard(1),  # q,
        Shard(1),  # k,
        Shard(1),  # v,
        Shard(1),  # kv_num_blocks,
        Shard(1),  # kv_indices,
    ]

    # When sharding on the num_heads dimension but mask's num_heads size is 1,
    # the mask is orthogonal to the num_heads dimension and can be replicated.
    num_heads_dim_sharding_mask_replicate: PlacementList = [
        Shard(1),  # output,
        Shard(1),  # logsumexp,
        Shard(1),  # max_logits
        Shard(1),  # q,
        Shard(1),  # k,
        Shard(1),  # v,
        Replicate(),  # kv_num_blocks,
        Replicate(),  # kv_indices,
    ]

    # Context parallel sharding, where Q is sharded but K and V are replicated.
    # An all-gather is needed to combine the Qs before the attention.
    seq_dim_sharding: PlacementList = [
        Shard(2),  # output,
        Shard(2),  # logsumexp,
        Shard(2),  # max_logits
        Shard(2),  # q,
        Replicate(),  # k,
        Replicate(),  # v,
        Shard(2),  # kv_num_blocks,
        Shard(2),  # kv_indices,
    ]

    single_mesh_dim_strategies = [
        replicate_sharding,
        batch_dim_sharding,
        batch_dim_sharding_mask_replicate,
        num_heads_dim_sharding,
        num_heads_dim_sharding_mask_replicate,
        seq_dim_sharding,
    ]

    op_strategy = expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=3
    )
    # We don't allow mask redistribution.
    # So if selecting this strategy results in mask redistribution,
    # we set the cost to be inf.
    for op_spec in op_strategy.strategies:
        costs = op_spec.redistribute_cost
        assert costs is not None
        # costs is a list of lists, one per input tensor
        # indices 3+ are the mask tensors (kv_num_blocks, kv_indices, etc.)
        for mask_idx in range(3, len(costs)):
            mask_costs = costs[mask_idx]
            # If any cost for this mask is > 0, mark all costs for this input as inf
            if any(cost > 0 for cost in mask_costs):
                costs[mask_idx] = [float("inf")] * len(mask_costs)

    return op_strategy


@register_op_strategy(flex_attention_backward_hop, schema_info=RuntimeSchemaInfo())  # type: ignore[arg-type]
def flex_attention_backward_strategy(op_schema: OpSchema) -> OpStrategy:
    mesh = op_schema.get_mesh_from_args(validate=False)

    # Replicate everything. The backward outputs follow the same sharding as the
    # forward inputs (query, key, value).
    replicate_sharding: PlacementList = [
        Replicate(),  # grad_query,
        Replicate(),  # grad_key,
        Replicate(),  # grad_value,
        Replicate(),  # query,
        Replicate(),  # key,
        Replicate(),  # value,
        Replicate(),  # out,
        Replicate(),  # logsumexp,
        Replicate(),  # grad_out,
        Replicate(),  # grad_logsumexp,
        Replicate(),  # q_num_blocks,
        Replicate(),  # q_indices,
    ]

    # When sharding on the batch dimension but mask's batch size is larger than 1,
    # the mask has to be sharded along the batch dimension as well.
    batch_dim_sharding: PlacementList = [
        Shard(0),  # grad_query,
        Shard(0),  # grad_key,
        Shard(0),  # grad_value,
        Shard(0),  # query,
        Shard(0),  # key,
        Shard(0),  # value,
        Shard(0),  # out,
        Shard(0),  # logsumexp,
        Shard(0),  # grad_out,
        Shard(0),  # grad_logsumexp,
        Shard(0),  # q_num_blocks,
        Shard(0),  # q_indices,
    ]

    # When sharding on the batch dimension but mask's batch size is 1, the mask
    # is orthogonal to the batch dimension and can be replicated.
    batch_dim_sharding_mask_replicate: PlacementList = [
        Shard(0),  # grad_query,
        Shard(0),  # grad_key,
        Shard(0),  # grad_value,
        Shard(0),  # query,
        Shard(0),  # key,
        Shard(0),  # value,
        Shard(0),  # out,
        Shard(0),  # logsumexp,
        Shard(0),  # grad_out,
        Shard(0),  # grad_logsumexp,
        Replicate(),  # q_num_blocks,
        Replicate(),  # q_indices,
    ]

    # When sharding on the num_heads dimension, the mask has to be sharded along
    # the num_heads dimension as well if masks' num_heads size is larger than 1.
    # TODO: is this legal? Can the mask shard on the head dimension?
    num_heads_dim_sharding: PlacementList = [
        Shard(1),  # grad_query,
        Shard(1),  # grad_key,
        Shard(1),  # grad_value,
        Shard(1),  # query,
        Shard(1),  # key,
        Shard(1),  # value,
        Shard(1),  # out,
        Shard(1),  # logsumexp,
        Shard(1),  # grad_out,
        Shard(1),  # grad_logsumexp,
        Shard(1),  # q_num_blocks,
        Shard(1),  # q_indices,
    ]

    # When sharding on the num_heads dimension but mask's num_heads size is 1,
    # the mask is orthogonal to the num_heads dimension and can be replicated.
    num_heads_dim_sharding_mask_replicate: PlacementList = [
        Shard(1),  # grad_query,
        Shard(1),  # grad_key,
        Shard(1),  # grad_value,
        Shard(1),  # query,
        Shard(1),  # key,
        Shard(1),  # value,
        Shard(1),  # out,
        Shard(1),  # logsumexp,
        Shard(1),  # grad_out,
        Shard(1),  # grad_logsumexp,
        Replicate(),  # q_num_blocks,
        Replicate(),  # q_indices,
    ]

    # Context parallel sharding, where Q is sharded but K and V are replicated.
    # For backward, grad_Q follows Q sharding (Shard(2)), while grad_K and
    # grad_V follow K and V sharding (Replicate()).
    seq_dim_sharding: PlacementList = [
        Shard(2),  # grad_query,
        _Partial(),  # grad_key,
        _Partial(),  # grad_value,
        Shard(2),  # query,
        Replicate(),  # key,
        Replicate(),  # value,
        Shard(2),  # out,
        Shard(2),  # logsumexp,
        Shard(2),  # grad_out,
        Shard(2),  # grad_logsumexp,
        Replicate(),  # q_num_blocks,
        Shard(3),  # q_indices,
    ]

    single_mesh_dim_strategies = [
        replicate_sharding,
        batch_dim_sharding,
        batch_dim_sharding_mask_replicate,
        num_heads_dim_sharding,
        num_heads_dim_sharding_mask_replicate,
        seq_dim_sharding,
    ]

    op_strategy = expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=3
    )
    # We don't allow mask redistribution.
    # So if selecting this strategy results in mask redistribution,
    # we set the cost to be inf.
    for op_spec in op_strategy.strategies:
        costs = op_spec.redistribute_cost
        assert costs is not None
        # costs is a list of lists, one per input tensor
        # For backward: indices 0-2 are outputs (grad_query, grad_key, grad_value),
        # indices 3-9 are tensor inputs, and indices
        # 10-11 are the mask tensors (q_num_blocks, q_indices)
        for mask_idx in range(10 - 3, len(costs)):
            mask_costs = costs[mask_idx]
            # If any cost for this mask is > 0, mark all costs for this input as inf
            if any(cost > 0 for cost in mask_costs):
                costs[mask_idx] = [float("inf")] * len(mask_costs)

    return op_strategy


def _propagate_sharding_and_redistribute(
    op_info: OpInfo,
    op_schema: OpSchema,
    out_tensor_meta: tuple[TensorMeta, TensorMeta, TensorMeta],
) -> OutputSharding:
    """
    Propagate sharding strategy for flex_attention and redistribute inputs.

    This function performs custom sharding propagation for flex_attention
    instead of using the standard propagator.propagate_op_sharding_non_cached()
    because the standard path fails for the following reasons:

    1. **Modified operator signature**: We modify the op signature to only
       include the necessary mask tensors for sharding validation, since mask
       tensors cannot be resharded. This simplification is needed because
       some mask tensors may be None and some are only used in either the
       forward or backward pass.

    2. **FakeTensorMode compatibility**: It's unclear whether
       flex_attention_hop() and flex_attention_backward_hop() can be safely
       invoked under FakeTensorMode for automatic output metadata inference.

    The function follows the standard sharding propagation workflow:
    - Selects an appropriate sharding strategy based on input placements
    - Determines if input redistribution is required
    - Creates output specifications with the provided tensor metadata
    - Performs input redistribution if necessary

    Args:
        op_info: OpInfo containing the device mesh, operator schema, and
            local tensor arguments.
        op_schema: OpSchema describing the operator signature with
            input/output specifications.
        out_tensor_meta: Pre-computed output tensor metadata as a tuple of
            (output, logsumexp, max_scores) TensorMeta objects.

    Returns:
        OutputSharding: Contains the output specifications, optional
            redistribution schema, and flags indicating whether
            redistribution is needed.
    """
    propagator = DTensor._op_dispatcher.sharding_propagator

    strategy_schema = propagator._wrap_with_op_strategy(op_schema)
    op_strategy = propagator.op_strategy_funcs[op_schema.op](strategy_schema)
    assert isinstance(op_strategy, OpStrategy)
    output_strategy = propagator._select_strategy(op_strategy, op_schema)
    # check if we need to redistribute the input
    needs_redistribute = False
    # check if we want to use args value from redistribute_schema
    use_val_from_redistribute_schema = False
    expected_input_specs: list[DTensorSpec] = []

    # in case where the op does not specify input_specs and output_specs
    # is a DTensorSpec, we use output_specs as the spec for each DTensor
    # input arg.
    if output_strategy.input_specs is None:
        raise AssertionError("output_strategy.input_specs should not be None")
        assert isinstance(output_strategy.output_specs, DTensorSpec)

    for idx, input_spec in enumerate(op_schema.args_spec):
        desired_spec = (
            output_strategy.output_spec
            if output_strategy.input_specs is None
            else output_strategy.input_specs[idx]
        )
        expected_input_specs.append(
            desired_spec.shallow_copy_with_tensor_meta(input_spec.tensor_meta)
        )
        if input_spec.placements != desired_spec.placements:
            needs_redistribute = True

    suggestion_schema = None
    if needs_redistribute:
        suggestion_schema = OpSchema(op_schema.op, tuple(expected_input_specs), {})
        suggestion_schema._inplace_rewrap_schema_suggestion(op_schema)

    output_specs: OutputSpecType = output_strategy.output_specs
    output_sharding = OutputSharding(
        output_specs,
        suggestion_schema,
        needs_redistribute=needs_redistribute,
        use_val_from_redistribute_schema=use_val_from_redistribute_schema,
    )

    new_output_spec = propagator._create_output_spec_with_new_tensor_meta(
        op_schema.op, output_sharding.output_spec, out_tensor_meta
    )
    output_sharding.output_spec = new_output_spec

    if output_sharding.needs_redistribute:
        DTensor._op_dispatcher.redistribute_local_args(
            op_info,
            output_sharding.redistribute_schema,  # type: ignore[arg-type]
            output_sharding.use_val_from_redistribute_schema,
        )

    return output_sharding


def _flex_propagate(
    query: DTensor,
    key: DTensor,
    value: DTensor,
    block_mask: tuple[DTensor, ...],
) -> OpInfo:
    compute_mesh = query.device_mesh

    block_mask_spec = tuple(b._spec for b in block_mask)
    op_schema = OpSchema(
        flex_attention_hop,  # type: ignore[arg-type]
        args_schema=(query._spec, key._spec, value._spec, *block_mask_spec),
        kwargs_schema={},
    )

    local_args = [query._local_tensor, key._local_tensor, value._local_tensor]
    local_args.extend(b._local_tensor for b in block_mask)
    op_info = OpInfo(
        compute_mesh,
        op_schema,
        flat_args_schema=(query._spec, key._spec, value._spec, *block_mask_spec),  # type: ignore[arg-type]
        local_args=tuple(local_args),
        local_kwargs={},
        args_tree_spec=None,
    )

    # Create output tensor metadata for sharding propagation
    fake_mode = detect_fake_mode() or FakeTensorMode(allow_non_fake_inputs=True)
    with fake_mode:
        batch_size, num_heads, seq_len_q, _q_head_dim = query.shape
        v_head_dim = value.size(-1)
        out_shape = (batch_size, num_heads, seq_len_q, v_head_dim)
        logsumexp = query.new_empty(
            batch_size, num_heads, seq_len_q, dtype=torch.float32
        )
        max_scores = query.new_empty(
            batch_size, num_heads, seq_len_q, dtype=torch.float32
        )
        out = query.new_empty(out_shape)
        out = _permute_strides(out, query.stride())

    out_tensor_meta = (
        TensorMeta(out.shape, out.stride(), out.dtype),
        TensorMeta(logsumexp.shape, logsumexp.stride(), logsumexp.dtype),
        TensorMeta(max_scores.shape, max_scores.stride(), max_scores.dtype),
    )

    # Propagate sharding and redistribute inputs
    output_sharding = _propagate_sharding_and_redistribute(
        op_info, op_schema, out_tensor_meta
    )

    op_info.output_sharding = output_sharding
    return op_info


def _flex_backward_propagate(
    query: DTensor,
    key: DTensor,
    value: DTensor,
    out: DTensor,
    logsumexp: DTensor,
    grad_out: DTensor,
    grad_logsumexp: DTensor,
    block_mask: tuple[DTensor, ...],
) -> OpInfo:
    compute_mesh = query.device_mesh

    block_mask_spec = tuple(b._spec for b in block_mask)
    op_schema = OpSchema(
        flex_attention_backward_hop,  # type: ignore[arg-type]
        args_schema=(
            query._spec,
            key._spec,
            value._spec,
            out._spec,
            logsumexp._spec,
            grad_out._spec,
            grad_logsumexp._spec,
            *block_mask_spec,
        ),
        kwargs_schema={},
    )

    local_args = [
        query._local_tensor,
        key._local_tensor,
        value._local_tensor,
        out._local_tensor,
        logsumexp._local_tensor,
        grad_out._local_tensor,
        grad_logsumexp._local_tensor,
    ]
    local_args.extend(b._local_tensor for b in block_mask)
    op_info = OpInfo(
        compute_mesh,
        op_schema,
        flat_args_schema=(
            query._spec,
            key._spec,
            value._spec,
            out._spec,
            logsumexp._spec,
            grad_out._spec,
            grad_logsumexp._spec,
            *block_mask_spec,
        ),  # type: ignore[arg-type]
        local_args=tuple(local_args),
        local_kwargs={},
        args_tree_spec=None,
    )

    # Create output tensor metadata for sharding propagation
    fake_mode = detect_fake_mode() or FakeTensorMode(allow_non_fake_inputs=True)
    with fake_mode:
        Bq, _, _, qk_head_dim = query.shape
        Bkv, Hkv, seq_len_kv, v_head_dim = value.shape

        grad_query = torch.empty_like(query)
        broadcasted_grad_key = key.new_empty((Bq, Hkv, seq_len_kv, qk_head_dim))
        broadcasted_grad_key = _permute_strides(broadcasted_grad_key, key.stride())

        broadcasted_grad_value = value.new_empty((Bq, Hkv, seq_len_kv, v_head_dim))
        broadcasted_grad_value = _permute_strides(
            broadcasted_grad_value, value.stride()
        )

        if Bq > 1 and Bkv == 1:
            grad_key = torch.sum(broadcasted_grad_key, dim=0, keepdim=True)
            grad_value = torch.sum(broadcasted_grad_value, dim=0, keepdim=True)
        else:
            grad_key = broadcasted_grad_key
            grad_value = broadcasted_grad_value

    out_tensor_meta = (
        TensorMeta(grad_query.shape, grad_query.stride(), grad_query.dtype),
        TensorMeta(grad_key.shape, grad_key.stride(), grad_key.dtype),
        TensorMeta(grad_value.shape, grad_value.stride(), grad_value.dtype),
    )

    # Propagate sharding and redistribute inputs
    output_sharding = _propagate_sharding_and_redistribute(
        op_info, op_schema, out_tensor_meta
    )

    op_info.output_sharding = output_sharding
    return op_info


@flex_attention_hop.py_impl(DTensor)
def dtensor_flex_attention(
    query: DTensor,
    key: DTensor,
    value: DTensor,
    score_mod: Callable,
    block_mask: tuple[DTensor | torch.Tensor, ...],
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple[torch.Tensor, ...] = (),
    mask_mod_other_buffers: tuple[DTensor | torch.Tensor, ...] = (),
) -> tuple[DTensor, DTensor, DTensor]:
    if score_mod_other_buffers:
        raise ValueError(
            "FlexAttention + DTensor doesn't support score_mod_other_buffers yet."
        )

    # Convert DTensor buffers in mask_mod_other_buffers to local tensors
    mask_mod_other_buffers_local = pytree.tree_map(
        lambda x: x._local_tensor if isinstance(x, DTensor) else x,
        mask_mod_other_buffers,
    )

    op_info = _flex_propagate(query, key, value, block_mask[2:4])
    local_query, local_key, local_value, *_ = op_info.local_args
    # Cast to torch.Tensor for type checker
    assert isinstance(local_query, torch.Tensor)
    assert isinstance(local_key, torch.Tensor)
    assert isinstance(local_value, torch.Tensor)

    block_mask_local = pytree.tree_map(
        lambda x: x._local_tensor if isinstance(x, DTensor) else x,
        block_mask,
    )
    outputs = flex_attention_hop(
        query=local_query,
        key=local_key,
        value=local_value,
        score_mod=score_mod,
        block_mask=block_mask_local,
        scale=scale,
        kernel_options=kernel_options,
        score_mod_other_buffers=score_mod_other_buffers,
        mask_mod_other_buffers=mask_mod_other_buffers_local,
    )
    assert op_info.output_sharding is not None
    return tuple(
        DTensor(out, spec, requires_grad=out.requires_grad)
        for out, spec in zip(outputs, op_info.output_sharding.output_spec)  # type: ignore[arg-type]
    )


@flex_attention_backward_hop.py_impl(DTensor)
def dtensor_flex_attention_backward(
    query: DTensor,
    key: DTensor,
    value: DTensor,
    out: DTensor,
    logsumexp: DTensor,
    grad_out: DTensor,
    grad_logsumexp: DTensor | torch.Tensor,
    fw_graph: Callable | GraphModule,
    joint_graph: GraphModule,
    block_mask: tuple[DTensor | torch.Tensor, ...],
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple[torch.Tensor, ...] = (),
    mask_mod_other_buffers: tuple[DTensor | torch.Tensor, ...] = (),
) -> tuple[
    DTensor,
    DTensor,
    DTensor,
    tuple[DTensor | None, ...],
]:
    if score_mod_other_buffers:
        raise ValueError(
            "FlexAttention + DTensor doesn't support score_mod_other_buffers yet."
        )

    # Convert DTensor buffers in mask_mod_other_buffers to local tensors
    mask_mod_other_buffers_local = pytree.tree_map(
        lambda x: x._local_tensor if isinstance(x, DTensor) else x,
        mask_mod_other_buffers,
    )

    if not isinstance(grad_logsumexp, DTensor):
        # TODO: Why is this not a DTensor? Is it because that logsumexp is not used
        # by the downstream ops in the forward pass?
        assert grad_logsumexp.shape == logsumexp.shape
        # TODO: we assume the grad_logsumexp are all zeros if it is not a DTensor,
        # but I'm not sure if we can safely assume this.
        for i, s in enumerate(logsumexp._local_tensor.shape):
            grad_logsumexp = grad_logsumexp.narrow(i, 0, s)
        grad_logsumexp = DTensor.from_local(
            grad_logsumexp,
            device_mesh=out.device_mesh,
            placements=out.placements,
        )

    op_info = _flex_backward_propagate(
        query, key, value, out, logsumexp, grad_out, grad_logsumexp, block_mask[6:8]
    )
    (
        local_query,
        local_key,
        local_value,
        local_out,
        local_logsumexp,
        local_grad_out,
        local_grad_logsumexp,
        *_,
    ) = op_info.local_args
    # Cast to torch.Tensor for type checker
    assert isinstance(local_query, torch.Tensor)
    assert isinstance(local_key, torch.Tensor)
    assert isinstance(local_value, torch.Tensor)
    assert isinstance(local_out, torch.Tensor)
    assert isinstance(local_logsumexp, torch.Tensor)
    assert isinstance(local_grad_out, torch.Tensor)
    assert isinstance(local_grad_logsumexp, torch.Tensor)

    block_mask_local = pytree.tree_map(
        lambda x: x._local_tensor if isinstance(x, DTensor) else x,
        block_mask,
    )
    outputs = flex_attention_backward_hop(
        query=local_query,
        key=local_key,
        value=local_value,
        out=local_out,
        logsumexp=local_logsumexp,
        grad_out=local_grad_out,
        grad_logsumexp=local_grad_logsumexp,
        fw_graph=fw_graph,
        joint_graph=joint_graph,
        block_mask=block_mask_local,
        scale=scale,
        kernel_options=kernel_options,
        score_mod_other_buffers=score_mod_other_buffers,
        mask_mod_other_buffers=mask_mod_other_buffers_local,
    )
    assert op_info.output_sharding is not None
    result = []
    assert len(outputs) == 4 and len(op_info.output_sharding.output_spec) == 3
    for out, spec in zip(outputs, op_info.output_sharding.output_spec, strict=False):  # type: ignore[arg-type]
        result.append(DTensor(out, spec, requires_grad=False))
    result.append(tuple())

    return tuple(result)
