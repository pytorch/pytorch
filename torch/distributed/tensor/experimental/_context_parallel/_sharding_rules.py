# Copyright (c) Meta Platforms, Inc. and affiliates
# Context Parallelism sharding rules for scaled_dot_product attention ops

from contextlib import contextmanager

import torch
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpStrategy,
    PlacementList,
    RuntimeSchemaInfo,
)
from torch.distributed.tensor._ops.utils import (
    expand_to_full_mesh_op_strategy,
    register_op_strategy,
)
from torch.distributed.tensor.placement_types import Replicate, Shard


aten = torch.ops.aten


@contextmanager
def _op_strategy_context(op_overload, strategy_func, schema_info=None):
    """
    Context manager for setting and clearing op strategies for Context Parallelism.

    Args:
        op_overload: The operator overload to set or clear the strategy for.
        strategy_func: The strategy function to set for the operator overload.
        schema_info: Optional schema information for the operator overload.

    Yields:
        None
    """
    from torch.distributed.tensor import DTensor

    propagator = DTensor._op_dispatcher.sharding_propagator
    _origin_op_strategy_funcs = None
    _origin_op_strategy_schema = None
    try:
        # Save original strategy if exists
        if op_overload in propagator.op_strategy_funcs:
            _origin_op_strategy_funcs = propagator.op_strategy_funcs[op_overload]
        if op_overload in propagator.op_to_schema_info:
            _origin_op_strategy_schema = propagator.op_to_schema_info[op_overload]

        # Register the new op strategy
        register_op_strategy(op_overload, schema_info=schema_info)(strategy_func)
        yield (_origin_op_strategy_funcs, _origin_op_strategy_schema)
    finally:
        # Restore original strategy
        if _origin_op_strategy_funcs is None:
            if op_overload in propagator.op_strategy_funcs:
                del propagator.op_strategy_funcs[op_overload]
        else:
            propagator.op_strategy_funcs[op_overload] = _origin_op_strategy_funcs

        if _origin_op_strategy_schema is None:
            if op_overload in propagator.op_to_schema_info:
                del propagator.op_to_schema_info[op_overload]
        else:
            propagator.op_to_schema_info[op_overload] = _origin_op_strategy_schema

        # Clear cache
        propagator.propagate_op_sharding.cache.cache_clear()


def _scaled_dot_product_flash_attention_cp_strategy(op_schema: OpSchema) -> OpStrategy:
    """
    Strategy for flash attention forward with Context Parallelism support.
    This includes the base strategies plus CP-specific sequence dimension sharding.
    """
    # Import here to avoid circular dependency
    from torch.distributed.tensor._ops._matrix_ops import (
        _scaled_dot_product_flash_attention_base_strategies,
    )

    # Get the base strategies (without CP modifications)
    mesh = op_schema.get_mesh_from_args()
    single_mesh_dim_strategies = _scaled_dot_product_flash_attention_base_strategies(
        op_schema
    )

    # Add Context Parallelism strategy: shards on the sequence dim
    return_debug_mask = len(op_schema.args_schema) >= 6 and op_schema.args_schema[5]
    debug_attn_mask_sharding = Shard(2) if return_debug_mask else Replicate()

    cp_strategy_placements: PlacementList = [
        Shard(2),  # output
        Shard(2),  # logsumexp
        None,  # cum_seq_q
        None,  # cum_seq_k
        None,  # max_q
        None,  # max_k
        Replicate(),  # rng_state
        None,  # unused
        debug_attn_mask_sharding,  # debugattn
        Shard(2),  # q
        Shard(2),  # k
        Shard(2),  # v
    ]
    single_mesh_dim_strategies.append(cp_strategy_placements)

    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=9
    )


def _scaled_dot_product_flash_attention_backward_cp_strategy(
    op_schema: OpSchema,
) -> OpStrategy:
    """
    Strategy for flash attention backward with Context Parallelism support.
    """
    from torch.distributed.tensor._op_schema import OpStrategy as OpStrat
    from torch.distributed.tensor._ops._matrix_ops import (
        _scaled_dot_product_flash_attention_backward_base_strategies,
    )

    mesh = op_schema.get_mesh_from_args(validate=False)
    single_mesh_dim_strategies = (
        _scaled_dot_product_flash_attention_backward_base_strategies(op_schema)
    )

    tensor_input_indices = [
        i
        for i, arg_spec in enumerate(op_schema.args_schema)
        if isinstance(arg_spec, OpStrat)
    ]
    num_tensor_inputs = len(tensor_input_indices)

    # Context Parallelism: shards on the sequence dim
    seq_dim_sharding: PlacementList = [
        Shard(2),  # grad_q
        Shard(2),  # grad_k
        Shard(2),  # grad_v
        Shard(2),  # grad_output
        Shard(2),  # q
        Shard(2),  # k
        Shard(2),  # v
        Shard(2),  # output
        Shard(2),  # logsumexp
    ]
    seq_dim_sharding.extend([Replicate()] * (num_tensor_inputs - 6))
    single_mesh_dim_strategies.append(seq_dim_sharding)

    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=3
    )


def _scaled_dot_product_efficient_attention_cp_strategy(
    op_schema: OpSchema,
) -> OpStrategy:
    """
    Strategy for efficient attention forward with Context Parallelism support.
    """
    from torch.distributed.tensor._ops._matrix_ops import (
        _scaled_dot_product_efficient_attention_base_strategies,
    )

    mesh = op_schema.get_mesh_from_args()
    single_mesh_dim_strategies = (
        _scaled_dot_product_efficient_attention_base_strategies(op_schema)
    )

    # Add Context Parallelism strategy
    has_attn_bias = op_schema.args_schema[3] is not None

    cp_strategy_placements: PlacementList = [
        Shard(2),  # output
        Shard(2),  # logsumexp
        None,  # philox_seed
        None,  # philox_offset
        Shard(2),  # q
        Shard(2),  # k
        Shard(2),  # v
    ]
    if has_attn_bias:
        cp_strategy_placements.append(Replicate())  # attn bias - not sharded for CP
    single_mesh_dim_strategies.append(cp_strategy_placements)

    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=4
    )


def _scaled_dot_product_efficient_attention_backward_cp_strategy(
    op_schema: OpSchema,
) -> OpStrategy:
    """
    Strategy for efficient attention backward with Context Parallelism support.
    """
    from torch.distributed.tensor._ops._matrix_ops import (
        _scaled_dot_product_efficient_attention_backward_base_strategies,
    )

    mesh = op_schema.get_mesh_from_args(validate=False)
    single_mesh_dim_strategies = (
        _scaled_dot_product_efficient_attention_backward_base_strategies(op_schema)
    )

    has_attn_bias = op_schema.args_schema[4] is not None

    # Context Parallelism: shards on the sequence dim
    seq_dim_sharding: PlacementList = [
        Shard(2),  # grad_q
        Shard(2),  # grad_k
        Shard(2),  # grad_v
        Shard(1) if has_attn_bias else None,  # grad_bias
        Shard(2),  # grad_output
        Shard(2),  # q
        Shard(2),  # k
        Shard(2),  # v
        Shard(2),  # output
        Shard(2),  # logsumexp
    ]
    if has_attn_bias:
        seq_dim_sharding.insert(8, Shard(1))  # attn_bias input
    seq_dim_sharding.extend([Replicate(), Replicate()])
    single_mesh_dim_strategies.append(seq_dim_sharding)

    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=4
    )


def _scaled_dot_product_cudnn_attention_cp_strategy(op_schema: OpSchema) -> OpStrategy:
    """
    Strategy for cudnn attention forward with Context Parallelism support.
    """
    from torch.distributed.tensor._ops._matrix_ops import (
        _scaled_dot_product_cudnn_attention_base_strategies,
    )

    mesh = op_schema.get_mesh_from_args()
    single_mesh_dim_strategies = _scaled_dot_product_cudnn_attention_base_strategies(
        op_schema
    )

    (
        query_strategy,
        _,
        _,
        attn_bias_strategy,
        compute_log_sumexp,
        *rest_args,
    ) = op_schema.args_schema
    return_debug_mask = len(op_schema.args_schema) >= 8 and rest_args[2]
    has_attn_bias = attn_bias_strategy is not None

    # Context Parallelism: shards on the sequence dim
    cp_sharding = Shard(2)
    logsumexp_sharding = cp_sharding if compute_log_sumexp else Replicate()
    debug_attn_mask_sharding = cp_sharding if return_debug_mask else None

    cp_strategy_placements: PlacementList = [
        cp_sharding,  # output
        logsumexp_sharding,  # logsumexp
        None,  # cum_seq_q
        None,  # cum_seq_k
        None,  # max_q
        None,  # max_k
        None,  # philox_seed
        None,  # philox_offset
        debug_attn_mask_sharding,  # debug_attn_mask
        cp_sharding,  # q
        cp_sharding,  # k
        cp_sharding,  # v
    ]
    if has_attn_bias:
        cp_strategy_placements.append(Replicate())  # attn_bias - not sharded for CP
    single_mesh_dim_strategies.append(cp_strategy_placements)

    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=9
    )


def _scaled_dot_product_cudnn_attention_backward_cp_strategy(
    op_schema: OpSchema,
) -> OpStrategy:
    """
    Strategy for cudnn attention backward with Context Parallelism support.
    """
    from torch.distributed.tensor._ops._matrix_ops import (
        _scaled_scaled_dot_product_cudnn_attention_backward_base_strategies,
    )

    mesh = op_schema.get_mesh_from_args(validate=False)
    single_mesh_dim_strategies = (
        _scaled_scaled_dot_product_cudnn_attention_backward_base_strategies(op_schema)
    )

    has_attn_bias = op_schema.args_schema[8] is not None
    has_scale = len(op_schema.args_schema) >= 16 and False

    # Context Parallelism: shards on the sequence dim
    context_parallel_sharding_out: PlacementList = [Shard(2)] * 3
    context_parallel_sharding_inp: PlacementList = [Shard(2)] * 6
    context_parallel_sharding_inp += [Replicate()] * 2  # philox_seed, philox_offset
    context_parallel_sharding_inp += [Shard(2) if has_attn_bias else None]
    context_parallel_sharding_inp += [None] * 6
    if has_scale:
        context_parallel_sharding_inp.append(None)

    context_parallel_sharding = (
        context_parallel_sharding_out + context_parallel_sharding_inp
    )
    single_mesh_dim_strategies.append(context_parallel_sharding)

    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=3
    )


# Store context managers and original strategies
_cp_strategy_contexts = {}
_original_strategies = {}


def register_cp_sharding_rules():
    """Register Context Parallelism sharding rules for all scaled_dot_product ops."""
    global _cp_strategy_contexts, _original_strategies

    # If already registered, don't register again
    if _cp_strategy_contexts:
        return

    # Define ops and their corresponding CP strategy functions
    cp_strategies = [
        (
            aten._scaled_dot_product_flash_attention.default,
            _scaled_dot_product_flash_attention_cp_strategy,
            RuntimeSchemaInfo(5),
        ),
        (
            aten._scaled_dot_product_flash_attention_backward.default,
            _scaled_dot_product_flash_attention_backward_cp_strategy,
            None,
        ),
        (
            aten._scaled_dot_product_efficient_attention.default,
            _scaled_dot_product_efficient_attention_cp_strategy,
            RuntimeSchemaInfo(4),
        ),
        (
            aten._scaled_dot_product_efficient_attention_backward.default,
            _scaled_dot_product_efficient_attention_backward_cp_strategy,
            None,
        ),
        (
            aten._scaled_dot_product_cudnn_attention.default,
            _scaled_dot_product_cudnn_attention_cp_strategy,
            RuntimeSchemaInfo(4),
        ),
        (
            aten._scaled_dot_product_cudnn_attention_backward.default,
            _scaled_dot_product_cudnn_attention_backward_cp_strategy,
            None,
        ),
    ]

    # Register each strategy
    for op_overload, strategy_func, schema_info in cp_strategies:
        ctx = _op_strategy_context(op_overload, strategy_func, schema_info)
        orig_funcs, orig_schema = ctx.__enter__()
        _cp_strategy_contexts[op_overload] = ctx
        _original_strategies[op_overload] = (orig_funcs, orig_schema)


def unregister_cp_sharding_rules():
    """Unregister Context Parallelism sharding rules and restore original strategies."""
    global _cp_strategy_contexts, _original_strategies

    # Exit all context managers
    for ctx in _cp_strategy_contexts.values():
        ctx.__exit__(None, None, None)

    _cp_strategy_contexts = {}
    _original_strategies = {}
