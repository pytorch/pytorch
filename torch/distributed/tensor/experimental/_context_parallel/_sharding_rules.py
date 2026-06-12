# Copyright (c) Meta Platforms, Inc. and affiliates
"""
Context Parallelism sharding rules for scaled_dot_product attention operators.

The sharding rules for CP cannot be embedded by default because Shard(2) is not
a valid sharding for SDPA without CP enabled. This module provides utilities to
dynamically install Shard(2) sharding rules when CP is activated.
"""

from contextlib import contextmanager

import torch
from torch.distributed.tensor._dtensor_spec import TensorMeta
from torch.distributed.tensor._op_schema import ArgsType, KwargsType, RuntimeSchemaInfo
from torch.distributed.tensor._ops.single_dim_strategy import (
    _ShardingPlaceholder,
    register_single_dim_strategy,
)
from torch.distributed.tensor.debug import (
    _clear_fast_path_sharding_prop_cache,
    _clear_python_sharding_prop_cache,
)
from torch.distributed.tensor.placement_types import Placement, Replicate


aten = torch.ops.aten

SEQ_DIM = 2
SingleDimPlacementList = list[Placement | _ShardingPlaceholder | None]


@contextmanager
def _single_dim_strategy_context(op_overload, strategy_func, schema_info=None):
    """Temporarily install a single-dim strategy for Context Parallelism."""
    from torch.distributed.tensor import DTensor

    propagator = DTensor._op_dispatcher.sharding_propagator
    _origin_single_dim_strategy = None
    _origin_op_strategy_schema = None
    try:
        if op_overload in propagator.op_single_dim_strategy_funcs:
            _origin_single_dim_strategy = propagator.op_single_dim_strategy_funcs[
                op_overload
            ]
        if op_overload in propagator.op_to_schema_info:
            _origin_op_strategy_schema = propagator.op_to_schema_info[op_overload]

        register_single_dim_strategy(op_overload, schema_info=schema_info)(
            strategy_func
        )
        yield (_origin_single_dim_strategy, _origin_op_strategy_schema)
    finally:
        if _origin_single_dim_strategy is None:
            if op_overload in propagator.op_single_dim_strategy_funcs:
                del propagator.op_single_dim_strategy_funcs[op_overload]
        else:
            propagator.op_single_dim_strategy_funcs[op_overload] = (
                _origin_single_dim_strategy
            )

        if _origin_op_strategy_schema is None:
            if op_overload in propagator.op_to_schema_info:
                del propagator.op_to_schema_info[op_overload]
        else:
            propagator.op_to_schema_info[op_overload] = _origin_op_strategy_schema

        # Ideally, we should clear the cache, but it is too expensive.
        # _clear_python_sharding_prop_cache()
        # _clear_fast_path_sharding_prop_cache()


def _cp_sharding() -> _ShardingPlaceholder:
    return _ShardingPlaceholder(SEQ_DIM)


# ==================== Flash Attention Strategies ====================


def _scaled_dot_product_flash_attention_cp_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: ArgsType,
    kwargs_schema: KwargsType,
) -> list[SingleDimPlacementList]:
    return_debug_mask = len(args_schema) >= 6 and args_schema[5]
    debug_attn_mask_sharding: Placement | _ShardingPlaceholder = (
        _cp_sharding() if return_debug_mask else Replicate()
    )

    return [
        [
            _cp_sharding(),  # output
            _cp_sharding(),  # logsumexp
            None,  # cum_seq_q
            None,  # cum_seq_k
            None,  # max_q
            None,  # max_k
            Replicate(),  # rng_state
            None,  # unused
            debug_attn_mask_sharding,
            _cp_sharding(),  # q
            _cp_sharding(),  # k
            _cp_sharding(),  # v
        ]
    ]


def _scaled_dot_product_flash_attention_backward_cp_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: ArgsType,
    kwargs_schema: KwargsType,
) -> list[SingleDimPlacementList]:
    num_tensor_inputs = sum(isinstance(arg, TensorMeta) for arg in args_schema)
    cp_strategy: SingleDimPlacementList = [
        _cp_sharding(),  # grad_q
        _cp_sharding(),  # grad_k
        _cp_sharding(),  # grad_v
        _cp_sharding(),  # grad_output
        _cp_sharding(),  # q
        _cp_sharding(),  # k
        _cp_sharding(),  # v
        _cp_sharding(),  # output
        _cp_sharding(),  # logsumexp
    ]
    cp_strategy.extend([Replicate()] * (num_tensor_inputs - 6))
    return [cp_strategy]


# ==================== Efficient Attention Strategies ====================


def _scaled_dot_product_efficient_attention_cp_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: ArgsType,
    kwargs_schema: KwargsType,
) -> list[SingleDimPlacementList]:
    has_attn_bias = args_schema[3] is not None

    cp_strategy: SingleDimPlacementList = [
        _cp_sharding(),  # output
        _cp_sharding(),  # logsumexp
        None,  # philox_seed
        None,  # philox_offset
        _cp_sharding(),  # q
        _cp_sharding(),  # k
        _cp_sharding(),  # v
    ]
    if has_attn_bias:
        cp_strategy.append(Replicate())
    return [cp_strategy]


def _scaled_dot_product_efficient_attention_backward_cp_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: ArgsType,
    kwargs_schema: KwargsType,
) -> list[SingleDimPlacementList]:
    has_attn_bias = args_schema[4] is not None

    cp_strategy: SingleDimPlacementList = [
        _cp_sharding(),  # grad_q
        _cp_sharding(),  # grad_k
        _cp_sharding(),  # grad_v
        _ShardingPlaceholder(1) if has_attn_bias else None,  # grad_bias
        _cp_sharding(),  # grad_output
        _cp_sharding(),  # q
        _cp_sharding(),  # k
        _cp_sharding(),  # v
        _cp_sharding(),  # output
        _cp_sharding(),  # logsumexp
    ]
    if has_attn_bias:
        cp_strategy.insert(8, _ShardingPlaceholder(1))  # attn_bias
    cp_strategy.extend([Replicate(), Replicate()])
    return [cp_strategy]


# ==================== cuDNN Attention Strategies ====================


def _scaled_dot_product_cudnn_attention_cp_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: ArgsType,
    kwargs_schema: KwargsType,
) -> list[SingleDimPlacementList]:
    attn_bias_meta = args_schema[3]
    compute_log_sumexp = args_schema[4]
    return_debug_mask = len(args_schema) >= 8 and args_schema[7]
    has_attn_bias = attn_bias_meta is not None

    logsumexp_sharding: Placement | _ShardingPlaceholder = (
        _cp_sharding() if compute_log_sumexp else Replicate()
    )
    debug_attn_mask_sharding = _cp_sharding() if return_debug_mask else None

    cp_strategy: SingleDimPlacementList = [
        _cp_sharding(),  # output
        logsumexp_sharding,
        None,  # cum_seq_q
        None,  # cum_seq_k
        None,  # max_q
        None,  # max_k
        None,  # philox_seed
        None,  # philox_offset
        debug_attn_mask_sharding,
        _cp_sharding(),  # q
        _cp_sharding(),  # k
        _cp_sharding(),  # v
    ]
    if has_attn_bias:
        cp_strategy.append(Replicate())
    return [cp_strategy]


def _scaled_dot_product_cudnn_attention_backward_cp_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: ArgsType,
    kwargs_schema: KwargsType,
) -> list[SingleDimPlacementList]:
    if len(args_schema) < 15:
        raise AssertionError(f"Expected at least 15 args, got {len(args_schema)}")

    for arg_index in range(6):
        arg = args_schema[arg_index]
        if not isinstance(arg, TensorMeta):
            raise AssertionError(f"Expected TensorMeta, got {type(arg)}")

    philox_placements: list[Placement] = []
    for arg_index in (6, 7):
        arg = args_schema[arg_index]
        if isinstance(arg, TensorMeta):
            philox_placements.append(Replicate())
        elif not isinstance(arg, torch.Tensor):
            raise AssertionError(f"Expected TensorMeta or Tensor, got {type(arg)}")

    has_attn_bias = args_schema[8] is not None
    if has_attn_bias and not isinstance(args_schema[8], (TensorMeta, torch.Tensor)):
        raise AssertionError(
            f"Expected TensorMeta or Tensor, got {type(args_schema[8])}"
        )

    cum_seq_placements: list[None] = []
    for arg_index in (9, 10):
        arg = args_schema[arg_index]
        if isinstance(arg, TensorMeta):
            cum_seq_placements.append(None)
        elif arg is None or isinstance(arg, torch.Tensor):
            pass
        else:
            raise AssertionError(f"Expected TensorMeta or Tensor, got {type(arg)}")

    cp_sharding: SingleDimPlacementList = [
        _cp_sharding(),  # grad_q
        _cp_sharding(),  # grad_k
        _cp_sharding(),  # grad_v
        _cp_sharding(),  # grad_out
        _cp_sharding(),  # q
        _cp_sharding(),  # k
        _cp_sharding(),  # v
        _cp_sharding(),  # output
        _cp_sharding(),  # logsumexp
    ]
    cp_sharding.extend(philox_placements)
    if has_attn_bias and isinstance(args_schema[8], TensorMeta):
        cp_sharding.append(_cp_sharding())
    cp_sharding.extend(cum_seq_placements)

    return [cp_sharding]


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
            _scaled_dot_product_flash_attention_cp_single_dim_strategy,
            RuntimeSchemaInfo(5),
        ),
        (
            aten._scaled_dot_product_flash_attention_backward.default,
            _scaled_dot_product_flash_attention_backward_cp_single_dim_strategy,
            None,
        ),
        (
            aten._scaled_dot_product_efficient_attention.default,
            _scaled_dot_product_efficient_attention_cp_single_dim_strategy,
            RuntimeSchemaInfo(4),
        ),
        (
            aten._scaled_dot_product_efficient_attention_backward.default,
            _scaled_dot_product_efficient_attention_backward_cp_single_dim_strategy,
            None,
        ),
        (
            aten._scaled_dot_product_cudnn_attention.default,
            _scaled_dot_product_cudnn_attention_cp_single_dim_strategy,
            RuntimeSchemaInfo(4),
        ),
        (
            aten._scaled_dot_product_cudnn_attention_backward.default,
            _scaled_dot_product_cudnn_attention_backward_cp_single_dim_strategy,
            None,
        ),
    ]

    # Register each strategy
    for op_overload, strategy_func, schema_info in cp_strategies:
        ctx = _single_dim_strategy_context(op_overload, strategy_func, schema_info)
        orig_funcs, orig_schema = ctx.__enter__()
        _cp_strategy_contexts[op_overload] = ctx
        _original_strategies[op_overload] = (orig_funcs, orig_schema)


def unregister_cp_sharding_rules(clear_the_cache=False):
    """Unregister Context Parallelism sharding rules and restore original strategies."""
    global _cp_strategy_contexts, _original_strategies

    # Exit all context managers
    for ctx in _cp_strategy_contexts.values():
        ctx.__exit__(None, None, None)

    if clear_the_cache:
        _clear_fast_path_sharding_prop_cache()
        _clear_python_sharding_prop_cache()

    _cp_strategy_contexts = {}
    _original_strategies = {}
