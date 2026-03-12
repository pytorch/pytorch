# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import cast

import torch
from torch._ops import OpOverload
from torch.distributed.tensor._op_schema import (
    ArgsType,
    KwargsType,
    OpSchema,
    OpSpec,
    OpStrategy,
    RuntimeSchemaInfo,
    StrategyType,
    TensorMeta,
)
from torch.distributed.tensor._ops.single_dim_strategy import (
    _ShardingPlaceholder,
    register_single_dim_strategy,
)
from torch.distributed.tensor._ops.utils import is_tensor_partial, register_op_strategy
from torch.distributed.tensor.placement_types import Placement


aten = torch.ops.aten


@register_op_strategy(
    [
        aten.normal_.default,
        aten.uniform_.default,
        aten.native_dropout.default,
        aten.bernoulli_.float,
        aten.bernoulli.default,
    ]
)
def random_op_strategy(op_schema: OpSchema) -> StrategyType:
    self_strategy = op_schema.args_schema[0]
    if not isinstance(self_strategy, OpStrategy):
        raise AssertionError

    random_strategy = OpStrategy([])
    for arg_strategy in self_strategy.strategies:
        arg_spec = arg_strategy.output_spec
        if is_tensor_partial(arg_spec):
            # TODO: figure out how inplace random op should behave when it's partial
            raise RuntimeError(f"{op_schema.op} with Partial is not supported yet!")
        random_strategy.strategies.append(
            OpSpec(
                output_specs=arg_spec,
                input_specs=(arg_spec,),
                redistribute_cost=[[0.0] * len(self_strategy.strategies)],
            )
        )

    return random_strategy


@register_single_dim_strategy(
    [aten.exponential_.default, aten.geometric_.default, aten.log_normal_.default],
)
def inplace_random_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    self_meta = cast(TensorMeta, args_schema[0])
    ndim = len(self_meta.shape)
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for d in range(ndim):
        strategies.append([_ShardingPlaceholder(d), _ShardingPlaceholder(d)])
    return strategies


@register_single_dim_strategy(aten.multinomial.default)
def multinomial_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    return []
