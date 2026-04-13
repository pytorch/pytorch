# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpSpec,
    OpStrategy,
    StrategyType,
)
from torch.distributed.tensor._ops.utils import (
    generate_redistribute_costs,
    is_tensor_partial,
    register_op_strategy,
)


aten = torch.ops.aten


@register_op_strategy(
    [
        aten.normal_.default,
        aten.uniform_.default,
        aten.native_dropout.default,
        aten.bernoulli_.float,
        aten.bernoulli_.Tensor,
        aten.bernoulli.default,
        aten.bernoulli.p,
        aten.bernoulli.Tensor,
    ]
)
def random_op_strategy(op_schema: OpSchema) -> StrategyType:
    self_strategy = op_schema.args_schema[0]
    if not isinstance(self_strategy, OpStrategy):
        raise AssertionError

    # bernoulli_.Tensor / bernoulli.Tensor have a second Tensor arg (probability p)
    p_strategy = None
    if len(op_schema.args_schema) > 1 and isinstance(
        op_schema.args_schema[1], OpStrategy
    ):
        p_strategy = op_schema.args_schema[1]

    random_strategy = OpStrategy([])
    for arg_strategy in self_strategy.strategies:
        arg_spec = arg_strategy.output_spec
        if is_tensor_partial(arg_spec):
            raise RuntimeError(f"{op_schema.op} with Partial is not supported yet!")

        if p_strategy is not None:
            input_specs = (arg_spec, arg_spec)
            redistribute_cost = [
                [0.0] * len(self_strategy.strategies),
                generate_redistribute_costs(p_strategy, arg_spec),
            ]
        else:
            input_specs = (arg_spec,)
            redistribute_cost = [[0.0] * len(self_strategy.strategies)]

        random_strategy.strategies.append(
            OpSpec(
                output_specs=arg_spec,
                input_specs=input_specs,
                redistribute_cost=redistribute_cost,
            )
        )

    return random_strategy
