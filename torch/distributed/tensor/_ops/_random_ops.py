# mypy: allow-untyped-decorators
# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpStrategy,
    PlacementStrategy,
    StrategyType,
)
from torch.distributed.tensor._ops.utils import is_tensor_partial, register_op_strategy


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
def random_op_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    self_strategy = op_schema.args_schema[0]
    assert isinstance(self_strategy, OpStrategy)

    random_strategy = OpStrategy([])
    for arg_strategy in self_strategy.strategies:
        arg_spec = arg_strategy.output_spec
        if is_tensor_partial(arg_spec):
            # TODO: figure out how inplace random op should behave when it's partial
            raise RuntimeError(f"{op_schema.op} with Partial is not supported yet!")
        random_strategy.strategies.append(PlacementStrategy(output_specs=arg_spec))

    return random_strategy
