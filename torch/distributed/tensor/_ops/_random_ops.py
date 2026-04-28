# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from torch._ops import OpOverload
from torch.distributed.tensor._dtensor_spec import TensorMeta
from torch.distributed.tensor._op_schema import ArgsType, KwargsType
from torch.distributed.tensor._ops.single_dim_strategy import (
    _ShardingPlaceholder,
    register_single_dim_strategy,
)
from torch.distributed.tensor.placement_types import Placement


aten = torch.ops.aten


def _random_inplace_single_dim_strategy(
    op: OpOverload,
    args_schema: ArgsType,
    kwargs_schema: KwargsType,
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Single-dim strategy for in-place random ops (single tensor input, output follows input).

    No Partial inputs: random sampling on partial tensors is undefined.
    """
    self_meta = args_schema[0]
    if not isinstance(self_meta, TensorMeta):
        raise AssertionError
    num_outputs = sum(1 for r in op._schema.returns if "Tensor" in str(r.type))
    placements: list[list[Placement | _ShardingPlaceholder]] = []
    for i in range(len(self_meta.shape)):
        rule: list[Placement | _ShardingPlaceholder] = [_ShardingPlaceholder(i)] * (
            num_outputs + 1
        )
        placements.append(rule)
    return placements


# In-place random sampling ops: output follows input sharding exactly.
_inplace_random_ops = [
    aten.normal_.default,
    aten.uniform_.default,
    aten.native_dropout.default,
    aten.bernoulli_.float,
    aten.bernoulli.default,
    aten.bernoulli.p,
    aten.log_normal_.default,
    aten.exponential_.default,
    aten.geometric_.default,
]

for _op in _inplace_random_ops:
    register_single_dim_strategy(_op, allow_uneven_sharding=True)(
        _random_inplace_single_dim_strategy
    )


@register_single_dim_strategy(
    [
        aten.bernoulli_.Tensor,
        aten.bernoulli.Tensor,
    ]
)
def random_op_with_p_tensor_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    self_meta = args_schema[0]
    if not isinstance(self_meta, TensorMeta):
        raise AssertionError(f"Expect TensorMeta but got {type(self_meta)}")
    return [
        [_ShardingPlaceholder(d), _ShardingPlaceholder(d), _ShardingPlaceholder(d)]
        for d in range(len(self_meta.shape))
    ]


@register_single_dim_strategy(aten.multinomial.default)
def multinomial_single_dim_strategy(
    op: OpOverload,
    args_schema: ArgsType,
    kwargs_schema: KwargsType,
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Single-dim strategy for multinomial.

    multinomial(self, num_samples, ...) -> Tensor
    Input: [*, n_categories], Output: [*, num_samples] (dtype=long)

    Only batch dims (all except the last) can be sharded — the last dim
    (categories) is consumed by the sampling and maps to a different
    semantic dim (num_samples) in the output.
    """
    self_meta = args_schema[0]
    if not isinstance(self_meta, TensorMeta):
        raise AssertionError
    placements: list[list[Placement | _ShardingPlaceholder]] = []
    for i in range(len(self_meta.shape) - 1):
        rule: list[Placement | _ShardingPlaceholder] = [
            _ShardingPlaceholder(i),
            _ShardingPlaceholder(i),
        ]
        placements.append(rule)
    return placements
