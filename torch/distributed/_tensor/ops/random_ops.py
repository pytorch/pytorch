# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import cast

import torch

from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import _Partial, DTensorSpec

aten = torch.ops.aten
random_ops = [
    aten.normal_.default,
    aten.uniform_.default,
]


def _register_non_deterministic_op(op):
    @register_prop_rule(op)
    def non_deterministic_rule(op_schema: OpSchema) -> OutputSharding:
        self_spec = cast(DTensorSpec, op_schema.args_schema[0])

        # NOTE: random op behavior on a partial tensor is TBD
        partial = False
        for placement in self_spec.placements:
            if isinstance(placement, _Partial):
                partial = True
                break

        if partial:
            return OutputSharding(
                None, failed_reason=f"{op} with _Partial is not supported yet!"
            )
        else:
            return OutputSharding(self_spec)


@register_prop_rule(aten.native_dropout.default)
def dropout_rule(op_schema: OpSchema) -> OutputSharding:
    self_spec = cast(DTensorSpec, op_schema.args_schema[0])

    # NOTE: dropout on a partial tensor should be similar to the case of a replicate tensor
    partial = False
    for placement in self_spec.placements:
        if isinstance(placement, _Partial):
            partial = True
            break

    if partial:
        return OutputSharding(
            None,
            failed_reason="aten.native_dropout.default with _Partial is not supported yet!",
        )
    else:
        return OutputSharding([self_spec, self_spec])


for op in random_ops:
    _register_non_deterministic_op(op)
