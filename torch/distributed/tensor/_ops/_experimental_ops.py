# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor

import torch
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpStrategy,
    PlacementStrategy,
    StrategyType,
)
from torch.distributed.tensor._ops.utils import get_mesh_from_args, register_op_strategy
from torch.distributed.tensor.placement_types import Replicate


aten = torch.ops.aten


@register_op_strategy(aten.slice_backward.default)
def slice_backward_rules(op_schema: OpSchema) -> StrategyType:
    """
    slice_backward is a new_zeros + slice_scatter, we only allow replication
    on the input/output for now since new_zeros would produce replication
    """
    mesh = get_mesh_from_args(op_schema, validate=False)
    replicate_spec = DTensorSpec(mesh, tuple([Replicate()] * mesh.ndim))
    return OpStrategy([PlacementStrategy(replicate_spec)])
