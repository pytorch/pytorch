# mypy: allow-untyped-decorators
# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor

import torch
from torch.distributed._tensor._op_schema import (
    OpSchema,
    OpStrategy,
    PlacementStrategy,
    StrategyType,
)
from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed._tensor.ops.utils import register_op_strategy
from torch.distributed._tensor.placement_types import DTensorSpec, Replicate


aten = torch.ops.aten


@register_op_strategy(aten.slice_backward.default)
def slice_backward_rules(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    """
    slice_backward is a new_zeros + slice_scatter, we only allow replication
    on the input/output for now since new_zeros would produce replication
    """
    replicate_spec = DTensorSpec(mesh, tuple([Replicate()] * mesh.ndim))
    return OpStrategy([PlacementStrategy(replicate_spec)])
