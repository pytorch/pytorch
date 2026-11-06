# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import _StridedShard, Replicate, Shard

from ..placement_contract import GlobalLayout, set_global_layout
from ..utils import _set_param_on_module


def dtensor_to_global_layout(module: nn.Module) -> None:
    """Replace DTensor parameters with local tensors that declare their layout.

    Call before ``flex_shard()`` on a model whose parameters are already
    DTensors, e.g. after expert or tensor parallelism. FlexShard then shards
    each local tensor, and the declared ``GlobalLayout`` records where it sits
    in the full parameter. Raises ``NotImplementedError`` for placements other
    than ``Shard`` and ``Replicate``.
    """
    for fqn, param in list(module.named_parameters(remove_duplicate=False)):
        if not isinstance(param, DTensor):
            continue
        if any(
            isinstance(p, _StridedShard) or not isinstance(p, (Shard, Replicate))
            for p in param.placements
        ):
            raise NotImplementedError(
                f"FlexShard does not support DTensor parameter {fqn!r} with "
                f"placements {param.placements!r}."
            )
        local_tensor = param.to_local().detach().contiguous()
        local_param = nn.Parameter(local_tensor, requires_grad=param.requires_grad)
        if param._has_grad_dtype_override:
            local_param.grad_dtype = param.grad_dtype
        set_global_layout(local_param, _dtensor_global_layout(param))
        _set_param_on_module(module, fqn, local_param)


def _dtensor_global_layout(param: DTensor) -> GlobalLayout:
    global_shape = tuple(param.shape)
    local_shape, global_offset = compute_local_shape_and_global_offset(
        param.shape, param.device_mesh, param.placements
    )
    if math.prod(local_shape) == 0:
        return GlobalLayout(global_shape, (), (), ())
    return GlobalLayout(
        global_shape=global_shape,
        global_offsets=(tuple(global_offset),),
        local_offsets=((0,) * len(local_shape),),
        local_sizes=(tuple(local_shape),),
    )
