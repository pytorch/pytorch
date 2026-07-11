# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

from typing import cast

import torch
from torch.distributed.tensor._dtensor_spec import TensorMeta
from torch.distributed.tensor._ops.single_dim_strategy import (
    _ShardingPlaceholder,
    register_single_dim_strategy,
)
from torch.distributed.tensor._ops.utils import normalize_dims
from torch.distributed.tensor.placement_types import Placement


aten = torch.ops.aten


@register_single_dim_strategy(
    [aten._fft_c2c.default],
    allow_uneven_sharding=True,
    allow_unbacked_sharding=False,
)
def fft_c2c_single_dim_strategy(
    _op: torch._ops.OpOverload,
    args_schema: tuple[object, ...],
    _kwargs_schema: dict[str, object],
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = cast(TensorMeta, args_schema[0])
    dims_arg = cast(int | list[int] | tuple[int, ...], args_schema[1])

    ndim = len(input_meta.shape)
    fft_dims = set(normalize_dims(dims_arg, ndim))

    return [
        [_ShardingPlaceholder(d), _ShardingPlaceholder(d)]
        for d in range(ndim)
        if d not in fft_dims
    ]
