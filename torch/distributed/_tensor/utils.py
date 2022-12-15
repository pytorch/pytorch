# Copyright (c) Meta Platforms, Inc. and affiliates

from typing import Dict, Optional, Sequence, Tuple, Union

import torch

import torch.distributed._tensor.api as dtensor
from torch.distributed._tensor.placement_types import DTensorSpec

ArgKwargsType = Union[Tuple[object, ...], Dict[str, object]]
# ATen op schemas could have Tensor, Tuple[Tensor] and List[Tensor], so output type sould
# be the same set of possiblities.
OutputSpecType = Optional[Union[DTensorSpec, Sequence[Optional[DTensorSpec]]]]


def unwrap_local_tensor(e: "dtensor.DTensor") -> torch.Tensor:
    return e._local_tensor if isinstance(e, dtensor.DTensor) else e


def unwrap_schema(e: object) -> object:
    return e._spec if isinstance(e, dtensor.DTensor) else e


def wrap(res: object, spec: OutputSpecType) -> object:
    if isinstance(res, torch.Tensor):
        assert spec is not None and isinstance(
            spec, DTensorSpec
        ), f"output spec does not match with output! Expected DTensorSpec, got {spec}."
        return dtensor.DTensor(
            res,
            spec.mesh,
            spec.placements,
            size=spec.shape,
            requires_grad=res.requires_grad,
        )
    elif isinstance(res, list):
        assert spec is not None and isinstance(
            spec, list
        ), f"output spec does not match with output! Expected list, got {spec}."
        return list(
            dtensor.DTensor(e, s.mesh, s.placements, size=s.shape)
            for e, s in zip(res, spec)
        )
    elif isinstance(res, tuple):
        assert spec is not None and isinstance(
            spec, tuple
        ), f"output spec does not match with output! Expected tuple, got {spec}"

        # NOTE: local results might return Optional Tensor from ATen op, so we need to
        # handle that case and make sure we don't wrap None with DTensor.
        # (i.e. native_layer_norm.backward)
        return tuple(
            dtensor.DTensor(e, s.mesh, s.placements, size=s.shape)
            if e is not None and s is not None else None
            for e, s in zip(res, spec)
        )
    else:
        # if the res contains only non tensor values, we simply return it without rewrapping
        return res
