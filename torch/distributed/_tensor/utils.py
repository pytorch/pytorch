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

