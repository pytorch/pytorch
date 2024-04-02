from typing import Optional, Tuple

import torch
from torch.distributed._tensor import DTensor as DistributedTensor
from torch.distributed._tensor.placement_types import DTensorSpec


def _flatten_tensor(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[DTensorSpec]]:
    if isinstance(tensor, DistributedTensor):
        tensor._local_tensor.requires_grad_()
        return tensor._local_tensor, tensor._spec
    return tensor, None


def _unflatten_tensor(tensor: torch.Tensor, spec: DTensorSpec) -> torch.Tensor:
    result = DistributedTensor.from_local(
        tensor,
        spec.mesh,
        spec.placements,
        run_check=False,
    )
    return result
