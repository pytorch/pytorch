from typing import Optional, Tuple

import torch
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed._tensor import DTensor
from torch.distributed._tensor.placement_types import DTensorSpec


def sync_grad_hook(grad):
    if isinstance(grad, AsyncCollectiveTensor):
        grad.wait()
    return grad


def _flatten_tensor(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[DTensorSpec]]:
    if isinstance(tensor, DTensor):
        tensor._local_tensor.requires_grad_()
        return tensor._local_tensor, tensor._spec
    return tensor, None


def _unflatten_tensor(tensor: torch.Tensor, spec: DTensorSpec) -> torch.Tensor:
    # unflatten would mainly be called everytime FSDP allgather parameters.
    result = DTensor.from_local(
        tensor,
        spec.mesh,
        spec.placements,
        run_check=False,
    )
    if tensor.requires_grad:
        tensor.register_hook(sync_grad_hook)
    return result
