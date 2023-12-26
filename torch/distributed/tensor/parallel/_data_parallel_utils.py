from functools import partial
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.distributed._functional_collectives import AsyncCollectiveTensor


def grad_layout_hook(param_placements, grad):
    # a gradient hook to ensure the gradient layout is the same as
    # the parameter layout, this is due to the fact that our current
    # FSDP have implicit assumption that param/grad sharding layout
    # should be the same after backward. However this is not always
    # the case for DTensor, i.e. we might have a replicated param
    # and a partial gradient and DTensor was relying on optimizer
    # who really consumes the gradient to convert the layout.
    if isinstance(grad, DTensor) and grad.placements != param_placements:
        # dist.all_reduce(grad._local_tensor, group=grad.device_mesh.get_group())
        grad = grad.redistribute(placements=param_placements)

    return grad



def _flatten_tensor(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[DTensorSpec]]:
    if isinstance(tensor, DTensor):
        tensor._local_tensor.requires_grad_()
        return tensor._local_tensor, tensor._spec
    return tensor, None


@torch._dynamo.disable
def _unflatten_tensor(tensor: torch.Tensor, spec: DTensorSpec) -> torch.Tensor:
    # unflatten would mainly be called everytime FSDP allgather parameters.
    result = DTensor.from_local(
        tensor,
        spec.mesh,
        spec.placements,
        run_check=False,
    )
    # if result.requires_grad:
    #     # only register the hook if the tensor requires grad
    #     result.register_hook(partial(grad_layout_hook, spec.placements))
    return result
