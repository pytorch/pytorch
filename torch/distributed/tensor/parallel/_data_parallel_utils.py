from functools import partial
from typing import no_type_check, Optional, Tuple

import torch
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import DTensorSpec


@no_type_check
def sync_grad_hook(grad, *, device_handle=None, compute_stream=None):
    if isinstance(grad, AsyncCollectiveTensor):
        if compute_stream is not None:
            with device_handle.stream(compute_stream):
                grad = grad.wait()
        else:
            grad = grad.wait()

    return grad


def _flatten_tensor(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[DTensorSpec]]:
    if isinstance(tensor, DTensor):
        tensor._local_tensor.requires_grad_()
        return tensor._local_tensor, tensor._spec
    return tensor, None


@no_type_check
def _unflatten_tensor(tensor, spec, *, device_handle=None, compute_stream=None):
    # unflatten would mainly be called everytime FSDP allgather parameters.
    result = DTensor.from_local(
        tensor,
        spec.mesh,
        spec.placements,
        run_check=False,
        shape=spec.shape,
        stride=spec.stride,
    )
    if tensor.requires_grad:
        # only register the hook if the tensor requires grad
        tensor.register_hook(
            partial(
                sync_grad_hook,
                device_handle=device_handle,
                compute_stream=compute_stream,
            )
        )
    return result
