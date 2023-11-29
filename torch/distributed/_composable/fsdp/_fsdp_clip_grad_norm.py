import functools
import math
import warnings
from typing import cast, Iterable, List, Union

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor


@torch.no_grad()
def _get_grad_norm(
    local_grads: List[torch.Tensor],
    norm_type: float,
    default_device: torch.device,
) -> torch.Tensor:
    """
    Returns the gradient norm of ``local_grads``, where the gradients are
    viewed as a single vector.
    """
    if len(local_grads) == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=default_device)
    vector_norm_fn = functools.partial(
        torch.linalg.vector_norm, ord=norm_type, dtype=torch.float32
    )
    grad_norm = vector_norm_fn(
        torch.stack([vector_norm_fn(grad) for grad in local_grads])
    )
    return grad_norm


@torch.no_grad()
def clip_grad_norm_(
    parameters: Union[DTensor, Iterable[DTensor]],
    max_norm: float,
    norm_type: float = 2.0,
) -> torch.Tensor:
    """
    Clip the gradient norm of all parameters in-place. The norm is computed
    over all gradients together, as if they were concatenated into a single
    vector, including communication for sharded gradients.

    Args:
        max_norm (float): Max norm of the gradients.
        norm_type (float): Type of the used p-norm. Can be ``float('inf')`` for
            infinity norm.

    Returns:
        torch.Tensor: Total norm of the gradients in ``torch.float32``.

    .. note:: If the passed in ``parameters`` is empty, then this returns the
        zero tensor on CPU, following the convention of
        :func:`torch.nn.utils.clip_grad_norm_`. Otherwise, this returns the
        total norm on the same device as the parameters' gradients.

    .. warning:: This needs to be called on all ranks since it uses collective
        communications.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [cast(DTensor, parameters)]
    else:
        parameters = list(parameters)
    if len(parameters) == 0:
        # Use default device following `torch.nn.utils.clip_grad_norm_()` but
        # use fp32 for consistency with the non-empty case
        zero_norm = torch.tensor(0.0, dtype=torch.float32)
        warnings.warn(
            f"Called clip_grad_norm_() on rank {dist.get_rank()} with empty "
            f"parameters -- returning zero norm on CPU in fp32"
        )
        return zero_norm

    for i, param in enumerate(parameters):
        if not isinstance(param, DTensor):
            raise ValueError(
                f"Distributed clip_grad_norm_() requires DTensors but got {type(param)}"
            )
        if (
            len(placements := param._spec.placements) != 1
            or not placements[0].is_shard()
        ):
            raise NotImplementedError(
                f"Only support 1D Shard placements but got {placements}"
            )
        if i == 0:
            mesh, device = (param._spec.mesh, param.device)
        else:
            if (curr_mesh := param._spec.mesh) != mesh:
                raise NotImplementedError(
                    f"Only supports parameters the same mesh but got {curr_mesh} and {mesh}"
                )
            if (curr_device := param.device) != device:
                raise NotImplementedError(
                    f"Only supports parameters the same device but got {curr_device} and {device}"
                )
    local_grads: List[torch.Tensor] = [
        cast(DTensor, param.grad)._local_tensor
        for param in parameters
        if param.grad is not None
    ]
    local_sharded_norm = _get_grad_norm(local_grads, norm_type, device)
    shard_dim = 0  # hard code since we assume all 1D `Shard` placements
    process_group = cast(dist.ProcessGroup, mesh.get_dim_groups(shard_dim))
    if norm_type == math.inf:
        total_norm = local_sharded_norm
        dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=process_group)
    else:
        # We perform redundant computation of x^1/p (before all-reduce) and x^p
        # (after all-reduce) to take advantage of the fused vector norm kernel
        total_norm = local_sharded_norm**norm_type
        dist.all_reduce(total_norm, group=process_group)
        total_norm = total_norm ** (1.0 / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for grad in local_grads:
        grad.mul_(clip_coef_clamped.to(grad.dtype))
    return total_norm
