# mypy: allow-untyped-decorators
from collections import defaultdict
from typing import Optional

import torch
from torch import Tensor
from torch.nn.utils.clip_grad import _no_grad, _tensor_or_tensors
from torch.utils._foreach_utils import _device_has_foreach_support, _has_foreach_support


__all__ = [
    "scale_grad_",
]


@_no_grad
def scale_grad_(
    parameters: _tensor_or_tensors,
    scaler: torch.Tensor,
    foreach: Optional[bool] = None,
) -> None:
    r"""Scale gradients of iterable parameters.

    This function is equivalent to :func:`torch.mul_` applied to each parameter.
    Gradients are modified in-place, multiplying by specified scaler.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients scaled
        scaler (Tensor): multiplier to scale gradients
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``
    Returns:
        None
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)
    _scale_grad_(parameters, scaler, foreach)


def _group_tensors_by_device_and_dtype(
    tensors: list[torch.Tensor],
) -> dict[tuple[torch.device, torch.dtype], list[Tensor]]:
    ret = defaultdict(list)
    for i, tensor in enumerate(tensors):
        ret[(tensor.device, tensor.dtype)].append(tensor)

    return ret


@_no_grad
def _scale_grad_(
    parameters: _tensor_or_tensors,
    scaler: torch.Tensor,
    foreach: Optional[bool] = None,
) -> None:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return
    grouped_grads = _group_tensors_by_device_and_dtype(grads)

    for (device, _), device_grads in grouped_grads.items():
        if (foreach is None and _has_foreach_support(device_grads, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            torch._foreach_mul_(device_grads, scaler.to(device))
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        else:
            scaler_device = scaler.to(device)
            for g in device_grads:
                g.mul_(scaler_device)
