import typing
import functools
from typing import Any, cast, List, Optional, Union
from typing_extensions import deprecated

import torch
from torch import Tensor
from torch.utils._foreach_utils import (
    _device_has_foreach_support,
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)
from torch.nn.utils.clip_grad import _tensor_or_tensors, _no_grad

__all__ = [
    "scale_grad_",
]

@_no_grad
def scale_grad_(
    parameters: _tensor_or_tensors,
    scaler: torch.Tensor,
    foreach: Optional[bool] = None,
) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    _scale_grad_(parameters, scaler, foreach)


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
    # grouped_grads: dict[
    #     tuple[torch.device, torch.dtype], tuple[list[list[Tensor]], list[int]]
    # ] = _group_tensors_by_device_and_dtype(
    #     [grads]
    # )

    grouped_grads = {(grads[0].device, None): ([grads], None)}

    for (device, _), ([device_grads], _) in grouped_grads.items():
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

class ScaleGradAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        params: List[Tensor],
        scaler: Tensor
    ) -> None:
        ctx.scaler = scaler

    @staticmethod
    def backward(ctx: Any, grads: List[Tensor]):
        # TODO: Implement inverse_groupping
        # grouped_grads: dict[
        #     tuple[torch.device, torch.dtype], tuple[list[list[Tensor]], list[int]]
        # ] = _group_tensors_by_device_and_dtype(
        #     [grads]
        # )

        device = grads[0].device
        scaled_grads = torch._foreach_mul(grads, scaler.to(device))
        return scaled_grads, None

def scale_grad_autograd_apply(params: List[Tensor], scaler: Tensor):
    return ScaleGradAutograd.apply(params, scaler)
