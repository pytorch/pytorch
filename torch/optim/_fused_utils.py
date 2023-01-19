from typing import cast, Dict, Optional

import torch
from torch import Tensor
from .optimizer import Optimizer


__all__ = []


# NOTE(crcrpar): Almost the same as `_MultiDeviceReplicator` defined in
# torch/cuda/amp/grad_scaler.py except for the key being str only for torch script.
class _MultiDeviceReplicator:
    main_tensor: Tensor
    _per_device_tensors: Dict[str, Tensor]

    def __init__(self, main_tensor: Tensor) -> None:
        self.main_tensor = main_tensor
        self._per_device_tensors = {str(main_tensor.device): main_tensor}

    def get(self, device: str):
        if device in self._per_device_tensors:
            return self._per_device_tensors[device]
        tensor = self.main_tensor.to(device=device, non_blocking=True, copy=True)
        self._per_device_tensors[device] = tensor
        return tensor


def _get_fp16AMP_params(
    *,
    optimizer: Optimizer,
    grad_scaler: Optional[torch.cuda.amp.GradScaler] = None,
    device: torch.device,
) -> Optional[_MultiDeviceReplicator]:
    if grad_scaler is None:
        return None
    found_inf_dict = grad_scaler._check_inf_per_device(optimizer)
    # Combines found_inf tensors from all devices. As in GradScaler.update(),
    # tensors are combined on the scale's device, which is an arbitrary but
    # reasonable choice that avoids new context creation.
    found_infs = [f.to(device, non_blocking=True) for f in found_inf_dict.values()]
    assert len(found_infs) > 0, "No inf checks were recorded in _check_inf_per_device."
    with torch.no_grad():
        found_inf_combined = cast(torch.Tensor, sum(found_infs))
    return _MultiDeviceReplicator(found_inf_combined)
