from typing import Optional

import torch
from torch.types import Device as _device_t


def _get_device_index(device: _device_t, optional: bool = False) -> int:
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        device = torch.device(device)
    device_index: Optional[int] = None
    if isinstance(device, torch.device):
        acc = torch.accelerator.current_accelerator()
        if acc is None:
            raise RuntimeError("Accelerator expected")
        if acc.type != device.type:
            raise ValueError(
                f"{device.type} doesn't match the current accelerator {torch.accelerator.current_accelerator()}."
            )
        device_index = device.index
    if device_index is None:
        if not optional:
            raise ValueError(
                f"Expected a torch.device with a specified index or an integer, but got:{device}"
            )
        return torch.accelerator.current_device_index()
    return device_index
