from typing import Any

import torch


def _get_device_index(
    device: Any, optional: bool = False, allow_cpu: bool = False
) -> int:
    r"""Get the device index from :attr:`device`, which can be a torch.device object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    is a MTIA device. Note that for a MTIA device without a specified index,
    i.e., ``torch.device('mtia')``, this will return the current default MTIA
    device if :attr:`optional` is ``True``. If :attr:`allow_cpu` is ``True``,
    CPU devices will be accepted and ``-1`` will be returned in this case.

    If :attr:`device` is a Python integer, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default MTIA
    device if :attr:`optional` is ``True``.
    """

    if device is None and optional:
        # If device is None (frequent), then we can can short-circuit the logic
        return torch._C._mtia_getDevice()
    if isinstance(device, int):
        return device
    if not torch.jit.is_scripting():
        if isinstance(device, torch.mtia.device):
            return device.idx
    if isinstance(device, str):
        device = torch.device(device)
    device_idx: int | None = None
    if isinstance(device, torch.device):
        if not allow_cpu and device.type == "cpu":
            raise ValueError(f"Expected a non cpu device, but got: {device}")
        if device.type not in ["mtia", "cpu"]:
            raise ValueError(f"Expected a mtia or cpu device, but got: {device}")
        device_idx = -1 if device.type == "cpu" else device.index
    if device_idx is None:
        if optional:
            device_idx = torch._C._mtia_getDevice()
        else:
            raise ValueError(
                f"Expected a torch.device with a specified index or an integer, but got: {device}"
            )
    return device_idx
