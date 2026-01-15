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
    match device:
        case int():
            return device

        case None:
            if optional:
                return torch._C._accelerator_hooks_get_current_device()
            raise ValueError(
                "Expected a torch.device with a specified index or an integer, but got: None"
            )

        case str():
            return _get_device_index(
                torch.device(device), optional=optional, allow_cpu=allow_cpu
            )

        case torch.device():
            if allow_cpu:
                if device.type not in ["mtia", "cpu"]:
                    raise ValueError(
                        f"Expected a mtia or cpu device, but got: {device}"
                    )
            elif device.type != "mtia":
                raise ValueError(f"Expected a mtia device, but got: {device}")

            if device.type == "cpu":
                return -1
            if device.index is not None:
                return device.index

            # Device is torch.device('mtia') without index
            if optional:
                return torch._C._accelerator_hooks_get_current_device()
            raise ValueError(
                f"Expected a torch.device with a specified index or an integer, but got: {device}"
            )

        case _:
            raise TypeError(
                f"Expected a torch.device, int, str, or None, but got: {type(device).__name__}"
            )
