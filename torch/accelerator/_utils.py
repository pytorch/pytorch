from collections.abc import Callable

import torch
from torch.types import Device


def _get_optional_device_index(device: Device = None) -> int | None:
    # Return None instead of the current device index, since querying it may trigger lazy init.
    if device is None or isinstance(device, int):
        return device
    device = torch.device(device)
    acc = torch.accelerator.current_accelerator()
    if acc is None:
        raise RuntimeError("Accelerator expected")
    if acc.type != device.type:
        raise ValueError(f"{device.type} doesn't match the current accelerator {acc}.")
    return device.index


def _get_device_index(device: Device, optional: bool = False) -> int:
    device_index = _get_optional_device_index(device)
    if device_index is None:
        if not optional:
            raise ValueError(
                f"Expected a torch.device with a specified index or an integer, but got:{device}"
            )
        return torch.accelerator.current_device_index()
    return device_index


def _lazy_call(callable: Callable[[], None], **kwargs) -> None:
    r"""Defer a callable until the :ref:`accelerator<accelerators>` runtime is initialized.

    If the runtime is already initialized or the backend does not support lazy
    initialization, the callable runs immediately. If no accelerator is
    available, the callable is silently dropped.

    Args:
        callable (Callable[[], None]): The function to be called.
        **kwargs: Additional keyword arguments forwarded to the backend's ``_lazy_call``
            (e.g., ``seed=True``, ``seed_all=True``).
    """
    acc = torch.accelerator.current_accelerator()
    if acc is None:
        return
    device_module = torch.get_device_module(acc)
    if hasattr(device_module, "_lazy_call"):
        device_module._lazy_call(callable, **kwargs)
    else:
        # Backend does not support lazy initialization; run immediately.
        callable()
