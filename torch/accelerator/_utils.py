from collections.abc import Callable

import torch
from torch.types import Device as _device_t


def _get_device_index(device: _device_t, optional: bool = False) -> int:
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        device = torch.device(device)
    device_index: int | None = None
    if isinstance(device, torch.device):
        acc = torch.accelerator.current_accelerator()
        if acc is None:
            raise RuntimeError("Accelerator expected")
        if acc.type != device.type:
            raise ValueError(
                f"{device.type} doesn't match the current accelerator {acc}."
            )
        device_index = device.index
    if device_index is None:
        if not optional:
            raise ValueError(
                f"Expected a torch.device with a specified index or an integer, but got:{device}"
            )
        return torch.accelerator.current_device_index()
    return device_index


def _lazy_call(callable: Callable[[], None], **kwargs) -> None:
    r"""Queue a callable for execution after accelerator initialization, or run it immediately if already initialized.

    If the backend supports lazy initialization (i.e., exposes ``_lazy_call``),
    the callable is deferred until the runtime is ready. Otherwise, it executes
    immediately. See :ref:`lazy-initialization-and-fork-safety-note`.

    Args:
        callable (Callable[[], None]): The function to be called.
        **kwargs: Additional keyword arguments forwarded to the backend's ``_lazy_call``
            (e.g., ``seed=True``, ``seed_all=True``).
    """
    acc = torch.accelerator.current_accelerator()
    if acc is None:
        raise RuntimeError("No accelerator is available; _lazy_call requires an active accelerator.")
    device_module = torch.get_device_module(acc)
    if hasattr(device_module, "_lazy_call"):
        device_module._lazy_call(callable, **kwargs)
    else:
        callable()
