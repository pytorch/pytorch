from collections.abc import Callable

import torch
from torch.types import Device as _device_t


def _get_device_index(device: _device_t, optional: bool = False) -> int:
    return torch._C._accelerator__utils_getDeviceIndex(device, optional)


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
