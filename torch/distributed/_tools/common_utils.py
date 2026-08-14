import functools
import warnings

import torch
from torch._custom_class_base import CustomClassBase
from torch.utils._python_dispatch import (
    _disable_current_modes,
    is_traceable_wrapper_subclass,
)


def get_untyped_storages(t: torch.Tensor) -> set[torch.UntypedStorage]:
    """
    Recursively extracts untyped storages from a tensor or its subclasses.

    Args:
        t (torch.Tensor): The tensor to extract storages from.

    Returns:
        Set[torch.UntypedStorage]: A set of untyped storages.
    """
    unflattened_tensors = [t]
    flattened_tensor_storages = set()
    while len(unflattened_tensors) > 0:
        obj = unflattened_tensors.pop()
        if is_traceable_wrapper_subclass(obj):
            attrs, _ = obj.__tensor_flatten__()
            for attr in attrs:
                match getattr(obj, attr):
                    case torch.Tensor() as v:
                        unflattened_tensors.append(v)
                    case CustomClassBase():
                        pass
                    case unexpected:
                        raise AssertionError(
                            f"expected Tensor or CustomClassBase, got {type(unexpected)}"
                        )
        else:
            if not hasattr(obj, "untyped_storage"):
                warnings.warn(
                    f"Expected a tensor or a traceable wrapper-subclass of tensor, but got {type(obj)}",
                    category=UserWarning,
                    stacklevel=2,
                )
            else:
                flattened_tensor_storages.add(obj.untyped_storage())
    return flattened_tensor_storages


@functools.cache
def get_allocation_granularity(device_type: str) -> int:
    """
    Returns the number of bytes that ``device_type``'s allocator rounds an allocation up to.

    The value is measured from the allocator's own statistics rather than hardcoded
    per backend, so any accelerator reporting ``requested_bytes``/``allocated_bytes``
    is covered, including out-of-tree PrivateUse1 backends.

    Args:
        device_type (str): The device type to query, e.g. ``"cuda"``.

    Returns:
        int: The allocation granularity in bytes. 1 if the allocator returns exactly
        the requested size, does not report the above statistics, or is bypassed.
    """
    acc = torch.accelerator.current_accelerator()
    if acc is None or acc.type != device_type:
        return 1

    def stat(key: str) -> int:
        return torch.accelerator.memory_stats().get(key, 0)

    # Modes must be off: this runs under ``__torch_dispatch__``, and a fake mode
    # would allocate nothing.
    with _disable_current_modes():
        before_req = stat("requested_bytes.all.current")
        before_alloc = stat("allocated_bytes.all.current")
        probe = torch.empty(1, dtype=torch.uint8, device=device_type)
        req = stat("requested_bytes.all.current") - before_req
        alloc = stat("allocated_bytes.all.current") - before_alloc
        del probe
    return alloc if req == 1 and alloc > 0 else 1
