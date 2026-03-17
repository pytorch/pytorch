import warnings

import torch
from torch._opaque_base import OpaqueBase
from torch.utils._python_dispatch import is_traceable_wrapper_subclass


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
                    case OpaqueBase():
                        pass
                    case unexpected:
                        raise AssertionError(
                            f"expected Tensor or OpaqueBase, got {type(unexpected)}"
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
