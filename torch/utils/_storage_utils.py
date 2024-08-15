import importlib
from functools import lru_cache
from typing import Any, Tuple, Union

import torch
from torch.utils._python_dispatch import is_traceable_wrapper_subclass


@lru_cache
def is_torch_tpu_available(check_device: bool = True) -> bool:
    """
    Checks if `torch_xla` is installed and potentially if a TPU is in the environment
    Taken from
    https://github.com/huggingface/transformers/blob/1ecf5f7c982d761b4daaa96719d162c324187c64/src/transformers/utils/import_utils.py#L463.
    """
    if importlib.util.find_spec("torch_xla") is not None:
        if check_device:
            # We need to check if `xla_device` can be found, will raise a RuntimeError if not
            try:
                import torch_xla.core.xla_model as xm

                _ = xm.xla_device()
                return True
            except RuntimeError:
                return False
        return True
    return False


def get_storage_id(tensor: torch.Tensor) -> Union[int, Tuple[Any, ...]]:
    """Returns a unique id for plain tensor
    or a (potentially nested) Tuple of unique id for the flattened Tensor
    if the input is a wrapper tensor subclass Tensor
    """
    if tensor.device.type == "xla" and is_torch_tpu_available():
        # NOTE: xla tensors dont have storage
        # use some other unique id to distinguish.
        # this is a XLA tensor, it must be created using torch_xla's
        # device. So the following import is safe:
        import torch_xla

        unique_id = torch_xla._XLAC._xla_get_tensor_id(tensor)
    elif is_traceable_wrapper_subclass(tensor):
        attrs, _ = tensor.__tensor_flatten__()
        unique_id = tuple(get_storage_id(getattr(tensor, attr)) for attr in attrs)
    else:
        unique_id = id(tensor.untyped_storage())

    return unique_id


def get_storage_size(tensor: torch.Tensor) -> int:
    """Get the storage size for the tensor in number of bytes
    for wrapper tensor subclass Tensors, we'll get the sum of the storage size for all tensor attributes
    """
    if is_traceable_wrapper_subclass(tensor):
        attrs, _ = tensor.__tensor_flatten__()
        return sum(get_storage_size(getattr(tensor, attr)) for attr in attrs)
    return tensor.untyped_storage().nbytes()
