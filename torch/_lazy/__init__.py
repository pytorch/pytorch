import pickle
from collections.abc import Sequence
from types import ModuleType
from typing import cast, TypeVar

import torch
import torch._C._lazy
from torch.serialization import DEFAULT_PROTOCOL
from torch.types import FileLike
from torch.utils._pytree import tree_flatten, tree_unflatten

from .closure import add_step_closure, run_step_closures


_T = TypeVar("_T")


def mark_step(device: str = "", wait: bool = False) -> None:
    """Triggers a mark step, which amounts to
    - collecting a group of 'live' lazy tensors to index into the compilation cache
      (lowering/compiling their IR graphs if not cached)
    - kicking off execution of the compiled function
    - (optionally, wait=True) waiting for cpu-side execution to complete (does not sync the accelerator)
    """
    # TODO(whc) expand this to include backend hooks and align with XLA backend needs
    torch._C._lazy._mark_step(device, [], wait=wait)

    run_step_closures()


def wait_device_ops(devices: list[str] | None = None) -> None:
    """Waits for all the async operations on the given devices to complete.
    Args:
      devices (string..., optional): The devices whose async ops need to be waited
        for. If empty, all the local devices will be waited for.
    """
    if devices is None:
        devices = []
    torch._C._lazy._wait_device_ops(devices=devices)


def sync_multi(tensors: Sequence[torch.Tensor], devices: list[str]) -> None:
    """
    Sync the list of lazy tensors so there IR get lowered for the activate backend
    and the compiled computation graph get cached.
    """
    torch._C._lazy._sync_multi(list(tensors), devices)


def get_tensor_id(tensor: torch.Tensor) -> int:
    """Return a unique id of the lazy tensor maintained by LTC"""
    return torch._C._lazy._get_tensor_id(tensor)


def to_cpu(tensors: _T, devices: list[str] | None = None) -> _T:
    devices = devices or ["lazy"]

    flattened, spec = tree_flatten(tensors)
    sync_multi(flattened, devices)
    return cast(_T, tree_unflatten([t.to("cpu") for t in flattened], spec))


def save(
    tensors: object,
    f: FileLike,
    pickle_module: ModuleType = pickle,
    pickle_protocol: int = DEFAULT_PROTOCOL,
    _use_new_zipfile_serialization: bool = True,
    _disable_byteorder_record: bool = False,
) -> None:
    torch.save(
        to_cpu(tensors),
        f,
        pickle_module=pickle_module,
        pickle_protocol=pickle_protocol,
        _use_new_zipfile_serialization=_use_new_zipfile_serialization,
        _disable_byteorder_record=_disable_byteorder_record,
    )
