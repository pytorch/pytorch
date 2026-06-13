# Copyright (c) Meta Platforms, Inc. and affiliates
from collections.abc import Iterator
from contextlib import contextmanager

import torch.distributed.config as dist_config
from torch.distributed.tensor._api import DTensor
from torch.distributed.tensor.experimental._attention import context_parallel
from torch.distributed.tensor.experimental._func_map import local_map
from torch.distributed.tensor.experimental._register_sharding import register_sharding


__all__ = [
    "context_parallel",
    "implicit_replication",
    "local_map",
    "register_sharding",
    "use_symmetric_memory",
]


@contextmanager
def implicit_replication() -> Iterator[None]:
    """
    This context manager allows :class:`DTensor` to implicitly treat all non-DTensors (``torch.Tensor``)
    in the program be replicate :class:`DTensor` s during the operator computation.

    .. warning:: This might possible lead to incorrect results if ``torch.Tensor`` s are not replicated
        in practice, please use it at your discretion.
    """
    try:
        DTensor._op_dispatcher._allow_implicit_replication = True
        yield
    finally:
        DTensor._op_dispatcher._allow_implicit_replication = False


@contextmanager
def use_symmetric_memory(enabled: bool = True) -> Iterator[None]:
    """
    Context manager that makes eligible DTensor construction APIs allocate local
    CUDA tensors from SymmetricMemory.
    """
    with dist_config.patch(dtensor_use_symmetric_memory=enabled):
        yield


# Set namespace for exposed private names
context_parallel.__module__ = "torch.distributed.tensor.experimental"
implicit_replication.__module__ = "torch.distributed.tensor.experimental"
local_map.__module__ = "torch.distributed.tensor.experimental"
register_sharding.__module__ = "torch.distributed.tensor.experimental"
use_symmetric_memory.__module__ = "torch.distributed.tensor.experimental"
