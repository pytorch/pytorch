# Copyright (c) Meta Platforms, Inc. and affiliates
import os
from collections.abc import Iterator
from contextlib import contextmanager

from torch.distributed.tensor._api import DTensor
from torch.distributed.tensor.experimental._attention import context_parallel
from torch.distributed.tensor.experimental._func_map import local_map
from torch.distributed.tensor.experimental._register_sharding import register_sharding


__all__ = ["context_parallel", "implicit_replication", "local_map", "register_sharding"]


@contextmanager
def implicit_replication() -> Iterator[None]:
    """
    This context manager allows :class:`DTensor` to implicitly treat all non-DTensors (``torch.Tensor``)
    in the program be replicate :class:`DTensor` s during the operator computation.

    .. warning:: This might possible lead to incorrect results if ``torch.Tensor`` s are not replicated
        in practice, please use it at your discretion.
    """
    # Store the original state to restore it safely afterward
    original_state = getattr(DTensor._op_dispatcher, "_allow_implicit_replication", False)
    
    try:
        # Enable implicit replication while the context manager is active
        DTensor._op_dispatcher._allow_implicit_replication = True
        yield
    finally:
        # Restore to True if the environment variable is set, otherwise restore to original state
        if os.environ.get("TORCH_DTENSOR_ALLOW_IMPLICIT_REPLICATION") == "1":
            DTensor._op_dispatcher._allow_implicit_replication = True
        else:
            DTensor._op_dispatcher._allow_implicit_replication = original_state


# Set namespace for exposed private names
context_parallel.__module__ = "torch.distributed.tensor.experimental"
implicit_replication.__module__ = "torch.distributed.tensor.experimental"
local_map.__module__ = "torch.distributed.tensor.experimental"
register_sharding.__module__ = "torch.distributed.tensor.experimental"
