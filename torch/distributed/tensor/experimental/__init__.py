# Copyright (c) Meta Platforms, Inc. and affiliates
import functools
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from torch.distributed.tensor._api import DTensor
from torch.distributed.tensor.experimental import _attention
from torch.distributed.tensor.experimental import _func_map
from torch.distributed.tensor.experimental import _register_sharding


__all__ = ["context_parallel", "implicit_replication", "local_map", "register_sharding"]


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


_ASSIGNMENTS = tuple(a for a in functools.WRAPPER_ASSIGNMENTS if a != "__module__")

@functools.wraps(_attention.context_parallel, assigned=_ASSIGNMENTS)
def context_parallel(*args: Any, **kwargs: Any) -> Any:
    return _attention.context_parallel(*args, **kwargs)

@functools.wraps(_func_map.local_map, assigned=_ASSIGNMENTS)
def local_map(*args: Any, **kwargs: Any) -> Any:
    return _func_map.local_map(*args, **kwargs)

@functools.wraps(_register_sharding.register_sharding, assigned=_ASSIGNMENTS)
def register_sharding(*args: Any, **kwargs: Any) -> Any:
    return _register_sharding.register_sharding(*args, **kwargs)
