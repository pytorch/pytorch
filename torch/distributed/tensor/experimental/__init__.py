# Copyright (c) Meta Platforms, Inc. and affiliates

from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from typing import ParamSpec, TypeVar
from torch.distributed.tensor._api import DTensor
from torch.distributed.tensor.experimental._attention import (
    context_parallel as _context_parallel,
)
from torch.distributed.tensor.experimental._func_map import local_map as _local_map
from torch.distributed.tensor.experimental._register_sharding import (
    register_sharding as _register_sharding,
)

__all__ = ["context_parallel", "implicit_replication", "local_map", "register_sharding"]

P = ParamSpec("P")
R = TypeVar("R")


@wraps(_context_parallel)
def context_parallel(*args: P.args, **kwargs: P.kwargs) -> R:
    return _context_parallel(*args, **kwargs)


@wraps(_local_map)
def local_map(*args: P.args, **kwargs: P.kwargs) -> R:
    return _local_map(*args, **kwargs)


@wraps(_register_sharding)
def register_sharding(*args: P.args, **kwargs: P.kwargs) -> R:
    return _register_sharding(*args, **kwargs)


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