# Copyright (c) Meta Platforms, Inc. and affiliates
from collections.abc import Iterator
from contextlib import contextmanager

from torch.distributed.tensor._api import DTensor
from torch.distributed.tensor.experimental._attention import (
    context_parallel as _context_parallel_impl,
)
from torch.distributed.tensor.experimental._func_map import local_map as _local_map_impl
from torch.distributed.tensor.experimental._register_sharding import (
    register_sharding as _register_sharding_impl,
)


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


def context_parallel(*args, **kwargs):
    """
    Wrapper for :func:`torch.distributed.tensor.experimental._attention.context_parallel`
    to expose it in the experimental namespace.
    """
    return _context_parallel_impl(*args, **kwargs)


def local_map(*args, **kwargs):
    """
    Wrapper for :func:`torch.distributed.tensor.experimental.local_map`
    to expose it in the experimental namespace.
    """
    return _local_map_impl(*args, **kwargs)


def register_sharding(*args, **kwargs):
    """
    Wrapper for :func:`torch.distributed.tensor.experimental.register_sharding`
    to expose it in the experimental namespace.
    """
    return _register_sharding_impl(*args, **kwargs)
