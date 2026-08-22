# Copyright (c) Meta Platforms, Inc. and affiliates
from collections.abc import Iterator
from contextlib import contextmanager

from torch.distributed.tensor._api import DTensor
from torch.distributed.tensor.experimental._attention import context_parallel
from torch.distributed.tensor.experimental._func_map import local_map
from torch.distributed.tensor.experimental._register_sharding import register_sharding
from typing_extensions import TypeAliasType


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


# Use TypeAliasType for re-exported names so type checkers infer the correct __module__
# without mutating it at runtime.
context_parallel: TypeAliasType = TypeAliasType(
    "context_parallel",
    _attention.context_parallel,
)

implicit_replication: TypeAliasType = TypeAliasType(
    "implicit_replication",
    implicit_replication,
)

local_map: TypeAliasType = TypeAliasType(
    "local_map",
    _func_map.local_map,
)

register_sharding: TypeAliasType = TypeAliasType(
    "register_sharding",
    _register_sharding.register_sharding,
)
