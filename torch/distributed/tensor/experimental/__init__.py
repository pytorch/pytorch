"""
torch.distributed.tensor.experimental

Experimental APIs for distributed tensor parallelism.

Note: See https://github.com/pytorch/pytorch/issues/171905
"""
# Copyright (c) Meta Platforms, Inc. and affiliates
from collections.abc import Iterator
from contextlib import contextmanager

import warnings

# These APIs are experimental and subject to change
warnings.warn(
    "torch.distributed.tensor.experimental APIs are unstable and subject to change. "
    "See https://github.com/pytorch/pytorch/issues/171905",
    UserWarning,
    stacklevel=2,
)


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
    try:
        DTensor._op_dispatcher._allow_implicit_replication = True
        yield
    finally:
        DTensor._op_dispatcher._allow_implicit_replication = False


# Set namespace for exposed private names
# NOTE: These __module__ assignments are a workaround to make type checkers
# correctly identify the origin of these classes.
# See: https://github.com/pytorch/pytorch/issues/171905
# TODO: Migrate to TypeAliasType (PEP 613) for better type checker support
context_parallel.__module__ = "torch.distributed.tensor.experimental"
implicit_replication.__module__ = "torch.distributed.tensor.experimental"
local_map.__module__ = "torch.distributed.tensor.experimental"
register_sharding.__module__ = "torch.distributed.tensor.experimental"
