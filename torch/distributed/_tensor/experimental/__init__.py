# Copyright (c) Meta Platforms, Inc. and affiliates
from contextlib import contextmanager

from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.experimental.dmap import dmap

__all__ = ["dmap"]


@contextmanager
def implicit_replication():
    try:
        DTensor._op_dispatcher._allow_implicit_replication = True
        yield
    finally:
        DTensor._op_dispatcher._allow_implicit_replication = False
