# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
from contextlib import contextmanager

from torch.distributed.tensor.api import DTensor
from torch.distributed.tensor.experimental.func_map import local_map
from torch.distributed.tensor.experimental.register_sharding import register_sharding


__all__ = ["implicit_replication", "local_map", "register_sharding"]


@contextmanager
def implicit_replication():
    try:
        DTensor._op_dispatcher._allow_implicit_replication = True
        yield
    finally:
        DTensor._op_dispatcher._allow_implicit_replication = False
