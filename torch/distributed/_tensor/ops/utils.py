# Copyright (c) Meta Platforms, Inc. and affiliates
import functools
import operator
from typing import cast, Iterable, List, Sequence, Union

import torch
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import DTensorSpec, Shard


# convenient wrapper to register sharding propagation rules
# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def register_prop_rule(op):
    # pyre-fixme[53]: Captured variable `func` is not annotated.
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def wrapper(impl):
        overloads = op if isinstance(op, list) else [op]
        for overload in overloads:
            DTensor._propagator.register_sharding_prop_rule(overload, impl)
        return impl

    return wrapper


def register_op_strategy(op):
    # pyre-fixme[53]: Captured variable `func` is not annotated.
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def wrapper(impl):
        overloads = op if isinstance(op, list) else [op]
        for overload in overloads:
            DTensor._propagator.register_op_strategy(overload, impl)
        return impl

    return wrapper


def as_list(
    x: Union[List[object], object]
    # pyre-fixme[11]: Annotation `immutable_list` is not defined as a type.
) -> Union[List[object], torch.fx.immutable_collections.immutable_list]:
    # During tracing, `aten.sum.dim_IntList` uses `immutable_list` for its args,
    # which is an object but treated as a list by the tracer. Therefore, keep
    # `immutable_list` intact here as well.
    if type(x) is list or isinstance(x, torch.fx.immutable_collections.immutable_list):
        return x
    else:
        return [x]


def normalize_dim(dim: int, ndim: int) -> int:
    return dim if dim >= 0 else dim + ndim


def normalize_dims(dims: Union[int, Sequence[int]], ndim: int) -> Sequence[int]:
    """
    normalize a dim or a sequence of dims, so that they
    are all positive.
    """
    if isinstance(dims, int):
        dims = (normalize_dim(dims, ndim),)
    elif isinstance(dims, list):
        dims = [normalize_dim(dim, ndim) for dim in dims]
    elif isinstance(dims, tuple):
        dims = tuple([normalize_dim(dim, ndim) for dim in dims])
    return dims


def prod(xs: Iterable[int]) -> int:
    return functools.reduce(operator.mul, xs, 1)


def is_tensor_shardable(shape: Sequence[int], spec: DTensorSpec) -> bool:
    """
    Check if the shape is shardable according to the spec.
    """
    # number of shards in each tensor dimension
    shards_map = [1] * len(shape)
    for i, placement in enumerate(spec.placements):
        if placement.is_shard():
            shard_dim = cast(Shard, placement).dim
            shards_map[shard_dim] *= spec.mesh.size(i)

    for i, dim_size in enumerate(shape):
        # TODO: maybe we should determine is_shardable based on
        #       whether it's evenly sharded or not
        if dim_size < shards_map[i]:
            return False

    return True


def is_tensor_dim_sharded(spec: DTensorSpec, dim: int) -> bool:
    """Return True if tensor dim is sharded"""
    return any(p.is_shard(dim) for p in spec.placements)


def is_tensor_partial(spec: DTensorSpec) -> bool:
    """Return True if tensor is partial on the mesh"""
    return any(p.is_partial() for p in spec.placements)
