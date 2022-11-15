# Copyright (c) Meta Platforms, Inc. and affiliates
import functools
import operator

import torch
from typing import List, Union, Sequence, Iterable
from torch.distributed._tensor.api import DTensor


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def unwrap_single_placement(e):
    if not isinstance(e, DTensor):
        return None
    assert len(e.placements) == 1, "more than one placement!"
    return e.placements[0]


# convenient wrapper to register custom operator impls
# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def register_impl(func):
    # pyre-fixme[53]: Captured variable `func` is not annotated.
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def wrapper(impl):
        DTensor._custom_dispatch_ops[func] = impl
        return impl

    return wrapper


# convenient wrapper to register sharding propagation rules
# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def register_prop_rule(func):
    # pyre-fixme[53]: Captured variable `func` is not annotated.
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def wrapper(impl):
        DTensor._op_to_rules[func] = impl
        return impl

    return wrapper


def as_list(
    x: Union[List[object], object]
    # pyre-fixme[11]: Annotation `immutable_list` is not defined as a type.
) -> Union[List[object], torch.fx.immutable_collections.immutable_list]:
    # During tracing, `aten.sum.dim_IntList` uses `immutable_list` for its args,
    # which is an object but treated as a list by the tracer. Therefore, keep
    # `immutable_list` intact here as well.
    if type(x) is list or isinstance(
        x, torch.fx.immutable_collections.immutable_list
    ):
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
