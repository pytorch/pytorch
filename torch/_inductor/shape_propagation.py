import functools
from collections.abc import Sequence
from typing import Callable, Optional, Protocol, Union

import sympy

import torch

from .virtualized import OpsValue


ShapeType = Optional[Sequence[Union[int, str]]]


class ShapeVar(Protocol):
    @property
    def shape(self) -> ShapeType: ...


ShapeArg = Union[ShapeVar, torch.types.Number, str, OpsValue]

# Inputs need to be cacheable (e.g., not a CSEVar) in order for the cache to be effective
# So first decompose CSEVars -> tuple before calling this


@functools.lru_cache(None)
def get_broadcasted_shape(a: ShapeType, b: ShapeType) -> ShapeType:
    if a is None:
        return b
    if b is None:
        return a
    assert isinstance(a, Sequence)
    assert isinstance(b, Sequence)
    if len(a) > len(b):
        return get_broadcasted_shape(a, (*b, *a[len(b) :]))
    elif len(a) < len(b):
        b, a = a, b
        return get_broadcasted_shape(a, (*b, *a[len(b) :]))
    else:

        def _get_broadcasted_dim(
            d1: Union[int, str], d2: Union[int, str]
        ) -> Union[int, str]:
            if str(d1) == "1":
                return d2
            elif str(d2) == "1":
                return d1
            assert str(d1) == str(d2)
            return d1

        return [_get_broadcasted_dim(d1, d2) for d1, d2 in zip(a, b)]


def broadcast_shapes_for_args(
    args: Sequence[ShapeArg],
) -> ShapeType:
    result_shape = None

    for arg in args:
        if shape := getattr(arg, "shape", None):
            result_shape = get_broadcasted_shape(result_shape, shape)

    return result_shape


class ShapePropagationOpsHandler:
    """
    Propagate shape from args to output
    """

    @staticmethod
    def constant(value: torch.types.Number, dtype: torch.dtype) -> ShapeType:
        return []

    @staticmethod
    def store_reduction(name: str, index: int, value: ShapeArg) -> None:
        return None

    @staticmethod
    def reduction(
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: str,
        value: Union[ShapeArg, tuple[ShapeArg, ...]],
    ) -> Union[ShapeType, tuple[ShapeType, ...]]:
        raise NotImplementedError

    @staticmethod
    def store(
        name: str, index: int, value: ShapeArg, mode: Optional[str] = None
    ) -> None:
        return None

    @staticmethod
    def index_expr(expr: sympy.Expr, dtype: torch.dtype) -> ShapeType:
        return []

    @staticmethod
    def indirect_indexing(
        var: ShapeArg,
        size: Union[sympy.Expr, int],
        check: bool = True,
        wrap_neg: bool = True,
    ) -> None:
        return None

    def __getattr__(self, name: str) -> Callable[..., ShapeType]:
        return lambda *args, **kwargs: broadcast_shapes_for_args(args)
