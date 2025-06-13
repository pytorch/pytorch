import functools
from collections.abc import Sequence
from typing import Callable, Optional, Protocol, Union

import sympy

import torch

from .virtualized import OpsValue, V


ShapeType = Optional[Sequence[Union[int, str]]]


class ShapeVar(Protocol):
    @property
    def shape(self) -> ShapeType: ...


ShapeArg = Union[ShapeVar, torch.types.Number, str, OpsValue, torch.dtype]

# Inputs need to be cacheable (e.g., not a CSEVar) in order for the cache to be effective
# So first decompose CSEVars -> tuple before calling this


@functools.lru_cache(None)
def get_broadcasted_shape(a: ShapeType, b: ShapeType) -> ShapeType:
    assert isinstance(a, Sequence)
    assert isinstance(b, Sequence)
    if len(a) > len(b):
        return get_broadcasted_shape(a, (*[1] * (len(a) - len(b)), *b))
    elif len(a) < len(b):
        b, a = a, b
        return get_broadcasted_shape(a, (*[1] * (len(a) - len(b)), *b))
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

        return tuple(_get_broadcasted_dim(d1, d2) for d1, d2 in zip(a, b))


def broadcast_shapes_for_args(args: Sequence[ShapeArg]) -> ShapeType:
    result_shape: ShapeType = None

    for arg in args:
        if hasattr(arg, "shape"):
            shape = arg.shape
            if shape is None:
                return None
            elif result_shape is None:
                result_shape = tuple(shape)
            else:
                result_shape = get_broadcasted_shape(result_shape, tuple(shape))
        elif isinstance(arg, (int, float)):
            if result_shape is None:
                result_shape = ()
        elif isinstance(arg, torch.dtype):
            continue
        else:
            return None

    return result_shape


class ShapePropagationOpsHandler:
    """
    Propagate shape from args to output
    """

    @staticmethod
    def constant(value: torch.types.Number, dtype: torch.dtype) -> ShapeType:
        # See implementation of constant for triton for the reason
        from torch._inductor.codegen.triton import TritonKernel

        if isinstance(V.kernel, TritonKernel):
            ndim = V.kernel.triton_tensor_ndim()
            return tuple([1] * ndim)
        else:
            return ()

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
    def to_dtype(
        value: ShapeVar,
        dtype: torch.dtype,
        src_dtype: Optional[torch.dtype] = None,
        use_compute_types: bool = True,
    ) -> ShapeType:
        return value.shape

    @staticmethod
    def index_expr(expr: sympy.Expr, dtype: torch.dtype) -> ShapeType:
        # shape is implicitly embedded in expr.
        return None

    @staticmethod
    def load_seed(name: str, offset: int) -> ShapeType:
        return ()

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
