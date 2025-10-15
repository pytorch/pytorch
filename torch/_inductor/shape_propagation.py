import functools
from collections.abc import Sequence
from typing import Callable, Optional, Protocol, Union

import sympy

import torch

from .virtualized import OpsValue, V


BlockShapeType = Optional[Sequence[Union[int, str]]]


class ShapeVar(Protocol):
    @property
    def shape(self) -> BlockShapeType: ...


ShapeArg = Union[ShapeVar, torch.types.Number, str, OpsValue, torch.dtype]

# Inputs need to be cacheable (e.g., not a CSEVar) in order for the cache to be effective
# So first decompose CSEVars -> tuple before calling this


@functools.lru_cache(None)
def get_broadcasted_shape(a: BlockShapeType, b: BlockShapeType) -> BlockShapeType:
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


def broadcast_shapes_for_args(args: Sequence[ShapeArg]) -> BlockShapeType:
    result_shape: BlockShapeType = None

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
            from torch._inductor.loop_body import LoopBody, LoopBodyBlock

            if isinstance(arg, (LoopBodyBlock, LoopBody, OpsValue)):
                # TODO: fix me
                return None
            raise TypeError(f"Unknown type: {type(arg)}")

    return result_shape


class ShapePropagationOpsHandler:
    """
    Propagate shape from args to output
    """

    @staticmethod
    def constant(value: torch.types.Number, dtype: torch.dtype) -> BlockShapeType:
        # See implementation of constant for triton for the reason
        from torch._inductor.codegen.triton import triton_compute_type, TritonKernel

        triton_type = triton_compute_type(dtype)

        if isinstance(V.kernel, TritonKernel) and triton_type != "tl.float32":
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
    ) -> Union[BlockShapeType, tuple[BlockShapeType, ...]]:
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
    ) -> BlockShapeType:
        return value.shape

    @staticmethod
    def index_expr(expr: sympy.Expr, dtype: torch.dtype) -> BlockShapeType:
        # shape is implicitly embedded in expr.
        return None

    @staticmethod
    def load_seed(name: str, offset: int) -> BlockShapeType:
        return ()

    @staticmethod
    def indirect_indexing(
        var: ShapeArg,
        size: Union[sympy.Expr, int],
        check: bool = True,
        wrap_neg: bool = True,
    ) -> None:
        return None

    def __getattr__(self, name: str) -> Callable[..., BlockShapeType]:
        return lambda *args, **kwargs: broadcast_shapes_for_args(args)

    @staticmethod
    def device_assert_async(cond: ShapeArg, msg: str) -> None:
        return None
