import itertools
from typing import Any, Callable, Optional, TYPE_CHECKING

import sympy

import torch
from torch._inductor.dtype_propagation import DtypePropagationOpsHandler
from torch._inductor.index_propagation import SymPyOps, TypedExpr

from .virtualized import StoreMode, V


if TYPE_CHECKING:
    from torch._inductor.scheduler import SchedulerNode


def construct_symbol(count: int, dtype: torch.dtype) -> sympy.Symbol:
    return sympy.Symbol(f"unknown_{count}")


class PreservesZeros(SymPyOps):
    """
    For prologue kernels where the loads are masked, does the final store of this kernel preserve
    the zeros.
    """

    def __init__(self) -> None:
        self.count = itertools.count(0)
        self.store_preserves_zeros: Optional[bool] = None
        self.dtype_prop = DtypePropagationOpsHandler()

    @staticmethod
    def load(name: str, index: sympy.Expr) -> TypedExpr:
        # In prologue fusion, all loads get broadcasted
        dtype = V.get_ops_handler().dtype_prop.load(name, index)
        return TypedExpr(
            sympy.Float(0) if dtype.is_floating_point else sympy.Integer(0), dtype
        )

    @staticmethod
    def store(
        name: str, index: sympy.Expr, value: TypedExpr, mode: "StoreMode" = None
    ) -> None:
        self = V.get_ops_handler()
        assert isinstance(self, PreservesZeros)
        # should only have a single store in prologue
        assert self.store_preserves_zeros is None
        self.store_preserves_zeros = value.is_constant() and value.expr == 0

    @staticmethod
    def indirect_indexing(*args: Any, **kwargs: Any) -> sympy.Expr:
        self = V.get_ops_handler()
        return construct_symbol(next(self.count), torch.int32)

    def __getattr__(self, name: str) -> Callable[..., Any]:
        from torch._inductor.codegen.common import OpDecompositions

        def inner(*args: Any, **kwargs: Any) -> TypedExpr:
            if hasattr(OpDecompositions, name):
                return getattr(OpDecompositions, name)(*args, **kwargs).value

            nonlocal self
            dtype = getattr(self.dtype_prop, name)(*args, **kwargs)
            return TypedExpr(construct_symbol(next(self.count), dtype), dtype)

        return inner


def prologue_preserves_zero_mask(node: "SchedulerNode") -> bool:
    """
    Does this prologue preserve zero masks
    """
    preserves_zeros = PreservesZeros()
    with V.set_ops_handler(preserves_zeros):
        node._body(*node.get_ranges())

    store_preserves_zeros = preserves_zeros.store_preserves_zeros
    assert isinstance(store_preserves_zeros, bool)
    return store_preserves_zeros
