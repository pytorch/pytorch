import dataclasses
import itertools
from typing import Any, Callable, Optional, TYPE_CHECKING

import sympy

import torch
from torch._inductor import config
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


def prologue_preserves_zero_mask(prologue: "SchedulerNode") -> bool:
    """
    Does this prologue preserve zero masks
    """
    preserves_zeros = PreservesZeros()
    with V.set_ops_handler(preserves_zeros):
        prologue._body(*prologue.get_ranges())

    store_preserves_zeros = preserves_zeros.store_preserves_zeros
    assert isinstance(store_preserves_zeros, bool)

    return store_preserves_zeros


@dataclasses.dataclass
class DTypeContainer:
    dtype: torch.dtype
    is_scalar: bool = False


class RecordLowPrecisionOps:
    def __init__(self) -> None:
        self.low_precision_numeric_op = False
        self.dtype_prop = DtypePropagationOpsHandler()
        self.non_numeric_ops = (
            "to_dtype",
            "constant",
        )

    @staticmethod
    def load(name: str, index: sympy.Expr) -> DTypeContainer:
        return DTypeContainer(V.get_ops_handler().dtype_prop.load(name, index))

    @staticmethod
    def store(
        name: str, index: sympy.Expr, value: TypedExpr, mode: "StoreMode" = None
    ) -> None:
        pass

    @staticmethod
    def indirect_indexing(*args: Any, **kwargs: Any) -> sympy.Expr:
        return sympy.S.Zero

    def __getattr__(self, name: str) -> Callable[..., Any]:
        def low_prec_float(dtype: torch.dtype) -> bool:
            return dtype.is_floating_point and dtype.itemsize < 4

        def inner(*args: Any, **kwargs: Any) -> DTypeContainer:
            out_dtype = getattr(self.dtype_prop, name)(*args, **kwargs)
            out = DTypeContainer(out_dtype, is_scalar=(name == "constant"))
            if name == "constant":
                out = DTypeContainer(torch.float, is_scalar=True)

            uses_low_prec = any(
                isinstance(dtype_cont, DTypeContainer)
                and low_prec_float(dtype_cont.dtype)
                for dtype_cont in itertools.chain((out,), args, kwargs.values())
            )

            if uses_low_prec and name not in self.non_numeric_ops:
                self.low_precision_numeric_op = True

            return out

        return inner


def can_codegen_without_upcasts(
    prologue: "SchedulerNode",
) -> bool:
    """
    Can this prologue be run without `upcast_to_fp32` while preserving numerics.

    This is only true if the node only contains dtype conversions, indexing, and other non-arithmetic operators.
    """

    low_prec_analysis = RecordLowPrecisionOps()

    # Need to turn off upcasting to do analysis of whether we can turn it off
    with config.patch("triton.codegen_upcast_to_fp32", False), V.set_ops_handler(
        low_prec_analysis
    ):
        prologue._body(*prologue.get_ranges())

    return not low_prec_analysis.low_precision_numeric_op
