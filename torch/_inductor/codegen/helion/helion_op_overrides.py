from __future__ import annotations

import math
from typing import cast, TYPE_CHECKING

import torch

from ...utils import upcast_compute_type
from ...virtualized import V
from ..common import CSEVariable, OpOverrides, PythonPrinter


if TYPE_CHECKING:
    from collections.abc import Callable

    import sympy

    from .helion_kernel import HelionKernel


def literal(value: bool | float | int) -> str:
    if isinstance(value, float) and not math.isfinite(value):
        return f"float({repr(value)!r})"
    return repr(value)


class HelionPrinter(PythonPrinter):
    """Prints Min/Max over tile index tensors as torch.minimum/maximum."""

    def _print_min_max(self, expr: sympy.Expr, name: str) -> str:
        values = []
        for arg in expr.args:
            value = self.doprint(arg, simplify=False)
            if not arg.free_symbols:
                dtype = torch.int64 if arg.is_integer else torch.float64
                value = f"hl.full([], {value}, dtype={dtype})"
            values.append(value)
        result = values[0]
        for value in values[1:]:
            result = f"torch.{name}({result}, {value})"
        return result

    def _print_Min(self, expr: sympy.Expr) -> str:
        return self._print_min_max(expr, "minimum")

    def _print_Max(self, expr: sympy.Expr) -> str:
        return self._print_min_max(expr, "maximum")


class HelionOverrides(OpOverrides):
    """Inductor ops as Helion device code, i.e. torch ops on tiles."""

    @staticmethod
    def constant(value: bool | float | int, dtype: torch.dtype) -> str:
        # A 0-d tensor, since torch.minimum/maximum take no Python scalar.
        return f"hl.full([], {literal(value)}, dtype={upcast_compute_type(dtype)})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def to_dtype(
        x: CSEVariable,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = True,
    ) -> CSEVariable | str:
        if use_compute_types:
            dtype = upcast_compute_type(dtype)
        if x.dtype == dtype:
            return x
        return f"{x}.to({dtype})"

    @staticmethod
    def index_expr(expr: sympy.Expr, dtype: torch.dtype) -> CSEVariable:
        return cast("HelionKernel", V.kernel).index_expr(expr, dtype)

    @staticmethod
    def value_expr(expr: sympy.Expr, dtype: torch.dtype) -> CSEVariable:
        return cast("HelionKernel", V.kernel).index_expr(expr, dtype)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def masked(
        mask: CSEVariable, body: Callable[[], CSEVariable], other: bool | float | int
    ) -> CSEVariable:
        return cast("HelionKernel", V.kernel).masked(mask, body, other)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def where(cond: str, a: str, b: str) -> str:
        return f"torch.where({cond}, {a}, {b})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def fma(x: str, y: str, z: str) -> str:
        return f"torch.addcmul({z}, {x}, {y})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def minimum(a: str, b: str) -> str:
        return f"torch.minimum({a}, {b})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def maximum(a: str, b: str) -> str:
        return f"torch.maximum({a}, {b})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def logical_and(a: str, b: str) -> str:
        return f"torch.logical_and({a}, {b})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def logical_or(a: str, b: str) -> str:
        return f"torch.logical_or({a}, {b})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def logical_not(x: str) -> str:
        return f"torch.logical_not({x})"

    @staticmethod
    def relu(x: str) -> str:
        return f"torch.relu({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def abs(x: str) -> str:
        return f"torch.abs({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def exp(x: str) -> str:
        return f"torch.exp({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def exp2(x: str) -> str:
        return f"torch.exp2({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def expm1(x: str) -> str:
        return f"torch.expm1({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def log(x: str) -> str:
        return f"torch.log({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def log1p(x: str) -> str:
        return f"torch.log1p({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def log2(x: str) -> str:
        return f"torch.log2({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def sqrt(x: str) -> str:
        return f"torch.sqrt({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def rsqrt(x: str) -> str:
        return f"torch.rsqrt({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def sin(x: str) -> str:
        return f"torch.sin({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def cos(x: str) -> str:
        return f"torch.cos({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def tanh(x: str) -> str:
        return f"torch.tanh({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def erf(x: str) -> str:
        return f"torch.erf({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def sigmoid(x: str) -> str:
        return f"torch.sigmoid({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def floor(x: str) -> str:
        return f"torch.floor({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def ceil(x: str) -> str:
        return f"torch.ceil({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def reciprocal(x: str) -> str:
        return f"torch.reciprocal({x})"
