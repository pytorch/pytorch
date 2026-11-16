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
    """Inductor ops as Helion device code, i.e. torch ops on tiles. The ops that
    are a single torch function are added from TORCH_FUNCTIONS below."""

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
    # pyrefly: ignore [bad-override]
    def to_dtype_bitcast(
        x: CSEVariable, dtype: torch.dtype, src_dtype: torch.dtype
    ) -> str:
        # A bf16 value is computed in FP32 (see to_dtype): cast it back before
        # reinterpreting its 16 bits.
        value = f"{x}" if x.dtype == src_dtype else f"{x}.to({src_dtype})"
        result = f"{value}.view({dtype})"
        compute_dtype = upcast_compute_type(dtype)
        return result if compute_dtype == dtype else f"{result}.to({compute_dtype})"

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
    def identity(x: CSEVariable) -> CSEVariable:
        return x

    @staticmethod
    # pyrefly: ignore [bad-override]
    def fma(x: str, y: str, z: str) -> str:
        return f"torch.addcmul({z}, {x}, {y})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def truncdiv(a: str, b: str) -> str:
        return f'torch.div({a}, {b}, rounding_mode="trunc")'

    @staticmethod
    # pyrefly: ignore [bad-override]
    def floordiv(a: str, b: str) -> str:
        return f'torch.div({a}, {b}, rounding_mode="floor")'


# Inductor ops that are one torch function of the same arguments, e.g.
# ops.sinh(x) -> torch.sinh(x) and ops.atan2(y, x) -> torch.atan2(y, x).
# fmt: off
TORCH_FUNCTIONS = {
    **{name: f"torch.{name}" for name in [
        "where", "minimum", "maximum", "logical_and", "logical_or", "logical_not",
        "logical_xor", "relu", "abs", "sign", "signbit", "isinf", "isnan", "reciprocal",
        "sqrt", "rsqrt", "exp", "exp2", "expm1", "log", "log1p", "log2", "log10",
        "sigmoid", "erf", "erfc", "erfinv", "lgamma", "i0", "sin", "cos", "tan", "sinh",
        "cosh", "tanh", "asin", "acos", "atan", "asinh", "acosh", "atanh", "atan2",
        "floor", "ceil", "trunc", "round", "copysign", "hypot", "nextafter", "fmod",
        "remainder", "bitwise_left_shift", "bitwise_right_shift",
    ]},
    **{name: f"torch.special.{name}" for name in [
        "bessel_j0", "bessel_j1", "bessel_y0", "bessel_y1", "erfcx", "i1",
        "modified_bessel_i0", "modified_bessel_i1",
    ]},
    # C-style, with the sign of a; a % b on torch tensors takes the sign of b.
    "mod": "torch.fmod",
}
# fmt: on


def _torch_call(function: str) -> Callable[..., str]:
    return lambda *args: f"{function}({', '.join(map(str, args))})"


for _name, _function in TORCH_FUNCTIONS.items():
    setattr(HelionOverrides, _name, staticmethod(_torch_call(_function)))
