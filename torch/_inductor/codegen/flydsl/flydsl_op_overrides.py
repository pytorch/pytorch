# mypy: allow-untyped-defs
"""Map Inductor pointwise ops onto FlyDSL source text for flex score mods.

Runs during lowering, so it must not import flydsl.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch

from ..common import CSEVariable, OpOverrides


if TYPE_CHECKING:
    from collections.abc import Sequence

    import sympy


RUNTIME_ALIAS = "fdu"
RUNTIME_MODULE = "torch._inductor.codegen.flydsl.flydsl_mod_runtime"


class FlyDSLCSEVariable(CSEVariable):
    """A FlyDSL value; ``index_expr`` names the coordinate it was loaded from."""

    def __init__(self, name, bounds, dtype=None, shape=None) -> None:
        super().__init__(name, bounds, dtype)
        self.index_expr: sympy.Expr | None = None
        self.shape = shape

    def __repr__(self) -> str:
        return str(self.name)


def _arg(x: Any) -> str:
    if isinstance(x, CSEVariable):
        return str(x.name)
    if isinstance(x, bool):
        return f"{RUNTIME_ALIAS}.const_bool({x})"
    if isinstance(x, int):
        return f"{RUNTIME_ALIAS}.const_i32({x})"
    if isinstance(x, float):
        return f"{RUNTIME_ALIAS}.const_f32(float({str(x)!r}))"
    return str(x)


_BOOL_RESULT = frozenset(
    [
        "lt",
        "le",
        "gt",
        "ge",
        "eq",
        "ne",
        "logical_and",
        "logical_or",
        "logical_xor",
        "logical_not",
    ]
)

# Unary ops other than neg, abs, relu, logical_not and bitwise_not, plus
# truediv, pow and atan2.
_FLOAT_RESULT = frozenset(
    [
        "reciprocal",
        "sqrt",
        "rsqrt",
        "exp",
        "exp2",
        "expm1",
        "log",
        "log2",
        "log10",
        "log1p",
        "floor",
        "ceil",
        "trunc",
        "round",
        "sin",
        "cos",
        "tanh",
        "sigmoid",
        "sinh",
        "cosh",
        "asin",
        "acos",
        "atan",
        "asinh",
        "acosh",
        "atanh",
        "erf",
        "erfc",
        "truediv",
        "pow",
        "atan2",
    ]
)


def _result_dtype(fn: str, args: Sequence[Any]) -> torch.dtype:
    if fn in _BOOL_RESULT:
        return torch.bool
    if fn in _FLOAT_RESULT:
        return torch.float32
    for a in args:
        if isinstance(a, CSEVariable):
            if a.dtype is not None and a.dtype.is_floating_point:
                return torch.float32
        elif isinstance(a, float):
            return torch.float32
    return torch.int32


def _call(fn: str, *args: Any) -> Any:
    from ...virtualized import V

    expr = f"{RUNTIME_ALIAS}.{fn}({', '.join(_arg(a) for a in args)})"
    kernel = getattr(V, "kernel", None)
    if kernel is None or not hasattr(kernel, "cse"):
        return expr
    from ..common import ValueRanges

    return kernel.cse.generate(
        kernel.body, expr, dtype=_result_dtype(fn, args), bounds=ValueRanges.unknown()
    )


class FlyDSLOpOverrides(OpOverrides):
    """Emit FlyDSL source for the pointwise ops a score_mod can contain."""

    @staticmethod
    def neg(a: Any) -> Any:
        return _call("neg", a)

    @staticmethod
    def abs(x0: Any) -> Any:
        return _call("abs", x0)

    @staticmethod
    def reciprocal(x: Any) -> Any:
        return _call("reciprocal", x)

    @staticmethod
    def sqrt(x0: Any) -> Any:
        return _call("sqrt", x0)

    @staticmethod
    def rsqrt(x0: Any) -> Any:
        return _call("rsqrt", x0)

    @staticmethod
    def exp(x0: Any) -> Any:
        return _call("exp", x0)

    @staticmethod
    def exp2(x: Any) -> Any:
        return _call("exp2", x)

    @staticmethod
    def expm1(x: Any) -> Any:
        return _call("expm1", x)

    @staticmethod
    def log(x0: Any) -> Any:
        return _call("log", x0)

    @staticmethod
    def log2(x: Any) -> Any:
        return _call("log2", x)

    @staticmethod
    def log10(x: Any) -> Any:
        return _call("log10", x)

    @staticmethod
    def log1p(x: Any) -> Any:
        return _call("log1p", x)

    @staticmethod
    def floor(x0: Any) -> Any:
        return _call("floor", x0)

    @staticmethod
    def ceil(x0: Any) -> Any:
        return _call("ceil", x0)

    @staticmethod
    def trunc(x0: Any) -> Any:
        return _call("trunc", x0)

    @staticmethod
    def round(x0: Any) -> Any:
        return _call("round", x0)

    @staticmethod
    def sin(x0: Any) -> Any:
        return _call("sin", x0)

    @staticmethod
    def cos(x0: Any) -> Any:
        return _call("cos", x0)

    @staticmethod
    def tanh(x0: Any) -> Any:
        return _call("tanh", x0)

    @staticmethod
    def sigmoid(x: Any) -> Any:
        return _call("sigmoid", x)

    @staticmethod
    def sinh(x0: Any) -> Any:
        return _call("sinh", x0)

    @staticmethod
    def cosh(x0: Any) -> Any:
        return _call("cosh", x0)

    @staticmethod
    def asin(x0: Any) -> Any:
        return _call("asin", x0)

    @staticmethod
    def acos(x0: Any) -> Any:
        return _call("acos", x0)

    @staticmethod
    def atan(x0: Any) -> Any:
        return _call("atan", x0)

    @staticmethod
    def asinh(x0: Any) -> Any:
        return _call("asinh", x0)

    @staticmethod
    def acosh(x0: Any) -> Any:
        return _call("acosh", x0)

    @staticmethod
    def atanh(x0: Any) -> Any:
        return _call("atanh", x0)

    @staticmethod
    def erf(x0: Any) -> Any:
        return _call("erf", x0)

    @staticmethod
    def erfc(x: Any) -> Any:
        return _call("erfc", x)

    @staticmethod
    def relu(x: Any) -> Any:
        return _call("relu", x)

    @staticmethod
    def logical_not(a: Any) -> Any:
        return _call("logical_not", a)

    @staticmethod
    def bitwise_not(x: Any) -> Any:
        return _call("bitwise_not", x)

    @staticmethod
    def add(a: Any, b: Any) -> Any:
        return _call("add", a, b)

    @staticmethod
    def sub(a: Any, b: Any) -> Any:
        return _call("sub", a, b)

    @staticmethod
    def mul(a: Any, b: Any) -> Any:
        return _call("mul", a, b)

    @staticmethod
    def truediv(a: Any, b: Any) -> Any:
        return _call("truediv", a, b)

    @staticmethod
    def floordiv(a: Any, b: Any) -> Any:
        return _call("floordiv", a, b)

    @staticmethod
    def mod(a: Any, b: Any) -> Any:
        return _call("mod", a, b)

    @staticmethod
    def remainder(a: Any, b: Any) -> Any:
        return _call("remainder", a, b)

    @staticmethod
    def maximum(x0: Any, x1: Any) -> Any:
        return _call("maximum", x0, x1)

    @staticmethod
    def minimum(x0: Any, x1: Any) -> Any:
        return _call("minimum", x0, x1)

    @staticmethod
    def pow(a: Any, b: Any) -> Any:
        return _call("pow", a, b)

    @staticmethod
    def lt(a: Any, b: Any) -> Any:
        return _call("lt", a, b)

    @staticmethod
    def le(a: Any, b: Any) -> Any:
        return _call("le", a, b)

    @staticmethod
    def gt(a: Any, b: Any) -> Any:
        return _call("gt", a, b)

    @staticmethod
    def ge(a: Any, b: Any) -> Any:
        return _call("ge", a, b)

    @staticmethod
    def eq(a: Any, b: Any) -> Any:
        return _call("eq", a, b)

    @staticmethod
    def ne(a: Any, b: Any) -> Any:
        return _call("ne", a, b)

    @staticmethod
    def logical_and(x0: Any, x1: Any) -> Any:
        return _call("logical_and", x0, x1)

    @staticmethod
    def logical_or(x0: Any, x1: Any) -> Any:
        return _call("logical_or", x0, x1)

    @staticmethod
    def logical_xor(x0: Any, x1: Any) -> Any:
        return _call("logical_xor", x0, x1)

    @staticmethod
    def bitwise_and(x: Any, y: Any) -> Any:
        return _call("bitwise_and", x, y)

    @staticmethod
    def bitwise_or(x: Any, y: Any) -> Any:
        return _call("bitwise_or", x, y)

    @staticmethod
    def bitwise_xor(x: Any, y: Any) -> Any:
        return _call("bitwise_xor", x, y)

    @staticmethod
    def bitwise_left_shift(x: Any, y: Any) -> Any:
        return _call("bitwise_left_shift", x, y)

    @staticmethod
    def bitwise_right_shift(x: Any, y: Any) -> Any:
        return _call("bitwise_right_shift", x, y)

    @staticmethod
    def atan2(x0: Any, x1: Any) -> Any:
        return _call("atan2", x0, x1)

    @staticmethod
    def erfinv(*args, **kwargs):
        raise NotImplementedError(
            f"`erfinv` does not lower to AMDGPU through FlyDSL and has no expansion "
            f"in {RUNTIME_MODULE}"
        )

    @staticmethod
    def lgamma(*args, **kwargs):
        raise NotImplementedError(
            f"`lgamma` does not lower to AMDGPU through FlyDSL and has no expansion "
            f"in {RUNTIME_MODULE}"
        )

    @staticmethod
    def digamma(*args, **kwargs):
        raise NotImplementedError(
            f"`digamma` does not lower to AMDGPU through FlyDSL and has no expansion "
            f"in {RUNTIME_MODULE}"
        )

    @staticmethod
    def constant(value: bool | float | int, dtype: torch.dtype) -> str:
        if dtype == torch.bool:
            return f"{RUNTIME_ALIAS}.const_bool({bool(value)})"
        if dtype.is_floating_point:
            return f"{RUNTIME_ALIAS}.const_f32(float({str(float(value))!r}))"
        return f"{RUNTIME_ALIAS}.const_i32({int(value)})"

    @staticmethod
    def index_expr(expr: sympy.Expr, dtype: torch.dtype) -> str:
        from ...virtualized import V

        rendered = V.kernel.kexpr(V.kernel.rename_indexing(expr))
        return _call("to_i32", rendered)

    @staticmethod
    def to_dtype(
        x: Any,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = True,
    ) -> str:
        # Mods compute in f32/i32/i1, like the Triton flex kernels.
        if dtype == torch.bool:
            return _call("to_bool", x)
        if dtype.is_floating_point:
            return _call("to_f32", x)
        return _call("to_i32", x)

    @staticmethod
    def to_dtype_bitcast(x: Any, dtype: torch.dtype, src_dtype: torch.dtype) -> str:
        raise NotImplementedError("bitcast is not supported in FlyDSL flex mods")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def where(cond: Any, a: Any, b: Any) -> str:
        return _call("where", cond, a, b)

    @staticmethod
    def masked(mask: Any, body: Any, other: Any) -> str:
        raise NotImplementedError("masked loads are not supported in FlyDSL flex mods")

    @staticmethod
    def indirect_indexing(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "indirect indexing is not supported in FlyDSL flex mods"
        )
