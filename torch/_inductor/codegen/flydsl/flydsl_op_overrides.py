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

    _UNARY = (
        "neg",
        "abs",
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
        "relu",
        "logical_not",
        "bitwise_not",
    )

    _BINARY = (
        "add",
        "sub",
        "mul",
        "truediv",
        "floordiv",
        "mod",
        "remainder",
        "maximum",
        "minimum",
        "pow",
        "lt",
        "le",
        "gt",
        "ge",
        "eq",
        "ne",
        "logical_and",
        "logical_or",
        "logical_xor",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "bitwise_left_shift",
        "bitwise_right_shift",
        "atan2",
    )

    @staticmethod
    def neg(*args):
        return _call("neg", *args)

    @staticmethod
    def abs(*args):
        return _call("abs", *args)

    @staticmethod
    def reciprocal(*args):
        return _call("reciprocal", *args)

    @staticmethod
    def sqrt(*args):
        return _call("sqrt", *args)

    @staticmethod
    def rsqrt(*args):
        return _call("rsqrt", *args)

    @staticmethod
    def exp(*args):
        return _call("exp", *args)

    @staticmethod
    def exp2(*args):
        return _call("exp2", *args)

    @staticmethod
    def expm1(*args):
        return _call("expm1", *args)

    @staticmethod
    def log(*args):
        return _call("log", *args)

    @staticmethod
    def log2(*args):
        return _call("log2", *args)

    @staticmethod
    def log10(*args):
        return _call("log10", *args)

    @staticmethod
    def log1p(*args):
        return _call("log1p", *args)

    @staticmethod
    def floor(*args):
        return _call("floor", *args)

    @staticmethod
    def ceil(*args):
        return _call("ceil", *args)

    @staticmethod
    def trunc(*args):
        return _call("trunc", *args)

    @staticmethod
    def round(*args):
        return _call("round", *args)

    @staticmethod
    def sin(*args):
        return _call("sin", *args)

    @staticmethod
    def cos(*args):
        return _call("cos", *args)

    @staticmethod
    def tanh(*args):
        return _call("tanh", *args)

    @staticmethod
    def sigmoid(*args):
        return _call("sigmoid", *args)

    @staticmethod
    def sinh(*args):
        return _call("sinh", *args)

    @staticmethod
    def cosh(*args):
        return _call("cosh", *args)

    @staticmethod
    def asin(*args):
        return _call("asin", *args)

    @staticmethod
    def acos(*args):
        return _call("acos", *args)

    @staticmethod
    def atan(*args):
        return _call("atan", *args)

    @staticmethod
    def asinh(*args):
        return _call("asinh", *args)

    @staticmethod
    def acosh(*args):
        return _call("acosh", *args)

    @staticmethod
    def atanh(*args):
        return _call("atanh", *args)

    @staticmethod
    def erf(*args):
        return _call("erf", *args)

    @staticmethod
    def erfc(*args):
        return _call("erfc", *args)

    @staticmethod
    def relu(*args):
        return _call("relu", *args)

    @staticmethod
    def logical_not(*args):
        return _call("logical_not", *args)

    @staticmethod
    def bitwise_not(*args):
        return _call("bitwise_not", *args)

    @staticmethod
    def add(*args):
        return _call("add", *args)

    @staticmethod
    def sub(*args):
        return _call("sub", *args)

    @staticmethod
    def mul(*args):
        return _call("mul", *args)

    @staticmethod
    def truediv(*args):
        return _call("truediv", *args)

    @staticmethod
    def floordiv(*args):
        return _call("floordiv", *args)

    @staticmethod
    def mod(*args):
        return _call("mod", *args)

    @staticmethod
    def remainder(*args):
        return _call("remainder", *args)

    @staticmethod
    def maximum(*args):
        return _call("maximum", *args)

    @staticmethod
    def minimum(*args):
        return _call("minimum", *args)

    @staticmethod
    def pow(*args):
        return _call("pow", *args)

    @staticmethod
    def lt(*args):
        return _call("lt", *args)

    @staticmethod
    def le(*args):
        return _call("le", *args)

    @staticmethod
    def gt(*args):
        return _call("gt", *args)

    @staticmethod
    def ge(*args):
        return _call("ge", *args)

    @staticmethod
    def eq(*args):
        return _call("eq", *args)

    @staticmethod
    def ne(*args):
        return _call("ne", *args)

    @staticmethod
    def logical_and(*args):
        return _call("logical_and", *args)

    @staticmethod
    def logical_or(*args):
        return _call("logical_or", *args)

    @staticmethod
    def logical_xor(*args):
        return _call("logical_xor", *args)

    @staticmethod
    def bitwise_and(*args):
        return _call("bitwise_and", *args)

    @staticmethod
    def bitwise_or(*args):
        return _call("bitwise_or", *args)

    @staticmethod
    def bitwise_xor(*args):
        return _call("bitwise_xor", *args)

    @staticmethod
    def bitwise_left_shift(*args):
        return _call("bitwise_left_shift", *args)

    @staticmethod
    def bitwise_right_shift(*args):
        return _call("bitwise_right_shift", *args)

    @staticmethod
    def atan2(*args):
        return _call("atan2", *args)

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


# Ops whose result is f32: every unary op but these, and truediv, pow, atan2.
_FLOAT_RESULT = (
    frozenset(FlyDSLOpOverrides._UNARY)
    - frozenset(["neg", "abs", "relu", "logical_not", "bitwise_not"])
    | frozenset(["truediv", "pow", "atan2"])
)
