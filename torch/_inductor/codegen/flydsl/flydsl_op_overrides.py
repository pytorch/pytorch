# mypy: allow-untyped-defs
"""Map Inductor pointwise ops onto FlyDSL source text for flex score mods.

Runs during lowering, so it must not import flydsl.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch
from torch.utils._ordered_set import OrderedSet

from ..common import CSEVariable, OpOverrides


if TYPE_CHECKING:
    from collections.abc import Sequence

    import sympy


RUNTIME_ALIAS = "fdu"
RUNTIME_MODULE = "torch._inductor.codegen.flydsl.flydsl_mod_runtime"

# No AMDGPU lowering and no expansion in the runtime shim.
UNSUPPORTED_OPS = frozenset(OrderedSet(["erfinv", "lgamma", "digamma"]))


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
    OrderedSet(
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
_FLOAT_RESULT = frozenset(FlyDSLOpOverrides._UNARY) - OrderedSet(
    [
        "neg",
        "abs",
        "relu",
        "logical_not",
        "bitwise_not",
    ]
) | OrderedSet(["truediv", "pow", "atan2"])


def _make_call(name: str):
    @staticmethod  # type: ignore[misc]
    def fn(*args):
        return _call(name, *args)

    fn.__name__ = name  # type: ignore[attr-defined]
    return fn


def _make_unsupported(name: str):
    @staticmethod  # type: ignore[misc]
    def fn(*args, **kwargs):
        raise NotImplementedError(
            f"`{name}` does not lower to AMDGPU through FlyDSL and has no expansion "
            f"in {RUNTIME_MODULE}"
        )

    fn.__name__ = name  # type: ignore[attr-defined]
    return fn


def _install_ops() -> None:
    for name in (*FlyDSLOpOverrides._UNARY, *FlyDSLOpOverrides._BINARY):
        setattr(FlyDSLOpOverrides, name, _make_call(name))
    for name in UNSUPPORTED_OPS:
        setattr(FlyDSLOpOverrides, name, _make_unsupported(name))


_install_ops()
