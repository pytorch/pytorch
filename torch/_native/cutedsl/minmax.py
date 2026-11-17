# Min/max with native NaN and signed-zero behavior.

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Float32
from cutlass._mlir.dialects import nvvm

from .types import NumericType, Scalar, ScalarResult


def _dt(a: Scalar, dtype: NumericType | None) -> NumericType:
    # Resolve types in trace-time Python, outside @cute.jit.
    return type(a) if dtype is None else dtype


@cute.jit
def _max_zero(a: Scalar, b: Scalar, result: ScalarResult) -> ScalarResult:
    return a + b if a == 0 and b == 0 else result


@cute.jit
def _min_zero(a: Scalar, b: Scalar, result: ScalarResult) -> ScalarResult:
    return -((-a) + (-b)) if a == 0 and b == 0 else result


@cute.jit
def fmax_nan(a: Scalar, b: Scalar, dtype: cutlass.Constexpr = None) -> ScalarResult:
    r"""Maximum with NaN propagation."""
    dt = _dt(a, dtype)
    if const_expr(dt is Float32):
        return Float32(nvvm.fmax(a.ir_value(), b.ir_value(), nan=True))
    if const_expr(getattr(dt, "is_integer", False)):
        return max(a, b)
    return _max_zero(a, b, b if ((b > a) or (b != b)) else a)


@cute.jit
def fmin_nan(a: Scalar, b: Scalar, dtype: cutlass.Constexpr = None) -> ScalarResult:
    r"""Minimum with NaN propagation."""
    dt = _dt(a, dtype)
    if const_expr(dt is Float32):
        return Float32(nvvm.fmin(a.ir_value(), b.ir_value(), nan=True))
    if const_expr(getattr(dt, "is_integer", False)):
        return min(a, b)
    return _min_zero(a, b, b if ((b < a) or (b != b)) else a)


@cute.jit
def fmax_suppress_nan(
    a: Scalar, b: Scalar, dtype: cutlass.Constexpr = None
) -> ScalarResult:
    r"""Maximum with NaN suppression."""
    dt = _dt(a, dtype)
    if const_expr(dt is Float32):
        return Float32(nvvm.fmax(a.ir_value(), b.ir_value()))
    if const_expr(getattr(dt, "is_integer", False)):
        return max(a, b)
    return _max_zero(a, b, b if ((b > a) or (a != a)) else a)


@cute.jit
def fmin_suppress_nan(
    a: Scalar, b: Scalar, dtype: cutlass.Constexpr = None
) -> ScalarResult:
    r"""Minimum with NaN suppression."""
    dt = _dt(a, dtype)
    if const_expr(dt is Float32):
        return Float32(nvvm.fmin(a.ir_value(), b.ir_value()))
    if const_expr(getattr(dt, "is_integer", False)):
        return min(a, b)
    return _min_zero(a, b, b if ((b < a) or (a != a)) else a)
