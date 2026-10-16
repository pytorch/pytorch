# NaN-propagating max / min in ONE instruction. torch's max/min family propagates NaN, and
# the comparison form costs four; PTX has had `max.NaN.f32` since sm_80 and NVVM exposes the
# flag, but cute.arch.fmax/fmin do not pass it, so this reaches the dialect op directly.
#
# FLOAT32 only: PTX has no NaN-propagating f64 max and integers have no NaN, so those keep
# the comparison form. On a tie PTX returns +0 for max(+0, -0) where the comparison form
# returns its first operand -- unobservable, since +-0 compare equal.

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Float32
from cutlass._mlir.dialects import nvvm


@cute.jit
def fmax_nan(a, b, dtype: cutlass.Constexpr):
    """max(a, b), NaN-propagating, for an accumulator of `dtype`."""
    if const_expr(dtype is Float32):
        return Float32(nvvm.fmax(a.ir_value(), b.ir_value(), nan=True))
    if const_expr(dtype.is_integer):
        return max(a, b)
    return b if ((b > a) or (b != b)) else a


@cute.jit
def fmin_nan(a, b, dtype: cutlass.Constexpr):
    """min(a, b), NaN-propagating, for an accumulator of `dtype`."""
    if const_expr(dtype is Float32):
        return Float32(nvvm.fmin(a.ir_value(), b.ir_value(), nan=True))
    if const_expr(dtype.is_integer):
        return min(a, b)
    return b if ((b < a) or (b != b)) else a
