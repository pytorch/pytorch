# Floating-point bit operations used by pointwise math.

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Float32, Float64, Int16, Int32, Int64, Uint32
from cutlass._mlir.dialects import arith
from cutlass._mlir.extras import types as T

from .types import Scalar, ScalarResult


@cute.jit
def signbit(x: Scalar, dtype: cutlass.Constexpr = None) -> ScalarResult:
    r"""Match native CUDA signbit, including Half NaN behavior."""
    dt = type(x) if dtype is None else dtype
    if const_expr(dt is Float32):
        return Int32(arith.bitcast(T.i32(), x.ir_value())) < Int32(0)
    if const_expr(dt is Float64):
        return Int64(arith.bitcast(T.i64(), x.ir_value())) < Int64(0)
    if const_expr(dt in (cutlass.Float16, cutlass.BFloat16)):
        bits = arith.bitcast(T.i16(), x.ir_value())
        if const_expr(dt is cutlass.Float16):
            # The unsigned range excludes positive values and all Half NaNs.
            raw = Uint32(arith.extui(T.i32(), bits))
            return raw - Uint32(0x8000) <= Uint32(0x7C00)
        return Int16(bits) < Int16(0)
    z = dt(0.0)
    return (x < z) or ((x == z) and (dt(1.0) / x < z))


@cute.jit
def copysign(x: Scalar, y: Scalar, dtype: cutlass.Constexpr = None) -> ScalarResult:
    r"""Return x's magnitude with y's sign bit."""
    dt = type(x) if dtype is None else dtype
    if const_expr(dt is Float32):
        xi = Int32(arith.bitcast(T.i32(), x.ir_value()))
        yi = Int32(arith.bitcast(T.i32(), y.ir_value()))
        r = (xi & Int32(0x7FFFFFFF)) + (yi & Int32(-0x80000000))
        return Float32(arith.bitcast(T.f32(), r.ir_value()))
    if const_expr(dt is Float64):
        xi = Int64(arith.bitcast(T.i64(), x.ir_value()))
        yi = Int64(arith.bitcast(T.i64(), y.ir_value()))
        # Avoid large integer literals in DSL symbol names.
        sign = Int64(1) << Int64(63)
        r = (xi & ~sign) + (yi & sign)
        return Float64(arith.bitcast(T.f64(), r.ir_value()))
    # Wrap absf's raw ArithValue before applying the sign.
    a = dt(cute.math.absf(x))
    z = dt(0.0)
    return -a if ((y < z) or ((y == z) and (dt(1.0) / y < z))) else a


@cute.jit
def frexp(x: Scalar) -> tuple[ScalarResult, ScalarResult]:
    r"""Decompose IEEE floats exactly, including subnormals and signed zero."""
    dt = type(x)
    if const_expr(dt is Float64):
        integer, mantissa_bits, exponent_max, bias = Int64, 52, 2047, 1022
        bits = Int64(arith.bitcast(T.i64(), x.ir_value()))
    else:
        integer, mantissa_bits, exponent_max, bias = Int32, 23, 255, 126
        bits = Int32(arith.bitcast(T.i32(), x.ir_value()))
    shift = integer(mantissa_bits)
    mask = integer(exponent_max)
    exponent = (bits >> shift) & mask
    original = x
    special = (x == dt(0.0)) or (exponent == mask)
    adjustment = integer(0)
    if exponent == integer(0):
        x = x * dt(2.0**mantissa_bits)
        if const_expr(dt is Float64):
            bits = Int64(arith.bitcast(T.i64(), x.ir_value()))
        else:
            bits = Int32(arith.bitcast(T.i32(), x.ir_value()))
        exponent = (bits >> shift) & mask
        adjustment = integer(-mantissa_bits)
    bits = (bits & ~(mask << shift)) | (integer(bias) << shift)
    mantissa = dt(arith.bitcast(x.ir_value().type, bits.ir_value()))
    result_exp = exponent - integer(bias) + adjustment
    return original if special else mantissa, integer(0) if special else result_exp
