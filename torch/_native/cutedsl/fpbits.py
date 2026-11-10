# CuTeDSL float helpers that need the sign bit itself.
#
# copysign is the whole reason this exists. Two of its inputs defeat arithmetic:
#   * -0.0 compares EQUAL to 0.0, so neither `y < 0` nor `y == 0` distinguishes them
#     (`1/y < 0` catches that one case, and nothing else);
#   * a NaN's sign is invisible to every comparison, since all of them are false.
# And |x| must map -0.0 to +0.0, which `x if x >= 0 else -x` does NOT do: -0.0 >= 0.0 is
# true, so it keeps the negative zero, and the later negation then flips BOTH branches --
# measured as every element's sign inverted in test_copysign's +-0.0/+-inf/nan case.
#
# cute.math.copysign says all of this in one instruction, but raises "Copysign is not
# supported on CTK 12.9" here, and there is no nvvm.copysign to fall back on. So the bits
# are masked directly through arith.bitcast: clear x's sign bit, contribute y's. That is
# exact for every input including both NaN signs, and the bitcasts are free (same register,
# no instruction).
#
# The two masked halves are combined with `+`, NOT `|`, and that is load-bearing. The masks
# are disjoint (bit 31 against bits 0..30), so the two are arithmetically identical -- but
# the `|` spelling is the textbook copysign idiom, and something downstream recognises and
# rewrites it, after which a NaN magnitude can come back with the wrong sign.
# Addition avoids that rewrite while preserving the disjoint sign and magnitude bits.

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Float32, Int32
from cutlass._mlir.dialects import arith
from cutlass._mlir.extras import types as T


@cute.jit
def signbit(x, dtype: cutlass.Constexpr = None):
    """True when x's SIGN BIT is set. Unlike `x < 0` this sees -0.0 (and a NaN's sign).

    pow needs it: C says pow(-0.0, -1) is -inf, so "is the base negative" has to include a
    negative zero, which every comparison reports as non-negative.
    """
    dt = type(x) if dtype is None else dtype
    if const_expr(dt is Float32):
        return Int32(arith.bitcast(T.i32(), x.ir_value())) < Int32(0)
    z = dt(0.0)
    return (x < z) or ((x == z) and (dt(1.0) / x < z))


@cute.jit
def copysign(x, y, dtype: cutlass.Constexpr = None):
    """|x| carrying y's sign bit: torch.copysign / C copysign semantics."""
    # Undecorated resolution, as in minmax._dt: this is trace-time Python.
    dt = type(x) if dtype is None else dtype
    if const_expr(dt is Float32):
        xi = Int32(arith.bitcast(T.i32(), x.ir_value()))
        yi = Int32(arith.bitcast(T.i32(), y.ir_value()))
        r = (xi & Int32(0x7FFFFFFF)) + (yi & Int32(-0x80000000))
        return Float32(arith.bitcast(T.f32(), r.ir_value()))
    # No other compute dtype reaches copysign (its promotion is INT_TO_FLOAT), but keep a
    # correct answer rather than a silent wrong one: absf fixes -0.0, and the comparison
    # covers every sign an integer value can have.
    a = dt(cute.math.absf(x))  # wrapped: absf yields a raw ArithValue
    z = dt(0.0)
    return -a if ((y < z) or ((y == z) and (dt(1.0) / y < z))) else a
