# mypy: allow-untyped-defs
"""Device helpers for FlyDSL flex score mods, imported by generated code as ``fdu``.

Ops promote mixed f32/i32 operands. Ops without an AMDGPU libcall are expanded over
exp2/log2/sqrt.
"""

from __future__ import annotations

import math as _math
import operator

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import math as fmath
from flydsl.expr.numeric import Float as _Float, Integer as _Integer
from flydsl.expr.utils.arith import ArithValue, current_fastmath


_LOG2E = _math.log2(_math.e)
_LN2 = 1.0 / _LOG2E
_LN10 = _math.log(10.0)
_PI = _math.pi
_PI_2 = _math.pi / 2.0


# Type coercion


def const_f32(v):
    return fx.Float32(float(v))


def const_i32(v):
    return fx.Int32(int(v))


def const_bool(v):
    one = fx.Int32(1)
    return ArithValue(one == fx.Int32(1 if v else 0))


def _is_bool(av) -> bool:
    try:
        return not av.is_float and av.type.width == 1
    except AttributeError:
        return False


def to_f32(x):
    # Integers go through fx.Int32 first: fx.Float32 reads signedness off the class,
    # and operator results come back as bare signless ArithValues.
    if isinstance(x, (bool, int, float)):
        return fx.Float32(float(x))
    if isinstance(x, (_Integer, _Float)):
        return fx.Float32(x)
    av = ArithValue(x)
    if _is_bool(av):
        return fx.Float32(av.select(const_f32(1.0), const_f32(0.0)))
    if not av.is_float:
        return fx.Float32(fx.Int32(av))
    return fx.Float32(av)


def to_i32(x):
    if isinstance(x, (bool, int, float)):
        return fx.Int32(int(x))
    if isinstance(x, (_Integer, _Float)):
        return fx.Int32(x)
    av = ArithValue(x)
    if _is_bool(av):
        return fx.Int32(av.select(const_i32(1), const_i32(0)))
    return fx.Int32(av)


def to_bool(x):
    if isinstance(x, (bool, int, float)):
        return const_bool(bool(x))
    av = ArithValue(x)
    if _is_bool(av):
        return av
    zero = const_f32(0.0) if av.is_float else const_i32(0)
    return ArithValue(av != zero)


def _is_float(x) -> bool:
    if isinstance(x, float):
        return True
    if isinstance(x, (bool, int)):
        return False
    try:
        return ArithValue(x).is_float
    except (TypeError, ValueError):
        return False


def _both_bool(a, b) -> bool:
    try:
        return _is_bool(ArithValue(a)) and _is_bool(ArithValue(b))
    except (TypeError, ValueError):
        return False


def _promote2(a, b):
    # Two i1 operands stay i1 so `&`/`|` of predicates remain selectable.
    if _both_bool(a, b):
        return ArithValue(a), ArithValue(b)
    if _is_float(a) or _is_float(b):
        return ArithValue(to_f32(a)), ArithValue(to_f32(b))
    return ArithValue(to_i32(a)), ArithValue(to_i32(b))


# Binary arithmetic


def _promoted(op):
    """``op`` on operands promoted by ``_promote2``."""

    def apply(a, b):
        x, y = _promote2(a, b)
        return ArithValue(op(x, y))

    return apply


add = _promoted(operator.add)
sub = _promoted(operator.sub)
mul = _promoted(operator.mul)
floordiv = _promoted(operator.floordiv)
mod = _promoted(operator.mod)
lt = _promoted(operator.lt)
le = _promoted(operator.le)
gt = _promoted(operator.gt)
ge = _promoted(operator.ge)
eq = _promoted(operator.eq)
ne = _promoted(operator.ne)
bitwise_and = _promoted(operator.and_)
bitwise_or = _promoted(operator.or_)
bitwise_xor = _promoted(operator.xor)
bitwise_left_shift = _promoted(operator.lshift)


def truediv(a, b):
    x, y = ArithValue(to_f32(a)), ArithValue(to_f32(b))
    return ArithValue(x / y)


def remainder(a, b):
    """Python/ATen remainder: the sign follows the divisor."""
    x, y = _promote2(a, b)
    r = ArithValue(x % y)
    zero = const_f32(0.0) if r.is_float else const_i32(0)
    wrong_sign = ArithValue(
        ArithValue(r != zero) & ArithValue((r < zero) != (y < zero))
    )
    return ArithValue(wrong_sign.select(ArithValue(r + y), r))


def maximum(a, b):
    x, y = _promote2(a, b)
    if x.is_float:
        return ArithValue(x.maximumf(y))
    return ArithValue(ArithValue(x > y).select(x, y))


def minimum(a, b):
    x, y = _promote2(a, b)
    if x.is_float:
        return ArithValue(x.minimumf(y))
    return ArithValue(ArithValue(x < y).select(x, y))


def pow(a, b):
    x, y = ArithValue(to_f32(a)), ArithValue(to_f32(b))
    return ArithValue(fmath.powf(x, y))


# Comparison and logic


def logical_and(a, b):
    return ArithValue(to_bool(a) & to_bool(b))


def logical_or(a, b):
    return ArithValue(to_bool(a) | to_bool(b))


def logical_xor(a, b):
    return ArithValue(to_bool(a) ^ to_bool(b))


def logical_not(a):
    return ArithValue(to_bool(a) == const_bool(False))


def bitwise_not(a):
    return ArithValue(~ArithValue(to_i32(a)))


def bitwise_right_shift(a, b):
    # Arithmetic shift, as for Python/ATen signed integers.
    x, y = _promote2(a, b)
    return ArithValue(ArithValue(x, signed=True) >> y)


def where(cond, a, b):
    x, y = _promote2(a, b)
    return ArithValue(to_bool(cond).select(x, y))


# Unary


def neg(x):
    return ArithValue(-ArithValue(x))


def abs(x):
    av = ArithValue(x)
    if av.is_float:
        return ArithValue(fmath.absf(av))
    return ArithValue(fmath.absi(av))


def reciprocal(x):
    return ArithValue(const_f32(1.0) / ArithValue(to_f32(x)))


def sqrt(x):
    return ArithValue(fmath.sqrt(to_f32(x)))


def rsqrt(x):
    return ArithValue(fmath.rsqrt(to_f32(x)))


def _v_exp_f32(x):
    """``v_exp_f32`` and an ``s_nop 0`` share one asm block, so no instruction
    reads the result in the slot after it: gfx950 needs that wait state, and
    LLVM counts any instruction toward it."""
    return _llvm.inline_asm(
        x.type, [x], "v_exp_f32 $0, $1\n\ts_nop 0", "=v,v", has_side_effects=False
    )


def exp2(x):
    """exp2 as AMDGPU expands it for f32 with denormals."""
    av = ArithValue(to_f32(x))
    if current_fastmath() is not None:
        return ArithValue(av.exp2())
    scaled = ArithValue(av < const_f32(-126.0))
    offset = ArithValue(scaled.select(const_f32(64.0), const_f32(0.0)))
    biased = ArithValue(av + offset)
    raw = _v_exp_f32(biased)
    power = ArithValue(scaled.select(const_i32(-64), const_i32(0)))
    return ArithValue(_llvm.intr_ldexp(fx.Float32.ir_type, raw, power))


def exp(x):
    return exp2(mul(x, const_f32(_LOG2E)))


def expm1(x):
    return sub(exp(x), const_f32(1.0))


def log2(x):
    return ArithValue(fmath.log2(to_f32(x)))


def log(x):
    return mul(log2(x), const_f32(_LN2))


def log10(x):
    return mul(log2(x), const_f32(_LN2 / _LN10))


def log1p(x):
    return log(add(x, const_f32(1.0)))


def floor(x):
    return ArithValue(fmath.floor(to_f32(x)))


def ceil(x):
    return ArithValue(fmath.ceil(to_f32(x)))


def trunc(x):
    return ArithValue(fmath.trunc(to_f32(x)))


def round(x):
    return ArithValue(fmath.roundeven(to_f32(x)))


def sin(x):
    return ArithValue(fmath.sin(to_f32(x)))


def cos(x):
    return ArithValue(fmath.cos(to_f32(x)))


def tanh(x):
    """sign(x) * (1 - 2 / (exp2(2|x| log2 e) + 1)); |x| keeps exp2 from overflowing."""
    av = ArithValue(to_f32(x))
    is_neg = ArithValue(av < const_f32(0.0))
    ax = ArithValue(is_neg.select(ArithValue(-av), av))
    # The argument is >= 0, so the result is never denormal.
    e = ArithValue(_v_exp_f32(ArithValue(ax * const_f32(2.0 * _LOG2E))))
    t = ArithValue(
        const_f32(1.0) - ArithValue(const_f32(2.0) / ArithValue(e + const_f32(1.0)))
    )
    return ArithValue(is_neg.select(ArithValue(-t), t))


def sigmoid(x):
    e = exp2(mul(neg(x), const_f32(_LOG2E)))
    return ArithValue(const_f32(1.0) / ArithValue(ArithValue(e) + const_f32(1.0)))


def _horner(z, coeffs):
    acc = const_f32(coeffs[0])
    for c in coeffs[1:]:
        acc = add(mul(acc, z), const_f32(c))
    return acc


def sinh(x):
    e = exp(abs(x))
    half = mul(sub(e, reciprocal(e)), const_f32(0.5))
    return where(lt(x, const_f32(0.0)), neg(half), half)


def cosh(x):
    e = exp(abs(x))
    return mul(add(e, reciprocal(e)), const_f32(0.5))


def asinh(x):
    ax = abs(x)
    r = log(add(ax, sqrt(add(mul(ax, ax), const_f32(1.0)))))
    return where(lt(x, const_f32(0.0)), neg(r), r)


def acosh(x):
    return log(add(x, sqrt(sub(mul(x, x), const_f32(1.0)))))


def atanh(x):
    return mul(
        log(truediv(add(const_f32(1.0), x), sub(const_f32(1.0), x))), const_f32(0.5)
    )


# Hastings' odd minimax polynomial for atan on |x| <= 1, max abs error ~1e-5.
_ATAN_COEFFS = (0.0208351, -0.0851330, 0.1801410, -0.3302995, 0.9998660)


def _atan_unit(x):
    return mul(x, _horner(mul(x, x), _ATAN_COEFFS))


def atan(x):
    ax = abs(x)
    big = gt(ax, const_f32(1.0))
    # Both branches are evaluated, so keep the reciprocal finite at x = 0.
    inner = where(big, reciprocal(maximum(ax, const_f32(1.0))), ax)
    r = _atan_unit(inner)
    r = where(big, sub(const_f32(_PI_2), r), r)
    return where(lt(x, const_f32(0.0)), neg(r), r)


def atan2(y, x):
    zero, one = const_f32(0.0), const_f32(1.0)
    ax, ay = abs(x), abs(y)
    ratio = truediv(ay, where(eq(ax, zero), one, ax))
    both_inf = logical_and(
        eq(ax, const_f32(float("inf"))), eq(ay, const_f32(float("inf")))
    )
    angle = atan(where(both_inf, one, ratio))
    angle = where(eq(ax, zero), where(eq(ay, zero), zero, const_f32(_PI_2)), angle)
    x_negative = lt(fmath.copysign(one, to_f32(x), fastmath="none"), zero)
    angle = where(x_negative, sub(const_f32(_PI), angle), angle)
    angle = where(logical_or(ne(x, x), ne(y, y)), add(x, y), angle)
    return fmath.copysign(to_f32(angle), to_f32(y), fastmath="none")


def asin(x):
    return atan2(x, sqrt(maximum(sub(const_f32(1.0), mul(x, x)), const_f32(0.0))))


def acos(x):
    return sub(const_f32(_PI_2), asin(x))


# Abramowitz & Stegun 7.1.26, max abs error 1.5e-7 on x >= 0.
_ERF_P = 0.3275911
_ERF_COEFFS = (1.061405429, -1.453152027, 1.421413741, -0.284496736, 0.254829592)


def erf(x):
    ax = abs(x)
    t = reciprocal(add(const_f32(1.0), mul(const_f32(_ERF_P), ax)))
    poly = mul(t, _horner(t, _ERF_COEFFS))
    r = sub(const_f32(1.0), mul(poly, exp(neg(mul(ax, ax)))))
    return where(lt(x, const_f32(0.0)), neg(r), r)


def erfc(x):
    # Absolute, not relative, accuracy in the right tail.
    return sub(const_f32(1.0), erf(x))


def relu(x):
    return maximum(x, const_f32(0.0))
