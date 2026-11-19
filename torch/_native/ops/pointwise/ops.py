# Typed scalar functors; narrow floating arithmetic uses FP32 intermediates.


import cutlass.cute as cute
from cutlass import (
    BFloat16,
    Boolean,
    const_expr,
    Float16,
    Float32,
    Float64,
    Int32,
    Int64,
)
from cutlass._mlir.dialects import arith

from ...cutedsl import fpbits as _fpbits, math as _math, minmax as _mm
from ...cutedsl.types import ComputeScalar, MathFn, StorageScalar
from .signature import Functor


_INV_SIX: float = 0.1666666716337204


@cute.jit
def _opmath(x: StorageScalar) -> ComputeScalar:
    if const_expr(isinstance(x, (Float16, BFloat16))):
        return Float32(x)
    return x


class _UnaryMath:
    def __init__(self, fn: MathFn) -> None:
        self.fn: MathFn = fn
        self.__name__: str = fn.__name__

    @cute.jit
    def __call__(self, x: StorageScalar) -> ComputeScalar:
        return self.fn(_opmath(x))


_exp = _UnaryMath(cute.math.exp)
_exp2 = _UnaryMath(cute.math.exp2)
_expm1 = _UnaryMath(cute.math.expm1)
_log = _UnaryMath(cute.math.log)
_log2 = _UnaryMath(cute.math.log2)
_log10 = _UnaryMath(cute.math.log10)
_log1p = _UnaryMath(cute.math.log1p)
_sqrt = _UnaryMath(cute.math.sqrt)
_rsqrt = _UnaryMath(cute.math.rsqrt)
_sin = _UnaryMath(cute.math.sin)
_cos = _UnaryMath(cute.math.cos)
_tan = _UnaryMath(cute.math.tan)
_asin = _UnaryMath(cute.math.asin)
_acos = _UnaryMath(cute.math.acos)
_atan = _UnaryMath(cute.math.atan)
_erfc = _UnaryMath(cute.math.erfc)
_cosh = _UnaryMath(cute.math.cosh)
_sinh = _UnaryMath(cute.math.sinh)
_asinh = _UnaryMath(cute.math.asinh)
_acosh = _UnaryMath(cute.math.acosh)
_atanh = _UnaryMath(cute.math.atanh)
_tanh = _UnaryMath(cute.math.tanh)
_erf = _UnaryMath(cute.math.erf)
_abs = _UnaryMath(cute.math.abs)
_round = _UnaryMath(cute.math.roundeven)
_floor = _UnaryMath(cute.math.floor)


@cute.jit
def _neg(x: StorageScalar) -> ComputeScalar:
    x = _opmath(x)
    return -x


@cute.jit
def _add(x: ComputeScalar, y: ComputeScalar, alpha: ComputeScalar) -> ComputeScalar:
    return x + alpha * y


@cute.jit
def _mul(x: ComputeScalar, y: ComputeScalar) -> ComputeScalar:
    return x * y


@cute.jit
def _sub(x: ComputeScalar, y: ComputeScalar, alpha: ComputeScalar) -> ComputeScalar:
    return x - alpha * y


@cute.jit
def _div(x: StorageScalar, y: StorageScalar) -> ComputeScalar:
    x, y = _opmath(x), _opmath(y)
    return x / y


@cute.jit
def _maximum(x: StorageScalar, y: StorageScalar) -> ComputeScalar:
    x, y = _opmath(x), _opmath(y)
    return _mm.fmax_nan(x, y)


@cute.jit
def _minimum(x: StorageScalar, y: StorageScalar) -> ComputeScalar:
    x, y = _opmath(x), _opmath(y)
    return _mm.fmin_nan(x, y)


@cute.jit
def _reciprocal(x: StorageScalar) -> ComputeScalar:
    x = _opmath(x)
    return x.__class__(1.0) / x


@cute.jit
def _atan2(x: StorageScalar, y: StorageScalar) -> ComputeScalar:
    x, y = _opmath(x), _opmath(y)
    return cute.math.atan2(x, y)


@cute.jit
def _deg2rad(x: StorageScalar) -> ComputeScalar:
    x = _opmath(x)
    return x * x.__class__(0.017453292519943295)  # pi/180


@cute.jit
def _rad2deg(x: StorageScalar) -> ComputeScalar:
    x = _opmath(x)
    return x * x.__class__(57.29577951308232)  # 180/pi


@cute.jit
def _logaddexp(x: StorageScalar, y: StorageScalar) -> ComputeScalar:
    x, y = _opmath(x), _opmath(y)
    d = x - y if x > y else y - x
    m = x if x > y else y  # noqa: FURB136 -- builtin max() does not lower in the DSL
    inf = x.__class__(float("inf"))
    same_inf = (x == y) and ((x == inf) or (x == -inf))
    return m if same_inf else m + cute.math.log1p(cute.math.exp(-d))


@cute.jit
def _xlogy(x: StorageScalar, y: StorageScalar) -> ComputeScalar:
    # NaN y takes precedence over zero x.
    x, y = _opmath(x), _opmath(y)
    z = x.__class__(0.0)
    return z if (x == z) and (y == y) else x * cute.math.log(y)


@cute.jit
def _hypot(x: StorageScalar, y: StorageScalar) -> ComputeScalar:
    x, y = _opmath(x), _opmath(y)
    return _math.hypot(x, y)


@cute.jit
def _pow(x: StorageScalar, y: StorageScalar) -> ComputeScalar:
    x, y = _opmath(x), _opmath(y)
    return cute.math.pow(x, y)


@cute.jit
def _sigmoid(x: StorageScalar) -> ComputeScalar:
    x = _opmath(x)
    one = x.__class__(1.0)
    return one / (one + cute.math.exp(-x))


@cute.jit
def _silu(x: StorageScalar) -> ComputeScalar:
    x = _opmath(x)
    one = x.__class__(1.0)
    return x / (one + cute.math.exp(-x))


@cute.jit
def _elu(
    x: StorageScalar,
    alpha: ComputeScalar,
    scale: ComputeScalar,
    input_scale: ComputeScalar,
) -> ComputeScalar:
    x = _opmath(x)
    z = x.__class__(0.0)
    return x * scale if x > z else cute.math.expm1(x * input_scale) * (alpha * scale)


@cute.jit
def _mish(x: StorageScalar) -> ComputeScalar:
    x = _opmath(x)
    sp = x if x > x.__class__(20.0) else cute.math.log1p(cute.math.exp(x))
    return x * cute.math.tanh(sp)


@cute.jit
def _softplus(
    x: StorageScalar, beta: ComputeScalar, threshold: ComputeScalar
) -> ComputeScalar:
    x = _opmath(x)
    bx = beta * x
    return x if bx > threshold else cute.math.log1p(cute.math.exp(bx)) / beta


@cute.jit
def _hardtanh(
    x: StorageScalar, min_val: ComputeScalar, max_val: ComputeScalar
) -> ComputeScalar:
    x = _opmath(x)
    return _clamp(x, min_val, max_val)


@cute.jit
def _hardsigmoid(x: StorageScalar) -> ComputeScalar:
    # ATen uses a rounded FP32 reciprocal even for FP64 inputs.
    x = _opmath(x)
    z = x.__class__(0.0)
    six = x.__class__(6.0)
    inv_six = x.__class__(_INV_SIX)
    return _clamp(x + x.__class__(3.0), z, six) * inv_six


@cute.jit
def _threshold(
    x: StorageScalar, threshold: StorageScalar, value: StorageScalar
) -> ComputeScalar:
    x = _opmath(x)
    threshold, value = _opmath(threshold), _opmath(value)
    return value if x <= threshold else x


@cute.jit
def _hardshrink(x: StorageScalar, lambd: StorageScalar) -> ComputeScalar:
    x = _opmath(x)
    lambd = _opmath(lambd)
    z = x.__class__(0.0)
    return z if ((x >= -lambd) and (x <= lambd)) else x


@cute.jit
def _softshrink(x: StorageScalar, lambd: StorageScalar) -> ComputeScalar:
    x = _opmath(x)
    lambd = _opmath(lambd)
    z = x.__class__(0.0)
    result = (x - lambd) if x > lambd else ((x + lambd) if x < -lambd else z)
    return x if x != x else result


@cute.jit
def _logit(x: StorageScalar, eps: ComputeScalar) -> ComputeScalar:
    x = _opmath(x)
    one = x.__class__(1.0)
    z = x.__class__(0.0)
    # Negative eps disables clamping, including omitted eps.
    neg = eps < z
    lo = x if neg else eps
    hi = x if neg else one - eps
    zc = lo if x < lo else (hi if x > hi else x)  # noqa: FURB136 -- min() does not lower
    return cute.math.log(zc / (one - zc))


@cute.jit
def _frac(x: StorageScalar) -> ComputeScalar:
    x = _opmath(x)
    return x - _trunc(x)


@cute.jit
def _ceil(x: StorageScalar) -> ComputeScalar:
    x = _opmath(x)
    return -cute.math.floor(-x)


@cute.jit
def _trunc(x: StorageScalar) -> ComputeScalar:
    # Avoid absf: its raw ArithValue is rejected by floor.
    x = _opmath(x)
    z = x.__class__(0.0)
    return cute.math.floor(x) if x >= z else -cute.math.floor(-x)


@cute.jit
def _sign(x: StorageScalar) -> ComputeScalar:
    x = _opmath(x)
    z = x.__class__(0.0)
    return x.__class__(1.0) if x > z else (x.__class__(-1.0) if x < z else z)


@cute.jit
def _relu(x: StorageScalar) -> ComputeScalar:
    x = _opmath(x)
    z = x.__class__(0.0)
    return z if x < z else x  # noqa: FURB136 - preserve NaNs and signed-zero ties


@cute.jit
def _gt(x: StorageScalar, y: StorageScalar) -> Boolean:
    x, y = _opmath(x), _opmath(y)
    return x > y


@cute.jit
def _lt(x: StorageScalar, y: StorageScalar) -> Boolean:
    x, y = _opmath(x), _opmath(y)
    return x < y


@cute.jit
def _ge(x: StorageScalar, y: StorageScalar) -> Boolean:
    x, y = _opmath(x), _opmath(y)
    return x >= y


@cute.jit
def _le(x: StorageScalar, y: StorageScalar) -> Boolean:
    x, y = _opmath(x), _opmath(y)
    return x <= y


@cute.jit
def _eq(x: StorageScalar, y: StorageScalar) -> Boolean:
    x, y = _opmath(x), _opmath(y)
    return x == y


@cute.jit
def _ne(x: StorageScalar, y: StorageScalar) -> Boolean:
    x, y = _opmath(x), _opmath(y)
    return x != y


@cute.jit
def _addcmul(
    x: StorageScalar, t1: StorageScalar, t2: StorageScalar, value: ComputeScalar
) -> ComputeScalar:
    x, t1, t2 = _opmath(x), _opmath(t1), _opmath(t2)
    if const_expr(x.__class__ in (Float32, Float64)):
        return (
            cute.math.fma(t1, t2, x)
            if value == x.__class__(1)
            else cute.math.fma(value, t1 * t2, x)
        )
    return x + value * (t1 * t2)


@cute.jit
def _rsub(x: ComputeScalar, y: ComputeScalar, alpha: ComputeScalar) -> ComputeScalar:
    return y - alpha * x


@cute.jit
def _fmax(x: StorageScalar, y: StorageScalar) -> ComputeScalar:
    x, y = _opmath(x), _opmath(y)
    return _mm.fmax_suppress_nan(x, y)


@cute.jit
def _fmin(x: StorageScalar, y: StorageScalar) -> ComputeScalar:
    x, y = _opmath(x), _opmath(y)
    return _mm.fmin_suppress_nan(x, y)


@cute.jit
def _clamp_min(x: StorageScalar, y: StorageScalar) -> ComputeScalar:
    x, y = _opmath(x), _opmath(y)
    return x if ((x >= y) or (x != x)) else y


@cute.jit
def _clamp_max(x: StorageScalar, y: StorageScalar) -> ComputeScalar:
    x, y = _opmath(x), _opmath(y)
    return x if ((x <= y) or (x != x)) else y


@cute.jit
def _clamp(x: StorageScalar, lo: ComputeScalar, hi: ComputeScalar) -> ComputeScalar:
    # The table supplies dtype limits for omitted bounds.
    x = _opmath(x)
    t = _clamp_min(x, lo)
    return _clamp_max(t, hi)


@cute.jit
def _lerp(x: StorageScalar, y: StorageScalar, w: StorageScalar) -> ComputeScalar:
    # Choose the stable form on either side of weight 0.5.
    x, y, w = _opmath(x), _opmath(y), _opmath(w)
    half = w.__class__(0.5)
    aw = w if w >= w.__class__(0.0) else -w
    near = x + w * (y - x)
    far = y - (y - x) * (w.__class__(1.0) - w)
    return near if aw < half else far


@cute.jit
def _lerp_scalar(x: StorageScalar, y: StorageScalar, w: ComputeScalar) -> ComputeScalar:
    return _lerp(x, y, w)


@cute.jit
def _copysign(x: StorageScalar, y: StorageScalar) -> ComputeScalar:
    x, y = _opmath(x), _opmath(y)
    return _fpbits.copysign(x, y)


@cute.jit
def _fmod_int(x: StorageScalar, y: StorageScalar) -> StorageScalar:
    if const_expr(x.__class__.signed):
        return x.__class__(arith.remsi(x.ir_value(), y.ir_value()))
    return x.__class__(arith.remui(x.ir_value(), y.ir_value()))


@cute.jit
def _remainder_int(x: StorageScalar, y: StorageScalar) -> StorageScalar:
    z = x.__class__(0)
    if const_expr(not x.__class__.signed):
        r = _fmod_int(x, y if y != z else x.__class__(1))
        return r if y != z else x.__class__((1 << x.__class__.width) - 1)
    r = _fmod_int(x, y)
    return r + y if (r != z) and ((r < z) != (y < z)) else r


@cute.jit
def _floor_divide_int(x: StorageScalar, y: StorageScalar) -> StorageScalar:
    z = x.__class__(0)
    valid = y != z
    if const_expr(x.__class__.signed):
        minimum = x.__class__(-(1 << (x.__class__.width - 1)))
        valid = valid and not (x == minimum and y == x.__class__(-1))
    q = x // (y if valid else x.__class__(1))
    return q if y != z else z


@cute.jit
def _div_trunc_int(x: StorageScalar, y: StorageScalar) -> StorageScalar:
    # Correct negative inexact floor quotients.
    z = x.__class__(0)
    q = x // y
    return q + x.__class__(1) if (q < z) and (q * y != x) else q


@cute.jit
def _logical_and(x: StorageScalar, y: StorageScalar) -> Boolean:
    x, y = _opmath(x), _opmath(y)
    z = x.__class__(0)
    return (x != z) and (y != z)


@cute.jit
def _logical_or(x: StorageScalar, y: StorageScalar) -> Boolean:
    x, y = _opmath(x), _opmath(y)
    z = x.__class__(0)
    return (x != z) or (y != z)


@cute.jit
def _logical_xor(x: StorageScalar, y: StorageScalar) -> Boolean:
    x, y = _opmath(x), _opmath(y)
    z = x.__class__(0)
    return (x != z) != (y != z)


@cute.jit
def _logical_not(x: StorageScalar) -> Boolean:
    x = _opmath(x)
    return x == x.__class__(0)


@cute.jit
def _signbit(x: StorageScalar) -> Boolean:
    return _fpbits.signbit(x)


@cute.jit
def _isnan(x: StorageScalar) -> Boolean:
    x = _opmath(x)
    return x != x


@cute.jit
def _isinf(x: StorageScalar) -> Boolean:
    x = _opmath(x)
    # inf iff x-x is nan while x itself is not nan.
    return ((x - x) != (x - x)) and (x == x)


@cute.jit
def _isfinite(x: StorageScalar) -> Boolean:
    x = _opmath(x)
    return (x - x) == (x - x)


@cute.jit
def _bitwise_and(x: StorageScalar, y: StorageScalar) -> StorageScalar:
    return x & y


@cute.jit
def _bitwise_or(x: StorageScalar, y: StorageScalar) -> StorageScalar:
    return x | y


@cute.jit
def _bitwise_xor(x: StorageScalar, y: StorageScalar) -> StorageScalar:
    return x ^ y


@cute.jit
def _bitwise_not(x: StorageScalar) -> StorageScalar:
    return ~x


@cute.jit
def _bitwise_left_shift(x: StorageScalar, y: StorageScalar) -> StorageScalar:
    valid = y >= y.__class__(0) and y < y.__class__(x.__class__.width)
    value = x if valid else x.__class__(0)
    # Bound the shift count even when its result is masked out.
    count = y & y.__class__(x.__class__.width - 1)
    return value << count


@cute.jit
def _bitwise_right_shift(x: StorageScalar, y: StorageScalar) -> StorageScalar:
    result = (
        x >> (x.__class__.width - 1)
        if const_expr(x.__class__.signed)
        else x.__class__(0)
    )
    if y >= 0 and y < x.__class__.width:
        result = x >> y
    return result


@cute.jit
def _frexp(x: StorageScalar) -> tuple[ComputeScalar, Int32]:
    x = _opmath(x)
    return _fpbits.frexp(x)


@cute.jit
def _identity(x: ComputeScalar) -> ComputeScalar:
    return x


@cute.jit
def _copy_float32(x: ComputeScalar) -> ComputeScalar:
    return _math.round_float32(x)


@cute.jit
def _fill(value: ComputeScalar) -> ComputeScalar:
    # Factories share this nullary functor.
    return value


@cute.jit
def _arange(ind: Int64, start: ComputeScalar, step: ComputeScalar) -> ComputeScalar:
    # with_index supplies the flat output index.
    return start + step * ind.to(start.__class__)


@cute.jit
def _linspace(
    ind: Int64,
    start: ComputeScalar,
    end: ComputeScalar,
    step: ComputeScalar,
    halfway: ComputeScalar,
    last: ComputeScalar,
) -> ComputeScalar:
    f = ind.to(start.__class__)
    fwd = start + step * f
    bwd = end - step * (last - ind).to(start.__class__)
    return fwd if ind < halfway else bwd


@cute.jit
def _to_bool(x: ComputeScalar) -> ComputeScalar:
    # Truncating numeric casts would discard nonzero fractional values.
    return x.__class__(0) if x == x.__class__(0) else x.__class__(1)


@cute.jit
def _angle(x: StorageScalar) -> ComputeScalar:
    # Native real angle uses the sign bit, including negative zero.
    x = _opmath(x)
    z = x.__class__(0.0)
    pi = x.__class__(3.141592653589793)
    return x if x != x else (pi if _fpbits.signbit(x) else z)


@cute.jit
def _isposinf(x: StorageScalar) -> Boolean:
    x = _opmath(x)
    # Positive infinity makes x - x NaN; exclude NaN inputs.
    z = x.__class__(0.0)
    return ((x - x) != (x - x)) and (x == x) and (x > z)


@cute.jit
def _isneginf(x: StorageScalar) -> Boolean:
    x = _opmath(x)
    z = x.__class__(0.0)
    return ((x - x) != (x - x)) and (x == x) and (x < z)


@cute.jit
def _sinc(x: StorageScalar) -> ComputeScalar:
    # The formula yields NaN at infinities, as ATen requires.
    x = _opmath(x)
    z = x.__class__(0.0)
    one = x.__class__(1.0)
    px = x * x.__class__(3.141592653589793)
    return one if x == z else cute.math.sin(px) / px


@cute.jit
def _heaviside(x: StorageScalar, y: StorageScalar) -> ComputeScalar:
    x, y = _opmath(x), _opmath(y)
    z = x.__class__(0)
    return y if x == z else (x.__class__(1) if x > z else z)


@cute.jit
def _logaddexp2(x: StorageScalar, y: StorageScalar) -> ComputeScalar:
    # Use the stable base-2 form, preserving equal infinities as in _logaddexp.
    x, y = _opmath(x), _opmath(y)
    d = x - y if x > y else y - x
    m = x if x > y else y  # noqa: FURB136 -- builtin max() does not lower in the DSL
    inf = x.__class__(float("inf"))
    same_inf = (x == y) and ((x == inf) or (x == -inf))
    inv_log2 = x.__class__(1.4426950408889634)
    return m if same_inf else m + cute.math.log1p(cute.math.exp2(-d)) * inv_log2


@cute.jit
def _entr(x: StorageScalar) -> ComputeScalar:
    # log(0) supplies the negative-input result without an infinity literal.
    x = _opmath(x)
    z = x.__class__(0.0)
    neg_inf = cute.math.log(z)
    return (
        x
        if x != x
        else ((-x * cute.math.log(x)) if x > z else (z if x == z else neg_inf))
    )


@cute.jit
def _xlog1py(x: StorageScalar, y: StorageScalar) -> ComputeScalar:
    x, y = _opmath(x), _opmath(y)
    z = x.__class__(0.0)
    return z if (x == z) and (y == y) else x * cute.math.log1p(y)


@cute.jit
def _hardswish(x: StorageScalar) -> ComputeScalar:
    x = _opmath(x)
    z = x.__class__(0.0)
    six = x.__class__(6.0)
    bounded = _clamp(x + x.__class__(3.0), z, six)
    return x * bounded * x.__class__(_INV_SIX)


@cute.jit
def _leaky_relu(x: StorageScalar, negative_slope: ComputeScalar) -> ComputeScalar:
    x = _opmath(x)
    z = x.__class__(0.0)
    return x if x > z else x * negative_slope


@cute.jit
def _addcdiv(
    x: StorageScalar, t1: StorageScalar, t2: StorageScalar, value: ComputeScalar
) -> ComputeScalar:
    x, t1, t2 = _opmath(x), _opmath(t1), _opmath(t2)
    return x + value * (t1 / t2)


@cute.jit
def _gelu_erf(x: StorageScalar) -> ComputeScalar:
    x = _opmath(x)
    half = x.__class__(0.5)
    e = cute.math.erf(x * x.__class__(0.7071067811865476))
    return x * half * (x.__class__(1.0) + e)


@cute.jit
def _gelu_tanh(x: StorageScalar) -> ComputeScalar:
    x = _opmath(x)
    x3 = x * x * x
    inner = x.__class__(0.7978845608028654) * (x + x.__class__(0.044715) * x3)
    return x * x.__class__(0.5) * (x.__class__(1.0) + cute.math.tanh(inner))


@cute.jit
def _nan_to_num(
    x: StorageScalar, nan: ComputeScalar, posinf: ComputeScalar, neginf: ComputeScalar
) -> ComputeScalar:
    x = _opmath(x)
    z = x.__class__(0.0)
    is_inf = ((x - x) != (x - x)) and (x == x)
    sat = posinf if x > z else neginf
    return nan if x != x else (sat if is_inf else x)


def get_fn(name: str | Functor) -> MathFn:
    # Resolve table names lazily after importing the DSL runtime.
    return globals()[name.name if isinstance(name, Functor) else name]
