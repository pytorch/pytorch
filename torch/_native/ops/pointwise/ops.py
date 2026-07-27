# The @cute.jit kernel functions referenced by the pointwise op table (see table.py,
# which is cutlass-free and holds the declarative rows). Each function is plain
# @cute.jit math over COMPUTE-dtype scalars:
#   fn(*input_vals, *scalar_consts) -> result | tuple-of-results
# Inputs arrive already converted to the compute dtype; baked scalar args (e.g. add's
# `alpha`) follow, as compute-dtype constants. The result is cast to the op's output
# dtype. A function references only DSL ops (operators, cute.math.*), never a
# user-class method (which would trip the IR flattener).
#
# This module imports `cutlass` (the @cute.jit decorator), so it must NOT be imported
# at `import torch` time -- overrides.py resolves a row's function lazily via get_fn()
# on the first real (non-declined) call. table.py carries the registration metadata.

from __future__ import annotations

import cutlass.cute as cute


# ---- op math (module-level named functions; reused/composed freely) ----


@cute.jit
def _neg(x):
    return -x


@cute.jit
def _add(x, y, alpha):
    return x + alpha * y


@cute.jit
def _mul(x, y):
    return x * y


@cute.jit
def _sub(x, y, alpha):
    return x - alpha * y


@cute.jit
def _div(x, y):
    return x / y


@cute.jit
def _maximum(x, y):
    # NaN-PROPAGATING max (matches torch.maximum, NOT fmax which suppresses): take y
    # when it is larger OR is NaN (y != y detects NaN), so a NaN in either operand
    # propagates. Same form as the reduction AMaxOps._maxnan. (Planned: Blackwell
    # PTX min/max with the NaN-propagating mode, later.)
    return y if ((y > x) or (y != y)) else x


@cute.jit
def _minimum(x, y):
    return y if ((y < x) or (y != y)) else x


# ---- unary math (INT_TO_FLOAT promotion) ----


@cute.jit
def _exp(x):
    return cute.math.exp(x)


@cute.jit
def _exp2(x):
    return cute.math.exp2(x)


@cute.jit
def _expm1(x):
    return cute.math.exp(x) - x.__class__(1.0)


@cute.jit
def _log(x):
    return cute.math.log(x)


@cute.jit
def _log2(x):
    return cute.math.log2(x)


@cute.jit
def _log10(x):
    return cute.math.log10(x)


@cute.jit
def _log1p(x):
    return cute.math.log(x + x.__class__(1.0))


@cute.jit
def _sqrt(x):
    return cute.math.sqrt(x)


@cute.jit
def _rsqrt(x):
    return cute.math.rsqrt(x)


@cute.jit
def _reciprocal(x):
    return x.__class__(1.0) / x


@cute.jit
def _sin(x):
    return cute.math.sin(x)


@cute.jit
def _cos(x):
    return cute.math.cos(x)


@cute.jit
def _tan(x):
    return cute.math.tan(x)


@cute.jit
def _asin(x):
    return cute.math.asin(x)


@cute.jit
def _acos(x):
    return cute.math.acos(x)


@cute.jit
def _atan(x):
    return cute.math.atan(x)


@cute.jit
def _atan2(x, y):
    return cute.math.atan2(x, y)


@cute.jit
def _erfc(x):
    return x.__class__(1.0) - cute.math.erf(x)


@cute.jit
def _cosh(x):
    # (e^x + e^-x)/2
    return (cute.math.exp(x) + cute.math.exp(-x)) * x.__class__(0.5)


@cute.jit
def _sinh(x):
    return (cute.math.exp(x) - cute.math.exp(-x)) * x.__class__(0.5)


@cute.jit
def _asinh(x):
    # log(x + sqrt(x^2 + 1))
    return cute.math.log(x + cute.math.sqrt(x * x + x.__class__(1.0)))


@cute.jit
def _acosh(x):
    return cute.math.log(x + cute.math.sqrt(x * x - x.__class__(1.0)))


@cute.jit
def _atanh(x):
    # 0.5 * log((1+x)/(1-x))
    one = x.__class__(1.0)
    return x.__class__(0.5) * cute.math.log((one + x) / (one - x))


@cute.jit
def _deg2rad(x):
    return x * x.__class__(0.017453292519943295)  # pi/180


@cute.jit
def _rad2deg(x):
    return x * x.__class__(57.29577951308232)  # 180/pi


@cute.jit
def _logaddexp(x, y):
    # log(e^x + e^y) = max + log1p(e^-|x-y|), the standard overflow-safe form.
    d = x - y if x > y else y - x
    m = x if x > y else y  # noqa: FURB136 -- builtin max() does not lower in the DSL
    return m + cute.math.log(x.__class__(1.0) + cute.math.exp(-d))


@cute.jit
def _xlogy(x, y):
    # 0 if x==0 (even y=0/inf -> 0 per aten), nan if y is nan handled by y!=y flow.
    z = x.__class__(0.0)
    return z if (x == z) and (y == y) else x * cute.math.log(y)


@cute.jit
def _hypot(x, y):
    # overflow-safe |(x,y)|: m*sqrt(1+(n/m)^2) with m=max(|x|,|y|)
    z = x.__class__(0.0)
    ax = x if x >= z else -x
    ay = y if y >= z else -y
    m = ax if ax > ay else ay  # noqa: FURB136 -- builtin max()/min() do not lower
    n = ay if ax > ay else ax  # noqa: FURB136
    one = x.__class__(1.0)
    r = n / m if m > z else z
    return m * cute.math.sqrt(one + r * r)


@cute.jit
def _pow(x, y):
    # Full float pow via exp2(y*log2|x|) plus the sign/edge dance:
    #   x>0            -> exp2(y*log2 x)
    #   x<0, y int     -> sign(-1^y) * exp2(y*log2|x|)   (odd y -> negative)
    #   x<0, y non-int -> nan  (injected via 0/0)
    #   y==0           -> 1 (any x, incl. 0 and nan-base? aten: pow(x,0)=1 always)
    #   x==0           -> exp2(y * -inf) = 0 (y>0) / inf (y<0), correct via the mag path.
    z = x.__class__(0.0)
    one = x.__class__(1.0)
    two = x.__class__(2.0)
    ax = x if x >= z else -x
    mag = cute.math.exp2(y * cute.math.log2(ax))
    y_int = y == cute.math.floor(y)
    y_odd = y_int and (cute.math.floor(y * x.__class__(0.5)) * two != y)
    signed = -mag if ((x < z) and y_odd) else mag
    nan_val = x.__class__(float("nan"))  # for the invalid neg-base^non-int case
    r = nan_val if ((x < z) and (not y_int)) else signed
    return one if y == z else r


@cute.jit
def _tanh(x):
    return cute.math.tanh(x)


@cute.jit
def _erf(x):
    return cute.math.erf(x)


@cute.jit
def _sigmoid(x):
    one = x.__class__(1.0)
    return one / (one + cute.math.exp(-x))


# ---- activations (INT_TO_FLOAT; scalar-parameterized ones bake their Scalars) ----


@cute.jit
def _silu(x):
    one = x.__class__(1.0)
    return x / (one + cute.math.exp(-x))


@cute.jit
def _elu(x, alpha, scale, input_scale):
    z = x.__class__(0.0)
    return scale * (
        x if x > z else alpha * (cute.math.exp(x * input_scale) - x.__class__(1.0))
    )


@cute.jit
def _celu(x, alpha):
    z = x.__class__(0.0)
    return x if x > z else alpha * (cute.math.exp(x / alpha) - x.__class__(1.0))


@cute.jit
def _selu(x):
    z = x.__class__(0.0)
    scale = x.__class__(1.0507009873554805)
    alpha = x.__class__(1.6732632423543772)
    return scale * (x if x > z else alpha * (cute.math.exp(x) - x.__class__(1.0)))


@cute.jit
def _mish(x):
    # x * tanh(softplus(x)); softplus at default beta=1 with the beta*x>20 passthrough.
    sp = (
        x
        if x > x.__class__(20.0)
        else cute.math.log(x.__class__(1.0) + cute.math.exp(x))
    )
    return x * cute.math.tanh(sp)


@cute.jit
def _softplus(x, beta, threshold):
    # (1/beta) * log1p(exp(beta*x)); passthrough to x when beta*x > threshold (aten).
    bx = beta * x
    return (
        x
        if bx > threshold
        else cute.math.log(x.__class__(1.0) + cute.math.exp(bx)) / beta
    )


@cute.jit
def _hardtanh(x, min_val, max_val):
    lo = x if x > min_val else min_val  # noqa: FURB136 -- no DSL builtin max/min
    return lo if lo < max_val else max_val  # noqa: FURB136


@cute.jit
def _hardsigmoid(x):
    # clamp(x/6 + 1/2, 0, 1)
    z = x.__class__(0.0)
    one = x.__class__(1.0)
    v = x * x.__class__(0.16666666666666666) + x.__class__(0.5)
    lo = v if v > z else z  # noqa: FURB136 -- no DSL builtin max/min
    return lo if lo < one else one  # noqa: FURB136


@cute.jit
def _relu6(x):
    z = x.__class__(0.0)
    six = x.__class__(6.0)
    lo = x if x > z else z  # noqa: FURB136 -- no DSL builtin max/min
    return lo if lo < six else six  # noqa: FURB136


@cute.jit
def _threshold(x, threshold, value):
    return x if x > threshold else value


@cute.jit
def _hardshrink(x, lambd):
    z = x.__class__(0.0)
    return z if ((x > -lambd) and (x < lambd)) else x


@cute.jit
def _softshrink(x, lambd):
    z = x.__class__(0.0)
    return (x - lambd) if x > lambd else ((x + lambd) if x < -lambd else z)


@cute.jit
def _logit(x):
    # log(x/(1-x)); eps=None (the only overload we register) -> no clamping.
    return cute.math.log(x / (x.__class__(1.0) - x))


# ---- rounding / sign (DEFAULT promotion; output dtype follows input) ----


@cute.jit
def _abs(x):
    z = x.__class__(0.0)
    return x if x >= z else -x


@cute.jit
def _square(x):
    return x * x


@cute.jit
def _frac(x):
    # x - trunc(x), sign follows x
    z = x.__class__(0.0)
    t = cute.math.floor(x) if x >= z else -cute.math.floor(-x)
    return x - t


@cute.jit
def _round(x):
    # round-half-to-EVEN (aten/IEEE): floor(x+0.5), stepping back 1 when x is exactly
    # a .5 half AND the rounded value is odd (verified vs torch.round on halves).
    h = x.__class__(0.5)
    one = x.__class__(1.0)
    two = x.__class__(2.0)
    r = cute.math.floor(x + h)
    is_half = (x - cute.math.floor(x)) == h
    r_even = cute.math.floor(r / two) * two == r
    return r if (not is_half) or r_even else r - one


@cute.jit
def _floor(x):
    return cute.math.floor(x)


@cute.jit
def _ceil(x):
    return -cute.math.floor(-x)


@cute.jit
def _trunc(x):
    # toward-zero rounding: floor for x>=0, ceil(-floor(-x)) for x<0. (Avoid chaining
    # cute.math.absf into floor -- absf yields an ArithValue that floor rejects.)
    z = x.__class__(0.0)
    return cute.math.floor(x) if x >= z else -cute.math.floor(-x)


@cute.jit
def _sign(x):
    z = x.__class__(0.0)
    return x.__class__(1.0) if x > z else (x.__class__(-1.0) if x < z else z)


@cute.jit
def _relu(x):
    # NB: builtin max() does not lower in the DSL -- keep the explicit select.
    z = x.__class__(0.0)
    return x if x > z else z  # noqa: FURB136


# ---- comparisons (ALWAYS_BOOL) ----


@cute.jit
def _gt(x, y):
    return x > y


@cute.jit
def _lt(x, y):
    return x < y


@cute.jit
def _ge(x, y):
    return x >= y


@cute.jit
def _le(x, y):
    return x <= y


@cute.jit
def _eq(x, y):
    return x == y


@cute.jit
def _ne(x, y):
    return x != y


@cute.jit
def _addcmul(x, t1, t2, value):
    # nin=3, scalar `value`: out = self + value * t1 * t2.
    return x + value * t1 * t2


# ---- more binaries ----


@cute.jit
def _rsub(x, y, alpha):
    return y - alpha * x


@cute.jit
def _fmax(x, y):
    # NaN-SUPPRESSING max (C fmax): if one arg is NaN take the other. y!=y detects NaN.
    return y if ((y > x) or (x != x)) else x


@cute.jit
def _fmin(x, y):
    return y if ((y < x) or (x != x)) else x


@cute.jit
def _clamp_min(x, y):
    # NaN in x propagates (aten): x!=x -> take x.
    return x if ((x > y) or (x != x)) else y


@cute.jit
def _clamp_max(x, y):
    return x if ((x < y) or (x != x)) else y


@cute.jit
def _copysign(x, y):
    # |x| with y's sign bit. cute.math.copysign needs CTK>=13, so branch on y with a
    # signed-zero-correct test: 1/y < 0 catches y = -0.0 (1/-0.0 = -inf).
    z = x.__class__(0.0)
    one = x.__class__(1.0)
    a = x if x >= z else -x
    y_neg = (y < z) or ((y == z) and (one / y < z))
    return -a if y_neg else a


@cute.jit
def _fmod(x, y):
    # C fmod: x - trunc(x/y)*y, result sign follows x.
    q = x / y
    z = q.__class__(0.0)
    tq = cute.math.floor(q) if q >= z else -cute.math.floor(-q)
    return x - tq * y


@cute.jit
def _remainder(x, y):
    # Python %: x - floor(x/y)*y, result sign follows y.
    return x - cute.math.floor(x / y) * y


@cute.jit
def _fmod_int(x, y):
    # Integer C fmod (sign follows x). NB the DSL's Int `/` is FLOAT true-division
    # (x - (x/y)*y would be exactly 0); `//` is genuine integer FLOOR division. So:
    # r = x - (x//y)*y is the floor-remainder (sign follows y); shift it by y when the
    # signs differ to get the truncation-remainder.
    z = x.__class__(0)
    r = x - (x // y) * y
    needs_fix = (r != z) and ((x < z) != (y < z))
    return r - y if needs_fix else r


@cute.jit
def _remainder_int(x, y):
    # Integer Python % (sign follows y) = the floor-remainder directly (`//` floors).
    return x - (x // y) * y


@cute.jit
def _floor_divide(x, y):
    return cute.math.floor(x / y)


@cute.jit
def _floor_divide_int(x, y):
    # The DSL's Int `//` is already floor division.
    return x // y


# ---- logical (ALWAYS_BOOL over any input dtype) ----


@cute.jit
def _logical_and(x, y):
    z = x.__class__(0)
    return (x != z) and (y != z)


@cute.jit
def _logical_or(x, y):
    z = x.__class__(0)
    return (x != z) or (y != z)


@cute.jit
def _logical_xor(x, y):
    z = x.__class__(0)
    return (x != z) != (y != z)


@cute.jit
def _logical_not(x):
    return x == x.__class__(0)


@cute.jit
def _signbit(x):
    # True for negative numbers INCLUDING -0.0 (signed-zero test via 1/x = -inf).
    z = x.__class__(0.0)
    return (x < z) or ((x == z) and (x.__class__(1.0) / x < z))


@cute.jit
def _isnan(x):
    return x != x


@cute.jit
def _isinf(x):
    # inf iff x-x is nan while x itself is not nan.
    return ((x - x) != (x - x)) and (x == x)


@cute.jit
def _isfinite(x):
    return (x - x) == (x - x)


# ---- bitwise (integer-only; DSL &,|,^,~,<<,>> lower on Int) ----


@cute.jit
def _bitwise_and(x, y):
    return x & y


@cute.jit
def _bitwise_or(x, y):
    return x | y


@cute.jit
def _bitwise_xor(x, y):
    return x ^ y


@cute.jit
def _bitwise_not(x):
    return ~x


@cute.jit
def _bitwise_left_shift(x, y):
    return x << y


@cute.jit
def _bitwise_right_shift(x, y):
    return x >> y


@cute.jit
def _frexp(x):
    # nout=2: (mantissa, exponent) with x == mantissa * 2**exponent, |mantissa| in
    # [0.5, 1). DSL has no frexp primitive; derive from log2|x|. We need log2 of the
    # MAGNITUDE, but every way of forming |x| with a single value (cute.math.absf, a
    # ternary select, copysign) yields an ArithValue that log2 rejects -- only pure
    # arithmetic on the value stays a Numeric. So compute log2(x) and log2(-x)
    # separately (each argument is a plain negation, still Numeric) and select the
    # RESULT by sign. e = floor(log2|x|)+1, m = x * 2**(-e); the exponent is projected
    # as Float and cast to int32 at the store (the row's out dtype). Constants use
    # x.__class__ so all ifexp branches share x's dtype (a bare Float32 literal breaks
    # fp64). NB: do NOT use log2(x*x)*0.5 -- x*x overflows to inf for large |x|
    # (fp16 above ~256) and corrupts the exponent.
    #
    # Only finite, nonzero x take the log2 path: zero would give log2(0) = -inf, and
    # inf/nan must pass through as (x, 0) to match aten (frexp(inf) = (inf, 0)). A
    # value is finite-and-nonzero iff (x - x == 0) -- false (nan) for inf/nan -- AND
    # x != 0.
    #
    # NOTE: exact for fp16/bf16/fp32 but NOT fp64 -- at exact powers of two the fp64
    # log2 lands just under the integer boundary and floor() picks the wrong exponent.
    # A correct fp64 frexp needs IEEE exponent-field bit extraction (Float64.bitcast);
    # deferred -> the row restricts dtypes to exclude fp64.
    flt = x.__class__
    zero = flt(0.0)
    use_log2 = (x - x == zero) and (x != zero)
    log2_abs = cute.math.log2(x) if x > zero else cute.math.log2(-x)
    e = cute.math.floor(log2_abs) + flt(1.0)
    e = e if use_log2 else zero
    m = x * cute.math.exp2(-e) if use_log2 else x
    return (m, e)


# ---- conversions (dtype casts: _to_copy / copy_; see overrides conversion impls) ----


@cute.jit
def _identity(x):
    # Pure dtype cast: the kernel's load-side packed convert to the COMPUTE dtype
    # (= the target dtype) IS the conversion; the store is then a no-op cast.
    return x


@cute.jit
def _to_bool(x):
    # bool TARGET: aten's bool cast is a NONZERO test, not a numeric cast (0.5 ->
    # True; a trunc-style convert would give False). Computed in the SOURCE dtype;
    # returns 0/1 which the store casts into the target's int8 (bool-byte) view.
    return x.__class__(0) if x == x.__class__(0) else x.__class__(1)


# ---- lazy fn resolver (table.py rows reference these by name) ----


def get_fn(name: str):
    # Resolve a table row's `fn` name to the @cute.jit callable defined above. Called
    # from overrides.py's impl on the first real dispatch, so importing this module
    # (and thus cutlass) is deferred off the `import torch` / registration path.
    return globals()[name]
