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
def _logit(x, eps):
    # log(z/(1-z)) where z is x clamped to [eps, 1-eps]. aten's logit_kernel_cuda uses a
    # NEGATIVE eps as the "no clamping" sentinel (the omitted-arg case), which the row's
    # optional_defaults supplies -- so this one body serves both overloads.
    one = x.__class__(1.0)
    z = x.__class__(0.0)
    # A negative eps must leave x untouched. Rather than branch on eps (a runtime scalar,
    # so the DSL needs a select) pick clamp bounds that cannot bite: x itself. A literal
    # -inf/+inf is not an option -- 1.0/0.0 folds at trace time and raises.
    neg = eps < z
    lo = x if neg else eps
    hi = x if neg else one - eps
    # EXACTLY aten's nested ternary, `x < lo ? lo : (x > hi ? hi : x)`, not a sequential
    # clamp-low-then-clamp-high. The two differ when the bounds CROSS (eps > 0.5 gives
    # lo > hi, e.g. eps=0.6 -> [0.6, 0.4]): aten returns lo and never re-clamps, while
    # applying hi afterwards would pull the result back down and flip the sign of the log.
    # A nan input takes neither branch (both comparisons are false) and so propagates.
    zc = lo if x < lo else (hi if x > hi else x)  # noqa: FURB136 -- min() does not lower
    return cute.math.log(zc / (one - zc))


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
def _clamp(x, lo, hi):
    # Two-sided clamp. An OMITTED bound is filled by the row's optional_defaults with
    # that dtype's extreme, so this one formula serves clamp, clamp_min and clamp_max.
    # NaN in x propagates (aten), hence the x!=x tests rather than plain comparisons.
    t = x if ((x > lo) or (x != x)) else lo
    return t if ((t < hi) or (t != t)) else hi


@cute.jit
def _lerp(x, y, w):
    # start + w*(end-start). aten uses this form for |w|<0.5 and the
    # end-(end-start)*(1-w) form otherwise, to keep precision when w approaches 1.
    half = w.__class__(0.5)
    aw = w if w >= w.__class__(0.0) else -w
    near = x + w * (y - x)
    far = y - (y - x) * (w.__class__(1.0) - w)
    return near if aw < half else far


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
    #
    # KNOWN LIMIT: this loses accuracy once |x/y| approaches the mantissa width,
    # because trunc(x/y)*y rounds away the low bits that the subtraction needs --
    # fmod(-1e20, 501) gives -6400 where aten (libm, exact multi-word reduction)
    # gives -388. Routing through _remainder's floor-remainder does NOT help: it
    # trades these for a larger number of small fp32 diffs (measured 34 -> 81
    # divergent OpInfo samples). A real fix needs exact reduction, not a rearranged
    # formula.
    q = x / y
    z = q.__class__(0.0)
    tq = cute.math.floor(q) if q >= z else -cute.math.floor(-q)
    return x - tq * y


@cute.jit
def _remainder(x, y):
    # Python %, result sign follows y. NOT x - floor(x/y)*y: that loses the identity
    # when x/y rounds across an integer (see _floor_divide). aten's remainder_cuda is
    # fmod plus a single sign fixup -- fmod is exact, so this is too.
    z = x.__class__(0.0)
    q = x / y
    tq = cute.math.floor(q) if q >= z else -cute.math.floor(-q)
    mod = x - tq * y  # fmod(x, y): sign follows x
    return mod + y if (mod != z) and ((y < z) != (mod < z)) else mod


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
    # NOT floor(x/y): aten's c10::div_floor_floating derives the quotient from the
    # REMAINDER so it stays exact when x/y rounds across an integer. floor(1.0/0.001)
    # gives 1000 because the division rounds UP to exactly 1000.0, but the true
    # quotient is 999.99...; aten returns 999. Mirror aten's algorithm:
    #   mod = fmod(x, y); div = (x - mod) / y   (exact -- x-mod is a multiple of y)
    #   div -= 1 when mod is nonzero and its sign differs from y (floor, not trunc)
    #   then snap div to the nearest integer when it is within 0.5 (rounding guard)
    # b == 0 keeps the plain IEEE result (inf/nan), as aten does.
    z = x.__class__(0.0)
    q = x / y
    tq = cute.math.floor(q) if q >= z else -cute.math.floor(-q)
    mod = x - tq * y  # fmod(x, y): sign follows x
    div = (x - mod) / y
    div = div - x.__class__(1.0) if (mod != z) and ((y < z) != (mod < z)) else div
    fd = cute.math.floor(div)
    fd = fd + x.__class__(1.0) if (div - fd) > x.__class__(0.5) else fd
    # div == 0 -> signed zero carrying q's sign (cute.math.copysign needs CTK>=13, so
    # negate manually; 1/q < 0 catches q == -0.0).
    one = x.__class__(1.0)
    q_neg = (q < z) or ((q == z) and (one / q < z))
    snapped = fd if div != z else (-z if q_neg else z)
    return q if y == z else snapped


@cute.jit
def _floor_divide_int(x, y):
    # The DSL's Int `//` is already floor division.
    return x // y


@cute.jit
def _div_trunc(x, y):
    # div(rounding_mode="trunc"): round the quotient TOWARD ZERO, unlike _floor_divide.
    q = x / y
    z = q.__class__(0.0)
    return cute.math.floor(q) if q >= z else -cute.math.floor(-q)


@cute.jit
def _div_trunc_int(x, y):
    # Integer trunc division. `//` FLOORS, which differs from truncation exactly when
    # the quotient is negative and the division is inexact -- add 1 back in that case.
    z = x.__class__(0)
    q = x // y
    return q + x.__class__(1) if (q < z) and (q * y != x) else q


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


# ---- nullary (nin == 0): no input; the value comes from scalars, or from the INDEX ----


@cute.jit
def _fill(value):
    # No input to read, so the kernel calls fn(*consts) with no loaded values and this
    # just returns the broadcast scalar. Backs aten's fill_ and everything built on it
    # (full / zeros / ones / full_like, all of which lower to empty() + fill_).
    return value


@cute.jit
def _arange(ind, start, step):
    # start + step*ind, the value form of aten's arange_cuda_out. `ind` is the thread's
    # flat output index (the kernel's with_index mode, aten's gpu_kernel_with_index).
    return start + step * ind.to(start.__class__)


@cute.jit
def _linspace(ind, start, end, step, halfway, last):
    # aten's linspace_cuda_out, INCLUDING its halfway split: the first half steps FORWARD
    # from `start`, the second half BACKWARD from `end`
    #   ind < halfway -> start + step*ind
    #   else          -> end - step*(steps - 1 - ind)
    # That is not an optimization. It is what pins BOTH endpoints exactly (stepping forward
    # throughout would drift by accumulated rounding and miss `end`), so it is reproduced
    # here -- verified: first/last are exact for every dtype and step count tested.
    # `last` is steps-1, precomputed on the host. All arithmetic is in the fp32+ compute
    # dtype; narrowing happens only on the store.
    f = ind.to(start.__class__)
    fwd = start + step * f
    bwd = end - step * (last - f)
    return fwd if ind < halfway else bwd


@cute.jit
def _to_bool(x):
    # bool TARGET: aten's bool cast is a NONZERO test, not a numeric cast (0.5 ->
    # True; a trunc-style convert would give False). Computed in the SOURCE dtype;
    # returns 0/1 which the store casts into the target's int8 (bool-byte) view.
    return x.__class__(0) if x == x.__class__(0) else x.__class__(1)


# ---- trivial additions (verified against aten; see table.py rows) ----


@cute.jit
def _angle(x):
    # REAL-input angle: nan -> nan, negative -> pi, else 0 (aten's angle_wrapper).
    # Complex input declines (no DSL complex).
    z = x.__class__(0.0)
    pi = x.__class__(3.141592653589793)
    return x if x != x else (pi if x < z else z)


@cute.jit
def _isposinf(x):
    # x == +inf, without needing an inf literal: inf is the only value where x-x is
    # nan while x itself is not nan (the covered _isinf test), and is positive.
    z = x.__class__(0.0)
    return ((x - x) != (x - x)) and (x == x) and (x > z)


@cute.jit
def _isneginf(x):
    z = x.__class__(0.0)
    return ((x - x) != (x - x)) and (x == x) and (x < z)


@cute.jit
def _sinc(x):
    # sin(pi x)/(pi x), 1 at x==0. aten yields NaN at +-inf (verified), which the
    # sin(inf)/inf form produces naturally -- no special case needed.
    z = x.__class__(0.0)
    one = x.__class__(1.0)
    px = x * x.__class__(3.141592653589793)
    return one if x == z else cute.math.sin(px) / px


@cute.jit
def _heaviside(x, y):
    # 0 for x<0, y at x==0, 1 for x>0. Integer-capable (no float-only primitive).
    z = x.__class__(0)
    return y if x == z else (x.__class__(1) if x > z else z)


@cute.jit
def _logaddexp2(x, y):
    # log2(2^x + 2^y), overflow-safe: m + log2(1 + 2^-|x-y|). Base-2 twin of the
    # covered _logaddexp.
    d = x - y if x > y else y - x
    m = x if x > y else y  # noqa: FURB136 -- builtin max() does not lower in the DSL
    return m + cute.math.log2(x.__class__(1.0) + cute.math.exp2(-d))


@cute.jit
def _entr(x):
    # -x*log(x) for x>0, 0 at x==0, -inf for x<0, nan for nan (aten's calc_entr).
    # -inf is produced arithmetically as log(0) since there is no inf literal.
    z = x.__class__(0.0)
    neg_inf = cute.math.log(z)
    return (
        x
        if x != x
        else ((-x * cute.math.log(x)) if x > z else (z if x == z else neg_inf))
    )


@cute.jit
def _xlog1py(x, y):
    # x*log1p(y), with x==0 short-circuiting to 0 unless y is nan (as for xlogy).
    z = x.__class__(0.0)
    return z if (x == z) and (y == y) else x * cute.math.log(x.__class__(1.0) + y)


@cute.jit
def _hardswish(x):
    # x * clamp(x+3, 0, 6) / 6
    z = x.__class__(0.0)
    six = x.__class__(6.0)
    t = x + x.__class__(3.0)
    lo = t if t > z else z  # noqa: FURB136 -- no DSL builtin max/min
    hi = lo if lo < six else six  # noqa: FURB136
    return x * hi / six


@cute.jit
def _leaky_relu(x, negative_slope):
    z = x.__class__(0.0)
    return x if x > z else x * negative_slope


@cute.jit
def _addcdiv(x, t1, t2, value):
    # nin=3: out = self + value * (t1 / t2). INT_TO_FLOAT twin of addcmul.
    return x + value * (t1 / t2)


@cute.jit
def _gelu_erf(x):
    # Exact gelu: x * 0.5 * (1 + erf(x / sqrt(2))).
    half = x.__class__(0.5)
    e = cute.math.erf(x * x.__class__(0.7071067811865476))
    return x * half * (x.__class__(1.0) + e)


@cute.jit
def _gelu_tanh(x):
    # approximate="tanh": 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3))).
    x3 = x * x * x
    inner = x.__class__(0.7978845608028654) * (x + x.__class__(0.044715) * x3)
    return x * x.__class__(0.5) * (x.__class__(1.0) + cute.math.tanh(inner))


@cute.jit
def _nan_to_num(x, nan, posinf, neginf):
    # nan -> `nan`; +inf -> `posinf`; -inf -> `neginf`; else unchanged. An omitted
    # posinf/neginf is filled by optional_defaults with the compute dtype's finite
    # max/lowest, matching aten. inf is detected without an inf literal: x-x is nan
    # only for inf (and for nan, excluded by the x==x test).
    z = x.__class__(0.0)
    is_inf = ((x - x) != (x - x)) and (x == x)
    sat = posinf if x > z else neginf
    return nan if x != x else (sat if is_inf else x)


# ---- lazy fn resolver (table.py rows reference these by name) ----


def get_fn(name: str):
    # Resolve a table row's `fn` name to the @cute.jit callable defined above. Called
    # from overrides.py's impl on the first real dispatch, so importing this module
    # (and thus cutlass) is deferred off the `import torch` / registration path.
    return globals()[name]
