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
def _tanh(x):
    return cute.math.tanh(x)


@cute.jit
def _erf(x):
    return cute.math.erf(x)


@cute.jit
def _sigmoid(x):
    one = x.__class__(1.0)
    return one / (one + cute.math.exp(-x))


# ---- rounding / sign (DEFAULT promotion; output dtype follows input) ----


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


# ---- lazy fn resolver (table.py rows reference these by name) ----


def get_fn(name: str):
    # Resolve a table row's `fn` name to the @cute.jit callable defined above. Called
    # from overrides.py's impl on the first real dispatch, so importing this module
    # (and thus cutlass) is deferred off the `import torch` / registration path.
    return globals()[name]
