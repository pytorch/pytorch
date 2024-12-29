"""
Adaptive numerical evaluation of SymPy expressions, using mpmath
for mathematical functions.
"""
from __future__ import annotations
from typing import Tuple as tTuple, Optional, Union as tUnion, Callable, List, Dict as tDict, Type, TYPE_CHECKING, \
    Any, overload

import math

import mpmath.libmp as libmp
from mpmath import (
    make_mpc, make_mpf, mp, mpc, mpf, nsum, quadts, quadosc, workprec)
from mpmath import inf as mpmath_inf
from mpmath.libmp import (from_int, from_man_exp, from_rational, fhalf,
        fnan, finf, fninf, fnone, fone, fzero, mpf_abs, mpf_add,
        mpf_atan, mpf_atan2, mpf_cmp, mpf_cos, mpf_e, mpf_exp, mpf_log, mpf_lt,
        mpf_mul, mpf_neg, mpf_pi, mpf_pow, mpf_pow_int, mpf_shift, mpf_sin,
        mpf_sqrt, normalize, round_nearest, to_int, to_str)
from mpmath.libmp import bitcount as mpmath_bitcount
from mpmath.libmp.backend import MPZ
from mpmath.libmp.libmpc import _infs_nan
from mpmath.libmp.libmpf import dps_to_prec, prec_to_dps

from .sympify import sympify
from .singleton import S
from sympy.external.gmpy import SYMPY_INTS
from sympy.utilities.iterables import is_sequence
from sympy.utilities.lambdify import lambdify
from sympy.utilities.misc import as_int

if TYPE_CHECKING:
    from sympy.core.expr import Expr
    from sympy.core.add import Add
    from sympy.core.mul import Mul
    from sympy.core.power import Pow
    from sympy.core.symbol import Symbol
    from sympy.integrals.integrals import Integral
    from sympy.concrete.summations import Sum
    from sympy.concrete.products import Product
    from sympy.functions.elementary.exponential import exp, log
    from sympy.functions.elementary.complexes import Abs, re, im
    from sympy.functions.elementary.integers import ceiling, floor
    from sympy.functions.elementary.trigonometric import atan
    from .numbers import Float, Rational, Integer, AlgebraicNumber, Number

LG10 = math.log2(10)
rnd = round_nearest


def bitcount(n):
    """Return smallest integer, b, such that |n|/2**b < 1.
    """
    return mpmath_bitcount(abs(int(n)))

# Used in a few places as placeholder values to denote exponents and
# precision levels, e.g. of exact numbers. Must be careful to avoid
# passing these to mpmath functions or returning them in final results.
INF = float(mpmath_inf)
MINUS_INF = float(-mpmath_inf)

# ~= 100 digits. Real men set this to INF.
DEFAULT_MAXPREC = 333


class PrecisionExhausted(ArithmeticError):
    pass

#----------------------------------------------------------------------------#
#                                                                            #
#              Helper functions for arithmetic and complex parts             #
#                                                                            #
#----------------------------------------------------------------------------#

"""
An mpf value tuple is a tuple of integers (sign, man, exp, bc)
representing a floating-point number: [1, -1][sign]*man*2**exp where
sign is 0 or 1 and bc should correspond to the number of bits used to
represent the mantissa (man) in binary notation, e.g.
"""
MPF_TUP = tTuple[int, int, int, int]  # mpf value tuple

"""
Explanation
===========

>>> from sympy.core.evalf import bitcount
>>> sign, man, exp, bc = 0, 5, 1, 3
>>> n = [1, -1][sign]*man*2**exp
>>> n, bitcount(man)
(10, 3)

A temporary result is a tuple (re, im, re_acc, im_acc) where
re and im are nonzero mpf value tuples representing approximate
numbers, or None to denote exact zeros.

re_acc, im_acc are integers denoting log2(e) where e is the estimated
relative accuracy of the respective complex part, but may be anything
if the corresponding complex part is None.

"""
TMP_RES = Any  # temporary result, should be some variant of
# tUnion[tTuple[Optional[MPF_TUP], Optional[MPF_TUP],
#               Optional[int], Optional[int]],
#        'ComplexInfinity']
# but mypy reports error because it doesn't know as we know
# 1. re and re_acc are either both None or both MPF_TUP
# 2. sometimes the result can't be zoo

# type of the "options" parameter in internal evalf functions
OPT_DICT = tDict[str, Any]


def fastlog(x: Optional[MPF_TUP]) -> tUnion[int, Any]:
    """Fast approximation of log2(x) for an mpf value tuple x.

    Explanation
    ===========

    Calculated as exponent + width of mantissa. This is an
    approximation for two reasons: 1) it gives the ceil(log2(abs(x)))
    value and 2) it is too high by 1 in the case that x is an exact
    power of 2. Although this is easy to remedy by testing to see if
    the odd mpf mantissa is 1 (indicating that one was dealing with
    an exact power of 2) that would decrease the speed and is not
    necessary as this is only being used as an approximation for the
    number of bits in x. The correct return value could be written as
    "x[2] + (x[3] if x[1] != 1 else 0)".
        Since mpf tuples always have an odd mantissa, no check is done
    to see if the mantissa is a multiple of 2 (in which case the
    result would be too large by 1).

    Examples
    ========

    >>> from sympy import log
    >>> from sympy.core.evalf import fastlog, bitcount
    >>> s, m, e = 0, 5, 1
    >>> bc = bitcount(m)
    >>> n = [1, -1][s]*m*2**e
    >>> n, (log(n)/log(2)).evalf(2), fastlog((s, m, e, bc))
    (10, 3.3, 4)
    """

    if not x or x == fzero:
        return MINUS_INF
    return x[2] + x[3]


def pure_complex(v: 'Expr', or_real=False) -> tuple['Number', 'Number'] | None:
    """Return a and b if v matches a + I*b where b is not zero and
    a and b are Numbers, else None. If `or_real` is True then 0 will
    be returned for `b` if `v` is a real number.

    Examples
    ========

    >>> from sympy.core.evalf import pure_complex
    >>> from sympy import sqrt, I, S
    >>> a, b, surd = S(2), S(3), sqrt(2)
    >>> pure_complex(a)
    >>> pure_complex(a, or_real=True)
    (2, 0)
    >>> pure_complex(surd)
    >>> pure_complex(a + b*I)
    (2, 3)
    >>> pure_complex(I)
    (0, 1)
    """
    h, t = v.as_coeff_Add()
    if t:
        c, i = t.as_coeff_Mul()
        if i is S.ImaginaryUnit:
            return h, c
    elif or_real:
        return h, S.Zero
    return None


# I don't know what this is, see function scaled_zero below
SCALED_ZERO_TUP = tTuple[List[int], int, int, int]


@overload
def scaled_zero(mag: SCALED_ZERO_TUP, sign=1) -> MPF_TUP:
    ...
@overload
def scaled_zero(mag: int, sign=1) -> tTuple[SCALED_ZERO_TUP, int]:
    ...
def scaled_zero(mag: tUnion[SCALED_ZERO_TUP, int], sign=1) -> \
        tUnion[MPF_TUP, tTuple[SCALED_ZERO_TUP, int]]:
    """Return an mpf representing a power of two with magnitude ``mag``
    and -1 for precision. Or, if ``mag`` is a scaled_zero tuple, then just
    remove the sign from within the list that it was initially wrapped
    in.

    Examples
    ========

    >>> from sympy.core.evalf import scaled_zero
    >>> from sympy import Float
    >>> z, p = scaled_zero(100)
    >>> z, p
    (([0], 1, 100, 1), -1)
    >>> ok = scaled_zero(z)
    >>> ok
    (0, 1, 100, 1)
    >>> Float(ok)
    1.26765060022823e+30
    >>> Float(ok, p)
    0.e+30
    >>> ok, p = scaled_zero(100, -1)
    >>> Float(scaled_zero(ok), p)
    -0.e+30
    """
    if isinstance(mag, tuple) and len(mag) == 4 and iszero(mag, scaled=True):
        return (mag[0][0],) + mag[1:]
    elif isinstance(mag, SYMPY_INTS):
        if sign not in [-1, 1]:
            raise ValueError('sign must be +/-1')
        rv, p = mpf_shift(fone, mag), -1
        s = 0 if sign == 1 else 1
        rv = ([s],) + rv[1:]
        return rv, p
    else:
        raise ValueError('scaled zero expects int or scaled_zero tuple.')


def iszero(mpf: tUnion[MPF_TUP, SCALED_ZERO_TUP, None], scaled=False) -> Optional[bool]:
    if not scaled:
        return not mpf or not mpf[1] and not mpf[-1]
    return mpf and isinstance(mpf[0], list) and mpf[1] == mpf[-1] == 1


def complex_accuracy(result: TMP_RES) -> tUnion[int, Any]:
    """
    Returns relative accuracy of a complex number with given accuracies
    for the real and imaginary parts. The relative accuracy is defined
    in the complex norm sense as ||z|+|error|| / |z| where error
    is equal to (real absolute error) + (imag absolute error)*i.

    The full expression for the (logarithmic) error can be approximated
    easily by using the max norm to approximate the complex norm.

    In the worst case (re and im equal), this is wrong by a factor
    sqrt(2), or by log2(sqrt(2)) = 0.5 bit.
    """
    if result is S.ComplexInfinity:
        return INF
    re, im, re_acc, im_acc = result
    if not im:
        if not re:
            return INF
        return re_acc
    if not re:
        return im_acc
    re_size = fastlog(re)
    im_size = fastlog(im)
    absolute_error = max(re_size - re_acc, im_size - im_acc)
    relative_error = absolute_error - max(re_size, im_size)
    return -relative_error


def get_abs(expr: 'Expr', prec: int, options: OPT_DICT) -> TMP_RES:
    result = evalf(expr, prec + 2, options)
    if result is S.ComplexInfinity:
        return finf, None, prec, None
    re, im, re_acc, im_acc = result
    if not re:
        re, re_acc, im, im_acc = im, im_acc, re, re_acc
    if im:
        if expr.is_number:
            abs_expr, _, acc, _ = evalf(abs(N(expr, prec + 2)),
                                        prec + 2, options)
            return abs_expr, None, acc, None
        else:
            if 'subs' in options:
                return libmp.mpc_abs((re, im), prec), None, re_acc, None
            return abs(expr), None, prec, None
    elif re:
        return mpf_abs(re), None, re_acc, None
    else:
        return None, None, None, None


def get_complex_part(expr: 'Expr', no: int, prec: int, options: OPT_DICT) -> TMP_RES:
    """no = 0 for real part, no = 1 for imaginary part"""
    workprec = prec
    i = 0
    while 1:
        res = evalf(expr, workprec, options)
        if res is S.ComplexInfinity:
            return fnan, None, prec, None
        value, accuracy = res[no::2]
        # XXX is the last one correct? Consider re((1+I)**2).n()
        if (not value) or accuracy >= prec or -value[2] > prec:
            return value, None, accuracy, None
        workprec += max(30, 2**i)
        i += 1


def evalf_abs(expr: 'Abs', prec: int, options: OPT_DICT) -> TMP_RES:
    return get_abs(expr.args[0], prec, options)


def evalf_re(expr: 're', prec: int, options: OPT_DICT) -> TMP_RES:
    return get_complex_part(expr.args[0], 0, prec, options)


def evalf_im(expr: 'im', prec: int, options: OPT_DICT) -> TMP_RES:
    return get_complex_part(expr.args[0], 1, prec, options)


def finalize_complex(re: MPF_TUP, im: MPF_TUP, prec: int) -> TMP_RES:
    if re == fzero and im == fzero:
        raise ValueError("got complex zero with unknown accuracy")
    elif re == fzero:
        return None, im, None, prec
    elif im == fzero:
        return re, None, prec, None

    size_re = fastlog(re)
    size_im = fastlog(im)
    if size_re > size_im:
        re_acc = prec
        im_acc = prec + min(-(size_re - size_im), 0)
    else:
        im_acc = prec
        re_acc = prec + min(-(size_im - size_re), 0)
    return re, im, re_acc, im_acc


def chop_parts(value: TMP_RES, prec: int) -> TMP_RES:
    """
    Chop off tiny real or complex parts.
    """
    if value is S.ComplexInfinity:
        return value
    re, im, re_acc, im_acc = value
    # Method 1: chop based on absolute value
    if re and re not in _infs_nan and (fastlog(re) < -prec + 4):
        re, re_acc = None, None
    if im and im not in _infs_nan and (fastlog(im) < -prec + 4):
        im, im_acc = None, None
    # Method 2: chop if inaccurate and relatively small
    if re and im:
        delta = fastlog(re) - fastlog(im)
        if re_acc < 2 and (delta - re_acc <= -prec + 4):
            re, re_acc = None, None
        if im_acc < 2 and (delta - im_acc >= prec - 4):
            im, im_acc = None, None
    return re, im, re_acc, im_acc


def check_target(expr: 'Expr', result: TMP_RES, prec: int):
    a = complex_accuracy(result)
    if a < prec:
        raise PrecisionExhausted("Failed to distinguish the expression: \n\n%s\n\n"
            "from zero. Try simplifying the input, using chop=True, or providing "
            "a higher maxn for evalf" % (expr))


def get_integer_part(expr: 'Expr', no: int, options: OPT_DICT, return_ints=False) -> \
        tUnion[TMP_RES, tTuple[int, int]]:
    """
    With no = 1, computes ceiling(expr)
    With no = -1, computes floor(expr)

    Note: this function either gives the exact result or signals failure.
    """
    from sympy.functions.elementary.complexes import re, im
    # The expression is likely less than 2^30 or so
    assumed_size = 30
    result = evalf(expr, assumed_size, options)
    if result is S.ComplexInfinity:
        raise ValueError("Cannot get integer part of Complex Infinity")
    ire, iim, ire_acc, iim_acc = result

    # We now know the size, so we can calculate how much extra precision
    # (if any) is needed to get within the nearest integer
    if ire and iim:
        gap = max(fastlog(ire) - ire_acc, fastlog(iim) - iim_acc)
    elif ire:
        gap = fastlog(ire) - ire_acc
    elif iim:
        gap = fastlog(iim) - iim_acc
    else:
        # ... or maybe the expression was exactly zero
        if return_ints:
            return 0, 0
        else:
            return None, None, None, None

    margin = 10

    if gap >= -margin:
        prec = margin + assumed_size + gap
        ire, iim, ire_acc, iim_acc = evalf(
            expr, prec, options)
    else:
        prec = assumed_size

    # We can now easily find the nearest integer, but to find floor/ceil, we
    # must also calculate whether the difference to the nearest integer is
    # positive or negative (which may fail if very close).
    def calc_part(re_im: 'Expr', nexpr: MPF_TUP):
        from .add import Add
        _, _, exponent, _ = nexpr
        is_int = exponent == 0
        nint = int(to_int(nexpr, rnd))
        if is_int:
            # make sure that we had enough precision to distinguish
            # between nint and the re or im part (re_im) of expr that
            # was passed to calc_part
            ire, iim, ire_acc, iim_acc = evalf(
                re_im - nint, 10, options)  # don't need much precision
            assert not iim
            size = -fastlog(ire) + 2  # -ve b/c ire is less than 1
            if size > prec:
                ire, iim, ire_acc, iim_acc = evalf(
                    re_im, size, options)
                assert not iim
                nexpr = ire
            nint = int(to_int(nexpr, rnd))
            _, _, new_exp, _ = ire
            is_int = new_exp == 0
        if not is_int:
            # if there are subs and they all contain integer re/im parts
            # then we can (hopefully) safely substitute them into the
            # expression
            s = options.get('subs', False)
            if s:
                # use strict=False with as_int because we take
                # 2.0 == 2
                def is_int_reim(x):
                    """Check for integer or integer + I*integer."""
                    try:
                        as_int(x, strict=False)
                        return True
                    except ValueError:
                        try:
                            [as_int(i, strict=False) for i in x.as_real_imag()]
                            return True
                        except ValueError:
                            return False

                if all(is_int_reim(v) for v in s.values()):
                    re_im = re_im.subs(s)

            re_im = Add(re_im, -nint, evaluate=False)
            x, _, x_acc, _ = evalf(re_im, 10, options)
            try:
                check_target(re_im, (x, None, x_acc, None), 3)
            except PrecisionExhausted:
                if not re_im.equals(0):
                    raise PrecisionExhausted
                x = fzero
            nint += int(no*(mpf_cmp(x or fzero, fzero) == no))
        nint = from_int(nint)
        return nint, INF

    re_, im_, re_acc, im_acc = None, None, None, None

    if ire:
        re_, re_acc = calc_part(re(expr, evaluate=False), ire)
    if iim:
        im_, im_acc = calc_part(im(expr, evaluate=False), iim)

    if return_ints:
        return int(to_int(re_ or fzero)), int(to_int(im_ or fzero))
    return re_, im_, re_acc, im_acc


def evalf_ceiling(expr: 'ceiling', prec: int, options: OPT_DICT) -> TMP_RES:
    return get_integer_part(expr.args[0], 1, options)


def evalf_floor(expr: 'floor', prec: int, options: OPT_DICT) -> TMP_RES:
    return get_integer_part(expr.args[0], -1, options)


def evalf_float(expr: 'Float', prec: int, options: OPT_DICT) -> TMP_RES:
    return expr._mpf_, None, prec, None


def evalf_rational(expr: 'Rational', prec: int, options: OPT_DICT) -> TMP_RES:
    return from_rational(expr.p, expr.q, prec), None, prec, None


def evalf_integer(expr: 'Integer', prec: int, options: OPT_DICT) -> TMP_RES:
    return from_int(expr.p, prec), None, prec, None

#----------------------------------------------------------------------------#
#                                                                            #
#                            Arithmetic operations                           #
#                                                                            #
#----------------------------------------------------------------------------#


def add_terms(terms: list, prec: int, target_prec: int) -> \
        tTuple[tUnion[MPF_TUP, SCALED_ZERO_TUP, None], Optional[int]]:
    """
    Helper for evalf_add. Adds a list of (mpfval, accuracy) terms.

    Returns
    =======

    - None, None if there are no non-zero terms;
    - terms[0] if there is only 1 term;
    - scaled_zero if the sum of the terms produces a zero by cancellation
      e.g. mpfs representing 1 and -1 would produce a scaled zero which need
      special handling since they are not actually zero and they are purposely
      malformed to ensure that they cannot be used in anything but accuracy
      calculations;
    - a tuple that is scaled to target_prec that corresponds to the
      sum of the terms.

    The returned mpf tuple will be normalized to target_prec; the input
    prec is used to define the working precision.

    XXX explain why this is needed and why one cannot just loop using mpf_add
    """

    terms = [t for t in terms if not iszero(t[0])]
    if not terms:
        return None, None
    elif len(terms) == 1:
        return terms[0]

    # see if any argument is NaN or oo and thus warrants a special return
    special = []
    from .numbers import Float
    for t in terms:
        arg = Float._new(t[0], 1)
        if arg is S.NaN or arg.is_infinite:
            special.append(arg)
    if special:
        from .add import Add
        rv = evalf(Add(*special), prec + 4, {})
        return rv[0], rv[2]

    working_prec = 2*prec
    sum_man, sum_exp = 0, 0
    absolute_err: List[int] = []

    for x, accuracy in terms:
        sign, man, exp, bc = x
        if sign:
            man = -man
        absolute_err.append(bc + exp - accuracy)
        delta = exp - sum_exp
        if exp >= sum_exp:
            # x much larger than existing sum?
            # first: quick test
            if ((delta > working_prec) and
                ((not sum_man) or
                 delta - bitcount(abs(sum_man)) > working_prec)):
                sum_man = man
                sum_exp = exp
            else:
                sum_man += (man << delta)
        else:
            delta = -delta
            # x much smaller than existing sum?
            if delta - bc > working_prec:
                if not sum_man:
                    sum_man, sum_exp = man, exp
            else:
                sum_man = (sum_man << delta) + man
                sum_exp = exp
    absolute_error = max(absolute_err)
    if not sum_man:
        return scaled_zero(absolute_error)
    if sum_man < 0:
        sum_sign = 1
        sum_man = -sum_man
    else:
        sum_sign = 0
    sum_bc = bitcount(sum_man)
    sum_accuracy = sum_exp + sum_bc - absolute_error
    r = normalize(sum_sign, sum_man, sum_exp, sum_bc, target_prec,
        rnd), sum_accuracy
    return r


def evalf_add(v: 'Add', prec: int, options: OPT_DICT) -> TMP_RES:
    res = pure_complex(v)
    if res:
        h, c = res
        re, _, re_acc, _ = evalf(h, prec, options)
        im, _, im_acc, _ = evalf(c, prec, options)
        return re, im, re_acc, im_acc

    oldmaxprec = options.get('maxprec', DEFAULT_MAXPREC)

    i = 0
    target_prec = prec
    while 1:
        options['maxprec'] = min(oldmaxprec, 2*prec)

        terms = [evalf(arg, prec + 10, options) for arg in v.args]
        n = terms.count(S.ComplexInfinity)
        if n >= 2:
            return fnan, None, prec, None
        re, re_acc = add_terms(
            [a[0::2] for a in terms if isinstance(a, tuple) and a[0]], prec, target_prec)
        im, im_acc = add_terms(
            [a[1::2] for a in terms if isinstance(a, tuple) and a[1]], prec, target_prec)
        if n == 1:
            if re in (finf, fninf, fnan) or im in (finf, fninf, fnan):
                return fnan, None, prec, None
            return S.ComplexInfinity
        acc = complex_accuracy((re, im, re_acc, im_acc))
        if acc >= target_prec:
            if options.get('verbose'):
                print("ADD: wanted", target_prec, "accurate bits, got", re_acc, im_acc)
            break
        else:
            if (prec - target_prec) > options['maxprec']:
                break

            prec = prec + max(10 + 2**i, target_prec - acc)
            i += 1
            if options.get('verbose'):
                print("ADD: restarting with prec", prec)

    options['maxprec'] = oldmaxprec
    if iszero(re, scaled=True):
        re = scaled_zero(re)
    if iszero(im, scaled=True):
        im = scaled_zero(im)
    return re, im, re_acc, im_acc


def evalf_mul(v: 'Mul', prec: int, options: OPT_DICT) -> TMP_RES:
    res = pure_complex(v)
    if res:
        # the only pure complex that is a mul is h*I
        _, h = res
        im, _, im_acc, _ = evalf(h, prec, options)
        return None, im, None, im_acc
    args = list(v.args)

    # see if any argument is NaN or oo and thus warrants a special return
    has_zero = False
    special = []
    from .numbers import Float
    for arg in args:
        result = evalf(arg, prec, options)
        if result is S.ComplexInfinity:
            special.append(result)
            continue
        if result[0] is None:
            if result[1] is None:
                has_zero = True
            continue
        num = Float._new(result[0], 1)
        if num is S.NaN:
            return fnan, None, prec, None
        if num.is_infinite:
            special.append(num)
    if special:
        if has_zero:
            return fnan, None, prec, None
        from .mul import Mul
        return evalf(Mul(*special), prec + 4, {})
    if has_zero:
        return None, None, None, None

    # With guard digits, multiplication in the real case does not destroy
    # accuracy. This is also true in the complex case when considering the
    # total accuracy; however accuracy for the real or imaginary parts
    # separately may be lower.
    acc = prec

    # XXX: big overestimate
    working_prec = prec + len(args) + 5

    # Empty product is 1
    start = man, exp, bc = MPZ(1), 0, 1

    # First, we multiply all pure real or pure imaginary numbers.
    # direction tells us that the result should be multiplied by
    # I**direction; all other numbers get put into complex_factors
    # to be multiplied out after the first phase.
    last = len(args)
    direction = 0
    args.append(S.One)
    complex_factors = []

    for i, arg in enumerate(args):
        if i != last and pure_complex(arg):
            args[-1] = (args[-1]*arg).expand()
            continue
        elif i == last and arg is S.One:
            continue
        re, im, re_acc, im_acc = evalf(arg, working_prec, options)
        if re and im:
            complex_factors.append((re, im, re_acc, im_acc))
            continue
        elif re:
            (s, m, e, b), w_acc = re, re_acc
        elif im:
            (s, m, e, b), w_acc = im, im_acc
            direction += 1
        else:
            return None, None, None, None
        direction += 2*s
        man *= m
        exp += e
        bc += b
        while bc > 3*working_prec:
            man >>= working_prec
            exp += working_prec
            bc -= working_prec
        acc = min(acc, w_acc)
    sign = (direction & 2) >> 1
    if not complex_factors:
        v = normalize(sign, man, exp, bitcount(man), prec, rnd)
        # multiply by i
        if direction & 1:
            return None, v, None, acc
        else:
            return v, None, acc, None
    else:
        # initialize with the first term
        if (man, exp, bc) != start:
            # there was a real part; give it an imaginary part
            re, im = (sign, man, exp, bitcount(man)), (0, MPZ(0), 0, 0)
            i0 = 0
        else:
            # there is no real part to start (other than the starting 1)
            wre, wim, wre_acc, wim_acc = complex_factors[0]
            acc = min(acc,
                      complex_accuracy((wre, wim, wre_acc, wim_acc)))
            re = wre
            im = wim
            i0 = 1

        for wre, wim, wre_acc, wim_acc in complex_factors[i0:]:
            # acc is the overall accuracy of the product; we aren't
            # computing exact accuracies of the product.
            acc = min(acc,
                      complex_accuracy((wre, wim, wre_acc, wim_acc)))

            use_prec = working_prec
            A = mpf_mul(re, wre, use_prec)
            B = mpf_mul(mpf_neg(im), wim, use_prec)
            C = mpf_mul(re, wim, use_prec)
            D = mpf_mul(im, wre, use_prec)
            re = mpf_add(A, B, use_prec)
            im = mpf_add(C, D, use_prec)
        if options.get('verbose'):
            print("MUL: wanted", prec, "accurate bits, got", acc)
        # multiply by I
        if direction & 1:
            re, im = mpf_neg(im), re
        return re, im, acc, acc


def evalf_pow(v: 'Pow', prec: int, options) -> TMP_RES:

    target_prec = prec
    base, exp = v.args

    # We handle x**n separately. This has two purposes: 1) it is much
    # faster, because we avoid calling evalf on the exponent, and 2) it
    # allows better handling of real/imaginary parts that are exactly zero
    if exp.is_Integer:
        p: int = exp.p  # type: ignore
        # Exact
        if not p:
            return fone, None, prec, None
        # Exponentiation by p magnifies relative error by |p|, so the
        # base must be evaluated with increased precision if p is large
        prec += int(math.log2(abs(p)))
        result = evalf(base, prec + 5, options)
        if result is S.ComplexInfinity:
            if p < 0:
                return None, None, None, None
            return result
        re, im, re_acc, im_acc = result
        # Real to integer power
        if re and not im:
            return mpf_pow_int(re, p, target_prec), None, target_prec, None
        # (x*I)**n = I**n * x**n
        if im and not re:
            z = mpf_pow_int(im, p, target_prec)
            case = p % 4
            if case == 0:
                return z, None, target_prec, None
            if case == 1:
                return None, z, None, target_prec
            if case == 2:
                return mpf_neg(z), None, target_prec, None
            if case == 3:
                return None, mpf_neg(z), None, target_prec
        # Zero raised to an integer power
        if not re:
            if p < 0:
                return S.ComplexInfinity
            return None, None, None, None
        # General complex number to arbitrary integer power
        re, im = libmp.mpc_pow_int((re, im), p, prec)
        # Assumes full accuracy in input
        return finalize_complex(re, im, target_prec)

    result = evalf(base, prec + 5, options)
    if result is S.ComplexInfinity:
        if exp.is_Rational:
            if exp < 0:
                return None, None, None, None
            return result
        raise NotImplementedError

    # Pure square root
    if exp is S.Half:
        xre, xim, _, _ = result
        # General complex square root
        if xim:
            re, im = libmp.mpc_sqrt((xre or fzero, xim), prec)
            return finalize_complex(re, im, prec)
        if not xre:
            return None, None, None, None
        # Square root of a negative real number
        if mpf_lt(xre, fzero):
            return None, mpf_sqrt(mpf_neg(xre), prec), None, prec
        # Positive square root
        return mpf_sqrt(xre, prec), None, prec, None

    # We first evaluate the exponent to find its magnitude
    # This determines the working precision that must be used
    prec += 10
    result = evalf(exp, prec, options)
    if result is S.ComplexInfinity:
        return fnan, None, prec, None
    yre, yim, _, _ = result
    # Special cases: x**0
    if not (yre or yim):
        return fone, None, prec, None

    ysize = fastlog(yre)
    # Restart if too big
    # XXX: prec + ysize might exceed maxprec
    if ysize > 5:
        prec += ysize
        yre, yim, _, _ = evalf(exp, prec, options)

    # Pure exponential function; no need to evalf the base
    if base is S.Exp1:
        if yim:
            re, im = libmp.mpc_exp((yre or fzero, yim), prec)
            return finalize_complex(re, im, target_prec)
        return mpf_exp(yre, target_prec), None, target_prec, None

    xre, xim, _, _ = evalf(base, prec + 5, options)
    # 0**y
    if not (xre or xim):
        if yim:
            return fnan, None, prec, None
        if yre[0] == 1:  # y < 0
            return S.ComplexInfinity
        return None, None, None, None

    # (real ** complex) or (complex ** complex)
    if yim:
        re, im = libmp.mpc_pow(
            (xre or fzero, xim or fzero), (yre or fzero, yim),
            target_prec)
        return finalize_complex(re, im, target_prec)
    # complex ** real
    if xim:
        re, im = libmp.mpc_pow_mpf((xre or fzero, xim), yre, target_prec)
        return finalize_complex(re, im, target_prec)
    # negative ** real
    elif mpf_lt(xre, fzero):
        re, im = libmp.mpc_pow_mpf((xre, fzero), yre, target_prec)
        return finalize_complex(re, im, target_prec)
    # positive ** real
    else:
        return mpf_pow(xre, yre, target_prec), None, target_prec, None


#----------------------------------------------------------------------------#
#                                                                            #
#                            Special functions                               #
#                                                                            #
#----------------------------------------------------------------------------#


def evalf_exp(expr: 'exp', prec: int, options: OPT_DICT) -> TMP_RES:
    from .power import Pow
    return evalf_pow(Pow(S.Exp1, expr.exp, evaluate=False), prec, options)


def evalf_trig(v: 'Expr', prec: int, options: OPT_DICT) -> TMP_RES:
    """
    This function handles sin and cos of complex arguments.

    TODO: should also handle tan of complex arguments.
    """
    from sympy.functions.elementary.trigonometric import cos, sin
    if isinstance(v, cos):
        func = mpf_cos
    elif isinstance(v, sin):
        func = mpf_sin
    else:
        raise NotImplementedError
    arg = v.args[0]
    # 20 extra bits is possibly overkill. It does make the need
    # to restart very unlikely
    xprec = prec + 20
    re, im, re_acc, im_acc = evalf(arg, xprec, options)
    if im:
        if 'subs' in options:
            v = v.subs(options['subs'])
        return evalf(v._eval_evalf(prec), prec, options)
    if not re:
        if isinstance(v, cos):
            return fone, None, prec, None
        elif isinstance(v, sin):
            return None, None, None, None
        else:
            raise NotImplementedError
    # For trigonometric functions, we are interested in the
    # fixed-point (absolute) accuracy of the argument.
    xsize = fastlog(re)
    # Magnitude <= 1.0. OK to compute directly, because there is no
    # danger of hitting the first root of cos (with sin, magnitude
    # <= 2.0 would actually be ok)
    if xsize < 1:
        return func(re, prec, rnd), None, prec, None
    # Very large
    if xsize >= 10:
        xprec = prec + xsize
        re, im, re_acc, im_acc = evalf(arg, xprec, options)
    # Need to repeat in case the argument is very close to a
    # multiple of pi (or pi/2), hitting close to a root
    while 1:
        y = func(re, prec, rnd)
        ysize = fastlog(y)
        gap = -ysize
        accuracy = (xprec - xsize) - gap
        if accuracy < prec:
            if options.get('verbose'):
                print("SIN/COS", accuracy, "wanted", prec, "gap", gap)
                print(to_str(y, 10))
            if xprec > options.get('maxprec', DEFAULT_MAXPREC):
                return y, None, accuracy, None
            xprec += gap
            re, im, re_acc, im_acc = evalf(arg, xprec, options)
            continue
        else:
            return y, None, prec, None


def evalf_log(expr: 'log', prec: int, options: OPT_DICT) -> TMP_RES:
    if len(expr.args)>1:
        expr = expr.doit()
        return evalf(expr, prec, options)
    arg = expr.args[0]
    workprec = prec + 10
    result = evalf(arg, workprec, options)
    if result is S.ComplexInfinity:
        return result
    xre, xim, xacc, _ = result

    # evalf can return NoneTypes if chop=True
    # issue 18516, 19623
    if xre is xim is None:
        # Dear reviewer, I do not know what -inf is;
        # it looks to be (1, 0, -789, -3)
        # but I'm not sure in general,
        # so we just let mpmath figure
        # it out by taking log of 0 directly.
        # It would be better to return -inf instead.
        xre = fzero

    if xim:
        from sympy.functions.elementary.complexes import Abs
        from sympy.functions.elementary.exponential import log

        # XXX: use get_abs etc instead
        re = evalf_log(
            log(Abs(arg, evaluate=False), evaluate=False), prec, options)
        im = mpf_atan2(xim, xre or fzero, prec)
        return re[0], im, re[2], prec

    imaginary_term = (mpf_cmp(xre, fzero) < 0)

    re = mpf_log(mpf_abs(xre), prec, rnd)
    size = fastlog(re)
    if prec - size > workprec and re != fzero:
        from .add import Add
        # We actually need to compute 1+x accurately, not x
        add = Add(S.NegativeOne, arg, evaluate=False)
        xre, xim, _, _ = evalf_add(add, prec, options)
        prec2 = workprec - fastlog(xre)
        # xre is now x - 1 so we add 1 back here to calculate x
        re = mpf_log(mpf_abs(mpf_add(xre, fone, prec2)), prec, rnd)

    re_acc = prec

    if imaginary_term:
        return re, mpf_pi(prec), re_acc, prec
    else:
        return re, None, re_acc, None


def evalf_atan(v: 'atan', prec: int, options: OPT_DICT) -> TMP_RES:
    arg = v.args[0]
    xre, xim, reacc, imacc = evalf(arg, prec + 5, options)
    if xre is xim is None:
        return (None,)*4
    if xim:
        raise NotImplementedError
    return mpf_atan(xre, prec, rnd), None, prec, None


def evalf_subs(prec: int, subs: dict) -> dict:
    """ Change all Float entries in `subs` to have precision prec. """
    newsubs = {}
    for a, b in subs.items():
        b = S(b)
        if b.is_Float:
            b = b._eval_evalf(prec)
        newsubs[a] = b
    return newsubs


def evalf_piecewise(expr: 'Expr', prec: int, options: OPT_DICT) -> TMP_RES:
    from .numbers import Float, Integer
    if 'subs' in options:
        expr = expr.subs(evalf_subs(prec, options['subs']))
        newopts = options.copy()
        del newopts['subs']
        if hasattr(expr, 'func'):
            return evalf(expr, prec, newopts)
        if isinstance(expr, float):
            return evalf(Float(expr), prec, newopts)
        if isinstance(expr, int):
            return evalf(Integer(expr), prec, newopts)

    # We still have undefined symbols
    raise NotImplementedError


def evalf_alg_num(a: 'AlgebraicNumber', prec: int, options: OPT_DICT) -> TMP_RES:
    return evalf(a.to_root(), prec, options)

#----------------------------------------------------------------------------#
#                                                                            #
#                            High-level operations                           #
#                                                                            #
#----------------------------------------------------------------------------#


def as_mpmath(x: Any, prec: int, options: OPT_DICT) -> tUnion[mpc, mpf]:
    from .numbers import Infinity, NegativeInfinity, Zero
    x = sympify(x)
    if isinstance(x, Zero) or x == 0.0:
        return mpf(0)
    if isinstance(x, Infinity):
        return mpf('inf')
    if isinstance(x, NegativeInfinity):
        return mpf('-inf')
    # XXX
    result = evalf(x, prec, options)
    return quad_to_mpmath(result)


def do_integral(expr: 'Integral', prec: int, options: OPT_DICT) -> TMP_RES:
    func = expr.args[0]
    x, xlow, xhigh = expr.args[1]
    if xlow == xhigh:
        xlow = xhigh = 0
    elif x not in func.free_symbols:
        # only the difference in limits matters in this case
        # so if there is a symbol in common that will cancel
        # out when taking the difference, then use that
        # difference
        if xhigh.free_symbols & xlow.free_symbols:
            diff = xhigh - xlow
            if diff.is_number:
                xlow, xhigh = 0, diff

    oldmaxprec = options.get('maxprec', DEFAULT_MAXPREC)
    options['maxprec'] = min(oldmaxprec, 2*prec)

    with workprec(prec + 5):
        xlow = as_mpmath(xlow, prec + 15, options)
        xhigh = as_mpmath(xhigh, prec + 15, options)

        # Integration is like summation, and we can phone home from
        # the integrand function to update accuracy summation style
        # Note that this accuracy is inaccurate, since it fails
        # to account for the variable quadrature weights,
        # but it is better than nothing

        from sympy.functions.elementary.trigonometric import cos, sin
        from .symbol import Wild

        have_part = [False, False]
        max_real_term: tUnion[float, int] = MINUS_INF
        max_imag_term: tUnion[float, int] = MINUS_INF

        def f(t: 'Expr') -> tUnion[mpc, mpf]:
            nonlocal max_real_term, max_imag_term
            re, im, re_acc, im_acc = evalf(func, mp.prec, {'subs': {x: t}})

            have_part[0] = re or have_part[0]
            have_part[1] = im or have_part[1]

            max_real_term = max(max_real_term, fastlog(re))
            max_imag_term = max(max_imag_term, fastlog(im))

            if im:
                return mpc(re or fzero, im)
            return mpf(re or fzero)

        if options.get('quad') == 'osc':
            A = Wild('A', exclude=[x])
            B = Wild('B', exclude=[x])
            D = Wild('D')
            m = func.match(cos(A*x + B)*D)
            if not m:
                m = func.match(sin(A*x + B)*D)
            if not m:
                raise ValueError("An integrand of the form sin(A*x+B)*f(x) "
                  "or cos(A*x+B)*f(x) is required for oscillatory quadrature")
            period = as_mpmath(2*S.Pi/m[A], prec + 15, options)
            result = quadosc(f, [xlow, xhigh], period=period)
            # XXX: quadosc does not do error detection yet
            quadrature_error = MINUS_INF
        else:
            result, quadrature_err = quadts(f, [xlow, xhigh], error=1)
            quadrature_error = fastlog(quadrature_err._mpf_)

    options['maxprec'] = oldmaxprec

    if have_part[0]:
        re: Optional[MPF_TUP] = result.real._mpf_
        re_acc: Optional[int]
        if re == fzero:
            re_s, re_acc = scaled_zero(int(-max(prec, max_real_term, quadrature_error)))
            re = scaled_zero(re_s)  # handled ok in evalf_integral
        else:
            re_acc = int(-max(max_real_term - fastlog(re) - prec, quadrature_error))
    else:
        re, re_acc = None, None

    if have_part[1]:
        im: Optional[MPF_TUP] = result.imag._mpf_
        im_acc: Optional[int]
        if im == fzero:
            im_s, im_acc = scaled_zero(int(-max(prec, max_imag_term, quadrature_error)))
            im = scaled_zero(im_s)  # handled ok in evalf_integral
        else:
            im_acc = int(-max(max_imag_term - fastlog(im) - prec, quadrature_error))
    else:
        im, im_acc = None, None

    result = re, im, re_acc, im_acc
    return result


def evalf_integral(expr: 'Integral', prec: int, options: OPT_DICT) -> TMP_RES:
    limits = expr.limits
    if len(limits) != 1 or len(limits[0]) != 3:
        raise NotImplementedError
    workprec = prec
    i = 0
    maxprec = options.get('maxprec', INF)
    while 1:
        result = do_integral(expr, workprec, options)
        accuracy = complex_accuracy(result)
        if accuracy >= prec:  # achieved desired precision
            break
        if workprec >= maxprec:  # can't increase accuracy any more
            break
        if accuracy == -1:
            # maybe the answer really is zero and maybe we just haven't increased
            # the precision enough. So increase by doubling to not take too long
            # to get to maxprec.
            workprec *= 2
        else:
            workprec += max(prec, 2**i)
        workprec = min(workprec, maxprec)
        i += 1
    return result


def check_convergence(numer: 'Expr', denom: 'Expr', n: 'Symbol') -> tTuple[int, Any, Any]:
    """
    Returns
    =======

    (h, g, p) where
    -- h is:
        > 0 for convergence of rate 1/factorial(n)**h
        < 0 for divergence of rate factorial(n)**(-h)
        = 0 for geometric or polynomial convergence or divergence

    -- abs(g) is:
        > 1 for geometric convergence of rate 1/h**n
        < 1 for geometric divergence of rate h**n
        = 1 for polynomial convergence or divergence

        (g < 0 indicates an alternating series)

    -- p is:
        > 1 for polynomial convergence of rate 1/n**h
        <= 1 for polynomial divergence of rate n**(-h)

    """
    from sympy.polys.polytools import Poly
    npol = Poly(numer, n)
    dpol = Poly(denom, n)
    p = npol.degree()
    q = dpol.degree()
    rate = q - p
    if rate:
        return rate, None, None
    constant = dpol.LC() / npol.LC()
    from .numbers import equal_valued
    if not equal_valued(abs(constant), 1):
        return rate, constant, None
    if npol.degree() == dpol.degree() == 0:
        return rate, constant, 0
    pc = npol.all_coeffs()[1]
    qc = dpol.all_coeffs()[1]
    return rate, constant, (qc - pc)/dpol.LC()


def hypsum(expr: 'Expr', n: 'Symbol', start: int, prec: int) -> mpf:
    """
    Sum a rapidly convergent infinite hypergeometric series with
    given general term, e.g. e = hypsum(1/factorial(n), n). The
    quotient between successive terms must be a quotient of integer
    polynomials.
    """
    from .numbers import Float, equal_valued
    from sympy.simplify.simplify import hypersimp

    if prec == float('inf'):
        raise NotImplementedError('does not support inf prec')

    if start:
        expr = expr.subs(n, n + start)
    hs = hypersimp(expr, n)
    if hs is None:
        raise NotImplementedError("a hypergeometric series is required")
    num, den = hs.as_numer_denom()

    func1 = lambdify(n, num)
    func2 = lambdify(n, den)

    h, g, p = check_convergence(num, den, n)

    if h < 0:
        raise ValueError("Sum diverges like (n!)^%i" % (-h))

    term = expr.subs(n, 0)
    if not term.is_Rational:
        raise NotImplementedError("Non rational term functionality is not implemented.")

    # Direct summation if geometric or faster
    if h > 0 or (h == 0 and abs(g) > 1):
        term = (MPZ(term.p) << prec) // term.q
        s = term
        k = 1
        while abs(term) > 5:
            term *= MPZ(func1(k - 1))
            term //= MPZ(func2(k - 1))
            s += term
            k += 1
        return from_man_exp(s, -prec)
    else:
        alt = g < 0
        if abs(g) < 1:
            raise ValueError("Sum diverges like (%i)^n" % abs(1/g))
        if p < 1 or (equal_valued(p, 1) and not alt):
            raise ValueError("Sum diverges like n^%i" % (-p))
        # We have polynomial convergence: use Richardson extrapolation
        vold = None
        ndig = prec_to_dps(prec)
        while True:
            # Need to use at least quad precision because a lot of cancellation
            # might occur in the extrapolation process; we check the answer to
            # make sure that the desired precision has been reached, too.
            prec2 = 4*prec
            term0 = (MPZ(term.p) << prec2) // term.q

            def summand(k, _term=[term0]):
                if k:
                    k = int(k)
                    _term[0] *= MPZ(func1(k - 1))
                    _term[0] //= MPZ(func2(k - 1))
                return make_mpf(from_man_exp(_term[0], -prec2))

            with workprec(prec):
                v = nsum(summand, [0, mpmath_inf], method='richardson')
            vf = Float(v, ndig)
            if vold is not None and vold == vf:
                break
            prec += prec  # double precision each time
            vold = vf

        return v._mpf_


def evalf_prod(expr: 'Product', prec: int, options: OPT_DICT) -> TMP_RES:
    if all((l[1] - l[2]).is_Integer for l in expr.limits):
        result = evalf(expr.doit(), prec=prec, options=options)
    else:
        from sympy.concrete.summations import Sum
        result = evalf(expr.rewrite(Sum), prec=prec, options=options)
    return result


def evalf_sum(expr: 'Sum', prec: int, options: OPT_DICT) -> TMP_RES:
    from .numbers import Float
    if 'subs' in options:
        expr = expr.subs(options['subs'])
    func = expr.function
    limits = expr.limits
    if len(limits) != 1 or len(limits[0]) != 3:
        raise NotImplementedError
    if func.is_zero:
        return None, None, prec, None
    prec2 = prec + 10
    try:
        n, a, b = limits[0]
        if b is not S.Infinity or a is S.NegativeInfinity or a != int(a):
            raise NotImplementedError
        # Use fast hypergeometric summation if possible
        v = hypsum(func, n, int(a), prec2)
        delta = prec - fastlog(v)
        if fastlog(v) < -10:
            v = hypsum(func, n, int(a), delta)
        return v, None, min(prec, delta), None
    except NotImplementedError:
        # Euler-Maclaurin summation for general series
        eps = Float(2.0)**(-prec)
        for i in range(1, 5):
            m = n = 2**i * prec
            s, err = expr.euler_maclaurin(m=m, n=n, eps=eps,
                eval_integral=False)
            err = err.evalf()
            if err is S.NaN:
                raise NotImplementedError
            if err <= eps:
                break
        err = fastlog(evalf(abs(err), 20, options)[0])
        re, im, re_acc, im_acc = evalf(s, prec2, options)
        if re_acc is None:
            re_acc = -err
        if im_acc is None:
            im_acc = -err
        return re, im, re_acc, im_acc


#----------------------------------------------------------------------------#
#                                                                            #
#                            Symbolic interface                              #
#                                                                            #
#----------------------------------------------------------------------------#

def evalf_symbol(x: 'Expr', prec: int, options: OPT_DICT) -> TMP_RES:
    val = options['subs'][x]
    if isinstance(val, mpf):
        if not val:
            return None, None, None, None
        return val._mpf_, None, prec, None
    else:
        if '_cache' not in options:
            options['_cache'] = {}
        cache = options['_cache']
        cached, cached_prec = cache.get(x, (None, MINUS_INF))
        if cached_prec >= prec:
            return cached
        v = evalf(sympify(val), prec, options)
        cache[x] = (v, prec)
        return v


evalf_table: tDict[Type['Expr'], Callable[['Expr', int, OPT_DICT], TMP_RES]] = {}


def _create_evalf_table():
    global evalf_table
    from sympy.concrete.products import Product
    from sympy.concrete.summations import Sum
    from .add import Add
    from .mul import Mul
    from .numbers import Exp1, Float, Half, ImaginaryUnit, Integer, NaN, NegativeOne, One, Pi, Rational, \
        Zero, ComplexInfinity, AlgebraicNumber
    from .power import Pow
    from .symbol import Dummy, Symbol
    from sympy.functions.elementary.complexes import Abs, im, re
    from sympy.functions.elementary.exponential import exp, log
    from sympy.functions.elementary.integers import ceiling, floor
    from sympy.functions.elementary.piecewise import Piecewise
    from sympy.functions.elementary.trigonometric import atan, cos, sin
    from sympy.integrals.integrals import Integral
    evalf_table = {
        Symbol: evalf_symbol,
        Dummy: evalf_symbol,
        Float: evalf_float,
        Rational: evalf_rational,
        Integer: evalf_integer,
        Zero: lambda x, prec, options: (None, None, prec, None),
        One: lambda x, prec, options: (fone, None, prec, None),
        Half: lambda x, prec, options: (fhalf, None, prec, None),
        Pi: lambda x, prec, options: (mpf_pi(prec), None, prec, None),
        Exp1: lambda x, prec, options: (mpf_e(prec), None, prec, None),
        ImaginaryUnit: lambda x, prec, options: (None, fone, None, prec),
        NegativeOne: lambda x, prec, options: (fnone, None, prec, None),
        ComplexInfinity: lambda x, prec, options: S.ComplexInfinity,
        NaN: lambda x, prec, options: (fnan, None, prec, None),

        exp: evalf_exp,

        cos: evalf_trig,
        sin: evalf_trig,

        Add: evalf_add,
        Mul: evalf_mul,
        Pow: evalf_pow,

        log: evalf_log,
        atan: evalf_atan,
        Abs: evalf_abs,

        re: evalf_re,
        im: evalf_im,
        floor: evalf_floor,
        ceiling: evalf_ceiling,

        Integral: evalf_integral,
        Sum: evalf_sum,
        Product: evalf_prod,
        Piecewise: evalf_piecewise,

        AlgebraicNumber: evalf_alg_num,
    }


def evalf(x: 'Expr', prec: int, options: OPT_DICT) -> TMP_RES:
    """
    Evaluate the ``Expr`` instance, ``x``
    to a binary precision of ``prec``. This
    function is supposed to be used internally.

    Parameters
    ==========

    x : Expr
        The formula to evaluate to a float.
    prec : int
        The binary precision that the output should have.
    options : dict
        A dictionary with the same entries as
        ``EvalfMixin.evalf`` and in addition,
        ``maxprec`` which is the maximum working precision.

    Returns
    =======

    An optional tuple, ``(re, im, re_acc, im_acc)``
    which are the real, imaginary, real accuracy
    and imaginary accuracy respectively. ``re`` is
    an mpf value tuple and so is ``im``. ``re_acc``
    and ``im_acc`` are ints.

    NB: all these return values can be ``None``.
    If all values are ``None``, then that represents 0.
    Note that 0 is also represented as ``fzero = (0, 0, 0, 0)``.
    """
    from sympy.functions.elementary.complexes import re as re_, im as im_
    try:
        rf = evalf_table[type(x)]
        r = rf(x, prec, options)
    except KeyError:
        # Fall back to ordinary evalf if possible
        if 'subs' in options:
            x = x.subs(evalf_subs(prec, options['subs']))
        xe = x._eval_evalf(prec)
        if xe is None:
            raise NotImplementedError
        as_real_imag = getattr(xe, "as_real_imag", None)
        if as_real_imag is None:
            raise NotImplementedError # e.g. FiniteSet(-1.0, 1.0).evalf()
        re, im = as_real_imag()
        if re.has(re_) or im.has(im_):
            raise NotImplementedError
        if re == 0.0:
            re = None
            reprec = None
        elif re.is_number:
            re = re._to_mpmath(prec, allow_ints=False)._mpf_
            reprec = prec
        else:
            raise NotImplementedError
        if im == 0.0:
            im = None
            imprec = None
        elif im.is_number:
            im = im._to_mpmath(prec, allow_ints=False)._mpf_
            imprec = prec
        else:
            raise NotImplementedError
        r = re, im, reprec, imprec

    if options.get("verbose"):
        print("### input", x)
        print("### output", to_str(r[0] or fzero, 50) if isinstance(r, tuple) else r)
        print("### raw", r) # r[0], r[2]
        print()
    chop = options.get('chop', False)
    if chop:
        if chop is True:
            chop_prec = prec
        else:
            # convert (approximately) from given tolerance;
            # the formula here will will make 1e-i rounds to 0 for
            # i in the range +/-27 while 2e-i will not be chopped
            chop_prec = int(round(-3.321*math.log10(chop) + 2.5))
            if chop_prec == 3:
                chop_prec -= 1
        r = chop_parts(r, chop_prec)
    if options.get("strict"):
        check_target(x, r, prec)
    return r


def quad_to_mpmath(q, ctx=None):
    """Turn the quad returned by ``evalf`` into an ``mpf`` or ``mpc``. """
    mpc = make_mpc if ctx is None else ctx.make_mpc
    mpf = make_mpf if ctx is None else ctx.make_mpf
    if q is S.ComplexInfinity:
        raise NotImplementedError
    re, im, _, _ = q
    if im:
        if not re:
            re = fzero
        return mpc((re, im))
    elif re:
        return mpf(re)
    else:
        return mpf(fzero)


class EvalfMixin:
    """Mixin class adding evalf capability."""

    __slots__ = ()  # type: tTuple[str, ...]

    def evalf(self, n=15, subs=None, maxn=100, chop=False, strict=False, quad=None, verbose=False):
        """
        Evaluate the given formula to an accuracy of *n* digits.

        Parameters
        ==========

        subs : dict, optional
            Substitute numerical values for symbols, e.g.
            ``subs={x:3, y:1+pi}``. The substitutions must be given as a
            dictionary.

        maxn : int, optional
            Allow a maximum temporary working precision of maxn digits.

        chop : bool or number, optional
            Specifies how to replace tiny real or imaginary parts in
            subresults by exact zeros.

            When ``True`` the chop value defaults to standard precision.

            Otherwise the chop value is used to determine the
            magnitude of "small" for purposes of chopping.

            >>> from sympy import N
            >>> x = 1e-4
            >>> N(x, chop=True)
            0.000100000000000000
            >>> N(x, chop=1e-5)
            0.000100000000000000
            >>> N(x, chop=1e-4)
            0

        strict : bool, optional
            Raise ``PrecisionExhausted`` if any subresult fails to
            evaluate to full accuracy, given the available maxprec.

        quad : str, optional
            Choose algorithm for numerical quadrature. By default,
            tanh-sinh quadrature is used. For oscillatory
            integrals on an infinite interval, try ``quad='osc'``.

        verbose : bool, optional
            Print debug information.

        Notes
        =====

        When Floats are naively substituted into an expression,
        precision errors may adversely affect the result. For example,
        adding 1e16 (a Float) to 1 will truncate to 1e16; if 1e16 is
        then subtracted, the result will be 0.
        That is exactly what happens in the following:

        >>> from sympy.abc import x, y, z
        >>> values = {x: 1e16, y: 1, z: 1e16}
        >>> (x + y - z).subs(values)
        0

        Using the subs argument for evalf is the accurate way to
        evaluate such an expression:

        >>> (x + y - z).evalf(subs=values)
        1.00000000000000
        """
        from .numbers import Float, Number
        n = n if n is not None else 15

        if subs and is_sequence(subs):
            raise TypeError('subs must be given as a dictionary')

        # for sake of sage that doesn't like evalf(1)
        if n == 1 and isinstance(self, Number):
            from .expr import _mag
            rv = self.evalf(2, subs, maxn, chop, strict, quad, verbose)
            m = _mag(rv)
            rv = rv.round(1 - m)
            return rv

        if not evalf_table:
            _create_evalf_table()
        prec = dps_to_prec(n)
        options = {'maxprec': max(prec, int(maxn*LG10)), 'chop': chop,
               'strict': strict, 'verbose': verbose}
        if subs is not None:
            options['subs'] = subs
        if quad is not None:
            options['quad'] = quad
        try:
            result = evalf(self, prec + 4, options)
        except NotImplementedError:
            # Fall back to the ordinary evalf
            if hasattr(self, 'subs') and subs is not None:  # issue 20291
                v = self.subs(subs)._eval_evalf(prec)
            else:
                v = self._eval_evalf(prec)
            if v is None:
                return self
            elif not v.is_number:
                return v
            try:
                # If the result is numerical, normalize it
                result = evalf(v, prec, options)
            except NotImplementedError:
                # Probably contains symbols or unknown functions
                return v
        if result is S.ComplexInfinity:
            return result
        re, im, re_acc, im_acc = result
        if re is S.NaN or im is S.NaN:
            return S.NaN
        if re:
            p = max(min(prec, re_acc), 1)
            re = Float._new(re, p)
        else:
            re = S.Zero
        if im:
            p = max(min(prec, im_acc), 1)
            im = Float._new(im, p)
            return re + im*S.ImaginaryUnit
        else:
            return re

    n = evalf

    def _evalf(self, prec):
        """Helper for evalf. Does the same thing but takes binary precision"""
        r = self._eval_evalf(prec)
        if r is None:
            r = self
        return r

    def _eval_evalf(self, prec):
        return

    def _to_mpmath(self, prec, allow_ints=True):
        # mpmath functions accept ints as input
        errmsg = "cannot convert to mpmath number"
        if allow_ints and self.is_Integer:
            return self.p
        if hasattr(self, '_as_mpf_val'):
            return make_mpf(self._as_mpf_val(prec))
        try:
            result = evalf(self, prec, {})
            return quad_to_mpmath(result)
        except NotImplementedError:
            v = self._eval_evalf(prec)
            if v is None:
                raise ValueError(errmsg)
            if v.is_Float:
                return make_mpf(v._mpf_)
            # Number + Number*I is also fine
            re, im = v.as_real_imag()
            if allow_ints and re.is_Integer:
                re = from_int(re.p)
            elif re.is_Float:
                re = re._mpf_
            else:
                raise ValueError(errmsg)
            if allow_ints and im.is_Integer:
                im = from_int(im.p)
            elif im.is_Float:
                im = im._mpf_
            else:
                raise ValueError(errmsg)
            return make_mpc((re, im))


def N(x, n=15, **options):
    r"""
    Calls x.evalf(n, \*\*options).

    Explanations
    ============

    Both .n() and N() are equivalent to .evalf(); use the one that you like better.
    See also the docstring of .evalf() for information on the options.

    Examples
    ========

    >>> from sympy import Sum, oo, N
    >>> from sympy.abc import k
    >>> Sum(1/k**k, (k, 1, oo))
    Sum(k**(-k), (k, 1, oo))
    >>> N(_, 4)
    1.291

    """
    # by using rational=True, any evaluation of a string
    # will be done using exact values for the Floats
    return sympify(x, rational=True).evalf(n, **options)


def _evalf_with_bounded_error(x: 'Expr', eps: 'Optional[Expr]' = None,
                              m: int = 0,
                              options: Optional[OPT_DICT] = None) -> TMP_RES:
    """
    Evaluate *x* to within a bounded absolute error.

    Parameters
    ==========

    x : Expr
        The quantity to be evaluated.
    eps : Expr, None, optional (default=None)
        Positive real upper bound on the acceptable error.
    m : int, optional (default=0)
        If *eps* is None, then use 2**(-m) as the upper bound on the error.
    options: OPT_DICT
        As in the ``evalf`` function.

    Returns
    =======

    A tuple ``(re, im, re_acc, im_acc)``, as returned by ``evalf``.

    See Also
    ========

    evalf

    """
    if eps is not None:
        if not (eps.is_Rational or eps.is_Float) or not eps > 0:
            raise ValueError("eps must be positive")
        r, _, _, _ = evalf(1/eps, 1, {})
        m = fastlog(r)

    c, d, _, _ = evalf(x, 1, {})
    # Note: If x = a + b*I, then |a| <= 2|c| and |b| <= 2|d|, with equality
    # only in the zero case.
    # If a is non-zero, then |c| = 2**nc for some integer nc, and c has
    # bitcount 1. Therefore 2**fastlog(c) = 2**(nc+1) = 2|c| is an upper bound
    # on |a|. Likewise for b and d.
    nr, ni = fastlog(c), fastlog(d)
    n = max(nr, ni) + 1
    # If x is 0, then n is MINUS_INF, and p will be 1. Otherwise,
    # n - 1 bits get us past the integer parts of a and b, and +1 accounts for
    # the factor of <= sqrt(2) that is |x|/max(|a|, |b|).
    p = max(1, m + n + 1)

    options = options or {}
    return evalf(x, p, options)
