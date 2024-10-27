from __future__ import annotations
from typing import Callable
from itertools import product

from .sympify import _sympify
from .cache import cacheit
from .singleton import S
from .expr import Expr
from .evalf import PrecisionExhausted
from .function import (expand_complex, expand_multinomial,
    expand_mul, _mexpand, PoleError)
from .logic import fuzzy_bool, fuzzy_not, fuzzy_and, fuzzy_or
from .parameters import global_parameters
from .relational import is_gt, is_lt
from .kind import NumberKind, UndefinedKind
from sympy.utilities.iterables import sift
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.misc import as_int
from sympy.multipledispatch import Dispatcher


class Pow(Expr):
    """
    Defines the expression x**y as "x raised to a power y"

    .. deprecated:: 1.7

       Using arguments that aren't subclasses of :class:`~.Expr` in core
       operators (:class:`~.Mul`, :class:`~.Add`, and :class:`~.Pow`) is
       deprecated. See :ref:`non-expr-args-deprecated` for details.

    Singleton definitions involving (0, 1, -1, oo, -oo, I, -I):

    +--------------+---------+-----------------------------------------------+
    | expr         | value   | reason                                        |
    +==============+=========+===============================================+
    | z**0         | 1       | Although arguments over 0**0 exist, see [2].  |
    +--------------+---------+-----------------------------------------------+
    | z**1         | z       |                                               |
    +--------------+---------+-----------------------------------------------+
    | (-oo)**(-1)  | 0       |                                               |
    +--------------+---------+-----------------------------------------------+
    | (-1)**-1     | -1      |                                               |
    +--------------+---------+-----------------------------------------------+
    | S.Zero**-1   | zoo     | This is not strictly true, as 0**-1 may be    |
    |              |         | undefined, but is convenient in some contexts |
    |              |         | where the base is assumed to be positive.     |
    +--------------+---------+-----------------------------------------------+
    | 1**-1        | 1       |                                               |
    +--------------+---------+-----------------------------------------------+
    | oo**-1       | 0       |                                               |
    +--------------+---------+-----------------------------------------------+
    | 0**oo        | 0       | Because for all complex numbers z near        |
    |              |         | 0, z**oo -> 0.                                |
    +--------------+---------+-----------------------------------------------+
    | 0**-oo       | zoo     | This is not strictly true, as 0**oo may be    |
    |              |         | oscillating between positive and negative     |
    |              |         | values or rotating in the complex plane.      |
    |              |         | It is convenient, however, when the base      |
    |              |         | is positive.                                  |
    +--------------+---------+-----------------------------------------------+
    | 1**oo        | nan     | Because there are various cases where         |
    | 1**-oo       |         | lim(x(t),t)=1, lim(y(t),t)=oo (or -oo),       |
    |              |         | but lim( x(t)**y(t), t) != 1.  See [3].       |
    +--------------+---------+-----------------------------------------------+
    | b**zoo       | nan     | Because b**z has no limit as z -> zoo         |
    +--------------+---------+-----------------------------------------------+
    | (-1)**oo     | nan     | Because of oscillations in the limit.         |
    | (-1)**(-oo)  |         |                                               |
    +--------------+---------+-----------------------------------------------+
    | oo**oo       | oo      |                                               |
    +--------------+---------+-----------------------------------------------+
    | oo**-oo      | 0       |                                               |
    +--------------+---------+-----------------------------------------------+
    | (-oo)**oo    | nan     |                                               |
    | (-oo)**-oo   |         |                                               |
    +--------------+---------+-----------------------------------------------+
    | oo**I        | nan     | oo**e could probably be best thought of as    |
    | (-oo)**I     |         | the limit of x**e for real x as x tends to    |
    |              |         | oo. If e is I, then the limit does not exist  |
    |              |         | and nan is used to indicate that.             |
    +--------------+---------+-----------------------------------------------+
    | oo**(1+I)    | zoo     | If the real part of e is positive, then the   |
    | (-oo)**(1+I) |         | limit of abs(x**e) is oo. So the limit value  |
    |              |         | is zoo.                                       |
    +--------------+---------+-----------------------------------------------+
    | oo**(-1+I)   | 0       | If the real part of e is negative, then the   |
    | -oo**(-1+I)  |         | limit is 0.                                   |
    +--------------+---------+-----------------------------------------------+

    Because symbolic computations are more flexible than floating point
    calculations and we prefer to never return an incorrect answer,
    we choose not to conform to all IEEE 754 conventions.  This helps
    us avoid extra test-case code in the calculation of limits.

    See Also
    ========

    sympy.core.numbers.Infinity
    sympy.core.numbers.NegativeInfinity
    sympy.core.numbers.NaN

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Exponentiation
    .. [2] https://en.wikipedia.org/wiki/Zero_to_the_power_of_zero
    .. [3] https://en.wikipedia.org/wiki/Indeterminate_forms

    """
    is_Pow = True

    __slots__ = ('is_commutative',)

    args: tuple[Expr, Expr]
    _args: tuple[Expr, Expr]

    @cacheit
    def __new__(cls, b, e, evaluate=None):
        if evaluate is None:
            evaluate = global_parameters.evaluate

        b = _sympify(b)
        e = _sympify(e)

        # XXX: This can be removed when non-Expr args are disallowed rather
        # than deprecated.
        from .relational import Relational
        if isinstance(b, Relational) or isinstance(e, Relational):
            raise TypeError('Relational cannot be used in Pow')

        # XXX: This should raise TypeError once deprecation period is over:
        for arg in [b, e]:
            if not isinstance(arg, Expr):
                sympy_deprecation_warning(
                    f"""
    Using non-Expr arguments in Pow is deprecated (in this case, one of the
    arguments is of type {type(arg).__name__!r}).

    If you really did intend to construct a power with this base, use the **
    operator instead.""",
                    deprecated_since_version="1.7",
                    active_deprecations_target="non-expr-args-deprecated",
                    stacklevel=4,
                )

        if evaluate:
            if e is S.ComplexInfinity:
                return S.NaN
            if e is S.Infinity:
                if is_gt(b, S.One):
                    return S.Infinity
                if is_gt(b, S.NegativeOne) and is_lt(b, S.One):
                    return S.Zero
                if is_lt(b, S.NegativeOne):
                    if b.is_finite:
                        return S.ComplexInfinity
                    if b.is_finite is False:
                        return S.NaN
            if e is S.Zero:
                return S.One
            elif e is S.One:
                return b
            elif e == -1 and not b:
                return S.ComplexInfinity
            elif e.__class__.__name__ == "AccumulationBounds":
                if b == S.Exp1:
                    from sympy.calculus.accumulationbounds import AccumBounds
                    return AccumBounds(Pow(b, e.min), Pow(b, e.max))
            # autosimplification if base is a number and exp odd/even
            # if base is Number then the base will end up positive; we
            # do not do this with arbitrary expressions since symbolic
            # cancellation might occur as in (x - 1)/(1 - x) -> -1. If
            # we returned Piecewise((-1, Ne(x, 1))) for such cases then
            # we could do this...but we don't
            elif (e.is_Symbol and e.is_integer or e.is_Integer
                    ) and (b.is_number and b.is_Mul or b.is_Number
                    ) and b.could_extract_minus_sign():
                if e.is_even:
                    b = -b
                elif e.is_odd:
                    return -Pow(-b, e)
            if S.NaN in (b, e):  # XXX S.NaN**x -> S.NaN under assumption that x != 0
                return S.NaN
            elif b is S.One:
                if abs(e).is_infinite:
                    return S.NaN
                return S.One
            else:
                # recognize base as E
                from sympy.functions.elementary.exponential import exp_polar
                if not e.is_Atom and b is not S.Exp1 and not isinstance(b, exp_polar):
                    from .exprtools import factor_terms
                    from sympy.functions.elementary.exponential import log
                    from sympy.simplify.radsimp import fraction
                    c, ex = factor_terms(e, sign=False).as_coeff_Mul()
                    num, den = fraction(ex)
                    if isinstance(den, log) and den.args[0] == b:
                        return S.Exp1**(c*num)
                    elif den.is_Add:
                        from sympy.functions.elementary.complexes import sign, im
                        s = sign(im(b))
                        if s.is_Number and s and den == \
                                log(-factor_terms(b, sign=False)) + s*S.ImaginaryUnit*S.Pi:
                            return S.Exp1**(c*num)

                obj = b._eval_power(e)
                if obj is not None:
                    return obj
        obj = Expr.__new__(cls, b, e)
        obj = cls._exec_constructor_postprocessors(obj)
        if not isinstance(obj, Pow):
            return obj
        obj.is_commutative = (b.is_commutative and e.is_commutative)
        return obj

    def inverse(self, argindex=1):
        if self.base == S.Exp1:
            from sympy.functions.elementary.exponential import log
            return log
        return None

    @property
    def base(self) -> Expr:
        return self._args[0]

    @property
    def exp(self) -> Expr:
        return self._args[1]

    @property
    def kind(self):
        if self.exp.kind is NumberKind:
            return self.base.kind
        else:
            return UndefinedKind

    @classmethod
    def class_key(cls):
        return 3, 2, cls.__name__

    def _eval_refine(self, assumptions):
        from sympy.assumptions.ask import ask, Q
        b, e = self.as_base_exp()
        if ask(Q.integer(e), assumptions) and b.could_extract_minus_sign():
            if ask(Q.even(e), assumptions):
                return Pow(-b, e)
            elif ask(Q.odd(e), assumptions):
                return -Pow(-b, e)

    def _eval_power(self, other):
        b, e = self.as_base_exp()
        if b is S.NaN:
            return (b**e)**other  # let __new__ handle it

        s = None
        if other.is_integer:
            s = 1
        elif b.is_polar:  # e.g. exp_polar, besselj, var('p', polar=True)...
            s = 1
        elif e.is_extended_real is not None:
            from sympy.functions.elementary.complexes import arg, im, re, sign
            from sympy.functions.elementary.exponential import exp, log
            from sympy.functions.elementary.integers import floor
            # helper functions ===========================
            def _half(e):
                """Return True if the exponent has a literal 2 as the
                denominator, else None."""
                if getattr(e, 'q', None) == 2:
                    return True
                n, d = e.as_numer_denom()
                if n.is_integer and d == 2:
                    return True
            def _n2(e):
                """Return ``e`` evaluated to a Number with 2 significant
                digits, else None."""
                try:
                    rv = e.evalf(2, strict=True)
                    if rv.is_Number:
                        return rv
                except PrecisionExhausted:
                    pass
            # ===================================================
            if e.is_extended_real:
                # we need _half(other) with constant floor or
                # floor(S.Half - e*arg(b)/2/pi) == 0


                # handle -1 as special case
                if e == -1:
                    # floor arg. is 1/2 + arg(b)/2/pi
                    if _half(other):
                        if b.is_negative is True:
                            return S.NegativeOne**other*Pow(-b, e*other)
                        elif b.is_negative is False:  # XXX ok if im(b) != 0?
                            return Pow(b, -other)
                elif e.is_even:
                    if b.is_extended_real:
                        b = abs(b)
                    if b.is_imaginary:
                        b = abs(im(b))*S.ImaginaryUnit

                if (abs(e) < 1) == True or e == 1:
                    s = 1  # floor = 0
                elif b.is_extended_nonnegative:
                    s = 1  # floor = 0
                elif re(b).is_extended_nonnegative and (abs(e) < 2) == True:
                    s = 1  # floor = 0
                elif _half(other):
                    s = exp(2*S.Pi*S.ImaginaryUnit*other*floor(
                        S.Half - e*arg(b)/(2*S.Pi)))
                    if s.is_extended_real and _n2(sign(s) - s) == 0:
                        s = sign(s)
                    else:
                        s = None
            else:
                # e.is_extended_real is False requires:
                #     _half(other) with constant floor or
                #     floor(S.Half - im(e*log(b))/2/pi) == 0
                try:
                    s = exp(2*S.ImaginaryUnit*S.Pi*other*
                        floor(S.Half - im(e*log(b))/2/S.Pi))
                    # be careful to test that s is -1 or 1 b/c sign(I) == I:
                    # so check that s is real
                    if s.is_extended_real and _n2(sign(s) - s) == 0:
                        s = sign(s)
                    else:
                        s = None
                except PrecisionExhausted:
                    s = None

        if s is not None:
            return s*Pow(b, e*other)

    def _eval_Mod(self, q):
        r"""A dispatched function to compute `b^e \bmod q`, dispatched
        by ``Mod``.

        Notes
        =====

        Algorithms:

        1. For unevaluated integer power, use built-in ``pow`` function
        with 3 arguments, if powers are not too large wrt base.

        2. For very large powers, use totient reduction if $e \ge \log(m)$.
        Bound on m, is for safe factorization memory wise i.e. $m^{1/4}$.
        For pollard-rho to be faster than built-in pow $\log(e) > m^{1/4}$
        check is added.

        3. For any unevaluated power found in `b` or `e`, the step 2
        will be recursed down to the base and the exponent
        such that the $b \bmod q$ becomes the new base and
        $\phi(q) + e \bmod \phi(q)$ becomes the new exponent, and then
        the computation for the reduced expression can be done.
        """

        base, exp = self.base, self.exp

        if exp.is_integer and exp.is_positive:
            if q.is_integer and base % q == 0:
                return S.Zero

            from sympy.functions.combinatorial.numbers import totient

            if base.is_Integer and exp.is_Integer and q.is_Integer:
                b, e, m = int(base), int(exp), int(q)
                mb = m.bit_length()
                if mb <= 80 and e >= mb and e.bit_length()**4 >= m:
                    phi = int(totient(m))
                    return Integer(pow(b, phi + e%phi, m))
                return Integer(pow(b, e, m))

            from .mod import Mod

            if isinstance(base, Pow) and base.is_integer and base.is_number:
                base = Mod(base, q)
                return Mod(Pow(base, exp, evaluate=False), q)

            if isinstance(exp, Pow) and exp.is_integer and exp.is_number:
                bit_length = int(q).bit_length()
                # XXX Mod-Pow actually attempts to do a hanging evaluation
                # if this dispatched function returns None.
                # May need some fixes in the dispatcher itself.
                if bit_length <= 80:
                    phi = totient(q)
                    exp = phi + Mod(exp, phi)
                    return Mod(Pow(base, exp, evaluate=False), q)

    def _eval_is_even(self):
        if self.exp.is_integer and self.exp.is_positive:
            return self.base.is_even

    def _eval_is_negative(self):
        ext_neg = Pow._eval_is_extended_negative(self)
        if ext_neg is True:
            return self.is_finite
        return ext_neg

    def _eval_is_extended_positive(self):
        if self.base == self.exp:
            if self.base.is_extended_nonnegative:
                return True
        elif self.base.is_positive:
            if self.exp.is_real:
                return True
        elif self.base.is_extended_negative:
            if self.exp.is_even:
                return True
            if self.exp.is_odd:
                return False
        elif self.base.is_zero:
            if self.exp.is_extended_real:
                return self.exp.is_zero
        elif self.base.is_extended_nonpositive:
            if self.exp.is_odd:
                return False
        elif self.base.is_imaginary:
            if self.exp.is_integer:
                m = self.exp % 4
                if m.is_zero:
                    return True
                if m.is_integer and m.is_zero is False:
                    return False
            if self.exp.is_imaginary:
                from sympy.functions.elementary.exponential import log
                return log(self.base).is_imaginary

    def _eval_is_extended_negative(self):
        if self.exp is S.Half:
            if self.base.is_complex or self.base.is_extended_real:
                return False
        if self.base.is_extended_negative:
            if self.exp.is_odd and self.base.is_finite:
                return True
            if self.exp.is_even:
                return False
        elif self.base.is_extended_positive:
            if self.exp.is_extended_real:
                return False
        elif self.base.is_zero:
            if self.exp.is_extended_real:
                return False
        elif self.base.is_extended_nonnegative:
            if self.exp.is_extended_nonnegative:
                return False
        elif self.base.is_extended_nonpositive:
            if self.exp.is_even:
                return False
        elif self.base.is_extended_real:
            if self.exp.is_even:
                return False

    def _eval_is_zero(self):
        if self.base.is_zero:
            if self.exp.is_extended_positive:
                return True
            elif self.exp.is_extended_nonpositive:
                return False
        elif self.base == S.Exp1:
            return self.exp is S.NegativeInfinity
        elif self.base.is_zero is False:
            if self.base.is_finite and self.exp.is_finite:
                return False
            elif self.exp.is_negative:
                return self.base.is_infinite
            elif self.exp.is_nonnegative:
                return False
            elif self.exp.is_infinite and self.exp.is_extended_real:
                if (1 - abs(self.base)).is_extended_positive:
                    return self.exp.is_extended_positive
                elif (1 - abs(self.base)).is_extended_negative:
                    return self.exp.is_extended_negative
        elif self.base.is_finite and self.exp.is_negative:
            # when self.base.is_zero is None
            return False

    def _eval_is_integer(self):
        b, e = self.args
        if b.is_rational:
            if b.is_integer is False and e.is_positive:
                return False  # rat**nonneg
        if b.is_integer and e.is_integer:
            if b is S.NegativeOne:
                return True
            if e.is_nonnegative or e.is_positive:
                return True
        if b.is_integer and e.is_negative and (e.is_finite or e.is_integer):
            if fuzzy_not((b - 1).is_zero) and fuzzy_not((b + 1).is_zero):
                return False
        if b.is_Number and e.is_Number:
            check = self.func(*self.args)
            return check.is_Integer
        if e.is_negative and b.is_positive and (b - 1).is_positive:
            return False
        if e.is_negative and b.is_negative and (b + 1).is_negative:
            return False

    def _eval_is_extended_real(self):
        if self.base is S.Exp1:
            if self.exp.is_extended_real:
                return True
            elif self.exp.is_imaginary:
                return (2*S.ImaginaryUnit*self.exp/S.Pi).is_even

        from sympy.functions.elementary.exponential import log, exp
        real_b = self.base.is_extended_real
        if real_b is None:
            if self.base.func == exp and self.base.exp.is_imaginary:
                return self.exp.is_imaginary
            if self.base.func == Pow and self.base.base is S.Exp1 and self.base.exp.is_imaginary:
                return self.exp.is_imaginary
            return
        real_e = self.exp.is_extended_real
        if real_e is None:
            return
        if real_b and real_e:
            if self.base.is_extended_positive:
                return True
            elif self.base.is_extended_nonnegative and self.exp.is_extended_nonnegative:
                return True
            elif self.exp.is_integer and self.base.is_extended_nonzero:
                return True
            elif self.exp.is_integer and self.exp.is_nonnegative:
                return True
            elif self.base.is_extended_negative:
                if self.exp.is_Rational:
                    return False
        if real_e and self.exp.is_extended_negative and self.base.is_zero is False:
            return Pow(self.base, -self.exp).is_extended_real
        im_b = self.base.is_imaginary
        im_e = self.exp.is_imaginary
        if im_b:
            if self.exp.is_integer:
                if self.exp.is_even:
                    return True
                elif self.exp.is_odd:
                    return False
            elif im_e and log(self.base).is_imaginary:
                return True
            elif self.exp.is_Add:
                c, a = self.exp.as_coeff_Add()
                if c and c.is_Integer:
                    return Mul(
                        self.base**c, self.base**a, evaluate=False).is_extended_real
            elif self.base in (-S.ImaginaryUnit, S.ImaginaryUnit):
                if (self.exp/2).is_integer is False:
                    return False
        if real_b and im_e:
            if self.base is S.NegativeOne:
                return True
            c = self.exp.coeff(S.ImaginaryUnit)
            if c:
                if self.base.is_rational and c.is_rational:
                    if self.base.is_nonzero and (self.base - 1).is_nonzero and c.is_nonzero:
                        return False
                ok = (c*log(self.base)/S.Pi).is_integer
                if ok is not None:
                    return ok

        if real_b is False and real_e: # we already know it's not imag
            if isinstance(self.exp, Rational) and self.exp.p == 1:
                return False
            from sympy.functions.elementary.complexes import arg
            i = arg(self.base)*self.exp/S.Pi
            if i.is_complex: # finite
                return i.is_integer

    def _eval_is_complex(self):

        if self.base == S.Exp1:
            return fuzzy_or([self.exp.is_complex, self.exp.is_extended_negative])

        if all(a.is_complex for a in self.args) and self._eval_is_finite():
            return True

    def _eval_is_imaginary(self):
        if self.base.is_commutative is False:
            return False

        if self.base.is_imaginary:
            if self.exp.is_integer:
                odd = self.exp.is_odd
                if odd is not None:
                    return odd
                return

        if self.base == S.Exp1:
            f = 2 * self.exp / (S.Pi*S.ImaginaryUnit)
            # exp(pi*integer) = 1 or -1, so not imaginary
            if f.is_even:
                return False
            # exp(pi*integer + pi/2) = I or -I, so it is imaginary
            if f.is_odd:
                return True
            return None

        if self.exp.is_imaginary:
            from sympy.functions.elementary.exponential import log
            imlog = log(self.base).is_imaginary
            if imlog is not None:
                return False  # I**i -> real; (2*I)**i -> complex ==> not imaginary

        if self.base.is_extended_real and self.exp.is_extended_real:
            if self.base.is_positive:
                return False
            else:
                rat = self.exp.is_rational
                if not rat:
                    return rat
                if self.exp.is_integer:
                    return False
                else:
                    half = (2*self.exp).is_integer
                    if half:
                        return self.base.is_negative
                    return half

        if self.base.is_extended_real is False:  # we already know it's not imag
            from sympy.functions.elementary.complexes import arg
            i = arg(self.base)*self.exp/S.Pi
            isodd = (2*i).is_odd
            if isodd is not None:
                return isodd

    def _eval_is_odd(self):
        if self.exp.is_integer:
            if self.exp.is_positive:
                return self.base.is_odd
            elif self.exp.is_nonnegative and self.base.is_odd:
                return True
            elif self.base is S.NegativeOne:
                return True

    def _eval_is_finite(self):
        if self.exp.is_negative:
            if self.base.is_zero:
                return False
            if self.base.is_infinite or self.base.is_nonzero:
                return True
        c1 = self.base.is_finite
        if c1 is None:
            return
        c2 = self.exp.is_finite
        if c2 is None:
            return
        if c1 and c2:
            if self.exp.is_nonnegative or fuzzy_not(self.base.is_zero):
                return True

    def _eval_is_prime(self):
        '''
        An integer raised to the n(>=2)-th power cannot be a prime.
        '''
        if self.base.is_integer and self.exp.is_integer and (self.exp - 1).is_positive:
            return False

    def _eval_is_composite(self):
        """
        A power is composite if both base and exponent are greater than 1
        """
        if (self.base.is_integer and self.exp.is_integer and
            ((self.base - 1).is_positive and (self.exp - 1).is_positive or
            (self.base + 1).is_negative and self.exp.is_positive and self.exp.is_even)):
            return True

    def _eval_is_polar(self):
        return self.base.is_polar

    def _eval_subs(self, old, new):
        from sympy.calculus.accumulationbounds import AccumBounds

        if isinstance(self.exp, AccumBounds):
            b = self.base.subs(old, new)
            e = self.exp.subs(old, new)
            if isinstance(e, AccumBounds):
                return e.__rpow__(b)
            return self.func(b, e)

        from sympy.functions.elementary.exponential import exp, log

        def _check(ct1, ct2, old):
            """Return (bool, pow, remainder_pow) where, if bool is True, then the
            exponent of Pow `old` will combine with `pow` so the substitution
            is valid, otherwise bool will be False.

            For noncommutative objects, `pow` will be an integer, and a factor
            `Pow(old.base, remainder_pow)` needs to be included. If there is
            no such factor, None is returned. For commutative objects,
            remainder_pow is always None.

            cti are the coefficient and terms of an exponent of self or old
            In this _eval_subs routine a change like (b**(2*x)).subs(b**x, y)
            will give y**2 since (b**x)**2 == b**(2*x); if that equality does
            not hold then the substitution should not occur so `bool` will be
            False.

            """
            coeff1, terms1 = ct1
            coeff2, terms2 = ct2
            if terms1 == terms2:
                if old.is_commutative:
                    # Allow fractional powers for commutative objects
                    pow = coeff1/coeff2
                    try:
                        as_int(pow, strict=False)
                        combines = True
                    except ValueError:
                        b, e = old.as_base_exp()
                        # These conditions ensure that (b**e)**f == b**(e*f) for any f
                        combines = b.is_positive and e.is_real or b.is_nonnegative and e.is_nonnegative

                    return combines, pow, None
                else:
                    # With noncommutative symbols, substitute only integer powers
                    if not isinstance(terms1, tuple):
                        terms1 = (terms1,)
                    if not all(term.is_integer for term in terms1):
                        return False, None, None

                    try:
                        # Round pow toward zero
                        pow, remainder = divmod(as_int(coeff1), as_int(coeff2))
                        if pow < 0 and remainder != 0:
                            pow += 1
                            remainder -= as_int(coeff2)

                        if remainder == 0:
                            remainder_pow = None
                        else:
                            remainder_pow = Mul(remainder, *terms1)

                        return True, pow, remainder_pow
                    except ValueError:
                        # Can't substitute
                        pass

            return False, None, None

        if old == self.base or (old == exp and self.base == S.Exp1):
            if new.is_Function and isinstance(new, Callable):
                return new(self.exp._subs(old, new))
            else:
                return new**self.exp._subs(old, new)

        # issue 10829: (4**x - 3*y + 2).subs(2**x, y) -> y**2 - 3*y + 2
        if isinstance(old, self.func) and self.exp == old.exp:
            l = log(self.base, old.base)
            if l.is_Number:
                return Pow(new, l)

        if isinstance(old, self.func) and self.base == old.base:
            if self.exp.is_Add is False:
                ct1 = self.exp.as_independent(Symbol, as_Add=False)
                ct2 = old.exp.as_independent(Symbol, as_Add=False)
                ok, pow, remainder_pow = _check(ct1, ct2, old)
                if ok:
                    # issue 5180: (x**(6*y)).subs(x**(3*y),z)->z**2
                    result = self.func(new, pow)
                    if remainder_pow is not None:
                        result = Mul(result, Pow(old.base, remainder_pow))
                    return result
            else:  # b**(6*x + a).subs(b**(3*x), y) -> y**2 * b**a
                # exp(exp(x) + exp(x**2)).subs(exp(exp(x)), w) -> w * exp(exp(x**2))
                oarg = old.exp
                new_l = []
                o_al = []
                ct2 = oarg.as_coeff_mul()
                for a in self.exp.args:
                    newa = a._subs(old, new)
                    ct1 = newa.as_coeff_mul()
                    ok, pow, remainder_pow = _check(ct1, ct2, old)
                    if ok:
                        new_l.append(new**pow)
                        if remainder_pow is not None:
                            o_al.append(remainder_pow)
                        continue
                    elif not old.is_commutative and not newa.is_integer:
                        # If any term in the exponent is non-integer,
                        # we do not do any substitutions in the noncommutative case
                        return
                    o_al.append(newa)
                if new_l:
                    expo = Add(*o_al)
                    new_l.append(Pow(self.base, expo, evaluate=False) if expo != 1 else self.base)
                    return Mul(*new_l)

        if (isinstance(old, exp) or (old.is_Pow and old.base is S.Exp1)) and self.exp.is_extended_real and self.base.is_positive:
            ct1 = old.exp.as_independent(Symbol, as_Add=False)
            ct2 = (self.exp*log(self.base)).as_independent(
                Symbol, as_Add=False)
            ok, pow, remainder_pow = _check(ct1, ct2, old)
            if ok:
                result = self.func(new, pow)  # (2**x).subs(exp(x*log(2)), z) -> z
                if remainder_pow is not None:
                    result = Mul(result, Pow(old.base, remainder_pow))
                return result

    def as_base_exp(self):
        """Return base and exp of self.

        Explanation
        ===========

        If base a Rational less than 1, then return 1/Rational, -exp.
        If this extra processing is not needed, the base and exp
        properties will give the raw arguments.

        Examples
        ========

        >>> from sympy import Pow, S
        >>> p = Pow(S.Half, 2, evaluate=False)
        >>> p.as_base_exp()
        (2, -2)
        >>> p.args
        (1/2, 2)
        >>> p.base, p.exp
        (1/2, 2)

        """

        b, e = self.args
        if b.is_Rational and b.p < b.q and b.p > 0:
            return 1/b, -e
        return b, e

    def _eval_adjoint(self):
        from sympy.functions.elementary.complexes import adjoint
        i, p = self.exp.is_integer, self.base.is_positive
        if i:
            return adjoint(self.base)**self.exp
        if p:
            return self.base**adjoint(self.exp)
        if i is False and p is False:
            expanded = expand_complex(self)
            if expanded != self:
                return adjoint(expanded)

    def _eval_conjugate(self):
        from sympy.functions.elementary.complexes import conjugate as c
        i, p = self.exp.is_integer, self.base.is_positive
        if i:
            return c(self.base)**self.exp
        if p:
            return self.base**c(self.exp)
        if i is False and p is False:
            expanded = expand_complex(self)
            if expanded != self:
                return c(expanded)
        if self.is_extended_real:
            return self

    def _eval_transpose(self):
        from sympy.functions.elementary.complexes import transpose
        if self.base == S.Exp1:
            return self.func(S.Exp1, self.exp.transpose())
        i, p = self.exp.is_integer, (self.base.is_complex or self.base.is_infinite)
        if p:
            return self.base**self.exp
        if i:
            return transpose(self.base)**self.exp
        if i is False and p is False:
            expanded = expand_complex(self)
            if expanded != self:
                return transpose(expanded)

    def _eval_expand_power_exp(self, **hints):
        """a**(n + m) -> a**n*a**m"""
        b = self.base
        e = self.exp
        if b == S.Exp1:
            from sympy.concrete.summations import Sum
            if isinstance(e, Sum) and e.is_commutative:
                from sympy.concrete.products import Product
                return Product(self.func(b, e.function), *e.limits)
        if e.is_Add and (hints.get('force', False) or
                b.is_zero is False or e._all_nonneg_or_nonppos()):
            if e.is_commutative:
                return Mul(*[self.func(b, x) for x in e.args])
            if b.is_commutative:
                c, nc = sift(e.args, lambda x: x.is_commutative, binary=True)
                if c:
                    return Mul(*[self.func(b, x) for x in c]
                        )*b**Add._from_args(nc)
        return self

    def _eval_expand_power_base(self, **hints):
        """(a*b)**n -> a**n * b**n"""
        force = hints.get('force', False)

        b = self.base
        e = self.exp
        if not b.is_Mul:
            return self

        cargs, nc = b.args_cnc(split_1=False)

        # expand each term - this is top-level-only
        # expansion but we have to watch out for things
        # that don't have an _eval_expand method
        if nc:
            nc = [i._eval_expand_power_base(**hints)
                if hasattr(i, '_eval_expand_power_base') else i
                for i in nc]

            if e.is_Integer:
                if e.is_positive:
                    rv = Mul(*nc*e)
                else:
                    rv = Mul(*[i**-1 for i in nc[::-1]]*-e)
                if cargs:
                    rv *= Mul(*cargs)**e
                return rv

            if not cargs:
                return self.func(Mul(*nc), e, evaluate=False)

            nc = [Mul(*nc)]

        # sift the commutative bases
        other, maybe_real = sift(cargs, lambda x: x.is_extended_real is False,
            binary=True)
        def pred(x):
            if x is S.ImaginaryUnit:
                return S.ImaginaryUnit
            polar = x.is_polar
            if polar:
                return True
            if polar is None:
                return fuzzy_bool(x.is_extended_nonnegative)
        sifted = sift(maybe_real, pred)
        nonneg = sifted[True]
        other += sifted[None]
        neg = sifted[False]
        imag = sifted[S.ImaginaryUnit]
        if imag:
            I = S.ImaginaryUnit
            i = len(imag) % 4
            if i == 0:
                pass
            elif i == 1:
                other.append(I)
            elif i == 2:
                if neg:
                    nonn = -neg.pop()
                    if nonn is not S.One:
                        nonneg.append(nonn)
                else:
                    neg.append(S.NegativeOne)
            else:
                if neg:
                    nonn = -neg.pop()
                    if nonn is not S.One:
                        nonneg.append(nonn)
                else:
                    neg.append(S.NegativeOne)
                other.append(I)
            del imag

        # bring out the bases that can be separated from the base

        if force or e.is_integer:
            # treat all commutatives the same and put nc in other
            cargs = nonneg + neg + other
            other = nc
        else:
            # this is just like what is happening automatically, except
            # that now we are doing it for an arbitrary exponent for which
            # no automatic expansion is done

            assert not e.is_Integer

            # handle negatives by making them all positive and putting
            # the residual -1 in other
            if len(neg) > 1:
                o = S.One
                if not other and neg[0].is_Number:
                    o *= neg.pop(0)
                if len(neg) % 2:
                    o = -o
                for n in neg:
                    nonneg.append(-n)
                if o is not S.One:
                    other.append(o)
            elif neg and other:
                if neg[0].is_Number and neg[0] is not S.NegativeOne:
                    other.append(S.NegativeOne)
                    nonneg.append(-neg[0])
                else:
                    other.extend(neg)
            else:
                other.extend(neg)
            del neg

            cargs = nonneg
            other += nc

        rv = S.One
        if cargs:
            if e.is_Rational:
                npow, cargs = sift(cargs, lambda x: x.is_Pow and
                    x.exp.is_Rational and x.base.is_number,
                    binary=True)
                rv = Mul(*[self.func(b.func(*b.args), e) for b in npow])
            rv *= Mul(*[self.func(b, e, evaluate=False) for b in cargs])
        if other:
            rv *= self.func(Mul(*other), e, evaluate=False)
        return rv

    def _eval_expand_multinomial(self, **hints):
        """(a + b + ..)**n -> a**n + n*a**(n-1)*b + .., n is nonzero integer"""

        base, exp = self.args
        result = self

        if exp.is_Rational and exp.p > 0 and base.is_Add:
            if not exp.is_Integer:
                n = Integer(exp.p // exp.q)

                if not n:
                    return result
                else:
                    radical, result = self.func(base, exp - n), []

                    expanded_base_n = self.func(base, n)
                    if expanded_base_n.is_Pow:
                        expanded_base_n = \
                            expanded_base_n._eval_expand_multinomial()
                    for term in Add.make_args(expanded_base_n):
                        result.append(term*radical)

                    return Add(*result)

            n = int(exp)

            if base.is_commutative:
                order_terms, other_terms = [], []

                for b in base.args:
                    if b.is_Order:
                        order_terms.append(b)
                    else:
                        other_terms.append(b)

                if order_terms:
                    # (f(x) + O(x^n))^m -> f(x)^m + m*f(x)^{m-1} *O(x^n)
                    f = Add(*other_terms)
                    o = Add(*order_terms)

                    if n == 2:
                        return expand_multinomial(f**n, deep=False) + n*f*o
                    else:
                        g = expand_multinomial(f**(n - 1), deep=False)
                        return expand_mul(f*g, deep=False) + n*g*o

                if base.is_number:
                    # Efficiently expand expressions of the form (a + b*I)**n
                    # where 'a' and 'b' are real numbers and 'n' is integer.
                    a, b = base.as_real_imag()

                    if a.is_Rational and b.is_Rational:
                        if not a.is_Integer:
                            if not b.is_Integer:
                                k = self.func(a.q * b.q, n)
                                a, b = a.p*b.q, a.q*b.p
                            else:
                                k = self.func(a.q, n)
                                a, b = a.p, a.q*b
                        elif not b.is_Integer:
                            k = self.func(b.q, n)
                            a, b = a*b.q, b.p
                        else:
                            k = 1

                        a, b, c, d = int(a), int(b), 1, 0

                        while n:
                            if n & 1:
                                c, d = a*c - b*d, b*c + a*d
                                n -= 1
                            a, b = a*a - b*b, 2*a*b
                            n //= 2

                        I = S.ImaginaryUnit

                        if k == 1:
                            return c + I*d
                        else:
                            return Integer(c)/k + I*d/k

                p = other_terms
                # (x + y)**3 -> x**3 + 3*x**2*y + 3*x*y**2 + y**3
                # in this particular example:
                # p = [x,y]; n = 3
                # so now it's easy to get the correct result -- we get the
                # coefficients first:
                from sympy.ntheory.multinomial import multinomial_coefficients
                from sympy.polys.polyutils import basic_from_dict
                expansion_dict = multinomial_coefficients(len(p), n)
                # in our example: {(3, 0): 1, (1, 2): 3, (0, 3): 1, (2, 1): 3}
                # and now construct the expression.
                return basic_from_dict(expansion_dict, *p)
            else:
                if n == 2:
                    return Add(*[f*g for f in base.args for g in base.args])
                else:
                    multi = (base**(n - 1))._eval_expand_multinomial()
                    if multi.is_Add:
                        return Add(*[f*g for f in base.args
                            for g in multi.args])
                    else:
                        # XXX can this ever happen if base was an Add?
                        return Add(*[f*multi for f in base.args])
        elif (exp.is_Rational and exp.p < 0 and base.is_Add and
                abs(exp.p) > exp.q):
            return 1 / self.func(base, -exp)._eval_expand_multinomial()
        elif exp.is_Add and base.is_Number and (hints.get('force', False) or
                base.is_zero is False or exp._all_nonneg_or_nonppos()):
            #  a + b      a  b
            #  n      --> n  n, where n, a, b are Numbers
            # XXX should be in expand_power_exp?
            coeff, tail = [], []
            for term in exp.args:
                if term.is_Number:
                    coeff.append(self.func(base, term))
                else:
                    tail.append(term)
            return Mul(*(coeff + [self.func(base, Add._from_args(tail))]))
        else:
            return result

    def as_real_imag(self, deep=True, **hints):
        if self.exp.is_Integer:
            from sympy.polys.polytools import poly

            exp = self.exp
            re_e, im_e = self.base.as_real_imag(deep=deep)
            if not im_e:
                return self, S.Zero
            a, b = symbols('a b', cls=Dummy)
            if exp >= 0:
                if re_e.is_Number and im_e.is_Number:
                    # We can be more efficient in this case
                    expr = expand_multinomial(self.base**exp)
                    if expr != self:
                        return expr.as_real_imag()

                expr = poly(
                    (a + b)**exp)  # a = re, b = im; expr = (a + b*I)**exp
            else:
                mag = re_e**2 + im_e**2
                re_e, im_e = re_e/mag, -im_e/mag
                if re_e.is_Number and im_e.is_Number:
                    # We can be more efficient in this case
                    expr = expand_multinomial((re_e + im_e*S.ImaginaryUnit)**-exp)
                    if expr != self:
                        return expr.as_real_imag()

                expr = poly((a + b)**-exp)

            # Terms with even b powers will be real
            r = [i for i in expr.terms() if not i[0][1] % 2]
            re_part = Add(*[cc*a**aa*b**bb for (aa, bb), cc in r])
            # Terms with odd b powers will be imaginary
            r = [i for i in expr.terms() if i[0][1] % 4 == 1]
            im_part1 = Add(*[cc*a**aa*b**bb for (aa, bb), cc in r])
            r = [i for i in expr.terms() if i[0][1] % 4 == 3]
            im_part3 = Add(*[cc*a**aa*b**bb for (aa, bb), cc in r])

            return (re_part.subs({a: re_e, b: S.ImaginaryUnit*im_e}),
            im_part1.subs({a: re_e, b: im_e}) + im_part3.subs({a: re_e, b: -im_e}))

        from sympy.functions.elementary.trigonometric import atan2, cos, sin

        if self.exp.is_Rational:
            re_e, im_e = self.base.as_real_imag(deep=deep)

            if im_e.is_zero and self.exp is S.Half:
                if re_e.is_extended_nonnegative:
                    return self, S.Zero
                if re_e.is_extended_nonpositive:
                    return S.Zero, (-self.base)**self.exp

            # XXX: This is not totally correct since for x**(p/q) with
            #      x being imaginary there are actually q roots, but
            #      only a single one is returned from here.
            r = self.func(self.func(re_e, 2) + self.func(im_e, 2), S.Half)

            t = atan2(im_e, re_e)

            rp, tp = self.func(r, self.exp), t*self.exp

            return rp*cos(tp), rp*sin(tp)
        elif self.base is S.Exp1:
            from sympy.functions.elementary.exponential import exp
            re_e, im_e = self.exp.as_real_imag()
            if deep:
                re_e = re_e.expand(deep, **hints)
                im_e = im_e.expand(deep, **hints)
            c, s = cos(im_e), sin(im_e)
            return exp(re_e)*c, exp(re_e)*s
        else:
            from sympy.functions.elementary.complexes import im, re
            if deep:
                hints['complex'] = False

                expanded = self.expand(deep, **hints)
                if hints.get('ignore') == expanded:
                    return None
                else:
                    return (re(expanded), im(expanded))
            else:
                return re(self), im(self)

    def _eval_derivative(self, s):
        from sympy.functions.elementary.exponential import log
        dbase = self.base.diff(s)
        dexp = self.exp.diff(s)
        return self * (dexp * log(self.base) + dbase * self.exp/self.base)

    def _eval_evalf(self, prec):
        base, exp = self.as_base_exp()
        if base == S.Exp1:
            # Use mpmath function associated to class "exp":
            from sympy.functions.elementary.exponential import exp as exp_function
            return exp_function(self.exp, evaluate=False)._eval_evalf(prec)
        base = base._evalf(prec)
        if not exp.is_Integer:
            exp = exp._evalf(prec)
        if exp.is_negative and base.is_number and base.is_extended_real is False:
            base = base.conjugate() / (base * base.conjugate())._evalf(prec)
            exp = -exp
            return self.func(base, exp).expand()
        return self.func(base, exp)

    def _eval_is_polynomial(self, syms):
        if self.exp.has(*syms):
            return False

        if self.base.has(*syms):
            return bool(self.base._eval_is_polynomial(syms) and
                self.exp.is_Integer and (self.exp >= 0))
        else:
            return True

    def _eval_is_rational(self):
        # The evaluation of self.func below can be very expensive in the case
        # of integer**integer if the exponent is large.  We should try to exit
        # before that if possible:
        if (self.exp.is_integer and self.base.is_rational
                and fuzzy_not(fuzzy_and([self.exp.is_negative, self.base.is_zero]))):
            return True
        p = self.func(*self.as_base_exp())  # in case it's unevaluated
        if not p.is_Pow:
            return p.is_rational
        b, e = p.as_base_exp()
        if e.is_Rational and b.is_Rational:
            # we didn't check that e is not an Integer
            # because Rational**Integer autosimplifies
            return False
        if e.is_integer:
            if b.is_rational:
                if fuzzy_not(b.is_zero) or e.is_nonnegative:
                    return True
                if b == e:  # always rational, even for 0**0
                    return True
            elif b.is_irrational:
                return e.is_zero
        if b is S.Exp1:
            if e.is_rational and e.is_nonzero:
                return False

    def _eval_is_algebraic(self):
        def _is_one(expr):
            try:
                return (expr - 1).is_zero
            except ValueError:
                # when the operation is not allowed
                return False

        if self.base.is_zero or _is_one(self.base):
            return True
        elif self.base is S.Exp1:
            s = self.func(*self.args)
            if s.func == self.func:
                if self.exp.is_nonzero:
                    if self.exp.is_algebraic:
                        return False
                    elif (self.exp/S.Pi).is_rational:
                        return False
                    elif (self.exp/(S.ImaginaryUnit*S.Pi)).is_rational:
                        return True
            else:
                return s.is_algebraic
        elif self.exp.is_rational:
            if self.base.is_algebraic is False:
                return self.exp.is_zero
            if self.base.is_zero is False:
                if self.exp.is_nonzero:
                    return self.base.is_algebraic
                elif self.base.is_algebraic:
                    return True
            if self.exp.is_positive:
                return self.base.is_algebraic
        elif self.base.is_algebraic and self.exp.is_algebraic:
            if ((fuzzy_not(self.base.is_zero)
                and fuzzy_not(_is_one(self.base)))
                or self.base.is_integer is False
                or self.base.is_irrational):
                return self.exp.is_rational

    def _eval_is_rational_function(self, syms):
        if self.exp.has(*syms):
            return False

        if self.base.has(*syms):
            return self.base._eval_is_rational_function(syms) and \
                self.exp.is_Integer
        else:
            return True

    def _eval_is_meromorphic(self, x, a):
        # f**g is meromorphic if g is an integer and f is meromorphic.
        # E**(log(f)*g) is meromorphic if log(f)*g is meromorphic
        # and finite.
        base_merom = self.base._eval_is_meromorphic(x, a)
        exp_integer = self.exp.is_Integer
        if exp_integer:
            return base_merom

        exp_merom = self.exp._eval_is_meromorphic(x, a)
        if base_merom is False:
            # f**g = E**(log(f)*g) may be meromorphic if the
            # singularities of log(f) and g cancel each other,
            # for example, if g = 1/log(f). Hence,
            return False if exp_merom else None
        elif base_merom is None:
            return None

        b = self.base.subs(x, a)
        # b is extended complex as base is meromorphic.
        # log(base) is finite and meromorphic when b != 0, zoo.
        b_zero = b.is_zero
        if b_zero:
            log_defined = False
        else:
            log_defined = fuzzy_and((b.is_finite, fuzzy_not(b_zero)))

        if log_defined is False: # zero or pole of base
            return exp_integer  # False or None
        elif log_defined is None:
            return None

        if not exp_merom:
            return exp_merom  # False or None

        return self.exp.subs(x, a).is_finite

    def _eval_is_algebraic_expr(self, syms):
        if self.exp.has(*syms):
            return False

        if self.base.has(*syms):
            return self.base._eval_is_algebraic_expr(syms) and \
                self.exp.is_Rational
        else:
            return True

    def _eval_rewrite_as_exp(self, base, expo, **kwargs):
        from sympy.functions.elementary.exponential import exp, log

        if base.is_zero or base.has(exp) or expo.has(exp):
            return base**expo

        evaluate = expo.has(Symbol)

        if base.has(Symbol):
            # delay evaluation if expo is non symbolic
            # (as exp(x*log(5)) automatically reduces to x**5)
            if global_parameters.exp_is_pow:
                return Pow(S.Exp1, log(base)*expo, evaluate=evaluate)
            else:
                return exp(log(base)*expo, evaluate=evaluate)

        else:
            from sympy.functions.elementary.complexes import arg, Abs
            return exp((log(Abs(base)) + S.ImaginaryUnit*arg(base))*expo)

    def as_numer_denom(self):
        if not self.is_commutative:
            return self, S.One
        base, exp = self.as_base_exp()
        n, d = base.as_numer_denom()
        # this should be the same as ExpBase.as_numer_denom wrt
        # exponent handling
        neg_exp = exp.is_negative
        if exp.is_Mul and not neg_exp and not exp.is_positive:
            neg_exp = exp.could_extract_minus_sign()
        int_exp = exp.is_integer
        # the denominator cannot be separated from the numerator if
        # its sign is unknown unless the exponent is an integer, e.g.
        # sqrt(a/b) != sqrt(a)/sqrt(b) when a=1 and b=-1. But if the
        # denominator is negative the numerator and denominator can
        # be negated and the denominator (now positive) separated.
        if not (d.is_extended_real or int_exp):
            n = base
            d = S.One
        dnonpos = d.is_nonpositive
        if dnonpos:
            n, d = -n, -d
        elif dnonpos is None and not int_exp:
            n = base
            d = S.One
        if neg_exp:
            n, d = d, n
            exp = -exp
        if exp.is_infinite:
            if n is S.One and d is not S.One:
                return n, self.func(d, exp)
            if n is not S.One and d is S.One:
                return self.func(n, exp), d
        return self.func(n, exp), self.func(d, exp)

    def matches(self, expr, repl_dict=None, old=False):
        expr = _sympify(expr)
        if repl_dict is None:
            repl_dict = {}

        # special case, pattern = 1 and expr.exp can match to 0
        if expr is S.One:
            d = self.exp.matches(S.Zero, repl_dict)
            if d is not None:
                return d

        # make sure the expression to be matched is an Expr
        if not isinstance(expr, Expr):
            return None

        b, e = expr.as_base_exp()

        # special case number
        sb, se = self.as_base_exp()
        if sb.is_Symbol and se.is_Integer and expr:
            if e.is_rational:
                return sb.matches(b**(e/se), repl_dict)
            return sb.matches(expr**(1/se), repl_dict)

        d = repl_dict.copy()
        d = self.base.matches(b, d)
        if d is None:
            return None

        d = self.exp.xreplace(d).matches(e, d)
        if d is None:
            return Expr.matches(self, expr, repl_dict)
        return d

    def _eval_nseries(self, x, n, logx, cdir=0):
        # NOTE! This function is an important part of the gruntz algorithm
        #       for computing limits. It has to return a generalized power
        #       series with coefficients in C(log, log(x)). In more detail:
        # It has to return an expression
        #     c_0*x**e_0 + c_1*x**e_1 + ... (finitely many terms)
        # where e_i are numbers (not necessarily integers) and c_i are
        # expressions involving only numbers, the log function, and log(x).
        # The series expansion of b**e is computed as follows:
        # 1) We express b as f*(1 + g) where f is the leading term of b.
        #    g has order O(x**d) where d is strictly positive.
        # 2) Then b**e = (f**e)*((1 + g)**e).
        #    (1 + g)**e is computed using binomial series.
        from sympy.functions.elementary.exponential import exp, log
        from sympy.series.limits import limit
        from sympy.series.order import Order
        from sympy.core.sympify import sympify
        if self.base is S.Exp1:
            e_series = self.exp.nseries(x, n=n, logx=logx)
            if e_series.is_Order:
                return 1 + e_series
            e0 = limit(e_series.removeO(), x, 0)
            if e0 is S.NegativeInfinity:
                return Order(x**n, x)
            if e0 is S.Infinity:
                return self
            t = e_series - e0
            exp_series = term = exp(e0)
            # series of exp(e0 + t) in t
            for i in range(1, n):
                term *= t/i
                term = term.nseries(x, n=n, logx=logx)
                exp_series += term
            exp_series += Order(t**n, x)
            from sympy.simplify.powsimp import powsimp
            return powsimp(exp_series, deep=True, combine='exp')
        from sympy.simplify.powsimp import powdenest
        from .numbers import _illegal
        self = powdenest(self, force=True).trigsimp()
        b, e = self.as_base_exp()

        if e.has(*_illegal):
            raise PoleError()

        if e.has(x):
            return exp(e*log(b))._eval_nseries(x, n=n, logx=logx, cdir=cdir)

        if logx is not None and b.has(log):
            from .symbol import Wild
            c, ex = symbols('c, ex', cls=Wild, exclude=[x])
            b = b.replace(log(c*x**ex), log(c) + ex*logx)
            self = b**e

        b = b.removeO()
        try:
            from sympy.functions.special.gamma_functions import polygamma
            if b.has(polygamma, S.EulerGamma) and logx is not None:
                raise ValueError()
            _, m = b.leadterm(x)
        except (ValueError, NotImplementedError, PoleError):
            b = b._eval_nseries(x, n=max(2, n), logx=logx, cdir=cdir).removeO()
            if b.has(S.NaN, S.ComplexInfinity):
                raise NotImplementedError()
            _, m = b.leadterm(x)

        if e.has(log):
            from sympy.simplify.simplify import logcombine
            e = logcombine(e).cancel()

        if not (m.is_zero or e.is_number and e.is_real):
            if self == self._eval_as_leading_term(x, logx=logx, cdir=cdir):
                res = exp(e*log(b))._eval_nseries(x, n=n, logx=logx, cdir=cdir)
                if res == exp(e*log(b)):
                    return self
                return res

        f = b.as_leading_term(x, logx=logx)
        g = (b/f - S.One).cancel(expand=False)
        if not m.is_number:
            raise NotImplementedError()
        maxpow = n - m*e
        if maxpow.has(Symbol):
            maxpow = sympify(n)

        if maxpow.is_negative:
            return Order(x**(m*e), x)

        if g.is_zero:
            r = f**e
            if r != self:
                r += Order(x**n, x)
            return r

        def coeff_exp(term, x):
            coeff, exp = S.One, S.Zero
            for factor in Mul.make_args(term):
                if factor.has(x):
                    base, exp = factor.as_base_exp()
                    if base != x:
                        try:
                            return term.leadterm(x)
                        except ValueError:
                            return term, S.Zero
                else:
                    coeff *= factor
            return coeff, exp

        def mul(d1, d2):
            res = {}
            for e1, e2 in product(d1, d2):
                ex = e1 + e2
                if ex < maxpow:
                    res[ex] = res.get(ex, S.Zero) + d1[e1]*d2[e2]
            return res

        try:
            c, d = g.leadterm(x, logx=logx)
        except (ValueError, NotImplementedError):
            if limit(g/x**maxpow, x, 0) == 0:
                # g has higher order zero
                return f**e + e*f**e*g  # first term of binomial series
            else:
                raise NotImplementedError()
        if c.is_Float and d == S.Zero:
            # Convert floats like 0.5 to exact SymPy numbers like S.Half, to
            # prevent rounding errors which can induce wrong values of d leading
            # to a NotImplementedError being returned from the block below.
            from sympy.simplify.simplify import nsimplify
            _, d = nsimplify(g).leadterm(x, logx=logx)
        if not d.is_positive:
            g = g.simplify()
            if g.is_zero:
                return f**e
            _, d = g.leadterm(x, logx=logx)
            if not d.is_positive:
                g = ((b - f)/f).expand()
                _, d = g.leadterm(x, logx=logx)
                if not d.is_positive:
                    raise NotImplementedError()

        from sympy.functions.elementary.integers import ceiling
        gpoly = g._eval_nseries(x, n=ceiling(maxpow), logx=logx, cdir=cdir).removeO()
        gterms = {}

        for term in Add.make_args(gpoly):
            co1, e1 = coeff_exp(term, x)
            gterms[e1] = gterms.get(e1, S.Zero) + co1

        k = S.One
        terms = {S.Zero: S.One}
        tk = gterms

        from sympy.functions.combinatorial.factorials import factorial, ff

        while (k*d - maxpow).is_negative:
            coeff = ff(e, k)/factorial(k)
            for ex in tk:
                terms[ex] = terms.get(ex, S.Zero) + coeff*tk[ex]
            tk = mul(tk, gterms)
            k += S.One

        from sympy.functions.elementary.complexes import im

        if not e.is_integer and m.is_zero and f.is_negative:
            ndir = (b - f).dir(x, cdir)
            if im(ndir).is_negative:
                inco, inex = coeff_exp(f**e*(-1)**(-2*e), x)
            elif im(ndir).is_zero:
                inco, inex = coeff_exp(exp(e*log(b)).as_leading_term(x, logx=logx, cdir=cdir), x)
            else:
                inco, inex = coeff_exp(f**e, x)
        else:
            inco, inex = coeff_exp(f**e, x)
        res = S.Zero

        for e1 in terms:
            ex = e1 + inex
            res += terms[e1]*inco*x**(ex)

        if not (e.is_integer and e.is_positive and (e*d - n).is_nonpositive and
                res == _mexpand(self)):
            try:
                res += Order(x**n, x)
            except NotImplementedError:
                return exp(e*log(b))._eval_nseries(x, n=n, logx=logx, cdir=cdir)
        return res

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        from sympy.functions.elementary.exponential import exp, log
        e = self.exp
        b = self.base
        if self.base is S.Exp1:
            arg = e.as_leading_term(x, logx=logx)
            arg0 = arg.subs(x, 0)
            if arg0 is S.NaN:
                arg0 = arg.limit(x, 0)
            if arg0.is_infinite is False:
                return S.Exp1**arg0
            raise PoleError("Cannot expand %s around 0" % (self))
        elif e.has(x):
            lt = exp(e * log(b))
            return lt.as_leading_term(x, logx=logx, cdir=cdir)
        else:
            from sympy.functions.elementary.complexes import im
            try:
                f = b.as_leading_term(x, logx=logx, cdir=cdir)
            except PoleError:
                return self
            if not e.is_integer and f.is_negative and not f.has(x):
                ndir = (b - f).dir(x, cdir)
                if im(ndir).is_negative:
                    # Normally, f**e would evaluate to exp(e*log(f)) but on branch cuts
                    # an other value is expected through the following computation
                    # exp(e*(log(f) - 2*pi*I)) == f**e*exp(-2*e*pi*I) == f**e*(-1)**(-2*e).
                    return self.func(f, e) * (-1)**(-2*e)
                elif im(ndir).is_zero:
                    log_leadterm = log(b)._eval_as_leading_term(x, logx=logx, cdir=cdir)
                    if log_leadterm.is_infinite is False:
                        return exp(e*log_leadterm)
            return self.func(f, e)

    @cacheit
    def _taylor_term(self, n, x, *previous_terms): # of (1 + x)**e
        from sympy.functions.combinatorial.factorials import binomial
        return binomial(self.exp, n) * self.func(x, n)

    def taylor_term(self, n, x, *previous_terms):
        if self.base is not S.Exp1:
            return super().taylor_term(n, x, *previous_terms)
        if n < 0:
            return S.Zero
        if n == 0:
            return S.One
        from .sympify import sympify
        x = sympify(x)
        if previous_terms:
            p = previous_terms[-1]
            if p is not None:
                return p * x / n
        from sympy.functions.combinatorial.factorials import factorial
        return x**n/factorial(n)

    def _eval_rewrite_as_sin(self, base, exp, **hints):
        if self.base is S.Exp1:
            from sympy.functions.elementary.trigonometric import sin
            return sin(S.ImaginaryUnit*self.exp + S.Pi/2) - S.ImaginaryUnit*sin(S.ImaginaryUnit*self.exp)

    def _eval_rewrite_as_cos(self, base, exp, **hints):
        if self.base is S.Exp1:
            from sympy.functions.elementary.trigonometric import cos
            return cos(S.ImaginaryUnit*self.exp) + S.ImaginaryUnit*cos(S.ImaginaryUnit*self.exp + S.Pi/2)

    def _eval_rewrite_as_tanh(self, base, exp, **hints):
        if self.base is S.Exp1:
            from sympy.functions.elementary.hyperbolic import tanh
            return (1 + tanh(self.exp/2))/(1 - tanh(self.exp/2))

    def _eval_rewrite_as_sqrt(self, base, exp, **kwargs):
        from sympy.functions.elementary.trigonometric import sin, cos
        if base is not S.Exp1:
            return None
        if exp.is_Mul:
            coeff = exp.coeff(S.Pi * S.ImaginaryUnit)
            if coeff and coeff.is_number:
                cosine, sine = cos(S.Pi*coeff), sin(S.Pi*coeff)
                if not isinstance(cosine, cos) and not isinstance (sine, sin):
                    return cosine + S.ImaginaryUnit*sine

    def as_content_primitive(self, radical=False, clear=True):
        """Return the tuple (R, self/R) where R is the positive Rational
        extracted from self.

        Examples
        ========

        >>> from sympy import sqrt
        >>> sqrt(4 + 4*sqrt(2)).as_content_primitive()
        (2, sqrt(1 + sqrt(2)))
        >>> sqrt(3 + 3*sqrt(2)).as_content_primitive()
        (1, sqrt(3)*sqrt(1 + sqrt(2)))

        >>> from sympy import expand_power_base, powsimp, Mul
        >>> from sympy.abc import x, y

        >>> ((2*x + 2)**2).as_content_primitive()
        (4, (x + 1)**2)
        >>> (4**((1 + y)/2)).as_content_primitive()
        (2, 4**(y/2))
        >>> (3**((1 + y)/2)).as_content_primitive()
        (1, 3**((y + 1)/2))
        >>> (3**((5 + y)/2)).as_content_primitive()
        (9, 3**((y + 1)/2))
        >>> eq = 3**(2 + 2*x)
        >>> powsimp(eq) == eq
        True
        >>> eq.as_content_primitive()
        (9, 3**(2*x))
        >>> powsimp(Mul(*_))
        3**(2*x + 2)

        >>> eq = (2 + 2*x)**y
        >>> s = expand_power_base(eq); s.is_Mul, s
        (False, (2*x + 2)**y)
        >>> eq.as_content_primitive()
        (1, (2*(x + 1))**y)
        >>> s = expand_power_base(_[1]); s.is_Mul, s
        (True, 2**y*(x + 1)**y)

        See docstring of Expr.as_content_primitive for more examples.
        """

        b, e = self.as_base_exp()
        b = _keep_coeff(*b.as_content_primitive(radical=radical, clear=clear))
        ce, pe = e.as_content_primitive(radical=radical, clear=clear)
        if b.is_Rational:
            #e
            #= ce*pe
            #= ce*(h + t)
            #= ce*h + ce*t
            #=> self
            #= b**(ce*h)*b**(ce*t)
            #= b**(cehp/cehq)*b**(ce*t)
            #= b**(iceh + r/cehq)*b**(ce*t)
            #= b**(iceh)*b**(r/cehq)*b**(ce*t)
            #= b**(iceh)*b**(ce*t + r/cehq)
            h, t = pe.as_coeff_Add()
            if h.is_Rational and b != S.Zero:
                ceh = ce*h
                c = self.func(b, ceh)
                r = S.Zero
                if not c.is_Rational:
                    iceh, r = divmod(ceh.p, ceh.q)
                    c = self.func(b, iceh)
                return c, self.func(b, _keep_coeff(ce, t + r/ce/ceh.q))
        e = _keep_coeff(ce, pe)
        # b**e = (h*t)**e = h**e*t**e = c*m*t**e
        if e.is_Rational and b.is_Mul:
            h, t = b.as_content_primitive(radical=radical, clear=clear)  # h is positive
            c, m = self.func(h, e).as_coeff_Mul()  # so c is positive
            m, me = m.as_base_exp()
            if m is S.One or me == e:  # probably always true
                # return the following, not return c, m*Pow(t, e)
                # which would change Pow into Mul; we let SymPy
                # decide what to do by using the unevaluated Mul, e.g
                # should it stay as sqrt(2 + 2*sqrt(5)) or become
                # sqrt(2)*sqrt(1 + sqrt(5))
                return c, self.func(_keep_coeff(m, t), e)
        return S.One, self.func(b, e)

    def is_constant(self, *wrt, **flags):
        expr = self
        if flags.get('simplify', True):
            expr = expr.simplify()
        b, e = expr.as_base_exp()
        bz = b.equals(0)
        if bz:  # recalculate with assumptions in case it's unevaluated
            new = b**e
            if new != expr:
                return new.is_constant()
        econ = e.is_constant(*wrt)
        bcon = b.is_constant(*wrt)
        if bcon:
            if econ:
                return True
            bz = b.equals(0)
            if bz is False:
                return False
        elif bcon is None:
            return None

        return e.equals(0)

    def _eval_difference_delta(self, n, step):
        b, e = self.args
        if e.has(n) and not b.has(n):
            new_e = e.subs(n, n + step)
            return (b**(new_e - e) - 1) * self

power = Dispatcher('power')
power.add((object, object), Pow)

from .add import Add
from .numbers import Integer, Rational
from .mul import Mul, _keep_coeff
from .symbol import Symbol, Dummy, symbols
