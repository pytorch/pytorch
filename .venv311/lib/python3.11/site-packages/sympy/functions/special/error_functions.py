""" This module contains various functions that are special cases
    of incomplete gamma functions. It should probably be renamed. """

from sympy.core import EulerGamma # Must be imported from core, not core.numbers
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.function import DefinedFunction, ArgumentIndexError, expand_mul
from sympy.core.logic import fuzzy_or
from sympy.core.numbers import I, pi, Rational, Integer
from sympy.core.relational import is_eq
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, uniquely_named_symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial, factorial2, RisingFactorial
from sympy.functions.elementary.complexes import  polar_lift, re, unpolarify
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.miscellaneous import sqrt, root
from sympy.functions.elementary.exponential import exp, log, exp_polar
from sympy.functions.elementary.hyperbolic import cosh, sinh
from sympy.functions.elementary.trigonometric import cos, sin, sinc
from sympy.functions.special.hyper import hyper, meijerg

# TODO series expansions
# TODO see the "Note:" in Ei

# Helper function
def real_to_real_as_real_imag(self, deep=True, **hints):
    if self.args[0].is_extended_real:
        if deep:
            hints['complex'] = False
            return (self.expand(deep, **hints), S.Zero)
        else:
            return (self, S.Zero)
    if deep:
        x, y = self.args[0].expand(deep, **hints).as_real_imag()
    else:
        x, y = self.args[0].as_real_imag()
    re = (self.func(x + I*y) + self.func(x - I*y))/2
    im = (self.func(x + I*y) - self.func(x - I*y))/(2*I)
    return (re, im)


###############################################################################
################################ ERROR FUNCTION ###############################
###############################################################################


class erf(DefinedFunction):
    r"""
    The Gauss error function.

    Explanation
    ===========

    This function is defined as:

    .. math ::
        \mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} \mathrm{d}t.

    Examples
    ========

    >>> from sympy import I, oo, erf
    >>> from sympy.abc import z

    Several special values are known:

    >>> erf(0)
    0
    >>> erf(oo)
    1
    >>> erf(-oo)
    -1
    >>> erf(I*oo)
    oo*I
    >>> erf(-I*oo)
    -oo*I

    In general one can pull out factors of -1 and $I$ from the argument:

    >>> erf(-z)
    -erf(z)

    The error function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(erf(z))
    erf(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(erf(z), z)
    2*exp(-z**2)/sqrt(pi)

    We can numerically evaluate the error function to arbitrary precision
    on the whole complex plane:

    >>> erf(4).evalf(30)
    0.999999984582742099719981147840

    >>> erf(-4*I).evalf(30)
    -1296959.73071763923152794095062*I

    See Also
    ========

    erfc: Complementary error function.
    erfi: Imaginary error function.
    erf2: Two-argument error function.
    erfinv: Inverse error function.
    erfcinv: Inverse Complementary error function.
    erf2inv: Inverse two-argument error function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Error_function
    .. [2] https://dlmf.nist.gov/7
    .. [3] https://mathworld.wolfram.com/Erf.html
    .. [4] https://functions.wolfram.com/GammaBetaErf/Erf

    """

    unbranched = True

    def fdiff(self, argindex=1):
        if argindex == 1:
            return 2*exp(-self.args[0]**2)/sqrt(pi)
        else:
            raise ArgumentIndexError(self, argindex)


    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.

        """
        return erfinv

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.One
            elif arg is S.NegativeInfinity:
                return S.NegativeOne
            elif arg.is_zero:
                return S.Zero

        if isinstance(arg, erfinv):
            return arg.args[0]

        if isinstance(arg, erfcinv):
            return S.One - arg.args[0]

        if arg.is_zero:
            return S.Zero

        # Only happens with unevaluated erf2inv
        if isinstance(arg, erf2inv) and arg.args[0].is_zero:
            return arg.args[1]

        # Try to pull out factors of I
        t = arg.extract_multiplicatively(I)
        if t in (S.Infinity, S.NegativeInfinity):
            return arg

        # Try to pull out factors of -1
        if arg.could_extract_minus_sign():
            return -cls(-arg)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            k = floor((n - 1)/S(2))
            if len(previous_terms) > 2:
                return -previous_terms[-2] * x**2 * (n - 2)/(n*k)
            else:
                return 2*S.NegativeOne**k * x**n/(n*factorial(k)*sqrt(pi))

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def _eval_is_real(self):
        if self.args[0].is_extended_real is True:
            return True
        # There are cases where erf(z) becomes a real number
        # even if z is a complex number

    def _eval_is_imaginary(self):
        if self.args[0].is_imaginary is True:
            return True

    def _eval_is_finite(self):
        z = self.args[0]
        return fuzzy_or([z.is_finite, z.is_extended_real])

    def _eval_is_zero(self):
        if self.args[0].is_extended_real is True:
            return self.args[0].is_zero

    def _eval_is_positive(self):
        if self.args[0].is_extended_real is True:
            return self.args[0].is_extended_positive

    def _eval_is_negative(self):
        if self.args[0].is_extended_real is True:
            return self.args[0].is_extended_negative

    def _eval_rewrite_as_uppergamma(self, z, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return sqrt(z**2)/z*(S.One - uppergamma(S.Half, z**2)/sqrt(pi))

    def _eval_rewrite_as_fresnels(self, z, **kwargs):
        arg = (S.One - I)*z/sqrt(pi)
        return (S.One + I)*(fresnelc(arg) - I*fresnels(arg))

    def _eval_rewrite_as_fresnelc(self, z, **kwargs):
        arg = (S.One - I)*z/sqrt(pi)
        return (S.One + I)*(fresnelc(arg) - I*fresnels(arg))

    def _eval_rewrite_as_meijerg(self, z, **kwargs):
        return z/sqrt(pi)*meijerg([S.Half], [], [0], [Rational(-1, 2)], z**2)

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        return 2*z/sqrt(pi)*hyper([S.Half], [3*S.Half], -z**2)

    def _eval_rewrite_as_expint(self, z, **kwargs):
        return sqrt(z**2)/z - z*expint(S.Half, z**2)/sqrt(pi)

    def _eval_rewrite_as_tractable(self, z, limitvar=None, **kwargs):
        from sympy.series.limits import limit
        if limitvar:
            lim = limit(z, limitvar, S.Infinity)
            if lim is S.NegativeInfinity:
                return S.NegativeOne + _erfs(-z)*exp(-z**2)
        return S.One - _erfs(z)*exp(-z**2)

    def _eval_rewrite_as_erfc(self, z, **kwargs):
        return S.One - erfc(z)

    def _eval_rewrite_as_erfi(self, z, **kwargs):
        return -I*erfi(I*z)

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        if arg0 is S.ComplexInfinity:
            arg0 = arg.limit(x, 0, dir='-' if cdir == -1 else '+')
        if x in arg.free_symbols and arg0.is_zero:
            return 2*arg/sqrt(pi)
        else:
            return self.func(arg0)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]

        if point in [S.Infinity, S.NegativeInfinity]:
            z = self.args[0]

            try:
                _, ex = z.leadterm(x)
            except (ValueError, NotImplementedError):
                return self

            ex = -ex # as x->1/x for aseries
            if ex.is_positive:
                newn = ceiling(n/ex)
                s = [S.NegativeOne**k * factorial2(2*k - 1) / (z**(2*k + 1) * 2**k)
                     for k in range(newn)] + [Order(1/z**newn, x)]
                return S.One - (exp(-z**2)/sqrt(pi)) * Add(*s)

        return super(erf, self)._eval_aseries(n, args0, x, logx)

    as_real_imag = real_to_real_as_real_imag


class erfc(DefinedFunction):
    r"""
    Complementary Error Function.

    Explanation
    ===========

    The function is defined as:

    .. math ::
        \mathrm{erfc}(x) = \frac{2}{\sqrt{\pi}} \int_x^\infty e^{-t^2} \mathrm{d}t

    Examples
    ========

    >>> from sympy import I, oo, erfc
    >>> from sympy.abc import z

    Several special values are known:

    >>> erfc(0)
    1
    >>> erfc(oo)
    0
    >>> erfc(-oo)
    2
    >>> erfc(I*oo)
    -oo*I
    >>> erfc(-I*oo)
    oo*I

    The error function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(erfc(z))
    erfc(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(erfc(z), z)
    -2*exp(-z**2)/sqrt(pi)

    It also follows

    >>> erfc(-z)
    2 - erfc(z)

    We can numerically evaluate the complementary error function to arbitrary
    precision on the whole complex plane:

    >>> erfc(4).evalf(30)
    0.0000000154172579002800188521596734869

    >>> erfc(4*I).evalf(30)
    1.0 - 1296959.73071763923152794095062*I

    See Also
    ========

    erf: Gaussian error function.
    erfi: Imaginary error function.
    erf2: Two-argument error function.
    erfinv: Inverse error function.
    erfcinv: Inverse Complementary error function.
    erf2inv: Inverse two-argument error function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Error_function
    .. [2] https://dlmf.nist.gov/7
    .. [3] https://mathworld.wolfram.com/Erfc.html
    .. [4] https://functions.wolfram.com/GammaBetaErf/Erfc

    """

    unbranched = True

    def fdiff(self, argindex=1):
        if argindex == 1:
            return -2*exp(-self.args[0]**2)/sqrt(pi)
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.

        """
        return erfcinv

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Zero
            elif arg.is_zero:
                return S.One

        if isinstance(arg, erfinv):
            return S.One - arg.args[0]

        if isinstance(arg, erfcinv):
            return arg.args[0]

        if arg.is_zero:
            return S.One

        # Try to pull out factors of I
        t = arg.extract_multiplicatively(I)
        if t in (S.Infinity, S.NegativeInfinity):
            return -arg

        # Try to pull out factors of -1
        if arg.could_extract_minus_sign():
            return 2 - cls(-arg)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return S.One
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            k = floor((n - 1)/S(2))
            if len(previous_terms) > 2:
                return -previous_terms[-2] * x**2 * (n - 2)/(n*k)
            else:
                return -2*S.NegativeOne**k * x**n/(n*factorial(k)*sqrt(pi))

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def _eval_is_real(self):
        if self.args[0].is_extended_real is True:
            return True
        if self.args[0].is_imaginary is True:
            return False

    def _eval_rewrite_as_tractable(self, z, limitvar=None, **kwargs):
        return self.rewrite(erf).rewrite("tractable", deep=True, limitvar=limitvar)

    def _eval_rewrite_as_erf(self, z, **kwargs):
        return S.One - erf(z)

    def _eval_rewrite_as_erfi(self, z, **kwargs):
        return S.One + I*erfi(I*z)

    def _eval_rewrite_as_fresnels(self, z, **kwargs):
        arg = (S.One - I)*z/sqrt(pi)
        return S.One - (S.One + I)*(fresnelc(arg) - I*fresnels(arg))

    def _eval_rewrite_as_fresnelc(self, z, **kwargs):
        arg = (S.One-I)*z/sqrt(pi)
        return S.One - (S.One + I)*(fresnelc(arg) - I*fresnels(arg))

    def _eval_rewrite_as_meijerg(self, z, **kwargs):
        return S.One - z/sqrt(pi)*meijerg([S.Half], [], [0], [Rational(-1, 2)], z**2)

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        return S.One - 2*z/sqrt(pi)*hyper([S.Half], [3*S.Half], -z**2)

    def _eval_rewrite_as_uppergamma(self, z, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return S.One - sqrt(z**2)/z*(S.One - uppergamma(S.Half, z**2)/sqrt(pi))

    def _eval_rewrite_as_expint(self, z, **kwargs):
        return S.One - sqrt(z**2)/z + z*expint(S.Half, z**2)/sqrt(pi)

    def _eval_expand_func(self, **hints):
        return self.rewrite(erf)

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        if arg0 is S.ComplexInfinity:
            arg0 = arg.limit(x, 0, dir='-' if cdir == -1 else '+')
        if arg0.is_zero:
            return S.One
        else:
            return self.func(arg0)

    as_real_imag = real_to_real_as_real_imag

    def _eval_aseries(self, n, args0, x, logx):
        return S.One - erf(*self.args)._eval_aseries(n, args0, x, logx)


class erfi(DefinedFunction):
    r"""
    Imaginary error function.

    Explanation
    ===========

    The function erfi is defined as:

    .. math ::
        \mathrm{erfi}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{t^2} \mathrm{d}t

    Examples
    ========

    >>> from sympy import I, oo, erfi
    >>> from sympy.abc import z

    Several special values are known:

    >>> erfi(0)
    0
    >>> erfi(oo)
    oo
    >>> erfi(-oo)
    -oo
    >>> erfi(I*oo)
    I
    >>> erfi(-I*oo)
    -I

    In general one can pull out factors of -1 and $I$ from the argument:

    >>> erfi(-z)
    -erfi(z)

    >>> from sympy import conjugate
    >>> conjugate(erfi(z))
    erfi(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(erfi(z), z)
    2*exp(z**2)/sqrt(pi)

    We can numerically evaluate the imaginary error function to arbitrary
    precision on the whole complex plane:

    >>> erfi(2).evalf(30)
    18.5648024145755525987042919132

    >>> erfi(-2*I).evalf(30)
    -0.995322265018952734162069256367*I

    See Also
    ========

    erf: Gaussian error function.
    erfc: Complementary error function.
    erf2: Two-argument error function.
    erfinv: Inverse error function.
    erfcinv: Inverse Complementary error function.
    erf2inv: Inverse two-argument error function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Error_function
    .. [2] https://mathworld.wolfram.com/Erfi.html
    .. [3] https://functions.wolfram.com/GammaBetaErf/Erfi

    """

    unbranched = True

    def fdiff(self, argindex=1):
        if argindex == 1:
            return 2*exp(self.args[0]**2)/sqrt(pi)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, z):
        if z.is_Number:
            if z is S.NaN:
                return S.NaN
            elif z.is_zero:
                return S.Zero
            elif z is S.Infinity:
                return S.Infinity

        if z.is_zero:
            return S.Zero

        # Try to pull out factors of -1
        if z.could_extract_minus_sign():
            return -cls(-z)

        # Try to pull out factors of I
        nz = z.extract_multiplicatively(I)
        if nz is not None:
            if nz is S.Infinity:
                return I
            if isinstance(nz, erfinv):
                return I*nz.args[0]
            if isinstance(nz, erfcinv):
                return I*(S.One - nz.args[0])
            # Only happens with unevaluated erf2inv
            if isinstance(nz, erf2inv) and nz.args[0].is_zero:
                return I*nz.args[1]

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            k = floor((n - 1)/S(2))
            if len(previous_terms) > 2:
                return previous_terms[-2] * x**2 * (n - 2)/(n*k)
            else:
                return 2 * x**n/(n*factorial(k)*sqrt(pi))

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real

    def _eval_is_zero(self):
        return self.args[0].is_zero

    def _eval_rewrite_as_tractable(self, z, limitvar=None, **kwargs):
        return self.rewrite(erf).rewrite("tractable", deep=True, limitvar=limitvar)

    def _eval_rewrite_as_erf(self, z, **kwargs):
        return -I*erf(I*z)

    def _eval_rewrite_as_erfc(self, z, **kwargs):
        return I*erfc(I*z) - I

    def _eval_rewrite_as_fresnels(self, z, **kwargs):
        arg = (S.One + I)*z/sqrt(pi)
        return (S.One - I)*(fresnelc(arg) - I*fresnels(arg))

    def _eval_rewrite_as_fresnelc(self, z, **kwargs):
        arg = (S.One + I)*z/sqrt(pi)
        return (S.One - I)*(fresnelc(arg) - I*fresnels(arg))

    def _eval_rewrite_as_meijerg(self, z, **kwargs):
        return z/sqrt(pi)*meijerg([S.Half], [], [0], [Rational(-1, 2)], -z**2)

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        return 2*z/sqrt(pi)*hyper([S.Half], [3*S.Half], z**2)

    def _eval_rewrite_as_uppergamma(self, z, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return sqrt(-z**2)/z*(uppergamma(S.Half, -z**2)/sqrt(pi) - S.One)

    def _eval_rewrite_as_expint(self, z, **kwargs):
        return sqrt(-z**2)/z - z*expint(S.Half, -z**2)/sqrt(pi)

    def _eval_expand_func(self, **hints):
        return self.rewrite(erf)

    as_real_imag = real_to_real_as_real_imag

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        if x in arg.free_symbols and arg0.is_zero:
            return 2*arg/sqrt(pi)
        elif arg0.is_finite:
            return self.func(arg0)
        return self.func(arg)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]

        if point is S.Infinity:
            z = self.args[0]
            s = [factorial2(2*k - 1) / (2**k * z**(2*k + 1))
                    for k in range(n)] + [Order(1/z**n, x)]
            return -I + (exp(z**2)/sqrt(pi)) * Add(*s)

        return super(erfi, self)._eval_aseries(n, args0, x, logx)


class erf2(DefinedFunction):
    r"""
    Two-argument error function.

    Explanation
    ===========

    This function is defined as:

    .. math ::
        \mathrm{erf2}(x, y) = \frac{2}{\sqrt{\pi}} \int_x^y e^{-t^2} \mathrm{d}t

    Examples
    ========

    >>> from sympy import oo, erf2
    >>> from sympy.abc import x, y

    Several special values are known:

    >>> erf2(0, 0)
    0
    >>> erf2(x, x)
    0
    >>> erf2(x, oo)
    1 - erf(x)
    >>> erf2(x, -oo)
    -erf(x) - 1
    >>> erf2(oo, y)
    erf(y) - 1
    >>> erf2(-oo, y)
    erf(y) + 1

    In general one can pull out factors of -1:

    >>> erf2(-x, -y)
    -erf2(x, y)

    The error function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(erf2(x, y))
    erf2(conjugate(x), conjugate(y))

    Differentiation with respect to $x$, $y$ is supported:

    >>> from sympy import diff
    >>> diff(erf2(x, y), x)
    -2*exp(-x**2)/sqrt(pi)
    >>> diff(erf2(x, y), y)
    2*exp(-y**2)/sqrt(pi)

    See Also
    ========

    erf: Gaussian error function.
    erfc: Complementary error function.
    erfi: Imaginary error function.
    erfinv: Inverse error function.
    erfcinv: Inverse Complementary error function.
    erf2inv: Inverse two-argument error function.

    References
    ==========

    .. [1] https://functions.wolfram.com/GammaBetaErf/Erf2/

    """


    def fdiff(self, argindex):
        x, y = self.args
        if argindex == 1:
            return -2*exp(-x**2)/sqrt(pi)
        elif argindex == 2:
            return 2*exp(-y**2)/sqrt(pi)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, x, y):
        chk = (S.Infinity, S.NegativeInfinity, S.Zero)
        if x is S.NaN or y is S.NaN:
            return S.NaN
        elif x == y:
            return S.Zero
        elif x in chk or y in chk:
            return erf(y) - erf(x)

        if isinstance(y, erf2inv) and y.args[0] == x:
            return y.args[1]

        if x.is_zero or y.is_zero or x.is_extended_real and x.is_infinite or \
                y.is_extended_real and y.is_infinite:
            return erf(y) - erf(x)

        #Try to pull out -1 factor
        sign_x = x.could_extract_minus_sign()
        sign_y = y.could_extract_minus_sign()
        if (sign_x and sign_y):
            return -cls(-x, -y)
        elif (sign_x or sign_y):
            return erf(y)-erf(x)

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate(), self.args[1].conjugate())

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real and self.args[1].is_extended_real

    def _eval_rewrite_as_erf(self, x, y, **kwargs):
        return erf(y) - erf(x)

    def _eval_rewrite_as_erfc(self, x, y, **kwargs):
        return erfc(x) - erfc(y)

    def _eval_rewrite_as_erfi(self, x, y, **kwargs):
        return I*(erfi(I*x)-erfi(I*y))

    def _eval_rewrite_as_fresnels(self, x, y, **kwargs):
        return erf(y).rewrite(fresnels) - erf(x).rewrite(fresnels)

    def _eval_rewrite_as_fresnelc(self, x, y, **kwargs):
        return erf(y).rewrite(fresnelc) - erf(x).rewrite(fresnelc)

    def _eval_rewrite_as_meijerg(self, x, y, **kwargs):
        return erf(y).rewrite(meijerg) - erf(x).rewrite(meijerg)

    def _eval_rewrite_as_hyper(self, x, y, **kwargs):
        return erf(y).rewrite(hyper) - erf(x).rewrite(hyper)

    def _eval_rewrite_as_uppergamma(self, x, y, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return (sqrt(y**2)/y*(S.One - uppergamma(S.Half, y**2)/sqrt(pi)) -
            sqrt(x**2)/x*(S.One - uppergamma(S.Half, x**2)/sqrt(pi)))

    def _eval_rewrite_as_expint(self, x, y, **kwargs):
        return erf(y).rewrite(expint) - erf(x).rewrite(expint)

    def _eval_expand_func(self, **hints):
        return self.rewrite(erf)

    def _eval_is_zero(self):
        return is_eq(*self.args)

class erfinv(DefinedFunction):
    r"""
    Inverse Error Function. The erfinv function is defined as:

    .. math ::
        \mathrm{erf}(x) = y \quad \Rightarrow \quad \mathrm{erfinv}(y) = x

    Examples
    ========

    >>> from sympy import erfinv
    >>> from sympy.abc import x

    Several special values are known:

    >>> erfinv(0)
    0
    >>> erfinv(1)
    oo

    Differentiation with respect to $x$ is supported:

    >>> from sympy import diff
    >>> diff(erfinv(x), x)
    sqrt(pi)*exp(erfinv(x)**2)/2

    We can numerically evaluate the inverse error function to arbitrary
    precision on [-1, 1]:

    >>> erfinv(0.2).evalf(30)
    0.179143454621291692285822705344

    See Also
    ========

    erf: Gaussian error function.
    erfc: Complementary error function.
    erfi: Imaginary error function.
    erf2: Two-argument error function.
    erfcinv: Inverse Complementary error function.
    erf2inv: Inverse two-argument error function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Error_function#Inverse_functions
    .. [2] https://functions.wolfram.com/GammaBetaErf/InverseErf/

    """


    def fdiff(self, argindex =1):
        if argindex == 1:
            return sqrt(pi)*exp(self.func(self.args[0])**2)*S.Half
        else :
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.

        """
        return erf

    @classmethod
    def eval(cls, z):
        if z is S.NaN:
            return S.NaN
        elif z is S.NegativeOne:
            return S.NegativeInfinity
        elif z.is_zero:
            return S.Zero
        elif z is S.One:
            return S.Infinity

        if isinstance(z, erf) and z.args[0].is_extended_real:
            return z.args[0]

        if z.is_zero:
            return S.Zero

        # Try to pull out factors of -1
        nz = z.extract_multiplicatively(-1)
        if nz is not None and (isinstance(nz, erf) and (nz.args[0]).is_extended_real):
            return -nz.args[0]

    def _eval_rewrite_as_erfcinv(self, z, **kwargs):
        return erfcinv(1-z)

    def _eval_is_zero(self):
        return self.args[0].is_zero


class erfcinv (DefinedFunction):
    r"""
    Inverse Complementary Error Function. The erfcinv function is defined as:

    .. math ::
        \mathrm{erfc}(x) = y \quad \Rightarrow \quad \mathrm{erfcinv}(y) = x

    Examples
    ========

    >>> from sympy import erfcinv
    >>> from sympy.abc import x

    Several special values are known:

    >>> erfcinv(1)
    0
    >>> erfcinv(0)
    oo

    Differentiation with respect to $x$ is supported:

    >>> from sympy import diff
    >>> diff(erfcinv(x), x)
    -sqrt(pi)*exp(erfcinv(x)**2)/2

    See Also
    ========

    erf: Gaussian error function.
    erfc: Complementary error function.
    erfi: Imaginary error function.
    erf2: Two-argument error function.
    erfinv: Inverse error function.
    erf2inv: Inverse two-argument error function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Error_function#Inverse_functions
    .. [2] https://functions.wolfram.com/GammaBetaErf/InverseErfc/

    """


    def fdiff(self, argindex =1):
        if argindex == 1:
            return -sqrt(pi)*exp(self.func(self.args[0])**2)*S.Half
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.

        """
        return erfc

    @classmethod
    def eval(cls, z):
        if z is S.NaN:
            return S.NaN
        elif z.is_zero:
            return S.Infinity
        elif z is S.One:
            return S.Zero
        elif z == 2:
            return S.NegativeInfinity

        if z.is_zero:
            return S.Infinity

    def _eval_rewrite_as_erfinv(self, z, **kwargs):
        return erfinv(1-z)

    def _eval_is_zero(self):
        return (self.args[0] - 1).is_zero

    def _eval_is_infinite(self):
        z = self.args[0]
        return fuzzy_or([z.is_zero, is_eq(z, Integer(2))])


class erf2inv(DefinedFunction):
    r"""
    Two-argument Inverse error function. The erf2inv function is defined as:

    .. math ::
        \mathrm{erf2}(x, w) = y \quad \Rightarrow \quad \mathrm{erf2inv}(x, y) = w

    Examples
    ========

    >>> from sympy import erf2inv, oo
    >>> from sympy.abc import x, y

    Several special values are known:

    >>> erf2inv(0, 0)
    0
    >>> erf2inv(1, 0)
    1
    >>> erf2inv(0, 1)
    oo
    >>> erf2inv(0, y)
    erfinv(y)
    >>> erf2inv(oo, y)
    erfcinv(-y)

    Differentiation with respect to $x$ and $y$ is supported:

    >>> from sympy import diff
    >>> diff(erf2inv(x, y), x)
    exp(-x**2 + erf2inv(x, y)**2)
    >>> diff(erf2inv(x, y), y)
    sqrt(pi)*exp(erf2inv(x, y)**2)/2

    See Also
    ========

    erf: Gaussian error function.
    erfc: Complementary error function.
    erfi: Imaginary error function.
    erf2: Two-argument error function.
    erfinv: Inverse error function.
    erfcinv: Inverse complementary error function.

    References
    ==========

    .. [1] https://functions.wolfram.com/GammaBetaErf/InverseErf2/

    """


    def fdiff(self, argindex):
        x, y = self.args
        if argindex == 1:
            return exp(self.func(x,y)**2-x**2)
        elif argindex == 2:
            return sqrt(pi)*S.Half*exp(self.func(x,y)**2)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, x, y):
        if x is S.NaN or y is S.NaN:
            return S.NaN
        elif x.is_zero and y.is_zero:
            return S.Zero
        elif x.is_zero and y is S.One:
            return S.Infinity
        elif x is S.One and y.is_zero:
            return S.One
        elif x.is_zero:
            return erfinv(y)
        elif x is S.Infinity:
            return erfcinv(-y)
        elif y.is_zero:
            return x
        elif y is S.Infinity:
            return erfinv(x)

        if x.is_zero:
            if y.is_zero:
                return S.Zero
            else:
                return erfinv(y)
        if y.is_zero:
            return x

    def _eval_is_zero(self):
        x, y = self.args
        if x.is_zero and y.is_zero:
            return True

###############################################################################
#################### EXPONENTIAL INTEGRALS ####################################
###############################################################################

class Ei(DefinedFunction):
    r"""
    The classical exponential integral.

    Explanation
    ===========

    For use in SymPy, this function is defined as

    .. math:: \operatorname{Ei}(x) = \sum_{n=1}^\infty \frac{x^n}{n\, n!}
                                     + \log(x) + \gamma,

    where $\gamma$ is the Euler-Mascheroni constant.

    If $x$ is a polar number, this defines an analytic function on the
    Riemann surface of the logarithm. Otherwise this defines an analytic
    function in the cut plane $\mathbb{C} \setminus (-\infty, 0]$.

    **Background**

    The name exponential integral comes from the following statement:

    .. math:: \operatorname{Ei}(x) = \int_{-\infty}^x \frac{e^t}{t} \mathrm{d}t

    If the integral is interpreted as a Cauchy principal value, this statement
    holds for $x > 0$ and $\operatorname{Ei}(x)$ as defined above.

    Examples
    ========

    >>> from sympy import Ei, polar_lift, exp_polar, I, pi
    >>> from sympy.abc import x

    >>> Ei(-1)
    Ei(-1)

    This yields a real value:

    >>> Ei(-1).n(chop=True)
    -0.219383934395520

    On the other hand the analytic continuation is not real:

    >>> Ei(polar_lift(-1)).n(chop=True)
    -0.21938393439552 + 3.14159265358979*I

    The exponential integral has a logarithmic branch point at the origin:

    >>> Ei(x*exp_polar(2*I*pi))
    Ei(x) + 2*I*pi

    Differentiation is supported:

    >>> Ei(x).diff(x)
    exp(x)/x

    The exponential integral is related to many other special functions.
    For example:

    >>> from sympy import expint, Shi
    >>> Ei(x).rewrite(expint)
    -expint(1, x*exp_polar(I*pi)) - I*pi
    >>> Ei(x).rewrite(Shi)
    Chi(x) + Shi(x)

    See Also
    ========

    expint: Generalised exponential integral.
    E1: Special case of the generalised exponential integral.
    li: Logarithmic integral.
    Li: Offset logarithmic integral.
    Si: Sine integral.
    Ci: Cosine integral.
    Shi: Hyperbolic sine integral.
    Chi: Hyperbolic cosine integral.
    uppergamma: Upper incomplete gamma function.

    References
    ==========

    .. [1] https://dlmf.nist.gov/6.6
    .. [2] https://en.wikipedia.org/wiki/Exponential_integral
    .. [3] Abramowitz & Stegun, section 5: https://web.archive.org/web/20201128173312/http://people.math.sfu.ca/~cbm/aands/page_228.htm

    """


    @classmethod
    def eval(cls, z):
        if z.is_zero:
            return S.NegativeInfinity
        elif z is S.Infinity:
            return S.Infinity
        elif z is S.NegativeInfinity:
            return S.Zero

        if z.is_zero:
            return S.NegativeInfinity

        nz, n = z.extract_branch_factor()
        if n:
            return Ei(nz) + 2*I*pi*n

    def fdiff(self, argindex=1):
        arg = unpolarify(self.args[0])
        if argindex == 1:
            return exp(arg)/arg
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_evalf(self, prec):
        if (self.args[0]/polar_lift(-1)).is_positive:
            return super()._eval_evalf(prec) + (I*pi)._eval_evalf(prec)
        return super()._eval_evalf(prec)

    def _eval_rewrite_as_uppergamma(self, z, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        # XXX this does not currently work usefully because uppergamma
        #     immediately turns into expint
        return -uppergamma(0, polar_lift(-1)*z) - I*pi

    def _eval_rewrite_as_expint(self, z, **kwargs):
        return -expint(1, polar_lift(-1)*z) - I*pi

    def _eval_rewrite_as_li(self, z, **kwargs):
        if isinstance(z, log):
            return li(z.args[0])
        # TODO:
        # Actually it only holds that:
        #  Ei(z) = li(exp(z))
        # for -pi < imag(z) <= pi
        return li(exp(z))

    def _eval_rewrite_as_Si(self, z, **kwargs):
        if z.is_negative:
            return Shi(z) + Chi(z) - I*pi
        else:
            return Shi(z) + Chi(z)
    _eval_rewrite_as_Ci = _eval_rewrite_as_Si
    _eval_rewrite_as_Chi = _eval_rewrite_as_Si
    _eval_rewrite_as_Shi = _eval_rewrite_as_Si

    def _eval_rewrite_as_tractable(self, z, limitvar=None, **kwargs):
        return exp(z) * _eis(z)

    def _eval_rewrite_as_Integral(self, z, **kwargs):
        from sympy.integrals.integrals import Integral
        t = Dummy(uniquely_named_symbol('t', [z]).name)
        return Integral(S.Exp1**t/t, (t, S.NegativeInfinity, z))

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy import re
        x0 = self.args[0].limit(x, 0)
        arg = self.args[0].as_leading_term(x, cdir=cdir)
        cdir = arg.dir(x, cdir)
        if x0.is_zero:
            c, e = arg.as_coeff_exponent(x)
            logx = log(x) if logx is None else logx
            return log(c) + e*logx + EulerGamma - (
                I*pi if re(cdir).is_negative else S.Zero)
        return super()._eval_as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_nseries(self, x, n, logx, cdir=0):
        x0 = self.args[0].limit(x, 0)
        if x0.is_zero:
            f = self._eval_rewrite_as_Si(*self.args)
            return f._eval_nseries(x, n, logx)
        return super()._eval_nseries(x, n, logx)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]

        if point in (S.Infinity, S.NegativeInfinity):
            z = self.args[0]
            s = [factorial(k) / (z)**k for k in range(n)] + \
                    [Order(1/z**n, x)]
            return (exp(z)/z) * Add(*s)

        return super(Ei, self)._eval_aseries(n, args0, x, logx)


class expint(DefinedFunction):
    r"""
    Generalized exponential integral.

    Explanation
    ===========

    This function is defined as

    .. math:: \operatorname{E}_\nu(z) = z^{\nu - 1} \Gamma(1 - \nu, z),

    where $\Gamma(1 - \nu, z)$ is the upper incomplete gamma function
    (``uppergamma``).

    Hence for $z$ with positive real part we have

    .. math:: \operatorname{E}_\nu(z)
              =   \int_1^\infty \frac{e^{-zt}}{t^\nu} \mathrm{d}t,

    which explains the name.

    The representation as an incomplete gamma function provides an analytic
    continuation for $\operatorname{E}_\nu(z)$. If $\nu$ is a
    non-positive integer, the exponential integral is thus an unbranched
    function of $z$, otherwise there is a branch point at the origin.
    Refer to the incomplete gamma function documentation for details of the
    branching behavior.

    Examples
    ========

    >>> from sympy import expint, S
    >>> from sympy.abc import nu, z

    Differentiation is supported. Differentiation with respect to $z$ further
    explains the name: for integral orders, the exponential integral is an
    iterated integral of the exponential function.

    >>> expint(nu, z).diff(z)
    -expint(nu - 1, z)

    Differentiation with respect to $\nu$ has no classical expression:

    >>> expint(nu, z).diff(nu)
    -z**(nu - 1)*meijerg(((), (1, 1)), ((0, 0, 1 - nu), ()), z)

    At non-postive integer orders, the exponential integral reduces to the
    exponential function:

    >>> expint(0, z)
    exp(-z)/z
    >>> expint(-1, z)
    exp(-z)/z + exp(-z)/z**2

    At half-integers it reduces to error functions:

    >>> expint(S(1)/2, z)
    sqrt(pi)*erfc(sqrt(z))/sqrt(z)

    At positive integer orders it can be rewritten in terms of exponentials
    and ``expint(1, z)``. Use ``expand_func()`` to do this:

    >>> from sympy import expand_func
    >>> expand_func(expint(5, z))
    z**4*expint(1, z)/24 + (-z**3 + z**2 - 2*z + 6)*exp(-z)/24

    The generalised exponential integral is essentially equivalent to the
    incomplete gamma function:

    >>> from sympy import uppergamma
    >>> expint(nu, z).rewrite(uppergamma)
    z**(nu - 1)*uppergamma(1 - nu, z)

    As such it is branched at the origin:

    >>> from sympy import exp_polar, pi, I
    >>> expint(4, z*exp_polar(2*pi*I))
    I*pi*z**3/3 + expint(4, z)
    >>> expint(nu, z*exp_polar(2*pi*I))
    z**(nu - 1)*(exp(2*I*pi*nu) - 1)*gamma(1 - nu) + expint(nu, z)

    See Also
    ========

    Ei: Another related function called exponential integral.
    E1: The classical case, returns expint(1, z).
    li: Logarithmic integral.
    Li: Offset logarithmic integral.
    Si: Sine integral.
    Ci: Cosine integral.
    Shi: Hyperbolic sine integral.
    Chi: Hyperbolic cosine integral.
    uppergamma

    References
    ==========

    .. [1] https://dlmf.nist.gov/8.19
    .. [2] https://functions.wolfram.com/GammaBetaErf/ExpIntegralE/
    .. [3] https://en.wikipedia.org/wiki/Exponential_integral

    """


    @classmethod
    def eval(cls, nu, z):
        from sympy.functions.special.gamma_functions import (gamma, uppergamma)
        nu2 = unpolarify(nu)
        if nu != nu2:
            return expint(nu2, z)
        if nu.is_Integer and nu <= 0 or (not nu.is_Integer and (2*nu).is_Integer):
            return unpolarify(expand_mul(z**(nu - 1)*uppergamma(1 - nu, z)))

        # Extract branching information. This can be deduced from what is
        # explained in lowergamma.eval().
        z, n = z.extract_branch_factor()
        if n is S.Zero:
            return
        if nu.is_integer:
            if not nu > 0:
                return
            return expint(nu, z) \
                - 2*pi*I*n*S.NegativeOne**(nu - 1)/factorial(nu - 1)*unpolarify(z)**(nu - 1)
        else:
            return (exp(2*I*pi*nu*n) - 1)*z**(nu - 1)*gamma(1 - nu) + expint(nu, z)

    def fdiff(self, argindex):
        nu, z = self.args
        if argindex == 1:
            return -z**(nu - 1)*meijerg([], [1, 1], [0, 0, 1 - nu], [], z)
        elif argindex == 2:
            return -expint(nu - 1, z)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_uppergamma(self, nu, z, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return z**(nu - 1)*uppergamma(1 - nu, z)

    def _eval_rewrite_as_Ei(self, nu, z, **kwargs):
        if nu == 1:
            return -Ei(z*exp_polar(-I*pi)) - I*pi
        elif nu.is_Integer and nu > 1:
            # DLMF, 8.19.7
            x = -unpolarify(z)
            return x**(nu - 1)/factorial(nu - 1)*E1(z).rewrite(Ei) + \
                exp(x)/factorial(nu - 1) * \
                Add(*[factorial(nu - k - 2)*x**k for k in range(nu - 1)])
        else:
            return self

    def _eval_expand_func(self, **hints):
        return self.rewrite(Ei).rewrite(expint, **hints)

    def _eval_rewrite_as_Si(self, nu, z, **kwargs):
        if nu != 1:
            return self
        return Shi(z) - Chi(z)
    _eval_rewrite_as_Ci = _eval_rewrite_as_Si
    _eval_rewrite_as_Chi = _eval_rewrite_as_Si
    _eval_rewrite_as_Shi = _eval_rewrite_as_Si

    def _eval_nseries(self, x, n, logx, cdir=0):
        if not self.args[0].has(x):
            nu = self.args[0]
            if nu == 1:
                f = self._eval_rewrite_as_Si(*self.args)
                return f._eval_nseries(x, n, logx)
            elif nu.is_Integer and nu > 1:
                f = self._eval_rewrite_as_Ei(*self.args)
                return f._eval_nseries(x, n, logx)
        return super()._eval_nseries(x, n, logx)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[1]
        nu = self.args[0]

        if point is S.Infinity:
            z = self.args[1]
            s = [S.NegativeOne**k * RisingFactorial(nu, k) / z**k for k in range(n)] + [Order(1/z**n, x)]
            return (exp(-z)/z) * Add(*s)

        return super(expint, self)._eval_aseries(n, args0, x, logx)

    def _eval_rewrite_as_Integral(self, *args, **kwargs):
        from sympy.integrals.integrals import Integral
        n, x = self.args
        t = Dummy(uniquely_named_symbol('t', args).name)
        return Integral(t**-n * exp(-t*x), (t, 1, S.Infinity))


def E1(z):
    """
    Classical case of the generalized exponential integral.

    Explanation
    ===========

    This is equivalent to ``expint(1, z)``.

    Examples
    ========

    >>> from sympy import E1
    >>> E1(0)
    expint(1, 0)

    >>> E1(5)
    expint(1, 5)

    See Also
    ========

    Ei: Exponential integral.
    expint: Generalised exponential integral.
    li: Logarithmic integral.
    Li: Offset logarithmic integral.
    Si: Sine integral.
    Ci: Cosine integral.
    Shi: Hyperbolic sine integral.
    Chi: Hyperbolic cosine integral.

    """
    return expint(1, z)


class li(DefinedFunction):
    r"""
    The classical logarithmic integral.

    Explanation
    ===========

    For use in SymPy, this function is defined as

    .. math:: \operatorname{li}(x) = \int_0^x \frac{1}{\log(t)} \mathrm{d}t \,.

    Examples
    ========

    >>> from sympy import I, oo, li
    >>> from sympy.abc import z

    Several special values are known:

    >>> li(0)
    0
    >>> li(1)
    -oo
    >>> li(oo)
    oo

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(li(z), z)
    1/log(z)

    Defining the ``li`` function via an integral:
    >>> from sympy import integrate
    >>> integrate(li(z))
    z*li(z) - Ei(2*log(z))

    >>> integrate(li(z),z)
    z*li(z) - Ei(2*log(z))


    The logarithmic integral can also be defined in terms of ``Ei``:

    >>> from sympy import Ei
    >>> li(z).rewrite(Ei)
    Ei(log(z))
    >>> diff(li(z).rewrite(Ei), z)
    1/log(z)

    We can numerically evaluate the logarithmic integral to arbitrary precision
    on the whole complex plane (except the singular points):

    >>> li(2).evalf(30)
    1.04516378011749278484458888919

    >>> li(2*I).evalf(30)
    1.0652795784357498247001125598 + 3.08346052231061726610939702133*I

    We can even compute Soldner's constant by the help of mpmath:

    >>> from mpmath import findroot
    >>> findroot(li, 2)
    1.45136923488338

    Further transformations include rewriting ``li`` in terms of
    the trigonometric integrals ``Si``, ``Ci``, ``Shi`` and ``Chi``:

    >>> from sympy import Si, Ci, Shi, Chi
    >>> li(z).rewrite(Si)
    -log(I*log(z)) - log(1/log(z))/2 + log(log(z))/2 + Ci(I*log(z)) + Shi(log(z))
    >>> li(z).rewrite(Ci)
    -log(I*log(z)) - log(1/log(z))/2 + log(log(z))/2 + Ci(I*log(z)) + Shi(log(z))
    >>> li(z).rewrite(Shi)
    -log(1/log(z))/2 + log(log(z))/2 + Chi(log(z)) - Shi(log(z))
    >>> li(z).rewrite(Chi)
    -log(1/log(z))/2 + log(log(z))/2 + Chi(log(z)) - Shi(log(z))

    See Also
    ========

    Li: Offset logarithmic integral.
    Ei: Exponential integral.
    expint: Generalised exponential integral.
    E1: Special case of the generalised exponential integral.
    Si: Sine integral.
    Ci: Cosine integral.
    Shi: Hyperbolic sine integral.
    Chi: Hyperbolic cosine integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Logarithmic_integral
    .. [2] https://mathworld.wolfram.com/LogarithmicIntegral.html
    .. [3] https://dlmf.nist.gov/6
    .. [4] https://mathworld.wolfram.com/SoldnersConstant.html

    """


    @classmethod
    def eval(cls, z):
        if z.is_zero:
            return S.Zero
        elif z is S.One:
            return S.NegativeInfinity
        elif z is S.Infinity:
            return S.Infinity
        if z.is_zero:
            return S.Zero

    def fdiff(self, argindex=1):
        arg = self.args[0]
        if argindex == 1:
            return S.One / log(arg)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_conjugate(self):
        z = self.args[0]
        # Exclude values on the branch cut (-oo, 0)
        if not z.is_extended_negative:
            return self.func(z.conjugate())

    def _eval_rewrite_as_Li(self, z, **kwargs):
        return Li(z) + li(2)

    def _eval_rewrite_as_Ei(self, z, **kwargs):
        return Ei(log(z))

    def _eval_rewrite_as_uppergamma(self, z, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return (-uppergamma(0, -log(z)) +
                S.Half*(log(log(z)) - log(S.One/log(z))) - log(-log(z)))

    def _eval_rewrite_as_Si(self, z, **kwargs):
        return (Ci(I*log(z)) - I*Si(I*log(z)) -
                S.Half*(log(S.One/log(z)) - log(log(z))) - log(I*log(z)))

    _eval_rewrite_as_Ci = _eval_rewrite_as_Si

    def _eval_rewrite_as_Shi(self, z, **kwargs):
        return (Chi(log(z)) - Shi(log(z)) - S.Half*(log(S.One/log(z)) - log(log(z))))

    _eval_rewrite_as_Chi = _eval_rewrite_as_Shi

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        return (log(z)*hyper((1, 1), (2, 2), log(z)) +
                S.Half*(log(log(z)) - log(S.One/log(z))) + EulerGamma)

    def _eval_rewrite_as_meijerg(self, z, **kwargs):
        return (-log(-log(z)) - S.Half*(log(S.One/log(z)) - log(log(z)))
                - meijerg(((), (1,)), ((0, 0), ()), -log(z)))

    def _eval_rewrite_as_tractable(self, z, limitvar=None, **kwargs):
        return z * _eis(log(z))

    def _eval_nseries(self, x, n, logx, cdir=0):
        z = self.args[0]
        s = [(log(z))**k / (factorial(k) * k) for k in range(1, n)]
        return EulerGamma + log(log(z)) + Add(*s)

    def _eval_is_zero(self):
        z = self.args[0]
        if z.is_zero:
            return True

class Li(DefinedFunction):
    r"""
    The offset logarithmic integral.

    Explanation
    ===========

    For use in SymPy, this function is defined as

    .. math:: \operatorname{Li}(x) = \operatorname{li}(x) - \operatorname{li}(2)

    Examples
    ========

    >>> from sympy import Li
    >>> from sympy.abc import z

    The following special value is known:

    >>> Li(2)
    0

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(Li(z), z)
    1/log(z)

    The shifted logarithmic integral can be written in terms of $li(z)$:

    >>> from sympy import li
    >>> Li(z).rewrite(li)
    li(z) - li(2)

    We can numerically evaluate the logarithmic integral to arbitrary precision
    on the whole complex plane (except the singular points):

    >>> Li(2).evalf(30)
    0

    >>> Li(4).evalf(30)
    1.92242131492155809316615998938

    See Also
    ========

    li: Logarithmic integral.
    Ei: Exponential integral.
    expint: Generalised exponential integral.
    E1: Special case of the generalised exponential integral.
    Si: Sine integral.
    Ci: Cosine integral.
    Shi: Hyperbolic sine integral.
    Chi: Hyperbolic cosine integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Logarithmic_integral
    .. [2] https://mathworld.wolfram.com/LogarithmicIntegral.html
    .. [3] https://dlmf.nist.gov/6

    """


    @classmethod
    def eval(cls, z):
        if z is S.Infinity:
            return S.Infinity
        elif z == S(2):
            return S.Zero

    def fdiff(self, argindex=1):
        arg = self.args[0]
        if argindex == 1:
            return S.One / log(arg)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_evalf(self, prec):
        return self.rewrite(li).evalf(prec)

    def _eval_rewrite_as_li(self, z, **kwargs):
        return li(z) - li(2)

    def _eval_rewrite_as_tractable(self, z, limitvar=None, **kwargs):
        return self.rewrite(li).rewrite("tractable", deep=True)

    def _eval_nseries(self, x, n, logx, cdir=0):
        f = self._eval_rewrite_as_li(*self.args)
        return f._eval_nseries(x, n, logx)

###############################################################################
#################### TRIGONOMETRIC INTEGRALS ##################################
###############################################################################

class TrigonometricIntegral(DefinedFunction):
    """ Base class for trigonometric integrals. """


    @classmethod
    def eval(cls, z):
        if z is S.Zero:
            return cls._atzero
        elif z is S.Infinity:
            return cls._atinf()
        elif z is S.NegativeInfinity:
            return cls._atneginf()

        if z.is_zero:
            return cls._atzero

        nz = z.extract_multiplicatively(polar_lift(I))
        if nz is None and cls._trigfunc(0) == 0:
            nz = z.extract_multiplicatively(I)
        if nz is not None:
            return cls._Ifactor(nz, 1)
        nz = z.extract_multiplicatively(polar_lift(-I))
        if nz is not None:
            return cls._Ifactor(nz, -1)

        nz = z.extract_multiplicatively(polar_lift(-1))
        if nz is None and cls._trigfunc(0) == 0:
            nz = z.extract_multiplicatively(-1)
        if nz is not None:
            return cls._minusfactor(nz)

        nz, n = z.extract_branch_factor()
        if n == 0 and nz == z:
            return
        return 2*pi*I*n*cls._trigfunc(0) + cls(nz)

    def fdiff(self, argindex=1):
        arg = unpolarify(self.args[0])
        if argindex == 1:
            return self._trigfunc(arg)/arg
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Ei(self, z, **kwargs):
        return self._eval_rewrite_as_expint(z).rewrite(Ei)

    def _eval_rewrite_as_uppergamma(self, z, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return self._eval_rewrite_as_expint(z).rewrite(uppergamma)

    def _eval_nseries(self, x, n, logx, cdir=0):
        # NOTE this is fairly inefficient
        if self.args[0].subs(x, 0) != 0:
            return super()._eval_nseries(x, n, logx)
        baseseries = self._trigfunc(x)._eval_nseries(x, n, logx)
        if self._trigfunc(0) != 0:
            baseseries -= 1
        baseseries = baseseries.replace(Pow, lambda t, n: t**n/n, simultaneous=False)
        if self._trigfunc(0) != 0:
            baseseries += EulerGamma + log(x)
        return baseseries.subs(x, self.args[0])._eval_nseries(x, n, logx)


class Si(TrigonometricIntegral):
    r"""
    Sine integral.

    Explanation
    ===========

    This function is defined by

    .. math:: \operatorname{Si}(z) = \int_0^z \frac{\sin{t}}{t} \mathrm{d}t.

    It is an entire function.

    Examples
    ========

    >>> from sympy import Si
    >>> from sympy.abc import z

    The sine integral is an antiderivative of $sin(z)/z$:

    >>> Si(z).diff(z)
    sin(z)/z

    It is unbranched:

    >>> from sympy import exp_polar, I, pi
    >>> Si(z*exp_polar(2*I*pi))
    Si(z)

    Sine integral behaves much like ordinary sine under multiplication by ``I``:

    >>> Si(I*z)
    I*Shi(z)
    >>> Si(-z)
    -Si(z)

    It can also be expressed in terms of exponential integrals, but beware
    that the latter is branched:

    >>> from sympy import expint
    >>> Si(z).rewrite(expint)
    -I*(-expint(1, z*exp_polar(-I*pi/2))/2 +
         expint(1, z*exp_polar(I*pi/2))/2) + pi/2

    It can be rewritten in the form of sinc function (by definition):

    >>> from sympy import sinc
    >>> Si(z).rewrite(sinc)
    Integral(sinc(_t), (_t, 0, z))

    See Also
    ========

    Ci: Cosine integral.
    Shi: Hyperbolic sine integral.
    Chi: Hyperbolic cosine integral.
    Ei: Exponential integral.
    expint: Generalised exponential integral.
    sinc: unnormalized sinc function
    E1: Special case of the generalised exponential integral.
    li: Logarithmic integral.
    Li: Offset logarithmic integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_integral

    """

    _trigfunc = sin
    _atzero = S.Zero

    @classmethod
    def _atinf(cls):
        return pi*S.Half

    @classmethod
    def _atneginf(cls):
        return -pi*S.Half

    @classmethod
    def _minusfactor(cls, z):
        return -Si(z)

    @classmethod
    def _Ifactor(cls, z, sign):
        return I*Shi(z)*sign

    def _eval_rewrite_as_expint(self, z, **kwargs):
        # XXX should we polarify z?
        return pi/2 + (E1(polar_lift(I)*z) - E1(polar_lift(-I)*z))/2/I

    def _eval_rewrite_as_Integral(self, z, **kwargs):
        from sympy.integrals.integrals import Integral
        t = Dummy(uniquely_named_symbol('t', [z]).name)
        return Integral(sinc(t), (t, 0, z))

    _eval_rewrite_as_sinc =  _eval_rewrite_as_Integral

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if arg0.is_zero:
            return arg
        elif not arg0.is_infinite:
            return self.func(arg0)
        else:
            return self

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]

        # Expansion at oo
        if point is S.Infinity:
            z = self.args[0]
            p = [S.NegativeOne**k * factorial(2*k) / z**(2*k + 1)
                    for k in range(n//2 + 1)] + [Order(1/z**n, x)]
            q = [S.NegativeOne**k * factorial(2*k + 1) / z**(2*(k + 1))
                    for k in range(n//2)] + [Order(1/z**n, x)]
            return pi/2 - cos(z)*Add(*p) - sin(z)*Add(*q)

        # All other points are not handled
        return super(Si, self)._eval_aseries(n, args0, x, logx)

    def _eval_is_zero(self):
        z = self.args[0]
        if z.is_zero:
            return True


class Ci(TrigonometricIntegral):
    r"""
    Cosine integral.

    Explanation
    ===========

    This function is defined for positive $x$ by

    .. math:: \operatorname{Ci}(x) = \gamma + \log{x}
                         + \int_0^x \frac{\cos{t} - 1}{t} \mathrm{d}t
           = -\int_x^\infty \frac{\cos{t}}{t} \mathrm{d}t,

    where $\gamma$ is the Euler-Mascheroni constant.

    We have

    .. math:: \operatorname{Ci}(z) =
        -\frac{\operatorname{E}_1\left(e^{i\pi/2} z\right)
               + \operatorname{E}_1\left(e^{-i \pi/2} z\right)}{2}

    which holds for all polar $z$ and thus provides an analytic
    continuation to the Riemann surface of the logarithm.

    The formula also holds as stated
    for $z \in \mathbb{C}$ with $\Re(z) > 0$.
    By lifting to the principal branch, we obtain an analytic function on the
    cut complex plane.

    Examples
    ========

    >>> from sympy import Ci
    >>> from sympy.abc import z

    The cosine integral is a primitive of $\cos(z)/z$:

    >>> Ci(z).diff(z)
    cos(z)/z

    It has a logarithmic branch point at the origin:

    >>> from sympy import exp_polar, I, pi
    >>> Ci(z*exp_polar(2*I*pi))
    Ci(z) + 2*I*pi

    The cosine integral behaves somewhat like ordinary $\cos$ under
    multiplication by $i$:

    >>> from sympy import polar_lift
    >>> Ci(polar_lift(I)*z)
    Chi(z) + I*pi/2
    >>> Ci(polar_lift(-1)*z)
    Ci(z) + I*pi

    It can also be expressed in terms of exponential integrals:

    >>> from sympy import expint
    >>> Ci(z).rewrite(expint)
    -expint(1, z*exp_polar(-I*pi/2))/2 - expint(1, z*exp_polar(I*pi/2))/2

    See Also
    ========

    Si: Sine integral.
    Shi: Hyperbolic sine integral.
    Chi: Hyperbolic cosine integral.
    Ei: Exponential integral.
    expint: Generalised exponential integral.
    E1: Special case of the generalised exponential integral.
    li: Logarithmic integral.
    Li: Offset logarithmic integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_integral

    """

    _trigfunc = cos
    _atzero = S.ComplexInfinity

    @classmethod
    def _atinf(cls):
        return S.Zero

    @classmethod
    def _atneginf(cls):
        return I*pi

    @classmethod
    def _minusfactor(cls, z):
        return Ci(z) + I*pi

    @classmethod
    def _Ifactor(cls, z, sign):
        return Chi(z) + I*pi/2*sign

    def _eval_rewrite_as_expint(self, z, **kwargs):
        return -(E1(polar_lift(I)*z) + E1(polar_lift(-I)*z))/2

    def _eval_rewrite_as_Integral(self, z, **kwargs):
        from sympy.integrals.integrals import Integral
        t = Dummy(uniquely_named_symbol('t', [z]).name)
        return S.EulerGamma + log(z) - Integral((1-cos(t))/t, (t, 0, z))

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if arg0.is_zero:
            c, e = arg.as_coeff_exponent(x)
            logx = log(x) if logx is None else logx
            return log(c) + e*logx + EulerGamma
        elif arg0.is_finite:
            return self.func(arg0)
        else:
            return self

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]

        if point in (S.Infinity, S.NegativeInfinity):
            z = self.args[0]
            p = [S.NegativeOne**k * factorial(2*k) / z**(2*k + 1)
                    for k in range(n//2 + 1)] + [Order(1/z**n, x)]
            q = [S.NegativeOne**k * factorial(2*k + 1) / z**(2*(k + 1))
                    for k in range(n//2)] + [Order(1/z**n, x)]
            result = sin(z)*(Add(*p)) - cos(z)*(Add(*q))

            if point is S.NegativeInfinity:
                result += I*pi
            return result

        return super(Ci, self)._eval_aseries(n, args0, x, logx)

class Shi(TrigonometricIntegral):
    r"""
    Sinh integral.

    Explanation
    ===========

    This function is defined by

    .. math:: \operatorname{Shi}(z) = \int_0^z \frac{\sinh{t}}{t} \mathrm{d}t.

    It is an entire function.

    Examples
    ========

    >>> from sympy import Shi
    >>> from sympy.abc import z

    The Sinh integral is a primitive of $\sinh(z)/z$:

    >>> Shi(z).diff(z)
    sinh(z)/z

    It is unbranched:

    >>> from sympy import exp_polar, I, pi
    >>> Shi(z*exp_polar(2*I*pi))
    Shi(z)

    The $\sinh$ integral behaves much like ordinary $\sinh$ under
    multiplication by $i$:

    >>> Shi(I*z)
    I*Si(z)
    >>> Shi(-z)
    -Shi(z)

    It can also be expressed in terms of exponential integrals, but beware
    that the latter is branched:

    >>> from sympy import expint
    >>> Shi(z).rewrite(expint)
    expint(1, z)/2 - expint(1, z*exp_polar(I*pi))/2 - I*pi/2

    See Also
    ========

    Si: Sine integral.
    Ci: Cosine integral.
    Chi: Hyperbolic cosine integral.
    Ei: Exponential integral.
    expint: Generalised exponential integral.
    E1: Special case of the generalised exponential integral.
    li: Logarithmic integral.
    Li: Offset logarithmic integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_integral

    """

    _trigfunc = sinh
    _atzero = S.Zero

    @classmethod
    def _atinf(cls):
        return S.Infinity

    @classmethod
    def _atneginf(cls):
        return S.NegativeInfinity

    @classmethod
    def _minusfactor(cls, z):
        return -Shi(z)

    @classmethod
    def _Ifactor(cls, z, sign):
        return I*Si(z)*sign

    def _eval_rewrite_as_expint(self, z, **kwargs):
        # XXX should we polarify z?
        return (E1(z) - E1(exp_polar(I*pi)*z))/2 - I*pi/2

    def _eval_is_zero(self):
        z = self.args[0]
        if z.is_zero:
            return True

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0].as_leading_term(x)
        arg0 = arg.subs(x, 0)

        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if arg0.is_zero:
            return arg
        elif not arg0.is_infinite:
            return self.func(arg0)
        else:
            return self


class Chi(TrigonometricIntegral):
    r"""
    Cosh integral.

    Explanation
    ===========

    This function is defined for positive $x$ by

    .. math:: \operatorname{Chi}(x) = \gamma + \log{x}
                         + \int_0^x \frac{\cosh{t} - 1}{t} \mathrm{d}t,

    where $\gamma$ is the Euler-Mascheroni constant.

    We have

    .. math:: \operatorname{Chi}(z) = \operatorname{Ci}\left(e^{i \pi/2}z\right)
                         - i\frac{\pi}{2},

    which holds for all polar $z$ and thus provides an analytic
    continuation to the Riemann surface of the logarithm.
    By lifting to the principal branch we obtain an analytic function on the
    cut complex plane.

    Examples
    ========

    >>> from sympy import Chi
    >>> from sympy.abc import z

    The $\cosh$ integral is a primitive of $\cosh(z)/z$:

    >>> Chi(z).diff(z)
    cosh(z)/z

    It has a logarithmic branch point at the origin:

    >>> from sympy import exp_polar, I, pi
    >>> Chi(z*exp_polar(2*I*pi))
    Chi(z) + 2*I*pi

    The $\cosh$ integral behaves somewhat like ordinary $\cosh$ under
    multiplication by $i$:

    >>> from sympy import polar_lift
    >>> Chi(polar_lift(I)*z)
    Ci(z) + I*pi/2
    >>> Chi(polar_lift(-1)*z)
    Chi(z) + I*pi

    It can also be expressed in terms of exponential integrals:

    >>> from sympy import expint
    >>> Chi(z).rewrite(expint)
    -expint(1, z)/2 - expint(1, z*exp_polar(I*pi))/2 - I*pi/2

    See Also
    ========

    Si: Sine integral.
    Ci: Cosine integral.
    Shi: Hyperbolic sine integral.
    Ei: Exponential integral.
    expint: Generalised exponential integral.
    E1: Special case of the generalised exponential integral.
    li: Logarithmic integral.
    Li: Offset logarithmic integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_integral

    """

    _trigfunc = cosh
    _atzero = S.ComplexInfinity

    @classmethod
    def _atinf(cls):
        return S.Infinity

    @classmethod
    def _atneginf(cls):
        return S.Infinity

    @classmethod
    def _minusfactor(cls, z):
        return Chi(z) + I*pi

    @classmethod
    def _Ifactor(cls, z, sign):
        return Ci(z) + I*pi/2*sign

    def _eval_rewrite_as_expint(self, z, **kwargs):
        return -I*pi/2 - (E1(z) + E1(exp_polar(I*pi)*z))/2

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if arg0.is_zero:
            c, e = arg.as_coeff_exponent(x)
            logx = log(x) if logx is None else logx
            return log(c) + e*logx + EulerGamma
        elif arg0.is_finite:
            return self.func(arg0)
        else:
            return self


###############################################################################
#################### FRESNEL INTEGRALS ########################################
###############################################################################

class FresnelIntegral(DefinedFunction):
    """ Base class for the Fresnel integrals."""

    unbranched = True

    @classmethod
    def eval(cls, z):
        # Values at positive infinities signs
        # if any were extracted automatically
        if z is S.Infinity:
            return S.Half

        # Value at zero
        if z.is_zero:
            return S.Zero

        # Try to pull out factors of -1 and I
        prefact = S.One
        newarg = z
        changed = False

        nz = newarg.extract_multiplicatively(-1)
        if nz is not None:
            prefact = -prefact
            newarg = nz
            changed = True

        nz = newarg.extract_multiplicatively(I)
        if nz is not None:
            prefact = cls._sign*I*prefact
            newarg = nz
            changed = True

        if changed:
            return prefact*cls(newarg)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return self._trigfunc(S.Half*pi*self.args[0]**2)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real

    _eval_is_finite = _eval_is_extended_real

    def _eval_is_zero(self):
        return self.args[0].is_zero

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    as_real_imag = real_to_real_as_real_imag


class fresnels(FresnelIntegral):
    r"""
    Fresnel integral S.

    Explanation
    ===========

    This function is defined by

    .. math:: \operatorname{S}(z) = \int_0^z \sin{\frac{\pi}{2} t^2} \mathrm{d}t.

    It is an entire function.

    Examples
    ========

    >>> from sympy import I, oo, fresnels
    >>> from sympy.abc import z

    Several special values are known:

    >>> fresnels(0)
    0
    >>> fresnels(oo)
    1/2
    >>> fresnels(-oo)
    -1/2
    >>> fresnels(I*oo)
    -I/2
    >>> fresnels(-I*oo)
    I/2

    In general one can pull out factors of -1 and $i$ from the argument:

    >>> fresnels(-z)
    -fresnels(z)
    >>> fresnels(I*z)
    -I*fresnels(z)

    The Fresnel S integral obeys the mirror symmetry
    $\overline{S(z)} = S(\bar{z})$:

    >>> from sympy import conjugate
    >>> conjugate(fresnels(z))
    fresnels(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(fresnels(z), z)
    sin(pi*z**2/2)

    Defining the Fresnel functions via an integral:

    >>> from sympy import integrate, pi, sin, expand_func
    >>> integrate(sin(pi*z**2/2), z)
    3*fresnels(z)*gamma(3/4)/(4*gamma(7/4))
    >>> expand_func(integrate(sin(pi*z**2/2), z))
    fresnels(z)

    We can numerically evaluate the Fresnel integral to arbitrary precision
    on the whole complex plane:

    >>> fresnels(2).evalf(30)
    0.343415678363698242195300815958

    >>> fresnels(-2*I).evalf(30)
    0.343415678363698242195300815958*I

    See Also
    ========

    fresnelc: Fresnel cosine integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fresnel_integral
    .. [2] https://dlmf.nist.gov/7
    .. [3] https://mathworld.wolfram.com/FresnelIntegrals.html
    .. [4] https://functions.wolfram.com/GammaBetaErf/FresnelS
    .. [5] The converging factors for the fresnel integrals
            by John W. Wrench Jr. and Vicki Alley

    """
    _trigfunc = sin
    _sign = -S.One

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) > 1:
                p = previous_terms[-1]
                return (-pi**2*x**4*(4*n - 1)/(8*n*(2*n + 1)*(4*n + 3))) * p
            else:
                return x**3 * (-x**4)**n * (S(2)**(-2*n - 1)*pi**(2*n + 1)) / ((4*n + 3)*factorial(2*n + 1))

    def _eval_rewrite_as_erf(self, z, **kwargs):
        return (S.One + I)/4 * (erf((S.One + I)/2*sqrt(pi)*z) - I*erf((S.One - I)/2*sqrt(pi)*z))

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        return pi*z**3/6 * hyper([Rational(3, 4)], [Rational(3, 2), Rational(7, 4)], -pi**2*z**4/16)

    def _eval_rewrite_as_meijerg(self, z, **kwargs):
        return (pi*z**Rational(9, 4) / (sqrt(2)*(z**2)**Rational(3, 4)*(-z)**Rational(3, 4))
                * meijerg([], [1], [Rational(3, 4)], [Rational(1, 4), 0], -pi**2*z**4/16))

    def _eval_rewrite_as_Integral(self, z, **kwargs):
        from sympy.integrals.integrals import Integral
        t = Dummy(uniquely_named_symbol('t', [z]).name)
        return Integral(sin(pi*t**2/2), (t, 0, z))

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.series.order import Order
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        if arg0 is S.ComplexInfinity:
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if arg0.is_zero:
            return pi*arg**3/6
        elif arg0 in [S.Infinity, S.NegativeInfinity]:
            s = 1 if arg0 is S.Infinity else -1
            return s*S.Half + Order(x, x)
        else:
            return self.func(arg0)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]

        # Expansion at oo and -oo
        if point in [S.Infinity, -S.Infinity]:
            z = self.args[0]

            # expansion of S(x) = S1(x*sqrt(pi/2)), see reference[5] page 1-8
            # as only real infinities are dealt with, sin and cos are O(1)
            p = [S.NegativeOne**k * factorial(4*k + 1) /
                 (2**(2*k + 2) * z**(4*k + 3) * 2**(2*k)*factorial(2*k))
                 for k in range(0, n) if 4*k + 3 < n]
            q = [1/(2*z)] + [S.NegativeOne**k * factorial(4*k - 1) /
                 (2**(2*k + 1) * z**(4*k + 1) * 2**(2*k - 1)*factorial(2*k - 1))
                 for k in range(1, n) if 4*k + 1 < n]

            p = [-sqrt(2/pi)*t for t in p]
            q = [-sqrt(2/pi)*t for t in q]
            s = 1 if point is S.Infinity else -1
            # The expansion at oo is 1/2 + some odd powers of z
            # To get the expansion at -oo, replace z by -z and flip the sign
            # The result -1/2 + the same odd powers of z as before.
            return s*S.Half + (sin(z**2)*Add(*p) + cos(z**2)*Add(*q)
                ).subs(x, sqrt(2/pi)*x) + Order(1/z**n, x)

        # All other points are not handled
        return super()._eval_aseries(n, args0, x, logx)


class fresnelc(FresnelIntegral):
    r"""
    Fresnel integral C.

    Explanation
    ===========

    This function is defined by

    .. math:: \operatorname{C}(z) = \int_0^z \cos{\frac{\pi}{2} t^2} \mathrm{d}t.

    It is an entire function.

    Examples
    ========

    >>> from sympy import I, oo, fresnelc
    >>> from sympy.abc import z

    Several special values are known:

    >>> fresnelc(0)
    0
    >>> fresnelc(oo)
    1/2
    >>> fresnelc(-oo)
    -1/2
    >>> fresnelc(I*oo)
    I/2
    >>> fresnelc(-I*oo)
    -I/2

    In general one can pull out factors of -1 and $i$ from the argument:

    >>> fresnelc(-z)
    -fresnelc(z)
    >>> fresnelc(I*z)
    I*fresnelc(z)

    The Fresnel C integral obeys the mirror symmetry
    $\overline{C(z)} = C(\bar{z})$:

    >>> from sympy import conjugate
    >>> conjugate(fresnelc(z))
    fresnelc(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(fresnelc(z), z)
    cos(pi*z**2/2)

    Defining the Fresnel functions via an integral:

    >>> from sympy import integrate, pi, cos, expand_func
    >>> integrate(cos(pi*z**2/2), z)
    fresnelc(z)*gamma(1/4)/(4*gamma(5/4))
    >>> expand_func(integrate(cos(pi*z**2/2), z))
    fresnelc(z)

    We can numerically evaluate the Fresnel integral to arbitrary precision
    on the whole complex plane:

    >>> fresnelc(2).evalf(30)
    0.488253406075340754500223503357

    >>> fresnelc(-2*I).evalf(30)
    -0.488253406075340754500223503357*I

    See Also
    ========

    fresnels: Fresnel sine integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fresnel_integral
    .. [2] https://dlmf.nist.gov/7
    .. [3] https://mathworld.wolfram.com/FresnelIntegrals.html
    .. [4] https://functions.wolfram.com/GammaBetaErf/FresnelC
    .. [5] The converging factors for the fresnel integrals
            by John W. Wrench Jr. and Vicki Alley

    """
    _trigfunc = cos
    _sign = S.One

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) > 1:
                p = previous_terms[-1]
                return (-pi**2*x**4*(4*n - 3)/(8*n*(2*n - 1)*(4*n + 1))) * p
            else:
                return x * (-x**4)**n * (S(2)**(-2*n)*pi**(2*n)) / ((4*n + 1)*factorial(2*n))

    def _eval_rewrite_as_erf(self, z, **kwargs):
        return (S.One - I)/4 * (erf((S.One + I)/2*sqrt(pi)*z) + I*erf((S.One - I)/2*sqrt(pi)*z))

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        return z * hyper([Rational(1, 4)], [S.Half, Rational(5, 4)], -pi**2*z**4/16)

    def _eval_rewrite_as_meijerg(self, z, **kwargs):
        return (pi*z**Rational(3, 4) / (sqrt(2)*root(z**2, 4)*root(-z, 4))
                * meijerg([], [1], [Rational(1, 4)], [Rational(3, 4), 0], -pi**2*z**4/16))

    def _eval_rewrite_as_Integral(self, z, **kwargs):
        from sympy.integrals.integrals import Integral
        t = Dummy(uniquely_named_symbol('t', [z]).name)
        return Integral(cos(pi*t**2/2), (t, 0, z))

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.series.order import Order
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        if arg0 is S.ComplexInfinity:
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if arg0.is_zero:
            return arg
        elif arg0 in [S.Infinity, S.NegativeInfinity]:
            s = 1 if arg0 is S.Infinity else -1
            return s*S.Half + Order(x, x)
        else:
            return self.func(arg0)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]

        # Expansion at oo
        if point in [S.Infinity, -S.Infinity]:
            z = self.args[0]

            # expansion of C(x) = C1(x*sqrt(pi/2)), see reference[5] page 1-8
            # as only real infinities are dealt with, sin and cos are O(1)
            p = [S.NegativeOne**k * factorial(4*k + 1) /
                 (2**(2*k + 2) * z**(4*k + 3) * 2**(2*k)*factorial(2*k))
                 for k in range(n) if 4*k + 3 < n]
            q = [1/(2*z)] + [S.NegativeOne**k * factorial(4*k - 1) /
                 (2**(2*k + 1) * z**(4*k + 1) * 2**(2*k - 1)*factorial(2*k - 1))
                 for k in range(1, n) if 4*k + 1 < n]

            p = [-sqrt(2/pi)*t for t in p]
            q = [ sqrt(2/pi)*t for t in q]
            s = 1 if point is S.Infinity else -1
            # The expansion at oo is 1/2 + some odd powers of z
            # To get the expansion at -oo, replace z by -z and flip the sign
            # The result -1/2 + the same odd powers of z as before.
            return s*S.Half + (cos(z**2)*Add(*p) + sin(z**2)*Add(*q)
                ).subs(x, sqrt(2/pi)*x) + Order(1/z**n, x)

        # All other points are not handled
        return super()._eval_aseries(n, args0, x, logx)


###############################################################################
#################### HELPER FUNCTIONS #########################################
###############################################################################


class _erfs(DefinedFunction):
    """
    Helper function to make the $\\mathrm{erf}(z)$ function
    tractable for the Gruntz algorithm.

    """
    @classmethod
    def eval(cls, arg):
        if arg.is_zero:
            return S.One

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]

        # Expansion at oo
        if point is S.Infinity:
            z = self.args[0]
            l = [1/sqrt(pi) * factorial(2*k)*(-S(
                 4))**(-k)/factorial(k) * (1/z)**(2*k + 1) for k in range(n)]
            o = Order(1/z**(2*n + 1), x)
            # It is very inefficient to first add the order and then do the nseries
            return (Add(*l))._eval_nseries(x, n, logx) + o

        # Expansion at I*oo
        t = point.extract_multiplicatively(I)
        if t is S.Infinity:
            z = self.args[0]
            # TODO: is the series really correct?
            l = [1/sqrt(pi) * factorial(2*k)*(-S(
                 4))**(-k)/factorial(k) * (1/z)**(2*k + 1) for k in range(n)]
            o = Order(1/z**(2*n + 1), x)
            # It is very inefficient to first add the order and then do the nseries
            return (Add(*l))._eval_nseries(x, n, logx) + o

        # All other points are not handled
        return super()._eval_aseries(n, args0, x, logx)

    def fdiff(self, argindex=1):
        if argindex == 1:
            z = self.args[0]
            return -2/sqrt(pi) + 2*z*_erfs(z)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_intractable(self, z, **kwargs):
        return (S.One - erf(z))*exp(z**2)


class _eis(DefinedFunction):
    """
    Helper function to make the $\\mathrm{Ei}(z)$ and $\\mathrm{li}(z)$
    functions tractable for the Gruntz algorithm.

    """


    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        if args0[0] not in (S.Infinity, S.NegativeInfinity):
            return super()._eval_aseries(n, args0, x, logx)

        z = self.args[0]
        l = [factorial(k) * (1/z)**(k + 1) for k in range(n)]
        o = Order(1/z**(n + 1), x)
        # It is very inefficient to first add the order and then do the nseries
        return (Add(*l))._eval_nseries(x, n, logx) + o


    def fdiff(self, argindex=1):
        if argindex == 1:
            z = self.args[0]
            return S.One / z - _eis(z)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_intractable(self, z, **kwargs):
        return exp(-z)*Ei(z)

    def _eval_as_leading_term(self, x, logx, cdir):
        x0 = self.args[0].limit(x, 0)
        if x0.is_zero:
            f = self._eval_rewrite_as_intractable(*self.args)
            return f._eval_as_leading_term(x, logx=logx, cdir=cdir)
        return super()._eval_as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_nseries(self, x, n, logx, cdir=0):
        x0 = self.args[0].limit(x, 0)
        if x0.is_zero:
            f = self._eval_rewrite_as_intractable(*self.args)
            return f._eval_nseries(x, n, logx)
        return super()._eval_nseries(x, n, logx)
