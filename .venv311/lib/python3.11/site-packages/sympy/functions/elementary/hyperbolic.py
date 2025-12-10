from sympy.core import S, sympify, cacheit
from sympy.core.add import Add
from sympy.core.function import DefinedFunction, ArgumentIndexError
from sympy.core.logic import fuzzy_or, fuzzy_and, fuzzy_not, FuzzyBool
from sympy.core.numbers import I, pi, Rational
from sympy.core.symbol import Dummy
from sympy.functions.combinatorial.factorials import (binomial, factorial,
                                                      RisingFactorial)
from sympy.functions.combinatorial.numbers import bernoulli, euler, nC
from sympy.functions.elementary.complexes import Abs, im, re
from sympy.functions.elementary.exponential import exp, log, match_real_imag
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (
    acos, acot, asin, atan, cos, cot, csc, sec, sin, tan,
    _imaginary_unit_as_coefficient)
from sympy.polys.specialpolys import symmetric_poly


def _rewrite_hyperbolics_as_exp(expr):
    return expr.xreplace({h: h.rewrite(exp)
        for h in expr.atoms(HyperbolicFunction)})


@cacheit
def _acosh_table():
    return {
        I: log(I*(1 + sqrt(2))),
        -I: log(-I*(1 + sqrt(2))),
        S.Half: pi/3,
        Rational(-1, 2): pi*Rational(2, 3),
        sqrt(2)/2: pi/4,
        -sqrt(2)/2: pi*Rational(3, 4),
        1/sqrt(2): pi/4,
        -1/sqrt(2): pi*Rational(3, 4),
        sqrt(3)/2: pi/6,
        -sqrt(3)/2: pi*Rational(5, 6),
        (sqrt(3) - 1)/sqrt(2**3): pi*Rational(5, 12),
        -(sqrt(3) - 1)/sqrt(2**3): pi*Rational(7, 12),
        sqrt(2 + sqrt(2))/2: pi/8,
        -sqrt(2 + sqrt(2))/2: pi*Rational(7, 8),
        sqrt(2 - sqrt(2))/2: pi*Rational(3, 8),
        -sqrt(2 - sqrt(2))/2: pi*Rational(5, 8),
        (1 + sqrt(3))/(2*sqrt(2)): pi/12,
        -(1 + sqrt(3))/(2*sqrt(2)): pi*Rational(11, 12),
        (sqrt(5) + 1)/4: pi/5,
        -(sqrt(5) + 1)/4: pi*Rational(4, 5)
    }


@cacheit
def _acsch_table():
    return {
            I: -pi / 2,
            I*(sqrt(2) + sqrt(6)): -pi / 12,
            I*(1 + sqrt(5)): -pi / 10,
            I*2 / sqrt(2 - sqrt(2)): -pi / 8,
            I*2: -pi / 6,
            I*sqrt(2 + 2/sqrt(5)): -pi / 5,
            I*sqrt(2): -pi / 4,
            I*(sqrt(5)-1): -3*pi / 10,
            I*2 / sqrt(3): -pi / 3,
            I*2 / sqrt(2 + sqrt(2)): -3*pi / 8,
            I*sqrt(2 - 2/sqrt(5)): -2*pi / 5,
            I*(sqrt(6) - sqrt(2)): -5*pi / 12,
            S(2): -I*log((1+sqrt(5))/2),
        }


@cacheit
def _asech_table():
        return {
            I: - (pi*I / 2) + log(1 + sqrt(2)),
            -I: (pi*I / 2) + log(1 + sqrt(2)),
            (sqrt(6) - sqrt(2)): pi / 12,
            (sqrt(2) - sqrt(6)): 11*pi / 12,
            sqrt(2 - 2/sqrt(5)): pi / 10,
            -sqrt(2 - 2/sqrt(5)): 9*pi / 10,
            2 / sqrt(2 + sqrt(2)): pi / 8,
            -2 / sqrt(2 + sqrt(2)): 7*pi / 8,
            2 / sqrt(3): pi / 6,
            -2 / sqrt(3): 5*pi / 6,
            (sqrt(5) - 1): pi / 5,
            (1 - sqrt(5)): 4*pi / 5,
            sqrt(2): pi / 4,
            -sqrt(2): 3*pi / 4,
            sqrt(2 + 2/sqrt(5)): 3*pi / 10,
            -sqrt(2 + 2/sqrt(5)): 7*pi / 10,
            S(2): pi / 3,
            -S(2): 2*pi / 3,
            sqrt(2*(2 + sqrt(2))): 3*pi / 8,
            -sqrt(2*(2 + sqrt(2))): 5*pi / 8,
            (1 + sqrt(5)): 2*pi / 5,
            (-1 - sqrt(5)): 3*pi / 5,
            (sqrt(6) + sqrt(2)): 5*pi / 12,
            (-sqrt(6) - sqrt(2)): 7*pi / 12,
            I*S.Infinity: -pi*I / 2,
            I*S.NegativeInfinity: pi*I / 2,
        }

###############################################################################
########################### HYPERBOLIC FUNCTIONS ##############################
###############################################################################


class HyperbolicFunction(DefinedFunction):
    """
    Base class for hyperbolic functions.

    See Also
    ========

    sinh, cosh, tanh, coth
    """

    unbranched = True


def _peeloff_ipi(arg):
    r"""
    Split ARG into two parts, a "rest" and a multiple of $I\pi$.
    This assumes ARG to be an ``Add``.
    The multiple of $I\pi$ returned in the second position is always a ``Rational``.

    Examples
    ========

    >>> from sympy.functions.elementary.hyperbolic import _peeloff_ipi as peel
    >>> from sympy import pi, I
    >>> from sympy.abc import x, y
    >>> peel(x + I*pi/2)
    (x, 1/2)
    >>> peel(x + I*2*pi/3 + I*pi*y)
    (x + I*pi*y + I*pi/6, 1/2)
    """
    ipi = pi*I
    for a in Add.make_args(arg):
        if a == ipi:
            K = S.One
            break
        elif a.is_Mul:
            K, p = a.as_two_terms()
            if p == ipi and K.is_Rational:
                break
    else:
        return arg, S.Zero

    m1 = (K % S.Half)
    m2 = K - m1
    return arg - m2*ipi, m2


class sinh(HyperbolicFunction):
    r"""
    ``sinh(x)`` is the hyperbolic sine of ``x``.

    The hyperbolic sine function is $\frac{e^x - e^{-x}}{2}$.

    Examples
    ========

    >>> from sympy import sinh
    >>> from sympy.abc import x
    >>> sinh(x)
    sinh(x)

    See Also
    ========

    cosh, tanh, asinh
    """

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            return cosh(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return asinh

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.NegativeInfinity
            elif arg.is_zero:
                return S.Zero
            elif arg.is_negative:
                return -cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                return S.NaN

            i_coeff = _imaginary_unit_as_coefficient(arg)

            if i_coeff is not None:
                return I * sin(i_coeff)
            else:
                if arg.could_extract_minus_sign():
                    return -cls(-arg)

            if arg.is_Add:
                x, m = _peeloff_ipi(arg)
                if m:
                    m = m*pi*I
                    return sinh(m)*cosh(x) + cosh(m)*sinh(x)

            if arg.is_zero:
                return S.Zero

            if arg.func == asinh:
                return arg.args[0]

            if arg.func == acosh:
                x = arg.args[0]
                return sqrt(x - 1) * sqrt(x + 1)

            if arg.func == atanh:
                x = arg.args[0]
                return x/sqrt(1 - x**2)

            if arg.func == acoth:
                x = arg.args[0]
                return 1/(sqrt(x - 1) * sqrt(x + 1))

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        """
        Returns the next term in the Taylor series expansion.
        """
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)

            if len(previous_terms) > 2:
                p = previous_terms[-2]
                return p * x**2 / (n*(n - 1))
            else:
                return x**(n) / factorial(n)

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def as_real_imag(self, deep=True, **hints):
        """
        Returns this function as a complex coordinate.
        """
        if self.args[0].is_extended_real:
            if deep:
                hints['complex'] = False
                return (self.expand(deep, **hints), S.Zero)
            else:
                return (self, S.Zero)
        if deep:
            re, im = self.args[0].expand(deep, **hints).as_real_imag()
        else:
            re, im = self.args[0].as_real_imag()
        return (sinh(re)*cos(im), cosh(re)*sin(im))

    def _eval_expand_complex(self, deep=True, **hints):
        re_part, im_part = self.as_real_imag(deep=deep, **hints)
        return re_part + im_part*I

    def _eval_expand_trig(self, deep=True, **hints):
        if deep:
            arg = self.args[0].expand(deep, **hints)
        else:
            arg = self.args[0]
        x = None
        if arg.is_Add: # TODO, implement more if deep stuff here
            x, y = arg.as_two_terms()
        else:
            coeff, terms = arg.as_coeff_Mul(rational=True)
            if coeff is not S.One and coeff.is_Integer and terms is not S.One:
                x = terms
                y = (coeff - 1)*x
        if x is not None:
            return (sinh(x)*cosh(y) + sinh(y)*cosh(x)).expand(trig=True)
        return sinh(arg)

    def _eval_rewrite_as_tractable(self, arg, limitvar=None, **kwargs):
        return (exp(arg) - exp(-arg)) / 2

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        return (exp(arg) - exp(-arg)) / 2

    def _eval_rewrite_as_sin(self, arg, **kwargs):
        return -I * sin(I * arg)

    def _eval_rewrite_as_csc(self, arg, **kwargs):
        return -I / csc(I * arg)

    def _eval_rewrite_as_cosh(self, arg, **kwargs):
        return -I*cosh(arg + pi*I/2)

    def _eval_rewrite_as_tanh(self, arg, **kwargs):
        tanh_half = tanh(S.Half*arg)
        return 2*tanh_half/(1 - tanh_half**2)

    def _eval_rewrite_as_coth(self, arg, **kwargs):
        coth_half = coth(S.Half*arg)
        return 2*coth_half/(coth_half**2 - 1)

    def _eval_rewrite_as_csch(self, arg, **kwargs):
        return 1 / csch(arg)

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0, dir='-' if cdir.is_negative else '+')
        if arg0.is_zero:
            return arg
        elif arg0.is_finite:
            return self.func(arg0)
        else:
            return self

    def _eval_is_real(self):
        arg = self.args[0]
        if arg.is_real:
            return True

        # if `im` is of the form n*pi
        # else, check if it is a number
        re, im = arg.as_real_imag()
        return (im%pi).is_zero

    def _eval_is_extended_real(self):
        if self.args[0].is_extended_real:
            return True

    def _eval_is_positive(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_positive

    def _eval_is_negative(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_negative

    def _eval_is_finite(self):
        arg = self.args[0]
        return arg.is_finite

    def _eval_is_zero(self):
        rest, ipi_mult = _peeloff_ipi(self.args[0])
        if rest.is_zero:
            return ipi_mult.is_integer


class cosh(HyperbolicFunction):
    r"""
    ``cosh(x)`` is the hyperbolic cosine of ``x``.

    The hyperbolic cosine function is $\frac{e^x + e^{-x}}{2}$.

    Examples
    ========

    >>> from sympy import cosh
    >>> from sympy.abc import x
    >>> cosh(x)
    cosh(x)

    See Also
    ========

    sinh, tanh, acosh
    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            return sinh(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        from sympy.functions.elementary.trigonometric import cos
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.Infinity
            elif arg.is_zero:
                return S.One
            elif arg.is_negative:
                return cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                return S.NaN

            i_coeff = _imaginary_unit_as_coefficient(arg)

            if i_coeff is not None:
                return cos(i_coeff)
            else:
                if arg.could_extract_minus_sign():
                    return cls(-arg)

            if arg.is_Add:
                x, m = _peeloff_ipi(arg)
                if m:
                    m = m*pi*I
                    return cosh(m)*cosh(x) + sinh(m)*sinh(x)

            if arg.is_zero:
                return S.One

            if arg.func == asinh:
                return sqrt(1 + arg.args[0]**2)

            if arg.func == acosh:
                return arg.args[0]

            if arg.func == atanh:
                return 1/sqrt(1 - arg.args[0]**2)

            if arg.func == acoth:
                x = arg.args[0]
                return x/(sqrt(x - 1) * sqrt(x + 1))

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 1:
            return S.Zero
        else:
            x = sympify(x)

            if len(previous_terms) > 2:
                p = previous_terms[-2]
                return p * x**2 / (n*(n - 1))
            else:
                return x**(n)/factorial(n)

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def as_real_imag(self, deep=True, **hints):
        if self.args[0].is_extended_real:
            if deep:
                hints['complex'] = False
                return (self.expand(deep, **hints), S.Zero)
            else:
                return (self, S.Zero)
        if deep:
            re, im = self.args[0].expand(deep, **hints).as_real_imag()
        else:
            re, im = self.args[0].as_real_imag()

        return (cosh(re)*cos(im), sinh(re)*sin(im))

    def _eval_expand_complex(self, deep=True, **hints):
        re_part, im_part = self.as_real_imag(deep=deep, **hints)
        return re_part + im_part*I

    def _eval_expand_trig(self, deep=True, **hints):
        if deep:
            arg = self.args[0].expand(deep, **hints)
        else:
            arg = self.args[0]
        x = None
        if arg.is_Add: # TODO, implement more if deep stuff here
            x, y = arg.as_two_terms()
        else:
            coeff, terms = arg.as_coeff_Mul(rational=True)
            if coeff is not S.One and coeff.is_Integer and terms is not S.One:
                x = terms
                y = (coeff - 1)*x
        if x is not None:
            return (cosh(x)*cosh(y) + sinh(x)*sinh(y)).expand(trig=True)
        return cosh(arg)

    def _eval_rewrite_as_tractable(self, arg, limitvar=None, **kwargs):
        return (exp(arg) + exp(-arg)) / 2

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        return (exp(arg) + exp(-arg)) / 2

    def _eval_rewrite_as_cos(self, arg, **kwargs):
        return cos(I * arg, evaluate=False)

    def _eval_rewrite_as_sec(self, arg, **kwargs):
        return 1 / sec(I * arg, evaluate=False)

    def _eval_rewrite_as_sinh(self, arg, **kwargs):
        return -I*sinh(arg + pi*I/2, evaluate=False)

    def _eval_rewrite_as_tanh(self, arg, **kwargs):
        tanh_half = tanh(S.Half*arg)**2
        return (1 + tanh_half)/(1 - tanh_half)

    def _eval_rewrite_as_coth(self, arg, **kwargs):
        coth_half = coth(S.Half*arg)**2
        return (coth_half + 1)/(coth_half - 1)

    def _eval_rewrite_as_sech(self, arg, **kwargs):
        return 1 / sech(arg)

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0, dir='-' if cdir.is_negative else '+')
        if arg0.is_zero:
            return S.One
        elif arg0.is_finite:
            return self.func(arg0)
        else:
            return self

    def _eval_is_real(self):
        arg = self.args[0]

        # `cosh(x)` is real for real OR purely imaginary `x`
        if arg.is_real or arg.is_imaginary:
            return True

        # cosh(a+ib) = cos(b)*cosh(a) + i*sin(b)*sinh(a)
        # the imaginary part can be an expression like n*pi
        # if not, check if the imaginary part is a number
        re, im = arg.as_real_imag()
        return (im%pi).is_zero

    def _eval_is_positive(self):
        # cosh(x+I*y) = cos(y)*cosh(x) + I*sin(y)*sinh(x)
        # cosh(z) is positive iff it is real and the real part is positive.
        # So we need sin(y)*sinh(x) = 0 which gives x=0 or y=n*pi
        # Case 1 (y=n*pi): cosh(z) = (-1)**n * cosh(x) -> positive for n even
        # Case 2 (x=0): cosh(z) = cos(y) -> positive when cos(y) is positive
        z = self.args[0]

        x, y = z.as_real_imag()
        ymod = y % (2*pi)

        yzero = ymod.is_zero
        # shortcut if ymod is zero
        if yzero:
            return True

        xzero = x.is_zero
        # shortcut x is not zero
        if xzero is False:
            return yzero

        return fuzzy_or([
                # Case 1:
                yzero,
                # Case 2:
                fuzzy_and([
                    xzero,
                    fuzzy_or([ymod < pi/2, ymod > 3*pi/2])
                ])
            ])


    def _eval_is_nonnegative(self):
        z = self.args[0]

        x, y = z.as_real_imag()
        ymod = y % (2*pi)

        yzero = ymod.is_zero
        # shortcut if ymod is zero
        if yzero:
            return True

        xzero = x.is_zero
        # shortcut x is not zero
        if xzero is False:
            return yzero

        return fuzzy_or([
                # Case 1:
                yzero,
                # Case 2:
                fuzzy_and([
                    xzero,
                    fuzzy_or([ymod <= pi/2, ymod >= 3*pi/2])
                ])
            ])

    def _eval_is_finite(self):
        arg = self.args[0]
        return arg.is_finite

    def _eval_is_zero(self):
        rest, ipi_mult = _peeloff_ipi(self.args[0])
        if ipi_mult and rest.is_zero:
            return (ipi_mult - S.Half).is_integer


class tanh(HyperbolicFunction):
    r"""
    ``tanh(x)`` is the hyperbolic tangent of ``x``.

    The hyperbolic tangent function is $\frac{\sinh(x)}{\cosh(x)}$.

    Examples
    ========

    >>> from sympy import tanh
    >>> from sympy.abc import x
    >>> tanh(x)
    tanh(x)

    See Also
    ========

    sinh, cosh, atanh
    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            return S.One - tanh(self.args[0])**2
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return atanh

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
            elif arg.is_negative:
                return -cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                return S.NaN

            i_coeff = _imaginary_unit_as_coefficient(arg)

            if i_coeff is not None:
                if i_coeff.could_extract_minus_sign():
                    return -I * tan(-i_coeff)
                return I * tan(i_coeff)
            else:
                if arg.could_extract_minus_sign():
                    return -cls(-arg)

            if arg.is_Add:
                x, m = _peeloff_ipi(arg)
                if m:
                    tanhm = tanh(m*pi*I)
                    if tanhm is S.ComplexInfinity:
                        return coth(x)
                    else: # tanhm == 0
                        return tanh(x)

            if arg.is_zero:
                return S.Zero

            if arg.func == asinh:
                x = arg.args[0]
                return x/sqrt(1 + x**2)

            if arg.func == acosh:
                x = arg.args[0]
                return sqrt(x - 1) * sqrt(x + 1) / x

            if arg.func == atanh:
                return arg.args[0]

            if arg.func == acoth:
                return 1/arg.args[0]

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)

            a = 2**(n + 1)

            B = bernoulli(n + 1)
            F = factorial(n + 1)

            return a*(a - 1) * B/F * x**n

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def as_real_imag(self, deep=True, **hints):
        if self.args[0].is_extended_real:
            if deep:
                hints['complex'] = False
                return (self.expand(deep, **hints), S.Zero)
            else:
                return (self, S.Zero)
        if deep:
            re, im = self.args[0].expand(deep, **hints).as_real_imag()
        else:
            re, im = self.args[0].as_real_imag()
        denom = sinh(re)**2 + cos(im)**2
        return (sinh(re)*cosh(re)/denom, sin(im)*cos(im)/denom)

    def _eval_expand_trig(self, **hints):
        arg = self.args[0]
        if arg.is_Add:
            n = len(arg.args)
            TX = [tanh(x, evaluate=False)._eval_expand_trig()
                for x in arg.args]
            p = [0, 0]  # [den, num]
            for i in range(n + 1):
                p[i % 2] += symmetric_poly(i, TX)
            return p[1]/p[0]
        elif arg.is_Mul:
            coeff, terms = arg.as_coeff_Mul()
            if coeff.is_Integer and coeff > 1:
                T = tanh(terms)
                n = [nC(range(coeff), k)*T**k for k in range(1, coeff + 1, 2)]
                d = [nC(range(coeff), k)*T**k for k in range(0, coeff + 1, 2)]
                return Add(*n)/Add(*d)
        return tanh(arg)

    def _eval_rewrite_as_tractable(self, arg, limitvar=None, **kwargs):
        neg_exp, pos_exp = exp(-arg), exp(arg)
        return (pos_exp - neg_exp)/(pos_exp + neg_exp)

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        neg_exp, pos_exp = exp(-arg), exp(arg)
        return (pos_exp - neg_exp)/(pos_exp + neg_exp)

    def _eval_rewrite_as_tan(self, arg, **kwargs):
        return -I * tan(I * arg, evaluate=False)

    def _eval_rewrite_as_cot(self, arg, **kwargs):
        return -I / cot(I * arg, evaluate=False)

    def _eval_rewrite_as_sinh(self, arg, **kwargs):
        return I*sinh(arg)/sinh(pi*I/2 - arg, evaluate=False)

    def _eval_rewrite_as_cosh(self, arg, **kwargs):
        return I*cosh(pi*I/2 - arg, evaluate=False)/cosh(arg)

    def _eval_rewrite_as_coth(self, arg, **kwargs):
        return 1/coth(arg)

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.series.order import Order
        arg = self.args[0].as_leading_term(x)

        if x in arg.free_symbols and Order(1, x).contains(arg):
            return arg
        else:
            return self.func(arg)

    def _eval_is_real(self):
        arg = self.args[0]
        if arg.is_real:
            return True

        re, im = arg.as_real_imag()

        # if denom = 0, tanh(arg) = zoo
        if re == 0 and im % pi == pi/2:
            return None

        # check if im is of the form n*pi/2 to make sin(2*im) = 0
        # if not, im could be a number, return False in that case
        return (im % (pi/2)).is_zero

    def _eval_is_extended_real(self):
        if self.args[0].is_extended_real:
            return True

    def _eval_is_positive(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_positive

    def _eval_is_negative(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_negative

    def _eval_is_finite(self):
        arg = self.args[0]

        re, im = arg.as_real_imag()
        denom = cos(im)**2 + sinh(re)**2
        if denom == 0:
            return False
        elif denom.is_number:
            return True
        if arg.is_extended_real:
            return True

    def _eval_is_zero(self):
        arg = self.args[0]
        if arg.is_zero:
            return True


class coth(HyperbolicFunction):
    r"""
    ``coth(x)`` is the hyperbolic cotangent of ``x``.

    The hyperbolic cotangent function is $\frac{\cosh(x)}{\sinh(x)}$.

    Examples
    ========

    >>> from sympy import coth
    >>> from sympy.abc import x
    >>> coth(x)
    coth(x)

    See Also
    ========

    sinh, cosh, acoth
    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            return -1/sinh(self.args[0])**2
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return acoth

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
                return S.ComplexInfinity
            elif arg.is_negative:
                return -cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                return S.NaN

            i_coeff = _imaginary_unit_as_coefficient(arg)

            if i_coeff is not None:
                if i_coeff.could_extract_minus_sign():
                    return I * cot(-i_coeff)
                return -I * cot(i_coeff)
            else:
                if arg.could_extract_minus_sign():
                    return -cls(-arg)

            if arg.is_Add:
                x, m = _peeloff_ipi(arg)
                if m:
                    cothm = coth(m*pi*I)
                    if cothm is S.ComplexInfinity:
                        return coth(x)
                    else: # cothm == 0
                        return tanh(x)

            if arg.is_zero:
                return S.ComplexInfinity

            if arg.func == asinh:
                x = arg.args[0]
                return sqrt(1 + x**2)/x

            if arg.func == acosh:
                x = arg.args[0]
                return x/(sqrt(x - 1) * sqrt(x + 1))

            if arg.func == atanh:
                return 1/arg.args[0]

            if arg.func == acoth:
                return arg.args[0]

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return 1 / sympify(x)
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)

            B = bernoulli(n + 1)
            F = factorial(n + 1)

            return 2**(n + 1) * B/F * x**n

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def as_real_imag(self, deep=True, **hints):
        from sympy.functions.elementary.trigonometric import (cos, sin)
        if self.args[0].is_extended_real:
            if deep:
                hints['complex'] = False
                return (self.expand(deep, **hints), S.Zero)
            else:
                return (self, S.Zero)
        if deep:
            re, im = self.args[0].expand(deep, **hints).as_real_imag()
        else:
            re, im = self.args[0].as_real_imag()
        denom = sinh(re)**2 + sin(im)**2
        return (sinh(re)*cosh(re)/denom, -sin(im)*cos(im)/denom)

    def _eval_rewrite_as_tractable(self, arg, limitvar=None, **kwargs):
        neg_exp, pos_exp = exp(-arg), exp(arg)
        return (pos_exp + neg_exp)/(pos_exp - neg_exp)

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        neg_exp, pos_exp = exp(-arg), exp(arg)
        return (pos_exp + neg_exp)/(pos_exp - neg_exp)

    def _eval_rewrite_as_sinh(self, arg, **kwargs):
        return -I*sinh(pi*I/2 - arg, evaluate=False)/sinh(arg)

    def _eval_rewrite_as_cosh(self, arg, **kwargs):
        return -I*cosh(arg)/cosh(pi*I/2 - arg, evaluate=False)

    def _eval_rewrite_as_tanh(self, arg, **kwargs):
        return 1/tanh(arg)

    def _eval_is_positive(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_positive

    def _eval_is_negative(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_negative

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.series.order import Order
        arg = self.args[0].as_leading_term(x)

        if x in arg.free_symbols and Order(1, x).contains(arg):
            return 1/arg
        else:
            return self.func(arg)

    def _eval_expand_trig(self, **hints):
        arg = self.args[0]
        if arg.is_Add:
            CX = [coth(x, evaluate=False)._eval_expand_trig() for x in arg.args]
            p = [[], []]
            n = len(arg.args)
            for i in range(n, -1, -1):
                p[(n - i) % 2].append(symmetric_poly(i, CX))
            return Add(*p[0])/Add(*p[1])
        elif arg.is_Mul:
            coeff, x = arg.as_coeff_Mul(rational=True)
            if coeff.is_Integer and coeff > 1:
                c = coth(x, evaluate=False)
                p = [[], []]
                for i in range(coeff, -1, -1):
                    p[(coeff - i) % 2].append(binomial(coeff, i)*c**i)
                return Add(*p[0])/Add(*p[1])
        return coth(arg)


class ReciprocalHyperbolicFunction(HyperbolicFunction):
    """Base class for reciprocal functions of hyperbolic functions. """

    #To be defined in class
    _reciprocal_of = None
    _is_even: FuzzyBool = None
    _is_odd: FuzzyBool = None

    @classmethod
    def eval(cls, arg):
        if arg.could_extract_minus_sign():
            if cls._is_even:
                return cls(-arg)
            if cls._is_odd:
                return -cls(-arg)

        t = cls._reciprocal_of.eval(arg)
        if hasattr(arg, 'inverse') and arg.inverse() == cls:
            return arg.args[0]
        return 1/t if t is not None else t

    def _call_reciprocal(self, method_name, *args, **kwargs):
        # Calls method_name on _reciprocal_of
        o = self._reciprocal_of(self.args[0])
        return getattr(o, method_name)(*args, **kwargs)

    def _calculate_reciprocal(self, method_name, *args, **kwargs):
        # If calling method_name on _reciprocal_of returns a value != None
        # then return the reciprocal of that value
        t = self._call_reciprocal(method_name, *args, **kwargs)
        return 1/t if t is not None else t

    def _rewrite_reciprocal(self, method_name, arg):
        # Special handling for rewrite functions. If reciprocal rewrite returns
        # unmodified expression, then return None
        t = self._call_reciprocal(method_name, arg)
        if t is not None and t != self._reciprocal_of(arg):
            return 1/t

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        return self._rewrite_reciprocal("_eval_rewrite_as_exp", arg)

    def _eval_rewrite_as_tractable(self, arg, limitvar=None, **kwargs):
        return self._rewrite_reciprocal("_eval_rewrite_as_tractable", arg)

    def _eval_rewrite_as_tanh(self, arg, **kwargs):
        return self._rewrite_reciprocal("_eval_rewrite_as_tanh", arg)

    def _eval_rewrite_as_coth(self, arg, **kwargs):
        return self._rewrite_reciprocal("_eval_rewrite_as_coth", arg)

    def as_real_imag(self, deep = True, **hints):
        return (1 / self._reciprocal_of(self.args[0])).as_real_imag(deep, **hints)

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def _eval_expand_complex(self, deep=True, **hints):
        re_part, im_part = self.as_real_imag(deep=True, **hints)
        return re_part + I*im_part

    def _eval_expand_trig(self, **hints):
        return self._calculate_reciprocal("_eval_expand_trig", **hints)

    def _eval_as_leading_term(self, x, logx, cdir):
        return (1/self._reciprocal_of(self.args[0]))._eval_as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_is_extended_real(self):
        return self._reciprocal_of(self.args[0]).is_extended_real

    def _eval_is_finite(self):
        return (1/self._reciprocal_of(self.args[0])).is_finite


class csch(ReciprocalHyperbolicFunction):
    r"""
    ``csch(x)`` is the hyperbolic cosecant of ``x``.

    The hyperbolic cosecant function is $\frac{2}{e^x - e^{-x}}$

    Examples
    ========

    >>> from sympy import csch
    >>> from sympy.abc import x
    >>> csch(x)
    csch(x)

    See Also
    ========

    sinh, cosh, tanh, sech, asinh, acosh
    """

    _reciprocal_of = sinh
    _is_odd = True

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function
        """
        if argindex == 1:
            return -coth(self.args[0]) * csch(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        """
        Returns the next term in the Taylor series expansion
        """
        if n == 0:
            return 1/sympify(x)
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)

            B = bernoulli(n + 1)
            F = factorial(n + 1)

            return 2 * (1 - 2**n) * B/F * x**n

    def _eval_rewrite_as_sin(self, arg, **kwargs):
        return I / sin(I * arg, evaluate=False)

    def _eval_rewrite_as_csc(self, arg, **kwargs):
        return I * csc(I * arg, evaluate=False)

    def _eval_rewrite_as_cosh(self, arg, **kwargs):
        return I / cosh(arg + I * pi / 2, evaluate=False)

    def _eval_rewrite_as_sinh(self, arg, **kwargs):
        return 1 / sinh(arg)

    def _eval_is_positive(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_positive

    def _eval_is_negative(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_negative


class sech(ReciprocalHyperbolicFunction):
    r"""
    ``sech(x)`` is the hyperbolic secant of ``x``.

    The hyperbolic secant function is $\frac{2}{e^x + e^{-x}}$

    Examples
    ========

    >>> from sympy import sech
    >>> from sympy.abc import x
    >>> sech(x)
    sech(x)

    See Also
    ========

    sinh, cosh, tanh, coth, csch, asinh, acosh
    """

    _reciprocal_of = cosh
    _is_even = True

    def fdiff(self, argindex=1):
        if argindex == 1:
            return - tanh(self.args[0])*sech(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 1:
            return S.Zero
        else:
            x = sympify(x)
            return euler(n) / factorial(n) * x**(n)

    def _eval_rewrite_as_cos(self, arg, **kwargs):
        return 1 / cos(I * arg, evaluate=False)

    def _eval_rewrite_as_sec(self, arg, **kwargs):
        return sec(I * arg, evaluate=False)

    def _eval_rewrite_as_sinh(self, arg, **kwargs):
        return I / sinh(arg + I * pi /2, evaluate=False)

    def _eval_rewrite_as_cosh(self, arg, **kwargs):
        return 1 / cosh(arg)

    def _eval_is_positive(self):
        if self.args[0].is_extended_real:
            return True


###############################################################################
############################# HYPERBOLIC INVERSES #############################
###############################################################################

class InverseHyperbolicFunction(DefinedFunction):
    """Base class for inverse hyperbolic functions."""

    pass


class asinh(InverseHyperbolicFunction):
    """
    ``asinh(x)`` is the inverse hyperbolic sine of ``x``.

    The inverse hyperbolic sine function.

    Examples
    ========

    >>> from sympy import asinh
    >>> from sympy.abc import x
    >>> asinh(x).diff(x)
    1/sqrt(x**2 + 1)
    >>> asinh(1)
    log(1 + sqrt(2))

    See Also
    ========

    acosh, atanh, sinh
    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            return 1/sqrt(self.args[0]**2 + 1)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.NegativeInfinity
            elif arg.is_zero:
                return S.Zero
            elif arg is S.One:
                return log(sqrt(2) + 1)
            elif arg is S.NegativeOne:
                return log(sqrt(2) - 1)
            elif arg.is_negative:
                return -cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                return S.ComplexInfinity

            if arg.is_zero:
                return S.Zero

            i_coeff = _imaginary_unit_as_coefficient(arg)

            if i_coeff is not None:
                return I * asin(i_coeff)
            else:
                if arg.could_extract_minus_sign():
                    return -cls(-arg)

        if isinstance(arg, sinh) and arg.args[0].is_number:
            z = arg.args[0]
            if z.is_real:
                return z
            r, i = match_real_imag(z)
            if r is not None and i is not None:
                f = floor((i + pi/2)/pi)
                m = z - I*pi*f
                even = f.is_even
                if even is True:
                    return m
                elif even is False:
                    return -m

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) >= 2 and n > 2:
                p = previous_terms[-2]
                return -p * (n - 2)**2/(n*(n - 1)) * x**2
            else:
                k = (n - 1) // 2
                R = RisingFactorial(S.Half, k)
                F = factorial(k)
                return S.NegativeOne**k * R / F * x**n / n

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        if x0.is_zero:
            return arg.as_leading_term(x)

        if x0 is S.NaN:
            expr = self.func(arg.as_leading_term(x))
            if expr.is_finite:
                return expr
            else:
                return self

        # Handling branch points
        if x0 in (-I, I, S.ComplexInfinity):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        # Handling points lying on branch cuts (-I*oo, -I) U (I, I*oo)
        if (1 + x0**2).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if re(ndir).is_positive:
                if im(x0).is_negative:
                    return -self.func(x0) - I*pi
            elif re(ndir).is_negative:
                if im(x0).is_positive:
                    return -self.func(x0) + I*pi
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):  # asinh
        arg = self.args[0]
        arg0 = arg.subs(x, 0)

        # Handling branch points
        if arg0 in (I, -I):
            return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)

        res = super()._eval_nseries(x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res

        # Handling points lying on branch cuts (-I*oo, -I) U (I, I*oo)
        if (1 + arg0**2).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if re(ndir).is_positive:
                if im(arg0).is_negative:
                    return -res - I*pi
            elif re(ndir).is_negative:
                if im(arg0).is_positive:
                    return -res + I*pi
            else:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def _eval_rewrite_as_log(self, x, **kwargs):
        return log(x + sqrt(x**2 + 1))

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_atanh(self, x, **kwargs):
        return atanh(x/sqrt(1 + x**2))

    def _eval_rewrite_as_acosh(self, x, **kwargs):
        ix = I*x
        return I*(sqrt(1 - ix)/sqrt(ix - 1) * acosh(ix) - pi/2)

    def _eval_rewrite_as_asin(self, x, **kwargs):
        return -I * asin(I * x, evaluate=False)

    def _eval_rewrite_as_acos(self, x, **kwargs):
        return I * acos(I * x, evaluate=False) - I*pi/2

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return sinh

    def _eval_is_zero(self):
        return self.args[0].is_zero

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real

    def _eval_is_finite(self):
        return self.args[0].is_finite


class acosh(InverseHyperbolicFunction):
    """
    ``acosh(x)`` is the inverse hyperbolic cosine of ``x``.

    The inverse hyperbolic cosine function.

    Examples
    ========

    >>> from sympy import acosh
    >>> from sympy.abc import x
    >>> acosh(x).diff(x)
    1/(sqrt(x - 1)*sqrt(x + 1))
    >>> acosh(1)
    0

    See Also
    ========

    asinh, atanh, cosh
    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            arg = self.args[0]
            return 1/(sqrt(arg - 1)*sqrt(arg + 1))
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.Infinity
            elif arg.is_zero:
                return pi*I / 2
            elif arg is S.One:
                return S.Zero
            elif arg is S.NegativeOne:
                return pi*I

        if arg.is_number:
            cst_table = _acosh_table()

            if arg in cst_table:
                if arg.is_extended_real:
                    return cst_table[arg]*I
                return cst_table[arg]

        if arg is S.ComplexInfinity:
            return S.ComplexInfinity
        if arg == I*S.Infinity:
            return S.Infinity + I*pi/2
        if arg == -I*S.Infinity:
            return S.Infinity - I*pi/2

        if arg.is_zero:
            return pi*I*S.Half

        if isinstance(arg, cosh) and arg.args[0].is_number:
            z = arg.args[0]
            if z.is_real:
                return Abs(z)
            r, i = match_real_imag(z)
            if r is not None and i is not None:
                f = floor(i/pi)
                m = z - I*pi*f
                even = f.is_even
                if even is True:
                    if r.is_nonnegative:
                        return m
                    elif r.is_negative:
                        return -m
                elif even is False:
                    m -= I*pi
                    if r.is_nonpositive:
                        return -m
                    elif r.is_positive:
                        return m

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return I*pi/2
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) >= 2 and n > 2:
                p = previous_terms[-2]
                return p * (n - 2)**2/(n*(n - 1)) * x**2
            else:
                k = (n - 1) // 2
                R = RisingFactorial(S.Half, k)
                F = factorial(k)
                return -R / F * I * x**n / n

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        # Handling branch points
        if x0 in (-S.One, S.Zero, S.One, S.ComplexInfinity):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)

        if x0 is S.NaN:
            expr = self.func(arg.as_leading_term(x))
            if expr.is_finite:
                return expr
            else:
                return self

        # Handling points lying on branch cuts (-oo, 1)
        if (x0 - 1).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if (x0 + 1).is_negative:
                    return self.func(x0) - 2*I*pi
                return -self.func(x0)
            elif not im(ndir).is_positive:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):  # acosh
        arg = self.args[0]
        arg0 = arg.subs(x, 0)

        # Handling branch points
        if arg0 in (S.One, S.NegativeOne):
            return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)

        res = super()._eval_nseries(x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res

        # Handling points lying on branch cuts (-oo, 1)
        if (arg0 - 1).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if (arg0 + 1).is_negative:
                    return res - 2*I*pi
                return -res
            elif not im(ndir).is_positive:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def _eval_rewrite_as_log(self, x, **kwargs):
        return log(x + sqrt(x + 1) * sqrt(x - 1))

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_acos(self, x, **kwargs):
        return sqrt(x - 1)/sqrt(1 - x) * acos(x)

    def _eval_rewrite_as_asin(self, x, **kwargs):
        return sqrt(x - 1)/sqrt(1 - x) * (pi/2 - asin(x))

    def _eval_rewrite_as_asinh(self, x, **kwargs):
        return sqrt(x - 1)/sqrt(1 - x) * (pi/2 + I*asinh(I*x, evaluate=False))

    def _eval_rewrite_as_atanh(self, x, **kwargs):
        sxm1 = sqrt(x - 1)
        s1mx = sqrt(1 - x)
        sx2m1 = sqrt(x**2 - 1)
        return (pi/2*sxm1/s1mx*(1 - x * sqrt(1/x**2)) +
                sxm1*sqrt(x + 1)/sx2m1 * atanh(sx2m1/x))

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return cosh

    def _eval_is_zero(self):
        if (self.args[0] - 1).is_zero:
            return True

    def _eval_is_extended_real(self):
        return fuzzy_and([self.args[0].is_extended_real, (self.args[0] - 1).is_extended_nonnegative])

    def _eval_is_finite(self):
        return self.args[0].is_finite


class atanh(InverseHyperbolicFunction):
    """
    ``atanh(x)`` is the inverse hyperbolic tangent of ``x``.

    The inverse hyperbolic tangent function.

    Examples
    ========

    >>> from sympy import atanh
    >>> from sympy.abc import x
    >>> atanh(x).diff(x)
    1/(1 - x**2)

    See Also
    ========

    asinh, acosh, tanh
    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            return 1/(1 - self.args[0]**2)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg.is_zero:
                return S.Zero
            elif arg is S.One:
                return S.Infinity
            elif arg is S.NegativeOne:
                return S.NegativeInfinity
            elif arg is S.Infinity:
                return -I * atan(arg)
            elif arg is S.NegativeInfinity:
                return I * atan(-arg)
            elif arg.is_negative:
                return -cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                from sympy.calculus.accumulationbounds import AccumBounds
                return I*AccumBounds(-pi/2, pi/2)

            i_coeff = _imaginary_unit_as_coefficient(arg)

            if i_coeff is not None:
                return I * atan(i_coeff)
            else:
                if arg.could_extract_minus_sign():
                    return -cls(-arg)

        if arg.is_zero:
            return S.Zero

        if isinstance(arg, tanh) and arg.args[0].is_number:
            z = arg.args[0]
            if z.is_real:
                return z
            r, i = match_real_imag(z)
            if r is not None and i is not None:
                f = floor(2*i/pi)
                even = f.is_even
                m = z - I*f*pi/2
                if even is True:
                    return m
                elif even is False:
                    return m - I*pi/2

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            return x**n / n

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        if x0.is_zero:
            return arg.as_leading_term(x)
        if x0 is S.NaN:
            expr = self.func(arg.as_leading_term(x))
            if expr.is_finite:
                return expr
            else:
                return self

        # Handling branch points
        if x0 in (-S.One, S.One, S.ComplexInfinity):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        # Handling points lying on branch cuts (-oo, -1] U [1, oo)
        if (1 - x0**2).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if x0.is_negative:
                    return self.func(x0) - I*pi
            elif im(ndir).is_positive:
                if x0.is_positive:
                    return self.func(x0) + I*pi
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):  # atanh
        arg = self.args[0]
        arg0 = arg.subs(x, 0)

        # Handling branch points
        if arg0 in (S.One, S.NegativeOne):
            return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)

        res = super()._eval_nseries(x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res

        # Handling points lying on branch cuts (-oo, -1] U [1, oo)
        if (1 - arg0**2).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if arg0.is_negative:
                    return res - I*pi
            elif im(ndir).is_positive:
                if arg0.is_positive:
                    return res + I*pi
            else:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def _eval_rewrite_as_log(self, x, **kwargs):
        return (log(1 + x) - log(1 - x)) / 2

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_asinh(self, x, **kwargs):
        f = sqrt(1/(x**2 - 1))
        return (pi*x/(2*sqrt(-x**2)) -
                sqrt(-x)*sqrt(1 - x**2)/sqrt(x)*f*asinh(f))

    def _eval_is_zero(self):
        if self.args[0].is_zero:
            return True

    def _eval_is_extended_real(self):
        return fuzzy_and([self.args[0].is_extended_real, (1 - self.args[0]).is_nonnegative, (self.args[0] + 1).is_nonnegative])

    def _eval_is_finite(self):
        return fuzzy_not(fuzzy_or([(self.args[0] - 1).is_zero, (self.args[0] + 1).is_zero]))

    def _eval_is_imaginary(self):
        return self.args[0].is_imaginary

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return tanh


class acoth(InverseHyperbolicFunction):
    """
    ``acoth(x)`` is the inverse hyperbolic cotangent of ``x``.

    The inverse hyperbolic cotangent function.

    Examples
    ========

    >>> from sympy import acoth
    >>> from sympy.abc import x
    >>> acoth(x).diff(x)
    1/(1 - x**2)

    See Also
    ========

    asinh, acosh, coth
    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            return 1/(1 - self.args[0]**2)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Zero
            elif arg is S.NegativeInfinity:
                return S.Zero
            elif arg.is_zero:
                return pi*I / 2
            elif arg is S.One:
                return S.Infinity
            elif arg is S.NegativeOne:
                return S.NegativeInfinity
            elif arg.is_negative:
                return -cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                return S.Zero

            i_coeff = _imaginary_unit_as_coefficient(arg)

            if i_coeff is not None:
                return -I * acot(i_coeff)
            else:
                if arg.could_extract_minus_sign():
                    return -cls(-arg)

        if arg.is_zero:
            return pi*I*S.Half

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return -I*pi/2
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            return x**n / n

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        if x0 is S.ComplexInfinity:
            return (1/arg).as_leading_term(x)
        if x0 is S.NaN:
            expr = self.func(arg.as_leading_term(x))
            if expr.is_finite:
                return expr
            else:
                return self

        # Handling branch points
        if x0 in (-S.One, S.One, S.Zero):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        # Handling points lying on branch cuts [-1, 1]
        if x0.is_real and (1 - x0**2).is_positive:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if x0.is_positive:
                    return self.func(x0) + I*pi
            elif im(ndir).is_positive:
                if x0.is_negative:
                    return self.func(x0) - I*pi
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):  # acoth
        arg = self.args[0]
        arg0 = arg.subs(x, 0)

        # Handling branch points
        if arg0 in (S.One, S.NegativeOne):
            return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)

        res = super()._eval_nseries(x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res

        # Handling points lying on branch cuts [-1, 1]
        if arg0.is_real and (1 - arg0**2).is_positive:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if arg0.is_positive:
                    return res + I*pi
            elif im(ndir).is_positive:
                if arg0.is_negative:
                    return res - I*pi
            else:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def _eval_rewrite_as_log(self, x, **kwargs):
        return (log(1 + 1/x) - log(1 - 1/x)) / 2

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_atanh(self, x, **kwargs):
        return atanh(1/x)

    def _eval_rewrite_as_asinh(self, x, **kwargs):
        return (pi*I/2*(sqrt((x - 1)/x)*sqrt(x/(x - 1)) - sqrt(1 + 1/x)*sqrt(x/(x + 1))) +
                x*sqrt(1/x**2)*asinh(sqrt(1/(x**2 - 1))))

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return coth

    def _eval_is_extended_real(self):
        return fuzzy_and([self.args[0].is_extended_real, fuzzy_or([(self.args[0] - 1).is_extended_nonnegative, (self.args[0] + 1).is_extended_nonpositive])])

    def _eval_is_finite(self):
        return fuzzy_not(fuzzy_or([(self.args[0] - 1).is_zero, (self.args[0] + 1).is_zero]))


class asech(InverseHyperbolicFunction):
    """
    ``asech(x)`` is the inverse hyperbolic secant of ``x``.

    The inverse hyperbolic secant function.

    Examples
    ========

    >>> from sympy import asech, sqrt, S
    >>> from sympy.abc import x
    >>> asech(x).diff(x)
    -1/(x*sqrt(1 - x**2))
    >>> asech(1).diff(x)
    0
    >>> asech(1)
    0
    >>> asech(S(2))
    I*pi/3
    >>> asech(-sqrt(2))
    3*I*pi/4
    >>> asech((sqrt(6) - sqrt(2)))
    I*pi/12

    See Also
    ========

    asinh, atanh, cosh, acoth

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hyperbolic_function
    .. [2] https://dlmf.nist.gov/4.37
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcSech/

    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            z = self.args[0]
            return -1/(z*sqrt(1 - z**2))
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return pi*I / 2
            elif arg is S.NegativeInfinity:
                return pi*I / 2
            elif arg.is_zero:
                return S.Infinity
            elif arg is S.One:
                return S.Zero
            elif arg is S.NegativeOne:
                return pi*I

        if arg.is_number:
            cst_table = _asech_table()

            if arg in cst_table:
                if arg.is_extended_real:
                    return cst_table[arg]*I
                return cst_table[arg]

        if arg is S.ComplexInfinity:
            from sympy.calculus.accumulationbounds import AccumBounds
            return I*AccumBounds(-pi/2, pi/2)

        if arg.is_zero:
            return S.Infinity

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return log(2 / x)
        elif n < 0 or n % 2 == 1:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) > 2 and n > 2:
                p = previous_terms[-2]
                return p * ((n - 1)*(n-2)) * x**2/(4 * (n//2)**2)
            else:
                k = n // 2
                R = RisingFactorial(S.Half, k) * n
                F = factorial(k) * n // 2 * n // 2
                return -1 * R / F * x**n / 4

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        # Handling branch points
        if x0 in (-S.One, S.Zero, S.One, S.ComplexInfinity):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)

        if x0 is S.NaN:
            expr = self.func(arg.as_leading_term(x))
            if expr.is_finite:
                return expr
            else:
                return self

        # Handling points lying on branch cuts (-oo, 0] U (1, oo)
        if x0.is_negative or (1 - x0).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_positive:
                if x0.is_positive or (x0 + 1).is_negative:
                    return -self.func(x0)
                return self.func(x0) - 2*I*pi
            elif not im(ndir).is_negative:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):  # asech
        from sympy.series.order import O
        arg = self.args[0]
        arg0 = arg.subs(x, 0)

        # Handling branch points
        if arg0 is S.One:
            t = Dummy('t', positive=True)
            ser = asech(S.One - t**2).rewrite(log).nseries(t, 0, 2*n)
            arg1 = S.One - self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f)/ f
            if not g.is_meromorphic(x, 0):   # cannot be expanded
                return O(1) if n == 0 else O(sqrt(x))
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO()*sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        if arg0 is S.NegativeOne:
            t = Dummy('t', positive=True)
            ser = asech(S.NegativeOne + t**2).rewrite(log).nseries(t, 0, 2*n)
            arg1 = S.One + self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f)/ f
            if not g.is_meromorphic(x, 0):   # cannot be expanded
                return O(1) if n == 0 else I*pi + O(sqrt(x))
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO()*sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        res = super()._eval_nseries(x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res

        # Handling points lying on branch cuts (-oo, 0] U (1, oo)
        if arg0.is_negative or (1 - arg0).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_positive:
                if arg0.is_positive or (arg0 + 1).is_negative:
                    return -res
                return res - 2*I*pi
            elif not im(ndir).is_negative:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return sech

    def _eval_rewrite_as_log(self, arg, **kwargs):
        return log(1/arg + sqrt(1/arg - 1) * sqrt(1/arg + 1))

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_acosh(self, arg, **kwargs):
        return acosh(1/arg)

    def _eval_rewrite_as_asinh(self, arg, **kwargs):
        return sqrt(1/arg - 1)/sqrt(1 - 1/arg)*(I*asinh(I/arg, evaluate=False)
                                                + pi*S.Half)

    def _eval_rewrite_as_atanh(self, x, **kwargs):
        return (I*pi*(1 - sqrt(x)*sqrt(1/x) - I/2*sqrt(-x)/sqrt(x) - I/2*sqrt(x**2)/sqrt(-x**2))
                + sqrt(1/(x + 1))*sqrt(x + 1)*atanh(sqrt(1 - x**2)))

    def _eval_rewrite_as_acsch(self, x, **kwargs):
        return sqrt(1/x - 1)/sqrt(1 - 1/x)*(pi/2 - I*acsch(I*x, evaluate=False))

    def _eval_is_extended_real(self):
        return fuzzy_and([self.args[0].is_extended_real, self.args[0].is_nonnegative, (1 - self.args[0]).is_nonnegative])

    def _eval_is_finite(self):
        return fuzzy_not(self.args[0].is_zero)


class acsch(InverseHyperbolicFunction):
    """
    ``acsch(x)`` is the inverse hyperbolic cosecant of ``x``.

    The inverse hyperbolic cosecant function.

    Examples
    ========

    >>> from sympy import acsch, sqrt, I
    >>> from sympy.abc import x
    >>> acsch(x).diff(x)
    -1/(x**2*sqrt(1 + x**(-2)))
    >>> acsch(1).diff(x)
    0
    >>> acsch(1)
    log(1 + sqrt(2))
    >>> acsch(I)
    -I*pi/2
    >>> acsch(-2*I)
    I*pi/6
    >>> acsch(I*(sqrt(6) - sqrt(2)))
    -5*I*pi/12

    See Also
    ========

    asinh

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hyperbolic_function
    .. [2] https://dlmf.nist.gov/4.37
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcCsch/

    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            z = self.args[0]
            return -1/(z**2*sqrt(1 + 1/z**2))
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Zero
            elif arg is S.NegativeInfinity:
                return S.Zero
            elif arg.is_zero:
                return S.ComplexInfinity
            elif arg is S.One:
                return log(1 + sqrt(2))
            elif arg is S.NegativeOne:
                return - log(1 + sqrt(2))

        if arg.is_number:
            cst_table = _acsch_table()

            if arg in cst_table:
                return cst_table[arg]*I

        if arg is S.ComplexInfinity:
            return S.Zero

        if arg.is_infinite:
            return S.Zero

        if arg.is_zero:
            return S.ComplexInfinity

        if arg.could_extract_minus_sign():
            return -cls(-arg)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return log(2 / x)
        elif n < 0 or n % 2 == 1:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) > 2 and n > 2:
                p = previous_terms[-2]
                return -p * ((n - 1)*(n-2)) * x**2/(4 * (n//2)**2)
            else:
                k = n // 2
                R = RisingFactorial(S.Half, k) *  n
                F = factorial(k) * n // 2 * n // 2
                return S.NegativeOne**(k +1) * R / F * x**n / 4

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        # Handling branch points
        if x0 in (-I, I, S.Zero):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)

        if x0 is S.NaN:
            expr = self.func(arg.as_leading_term(x))
            if expr.is_finite:
                return expr
            else:
                return self

        if x0 is S.ComplexInfinity:
            return (1/arg).as_leading_term(x)
        # Handling points lying on branch cuts (-I, I)
        if x0.is_imaginary and (1 + x0**2).is_positive:
            ndir = arg.dir(x, cdir if cdir else 1)
            if re(ndir).is_positive:
                if im(x0).is_positive:
                    return -self.func(x0) - I*pi
            elif re(ndir).is_negative:
                if im(x0).is_negative:
                    return -self.func(x0) + I*pi
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):  # acsch
        from sympy.series.order import O
        arg = self.args[0]
        arg0 = arg.subs(x, 0)

        # Handling branch points
        if arg0 is I:
            t = Dummy('t', positive=True)
            ser = acsch(I + t**2).rewrite(log).nseries(t, 0, 2*n)
            arg1 = -I + self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f)/ f
            if not g.is_meromorphic(x, 0):   # cannot be expanded
                return O(1) if n == 0 else -I*pi/2 + O(sqrt(x))
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO()*sqrt(f)).expand()
            res = ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)
            return res

        if arg0 == S.NegativeOne*I:
            t = Dummy('t', positive=True)
            ser = acsch(-I + t**2).rewrite(log).nseries(t, 0, 2*n)
            arg1 = I + self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f)/ f
            if not g.is_meromorphic(x, 0):   # cannot be expanded
                return O(1) if n == 0 else I*pi/2 + O(sqrt(x))
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO()*sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        res = super()._eval_nseries(x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res

        # Handling points lying on branch cuts (-I, I)
        if arg0.is_imaginary and (1 + arg0**2).is_positive:
            ndir = self.args[0].dir(x, cdir if cdir else 1)
            if re(ndir).is_positive:
                if im(arg0).is_positive:
                    return -res - I*pi
            elif re(ndir).is_negative:
                if im(arg0).is_negative:
                    return -res + I*pi
            else:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return csch

    def _eval_rewrite_as_log(self, arg, **kwargs):
        return log(1/arg + sqrt(1/arg**2 + 1))

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_asinh(self, arg, **kwargs):
        return asinh(1/arg)

    def _eval_rewrite_as_acosh(self, arg, **kwargs):
        return I*(sqrt(1 - I/arg)/sqrt(I/arg - 1)*
                                acosh(I/arg, evaluate=False) - pi*S.Half)

    def _eval_rewrite_as_atanh(self, arg, **kwargs):
        arg2 = arg**2
        arg2p1 = arg2 + 1
        return sqrt(-arg2)/arg*(pi*S.Half -
                                sqrt(-arg2p1**2)/arg2p1*atanh(sqrt(arg2p1)))

    def _eval_is_zero(self):
        return self.args[0].is_infinite

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real

    def _eval_is_finite(self):
        return fuzzy_not(self.args[0].is_zero)
