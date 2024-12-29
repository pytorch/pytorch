from functools import wraps

from sympy.core import S
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.function import Function, ArgumentIndexError, _mexpand
from sympy.core.logic import fuzzy_or, fuzzy_not
from sympy.core.numbers import Rational, pi, I
from sympy.core.power import Pow
from sympy.core.symbol import Dummy, uniquely_named_symbol, Wild
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.trigonometric import sin, cos, csc, cot
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import cbrt, sqrt, root
from sympy.functions.elementary.complexes import (Abs, re, im, polar_lift, unpolarify)
from sympy.functions.special.gamma_functions import gamma, digamma, uppergamma
from sympy.functions.special.hyper import hyper
from sympy.polys.orthopolys import spherical_bessel_fn

from mpmath import mp, workprec

# TODO
# o Scorer functions G1 and G2
# o Asymptotic expansions
#   These are possible, e.g. for fixed order, but since the bessel type
#   functions are oscillatory they are not actually tractable at
#   infinity, so this is not particularly useful right now.
# o Nicer series expansions.
# o More rewriting.
# o Add solvers to ode.py (or rather add solvers for the hypergeometric equation).


class BesselBase(Function):
    """
    Abstract base class for Bessel-type functions.

    This class is meant to reduce code duplication.
    All Bessel-type functions can 1) be differentiated, with the derivatives
    expressed in terms of similar functions, and 2) be rewritten in terms
    of other Bessel-type functions.

    Here, Bessel-type functions are assumed to have one complex parameter.

    To use this base class, define class attributes ``_a`` and ``_b`` such that
    ``2*F_n' = -_a*F_{n+1} + b*F_{n-1}``.

    """

    @property
    def order(self):
        """ The order of the Bessel-type function. """
        return self.args[0]

    @property
    def argument(self):
        """ The argument of the Bessel-type function. """
        return self.args[1]

    @classmethod
    def eval(cls, nu, z):
        return

    def fdiff(self, argindex=2):
        if argindex != 2:
            raise ArgumentIndexError(self, argindex)
        return (self._b/2 * self.__class__(self.order - 1, self.argument) -
                self._a/2 * self.__class__(self.order + 1, self.argument))

    def _eval_conjugate(self):
        z = self.argument
        if z.is_extended_negative is False:
            return self.__class__(self.order.conjugate(), z.conjugate())

    def _eval_is_meromorphic(self, x, a):
        nu, z = self.order, self.argument

        if nu.has(x):
            return False
        if not z._eval_is_meromorphic(x, a):
            return None
        z0 = z.subs(x, a)
        if nu.is_integer:
            if isinstance(self, (besselj, besseli, hn1, hn2, jn, yn)) or not nu.is_zero:
                return fuzzy_not(z0.is_infinite)
        return fuzzy_not(fuzzy_or([z0.is_zero, z0.is_infinite]))

    def _eval_expand_func(self, **hints):
        nu, z, f = self.order, self.argument, self.__class__
        if nu.is_real:
            if (nu - 1).is_positive:
                return (-self._a*self._b*f(nu - 2, z)._eval_expand_func() +
                        2*self._a*(nu - 1)*f(nu - 1, z)._eval_expand_func()/z)
            elif (nu + 1).is_negative:
                return (2*self._b*(nu + 1)*f(nu + 1, z)._eval_expand_func()/z -
                        self._a*self._b*f(nu + 2, z)._eval_expand_func())
        return self

    def _eval_simplify(self, **kwargs):
        from sympy.simplify.simplify import besselsimp
        return besselsimp(self)


class besselj(BesselBase):
    r"""
    Bessel function of the first kind.

    Explanation
    ===========

    The Bessel $J$ function of order $\nu$ is defined to be the function
    satisfying Bessel's differential equation

    .. math ::
        z^2 \frac{\mathrm{d}^2 w}{\mathrm{d}z^2}
        + z \frac{\mathrm{d}w}{\mathrm{d}z} + (z^2 - \nu^2) w = 0,

    with Laurent expansion

    .. math ::
        J_\nu(z) = z^\nu \left(\frac{1}{\Gamma(\nu + 1) 2^\nu} + O(z^2) \right),

    if $\nu$ is not a negative integer. If $\nu=-n \in \mathbb{Z}_{<0}$
    *is* a negative integer, then the definition is

    .. math ::
        J_{-n}(z) = (-1)^n J_n(z).

    Examples
    ========

    Create a Bessel function object:

    >>> from sympy import besselj, jn
    >>> from sympy.abc import z, n
    >>> b = besselj(n, z)

    Differentiate it:

    >>> b.diff(z)
    besselj(n - 1, z)/2 - besselj(n + 1, z)/2

    Rewrite in terms of spherical Bessel functions:

    >>> b.rewrite(jn)
    sqrt(2)*sqrt(z)*jn(n - 1/2, z)/sqrt(pi)

    Access the parameter and argument:

    >>> b.order
    n
    >>> b.argument
    z

    See Also
    ========

    bessely, besseli, besselk

    References
    ==========

    .. [1] Abramowitz, Milton; Stegun, Irene A., eds. (1965), "Chapter 9",
           Handbook of Mathematical Functions with Formulas, Graphs, and
           Mathematical Tables
    .. [2] Luke, Y. L. (1969), The Special Functions and Their
           Approximations, Volume 1
    .. [3] https://en.wikipedia.org/wiki/Bessel_function
    .. [4] https://functions.wolfram.com/Bessel-TypeFunctions/BesselJ/

    """

    _a = S.One
    _b = S.One

    @classmethod
    def eval(cls, nu, z):
        if z.is_zero:
            if nu.is_zero:
                return S.One
            elif (nu.is_integer and nu.is_zero is False) or re(nu).is_positive:
                return S.Zero
            elif re(nu).is_negative and not (nu.is_integer is True):
                return S.ComplexInfinity
            elif nu.is_imaginary:
                return S.NaN
        if z in (S.Infinity, S.NegativeInfinity):
            return S.Zero

        if z.could_extract_minus_sign():
            return (z)**nu*(-z)**(-nu)*besselj(nu, -z)
        if nu.is_integer:
            if nu.could_extract_minus_sign():
                return S.NegativeOne**(-nu)*besselj(-nu, z)
            newz = z.extract_multiplicatively(I)
            if newz:  # NOTE we don't want to change the function if z==0
                return I**(nu)*besseli(nu, newz)

        # branch handling:
        if nu.is_integer:
            newz = unpolarify(z)
            if newz != z:
                return besselj(nu, newz)
        else:
            newz, n = z.extract_branch_factor()
            if n != 0:
                return exp(2*n*pi*nu*I)*besselj(nu, newz)
        nnu = unpolarify(nu)
        if nu != nnu:
            return besselj(nnu, z)

    def _eval_rewrite_as_besseli(self, nu, z, **kwargs):
        return exp(I*pi*nu/2)*besseli(nu, polar_lift(-I)*z)

    def _eval_rewrite_as_bessely(self, nu, z, **kwargs):
        if nu.is_integer is False:
            return csc(pi*nu)*bessely(-nu, z) - cot(pi*nu)*bessely(nu, z)

    def _eval_rewrite_as_jn(self, nu, z, **kwargs):
        return sqrt(2*z/pi)*jn(nu - S.Half, self.argument)

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        nu, z = self.args
        try:
            arg = z.as_leading_term(x)
        except NotImplementedError:
            return self
        c, e = arg.as_coeff_exponent(x)

        if e.is_positive:
            return arg**nu/(2**nu*gamma(nu + 1))
        elif e.is_negative:
            cdir = 1 if cdir == 0 else cdir
            sign = c*cdir**e
            if not sign.is_negative:
                # Refer Abramowitz and Stegun 1965, p. 364 for more information on
                # asymptotic approximation of besselj function.
                return sqrt(2)*cos(z - pi*(2*nu + 1)/4)/sqrt(pi*z)
            return self

        return super(besselj, self)._eval_as_leading_term(x, logx, cdir)

    def _eval_is_extended_real(self):
        nu, z = self.args
        if nu.is_integer and z.is_extended_real:
            return True

    def _eval_nseries(self, x, n, logx, cdir=0):
        # Refer https://functions.wolfram.com/Bessel-TypeFunctions/BesselJ/06/01/04/01/01/0003/
        # for more information on nseries expansion of besselj function.
        from sympy.series.order import Order
        nu, z = self.args

        # In case of powers less than 1, number of terms need to be computed
        # separately to avoid repeated callings of _eval_nseries with wrong n
        try:
            _, exp = z.leadterm(x)
        except (ValueError, NotImplementedError):
            return self

        if exp.is_positive:
            newn = ceiling(n/exp)
            o = Order(x**n, x)
            r = (z/2)._eval_nseries(x, n, logx, cdir).removeO()
            if r is S.Zero:
                return o
            t = (_mexpand(r**2) + o).removeO()

            term = r**nu/gamma(nu + 1)
            s = [term]
            for k in range(1, (newn + 1)//2):
                term *= -t/(k*(nu + k))
                term = (_mexpand(term) + o).removeO()
                s.append(term)
            return Add(*s) + o

        return super(besselj, self)._eval_nseries(x, n, logx, cdir)


class bessely(BesselBase):
    r"""
    Bessel function of the second kind.

    Explanation
    ===========

    The Bessel $Y$ function of order $\nu$ is defined as

    .. math ::
        Y_\nu(z) = \lim_{\mu \to \nu} \frac{J_\mu(z) \cos(\pi \mu)
                                            - J_{-\mu}(z)}{\sin(\pi \mu)},

    where $J_\mu(z)$ is the Bessel function of the first kind.

    It is a solution to Bessel's equation, and linearly independent from
    $J_\nu$.

    Examples
    ========

    >>> from sympy import bessely, yn
    >>> from sympy.abc import z, n
    >>> b = bessely(n, z)
    >>> b.diff(z)
    bessely(n - 1, z)/2 - bessely(n + 1, z)/2
    >>> b.rewrite(yn)
    sqrt(2)*sqrt(z)*yn(n - 1/2, z)/sqrt(pi)

    See Also
    ========

    besselj, besseli, besselk

    References
    ==========

    .. [1] https://functions.wolfram.com/Bessel-TypeFunctions/BesselY/

    """

    _a = S.One
    _b = S.One

    @classmethod
    def eval(cls, nu, z):
        if z.is_zero:
            if nu.is_zero:
                return S.NegativeInfinity
            elif re(nu).is_zero is False:
                return S.ComplexInfinity
            elif re(nu).is_zero:
                return S.NaN
        if z in (S.Infinity, S.NegativeInfinity):
            return S.Zero
        if z == I*S.Infinity:
            return exp(I*pi*(nu + 1)/2) * S.Infinity
        if z == I*S.NegativeInfinity:
            return exp(-I*pi*(nu + 1)/2) * S.Infinity

        if nu.is_integer:
            if nu.could_extract_minus_sign():
                return S.NegativeOne**(-nu)*bessely(-nu, z)

    def _eval_rewrite_as_besselj(self, nu, z, **kwargs):
        if nu.is_integer is False:
            return csc(pi*nu)*(cos(pi*nu)*besselj(nu, z) - besselj(-nu, z))

    def _eval_rewrite_as_besseli(self, nu, z, **kwargs):
        aj = self._eval_rewrite_as_besselj(*self.args)
        if aj:
            return aj.rewrite(besseli)

    def _eval_rewrite_as_yn(self, nu, z, **kwargs):
        return sqrt(2*z/pi) * yn(nu - S.Half, self.argument)

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        nu, z = self.args
        try:
            arg = z.as_leading_term(x)
        except NotImplementedError:
            return self
        c, e = arg.as_coeff_exponent(x)

        if e.is_positive:
            term_one = ((2/pi)*log(z/2)*besselj(nu, z))
            term_two = -(z/2)**(-nu)*factorial(nu - 1)/pi if (nu).is_positive else S.Zero
            term_three = -(z/2)**nu/(pi*factorial(nu))*(digamma(nu + 1) - S.EulerGamma)
            arg = Add(*[term_one, term_two, term_three]).as_leading_term(x, logx=logx)
            return arg
        elif e.is_negative:
            cdir = 1 if cdir == 0 else cdir
            sign = c*cdir**e
            if not sign.is_negative:
                # Refer Abramowitz and Stegun 1965, p. 364 for more information on
                # asymptotic approximation of bessely function.
                return sqrt(2)*(-sin(pi*nu/2 - z + pi/4) + 3*cos(pi*nu/2 - z + pi/4)/(8*z))*sqrt(1/z)/sqrt(pi)
            return self

        return super(bessely, self)._eval_as_leading_term(x, logx, cdir)

    def _eval_is_extended_real(self):
        nu, z = self.args
        if nu.is_integer and z.is_positive:
            return True

    def _eval_nseries(self, x, n, logx, cdir=0):
        # Refer https://functions.wolfram.com/Bessel-TypeFunctions/BesselY/06/01/04/01/02/0008/
        # for more information on nseries expansion of bessely function.
        from sympy.series.order import Order
        nu, z = self.args

        # In case of powers less than 1, number of terms need to be computed
        # separately to avoid repeated callings of _eval_nseries with wrong n
        try:
            _, exp = z.leadterm(x)
        except (ValueError, NotImplementedError):
            return self

        if exp.is_positive and nu.is_integer:
            newn = ceiling(n/exp)
            bn = besselj(nu, z)
            a = ((2/pi)*log(z/2)*bn)._eval_nseries(x, n, logx, cdir)

            b, c = [], []
            o = Order(x**n, x)
            r = (z/2)._eval_nseries(x, n, logx, cdir).removeO()
            if r is S.Zero:
                return o
            t = (_mexpand(r**2) + o).removeO()

            if nu > S.Zero:
                term = r**(-nu)*factorial(nu - 1)/pi
                b.append(term)
                for k in range(1, nu):
                    denom = (nu - k)*k
                    if denom == S.Zero:
                        term *= t/k
                    else:
                        term *= t/denom
                    term = (_mexpand(term) + o).removeO()
                    b.append(term)

            p = r**nu/(pi*factorial(nu))
            term = p*(digamma(nu + 1) - S.EulerGamma)
            c.append(term)
            for k in range(1, (newn + 1)//2):
                p *= -t/(k*(k + nu))
                p = (_mexpand(p) + o).removeO()
                term = p*(digamma(k + nu + 1) + digamma(k + 1))
                c.append(term)
            return a - Add(*b) - Add(*c) # Order term comes from a

        return super(bessely, self)._eval_nseries(x, n, logx, cdir)


class besseli(BesselBase):
    r"""
    Modified Bessel function of the first kind.

    Explanation
    ===========

    The Bessel $I$ function is a solution to the modified Bessel equation

    .. math ::
        z^2 \frac{\mathrm{d}^2 w}{\mathrm{d}z^2}
        + z \frac{\mathrm{d}w}{\mathrm{d}z} + (z^2 + \nu^2)^2 w = 0.

    It can be defined as

    .. math ::
        I_\nu(z) = i^{-\nu} J_\nu(iz),

    where $J_\nu(z)$ is the Bessel function of the first kind.

    Examples
    ========

    >>> from sympy import besseli
    >>> from sympy.abc import z, n
    >>> besseli(n, z).diff(z)
    besseli(n - 1, z)/2 + besseli(n + 1, z)/2

    See Also
    ========

    besselj, bessely, besselk

    References
    ==========

    .. [1] https://functions.wolfram.com/Bessel-TypeFunctions/BesselI/

    """

    _a = -S.One
    _b = S.One

    @classmethod
    def eval(cls, nu, z):
        if z.is_zero:
            if nu.is_zero:
                return S.One
            elif (nu.is_integer and nu.is_zero is False) or re(nu).is_positive:
                return S.Zero
            elif re(nu).is_negative and not (nu.is_integer is True):
                return S.ComplexInfinity
            elif nu.is_imaginary:
                return S.NaN
        if im(z) in (S.Infinity, S.NegativeInfinity):
            return S.Zero
        if z is S.Infinity:
            return S.Infinity
        if z is S.NegativeInfinity:
            return (-1)**nu*S.Infinity

        if z.could_extract_minus_sign():
            return (z)**nu*(-z)**(-nu)*besseli(nu, -z)
        if nu.is_integer:
            if nu.could_extract_minus_sign():
                return besseli(-nu, z)
            newz = z.extract_multiplicatively(I)
            if newz:  # NOTE we don't want to change the function if z==0
                return I**(-nu)*besselj(nu, -newz)

        # branch handling:
        if nu.is_integer:
            newz = unpolarify(z)
            if newz != z:
                return besseli(nu, newz)
        else:
            newz, n = z.extract_branch_factor()
            if n != 0:
                return exp(2*n*pi*nu*I)*besseli(nu, newz)
        nnu = unpolarify(nu)
        if nu != nnu:
            return besseli(nnu, z)

    def _eval_rewrite_as_besselj(self, nu, z, **kwargs):
        return exp(-I*pi*nu/2)*besselj(nu, polar_lift(I)*z)

    def _eval_rewrite_as_bessely(self, nu, z, **kwargs):
        aj = self._eval_rewrite_as_besselj(*self.args)
        if aj:
            return aj.rewrite(bessely)

    def _eval_rewrite_as_jn(self, nu, z, **kwargs):
        return self._eval_rewrite_as_besselj(*self.args).rewrite(jn)

    def _eval_is_extended_real(self):
        nu, z = self.args
        if nu.is_integer and z.is_extended_real:
            return True

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        nu, z = self.args
        try:
            arg = z.as_leading_term(x)
        except NotImplementedError:
            return self
        c, e = arg.as_coeff_exponent(x)

        if e.is_positive:
            return arg**nu/(2**nu*gamma(nu + 1))
        elif e.is_negative:
            cdir = 1 if cdir == 0 else cdir
            sign = c*cdir**e
            if not sign.is_negative:
                # Refer Abramowitz and Stegun 1965, p. 377 for more information on
                # asymptotic approximation of besseli function.
                return exp(z)/sqrt(2*pi*z)
            return self

        return super(besseli, self)._eval_as_leading_term(x, logx, cdir)

    def _eval_nseries(self, x, n, logx, cdir=0):
        # Refer https://functions.wolfram.com/Bessel-TypeFunctions/BesselI/06/01/04/01/01/0003/
        # for more information on nseries expansion of besseli function.
        from sympy.series.order import Order
        nu, z = self.args

        # In case of powers less than 1, number of terms need to be computed
        # separately to avoid repeated callings of _eval_nseries with wrong n
        try:
            _, exp = z.leadterm(x)
        except (ValueError, NotImplementedError):
            return self

        if exp.is_positive:
            newn = ceiling(n/exp)
            o = Order(x**n, x)
            r = (z/2)._eval_nseries(x, n, logx, cdir).removeO()
            if r is S.Zero:
                return o
            t = (_mexpand(r**2) + o).removeO()

            term = r**nu/gamma(nu + 1)
            s = [term]
            for k in range(1, (newn + 1)//2):
                term *= t/(k*(nu + k))
                term = (_mexpand(term) + o).removeO()
                s.append(term)
            return Add(*s) + o

        return super(besseli, self)._eval_nseries(x, n, logx, cdir)


class besselk(BesselBase):
    r"""
    Modified Bessel function of the second kind.

    Explanation
    ===========

    The Bessel $K$ function of order $\nu$ is defined as

    .. math ::
        K_\nu(z) = \lim_{\mu \to \nu} \frac{\pi}{2}
                   \frac{I_{-\mu}(z) -I_\mu(z)}{\sin(\pi \mu)},

    where $I_\mu(z)$ is the modified Bessel function of the first kind.

    It is a solution of the modified Bessel equation, and linearly independent
    from $Y_\nu$.

    Examples
    ========

    >>> from sympy import besselk
    >>> from sympy.abc import z, n
    >>> besselk(n, z).diff(z)
    -besselk(n - 1, z)/2 - besselk(n + 1, z)/2

    See Also
    ========

    besselj, besseli, bessely

    References
    ==========

    .. [1] https://functions.wolfram.com/Bessel-TypeFunctions/BesselK/

    """

    _a = S.One
    _b = -S.One

    @classmethod
    def eval(cls, nu, z):
        if z.is_zero:
            if nu.is_zero:
                return S.Infinity
            elif re(nu).is_zero is False:
                return S.ComplexInfinity
            elif re(nu).is_zero:
                return S.NaN
        if z in (S.Infinity, I*S.Infinity, I*S.NegativeInfinity):
            return S.Zero

        if nu.is_integer:
            if nu.could_extract_minus_sign():
                return besselk(-nu, z)

    def _eval_rewrite_as_besseli(self, nu, z, **kwargs):
        if nu.is_integer is False:
            return pi*csc(pi*nu)*(besseli(-nu, z) - besseli(nu, z))/2

    def _eval_rewrite_as_besselj(self, nu, z, **kwargs):
        ai = self._eval_rewrite_as_besseli(*self.args)
        if ai:
            return ai.rewrite(besselj)

    def _eval_rewrite_as_bessely(self, nu, z, **kwargs):
        aj = self._eval_rewrite_as_besselj(*self.args)
        if aj:
            return aj.rewrite(bessely)

    def _eval_rewrite_as_yn(self, nu, z, **kwargs):
        ay = self._eval_rewrite_as_bessely(*self.args)
        if ay:
            return ay.rewrite(yn)

    def _eval_is_extended_real(self):
        nu, z = self.args
        if nu.is_integer and z.is_positive:
            return True

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        nu, z = self.args
        try:
            arg = z.as_leading_term(x)
        except NotImplementedError:
            return self
        _, e = arg.as_coeff_exponent(x)

        if e.is_positive:
            term_one = ((-1)**(nu -1)*log(z/2)*besseli(nu, z))
            term_two = (z/2)**(-nu)*factorial(nu - 1)/2 if (nu).is_positive else S.Zero
            term_three = (-1)**nu*(z/2)**nu/(2*factorial(nu))*(digamma(nu + 1) - S.EulerGamma)
            arg = Add(*[term_one, term_two, term_three]).as_leading_term(x, logx=logx)
            return arg
        elif e.is_negative:
            # Refer Abramowitz and Stegun 1965, p. 378 for more information on
            # asymptotic approximation of besselk function.
            return sqrt(pi)*exp(-z)/sqrt(2*z)

        return super(besselk, self)._eval_as_leading_term(x, logx, cdir)

    def _eval_nseries(self, x, n, logx, cdir=0):
        # Refer https://functions.wolfram.com/Bessel-TypeFunctions/BesselK/06/01/04/01/02/0008/
        # for more information on nseries expansion of besselk function.
        from sympy.series.order import Order
        nu, z = self.args

        # In case of powers less than 1, number of terms need to be computed
        # separately to avoid repeated callings of _eval_nseries with wrong n
        try:
            _, exp = z.leadterm(x)
        except (ValueError, NotImplementedError):
            return self

        if exp.is_positive and nu.is_integer:
            newn = ceiling(n/exp)
            bn = besseli(nu, z)
            a = ((-1)**(nu - 1)*log(z/2)*bn)._eval_nseries(x, n, logx, cdir)

            b, c = [], []
            o = Order(x**n, x)
            r = (z/2)._eval_nseries(x, n, logx, cdir).removeO()
            if r is S.Zero:
                return o
            t = (_mexpand(r**2) + o).removeO()

            if nu > S.Zero:
                term = r**(-nu)*factorial(nu - 1)/2
                b.append(term)
                for k in range(1, nu):
                    denom = (k - nu)*k
                    if denom == S.Zero:
                        term *= t/k
                    else:
                        term *= t/denom
                    term = (_mexpand(term) + o).removeO()
                    b.append(term)

            p = r**nu*(-1)**nu/(2*factorial(nu))
            term = p*(digamma(nu + 1) - S.EulerGamma)
            c.append(term)
            for k in range(1, (newn + 1)//2):
                p *= t/(k*(k + nu))
                p = (_mexpand(p) + o).removeO()
                term = p*(digamma(k + nu + 1) + digamma(k + 1))
                c.append(term)
            return a + Add(*b) + Add(*c) # Order term comes from a

        return super(besselk, self)._eval_nseries(x, n, logx, cdir)


class hankel1(BesselBase):
    r"""
    Hankel function of the first kind.

    Explanation
    ===========

    This function is defined as

    .. math ::
        H_\nu^{(1)} = J_\nu(z) + iY_\nu(z),

    where $J_\nu(z)$ is the Bessel function of the first kind, and
    $Y_\nu(z)$ is the Bessel function of the second kind.

    It is a solution to Bessel's equation.

    Examples
    ========

    >>> from sympy import hankel1
    >>> from sympy.abc import z, n
    >>> hankel1(n, z).diff(z)
    hankel1(n - 1, z)/2 - hankel1(n + 1, z)/2

    See Also
    ========

    hankel2, besselj, bessely

    References
    ==========

    .. [1] https://functions.wolfram.com/Bessel-TypeFunctions/HankelH1/

    """

    _a = S.One
    _b = S.One

    def _eval_conjugate(self):
        z = self.argument
        if z.is_extended_negative is False:
            return hankel2(self.order.conjugate(), z.conjugate())


class hankel2(BesselBase):
    r"""
    Hankel function of the second kind.

    Explanation
    ===========

    This function is defined as

    .. math ::
        H_\nu^{(2)} = J_\nu(z) - iY_\nu(z),

    where $J_\nu(z)$ is the Bessel function of the first kind, and
    $Y_\nu(z)$ is the Bessel function of the second kind.

    It is a solution to Bessel's equation, and linearly independent from
    $H_\nu^{(1)}$.

    Examples
    ========

    >>> from sympy import hankel2
    >>> from sympy.abc import z, n
    >>> hankel2(n, z).diff(z)
    hankel2(n - 1, z)/2 - hankel2(n + 1, z)/2

    See Also
    ========

    hankel1, besselj, bessely

    References
    ==========

    .. [1] https://functions.wolfram.com/Bessel-TypeFunctions/HankelH2/

    """

    _a = S.One
    _b = S.One

    def _eval_conjugate(self):
        z = self.argument
        if z.is_extended_negative is False:
            return hankel1(self.order.conjugate(), z.conjugate())


def assume_integer_order(fn):
    @wraps(fn)
    def g(self, nu, z):
        if nu.is_integer:
            return fn(self, nu, z)
    return g


class SphericalBesselBase(BesselBase):
    """
    Base class for spherical Bessel functions.

    These are thin wrappers around ordinary Bessel functions,
    since spherical Bessel functions differ from the ordinary
    ones just by a slight change in order.

    To use this class, define the ``_eval_evalf()`` and ``_expand()`` methods.

    """

    def _expand(self, **hints):
        """ Expand self into a polynomial. Nu is guaranteed to be Integer. """
        raise NotImplementedError('expansion')

    def _eval_expand_func(self, **hints):
        if self.order.is_Integer:
            return self._expand(**hints)
        return self

    def fdiff(self, argindex=2):
        if argindex != 2:
            raise ArgumentIndexError(self, argindex)
        return self.__class__(self.order - 1, self.argument) - \
            self * (self.order + 1)/self.argument


def _jn(n, z):
    return (spherical_bessel_fn(n, z)*sin(z) +
            S.NegativeOne**(n + 1)*spherical_bessel_fn(-n - 1, z)*cos(z))


def _yn(n, z):
    # (-1)**(n + 1) * _jn(-n - 1, z)
    return (S.NegativeOne**(n + 1) * spherical_bessel_fn(-n - 1, z)*sin(z) -
            spherical_bessel_fn(n, z)*cos(z))


class jn(SphericalBesselBase):
    r"""
    Spherical Bessel function of the first kind.

    Explanation
    ===========

    This function is a solution to the spherical Bessel equation

    .. math ::
        z^2 \frac{\mathrm{d}^2 w}{\mathrm{d}z^2}
          + 2z \frac{\mathrm{d}w}{\mathrm{d}z} + (z^2 - \nu(\nu + 1)) w = 0.

    It can be defined as

    .. math ::
        j_\nu(z) = \sqrt{\frac{\pi}{2z}} J_{\nu + \frac{1}{2}}(z),

    where $J_\nu(z)$ is the Bessel function of the first kind.

    The spherical Bessel functions of integral order are
    calculated using the formula:

    .. math:: j_n(z) = f_n(z) \sin{z} + (-1)^{n+1} f_{-n-1}(z) \cos{z},

    where the coefficients $f_n(z)$ are available as
    :func:`sympy.polys.orthopolys.spherical_bessel_fn`.

    Examples
    ========

    >>> from sympy import Symbol, jn, sin, cos, expand_func, besselj, bessely
    >>> z = Symbol("z")
    >>> nu = Symbol("nu", integer=True)
    >>> print(expand_func(jn(0, z)))
    sin(z)/z
    >>> expand_func(jn(1, z)) == sin(z)/z**2 - cos(z)/z
    True
    >>> expand_func(jn(3, z))
    (-6/z**2 + 15/z**4)*sin(z) + (1/z - 15/z**3)*cos(z)
    >>> jn(nu, z).rewrite(besselj)
    sqrt(2)*sqrt(pi)*sqrt(1/z)*besselj(nu + 1/2, z)/2
    >>> jn(nu, z).rewrite(bessely)
    (-1)**nu*sqrt(2)*sqrt(pi)*sqrt(1/z)*bessely(-nu - 1/2, z)/2
    >>> jn(2, 5.2+0.3j).evalf(20)
    0.099419756723640344491 - 0.054525080242173562897*I

    See Also
    ========

    besselj, bessely, besselk, yn

    References
    ==========

    .. [1] https://dlmf.nist.gov/10.47

    """
    @classmethod
    def eval(cls, nu, z):
        if z.is_zero:
            if nu.is_zero:
                return S.One
            elif nu.is_integer:
                if nu.is_positive:
                    return S.Zero
                else:
                    return S.ComplexInfinity
        if z in (S.NegativeInfinity, S.Infinity):
            return S.Zero

    def _eval_rewrite_as_besselj(self, nu, z, **kwargs):
        return sqrt(pi/(2*z)) * besselj(nu + S.Half, z)

    def _eval_rewrite_as_bessely(self, nu, z, **kwargs):
        return S.NegativeOne**nu * sqrt(pi/(2*z)) * bessely(-nu - S.Half, z)

    def _eval_rewrite_as_yn(self, nu, z, **kwargs):
        return S.NegativeOne**(nu) * yn(-nu - 1, z)

    def _expand(self, **hints):
        return _jn(self.order, self.argument)

    def _eval_evalf(self, prec):
        if self.order.is_Integer:
            return self.rewrite(besselj)._eval_evalf(prec)


class yn(SphericalBesselBase):
    r"""
    Spherical Bessel function of the second kind.

    Explanation
    ===========

    This function is another solution to the spherical Bessel equation, and
    linearly independent from $j_n$. It can be defined as

    .. math ::
        y_\nu(z) = \sqrt{\frac{\pi}{2z}} Y_{\nu + \frac{1}{2}}(z),

    where $Y_\nu(z)$ is the Bessel function of the second kind.

    For integral orders $n$, $y_n$ is calculated using the formula:

    .. math:: y_n(z) = (-1)^{n+1} j_{-n-1}(z)

    Examples
    ========

    >>> from sympy import Symbol, yn, sin, cos, expand_func, besselj, bessely
    >>> z = Symbol("z")
    >>> nu = Symbol("nu", integer=True)
    >>> print(expand_func(yn(0, z)))
    -cos(z)/z
    >>> expand_func(yn(1, z)) == -cos(z)/z**2-sin(z)/z
    True
    >>> yn(nu, z).rewrite(besselj)
    (-1)**(nu + 1)*sqrt(2)*sqrt(pi)*sqrt(1/z)*besselj(-nu - 1/2, z)/2
    >>> yn(nu, z).rewrite(bessely)
    sqrt(2)*sqrt(pi)*sqrt(1/z)*bessely(nu + 1/2, z)/2
    >>> yn(2, 5.2+0.3j).evalf(20)
    0.18525034196069722536 + 0.014895573969924817587*I

    See Also
    ========

    besselj, bessely, besselk, jn

    References
    ==========

    .. [1] https://dlmf.nist.gov/10.47

    """
    @assume_integer_order
    def _eval_rewrite_as_besselj(self, nu, z, **kwargs):
        return S.NegativeOne**(nu+1) * sqrt(pi/(2*z)) * besselj(-nu - S.Half, z)

    @assume_integer_order
    def _eval_rewrite_as_bessely(self, nu, z, **kwargs):
        return sqrt(pi/(2*z)) * bessely(nu + S.Half, z)

    def _eval_rewrite_as_jn(self, nu, z, **kwargs):
        return S.NegativeOne**(nu + 1) * jn(-nu - 1, z)

    def _expand(self, **hints):
        return _yn(self.order, self.argument)

    def _eval_evalf(self, prec):
        if self.order.is_Integer:
            return self.rewrite(bessely)._eval_evalf(prec)


class SphericalHankelBase(SphericalBesselBase):

    @assume_integer_order
    def _eval_rewrite_as_besselj(self, nu, z, **kwargs):
        # jn +- I*yn
        # jn as beeselj: sqrt(pi/(2*z)) * besselj(nu + S.Half, z)
        # yn as besselj: (-1)**(nu+1) * sqrt(pi/(2*z)) * besselj(-nu - S.Half, z)
        hks = self._hankel_kind_sign
        return sqrt(pi/(2*z))*(besselj(nu + S.Half, z) +
                               hks*I*S.NegativeOne**(nu+1)*besselj(-nu - S.Half, z))

    @assume_integer_order
    def _eval_rewrite_as_bessely(self, nu, z, **kwargs):
        # jn +- I*yn
        # jn as bessely: (-1)**nu * sqrt(pi/(2*z)) * bessely(-nu - S.Half, z)
        # yn as bessely: sqrt(pi/(2*z)) * bessely(nu + S.Half, z)
        hks = self._hankel_kind_sign
        return sqrt(pi/(2*z))*(S.NegativeOne**nu*bessely(-nu - S.Half, z) +
                               hks*I*bessely(nu + S.Half, z))

    def _eval_rewrite_as_yn(self, nu, z, **kwargs):
        hks = self._hankel_kind_sign
        return jn(nu, z).rewrite(yn) + hks*I*yn(nu, z)

    def _eval_rewrite_as_jn(self, nu, z, **kwargs):
        hks = self._hankel_kind_sign
        return jn(nu, z) + hks*I*yn(nu, z).rewrite(jn)

    def _eval_expand_func(self, **hints):
        if self.order.is_Integer:
            return self._expand(**hints)
        else:
            nu = self.order
            z = self.argument
            hks = self._hankel_kind_sign
            return jn(nu, z) + hks*I*yn(nu, z)

    def _expand(self, **hints):
        n = self.order
        z = self.argument
        hks = self._hankel_kind_sign

        # fully expanded version
        # return ((fn(n, z) * sin(z) +
        #          (-1)**(n + 1) * fn(-n - 1, z) * cos(z)) +  # jn
        #         (hks * I * (-1)**(n + 1) *
        #          (fn(-n - 1, z) * hk * I * sin(z) +
        #           (-1)**(-n) * fn(n, z) * I * cos(z)))  # +-I*yn
        #         )

        return (_jn(n, z) + hks*I*_yn(n, z)).expand()

    def _eval_evalf(self, prec):
        if self.order.is_Integer:
            return self.rewrite(besselj)._eval_evalf(prec)


class hn1(SphericalHankelBase):
    r"""
    Spherical Hankel function of the first kind.

    Explanation
    ===========

    This function is defined as

    .. math:: h_\nu^(1)(z) = j_\nu(z) + i y_\nu(z),

    where $j_\nu(z)$ and $y_\nu(z)$ are the spherical
    Bessel function of the first and second kinds.

    For integral orders $n$, $h_n^(1)$ is calculated using the formula:

    .. math:: h_n^(1)(z) = j_{n}(z) + i (-1)^{n+1} j_{-n-1}(z)

    Examples
    ========

    >>> from sympy import Symbol, hn1, hankel1, expand_func, yn, jn
    >>> z = Symbol("z")
    >>> nu = Symbol("nu", integer=True)
    >>> print(expand_func(hn1(nu, z)))
    jn(nu, z) + I*yn(nu, z)
    >>> print(expand_func(hn1(0, z)))
    sin(z)/z - I*cos(z)/z
    >>> print(expand_func(hn1(1, z)))
    -I*sin(z)/z - cos(z)/z + sin(z)/z**2 - I*cos(z)/z**2
    >>> hn1(nu, z).rewrite(jn)
    (-1)**(nu + 1)*I*jn(-nu - 1, z) + jn(nu, z)
    >>> hn1(nu, z).rewrite(yn)
    (-1)**nu*yn(-nu - 1, z) + I*yn(nu, z)
    >>> hn1(nu, z).rewrite(hankel1)
    sqrt(2)*sqrt(pi)*sqrt(1/z)*hankel1(nu, z)/2

    See Also
    ========

    hn2, jn, yn, hankel1, hankel2

    References
    ==========

    .. [1] https://dlmf.nist.gov/10.47

    """

    _hankel_kind_sign = S.One

    @assume_integer_order
    def _eval_rewrite_as_hankel1(self, nu, z, **kwargs):
        return sqrt(pi/(2*z))*hankel1(nu, z)


class hn2(SphericalHankelBase):
    r"""
    Spherical Hankel function of the second kind.

    Explanation
    ===========

    This function is defined as

    .. math:: h_\nu^(2)(z) = j_\nu(z) - i y_\nu(z),

    where $j_\nu(z)$ and $y_\nu(z)$ are the spherical
    Bessel function of the first and second kinds.

    For integral orders $n$, $h_n^(2)$ is calculated using the formula:

    .. math:: h_n^(2)(z) = j_{n} - i (-1)^{n+1} j_{-n-1}(z)

    Examples
    ========

    >>> from sympy import Symbol, hn2, hankel2, expand_func, jn, yn
    >>> z = Symbol("z")
    >>> nu = Symbol("nu", integer=True)
    >>> print(expand_func(hn2(nu, z)))
    jn(nu, z) - I*yn(nu, z)
    >>> print(expand_func(hn2(0, z)))
    sin(z)/z + I*cos(z)/z
    >>> print(expand_func(hn2(1, z)))
    I*sin(z)/z - cos(z)/z + sin(z)/z**2 + I*cos(z)/z**2
    >>> hn2(nu, z).rewrite(hankel2)
    sqrt(2)*sqrt(pi)*sqrt(1/z)*hankel2(nu, z)/2
    >>> hn2(nu, z).rewrite(jn)
    -(-1)**(nu + 1)*I*jn(-nu - 1, z) + jn(nu, z)
    >>> hn2(nu, z).rewrite(yn)
    (-1)**nu*yn(-nu - 1, z) - I*yn(nu, z)

    See Also
    ========

    hn1, jn, yn, hankel1, hankel2

    References
    ==========

    .. [1] https://dlmf.nist.gov/10.47

    """

    _hankel_kind_sign = -S.One

    @assume_integer_order
    def _eval_rewrite_as_hankel2(self, nu, z, **kwargs):
        return sqrt(pi/(2*z))*hankel2(nu, z)


def jn_zeros(n, k, method="sympy", dps=15):
    """
    Zeros of the spherical Bessel function of the first kind.

    Explanation
    ===========

    This returns an array of zeros of $jn$ up to the $k$-th zero.

    * method = "sympy": uses `mpmath.besseljzero
      <https://mpmath.org/doc/current/functions/bessel.html#mpmath.besseljzero>`_
    * method = "scipy": uses the
      `SciPy's sph_jn <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jn_zeros.html>`_
      and
      `newton <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html>`_
      to find all
      roots, which is faster than computing the zeros using a general
      numerical solver, but it requires SciPy and only works with low
      precision floating point numbers. (The function used with
      method="sympy" is a recent addition to mpmath; before that a general
      solver was used.)

    Examples
    ========

    >>> from sympy import jn_zeros
    >>> jn_zeros(2, 4, dps=5)
    [5.7635, 9.095, 12.323, 15.515]

    See Also
    ========

    jn, yn, besselj, besselk, bessely

    Parameters
    ==========

    n : integer
        order of Bessel function

    k : integer
        number of zeros to return


    """
    from math import pi as math_pi

    if method == "sympy":
        from mpmath import besseljzero
        from mpmath.libmp.libmpf import dps_to_prec
        prec = dps_to_prec(dps)
        return [Expr._from_mpmath(besseljzero(S(n + 0.5)._to_mpmath(prec),
                                              int(l)), prec)
                for l in range(1, k + 1)]
    elif method == "scipy":
        from scipy.optimize import newton
        try:
            from scipy.special import spherical_jn
            f = lambda x: spherical_jn(n, x)
        except ImportError:
            from scipy.special import sph_jn
            f = lambda x: sph_jn(n, x)[0][-1]
    else:
        raise NotImplementedError("Unknown method.")

    def solver(f, x):
        if method == "scipy":
            root = newton(f, x)
        else:
            raise NotImplementedError("Unknown method.")
        return root

    # we need to approximate the position of the first root:
    root = n + math_pi
    # determine the first root exactly:
    root = solver(f, root)
    roots = [root]
    for i in range(k - 1):
        # estimate the position of the next root using the last root + pi:
        root = solver(f, root + math_pi)
        roots.append(root)
    return roots


class AiryBase(Function):
    """
    Abstract base class for Airy functions.

    This class is meant to reduce code duplication.

    """

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real

    def as_real_imag(self, deep=True, **hints):
        z = self.args[0]
        zc = z.conjugate()
        f = self.func
        u = (f(z)+f(zc))/2
        v = I*(f(zc)-f(z))/2
        return u, v

    def _eval_expand_complex(self, deep=True, **hints):
        re_part, im_part = self.as_real_imag(deep=deep, **hints)
        return re_part + im_part*I


class airyai(AiryBase):
    r"""
    The Airy function $\operatorname{Ai}$ of the first kind.

    Explanation
    ===========

    The Airy function $\operatorname{Ai}(z)$ is defined to be the function
    satisfying Airy's differential equation

    .. math::
        \frac{\mathrm{d}^2 w(z)}{\mathrm{d}z^2} - z w(z) = 0.

    Equivalently, for real $z$

    .. math::
        \operatorname{Ai}(z) := \frac{1}{\pi}
        \int_0^\infty \cos\left(\frac{t^3}{3} + z t\right) \mathrm{d}t.

    Examples
    ========

    Create an Airy function object:

    >>> from sympy import airyai
    >>> from sympy.abc import z

    >>> airyai(z)
    airyai(z)

    Several special values are known:

    >>> airyai(0)
    3**(1/3)/(3*gamma(2/3))
    >>> from sympy import oo
    >>> airyai(oo)
    0
    >>> airyai(-oo)
    0

    The Airy function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(airyai(z))
    airyai(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(airyai(z), z)
    airyaiprime(z)
    >>> diff(airyai(z), z, 2)
    z*airyai(z)

    Series expansion is also supported:

    >>> from sympy import series
    >>> series(airyai(z), z, 0, 3)
    3**(5/6)*gamma(1/3)/(6*pi) - 3**(1/6)*z*gamma(2/3)/(2*pi) + O(z**3)

    We can numerically evaluate the Airy function to arbitrary precision
    on the whole complex plane:

    >>> airyai(-2).evalf(50)
    0.22740742820168557599192443603787379946077222541710

    Rewrite $\operatorname{Ai}(z)$ in terms of hypergeometric functions:

    >>> from sympy import hyper
    >>> airyai(z).rewrite(hyper)
    -3**(2/3)*z*hyper((), (4/3,), z**3/9)/(3*gamma(1/3)) + 3**(1/3)*hyper((), (2/3,), z**3/9)/(3*gamma(2/3))

    See Also
    ========

    airybi: Airy function of the second kind.
    airyaiprime: Derivative of the Airy function of the first kind.
    airybiprime: Derivative of the Airy function of the second kind.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Airy_function
    .. [2] https://dlmf.nist.gov/9
    .. [3] https://encyclopediaofmath.org/wiki/Airy_functions
    .. [4] https://mathworld.wolfram.com/AiryFunctions.html

    """

    nargs = 1
    unbranched = True

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
                return S.One / (3**Rational(2, 3) * gamma(Rational(2, 3)))
        if arg.is_zero:
            return S.One / (3**Rational(2, 3) * gamma(Rational(2, 3)))

    def fdiff(self, argindex=1):
        if argindex == 1:
            return airyaiprime(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) > 1:
                p = previous_terms[-1]
                return ((cbrt(3)*x)**(-n)*(cbrt(3)*x)**(n + 1)*sin(pi*(n*Rational(2, 3) + Rational(4, 3)))*factorial(n) *
                        gamma(n/3 + Rational(2, 3))/(sin(pi*(n*Rational(2, 3) + Rational(2, 3)))*factorial(n + 1)*gamma(n/3 + Rational(1, 3))) * p)
            else:
                return (S.One/(3**Rational(2, 3)*pi) * gamma((n+S.One)/S(3)) * sin(Rational(2, 3)*pi*(n+S.One)) /
                        factorial(n) * (cbrt(3)*x)**n)

    def _eval_rewrite_as_besselj(self, z, **kwargs):
        ot = Rational(1, 3)
        tt = Rational(2, 3)
        a = Pow(-z, Rational(3, 2))
        if re(z).is_negative:
            return ot*sqrt(-z) * (besselj(-ot, tt*a) + besselj(ot, tt*a))

    def _eval_rewrite_as_besseli(self, z, **kwargs):
        ot = Rational(1, 3)
        tt = Rational(2, 3)
        a = Pow(z, Rational(3, 2))
        if re(z).is_positive:
            return ot*sqrt(z) * (besseli(-ot, tt*a) - besseli(ot, tt*a))
        else:
            return ot*(Pow(a, ot)*besseli(-ot, tt*a) - z*Pow(a, -ot)*besseli(ot, tt*a))

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        pf1 = S.One / (3**Rational(2, 3)*gamma(Rational(2, 3)))
        pf2 = z / (root(3, 3)*gamma(Rational(1, 3)))
        return pf1 * hyper([], [Rational(2, 3)], z**3/9) - pf2 * hyper([], [Rational(4, 3)], z**3/9)

    def _eval_expand_func(self, **hints):
        arg = self.args[0]
        symbs = arg.free_symbols

        if len(symbs) == 1:
            z = symbs.pop()
            c = Wild("c", exclude=[z])
            d = Wild("d", exclude=[z])
            m = Wild("m", exclude=[z])
            n = Wild("n", exclude=[z])
            M = arg.match(c*(d*z**n)**m)
            if M is not None:
                m = M[m]
                # The transformation is given by 03.05.16.0001.01
                # https://functions.wolfram.com/Bessel-TypeFunctions/AiryAi/16/01/01/0001/
                if (3*m).is_integer:
                    c = M[c]
                    d = M[d]
                    n = M[n]
                    pf = (d * z**n)**m / (d**m * z**(m*n))
                    newarg = c * d**m * z**(m*n)
                    return S.Half * ((pf + S.One)*airyai(newarg) - (pf - S.One)/sqrt(3)*airybi(newarg))


class airybi(AiryBase):
    r"""
    The Airy function $\operatorname{Bi}$ of the second kind.

    Explanation
    ===========

    The Airy function $\operatorname{Bi}(z)$ is defined to be the function
    satisfying Airy's differential equation

    .. math::
        \frac{\mathrm{d}^2 w(z)}{\mathrm{d}z^2} - z w(z) = 0.

    Equivalently, for real $z$

    .. math::
        \operatorname{Bi}(z) := \frac{1}{\pi}
                 \int_0^\infty
                   \exp\left(-\frac{t^3}{3} + z t\right)
                   + \sin\left(\frac{t^3}{3} + z t\right) \mathrm{d}t.

    Examples
    ========

    Create an Airy function object:

    >>> from sympy import airybi
    >>> from sympy.abc import z

    >>> airybi(z)
    airybi(z)

    Several special values are known:

    >>> airybi(0)
    3**(5/6)/(3*gamma(2/3))
    >>> from sympy import oo
    >>> airybi(oo)
    oo
    >>> airybi(-oo)
    0

    The Airy function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(airybi(z))
    airybi(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(airybi(z), z)
    airybiprime(z)
    >>> diff(airybi(z), z, 2)
    z*airybi(z)

    Series expansion is also supported:

    >>> from sympy import series
    >>> series(airybi(z), z, 0, 3)
    3**(1/3)*gamma(1/3)/(2*pi) + 3**(2/3)*z*gamma(2/3)/(2*pi) + O(z**3)

    We can numerically evaluate the Airy function to arbitrary precision
    on the whole complex plane:

    >>> airybi(-2).evalf(50)
    -0.41230258795639848808323405461146104203453483447240

    Rewrite $\operatorname{Bi}(z)$ in terms of hypergeometric functions:

    >>> from sympy import hyper
    >>> airybi(z).rewrite(hyper)
    3**(1/6)*z*hyper((), (4/3,), z**3/9)/gamma(1/3) + 3**(5/6)*hyper((), (2/3,), z**3/9)/(3*gamma(2/3))

    See Also
    ========

    airyai: Airy function of the first kind.
    airyaiprime: Derivative of the Airy function of the first kind.
    airybiprime: Derivative of the Airy function of the second kind.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Airy_function
    .. [2] https://dlmf.nist.gov/9
    .. [3] https://encyclopediaofmath.org/wiki/Airy_functions
    .. [4] https://mathworld.wolfram.com/AiryFunctions.html

    """

    nargs = 1
    unbranched = True

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.Zero
            elif arg.is_zero:
                return S.One / (3**Rational(1, 6) * gamma(Rational(2, 3)))

        if arg.is_zero:
            return S.One / (3**Rational(1, 6) * gamma(Rational(2, 3)))

    def fdiff(self, argindex=1):
        if argindex == 1:
            return airybiprime(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) > 1:
                p = previous_terms[-1]
                return (cbrt(3)*x * Abs(sin(Rational(2, 3)*pi*(n + S.One))) * factorial((n - S.One)/S(3)) /
                        ((n + S.One) * Abs(cos(Rational(2, 3)*pi*(n + S.Half))) * factorial((n - 2)/S(3))) * p)
            else:
                return (S.One/(root(3, 6)*pi) * gamma((n + S.One)/S(3)) * Abs(sin(Rational(2, 3)*pi*(n + S.One))) /
                        factorial(n) * (cbrt(3)*x)**n)

    def _eval_rewrite_as_besselj(self, z, **kwargs):
        ot = Rational(1, 3)
        tt = Rational(2, 3)
        a = Pow(-z, Rational(3, 2))
        if re(z).is_negative:
            return sqrt(-z/3) * (besselj(-ot, tt*a) - besselj(ot, tt*a))

    def _eval_rewrite_as_besseli(self, z, **kwargs):
        ot = Rational(1, 3)
        tt = Rational(2, 3)
        a = Pow(z, Rational(3, 2))
        if re(z).is_positive:
            return sqrt(z)/sqrt(3) * (besseli(-ot, tt*a) + besseli(ot, tt*a))
        else:
            b = Pow(a, ot)
            c = Pow(a, -ot)
            return sqrt(ot)*(b*besseli(-ot, tt*a) + z*c*besseli(ot, tt*a))

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        pf1 = S.One / (root(3, 6)*gamma(Rational(2, 3)))
        pf2 = z*root(3, 6) / gamma(Rational(1, 3))
        return pf1 * hyper([], [Rational(2, 3)], z**3/9) + pf2 * hyper([], [Rational(4, 3)], z**3/9)

    def _eval_expand_func(self, **hints):
        arg = self.args[0]
        symbs = arg.free_symbols

        if len(symbs) == 1:
            z = symbs.pop()
            c = Wild("c", exclude=[z])
            d = Wild("d", exclude=[z])
            m = Wild("m", exclude=[z])
            n = Wild("n", exclude=[z])
            M = arg.match(c*(d*z**n)**m)
            if M is not None:
                m = M[m]
                # The transformation is given by 03.06.16.0001.01
                # https://functions.wolfram.com/Bessel-TypeFunctions/AiryBi/16/01/01/0001/
                if (3*m).is_integer:
                    c = M[c]
                    d = M[d]
                    n = M[n]
                    pf = (d * z**n)**m / (d**m * z**(m*n))
                    newarg = c * d**m * z**(m*n)
                    return S.Half * (sqrt(3)*(S.One - pf)*airyai(newarg) + (S.One + pf)*airybi(newarg))


class airyaiprime(AiryBase):
    r"""
    The derivative $\operatorname{Ai}^\prime$ of the Airy function of the first
    kind.

    Explanation
    ===========

    The Airy function $\operatorname{Ai}^\prime(z)$ is defined to be the
    function

    .. math::
        \operatorname{Ai}^\prime(z) := \frac{\mathrm{d} \operatorname{Ai}(z)}{\mathrm{d} z}.

    Examples
    ========

    Create an Airy function object:

    >>> from sympy import airyaiprime
    >>> from sympy.abc import z

    >>> airyaiprime(z)
    airyaiprime(z)

    Several special values are known:

    >>> airyaiprime(0)
    -3**(2/3)/(3*gamma(1/3))
    >>> from sympy import oo
    >>> airyaiprime(oo)
    0

    The Airy function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(airyaiprime(z))
    airyaiprime(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(airyaiprime(z), z)
    z*airyai(z)
    >>> diff(airyaiprime(z), z, 2)
    z*airyaiprime(z) + airyai(z)

    Series expansion is also supported:

    >>> from sympy import series
    >>> series(airyaiprime(z), z, 0, 3)
    -3**(2/3)/(3*gamma(1/3)) + 3**(1/3)*z**2/(6*gamma(2/3)) + O(z**3)

    We can numerically evaluate the Airy function to arbitrary precision
    on the whole complex plane:

    >>> airyaiprime(-2).evalf(50)
    0.61825902074169104140626429133247528291577794512415

    Rewrite $\operatorname{Ai}^\prime(z)$ in terms of hypergeometric functions:

    >>> from sympy import hyper
    >>> airyaiprime(z).rewrite(hyper)
    3**(1/3)*z**2*hyper((), (5/3,), z**3/9)/(6*gamma(2/3)) - 3**(2/3)*hyper((), (1/3,), z**3/9)/(3*gamma(1/3))

    See Also
    ========

    airyai: Airy function of the first kind.
    airybi: Airy function of the second kind.
    airybiprime: Derivative of the Airy function of the second kind.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Airy_function
    .. [2] https://dlmf.nist.gov/9
    .. [3] https://encyclopediaofmath.org/wiki/Airy_functions
    .. [4] https://mathworld.wolfram.com/AiryFunctions.html

    """

    nargs = 1
    unbranched = True

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Zero

        if arg.is_zero:
            return S.NegativeOne / (3**Rational(1, 3) * gamma(Rational(1, 3)))

    def fdiff(self, argindex=1):
        if argindex == 1:
            return self.args[0]*airyai(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_evalf(self, prec):
        z = self.args[0]._to_mpmath(prec)
        with workprec(prec):
            res = mp.airyai(z, derivative=1)
        return Expr._from_mpmath(res, prec)

    def _eval_rewrite_as_besselj(self, z, **kwargs):
        tt = Rational(2, 3)
        a = Pow(-z, Rational(3, 2))
        if re(z).is_negative:
            return z/3 * (besselj(-tt, tt*a) - besselj(tt, tt*a))

    def _eval_rewrite_as_besseli(self, z, **kwargs):
        ot = Rational(1, 3)
        tt = Rational(2, 3)
        a = tt * Pow(z, Rational(3, 2))
        if re(z).is_positive:
            return z/3 * (besseli(tt, a) - besseli(-tt, a))
        else:
            a = Pow(z, Rational(3, 2))
            b = Pow(a, tt)
            c = Pow(a, -tt)
            return ot * (z**2*c*besseli(tt, tt*a) - b*besseli(-ot, tt*a))

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        pf1 = z**2 / (2*3**Rational(2, 3)*gamma(Rational(2, 3)))
        pf2 = 1 / (root(3, 3)*gamma(Rational(1, 3)))
        return pf1 * hyper([], [Rational(5, 3)], z**3/9) - pf2 * hyper([], [Rational(1, 3)], z**3/9)

    def _eval_expand_func(self, **hints):
        arg = self.args[0]
        symbs = arg.free_symbols

        if len(symbs) == 1:
            z = symbs.pop()
            c = Wild("c", exclude=[z])
            d = Wild("d", exclude=[z])
            m = Wild("m", exclude=[z])
            n = Wild("n", exclude=[z])
            M = arg.match(c*(d*z**n)**m)
            if M is not None:
                m = M[m]
                # The transformation is in principle
                # given by 03.07.16.0001.01 but note
                # that there is an error in this formula.
                # https://functions.wolfram.com/Bessel-TypeFunctions/AiryAiPrime/16/01/01/0001/
                if (3*m).is_integer:
                    c = M[c]
                    d = M[d]
                    n = M[n]
                    pf = (d**m * z**(n*m)) / (d * z**n)**m
                    newarg = c * d**m * z**(n*m)
                    return S.Half * ((pf + S.One)*airyaiprime(newarg) + (pf - S.One)/sqrt(3)*airybiprime(newarg))


class airybiprime(AiryBase):
    r"""
    The derivative $\operatorname{Bi}^\prime$ of the Airy function of the first
    kind.

    Explanation
    ===========

    The Airy function $\operatorname{Bi}^\prime(z)$ is defined to be the
    function

    .. math::
        \operatorname{Bi}^\prime(z) := \frac{\mathrm{d} \operatorname{Bi}(z)}{\mathrm{d} z}.

    Examples
    ========

    Create an Airy function object:

    >>> from sympy import airybiprime
    >>> from sympy.abc import z

    >>> airybiprime(z)
    airybiprime(z)

    Several special values are known:

    >>> airybiprime(0)
    3**(1/6)/gamma(1/3)
    >>> from sympy import oo
    >>> airybiprime(oo)
    oo
    >>> airybiprime(-oo)
    0

    The Airy function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(airybiprime(z))
    airybiprime(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(airybiprime(z), z)
    z*airybi(z)
    >>> diff(airybiprime(z), z, 2)
    z*airybiprime(z) + airybi(z)

    Series expansion is also supported:

    >>> from sympy import series
    >>> series(airybiprime(z), z, 0, 3)
    3**(1/6)/gamma(1/3) + 3**(5/6)*z**2/(6*gamma(2/3)) + O(z**3)

    We can numerically evaluate the Airy function to arbitrary precision
    on the whole complex plane:

    >>> airybiprime(-2).evalf(50)
    0.27879516692116952268509756941098324140300059345163

    Rewrite $\operatorname{Bi}^\prime(z)$ in terms of hypergeometric functions:

    >>> from sympy import hyper
    >>> airybiprime(z).rewrite(hyper)
    3**(5/6)*z**2*hyper((), (5/3,), z**3/9)/(6*gamma(2/3)) + 3**(1/6)*hyper((), (1/3,), z**3/9)/gamma(1/3)

    See Also
    ========

    airyai: Airy function of the first kind.
    airybi: Airy function of the second kind.
    airyaiprime: Derivative of the Airy function of the first kind.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Airy_function
    .. [2] https://dlmf.nist.gov/9
    .. [3] https://encyclopediaofmath.org/wiki/Airy_functions
    .. [4] https://mathworld.wolfram.com/AiryFunctions.html

    """

    nargs = 1
    unbranched = True

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.Zero
            elif arg.is_zero:
                return 3**Rational(1, 6) / gamma(Rational(1, 3))

        if arg.is_zero:
            return 3**Rational(1, 6) / gamma(Rational(1, 3))


    def fdiff(self, argindex=1):
        if argindex == 1:
            return self.args[0]*airybi(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_evalf(self, prec):
        z = self.args[0]._to_mpmath(prec)
        with workprec(prec):
            res = mp.airybi(z, derivative=1)
        return Expr._from_mpmath(res, prec)

    def _eval_rewrite_as_besselj(self, z, **kwargs):
        tt = Rational(2, 3)
        a = tt * Pow(-z, Rational(3, 2))
        if re(z).is_negative:
            return -z/sqrt(3) * (besselj(-tt, a) + besselj(tt, a))

    def _eval_rewrite_as_besseli(self, z, **kwargs):
        ot = Rational(1, 3)
        tt = Rational(2, 3)
        a = tt * Pow(z, Rational(3, 2))
        if re(z).is_positive:
            return z/sqrt(3) * (besseli(-tt, a) + besseli(tt, a))
        else:
            a = Pow(z, Rational(3, 2))
            b = Pow(a, tt)
            c = Pow(a, -tt)
            return sqrt(ot) * (b*besseli(-tt, tt*a) + z**2*c*besseli(tt, tt*a))

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        pf1 = z**2 / (2*root(3, 6)*gamma(Rational(2, 3)))
        pf2 = root(3, 6) / gamma(Rational(1, 3))
        return pf1 * hyper([], [Rational(5, 3)], z**3/9) + pf2 * hyper([], [Rational(1, 3)], z**3/9)

    def _eval_expand_func(self, **hints):
        arg = self.args[0]
        symbs = arg.free_symbols

        if len(symbs) == 1:
            z = symbs.pop()
            c = Wild("c", exclude=[z])
            d = Wild("d", exclude=[z])
            m = Wild("m", exclude=[z])
            n = Wild("n", exclude=[z])
            M = arg.match(c*(d*z**n)**m)
            if M is not None:
                m = M[m]
                # The transformation is in principle
                # given by 03.08.16.0001.01 but note
                # that there is an error in this formula.
                # https://functions.wolfram.com/Bessel-TypeFunctions/AiryBiPrime/16/01/01/0001/
                if (3*m).is_integer:
                    c = M[c]
                    d = M[d]
                    n = M[n]
                    pf = (d**m * z**(n*m)) / (d * z**n)**m
                    newarg = c * d**m * z**(n*m)
                    return S.Half * (sqrt(3)*(pf - S.One)*airyaiprime(newarg) + (pf + S.One)*airybiprime(newarg))


class marcumq(Function):
    r"""
    The Marcum Q-function.

    Explanation
    ===========

    The Marcum Q-function is defined by the meromorphic continuation of

    .. math::
        Q_m(a, b) = a^{- m + 1} \int_{b}^{\infty} x^{m} e^{- \frac{a^{2}}{2} - \frac{x^{2}}{2}} I_{m - 1}\left(a x\right)\, dx

    Examples
    ========

    >>> from sympy import marcumq
    >>> from sympy.abc import m, a, b
    >>> marcumq(m, a, b)
    marcumq(m, a, b)

    Special values:

    >>> marcumq(m, 0, b)
    uppergamma(m, b**2/2)/gamma(m)
    >>> marcumq(0, 0, 0)
    0
    >>> marcumq(0, a, 0)
    1 - exp(-a**2/2)
    >>> marcumq(1, a, a)
    1/2 + exp(-a**2)*besseli(0, a**2)/2
    >>> marcumq(2, a, a)
    1/2 + exp(-a**2)*besseli(0, a**2)/2 + exp(-a**2)*besseli(1, a**2)

    Differentiation with respect to $a$ and $b$ is supported:

    >>> from sympy import diff
    >>> diff(marcumq(m, a, b), a)
    a*(-marcumq(m, a, b) + marcumq(m + 1, a, b))
    >>> diff(marcumq(m, a, b), b)
    -a**(1 - m)*b**m*exp(-a**2/2 - b**2/2)*besseli(m - 1, a*b)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Marcum_Q-function
    .. [2] https://mathworld.wolfram.com/MarcumQ-Function.html

    """

    @classmethod
    def eval(cls, m, a, b):
        if a is S.Zero:
            if m is S.Zero and b is S.Zero:
                return S.Zero
            return uppergamma(m, b**2 * S.Half) / gamma(m)

        if m is S.Zero and b is S.Zero:
            return 1 - 1 / exp(a**2 * S.Half)

        if a == b:
            if m is S.One:
                return (1 + exp(-a**2) * besseli(0, a**2))*S.Half
            if m == 2:
                return S.Half + S.Half * exp(-a**2) * besseli(0, a**2) + exp(-a**2) * besseli(1, a**2)

        if a.is_zero:
            if m.is_zero and b.is_zero:
                return S.Zero
            return uppergamma(m, b**2*S.Half) / gamma(m)

        if m.is_zero and b.is_zero:
            return 1 - 1 / exp(a**2*S.Half)

    def fdiff(self, argindex=2):
        m, a, b = self.args
        if argindex == 2:
            return a * (-marcumq(m, a, b) + marcumq(1+m, a, b))
        elif argindex == 3:
            return (-b**m / a**(m-1)) * exp(-(a**2 + b**2)/2) * besseli(m-1, a*b)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Integral(self, m, a, b, **kwargs):
        from sympy.integrals.integrals import Integral
        x = kwargs.get('x', Dummy(uniquely_named_symbol('x').name))
        return a ** (1 - m) * \
               Integral(x**m * exp(-(x**2 + a**2)/2) * besseli(m-1, a*x), [x, b, S.Infinity])

    def _eval_rewrite_as_Sum(self, m, a, b, **kwargs):
        from sympy.concrete.summations import Sum
        k = kwargs.get('k', Dummy('k'))
        return exp(-(a**2 + b**2) / 2) * Sum((a/b)**k * besseli(k, a*b), [k, 1-m, S.Infinity])

    def _eval_rewrite_as_besseli(self, m, a, b, **kwargs):
        if a == b:
            if m == 1:
                return (1 + exp(-a**2) * besseli(0, a**2)) / 2
            if m.is_Integer and m >= 2:
                s = sum(besseli(i, a**2) for i in range(1, m))
                return S.Half + exp(-a**2) * besseli(0, a**2) / 2 + exp(-a**2) * s

    def _eval_is_zero(self):
        if all(arg.is_zero for arg in self.args):
            return True
