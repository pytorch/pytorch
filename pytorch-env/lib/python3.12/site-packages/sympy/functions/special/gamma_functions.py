from math import prod

from sympy.core import Add, S, Dummy, expand_func
from sympy.core.expr import Expr
from sympy.core.function import Function, ArgumentIndexError, PoleError
from sympy.core.logic import fuzzy_and, fuzzy_not
from sympy.core.numbers import Rational, pi, oo, I
from sympy.core.power import Pow
from sympy.functions.special.zeta_functions import zeta
from sympy.functions.special.error_functions import erf, erfc, Ei
from sympy.functions.elementary.complexes import re, unpolarify
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin, cos, cot
from sympy.functions.combinatorial.numbers import bernoulli, harmonic
from sympy.functions.combinatorial.factorials import factorial, rf, RisingFactorial
from sympy.utilities.misc import as_int

from mpmath import mp, workprec
from mpmath.libmp.libmpf import prec_to_dps

def intlike(n):
    try:
        as_int(n, strict=False)
        return True
    except ValueError:
        return False

###############################################################################
############################ COMPLETE GAMMA FUNCTION ##########################
###############################################################################

class gamma(Function):
    r"""
    The gamma function

    .. math::
        \Gamma(x) := \int^{\infty}_{0} t^{x-1} e^{-t} \mathrm{d}t.

    Explanation
    ===========

    The ``gamma`` function implements the function which passes through the
    values of the factorial function (i.e., $\Gamma(n) = (n - 1)!$ when n is
    an integer). More generally, $\Gamma(z)$ is defined in the whole complex
    plane except at the negative integers where there are simple poles.

    Examples
    ========

    >>> from sympy import S, I, pi, gamma
    >>> from sympy.abc import x

    Several special values are known:

    >>> gamma(1)
    1
    >>> gamma(4)
    6
    >>> gamma(S(3)/2)
    sqrt(pi)/2

    The ``gamma`` function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(gamma(x))
    gamma(conjugate(x))

    Differentiation with respect to $x$ is supported:

    >>> from sympy import diff
    >>> diff(gamma(x), x)
    gamma(x)*polygamma(0, x)

    Series expansion is also supported:

    >>> from sympy import series
    >>> series(gamma(x), x, 0, 3)
    1/x - EulerGamma + x*(EulerGamma**2/2 + pi**2/12) + x**2*(-EulerGamma*pi**2/12 - zeta(3)/3 - EulerGamma**3/6) + O(x**3)

    We can numerically evaluate the ``gamma`` function to arbitrary precision
    on the whole complex plane:

    >>> gamma(pi).evalf(40)
    2.288037795340032417959588909060233922890
    >>> gamma(1+I).evalf(20)
    0.49801566811835604271 - 0.15494982830181068512*I

    See Also
    ========

    lowergamma: Lower incomplete gamma function.
    uppergamma: Upper incomplete gamma function.
    polygamma: Polygamma function.
    loggamma: Log Gamma function.
    digamma: Digamma function.
    trigamma: Trigamma function.
    sympy.functions.special.beta_functions.beta: Euler Beta function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gamma_function
    .. [2] https://dlmf.nist.gov/5
    .. [3] https://mathworld.wolfram.com/GammaFunction.html
    .. [4] https://functions.wolfram.com/GammaBetaErf/Gamma/

    """

    unbranched = True
    _singularities = (S.ComplexInfinity,)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return self.func(self.args[0])*polygamma(0, self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is oo:
                return oo
            elif intlike(arg):
                if arg.is_positive:
                    return factorial(arg - 1)
                else:
                    return S.ComplexInfinity
            elif arg.is_Rational:
                if arg.q == 2:
                    n = abs(arg.p) // arg.q

                    if arg.is_positive:
                        k, coeff = n, S.One
                    else:
                        n = k = n + 1

                        if n & 1 == 0:
                            coeff = S.One
                        else:
                            coeff = S.NegativeOne

                    coeff *= prod(range(3, 2*k, 2))

                    if arg.is_positive:
                        return coeff*sqrt(pi) / 2**n
                    else:
                        return 2**n*sqrt(pi) / coeff

    def _eval_expand_func(self, **hints):
        arg = self.args[0]
        if arg.is_Rational:
            if abs(arg.p) > arg.q:
                x = Dummy('x')
                n = arg.p // arg.q
                p = arg.p - n*arg.q
                return self.func(x + n)._eval_expand_func().subs(x, Rational(p, arg.q))

        if arg.is_Add:
            coeff, tail = arg.as_coeff_add()
            if coeff and coeff.q != 1:
                intpart = floor(coeff)
                tail = (coeff - intpart,) + tail
                coeff = intpart
            tail = arg._new_rawargs(*tail, reeval=False)
            return self.func(tail)*RisingFactorial(tail, coeff)

        return self.func(*self.args)

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def _eval_is_real(self):
        x = self.args[0]
        if x.is_nonpositive and x.is_integer:
            return False
        if intlike(x) and x <= 0:
            return False
        if x.is_positive or x.is_noninteger:
            return True

    def _eval_is_positive(self):
        x = self.args[0]
        if x.is_positive:
            return True
        elif x.is_noninteger:
            return floor(x).is_even

    def _eval_rewrite_as_tractable(self, z, limitvar=None, **kwargs):
        return exp(loggamma(z))

    def _eval_rewrite_as_factorial(self, z, **kwargs):
        return factorial(z - 1)

    def _eval_nseries(self, x, n, logx, cdir=0):
        x0 = self.args[0].limit(x, 0)
        if not (x0.is_Integer and x0 <= 0):
            return super()._eval_nseries(x, n, logx)
        t = self.args[0] - x0
        return (self.func(t + 1)/rf(self.args[0], -x0 + 1))._eval_nseries(x, n, logx)

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        arg = self.args[0]
        x0 = arg.subs(x, 0)

        if x0.is_integer and x0.is_nonpositive:
            n = -x0
            res = S.NegativeOne**n/self.func(n + 1)
            return res/(arg + n).as_leading_term(x)
        elif not x0.is_infinite:
            return self.func(x0)
        raise PoleError()


###############################################################################
################## LOWER and UPPER INCOMPLETE GAMMA FUNCTIONS #################
###############################################################################

class lowergamma(Function):
    r"""
    The lower incomplete gamma function.

    Explanation
    ===========

    It can be defined as the meromorphic continuation of

    .. math::
        \gamma(s, x) := \int_0^x t^{s-1} e^{-t} \mathrm{d}t = \Gamma(s) - \Gamma(s, x).

    This can be shown to be the same as

    .. math::
        \gamma(s, x) = \frac{x^s}{s} {}_1F_1\left({s \atop s+1} \middle| -x\right),

    where ${}_1F_1$ is the (confluent) hypergeometric function.

    Examples
    ========

    >>> from sympy import lowergamma, S
    >>> from sympy.abc import s, x
    >>> lowergamma(s, x)
    lowergamma(s, x)
    >>> lowergamma(3, x)
    -2*(x**2/2 + x + 1)*exp(-x) + 2
    >>> lowergamma(-S(1)/2, x)
    -2*sqrt(pi)*erf(sqrt(x)) - 2*exp(-x)/sqrt(x)

    See Also
    ========

    gamma: Gamma function.
    uppergamma: Upper incomplete gamma function.
    polygamma: Polygamma function.
    loggamma: Log Gamma function.
    digamma: Digamma function.
    trigamma: Trigamma function.
    sympy.functions.special.beta_functions.beta: Euler Beta function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Incomplete_gamma_function#Lower_incomplete_gamma_function
    .. [2] Abramowitz, Milton; Stegun, Irene A., eds. (1965), Chapter 6,
           Section 5, Handbook of Mathematical Functions with Formulas, Graphs,
           and Mathematical Tables
    .. [3] https://dlmf.nist.gov/8
    .. [4] https://functions.wolfram.com/GammaBetaErf/Gamma2/
    .. [5] https://functions.wolfram.com/GammaBetaErf/Gamma3/

    """


    def fdiff(self, argindex=2):
        from sympy.functions.special.hyper import meijerg
        if argindex == 2:
            a, z = self.args
            return exp(-unpolarify(z))*z**(a - 1)
        elif argindex == 1:
            a, z = self.args
            return gamma(a)*digamma(a) - log(z)*uppergamma(a, z) \
                - meijerg([], [1, 1], [0, 0, a], [], z)

        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, a, x):
        # For lack of a better place, we use this one to extract branching
        # information. The following can be
        # found in the literature (c/f references given above), albeit scattered:
        # 1) For fixed x != 0, lowergamma(s, x) is an entire function of s
        # 2) For fixed positive integers s, lowergamma(s, x) is an entire
        #    function of x.
        # 3) For fixed non-positive integers s,
        #    lowergamma(s, exp(I*2*pi*n)*x) =
        #              2*pi*I*n*(-1)**(-s)/factorial(-s) + lowergamma(s, x)
        #    (this follows from lowergamma(s, x).diff(x) = x**(s-1)*exp(-x)).
        # 4) For fixed non-integral s,
        #    lowergamma(s, x) = x**s*gamma(s)*lowergamma_unbranched(s, x),
        #    where lowergamma_unbranched(s, x) is an entire function (in fact
        #    of both s and x), i.e.
        #    lowergamma(s, exp(2*I*pi*n)*x) = exp(2*pi*I*n*a)*lowergamma(a, x)
        if x is S.Zero:
            return S.Zero
        nx, n = x.extract_branch_factor()
        if a.is_integer and a.is_positive:
            nx = unpolarify(x)
            if nx != x:
                return lowergamma(a, nx)
        elif a.is_integer and a.is_nonpositive:
            if n != 0:
                return 2*pi*I*n*S.NegativeOne**(-a)/factorial(-a) + lowergamma(a, nx)
        elif n != 0:
            return exp(2*pi*I*n*a)*lowergamma(a, nx)

        # Special values.
        if a.is_Number:
            if a is S.One:
                return S.One - exp(-x)
            elif a is S.Half:
                return sqrt(pi)*erf(sqrt(x))
            elif a.is_Integer or (2*a).is_Integer:
                b = a - 1
                if b.is_positive:
                    if a.is_integer:
                        return factorial(b) - exp(-x) * factorial(b) * Add(*[x ** k / factorial(k) for k in range(a)])
                    else:
                        return gamma(a)*(lowergamma(S.Half, x)/sqrt(pi) - exp(-x)*Add(*[x**(k - S.Half)/gamma(S.Half + k) for k in range(1, a + S.Half)]))

                if not a.is_Integer:
                    return S.NegativeOne**(S.Half - a)*pi*erf(sqrt(x))/gamma(1 - a) + exp(-x)*Add(*[x**(k + a - 1)*gamma(a)/gamma(a + k) for k in range(1, Rational(3, 2) - a)])

        if x.is_zero:
            return S.Zero

    def _eval_evalf(self, prec):
        if all(x.is_number for x in self.args):
            a = self.args[0]._to_mpmath(prec)
            z = self.args[1]._to_mpmath(prec)
            with workprec(prec):
                res = mp.gammainc(a, 0, z)
            return Expr._from_mpmath(res, prec)
        else:
            return self

    def _eval_conjugate(self):
        x = self.args[1]
        if x not in (S.Zero, S.NegativeInfinity):
            return self.func(self.args[0].conjugate(), x.conjugate())

    def _eval_is_meromorphic(self, x, a):
        # By https://en.wikipedia.org/wiki/Incomplete_gamma_function#Holomorphic_extension,
        #    lowergamma(s, z) = z**s*gamma(s)*gammastar(s, z),
        # where gammastar(s, z) is holomorphic for all s and z.
        # Hence the singularities of lowergamma are z = 0  (branch
        # point) and nonpositive integer values of s (poles of gamma(s)).
        s, z = self.args
        args_merom = fuzzy_and([z._eval_is_meromorphic(x, a),
            s._eval_is_meromorphic(x, a)])
        if not args_merom:
            return args_merom
        z0 = z.subs(x, a)
        if s.is_integer:
            return fuzzy_and([s.is_positive, z0.is_finite])
        s0 = s.subs(x, a)
        return fuzzy_and([s0.is_finite, z0.is_finite, fuzzy_not(z0.is_zero)])

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import O
        s, z = self.args
        if args0[0] is oo and not z.has(x):
            coeff = z**s*exp(-z)
            sum_expr = sum(z**k/rf(s, k + 1) for k in range(n - 1))
            o = O(z**s*s**(-n))
            return coeff*sum_expr + o
        return super()._eval_aseries(n, args0, x, logx)

    def _eval_rewrite_as_uppergamma(self, s, x, **kwargs):
        return gamma(s) - uppergamma(s, x)

    def _eval_rewrite_as_expint(self, s, x, **kwargs):
        from sympy.functions.special.error_functions import expint
        if s.is_integer and s.is_nonpositive:
            return self
        return self.rewrite(uppergamma).rewrite(expint)

    def _eval_is_zero(self):
        x = self.args[1]
        if x.is_zero:
            return True


class uppergamma(Function):
    r"""
    The upper incomplete gamma function.

    Explanation
    ===========

    It can be defined as the meromorphic continuation of

    .. math::
        \Gamma(s, x) := \int_x^\infty t^{s-1} e^{-t} \mathrm{d}t = \Gamma(s) - \gamma(s, x).

    where $\gamma(s, x)$ is the lower incomplete gamma function,
    :class:`lowergamma`. This can be shown to be the same as

    .. math::
        \Gamma(s, x) = \Gamma(s) - \frac{x^s}{s} {}_1F_1\left({s \atop s+1} \middle| -x\right),

    where ${}_1F_1$ is the (confluent) hypergeometric function.

    The upper incomplete gamma function is also essentially equivalent to the
    generalized exponential integral:

    .. math::
        \operatorname{E}_{n}(x) = \int_{1}^{\infty}{\frac{e^{-xt}}{t^n} \, dt} = x^{n-1}\Gamma(1-n,x).

    Examples
    ========

    >>> from sympy import uppergamma, S
    >>> from sympy.abc import s, x
    >>> uppergamma(s, x)
    uppergamma(s, x)
    >>> uppergamma(3, x)
    2*(x**2/2 + x + 1)*exp(-x)
    >>> uppergamma(-S(1)/2, x)
    -2*sqrt(pi)*erfc(sqrt(x)) + 2*exp(-x)/sqrt(x)
    >>> uppergamma(-2, x)
    expint(3, x)/x**2

    See Also
    ========

    gamma: Gamma function.
    lowergamma: Lower incomplete gamma function.
    polygamma: Polygamma function.
    loggamma: Log Gamma function.
    digamma: Digamma function.
    trigamma: Trigamma function.
    sympy.functions.special.beta_functions.beta: Euler Beta function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Incomplete_gamma_function#Upper_incomplete_gamma_function
    .. [2] Abramowitz, Milton; Stegun, Irene A., eds. (1965), Chapter 6,
           Section 5, Handbook of Mathematical Functions with Formulas, Graphs,
           and Mathematical Tables
    .. [3] https://dlmf.nist.gov/8
    .. [4] https://functions.wolfram.com/GammaBetaErf/Gamma2/
    .. [5] https://functions.wolfram.com/GammaBetaErf/Gamma3/
    .. [6] https://en.wikipedia.org/wiki/Exponential_integral#Relation_with_other_functions

    """


    def fdiff(self, argindex=2):
        from sympy.functions.special.hyper import meijerg
        if argindex == 2:
            a, z = self.args
            return -exp(-unpolarify(z))*z**(a - 1)
        elif argindex == 1:
            a, z = self.args
            return uppergamma(a, z)*log(z) + meijerg([], [1, 1], [0, 0, a], [], z)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_evalf(self, prec):
        if all(x.is_number for x in self.args):
            a = self.args[0]._to_mpmath(prec)
            z = self.args[1]._to_mpmath(prec)
            with workprec(prec):
                res = mp.gammainc(a, z, mp.inf)
            return Expr._from_mpmath(res, prec)
        return self

    @classmethod
    def eval(cls, a, z):
        from sympy.functions.special.error_functions import expint
        if z.is_Number:
            if z is S.NaN:
                return S.NaN
            elif z is oo:
                return S.Zero
            elif z.is_zero:
                if re(a).is_positive:
                    return gamma(a)

        # We extract branching information here. C/f lowergamma.
        nx, n = z.extract_branch_factor()
        if a.is_integer and a.is_positive:
            nx = unpolarify(z)
            if z != nx:
                return uppergamma(a, nx)
        elif a.is_integer and a.is_nonpositive:
            if n != 0:
                return -2*pi*I*n*S.NegativeOne**(-a)/factorial(-a) + uppergamma(a, nx)
        elif n != 0:
            return gamma(a)*(1 - exp(2*pi*I*n*a)) + exp(2*pi*I*n*a)*uppergamma(a, nx)

        # Special values.
        if a.is_Number:
            if a is S.Zero and z.is_positive:
                return -Ei(-z)
            elif a is S.One:
                return exp(-z)
            elif a is S.Half:
                return sqrt(pi)*erfc(sqrt(z))
            elif a.is_Integer or (2*a).is_Integer:
                b = a - 1
                if b.is_positive:
                    if a.is_integer:
                        return exp(-z) * factorial(b) * Add(*[z**k / factorial(k)
                                                              for k in range(a)])
                    else:
                        return (gamma(a) * erfc(sqrt(z)) +
                                S.NegativeOne**(a - S(3)/2) * exp(-z) * sqrt(z)
                                * Add(*[gamma(-S.Half - k) * (-z)**k / gamma(1-a)
                                        for k in range(a - S.Half)]))
                elif b.is_Integer:
                    return expint(-b, z)*unpolarify(z)**(b + 1)

                if not a.is_Integer:
                    return (S.NegativeOne**(S.Half - a) * pi*erfc(sqrt(z))/gamma(1-a)
                            - z**a * exp(-z) * Add(*[z**k * gamma(a) / gamma(a+k+1)
                                                     for k in range(S.Half - a)]))

        if a.is_zero and z.is_positive:
            return -Ei(-z)

        if z.is_zero and re(a).is_positive:
            return gamma(a)

    def _eval_conjugate(self):
        z = self.args[1]
        if z not in (S.Zero, S.NegativeInfinity):
            return self.func(self.args[0].conjugate(), z.conjugate())

    def _eval_is_meromorphic(self, x, a):
        return lowergamma._eval_is_meromorphic(self, x, a)

    def _eval_rewrite_as_lowergamma(self, s, x, **kwargs):
        return gamma(s) - lowergamma(s, x)

    def _eval_rewrite_as_tractable(self, s, x, **kwargs):
        return exp(loggamma(s)) - lowergamma(s, x)

    def _eval_rewrite_as_expint(self, s, x, **kwargs):
        from sympy.functions.special.error_functions import expint
        return expint(1 - s, x)*x**s


###############################################################################
###################### POLYGAMMA and LOGGAMMA FUNCTIONS #######################
###############################################################################

class polygamma(Function):
    r"""
    The function ``polygamma(n, z)`` returns ``log(gamma(z)).diff(n + 1)``.

    Explanation
    ===========

    It is a meromorphic function on $\mathbb{C}$ and defined as the $(n+1)$-th
    derivative of the logarithm of the gamma function:

    .. math::
        \psi^{(n)} (z) := \frac{\mathrm{d}^{n+1}}{\mathrm{d} z^{n+1}} \log\Gamma(z).

    For `n` not a nonnegative integer the generalization by Espinosa and Moll [5]_
    is used:

    .. math:: \psi(s,z) = \frac{\zeta'(s+1, z) + (\gamma + \psi(-s)) \zeta(s+1, z)}
        {\Gamma(-s)}

    Examples
    ========

    Several special values are known:

    >>> from sympy import S, polygamma
    >>> polygamma(0, 1)
    -EulerGamma
    >>> polygamma(0, 1/S(2))
    -2*log(2) - EulerGamma
    >>> polygamma(0, 1/S(3))
    -log(3) - sqrt(3)*pi/6 - EulerGamma - log(sqrt(3))
    >>> polygamma(0, 1/S(4))
    -pi/2 - log(4) - log(2) - EulerGamma
    >>> polygamma(0, 2)
    1 - EulerGamma
    >>> polygamma(0, 23)
    19093197/5173168 - EulerGamma

    >>> from sympy import oo, I
    >>> polygamma(0, oo)
    oo
    >>> polygamma(0, -oo)
    oo
    >>> polygamma(0, I*oo)
    oo
    >>> polygamma(0, -I*oo)
    oo

    Differentiation with respect to $x$ is supported:

    >>> from sympy import Symbol, diff
    >>> x = Symbol("x")
    >>> diff(polygamma(0, x), x)
    polygamma(1, x)
    >>> diff(polygamma(0, x), x, 2)
    polygamma(2, x)
    >>> diff(polygamma(0, x), x, 3)
    polygamma(3, x)
    >>> diff(polygamma(1, x), x)
    polygamma(2, x)
    >>> diff(polygamma(1, x), x, 2)
    polygamma(3, x)
    >>> diff(polygamma(2, x), x)
    polygamma(3, x)
    >>> diff(polygamma(2, x), x, 2)
    polygamma(4, x)

    >>> n = Symbol("n")
    >>> diff(polygamma(n, x), x)
    polygamma(n + 1, x)
    >>> diff(polygamma(n, x), x, 2)
    polygamma(n + 2, x)

    We can rewrite ``polygamma`` functions in terms of harmonic numbers:

    >>> from sympy import harmonic
    >>> polygamma(0, x).rewrite(harmonic)
    harmonic(x - 1) - EulerGamma
    >>> polygamma(2, x).rewrite(harmonic)
    2*harmonic(x - 1, 3) - 2*zeta(3)
    >>> ni = Symbol("n", integer=True)
    >>> polygamma(ni, x).rewrite(harmonic)
    (-1)**(n + 1)*(-harmonic(x - 1, n + 1) + zeta(n + 1))*factorial(n)

    See Also
    ========

    gamma: Gamma function.
    lowergamma: Lower incomplete gamma function.
    uppergamma: Upper incomplete gamma function.
    loggamma: Log Gamma function.
    digamma: Digamma function.
    trigamma: Trigamma function.
    sympy.functions.special.beta_functions.beta: Euler Beta function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Polygamma_function
    .. [2] https://mathworld.wolfram.com/PolygammaFunction.html
    .. [3] https://functions.wolfram.com/GammaBetaErf/PolyGamma/
    .. [4] https://functions.wolfram.com/GammaBetaErf/PolyGamma2/
    .. [5] O. Espinosa and V. Moll, "A generalized polygamma function",
           *Integral Transforms and Special Functions* (2004), 101-115.

    """

    @classmethod
    def eval(cls, n, z):
        if n is S.NaN or z is S.NaN:
            return S.NaN
        elif z is oo:
            return oo if n.is_zero else S.Zero
        elif z.is_Integer and z.is_nonpositive:
            return S.ComplexInfinity
        elif n is S.NegativeOne:
            return loggamma(z) - log(2*pi) / 2
        elif n.is_zero:
            if z is -oo or z.extract_multiplicatively(I) in (oo, -oo):
                return oo
            elif z.is_Integer:
                return harmonic(z-1) - S.EulerGamma
            elif z.is_Rational:
                # TODO n == 1 also can do some rational z
                p, q = z.as_numer_denom()
                # only expand for small denominators to avoid creating long expressions
                if q <= 6:
                    return expand_func(polygamma(S.Zero, z, evaluate=False))
        elif n.is_integer and n.is_nonnegative:
            nz = unpolarify(z)
            if z != nz:
                return polygamma(n, nz)
            if z.is_Integer:
                return S.NegativeOne**(n+1) * factorial(n) * zeta(n+1, z)
            elif z is S.Half:
                return S.NegativeOne**(n+1) * factorial(n) * (2**(n+1)-1) * zeta(n+1)

    def _eval_is_real(self):
        if self.args[0].is_positive and self.args[1].is_positive:
            return True

    def _eval_is_complex(self):
        z = self.args[1]
        is_negative_integer = fuzzy_and([z.is_negative, z.is_integer])
        return fuzzy_and([z.is_complex, fuzzy_not(is_negative_integer)])

    def _eval_is_positive(self):
        n, z = self.args
        if n.is_positive:
            if n.is_odd and z.is_real:
                return True
            if n.is_even and z.is_positive:
                return False

    def _eval_is_negative(self):
        n, z = self.args
        if n.is_positive:
            if n.is_even and z.is_positive:
                return True
            if n.is_odd and z.is_real:
                return False

    def _eval_expand_func(self, **hints):
        n, z = self.args

        if n.is_Integer and n.is_nonnegative:
            if z.is_Add:
                coeff = z.args[0]
                if coeff.is_Integer:
                    e = -(n + 1)
                    if coeff > 0:
                        tail = Add(*[Pow(
                            z - i, e) for i in range(1, int(coeff) + 1)])
                    else:
                        tail = -Add(*[Pow(
                            z + i, e) for i in range(int(-coeff))])
                    return polygamma(n, z - coeff) + S.NegativeOne**n*factorial(n)*tail

            elif z.is_Mul:
                coeff, z = z.as_two_terms()
                if coeff.is_Integer and coeff.is_positive:
                    tail = [polygamma(n, z + Rational(
                        i, coeff)) for i in range(int(coeff))]
                    if n == 0:
                        return Add(*tail)/coeff + log(coeff)
                    else:
                        return Add(*tail)/coeff**(n + 1)
                z *= coeff

        if n == 0 and z.is_Rational:
            p, q = z.as_numer_denom()

            # Reference:
            #   Values of the polygamma functions at rational arguments, J. Choi, 2007
            part_1 = -S.EulerGamma - pi * cot(p * pi / q) / 2 - log(q) + Add(
                *[cos(2 * k * pi * p / q) * log(2 * sin(k * pi / q)) for k in range(1, q)])

            if z > 0:
                n = floor(z)
                z0 = z - n
                return part_1 + Add(*[1 / (z0 + k) for k in range(n)])
            elif z < 0:
                n = floor(1 - z)
                z0 = z + n
                return part_1 - Add(*[1 / (z0 - 1 - k) for k in range(n)])

        if n == -1:
            return loggamma(z) - log(2*pi) / 2
        if n.is_integer is False or n.is_nonnegative is False:
            s = Dummy("s")
            dzt = zeta(s, z).diff(s).subs(s, n+1)
            return (dzt + (S.EulerGamma + digamma(-n)) * zeta(n+1, z)) / gamma(-n)

        return polygamma(n, z)

    def _eval_rewrite_as_zeta(self, n, z, **kwargs):
        if n.is_integer and n.is_positive:
            return S.NegativeOne**(n + 1)*factorial(n)*zeta(n + 1, z)

    def _eval_rewrite_as_harmonic(self, n, z, **kwargs):
        if n.is_integer:
            if n.is_zero:
                return harmonic(z - 1) - S.EulerGamma
            else:
                return S.NegativeOne**(n+1) * factorial(n) * (zeta(n+1) - harmonic(z-1, n+1))

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        from sympy.series.order import Order
        n, z = [a.as_leading_term(x) for a in self.args]
        o = Order(z, x)
        if n == 0 and o.contains(1/x):
            logx = log(x) if logx is None else logx
            return o.getn() * logx
        else:
            return self.func(n, z)

    def fdiff(self, argindex=2):
        if argindex == 2:
            n, z = self.args[:2]
            return polygamma(n + 1, z)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        if args0[1] != oo or not \
                (self.args[0].is_Integer and self.args[0].is_nonnegative):
            return super()._eval_aseries(n, args0, x, logx)
        z = self.args[1]
        N = self.args[0]

        if N == 0:
            # digamma function series
            # Abramowitz & Stegun, p. 259, 6.3.18
            r = log(z) - 1/(2*z)
            o = None
            if n < 2:
                o = Order(1/z, x)
            else:
                m = ceiling((n + 1)//2)
                l = [bernoulli(2*k) / (2*k*z**(2*k)) for k in range(1, m)]
                r -= Add(*l)
                o = Order(1/z**n, x)
            return r._eval_nseries(x, n, logx) + o
        else:
            # proper polygamma function
            # Abramowitz & Stegun, p. 260, 6.4.10
            # We return terms to order higher than O(x**n) on purpose
            # -- otherwise we would not be able to return any terms for
            #    quite a long time!
            fac = gamma(N)
            e0 = fac + N*fac/(2*z)
            m = ceiling((n + 1)//2)
            for k in range(1, m):
                fac = fac*(2*k + N - 1)*(2*k + N - 2) / ((2*k)*(2*k - 1))
                e0 += bernoulli(2*k)*fac/z**(2*k)
            o = Order(1/z**(2*m), x)
            if n == 0:
                o = Order(1/z, x)
            elif n == 1:
                o = Order(1/z**2, x)
            r = e0._eval_nseries(z, n, logx) + o
            return (-1 * (-1/z)**N * r)._eval_nseries(x, n, logx)

    def _eval_evalf(self, prec):
        if not all(i.is_number for i in self.args):
            return
        s = self.args[0]._to_mpmath(prec+12)
        z = self.args[1]._to_mpmath(prec+12)
        if mp.isint(z) and z <= 0:
            return S.ComplexInfinity
        with workprec(prec+12):
            if mp.isint(s) and s >= 0:
                res = mp.polygamma(s, z)
            else:
                zt = mp.zeta(s+1, z)
                dzt = mp.zeta(s+1, z, 1)
                res = (dzt + (mp.euler + mp.digamma(-s)) * zt) * mp.rgamma(-s)
        return Expr._from_mpmath(res, prec)


class loggamma(Function):
    r"""
    The ``loggamma`` function implements the logarithm of the
    gamma function (i.e., $\log\Gamma(x)$).

    Examples
    ========

    Several special values are known. For numerical integral
    arguments we have:

    >>> from sympy import loggamma
    >>> loggamma(-2)
    oo
    >>> loggamma(0)
    oo
    >>> loggamma(1)
    0
    >>> loggamma(2)
    0
    >>> loggamma(3)
    log(2)

    And for symbolic values:

    >>> from sympy import Symbol
    >>> n = Symbol("n", integer=True, positive=True)
    >>> loggamma(n)
    log(gamma(n))
    >>> loggamma(-n)
    oo

    For half-integral values:

    >>> from sympy import S
    >>> loggamma(S(5)/2)
    log(3*sqrt(pi)/4)
    >>> loggamma(n/2)
    log(2**(1 - n)*sqrt(pi)*gamma(n)/gamma(n/2 + 1/2))

    And general rational arguments:

    >>> from sympy import expand_func
    >>> L = loggamma(S(16)/3)
    >>> expand_func(L).doit()
    -5*log(3) + loggamma(1/3) + log(4) + log(7) + log(10) + log(13)
    >>> L = loggamma(S(19)/4)
    >>> expand_func(L).doit()
    -4*log(4) + loggamma(3/4) + log(3) + log(7) + log(11) + log(15)
    >>> L = loggamma(S(23)/7)
    >>> expand_func(L).doit()
    -3*log(7) + log(2) + loggamma(2/7) + log(9) + log(16)

    The ``loggamma`` function has the following limits towards infinity:

    >>> from sympy import oo
    >>> loggamma(oo)
    oo
    >>> loggamma(-oo)
    zoo

    The ``loggamma`` function obeys the mirror symmetry
    if $x \in \mathbb{C} \setminus \{-\infty, 0\}$:

    >>> from sympy.abc import x
    >>> from sympy import conjugate
    >>> conjugate(loggamma(x))
    loggamma(conjugate(x))

    Differentiation with respect to $x$ is supported:

    >>> from sympy import diff
    >>> diff(loggamma(x), x)
    polygamma(0, x)

    Series expansion is also supported:

    >>> from sympy import series
    >>> series(loggamma(x), x, 0, 4).cancel()
    -log(x) - EulerGamma*x + pi**2*x**2/12 - x**3*zeta(3)/3 + O(x**4)

    We can numerically evaluate the ``loggamma`` function
    to arbitrary precision on the whole complex plane:

    >>> from sympy import I
    >>> loggamma(5).evalf(30)
    3.17805383034794561964694160130
    >>> loggamma(I).evalf(20)
    -0.65092319930185633889 - 1.8724366472624298171*I

    See Also
    ========

    gamma: Gamma function.
    lowergamma: Lower incomplete gamma function.
    uppergamma: Upper incomplete gamma function.
    polygamma: Polygamma function.
    digamma: Digamma function.
    trigamma: Trigamma function.
    sympy.functions.special.beta_functions.beta: Euler Beta function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gamma_function
    .. [2] https://dlmf.nist.gov/5
    .. [3] https://mathworld.wolfram.com/LogGammaFunction.html
    .. [4] https://functions.wolfram.com/GammaBetaErf/LogGamma/

    """
    @classmethod
    def eval(cls, z):
        if z.is_integer:
            if z.is_nonpositive:
                return oo
            elif z.is_positive:
                return log(gamma(z))
        elif z.is_rational:
            p, q = z.as_numer_denom()
            # Half-integral values:
            if p.is_positive and q == 2:
                return log(sqrt(pi) * 2**(1 - p) * gamma(p) / gamma((p + 1)*S.Half))

        if z is oo:
            return oo
        elif abs(z) is oo:
            return S.ComplexInfinity
        if z is S.NaN:
            return S.NaN

    def _eval_expand_func(self, **hints):
        from sympy.concrete.summations import Sum
        z = self.args[0]

        if z.is_Rational:
            p, q = z.as_numer_denom()
            # General rational arguments (u + p/q)
            # Split z as n + p/q with p < q
            n = p // q
            p = p - n*q
            if p.is_positive and q.is_positive and p < q:
                k = Dummy("k")
                if n.is_positive:
                    return loggamma(p / q) - n*log(q) + Sum(log((k - 1)*q + p), (k, 1, n))
                elif n.is_negative:
                    return loggamma(p / q) - n*log(q) + pi*I*n - Sum(log(k*q - p), (k, 1, -n))
                elif n.is_zero:
                    return loggamma(p / q)

        return self

    def _eval_nseries(self, x, n, logx=None, cdir=0):
        x0 = self.args[0].limit(x, 0)
        if x0.is_zero:
            f = self._eval_rewrite_as_intractable(*self.args)
            return f._eval_nseries(x, n, logx)
        return super()._eval_nseries(x, n, logx)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        if args0[0] != oo:
            return super()._eval_aseries(n, args0, x, logx)
        z = self.args[0]
        r = log(z)*(z - S.Half) - z + log(2*pi)/2
        l = [bernoulli(2*k) / (2*k*(2*k - 1)*z**(2*k - 1)) for k in range(1, n)]
        o = None
        if n == 0:
            o = Order(1, x)
        else:
            o = Order(1/z**n, x)
        # It is very inefficient to first add the order and then do the nseries
        return (r + Add(*l))._eval_nseries(x, n, logx) + o

    def _eval_rewrite_as_intractable(self, z, **kwargs):
        return log(gamma(z))

    def _eval_is_real(self):
        z = self.args[0]
        if z.is_positive:
            return True
        elif z.is_nonpositive:
            return False

    def _eval_conjugate(self):
        z = self.args[0]
        if z not in (S.Zero, S.NegativeInfinity):
            return self.func(z.conjugate())

    def fdiff(self, argindex=1):
        if argindex == 1:
            return polygamma(0, self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)


class digamma(Function):
    r"""
    The ``digamma`` function is the first derivative of the ``loggamma``
    function

    .. math::
        \psi(x) := \frac{\mathrm{d}}{\mathrm{d} z} \log\Gamma(z)
                = \frac{\Gamma'(z)}{\Gamma(z) }.

    In this case, ``digamma(z) = polygamma(0, z)``.

    Examples
    ========

    >>> from sympy import digamma
    >>> digamma(0)
    zoo
    >>> from sympy import Symbol
    >>> z = Symbol('z')
    >>> digamma(z)
    polygamma(0, z)

    To retain ``digamma`` as it is:

    >>> digamma(0, evaluate=False)
    digamma(0)
    >>> digamma(z, evaluate=False)
    digamma(z)

    See Also
    ========

    gamma: Gamma function.
    lowergamma: Lower incomplete gamma function.
    uppergamma: Upper incomplete gamma function.
    polygamma: Polygamma function.
    loggamma: Log Gamma function.
    trigamma: Trigamma function.
    sympy.functions.special.beta_functions.beta: Euler Beta function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Digamma_function
    .. [2] https://mathworld.wolfram.com/DigammaFunction.html
    .. [3] https://functions.wolfram.com/GammaBetaErf/PolyGamma2/

    """
    def _eval_evalf(self, prec):
        z = self.args[0]
        nprec = prec_to_dps(prec)
        return polygamma(0, z).evalf(n=nprec)

    def fdiff(self, argindex=1):
        z = self.args[0]
        return polygamma(0, z).fdiff()

    def _eval_is_real(self):
        z = self.args[0]
        return polygamma(0, z).is_real

    def _eval_is_positive(self):
        z = self.args[0]
        return polygamma(0, z).is_positive

    def _eval_is_negative(self):
        z = self.args[0]
        return polygamma(0, z).is_negative

    def _eval_aseries(self, n, args0, x, logx):
        as_polygamma = self.rewrite(polygamma)
        args0 = [S.Zero,] + args0
        return as_polygamma._eval_aseries(n, args0, x, logx)

    @classmethod
    def eval(cls, z):
        return polygamma(0, z)

    def _eval_expand_func(self, **hints):
        z = self.args[0]
        return polygamma(0, z).expand(func=True)

    def _eval_rewrite_as_harmonic(self, z, **kwargs):
        return harmonic(z - 1) - S.EulerGamma

    def _eval_rewrite_as_polygamma(self, z, **kwargs):
        return polygamma(0, z)

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        z = self.args[0]
        return polygamma(0, z).as_leading_term(x)



class trigamma(Function):
    r"""
    The ``trigamma`` function is the second derivative of the ``loggamma``
    function

    .. math::
        \psi^{(1)}(z) := \frac{\mathrm{d}^{2}}{\mathrm{d} z^{2}} \log\Gamma(z).

    In this case, ``trigamma(z) = polygamma(1, z)``.

    Examples
    ========

    >>> from sympy import trigamma
    >>> trigamma(0)
    zoo
    >>> from sympy import Symbol
    >>> z = Symbol('z')
    >>> trigamma(z)
    polygamma(1, z)

    To retain ``trigamma`` as it is:

    >>> trigamma(0, evaluate=False)
    trigamma(0)
    >>> trigamma(z, evaluate=False)
    trigamma(z)


    See Also
    ========

    gamma: Gamma function.
    lowergamma: Lower incomplete gamma function.
    uppergamma: Upper incomplete gamma function.
    polygamma: Polygamma function.
    loggamma: Log Gamma function.
    digamma: Digamma function.
    sympy.functions.special.beta_functions.beta: Euler Beta function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigamma_function
    .. [2] https://mathworld.wolfram.com/TrigammaFunction.html
    .. [3] https://functions.wolfram.com/GammaBetaErf/PolyGamma2/

    """
    def _eval_evalf(self, prec):
        z = self.args[0]
        nprec = prec_to_dps(prec)
        return polygamma(1, z).evalf(n=nprec)

    def fdiff(self, argindex=1):
        z = self.args[0]
        return polygamma(1, z).fdiff()

    def _eval_is_real(self):
        z = self.args[0]
        return polygamma(1, z).is_real

    def _eval_is_positive(self):
        z = self.args[0]
        return polygamma(1, z).is_positive

    def _eval_is_negative(self):
        z = self.args[0]
        return polygamma(1, z).is_negative

    def _eval_aseries(self, n, args0, x, logx):
        as_polygamma = self.rewrite(polygamma)
        args0 = [S.One,] + args0
        return as_polygamma._eval_aseries(n, args0, x, logx)

    @classmethod
    def eval(cls, z):
        return polygamma(1, z)

    def _eval_expand_func(self, **hints):
        z = self.args[0]
        return polygamma(1, z).expand(func=True)

    def _eval_rewrite_as_zeta(self, z, **kwargs):
        return zeta(2, z)

    def _eval_rewrite_as_polygamma(self, z, **kwargs):
        return polygamma(1, z)

    def _eval_rewrite_as_harmonic(self, z, **kwargs):
        return -harmonic(z - 1, 2) + pi**2 / 6

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        z = self.args[0]
        return polygamma(1, z).as_leading_term(x)


###############################################################################
##################### COMPLETE MULTIVARIATE GAMMA FUNCTION ####################
###############################################################################


class multigamma(Function):
    r"""
    The multivariate gamma function is a generalization of the gamma function

    .. math::
        \Gamma_p(z) = \pi^{p(p-1)/4}\prod_{k=1}^p \Gamma[z + (1 - k)/2].

    In a special case, ``multigamma(x, 1) = gamma(x)``.

    Examples
    ========

    >>> from sympy import S, multigamma
    >>> from sympy import Symbol
    >>> x = Symbol('x')
    >>> p = Symbol('p', positive=True, integer=True)

    >>> multigamma(x, p)
    pi**(p*(p - 1)/4)*Product(gamma(-_k/2 + x + 1/2), (_k, 1, p))

    Several special values are known:

    >>> multigamma(1, 1)
    1
    >>> multigamma(4, 1)
    6
    >>> multigamma(S(3)/2, 1)
    sqrt(pi)/2

    Writing ``multigamma`` in terms of the ``gamma`` function:

    >>> multigamma(x, 1)
    gamma(x)

    >>> multigamma(x, 2)
    sqrt(pi)*gamma(x)*gamma(x - 1/2)

    >>> multigamma(x, 3)
    pi**(3/2)*gamma(x)*gamma(x - 1)*gamma(x - 1/2)

    Parameters
    ==========

    p : order or dimension of the multivariate gamma function

    See Also
    ========

    gamma, lowergamma, uppergamma, polygamma, loggamma, digamma, trigamma,
    sympy.functions.special.beta_functions.beta

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Multivariate_gamma_function

    """
    unbranched = True

    def fdiff(self, argindex=2):
        from sympy.concrete.summations import Sum
        if argindex == 2:
            x, p = self.args
            k = Dummy("k")
            return self.func(x, p)*Sum(polygamma(0, x + (1 - k)/2), (k, 1, p))
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, x, p):
        from sympy.concrete.products import Product
        if p.is_positive is False or p.is_integer is False:
            raise ValueError('Order parameter p must be positive integer.')
        k = Dummy("k")
        return (pi**(p*(p - 1)/4)*Product(gamma(x + (1 - k)/2),
                                          (k, 1, p))).doit()

    def _eval_conjugate(self):
        x, p = self.args
        return self.func(x.conjugate(), p)

    def _eval_is_real(self):
        x, p = self.args
        y = 2*x
        if y.is_integer and (y <= (p - 1)) is True:
            return False
        if intlike(y) and (y <= (p - 1)):
            return False
        if y > (p - 1) or y.is_noninteger:
            return True
