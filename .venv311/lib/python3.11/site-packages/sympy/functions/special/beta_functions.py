from sympy.core import S
from sympy.core.function import DefinedFunction, ArgumentIndexError
from sympy.core.symbol import Dummy, uniquely_named_symbol
from sympy.functions.special.gamma_functions import gamma, digamma
from sympy.functions.combinatorial.numbers import catalan
from sympy.functions.elementary.complexes import conjugate

# See mpmath #569 and SymPy #20569
def betainc_mpmath_fix(a, b, x1, x2, reg=0):
    from mpmath import betainc, mpf
    if x1 == x2:
        return mpf(0)
    else:
        return betainc(a, b, x1, x2, reg)

###############################################################################
############################ COMPLETE BETA  FUNCTION ##########################
###############################################################################

class beta(DefinedFunction):
    r"""
    The beta integral is called the Eulerian integral of the first kind by
    Legendre:

    .. math::
        \mathrm{B}(x,y)  \int^{1}_{0} t^{x-1} (1-t)^{y-1} \mathrm{d}t.

    Explanation
    ===========

    The Beta function or Euler's first integral is closely associated
    with the gamma function. The Beta function is often used in probability
    theory and mathematical statistics. It satisfies properties like:

    .. math::
        \mathrm{B}(a,1) = \frac{1}{a} \\
        \mathrm{B}(a,b) = \mathrm{B}(b,a)  \\
        \mathrm{B}(a,b) = \frac{\Gamma(a) \Gamma(b)}{\Gamma(a+b)}

    Therefore for integral values of $a$ and $b$:

    .. math::
        \mathrm{B} = \frac{(a-1)! (b-1)!}{(a+b-1)!}

    A special case of the Beta function when `x = y` is the
    Central Beta function. It satisfies properties like:

    .. math::
        \mathrm{B}(x) = 2^{1 - 2x}\mathrm{B}(x, \frac{1}{2})
        \mathrm{B}(x) = 2^{1 - 2x} cos(\pi x) \mathrm{B}(\frac{1}{2} - x, x)
        \mathrm{B}(x) = \int_{0}^{1} \frac{t^x}{(1 + t)^{2x}} dt
        \mathrm{B}(x) = \frac{2}{x} \prod_{n = 1}^{\infty} \frac{n(n + 2x)}{(n + x)^2}

    Examples
    ========

    >>> from sympy import I, pi
    >>> from sympy.abc import x, y

    The Beta function obeys the mirror symmetry:

    >>> from sympy import beta, conjugate
    >>> conjugate(beta(x, y))
    beta(conjugate(x), conjugate(y))

    Differentiation with respect to both $x$ and $y$ is supported:

    >>> from sympy import beta, diff
    >>> diff(beta(x, y), x)
    (polygamma(0, x) - polygamma(0, x + y))*beta(x, y)

    >>> diff(beta(x, y), y)
    (polygamma(0, y) - polygamma(0, x + y))*beta(x, y)

    >>> diff(beta(x), x)
    2*(polygamma(0, x) - polygamma(0, 2*x))*beta(x, x)

    We can numerically evaluate the Beta function to
    arbitrary precision for any complex numbers x and y:

    >>> from sympy import beta
    >>> beta(pi).evalf(40)
    0.02671848900111377452242355235388489324562

    >>> beta(1 + I).evalf(20)
    -0.2112723729365330143 - 0.7655283165378005676*I

    See Also
    ========

    gamma: Gamma function.
    uppergamma: Upper incomplete gamma function.
    lowergamma: Lower incomplete gamma function.
    polygamma: Polygamma function.
    loggamma: Log Gamma function.
    digamma: Digamma function.
    trigamma: Trigamma function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Beta_function
    .. [2] https://mathworld.wolfram.com/BetaFunction.html
    .. [3] https://dlmf.nist.gov/5.12

    """
    unbranched = True

    def fdiff(self, argindex):
        x, y = self.args
        if argindex == 1:
            # Diff wrt x
            return beta(x, y)*(digamma(x) - digamma(x + y))
        elif argindex == 2:
            # Diff wrt y
            return beta(x, y)*(digamma(y) - digamma(x + y))
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, x, y=None):
        if y is None:
            return beta(x, x)
        if x.is_Number and y.is_Number:
            return beta(x, y, evaluate=False).doit()

    def doit(self, **hints):
        x = xold = self.args[0]
        # Deal with unevaluated single argument beta
        single_argument = len(self.args) == 1
        y = yold = self.args[0] if single_argument else self.args[1]
        if hints.get('deep', True):
            x = x.doit(**hints)
            y = y.doit(**hints)
        if y.is_zero or x.is_zero:
            return S.ComplexInfinity
        if y is S.One:
            return 1/x
        if x is S.One:
            return 1/y
        if y == x + 1:
            return 1/(x*y*catalan(x))
        s = x + y
        if (s.is_integer and s.is_negative and x.is_integer is False and
            y.is_integer is False):
            return S.Zero
        if x == xold and y == yold and not single_argument:
            return self
        return beta(x, y)

    def _eval_expand_func(self, **hints):
        x, y = self.args
        return gamma(x)*gamma(y) / gamma(x + y)

    def _eval_is_real(self):
        return self.args[0].is_real and self.args[1].is_real

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate(), self.args[1].conjugate())

    def _eval_rewrite_as_gamma(self, x, y, piecewise=True, **kwargs):
        return self._eval_expand_func(**kwargs)

    def _eval_rewrite_as_Integral(self, x, y, **kwargs):
        from sympy.integrals.integrals import Integral
        t = Dummy(uniquely_named_symbol('t', [x, y]).name)
        return Integral(t**(x - 1)*(1 - t)**(y - 1), (t, 0, 1))

###############################################################################
########################## INCOMPLETE BETA FUNCTION ###########################
###############################################################################

class betainc(DefinedFunction):
    r"""
    The Generalized Incomplete Beta function is defined as

    .. math::
        \mathrm{B}_{(x_1, x_2)}(a, b) = \int_{x_1}^{x_2} t^{a - 1} (1 - t)^{b - 1} dt

    The Incomplete Beta function is a special case
    of the Generalized Incomplete Beta function :

    .. math:: \mathrm{B}_z (a, b) = \mathrm{B}_{(0, z)}(a, b)

    The Incomplete Beta function satisfies :

    .. math:: \mathrm{B}_z (a, b) = (-1)^a \mathrm{B}_{\frac{z}{z - 1}} (a, 1 - a - b)

    The Beta function is a special case of the Incomplete Beta function :

    .. math:: \mathrm{B}(a, b) = \mathrm{B}_{1}(a, b)

    Examples
    ========

    >>> from sympy import betainc, symbols, conjugate
    >>> a, b, x, x1, x2 = symbols('a b x x1 x2')

    The Generalized Incomplete Beta function is given by:

    >>> betainc(a, b, x1, x2)
    betainc(a, b, x1, x2)

    The Incomplete Beta function can be obtained as follows:

    >>> betainc(a, b, 0, x)
    betainc(a, b, 0, x)

    The Incomplete Beta function obeys the mirror symmetry:

    >>> conjugate(betainc(a, b, x1, x2))
    betainc(conjugate(a), conjugate(b), conjugate(x1), conjugate(x2))

    We can numerically evaluate the Incomplete Beta function to
    arbitrary precision for any complex numbers a, b, x1 and x2:

    >>> from sympy import betainc, I
    >>> betainc(2, 3, 4, 5).evalf(10)
    56.08333333
    >>> betainc(0.75, 1 - 4*I, 0, 2 + 3*I).evalf(25)
    0.2241657956955709603655887 + 0.3619619242700451992411724*I

    The Generalized Incomplete Beta function can be expressed
    in terms of the Generalized Hypergeometric function.

    >>> from sympy import hyper
    >>> betainc(a, b, x1, x2).rewrite(hyper)
    (-x1**a*hyper((a, 1 - b), (a + 1,), x1) + x2**a*hyper((a, 1 - b), (a + 1,), x2))/a

    See Also
    ========

    beta: Beta function
    hyper: Generalized Hypergeometric function

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function
    .. [2] https://dlmf.nist.gov/8.17
    .. [3] https://functions.wolfram.com/GammaBetaErf/Beta4/
    .. [4] https://functions.wolfram.com/GammaBetaErf/BetaRegularized4/02/

    """
    nargs = 4
    unbranched = True

    def fdiff(self, argindex):
        a, b, x1, x2 = self.args
        if argindex == 3:
            # Diff wrt x1
            return -(1 - x1)**(b - 1)*x1**(a - 1)
        elif argindex == 4:
            # Diff wrt x2
            return (1 - x2)**(b - 1)*x2**(a - 1)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_mpmath(self):
        return betainc_mpmath_fix, self.args

    def _eval_is_real(self):
        if all(arg.is_real for arg in self.args):
            return True

    def _eval_conjugate(self):
        return self.func(*map(conjugate, self.args))

    def _eval_rewrite_as_Integral(self, a, b, x1, x2, **kwargs):
        from sympy.integrals.integrals import Integral
        t = Dummy(uniquely_named_symbol('t', [a, b, x1, x2]).name)
        return Integral(t**(a - 1)*(1 - t)**(b - 1), (t, x1, x2))

    def _eval_rewrite_as_hyper(self, a, b, x1, x2, **kwargs):
        from sympy.functions.special.hyper import hyper
        return (x2**a * hyper((a, 1 - b), (a + 1,), x2) - x1**a * hyper((a, 1 - b), (a + 1,), x1)) / a

###############################################################################
#################### REGULARIZED INCOMPLETE BETA FUNCTION #####################
###############################################################################

class betainc_regularized(DefinedFunction):
    r"""
    The Generalized Regularized Incomplete Beta function is given by

    .. math::
        \mathrm{I}_{(x_1, x_2)}(a, b) = \frac{\mathrm{B}_{(x_1, x_2)}(a, b)}{\mathrm{B}(a, b)}

    The Regularized Incomplete Beta function is a special case
    of the Generalized Regularized Incomplete Beta function :

    .. math:: \mathrm{I}_z (a, b) = \mathrm{I}_{(0, z)}(a, b)

    The Regularized Incomplete Beta function is the cumulative distribution
    function of the beta distribution.

    Examples
    ========

    >>> from sympy import betainc_regularized, symbols, conjugate
    >>> a, b, x, x1, x2 = symbols('a b x x1 x2')

    The Generalized Regularized Incomplete Beta
    function is given by:

    >>> betainc_regularized(a, b, x1, x2)
    betainc_regularized(a, b, x1, x2)

    The Regularized Incomplete Beta function
    can be obtained as follows:

    >>> betainc_regularized(a, b, 0, x)
    betainc_regularized(a, b, 0, x)

    The Regularized Incomplete Beta function
    obeys the mirror symmetry:

    >>> conjugate(betainc_regularized(a, b, x1, x2))
    betainc_regularized(conjugate(a), conjugate(b), conjugate(x1), conjugate(x2))

    We can numerically evaluate the Regularized Incomplete Beta function
    to arbitrary precision for any complex numbers a, b, x1 and x2:

    >>> from sympy import betainc_regularized, pi, E
    >>> betainc_regularized(1, 2, 0, 0.25).evalf(10)
    0.4375000000
    >>> betainc_regularized(pi, E, 0, 1).evalf(5)
    1.00000

    The Generalized Regularized Incomplete Beta function can be
    expressed in terms of the Generalized Hypergeometric function.

    >>> from sympy import hyper
    >>> betainc_regularized(a, b, x1, x2).rewrite(hyper)
    (-x1**a*hyper((a, 1 - b), (a + 1,), x1) + x2**a*hyper((a, 1 - b), (a + 1,), x2))/(a*beta(a, b))

    See Also
    ========

    beta: Beta function
    hyper: Generalized Hypergeometric function

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function
    .. [2] https://dlmf.nist.gov/8.17
    .. [3] https://functions.wolfram.com/GammaBetaErf/Beta4/
    .. [4] https://functions.wolfram.com/GammaBetaErf/BetaRegularized4/02/

    """
    nargs = 4
    unbranched = True

    def __new__(cls, a, b, x1, x2):
        return super().__new__(cls, a, b, x1, x2)

    def _eval_mpmath(self):
        return betainc_mpmath_fix, (*self.args, S(1))

    def fdiff(self, argindex):
        a, b, x1, x2 = self.args
        if argindex == 3:
            # Diff wrt x1
            return -(1 - x1)**(b - 1)*x1**(a - 1) / beta(a, b)
        elif argindex == 4:
            # Diff wrt x2
            return (1 - x2)**(b - 1)*x2**(a - 1) / beta(a, b)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_is_real(self):
        if all(arg.is_real for arg in self.args):
            return True

    def _eval_conjugate(self):
        return self.func(*map(conjugate, self.args))

    def _eval_rewrite_as_Integral(self, a, b, x1, x2, **kwargs):
        from sympy.integrals.integrals import Integral
        t = Dummy(uniquely_named_symbol('t', [a, b, x1, x2]).name)
        integrand = t**(a - 1)*(1 - t)**(b - 1)
        expr = Integral(integrand, (t, x1, x2))
        return expr / Integral(integrand, (t, 0, 1))

    def _eval_rewrite_as_hyper(self, a, b, x1, x2, **kwargs):
        from sympy.functions.special.hyper import hyper
        expr = (x2**a * hyper((a, 1 - b), (a + 1,), x2) - x1**a * hyper((a, 1 - b), (a + 1,), x1)) / a
        return expr / beta(a, b)
