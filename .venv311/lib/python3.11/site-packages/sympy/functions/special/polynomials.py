"""
This module mainly implements special orthogonal polynomials.

See also functions.combinatorial.numbers which contains some
combinatorial polynomials.

"""

from sympy.core import Rational
from sympy.core.function import DefinedFunction, ArgumentIndexError
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.functions.combinatorial.factorials import binomial, factorial, RisingFactorial
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sec
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper
from sympy.polys.orthopolys import (chebyshevt_poly, chebyshevu_poly,
                                    gegenbauer_poly, hermite_poly, hermite_prob_poly,
                                    jacobi_poly, laguerre_poly, legendre_poly)

_x = Dummy('x')


class OrthogonalPolynomial(DefinedFunction):
    """Base class for orthogonal polynomials.
    """

    @classmethod
    def _eval_at_order(cls, n, x):
        if n.is_integer and n >= 0:
            return cls._ortho_poly(int(n), _x).subs(_x, x)

    def _eval_conjugate(self):
        return self.func(self.args[0], self.args[1].conjugate())

#----------------------------------------------------------------------------
# Jacobi polynomials
#


class jacobi(OrthogonalPolynomial):
    r"""
    Jacobi polynomial $P_n^{\left(\alpha, \beta\right)}(x)$.

    Explanation
    ===========

    ``jacobi(n, alpha, beta, x)`` gives the $n$th Jacobi polynomial
    in $x$, $P_n^{\left(\alpha, \beta\right)}(x)$.

    The Jacobi polynomials are orthogonal on $[-1, 1]$ with respect
    to the weight $\left(1-x\right)^\alpha \left(1+x\right)^\beta$.

    Examples
    ========

    >>> from sympy import jacobi, S, conjugate, diff
    >>> from sympy.abc import a, b, n, x

    >>> jacobi(0, a, b, x)
    1
    >>> jacobi(1, a, b, x)
    a/2 - b/2 + x*(a/2 + b/2 + 1)
    >>> jacobi(2, a, b, x)
    a**2/8 - a*b/4 - a/8 + b**2/8 - b/8 + x**2*(a**2/8 + a*b/4 + 7*a/8 + b**2/8 + 7*b/8 + 3/2) + x*(a**2/4 + 3*a/4 - b**2/4 - 3*b/4) - 1/2

    >>> jacobi(n, a, b, x)
    jacobi(n, a, b, x)

    >>> jacobi(n, a, a, x)
    RisingFactorial(a + 1, n)*gegenbauer(n,
        a + 1/2, x)/RisingFactorial(2*a + 1, n)

    >>> jacobi(n, 0, 0, x)
    legendre(n, x)

    >>> jacobi(n, S(1)/2, S(1)/2, x)
    RisingFactorial(3/2, n)*chebyshevu(n, x)/factorial(n + 1)

    >>> jacobi(n, -S(1)/2, -S(1)/2, x)
    RisingFactorial(1/2, n)*chebyshevt(n, x)/factorial(n)

    >>> jacobi(n, a, b, -x)
    (-1)**n*jacobi(n, b, a, x)

    >>> jacobi(n, a, b, 0)
    gamma(a + n + 1)*hyper((-n, -b - n), (a + 1,), -1)/(2**n*factorial(n)*gamma(a + 1))
    >>> jacobi(n, a, b, 1)
    RisingFactorial(a + 1, n)/factorial(n)

    >>> conjugate(jacobi(n, a, b, x))
    jacobi(n, conjugate(a), conjugate(b), conjugate(x))

    >>> diff(jacobi(n,a,b,x), x)
    (a/2 + b/2 + n/2 + 1/2)*jacobi(n - 1, a + 1, b + 1, x)

    See Also
    ========

    gegenbauer,
    chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly,
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Jacobi_polynomials
    .. [2] https://mathworld.wolfram.com/JacobiPolynomial.html
    .. [3] https://functions.wolfram.com/Polynomials/JacobiP/

    """

    @classmethod
    def eval(cls, n, a, b, x):
        # Simplify to other polynomials
        # P^{a, a}_n(x)
        if a == b:
            if a == Rational(-1, 2):
                return RisingFactorial(S.Half, n) / factorial(n) * chebyshevt(n, x)
            elif a.is_zero:
                return legendre(n, x)
            elif a == S.Half:
                return RisingFactorial(3*S.Half, n) / factorial(n + 1) * chebyshevu(n, x)
            else:
                return RisingFactorial(a + 1, n) / RisingFactorial(2*a + 1, n) * gegenbauer(n, a + S.Half, x)
        elif b == -a:
            # P^{a, -a}_n(x)
            return gamma(n + a + 1) / gamma(n + 1) * (1 + x)**(a/2) / (1 - x)**(a/2) * assoc_legendre(n, -a, x)

        if not n.is_Number:
            # Symbolic result P^{a,b}_n(x)
            # P^{a,b}_n(-x)  --->  (-1)**n * P^{b,a}_n(-x)
            if x.could_extract_minus_sign():
                return S.NegativeOne**n * jacobi(n, b, a, -x)
            # We can evaluate for some special values of x
            if x.is_zero:
                return (2**(-n) * gamma(a + n + 1) / (gamma(a + 1) * factorial(n)) *
                        hyper([-b - n, -n], [a + 1], -1))
            if x == S.One:
                return RisingFactorial(a + 1, n) / factorial(n)
            elif x is S.Infinity:
                if n.is_positive:
                    # Make sure a+b+2*n \notin Z
                    if (a + b + 2*n).is_integer:
                        raise ValueError("Error. a + b + 2*n should not be an integer.")
                    return RisingFactorial(a + b + n + 1, n) * S.Infinity
        else:
            # n is a given fixed integer, evaluate into polynomial
            return jacobi_poly(n, a, b, x)

    def fdiff(self, argindex=4):
        from sympy.concrete.summations import Sum
        if argindex == 1:
            # Diff wrt n
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # Diff wrt a
            n, a, b, x = self.args
            k = Dummy("k")
            f1 = 1 / (a + b + n + k + 1)
            f2 = ((a + b + 2*k + 1) * RisingFactorial(b + k + 1, n - k) /
                  ((n - k) * RisingFactorial(a + b + k + 1, n - k)))
            return Sum(f1 * (jacobi(n, a, b, x) + f2*jacobi(k, a, b, x)), (k, 0, n - 1))
        elif argindex == 3:
            # Diff wrt b
            n, a, b, x = self.args
            k = Dummy("k")
            f1 = 1 / (a + b + n + k + 1)
            f2 = (-1)**(n - k) * ((a + b + 2*k + 1) * RisingFactorial(a + k + 1, n - k) /
                  ((n - k) * RisingFactorial(a + b + k + 1, n - k)))
            return Sum(f1 * (jacobi(n, a, b, x) + f2*jacobi(k, a, b, x)), (k, 0, n - 1))
        elif argindex == 4:
            # Diff wrt x
            n, a, b, x = self.args
            return S.Half * (a + b + n + 1) * jacobi(n - 1, a + 1, b + 1, x)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, a, b, x, **kwargs):
        from sympy.concrete.summations import Sum
        # Make sure n \in N
        if n.is_negative or n.is_integer is False:
            raise ValueError("Error: n should be a non-negative integer.")
        k = Dummy("k")
        kern = (RisingFactorial(-n, k) * RisingFactorial(a + b + n + 1, k) * RisingFactorial(a + k + 1, n - k) /
                factorial(k) * ((1 - x)/2)**k)
        return 1 / factorial(n) * Sum(kern, (k, 0, n))

    def _eval_rewrite_as_polynomial(self, n, a, b, x, **kwargs):
        # This function is just kept for backwards compatibility
        # but should not be used
        return self._eval_rewrite_as_Sum(n, a, b, x, **kwargs)

    def _eval_conjugate(self):
        n, a, b, x = self.args
        return self.func(n, a.conjugate(), b.conjugate(), x.conjugate())


def jacobi_normalized(n, a, b, x):
    r"""
    Jacobi polynomial $P_n^{\left(\alpha, \beta\right)}(x)$.

    Explanation
    ===========

    ``jacobi_normalized(n, alpha, beta, x)`` gives the $n$th
    Jacobi polynomial in $x$, $P_n^{\left(\alpha, \beta\right)}(x)$.

    The Jacobi polynomials are orthogonal on $[-1, 1]$ with respect
    to the weight $\left(1-x\right)^\alpha \left(1+x\right)^\beta$.

    This functions returns the polynomials normilzed:

    .. math::

        \int_{-1}^{1}
          P_m^{\left(\alpha, \beta\right)}(x)
          P_n^{\left(\alpha, \beta\right)}(x)
          (1-x)^{\alpha} (1+x)^{\beta} \mathrm{d}x
        = \delta_{m,n}

    Examples
    ========

    >>> from sympy import jacobi_normalized
    >>> from sympy.abc import n,a,b,x

    >>> jacobi_normalized(n, a, b, x)
    jacobi(n, a, b, x)/sqrt(2**(a + b + 1)*gamma(a + n + 1)*gamma(b + n + 1)/((a + b + 2*n + 1)*factorial(n)*gamma(a + b + n + 1)))

    Parameters
    ==========

    n : integer degree of polynomial

    a : alpha value

    b : beta value

    x : symbol

    See Also
    ========

    gegenbauer,
    chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly,
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Jacobi_polynomials
    .. [2] https://mathworld.wolfram.com/JacobiPolynomial.html
    .. [3] https://functions.wolfram.com/Polynomials/JacobiP/

    """
    nfactor = (S(2)**(a + b + 1) * (gamma(n + a + 1) * gamma(n + b + 1))
               / (2*n + a + b + 1) / (factorial(n) * gamma(n + a + b + 1)))

    return jacobi(n, a, b, x) / sqrt(nfactor)


#----------------------------------------------------------------------------
# Gegenbauer polynomials
#


class gegenbauer(OrthogonalPolynomial):
    r"""
    Gegenbauer polynomial $C_n^{\left(\alpha\right)}(x)$.

    Explanation
    ===========

    ``gegenbauer(n, alpha, x)`` gives the $n$th Gegenbauer polynomial
    in $x$, $C_n^{\left(\alpha\right)}(x)$.

    The Gegenbauer polynomials are orthogonal on $[-1, 1]$ with
    respect to the weight $\left(1-x^2\right)^{\alpha-\frac{1}{2}}$.

    Examples
    ========

    >>> from sympy import gegenbauer, conjugate, diff
    >>> from sympy.abc import n,a,x
    >>> gegenbauer(0, a, x)
    1
    >>> gegenbauer(1, a, x)
    2*a*x
    >>> gegenbauer(2, a, x)
    -a + x**2*(2*a**2 + 2*a)
    >>> gegenbauer(3, a, x)
    x**3*(4*a**3/3 + 4*a**2 + 8*a/3) + x*(-2*a**2 - 2*a)

    >>> gegenbauer(n, a, x)
    gegenbauer(n, a, x)
    >>> gegenbauer(n, a, -x)
    (-1)**n*gegenbauer(n, a, x)

    >>> gegenbauer(n, a, 0)
    2**n*sqrt(pi)*gamma(a + n/2)/(gamma(a)*gamma(1/2 - n/2)*gamma(n + 1))
    >>> gegenbauer(n, a, 1)
    gamma(2*a + n)/(gamma(2*a)*gamma(n + 1))

    >>> conjugate(gegenbauer(n, a, x))
    gegenbauer(n, conjugate(a), conjugate(x))

    >>> diff(gegenbauer(n, a, x), x)
    2*a*gegenbauer(n - 1, a + 1, x)

    See Also
    ========

    jacobi,
    chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gegenbauer_polynomials
    .. [2] https://mathworld.wolfram.com/GegenbauerPolynomial.html
    .. [3] https://functions.wolfram.com/Polynomials/GegenbauerC3/

    """

    @classmethod
    def eval(cls, n, a, x):
        # For negative n the polynomials vanish
        # See https://functions.wolfram.com/Polynomials/GegenbauerC3/03/01/03/0012/
        if n.is_negative:
            return S.Zero

        # Some special values for fixed a
        if a == S.Half:
            return legendre(n, x)
        elif a == S.One:
            return chebyshevu(n, x)
        elif a == S.NegativeOne:
            return S.Zero

        if not n.is_Number:
            # Handle this before the general sign extraction rule
            if x == S.NegativeOne:
                if (re(a) > S.Half) == True:
                    return S.ComplexInfinity
                else:
                    return (cos(S.Pi*(a+n)) * sec(S.Pi*a) * gamma(2*a+n) /
                                (gamma(2*a) * gamma(n+1)))

            # Symbolic result C^a_n(x)
            # C^a_n(-x)  --->  (-1)**n * C^a_n(x)
            if x.could_extract_minus_sign():
                return S.NegativeOne**n * gegenbauer(n, a, -x)
            # We can evaluate for some special values of x
            if x.is_zero:
                return (2**n * sqrt(S.Pi) * gamma(a + S.Half*n) /
                        (gamma((1 - n)/2) * gamma(n + 1) * gamma(a)) )
            if x == S.One:
                return gamma(2*a + n) / (gamma(2*a) * gamma(n + 1))
            elif x is S.Infinity:
                if n.is_positive:
                    return RisingFactorial(a, n) * S.Infinity
        else:
            # n is a given fixed integer, evaluate into polynomial
            return gegenbauer_poly(n, a, x)

    def fdiff(self, argindex=3):
        from sympy.concrete.summations import Sum
        if argindex == 1:
            # Diff wrt n
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # Diff wrt a
            n, a, x = self.args
            k = Dummy("k")
            factor1 = 2 * (1 + (-1)**(n - k)) * (k + a) / ((k +
                           n + 2*a) * (n - k))
            factor2 = 2*(k + 1) / ((k + 2*a) * (2*k + 2*a + 1)) + \
                2 / (k + n + 2*a)
            kern = factor1*gegenbauer(k, a, x) + factor2*gegenbauer(n, a, x)
            return Sum(kern, (k, 0, n - 1))
        elif argindex == 3:
            # Diff wrt x
            n, a, x = self.args
            return 2*a*gegenbauer(n - 1, a + 1, x)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, a, x, **kwargs):
        from sympy.concrete.summations import Sum
        k = Dummy("k")
        kern = ((-1)**k * RisingFactorial(a, n - k) * (2*x)**(n - 2*k) /
                (factorial(k) * factorial(n - 2*k)))
        return Sum(kern, (k, 0, floor(n/2)))

    def _eval_rewrite_as_polynomial(self, n, a, x, **kwargs):
        # This function is just kept for backwards compatibility
        # but should not be used
        return self._eval_rewrite_as_Sum(n, a, x, **kwargs)

    def _eval_conjugate(self):
        n, a, x = self.args
        return self.func(n, a.conjugate(), x.conjugate())

#----------------------------------------------------------------------------
# Chebyshev polynomials of first and second kind
#


class chebyshevt(OrthogonalPolynomial):
    r"""
    Chebyshev polynomial of the first kind, $T_n(x)$.

    Explanation
    ===========

    ``chebyshevt(n, x)`` gives the $n$th Chebyshev polynomial (of the first
    kind) in $x$, $T_n(x)$.

    The Chebyshev polynomials of the first kind are orthogonal on
    $[-1, 1]$ with respect to the weight $\frac{1}{\sqrt{1-x^2}}$.

    Examples
    ========

    >>> from sympy import chebyshevt, diff
    >>> from sympy.abc import n,x
    >>> chebyshevt(0, x)
    1
    >>> chebyshevt(1, x)
    x
    >>> chebyshevt(2, x)
    2*x**2 - 1

    >>> chebyshevt(n, x)
    chebyshevt(n, x)
    >>> chebyshevt(n, -x)
    (-1)**n*chebyshevt(n, x)
    >>> chebyshevt(-n, x)
    chebyshevt(n, x)

    >>> chebyshevt(n, 0)
    cos(pi*n/2)
    >>> chebyshevt(n, -1)
    (-1)**n

    >>> diff(chebyshevt(n, x), x)
    n*chebyshevu(n - 1, x)

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Chebyshev_polynomial
    .. [2] https://mathworld.wolfram.com/ChebyshevPolynomialoftheFirstKind.html
    .. [3] https://mathworld.wolfram.com/ChebyshevPolynomialoftheSecondKind.html
    .. [4] https://functions.wolfram.com/Polynomials/ChebyshevT/
    .. [5] https://functions.wolfram.com/Polynomials/ChebyshevU/

    """

    _ortho_poly = staticmethod(chebyshevt_poly)

    @classmethod
    def eval(cls, n, x):
        if not n.is_Number:
            # Symbolic result T_n(x)
            # T_n(-x)  --->  (-1)**n * T_n(x)
            if x.could_extract_minus_sign():
                return S.NegativeOne**n * chebyshevt(n, -x)
            # T_{-n}(x)  --->  T_n(x)
            if n.could_extract_minus_sign():
                return chebyshevt(-n, x)
            # We can evaluate for some special values of x
            if x.is_zero:
                return cos(S.Half * S.Pi * n)
            if x == S.One:
                return S.One
            elif x is S.Infinity:
                return S.Infinity
        else:
            # n is a given fixed integer, evaluate into polynomial
            if n.is_negative:
                # T_{-n}(x) == T_n(x)
                return cls._eval_at_order(-n, x)
            else:
                return cls._eval_at_order(n, x)

    def fdiff(self, argindex=2):
        if argindex == 1:
            # Diff wrt n
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # Diff wrt x
            n, x = self.args
            return n * chebyshevu(n - 1, x)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, x, **kwargs):
        from sympy.concrete.summations import Sum
        k = Dummy("k")
        kern = binomial(n, 2*k) * (x**2 - 1)**k * x**(n - 2*k)
        return Sum(kern, (k, 0, floor(n/2)))

    def _eval_rewrite_as_polynomial(self, n, x, **kwargs):
        # This function is just kept for backwards compatibility
        # but should not be used
        return self._eval_rewrite_as_Sum(n, x, **kwargs)


class chebyshevu(OrthogonalPolynomial):
    r"""
    Chebyshev polynomial of the second kind, $U_n(x)$.

    Explanation
    ===========

    ``chebyshevu(n, x)`` gives the $n$th Chebyshev polynomial of the second
    kind in x, $U_n(x)$.

    The Chebyshev polynomials of the second kind are orthogonal on
    $[-1, 1]$ with respect to the weight $\sqrt{1-x^2}$.

    Examples
    ========

    >>> from sympy import chebyshevu, diff
    >>> from sympy.abc import n,x
    >>> chebyshevu(0, x)
    1
    >>> chebyshevu(1, x)
    2*x
    >>> chebyshevu(2, x)
    4*x**2 - 1

    >>> chebyshevu(n, x)
    chebyshevu(n, x)
    >>> chebyshevu(n, -x)
    (-1)**n*chebyshevu(n, x)
    >>> chebyshevu(-n, x)
    -chebyshevu(n - 2, x)

    >>> chebyshevu(n, 0)
    cos(pi*n/2)
    >>> chebyshevu(n, 1)
    n + 1

    >>> diff(chebyshevu(n, x), x)
    (-x*chebyshevu(n, x) + (n + 1)*chebyshevt(n + 1, x))/(x**2 - 1)

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevt_root, chebyshevu_root,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Chebyshev_polynomial
    .. [2] https://mathworld.wolfram.com/ChebyshevPolynomialoftheFirstKind.html
    .. [3] https://mathworld.wolfram.com/ChebyshevPolynomialoftheSecondKind.html
    .. [4] https://functions.wolfram.com/Polynomials/ChebyshevT/
    .. [5] https://functions.wolfram.com/Polynomials/ChebyshevU/

    """

    _ortho_poly = staticmethod(chebyshevu_poly)

    @classmethod
    def eval(cls, n, x):
        if not n.is_Number:
            # Symbolic result U_n(x)
            # U_n(-x)  --->  (-1)**n * U_n(x)
            if x.could_extract_minus_sign():
                return S.NegativeOne**n * chebyshevu(n, -x)
            # U_{-n}(x)  --->  -U_{n-2}(x)
            if n.could_extract_minus_sign():
                if n == S.NegativeOne:
                    # n can not be -1 here
                    return S.Zero
                elif not (-n - 2).could_extract_minus_sign():
                    return -chebyshevu(-n - 2, x)
            # We can evaluate for some special values of x
            if x.is_zero:
                return cos(S.Half * S.Pi * n)
            if x == S.One:
                return S.One + n
            elif x is S.Infinity:
                return S.Infinity
        else:
            # n is a given fixed integer, evaluate into polynomial
            if n.is_negative:
                # U_{-n}(x)  --->  -U_{n-2}(x)
                if n == S.NegativeOne:
                    return S.Zero
                else:
                    return -cls._eval_at_order(-n - 2, x)
            else:
                return cls._eval_at_order(n, x)

    def fdiff(self, argindex=2):
        if argindex == 1:
            # Diff wrt n
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # Diff wrt x
            n, x = self.args
            return ((n + 1) * chebyshevt(n + 1, x) - x * chebyshevu(n, x)) / (x**2 - 1)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, x, **kwargs):
        from sympy.concrete.summations import Sum
        k = Dummy("k")
        kern = S.NegativeOne**k * factorial(
            n - k) * (2*x)**(n - 2*k) / (factorial(k) * factorial(n - 2*k))
        return Sum(kern, (k, 0, floor(n/2)))

    def _eval_rewrite_as_polynomial(self, n, x, **kwargs):
        # This function is just kept for backwards compatibility
        # but should not be used
        return self._eval_rewrite_as_Sum(n, x, **kwargs)


class chebyshevt_root(DefinedFunction):
    r"""
    ``chebyshev_root(n, k)`` returns the $k$th root (indexed from zero) of
    the $n$th Chebyshev polynomial of the first kind; that is, if
    $0 \le k < n$, ``chebyshevt(n, chebyshevt_root(n, k)) == 0``.

    Examples
    ========

    >>> from sympy import chebyshevt, chebyshevt_root
    >>> chebyshevt_root(3, 2)
    -sqrt(3)/2
    >>> chebyshevt(3, chebyshevt_root(3, 2))
    0

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly
    """

    @classmethod
    def eval(cls, n, k):
        if not ((0 <= k) and (k < n)):
            raise ValueError("must have 0 <= k < n, "
                "got k = %s and n = %s" % (k, n))
        return cos(S.Pi*(2*k + 1)/(2*n))


class chebyshevu_root(DefinedFunction):
    r"""
    ``chebyshevu_root(n, k)`` returns the $k$th root (indexed from zero) of the
    $n$th Chebyshev polynomial of the second kind; that is, if $0 \le k < n$,
    ``chebyshevu(n, chebyshevu_root(n, k)) == 0``.

    Examples
    ========

    >>> from sympy import chebyshevu, chebyshevu_root
    >>> chebyshevu_root(3, 2)
    -sqrt(2)/2
    >>> chebyshevu(3, chebyshevu_root(3, 2))
    0

    See Also
    ========

    chebyshevt, chebyshevt_root, chebyshevu,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly
    """


    @classmethod
    def eval(cls, n, k):
        if not ((0 <= k) and (k < n)):
            raise ValueError("must have 0 <= k < n, "
                "got k = %s and n = %s" % (k, n))
        return cos(S.Pi*(k + 1)/(n + 1))

#----------------------------------------------------------------------------
# Legendre polynomials and Associated Legendre polynomials
#


class legendre(OrthogonalPolynomial):
    r"""
    ``legendre(n, x)`` gives the $n$th Legendre polynomial of $x$, $P_n(x)$

    Explanation
    ===========

    The Legendre polynomials are orthogonal on $[-1, 1]$ with respect to
    the constant weight 1. They satisfy $P_n(1) = 1$ for all $n$; further,
    $P_n$ is odd for odd $n$ and even for even $n$.

    Examples
    ========

    >>> from sympy import legendre, diff
    >>> from sympy.abc import x, n
    >>> legendre(0, x)
    1
    >>> legendre(1, x)
    x
    >>> legendre(2, x)
    3*x**2/2 - 1/2
    >>> legendre(n, x)
    legendre(n, x)
    >>> diff(legendre(n,x), x)
    n*(x*legendre(n, x) - legendre(n - 1, x))/(x**2 - 1)

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevt_root, chebyshevu, chebyshevu_root,
    assoc_legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Legendre_polynomial
    .. [2] https://mathworld.wolfram.com/LegendrePolynomial.html
    .. [3] https://functions.wolfram.com/Polynomials/LegendreP/
    .. [4] https://functions.wolfram.com/Polynomials/LegendreP2/

    """

    _ortho_poly = staticmethod(legendre_poly)

    @classmethod
    def eval(cls, n, x):
        if not n.is_Number:
            # Symbolic result L_n(x)
            # L_n(-x)  --->  (-1)**n * L_n(x)
            if x.could_extract_minus_sign():
                return S.NegativeOne**n * legendre(n, -x)
            # L_{-n}(x)  --->  L_{n-1}(x)
            if n.could_extract_minus_sign() and not(-n - 1).could_extract_minus_sign():
                return legendre(-n - S.One, x)
            # We can evaluate for some special values of x
            if x.is_zero:
                return sqrt(S.Pi)/(gamma(S.Half - n/2)*gamma(S.One + n/2))
            elif x == S.One:
                return S.One
            elif x is S.Infinity:
                return S.Infinity
        else:
            # n is a given fixed integer, evaluate into polynomial;
            # L_{-n}(x)  --->  L_{n-1}(x)
            if n.is_negative:
                n = -n - S.One
            return cls._eval_at_order(n, x)

    def fdiff(self, argindex=2):
        if argindex == 1:
            # Diff wrt n
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # Diff wrt x
            # Find better formula, this is unsuitable for x = +/-1
            # https://www.autodiff.org/ad16/Oral/Buecker_Legendre.pdf says
            # at x = 1:
            #    n*(n + 1)/2            , m = 0
            #    oo                     , m = 1
            #    -(n-1)*n*(n+1)*(n+2)/4 , m = 2
            #    0                      , m = 3, 4, ..., n
            #
            # at x = -1
            #    (-1)**(n+1)*n*(n + 1)/2       , m = 0
            #    (-1)**n*oo                    , m = 1
            #    (-1)**n*(n-1)*n*(n+1)*(n+2)/4 , m = 2
            #    0                             , m = 3, 4, ..., n
            n, x = self.args
            return n/(x**2 - 1)*(x*legendre(n, x) - legendre(n - 1, x))
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, x, **kwargs):
        from sympy.concrete.summations import Sum
        k = Dummy("k")
        kern = S.NegativeOne**k*binomial(n, k)**2*((1 + x)/2)**(n - k)*((1 - x)/2)**k
        return Sum(kern, (k, 0, n))

    def _eval_rewrite_as_polynomial(self, n, x, **kwargs):
        # This function is just kept for backwards compatibility
        # but should not be used
        return self._eval_rewrite_as_Sum(n, x, **kwargs)


class assoc_legendre(DefinedFunction):
    r"""
    ``assoc_legendre(n, m, x)`` gives $P_n^m(x)$, where $n$ and $m$ are
    the degree and order or an expression which is related to the nth
    order Legendre polynomial, $P_n(x)$ in the following manner:

    .. math::
        P_n^m(x) = (-1)^m (1 - x^2)^{\frac{m}{2}}
                   \frac{\mathrm{d}^m P_n(x)}{\mathrm{d} x^m}

    Explanation
    ===========

    Associated Legendre polynomials are orthogonal on $[-1, 1]$ with:

    - weight $= 1$            for the same $m$ and different $n$.
    - weight $= \frac{1}{1-x^2}$   for the same $n$ and different $m$.

    Examples
    ========

    >>> from sympy import assoc_legendre
    >>> from sympy.abc import x, m, n
    >>> assoc_legendre(0,0, x)
    1
    >>> assoc_legendre(1,0, x)
    x
    >>> assoc_legendre(1,1, x)
    -sqrt(1 - x**2)
    >>> assoc_legendre(n,m,x)
    assoc_legendre(n, m, x)

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Associated_Legendre_polynomials
    .. [2] https://mathworld.wolfram.com/LegendrePolynomial.html
    .. [3] https://functions.wolfram.com/Polynomials/LegendreP/
    .. [4] https://functions.wolfram.com/Polynomials/LegendreP2/

    """

    @classmethod
    def _eval_at_order(cls, n, m):
        P = legendre_poly(n, _x, polys=True).diff((_x, m))
        return S.NegativeOne**m * (1 - _x**2)**Rational(m, 2) * P.as_expr()

    @classmethod
    def eval(cls, n, m, x):
        if m.could_extract_minus_sign():
            # P^{-m}_n  --->  F * P^m_n
            return S.NegativeOne**(-m) * (factorial(m + n)/factorial(n - m)) * assoc_legendre(n, -m, x)
        if m == 0:
            # P^0_n  --->  L_n
            return legendre(n, x)
        if x == 0:
            return 2**m*sqrt(S.Pi) / (gamma((1 - m - n)/2)*gamma(1 - (m - n)/2))
        if n.is_Number and m.is_Number and n.is_integer and m.is_integer:
            if n.is_negative:
                raise ValueError("%s : 1st index must be nonnegative integer (got %r)" % (cls, n))
            if abs(m) > n:
                raise ValueError("%s : abs('2nd index') must be <= '1st index' (got %r, %r)" % (cls, n, m))
            return cls._eval_at_order(int(n), abs(int(m))).subs(_x, x)

    def fdiff(self, argindex=3):
        if argindex == 1:
            # Diff wrt n
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # Diff wrt m
            raise ArgumentIndexError(self, argindex)
        elif argindex == 3:
            # Diff wrt x
            # Find better formula, this is unsuitable for x = 1
            n, m, x = self.args
            return 1/(x**2 - 1)*(x*n*assoc_legendre(n, m, x) - (m + n)*assoc_legendre(n - 1, m, x))
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, m, x, **kwargs):
        from sympy.concrete.summations import Sum
        k = Dummy("k")
        kern = factorial(2*n - 2*k)/(2**n*factorial(n - k)*factorial(
            k)*factorial(n - 2*k - m))*S.NegativeOne**k*x**(n - m - 2*k)
        return (1 - x**2)**(m/2) * Sum(kern, (k, 0, floor((n - m)*S.Half)))

    def _eval_rewrite_as_polynomial(self, n, m, x, **kwargs):
        # This function is just kept for backwards compatibility
        # but should not be used
        return self._eval_rewrite_as_Sum(n, m, x, **kwargs)

    def _eval_conjugate(self):
        n, m, x = self.args
        return self.func(n, m.conjugate(), x.conjugate())

#----------------------------------------------------------------------------
# Hermite polynomials
#


class hermite(OrthogonalPolynomial):
    r"""
    ``hermite(n, x)`` gives the $n$th Hermite polynomial in $x$, $H_n(x)$.

    Explanation
    ===========

    The Hermite polynomials are orthogonal on $(-\infty, \infty)$
    with respect to the weight $\exp\left(-x^2\right)$.

    Examples
    ========

    >>> from sympy import hermite, diff
    >>> from sympy.abc import x, n
    >>> hermite(0, x)
    1
    >>> hermite(1, x)
    2*x
    >>> hermite(2, x)
    4*x**2 - 2
    >>> hermite(n, x)
    hermite(n, x)
    >>> diff(hermite(n,x), x)
    2*n*hermite(n - 1, x)
    >>> hermite(n, -x)
    (-1)**n*hermite(n, x)

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hermite_polynomial
    .. [2] https://mathworld.wolfram.com/HermitePolynomial.html
    .. [3] https://functions.wolfram.com/Polynomials/HermiteH/

    """

    _ortho_poly = staticmethod(hermite_poly)

    @classmethod
    def eval(cls, n, x):
        if not n.is_Number:
            # Symbolic result H_n(x)
            # H_n(-x)  --->  (-1)**n * H_n(x)
            if x.could_extract_minus_sign():
                return S.NegativeOne**n * hermite(n, -x)
            # We can evaluate for some special values of x
            if x.is_zero:
                return 2**n * sqrt(S.Pi) / gamma((S.One - n)/2)
            elif x is S.Infinity:
                return S.Infinity
        else:
            # n is a given fixed integer, evaluate into polynomial
            if n.is_negative:
                raise ValueError(
                    "The index n must be nonnegative integer (got %r)" % n)
            else:
                return cls._eval_at_order(n, x)

    def fdiff(self, argindex=2):
        if argindex == 1:
            # Diff wrt n
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # Diff wrt x
            n, x = self.args
            return 2*n*hermite(n - 1, x)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, x, **kwargs):
        from sympy.concrete.summations import Sum
        k = Dummy("k")
        kern = S.NegativeOne**k / (factorial(k)*factorial(n - 2*k)) * (2*x)**(n - 2*k)
        return factorial(n)*Sum(kern, (k, 0, floor(n/2)))

    def _eval_rewrite_as_polynomial(self, n, x, **kwargs):
        # This function is just kept for backwards compatibility
        # but should not be used
        return self._eval_rewrite_as_Sum(n, x, **kwargs)

    def _eval_rewrite_as_hermite_prob(self, n, x, **kwargs):
        return sqrt(2)**n * hermite_prob(n, x*sqrt(2))


class hermite_prob(OrthogonalPolynomial):
    r"""
    ``hermite_prob(n, x)`` gives the $n$th probabilist's Hermite polynomial
    in $x$, $He_n(x)$.

    Explanation
    ===========

    The probabilist's Hermite polynomials are orthogonal on $(-\infty, \infty)$
    with respect to the weight $\exp\left(-\frac{x^2}{2}\right)$. They are monic
    polynomials, related to the plain Hermite polynomials (:py:class:`~.hermite`) by

    .. math :: He_n(x) = 2^{-n/2} H_n(x/\sqrt{2})

    Examples
    ========

    >>> from sympy import hermite_prob, diff, I
    >>> from sympy.abc import x, n
    >>> hermite_prob(1, x)
    x
    >>> hermite_prob(5, x)
    x**5 - 10*x**3 + 15*x
    >>> diff(hermite_prob(n,x), x)
    n*hermite_prob(n - 1, x)
    >>> hermite_prob(n, -x)
    (-1)**n*hermite_prob(n, x)

    The sum of absolute values of coefficients of $He_n(x)$ is the number of
    matchings in the complete graph $K_n$ or telephone number, A000085 in the OEIS:

    >>> [hermite_prob(n,I) / I**n for n in range(11)]
    [1, 1, 2, 4, 10, 26, 76, 232, 764, 2620, 9496]

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hermite_polynomial
    .. [2] https://mathworld.wolfram.com/HermitePolynomial.html
    """

    _ortho_poly = staticmethod(hermite_prob_poly)

    @classmethod
    def eval(cls, n, x):
        if not n.is_Number:
            if x.could_extract_minus_sign():
                return S.NegativeOne**n * hermite_prob(n, -x)
            if x.is_zero:
                return sqrt(S.Pi) / gamma((S.One-n) / 2)
            elif x is S.Infinity:
                return S.Infinity
        else:
            if n.is_negative:
                ValueError("n must be a nonnegative integer, not %r" % n)
            else:
                return cls._eval_at_order(n, x)

    def fdiff(self, argindex=2):
        if argindex == 2:
            n, x = self.args
            return n*hermite_prob(n-1, x)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, x, **kwargs):
        from sympy.concrete.summations import Sum
        k = Dummy("k")
        kern = (-S.Half)**k * x**(n-2*k) / (factorial(k) * factorial(n-2*k))
        return factorial(n)*Sum(kern, (k, 0, floor(n/2)))

    def _eval_rewrite_as_polynomial(self, n, x, **kwargs):
        # This function is just kept for backwards compatibility
        # but should not be used
        return self._eval_rewrite_as_Sum(n, x, **kwargs)

    def _eval_rewrite_as_hermite(self, n, x, **kwargs):
        return sqrt(2)**(-n) * hermite(n, x/sqrt(2))


#----------------------------------------------------------------------------
# Laguerre polynomials
#


class laguerre(OrthogonalPolynomial):
    r"""
    Returns the $n$th Laguerre polynomial in $x$, $L_n(x)$.

    Examples
    ========

    >>> from sympy import laguerre, diff
    >>> from sympy.abc import x, n
    >>> laguerre(0, x)
    1
    >>> laguerre(1, x)
    1 - x
    >>> laguerre(2, x)
    x**2/2 - 2*x + 1
    >>> laguerre(3, x)
    -x**3/6 + 3*x**2/2 - 3*x + 1

    >>> laguerre(n, x)
    laguerre(n, x)

    >>> diff(laguerre(n, x), x)
    -assoc_laguerre(n - 1, 1, x)

    Parameters
    ==========

    n : int
        Degree of Laguerre polynomial. Must be `n \ge 0`.

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Laguerre_polynomial
    .. [2] https://mathworld.wolfram.com/LaguerrePolynomial.html
    .. [3] https://functions.wolfram.com/Polynomials/LaguerreL/
    .. [4] https://functions.wolfram.com/Polynomials/LaguerreL3/

    """

    _ortho_poly = staticmethod(laguerre_poly)

    @classmethod
    def eval(cls, n, x):
        if n.is_integer is False:
            raise ValueError("Error: n should be an integer.")
        if not n.is_Number:
            # Symbolic result L_n(x)
            # L_{n}(-x)  --->  exp(-x) * L_{-n-1}(x)
            # L_{-n}(x)  --->  exp(x) * L_{n-1}(-x)
            if n.could_extract_minus_sign() and not(-n - 1).could_extract_minus_sign():
                return exp(x)*laguerre(-n - 1, -x)
            # We can evaluate for some special values of x
            if x.is_zero:
                return S.One
            elif x is S.NegativeInfinity:
                return S.Infinity
            elif x is S.Infinity:
                return S.NegativeOne**n * S.Infinity
        else:
            if n.is_negative:
                return exp(x)*laguerre(-n - 1, -x)
            else:
                return cls._eval_at_order(n, x)

    def fdiff(self, argindex=2):
        if argindex == 1:
            # Diff wrt n
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # Diff wrt x
            n, x = self.args
            return -assoc_laguerre(n - 1, 1, x)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, x, **kwargs):
        from sympy.concrete.summations import Sum
        # Make sure n \in N_0
        if n.is_negative:
            return exp(x) * self._eval_rewrite_as_Sum(-n - 1, -x, **kwargs)
        if n.is_integer is False:
            raise ValueError("Error: n should be an integer.")
        k = Dummy("k")
        kern = RisingFactorial(-n, k) / factorial(k)**2 * x**k
        return Sum(kern, (k, 0, n))

    def _eval_rewrite_as_polynomial(self, n, x, **kwargs):
        # This function is just kept for backwards compatibility
        # but should not be used
        return self._eval_rewrite_as_Sum(n, x, **kwargs)


class assoc_laguerre(OrthogonalPolynomial):
    r"""
    Returns the $n$th generalized Laguerre polynomial in $x$, $L_n(x)$.

    Examples
    ========

    >>> from sympy import assoc_laguerre, diff
    >>> from sympy.abc import x, n, a
    >>> assoc_laguerre(0, a, x)
    1
    >>> assoc_laguerre(1, a, x)
    a - x + 1
    >>> assoc_laguerre(2, a, x)
    a**2/2 + 3*a/2 + x**2/2 + x*(-a - 2) + 1
    >>> assoc_laguerre(3, a, x)
    a**3/6 + a**2 + 11*a/6 - x**3/6 + x**2*(a/2 + 3/2) +
        x*(-a**2/2 - 5*a/2 - 3) + 1

    >>> assoc_laguerre(n, a, 0)
    binomial(a + n, a)

    >>> assoc_laguerre(n, a, x)
    assoc_laguerre(n, a, x)

    >>> assoc_laguerre(n, 0, x)
    laguerre(n, x)

    >>> diff(assoc_laguerre(n, a, x), x)
    -assoc_laguerre(n - 1, a + 1, x)

    >>> diff(assoc_laguerre(n, a, x), a)
    Sum(assoc_laguerre(_k, a, x)/(-a + n), (_k, 0, n - 1))

    Parameters
    ==========

    n : int
        Degree of Laguerre polynomial. Must be `n \ge 0`.

    alpha : Expr
        Arbitrary expression. For ``alpha=0`` regular Laguerre
        polynomials will be generated.

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Laguerre_polynomial#Generalized_Laguerre_polynomials
    .. [2] https://mathworld.wolfram.com/AssociatedLaguerrePolynomial.html
    .. [3] https://functions.wolfram.com/Polynomials/LaguerreL/
    .. [4] https://functions.wolfram.com/Polynomials/LaguerreL3/

    """

    @classmethod
    def eval(cls, n, alpha, x):
        # L_{n}^{0}(x)  --->  L_{n}(x)
        if alpha.is_zero:
            return laguerre(n, x)

        if not n.is_Number:
            # We can evaluate for some special values of x
            if x.is_zero:
                return binomial(n + alpha, alpha)
            elif x is S.Infinity and n > 0:
                return S.NegativeOne**n * S.Infinity
            elif x is S.NegativeInfinity and n > 0:
                return S.Infinity
        else:
            # n is a given fixed integer, evaluate into polynomial
            if n.is_negative:
                raise ValueError(
                    "The index n must be nonnegative integer (got %r)" % n)
            else:
                return laguerre_poly(n, x, alpha)

    def fdiff(self, argindex=3):
        from sympy.concrete.summations import Sum
        if argindex == 1:
            # Diff wrt n
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # Diff wrt alpha
            n, alpha, x = self.args
            k = Dummy("k")
            return Sum(assoc_laguerre(k, alpha, x) / (n - alpha), (k, 0, n - 1))
        elif argindex == 3:
            # Diff wrt x
            n, alpha, x = self.args
            return -assoc_laguerre(n - 1, alpha + 1, x)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, alpha, x, **kwargs):
        from sympy.concrete.summations import Sum
        # Make sure n \in N_0
        if n.is_negative or n.is_integer is False:
            raise ValueError("Error: n should be a non-negative integer.")
        k = Dummy("k")
        kern = RisingFactorial(
            -n, k) / (gamma(k + alpha + 1) * factorial(k)) * x**k
        return gamma(n + alpha + 1) / factorial(n) * Sum(kern, (k, 0, n))

    def _eval_rewrite_as_polynomial(self, n, alpha, x, **kwargs):
        # This function is just kept for backwards compatibility
        # but should not be used
        return self._eval_rewrite_as_Sum(n, alpha, x, **kwargs)

    def _eval_conjugate(self):
        n, alpha, x = self.args
        return self.func(n, alpha.conjugate(), x.conjugate())
