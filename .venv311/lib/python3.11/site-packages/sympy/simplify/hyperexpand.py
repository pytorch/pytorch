"""
Expand Hypergeometric (and Meijer G) functions into named
special functions.

The algorithm for doing this uses a collection of lookup tables of
hypergeometric functions, and various of their properties, to expand
many hypergeometric functions in terms of special functions.

It is based on the following paper:
      Kelly B. Roach.  Meijer G Function Representations.
      In: Proceedings of the 1997 International Symposium on Symbolic and
      Algebraic Computation, pages 205-211, New York, 1997. ACM.

It is described in great(er) detail in the Sphinx documentation.
"""
# SUMMARY OF EXTENSIONS FOR MEIJER G FUNCTIONS
#
# o z**rho G(ap, bq; z) = G(ap + rho, bq + rho; z)
#
# o denote z*d/dz by D
#
# o It is helpful to keep in mind that ap and bq play essentially symmetric
#   roles: G(1/z) has slightly altered parameters, with ap and bq interchanged.
#
# o There are four shift operators:
#   A_J = b_J - D,     J = 1, ..., n
#   B_J = 1 - a_j + D, J = 1, ..., m
#   C_J = -b_J + D,    J = m+1, ..., q
#   D_J = a_J - 1 - D, J = n+1, ..., p
#
#   A_J, C_J increment b_J
#   B_J, D_J decrement a_J
#
# o The corresponding four inverse-shift operators are defined if there
#   is no cancellation. Thus e.g. an index a_J (upper or lower) can be
#   incremented if a_J != b_i for i = 1, ..., q.
#
# o Order reduction: if b_j - a_i is a non-negative integer, where
#   j <= m and i > n, the corresponding quotient of gamma functions reduces
#   to a polynomial. Hence the G function can be expressed using a G-function
#   of lower order.
#   Similarly if j > m and i <= n.
#
#   Secondly, there are paired index theorems [Adamchik, The evaluation of
#   integrals of Bessel functions via G-function identities]. Suppose there
#   are three parameters a, b, c, where a is an a_i, i <= n, b is a b_j,
#   j <= m and c is a denominator parameter (i.e. a_i, i > n or b_j, j > m).
#   Suppose further all three differ by integers.
#   Then the order can be reduced.
#   TODO work this out in detail.
#
# o An index quadruple is called suitable if its order cannot be reduced.
#   If there exists a sequence of shift operators transforming one index
#   quadruple into another, we say one is reachable from the other.
#
# o Deciding if one index quadruple is reachable from another is tricky. For
#   this reason, we use hand-built routines to match and instantiate formulas.
#
from collections import defaultdict
from itertools import product
from functools import reduce
from math import prod

from sympy import SYMPY_DEBUG
from sympy.core import (S, Dummy, symbols, sympify, Tuple, expand, I, pi, Mul,
    EulerGamma, oo, zoo, expand_func, Add, nan, Expr, Rational)
from sympy.core.mod import Mod
from sympy.core.sorting import default_sort_key
from sympy.functions import (exp, sqrt, root, log, lowergamma, cos,
        besseli, gamma, uppergamma, expint, erf, sin, besselj, Ei, Ci, Si, Shi,
        sinh, cosh, Chi, fresnels, fresnelc, polar_lift, exp_polar, floor, ceiling,
        rf, factorial, lerchphi, Piecewise, re, elliptic_k, elliptic_e)
from sympy.functions.elementary.complexes import polarify, unpolarify
from sympy.functions.special.hyper import (hyper, HyperRep_atanh,
        HyperRep_power1, HyperRep_power2, HyperRep_log1, HyperRep_asin1,
        HyperRep_asin2, HyperRep_sqrts1, HyperRep_sqrts2, HyperRep_log2,
        HyperRep_cosasin, HyperRep_sinasin, meijerg)
from sympy.matrices import Matrix, eye, zeros
from sympy.polys import apart, poly, Poly
from sympy.series import residue
from sympy.simplify.powsimp import powdenest
from sympy.utilities.iterables import sift

# function to define "buckets"
def _mod1(x):
    # TODO see if this can work as Mod(x, 1); this will require
    # different handling of the "buckets" since these need to
    # be sorted and that fails when there is a mixture of
    # integers and expressions with parameters. With the current
    # Mod behavior, Mod(k, 1) == Mod(1, 1) == 0 if k is an integer.
    # Although the sorting can be done with Basic.compare, this may
    # still require different handling of the sorted buckets.
    if x.is_Number:
        return Mod(x, 1)
    c, x = x.as_coeff_Add()
    return Mod(c, 1) + x


# leave add formulae at the top for easy reference
def add_formulae(formulae):
    """ Create our knowledge base. """
    a, b, c, z = symbols('a b c, z', cls=Dummy)

    def add(ap, bq, res):
        func = Hyper_Function(ap, bq)
        formulae.append(Formula(func, z, res, (a, b, c)))

    def addb(ap, bq, B, C, M):
        func = Hyper_Function(ap, bq)
        formulae.append(Formula(func, z, None, (a, b, c), B, C, M))

    # Luke, Y. L. (1969), The Special Functions and Their Approximations,
    # Volume 1, section 6.2

    # 0F0
    add((), (), exp(z))

    # 1F0
    add((a, ), (), HyperRep_power1(-a, z))

    # 2F1
    addb((a, a - S.Half), (2*a, ),
         Matrix([HyperRep_power2(a, z),
                 HyperRep_power2(a + S.Half, z)/2]),
         Matrix([[1, 0]]),
         Matrix([[(a - S.Half)*z/(1 - z), (S.Half - a)*z/(1 - z)],
                 [a/(1 - z), a*(z - 2)/(1 - z)]]))
    addb((1, 1), (2, ),
         Matrix([HyperRep_log1(z), 1]), Matrix([[-1/z, 0]]),
         Matrix([[0, z/(z - 1)], [0, 0]]))
    addb((S.Half, 1), (S('3/2'), ),
         Matrix([HyperRep_atanh(z), 1]),
         Matrix([[1, 0]]),
         Matrix([[Rational(-1, 2), 1/(1 - z)/2], [0, 0]]))
    addb((S.Half, S.Half), (S('3/2'), ),
         Matrix([HyperRep_asin1(z), HyperRep_power1(Rational(-1, 2), z)]),
         Matrix([[1, 0]]),
         Matrix([[Rational(-1, 2), S.Half], [0, z/(1 - z)/2]]))
    addb((a, S.Half + a), (S.Half, ),
         Matrix([HyperRep_sqrts1(-a, z), -HyperRep_sqrts2(-a - S.Half, z)]),
         Matrix([[1, 0]]),
         Matrix([[0, -a],
                 [z*(-2*a - 1)/2/(1 - z), S.Half - z*(-2*a - 1)/(1 - z)]]))

    # A. P. Prudnikov, Yu. A. Brychkov and O. I. Marichev (1990).
    # Integrals and Series: More Special Functions, Vol. 3,.
    # Gordon and Breach Science Publisher
    addb([a, -a], [S.Half],
         Matrix([HyperRep_cosasin(a, z), HyperRep_sinasin(a, z)]),
         Matrix([[1, 0]]),
         Matrix([[0, -a], [a*z/(1 - z), 1/(1 - z)/2]]))
    addb([1, 1], [3*S.Half],
         Matrix([HyperRep_asin2(z), 1]), Matrix([[1, 0]]),
         Matrix([[(z - S.Half)/(1 - z), 1/(1 - z)/2], [0, 0]]))

    # Complete elliptic integrals K(z) and E(z), both a 2F1 function
    addb([S.Half, S.Half], [S.One],
         Matrix([elliptic_k(z), elliptic_e(z)]),
         Matrix([[2/pi, 0]]),
         Matrix([[Rational(-1, 2), -1/(2*z-2)],
                 [Rational(-1, 2), S.Half]]))
    addb([Rational(-1, 2), S.Half], [S.One],
         Matrix([elliptic_k(z), elliptic_e(z)]),
         Matrix([[0, 2/pi]]),
         Matrix([[Rational(-1, 2), -1/(2*z-2)],
                 [Rational(-1, 2), S.Half]]))

    # 3F2
    addb([Rational(-1, 2), 1, 1], [S.Half, 2],
         Matrix([z*HyperRep_atanh(z), HyperRep_log1(z), 1]),
         Matrix([[Rational(-2, 3), -S.One/(3*z), Rational(2, 3)]]),
         Matrix([[S.Half, 0, z/(1 - z)/2],
                 [0, 0, z/(z - 1)],
                 [0, 0, 0]]))
    # actually the formula for 3/2 is much nicer ...
    addb([Rational(-1, 2), 1, 1], [2, 2],
         Matrix([HyperRep_power1(S.Half, z), HyperRep_log2(z), 1]),
         Matrix([[Rational(4, 9) - 16/(9*z), 4/(3*z), 16/(9*z)]]),
         Matrix([[z/2/(z - 1), 0, 0], [1/(2*(z - 1)), 0, S.Half], [0, 0, 0]]))

    # 1F1
    addb([1], [b], Matrix([z**(1 - b) * exp(z) * lowergamma(b - 1, z), 1]),
         Matrix([[b - 1, 0]]), Matrix([[1 - b + z, 1], [0, 0]]))
    addb([a], [2*a],
         Matrix([z**(S.Half - a)*exp(z/2)*besseli(a - S.Half, z/2)
                 * gamma(a + S.Half)/4**(S.Half - a),
                 z**(S.Half - a)*exp(z/2)*besseli(a + S.Half, z/2)
                 * gamma(a + S.Half)/4**(S.Half - a)]),
         Matrix([[1, 0]]),
         Matrix([[z/2, z/2], [z/2, (z/2 - 2*a)]]))
    mz = polar_lift(-1)*z
    addb([a], [a + 1],
         Matrix([mz**(-a)*a*lowergamma(a, mz), a*exp(z)]),
         Matrix([[1, 0]]),
         Matrix([[-a, 1], [0, z]]))
    # This one is redundant.
    add([Rational(-1, 2)], [S.Half], exp(z) - sqrt(pi*z)*(-I)*erf(I*sqrt(z)))

    # Added to get nice results for Laplace transform of Fresnel functions
    # https://functions.wolfram.com/07.22.03.6437.01
    # Basic rule
    #add([1], [Rational(3, 4), Rational(5, 4)],
    #    sqrt(pi) * (cos(2*sqrt(polar_lift(-1)*z))*fresnelc(2*root(polar_lift(-1)*z,4)/sqrt(pi)) +
    #                sin(2*sqrt(polar_lift(-1)*z))*fresnels(2*root(polar_lift(-1)*z,4)/sqrt(pi)))
    #    / (2*root(polar_lift(-1)*z,4)))
    # Manually tuned rule
    addb([1], [Rational(3, 4), Rational(5, 4)],
         Matrix([ sqrt(pi)*(I*sinh(2*sqrt(z))*fresnels(2*root(z, 4)*exp(I*pi/4)/sqrt(pi))
                            + cosh(2*sqrt(z))*fresnelc(2*root(z, 4)*exp(I*pi/4)/sqrt(pi)))
                  * exp(-I*pi/4)/(2*root(z, 4)),
                  sqrt(pi)*root(z, 4)*(sinh(2*sqrt(z))*fresnelc(2*root(z, 4)*exp(I*pi/4)/sqrt(pi))
                                      + I*cosh(2*sqrt(z))*fresnels(2*root(z, 4)*exp(I*pi/4)/sqrt(pi)))
                  *exp(-I*pi/4)/2,
                  1 ]),
         Matrix([[1, 0, 0]]),
         Matrix([[Rational(-1, 4),              1, Rational(1, 4)],
                 [              z, Rational(1, 4),              0],
                 [              0,              0,              0]]))

    # 2F2
    addb([S.Half, a], [Rational(3, 2), a + 1],
         Matrix([a/(2*a - 1)*(-I)*sqrt(pi/z)*erf(I*sqrt(z)),
                 a/(2*a - 1)*(polar_lift(-1)*z)**(-a)*
                 lowergamma(a, polar_lift(-1)*z),
                 a/(2*a - 1)*exp(z)]),
         Matrix([[1, -1, 0]]),
         Matrix([[Rational(-1, 2), 0, 1], [0, -a, 1], [0, 0, z]]))
    # We make a "basis" of four functions instead of three, and give EulerGamma
    # an extra slot (it could just be a coefficient to 1). The advantage is
    # that this way Polys will not see multivariate polynomials (it treats
    # EulerGamma as an indeterminate), which is *way* faster.
    addb([1, 1], [2, 2],
         Matrix([Ei(z) - log(z), exp(z), 1, EulerGamma]),
         Matrix([[1/z, 0, 0, -1/z]]),
         Matrix([[0, 1, -1, 0], [0, z, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))

    # 0F1
    add((), (S.Half, ), cosh(2*sqrt(z)))
    addb([], [b],
         Matrix([gamma(b)*z**((1 - b)/2)*besseli(b - 1, 2*sqrt(z)),
                 gamma(b)*z**(1 - b/2)*besseli(b, 2*sqrt(z))]),
         Matrix([[1, 0]]), Matrix([[0, 1], [z, (1 - b)]]))

    # 0F3
    x = 4*z**Rational(1, 4)

    def fp(a, z):
        return besseli(a, x) + besselj(a, x)

    def fm(a, z):
        return besseli(a, x) - besselj(a, x)

    # TODO branching
    addb([], [S.Half, a, a + S.Half],
         Matrix([fp(2*a - 1, z), fm(2*a, z)*z**Rational(1, 4),
                 fm(2*a - 1, z)*sqrt(z), fp(2*a, z)*z**Rational(3, 4)])
         * 2**(-2*a)*gamma(2*a)*z**((1 - 2*a)/4),
         Matrix([[1, 0, 0, 0]]),
         Matrix([[0, 1, 0, 0],
                 [0, S.Half - a, 1, 0],
                 [0, 0, S.Half, 1],
                 [z, 0, 0, 1 - a]]))
    x = 2*(4*z)**Rational(1, 4)*exp_polar(I*pi/4)
    addb([], [a, a + S.Half, 2*a],
         (2*sqrt(polar_lift(-1)*z))**(1 - 2*a)*gamma(2*a)**2 *
         Matrix([besselj(2*a - 1, x)*besseli(2*a - 1, x),
                 x*(besseli(2*a, x)*besselj(2*a - 1, x)
                    - besseli(2*a - 1, x)*besselj(2*a, x)),
                 x**2*besseli(2*a, x)*besselj(2*a, x),
                 x**3*(besseli(2*a, x)*besselj(2*a - 1, x)
                       + besseli(2*a - 1, x)*besselj(2*a, x))]),
         Matrix([[1, 0, 0, 0]]),
         Matrix([[0, Rational(1, 4), 0, 0],
                 [0, (1 - 2*a)/2, Rational(-1, 2), 0],
                 [0, 0, 1 - 2*a, Rational(1, 4)],
                 [-32*z, 0, 0, 1 - a]]))

    # 1F2
    addb([a], [a - S.Half, 2*a],
         Matrix([z**(S.Half - a)*besseli(a - S.Half, sqrt(z))**2,
                 z**(1 - a)*besseli(a - S.Half, sqrt(z))
                 *besseli(a - Rational(3, 2), sqrt(z)),
                 z**(Rational(3, 2) - a)*besseli(a - Rational(3, 2), sqrt(z))**2]),
         Matrix([[-gamma(a + S.Half)**2/4**(S.Half - a),
                 2*gamma(a - S.Half)*gamma(a + S.Half)/4**(1 - a),
                 0]]),
         Matrix([[1 - 2*a, 1, 0], [z/2, S.Half - a, S.Half], [0, z, 0]]))
    addb([S.Half], [b, 2 - b],
         pi*(1 - b)/sin(pi*b)*
         Matrix([besseli(1 - b, sqrt(z))*besseli(b - 1, sqrt(z)),
                 sqrt(z)*(besseli(-b, sqrt(z))*besseli(b - 1, sqrt(z))
                          + besseli(1 - b, sqrt(z))*besseli(b, sqrt(z))),
                 besseli(-b, sqrt(z))*besseli(b, sqrt(z))]),
         Matrix([[1, 0, 0]]),
         Matrix([[b - 1, S.Half, 0],
                 [z, 0, z],
                 [0, S.Half, -b]]))
    addb([S.Half], [Rational(3, 2), Rational(3, 2)],
         Matrix([Shi(2*sqrt(z))/2/sqrt(z), sinh(2*sqrt(z))/2/sqrt(z),
                 cosh(2*sqrt(z))]),
         Matrix([[1, 0, 0]]),
         Matrix([[Rational(-1, 2), S.Half, 0], [0, Rational(-1, 2), S.Half], [0, 2*z, 0]]))

    # FresnelS
    # Basic rule
    #add([Rational(3, 4)], [Rational(3, 2),Rational(7, 4)], 6*fresnels( exp(pi*I/4)*root(z,4)*2/sqrt(pi) ) / ( pi * (exp(pi*I/4)*root(z,4)*2/sqrt(pi))**3 ) )
    # Manually tuned rule
    addb([Rational(3, 4)], [Rational(3, 2), Rational(7, 4)],
         Matrix(
             [ fresnels(
                 exp(
                     pi*I/4)*root(
                         z, 4)*2/sqrt(
                             pi) ) / (
                                 pi * (exp(pi*I/4)*root(z, 4)*2/sqrt(pi))**3 ),
               sinh(2*sqrt(z))/sqrt(z),
               cosh(2*sqrt(z)) ]),
         Matrix([[6, 0, 0]]),
         Matrix([[Rational(-3, 4),  Rational(1, 16), 0],
                 [ 0,      Rational(-1, 2),  1],
                 [ 0,       z,       0]]))

    # FresnelC
    # Basic rule
    #add([Rational(1, 4)], [S.Half,Rational(5, 4)], fresnelc( exp(pi*I/4)*root(z,4)*2/sqrt(pi) ) / ( exp(pi*I/4)*root(z,4)*2/sqrt(pi) ) )
    # Manually tuned rule
    addb([Rational(1, 4)], [S.Half, Rational(5, 4)],
         Matrix(
             [ sqrt(
                 pi)*exp(
                     -I*pi/4)*fresnelc(
                         2*root(z, 4)*exp(I*pi/4)/sqrt(pi))/(2*root(z, 4)),
               cosh(2*sqrt(z)),
               sinh(2*sqrt(z))*sqrt(z) ]),
         Matrix([[1, 0, 0]]),
         Matrix([[Rational(-1, 4),  Rational(1, 4), 0     ],
                 [ 0,       0,      1     ],
                 [ 0,       z,      S.Half]]))

    # 2F3
    # XXX with this five-parameter formula is pretty slow with the current
    #     Formula.find_instantiations (creates 2!*3!*3**(2+3) ~ 3000
    #     instantiations ... But it's not too bad.
    addb([a, a + S.Half], [2*a, b, 2*a - b + 1],
         gamma(b)*gamma(2*a - b + 1) * (sqrt(z)/2)**(1 - 2*a) *
         Matrix([besseli(b - 1, sqrt(z))*besseli(2*a - b, sqrt(z)),
                 sqrt(z)*besseli(b, sqrt(z))*besseli(2*a - b, sqrt(z)),
                 sqrt(z)*besseli(b - 1, sqrt(z))*besseli(2*a - b + 1, sqrt(z)),
                 besseli(b, sqrt(z))*besseli(2*a - b + 1, sqrt(z))]),
         Matrix([[1, 0, 0, 0]]),
         Matrix([[0, S.Half, S.Half, 0],
                 [z/2, 1 - b, 0, z/2],
                 [z/2, 0, b - 2*a, z/2],
                 [0, S.Half, S.Half, -2*a]]))
    # (C/f above comment about eulergamma in the basis).
    addb([1, 1], [2, 2, Rational(3, 2)],
         Matrix([Chi(2*sqrt(z)) - log(2*sqrt(z)),
                 cosh(2*sqrt(z)), sqrt(z)*sinh(2*sqrt(z)), 1, EulerGamma]),
         Matrix([[1/z, 0, 0, 0, -1/z]]),
         Matrix([[0, S.Half, 0, Rational(-1, 2), 0],
                 [0, 0, 1, 0, 0],
                 [0, z, S.Half, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]]))

    # 3F3
    # This is rule: https://functions.wolfram.com/07.31.03.0134.01
    # Initial reason to add it was a nice solution for
    # integrate(erf(a*z)/z**2, z) and same for erfc and erfi.
    # Basic rule
    # add([1, 1, a], [2, 2, a+1], (a/(z*(a-1)**2)) *
    #     (1 - (-z)**(1-a) * (gamma(a) - uppergamma(a,-z))
    #      - (a-1) * (EulerGamma + uppergamma(0,-z) + log(-z))
    #      - exp(z)))
    # Manually tuned rule
    addb([1, 1, a], [2, 2, a+1],
         Matrix([a*(log(-z) + expint(1, -z) + EulerGamma)/(z*(a**2 - 2*a + 1)),
                 a*(-z)**(-a)*(gamma(a) - uppergamma(a, -z))/(a - 1)**2,
                 a*exp(z)/(a**2 - 2*a + 1),
                 a/(z*(a**2 - 2*a + 1))]),
         Matrix([[1-a, 1, -1/z, 1]]),
         Matrix([[-1,0,-1/z,1],
                 [0,-a,1,0],
                 [0,0,z,0],
                 [0,0,0,-1]]))


def add_meijerg_formulae(formulae):
    a, b, c, z = list(map(Dummy, 'abcz'))
    rho = Dummy('rho')

    def add(an, ap, bm, bq, B, C, M, matcher):
        formulae.append(MeijerFormula(an, ap, bm, bq, z, [a, b, c, rho],
                                      B, C, M, matcher))

    def detect_uppergamma(func):
        x = func.an[0]
        y, z = func.bm
        swapped = False
        if not _mod1((x - y).simplify()):
            swapped = True
            (y, z) = (z, y)
        if _mod1((x - z).simplify()) or x - z > 0:
            return None
        l = [y, x]
        if swapped:
            l = [x, y]
        return {rho: y, a: x - y}, G_Function([x], [], l, [])

    add([a + rho], [], [rho, a + rho], [],
        Matrix([gamma(1 - a)*z**rho*exp(z)*uppergamma(a, z),
                gamma(1 - a)*z**(a + rho)]),
        Matrix([[1, 0]]),
        Matrix([[rho + z, -1], [0, a + rho]]),
        detect_uppergamma)

    def detect_3113(func):
        """https://functions.wolfram.com/07.34.03.0984.01"""
        x = func.an[0]
        u, v, w = func.bm
        if _mod1((u - v).simplify()) == 0:
            if _mod1((v - w).simplify()) == 0:
                return
            sig = (S.Half, S.Half, S.Zero)
            x1, x2, y = u, v, w
        else:
            if _mod1((x - u).simplify()) == 0:
                sig = (S.Half, S.Zero, S.Half)
                x1, y, x2 = u, v, w
            else:
                sig = (S.Zero, S.Half, S.Half)
                y, x1, x2 = u, v, w

        if (_mod1((x - x1).simplify()) != 0 or
            _mod1((x - x2).simplify()) != 0 or
            _mod1((x - y).simplify()) != S.Half or
                x - x1 > 0 or x - x2 > 0):
            return

        return {a: x}, G_Function([x], [], [x - S.Half + t for t in sig], [])

    s = sin(2*sqrt(z))
    c_ = cos(2*sqrt(z))
    S_ = Si(2*sqrt(z)) - pi/2
    C = Ci(2*sqrt(z))
    add([a], [], [a, a, a - S.Half], [],
        Matrix([sqrt(pi)*z**(a - S.Half)*(c_*S_ - s*C),
                sqrt(pi)*z**a*(s*S_ + c_*C),
                sqrt(pi)*z**a]),
        Matrix([[-2, 0, 0]]),
        Matrix([[a - S.Half, -1, 0], [z, a, S.Half], [0, 0, a]]),
        detect_3113)


def make_simp(z):
    """ Create a function that simplifies rational functions in ``z``. """

    def simp(expr):
        """ Efficiently simplify the rational function ``expr``. """
        numer, denom = expr.as_numer_denom()
        numer = numer.expand()
        # denom = denom.expand()  # is this needed?
        c, numer, denom = poly(numer, z).cancel(poly(denom, z))
        return c * numer.as_expr() / denom.as_expr()

    return simp


def debug(*args):
    if SYMPY_DEBUG:
        for a in args:
            print(a, end="")
        print()


class Hyper_Function(Expr):
    """ A generalized hypergeometric function. """

    def __new__(cls, ap, bq):
        obj = super().__new__(cls)
        obj.ap = Tuple(*list(map(expand, ap)))
        obj.bq = Tuple(*list(map(expand, bq)))
        return obj

    @property
    def args(self):
        return (self.ap, self.bq)

    @property
    def sizes(self):
        return (len(self.ap), len(self.bq))

    @property
    def gamma(self):
        """
        Number of upper parameters that are negative integers

        This is a transformation invariant.
        """
        return sum(bool(x.is_integer and x.is_negative) for x in self.ap)

    def _hashable_content(self):
        return super()._hashable_content() + (self.ap,
                self.bq)

    def __call__(self, arg):
        return hyper(self.ap, self.bq, arg)

    def build_invariants(self):
        """
        Compute the invariant vector.

        Explanation
        ===========

        The invariant vector is:
            (gamma, ((s1, n1), ..., (sk, nk)), ((t1, m1), ..., (tr, mr)))
        where gamma is the number of integer a < 0,
              s1 < ... < sk
              nl is the number of parameters a_i congruent to sl mod 1
              t1 < ... < tr
              ml is the number of parameters b_i congruent to tl mod 1

        If the index pair contains parameters, then this is not truly an
        invariant, since the parameters cannot be sorted uniquely mod1.

        Examples
        ========

        >>> from sympy.simplify.hyperexpand import Hyper_Function
        >>> from sympy import S
        >>> ap = (S.Half, S.One/3, S(-1)/2, -2)
        >>> bq = (1, 2)

        Here gamma = 1,
             k = 3, s1 = 0, s2 = 1/3, s3 = 1/2
                    n1 = 1, n2 = 1,   n2 = 2
             r = 1, t1 = 0
                    m1 = 2:

        >>> Hyper_Function(ap, bq).build_invariants()
        (1, ((0, 1), (1/3, 1), (1/2, 2)), ((0, 2),))
        """
        abuckets, bbuckets = sift(self.ap, _mod1), sift(self.bq, _mod1)

        def tr(bucket):
            bucket = list(bucket.items())
            if not any(isinstance(x[0], Mod) for x in bucket):
                bucket.sort(key=lambda x: default_sort_key(x[0]))
            bucket = tuple([(mod, len(values)) for mod, values in bucket if
                    values])
            return bucket

        return (self.gamma, tr(abuckets), tr(bbuckets))

    def difficulty(self, func):
        """ Estimate how many steps it takes to reach ``func`` from self.
            Return -1 if impossible. """
        if self.gamma != func.gamma:
            return -1
        oabuckets, obbuckets, abuckets, bbuckets = [sift(params, _mod1) for
                params in (self.ap, self.bq, func.ap, func.bq)]

        diff = 0
        for bucket, obucket in [(abuckets, oabuckets), (bbuckets, obbuckets)]:
            for mod in set(list(bucket.keys()) + list(obucket.keys())):
                if (mod not in bucket) or (mod not in obucket) \
                        or len(bucket[mod]) != len(obucket[mod]):
                    return -1
                l1 = list(bucket[mod])
                l2 = list(obucket[mod])
                l1.sort()
                l2.sort()
                for i, j in zip(l1, l2):
                    diff += abs(i - j)

        return diff

    def _is_suitable_origin(self):
        """
        Decide if ``self`` is a suitable origin.

        Explanation
        ===========

        A function is a suitable origin iff:
        * none of the ai equals bj + n, with n a non-negative integer
        * none of the ai is zero
        * none of the bj is a non-positive integer

        Note that this gives meaningful results only when none of the indices
        are symbolic.

        """
        for a in self.ap:
            for b in self.bq:
                if (a - b).is_integer and (a - b).is_negative is False:
                    return False
        for a in self.ap:
            if a == 0:
                return False
        for b in self.bq:
            if b.is_integer and b.is_nonpositive:
                return False
        return True


class G_Function(Expr):
    """ A Meijer G-function. """

    def __new__(cls, an, ap, bm, bq):
        obj = super().__new__(cls)
        obj.an = Tuple(*list(map(expand, an)))
        obj.ap = Tuple(*list(map(expand, ap)))
        obj.bm = Tuple(*list(map(expand, bm)))
        obj.bq = Tuple(*list(map(expand, bq)))
        return obj

    @property
    def args(self):
        return (self.an, self.ap, self.bm, self.bq)

    def _hashable_content(self):
        return super()._hashable_content() + self.args

    def __call__(self, z):
        return meijerg(self.an, self.ap, self.bm, self.bq, z)

    def compute_buckets(self):
        """
        Compute buckets for the fours sets of parameters.

        Explanation
        ===========

        We guarantee that any two equal Mod objects returned are actually the
        same, and that the buckets are sorted by real part (an and bq
        descendending, bm and ap ascending).

        Examples
        ========

        >>> from sympy.simplify.hyperexpand import G_Function
        >>> from sympy.abc import y
        >>> from sympy import S

        >>> a, b = [1, 3, 2, S(3)/2], [1 + y, y, 2, y + 3]
        >>> G_Function(a, b, [2], [y]).compute_buckets()
        ({0: [3, 2, 1], 1/2: [3/2]},
        {0: [2], y: [y, y + 1, y + 3]}, {0: [2]}, {y: [y]})

        """
        dicts = pan, pap, pbm, pbq = [defaultdict(list) for i in range(4)]
        for dic, lis in zip(dicts, (self.an, self.ap, self.bm, self.bq)):
            for x in lis:
                dic[_mod1(x)].append(x)

        for dic, flip in zip(dicts, (True, False, False, True)):
            for m, items in dic.items():
                x0 = items[0]
                items.sort(key=lambda x: x - x0, reverse=flip)
                dic[m] = items

        return tuple([dict(w) for w in dicts])

    @property
    def signature(self):
        return (len(self.an), len(self.ap), len(self.bm), len(self.bq))


# Dummy variable.
_x = Dummy('x')

class Formula:
    """
    This class represents hypergeometric formulae.

    Explanation
    ===========

    Its data members are:
    - z, the argument
    - closed_form, the closed form expression
    - symbols, the free symbols (parameters) in the formula
    - func, the function
    - B, C, M (see _compute_basis)

    Examples
    ========

    >>> from sympy.abc import a, b, z
    >>> from sympy.simplify.hyperexpand import Formula, Hyper_Function
    >>> func = Hyper_Function((a/2, a/3 + b, (1+a)/2), (a, b, (a+b)/7))
    >>> f = Formula(func, z, None, [a, b])

    """

    def _compute_basis(self, closed_form):
        """
        Compute a set of functions B=(f1, ..., fn), a nxn matrix M
        and a 1xn matrix C such that:
           closed_form = C B
           z d/dz B = M B.
        """
        afactors = [_x + a for a in self.func.ap]
        bfactors = [_x + b - 1 for b in self.func.bq]
        expr = _x*Mul(*bfactors) - self.z*Mul(*afactors)
        poly = Poly(expr, _x)

        n = poly.degree() - 1
        b = [closed_form]
        for _ in range(n):
            b.append(self.z*b[-1].diff(self.z))

        self.B = Matrix(b)
        self.C = Matrix([[1] + [0]*n])

        m = eye(n)
        m = m.col_insert(0, zeros(n, 1))
        l = poly.all_coeffs()[1:]
        l.reverse()
        self.M = m.row_insert(n, -Matrix([l])/poly.all_coeffs()[0])

    def __init__(self, func, z, res, symbols, B=None, C=None, M=None):
        z = sympify(z)
        res = sympify(res)
        symbols = [x for x in sympify(symbols) if func.has(x)]

        self.z = z
        self.symbols = symbols
        self.B = B
        self.C = C
        self.M = M
        self.func = func

        # TODO with symbolic parameters, it could be advantageous
        #      (for prettier answers) to compute a basis only *after*
        #      instantiation
        if res is not None:
            self._compute_basis(res)

    @property
    def closed_form(self):
        return reduce(lambda s,m: s+m[0]*m[1], zip(self.C, self.B), S.Zero)

    def find_instantiations(self, func):
        """
        Find substitutions of the free symbols that match ``func``.

        Return the substitution dictionaries as a list. Note that the returned
        instantiations need not actually match, or be valid!

        """
        from sympy.solvers import solve
        ap = func.ap
        bq = func.bq
        if len(ap) != len(self.func.ap) or len(bq) != len(self.func.bq):
            raise TypeError('Cannot instantiate other number of parameters')
        symbol_values = []
        for a in self.symbols:
            if a in self.func.ap.args:
                symbol_values.append(ap)
            elif a in self.func.bq.args:
                symbol_values.append(bq)
            else:
                raise ValueError("At least one of the parameters of the "
                        "formula must be equal to %s" % (a,))
        base_repl = [dict(list(zip(self.symbols, values)))
                for values in product(*symbol_values)]
        abuckets, bbuckets = [sift(params, _mod1) for params in [ap, bq]]
        a_inv, b_inv = [{a: len(vals) for a, vals in bucket.items()}
                for bucket in [abuckets, bbuckets]]
        critical_values = [[0] for _ in self.symbols]
        result = []
        _n = Dummy()
        for repl in base_repl:
            symb_a, symb_b = [sift(params, lambda x: _mod1(x.xreplace(repl)))
                for params in [self.func.ap, self.func.bq]]
            for bucket, obucket in [(abuckets, symb_a), (bbuckets, symb_b)]:
                for mod in set(list(bucket.keys()) + list(obucket.keys())):
                    if (mod not in bucket) or (mod not in obucket) \
                            or len(bucket[mod]) != len(obucket[mod]):
                        break
                    for a, vals in zip(self.symbols, critical_values):
                        if repl[a].free_symbols:
                            continue
                        exprs = [expr for expr in obucket[mod] if expr.has(a)]
                        repl0 = repl.copy()
                        repl0[a] += _n
                        for expr in exprs:
                            for target in bucket[mod]:
                                n0, = solve(expr.xreplace(repl0) - target, _n)
                                if n0.free_symbols:
                                    raise ValueError("Value should not be true")
                                vals.append(n0)
            else:
                values = []
                for a, vals in zip(self.symbols, critical_values):
                    a0 = repl[a]
                    min_ = floor(min(vals))
                    max_ = ceiling(max(vals))
                    values.append([a0 + n for n in range(min_, max_ + 1)])
                result.extend(dict(list(zip(self.symbols, l))) for l in product(*values))
        return result




class FormulaCollection:
    """ A collection of formulae to use as origins. """

    def __init__(self):
        """ Doing this globally at module init time is a pain ... """
        self.symbolic_formulae = {}
        self.concrete_formulae = {}
        self.formulae = []

        add_formulae(self.formulae)

        # Now process the formulae into a helpful form.
        # These dicts are indexed by (p, q).

        for f in self.formulae:
            sizes = f.func.sizes
            if len(f.symbols) > 0:
                self.symbolic_formulae.setdefault(sizes, []).append(f)
            else:
                inv = f.func.build_invariants()
                self.concrete_formulae.setdefault(sizes, {})[inv] = f

    def lookup_origin(self, func):
        """
        Given the suitable target ``func``, try to find an origin in our
        knowledge base.

        Examples
        ========

        >>> from sympy.simplify.hyperexpand import (FormulaCollection,
        ...     Hyper_Function)
        >>> f = FormulaCollection()
        >>> f.lookup_origin(Hyper_Function((), ())).closed_form
        exp(_z)
        >>> f.lookup_origin(Hyper_Function([1], ())).closed_form
        HyperRep_power1(-1, _z)

        >>> from sympy import S
        >>> i = Hyper_Function([S('1/4'), S('3/4 + 4')], [S.Half])
        >>> f.lookup_origin(i).closed_form
        HyperRep_sqrts1(-1/4, _z)
        """
        inv = func.build_invariants()
        sizes = func.sizes
        if sizes in self.concrete_formulae and \
                inv in self.concrete_formulae[sizes]:
            return self.concrete_formulae[sizes][inv]

        # We don't have a concrete formula. Try to instantiate.
        if sizes not in self.symbolic_formulae:
            return None  # Too bad...

        possible = []
        for f in self.symbolic_formulae[sizes]:
            repls = f.find_instantiations(func)
            for repl in repls:
                func2 = f.func.xreplace(repl)
                if not func2._is_suitable_origin():
                    continue
                diff = func2.difficulty(func)
                if diff == -1:
                    continue
                possible.append((diff, repl, f, func2))

        # find the nearest origin
        possible.sort(key=lambda x: x[0])
        for _, repl, f, func2 in possible:
            f2 = Formula(func2, f.z, None, [], f.B.subs(repl),
                    f.C.subs(repl), f.M.subs(repl))
            if not any(e.has(S.NaN, oo, -oo, zoo) for e in [f2.B, f2.M, f2.C]):
                return f2

        return None


class MeijerFormula:
    """
    This class represents a Meijer G-function formula.

    Its data members are:
    - z, the argument
    - symbols, the free symbols (parameters) in the formula
    - func, the function
    - B, C, M (c/f ordinary Formula)
    """

    def __init__(self, an, ap, bm, bq, z, symbols, B, C, M, matcher):
        an, ap, bm, bq = [Tuple(*list(map(expand, w))) for w in [an, ap, bm, bq]]
        self.func = G_Function(an, ap, bm, bq)
        self.z = z
        self.symbols = symbols
        self._matcher = matcher
        self.B = B
        self.C = C
        self.M = M

    @property
    def closed_form(self):
        return reduce(lambda s,m: s+m[0]*m[1], zip(self.C, self.B), S.Zero)

    def try_instantiate(self, func):
        """
        Try to instantiate the current formula to (almost) match func.
        This uses the _matcher passed on init.
        """
        if func.signature != self.func.signature:
            return None
        res = self._matcher(func)
        if res is not None:
            subs, newfunc = res
            return MeijerFormula(newfunc.an, newfunc.ap, newfunc.bm, newfunc.bq,
                                 self.z, [],
                                 self.B.subs(subs), self.C.subs(subs),
                                 self.M.subs(subs), None)


class MeijerFormulaCollection:
    """
    This class holds a collection of meijer g formulae.
    """

    def __init__(self):
        formulae = []
        add_meijerg_formulae(formulae)
        self.formulae = defaultdict(list)
        for formula in formulae:
            self.formulae[formula.func.signature].append(formula)
        self.formulae = dict(self.formulae)

    def lookup_origin(self, func):
        """ Try to find a formula that matches func. """
        if func.signature not in self.formulae:
            return None
        for formula in self.formulae[func.signature]:
            res = formula.try_instantiate(func)
            if res is not None:
                return res


class Operator:
    """
    Base class for operators to be applied to our functions.

    Explanation
    ===========

    These operators are differential operators. They are by convention
    expressed in the variable D = z*d/dz (although this base class does
    not actually care).
    Note that when the operator is applied to an object, we typically do
    *not* blindly differentiate but instead use a different representation
    of the z*d/dz operator (see make_derivative_operator).

    To subclass from this, define a __init__ method that initializes a
    self._poly variable. This variable stores a polynomial. By convention
    the generator is z*d/dz, and acts to the right of all coefficients.

    Thus this poly
        x**2 + 2*z*x + 1
    represents the differential operator
        (z*d/dz)**2 + 2*z**2*d/dz.

    This class is used only in the implementation of the hypergeometric
    function expansion algorithm.
    """

    def apply(self, obj, op):
        """
        Apply ``self`` to the object ``obj``, where the generator is ``op``.

        Examples
        ========

        >>> from sympy.simplify.hyperexpand import Operator
        >>> from sympy.polys.polytools import Poly
        >>> from sympy.abc import x, y, z
        >>> op = Operator()
        >>> op._poly = Poly(x**2 + z*x + y, x)
        >>> op.apply(z**7, lambda f: f.diff(z))
        y*z**7 + 7*z**7 + 42*z**5
        """
        coeffs = self._poly.all_coeffs()
        coeffs.reverse()
        diffs = [obj]
        for c in coeffs[1:]:
            diffs.append(op(diffs[-1]))
        r = coeffs[0]*diffs[0]
        for c, d in zip(coeffs[1:], diffs[1:]):
            r += c*d
        return r


class MultOperator(Operator):
    """ Simply multiply by a "constant" """

    def __init__(self, p):
        self._poly = Poly(p, _x)


class ShiftA(Operator):
    """ Increment an upper index. """

    def __init__(self, ai):
        ai = sympify(ai)
        if ai == 0:
            raise ValueError('Cannot increment zero upper index.')
        self._poly = Poly(_x/ai + 1, _x)

    def __str__(self):
        return '<Increment upper %s.>' % (1/self._poly.all_coeffs()[0])


class ShiftB(Operator):
    """ Decrement a lower index. """

    def __init__(self, bi):
        bi = sympify(bi)
        if bi == 1:
            raise ValueError('Cannot decrement unit lower index.')
        self._poly = Poly(_x/(bi - 1) + 1, _x)

    def __str__(self):
        return '<Decrement lower %s.>' % (1/self._poly.all_coeffs()[0] + 1)


class UnShiftA(Operator):
    """ Decrement an upper index. """

    def __init__(self, ap, bq, i, z):
        """ Note: i counts from zero! """
        ap, bq, i = list(map(sympify, [ap, bq, i]))

        self._ap = ap
        self._bq = bq
        self._i = i

        ap = list(ap)
        bq = list(bq)
        ai = ap.pop(i) - 1

        if ai == 0:
            raise ValueError('Cannot decrement unit upper index.')

        m = Poly(z*ai, _x)
        for a in ap:
            m *= Poly(_x + a, _x)

        A = Dummy('A')
        n = D = Poly(ai*A - ai, A)
        for b in bq:
            n *= D + (b - 1).as_poly(A)

        b0 = -n.nth(0)
        if b0 == 0:
            raise ValueError('Cannot decrement upper index: '
                             'cancels with lower')

        n = Poly(Poly(n.all_coeffs()[:-1], A).as_expr().subs(A, _x/ai + 1), _x)

        self._poly = Poly((n - m)/b0, _x)

    def __str__(self):
        return '<Decrement upper index #%s of %s, %s.>' % (self._i,
                                                        self._ap, self._bq)


class UnShiftB(Operator):
    """ Increment a lower index. """

    def __init__(self, ap, bq, i, z):
        """ Note: i counts from zero! """
        ap, bq, i = list(map(sympify, [ap, bq, i]))

        self._ap = ap
        self._bq = bq
        self._i = i

        ap = list(ap)
        bq = list(bq)
        bi = bq.pop(i) + 1

        if bi == 0:
            raise ValueError('Cannot increment -1 lower index.')

        m = Poly(_x*(bi - 1), _x)
        for b in bq:
            m *= Poly(_x + b - 1, _x)

        B = Dummy('B')
        D = Poly((bi - 1)*B - bi + 1, B)
        n = Poly(z, B)
        for a in ap:
            n *= (D + a.as_poly(B))

        b0 = n.nth(0)
        if b0 == 0:
            raise ValueError('Cannot increment index: cancels with upper')

        n = Poly(Poly(n.all_coeffs()[:-1], B).as_expr().subs(
            B, _x/(bi - 1) + 1), _x)

        self._poly = Poly((m - n)/b0, _x)

    def __str__(self):
        return '<Increment lower index #%s of %s, %s.>' % (self._i,
                                                        self._ap, self._bq)


class MeijerShiftA(Operator):
    """ Increment an upper b index. """

    def __init__(self, bi):
        bi = sympify(bi)
        self._poly = Poly(bi - _x, _x)

    def __str__(self):
        return '<Increment upper b=%s.>' % (self._poly.all_coeffs()[1])


class MeijerShiftB(Operator):
    """ Decrement an upper a index. """

    def __init__(self, bi):
        bi = sympify(bi)
        self._poly = Poly(1 - bi + _x, _x)

    def __str__(self):
        return '<Decrement upper a=%s.>' % (1 - self._poly.all_coeffs()[1])


class MeijerShiftC(Operator):
    """ Increment a lower b index. """

    def __init__(self, bi):
        bi = sympify(bi)
        self._poly = Poly(-bi + _x, _x)

    def __str__(self):
        return '<Increment lower b=%s.>' % (-self._poly.all_coeffs()[1])


class MeijerShiftD(Operator):
    """ Decrement a lower a index. """

    def __init__(self, bi):
        bi = sympify(bi)
        self._poly = Poly(bi - 1 - _x, _x)

    def __str__(self):
        return '<Decrement lower a=%s.>' % (self._poly.all_coeffs()[1] + 1)


class MeijerUnShiftA(Operator):
    """ Decrement an upper b index. """

    def __init__(self, an, ap, bm, bq, i, z):
        """ Note: i counts from zero! """
        an, ap, bm, bq, i = list(map(sympify, [an, ap, bm, bq, i]))

        self._an = an
        self._ap = ap
        self._bm = bm
        self._bq = bq
        self._i = i

        an = list(an)
        ap = list(ap)
        bm = list(bm)
        bq = list(bq)
        bi = bm.pop(i) - 1

        m = Poly(1, _x) * prod(Poly(b - _x, _x) for b in bm) * prod(Poly(_x - b, _x) for b in bq)

        A = Dummy('A')
        D = Poly(bi - A, A)
        n = Poly(z, A) * prod((D + 1 - a) for a in an) * prod((-D + a - 1) for a in ap)

        b0 = n.nth(0)
        if b0 == 0:
            raise ValueError('Cannot decrement upper b index (cancels)')

        n = Poly(Poly(n.all_coeffs()[:-1], A).as_expr().subs(A, bi - _x), _x)

        self._poly = Poly((m - n)/b0, _x)

    def __str__(self):
        return '<Decrement upper b index #%s of %s, %s, %s, %s.>' % (self._i,
                                      self._an, self._ap, self._bm, self._bq)


class MeijerUnShiftB(Operator):
    """ Increment an upper a index. """

    def __init__(self, an, ap, bm, bq, i, z):
        """ Note: i counts from zero! """
        an, ap, bm, bq, i = list(map(sympify, [an, ap, bm, bq, i]))

        self._an = an
        self._ap = ap
        self._bm = bm
        self._bq = bq
        self._i = i

        an = list(an)
        ap = list(ap)
        bm = list(bm)
        bq = list(bq)
        ai = an.pop(i) + 1

        m = Poly(z, _x)
        for a in an:
            m *= Poly(1 - a + _x, _x)
        for a in ap:
            m *= Poly(a - 1 - _x, _x)

        B = Dummy('B')
        D = Poly(B + ai - 1, B)
        n = Poly(1, B)
        for b in bm:
            n *= (-D + b)
        for b in bq:
            n *= (D - b)

        b0 = n.nth(0)
        if b0 == 0:
            raise ValueError('Cannot increment upper a index (cancels)')

        n = Poly(Poly(n.all_coeffs()[:-1], B).as_expr().subs(
            B, 1 - ai + _x), _x)

        self._poly = Poly((m - n)/b0, _x)

    def __str__(self):
        return '<Increment upper a index #%s of %s, %s, %s, %s.>' % (self._i,
                                      self._an, self._ap, self._bm, self._bq)


class MeijerUnShiftC(Operator):
    """ Decrement a lower b index. """
    # XXX this is "essentially" the same as MeijerUnShiftA. This "essentially"
    #     can be made rigorous using the functional equation G(1/z) = G'(z),
    #     where G' denotes a G function of slightly altered parameters.
    #     However, sorting out the details seems harder than just coding it
    #     again.

    def __init__(self, an, ap, bm, bq, i, z):
        """ Note: i counts from zero! """
        an, ap, bm, bq, i = list(map(sympify, [an, ap, bm, bq, i]))

        self._an = an
        self._ap = ap
        self._bm = bm
        self._bq = bq
        self._i = i

        an = list(an)
        ap = list(ap)
        bm = list(bm)
        bq = list(bq)
        bi = bq.pop(i) - 1

        m = Poly(1, _x)
        for b in bm:
            m *= Poly(b - _x, _x)
        for b in bq:
            m *= Poly(_x - b, _x)

        C = Dummy('C')
        D = Poly(bi + C, C)
        n = Poly(z, C)
        for a in an:
            n *= (D + 1 - a)
        for a in ap:
            n *= (-D + a - 1)

        b0 = n.nth(0)
        if b0 == 0:
            raise ValueError('Cannot decrement lower b index (cancels)')

        n = Poly(Poly(n.all_coeffs()[:-1], C).as_expr().subs(C, _x - bi), _x)

        self._poly = Poly((m - n)/b0, _x)

    def __str__(self):
        return '<Decrement lower b index #%s of %s, %s, %s, %s.>' % (self._i,
                                      self._an, self._ap, self._bm, self._bq)


class MeijerUnShiftD(Operator):
    """ Increment a lower a index. """
    # XXX This is essentially the same as MeijerUnShiftA.
    #     See comment at MeijerUnShiftC.

    def __init__(self, an, ap, bm, bq, i, z):
        """ Note: i counts from zero! """
        an, ap, bm, bq, i = list(map(sympify, [an, ap, bm, bq, i]))

        self._an = an
        self._ap = ap
        self._bm = bm
        self._bq = bq
        self._i = i

        an = list(an)
        ap = list(ap)
        bm = list(bm)
        bq = list(bq)
        ai = ap.pop(i) + 1

        m = Poly(z, _x)
        for a in an:
            m *= Poly(1 - a + _x, _x)
        for a in ap:
            m *= Poly(a - 1 - _x, _x)

        B = Dummy('B')  # - this is the shift operator `D_I`
        D = Poly(ai - 1 - B, B)
        n = Poly(1, B)
        for b in bm:
            n *= (-D + b)
        for b in bq:
            n *= (D - b)

        b0 = n.nth(0)
        if b0 == 0:
            raise ValueError('Cannot increment lower a index (cancels)')

        n = Poly(Poly(n.all_coeffs()[:-1], B).as_expr().subs(
            B, ai - 1 - _x), _x)

        self._poly = Poly((m - n)/b0, _x)

    def __str__(self):
        return '<Increment lower a index #%s of %s, %s, %s, %s.>' % (self._i,
                                      self._an, self._ap, self._bm, self._bq)


class ReduceOrder(Operator):
    """ Reduce Order by cancelling an upper and a lower index. """

    def __new__(cls, ai, bj):
        """ For convenience if reduction is not possible, return None. """
        ai = sympify(ai)
        bj = sympify(bj)
        n = ai - bj
        if not n.is_Integer or n < 0:
            return None
        if bj.is_integer and bj.is_nonpositive:
            return None

        expr = Operator.__new__(cls)

        p = S.One
        for k in range(n):
            p *= (_x + bj + k)/(bj + k)

        expr._poly = Poly(p, _x)
        expr._a = ai
        expr._b = bj

        return expr

    @classmethod
    def _meijer(cls, b, a, sign):
        """ Cancel b + sign*s and a + sign*s
            This is for meijer G functions. """
        b = sympify(b)
        a = sympify(a)
        n = b - a
        if n.is_negative or not n.is_Integer:
            return None

        expr = Operator.__new__(cls)

        p = S.One
        for k in range(n):
            p *= (sign*_x + a + k)

        expr._poly = Poly(p, _x)
        if sign == -1:
            expr._a = b
            expr._b = a
        else:
            expr._b = Add(1, a - 1, evaluate=False)
            expr._a = Add(1, b - 1, evaluate=False)

        return expr

    @classmethod
    def meijer_minus(cls, b, a):
        return cls._meijer(b, a, -1)

    @classmethod
    def meijer_plus(cls, a, b):
        return cls._meijer(1 - a, 1 - b, 1)

    def __str__(self):
        return '<Reduce order by cancelling upper %s with lower %s.>' % \
            (self._a, self._b)


def _reduce_order(ap, bq, gen, key):
    """ Order reduction algorithm used in Hypergeometric and Meijer G """
    ap = list(ap)
    bq = list(bq)

    ap.sort(key=key)
    bq.sort(key=key)

    nap = []
    # we will edit bq in place
    operators = []
    for a in ap:
        op = None
        for i in range(len(bq)):
            op = gen(a, bq[i])
            if op is not None:
                bq.pop(i)
                break
        if op is None:
            nap.append(a)
        else:
            operators.append(op)

    return nap, bq, operators


def reduce_order(func):
    """
    Given the hypergeometric function ``func``, find a sequence of operators to
    reduces order as much as possible.

    Explanation
    ===========

    Return (newfunc, [operators]), where applying the operators to the
    hypergeometric function newfunc yields func.

    Examples
    ========

    >>> from sympy.simplify.hyperexpand import reduce_order, Hyper_Function
    >>> reduce_order(Hyper_Function((1, 2), (3, 4)))
    (Hyper_Function((1, 2), (3, 4)), [])
    >>> reduce_order(Hyper_Function((1,), (1,)))
    (Hyper_Function((), ()), [<Reduce order by cancelling upper 1 with lower 1.>])
    >>> reduce_order(Hyper_Function((2, 4), (3, 3)))
    (Hyper_Function((2,), (3,)), [<Reduce order by cancelling
    upper 4 with lower 3.>])
    """
    nap, nbq, operators = _reduce_order(func.ap, func.bq, ReduceOrder, default_sort_key)

    return Hyper_Function(Tuple(*nap), Tuple(*nbq)), operators


def reduce_order_meijer(func):
    """
    Given the Meijer G function parameters, ``func``, find a sequence of
    operators that reduces order as much as possible.

    Return newfunc, [operators].

    Examples
    ========

    >>> from sympy.simplify.hyperexpand import (reduce_order_meijer,
    ...                                         G_Function)
    >>> reduce_order_meijer(G_Function([3, 4], [5, 6], [3, 4], [1, 2]))[0]
    G_Function((4, 3), (5, 6), (3, 4), (2, 1))
    >>> reduce_order_meijer(G_Function([3, 4], [5, 6], [3, 4], [1, 8]))[0]
    G_Function((3,), (5, 6), (3, 4), (1,))
    >>> reduce_order_meijer(G_Function([3, 4], [5, 6], [7, 5], [1, 5]))[0]
    G_Function((3,), (), (), (1,))
    >>> reduce_order_meijer(G_Function([3, 4], [5, 6], [7, 5], [5, 3]))[0]
    G_Function((), (), (), ())
    """

    nan, nbq, ops1 = _reduce_order(func.an, func.bq, ReduceOrder.meijer_plus,
                                   lambda x: default_sort_key(-x))
    nbm, nap, ops2 = _reduce_order(func.bm, func.ap, ReduceOrder.meijer_minus,
                                   default_sort_key)

    return G_Function(nan, nap, nbm, nbq), ops1 + ops2


def make_derivative_operator(M, z):
    """ Create a derivative operator, to be passed to Operator.apply. """
    def doit(C):
        r = z*C.diff(z) + C*M
        r = r.applyfunc(make_simp(z))
        return r
    return doit


def apply_operators(obj, ops, op):
    """
    Apply the list of operators ``ops`` to object ``obj``, substituting
    ``op`` for the generator.
    """
    res = obj
    for o in reversed(ops):
        res = o.apply(res, op)
    return res


def devise_plan(target, origin, z):
    """
    Devise a plan (consisting of shift and un-shift operators) to be applied
    to the hypergeometric function ``target`` to yield ``origin``.
    Returns a list of operators.

    Examples
    ========

    >>> from sympy.simplify.hyperexpand import devise_plan, Hyper_Function
    >>> from sympy.abc import z

    Nothing to do:

    >>> devise_plan(Hyper_Function((1, 2), ()), Hyper_Function((1, 2), ()), z)
    []
    >>> devise_plan(Hyper_Function((), (1, 2)), Hyper_Function((), (1, 2)), z)
    []

    Very simple plans:

    >>> devise_plan(Hyper_Function((2,), ()), Hyper_Function((1,), ()), z)
    [<Increment upper 1.>]
    >>> devise_plan(Hyper_Function((), (2,)), Hyper_Function((), (1,)), z)
    [<Increment lower index #0 of [], [1].>]

    Several buckets:

    >>> from sympy import S
    >>> devise_plan(Hyper_Function((1, S.Half), ()),
    ...             Hyper_Function((2, S('3/2')), ()), z) #doctest: +NORMALIZE_WHITESPACE
    [<Decrement upper index #0 of [3/2, 1], [].>,
    <Decrement upper index #0 of [2, 3/2], [].>]

    A slightly more complicated plan:

    >>> devise_plan(Hyper_Function((1, 3), ()), Hyper_Function((2, 2), ()), z)
    [<Increment upper 2.>, <Decrement upper index #0 of [2, 2], [].>]

    Another more complicated plan: (note that the ap have to be shifted first!)

    >>> devise_plan(Hyper_Function((1, -1), (2,)), Hyper_Function((3, -2), (4,)), z)
    [<Decrement lower 3.>, <Decrement lower 4.>,
    <Decrement upper index #1 of [-1, 2], [4].>,
    <Decrement upper index #1 of [-1, 3], [4].>, <Increment upper -2.>]
    """
    abuckets, bbuckets, nabuckets, nbbuckets = [sift(params, _mod1) for
            params in (target.ap, target.bq, origin.ap, origin.bq)]

    if len(list(abuckets.keys())) != len(list(nabuckets.keys())) or \
            len(list(bbuckets.keys())) != len(list(nbbuckets.keys())):
        raise ValueError('%s not reachable from %s' % (target, origin))

    ops = []

    def do_shifts(fro, to, inc, dec):
        ops = []
        for i in range(len(fro)):
            if to[i] - fro[i] > 0:
                sh = inc
                ch = 1
            else:
                sh = dec
                ch = -1

            while to[i] != fro[i]:
                ops += [sh(fro, i)]
                fro[i] += ch

        return ops

    def do_shifts_a(nal, nbk, al, aother, bother):
        """ Shift us from (nal, nbk) to (al, nbk). """
        return do_shifts(nal, al, lambda p, i: ShiftA(p[i]),
                         lambda p, i: UnShiftA(p + aother, nbk + bother, i, z))

    def do_shifts_b(nal, nbk, bk, aother, bother):
        """ Shift us from (nal, nbk) to (nal, bk). """
        return do_shifts(nbk, bk,
                         lambda p, i: UnShiftB(nal + aother, p + bother, i, z),
                         lambda p, i: ShiftB(p[i]))

    for r in sorted(list(abuckets.keys()) + list(bbuckets.keys()), key=default_sort_key):
        al = ()
        nal = ()
        bk = ()
        nbk = ()
        if r in abuckets:
            al = abuckets[r]
            nal = nabuckets[r]
        if r in bbuckets:
            bk = bbuckets[r]
            nbk = nbbuckets[r]
        if len(al) != len(nal) or len(bk) != len(nbk):
            raise ValueError('%s not reachable from %s' % (target, origin))

        al, nal, bk, nbk = [sorted(w, key=default_sort_key)
            for w in [al, nal, bk, nbk]]

        def others(dic, key):
            l = []
            for k in dic:
                if k != key:
                    l.extend(dic[k])
            return l
        aother = others(nabuckets, r)
        bother = others(nbbuckets, r)

        if len(al) == 0:
            # there can be no complications, just shift the bs as we please
            ops += do_shifts_b([], nbk, bk, aother, bother)
        elif len(bk) == 0:
            # there can be no complications, just shift the as as we please
            ops += do_shifts_a(nal, [], al, aother, bother)
        else:
            namax = nal[-1]
            amax = al[-1]

            if nbk[0] - namax <= 0 or bk[0] - amax <= 0:
                raise ValueError('Non-suitable parameters.')

            if namax - amax > 0:
                # we are going to shift down - first do the as, then the bs
                ops += do_shifts_a(nal, nbk, al, aother, bother)
                ops += do_shifts_b(al, nbk, bk, aother, bother)
            else:
                # we are going to shift up - first do the bs, then the as
                ops += do_shifts_b(nal, nbk, bk, aother, bother)
                ops += do_shifts_a(nal, bk, al, aother, bother)

        nabuckets[r] = al
        nbbuckets[r] = bk

    ops.reverse()
    return ops


def try_shifted_sum(func, z):
    """ Try to recognise a hypergeometric sum that starts from k > 0. """
    abuckets, bbuckets = sift(func.ap, _mod1), sift(func.bq, _mod1)
    if len(abuckets[S.Zero]) != 1:
        return None
    r = abuckets[S.Zero][0]
    if r <= 0:
        return None
    if S.Zero not in bbuckets:
        return None
    l = list(bbuckets[S.Zero])
    l.sort()
    k = l[0]
    if k <= 0:
        return None

    nap = list(func.ap)
    nap.remove(r)
    nbq = list(func.bq)
    nbq.remove(k)
    k -= 1
    nap = [x - k for x in nap]
    nbq = [x - k for x in nbq]

    ops = []
    for n in range(r - 1):
        ops.append(ShiftA(n + 1))
    ops.reverse()

    fac = factorial(k)/z**k
    fac *= Mul(*[rf(b, k) for b in nbq])
    fac /= Mul(*[rf(a, k) for a in nap])

    ops += [MultOperator(fac)]

    p = 0
    for n in range(k):
        m = z**n/factorial(n)
        m *= Mul(*[rf(a, n) for a in nap])
        m /= Mul(*[rf(b, n) for b in nbq])
        p += m

    return Hyper_Function(nap, nbq), ops, -p


def try_polynomial(func, z):
    """ Recognise polynomial cases. Returns None if not such a case.
        Requires order to be fully reduced. """
    abuckets, bbuckets = sift(func.ap, _mod1), sift(func.bq, _mod1)
    a0 = abuckets[S.Zero]
    b0 = bbuckets[S.Zero]
    a0.sort()
    b0.sort()
    al0 = [x for x in a0 if x <= 0]
    bl0 = [x for x in b0 if x <= 0]

    if bl0 and all(a < bl0[-1] for a in al0):
        return oo
    if not al0:
        return None

    a = al0[-1]
    fac = 1
    res = S.One
    for n in Tuple(*list(range(-a))):
        fac *= z
        fac /= n + 1
        fac *= Mul(*[a + n for a in func.ap])
        fac /= Mul(*[b + n for b in func.bq])
        res += fac
    return res


def try_lerchphi(func):
    """
    Try to find an expression for Hyper_Function ``func`` in terms of Lerch
    Transcendents.

    Return None if no such expression can be found.
    """
    # This is actually quite simple, and is described in Roach's paper,
    # section 18.
    # We don't need to implement the reduction to polylog here, this
    # is handled by expand_func.

    # First we need to figure out if the summation coefficient is a rational
    # function of the summation index, and construct that rational function.
    abuckets, bbuckets = sift(func.ap, _mod1), sift(func.bq, _mod1)

    paired = {}
    for key, value in abuckets.items():
        if key != 0 and key not in bbuckets:
            return None
        bvalue = bbuckets[key]
        paired[key] = (list(value), list(bvalue))
        bbuckets.pop(key, None)
    if bbuckets != {}:
        return None
    if S.Zero not in abuckets:
        return None
    aints, bints = paired[S.Zero]
    # Account for the additional n! in denominator
    paired[S.Zero] = (aints, bints + [1])

    t = Dummy('t')
    numer = S.One
    denom = S.One
    for key, (avalue, bvalue) in paired.items():
        if len(avalue) != len(bvalue):
            return None
        # Note that since order has been reduced fully, all the b are
        # bigger than all the a they differ from by an integer. In particular
        # if there are any negative b left, this function is not well-defined.
        for a, b in zip(avalue, bvalue):
            if (a - b).is_positive:
                k = a - b
                numer *= rf(b + t, k)
                denom *= rf(b, k)
            else:
                k = b - a
                numer *= rf(a, k)
                denom *= rf(a + t, k)

    # Now do a partial fraction decomposition.
    # We assemble two structures: a list monomials of pairs (a, b) representing
    # a*t**b (b a non-negative integer), and a dict terms, where
    # terms[a] = [(b, c)] means that there is a term b/(t-a)**c.
    part = apart(numer/denom, t)
    args = Add.make_args(part)
    monomials = []
    terms = {}
    for arg in args:
        numer, denom = arg.as_numer_denom()
        if not denom.has(t):
            p = Poly(numer, t)
            if not p.is_monomial:
                raise TypeError("p should be monomial")
            ((b, ), a) = p.LT()
            monomials += [(a/denom, b)]
            continue
        if numer.has(t):
            raise NotImplementedError('Need partial fraction decomposition'
                                      ' with linear denominators')
        indep, [dep] = denom.as_coeff_mul(t)
        n = 1
        if dep.is_Pow:
            n = dep.exp
            dep = dep.base
        if dep == t:
            a = 0
        elif dep.is_Add:
            a, tmp = dep.as_independent(t)
            b = 1
            if tmp != t:
                b, _ = tmp.as_independent(t)
            if dep != b*t + a:
                raise NotImplementedError('unrecognised form %s' % dep)
            a /= b
            indep *= b**n
        else:
            raise NotImplementedError('unrecognised form of partial fraction')
        terms.setdefault(a, []).append((numer/indep, n))

    # Now that we have this information, assemble our formula. All the
    # monomials yield rational functions and go into one basis element.
    # The terms[a] are related by differentiation. If the largest exponent is
    # n, we need lerchphi(z, k, a) for k = 1, 2, ..., n.
    # deriv maps a basis to its derivative, expressed as a C(z)-linear
    # combination of other basis elements.
    deriv = {}
    coeffs = {}
    z = Dummy('z')
    monomials.sort(key=lambda x: x[1])
    mon = {0: 1/(1 - z)}
    if monomials:
        for k in range(monomials[-1][1]):
            mon[k + 1] = z*mon[k].diff(z)
    for a, n in monomials:
        coeffs.setdefault(S.One, []).append(a*mon[n])
    for a, l in terms.items():
        for c, k in l:
            coeffs.setdefault(lerchphi(z, k, a), []).append(c)
        l.sort(key=lambda x: x[1])
        for k in range(2, l[-1][1] + 1):
            deriv[lerchphi(z, k, a)] = [(-a, lerchphi(z, k, a)),
                                        (1, lerchphi(z, k - 1, a))]
        deriv[lerchphi(z, 1, a)] = [(-a, lerchphi(z, 1, a)),
                                    (1/(1 - z), S.One)]
    trans = {}
    for n, b in enumerate([S.One] + list(deriv.keys())):
        trans[b] = n
    basis = [expand_func(b) for (b, _) in sorted(trans.items(),
                                                 key=lambda x:x[1])]
    B = Matrix(basis)
    C = Matrix([[0]*len(B)])
    for b, c in coeffs.items():
        C[trans[b]] = Add(*c)
    M = zeros(len(B))
    for b, l in deriv.items():
        for c, b2 in l:
            M[trans[b], trans[b2]] = c
    return Formula(func, z, None, [], B, C, M)


def build_hypergeometric_formula(func):
    """
    Create a formula object representing the hypergeometric function ``func``.

    """
    # We know that no `ap` are negative integers, otherwise "detect poly"
    # would have kicked in. However, `ap` could be empty. In this case we can
    # use a different basis.
    # I'm not aware of a basis that works in all cases.
    z = Dummy('z')
    if func.ap:
        afactors = [_x + a for a in func.ap]
        bfactors = [_x + b - 1 for b in func.bq]
        expr = _x*Mul(*bfactors) - z*Mul(*afactors)
        poly = Poly(expr, _x)
        n = poly.degree()
        basis = []
        M = zeros(n)
        for k in range(n):
            a = func.ap[0] + k
            basis += [hyper([a] + list(func.ap[1:]), func.bq, z)]
            if k < n - 1:
                M[k, k] = -a
                M[k, k + 1] = a
        B = Matrix(basis)
        C = Matrix([[1] + [0]*(n - 1)])
        derivs = [eye(n)]
        for k in range(n):
            derivs.append(M*derivs[k])
        l = poly.all_coeffs()
        l.reverse()
        res = [0]*n
        for k, c in enumerate(l):
            for r, d in enumerate(C*derivs[k]):
                res[r] += c*d
        for k, c in enumerate(res):
            M[n - 1, k] = -c/derivs[n - 1][0, n - 1]/poly.all_coeffs()[0]
        return Formula(func, z, None, [], B, C, M)
    else:
        # Since there are no `ap`, none of the `bq` can be non-positive
        # integers.
        basis = []
        bq = list(func.bq[:])
        for i in range(len(bq)):
            basis += [hyper([], bq, z)]
            bq[i] += 1
        basis += [hyper([], bq, z)]
        B = Matrix(basis)
        n = len(B)
        C = Matrix([[1] + [0]*(n - 1)])
        M = zeros(n)
        M[0, n - 1] = z/Mul(*func.bq)
        for k in range(1, n):
            M[k, k - 1] = func.bq[k - 1]
            M[k, k] = -func.bq[k - 1]
        return Formula(func, z, None, [], B, C, M)


def hyperexpand_special(ap, bq, z):
    """
    Try to find a closed-form expression for hyper(ap, bq, z), where ``z``
    is supposed to be a "special" value, e.g. 1.

    This function tries various of the classical summation formulae
    (Gauss, Saalschuetz, etc).
    """
    # This code is very ad-hoc. There are many clever algorithms
    # (notably Zeilberger's) related to this problem.
    # For now we just want a few simple cases to work.
    p, q = len(ap), len(bq)
    z_ = z
    z = unpolarify(z)
    if z == 0:
        return S.One
    from sympy.simplify.simplify import simplify
    if p == 2 and q == 1:
        # 2F1
        a, b, c = ap + bq
        if z == 1:
            # Gauss
            return gamma(c - a - b)*gamma(c)/gamma(c - a)/gamma(c - b)
        if z == -1 and simplify(b - a + c) == 1:
            b, a = a, b
        if z == -1 and simplify(a - b + c) == 1:
            # Kummer
            if b.is_integer and b.is_negative:
                return 2*cos(pi*b/2)*gamma(-b)*gamma(b - a + 1) \
                    /gamma(-b/2)/gamma(b/2 - a + 1)
            else:
                return gamma(b/2 + 1)*gamma(b - a + 1) \
                    /gamma(b + 1)/gamma(b/2 - a + 1)
    # TODO tons of more formulae
    #      investigate what algorithms exist
    return hyper(ap, bq, z_)

_collection = None


def _hyperexpand(func, z, ops0=[], z0=Dummy('z0'), premult=1, prem=0,
                 rewrite='default'):
    """
    Try to find an expression for the hypergeometric function ``func``.

    Explanation
    ===========

    The result is expressed in terms of a dummy variable ``z0``. Then it
    is multiplied by ``premult``. Then ``ops0`` is applied.
    ``premult`` must be a*z**prem for some a independent of ``z``.
    """

    if z.is_zero:
        return S.One

    from sympy.simplify.simplify import simplify

    z = polarify(z, subs=False)
    if rewrite == 'default':
        rewrite = 'nonrepsmall'

    def carryout_plan(f, ops):
        C = apply_operators(f.C.subs(f.z, z0), ops,
                            make_derivative_operator(f.M.subs(f.z, z0), z0))
        C = apply_operators(C, ops0,
                            make_derivative_operator(f.M.subs(f.z, z0)
                                         + prem*eye(f.M.shape[0]), z0))

        if premult == 1:
            C = C.applyfunc(make_simp(z0))
        r = reduce(lambda s,m: s+m[0]*m[1], zip(C, f.B.subs(f.z, z0)), S.Zero)*premult
        res = r.subs(z0, z)
        if rewrite:
            res = res.rewrite(rewrite)
        return res

    # TODO
    # The following would be possible:
    # *) PFD Duplication (see Kelly Roach's paper)
    # *) In a similar spirit, try_lerchphi() can be generalised considerably.

    global _collection
    if _collection is None:
        _collection = FormulaCollection()

    debug('Trying to expand hypergeometric function ', func)

    # First reduce order as much as possible.
    func, ops = reduce_order(func)
    if ops:
        debug('  Reduced order to ', func)
    else:
        debug('  Could not reduce order.')

    # Now try polynomial cases
    res = try_polynomial(func, z0)
    if res is not None:
        debug('  Recognised polynomial.')
        p = apply_operators(res, ops, lambda f: z0*f.diff(z0))
        p = apply_operators(p*premult, ops0, lambda f: z0*f.diff(z0))
        return unpolarify(simplify(p).subs(z0, z))

    # Try to recognise a shifted sum.
    p = S.Zero
    res = try_shifted_sum(func, z0)
    if res is not None:
        func, nops, p = res
        debug('  Recognised shifted sum, reduced order to ', func)
        ops += nops

    # apply the plan for poly
    p = apply_operators(p, ops, lambda f: z0*f.diff(z0))
    p = apply_operators(p*premult, ops0, lambda f: z0*f.diff(z0))
    p = simplify(p).subs(z0, z)

    # Try special expansions early.
    if unpolarify(z) in [1, -1] and (len(func.ap), len(func.bq)) == (2, 1):
        f = build_hypergeometric_formula(func)
        r = carryout_plan(f, ops).replace(hyper, hyperexpand_special)
        if not r.has(hyper):
            return r + p

    # Try to find a formula in our collection
    formula = _collection.lookup_origin(func)

    # Now try a lerch phi formula
    if formula is None:
        formula = try_lerchphi(func)

    if formula is None:
        debug('  Could not find an origin. ',
              'Will return answer in terms of '
              'simpler hypergeometric functions.')
        formula = build_hypergeometric_formula(func)

    debug('  Found an origin: ', formula.closed_form, ' ', formula.func)

    # We need to find the operators that convert formula into func.
    ops += devise_plan(func, formula.func, z0)

    # Now carry out the plan.
    r = carryout_plan(formula, ops) + p

    return powdenest(r, polar=True).replace(hyper, hyperexpand_special)


def devise_plan_meijer(fro, to, z):
    """
    Find operators to convert G-function ``fro`` into G-function ``to``.

    Explanation
    ===========

    It is assumed that ``fro`` and ``to`` have the same signatures, and that in fact
    any corresponding pair of parameters differs by integers, and a direct path
    is possible. I.e. if there are parameters a1 b1 c1  and a2 b2 c2 it is
    assumed that a1 can be shifted to a2, etc. The only thing this routine
    determines is the order of shifts to apply, nothing clever will be tried.
    It is also assumed that ``fro`` is suitable.

    Examples
    ========

    >>> from sympy.simplify.hyperexpand import (devise_plan_meijer,
    ...                                         G_Function)
    >>> from sympy.abc import z

    Empty plan:

    >>> devise_plan_meijer(G_Function([1], [2], [3], [4]),
    ...                    G_Function([1], [2], [3], [4]), z)
    []

    Very simple plans:

    >>> devise_plan_meijer(G_Function([0], [], [], []),
    ...                    G_Function([1], [], [], []), z)
    [<Increment upper a index #0 of [0], [], [], [].>]
    >>> devise_plan_meijer(G_Function([0], [], [], []),
    ...                    G_Function([-1], [], [], []), z)
    [<Decrement upper a=0.>]
    >>> devise_plan_meijer(G_Function([], [1], [], []),
    ...                    G_Function([], [2], [], []), z)
    [<Increment lower a index #0 of [], [1], [], [].>]

    Slightly more complicated plans:

    >>> devise_plan_meijer(G_Function([0], [], [], []),
    ...                    G_Function([2], [], [], []), z)
    [<Increment upper a index #0 of [1], [], [], [].>,
    <Increment upper a index #0 of [0], [], [], [].>]
    >>> devise_plan_meijer(G_Function([0], [], [0], []),
    ...                    G_Function([-1], [], [1], []), z)
    [<Increment upper b=0.>, <Decrement upper a=0.>]

    Order matters:

    >>> devise_plan_meijer(G_Function([0], [], [0], []),
    ...                    G_Function([1], [], [1], []), z)
    [<Increment upper a index #0 of [0], [], [1], [].>, <Increment upper b=0.>]
    """
    # TODO for now, we use the following simple heuristic: inverse-shift
    #      when possible, shift otherwise. Give up if we cannot make progress.

    def try_shift(f, t, shifter, diff, counter):
        """ Try to apply ``shifter`` in order to bring some element in ``f``
            nearer to its counterpart in ``to``. ``diff`` is +/- 1 and
            determines the effect of ``shifter``. Counter is a list of elements
            blocking the shift.

            Return an operator if change was possible, else None.
        """
        for idx, (a, b) in enumerate(zip(f, t)):
            if (
                (a - b).is_integer and (b - a)/diff > 0 and
                    all(a != x for x in counter)):
                sh = shifter(idx)
                f[idx] += diff
                return sh
    fan = list(fro.an)
    fap = list(fro.ap)
    fbm = list(fro.bm)
    fbq = list(fro.bq)
    ops = []
    change = True
    while change:
        change = False
        op = try_shift(fan, to.an,
                       lambda i: MeijerUnShiftB(fan, fap, fbm, fbq, i, z),
                       1, fbm + fbq)
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fap, to.ap,
                       lambda i: MeijerUnShiftD(fan, fap, fbm, fbq, i, z),
                       1, fbm + fbq)
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fbm, to.bm,
                       lambda i: MeijerUnShiftA(fan, fap, fbm, fbq, i, z),
                       -1, fan + fap)
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fbq, to.bq,
                       lambda i: MeijerUnShiftC(fan, fap, fbm, fbq, i, z),
                       -1, fan + fap)
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fan, to.an, lambda i: MeijerShiftB(fan[i]), -1, [])
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fap, to.ap, lambda i: MeijerShiftD(fap[i]), -1, [])
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fbm, to.bm, lambda i: MeijerShiftA(fbm[i]), 1, [])
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fbq, to.bq, lambda i: MeijerShiftC(fbq[i]), 1, [])
        if op is not None:
            ops += [op]
            change = True
            continue
    if fan != list(to.an) or fap != list(to.ap) or fbm != list(to.bm) or \
            fbq != list(to.bq):
        raise NotImplementedError('Could not devise plan.')
    ops.reverse()
    return ops

_meijercollection = None


def _meijergexpand(func, z0, allow_hyper=False, rewrite='default',
                   place=None):
    """
    Try to find an expression for the Meijer G function specified
    by the G_Function ``func``. If ``allow_hyper`` is True, then returning
    an expression in terms of hypergeometric functions is allowed.

    Currently this just does Slater's theorem.
    If expansions exist both at zero and at infinity, ``place``
    can be set to ``0`` or ``zoo`` for the preferred choice.
    """
    global _meijercollection
    if _meijercollection is None:
        _meijercollection = MeijerFormulaCollection()
    if rewrite == 'default':
        rewrite = None

    func0 = func
    debug('Try to expand Meijer G function corresponding to ', func)

    # We will play games with analytic continuation - rather use a fresh symbol
    z = Dummy('z')

    func, ops = reduce_order_meijer(func)
    if ops:
        debug('  Reduced order to ', func)
    else:
        debug('  Could not reduce order.')

    # Try to find a direct formula
    f = _meijercollection.lookup_origin(func)
    if f is not None:
        debug('  Found a Meijer G formula: ', f.func)
        ops += devise_plan_meijer(f.func, func, z)

        # Now carry out the plan.
        C = apply_operators(f.C.subs(f.z, z), ops,
                            make_derivative_operator(f.M.subs(f.z, z), z))

        C = C.applyfunc(make_simp(z))
        r = C*f.B.subs(f.z, z)
        r = r[0].subs(z, z0)
        return powdenest(r, polar=True)

    debug("  Could not find a direct formula. Trying Slater's theorem.")

    # TODO the following would be possible:
    # *) Paired Index Theorems
    # *) PFD Duplication
    #    (See Kelly Roach's paper for details on either.)
    #
    # TODO Also, we tend to create combinations of gamma functions that can be
    #      simplified.

    def can_do(pbm, pap):
        """ Test if slater applies. """
        for i in pbm:
            if len(pbm[i]) > 1:
                l = 0
                if i in pap:
                    l = len(pap[i])
                if l + 1 < len(pbm[i]):
                    return False
        return True

    def do_slater(an, bm, ap, bq, z, zfinal):
        # zfinal is the value that will eventually be substituted for z.
        # We pass it to _hyperexpand to improve performance.
        func = G_Function(an, bm, ap, bq)
        _, pbm, pap, _ = func.compute_buckets()
        if not can_do(pbm, pap):
            return S.Zero, False

        cond = len(an) + len(ap) < len(bm) + len(bq)
        if len(an) + len(ap) == len(bm) + len(bq):
            cond = abs(z) < 1
        if cond is False:
            return S.Zero, False

        res = S.Zero
        for m in pbm:
            if len(pbm[m]) == 1:
                bh = pbm[m][0]
                fac = 1
                bo = list(bm)
                bo.remove(bh)
                for bj in bo:
                    fac *= gamma(bj - bh)
                for aj in an:
                    fac *= gamma(1 + bh - aj)
                for bj in bq:
                    fac /= gamma(1 + bh - bj)
                for aj in ap:
                    fac /= gamma(aj - bh)
                nap = [1 + bh - a for a in list(an) + list(ap)]
                nbq = [1 + bh - b for b in list(bo) + list(bq)]

                k = polar_lift(S.NegativeOne**(len(ap) - len(bm)))
                harg = k*zfinal
                # NOTE even though k "is" +-1, this has to be t/k instead of
                #      t*k ... we are using polar numbers for consistency!
                premult = (t/k)**bh
                hyp = _hyperexpand(Hyper_Function(nap, nbq), harg, ops,
                                   t, premult, bh, rewrite=None)
                res += fac * hyp
            else:
                b_ = pbm[m][0]
                ki = [bi - b_ for bi in pbm[m][1:]]
                u = len(ki)
                li = [ai - b_ for ai in pap[m][:u + 1]]
                bo = list(bm)
                for b in pbm[m]:
                    bo.remove(b)
                ao = list(ap)
                for a in pap[m][:u]:
                    ao.remove(a)
                lu = li[-1]
                di = [l - k for (l, k) in zip(li, ki)]

                # We first work out the integrand:
                s = Dummy('s')
                integrand = z**s
                for b in bm:
                    if not Mod(b, 1) and b.is_Number:
                        b = int(round(b))
                    integrand *= gamma(b - s)
                for a in an:
                    integrand *= gamma(1 - a + s)
                for b in bq:
                    integrand /= gamma(1 - b + s)
                for a in ap:
                    integrand /= gamma(a - s)

                # Now sum the finitely many residues:
                # XXX This speeds up some cases - is it a good idea?
                integrand = expand_func(integrand)
                for r in range(int(round(lu))):
                    resid = residue(integrand, s, b_ + r)
                    resid = apply_operators(resid, ops, lambda f: z*f.diff(z))
                    res -= resid

                # Now the hypergeometric term.
                au = b_ + lu
                k = polar_lift(S.NegativeOne**(len(ao) + len(bo) + 1))
                harg = k*zfinal
                premult = (t/k)**au
                nap = [1 + au - a for a in list(an) + list(ap)] + [1]
                nbq = [1 + au - b for b in list(bm) + list(bq)]

                hyp = _hyperexpand(Hyper_Function(nap, nbq), harg, ops,
                                   t, premult, au, rewrite=None)

                C = S.NegativeOne**(lu)/factorial(lu)
                for i in range(u):
                    C *= S.NegativeOne**di[i]/rf(lu - li[i] + 1, di[i])
                for a in an:
                    C *= gamma(1 - a + au)
                for b in bo:
                    C *= gamma(b - au)
                for a in ao:
                    C /= gamma(a - au)
                for b in bq:
                    C /= gamma(1 - b + au)

                res += C*hyp

        return res, cond

    t = Dummy('t')
    slater1, cond1 = do_slater(func.an, func.bm, func.ap, func.bq, z, z0)

    def tr(l):
        return [1 - x for x in l]

    for op in ops:
        op._poly = Poly(op._poly.subs({z: 1/t, _x: -_x}), _x)
    slater2, cond2 = do_slater(tr(func.bm), tr(func.an), tr(func.bq), tr(func.ap),
                               t, 1/z0)

    slater1 = powdenest(slater1.subs(z, z0), polar=True)
    slater2 = powdenest(slater2.subs(t, 1/z0), polar=True)
    if not isinstance(cond2, bool):
        cond2 = cond2.subs(t, 1/z)

    m = func(z)
    if m.delta > 0 or \
        (m.delta == 0 and len(m.ap) == len(m.bq) and
            (re(m.nu) < -1) is not False and polar_lift(z0) == polar_lift(1)):
        # The condition delta > 0 means that the convergence region is
        # connected. Any expression we find can be continued analytically
        # to the entire convergence region.
        # The conditions delta==0, p==q, re(nu) < -1 imply that G is continuous
        # on the positive reals, so the values at z=1 agree.
        if cond1 is not False:
            cond1 = True
        if cond2 is not False:
            cond2 = True

    if cond1 is True:
        slater1 = slater1.rewrite(rewrite or 'nonrep')
    else:
        slater1 = slater1.rewrite(rewrite or 'nonrepsmall')
    if cond2 is True:
        slater2 = slater2.rewrite(rewrite or 'nonrep')
    else:
        slater2 = slater2.rewrite(rewrite or 'nonrepsmall')

    if cond1 is not False and cond2 is not False:
        # If one condition is False, there is no choice.
        if place == 0:
            cond2 = False
        if place == zoo:
            cond1 = False

    if not isinstance(cond1, bool):
        cond1 = cond1.subs(z, z0)
    if not isinstance(cond2, bool):
        cond2 = cond2.subs(z, z0)

    def weight(expr, cond):
        if cond is True:
            c0 = 0
        elif cond is False:
            c0 = 1
        else:
            c0 = 2
        if expr.has(oo, zoo, -oo, nan):
            # XXX this actually should not happen, but consider
            # S('meijerg(((0, -1/2, 0, -1/2, 1/2), ()), ((0,),
            #   (-1/2, -1/2, -1/2, -1)), exp_polar(I*pi))/4')
            c0 = 3
        return (c0, expr.count(hyper), expr.count_ops())

    w1 = weight(slater1, cond1)
    w2 = weight(slater2, cond2)
    if min(w1, w2) <= (0, 1, oo):
        if w1 < w2:
            return slater1
        else:
            return slater2
    if max(w1[0], w2[0]) <= 1 and max(w1[1], w2[1]) <= 1:
        return Piecewise((slater1, cond1), (slater2, cond2), (func0(z0), True))

    # We couldn't find an expression without hypergeometric functions.
    # TODO it would be helpful to give conditions under which the integral
    #      is known to diverge.
    r = Piecewise((slater1, cond1), (slater2, cond2), (func0(z0), True))
    if r.has(hyper) and not allow_hyper:
        debug('  Could express using hypergeometric functions, '
              'but not allowed.')
    if not r.has(hyper) or allow_hyper:
        return r

    return func0(z0)


def hyperexpand(f, allow_hyper=False, rewrite='default', place=None):
    """
    Expand hypergeometric functions. If allow_hyper is True, allow partial
    simplification (that is a result different from input,
    but still containing hypergeometric functions).

    If a G-function has expansions both at zero and at infinity,
    ``place`` can be set to ``0`` or ``zoo`` to indicate the
    preferred choice.

    Examples
    ========

    >>> from sympy.simplify.hyperexpand import hyperexpand
    >>> from sympy.functions import hyper
    >>> from sympy.abc import z
    >>> hyperexpand(hyper([], [], z))
    exp(z)

    Non-hyperegeometric parts of the expression and hypergeometric expressions
    that are not recognised are left unchanged:

    >>> hyperexpand(1 + hyper([1, 1, 1], [], z))
    hyper((1, 1, 1), (), z) + 1
    """
    f = sympify(f)

    def do_replace(ap, bq, z):
        r = _hyperexpand(Hyper_Function(ap, bq), z, rewrite=rewrite)
        if r is None:
            return hyper(ap, bq, z)
        else:
            return r

    def do_meijer(ap, bq, z):
        r = _meijergexpand(G_Function(ap[0], ap[1], bq[0], bq[1]), z,
                   allow_hyper, rewrite=rewrite, place=place)
        if not r.has(nan, zoo, oo, -oo):
            return r
    return f.replace(hyper, do_replace).replace(meijerg, do_meijer)
