from sympy.integrals.laplace import (
    laplace_transform, inverse_laplace_transform,
    LaplaceTransform, InverseLaplaceTransform,
    _laplace_deep_collect, laplace_correspondence,
    laplace_initial_conds)
from sympy.core.function import Function, expand_mul
from sympy.core import EulerGamma, Subs, Derivative, diff
from sympy.core.exprtools import factor_terms
from sympy.core.numbers import I, oo, pi
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, symbols
from sympy.simplify.simplify import simplify
from sympy.functions.elementary.complexes import Abs, re
from sympy.functions.elementary.exponential import exp, log, exp_polar
from sympy.functions.elementary.hyperbolic import cosh, sinh, coth, asinh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import atan, cos, sin
from sympy.logic.boolalg import And
from sympy.functions.special.gamma_functions import (
    lowergamma, gamma, uppergamma)
from sympy.functions.special.delta_functions import DiracDelta, Heaviside
from sympy.functions.special.singularity_functions import SingularityFunction
from sympy.functions.special.zeta_functions import lerchphi
from sympy.functions.special.error_functions import (
    fresnelc, fresnels, erf, erfc, Ei, Ci, expint, E1)
from sympy.functions.special.bessel import besseli, besselj, besselk, bessely
from sympy.testing.pytest import slow, warns_deprecated_sympy
from sympy.matrices import Matrix, eye
from sympy.abc import s


@slow
def test_laplace_transform():
    LT = laplace_transform
    ILT = inverse_laplace_transform
    a, b, c = symbols('a, b, c', positive=True)
    np = symbols('np', integer=True, positive=True)
    t, w, x = symbols('t, w, x')
    f = Function('f')
    F = Function('F')
    g = Function('g')
    y = Function('y')
    Y = Function('Y')

    # Test helper functions
    assert (
        _laplace_deep_collect(exp((t+a)*(t+b)) +
                              besselj(2, exp((t+a)*(t+b)-t**2)), t) ==
        exp(a*b + t**2 + t*(a + b)) + besselj(2, exp(a*b + t*(a + b))))
    L = laplace_transform(diff(y(t), t, 3), t, s, noconds=True)
    L = laplace_correspondence(L, {y: Y})
    L = laplace_initial_conds(L, t, {y: [2, 4, 8, 16, 32]})
    assert L == s**3*Y(s) - 2*s**2 - 4*s - 8
    # Test whether `noconds=True` in `doit`:
    assert (2*LaplaceTransform(exp(t), t, s) - 1).doit() == -1 + 2/(s - 1)
    assert (LT(a*t+t**2+t**(S(5)/2), t, s) ==
            (a/s**2 + 2/s**3 + 15*sqrt(pi)/(8*s**(S(7)/2)), 0, True))
    assert LT(b/(t+a), t, s) == (-b*exp(-a*s)*Ei(-a*s), 0, True)
    assert (LT(1/sqrt(t+a), t, s) ==
            (sqrt(pi)*sqrt(1/s)*exp(a*s)*erfc(sqrt(a)*sqrt(s)), 0, True))
    assert (LT(sqrt(t)/(t+a), t, s) ==
            (-pi*sqrt(a)*exp(a*s)*erfc(sqrt(a)*sqrt(s)) + sqrt(pi)*sqrt(1/s),
             0, True))
    assert (LT((t+a)**(-S(3)/2), t, s) ==
            (-2*sqrt(pi)*sqrt(s)*exp(a*s)*erfc(sqrt(a)*sqrt(s)) + 2/sqrt(a),
             0, True))
    assert (LT(t**(S(1)/2)*(t+a)**(-1), t, s) ==
            (-pi*sqrt(a)*exp(a*s)*erfc(sqrt(a)*sqrt(s)) + sqrt(pi)*sqrt(1/s),
             0, True))
    assert (LT(1/(a*sqrt(t) + t**(3/2)), t, s) ==
            (pi*sqrt(a)*exp(a*s)*erfc(sqrt(a)*sqrt(s)), 0, True))
    assert (LT((t+a)**b, t, s) ==
            (s**(-b - 1)*exp(-a*s)*uppergamma(b + 1, a*s), 0, True))
    assert LT(t**5/(t+a), t, s) == (120*a**5*uppergamma(-5, a*s), 0, True)
    assert LT(exp(t), t, s) == (1/(s - 1), 1, True)
    assert LT(exp(2*t), t, s) == (1/(s - 2), 2, True)
    assert LT(exp(a*t), t, s) == (1/(s - a), a, True)
    assert LT(exp(a*(t-b)), t, s) == (exp(-a*b)/(-a + s), a, True)
    assert LT(t*exp(-a*(t)), t, s) == ((a + s)**(-2), -a, True)
    assert LT(t*exp(-a*(t-b)), t, s) == (exp(a*b)/(a + s)**2, -a, True)
    assert LT(b*t*exp(-a*t), t, s) == (b/(a + s)**2, -a, True)
    assert LT(exp(-a*exp(-t)), t, s) == (lowergamma(s, a)/a**s, 0, True)
    assert LT(exp(-a*exp(t)), t, s) == (a**s*uppergamma(-s, a), 0, True)
    assert (LT(t**(S(7)/4)*exp(-8*t)/gamma(S(11)/4), t, s) ==
            ((s + 8)**(-S(11)/4), -8, True))
    assert (LT(t**(S(3)/2)*exp(-8*t), t, s) ==
            (3*sqrt(pi)/(4*(s + 8)**(S(5)/2)), -8, True))
    assert LT(t**a*exp(-a*t), t, s) == ((a+s)**(-a-1)*gamma(a+1), -a, True)
    assert (LT(b*exp(-a*t**2), t, s) ==
            (sqrt(pi)*b*exp(s**2/(4*a))*erfc(s/(2*sqrt(a)))/(2*sqrt(a)),
             0, True))
    assert (LT(exp(-2*t**2), t, s) ==
            (sqrt(2)*sqrt(pi)*exp(s**2/8)*erfc(sqrt(2)*s/4)/4, 0, True))
    assert (LT(b*exp(2*t**2), t, s) ==
            (b*LaplaceTransform(exp(2*t**2), t, s), -oo, True))
    assert (LT(t*exp(-a*t**2), t, s) ==
            (1/(2*a) - s*erfc(s/(2*sqrt(a)))/(4*sqrt(pi)*a**(S(3)/2)),
             0, True))
    assert (LT(exp(-a/t), t, s) ==
            (2*sqrt(a)*sqrt(1/s)*besselk(1, 2*sqrt(a)*sqrt(s)), 0, True))
    assert LT(sqrt(t)*exp(-a/t), t, s, simplify=True) == (
        sqrt(pi)*(sqrt(a)*sqrt(s) + 1/S(2))*sqrt(s**(-3)) *
        exp(-2*sqrt(a)*sqrt(s)), 0, True)
    assert (LT(exp(-a/t)/sqrt(t), t, s) ==
            (sqrt(pi)*sqrt(1/s)*exp(-2*sqrt(a)*sqrt(s)), 0, True))
    assert (LT(exp(-a/t)/(t*sqrt(t)), t, s) ==
            (sqrt(pi)*sqrt(1/a)*exp(-2*sqrt(a)*sqrt(s)), 0, True))
    # TODO: rules with sqrt(a*t) and sqrt(a/t) have stopped working after
    #       changes to as_base_exp
    # assert (
    #     LT(exp(-2*sqrt(a*t)), t, s) ==
    #     (1/s - sqrt(pi)*sqrt(a) * exp(a/s)*erfc(sqrt(a)*sqrt(1/s)) /
    #      s**(S(3)/2), 0, True))
    # assert LT(exp(-2*sqrt(a*t))/sqrt(t), t, s) == (
    #     exp(a/s)*erfc(sqrt(a) * sqrt(1/s))*(sqrt(pi)*sqrt(1/s)), 0, True)
    assert (LT(t**4*exp(-2/t), t, s) ==
            (8*sqrt(2)*(1/s)**(S(5)/2)*besselk(5, 2*sqrt(2)*sqrt(s)),
             0, True))
    assert LT(sinh(a*t), t, s) == (a/(-a**2 + s**2), a, True)
    assert (LT(b*sinh(a*t)**2, t, s) ==
            (2*a**2*b/(-4*a**2*s + s**3), 2*a, True))
    assert (LT(b*sinh(a*t)**2, t, s, simplify=True) ==
            (2*a**2*b/(s*(-4*a**2 + s**2)), 2*a, True))
    # The following line confirms that issue #21202 is solved
    assert LT(cosh(2*t), t, s) == (s/(-4 + s**2), 2, True)
    assert LT(cosh(a*t), t, s) == (s/(-a**2 + s**2), a, True)
    assert (LT(cosh(a*t)**2, t, s, simplify=True) ==
            ((2*a**2 - s**2)/(s*(4*a**2 - s**2)), 2*a, True))
    assert (LT(sinh(x+3), x, s, simplify=True) ==
            ((s*sinh(3) + cosh(3))/(s**2 - 1), 1, True))
    L, _, _ = LT(42*sin(w*t+x)**2, t, s)
    assert (
        L -
        21*(s**2 + s*(-s*cos(2*x) + 2*w*sin(2*x)) +
            4*w**2)/(s*(s**2 + 4*w**2))).simplify() == 0
    # The following line replaces the old test test_issue_7173()
    assert LT(sinh(a*t)*cosh(a*t), t, s, simplify=True) == (a/(-4*a**2 + s**2),
                                                            2*a, True)
    assert LT(sinh(a*t)/t, t, s) == (log((a + s)/(-a + s))/2, a, True)
    assert (LT(t**(-S(3)/2)*sinh(a*t), t, s) ==
            (-sqrt(pi)*(sqrt(-a + s) - sqrt(a + s)), a, True))
    # assert (LT(sinh(2*sqrt(a*t)), t, s) ==
    #         (sqrt(pi)*sqrt(a)*exp(a/s)/s**(S(3)/2), 0, True))
    # assert (LT(sqrt(t)*sinh(2*sqrt(a*t)), t, s, simplify=True) ==
    #         ((-sqrt(a)*s**(S(5)/2) + sqrt(pi)*s**2*(2*a + s)*exp(a/s) *
    #           erf(sqrt(a)*sqrt(1/s))/2)/s**(S(9)/2), 0, True))
    # assert (LT(sinh(2*sqrt(a*t))/sqrt(t), t, s) ==
    #         (sqrt(pi)*exp(a/s)*erf(sqrt(a)*sqrt(1/s))/sqrt(s), 0, True))
    # assert (LT(sinh(sqrt(a*t))**2/sqrt(t), t, s) ==
    #         (sqrt(pi)*(exp(a/s) - 1)/(2*sqrt(s)), 0, True))
    assert (LT(t**(S(3)/7)*cosh(a*t), t, s) ==
            (((a + s)**(-S(10)/7) + (-a+s)**(-S(10)/7))*gamma(S(10)/7)/2,
             a, True))
    # assert (LT(cosh(2*sqrt(a*t)), t, s) ==
    #         (sqrt(pi)*sqrt(a)*exp(a/s)*erf(sqrt(a)*sqrt(1/s))/s**(S(3)/2) +
    #          1/s, 0, True))
    # assert (LT(sqrt(t)*cosh(2*sqrt(a*t)), t, s) ==
    #         (sqrt(pi)*(a + s/2)*exp(a/s)/s**(S(5)/2), 0, True))
    # assert (LT(cosh(2*sqrt(a*t))/sqrt(t), t, s) ==
    #         (sqrt(pi)*exp(a/s)/sqrt(s), 0, True))
    # assert (LT(cosh(sqrt(a*t))**2/sqrt(t), t, s) ==
    #         (sqrt(pi)*(exp(a/s) + 1)/(2*sqrt(s)), 0, True))
    assert LT(log(t), t, s, simplify=True) == (
        (-log(s) - EulerGamma)/s, 0, True)
    assert (LT(-log(t/a), t, s, simplify=True) ==
            ((log(a) + log(s) + EulerGamma)/s, 0, True))
    assert LT(log(1+a*t), t, s) == (-exp(s/a)*Ei(-s/a)/s, 0, True)
    assert (LT(log(t+a), t, s, simplify=True) ==
            ((s*log(a) - exp(s/a)*Ei(-s/a))/s**2, 0, True))
    assert (LT(log(t)/sqrt(t), t, s, simplify=True) ==
            (sqrt(pi)*(-log(s) - log(4) - EulerGamma)/sqrt(s), 0, True))
    assert (LT(t**(S(5)/2)*log(t), t, s, simplify=True) ==
            (sqrt(pi)*(-15*log(s) - log(1073741824) - 15*EulerGamma + 46) /
             (8*s**(S(7)/2)), 0, True))
    assert (LT(t**3*log(t), t, s, noconds=True, simplify=True) -
            6*(-log(s) - S.EulerGamma + S(11)/6)/s**4).simplify() == S.Zero
    assert (LT(log(t)**2, t, s, simplify=True) ==
            (((log(s) + EulerGamma)**2 + pi**2/6)/s, 0, True))
    assert (LT(exp(-a*t)*log(t), t, s, simplify=True) ==
            ((-log(a + s) - EulerGamma)/(a + s), -a, True))
    assert LT(sin(a*t), t, s) == (a/(a**2 + s**2), 0, True)
    assert (LT(Abs(sin(a*t)), t, s) ==
            (a*coth(pi*s/(2*a))/(a**2 + s**2), 0, True))
    assert LT(sin(a*t)/t, t, s) == (atan(a/s), 0, True)
    assert LT(sin(a*t)**2/t, t, s) == (log(4*a**2/s**2 + 1)/4, 0, True)
    assert (LT(sin(a*t)**2/t**2, t, s) ==
            (a*atan(2*a/s) - s*log(4*a**2/s**2 + 1)/4, 0, True))
    # assert (LT(sin(2*sqrt(a*t)), t, s) ==
    #         (sqrt(pi)*sqrt(a)*exp(-a/s)/s**(S(3)/2), 0, True))
    # assert LT(sin(2*sqrt(a*t))/t, t, s) == (pi*erf(sqrt(a)*sqrt(1/s)), 0, True)
    assert LT(cos(a*t), t, s) == (s/(a**2 + s**2), 0, True)
    assert (LT(cos(a*t)**2, t, s) ==
            ((2*a**2 + s**2)/(s*(4*a**2 + s**2)), 0, True))
    # assert (LT(sqrt(t)*cos(2*sqrt(a*t)), t, s, simplify=True) ==
    #         (sqrt(pi)*(-a + s/2)*exp(-a/s)/s**(S(5)/2), 0, True))
    # assert (LT(cos(2*sqrt(a*t))/sqrt(t), t, s) ==
    #         (sqrt(pi)*sqrt(1/s)*exp(-a/s), 0, True))
    assert (LT(sin(a*t)*sin(b*t), t, s) ==
            (2*a*b*s/((s**2 + (a - b)**2)*(s**2 + (a + b)**2)), 0, True))
    assert (LT(cos(a*t)*sin(b*t), t, s) ==
            (b*(-a**2 + b**2 + s**2)/((s**2 + (a - b)**2)*(s**2 + (a + b)**2)),
             0, True))
    assert (LT(cos(a*t)*cos(b*t), t, s) ==
            (s*(a**2 + b**2 + s**2)/((s**2 + (a - b)**2)*(s**2 + (a + b)**2)),
             0, True))
    assert (LT(-a*t*cos(a*t) + sin(a*t), t, s, simplify=True) ==
            (2*a**3/(a**4 + 2*a**2*s**2 + s**4), 0, True))
    assert LT(c*exp(-b*t)*sin(a*t), t, s) == (a *
                                              c/(a**2 + (b + s)**2), -b, True)
    assert LT(c*exp(-b*t)*cos(a*t), t, s) == (c*(b + s)/(a**2 + (b + s)**2),
                                              -b, True)
    L, plane, cond = LT(cos(x + 3), x, s, simplify=True)
    assert plane == 0
    assert L - (s*cos(3) - sin(3))/(s**2 + 1) == 0
    # Error functions (laplace7.pdf)
    assert LT(erf(a*t), t, s) == (exp(s**2/(4*a**2))*erfc(s/(2*a))/s, 0, True)
    # assert LT(erf(sqrt(a*t)), t, s) == (sqrt(a)/(s*sqrt(a + s)), 0, True)
    # assert (LT(exp(a*t)*erf(sqrt(a*t)), t, s, simplify=True) ==
    #         (-sqrt(a)/(sqrt(s)*(a - s)), a, True))
    # assert (LT(erf(sqrt(a/t)/2), t, s, simplify=True) ==
    #         (1/s - exp(-sqrt(a)*sqrt(s))/s, 0, True))
    # assert (LT(erfc(sqrt(a*t)), t, s, simplify=True) ==
    #         (-sqrt(a)/(s*sqrt(a + s)) + 1/s, -a, True))
    # assert (LT(exp(a*t)*erfc(sqrt(a*t)), t, s) ==
    #         (1/(sqrt(a)*sqrt(s) + s), 0, True))
    # assert LT(erfc(sqrt(a/t)/2), t, s) == (exp(-sqrt(a)*sqrt(s))/s, 0, True)
    # Bessel functions (laplace8.pdf)
    assert LT(besselj(0, a*t), t, s) == (1/sqrt(a**2 + s**2), 0, True)
    assert (LT(besselj(1, a*t), t, s, simplify=True) ==
            (a/(a**2 + s**2 + s*sqrt(a**2 + s**2)), 0, True))
    assert (LT(besselj(2, a*t), t, s, simplify=True) ==
            (a**2/(sqrt(a**2 + s**2)*(s + sqrt(a**2 + s**2))**2), 0, True))
    assert (LT(t*besselj(0, a*t), t, s) ==
            (s/(a**2 + s**2)**(S(3)/2), 0, True))
    assert (LT(t*besselj(1, a*t), t, s) ==
            (a/(a**2 + s**2)**(S(3)/2), 0, True))
    assert (LT(t**2*besselj(2, a*t), t, s) ==
            (3*a**2/(a**2 + s**2)**(S(5)/2), 0, True))
    # assert LT(besselj(0, 2*sqrt(a*t)), t, s) == (exp(-a/s)/s, 0, True)
    # assert (LT(t**(S(3)/2)*besselj(3, 2*sqrt(a*t)), t, s) ==
    #         (a**(S(3)/2)*exp(-a/s)/s**4, 0, True))
    assert (LT(besselj(0, a*sqrt(t**2+b*t)), t, s, simplify=True) ==
            (exp(b*(s - sqrt(a**2 + s**2)))/sqrt(a**2 + s**2), 0, True))
    assert LT(besseli(0, a*t), t, s) == (1/sqrt(-a**2 + s**2), a, True)
    assert (LT(besseli(1, a*t), t, s, simplify=True) ==
            (a/(-a**2 + s**2 + s*sqrt(-a**2 + s**2)), a, True))
    assert (LT(besseli(2, a*t), t, s, simplify=True) ==
            (a**2/(sqrt(-a**2 + s**2)*(s + sqrt(-a**2 + s**2))**2), a, True))
    assert LT(t*besseli(0, a*t), t, s) == (s/(-a**2 + s**2)**(S(3)/2), a, True)
    assert LT(t*besseli(1, a*t), t, s) == (a/(-a**2 + s**2)**(S(3)/2), a, True)
    assert (LT(t**2*besseli(2, a*t), t, s) ==
            (3*a**2/(-a**2 + s**2)**(S(5)/2), a, True))
    # assert (LT(t**(S(3)/2)*besseli(3, 2*sqrt(a*t)), t, s) ==
    #         (a**(S(3)/2)*exp(a/s)/s**4, 0, True))
    assert (LT(bessely(0, a*t), t, s) ==
            (-2*asinh(s/a)/(pi*sqrt(a**2 + s**2)), 0, True))
    assert (LT(besselk(0, a*t), t, s) ==
            (log((s + sqrt(-a**2 + s**2))/a)/sqrt(-a**2 + s**2), -a, True))
    assert (LT(sin(a*t)**4, t, s, simplify=True) ==
            (24*a**4/(s*(64*a**4 + 20*a**2*s**2 + s**4)), 0, True))
    # Test general rules and unevaluated forms
    # These all also test whether issue #7219 is solved.
    assert LT(Heaviside(t-1)*cos(t-1), t, s) == (s*exp(-s)/(s**2 + 1), 0, True)
    assert LT(a*f(t), t, w) == (a*LaplaceTransform(f(t), t, w), -oo, True)
    assert (LT(a*Heaviside(t+1)*f(t+1), t, s) ==
            (a*LaplaceTransform(f(t + 1), t, s), -oo, True))
    assert (LT(a*Heaviside(t-1)*f(t-1), t, s) ==
            (a*LaplaceTransform(f(t), t, s)*exp(-s), -oo, True))
    assert (LT(b*f(t/a), t, s) ==
            (a*b*LaplaceTransform(f(t), t, a*s), -oo, True))
    assert LT(exp(-f(x)*t), t, s) == (1/(s + f(x)), -re(f(x)), True)
    assert (LT(exp(-a*t)*f(t), t, s) ==
            (LaplaceTransform(f(t), t, a + s), -oo, True))
    # assert (LT(exp(-a*t)*erfc(sqrt(b/t)/2), t, s) ==
    #         (exp(-sqrt(b)*sqrt(a + s))/(a + s), -a, True))
    assert (LT(sinh(a*t)*f(t), t, s) ==
            (LaplaceTransform(f(t), t, -a + s)/2 -
             LaplaceTransform(f(t), t, a + s)/2, -oo, True))
    assert (LT(sinh(a*t)*t, t, s, simplify=True) ==
            (2*a*s/(a**4 - 2*a**2*s**2 + s**4), a, True))
    assert (LT(cosh(a*t)*f(t), t, s) ==
            (LaplaceTransform(f(t), t, -a + s)/2 +
             LaplaceTransform(f(t), t, a + s)/2, -oo, True))
    assert (LT(cosh(a*t)*t, t, s, simplify=True) ==
            (1/(2*(a + s)**2) + 1/(2*(a - s)**2), a, True))
    assert (LT(sin(a*t)*f(t), t, s, simplify=True) ==
            (I*(-LaplaceTransform(f(t), t, -I*a + s) +
                LaplaceTransform(f(t), t, I*a + s))/2, -oo, True))
    assert (LT(sin(f(t)), t, s) ==
            (LaplaceTransform(sin(f(t)), t, s), -oo, True))
    assert (LT(sin(a*t)*t, t, s, simplify=True) ==
            (2*a*s/(a**4 + 2*a**2*s**2 + s**4), 0, True))
    assert (LT(cos(a*t)*f(t), t, s) ==
            (LaplaceTransform(f(t), t, -I*a + s)/2 +
             LaplaceTransform(f(t), t, I*a + s)/2, -oo, True))
    assert (LT(cos(a*t)*t, t, s, simplify=True) ==
            ((-a**2 + s**2)/(a**4 + 2*a**2*s**2 + s**4), 0, True))
    L, plane, _ = LT(sin(a*t+b)**2*f(t), t, s)
    assert plane == -oo
    assert (
        -L + (
            LaplaceTransform(f(t), t, s)/2 -
            LaplaceTransform(f(t), t, -2*I*a + s)*exp(2*I*b)/4 -
            LaplaceTransform(f(t), t, 2*I*a + s)*exp(-2*I*b)/4)) == 0
    L = LT(sin(a*t+b)**2*f(t), t, s, noconds=True)
    assert (
        laplace_correspondence(L, {f: F}) ==
        F(s)/2 - F(-2*I*a + s)*exp(2*I*b)/4 -
        F(2*I*a + s)*exp(-2*I*b)/4)
    L, plane, _ = LT(sin(a*t)**3*cosh(b*t), t, s)
    assert plane == b
    assert (
        -L - 3*a/(8*(9*a**2 + b**2 + 2*b*s + s**2)) -
        3*a/(8*(9*a**2 + b**2 - 2*b*s + s**2)) +
        3*a/(8*(a**2 + b**2 + 2*b*s + s**2)) +
        3*a/(8*(a**2 + b**2 - 2*b*s + s**2))).simplify() == 0
    assert (LT(t**2*exp(-t**2), t, s) ==
            (sqrt(pi)*s**2*exp(s**2/4)*erfc(s/2)/8 - s/4 +
             sqrt(pi)*exp(s**2/4)*erfc(s/2)/4, 0, True))
    assert (LT((a*t**2 + b*t + c)*f(t), t, s) ==
            (a*Derivative(LaplaceTransform(f(t), t, s), (s, 2)) -
             b*Derivative(LaplaceTransform(f(t), t, s), s) +
            c*LaplaceTransform(f(t), t, s), -oo, True))
    assert (LT(t**np*g(t), t, s) ==
            ((-1)**np*Derivative(LaplaceTransform(g(t), t, s), (s, np)),
             -oo, True))
    # The following tests check whether _piecewise_to_heaviside works:
    x1 = Piecewise((0, t <= 0), (1, t <= 1), (0, True))
    X1 = LT(x1, t, s)[0]
    assert X1 == 1/s - exp(-s)/s
    y1 = ILT(X1, s, t)
    assert y1 == Heaviside(t) - Heaviside(t - 1)
    x1 = Piecewise((0, t <= 0), (t, t <= 1), (2-t, t <= 2), (0, True))
    X1 = LT(x1, t, s)[0].simplify()
    assert X1 == (exp(2*s) - 2*exp(s) + 1)*exp(-2*s)/s**2
    y1 = ILT(X1, s, t)
    assert (
        -y1 + t*Heaviside(t) + (t - 2)*Heaviside(t - 2) -
        2*(t - 1)*Heaviside(t - 1)).simplify() == 0
    x1 = Piecewise((exp(t), t <= 0), (1, t <= 1), (exp(-(t)), True))
    X1 = LT(x1, t, s)[0]
    assert X1 == exp(-1)*exp(-s)/(s + 1) + 1/s - exp(-s)/s
    y1 = ILT(X1, s, t)
    assert y1 == (
        exp(-1)*exp(1 - t)*Heaviside(t - 1) + Heaviside(t) - Heaviside(t - 1))
    x1 = Piecewise((0, x <= 0), (1, x <= 1), (0, True))
    X1 = LT(x1, t, s)[0]
    assert X1 == Piecewise((0, x <= 0), (1, x <= 1), (0, True))/s
    x1 = [
        a*Piecewise((1, And(t > 1, t <= 3)), (2, True)),
        a*Piecewise((1, And(t >= 1, t <= 3)), (2, True)),
        a*Piecewise((1, And(t >= 1, t < 3)), (2, True)),
        a*Piecewise((1, And(t > 1, t < 3)), (2, True))]
    for x2 in x1:
        assert LT(x2, t, s)[0].expand() == 2*a/s - a*exp(-s)/s + a*exp(-3*s)/s
    assert (
        LT(Piecewise((1, Eq(t, 1)), (2, True)), t, s)[0] ==
        LaplaceTransform(Piecewise((1, Eq(t, 1)), (2, True)), t, s))
    # The following lines test whether _laplace_transform successfully
    # removes Heaviside(1) before processing espressions. It fails if
    # Heaviside(t) remains because then meijerg functions will appear.
    X1 = 1/sqrt(a*s**2-b)
    x1 = ILT(X1, s, t)
    Y1 = LT(x1, t, s)[0]
    Z1 = (Y1**2/X1**2).simplify()
    assert Z1 == 1
    # The following two lines test whether issues #5813 and #7176 are solved.
    assert (LT(diff(f(t), (t, 1)), t, s, noconds=True) ==
            s*LaplaceTransform(f(t), t, s) - f(0))
    assert (LT(diff(f(t), (t, 3)), t, s, noconds=True) ==
            s**3*LaplaceTransform(f(t), t, s) - s**2*f(0) -
            s*Subs(Derivative(f(t), t), t, 0) -
            Subs(Derivative(f(t), (t, 2)), t, 0))
    # Issue #7219
    assert (LT(diff(f(x, t, w), t, 2), t, s) ==
            (s**2*LaplaceTransform(f(x, t, w), t, s) - s*f(x, 0, w) -
             Subs(Derivative(f(x, t, w), t), t, 0), -oo, True))
    # Issue #23307
    assert (LT(10*diff(f(t), (t, 1)), t, s, noconds=True) ==
            10*s*LaplaceTransform(f(t), t, s) - 10*f(0))
    assert (LT(a*f(b*t)+g(c*t), t, s, noconds=True) ==
            a*LaplaceTransform(f(t), t, s/b)/b +
            LaplaceTransform(g(t), t, s/c)/c)
    assert inverse_laplace_transform(
        f(w), w, t, plane=0) == InverseLaplaceTransform(f(w), w, t, 0)
    assert (LT(f(t)*g(t), t, s, noconds=True) ==
            LaplaceTransform(f(t)*g(t), t, s))
    # Issue #24294
    assert (LT(b*f(a*t), t, s, noconds=True) ==
            b*LaplaceTransform(f(t), t, s/a)/a)
    assert LT(3*exp(t)*Heaviside(t), t, s) == (3/(s - 1), 1, True)
    assert (LT(2*sin(t)*Heaviside(t), t, s, simplify=True) ==
            (2/(s**2 + 1), 0, True))
    # Issue #25293
    assert (
        LT((1/(t-1))*sin(4*pi*(t-1))*DiracDelta(t-1) *
           (Heaviside(t-1/4) - Heaviside(t-2)), t, s)[0] == 4*pi*exp(-s))
    # additional basic tests from wikipedia
    assert (LT((t - a)**b*exp(-c*(t - a))*Heaviside(t - a), t, s) ==
            ((c + s)**(-b - 1)*exp(-a*s)*gamma(b + 1), -c, True))
    assert (
        LT((exp(2*t)-1)*exp(-b-t)*Heaviside(t)/2, t, s, noconds=True,
           simplify=True) ==
        exp(-b)/(s**2 - 1))
    # DiracDelta function: standard cases
    assert LT(DiracDelta(t), t, s) == (1, -oo, True)
    assert LT(DiracDelta(a*t), t, s) == (1/a, -oo, True)
    assert LT(DiracDelta(t/42), t, s) == (42, -oo, True)
    assert LT(DiracDelta(t+42), t, s) == (0, -oo, True)
    assert (LT(DiracDelta(t)+DiracDelta(t-42), t, s) ==
            (1 + exp(-42*s), -oo, True))
    assert (LT(DiracDelta(t)-a*exp(-a*t), t, s, simplify=True) ==
            (s/(a + s), -a, True))
    assert (
        LT(exp(-t)*(DiracDelta(t)+DiracDelta(t-42)), t, s, simplify=True) ==
        (exp(-42*s - 42) + 1, -oo, True))
    assert LT(f(t)*DiracDelta(t-42), t, s) == (f(42)*exp(-42*s), -oo, True)
    assert LT(f(t)*DiracDelta(b*t-a), t, s) == (f(a/b)*exp(-a*s/b)/b,
                                                -oo, True)
    assert LT(f(t)*DiracDelta(b*t+a), t, s) == (0, -oo, True)
    # SingularityFunction
    assert LT(SingularityFunction(t, a, -1), t, s)[0] == exp(-a*s)
    assert LT(SingularityFunction(t, a, 1), t, s)[0] == exp(-a*s)/s**2
    assert LT(SingularityFunction(t, a, x), t, s)[0] == (
        LaplaceTransform(SingularityFunction(t, a, x), t, s))
    # Collection of cases that cannot be fully evaluated and/or would catch
    # some common implementation errors
    assert (LT(DiracDelta(t**2), t, s, noconds=True) ==
            LaplaceTransform(DiracDelta(t**2), t, s))
    assert LT(DiracDelta(t**2 - 1), t, s) == (exp(-s)/2, -oo, True)
    assert LT(DiracDelta(t*(1 - t)), t, s) == (1 - exp(-s), -oo, True)
    assert (LT((DiracDelta(t) + 1)*(DiracDelta(t - 1) + 1), t, s) ==
            (LaplaceTransform(DiracDelta(t)*DiracDelta(t - 1), t, s) +
             1 + exp(-s) + 1/s, 0, True))
    assert LT(DiracDelta(2*t-2*exp(a)), t, s) == (exp(-s*exp(a))/2, -oo, True)
    assert LT(DiracDelta(-2*t+2*exp(a)), t, s) == (exp(-s*exp(a))/2, -oo, True)
    # Heaviside tests
    assert LT(Heaviside(t), t, s) == (1/s, 0, True)
    assert LT(Heaviside(t - a), t, s) == (exp(-a*s)/s, 0, True)
    assert LT(Heaviside(t-1), t, s) == (exp(-s)/s, 0, True)
    assert LT(Heaviside(2*t-4), t, s) == (exp(-2*s)/s, 0, True)
    assert LT(Heaviside(2*t+4), t, s) == (1/s, 0, True)
    assert (LT(Heaviside(-2*t+4), t, s, simplify=True) ==
            (1/s - exp(-2*s)/s, 0, True))
    assert (LT(g(t)*Heaviside(t - w), t, s) ==
            (LaplaceTransform(g(t)*Heaviside(t - w), t, s), -oo, True))
    assert (
        LT(Heaviside(t-a)*g(t), t, s) ==
        (LaplaceTransform(g(a + t), t, s)*exp(-a*s), -oo, True))
    assert (
        LT(Heaviside(t+a)*g(t), t, s) ==
        (LaplaceTransform(g(t), t, s), -oo, True))
    assert (
        LT(Heaviside(-t+a)*g(t), t, s) ==
        (LaplaceTransform(g(t), t, s) -
         LaplaceTransform(g(a + t), t, s)*exp(-a*s), -oo, True))
    assert (
        LT(Heaviside(-t-a)*g(t), t, s) == (0, 0, True))
    # Fresnel functions
    assert (laplace_transform(fresnels(t), t, s, simplify=True) ==
            ((-sin(s**2/(2*pi))*fresnels(s/pi) +
              sqrt(2)*sin(s**2/(2*pi) + pi/4)/2 -
              cos(s**2/(2*pi))*fresnelc(s/pi))/s, 0, True))
    assert (laplace_transform(fresnelc(t), t, s, simplify=True) ==
            ((sin(s**2/(2*pi))*fresnelc(s/pi) -
              cos(s**2/(2*pi))*fresnels(s/pi) +
              sqrt(2)*cos(s**2/(2*pi) + pi/4)/2)/s, 0, True))
    # Matrix tests
    Mt = Matrix([[exp(t), t*exp(-t)], [t*exp(-t), exp(t)]])
    Ms = Matrix([[1/(s - 1), (s + 1)**(-2)],
                 [(s + 1)**(-2),     1/(s - 1)]])
    # The default behaviour for Laplace transform of a Matrix returns a Matrix
    # of Tuples and is deprecated:
    with warns_deprecated_sympy():
        Ms_conds = Matrix(
            [[(1/(s - 1), 1, True), ((s + 1)**(-2), -1, True)],
             [((s + 1)**(-2), -1, True), (1/(s - 1), 1, True)]])
    with warns_deprecated_sympy():
        assert LT(Mt, t, s) == Ms_conds
    # The new behavior is to return a tuple of a Matrix and the convergence
    # conditions for the matrix as a whole:
    assert LT(Mt, t, s, legacy_matrix=False) == (Ms, 1, True)
    # With noconds=True the transformed matrix is returned without conditions
    # either way:
    assert LT(Mt, t, s, noconds=True) == Ms
    assert LT(Mt, t, s, legacy_matrix=False, noconds=True) == Ms


@slow
def test_inverse_laplace_transform():
    s = symbols('s')
    k, n, t = symbols('k, n, t', real=True)
    a, b, c, d = symbols('a, b, c, d', positive=True)
    f = Function('f')
    F = Function('F')

    def ILT(g):
        return inverse_laplace_transform(g, s, t)

    def ILTS(g):
        return inverse_laplace_transform(g, s, t, simplify=True)

    def ILTF(g):
        return laplace_correspondence(
            inverse_laplace_transform(g, s, t), {f: F})

    # Tests for the rules in Bateman54.

    # Section 4.1: Some of the Laplace transform rules can also be used well
    #     in the inverse transform.
    assert ILTF(exp(-a*s)*F(s)) == f(-a + t)
    assert ILTF(k*F(s-a)) == k*f(t)*exp(-a*t)
    assert ILTF(diff(F(s), s, 3)) == -t**3*f(t)
    assert ILTF(diff(F(s), s, 4)) == t**4*f(t)

    # Section 5.1: Most rules are impractical for a computer algebra system.

    # Section 5.2: Rational functions
    assert ILT(2) == 2*DiracDelta(t)
    assert ILT(1/s) == Heaviside(t)
    assert ILT(1/s**2) == t*Heaviside(t)
    assert ILT(1/s**5) == t**4*Heaviside(t)/24
    assert ILT(1/s**n) == t**(n - 1)*Heaviside(t)/gamma(n)
    assert ILT(a/(a + s)) == a*exp(-a*t)*Heaviside(t)
    assert ILT(s/(a + s)) == -a*exp(-a*t)*Heaviside(t) + DiracDelta(t)
    assert (ILT(b*s/(s+a)**2) ==
            b*(-a*t*exp(-a*t)*Heaviside(t) + exp(-a*t)*Heaviside(t)))
    assert (ILTS(c/((s+a)*(s+b))) ==
            c*(exp(a*t) - exp(b*t))*exp(-t*(a + b))*Heaviside(t)/(a - b))
    assert (ILTS(c*s/((s+a)*(s+b))) ==
            c*(a*exp(b*t) - b*exp(a*t))*exp(-t*(a + b))*Heaviside(t)/(a - b))
    assert ILTS(s/(a + s)**3) == t*(-a*t + 2)*exp(-a*t)*Heaviside(t)/2
    assert ILTS(1/(s*(a + s)**3)) == (
        -a**2*t**2 - 2*a*t + 2*exp(a*t) - 2)*exp(-a*t)*Heaviside(t)/(2*a**3)
    assert ILT(1/(s*(a + s)**n)) == (
        Heaviside(t)*lowergamma(n, a*t)/(a**n*gamma(n)))
    assert ILT((s-a)**(-b)) == t**(b - 1)*exp(a*t)*Heaviside(t)/gamma(b)
    assert ILT((a + s)**(-2)) == t*exp(-a*t)*Heaviside(t)
    assert ILT((a + s)**(-5)) == t**4*exp(-a*t)*Heaviside(t)/24
    assert ILT(s**2/(s**2 + 1)) == -sin(t)*Heaviside(t) + DiracDelta(t)
    assert ILT(1 - 1/(s**2 + 1)) == -sin(t)*Heaviside(t) + DiracDelta(t)
    assert ILT(a/(a**2 + s**2)) == sin(a*t)*Heaviside(t)
    assert ILT(s/(s**2 + a**2)) == cos(a*t)*Heaviside(t)
    assert ILT(b/(b**2 + (a + s)**2)) == exp(-a*t)*sin(b*t)*Heaviside(t)
    assert (ILT(b*s/(b**2 + (a + s)**2)) ==
            b*(-a*exp(-a*t)*sin(b*t)/b + exp(-a*t)*cos(b*t))*Heaviside(t))
    assert ILT(1/(s**2*(s**2 + 1))) == t*Heaviside(t) - sin(t)*Heaviside(t)
    assert (ILTS(c*s/(d**2*(s+a)**2+b**2)) ==
            c*(-a*d*sin(b*t/d) + b*cos(b*t/d))*exp(-a*t)*Heaviside(t)/(b*d**2))
    assert ILTS((b*s**2 + d)/(a**2 + s**2)**2) == (
        2*a**2*b*sin(a*t) + (a**2*b - d)*(a*t*cos(a*t) -
                                          sin(a*t)))*Heaviside(t)/(2*a**3)
    assert ILTS(b/(s**2-a**2)) == b*sinh(a*t)*Heaviside(t)/a
    assert (ILT(b/(s**2-a**2)) ==
            b*(exp(a*t)*Heaviside(t)/(2*a) - exp(-a*t)*Heaviside(t)/(2*a)))
    assert ILTS(b*s/(s**2-a**2)) == b*cosh(a*t)*Heaviside(t)
    assert (ILT(b/(s*(s+a))) ==
            b*(Heaviside(t)/a - exp(-a*t)*Heaviside(t)/a))
    # Issue #24424
    assert (ILTS((s + 8)/((s + 2)*(s**2 + 2*s + 10))) ==
            ((8*sin(3*t) - 9*cos(3*t))*exp(t) + 9)*exp(-2*t)*Heaviside(t)/15)
    # Issue #8514; this is not important anymore, since this function
    # is not solved by integration anymore
    assert (ILT(1/(a*s**2+b*s+c)) ==
            2*exp(-b*t/(2*a))*sin(t*sqrt(4*a*c - b**2)/(2*a)) *
            Heaviside(t)/sqrt(4*a*c - b**2))

    # Section 5.3: Irrational algebraic functions
    assert (  # (1)
        ILT(1/sqrt(s)/(b*s-a)) ==
        exp(a*t/b)*Heaviside(t)*erf(sqrt(a)*sqrt(t)/sqrt(b))/(sqrt(a)*sqrt(b)))
    assert (  # (2)
        ILT(1/sqrt(k*s)/(c*s-a)/s) ==
        (-2*c*sqrt(t)/(sqrt(pi)*a) +
         c**(S(3)/2)*exp(a*t/c)*erf(sqrt(a)*sqrt(t)/sqrt(c))/a**(S(3)/2)) *
        Heaviside(t)/(c*sqrt(k)))
    assert (  # (4)
        ILT(1/(sqrt(c*s)+a)) == (-a*exp(a**2*t/c)*erfc(a*sqrt(t)/sqrt(c))/c +
                                 1/(sqrt(pi)*sqrt(c)*sqrt(t)))*Heaviside(t))
    assert (  # (5)
        ILT(a/s/(b*sqrt(s)+a)) ==
        (-exp(a**2*t/b**2)*erfc(a*sqrt(t)/b) + 1)*Heaviside(t))
    assert (  # (6)
            ILT((a-b)*sqrt(s)/(sqrt(s)+sqrt(a))/(s-b)) ==
            (sqrt(a)*sqrt(b)*exp(b*t)*erfc(sqrt(b)*sqrt(t)) +
             a*exp(a*t)*erfc(sqrt(a)*sqrt(t)) - b*exp(b*t))*Heaviside(t))
    assert (  # (7)
            ILT(1/sqrt(s)/(sqrt(b*s)+a)) ==
            exp(a**2*t/b)*Heaviside(t)*erfc(a*sqrt(t)/sqrt(b))/sqrt(b))
    assert (  # (8)
            ILT(a**2/(sqrt(s)+a)/s**(S(3)/2)) ==
            (2*a*sqrt(t)/sqrt(pi) + exp(a**2*t)*erfc(a*sqrt(t)) - 1) *
            Heaviside(t))
    assert (  # (9)
            ILT((a-b)*sqrt(b)/(s-b)/sqrt(s)/(sqrt(s)+sqrt(a))) ==
            (sqrt(a)*exp(b*t)*erf(sqrt(b)*sqrt(t)) +
             sqrt(b)*exp(a*t)*erfc(sqrt(a)*sqrt(t)) -
             sqrt(b)*exp(b*t))*Heaviside(t))
    assert (  # (10)
            ILT(1/(sqrt(s)+sqrt(a))**2) ==
            (-2*sqrt(a)*sqrt(t)/sqrt(pi) +
             (-2*a*t + 1)*(erf(sqrt(a)*sqrt(t)) -
                           1)*exp(a*t) + 1)*Heaviside(t))
    assert (  # (11)
            ILT(1/(sqrt(s)+sqrt(a))**2/s) ==
            ((2*t - 1/a)*exp(a*t)*erfc(sqrt(a)*sqrt(t)) + 1/a -
             2*sqrt(t)/(sqrt(pi)*sqrt(a)))*Heaviside(t))
    assert (  # (12)
            ILT(1/(sqrt(s)+a)**2/sqrt(s)) ==
            (-2*a*t*exp(a**2*t)*erfc(a*sqrt(t)) +
             2*sqrt(t)/sqrt(pi))*Heaviside(t))
    assert (  # (13)
            ILT(1/(sqrt(s)+a)**3) ==
            (-a*t*(2*a**2*t + 3)*exp(a**2*t)*erfc(a*sqrt(t)) +
             2*sqrt(t)*(a**2*t + 1)/sqrt(pi))*Heaviside(t))
    x = (
        - ILT(sqrt(s)/(sqrt(s)+a)**3) +
        2*(sqrt(pi)*a**2*t*(-2*sqrt(pi)*erfc(a*sqrt(t)) +
                            2*exp(-a**2*t)/(a*sqrt(t))) *
           (-a**4*t**2 - 5*a**2*t/2 - S.Half) * exp(a**2*t)/2 +
           sqrt(pi)*a*sqrt(t)*(a**2*t + 1)/2) *
        Heaviside(t)/(pi*a**2*t)).simplify()
    assert (  # (14)
            x == 0)
    x = (
        - ILT(1/sqrt(s)/(sqrt(s)+a)**3) +
        Heaviside(t)*(sqrt(t)*((2*a**2*t + 1) *
                               (sqrt(pi)*a*sqrt(t)*exp(a**2*t) *
                                erfc(a*sqrt(t)) - 1) + 1) /
                      (sqrt(pi)*a))).simplify()
    assert (  # (15)
            x == 0)
    assert (  # (16)
            factor_terms(ILT(3/(sqrt(s)+a)**4)) ==
            3*(-2*a**3*t**(S(5)/2)*(2*a**2*t + 5)/(3*sqrt(pi)) +
               t*(4*a**4*t**2 + 12*a**2*t + 3)*exp(a**2*t) *
               erfc(a*sqrt(t))/3)*Heaviside(t))
    assert (  # (17)
            ILT((sqrt(s)-a)/(s*(sqrt(s)+a))) ==
            (2*exp(a**2*t)*erfc(a*sqrt(t))-1)*Heaviside(t))
    assert (  # (18)
            ILT((sqrt(s)-a)**2/(s*(sqrt(s)+a)**2)) == (
                1 + 8*a**2*t*exp(a**2*t)*erfc(a*sqrt(t)) -
                8/sqrt(pi)*a*sqrt(t))*Heaviside(t))
    assert (  # (19)
            ILT((sqrt(s)-a)**3/(s*(sqrt(s)+a)**3)) == Heaviside(t)*(
                2*(8*a**4*t**2+8*a**2*t+1)*exp(a**2*t) *
                erfc(a*sqrt(t))-8/sqrt(pi)*a*sqrt(t)*(2*a**2*t+1)-1))
    assert (  # (22)
            ILT(sqrt(s+a)/(s+b)) == Heaviside(t)*(
                exp(-a*t)/sqrt(t)/sqrt(pi) +
                sqrt(a-b)*exp(-b*t)*erf(sqrt(a-b)*sqrt(t))))
    assert (  # (23)
            ILT(1/sqrt(s+b)/(s+a)) == Heaviside(t)*(
                1/sqrt(b-a)*exp(-a*t)*erf(sqrt(b-a)*sqrt(t))))
    assert (  # (35)
            ILT(1/sqrt(s**2+a**2)) == Heaviside(t)*(
                besselj(0, a*t)))
    assert (  # (44)
            ILT(1/sqrt(s**2-a**2)) == Heaviside(t)*(
                besseli(0, a*t)))

    # Miscellaneous tests
    # Can _inverse_laplace_time_shift deal with positive exponents?
    assert (
        - ILT((s**2*exp(2*s) + 4*exp(s) - 4)*exp(-2*s)/(s*(s**2 + 1))) +
        cos(t)*Heaviside(t) + 4*cos(t - 2)*Heaviside(t - 2) -
        4*cos(t - 1)*Heaviside(t - 1) - 4*Heaviside(t - 2) +
        4*Heaviside(t - 1)).simplify() == 0


@slow
def test_inverse_laplace_transform_old():
    from sympy.functions.special.delta_functions import DiracDelta
    ILT = inverse_laplace_transform
    a, b, c, d = symbols('a b c d', positive=True)
    n, r = symbols('n, r', real=True)
    t, z = symbols('t z')
    f = Function('f')
    F = Function('F')

    def simp_hyp(expr):
        return factor_terms(expand_mul(expr)).rewrite(sin)

    L = ILT(F(s), s, t)
    assert laplace_correspondence(L, {f: F}) == f(t)
    assert ILT(exp(-a*s)/s, s, t) == Heaviside(-a + t)
    assert ILT(exp(-a*s)/(b + s), s, t) == exp(-b*(-a + t))*Heaviside(-a + t)
    assert (ILT((b + s)/(a**2 + (b + s)**2), s, t) ==
            exp(-b*t)*cos(a*t)*Heaviside(t))
    assert (ILT(exp(-a*s)/s**b, s, t) ==
            (-a + t)**(b - 1)*Heaviside(-a + t)/gamma(b))
    assert (ILT(exp(-a*s)/sqrt(s**2 + 1), s, t) ==
            Heaviside(-a + t)*besselj(0, a - t))
    assert ILT(1/(s*sqrt(s + 1)), s, t) == Heaviside(t)*erf(sqrt(t))
    # TODO sinh/cosh shifted come out a mess. also delayed trig is a mess
    # TODO should this simplify further?
    assert (ILT(exp(-a*s)/s**b, s, t) ==
            (t - a)**(b - 1)*Heaviside(t - a)/gamma(b))
    assert (ILT(exp(-a*s)/sqrt(1 + s**2), s, t) ==
            Heaviside(t - a)*besselj(0, a - t))  # note: besselj(0, x) is even
    # XXX ILT turns these branch factor into trig functions ...
    assert (
        simplify(ILT(a**b*(s + sqrt(s**2 - a**2))**(-b)/sqrt(s**2 - a**2),
                     s, t).rewrite(exp)) ==
        Heaviside(t)*besseli(b, a*t))
    assert (
        ILT(a**b*(s + sqrt(s**2 + a**2))**(-b)/sqrt(s**2 + a**2),
            s, t, simplify=True).rewrite(exp) ==
        Heaviside(t)*besselj(b, a*t))
    assert ILT(1/(s*sqrt(s + 1)), s, t) == Heaviside(t)*erf(sqrt(t))
    # TODO can we make erf(t) work?
    assert (ILT((s * eye(2) - Matrix([[1, 0], [0, 2]])).inv(), s, t) ==
            Matrix([[exp(t)*Heaviside(t), 0], [0, exp(2*t)*Heaviside(t)]]))
    # Test time_diff rule
    assert (ILT(s**42*f(s), s, t) ==
            Derivative(InverseLaplaceTransform(f(s), s, t, None), (t, 42)))
    assert ILT(cos(s), s, t) == InverseLaplaceTransform(cos(s), s, t, None)
    # Rules for testing different DiracDelta cases
    assert (
        ILT(1 + 2*s + 3*s**2 + 5*s**3, s, t) == DiracDelta(t) +
        2*DiracDelta(t, 1) + 3*DiracDelta(t, 2) + 5*DiracDelta(t, 3))
    assert (ILT(2*exp(3*s) - 5*exp(-7*s), s, t) ==
            2*InverseLaplaceTransform(exp(3*s), s, t, None) -
            5*DiracDelta(t - 7))
    a = cos(sin(7)/2)
    assert ILT(a*exp(-3*s), s, t) == a*DiracDelta(t - 3)
    assert ILT(exp(2*s), s, t) == InverseLaplaceTransform(exp(2*s), s, t, None)
    r = Symbol('r', real=True)
    assert ILT(exp(r*s), s, t) == InverseLaplaceTransform(exp(r*s), s, t, None)
    # Rules for testing whether Heaviside(t) is treated properly in diff rule
    assert ILT(s**2/(a**2 + s**2), s, t) == (
        -a*sin(a*t)*Heaviside(t) + DiracDelta(t))
    assert ILT(s**2*(f(s) + 1/(a**2 + s**2)), s, t) == (
        -a*sin(a*t)*Heaviside(t) + DiracDelta(t) +
        Derivative(InverseLaplaceTransform(f(s), s, t, None), (t, 2)))
    # Rules from the previous test_inverse_laplace_transform_delta_cond():
    assert (ILT(exp(r*s), s, t, noconds=False) ==
            (InverseLaplaceTransform(exp(r*s), s, t, None), True))
    # inversion does not exist: verify it doesn't evaluate to DiracDelta
    for z in (Symbol('z', extended_real=False),
              Symbol('z', imaginary=True, zero=False)):
        f = ILT(exp(z*s), s, t, noconds=False)
        f = f[0] if isinstance(f, tuple) else f
        assert f.func != DiracDelta


@slow
def test_expint():
    x = Symbol('x')
    a = Symbol('a')
    u = Symbol('u', polar=True)

    # TODO LT of Si, Shi, Chi is a mess ...
    assert laplace_transform(Ci(x), x, s) == (-log(1 + s**2)/2/s, 0, True)
    assert (laplace_transform(expint(a, x), x, s, simplify=True) ==
            (lerchphi(s*exp_polar(I*pi), 1, a), 0, re(a) > S.Zero))
    assert (laplace_transform(expint(1, x), x, s, simplify=True) ==
            (log(s + 1)/s, 0, True))
    assert (laplace_transform(expint(2, x), x, s, simplify=True) ==
            ((s - log(s + 1))/s**2, 0, True))
    assert (inverse_laplace_transform(-log(1 + s**2)/2/s, s, u).expand() ==
            Heaviside(u)*Ci(u))
    assert (
        inverse_laplace_transform(log(s + 1)/s, s, x,
                                  simplify=True).rewrite(expint) ==
        Heaviside(x)*E1(x))
    assert (
        inverse_laplace_transform(
            (s - log(s + 1))/s**2, s, x,
            simplify=True).rewrite(expint).expand() ==
        (expint(2, x)*Heaviside(x)).rewrite(Ei).rewrite(expint).expand())
