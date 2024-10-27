from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.function import (Derivative, Function, diff, expand)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (asinh, csch, cosh, coth, sech, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise, piecewise_fold
from sympy.functions.elementary.trigonometric import (acos, acot, acsc, asec, asin, atan, cos, cot, csc, sec, sin, tan)
from sympy.functions.special.delta_functions import Heaviside, DiracDelta
from sympy.functions.special.elliptic_integrals import (elliptic_e, elliptic_f)
from sympy.functions.special.error_functions import (Chi, Ci, Ei, Shi, Si, erf, erfi, fresnelc, fresnels, li)
from sympy.functions.special.gamma_functions import uppergamma
from sympy.functions.special.polynomials import (assoc_laguerre, chebyshevt, chebyshevu, gegenbauer, hermite, jacobi, laguerre, legendre)
from sympy.functions.special.zeta_functions import polylog
from sympy.integrals.integrals import (Integral, integrate)
from sympy.logic.boolalg import And
from sympy.integrals.manualintegrate import (manualintegrate, find_substitutions,
    _parts_rule, integral_steps, manual_subs)
from sympy.testing.pytest import raises, slow

x, y, z, u, n, a, b, c, d, e = symbols('x y z u n a b c d e')
f = Function('f')


def assert_is_integral_of(f: Expr, F: Expr):
    assert manualintegrate(f, x) == F
    assert F.diff(x).equals(f)


def test_find_substitutions():
    assert find_substitutions((cot(x)**2 + 1)**2*csc(x)**2*cot(x)**2, x, u) == \
        [(cot(x), 1, -u**6 - 2*u**4 - u**2)]
    assert find_substitutions((sec(x)**2 + tan(x) * sec(x)) / (sec(x) + tan(x)),
                              x, u) == [(sec(x) + tan(x), 1, 1/u)]
    assert (-x**2, Rational(-1, 2), exp(u)) in find_substitutions(x * exp(-x**2), x, u)
    assert not find_substitutions(Derivative(f(x), x)**2, x, u)


def test_manualintegrate_polynomials():
    assert manualintegrate(y, x) == x*y
    assert manualintegrate(exp(2), x) == x * exp(2)
    assert manualintegrate(x**2, x) == x**3 / 3
    assert manualintegrate(3 * x**2 + 4 * x**3, x) == x**3 + x**4

    assert manualintegrate((x + 2)**3, x) == (x + 2)**4 / 4
    assert manualintegrate((3*x + 4)**2, x) == (3*x + 4)**3 / 9

    assert manualintegrate((u + 2)**3, u) == (u + 2)**4 / 4
    assert manualintegrate((3*u + 4)**2, u) == (3*u + 4)**3 / 9


def test_manualintegrate_exponentials():
    assert manualintegrate(exp(2*x), x) == exp(2*x) / 2
    assert manualintegrate(2**x, x) == (2 ** x) / log(2)
    assert_is_integral_of(1/sqrt(1-exp(2*x)),
                          log(sqrt(1 - exp(2*x)) - 1)/2 - log(sqrt(1 - exp(2*x)) + 1)/2)

    assert manualintegrate(1 / x, x) == log(x)
    assert manualintegrate(1 / (2*x + 3), x) == log(2*x + 3) / 2
    assert manualintegrate(log(x)**2 / x, x) == log(x)**3 / 3

    assert_is_integral_of(x**x*(log(x)+1), x**x)


def test_manualintegrate_parts():
    assert manualintegrate(exp(x) * sin(x), x) == \
        (exp(x) * sin(x)) / 2 - (exp(x) * cos(x)) / 2
    assert manualintegrate(2*x*cos(x), x) == 2*x*sin(x) + 2*cos(x)
    assert manualintegrate(x * log(x), x) == x**2*log(x)/2 - x**2/4
    assert manualintegrate(log(x), x) == x * log(x) - x
    assert manualintegrate((3*x**2 + 5) * exp(x), x) == \
        3*x**2*exp(x) - 6*x*exp(x) + 11*exp(x)
    assert manualintegrate(atan(x), x) == x*atan(x) - log(x**2 + 1)/2

    # Make sure _parts_rule doesn't pick u = constant but can pick dv =
    # constant if necessary, e.g. for integrate(atan(x))
    assert _parts_rule(cos(x), x) == None
    assert _parts_rule(exp(x), x) == None
    assert _parts_rule(x**2, x) == None
    result = _parts_rule(atan(x), x)
    assert result[0] == atan(x) and result[1] == 1


def test_manualintegrate_trigonometry():
    assert manualintegrate(sin(x), x) == -cos(x)
    assert manualintegrate(tan(x), x) == -log(cos(x))

    assert manualintegrate(sec(x), x) == log(sec(x) + tan(x))
    assert manualintegrate(csc(x), x) == -log(csc(x) + cot(x))

    assert manualintegrate(sin(x) * cos(x), x) in [sin(x) ** 2 / 2, -cos(x)**2 / 2]
    assert manualintegrate(-sec(x) * tan(x), x) == -sec(x)
    assert manualintegrate(csc(x) * cot(x), x) == -csc(x)
    assert manualintegrate(sec(x)**2, x) == tan(x)
    assert manualintegrate(csc(x)**2, x) == -cot(x)

    assert manualintegrate(x * sec(x**2), x) == log(tan(x**2) + sec(x**2))/2
    assert manualintegrate(cos(x)*csc(sin(x)), x) == -log(cot(sin(x)) + csc(sin(x)))
    assert manualintegrate(cos(3*x)*sec(x), x) == -x + sin(2*x)
    assert manualintegrate(sin(3*x)*sec(x), x) == \
        -3*log(cos(x)) + 2*log(cos(x)**2) - 2*cos(x)**2

    assert_is_integral_of(sinh(2*x), cosh(2*x)/2)
    assert_is_integral_of(x*cosh(x**2), sinh(x**2)/2)
    assert_is_integral_of(tanh(x), log(cosh(x)))
    assert_is_integral_of(coth(x), log(sinh(x)))
    f, F = sech(x), 2*atan(tanh(x/2))
    assert manualintegrate(f, x) == F
    assert (F.diff(x) - f).rewrite(exp).simplify() == 0  # todo: equals returns None
    f, F = csch(x), log(tanh(x/2))
    assert manualintegrate(f, x) == F
    assert (F.diff(x) - f).rewrite(exp).simplify() == 0


@slow
def test_manualintegrate_trigpowers():
    assert manualintegrate(sin(x)**2 * cos(x), x) == sin(x)**3 / 3
    assert manualintegrate(sin(x)**2 * cos(x) **2, x) == \
        x / 8 - sin(4*x) / 32
    assert manualintegrate(sin(x) * cos(x)**3, x) == -cos(x)**4 / 4
    assert manualintegrate(sin(x)**3 * cos(x)**2, x) == \
        cos(x)**5 / 5 - cos(x)**3 / 3

    assert manualintegrate(tan(x)**3 * sec(x), x) == sec(x)**3/3 - sec(x)
    assert manualintegrate(tan(x) * sec(x) **2, x) == sec(x)**2/2

    assert manualintegrate(cot(x)**5 * csc(x), x) == \
        -csc(x)**5/5 + 2*csc(x)**3/3 - csc(x)
    assert manualintegrate(cot(x)**2 * csc(x)**6, x) == \
        -cot(x)**7/7 - 2*cot(x)**5/5 - cot(x)**3/3


@slow
def test_manualintegrate_inversetrig():
    # atan
    assert manualintegrate(exp(x) / (1 + exp(2*x)), x) == atan(exp(x))
    assert manualintegrate(1 / (4 + 9 * x**2), x) == atan(3 * x/2) / 6
    assert manualintegrate(1 / (16 + 16 * x**2), x) == atan(x) / 16
    assert manualintegrate(1 / (4 + x**2), x) == atan(x / 2) / 2
    assert manualintegrate(1 / (1 + 4 * x**2), x) == atan(2*x) / 2
    ra = Symbol('a', real=True)
    rb = Symbol('b', real=True)
    assert manualintegrate(1/(ra + rb*x**2), x) == \
        Piecewise((atan(x/sqrt(ra/rb))/(rb*sqrt(ra/rb)), ra/rb > 0),
                  ((log(x - sqrt(-ra/rb)) - log(x + sqrt(-ra/rb)))/(2*sqrt(rb)*sqrt(-ra)), True))
    assert manualintegrate(1/(4 + rb*x**2), x) == \
        Piecewise((atan(x/(2*sqrt(1/rb)))/(2*rb*sqrt(1/rb)), 1/rb > 0),
                  (-I*(log(x - 2*sqrt(-1/rb)) - log(x + 2*sqrt(-1/rb)))/(4*sqrt(rb)), True))
    assert manualintegrate(1/(ra + 4*x**2), x) == \
        Piecewise((atan(2*x/sqrt(ra))/(2*sqrt(ra)), ra > 0),
                  ((log(x - sqrt(-ra)/2) - log(x + sqrt(-ra)/2))/(4*sqrt(-ra)), True))
    assert manualintegrate(1/(4 + 4*x**2), x) == atan(x) / 4

    assert manualintegrate(1/(a + b*x**2), x) == Piecewise((atan(x/sqrt(a/b))/(b*sqrt(a/b)), Ne(a, 0)),
                                                           (-1/(b*x), True))

    # asin
    assert manualintegrate(1/sqrt(1-x**2), x) == asin(x)
    assert manualintegrate(1/sqrt(4-4*x**2), x) == asin(x)/2
    assert manualintegrate(3/sqrt(1-9*x**2), x) == asin(3*x)
    assert manualintegrate(1/sqrt(4-9*x**2), x) == asin(x*Rational(3, 2))/3

    # asinh
    assert manualintegrate(1/sqrt(x**2 + 1), x) == \
        asinh(x)
    assert manualintegrate(1/sqrt(x**2 + 4), x) == \
        asinh(x/2)
    assert manualintegrate(1/sqrt(4*x**2 + 4), x) == \
        asinh(x)/2
    assert manualintegrate(1/sqrt(4*x**2 + 1), x) == \
        asinh(2*x)/2
    assert manualintegrate(1/sqrt(ra*x**2 + 1), x) == \
        Piecewise((asin(x*sqrt(-ra))/sqrt(-ra), ra < 0), (asinh(sqrt(ra)*x)/sqrt(ra), ra > 0), (x, True))
    assert manualintegrate(1/sqrt(ra + x**2), x) == \
        Piecewise((asinh(x*sqrt(1/ra)), ra > 0), (log(2*x + 2*sqrt(ra + x**2)), True))

    # log
    assert manualintegrate(1/sqrt(x**2 - 1), x) == log(2*x + 2*sqrt(x**2 - 1))
    assert manualintegrate(1/sqrt(x**2 - 4), x) == log(2*x + 2*sqrt(x**2 - 4))
    assert manualintegrate(1/sqrt(4*x**2 - 4), x) == log(8*x + 4*sqrt(4*x**2 - 4))/2
    assert manualintegrate(1/sqrt(9*x**2 - 1), x) == log(18*x + 6*sqrt(9*x**2 - 1))/3
    assert manualintegrate(1/sqrt(ra*x**2 - 4), x) == \
           Piecewise((log(2*sqrt(ra)*sqrt(ra*x**2 - 4) + 2*ra*x)/sqrt(ra), Ne(ra, 0)), (-I*x/2, True))
    assert manualintegrate(1/sqrt(-ra + 4*x**2), x) == \
        Piecewise((asinh(2*x*sqrt(-1/ra))/2, ra < 0), (log(8*x + 4*sqrt(-ra + 4*x**2))/2, True))

    # From https://www.wikiwand.com/en/List_of_integrals_of_inverse_trigonometric_functions
    # asin
    assert manualintegrate(asin(x), x) == x*asin(x) + sqrt(1 - x**2)
    assert manualintegrate(asin(a*x), x) == Piecewise(((a*x*asin(a*x) + sqrt(-a**2*x**2 + 1))/a, Ne(a, 0)), (0, True))
    assert manualintegrate(x*asin(a*x), x) == \
           -a*Piecewise((-x*sqrt(-a**2*x**2 + 1)/(2*a**2) +
                         log(-2*a**2*x + 2*sqrt(-a**2)*sqrt(-a**2*x**2 + 1))/(2*a**2*sqrt(-a**2)), Ne(a**2, 0)),
                        (x**3/3, True))/2 + x**2*asin(a*x)/2
    # acos
    assert manualintegrate(acos(x), x) == x*acos(x) - sqrt(1 - x**2)
    assert manualintegrate(acos(a*x), x) == Piecewise(((a*x*acos(a*x) - sqrt(-a**2*x**2 + 1))/a, Ne(a, 0)), (pi*x/2, True))
    assert manualintegrate(x*acos(a*x), x) == \
           a*Piecewise((-x*sqrt(-a**2*x**2 + 1)/(2*a**2) +
                        log(-2*a**2*x + 2*sqrt(-a**2)*sqrt(-a**2*x**2 + 1))/(2*a**2*sqrt(-a**2)), Ne(a**2, 0)),
                       (x**3/3, True))/2 + x**2*acos(a*x)/2
    # atan
    assert manualintegrate(atan(x), x) == x*atan(x) - log(x**2 + 1)/2
    assert manualintegrate(atan(a*x), x) == Piecewise(((a*x*atan(a*x) - log(a**2*x**2 + 1)/2)/a, Ne(a, 0)), (0, True))
    assert manualintegrate(x*atan(a*x), x) == -a*(x/a**2 - atan(x/sqrt(a**(-2)))/(a**4*sqrt(a**(-2))))/2 + x**2*atan(a*x)/2
    # acsc
    assert manualintegrate(acsc(x), x) == x*acsc(x) + Integral(1/(x*sqrt(1 - 1/x**2)), x)
    assert manualintegrate(acsc(a*x), x) == x*acsc(a*x) + Integral(1/(x*sqrt(1 - 1/(a**2*x**2))), x)/a
    assert manualintegrate(x*acsc(a*x), x) == x**2*acsc(a*x)/2 + Integral(1/sqrt(1 - 1/(a**2*x**2)), x)/(2*a)
    # asec
    assert manualintegrate(asec(x), x) == x*asec(x) - Integral(1/(x*sqrt(1 - 1/x**2)), x)
    assert manualintegrate(asec(a*x), x) == x*asec(a*x) - Integral(1/(x*sqrt(1 - 1/(a**2*x**2))), x)/a
    assert manualintegrate(x*asec(a*x), x) == x**2*asec(a*x)/2 - Integral(1/sqrt(1 - 1/(a**2*x**2)), x)/(2*a)
    # acot
    assert manualintegrate(acot(x), x) == x*acot(x) + log(x**2 + 1)/2
    assert manualintegrate(acot(a*x), x) == Piecewise(((a*x*acot(a*x) + log(a**2*x**2 + 1)/2)/a, Ne(a, 0)), (pi*x/2, True))
    assert manualintegrate(x*acot(a*x), x) == a*(x/a**2 - atan(x/sqrt(a**(-2)))/(a**4*sqrt(a**(-2))))/2 + x**2*acot(a*x)/2

    # piecewise
    assert manualintegrate(1/sqrt(ra-rb*x**2), x) == \
        Piecewise((asin(x*sqrt(rb/ra))/sqrt(rb), And(-rb < 0, ra > 0)),
                  (asinh(x*sqrt(-rb/ra))/sqrt(-rb), And(-rb > 0, ra > 0)),
                  (log(-2*rb*x + 2*sqrt(-rb)*sqrt(ra - rb*x**2))/sqrt(-rb), Ne(rb, 0)),
                  (x/sqrt(ra), True))
    assert manualintegrate(1/sqrt(ra + rb*x**2), x) == \
        Piecewise((asin(x*sqrt(-rb/ra))/sqrt(-rb), And(ra > 0, rb < 0)),
                  (asinh(x*sqrt(rb/ra))/sqrt(rb), And(ra > 0, rb > 0)),
                  (log(2*sqrt(rb)*sqrt(ra + rb*x**2) + 2*rb*x)/sqrt(rb), Ne(rb, 0)),
                  (x/sqrt(ra), True))


def test_manualintegrate_trig_substitution():
    assert manualintegrate(sqrt(16*x**2 - 9)/x, x) == \
        Piecewise((sqrt(16*x**2 - 9) - 3*acos(3/(4*x)),
                   And(x < Rational(3, 4), x > Rational(-3, 4))))
    assert manualintegrate(1/(x**4 * sqrt(25-x**2)), x) == \
        Piecewise((-sqrt(-x**2/25 + 1)/(125*x) -
                   (-x**2/25 + 1)**(3*S.Half)/(15*x**3), And(x < 5, x > -5)))
    assert manualintegrate(x**7/(49*x**2 + 1)**(3 * S.Half), x) == \
        ((49*x**2 + 1)**(5*S.Half)/28824005 -
         (49*x**2 + 1)**(3*S.Half)/5764801 +
         3*sqrt(49*x**2 + 1)/5764801 + 1/(5764801*sqrt(49*x**2 + 1)))

def test_manualintegrate_trivial_substitution():
    assert manualintegrate((exp(x) - exp(-x))/x, x) == -Ei(-x) + Ei(x)
    f = Function('f')
    assert manualintegrate((f(x) - f(-x))/x, x) == \
        -Integral(f(-x)/x, x) + Integral(f(x)/x, x)


def test_manualintegrate_rational():
    assert manualintegrate(1/(4 - x**2), x) == -log(x - 2)/4 + log(x + 2)/4
    assert manualintegrate(1/(-1 + x**2), x) == log(x - 1)/2 - log(x + 1)/2


def test_manualintegrate_special():
    f, F = 4*exp(-x**2/3), 2*sqrt(3)*sqrt(pi)*erf(sqrt(3)*x/3)
    assert_is_integral_of(f, F)
    f, F = 3*exp(4*x**2), 3*sqrt(pi)*erfi(2*x)/4
    assert_is_integral_of(f, F)
    f, F = x**Rational(1, 3)*exp(-x/8), -16*uppergamma(Rational(4, 3), x/8)
    assert_is_integral_of(f, F)
    f, F = exp(2*x)/x, Ei(2*x)
    assert_is_integral_of(f, F)
    f, F = exp(1 + 2*x - x**2), sqrt(pi)*exp(2)*erf(x - 1)/2
    assert_is_integral_of(f, F)
    f = sin(x**2 + 4*x + 1)
    F = (sqrt(2)*sqrt(pi)*(-sin(3)*fresnelc(sqrt(2)*(2*x + 4)/(2*sqrt(pi))) +
        cos(3)*fresnels(sqrt(2)*(2*x + 4)/(2*sqrt(pi))))/2)
    assert_is_integral_of(f, F)
    f, F = cos(4*x**2), sqrt(2)*sqrt(pi)*fresnelc(2*sqrt(2)*x/sqrt(pi))/4
    assert_is_integral_of(f, F)
    f, F = sin(3*x + 2)/x, sin(2)*Ci(3*x) + cos(2)*Si(3*x)
    assert_is_integral_of(f, F)
    f, F = sinh(3*x - 2)/x, -sinh(2)*Chi(3*x) + cosh(2)*Shi(3*x)
    assert_is_integral_of(f, F)
    f, F = 5*cos(2*x - 3)/x, 5*cos(3)*Ci(2*x) + 5*sin(3)*Si(2*x)
    assert_is_integral_of(f, F)
    f, F = cosh(x/2)/x, Chi(x/2)
    assert_is_integral_of(f, F)
    f, F = cos(x**2)/x, Ci(x**2)/2
    assert_is_integral_of(f, F)
    f, F = 1/log(2*x + 1), li(2*x + 1)/2
    assert_is_integral_of(f, F)
    f, F = polylog(2, 5*x)/x, polylog(3, 5*x)
    assert_is_integral_of(f, F)
    f, F = 5/sqrt(3 - 2*sin(x)**2), 5*sqrt(3)*elliptic_f(x, Rational(2, 3))/3
    assert_is_integral_of(f, F)
    f, F = sqrt(4 + 9*sin(x)**2), 2*elliptic_e(x, Rational(-9, 4))
    assert_is_integral_of(f, F)


def test_manualintegrate_derivative():
    assert manualintegrate(pi * Derivative(x**2 + 2*x + 3), x) == \
        pi * (x**2 + 2*x + 3)
    assert manualintegrate(Derivative(x**2 + 2*x + 3, y), x) == \
        Integral(Derivative(x**2 + 2*x + 3, y))
    assert manualintegrate(Derivative(sin(x), x, x, x, y), x) == \
        Derivative(sin(x), x, x, y)


def test_manualintegrate_Heaviside():
    assert_is_integral_of(DiracDelta(3*x+2), Heaviside(3*x+2)/3)
    assert_is_integral_of(DiracDelta(3*x, 0), Heaviside(3*x)/3)
    assert manualintegrate(DiracDelta(a+b*x, 1), x) == \
        Piecewise((DiracDelta(a + b*x)/b, Ne(b, 0)), (x*DiracDelta(a, 1), True))
    assert_is_integral_of(DiracDelta(x/3-1, 2), 3*DiracDelta(x/3-1, 1))
    assert manualintegrate(Heaviside(x), x) == x*Heaviside(x)
    assert manualintegrate(x*Heaviside(2), x) == x**2/2
    assert manualintegrate(x*Heaviside(-2), x) == 0
    assert manualintegrate(x*Heaviside( x), x) == x**2*Heaviside( x)/2
    assert manualintegrate(x*Heaviside(-x), x) == x**2*Heaviside(-x)/2
    assert manualintegrate(Heaviside(2*x + 4), x) == (x+2)*Heaviside(2*x + 4)
    assert manualintegrate(x*Heaviside(x), x) == x**2*Heaviside(x)/2
    assert manualintegrate(Heaviside(x + 1)*Heaviside(1 - x)*x**2, x) == \
        ((x**3/3 + Rational(1, 3))*Heaviside(x + 1) - Rational(2, 3))*Heaviside(-x + 1)

    y = Symbol('y')
    assert manualintegrate(sin(7 + x)*Heaviside(3*x - 7), x) == \
            (- cos(x + 7) + cos(Rational(28, 3)))*Heaviside(3*x - S(7))

    assert manualintegrate(sin(y + x)*Heaviside(3*x - y), x) == \
            (cos(y*Rational(4, 3)) - cos(x + y))*Heaviside(3*x - y)


def test_manualintegrate_orthogonal_poly():
    n = symbols('n')
    a, b = 7, Rational(5, 3)
    polys = [jacobi(n, a, b, x), gegenbauer(n, a, x), chebyshevt(n, x),
        chebyshevu(n, x), legendre(n, x), hermite(n, x), laguerre(n, x),
        assoc_laguerre(n, a, x)]
    for p in polys:
        integral = manualintegrate(p, x)
        for deg in [-2, -1, 0, 1, 3, 5, 8]:
            # some accept negative "degree", some do not
            try:
                p_subbed = p.subs(n, deg)
            except ValueError:
                continue
            assert (integral.subs(n, deg).diff(x) - p_subbed).expand() == 0

        # can also integrate simple expressions with these polynomials
        q = x*p.subs(x, 2*x + 1)
        integral = manualintegrate(q, x)
        for deg in [2, 4, 7]:
            assert (integral.subs(n, deg).diff(x) - q.subs(n, deg)).expand() == 0

        # cannot integrate with respect to any other parameter
        t = symbols('t')
        for i in range(len(p.args) - 1):
            new_args = list(p.args)
            new_args[i] = t
            assert isinstance(manualintegrate(p.func(*new_args), t), Integral)


@slow
def test_issue_6799():
    r, x, phi = map(Symbol, 'r x phi'.split())
    n = Symbol('n', integer=True, positive=True)

    integrand = (cos(n*(x-phi))*cos(n*x))
    limits = (x, -pi, pi)
    assert manualintegrate(integrand, x) == \
        ((n*x/2 + sin(2*n*x)/4)*cos(n*phi) - sin(n*phi)*cos(n*x)**2/2)/n
    assert r * integrate(integrand, limits).trigsimp() / pi == r * cos(n * phi)
    assert not integrate(integrand, limits).has(Dummy)


def test_issue_12251():
    assert manualintegrate(x**y, x) == Piecewise(
        (x**(y + 1)/(y + 1), Ne(y, -1)), (log(x), True))


def test_issue_3796():
    assert manualintegrate(diff(exp(x + x**2)), x) == exp(x + x**2)
    assert integrate(x * exp(x**4), x, risch=False) == -I*sqrt(pi)*erf(I*x**2)/4


def test_manual_true():
    assert integrate(exp(x) * sin(x), x, manual=True) == \
        (exp(x) * sin(x)) / 2 - (exp(x) * cos(x)) / 2
    assert integrate(sin(x) * cos(x), x, manual=True) in \
        [sin(x) ** 2 / 2, -cos(x)**2 / 2]


def test_issue_6746():
    y = Symbol('y')
    n = Symbol('n')
    assert manualintegrate(y**x, x) == Piecewise(
        (y**x/log(y), Ne(log(y), 0)), (x, True))
    assert manualintegrate(y**(n*x), x) == Piecewise(
        (Piecewise(
            (y**(n*x)/log(y), Ne(log(y), 0)),
            (n*x, True)
        )/n, Ne(n, 0)),
        (x, True))
    assert manualintegrate(exp(n*x), x) == Piecewise(
        (exp(n*x)/n, Ne(n, 0)), (x, True))

    y = Symbol('y', positive=True)
    assert manualintegrate((y + 1)**x, x) == (y + 1)**x/log(y + 1)
    y = Symbol('y', zero=True)
    assert manualintegrate((y + 1)**x, x) == x
    y = Symbol('y')
    n = Symbol('n', nonzero=True)
    assert manualintegrate(y**(n*x), x) == Piecewise(
        (y**(n*x)/log(y), Ne(log(y), 0)), (n*x, True))/n
    y = Symbol('y', positive=True)
    assert manualintegrate((y + 1)**(n*x), x) == \
        (y + 1)**(n*x)/(n*log(y + 1))
    a = Symbol('a', negative=True)
    b = Symbol('b')
    assert manualintegrate(1/(a + b*x**2), x) == atan(x/sqrt(a/b))/(b*sqrt(a/b))
    b = Symbol('b', negative=True)
    assert manualintegrate(1/(a + b*x**2), x) == \
        atan(x/(sqrt(-a)*sqrt(-1/b)))/(b*sqrt(-a)*sqrt(-1/b))
    assert manualintegrate(1/((x**a + y**b + 4)*sqrt(a*x**2 + 1)), x) == \
        y**(-b)*Integral(x**(-a)/(y**(-b)*sqrt(a*x**2 + 1) +
        x**(-a)*sqrt(a*x**2 + 1) + 4*x**(-a)*y**(-b)*sqrt(a*x**2 + 1)), x)
    assert manualintegrate(1/((x**2 + 4)*sqrt(4*x**2 + 1)), x) == \
        Integral(1/((x**2 + 4)*sqrt(4*x**2 + 1)), x)
    assert manualintegrate(1/(x - a**x + x*b**2), x) == \
        Integral(1/(-a**x + b**2*x + x), x)


@slow
def test_issue_2850():
    assert manualintegrate(asin(x)*log(x), x) == -x*asin(x) - sqrt(-x**2 + 1) \
            + (x*asin(x) + sqrt(-x**2 + 1))*log(x) - Integral(sqrt(-x**2 + 1)/x, x)
    assert manualintegrate(acos(x)*log(x), x) == -x*acos(x) + sqrt(-x**2 + 1) + \
        (x*acos(x) - sqrt(-x**2 + 1))*log(x) + Integral(sqrt(-x**2 + 1)/x, x)
    assert manualintegrate(atan(x)*log(x), x) == -x*atan(x) + (x*atan(x) - \
            log(x**2 + 1)/2)*log(x) + log(x**2 + 1)/2 + Integral(log(x**2 + 1)/x, x)/2


def test_issue_9462():
    assert manualintegrate(sin(2*x)*exp(x), x) == exp(x)*sin(2*x)/5 - 2*exp(x)*cos(2*x)/5
    assert not integral_steps(sin(2*x)*exp(x), x).contains_dont_know()
    assert manualintegrate((x - 3) / (x**2 - 2*x + 2)**2, x) == \
                           Integral(x/(x**4 - 4*x**3 + 8*x**2 - 8*x + 4), x) \
                           - 3*Integral(1/(x**4 - 4*x**3 + 8*x**2 - 8*x + 4), x)


def test_cyclic_parts():
    f = cos(x)*exp(x/4)
    F = 16*exp(x/4)*sin(x)/17 + 4*exp(x/4)*cos(x)/17
    assert manualintegrate(f, x) == F and F.diff(x) == f
    f = x*cos(x)*exp(x/4)
    F = (x*(16*exp(x/4)*sin(x)/17 + 4*exp(x/4)*cos(x)/17) -
        128*exp(x/4)*sin(x)/289 + 240*exp(x/4)*cos(x)/289)
    assert manualintegrate(f, x) == F and F.diff(x) == f


@slow
def test_issue_10847_slow():
    assert manualintegrate((4*x**4 + 4*x**3 + 16*x**2 + 12*x + 8)
                           / (x**6 + 2*x**5 + 3*x**4 + 4*x**3 + 3*x**2 + 2*x + 1), x) == \
                           2*x/(x**2 + 1) + 3*atan(x) - 1/(x**2 + 1) - 3/(x + 1)


@slow
def test_issue_10847():

    assert manualintegrate(x**2 / (x**2 - c), x) == \
        c*Piecewise((atan(x/sqrt(-c))/sqrt(-c), Ne(c, 0)), (-1/x, True)) + x

    rc = Symbol('c', real=True)
    assert manualintegrate(x**2 / (x**2 - rc), x) == \
        rc*Piecewise((atan(x/sqrt(-rc))/sqrt(-rc), rc < 0),
                     ((log(-sqrt(rc) + x) - log(sqrt(rc) + x))/(2*sqrt(rc)), True)) + x

    assert manualintegrate(sqrt(x - y) * log(z / x), x) == \
        4*y**2*Piecewise((atan(sqrt(x - y)/sqrt(y))/sqrt(y), Ne(y, 0)),
                         (-1/sqrt(x - y), True))/3 - 4*y*sqrt(x - y)/3 + \
        2*(x - y)**Rational(3, 2)*log(z/x)/3 + 4*(x - y)**Rational(3, 2)/9
    ry = Symbol('y', real=True)
    rz = Symbol('z', real=True)
    assert manualintegrate(sqrt(x - ry) * log(rz / x), x) == \
        4*ry**2*Piecewise((atan(sqrt(x - ry)/sqrt(ry))/sqrt(ry), ry > 0),
                         ((log(-sqrt(-ry) + sqrt(x - ry)) - log(sqrt(-ry) + sqrt(x - ry)))/(2*sqrt(-ry)), True))/3 \
                         - 4*ry*sqrt(x - ry)/3 + 2*(x - ry)**Rational(3, 2)*log(rz/x)/3 \
                         + 4*(x - ry)**Rational(3, 2)/9

    assert manualintegrate(sqrt(x) * log(x), x) == 2*x**Rational(3, 2)*log(x)/3 - 4*x**Rational(3, 2)/9

    result = manualintegrate(sqrt(a*x + b) / x, x)
    assert result == Piecewise((-2*b*Piecewise(
        (-atan(sqrt(a*x + b)/sqrt(-b))/sqrt(-b), Ne(b, 0)),
        (1/sqrt(a*x + b), True)) + 2*sqrt(a*x + b), Ne(a, 0)),
        (sqrt(b)*log(x), True))
    assert piecewise_fold(result) == Piecewise(
        (2*b*atan(sqrt(a*x + b)/sqrt(-b))/sqrt(-b) + 2*sqrt(a*x + b), Ne(a, 0) & Ne(b, 0)),
        (-2*b/sqrt(a*x + b) + 2*sqrt(a*x + b), Ne(a, 0)),
        (sqrt(b)*log(x), True))

    ra = Symbol('a', real=True)
    rb = Symbol('b', real=True)
    assert manualintegrate(sqrt(ra*x + rb) / x, x) == \
        Piecewise(
            (-2*rb*Piecewise(
                (-atan(sqrt(ra*x + rb)/sqrt(-rb))/sqrt(-rb), rb < 0),
                (-I*(log(-sqrt(rb) + sqrt(ra*x + rb)) - log(sqrt(rb) + sqrt(ra*x + rb)))/(2*sqrt(-rb)), True)) +
             2*sqrt(ra*x + rb), Ne(ra, 0)),
            (sqrt(rb)*log(x), True))

    assert expand(manualintegrate(sqrt(ra*x + rb) / (x + rc), x)) == \
           Piecewise((-2*ra*rc*Piecewise((atan(sqrt(ra*x + rb)/sqrt(ra*rc - rb))/sqrt(ra*rc - rb), ra*rc - rb > 0),
                                       (log(-sqrt(-ra*rc + rb) + sqrt(ra*x + rb))/(2*sqrt(-ra*rc + rb)) -
                                        log(sqrt(-ra*rc + rb) + sqrt(ra*x + rb))/(2*sqrt(-ra*rc + rb)), True)) +
                      2*rb*Piecewise((atan(sqrt(ra*x + rb)/sqrt(ra*rc - rb))/sqrt(ra*rc - rb), ra*rc - rb > 0),
                                    (log(-sqrt(-ra*rc + rb) + sqrt(ra*x + rb))/(2*sqrt(-ra*rc + rb)) -
                                     log(sqrt(-ra*rc + rb) + sqrt(ra*x + rb))/(2*sqrt(-ra*rc + rb)), True)) +
                      2*sqrt(ra*x + rb), Ne(ra, 0)), (sqrt(rb)*log(rc + x), True))

    assert manualintegrate(sqrt(2*x + 3) / (x + 1), x) == 2*sqrt(2*x + 3) - log(sqrt(2*x + 3) + 1) + log(sqrt(2*x + 3) - 1)
    assert manualintegrate(sqrt(2*x + 3) / 2 * x, x) == (2*x + 3)**Rational(5, 2)/20 - (2*x + 3)**Rational(3, 2)/4
    assert manualintegrate(x**Rational(3,2) * log(x), x) == 2*x**Rational(5,2)*log(x)/5 - 4*x**Rational(5,2)/25
    assert manualintegrate(x**(-3) * log(x), x) == -log(x)/(2*x**2) - 1/(4*x**2)
    assert manualintegrate(log(y)/(y**2*(1 - 1/y)), y) == \
        log(y)*log(-1 + 1/y) - Integral(log(-1 + 1/y)/y, y)


def test_issue_12899():
    assert manualintegrate(f(x,y).diff(x),y) == Integral(Derivative(f(x,y),x),y)
    assert manualintegrate(f(x,y).diff(y).diff(x),y) == Derivative(f(x,y),x)


def test_constant_independent_of_symbol():
    assert manualintegrate(Integral(y, (x, 1, 2)), x) == \
        x*Integral(y, (x, 1, 2))


def test_issue_12641():
    assert manualintegrate(sin(2*x), x) == -cos(2*x)/2
    assert manualintegrate(cos(x)*sin(2*x), x) == -2*cos(x)**3/3
    assert manualintegrate((sin(2*x)*cos(x))/(1 + cos(x)), x) == \
        -2*log(cos(x) + 1) - cos(x)**2 + 2*cos(x)


@slow
def test_issue_13297():
    assert manualintegrate(sin(x) * cos(x)**5, x) == -cos(x)**6 / 6


def test_issue_14470():
    assert_is_integral_of(1/(x*sqrt(x + 1)), log(sqrt(x + 1) - 1) - log(sqrt(x + 1) + 1))


@slow
def test_issue_9858():
    assert manualintegrate(exp(x)*cos(exp(x)), x) == sin(exp(x))
    assert manualintegrate(exp(2*x)*cos(exp(x)), x) == \
        exp(x)*sin(exp(x)) + cos(exp(x))
    res = manualintegrate(exp(10*x)*sin(exp(x)), x)
    assert not res.has(Integral)
    assert res.diff(x) == exp(10*x)*sin(exp(x))
    # an example with many similar integrations by parts
    assert manualintegrate(sum(x*exp(k*x) for k in range(1, 8)), x) == (
        x*exp(7*x)/7 + x*exp(6*x)/6 + x*exp(5*x)/5 + x*exp(4*x)/4 +
        x*exp(3*x)/3 + x*exp(2*x)/2 + x*exp(x) - exp(7*x)/49 -exp(6*x)/36 -
        exp(5*x)/25 - exp(4*x)/16 - exp(3*x)/9 - exp(2*x)/4 - exp(x))


def test_issue_8520():
    assert manualintegrate(x/(x**4 + 1), x) == atan(x**2)/2
    assert manualintegrate(x**2/(x**6 + 25), x) == atan(x**3/5)/15
    f = x/(9*x**4 + 4)**2
    assert manualintegrate(f, x).diff(x).factor() == f


def test_manual_subs():
    x, y = symbols('x y')
    expr = log(x) + exp(x)
    # if log(x) is y, then exp(y) is x
    assert manual_subs(expr, log(x), y) == y + exp(exp(y))
    # if exp(x) is y, then log(y) need not be x
    assert manual_subs(expr, exp(x), y) == log(x) + y

    raises(ValueError, lambda: manual_subs(expr, x))
    raises(ValueError, lambda: manual_subs(expr, exp(x), x, y))


@slow
def test_issue_15471():
    f = log(x)*cos(log(x))/x**Rational(3, 4)
    F = -128*x**Rational(1, 4)*sin(log(x))/289 + 240*x**Rational(1, 4)*cos(log(x))/289 + (16*x**Rational(1, 4)*sin(log(x))/17 + 4*x**Rational(1, 4)*cos(log(x))/17)*log(x)
    assert_is_integral_of(f, F)


def test_quadratic_denom():
    f = (5*x + 2)/(3*x**2 - 2*x + 8)
    assert manualintegrate(f, x) == 5*log(3*x**2 - 2*x + 8)/6 + 11*sqrt(23)*atan(3*sqrt(23)*(x - Rational(1, 3))/23)/69
    g = 3/(2*x**2 + 3*x + 1)
    assert manualintegrate(g, x) == 3*log(4*x + 2) - 3*log(4*x + 4)

def test_issue_22757():
    assert manualintegrate(sin(x), y) == y * sin(x)


def test_issue_23348():
    steps = integral_steps(tan(x), x)
    constant_times_step = steps.substep.substep
    assert constant_times_step.integrand == constant_times_step.constant * constant_times_step.other


def test_issue_23566():
    i = Integral(1/sqrt(x**2 - 1), (x, -2, -1)).doit(manual=True)
    assert i == -log(4 - 2*sqrt(3)) + log(2)
    assert str(i.n()) == '1.31695789692482'


def test_issue_25093():
    ap = Symbol('ap', positive=True)
    an = Symbol('an', negative=True)
    assert manualintegrate(exp(a*x**2 + b), x) == sqrt(pi)*exp(b)*erfi(sqrt(a)*x)/(2*sqrt(a))
    assert manualintegrate(exp(ap*x**2 + b), x) == sqrt(pi)*exp(b)*erfi(sqrt(ap)*x)/(2*sqrt(ap))
    assert manualintegrate(exp(an*x**2 + b), x) == -sqrt(pi)*exp(b)*erf(an*x/sqrt(-an))/(2*sqrt(-an))
    assert manualintegrate(sin(a*x**2 + b), x) == (
        sqrt(2)*sqrt(pi)*(sin(b)*fresnelc(sqrt(2)*sqrt(a)*x/sqrt(pi))
        + cos(b)*fresnels(sqrt(2)*sqrt(a)*x/sqrt(pi)))/(2*sqrt(a)))
    assert manualintegrate(cos(a*x**2 + b), x) == (
        sqrt(2)*sqrt(pi)*(-sin(b)*fresnels(sqrt(2)*sqrt(a)*x/sqrt(pi))
        + cos(b)*fresnelc(sqrt(2)*sqrt(a)*x/sqrt(pi)))/(2*sqrt(a)))


def test_nested_pow():
    assert_is_integral_of(sqrt(x**2), x*sqrt(x**2)/2)
    assert_is_integral_of(sqrt(x**(S(5)/3)), 6*x*sqrt(x**(S(5)/3))/11)
    assert_is_integral_of(1/sqrt(x**2), x*log(x)/sqrt(x**2))
    assert_is_integral_of(x*sqrt(x**(-4)), x**2*sqrt(x**-4)*log(x))
    f = (c*(a+b*x)**d)**e
    F1 = (c*(a + b*x)**d)**e*(a/b + x)/(d*e + 1)
    F2 = (c*(a + b*x)**d)**e*(a/b + x)*log(a/b + x)
    assert manualintegrate(f, x) == \
        Piecewise((Piecewise((F1, Ne(d*e, -1)), (F2, True)), Ne(b, 0)), (x*(a**d*c)**e, True))
    assert F1.diff(x).equals(f)
    assert F2.diff(x).subs(d*e, -1).equals(f)


def test_manualintegrate_sqrt_linear():
    assert_is_integral_of((5*x**3+4)/sqrt(2+3*x),
                          10*(3*x + 2)**(S(7)/2)/567 - 4*(3*x + 2)**(S(5)/2)/27 +
                          40*(3*x + 2)**(S(3)/2)/81 + 136*sqrt(3*x + 2)/81)
    assert manualintegrate(x/sqrt(a+b*x)**3, x) == \
        Piecewise((Mul(2, b**-2, a/sqrt(a + b*x) + sqrt(a + b*x)), Ne(b, 0)), (x**2/(2*a**(S(3)/2)), True))
    assert_is_integral_of((sqrt(3*x+3)+1)/((2*x+2)**(1/S(3))+1),
                          3*sqrt(6)*(2*x + 2)**(S(7)/6)/14 - 3*sqrt(6)*(2*x + 2)**(S(5)/6)/10 -
                          3*sqrt(6)*(2*x + 2)**(S.One/6)/2 + 3*(2*x + 2)**(S(2)/3)/4 - 3*(2*x + 2)**(S.One/3)/2 +
                          sqrt(6)*sqrt(2*x + 2)/2 + 3*log((2*x + 2)**(S.One/3) + 1)/2 +
                          3*sqrt(6)*atan((2*x + 2)**(S.One/6))/2)
    assert_is_integral_of(sqrt(x+sqrt(x)),
                          2*sqrt(sqrt(x) + x)*(sqrt(x)/12 + x/3 - S(1)/8) + log(2*sqrt(x) + 2*sqrt(sqrt(x) + x) + 1)/8)
    assert_is_integral_of(sqrt(2*x+3+sqrt(4*x+5))**3,
                          sqrt(2*x + sqrt(4*x + 5) + 3) *
                          (9*x/10 + 11*(4*x + 5)**(S(3)/2)/40 + sqrt(4*x + 5)/40 + (4*x + 5)**2/10 + S(11)/10)/2)


def test_manualintegrate_sqrt_quadratic():
    assert_is_integral_of(1/sqrt((x - I)**2-1), log(2*x + 2*sqrt(x**2 - 2*I*x - 2) - 2*I))
    assert_is_integral_of(1/sqrt(3*x**2+4*x+5), sqrt(3)*asinh(3*sqrt(11)*(x + S(2)/3)/11)/3)
    assert_is_integral_of(1/sqrt(-3*x**2+4*x+5), sqrt(3)*asin(3*sqrt(19)*(x - S(2)/3)/19)/3)
    assert_is_integral_of(1/sqrt(3*x**2+4*x-5), sqrt(3)*log(6*x + 2*sqrt(3)*sqrt(3*x**2 + 4*x - 5) + 4)/3)
    assert_is_integral_of(1/sqrt(4*x**2-4*x+1), (x - S.Half)*log(x - S.Half)/(2*sqrt((x - S.Half)**2)))
    assert manualintegrate(1/sqrt(a+b*x+c*x**2), x) == \
        Piecewise((log(b + 2*sqrt(c)*sqrt(a + b*x + c*x**2) + 2*c*x)/sqrt(c), Ne(c, 0) & Ne(a - b**2/(4*c), 0)),
                  ((b/(2*c) + x)*log(b/(2*c) + x)/sqrt(c*(b/(2*c) + x)**2), Ne(c, 0)),
                  (2*sqrt(a + b*x)/b, Ne(b, 0)), (x/sqrt(a), True))

    assert_is_integral_of((7*x+6)/sqrt(3*x**2+4*x+5),
                          7*sqrt(3*x**2 + 4*x + 5)/3 + 4*sqrt(3)*asinh(3*sqrt(11)*(x + S(2)/3)/11)/9)
    assert_is_integral_of((7*x+6)/sqrt(-3*x**2+4*x+5),
                          -7*sqrt(-3*x**2 + 4*x + 5)/3 + 32*sqrt(3)*asin(3*sqrt(19)*(x - S(2)/3)/19)/9)
    assert_is_integral_of((7*x+6)/sqrt(3*x**2+4*x-5),
                          7*sqrt(3*x**2 + 4*x - 5)/3 + 4*sqrt(3)*log(6*x + 2*sqrt(3)*sqrt(3*x**2 + 4*x - 5) + 4)/9)
    assert manualintegrate((d+e*x)/sqrt(a+b*x+c*x**2), x) == \
        Piecewise(((-b*e/(2*c) + d) *
                   Piecewise((log(b + 2*sqrt(c)*sqrt(a + b*x + c*x**2) + 2*c*x)/sqrt(c), Ne(a - b**2/(4*c), 0)),
                             ((b/(2*c) + x)*log(b/(2*c) + x)/sqrt(c*(b/(2*c) + x)**2), True)) +
                   e*sqrt(a + b*x + c*x**2)/c, Ne(c, 0)),
                  ((2*d*sqrt(a + b*x) + 2*e*(-a*sqrt(a + b*x) + (a + b*x)**(S(3)/2)/3)/b)/b, Ne(b, 0)),
                  ((d*x + e*x**2/2)/sqrt(a), True))

    assert manualintegrate((3*x**3-x**2+2*x-4)/sqrt(x**2-3*x+2), x) == \
        sqrt(x**2 - 3*x + 2)*(x**2 + 13*x/4 + S(101)/8) + 135*log(2*x + 2*sqrt(x**2 - 3*x + 2) - 3)/16

    assert_is_integral_of(sqrt(53225*x**2-66732*x+23013),
                          (x/2 - S(16683)/53225)*sqrt(53225*x**2 - 66732*x + 23013) +
                          111576969*sqrt(2129)*asinh(53225*x/10563 - S(11122)/3521)/1133160250)
    assert manualintegrate(sqrt(a+c*x**2), x) == \
        Piecewise((a*Piecewise((log(2*sqrt(c)*sqrt(a + c*x**2) + 2*c*x)/sqrt(c), Ne(a, 0)),
                               (x*log(x)/sqrt(c*x**2), True))/2 + x*sqrt(a + c*x**2)/2, Ne(c, 0)),
                  (sqrt(a)*x, True))
    assert manualintegrate(sqrt(a+b*x+c*x**2), x) == \
        Piecewise(((a/2 - b**2/(8*c)) *
                   Piecewise((log(b + 2*sqrt(c)*sqrt(a + b*x + c*x**2) + 2*c*x)/sqrt(c), Ne(a - b**2/(4*c), 0)),
                             ((b/(2*c) + x)*log(b/(2*c) + x)/sqrt(c*(b/(2*c) + x)**2), True)) +
                   (b/(4*c) + x/2)*sqrt(a + b*x + c*x**2), Ne(c, 0)),
                  (2*(a + b*x)**(S(3)/2)/(3*b), Ne(b, 0)),
                  (sqrt(a)*x, True))

    assert_is_integral_of(x*sqrt(x**2+2*x+4),
                          (x**2/3 + x/6 + S(5)/6)*sqrt(x**2 + 2*x + 4) - 3*asinh(sqrt(3)*(x + 1)/3)/2)


def test_mul_pow_derivative():
    assert_is_integral_of(x*sec(x)*tan(x), x*sec(x) - log(tan(x) + sec(x)))
    assert_is_integral_of(x*sec(x)**2, x*tan(x) + log(cos(x)))
    assert_is_integral_of(x**3*Derivative(f(x), (x, 4)),
                          x**3*Derivative(f(x), (x, 3)) - 3*x**2*Derivative(f(x), (x, 2)) +
                          6*x*Derivative(f(x), x) - 6*f(x))
