from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Eq, Ne
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (asinh, cosh, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin, tan)
from sympy.functions.special.bessel import (besselj, besselk, bessely, jn)
from sympy.functions.special.error_functions import erf
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import And
from sympy.matrices import Matrix
from sympy.simplify.ratsimp import ratsimp
from sympy.simplify.simplify import simplify
from sympy.integrals.heurisch import components, heurisch, heurisch_wrapper
from sympy.testing.pytest import XFAIL, slow
from sympy.integrals.integrals import integrate
x, y, z, nu = symbols('x,y,z,nu')
f = Function('f')


def test_components():
    assert components(x*y, x) == {x}
    assert components(1/(x + y), x) == {x}
    assert components(sin(x), x) == {sin(x), x}
    assert components(sin(x)*sqrt(log(x)), x) == \
        {log(x), sin(x), sqrt(log(x)), x}
    assert components(x*sin(exp(x)*y), x) == \
        {sin(y*exp(x)), x, exp(x)}
    assert components(x**Rational(17, 54)/sqrt(sin(x)), x) == \
        {sin(x), x**Rational(1, 54), sqrt(sin(x)), x}

    assert components(f(x), x) == \
        {x, f(x)}
    assert components(Derivative(f(x), x), x) == \
        {x, f(x), Derivative(f(x), x)}
    assert components(f(x)*diff(f(x), x), x) == \
        {x, f(x), Derivative(f(x), x), Derivative(f(x), x)}


def test_issue_10680():
    assert isinstance(integrate(x**log(x**log(x**log(x))),x), Integral)


def test_issue_21166():
    assert integrate(sin(x/sqrt(abs(x))), (x, -1, 1)) == 0


def test_heurisch_polynomials():
    assert heurisch(1, x) == x
    assert heurisch(x, x) == x**2/2
    assert heurisch(x**17, x) == x**18/18
    # For coverage
    assert heurisch_wrapper(y, x) == y*x


def test_heurisch_fractions():
    assert heurisch(1/x, x) == log(x)
    assert heurisch(1/(2 + x), x) == log(x + 2)
    assert heurisch(1/(x + sin(y)), x) == log(x + sin(y))

    # Up to a constant, where C = pi*I*Rational(5, 12), Mathematica gives identical
    # result in the first case. The difference is because SymPy changes
    # signs of expressions without any care.
    # XXX ^ ^ ^ is this still correct?
    assert heurisch(5*x**5/(
        2*x**6 - 5), x) in [5*log(2*x**6 - 5) / 12, 5*log(-2*x**6 + 5) / 12]
    assert heurisch(5*x**5/(2*x**6 + 5), x) == 5*log(2*x**6 + 5) / 12

    assert heurisch(1/x**2, x) == -1/x
    assert heurisch(-1/x**5, x) == 1/(4*x**4)


def test_heurisch_log():
    assert heurisch(log(x), x) == x*log(x) - x
    assert heurisch(log(3*x), x) == -x + x*log(3) + x*log(x)
    assert heurisch(log(x**2), x) in [x*log(x**2) - 2*x, 2*x*log(x) - 2*x]


def test_heurisch_exp():
    assert heurisch(exp(x), x) == exp(x)
    assert heurisch(exp(-x), x) == -exp(-x)
    assert heurisch(exp(17*x), x) == exp(17*x) / 17
    assert heurisch(x*exp(x), x) == x*exp(x) - exp(x)
    assert heurisch(x*exp(x**2), x) == exp(x**2) / 2

    assert heurisch(exp(-x**2), x) is None

    assert heurisch(2**x, x) == 2**x/log(2)
    assert heurisch(x*2**x, x) == x*2**x/log(2) - 2**x*log(2)**(-2)

    assert heurisch(Integral(x**z*y, (y, 1, 2), (z, 2, 3)).function, x) == (x*x**z*y)/(z+1)
    assert heurisch(Sum(x**z, (z, 1, 2)).function, z) == x**z/log(x)

    # https://github.com/sympy/sympy/issues/23707
    anti = -exp(z)/(sqrt(x - y)*exp(z*sqrt(x - y)) - exp(z*sqrt(x - y)))
    assert heurisch(exp(z)*exp(-z*sqrt(x - y)), z) == anti


def test_heurisch_trigonometric():
    assert heurisch(sin(x), x) == -cos(x)
    assert heurisch(pi*sin(x) + 1, x) == x - pi*cos(x)

    assert heurisch(cos(x), x) == sin(x)
    assert heurisch(tan(x), x) in [
        log(1 + tan(x)**2)/2,
        log(tan(x) + I) + I*x,
        log(tan(x) - I) - I*x,
    ]

    assert heurisch(sin(x)*sin(y), x) == -cos(x)*sin(y)
    assert heurisch(sin(x)*sin(y), y) == -cos(y)*sin(x)

    # gives sin(x) in answer when run via setup.py and cos(x) when run via py.test
    assert heurisch(sin(x)*cos(x), x) in [sin(x)**2 / 2, -cos(x)**2 / 2]
    assert heurisch(cos(x)/sin(x), x) == log(sin(x))

    assert heurisch(x*sin(7*x), x) == sin(7*x) / 49 - x*cos(7*x) / 7
    assert heurisch(1/pi/4 * x**2*cos(x), x) == 1/pi/4*(x**2*sin(x) -
                    2*sin(x) + 2*x*cos(x))

    assert heurisch(acos(x/4) * asin(x/4), x) == 2*x - (sqrt(16 - x**2))*asin(x/4) \
        + (sqrt(16 - x**2))*acos(x/4) + x*asin(x/4)*acos(x/4)

    assert heurisch(sin(x)/(cos(x)**2+1), x) == -atan(cos(x)) #fixes issue 13723
    assert heurisch(1/(cos(x)+2), x) == 2*sqrt(3)*atan(sqrt(3)*tan(x/2)/3)/3
    assert heurisch(2*sin(x)*cos(x)/(sin(x)**4 + 1), x) == atan(sqrt(2)*sin(x)
        - 1) - atan(sqrt(2)*sin(x) + 1)

    assert heurisch(1/cosh(x), x) == 2*atan(tanh(x/2))


def test_heurisch_hyperbolic():
    assert heurisch(sinh(x), x) == cosh(x)
    assert heurisch(cosh(x), x) == sinh(x)

    assert heurisch(x*sinh(x), x) == x*cosh(x) - sinh(x)
    assert heurisch(x*cosh(x), x) == x*sinh(x) - cosh(x)

    assert heurisch(
        x*asinh(x/2), x) == x**2*asinh(x/2)/2 + asinh(x/2) - x*sqrt(4 + x**2)/4


def test_heurisch_mixed():
    assert heurisch(sin(x)*exp(x), x) == exp(x)*sin(x)/2 - exp(x)*cos(x)/2
    assert heurisch(sin(x/sqrt(-x)), x) == 2*x*cos(x/sqrt(-x))/sqrt(-x) - 2*sin(x/sqrt(-x))


def test_heurisch_radicals():
    assert heurisch(1/sqrt(x), x) == 2*sqrt(x)
    assert heurisch(1/sqrt(x)**3, x) == -2/sqrt(x)
    assert heurisch(sqrt(x)**3, x) == 2*sqrt(x)**5/5

    assert heurisch(sin(x)*sqrt(cos(x)), x) == -2*sqrt(cos(x))**3/3
    y = Symbol('y')
    assert heurisch(sin(y*sqrt(x)), x) == 2/y**2*sin(y*sqrt(x)) - \
        2*sqrt(x)*cos(y*sqrt(x))/y
    assert heurisch_wrapper(sin(y*sqrt(x)), x) == Piecewise(
        (-2*sqrt(x)*cos(sqrt(x)*y)/y + 2*sin(sqrt(x)*y)/y**2, Ne(y, 0)),
        (0, True))
    y = Symbol('y', positive=True)
    assert heurisch_wrapper(sin(y*sqrt(x)), x) == 2/y**2*sin(y*sqrt(x)) - \
        2*sqrt(x)*cos(y*sqrt(x))/y


def test_heurisch_special():
    assert heurisch(erf(x), x) == x*erf(x) + exp(-x**2)/sqrt(pi)
    assert heurisch(exp(-x**2)*erf(x), x) == sqrt(pi)*erf(x)**2 / 4


def test_heurisch_symbolic_coeffs():
    assert heurisch(1/(x + y), x) == log(x + y)
    assert heurisch(1/(x + sqrt(2)), x) == log(x + sqrt(2))
    assert simplify(diff(heurisch(log(x + y + z), y), y)) == log(x + y + z)


def test_heurisch_symbolic_coeffs_1130():
    y = Symbol('y')
    assert heurisch_wrapper(1/(x**2 + y), x) == Piecewise(
    (log(x - sqrt(-y))/(2*sqrt(-y)) - log(x + sqrt(-y))/(2*sqrt(-y)),
    Ne(y, 0)), (-1/x, True))
    y = Symbol('y', positive=True)
    assert heurisch_wrapper(1/(x**2 + y), x) == (atan(x/sqrt(y))/sqrt(y))


def test_heurisch_hacking():
    assert heurisch(sqrt(1 + 7*x**2), x, hints=[]) == \
        x*sqrt(1 + 7*x**2)/2 + sqrt(7)*asinh(sqrt(7)*x)/14
    assert heurisch(sqrt(1 - 7*x**2), x, hints=[]) == \
        x*sqrt(1 - 7*x**2)/2 + sqrt(7)*asin(sqrt(7)*x)/14

    assert heurisch(1/sqrt(1 + 7*x**2), x, hints=[]) == \
        sqrt(7)*asinh(sqrt(7)*x)/7
    assert heurisch(1/sqrt(1 - 7*x**2), x, hints=[]) == \
        sqrt(7)*asin(sqrt(7)*x)/7

    assert heurisch(exp(-7*x**2), x, hints=[]) == \
        sqrt(7*pi)*erf(sqrt(7)*x)/14

    assert heurisch(1/sqrt(9 - 4*x**2), x, hints=[]) == \
        asin(x*Rational(2, 3))/2

    assert heurisch(1/sqrt(9 + 4*x**2), x, hints=[]) == \
        asinh(x*Rational(2, 3))/2

    assert heurisch(1/sqrt(3*x**2-4), x, hints=[]) == \
           sqrt(3)*log(3*x + sqrt(3)*sqrt(3*x**2 - 4))/3


def test_heurisch_function():
    assert heurisch(f(x), x) is None

@XFAIL
def test_heurisch_function_derivative():
    # TODO: it looks like this used to work just by coincindence and
    # thanks to sloppy implementation. Investigate why this used to
    # work at all and if support for this can be restored.

    df = diff(f(x), x)

    assert heurisch(f(x)*df, x) == f(x)**2/2
    assert heurisch(f(x)**2*df, x) == f(x)**3/3
    assert heurisch(df/f(x), x) == log(f(x))


def test_heurisch_wrapper():
    f = 1/(y + x)
    assert heurisch_wrapper(f, x) == log(x + y)
    f = 1/(y - x)
    assert heurisch_wrapper(f, x) == -log(x - y)
    f = 1/((y - x)*(y + x))
    assert heurisch_wrapper(f, x) == Piecewise(
        (-log(x - y)/(2*y) + log(x + y)/(2*y), Ne(y, 0)), (1/x, True))
    # issue 6926
    f = sqrt(x**2/((y - x)*(y + x)))
    assert heurisch_wrapper(f, x) == x*sqrt(-x**2/(x**2 - y**2)) \
    - y**2*sqrt(-x**2/(x**2 - y**2))/x


def test_issue_3609():
    assert heurisch(1/(x * (1 + log(x)**2)), x) == atan(log(x))

### These are examples from the Poor Man's Integrator
### http://www-sop.inria.fr/cafe/Manuel.Bronstein/pmint/examples/


def test_pmint_rat():
    # TODO: heurisch() is off by a constant: -3/4. Possibly different permutation
    # would give the optimal result?

    def drop_const(expr, x):
        if expr.is_Add:
            return Add(*[ arg for arg in expr.args if arg.has(x) ])
        else:
            return expr

    f = (x**7 - 24*x**4 - 4*x**2 + 8*x - 8)/(x**8 + 6*x**6 + 12*x**4 + 8*x**2)
    g = (4 + 8*x**2 + 6*x + 3*x**3)/(x**5 + 4*x**3 + 4*x) + log(x)

    assert drop_const(ratsimp(heurisch(f, x)), x) == g


def test_pmint_trig():
    f = (x - tan(x)) / tan(x)**2 + tan(x)
    g = -x**2/2 - x/tan(x) + log(tan(x)**2 + 1)/2

    assert heurisch(f, x) == g


def test_pmint_logexp():
    f = (1 + x + x*exp(x))*(x + log(x) + exp(x) - 1)/(x + log(x) + exp(x))**2/x
    g = log(x + exp(x) + log(x)) + 1/(x + exp(x) + log(x))

    assert ratsimp(heurisch(f, x)) == g


def test_pmint_erf():
    f = exp(-x**2)*erf(x)/(erf(x)**3 - erf(x)**2 - erf(x) + 1)
    g = sqrt(pi)*log(erf(x) - 1)/8 - sqrt(pi)*log(erf(x) + 1)/8 - sqrt(pi)/(4*erf(x) - 4)

    assert ratsimp(heurisch(f, x)) == g


def test_pmint_LambertW():
    f = LambertW(x)
    g = x*LambertW(x) - x + x/LambertW(x)

    assert heurisch(f, x) == g


def test_pmint_besselj():
    f = besselj(nu + 1, x)/besselj(nu, x)
    g = nu*log(x) - log(besselj(nu, x))

    assert heurisch(f, x) == g

    f = (nu*besselj(nu, x) - x*besselj(nu + 1, x))/x
    g = besselj(nu, x)

    assert heurisch(f, x) == g

    f = jn(nu + 1, x)/jn(nu, x)
    g = nu*log(x) - log(jn(nu, x))

    assert heurisch(f, x) == g


@slow
def test_pmint_bessel_products():
    f = x*besselj(nu, x)*bessely(nu, 2*x)
    g = -2*x*besselj(nu, x)*bessely(nu - 1, 2*x)/3 + x*besselj(nu - 1, x)*bessely(nu, 2*x)/3

    assert heurisch(f, x) == g

    f = x*besselj(nu, x)*besselk(nu, 2*x)
    g = -2*x*besselj(nu, x)*besselk(nu - 1, 2*x)/5 - x*besselj(nu - 1, x)*besselk(nu, 2*x)/5

    assert heurisch(f, x) == g


def test_pmint_WrightOmega():
    def omega(x):
        return LambertW(exp(x))

    f = (1 + omega(x) * (2 + cos(omega(x)) * (x + omega(x))))/(1 + omega(x))/(x + omega(x))
    g = log(x + LambertW(exp(x))) + sin(LambertW(exp(x)))

    assert heurisch(f, x) == g


def test_RR():
    # Make sure the algorithm does the right thing if the ring is RR. See
    # issue 8685.
    assert heurisch(sqrt(1 + 0.25*x**2), x, hints=[]) == \
        0.5*x*sqrt(0.25*x**2 + 1) + 1.0*asinh(0.5*x)

# TODO: convert the rest of PMINT tests:
# Airy functions
# f = (x - AiryAi(x)*AiryAi(1, x)) / (x**2 - AiryAi(x)**2)
# g = Rational(1,2)*ln(x + AiryAi(x)) + Rational(1,2)*ln(x - AiryAi(x))
# f = x**2 * AiryAi(x)
# g = -AiryAi(x) + AiryAi(1, x)*x
# Whittaker functions
# f = WhittakerW(mu + 1, nu, x) / (WhittakerW(mu, nu, x) * x)
# g = x/2 - mu*ln(x) - ln(WhittakerW(mu, nu, x))


def test_issue_22527():
    t, R = symbols(r't R')
    z = Function('z')(t)
    def f(x):
      return x/sqrt(R**2 - x**2)
    Uz = integrate(f(z), z)
    Ut = integrate(f(t), t)
    assert Ut == Uz.subs(z, t)


def test_heurisch_complex_erf_issue_26338():
    r = symbols('r', real=True)
    a = exp(-r**2/(2*(2 - I)**2))
    assert heurisch(a, r, hints=[]) is None  # None, not a wrong soln
    a = sqrt(pi)*erf((1 + I)/2)/2
    assert integrate(exp(-I*r**2/2), (r, 0, 1)) == a - I*a

    a = exp(-x**2/(2*(2 - I)**2))
    assert heurisch(a, x, hints=[]) is None  # None, not a wrong soln
    a = sqrt(pi)*erf((1 + I)/2)/2
    assert integrate(exp(-I*x**2/2), (x, 0, 1)) == a - I*a


def test_issue_15498():
    Z0 = Function('Z0')
    k01, k10, t, s= symbols('k01 k10 t s', real=True, positive=True)
    m = Matrix([[exp(-k10*t)]])
    _83 = Rational(83, 100)  # 0.83 works, too
    [a, b, c, d, e, f, g] = [100, 0.5, _83, 50, 0.6, 2, 120]
    AIF_btf = a*(d*e*(1 - exp(-(t - b)/e)) + f*g*(1 - exp(-(t - b)/g)))
    AIF_atf = a*(d*e*exp(-(t - b)/e)*(exp((c - b)/e) - 1
        ) + f*g*exp(-(t - b)/g)*(exp((c - b)/g) - 1))
    AIF_sym = Piecewise((0, t < b), (AIF_btf, And(b <= t, t < c)), (AIF_atf, c <= t))
    aif_eq = Eq(Z0(t), AIF_sym)
    f_vec = Matrix([[k01*Z0(t)]])
    integrand = m*m.subs(t, s)**-1*f_vec.subs(aif_eq.lhs, aif_eq.rhs).subs(t, s)
    solution = integrate(integrand[0], (s, 0, t))
    assert solution is not None  # does not hang and takes less than 10 s
