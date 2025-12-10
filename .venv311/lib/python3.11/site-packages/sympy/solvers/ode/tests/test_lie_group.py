from sympy.core.function import Function
from sympy.core.numbers import Rational
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (atan, sin, tan)

from sympy.solvers.ode import (classify_ode, checkinfsol, dsolve, infinitesimals)

from sympy.solvers.ode.subscheck import checkodesol

from sympy.testing.pytest import XFAIL


C1 = Symbol('C1')
x, y = symbols("x y")
f = Function('f')
xi = Function('xi')
eta = Function('eta')


def test_heuristic1():
    a, b, c, a4, a3, a2, a1, a0 = symbols("a b c a4 a3 a2 a1 a0")
    df = f(x).diff(x)
    eq = Eq(df, x**2*f(x))
    eq1 = f(x).diff(x) + a*f(x) - c*exp(b*x)
    eq2 = f(x).diff(x) + 2*x*f(x) - x*exp(-x**2)
    eq3 = (1 + 2*x)*df + 2 - 4*exp(-f(x))
    eq4 = f(x).diff(x) - (a4*x**4 + a3*x**3 + a2*x**2 + a1*x + a0)**Rational(-1, 2)
    eq5 = x**2*df - f(x) + x**2*exp(x - (1/x))
    eqlist = [eq, eq1, eq2, eq3, eq4, eq5]

    i = infinitesimals(eq, hint='abaco1_simple')
    assert i == [{eta(x, f(x)): exp(x**3/3), xi(x, f(x)): 0},
        {eta(x, f(x)): f(x), xi(x, f(x)): 0},
        {eta(x, f(x)): 0, xi(x, f(x)): x**(-2)}]
    i1 = infinitesimals(eq1, hint='abaco1_simple')
    assert i1 == [{eta(x, f(x)): exp(-a*x), xi(x, f(x)): 0}]
    i2 = infinitesimals(eq2, hint='abaco1_simple')
    assert i2 == [{eta(x, f(x)): exp(-x**2), xi(x, f(x)): 0}]
    i3 = infinitesimals(eq3, hint='abaco1_simple')
    assert i3 == [{eta(x, f(x)): 0, xi(x, f(x)): 2*x + 1},
        {eta(x, f(x)): 0, xi(x, f(x)): 1/(exp(f(x)) - 2)}]
    i4 = infinitesimals(eq4, hint='abaco1_simple')
    assert i4 == [{eta(x, f(x)): 1, xi(x, f(x)): 0},
        {eta(x, f(x)): 0,
        xi(x, f(x)): sqrt(a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4)}]
    i5 = infinitesimals(eq5, hint='abaco1_simple')
    assert i5 == [{xi(x, f(x)): 0, eta(x, f(x)): exp(-1/x)}]

    ilist = [i, i1, i2, i3, i4, i5]
    for eq, i in (zip(eqlist, ilist)):
        check = checkinfsol(eq, i)
        assert check[0]

    # This ODE can be solved by the Lie Group method, when there are
    # better assumptions
    eq6 = df - (f(x)/x)*(x*log(x**2/f(x)) + 2)
    i = infinitesimals(eq6, hint='abaco1_product')
    assert i == [{eta(x, f(x)): f(x)*exp(-x), xi(x, f(x)): 0}]
    assert checkinfsol(eq6, i)[0]

    eq7 = x*(f(x).diff(x)) + 1 - f(x)**2
    i = infinitesimals(eq7, hint='chi')
    assert checkinfsol(eq7, i)[0]


def test_heuristic3():
    a, b = symbols("a b")
    df = f(x).diff(x)

    eq = x**2*df + x*f(x) + f(x)**2 + x**2
    i = infinitesimals(eq, hint='bivariate')
    assert i == [{eta(x, f(x)): f(x), xi(x, f(x)): x}]
    assert checkinfsol(eq, i)[0]

    eq = x**2*(-f(x)**2 + df)- a*x**2*f(x) + 2 - a*x
    i = infinitesimals(eq, hint='bivariate')
    assert checkinfsol(eq, i)[0]


def test_heuristic_function_sum():
    eq = f(x).diff(x) - (3*(1 + x**2/f(x)**2)*atan(f(x)/x) + (1 - 2*f(x))/x +
       (1 - 3*f(x))*(x/f(x)**2))
    i = infinitesimals(eq, hint='function_sum')
    assert i == [{eta(x, f(x)): f(x)**(-2) + x**(-2), xi(x, f(x)): 0}]
    assert checkinfsol(eq, i)[0]


def test_heuristic_abaco2_similar():
    a, b = symbols("a b")
    F = Function('F')
    eq = f(x).diff(x) - F(a*x + b*f(x))
    i = infinitesimals(eq, hint='abaco2_similar')
    assert i == [{eta(x, f(x)): -a/b, xi(x, f(x)): 1}]
    assert checkinfsol(eq, i)[0]

    eq = f(x).diff(x) - (f(x)**2 / (sin(f(x) - x) - x**2 + 2*x*f(x)))
    i = infinitesimals(eq, hint='abaco2_similar')
    assert i == [{eta(x, f(x)): f(x)**2, xi(x, f(x)): f(x)**2}]
    assert checkinfsol(eq, i)[0]


def test_heuristic_abaco2_unique_unknown():

    a, b = symbols("a b")
    F = Function('F')
    eq = f(x).diff(x) - x**(a - 1)*(f(x)**(1 - b))*F(x**a/a + f(x)**b/b)
    i = infinitesimals(eq, hint='abaco2_unique_unknown')
    assert i == [{eta(x, f(x)): -f(x)*f(x)**(-b), xi(x, f(x)): x*x**(-a)}]
    assert checkinfsol(eq, i)[0]

    eq = f(x).diff(x) + tan(F(x**2 + f(x)**2) + atan(x/f(x)))
    i = infinitesimals(eq, hint='abaco2_unique_unknown')
    assert i == [{eta(x, f(x)): x, xi(x, f(x)): -f(x)}]
    assert checkinfsol(eq, i)[0]

    eq = (x*f(x).diff(x) + f(x) + 2*x)**2 -4*x*f(x) -4*x**2 -4*a
    i = infinitesimals(eq, hint='abaco2_unique_unknown')
    assert checkinfsol(eq, i)[0]


def test_heuristic_linear():
    a, b, m, n = symbols("a b m n")

    eq = x**(n*(m + 1) - m)*(f(x).diff(x)) - a*f(x)**n -b*x**(n*(m + 1))
    i = infinitesimals(eq, hint='linear')
    assert checkinfsol(eq, i)[0]


@XFAIL
def test_kamke():
    a, b, alpha, c = symbols("a b alpha c")
    eq = x**2*(a*f(x)**2+(f(x).diff(x))) + b*x**alpha + c
    i = infinitesimals(eq, hint='sum_function')  # XFAIL
    assert checkinfsol(eq, i)[0]


def test_user_infinitesimals():
    x = Symbol("x") # assuming x is real generates an error
    eq = x*(f(x).diff(x)) + 1 - f(x)**2
    sol = Eq(f(x), (C1 + x**2)/(C1 - x**2))
    infinitesimals = {'xi':sqrt(f(x) - 1)/sqrt(f(x) + 1), 'eta':0}
    assert dsolve(eq, hint='lie_group', **infinitesimals) == sol
    assert checkodesol(eq, sol) == (True, 0)


@XFAIL
def test_lie_group_issue15219():
    eqn = exp(f(x).diff(x)-f(x))
    assert 'lie_group' not in classify_ode(eqn, f(x))
