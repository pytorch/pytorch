from sympy.core.function import (Derivative, Function, diff)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.error_functions import (Ei, erf, erfi)
from sympy.integrals.integrals import Integral

from sympy.solvers.ode.subscheck import checkodesol, checksysodesol

from sympy.functions import besselj, bessely

from sympy.testing.pytest import raises, slow


C0, C1, C2, C3, C4 = symbols('C0:5')
u, x, y, z = symbols('u,x:z', real=True)
f = Function('f')
g = Function('g')
h = Function('h')


@slow
def test_checkodesol():
    # For the most part, checkodesol is well tested in the tests below.
    # These tests only handle cases not checked below.
    raises(ValueError, lambda: checkodesol(f(x, y).diff(x), Eq(f(x, y), x)))
    raises(ValueError, lambda: checkodesol(f(x).diff(x), Eq(f(x, y),
           x), f(x, y)))
    assert checkodesol(f(x).diff(x), Eq(f(x, y), x)) == \
        (False, -f(x).diff(x) + f(x, y).diff(x) - 1)
    assert checkodesol(f(x).diff(x), Eq(f(x), x)) is not True
    assert checkodesol(f(x).diff(x), Eq(f(x), x)) == (False, 1)
    sol1 = Eq(f(x)**5 + 11*f(x) - 2*f(x) + x, 0)
    assert checkodesol(diff(sol1.lhs, x), sol1) == (True, 0)
    assert checkodesol(diff(sol1.lhs, x)*exp(f(x)), sol1) == (True, 0)
    assert checkodesol(diff(sol1.lhs, x, 2), sol1) == (True, 0)
    assert checkodesol(diff(sol1.lhs, x, 2)*exp(f(x)), sol1) == (True, 0)
    assert checkodesol(diff(sol1.lhs, x, 3), sol1) == (True, 0)
    assert checkodesol(diff(sol1.lhs, x, 3)*exp(f(x)), sol1) == (True, 0)
    assert checkodesol(diff(sol1.lhs, x, 3), Eq(f(x), x*log(x))) == \
        (False, 60*x**4*((log(x) + 1)**2 + log(x))*(
        log(x) + 1)*log(x)**2 - 5*x**4*log(x)**4 - 9)
    assert checkodesol(diff(exp(f(x)) + x, x)*x, Eq(exp(f(x)) + x, 0)) == \
        (True, 0)
    assert checkodesol(diff(exp(f(x)) + x, x)*x, Eq(exp(f(x)) + x, 0),
        solve_for_func=False) == (True, 0)
    assert checkodesol(f(x).diff(x, 2), [Eq(f(x), C1 + C2*x),
        Eq(f(x), C2 + C1*x), Eq(f(x), C1*x + C2*x**2)]) == \
        [(True, 0), (True, 0), (False, C2)]
    assert checkodesol(f(x).diff(x, 2), {Eq(f(x), C1 + C2*x),
        Eq(f(x), C2 + C1*x), Eq(f(x), C1*x + C2*x**2)}) == \
        {(True, 0), (True, 0), (False, C2)}
    assert checkodesol(f(x).diff(x) - 1/f(x)/2, Eq(f(x)**2, x)) == \
        [(True, 0), (True, 0)]
    assert checkodesol(f(x).diff(x) - f(x), Eq(C1*exp(x), f(x))) == (True, 0)
    # Based on test_1st_homogeneous_coeff_ode2_eq3sol.  Make sure that
    # checkodesol tries back substituting f(x) when it can.
    eq3 = x*exp(f(x)/x) + f(x) - x*f(x).diff(x)
    sol3 = Eq(f(x), log(log(C1/x)**(-x)))
    assert not checkodesol(eq3, sol3)[1].has(f(x))
    # This case was failing intermittently depending on hash-seed:
    eqn = Eq(Derivative(x*Derivative(f(x), x), x)/x, exp(x))
    sol = Eq(f(x), C1 + C2*log(x) + exp(x) - Ei(x))
    assert checkodesol(eqn, sol, order=2, solve_for_func=False)[0]
    eq = x**2*(f(x).diff(x, 2)) + x*(f(x).diff(x)) + (2*x**2 +25)*f(x)
    sol = Eq(f(x), C1*besselj(5*I, sqrt(2)*x) + C2*bessely(5*I, sqrt(2)*x))
    assert checkodesol(eq, sol) == (True, 0)

    eqs = [Eq(f(x).diff(x), f(x) + g(x)), Eq(g(x).diff(x), f(x) + g(x))]
    sol = [Eq(f(x), -C1 + C2*exp(2*x)), Eq(g(x), C1 + C2*exp(2*x))]
    assert checkodesol(eqs, sol) == (True, [0, 0])


def test_checksysodesol():
    x, y, z = symbols('x, y, z', cls=Function)
    t = Symbol('t')
    eq = (Eq(diff(x(t),t), 9*y(t)), Eq(diff(y(t),t), 12*x(t)))
    sol = [Eq(x(t), 9*C1*exp(-6*sqrt(3)*t) + 9*C2*exp(6*sqrt(3)*t)), \
    Eq(y(t), -6*sqrt(3)*C1*exp(-6*sqrt(3)*t) + 6*sqrt(3)*C2*exp(6*sqrt(3)*t))]
    assert checksysodesol(eq, sol) == (True, [0, 0])

    eq = (Eq(diff(x(t),t), 2*x(t) + 4*y(t)), Eq(diff(y(t),t), 12*x(t) + 41*y(t)))
    sol = [Eq(x(t), 4*C1*exp(t*(-sqrt(1713)/2 + Rational(43, 2))) + 4*C2*exp(t*(sqrt(1713)/2 + \
    Rational(43, 2)))), Eq(y(t), C1*(-sqrt(1713)/2 + Rational(39, 2))*exp(t*(-sqrt(1713)/2 + \
    Rational(43, 2))) + C2*(Rational(39, 2) + sqrt(1713)/2)*exp(t*(sqrt(1713)/2 + Rational(43, 2))))]
    assert checksysodesol(eq, sol) == (True, [0, 0])

    eq = (Eq(diff(x(t),t), x(t) + y(t)), Eq(diff(y(t),t), -2*x(t) + 2*y(t)))
    sol = [Eq(x(t), (C1*sin(sqrt(7)*t/2) + C2*cos(sqrt(7)*t/2))*exp(t*Rational(3, 2))), \
    Eq(y(t), ((C1/2 - sqrt(7)*C2/2)*sin(sqrt(7)*t/2) + (sqrt(7)*C1/2 + \
    C2/2)*cos(sqrt(7)*t/2))*exp(t*Rational(3, 2)))]
    assert checksysodesol(eq, sol) == (True, [0, 0])

    eq = (Eq(diff(x(t),t), x(t) + y(t) + 9), Eq(diff(y(t),t), 2*x(t) + 5*y(t) + 23))
    sol = [Eq(x(t), C1*exp(t*(-sqrt(6) + 3)) + C2*exp(t*(sqrt(6) + 3)) - \
    Rational(22, 3)), Eq(y(t), C1*(-sqrt(6) + 2)*exp(t*(-sqrt(6) + 3)) + C2*(2 + \
    sqrt(6))*exp(t*(sqrt(6) + 3)) - Rational(5, 3))]
    assert checksysodesol(eq, sol) == (True, [0, 0])

    eq = (Eq(diff(x(t),t), x(t) + y(t) + 81), Eq(diff(y(t),t), -2*x(t) + y(t) + 23))
    sol = [Eq(x(t), (C1*sin(sqrt(2)*t) + C2*cos(sqrt(2)*t))*exp(t) - Rational(58, 3)), \
    Eq(y(t), (sqrt(2)*C1*cos(sqrt(2)*t) - sqrt(2)*C2*sin(sqrt(2)*t))*exp(t) - Rational(185, 3))]
    assert checksysodesol(eq, sol) == (True, [0, 0])

    eq = (Eq(diff(x(t),t), 5*t*x(t) + 2*y(t)), Eq(diff(y(t),t), 2*x(t) + 5*t*y(t)))
    sol = [Eq(x(t), (C1*exp(Integral(2, t).doit()) + C2*exp(-(Integral(2, t)).doit()))*\
    exp((Integral(5*t, t)).doit())), Eq(y(t), (C1*exp((Integral(2, t)).doit()) - \
    C2*exp(-(Integral(2, t)).doit()))*exp((Integral(5*t, t)).doit()))]
    assert checksysodesol(eq, sol) == (True, [0, 0])

    eq = (Eq(diff(x(t),t), 5*t*x(t) + t**2*y(t)), Eq(diff(y(t),t), -t**2*x(t) + 5*t*y(t)))
    sol = [Eq(x(t), (C1*cos((Integral(t**2, t)).doit()) + C2*sin((Integral(t**2, t)).doit()))*\
    exp((Integral(5*t, t)).doit())), Eq(y(t), (-C1*sin((Integral(t**2, t)).doit()) + \
    C2*cos((Integral(t**2, t)).doit()))*exp((Integral(5*t, t)).doit()))]
    assert checksysodesol(eq, sol) == (True, [0, 0])

    eq = (Eq(diff(x(t),t), 5*t*x(t) + t**2*y(t)), Eq(diff(y(t),t), -t**2*x(t) + (5*t+9*t**2)*y(t)))
    sol = [Eq(x(t), (C1*exp((-sqrt(77)/2 + Rational(9, 2))*(Integral(t**2, t)).doit()) + \
    C2*exp((sqrt(77)/2 + Rational(9, 2))*(Integral(t**2, t)).doit()))*exp((Integral(5*t, t)).doit())), \
    Eq(y(t), (C1*(-sqrt(77)/2 + Rational(9, 2))*exp((-sqrt(77)/2 + Rational(9, 2))*(Integral(t**2, t)).doit()) + \
    C2*(sqrt(77)/2 + Rational(9, 2))*exp((sqrt(77)/2 + Rational(9, 2))*(Integral(t**2, t)).doit()))*exp((Integral(5*t, t)).doit()))]
    assert checksysodesol(eq, sol) == (True, [0, 0])

    eq = (Eq(diff(x(t),t,t), 5*x(t) + 43*y(t)), Eq(diff(y(t),t,t), x(t) + 9*y(t)))
    root0 = -sqrt(-sqrt(47) + 7)
    root1 = sqrt(-sqrt(47) + 7)
    root2 = -sqrt(sqrt(47) + 7)
    root3 = sqrt(sqrt(47) + 7)
    sol = [Eq(x(t), 43*C1*exp(t*root0) + 43*C2*exp(t*root1) + 43*C3*exp(t*root2) + 43*C4*exp(t*root3)), \
    Eq(y(t), C1*(root0**2 - 5)*exp(t*root0) + C2*(root1**2 - 5)*exp(t*root1) + \
    C3*(root2**2 - 5)*exp(t*root2) + C4*(root3**2 - 5)*exp(t*root3))]
    assert checksysodesol(eq, sol) == (True, [0, 0])

    eq = (Eq(diff(x(t),t,t), 8*x(t)+3*y(t)+31), Eq(diff(y(t),t,t), 9*x(t)+7*y(t)+12))
    root0 = -sqrt(-sqrt(109)/2 + Rational(15, 2))
    root1 = sqrt(-sqrt(109)/2 + Rational(15, 2))
    root2 = -sqrt(sqrt(109)/2 + Rational(15, 2))
    root3 = sqrt(sqrt(109)/2 + Rational(15, 2))
    sol = [Eq(x(t), 3*C1*exp(t*root0) + 3*C2*exp(t*root1) + 3*C3*exp(t*root2) + 3*C4*exp(t*root3) - Rational(181, 29)), \
    Eq(y(t), C1*(root0**2 - 8)*exp(t*root0) + C2*(root1**2 - 8)*exp(t*root1) + \
    C3*(root2**2 - 8)*exp(t*root2) + C4*(root3**2 - 8)*exp(t*root3) + Rational(183, 29))]
    assert checksysodesol(eq, sol) == (True, [0, 0])

    eq = (Eq(diff(x(t),t,t) - 9*diff(y(t),t) + 7*x(t),0), Eq(diff(y(t),t,t) + 9*diff(x(t),t) + 7*y(t),0))
    sol = [Eq(x(t), C1*cos(t*(Rational(9, 2) + sqrt(109)/2)) + C2*sin(t*(Rational(9, 2) + sqrt(109)/2)) + \
    C3*cos(t*(-sqrt(109)/2 + Rational(9, 2))) + C4*sin(t*(-sqrt(109)/2 + Rational(9, 2)))), Eq(y(t), -C1*sin(t*(Rational(9, 2) + sqrt(109)/2)) \
    + C2*cos(t*(Rational(9, 2) + sqrt(109)/2)) - C3*sin(t*(-sqrt(109)/2 + Rational(9, 2))) + C4*cos(t*(-sqrt(109)/2 + Rational(9, 2))))]
    assert checksysodesol(eq, sol) == (True, [0, 0])

    eq = (Eq(diff(x(t),t,t), 9*t*diff(y(t),t)-9*y(t)), Eq(diff(y(t),t,t),7*t*diff(x(t),t)-7*x(t)))
    I1 = sqrt(6)*7**Rational(1, 4)*sqrt(pi)*erfi(sqrt(6)*7**Rational(1, 4)*t/2)/2 - exp(3*sqrt(7)*t**2/2)/t
    I2 = -sqrt(6)*7**Rational(1, 4)*sqrt(pi)*erf(sqrt(6)*7**Rational(1, 4)*t/2)/2 - exp(-3*sqrt(7)*t**2/2)/t
    sol = [Eq(x(t), C3*t + t*(9*C1*I1 + 9*C2*I2)), Eq(y(t), C4*t + t*(3*sqrt(7)*C1*I1 - 3*sqrt(7)*C2*I2))]
    assert checksysodesol(eq, sol) == (True, [0, 0])

    eq = (Eq(diff(x(t),t), 21*x(t)), Eq(diff(y(t),t), 17*x(t)+3*y(t)), Eq(diff(z(t),t), 5*x(t)+7*y(t)+9*z(t)))
    sol = [Eq(x(t), C1*exp(21*t)), Eq(y(t), 17*C1*exp(21*t)/18 + C2*exp(3*t)), \
    Eq(z(t), 209*C1*exp(21*t)/216 - 7*C2*exp(3*t)/6 + C3*exp(9*t))]
    assert checksysodesol(eq, sol) == (True, [0, 0, 0])

    eq = (Eq(diff(x(t),t),3*y(t)-11*z(t)),Eq(diff(y(t),t),7*z(t)-3*x(t)),Eq(diff(z(t),t),11*x(t)-7*y(t)))
    sol = [Eq(x(t), 7*C0 + sqrt(179)*C1*cos(sqrt(179)*t) + (77*C1/3 + 130*C2/3)*sin(sqrt(179)*t)), \
    Eq(y(t), 11*C0 + sqrt(179)*C2*cos(sqrt(179)*t) + (-58*C1/3 - 77*C2/3)*sin(sqrt(179)*t)), \
    Eq(z(t), 3*C0 + sqrt(179)*(-7*C1/3 - 11*C2/3)*cos(sqrt(179)*t) + (11*C1 - 7*C2)*sin(sqrt(179)*t))]
    assert checksysodesol(eq, sol) == (True, [0, 0, 0])

    eq = (Eq(3*diff(x(t),t),4*5*(y(t)-z(t))),Eq(4*diff(y(t),t),3*5*(z(t)-x(t))),Eq(5*diff(z(t),t),3*4*(x(t)-y(t))))
    sol = [Eq(x(t), C0 + 5*sqrt(2)*C1*cos(5*sqrt(2)*t) + (12*C1/5 + 164*C2/15)*sin(5*sqrt(2)*t)), \
    Eq(y(t), C0 + 5*sqrt(2)*C2*cos(5*sqrt(2)*t) + (-51*C1/10 - 12*C2/5)*sin(5*sqrt(2)*t)), \
    Eq(z(t), C0 + 5*sqrt(2)*(-9*C1/25 - 16*C2/25)*cos(5*sqrt(2)*t) + (12*C1/5 - 12*C2/5)*sin(5*sqrt(2)*t))]
    assert checksysodesol(eq, sol) == (True, [0, 0, 0])

    eq = (Eq(diff(x(t),t),4*x(t) - z(t)),Eq(diff(y(t),t),2*x(t)+2*y(t)-z(t)),Eq(diff(z(t),t),3*x(t)+y(t)))
    sol = [Eq(x(t), C1*exp(2*t) + C2*t*exp(2*t) + C2*exp(2*t) + C3*t**2*exp(2*t)/2 + C3*t*exp(2*t) + C3*exp(2*t)), \
    Eq(y(t), C1*exp(2*t) + C2*t*exp(2*t) + C2*exp(2*t) + C3*t**2*exp(2*t)/2 + C3*t*exp(2*t)), \
    Eq(z(t), 2*C1*exp(2*t) + 2*C2*t*exp(2*t) + C2*exp(2*t) + C3*t**2*exp(2*t) + C3*t*exp(2*t) + C3*exp(2*t))]
    assert checksysodesol(eq, sol) == (True, [0, 0, 0])

    eq = (Eq(diff(x(t),t),4*x(t) - y(t) - 2*z(t)),Eq(diff(y(t),t),2*x(t) + y(t)- 2*z(t)),Eq(diff(z(t),t),5*x(t)-3*z(t)))
    sol = [Eq(x(t), C1*exp(2*t) + C2*(-sin(t) + 3*cos(t)) + C3*(3*sin(t) + cos(t))), \
    Eq(y(t), C2*(-sin(t) + 3*cos(t)) + C3*(3*sin(t) + cos(t))), Eq(z(t), C1*exp(2*t) + 5*C2*cos(t) + 5*C3*sin(t))]
    assert checksysodesol(eq, sol) == (True, [0, 0, 0])

    eq = (Eq(diff(x(t),t),x(t)*y(t)**3), Eq(diff(y(t),t),y(t)**5))
    sol = [Eq(x(t), C1*exp((-1/(4*C2 + 4*t))**(Rational(-1, 4)))), Eq(y(t), -(-1/(4*C2 + 4*t))**Rational(1, 4)), \
    Eq(x(t), C1*exp(-1/(-1/(4*C2 + 4*t))**Rational(1, 4))), Eq(y(t), (-1/(4*C2 + 4*t))**Rational(1, 4)), \
    Eq(x(t), C1*exp(-I/(-1/(4*C2 + 4*t))**Rational(1, 4))), Eq(y(t), -I*(-1/(4*C2 + 4*t))**Rational(1, 4)), \
    Eq(x(t), C1*exp(I/(-1/(4*C2 + 4*t))**Rational(1, 4))), Eq(y(t), I*(-1/(4*C2 + 4*t))**Rational(1, 4))]
    assert checksysodesol(eq, sol) == (True, [0, 0])

    eq = (Eq(diff(x(t),t), exp(3*x(t))*y(t)**3),Eq(diff(y(t),t), y(t)**5))
    sol = [Eq(x(t), -log(C1 - 3/(-1/(4*C2 + 4*t))**Rational(1, 4))/3), Eq(y(t), -(-1/(4*C2 + 4*t))**Rational(1, 4)), \
    Eq(x(t), -log(C1 + 3/(-1/(4*C2 + 4*t))**Rational(1, 4))/3), Eq(y(t), (-1/(4*C2 + 4*t))**Rational(1, 4)), \
    Eq(x(t), -log(C1 + 3*I/(-1/(4*C2 + 4*t))**Rational(1, 4))/3), Eq(y(t), -I*(-1/(4*C2 + 4*t))**Rational(1, 4)), \
    Eq(x(t), -log(C1 - 3*I/(-1/(4*C2 + 4*t))**Rational(1, 4))/3), Eq(y(t), I*(-1/(4*C2 + 4*t))**Rational(1, 4))]
    assert checksysodesol(eq, sol) == (True, [0, 0])

    eq = (Eq(x(t),t*diff(x(t),t)+diff(x(t),t)*diff(y(t),t)), Eq(y(t),t*diff(y(t),t)+diff(y(t),t)**2))
    sol = {Eq(x(t), C1*C2 + C1*t), Eq(y(t), C2**2 + C2*t)}
    assert checksysodesol(eq, sol) == (True, [0, 0])
