from sympy.core.function import (Derivative as D, Function)
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.core import S
from sympy.solvers.pde import (pde_separate, pde_separate_add, pde_separate_mul,
    pdsolve, classify_pde, checkpdesol)
from sympy.testing.pytest import raises


a, b, c, x, y = symbols('a b c x y')

def test_pde_separate_add():
    x, y, z, t = symbols("x,y,z,t")
    F, T, X, Y, Z, u = map(Function, 'FTXYZu')

    eq = Eq(D(u(x, t), x), D(u(x, t), t)*exp(u(x, t)))
    res = pde_separate_add(eq, u(x, t), [X(x), T(t)])
    assert res == [D(X(x), x)*exp(-X(x)), D(T(t), t)*exp(T(t))]


def test_pde_separate():
    x, y, z, t = symbols("x,y,z,t")
    F, T, X, Y, Z, u = map(Function, 'FTXYZu')

    eq = Eq(D(u(x, t), x), D(u(x, t), t)*exp(u(x, t)))
    raises(ValueError, lambda: pde_separate(eq, u(x, t), [X(x), T(t)], 'div'))


def test_pde_separate_mul():
    x, y, z, t = symbols("x,y,z,t")
    c = Symbol("C", real=True)
    Phi = Function('Phi')
    F, R, T, X, Y, Z, u = map(Function, 'FRTXYZu')
    r, theta, z = symbols('r,theta,z')

    # Something simple :)
    eq = Eq(D(F(x, y, z), x) + D(F(x, y, z), y) + D(F(x, y, z), z), 0)

    # Duplicate arguments in functions
    raises(
        ValueError, lambda: pde_separate_mul(eq, F(x, y, z), [X(x), u(z, z)]))
    # Wrong number of arguments
    raises(ValueError, lambda: pde_separate_mul(eq, F(x, y, z), [X(x), Y(y)]))
    # Wrong variables: [x, y] -> [x, z]
    raises(
        ValueError, lambda: pde_separate_mul(eq, F(x, y, z), [X(t), Y(x, y)]))

    assert pde_separate_mul(eq, F(x, y, z), [Y(y), u(x, z)]) == \
        [D(Y(y), y)/Y(y), -D(u(x, z), x)/u(x, z) - D(u(x, z), z)/u(x, z)]
    assert pde_separate_mul(eq, F(x, y, z), [X(x), Y(y), Z(z)]) == \
        [D(X(x), x)/X(x), -D(Z(z), z)/Z(z) - D(Y(y), y)/Y(y)]

    # wave equation
    wave = Eq(D(u(x, t), t, t), c**2*D(u(x, t), x, x))
    res = pde_separate_mul(wave, u(x, t), [X(x), T(t)])
    assert res == [D(X(x), x, x)/X(x), D(T(t), t, t)/(c**2*T(t))]

    # Laplace equation in cylindrical coords
    eq = Eq(1/r * D(Phi(r, theta, z), r) + D(Phi(r, theta, z), r, 2) +
            1/r**2 * D(Phi(r, theta, z), theta, 2) + D(Phi(r, theta, z), z, 2), 0)
    # Separate z
    res = pde_separate_mul(eq, Phi(r, theta, z), [Z(z), u(theta, r)])
    assert res == [D(Z(z), z, z)/Z(z),
            -D(u(theta, r), r, r)/u(theta, r) -
        D(u(theta, r), r)/(r*u(theta, r)) -
        D(u(theta, r), theta, theta)/(r**2*u(theta, r))]
    # Lets use the result to create a new equation...
    eq = Eq(res[1], c)
    # ...and separate theta...
    res = pde_separate_mul(eq, u(theta, r), [T(theta), R(r)])
    assert res == [D(T(theta), theta, theta)/T(theta),
            -r*D(R(r), r)/R(r) - r**2*D(R(r), r, r)/R(r) - c*r**2]
    # ...or r...
    res = pde_separate_mul(eq, u(theta, r), [R(r), T(theta)])
    assert res == [r*D(R(r), r)/R(r) + r**2*D(R(r), r, r)/R(r) + c*r**2,
            -D(T(theta), theta, theta)/T(theta)]


def test_issue_11726():
    x, t = symbols("x t")
    f  = symbols("f", cls=Function)
    X, T = symbols("X T", cls=Function)

    u = f(x, t)
    eq = u.diff(x, 2) - u.diff(t, 2)
    res = pde_separate(eq, u, [T(x), X(t)])
    assert res == [D(T(x), x, x)/T(x),D(X(t), t, t)/X(t)]


def test_pde_classify():
    # When more number of hints are added, add tests for classifying here.
    f = Function('f')
    eq1 = a*f(x,y) + b*f(x,y).diff(x) + c*f(x,y).diff(y)
    eq2 = 3*f(x,y) + 2*f(x,y).diff(x) + f(x,y).diff(y)
    eq3 = a*f(x,y) + b*f(x,y).diff(x) + 2*f(x,y).diff(y)
    eq4 = x*f(x,y) + f(x,y).diff(x) + 3*f(x,y).diff(y)
    eq5 = x**2*f(x,y) + x*f(x,y).diff(x) + x*y*f(x,y).diff(y)
    eq6 = y*x**2*f(x,y) + y*f(x,y).diff(x) + f(x,y).diff(y)
    for eq in [eq1, eq2, eq3]:
        assert classify_pde(eq) == ('1st_linear_constant_coeff_homogeneous',)
    for eq in [eq4, eq5, eq6]:
        assert classify_pde(eq) == ('1st_linear_variable_coeff',)


def test_checkpdesol():
    f, F = map(Function, ['f', 'F'])
    eq1 = a*f(x,y) + b*f(x,y).diff(x) + c*f(x,y).diff(y)
    eq2 = 3*f(x,y) + 2*f(x,y).diff(x) + f(x,y).diff(y)
    eq3 = a*f(x,y) + b*f(x,y).diff(x) + 2*f(x,y).diff(y)
    for eq in [eq1, eq2, eq3]:
        assert checkpdesol(eq, pdsolve(eq))[0]
    eq4 = x*f(x,y) + f(x,y).diff(x) + 3*f(x,y).diff(y)
    eq5 = 2*f(x,y) + 1*f(x,y).diff(x) + 3*f(x,y).diff(y)
    eq6 = f(x,y) + 1*f(x,y).diff(x) + 3*f(x,y).diff(y)
    assert checkpdesol(eq4, [pdsolve(eq5), pdsolve(eq6)]) == [
        (False, (x - 2)*F(3*x - y)*exp(-x/S(5) - 3*y/S(5))),
         (False, (x - 1)*F(3*x - y)*exp(-x/S(10) - 3*y/S(10)))]
    for eq in [eq4, eq5, eq6]:
        assert checkpdesol(eq, pdsolve(eq))[0]
    sol = pdsolve(eq4)
    sol4 = Eq(sol.lhs - sol.rhs, 0)
    raises(NotImplementedError, lambda:
        checkpdesol(eq4, sol4, solve_for_func=False))


def test_solvefun():
    f, F, G, H = map(Function, ['f', 'F', 'G', 'H'])
    eq1 = f(x,y) + f(x,y).diff(x) + f(x,y).diff(y)
    assert pdsolve(eq1) == Eq(f(x, y), F(x - y)*exp(-x/2 - y/2))
    assert pdsolve(eq1, solvefun=G) == Eq(f(x, y), G(x - y)*exp(-x/2 - y/2))
    assert pdsolve(eq1, solvefun=H) == Eq(f(x, y), H(x - y)*exp(-x/2 - y/2))


def test_pde_1st_linear_constant_coeff_homogeneous():
    f, F = map(Function, ['f', 'F'])
    u = f(x, y)
    eq = 2*u + u.diff(x) + u.diff(y)
    assert classify_pde(eq) == ('1st_linear_constant_coeff_homogeneous',)
    sol = pdsolve(eq)
    assert sol == Eq(u, F(x - y)*exp(-x - y))
    assert checkpdesol(eq, sol)[0]

    eq = 4 + (3*u.diff(x)/u) + (2*u.diff(y)/u)
    assert classify_pde(eq) == ('1st_linear_constant_coeff_homogeneous',)
    sol = pdsolve(eq)
    assert sol == Eq(u, F(2*x - 3*y)*exp(-S(12)*x/13 - S(8)*y/13))
    assert checkpdesol(eq, sol)[0]

    eq = u + (6*u.diff(x)) + (7*u.diff(y))
    assert classify_pde(eq) == ('1st_linear_constant_coeff_homogeneous',)
    sol = pdsolve(eq)
    assert sol == Eq(u, F(7*x - 6*y)*exp(-6*x/S(85) - 7*y/S(85)))
    assert checkpdesol(eq, sol)[0]

    eq = a*u + b*u.diff(x) + c*u.diff(y)
    sol = pdsolve(eq)
    assert checkpdesol(eq, sol)[0]


def test_pde_1st_linear_constant_coeff():
    f, F = map(Function, ['f', 'F'])
    u = f(x,y)
    eq = -2*u.diff(x) + 4*u.diff(y) + 5*u - exp(x + 3*y)
    sol = pdsolve(eq)
    assert sol == Eq(f(x,y),
    (F(4*x + 2*y)*exp(x/2) + exp(x + 4*y)/15)*exp(-y))
    assert classify_pde(eq) == ('1st_linear_constant_coeff',
    '1st_linear_constant_coeff_Integral')
    assert checkpdesol(eq, sol)[0]

    eq = (u.diff(x)/u) + (u.diff(y)/u) + 1 - (exp(x + y)/u)
    sol = pdsolve(eq)
    assert sol == Eq(f(x, y), F(x - y)*exp(-x/2 - y/2) + exp(x + y)/3)
    assert classify_pde(eq) == ('1st_linear_constant_coeff',
    '1st_linear_constant_coeff_Integral')
    assert checkpdesol(eq, sol)[0]

    eq = 2*u + -u.diff(x) + 3*u.diff(y) + sin(x)
    sol = pdsolve(eq)
    assert sol == Eq(f(x, y),
         F(3*x + y)*exp(x/5 - 3*y/5) - 2*sin(x)/5 - cos(x)/5)
    assert classify_pde(eq) == ('1st_linear_constant_coeff',
    '1st_linear_constant_coeff_Integral')
    assert checkpdesol(eq, sol)[0]

    eq = u + u.diff(x) + u.diff(y) + x*y
    sol = pdsolve(eq)
    assert sol.expand() == Eq(f(x, y),
        x + y + (x - y)**2/4 - (x + y)**2/4 + F(x - y)*exp(-x/2 - y/2) - 2).expand()
    assert classify_pde(eq) == ('1st_linear_constant_coeff',
    '1st_linear_constant_coeff_Integral')
    assert checkpdesol(eq, sol)[0]
    eq = u + u.diff(x) + u.diff(y) + log(x)
    assert classify_pde(eq) == ('1st_linear_constant_coeff',
    '1st_linear_constant_coeff_Integral')


def test_pdsolve_all():
    f, F = map(Function, ['f', 'F'])
    u = f(x,y)
    eq = u + u.diff(x) + u.diff(y) + x**2*y
    sol = pdsolve(eq, hint = 'all')
    keys = ['1st_linear_constant_coeff',
        '1st_linear_constant_coeff_Integral', 'default', 'order']
    assert sorted(sol.keys()) == keys
    assert sol['order'] == 1
    assert sol['default'] == '1st_linear_constant_coeff'
    assert sol['1st_linear_constant_coeff'].expand() == Eq(f(x, y),
        -x**2*y + x**2 + 2*x*y - 4*x - 2*y + F(x - y)*exp(-x/2 - y/2) + 6).expand()


def test_pdsolve_variable_coeff():
    f, F = map(Function, ['f', 'F'])
    u = f(x, y)
    eq = x*(u.diff(x)) - y*(u.diff(y)) + y**2*u - y**2
    sol = pdsolve(eq, hint="1st_linear_variable_coeff")
    assert sol == Eq(u, F(x*y)*exp(y**2/2) + 1)
    assert checkpdesol(eq, sol)[0]

    eq = x**2*u + x*u.diff(x) + x*y*u.diff(y)
    sol = pdsolve(eq, hint='1st_linear_variable_coeff')
    assert sol == Eq(u, F(y*exp(-x))*exp(-x**2/2))
    assert checkpdesol(eq, sol)[0]

    eq = y*x**2*u + y*u.diff(x) + u.diff(y)
    sol = pdsolve(eq, hint='1st_linear_variable_coeff')
    assert sol == Eq(u, F(-2*x + y**2)*exp(-x**3/3))
    assert checkpdesol(eq, sol)[0]

    eq = exp(x)**2*(u.diff(x)) + y
    sol = pdsolve(eq, hint='1st_linear_variable_coeff')
    assert sol == Eq(u, y*exp(-2*x)/2 + F(y))
    assert checkpdesol(eq, sol)[0]

    eq = exp(2*x)*(u.diff(y)) + y*u - u
    sol = pdsolve(eq, hint='1st_linear_variable_coeff')
    assert sol == Eq(u, F(x)*exp(-y*(y - 2)*exp(-2*x)/2))
