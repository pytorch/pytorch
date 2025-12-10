from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import unchanged
from sympy.core.function import (count_ops, diff, expand, expand_multinomial, Function, Derivative)
from sympy.core.mul import Mul, _keep_coeff
from sympy.core import GoldenRatio
from sympy.core.numbers import (E, Float, I, oo, pi, Rational, zoo)
from sympy.core.relational import (Eq, Lt, Gt, Ge, Le)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.complexes import (Abs, sign)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import (cosh, csch, sinh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin, sinc, tan)
from sympy.functions.special.error_functions import erf
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.geometry.polygon import rad
from sympy.integrals.integrals import (Integral, integrate)
from sympy.logic.boolalg import (And, Or)
from sympy.matrices.dense import (Matrix, eye)
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.polys.polytools import (factor, Poly)
from sympy.simplify.simplify import (besselsimp, hypersimp, inversecombine, logcombine, nsimplify, nthroot, posify, separatevars, signsimp, simplify)
from sympy.solvers.solvers import solve

from sympy.testing.pytest import XFAIL, slow, _both_exp_pow
from sympy.abc import x, y, z, t, a, b, c, d, e, f, g, h, i, n


def test_issue_7263():
    assert abs((simplify(30.8**2 - 82.5**2 * sin(rad(11.6))**2)).evalf() - \
            673.447451402970) < 1e-12


def test_factorial_simplify():
    # There are more tests in test_factorials.py.
    x = Symbol('x')
    assert simplify(factorial(x)/x) == gamma(x)
    assert simplify(factorial(factorial(x))) == factorial(factorial(x))


def test_simplify_expr():
    x, y, z, k, n, m, w, s, A = symbols('x,y,z,k,n,m,w,s,A')
    f = Function('f')

    assert all(simplify(tmp) == tmp for tmp in [I, E, oo, x, -x, -oo, -E, -I])

    e = 1/x + 1/y
    assert e != (x + y)/(x*y)
    assert simplify(e) == (x + y)/(x*y)

    e = A**2*s**4/(4*pi*k*m**3)
    assert simplify(e) == e

    e = (4 + 4*x - 2*(2 + 2*x))/(2 + 2*x)
    assert simplify(e) == 0

    e = (-4*x*y**2 - 2*y**3 - 2*x**2*y)/(x + y)**2
    assert simplify(e) == -2*y

    e = -x - y - (x + y)**(-1)*y**2 + (x + y)**(-1)*x**2
    assert simplify(e) == -2*y

    e = (x + x*y)/x
    assert simplify(e) == 1 + y

    e = (f(x) + y*f(x))/f(x)
    assert simplify(e) == 1 + y

    e = (2 * (1/n - cos(n * pi)/n))/pi
    assert simplify(e) == (-cos(pi*n) + 1)/(pi*n)*2

    e = integrate(1/(x**3 + 1), x).diff(x)
    assert simplify(e) == 1/(x**3 + 1)

    e = integrate(x/(x**2 + 3*x + 1), x).diff(x)
    assert simplify(e) == x/(x**2 + 3*x + 1)

    f = Symbol('f')
    A = Matrix([[2*k - m*w**2, -k], [-k, k - m*w**2]]).inv()
    assert simplify((A*Matrix([0, f]))[1] -
            (-f*(2*k - m*w**2)/(k**2 - (k - m*w**2)*(2*k - m*w**2)))) == 0

    f = -x + y/(z + t) + z*x/(z + t) + z*a/(z + t) + t*x/(z + t)
    assert simplify(f) == (y + a*z)/(z + t)

    # issue 10347
    expr = -x*(y**2 - 1)*(2*y**2*(x**2 - 1)/(a*(x**2 - y**2)**2) + (x**2 - 1)
        /(a*(x**2 - y**2)))/(a*(x**2 - y**2)) + x*(-2*x**2*sqrt(-x**2*y**2 + x**2
        + y**2 - 1)*sin(z)/(a*(x**2 - y**2)**2) - x**2*sqrt(-x**2*y**2 + x**2 +
        y**2 - 1)*sin(z)/(a*(x**2 - 1)*(x**2 - y**2)) + (x**2*sqrt((-x**2 + 1)*
        (y**2 - 1))*sqrt(-x**2*y**2 + x**2 + y**2 - 1)*sin(z)/(x**2 - 1) + sqrt(
        (-x**2 + 1)*(y**2 - 1))*(x*(-x*y**2 + x)/sqrt(-x**2*y**2 + x**2 + y**2 -
        1) + sqrt(-x**2*y**2 + x**2 + y**2 - 1))*sin(z))/(a*sqrt((-x**2 + 1)*(
        y**2 - 1))*(x**2 - y**2)))*sqrt(-x**2*y**2 + x**2 + y**2 - 1)*sin(z)/(a*
        (x**2 - y**2)) + x*(-2*x**2*sqrt(-x**2*y**2 + x**2 + y**2 - 1)*cos(z)/(a*
        (x**2 - y**2)**2) - x**2*sqrt(-x**2*y**2 + x**2 + y**2 - 1)*cos(z)/(a*
        (x**2 - 1)*(x**2 - y**2)) + (x**2*sqrt((-x**2 + 1)*(y**2 - 1))*sqrt(-x**2
        *y**2 + x**2 + y**2 - 1)*cos(z)/(x**2 - 1) + x*sqrt((-x**2 + 1)*(y**2 -
        1))*(-x*y**2 + x)*cos(z)/sqrt(-x**2*y**2 + x**2 + y**2 - 1) + sqrt((-x**2
        + 1)*(y**2 - 1))*sqrt(-x**2*y**2 + x**2 + y**2 - 1)*cos(z))/(a*sqrt((-x**2
        + 1)*(y**2 - 1))*(x**2 - y**2)))*sqrt(-x**2*y**2 + x**2 + y**2 - 1)*cos(
        z)/(a*(x**2 - y**2)) - y*sqrt((-x**2 + 1)*(y**2 - 1))*(-x*y*sqrt(-x**2*
        y**2 + x**2 + y**2 - 1)*sin(z)/(a*(x**2 - y**2)*(y**2 - 1)) + 2*x*y*sqrt(
        -x**2*y**2 + x**2 + y**2 - 1)*sin(z)/(a*(x**2 - y**2)**2) + (x*y*sqrt((
        -x**2 + 1)*(y**2 - 1))*sqrt(-x**2*y**2 + x**2 + y**2 - 1)*sin(z)/(y**2 -
        1) + x*sqrt((-x**2 + 1)*(y**2 - 1))*(-x**2*y + y)*sin(z)/sqrt(-x**2*y**2
        + x**2 + y**2 - 1))/(a*sqrt((-x**2 + 1)*(y**2 - 1))*(x**2 - y**2)))*sin(
        z)/(a*(x**2 - y**2)) + y*(x**2 - 1)*(-2*x*y*(x**2 - 1)/(a*(x**2 - y**2)
        **2) + 2*x*y/(a*(x**2 - y**2)))/(a*(x**2 - y**2)) + y*(x**2 - 1)*(y**2 -
        1)*(-x*y*sqrt(-x**2*y**2 + x**2 + y**2 - 1)*cos(z)/(a*(x**2 - y**2)*(y**2
        - 1)) + 2*x*y*sqrt(-x**2*y**2 + x**2 + y**2 - 1)*cos(z)/(a*(x**2 - y**2)
        **2) + (x*y*sqrt((-x**2 + 1)*(y**2 - 1))*sqrt(-x**2*y**2 + x**2 + y**2 -
        1)*cos(z)/(y**2 - 1) + x*sqrt((-x**2 + 1)*(y**2 - 1))*(-x**2*y + y)*cos(
        z)/sqrt(-x**2*y**2 + x**2 + y**2 - 1))/(a*sqrt((-x**2 + 1)*(y**2 - 1)
        )*(x**2 - y**2)))*cos(z)/(a*sqrt((-x**2 + 1)*(y**2 - 1))*(x**2 - y**2)
        ) - x*sqrt((-x**2 + 1)*(y**2 - 1))*sqrt(-x**2*y**2 + x**2 + y**2 - 1)*sin(
        z)**2/(a**2*(x**2 - 1)*(x**2 - y**2)*(y**2 - 1)) - x*sqrt((-x**2 + 1)*(
        y**2 - 1))*sqrt(-x**2*y**2 + x**2 + y**2 - 1)*cos(z)**2/(a**2*(x**2 - 1)*(
        x**2 - y**2)*(y**2 - 1))
    assert simplify(expr) == 2*x/(a**2*(x**2 - y**2))

    #issue 17631
    assert simplify('((-1/2)*Boole(True)*Boole(False)-1)*Boole(True)') == \
            Mul(sympify('(2 + Boole(True)*Boole(False))'), sympify('-Boole(True)/2'))

    A, B = symbols('A,B', commutative=False)

    assert simplify(A*B - B*A) == A*B - B*A
    assert simplify(A/(1 + y/x)) == x*A/(x + y)
    assert simplify(A*(1/x + 1/y)) == A/x + A/y  #(x + y)*A/(x*y)

    assert simplify(log(2) + log(3)) == log(6)
    assert simplify(log(2*x) - log(2)) == log(x)

    assert simplify(hyper([], [], x)) == exp(x)


def test_issue_3557():
    f_1 = x*a + y*b + z*c - 1
    f_2 = x*d + y*e + z*f - 1
    f_3 = x*g + y*h + z*i - 1

    solutions = solve([f_1, f_2, f_3], x, y, z, simplify=False)

    assert simplify(solutions[y]) == \
        (a*i + c*d + f*g - a*f - c*g - d*i)/ \
        (a*e*i + b*f*g + c*d*h - a*f*h - b*d*i - c*e*g)


def test_simplify_other():
    assert simplify(sin(x)**2 + cos(x)**2) == 1
    assert simplify(gamma(x + 1)/gamma(x)) == x
    assert simplify(sin(x)**2 + cos(x)**2 + factorial(x)/gamma(x)) == 1 + x
    assert simplify(
        Eq(sin(x)**2 + cos(x)**2, factorial(x)/gamma(x))) == Eq(x, 1)
    nc = symbols('nc', commutative=False)
    assert simplify(x + x*nc) == x*(1 + nc)
    # issue 6123
    # f = exp(-I*(k*sqrt(t) + x/(2*sqrt(t)))**2)
    # ans = integrate(f, (k, -oo, oo), conds='none')
    ans = I*(-pi*x*exp(I*pi*Rational(-3, 4) + I*x**2/(4*t))*erf(x*exp(I*pi*Rational(-3, 4))/
        (2*sqrt(t)))/(2*sqrt(t)) + pi*x*exp(I*pi*Rational(-3, 4) + I*x**2/(4*t))/
        (2*sqrt(t)))*exp(-I*x**2/(4*t))/(sqrt(pi)*x) - I*sqrt(pi) * \
        (-erf(x*exp(I*pi/4)/(2*sqrt(t))) + 1)*exp(I*pi/4)/(2*sqrt(t))
    assert simplify(ans) == -(-1)**Rational(3, 4)*sqrt(pi)/sqrt(t)
    # issue 6370
    assert simplify(2**(2 + x)/4) == 2**x


@_both_exp_pow
def test_simplify_complex():
    cosAsExp = cos(x)._eval_rewrite_as_exp(x)
    tanAsExp = tan(x)._eval_rewrite_as_exp(x)
    assert simplify(cosAsExp*tanAsExp) == sin(x) # issue 4341

    # issue 10124
    assert simplify(exp(Matrix([[0, -1], [1, 0]]))) == Matrix([[cos(1),
        -sin(1)], [sin(1), cos(1)]])


def test_simplify_ratio():
    # roots of x**3-3*x+5
    roots = ['(1/2 - sqrt(3)*I/2)*(sqrt(21)/2 + 5/2)**(1/3) + 1/((1/2 - '
             'sqrt(3)*I/2)*(sqrt(21)/2 + 5/2)**(1/3))',
             '1/((1/2 + sqrt(3)*I/2)*(sqrt(21)/2 + 5/2)**(1/3)) + '
             '(1/2 + sqrt(3)*I/2)*(sqrt(21)/2 + 5/2)**(1/3)',
             '-(sqrt(21)/2 + 5/2)**(1/3) - 1/(sqrt(21)/2 + 5/2)**(1/3)']

    for r in roots:
        r = S(r)
        assert count_ops(simplify(r, ratio=1)) <= count_ops(r)
        # If ratio=oo, simplify() is always applied:
        assert simplify(r, ratio=oo) is not r


def test_simplify_measure():
    measure1 = lambda expr: len(str(expr))
    measure2 = lambda expr: -count_ops(expr)
                                       # Return the most complicated result
    expr = (x + 1)/(x + sin(x)**2 + cos(x)**2)
    assert measure1(simplify(expr, measure=measure1)) <= measure1(expr)
    assert measure2(simplify(expr, measure=measure2)) <= measure2(expr)

    expr2 = Eq(sin(x)**2 + cos(x)**2, 1)
    assert measure1(simplify(expr2, measure=measure1)) <= measure1(expr2)
    assert measure2(simplify(expr2, measure=measure2)) <= measure2(expr2)


def test_simplify_rational():
    expr = 2**x*2.**y
    assert simplify(expr, rational = True) == 2**(x+y)
    assert simplify(expr, rational = None) == 2.0**(x+y)
    assert simplify(expr, rational = False) == expr
    assert simplify('0.9 - 0.8 - 0.1', rational = True) == 0


def test_simplify_issue_1308():
    assert simplify(exp(Rational(-1, 2)) + exp(Rational(-3, 2))) == \
        (1 + E)*exp(Rational(-3, 2))


def test_issue_5652():
    assert simplify(E + exp(-E)) == exp(-E) + E
    n = symbols('n', commutative=False)
    assert simplify(n + n**(-n)) == n + n**(-n)

def test_issue_27380():
    assert simplify(1.0**(x+1)/1.0**x) == 1.0

def test_simplify_fail1():
    x = Symbol('x')
    y = Symbol('y')
    e = (x + y)**2/(-4*x*y**2 - 2*y**3 - 2*x**2*y)
    assert simplify(e) == 1 / (-2*y)


def test_nthroot():
    assert nthroot(90 + 34*sqrt(7), 3) == sqrt(7) + 3
    q = 1 + sqrt(2) - 2*sqrt(3) + sqrt(6) + sqrt(7)
    assert nthroot(expand_multinomial(q**3), 3) == q
    assert nthroot(41 + 29*sqrt(2), 5) == 1 + sqrt(2)
    assert nthroot(-41 - 29*sqrt(2), 5) == -1 - sqrt(2)
    expr = 1320*sqrt(10) + 4216 + 2576*sqrt(6) + 1640*sqrt(15)
    assert nthroot(expr, 5) == 1 + sqrt(6) + sqrt(15)
    q = 1 + sqrt(2) + sqrt(3) + sqrt(5)
    assert expand_multinomial(nthroot(expand_multinomial(q**5), 5)) == q
    q = 1 + sqrt(2) + 7*sqrt(6) + 2*sqrt(10)
    assert nthroot(expand_multinomial(q**5), 5, 8) == q
    q = 1 + sqrt(2) - 2*sqrt(3) + 1171*sqrt(6)
    assert nthroot(expand_multinomial(q**3), 3) == q
    assert nthroot(expand_multinomial(q**6), 6) == q


def test_nthroot1():
    q = 1 + sqrt(2) + sqrt(3) + S.One/10**20
    p = expand_multinomial(q**5)
    assert nthroot(p, 5) == q
    q = 1 + sqrt(2) + sqrt(3) + S.One/10**30
    p = expand_multinomial(q**5)
    assert nthroot(p, 5) == q


@_both_exp_pow
def test_separatevars():
    x, y, z, n = symbols('x,y,z,n')
    assert separatevars(2*n*x*z + 2*x*y*z) == 2*x*z*(n + y)
    assert separatevars(x*z + x*y*z) == x*z*(1 + y)
    assert separatevars(pi*x*z + pi*x*y*z) == pi*x*z*(1 + y)
    assert separatevars(x*y**2*sin(x) + x*sin(x)*sin(y)) == \
        x*(sin(y) + y**2)*sin(x)
    assert separatevars(x*exp(x + y) + x*exp(x)) == x*(1 + exp(y))*exp(x)
    assert separatevars((x*(y + 1))**z).is_Pow  # != x**z*(1 + y)**z
    assert separatevars(1 + x + y + x*y) == (x + 1)*(y + 1)
    assert separatevars(y/pi*exp(-(z - x)/cos(n))) == \
        y*exp(x/cos(n))*exp(-z/cos(n))/pi
    assert separatevars((x + y)*(x - y) + y**2 + 2*x + 1) == (x + 1)**2
    # issue 4858
    p = Symbol('p', positive=True)
    assert separatevars(sqrt(p**2 + x*p**2)) == p*sqrt(1 + x)
    assert separatevars(sqrt(y*(p**2 + x*p**2))) == p*sqrt(y*(1 + x))
    assert separatevars(sqrt(y*(p**2 + x*p**2)), force=True) == \
        p*sqrt(y)*sqrt(1 + x)
    # issue 4865
    assert separatevars(sqrt(x*y)).is_Pow
    assert separatevars(sqrt(x*y), force=True) == sqrt(x)*sqrt(y)
    # issue 4957
    # any type sequence for symbols is fine
    assert separatevars(((2*x + 2)*y), dict=True, symbols=()) == \
        {'coeff': 1, x: 2*x + 2, y: y}
    # separable
    assert separatevars(((2*x + 2)*y), dict=True, symbols=[x]) == \
        {'coeff': y, x: 2*x + 2}
    assert separatevars(((2*x + 2)*y), dict=True, symbols=[]) == \
        {'coeff': 1, x: 2*x + 2, y: y}
    assert separatevars(((2*x + 2)*y), dict=True) == \
        {'coeff': 1, x: 2*x + 2, y: y}
    assert separatevars(((2*x + 2)*y), dict=True, symbols=None) == \
        {'coeff': y*(2*x + 2)}
    # not separable
    assert separatevars(3, dict=True) is None
    assert separatevars(2*x + y, dict=True, symbols=()) is None
    assert separatevars(2*x + y, dict=True) is None
    assert separatevars(2*x + y, dict=True, symbols=None) == {'coeff': 2*x + y}
    # issue 4808
    n, m = symbols('n,m', commutative=False)
    assert separatevars(m + n*m) == (1 + n)*m
    assert separatevars(x + x*n) == x*(1 + n)
    # issue 4910
    f = Function('f')
    assert separatevars(f(x) + x*f(x)) == f(x) + x*f(x)
    # a noncommutable object present
    eq = x*(1 + hyper((), (), y*z))
    assert separatevars(eq) == eq

    s = separatevars(abs(x*y))
    assert s == abs(x)*abs(y) and s.is_Mul
    z = cos(1)**2 + sin(1)**2 - 1
    a = abs(x*z)
    s = separatevars(a)
    assert not a.is_Mul and s.is_Mul and s == abs(x)*abs(z)
    s = separatevars(abs(x*y*z))
    assert s == abs(x)*abs(y)*abs(z)

    # abs(x+y)/abs(z) would be better but we test this here to
    # see that it doesn't raise
    assert separatevars(abs((x+y)/z)) == abs((x+y)/z)


def test_separatevars_advanced_factor():
    x, y, z = symbols('x,y,z')
    assert separatevars(1 + log(x)*log(y) + log(x) + log(y)) == \
        (log(x) + 1)*(log(y) + 1)
    assert separatevars(1 + x - log(z) - x*log(z) - exp(y)*log(z) -
        x*exp(y)*log(z) + x*exp(y) + exp(y)) == \
        -((x + 1)*(log(z) - 1)*(exp(y) + 1))
    x, y = symbols('x,y', positive=True)
    assert separatevars(1 + log(x**log(y)) + log(x*y)) == \
        (log(x) + 1)*(log(y) + 1)


def test_hypersimp():
    n, k = symbols('n,k', integer=True)

    assert hypersimp(factorial(k), k) == k + 1
    assert hypersimp(factorial(k**2), k) is None

    assert hypersimp(1/factorial(k), k) == 1/(k + 1)

    assert hypersimp(2**k/factorial(k)**2, k) == 2/(k + 1)**2

    assert hypersimp(binomial(n, k), k) == (n - k)/(k + 1)
    assert hypersimp(binomial(n + 1, k), k) == (n - k + 1)/(k + 1)

    term = (4*k + 1)*factorial(k)/factorial(2*k + 1)
    assert hypersimp(term, k) == S.Half*((4*k + 5)/(3 + 14*k + 8*k**2))

    term = 1/((2*k - 1)*factorial(2*k + 1))
    assert hypersimp(term, k) == (k - S.Half)/((k + 1)*(2*k + 1)*(2*k + 3))

    term = binomial(n, k)*(-1)**k/factorial(k)
    assert hypersimp(term, k) == (k - n)/(k + 1)**2


def test_nsimplify():
    x = Symbol("x")
    assert nsimplify(0) == 0
    assert nsimplify(-1) == -1
    assert nsimplify(1) == 1
    assert nsimplify(1 + x) == 1 + x
    assert nsimplify(2.7) == Rational(27, 10)
    assert nsimplify(1 - GoldenRatio) == (1 - sqrt(5))/2
    assert nsimplify((1 + sqrt(5))/4, [GoldenRatio]) == GoldenRatio/2
    assert nsimplify(2/GoldenRatio, [GoldenRatio]) == 2*GoldenRatio - 2
    assert nsimplify(exp(pi*I*Rational(5, 3), evaluate=False)) == \
        sympify('1/2 - sqrt(3)*I/2')
    assert nsimplify(sin(pi*Rational(3, 5), evaluate=False)) == \
        sympify('sqrt(sqrt(5)/8 + 5/8)')
    assert nsimplify(sqrt(atan('1', evaluate=False))*(2 + I), [pi]) == \
        sqrt(pi) + sqrt(pi)/2*I
    assert nsimplify(2 + exp(2*atan('1/4')*I)) == sympify('49/17 + 8*I/17')
    assert nsimplify(pi, tolerance=0.01) == Rational(22, 7)
    assert nsimplify(pi, tolerance=0.001) == Rational(355, 113)
    assert nsimplify(0.33333, tolerance=1e-4) == Rational(1, 3)
    assert nsimplify(2.0**(1/3.), tolerance=0.001) == Rational(635, 504)
    assert nsimplify(2.0**(1/3.), tolerance=0.001, full=True) == \
        2**Rational(1, 3)
    assert nsimplify(x + .5, rational=True) == S.Half + x
    assert nsimplify(1/.3 + x, rational=True) == Rational(10, 3) + x
    assert nsimplify(log(3).n(), rational=True) == \
        sympify('109861228866811/100000000000000')
    assert nsimplify(Float(0.272198261287950), [pi, log(2)]) == pi*log(2)/8
    assert nsimplify(Float(0.272198261287950).n(3), [pi, log(2)]) == \
        -pi/4 - log(2) + Rational(7, 4)
    assert nsimplify(x/7.0) == x/7
    assert nsimplify(pi/1e2) == pi/100
    assert nsimplify(pi/1e2, rational=False) == pi/100.0
    assert nsimplify(pi/1e-7) == 10000000*pi
    assert not nsimplify(
        factor(-3.0*z**2*(z**2)**(-2.5) + 3*(z**2)**(-1.5))).atoms(Float)
    e = x**0.0
    assert e.is_Pow and nsimplify(x**0.0) == 1
    assert nsimplify(3.333333, tolerance=0.1, rational=True) == Rational(10, 3)
    assert nsimplify(3.333333, tolerance=0.01, rational=True) == Rational(10, 3)
    assert nsimplify(3.666666, tolerance=0.1, rational=True) == Rational(11, 3)
    assert nsimplify(3.666666, tolerance=0.01, rational=True) == Rational(11, 3)
    assert nsimplify(33, tolerance=10, rational=True) == Rational(33)
    assert nsimplify(33.33, tolerance=10, rational=True) == Rational(30)
    assert nsimplify(37.76, tolerance=10, rational=True) == Rational(40)
    assert nsimplify(-203.1) == Rational(-2031, 10)
    assert nsimplify(.2, tolerance=0) == Rational(1, 5)
    assert nsimplify(-.2, tolerance=0) == Rational(-1, 5)
    assert nsimplify(.2222, tolerance=0) == Rational(1111, 5000)
    assert nsimplify(-.2222, tolerance=0) == Rational(-1111, 5000)
    # issue 7211, PR 4112
    assert nsimplify(S(2e-8)) == Rational(1, 50000000)
    # issue 7322 direct test
    assert nsimplify(1e-42, rational=True) != 0
    # issue 10336
    inf = Float('inf')
    infs = (-oo, oo, inf, -inf)
    for zi in infs:
        ans = sign(zi)*oo
        assert nsimplify(zi) == ans
        assert nsimplify(zi + x) == x + ans

    assert nsimplify(0.33333333, rational=True, rational_conversion='exact') == Rational(0.33333333)

    # Make sure nsimplify on expressions uses full precision
    assert nsimplify(pi.evalf(100)*x, rational_conversion='exact').evalf(100) == pi.evalf(100)*x


def test_issue_9448():
    tmp = sympify("1/(1 - (-1)**(2/3) - (-1)**(1/3)) + 1/(1 + (-1)**(2/3) + (-1)**(1/3))")
    assert nsimplify(tmp) == S.Half


def test_extract_minus_sign():
    x = Symbol("x")
    y = Symbol("y")
    a = Symbol("a")
    b = Symbol("b")
    assert simplify(-x/-y) == x/y
    assert simplify(-x/y) == -x/y
    assert simplify(x/y) == x/y
    assert simplify(x/-y) == -x/y
    assert simplify(-x/0) == zoo*x
    assert simplify(Rational(-5, 0)) is zoo
    assert simplify(-a*x/(-y - b)) == a*x/(b + y)


def test_diff():
    x = Symbol("x")
    y = Symbol("y")
    f = Function("f")
    g = Function("g")
    assert simplify(g(x).diff(x)*f(x).diff(x) - f(x).diff(x)*g(x).diff(x)) == 0
    assert simplify(2*f(x)*f(x).diff(x) - diff(f(x)**2, x)) == 0
    assert simplify(diff(1/f(x), x) + f(x).diff(x)/f(x)**2) == 0
    assert simplify(f(x).diff(x, y) - f(x).diff(y, x)) == 0


def test_logcombine_1():
    x, y = symbols("x,y")
    a = Symbol("a")
    z, w = symbols("z,w", positive=True)
    b = Symbol("b", real=True)
    assert logcombine(log(x) + 2*log(y)) == log(x) + 2*log(y)
    assert logcombine(log(x) + 2*log(y), force=True) == log(x*y**2)
    assert logcombine(a*log(w) + log(z)) == a*log(w) + log(z)
    assert logcombine(b*log(z) + b*log(x)) == log(z**b) + b*log(x)
    assert logcombine(b*log(z) - log(w)) == log(z**b/w)
    assert logcombine(log(x)*log(z)) == log(x)*log(z)
    assert logcombine(log(w)*log(x)) == log(w)*log(x)
    assert logcombine(cos(-2*log(z) + b*log(w))) in [cos(log(w**b/z**2)),
                                                   cos(log(z**2/w**b))]
    assert logcombine(log(log(x) - log(y)) - log(z), force=True) == \
        log(log(x/y)/z)
    assert logcombine((2 + I)*log(x), force=True) == (2 + I)*log(x)
    assert logcombine((x**2 + log(x) - log(y))/(x*y), force=True) == \
        (x**2 + log(x/y))/(x*y)
    # the following could also give log(z*x**log(y**2)), what we
    # are testing is that a canonical result is obtained
    assert logcombine(log(x)*2*log(y) + log(z), force=True) == \
        log(z*y**log(x**2))
    assert logcombine((x*y + sqrt(x**4 + y**4) + log(x) - log(y))/(pi*x**Rational(2, 3)*
            sqrt(y)**3), force=True) == (
            x*y + sqrt(x**4 + y**4) + log(x/y))/(pi*x**Rational(2, 3)*y**Rational(3, 2))
    assert logcombine(gamma(-log(x/y))*acos(-log(x/y)), force=True) == \
        acos(-log(x/y))*gamma(-log(x/y))

    assert logcombine(2*log(z)*log(w)*log(x) + log(z) + log(w)) == \
        log(z**log(w**2))*log(x) + log(w*z)
    assert logcombine(3*log(w) + 3*log(z)) == log(w**3*z**3)
    assert logcombine(x*(y + 1) + log(2) + log(3)) == x*(y + 1) + log(6)
    assert logcombine((x + y)*log(w) + (-x - y)*log(3)) == (x + y)*log(w/3)
    # a single unknown can combine
    assert logcombine(log(x) + log(2)) == log(2*x)
    eq = log(abs(x)) + log(abs(y))
    assert logcombine(eq) == eq
    reps = {x: 0, y: 0}
    assert log(abs(x)*abs(y)).subs(reps) != eq.subs(reps)


def test_logcombine_complex_coeff():
    i = Integral((sin(x**2) + cos(x**3))/x, x)
    assert logcombine(i, force=True) == i
    assert logcombine(i + 2*log(x), force=True) == \
        i + log(x**2)


def test_issue_5950():
    x, y = symbols("x,y", positive=True)
    assert logcombine(log(3) - log(2)) == log(Rational(3,2), evaluate=False)
    assert logcombine(log(x) - log(y)) == log(x/y)
    assert logcombine(log(Rational(3,2), evaluate=False) - log(2)) == \
        log(Rational(3,4), evaluate=False)


def test_posify():
    x = symbols('x')

    assert str(posify(
        x +
        Symbol('p', positive=True) +
        Symbol('n', negative=True))) == '(_x + n + p, {_x: x})'

    eq, rep = posify(1/x)
    assert log(eq).expand().subs(rep) == -log(x)
    assert str(posify([x, 1 + x])) == '([_x, _x + 1], {_x: x})'

    p = symbols('p', positive=True)
    n = symbols('n', negative=True)
    orig = [x, n, p]
    modified, reps = posify(orig)
    assert str(modified) == '[_x, n, p]'
    assert [w.subs(reps) for w in modified] == orig

    assert str(Integral(posify(1/x + y)[0], (y, 1, 3)).expand()) == \
        'Integral(1/_x, (y, 1, 3)) + Integral(_y, (y, 1, 3))'
    assert str(Sum(posify(1/x**n)[0], (n,1,3)).expand()) == \
        'Sum(_x**(-n), (n, 1, 3))'

    A = Matrix([[1, 2, 3], [4, 5, 6 * Abs(x)]])
    Ap, rep = posify(A)
    assert Ap == A.subs(*reversed(rep.popitem()))

    # issue 16438
    k = Symbol('k', finite=True)
    eq, rep = posify(k)
    assert eq.assumptions0 == {'positive': True, 'zero': False, 'imaginary': False,
     'nonpositive': False, 'commutative': True, 'hermitian': True, 'real': True, 'nonzero': True,
     'nonnegative': True, 'negative': False, 'complex': True, 'finite': True,
     'infinite': False, 'extended_real':True, 'extended_negative': False,
     'extended_nonnegative': True, 'extended_nonpositive': False,
     'extended_nonzero': True, 'extended_positive': True}


def test_issue_4194():
    # simplify should call cancel
    f = Function('f')
    assert simplify((4*x + 6*f(y))/(2*x + 3*f(y))) == 2


@XFAIL
def test_simplify_float_vs_integer():
    # Test for issue 4473:
    # https://github.com/sympy/sympy/issues/4473
    assert simplify(x**2.0 - x**2) == 0
    assert simplify(x**2 - x**2.0) == 0


def test_as_content_primitive():
    assert (x/2 + y).as_content_primitive() == (S.Half, x + 2*y)
    assert (x/2 + y).as_content_primitive(clear=False) == (S.One, x/2 + y)
    assert (y*(x/2 + y)).as_content_primitive() == (S.Half, y*(x + 2*y))
    assert (y*(x/2 + y)).as_content_primitive(clear=False) == (S.One, y*(x/2 + y))

    # although the _as_content_primitive methods do not alter the underlying structure,
    # the as_content_primitive function will touch up the expression and join
    # bases that would otherwise have not been joined.
    assert (x*(2 + 2*x)*(3*x + 3)**2).as_content_primitive() == \
        (18, x*(x + 1)**3)
    assert (2 + 2*x + 2*y*(3 + 3*y)).as_content_primitive() == \
        (2, x + 3*y*(y + 1) + 1)
    assert ((2 + 6*x)**2).as_content_primitive() == \
        (4, (3*x + 1)**2)
    assert ((2 + 6*x)**(2*y)).as_content_primitive() == \
        (1, (_keep_coeff(S(2), (3*x + 1)))**(2*y))
    assert (5 + 10*x + 2*y*(3 + 3*y)).as_content_primitive() == \
        (1, 10*x + 6*y*(y + 1) + 5)
    assert (5*(x*(1 + y)) + 2*x*(3 + 3*y)).as_content_primitive() == \
        (11, x*(y + 1))
    assert ((5*(x*(1 + y)) + 2*x*(3 + 3*y))**2).as_content_primitive() == \
        (121, x**2*(y + 1)**2)
    assert (y**2).as_content_primitive() == \
        (1, y**2)
    assert (S.Infinity).as_content_primitive() == (1, oo)
    eq = x**(2 + y)
    assert (eq).as_content_primitive() == (1, eq)
    assert (S.Half**(2 + x)).as_content_primitive() == (Rational(1, 4), 2**-x)
    assert (Rational(-1, 2)**(2 + x)).as_content_primitive() == \
           (Rational(1, 4), (Rational(-1, 2))**x)
    assert (Rational(-1, 2)**(2 + x)).as_content_primitive() == \
           (Rational(1, 4), Rational(-1, 2)**x)
    assert (4**((1 + y)/2)).as_content_primitive() == (2, 4**(y/2))
    assert (3**((1 + y)/2)).as_content_primitive() == \
           (1, 3**(Mul(S.Half, 1 + y, evaluate=False)))
    assert (5**Rational(3, 4)).as_content_primitive() == (1, 5**Rational(3, 4))
    assert (5**Rational(7, 4)).as_content_primitive() == (5, 5**Rational(3, 4))
    assert Add(z*Rational(5, 7), 0.5*x, y*Rational(3, 2), evaluate=False).as_content_primitive() == \
              (Rational(1, 14), 7.0*x + 21*y + 10*z)
    assert (2**Rational(3, 4) + 2**Rational(1, 4)*sqrt(3)).as_content_primitive(radical=True) == \
           (1, 2**Rational(1, 4)*(sqrt(2) + sqrt(3)))


def test_signsimp():
    e = x*(-x + 1) + x*(x - 1)
    assert signsimp(Eq(e, 0)) is S.true
    assert Abs(x - 1) == Abs(1 - x)
    assert signsimp(y - x) == y - x
    assert signsimp(y - x, evaluate=False) == Mul(-1, x - y, evaluate=False)


def test_besselsimp():
    from sympy.functions.special.bessel import (besseli, besselj, bessely)
    from sympy.integrals.transforms import cosine_transform
    assert besselsimp(exp(-I*pi*y/2)*besseli(y, z*exp_polar(I*pi/2))) == \
        besselj(y, z)
    assert besselsimp(exp(-I*pi*a/2)*besseli(a, 2*sqrt(x)*exp_polar(I*pi/2))) == \
        besselj(a, 2*sqrt(x))
    assert besselsimp(sqrt(2)*sqrt(pi)*x**Rational(1, 4)*exp(I*pi/4)*exp(-I*pi*a/2) *
                      besseli(Rational(-1, 2), sqrt(x)*exp_polar(I*pi/2)) *
                      besseli(a, sqrt(x)*exp_polar(I*pi/2))/2) == \
        besselj(a, sqrt(x)) * cos(sqrt(x))
    assert besselsimp(besseli(Rational(-1, 2), z)) == \
        sqrt(2)*cosh(z)/(sqrt(pi)*sqrt(z))
    assert besselsimp(besseli(a, z*exp_polar(-I*pi/2))) == \
        exp(-I*pi*a/2)*besselj(a, z)
    assert cosine_transform(1/t*sin(a/t), t, y) == \
        sqrt(2)*sqrt(pi)*besselj(0, 2*sqrt(a)*sqrt(y))/2

    assert besselsimp(x**2*(a*(-2*besselj(5*I, x) + besselj(-2 + 5*I, x) +
    besselj(2 + 5*I, x)) + b*(-2*bessely(5*I, x) + bessely(-2 + 5*I, x) +
    bessely(2 + 5*I, x)))/4 + x*(a*(besselj(-1 + 5*I, x)/2 - besselj(1 + 5*I, x)/2)
    + b*(bessely(-1 + 5*I, x)/2 - bessely(1 + 5*I, x)/2)) + (x**2 + 25)*(a*besselj(5*I, x)
    + b*bessely(5*I, x))) == 0

    assert besselsimp(81*x**2*(a*(besselj(Rational(-5, 3), 9*x) - 2*besselj(Rational(1, 3), 9*x) + besselj(Rational(7, 3), 9*x))
    + b*(bessely(Rational(-5, 3), 9*x) - 2*bessely(Rational(1, 3), 9*x) + bessely(Rational(7, 3), 9*x)))/4 + x*(a*(9*besselj(Rational(-2, 3), 9*x)/2
    - 9*besselj(Rational(4, 3), 9*x)/2) + b*(9*bessely(Rational(-2, 3), 9*x)/2 - 9*bessely(Rational(4, 3), 9*x)/2)) +
    (81*x**2 - Rational(1, 9))*(a*besselj(Rational(1, 3), 9*x) + b*bessely(Rational(1, 3), 9*x))) == 0

    assert besselsimp(besselj(a-1,x) + besselj(a+1, x) - 2*a*besselj(a, x)/x) == 0

    assert besselsimp(besselj(a-1,x) + besselj(a+1, x) + besselj(a, x)) == (2*a + x)*besselj(a, x)/x

    assert besselsimp(x**2* besselj(a,x) + x**3*besselj(a+1, x) + besselj(a+2, x)) == \
    2*a*x*besselj(a + 1, x) + x**3*besselj(a + 1, x) - x**2*besselj(a + 2, x) + 2*x*besselj(a + 1, x) + besselj(a + 2, x)

def test_Piecewise():
    e1 = x*(x + y) - y*(x + y)
    e2 = sin(x)**2 + cos(x)**2
    e3 = expand((x + y)*y/x)
    s1 = simplify(e1)
    s2 = simplify(e2)
    s3 = simplify(e3)
    assert simplify(Piecewise((e1, x < e2), (e3, True))) == \
        Piecewise((s1, x < s2), (s3, True))


def test_polymorphism():
    class A(Basic):
        def _eval_simplify(x, **kwargs):
            return S.One

    a = A(S(5), S(2))
    assert simplify(a) == 1


def test_issue_from_PR1599():
    n1, n2, n3, n4 = symbols('n1 n2 n3 n4', negative=True)
    assert simplify(I*sqrt(n1)) == -sqrt(-n1)


def test_issue_6811():
    eq = (x + 2*y)*(2*x + 2)
    assert simplify(eq) == (x + 1)*(x + 2*y)*2
    # reject the 2-arg Mul -- these are a headache for test writing
    assert simplify(eq.expand()) == \
        2*x**2 + 4*x*y + 2*x + 4*y


def test_issue_6920():
    e = [cos(x) + I*sin(x), cos(x) - I*sin(x),
        cosh(x) - sinh(x), cosh(x) + sinh(x)]
    ok = [exp(I*x), exp(-I*x), exp(-x), exp(x)]
    # wrap in f to show that the change happens wherever ei occurs
    f = Function('f')
    assert [simplify(f(ei)).args[0] for ei in e] == ok


def test_issue_7001():
    from sympy.abc import r, R
    assert simplify(-(r*Piecewise((pi*Rational(4, 3), r <= R),
        (-8*pi*R**3/(3*r**3), True)) + 2*Piecewise((pi*r*Rational(4, 3), r <= R),
        (4*pi*R**3/(3*r**2), True)))/(4*pi*r)) == \
        Piecewise((-1, r <= R), (0, True))


def test_inequality_no_auto_simplify():
    # no simplify on creation but can be simplified
    lhs = cos(x)**2 + sin(x)**2
    rhs = 2
    e = Lt(lhs, rhs, evaluate=False)
    assert e is not S.true
    assert simplify(e)


def test_issue_9398():
    from sympy.core.numbers import Number
    from sympy.polys.polytools import cancel
    assert cancel(1e-14) != 0
    assert cancel(1e-14*I) != 0

    assert simplify(1e-14) != 0
    assert simplify(1e-14*I) != 0

    assert (I*Number(1.)*Number(10)**Number(-14)).simplify() != 0

    assert cancel(1e-20) != 0
    assert cancel(1e-20*I) != 0

    assert simplify(1e-20) != 0
    assert simplify(1e-20*I) != 0

    assert cancel(1e-100) != 0
    assert cancel(1e-100*I) != 0

    assert simplify(1e-100) != 0
    assert simplify(1e-100*I) != 0

    f = Float("1e-1000")
    assert cancel(f) != 0
    assert cancel(f*I) != 0

    assert simplify(f) != 0
    assert simplify(f*I) != 0


def test_issue_9324_simplify():
    M = MatrixSymbol('M', 10, 10)
    e = M[0, 0] + M[5, 4] + 1304
    assert simplify(e) == e


def test_issue_9817_simplify():
    # simplify on trace of substituted explicit quadratic form of matrix
    # expressions (a scalar) should return without errors (AttributeError)
    # See issue #9817 and #9190 for the original bug more discussion on this
    from sympy.matrices.expressions import Identity, trace
    v = MatrixSymbol('v', 3, 1)
    A = MatrixSymbol('A', 3, 3)
    x = Matrix([i + 1 for i in range(3)])
    X = Identity(3)
    quadratic = v.T * A * v
    assert simplify((trace(quadratic.as_explicit())).xreplace({v:x, A:X})) == 14


def test_issue_13474():
    x = Symbol('x')
    assert simplify(x + csch(sinc(1))) == x + csch(sinc(1))


@_both_exp_pow
def test_simplify_function_inverse():
    # "inverse" attribute does not guarantee that f(g(x)) is x
    # so this simplification should not happen automatically.
    # See issue #12140
    x, y = symbols('x, y')
    g = Function('g')

    class f(Function):
        def inverse(self, argindex=1):
            return g

    assert simplify(f(g(x))) == f(g(x))
    assert inversecombine(f(g(x))) == x
    assert simplify(f(g(x)), inverse=True) == x
    assert simplify(f(g(sin(x)**2 + cos(x)**2)), inverse=True) == 1
    assert simplify(f(g(x, y)), inverse=True) == f(g(x, y))
    assert unchanged(asin, sin(x))
    assert simplify(asin(sin(x))) == asin(sin(x))
    assert simplify(2*asin(sin(3*x)), inverse=True) == 6*x
    assert simplify(log(exp(x))) == log(exp(x))
    assert simplify(log(exp(x)), inverse=True) == x
    assert simplify(exp(log(x)), inverse=True) == x
    assert simplify(log(exp(x), 2), inverse=True) == x/log(2)
    assert simplify(log(exp(x), 2, evaluate=False), inverse=True) == x/log(2)


def test_clear_coefficients():
    from sympy.simplify.simplify import clear_coefficients
    assert clear_coefficients(4*y*(6*x + 3)) == (y*(2*x + 1), 0)
    assert clear_coefficients(4*y*(6*x + 3) - 2) == (y*(2*x + 1), Rational(1, 6))
    assert clear_coefficients(4*y*(6*x + 3) - 2, x) == (y*(2*x + 1), x/12 + Rational(1, 6))
    assert clear_coefficients(sqrt(2) - 2) == (sqrt(2), 2)
    assert clear_coefficients(4*sqrt(2) - 2) == (sqrt(2), S.Half)
    assert clear_coefficients(S(3), x) == (0, x - 3)
    assert clear_coefficients(S.Infinity, x) == (S.Infinity, x)
    assert clear_coefficients(-S.Pi, x) == (S.Pi, -x)
    assert clear_coefficients(2 - S.Pi/3, x) == (pi, -3*x + 6)

def test_nc_simplify():
    from sympy.simplify.simplify import nc_simplify
    from sympy.matrices.expressions import MatPow, Identity
    from sympy.core import Pow
    from functools import reduce

    a, b, c, d = symbols('a b c d', commutative = False)
    x = Symbol('x')
    A = MatrixSymbol("A", x, x)
    B = MatrixSymbol("B", x, x)
    C = MatrixSymbol("C", x, x)
    D = MatrixSymbol("D", x, x)
    subst = {a: A, b: B, c: C, d:D}
    funcs = {Add: lambda x,y: x+y, Mul: lambda x,y: x*y }

    def _to_matrix(expr):
        if expr in subst:
            return subst[expr]
        if isinstance(expr, Pow):
            return MatPow(_to_matrix(expr.args[0]), expr.args[1])
        elif isinstance(expr, (Add, Mul)):
            return reduce(funcs[expr.func],[_to_matrix(a) for a in expr.args])
        else:
            return expr*Identity(x)

    def _check(expr, simplified, deep=True, matrix=True):
        assert nc_simplify(expr, deep=deep) == simplified
        assert expand(expr) == expand(simplified)
        if matrix:
            m_simp = _to_matrix(simplified).doit(inv_expand=False)
            assert nc_simplify(_to_matrix(expr), deep=deep) == m_simp

    _check(a*b*a*b*a*b*c*(a*b)**3*c, ((a*b)**3*c)**2)
    _check(a*b*(a*b)**-2*a*b, 1)
    _check(a**2*b*a*b*a*b*(a*b)**-1, a*(a*b)**2, matrix=False)
    _check(b*a*b**2*a*b**2*a*b**2, b*(a*b**2)**3)
    _check(a*b*a**2*b*a**2*b*a**3, (a*b*a)**3*a**2)
    _check(a**2*b*a**4*b*a**4*b*a**2, (a**2*b*a**2)**3)
    _check(a**3*b*a**4*b*a**4*b*a, a**3*(b*a**4)**3*a**-3)
    _check(a*b*a*b + a*b*c*x*a*b*c, (a*b)**2 + x*(a*b*c)**2)
    _check(a*b*a*b*c*a*b*a*b*c, ((a*b)**2*c)**2)
    _check(b**-1*a**-1*(a*b)**2, a*b)
    _check(a**-1*b*c**-1, (c*b**-1*a)**-1)
    expr = a**3*b*a**4*b*a**4*b*a**2*b*a**2*(b*a**2)**2*b*a**2*b*a**2
    for _ in range(10):
        expr *= a*b
    _check(expr, a**3*(b*a**4)**2*(b*a**2)**6*(a*b)**10)
    _check((a*b*a*b)**2, (a*b*a*b)**2, deep=False)
    _check(a*b*(c*d)**2, a*b*(c*d)**2)
    expr = b**-1*(a**-1*b**-1 - a**-1*c*b**-1)**-1*a**-1
    assert nc_simplify(expr) == (1-c)**-1
    # commutative expressions should be returned without an error
    assert nc_simplify(2*x**2) == 2*x**2

def test_issue_15965():
    A = Sum(z*x**y, (x, 1, a))
    anew = z*Sum(x**y, (x, 1, a))
    B = Integral(x*y, x)
    bdo = x**2*y/2
    assert simplify(A + B) == anew + bdo
    assert simplify(A) == anew
    assert simplify(B) == bdo
    assert simplify(B, doit=False) == y*Integral(x, x)


def test_issue_17137():
    assert simplify(cos(x)**I) == cos(x)**I
    assert simplify(cos(x)**(2 + 3*I)) == cos(x)**(2 + 3*I)


def test_issue_21869():
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    expr = And(Eq(x**2, 4), Le(x, y))
    assert expr.simplify() == expr

    expr = And(Eq(x**2, 4), Eq(x, 2))
    assert expr.simplify() == Eq(x, 2)

    expr = And(Eq(x**3, x**2), Eq(x, 1))
    assert expr.simplify() == Eq(x, 1)

    expr = And(Eq(sin(x), x**2), Eq(x, 0))
    assert expr.simplify() == Eq(x, 0)

    expr = And(Eq(x**3, x**2), Eq(x, 2))
    assert expr.simplify() == S.false

    expr = And(Eq(y, x**2), Eq(x, 1))
    assert expr.simplify() == And(Eq(y,1), Eq(x, 1))

    expr = And(Eq(y**2, 1), Eq(y, x**2), Eq(x, 1))
    assert expr.simplify() == And(Eq(y,1), Eq(x, 1))

    expr = And(Eq(y**2, 4), Eq(y, 2*x**2), Eq(x, 1))
    assert expr.simplify() == And(Eq(y,2), Eq(x, 1))

    expr = And(Eq(y**2, 4), Eq(y, x**2), Eq(x, 1))
    assert expr.simplify() == S.false


def test_issue_7971_21740():
    z = Integral(x, (x, 1, 1))
    assert z != 0
    assert simplify(z) is S.Zero
    assert simplify(S.Zero) is S.Zero
    z = simplify(Float(0))
    assert z is not S.Zero and z == 0.0


@slow
def test_issue_17141_slow():
    # Should not give RecursionError
    assert simplify((2**acos(I+1)**2).rewrite('log')) == 2**((pi + 2*I*log(-1 +
                   sqrt(1 - 2*I) + I))**2/4)


def test_issue_17141():
    # Check that there is no RecursionError
    assert simplify(x**(1 / acos(I))) == x**(2/(pi - 2*I*log(1 + sqrt(2))))
    assert simplify(acos(-I)**2*acos(I)**2) == \
           log(1 + sqrt(2))**4 + pi**2*log(1 + sqrt(2))**2/2 + pi**4/16
    assert simplify(2**acos(I)**2) == 2**((pi - 2*I*log(1 + sqrt(2)))**2/4)
    p = 2**acos(I+1)**2
    assert simplify(p) == p


def test_simplify_kroneckerdelta():
    i, j = symbols("i j")
    K = KroneckerDelta

    assert simplify(K(i, j)) == K(i, j)
    assert simplify(K(0, j)) == K(0, j)
    assert simplify(K(i, 0)) == K(i, 0)

    assert simplify(K(0, j).rewrite(Piecewise) * K(1, j)) == 0
    assert simplify(K(1, i) + Piecewise((1, Eq(j, 2)), (0, True))) == K(1, i) + K(2, j)

    # issue 17214
    assert simplify(K(0, j) * K(1, j)) == 0

    n = Symbol('n', integer=True)
    assert simplify(K(0, n) * K(1, n)) == 0

    M = Matrix(4, 4, lambda i, j: K(j - i, n) if i <= j else 0)
    assert simplify(M**2) == Matrix([[K(0, n), 0, K(1, n), 0],
                                     [0, K(0, n), 0, K(1, n)],
                                     [0, 0, K(0, n), 0],
                                     [0, 0, 0, K(0, n)]])
    assert simplify(eye(1) * KroneckerDelta(0, n) *
                    KroneckerDelta(1, n)) == Matrix([[0]])

    assert simplify(S.Infinity * KroneckerDelta(0, n) *
                    KroneckerDelta(1, n)) is S.NaN


def test_issue_17292():
    assert simplify(abs(x)/abs(x**2)) == 1/abs(x)
    # this is bigger than the issue: check that deep processing works
    assert simplify(5*abs((x**2 - 1)/(x - 1))) == 5*Abs(x + 1)


def test_issue_19822():
    expr = And(Gt(n-2, 1), Gt(n, 1))
    assert simplify(expr) == Gt(n, 3)


def test_issue_18645():
    expr = And(Ge(x, 3), Le(x, 3))
    assert simplify(expr) == Eq(x, 3)
    expr = And(Eq(x, 3), Le(x, 3))
    assert simplify(expr) == Eq(x, 3)


@XFAIL
def test_issue_18642():
    i = Symbol("i", integer=True)
    n = Symbol("n", integer=True)
    expr = And(Eq(i, 2 * n), Le(i, 2*n -1))
    assert simplify(expr) == S.false


@XFAIL
def test_issue_18389():
    n = Symbol("n", integer=True)
    expr = Eq(n, 0) | (n >= 1)
    assert simplify(expr) == Ge(n, 0)


def test_issue_8373():
    x = Symbol('x', real=True)
    assert simplify(Or(x < 1, x >= 1)) == S.true


def test_issue_7950():
    expr = And(Eq(x, 1), Eq(x, 2))
    assert simplify(expr) == S.false


def test_issue_22020():
    expr = I*pi/2 -oo
    assert simplify(expr) == expr
    # Used to throw an error


def test_issue_19484():
    assert simplify(sign(x) * Abs(x)) == x

    e = x + sign(x + x**3)
    assert simplify(Abs(x + x**3)*e) == x**3 + x*Abs(x**3 + x) + x

    e = x**2 + sign(x**3 + 1)
    assert simplify(Abs(x**3 + 1) * e) == x**3 + x**2*Abs(x**3 + 1) + 1

    f = Function('f')
    e = x + sign(x + f(x)**3)
    assert simplify(Abs(x + f(x)**3) * e) == x*Abs(x + f(x)**3) + x + f(x)**3


def test_issue_23543():
    # Used to give an error
    x, y, z = symbols("x y z", commutative=False)
    assert (x*(y + z/2)).simplify() == x*(2*y + z)/2


def test_issue_11004():

    def f(n):
        return sqrt(2*pi*n) * (n/E)**n

    def m(n, k):
        return  f(n) / (f(n/k)**k)

    def p(n,k):
        return m(n, k) / (k**n)

    N, k = symbols('N k')
    half = Float('0.5', 4)
    z = log(p(n, k) / p(n, k + 1)).expand(force=True)
    r = simplify(z.subs(n, N).n(4))
    assert r == (
        half*k*log(k)
        - half*k*log(k + 1)
        + half*log(N)
        - half*log(k + 1)
        + Float(0.9189224, 4)
    )


def test_issue_19161():
    polynomial = Poly('x**2').simplify()
    assert (polynomial-x**2).simplify() == 0


def test_issue_22210():
    d = Symbol('d', integer=True)
    expr = 2*Derivative(sin(x), (x, d))
    assert expr.simplify() == expr


def test_reduce_inverses_nc_pow():
    x, y = symbols("x y", commutative=True)
    Z = symbols("Z", commutative=False)
    assert simplify(2**Z * y**Z) == 2**Z * y**Z
    assert simplify(x**Z * y**Z) == x**Z * y**Z
    x, y = symbols("x y", positive=True)
    assert expand((x*y)**Z) == x**Z * y**Z
    assert simplify(x**Z * y**Z) == expand((x*y)**Z)

def test_nc_recursion_coeff():
    X = symbols("X", commutative = False)
    assert (2 * cos(pi/3) * X).simplify() == X
    assert (2.0 * cos(pi/3) * X).simplify() == X
