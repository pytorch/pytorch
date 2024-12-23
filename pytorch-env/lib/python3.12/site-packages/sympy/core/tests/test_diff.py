from sympy.concrete.summations import Sum
from sympy.core.expr import Expr
from sympy.core.function import (Derivative, Function, diff, Subs)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, cot, sin, tan)
from sympy.tensor.array.ndim_array import NDimArray
from sympy.testing.pytest import raises
from sympy.abc import a, b, c, x, y, z

def test_diff():
    assert Rational(1, 3).diff(x) is S.Zero
    assert I.diff(x) is S.Zero
    assert pi.diff(x) is S.Zero
    assert x.diff(x, 0) == x
    assert (x**2).diff(x, 2, x) == 0
    assert (x**2).diff((x, 2), x) == 0
    assert (x**2).diff((x, 1), x) == 2
    assert (x**2).diff((x, 1), (x, 1)) == 2
    assert (x**2).diff((x, 2)) == 2
    assert (x**2).diff(x, y, 0) == 2*x
    assert (x**2).diff(x, (y, 0)) == 2*x
    assert (x**2).diff(x, y) == 0
    raises(ValueError, lambda: x.diff(1, x))

    p = Rational(5)
    e = a*b + b**p
    assert e.diff(a) == b
    assert e.diff(b) == a + 5*b**4
    assert e.diff(b).diff(a) == Rational(1)
    e = a*(b + c)
    assert e.diff(a) == b + c
    assert e.diff(b) == a
    assert e.diff(b).diff(a) == Rational(1)
    e = c**p
    assert e.diff(c, 6) == Rational(0)
    assert e.diff(c, 5) == Rational(120)
    e = c**Rational(2)
    assert e.diff(c) == 2*c
    e = a*b*c
    assert e.diff(c) == a*b


def test_diff2():
    n3 = Rational(3)
    n2 = Rational(2)
    n6 = Rational(6)

    e = n3*(-n2 + x**n2)*cos(x) + x*(-n6 + x**n2)*sin(x)
    assert e == 3*(-2 + x**2)*cos(x) + x*(-6 + x**2)*sin(x)
    assert e.diff(x).expand() == x**3*cos(x)

    e = (x + 1)**3
    assert e.diff(x) == 3*(x + 1)**2
    e = x*(x + 1)**3
    assert e.diff(x) == (x + 1)**3 + 3*x*(x + 1)**2
    e = 2*exp(x*x)*x
    assert e.diff(x) == 2*exp(x**2) + 4*x**2*exp(x**2)


def test_diff3():
    p = Rational(5)
    e = a*b + sin(b**p)
    assert e == a*b + sin(b**5)
    assert e.diff(a) == b
    assert e.diff(b) == a + 5*b**4*cos(b**5)
    e = tan(c)
    assert e == tan(c)
    assert e.diff(c) in [cos(c)**(-2), 1 + sin(c)**2/cos(c)**2, 1 + tan(c)**2]
    e = c*log(c) - c
    assert e == -c + c*log(c)
    assert e.diff(c) == log(c)
    e = log(sin(c))
    assert e == log(sin(c))
    assert e.diff(c) in [sin(c)**(-1)*cos(c), cot(c)]
    e = (Rational(2)**a/log(Rational(2)))
    assert e == 2**a*log(Rational(2))**(-1)
    assert e.diff(a) == 2**a


def test_diff_no_eval_derivative():
    class My(Expr):
        def __new__(cls, x):
            return Expr.__new__(cls, x)

    # My doesn't have its own _eval_derivative method
    assert My(x).diff(x).func is Derivative
    assert My(x).diff(x, 3).func is Derivative
    assert re(x).diff(x, 2) == Derivative(re(x), (x, 2))  # issue 15518
    assert diff(NDimArray([re(x), im(x)]), (x, 2)) == NDimArray(
        [Derivative(re(x), (x, 2)), Derivative(im(x), (x, 2))])
    # it doesn't have y so it shouldn't need a method for this case
    assert My(x).diff(y) == 0


def test_speed():
    # this should return in 0.0s. If it takes forever, it's wrong.
    assert x.diff(x, 10**8) == 0


def test_deriv_noncommutative():
    A = Symbol("A", commutative=False)
    f = Function("f")
    assert A*f(x)*A == f(x)*A**2
    assert A*f(x).diff(x)*A == f(x).diff(x) * A**2


def test_diff_nth_derivative():
    f =  Function("f")
    n = Symbol("n", integer=True)

    expr = diff(sin(x), (x, n))
    expr2 = diff(f(x), (x, 2))
    expr3 = diff(f(x), (x, n))

    assert expr.subs(sin(x), cos(-x)) == Derivative(cos(-x), (x, n))
    assert expr.subs(n, 1).doit() == cos(x)
    assert expr.subs(n, 2).doit() == -sin(x)

    assert expr2.subs(Derivative(f(x), x), y) == Derivative(y, x)
    # Currently not supported (cannot determine if `n > 1`):
    #assert expr3.subs(Derivative(f(x), x), y) == Derivative(y, (x, n-1))
    assert expr3 == Derivative(f(x), (x, n))

    assert diff(x, (x, n)) == Piecewise((x, Eq(n, 0)), (1, Eq(n, 1)), (0, True))
    assert diff(2*x, (x, n)).dummy_eq(
        Sum(Piecewise((2*x*factorial(n)/(factorial(y)*factorial(-y + n)),
        Eq(y, 0) & Eq(Max(0, -y + n), 0)),
        (2*factorial(n)/(factorial(y)*factorial(-y + n)), Eq(y, 0) & Eq(Max(0,
        -y + n), 1)), (0, True)), (y, 0, n)))
    # TODO: assert diff(x**2, (x, n)) == x**(2-n)*ff(2, n)
    exprm = x*sin(x)
    mul_diff = diff(exprm, (x, n))
    assert isinstance(mul_diff, Sum)
    for i in range(5):
        assert mul_diff.subs(n, i).doit() == exprm.diff((x, i)).expand()

    exprm2 = 2*y*x*sin(x)*cos(x)*log(x)*exp(x)
    dex = exprm2.diff((x, n))
    assert isinstance(dex, Sum)
    for i in range(7):
        assert dex.subs(n, i).doit().expand() == \
        exprm2.diff((x, i)).expand()

    assert (cos(x)*sin(y)).diff([[x, y, z]]) == NDimArray([
        -sin(x)*sin(y), cos(x)*cos(y), 0])


def test_issue_16160():
    assert Derivative(x**3, (x, x)).subs(x, 2) == Subs(
        Derivative(x**3, (x, 2)), x, 2)
    assert Derivative(1 + x**3, (x, x)).subs(x, 0
        ) == Derivative(1 + y**3, (y, 0)).subs(y, 0)
