from sympy.core.numbers import (Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.codegen.cfunctions import (
    expm1, log1p, exp2, log2, fma, log10, Sqrt, Cbrt, hypot, isnan, isinf
)
from sympy.core.function import expand_log


def test_expm1():
    # Eval
    assert expm1(0) == 0

    x = Symbol('x', real=True)

    # Expand and rewrite
    assert expm1(x).expand(func=True) - exp(x) == -1
    assert expm1(x).rewrite('tractable') - exp(x) == -1
    assert expm1(x).rewrite('exp') - exp(x) == -1

    # Precision
    assert not ((exp(1e-10).evalf() - 1) - 1e-10 - 5e-21) < 1e-22  # for comparison
    assert abs(expm1(1e-10).evalf() - 1e-10 - 5e-21) < 1e-22

    # Properties
    assert expm1(x).is_real
    assert expm1(x).is_finite

    # Diff
    assert expm1(42*x).diff(x) - 42*exp(42*x) == 0
    assert expm1(42*x).diff(x) - expm1(42*x).expand(func=True).diff(x) == 0


def test_log1p():
    # Eval
    assert log1p(0) == 0
    d = S(10)
    assert expand_log(log1p(d**-1000) - log(d**1000 + 1) + log(d**1000)) == 0

    x = Symbol('x', real=True)

    # Expand and rewrite
    assert log1p(x).expand(func=True) - log(x + 1) == 0
    assert log1p(x).rewrite('tractable') - log(x + 1) == 0
    assert log1p(x).rewrite('log') - log(x + 1) == 0

    # Precision
    assert not abs(log(1e-99 + 1).evalf() - 1e-99) < 1e-100  # for comparison
    assert abs(expand_log(log1p(1e-99)).evalf() - 1e-99) < 1e-100

    # Properties
    assert log1p(-2**Rational(-1, 2)).is_real

    assert not log1p(-1).is_finite
    assert log1p(pi).is_finite

    assert not log1p(x).is_positive
    assert log1p(Symbol('y', positive=True)).is_positive

    assert not log1p(x).is_zero
    assert log1p(Symbol('z', zero=True)).is_zero

    assert not log1p(x).is_nonnegative
    assert log1p(Symbol('o', nonnegative=True)).is_nonnegative

    # Diff
    assert log1p(42*x).diff(x) - 42/(42*x + 1) == 0
    assert log1p(42*x).diff(x) - log1p(42*x).expand(func=True).diff(x) == 0


def test_exp2():
    # Eval
    assert exp2(2) == 4

    x = Symbol('x', real=True)

    # Expand
    assert exp2(x).expand(func=True) - 2**x == 0

    # Diff
    assert exp2(42*x).diff(x) - 42*exp2(42*x)*log(2) == 0
    assert exp2(42*x).diff(x) - exp2(42*x).diff(x) == 0


def test_log2():
    # Eval
    assert log2(8) == 3
    assert log2(pi) != log(pi)/log(2)  # log2 should *save* (CPU) instructions

    x = Symbol('x', real=True)
    assert log2(x) != log(x)/log(2)
    assert log2(2**x) == x

    # Expand
    assert log2(x).expand(func=True) - log(x)/log(2) == 0

    # Diff
    assert log2(42*x).diff() - 1/(log(2)*x) == 0
    assert log2(42*x).diff() - log2(42*x).expand(func=True).diff(x) == 0


def test_fma():
    x, y, z = symbols('x y z')

    # Expand
    assert fma(x, y, z).expand(func=True) - x*y - z == 0

    expr = fma(17*x, 42*y, 101*z)

    # Diff
    assert expr.diff(x) - expr.expand(func=True).diff(x) == 0
    assert expr.diff(y) - expr.expand(func=True).diff(y) == 0
    assert expr.diff(z) - expr.expand(func=True).diff(z) == 0

    assert expr.diff(x) - 17*42*y == 0
    assert expr.diff(y) - 17*42*x == 0
    assert expr.diff(z) - 101 == 0


def test_log10():
    x = Symbol('x')

    # Expand
    assert log10(x).expand(func=True) - log(x)/log(10) == 0

    # Diff
    assert log10(42*x).diff(x) - 1/(log(10)*x) == 0
    assert log10(42*x).diff(x) - log10(42*x).expand(func=True).diff(x) == 0


def test_Cbrt():
    x = Symbol('x')

    # Expand
    assert Cbrt(x).expand(func=True) - x**Rational(1, 3) == 0

    # Diff
    assert Cbrt(42*x).diff(x) - 42*(42*x)**(Rational(1, 3) - 1)/3 == 0
    assert Cbrt(42*x).diff(x) - Cbrt(42*x).expand(func=True).diff(x) == 0


def test_Sqrt():
    x = Symbol('x')

    # Expand
    assert Sqrt(x).expand(func=True) - x**S.Half == 0

    # Diff
    assert Sqrt(42*x).diff(x) - 42*(42*x)**(S.Half - 1)/2 == 0
    assert Sqrt(42*x).diff(x) - Sqrt(42*x).expand(func=True).diff(x) == 0


def test_hypot():
    x, y = symbols('x y')

    # Expand
    assert hypot(x, y).expand(func=True) - (x**2 + y**2)**S.Half == 0

    # Diff
    assert hypot(17*x, 42*y).diff(x).expand(func=True) - hypot(17*x, 42*y).expand(func=True).diff(x) == 0
    assert hypot(17*x, 42*y).diff(y).expand(func=True) - hypot(17*x, 42*y).expand(func=True).diff(y) == 0

    assert hypot(17*x, 42*y).diff(x).expand(func=True) - 2*17*17*x*((17*x)**2 + (42*y)**2)**Rational(-1, 2)/2 == 0
    assert hypot(17*x, 42*y).diff(y).expand(func=True) - 2*42*42*y*((17*x)**2 + (42*y)**2)**Rational(-1, 2)/2 == 0


def test_isnan_isinf():
    x = Symbol('x')

    # isinf
    assert isinf(+S.Infinity) == True
    assert isinf(-S.Infinity) == True
    assert isinf(S.Pi) == False
    isinfx = isinf(x)
    assert isinfx not in (False, True)
    assert isinfx.func is isinf
    assert isinfx.args == (x,)

    # isnan
    assert isnan(S.NaN) == True
    assert isnan(S.Pi) == False
    isnanx = isnan(x)
    assert isnanx not in (False, True)
    assert isnanx.func is isnan
    assert isnanx.args == (x,)
