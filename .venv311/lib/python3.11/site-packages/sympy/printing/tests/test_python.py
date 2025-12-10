from sympy.core.function import (Derivative, Function)
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (Abs, conjugate)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.integrals.integrals import Integral
from sympy.matrices.dense import Matrix
from sympy.series.limits import limit

from sympy.printing.python import python

from sympy.testing.pytest import raises, XFAIL

x, y = symbols('x,y')
th = Symbol('theta')
ph = Symbol('phi')


def test_python_basic():
    # Simple numbers/symbols
    assert python(-Rational(1)/2) == "e = Rational(-1, 2)"
    assert python(-Rational(13)/22) == "e = Rational(-13, 22)"
    assert python(oo) == "e = oo"

    # Powers
    assert python(x**2) == "x = Symbol(\'x\')\ne = x**2"
    assert python(1/x) == "x = Symbol('x')\ne = 1/x"
    assert python(y*x**-2) == "y = Symbol('y')\nx = Symbol('x')\ne = y/x**2"
    assert python(
        x**Rational(-5, 2)) == "x = Symbol('x')\ne = x**Rational(-5, 2)"

    # Sums of terms
    assert python(x**2 + x + 1) in [
        "x = Symbol('x')\ne = 1 + x + x**2",
        "x = Symbol('x')\ne = x + x**2 + 1",
        "x = Symbol('x')\ne = x**2 + x + 1", ]
    assert python(1 - x) in [
        "x = Symbol('x')\ne = 1 - x",
        "x = Symbol('x')\ne = -x + 1"]
    assert python(1 - 2*x) in [
        "x = Symbol('x')\ne = 1 - 2*x",
        "x = Symbol('x')\ne = -2*x + 1"]
    assert python(1 - Rational(3, 2)*y/x) in [
        "y = Symbol('y')\nx = Symbol('x')\ne = 1 - 3/2*y/x",
        "y = Symbol('y')\nx = Symbol('x')\ne = -3/2*y/x + 1",
        "y = Symbol('y')\nx = Symbol('x')\ne = 1 - 3*y/(2*x)"]

    # Multiplication
    assert python(x/y) == "x = Symbol('x')\ny = Symbol('y')\ne = x/y"
    assert python(-x/y) == "x = Symbol('x')\ny = Symbol('y')\ne = -x/y"
    assert python((x + 2)/y) in [
        "y = Symbol('y')\nx = Symbol('x')\ne = 1/y*(2 + x)",
        "y = Symbol('y')\nx = Symbol('x')\ne = 1/y*(x + 2)",
        "x = Symbol('x')\ny = Symbol('y')\ne = 1/y*(2 + x)",
        "x = Symbol('x')\ny = Symbol('y')\ne = (2 + x)/y",
        "x = Symbol('x')\ny = Symbol('y')\ne = (x + 2)/y"]
    assert python((1 + x)*y) in [
        "y = Symbol('y')\nx = Symbol('x')\ne = y*(1 + x)",
        "y = Symbol('y')\nx = Symbol('x')\ne = y*(x + 1)", ]

    # Check for proper placement of negative sign
    assert python(-5*x/(x + 10)) == "x = Symbol('x')\ne = -5*x/(x + 10)"
    assert python(1 - Rational(3, 2)*(x + 1)) in [
        "x = Symbol('x')\ne = Rational(-3, 2)*x + Rational(-1, 2)",
        "x = Symbol('x')\ne = -3*x/2 + Rational(-1, 2)",
        "x = Symbol('x')\ne = -3*x/2 + Rational(-1, 2)"
    ]


def test_python_keyword_symbol_name_escaping():
    # Check for escaping of keywords
    assert python(
        5*Symbol("lambda")) == "lambda_ = Symbol('lambda')\ne = 5*lambda_"
    assert (python(5*Symbol("lambda") + 7*Symbol("lambda_")) ==
            "lambda__ = Symbol('lambda')\nlambda_ = Symbol('lambda_')\ne = 7*lambda_ + 5*lambda__")
    assert (python(5*Symbol("for") + Function("for_")(8)) ==
            "for__ = Symbol('for')\nfor_ = Function('for_')\ne = 5*for__ + for_(8)")


def test_python_keyword_function_name_escaping():
    assert python(
        5*Function("for")(8)) == "for_ = Function('for')\ne = 5*for_(8)"


def test_python_relational():
    assert python(Eq(x, y)) == "x = Symbol('x')\ny = Symbol('y')\ne = Eq(x, y)"
    assert python(Ge(x, y)) == "x = Symbol('x')\ny = Symbol('y')\ne = x >= y"
    assert python(Le(x, y)) == "x = Symbol('x')\ny = Symbol('y')\ne = x <= y"
    assert python(Gt(x, y)) == "x = Symbol('x')\ny = Symbol('y')\ne = x > y"
    assert python(Lt(x, y)) == "x = Symbol('x')\ny = Symbol('y')\ne = x < y"
    assert python(Ne(x/(y + 1), y**2)) in [
        "x = Symbol('x')\ny = Symbol('y')\ne = Ne(x/(1 + y), y**2)",
        "x = Symbol('x')\ny = Symbol('y')\ne = Ne(x/(y + 1), y**2)"]


def test_python_functions():
    # Simple
    assert python(2*x + exp(x)) in "x = Symbol('x')\ne = 2*x + exp(x)"
    assert python(sqrt(2)) == 'e = sqrt(2)'
    assert python(2**Rational(1, 3)) == 'e = 2**Rational(1, 3)'
    assert python(sqrt(2 + pi)) == 'e = sqrt(2 + pi)'
    assert python((2 + pi)**Rational(1, 3)) == 'e = (2 + pi)**Rational(1, 3)'
    assert python(2**Rational(1, 4)) == 'e = 2**Rational(1, 4)'
    assert python(Abs(x)) == "x = Symbol('x')\ne = Abs(x)"
    assert python(
        Abs(x/(x**2 + 1))) in ["x = Symbol('x')\ne = Abs(x/(1 + x**2))",
            "x = Symbol('x')\ne = Abs(x/(x**2 + 1))"]

    # Univariate/Multivariate functions
    f = Function('f')
    assert python(f(x)) == "x = Symbol('x')\nf = Function('f')\ne = f(x)"
    assert python(f(x, y)) == "x = Symbol('x')\ny = Symbol('y')\nf = Function('f')\ne = f(x, y)"
    assert python(f(x/(y + 1), y)) in [
        "x = Symbol('x')\ny = Symbol('y')\nf = Function('f')\ne = f(x/(1 + y), y)",
        "x = Symbol('x')\ny = Symbol('y')\nf = Function('f')\ne = f(x/(y + 1), y)"]

    # Nesting of square roots
    assert python(sqrt((sqrt(x + 1)) + 1)) in [
        "x = Symbol('x')\ne = sqrt(1 + sqrt(1 + x))",
        "x = Symbol('x')\ne = sqrt(sqrt(x + 1) + 1)"]

    # Nesting of powers
    assert python((((x + 1)**Rational(1, 3)) + 1)**Rational(1, 3)) in [
        "x = Symbol('x')\ne = (1 + (1 + x)**Rational(1, 3))**Rational(1, 3)",
        "x = Symbol('x')\ne = ((x + 1)**Rational(1, 3) + 1)**Rational(1, 3)"]

    # Function powers
    assert python(sin(x)**2) == "x = Symbol('x')\ne = sin(x)**2"


@XFAIL
def test_python_functions_conjugates():
    a, b = map(Symbol, 'ab')
    assert python( conjugate(a + b*I) ) == '_     _\na - I*b'
    assert python( conjugate(exp(a + b*I)) ) == ' _     _\n a - I*b\ne       '


def test_python_derivatives():
    # Simple
    f_1 = Derivative(log(x), x, evaluate=False)
    assert python(f_1) == "x = Symbol('x')\ne = Derivative(log(x), x)"

    f_2 = Derivative(log(x), x, evaluate=False) + x
    assert python(f_2) == "x = Symbol('x')\ne = x + Derivative(log(x), x)"

    # Multiple symbols
    f_3 = Derivative(log(x) + x**2, x, y, evaluate=False)
    assert python(f_3) == \
        "x = Symbol('x')\ny = Symbol('y')\ne = Derivative(x**2 + log(x), x, y)"

    f_4 = Derivative(2*x*y, y, x, evaluate=False) + x**2
    assert python(f_4) in [
        "x = Symbol('x')\ny = Symbol('y')\ne = x**2 + Derivative(2*x*y, y, x)",
        "x = Symbol('x')\ny = Symbol('y')\ne = Derivative(2*x*y, y, x) + x**2"]


def test_python_integrals():
    # Simple
    f_1 = Integral(log(x), x)
    assert python(f_1) == "x = Symbol('x')\ne = Integral(log(x), x)"

    f_2 = Integral(x**2, x)
    assert python(f_2) == "x = Symbol('x')\ne = Integral(x**2, x)"

    # Double nesting of pow
    f_3 = Integral(x**(2**x), x)
    assert python(f_3) == "x = Symbol('x')\ne = Integral(x**(2**x), x)"

    # Definite integrals
    f_4 = Integral(x**2, (x, 1, 2))
    assert python(f_4) == "x = Symbol('x')\ne = Integral(x**2, (x, 1, 2))"

    f_5 = Integral(x**2, (x, Rational(1, 2), 10))
    assert python(
        f_5) == "x = Symbol('x')\ne = Integral(x**2, (x, Rational(1, 2), 10))"

    # Nested integrals
    f_6 = Integral(x**2*y**2, x, y)
    assert python(f_6) == "x = Symbol('x')\ny = Symbol('y')\ne = Integral(x**2*y**2, x, y)"


def test_python_matrix():
    p = python(Matrix([[x**2+1, 1], [y, x+y]]))
    s = "x = Symbol('x')\ny = Symbol('y')\ne = MutableDenseMatrix([[x**2 + 1, 1], [y, x + y]])"
    assert p == s

def test_python_limits():
    assert python(limit(x, x, oo)) == 'e = oo'
    assert python(limit(x**2, x, 0)) == 'e = 0'

def test_issue_20762():
    # Make sure Python removes curly braces from subscripted variables
    a_b = Symbol('a_{b}')
    b = Symbol('b')
    expr = a_b*b
    assert python(expr) == "a_b = Symbol('a_{b}')\nb = Symbol('b')\ne = a_b*b"


def test_settings():
    raises(TypeError, lambda: python(x, method="garbage"))
