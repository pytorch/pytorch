from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.function import Derivative, Function
from sympy.core.numbers import Integer, Rational, Float, oo
from sympy.core.relational import Rel
from sympy.core.symbol import symbols
from sympy.functions import sin
from sympy.integrals.integrals import Integral
from sympy.series.order import Order

from sympy.printing.precedence import precedence, PRECEDENCE

x, y = symbols("x,y")


def test_Add():
    assert precedence(x + y) == PRECEDENCE["Add"]
    assert precedence(x*y + 1) == PRECEDENCE["Add"]


def test_Function():
    assert precedence(sin(x)) == PRECEDENCE["Func"]

def test_Derivative():
    assert precedence(Derivative(x, y)) == PRECEDENCE["Atom"]

def test_Integral():
    assert precedence(Integral(x, y)) == PRECEDENCE["Atom"]


def test_Mul():
    assert precedence(x*y) == PRECEDENCE["Mul"]
    assert precedence(-x*y) == PRECEDENCE["Add"]


def test_Number():
    assert precedence(Integer(0)) == PRECEDENCE["Atom"]
    assert precedence(Integer(1)) == PRECEDENCE["Atom"]
    assert precedence(Integer(-1)) == PRECEDENCE["Add"]
    assert precedence(Integer(10)) == PRECEDENCE["Atom"]
    assert precedence(Rational(5, 2)) == PRECEDENCE["Mul"]
    assert precedence(Rational(-5, 2)) == PRECEDENCE["Add"]
    assert precedence(Float(5)) == PRECEDENCE["Atom"]
    assert precedence(Float(-5)) == PRECEDENCE["Add"]
    assert precedence(oo) == PRECEDENCE["Atom"]
    assert precedence(-oo) == PRECEDENCE["Add"]


def test_Order():
    assert precedence(Order(x)) == PRECEDENCE["Atom"]


def test_Pow():
    assert precedence(x**y) == PRECEDENCE["Pow"]
    assert precedence(-x**y) == PRECEDENCE["Add"]
    assert precedence(x**-y) == PRECEDENCE["Pow"]


def test_Product():
    assert precedence(Product(x, (x, y, y + 1))) == PRECEDENCE["Atom"]


def test_Relational():
    assert precedence(Rel(x + y, y, "<")) == PRECEDENCE["Relational"]


def test_Sum():
    assert precedence(Sum(x, (x, y, y + 1))) == PRECEDENCE["Atom"]


def test_Symbol():
    assert precedence(x) == PRECEDENCE["Atom"]


def test_And_Or():
    # precedence relations between logical operators, ...
    assert precedence(x & y) > precedence(x | y)
    assert precedence(~y) > precedence(x & y)
    # ... and with other operators (cfr. other programming languages)
    assert precedence(x + y) > precedence(x | y)
    assert precedence(x + y) > precedence(x & y)
    assert precedence(x*y) > precedence(x | y)
    assert precedence(x*y) > precedence(x & y)
    assert precedence(~y) > precedence(x*y)
    assert precedence(~y) > precedence(x - y)
    # double checks
    assert precedence(x & y) == PRECEDENCE["And"]
    assert precedence(x | y) == PRECEDENCE["Or"]
    assert precedence(~y) == PRECEDENCE["Not"]


def test_custom_function_precedence_comparison():
    """
    Test cases for custom functions with different precedence values,
    specifically handling:
    1. Functions with precedence < PRECEDENCE["Mul"] (50)
    2. Functions with precedence = Func (70)

    Key distinction:
    1. Lower precedence functions (45) need parentheses: -2*(x F y)
    2. Higher precedence functions (70) don't: -2*x F y
    """
    class LowPrecedenceF(Function):
        precedence = PRECEDENCE["Mul"] - 5
        def _sympystr(self, printer):
            return f"{printer._print(self.args[0])} F {printer._print(self.args[1])}"

    class HighPrecedenceF(Function):
        precedence = PRECEDENCE["Func"]
        def _sympystr(self, printer):
            return f"{printer._print(self.args[0])} F {printer._print(self.args[1])}"

    def test_low_precedence():
        expr1 = 2 * LowPrecedenceF(x, y)
        assert str(expr1) == "2*(x F y)"

        expr2 = -2 * LowPrecedenceF(x, y)
        assert str(expr2) == "-2*(x F y)"

    def test_high_precedence():
        expr1 = 2 * HighPrecedenceF(x, y)
        assert str(expr1) == "2*x F y"

        expr2 = -2 * HighPrecedenceF(x, y)
        assert str(expr2) == "-2*x F y"

    test_low_precedence()
    test_high_precedence()
