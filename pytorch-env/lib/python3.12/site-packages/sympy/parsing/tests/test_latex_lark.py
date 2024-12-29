from sympy.testing.pytest import XFAIL
from sympy.parsing.latex.lark import parse_latex_lark
from sympy.external import import_module

from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.function import Derivative, Function
from sympy.core.numbers import E, oo, Rational
from sympy.core.power import Pow
from sympy.core.parameters import evaluate
from sympy.core.relational import GreaterThan, LessThan, StrictGreaterThan, StrictLessThan, Unequality
from sympy.core.symbol import Symbol
from sympy.functions.combinatorial.factorials import binomial, factorial
from sympy.functions.elementary.complexes import Abs, conjugate
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.miscellaneous import root, sqrt, Min, Max
from sympy.functions.elementary.trigonometric import asin, cos, csc, sec, sin, tan
from sympy.integrals.integrals import Integral
from sympy.series.limits import Limit

from sympy.core.relational import Eq, Ne, Lt, Le, Gt, Ge
from sympy.physics.quantum import Bra, Ket, InnerProduct
from sympy.abc import x, y, z, a, b, c, d, t, k, n

from .test_latex import theta, f, _Add, _Mul, _Pow, _Sqrt, _Conjugate, _Abs, _factorial, _exp, _binomial

lark = import_module("lark")

# disable tests if lark is not present
disabled = lark is None

# shorthand definitions that are only needed for the Lark LaTeX parser
def _Min(*args):
    return Min(*args, evaluate=False)


def _Max(*args):
    return Max(*args, evaluate=False)


def _log(a, b=E):
    if b == E:
        return log(a, evaluate=False)
    else:
        return log(a, b, evaluate=False)


# These LaTeX strings should parse to the corresponding SymPy expression
SYMBOL_EXPRESSION_PAIRS = [
    (r"x_0", Symbol('x_{0}')),
    (r"x_{1}", Symbol('x_{1}')),
    (r"x_a", Symbol('x_{a}')),
    (r"x_{b}", Symbol('x_{b}')),
    (r"h_\theta", Symbol('h_{theta}')),
    (r"h_{\theta}", Symbol('h_{theta}')),
    (r"y''_1", Symbol("y_{1}''")),
    (r"y_1''", Symbol("y_{1}''")),
    (r"\mathit{x}", Symbol('x')),
    (r"\mathit{test}", Symbol('test')),
    (r"\mathit{TEST}", Symbol('TEST')),
    (r"\mathit{HELLO world}", Symbol('HELLO world'))
]

UNEVALUATED_SIMPLE_EXPRESSION_PAIRS = [
    (r"0", 0),
    (r"1", 1),
    (r"-3.14", -3.14),
    (r"(-7.13)(1.5)", _Mul(-7.13, 1.5)),
    (r"1+1", _Add(1, 1)),
    (r"0+1", _Add(0, 1)),
    (r"1*2", _Mul(1, 2)),
    (r"0*1", _Mul(0, 1)),
    (r"x", x),
    (r"2x", 2 * x),
    (r"3x - 1", _Add(_Mul(3, x), -1)),
    (r"-c", -c),
    (r"\infty", oo),
    (r"a \cdot b", a * b),
    (r"1 \times 2 ", _Mul(1, 2)),
    (r"a / b", a / b),
    (r"a \div b", a / b),
    (r"a + b", a + b),
    (r"a + b - a", _Add(a + b, -a)),
    (r"(x + y) z", _Mul(_Add(x, y), z)),
    (r"a'b+ab'", _Add(_Mul(Symbol("a'"), b), _Mul(a, Symbol("b'"))))
]

EVALUATED_SIMPLE_EXPRESSION_PAIRS = [
    (r"(-7.13)(1.5)", -10.695),
    (r"1+1", 2),
    (r"0+1", 1),
    (r"1*2", 2),
    (r"0*1", 0),
    (r"2x", 2 * x),
    (r"3x - 1", 3 * x - 1),
    (r"-c", -c),
    (r"a \cdot b", a * b),
    (r"1 \times 2 ", 2),
    (r"a / b", a / b),
    (r"a \div b", a / b),
    (r"a + b", a + b),
    (r"a + b - a", b),
    (r"(x + y) z", (x + y) * z),
]

UNEVALUATED_FRACTION_EXPRESSION_PAIRS = [
    (r"\frac{a}{b}", a / b),
    (r"\dfrac{a}{b}", a / b),
    (r"\tfrac{a}{b}", a / b),
    (r"\frac12", _Mul(1, _Pow(2, -1))),
    (r"\frac12y", _Mul(_Mul(1, _Pow(2, -1)), y)),
    (r"\frac1234", _Mul(_Mul(1, _Pow(2, -1)), 34)),
    (r"\frac2{3}", _Mul(2, _Pow(3, -1))),
    (r"\frac{a + b}{c}", _Mul(a + b, _Pow(c, -1))),
    (r"\frac{7}{3}", _Mul(7, _Pow(3, -1)))
]

EVALUATED_FRACTION_EXPRESSION_PAIRS = [
    (r"\frac{a}{b}", a / b),
    (r"\dfrac{a}{b}", a / b),
    (r"\tfrac{a}{b}", a / b),
    (r"\frac12", Rational(1, 2)),
    (r"\frac12y", y / 2),
    (r"\frac1234", 17),
    (r"\frac2{3}", Rational(2, 3)),
    (r"\frac{a + b}{c}", (a + b) / c),
    (r"\frac{7}{3}", Rational(7, 3))
]

RELATION_EXPRESSION_PAIRS = [
    (r"x = y", Eq(x, y)),
    (r"x \neq y", Ne(x, y)),
    (r"x < y", Lt(x, y)),
    (r"x > y", Gt(x, y)),
    (r"x \leq y", Le(x, y)),
    (r"x \geq y", Ge(x, y)),
    (r"x \le y", Le(x, y)),
    (r"x \ge y", Ge(x, y)),
    (r"x < y", StrictLessThan(x, y)),
    (r"x \leq y", LessThan(x, y)),
    (r"x > y", StrictGreaterThan(x, y)),
    (r"x \geq y", GreaterThan(x, y)),
    (r"x \neq y", Unequality(x, y)), # same as 2nd one in the list
    (r"a^2 + b^2 = c^2", Eq(a**2 + b**2, c**2))
]

UNEVALUATED_POWER_EXPRESSION_PAIRS = [
    (r"x^2", x ** 2),
    (r"x^\frac{1}{2}", _Pow(x, _Mul(1, _Pow(2, -1)))),
    (r"x^{3 + 1}", x ** _Add(3, 1)),
    (r"\pi^{|xy|}", Symbol('pi') ** _Abs(x * y)),
    (r"5^0 - 4^0", _Add(_Pow(5, 0), _Mul(-1, _Pow(4, 0))))
]

EVALUATED_POWER_EXPRESSION_PAIRS = [
    (r"x^2", x ** 2),
    (r"x^\frac{1}{2}", sqrt(x)),
    (r"x^{3 + 1}", x ** 4),
    (r"\pi^{|xy|}", Symbol('pi') ** _Abs(x * y)),
    (r"5^0 - 4^0", 0)
]

UNEVALUATED_INTEGRAL_EXPRESSION_PAIRS = [
    (r"\int x dx", Integral(_Mul(1, x), x)),
    (r"\int x \, dx", Integral(_Mul(1, x), x)),
    (r"\int x d\theta", Integral(_Mul(1, x), theta)),
    (r"\int (x^2 - y)dx", Integral(_Mul(1, x ** 2 - y), x)),
    (r"\int x + a dx", Integral(_Mul(1, _Add(x, a)), x)),
    (r"\int da", Integral(_Mul(1, 1), a)),
    (r"\int_0^7 dx", Integral(_Mul(1, 1), (x, 0, 7))),
    (r"\int\limits_{0}^{1} x dx", Integral(_Mul(1, x), (x, 0, 1))),
    (r"\int_a^b x dx", Integral(_Mul(1, x), (x, a, b))),
    (r"\int^b_a x dx", Integral(_Mul(1, x), (x, a, b))),
    (r"\int_{a}^b x dx", Integral(_Mul(1, x), (x, a, b))),
    (r"\int^{b}_a x dx", Integral(_Mul(1, x), (x, a, b))),
    (r"\int_{a}^{b} x dx", Integral(_Mul(1, x), (x, a, b))),
    (r"\int^{b}_{a} x dx", Integral(_Mul(1, x), (x, a, b))),
    (r"\int_{f(a)}^{f(b)} f(z) dz", Integral(f(z), (z, f(a), f(b)))),
    (r"\int a + b + c dx", Integral(_Mul(1, _Add(_Add(a, b), c)), x)),
    (r"\int \frac{dz}{z}", Integral(_Mul(1, _Mul(1, Pow(z, -1))), z)),
    (r"\int \frac{3 dz}{z}", Integral(_Mul(1, _Mul(3, _Pow(z, -1))), z)),
    (r"\int \frac{1}{x} dx", Integral(_Mul(1, _Mul(1, Pow(x, -1))), x)),
    (r"\int \frac{1}{a} + \frac{1}{b} dx",
     Integral(_Mul(1, _Add(_Mul(1, _Pow(a, -1)), _Mul(1, Pow(b, -1)))), x)),
    (r"\int \frac{1}{x} + 1 dx", Integral(_Mul(1, _Add(_Mul(1, _Pow(x, -1)), 1)), x))
]

EVALUATED_INTEGRAL_EXPRESSION_PAIRS = [
    (r"\int x dx", Integral(x, x)),
    (r"\int x \, dx", Integral(x, x)),
    (r"\int x d\theta", Integral(x, theta)),
    (r"\int (x^2 - y)dx", Integral(x ** 2 - y, x)),
    (r"\int x + a dx", Integral(x + a, x)),
    (r"\int da", Integral(1, a)),
    (r"\int_0^7 dx", Integral(1, (x, 0, 7))),
    (r"\int\limits_{0}^{1} x dx", Integral(x, (x, 0, 1))),
    (r"\int_a^b x dx", Integral(x, (x, a, b))),
    (r"\int^b_a x dx", Integral(x, (x, a, b))),
    (r"\int_{a}^b x dx", Integral(x, (x, a, b))),
    (r"\int^{b}_a x dx", Integral(x, (x, a, b))),
    (r"\int_{a}^{b} x dx", Integral(x, (x, a, b))),
    (r"\int^{b}_{a} x dx", Integral(x, (x, a, b))),
    (r"\int_{f(a)}^{f(b)} f(z) dz", Integral(f(z), (z, f(a), f(b)))),
    (r"\int a + b + c dx", Integral(a + b + c, x)),
    (r"\int \frac{dz}{z}", Integral(Pow(z, -1), z)),
    (r"\int \frac{3 dz}{z}", Integral(3 * Pow(z, -1), z)),
    (r"\int \frac{1}{x} dx", Integral(1 / x, x)),
    (r"\int \frac{1}{a} + \frac{1}{b} dx", Integral(1 / a + 1 / b, x)),
    (r"\int \frac{1}{x} + 1 dx", Integral(1 / x + 1, x))
]

DERIVATIVE_EXPRESSION_PAIRS = [
    (r"\frac{d}{dx} x", Derivative(x, x)),
    (r"\frac{d}{dt} x", Derivative(x, t)),
    (r"\frac{d}{dx} ( \tan x )", Derivative(tan(x), x)),
    (r"\frac{d f(x)}{dx}", Derivative(f(x), x)),
    (r"\frac{d\theta(x)}{dx}", Derivative(Function('theta')(x), x))
]

TRIGONOMETRIC_EXPRESSION_PAIRS = [
    (r"\sin \theta", sin(theta)),
    (r"\sin(\theta)", sin(theta)),
    (r"\sin^{-1} a", asin(a)),
    (r"\sin a \cos b", _Mul(sin(a), cos(b))),
    (r"\sin \cos \theta", sin(cos(theta))),
    (r"\sin(\cos \theta)", sin(cos(theta))),
    (r"(\csc x)(\sec y)", csc(x) * sec(y)),
    (r"\frac{\sin{x}}2", _Mul(sin(x), _Pow(2, -1)))
]

UNEVALUATED_LIMIT_EXPRESSION_PAIRS = [
    (r"\lim_{x \to 3} a", Limit(a, x, 3, dir="+-")),
    (r"\lim_{x \rightarrow 3} a", Limit(a, x, 3, dir="+-")),
    (r"\lim_{x \Rightarrow 3} a", Limit(a, x, 3, dir="+-")),
    (r"\lim_{x \longrightarrow 3} a", Limit(a, x, 3, dir="+-")),
    (r"\lim_{x \Longrightarrow 3} a", Limit(a, x, 3, dir="+-")),
    (r"\lim_{x \to 3^{+}} a", Limit(a, x, 3, dir="+")),
    (r"\lim_{x \to 3^{-}} a", Limit(a, x, 3, dir="-")),
    (r"\lim_{x \to 3^+} a", Limit(a, x, 3, dir="+")),
    (r"\lim_{x \to 3^-} a", Limit(a, x, 3, dir="-")),
    (r"\lim_{x \to \infty} \frac{1}{x}", Limit(_Mul(1, _Pow(x, -1)), x, oo))
]

EVALUATED_LIMIT_EXPRESSION_PAIRS = [
    (r"\lim_{x \to \infty} \frac{1}{x}", Limit(1 / x, x, oo))
]

UNEVALUATED_SQRT_EXPRESSION_PAIRS = [
    (r"\sqrt{x}", sqrt(x)),
    (r"\sqrt{x + b}", sqrt(_Add(x, b))),
    (r"\sqrt[3]{\sin x}", _Pow(sin(x), _Pow(3, -1))),
    # the above test needed to be handled differently than the ones below because root
    # acts differently if its second argument is a number
    (r"\sqrt[y]{\sin x}", root(sin(x), y)),
    (r"\sqrt[\theta]{\sin x}", root(sin(x), theta)),
    (r"\sqrt{\frac{12}{6}}", _Sqrt(_Mul(12, _Pow(6, -1))))
]

EVALUATED_SQRT_EXPRESSION_PAIRS = [
    (r"\sqrt{x}", sqrt(x)),
    (r"\sqrt{x + b}", sqrt(x + b)),
    (r"\sqrt[3]{\sin x}", root(sin(x), 3)),
    (r"\sqrt[y]{\sin x}", root(sin(x), y)),
    (r"\sqrt[\theta]{\sin x}", root(sin(x), theta)),
    (r"\sqrt{\frac{12}{6}}", sqrt(2))
]

UNEVALUATED_FACTORIAL_EXPRESSION_PAIRS = [
    (r"x!", _factorial(x)),
    (r"100!", _factorial(100)),
    (r"\theta!", _factorial(theta)),
    (r"(x + 1)!", _factorial(_Add(x, 1))),
    (r"(x!)!", _factorial(_factorial(x))),
    (r"x!!!", _factorial(_factorial(_factorial(x)))),
    (r"5!7!", _Mul(_factorial(5), _factorial(7)))
]

EVALUATED_FACTORIAL_EXPRESSION_PAIRS = [
    (r"x!", factorial(x)),
    (r"100!", factorial(100)),
    (r"\theta!", factorial(theta)),
    (r"(x + 1)!", factorial(x + 1)),
    (r"(x!)!", factorial(factorial(x))),
    (r"x!!!", factorial(factorial(factorial(x)))),
    (r"5!7!", factorial(5) * factorial(7))
]

UNEVALUATED_SUM_EXPRESSION_PAIRS = [
    (r"\sum_{k = 1}^{3} c", Sum(_Mul(1, c), (k, 1, 3))),
    (r"\sum_{k = 1}^3 c", Sum(_Mul(1, c), (k, 1, 3))),
    (r"\sum^{3}_{k = 1} c", Sum(_Mul(1, c), (k, 1, 3))),
    (r"\sum^3_{k = 1} c", Sum(_Mul(1, c), (k, 1, 3))),
    (r"\sum_{k = 1}^{10} k^2", Sum(_Mul(1, k ** 2), (k, 1, 10))),
    (r"\sum_{n = 0}^{\infty} \frac{1}{n!}",
     Sum(_Mul(1, _Mul(1, _Pow(_factorial(n), -1))), (n, 0, oo)))
]

EVALUATED_SUM_EXPRESSION_PAIRS = [
    (r"\sum_{k = 1}^{3} c", Sum(c, (k, 1, 3))),
    (r"\sum_{k = 1}^3 c", Sum(c, (k, 1, 3))),
    (r"\sum^{3}_{k = 1} c", Sum(c, (k, 1, 3))),
    (r"\sum^3_{k = 1} c", Sum(c, (k, 1, 3))),
    (r"\sum_{k = 1}^{10} k^2", Sum(k ** 2, (k, 1, 10))),
    (r"\sum_{n = 0}^{\infty} \frac{1}{n!}", Sum(1 / factorial(n), (n, 0, oo)))
]

UNEVALUATED_PRODUCT_EXPRESSION_PAIRS = [
    (r"\prod_{a = b}^{c} x", Product(x, (a, b, c))),
    (r"\prod_{a = b}^c x", Product(x, (a, b, c))),
    (r"\prod^{c}_{a = b} x", Product(x, (a, b, c))),
    (r"\prod^c_{a = b} x", Product(x, (a, b, c)))
]

APPLIED_FUNCTION_EXPRESSION_PAIRS = [
    (r"f(x)", f(x)),
    (r"f(x, y)", f(x, y)),
    (r"f(x, y, z)", f(x, y, z)),
    (r"f'_1(x)", Function("f_{1}'")(x)),
    (r"f_{1}''(x+y)", Function("f_{1}''")(x + y)),
    (r"h_{\theta}(x_0, x_1)",
     Function('h_{theta}')(Symbol('x_{0}'), Symbol('x_{1}')))
]

UNEVALUATED_COMMON_FUNCTION_EXPRESSION_PAIRS = [
    (r"|x|", _Abs(x)),
    (r"||x||", _Abs(Abs(x))),
    (r"|x||y|", _Abs(x) * _Abs(y)),
    (r"||x||y||", _Abs(_Abs(x) * _Abs(y))),
    (r"\lfloor x \rfloor", floor(x)),
    (r"\lceil x \rceil", ceiling(x)),
    (r"\exp x", _exp(x)),
    (r"\exp(x)", _exp(x)),
    (r"\lg x", _log(x, 10)),
    (r"\ln x", _log(x)),
    (r"\ln xy", _log(x * y)),
    (r"\log x", _log(x)),
    (r"\log xy", _log(x * y)),
    (r"\log_{2} x", _log(x, 2)),
    (r"\log_{a} x", _log(x, a)),
    (r"\log_{11} x", _log(x, 11)),
    (r"\log_{a^2} x", _log(x, _Pow(a, 2))),
    (r"\log_2 x", _log(x, 2)),
    (r"\log_a x", _log(x, a)),
    (r"\overline{z}", _Conjugate(z)),
    (r"\overline{\overline{z}}", _Conjugate(_Conjugate(z))),
    (r"\overline{x + y}", _Conjugate(_Add(x, y))),
    (r"\overline{x} + \overline{y}", _Conjugate(x) + _Conjugate(y)),
    (r"\min(a, b)", _Min(a, b)),
    (r"\min(a, b, c - d, xy)", _Min(a, b, c - d, x * y)),
    (r"\max(a, b)", _Max(a, b)),
    (r"\max(a, b, c - d, xy)", _Max(a, b, c - d, x * y)),
    # physics things don't have an `evaluate=False` variant
    (r"\langle x |", Bra('x')),
    (r"| x \rangle", Ket('x')),
    (r"\langle x | y \rangle", InnerProduct(Bra('x'), Ket('y'))),
]

EVALUATED_COMMON_FUNCTION_EXPRESSION_PAIRS = [
    (r"|x|", Abs(x)),
    (r"||x||", Abs(Abs(x))),
    (r"|x||y|", Abs(x) * Abs(y)),
    (r"||x||y||", Abs(Abs(x) * Abs(y))),
    (r"\lfloor x \rfloor", floor(x)),
    (r"\lceil x \rceil", ceiling(x)),
    (r"\exp x", exp(x)),
    (r"\exp(x)", exp(x)),
    (r"\lg x", log(x, 10)),
    (r"\ln x", log(x)),
    (r"\ln xy", log(x * y)),
    (r"\log x", log(x)),
    (r"\log xy", log(x * y)),
    (r"\log_{2} x", log(x, 2)),
    (r"\log_{a} x", log(x, a)),
    (r"\log_{11} x", log(x, 11)),
    (r"\log_{a^2} x", log(x, _Pow(a, 2))),
    (r"\log_2 x", log(x, 2)),
    (r"\log_a x", log(x, a)),
    (r"\overline{z}", conjugate(z)),
    (r"\overline{\overline{z}}", conjugate(conjugate(z))),
    (r"\overline{x + y}", conjugate(x + y)),
    (r"\overline{x} + \overline{y}", conjugate(x) + conjugate(y)),
    (r"\min(a, b)", Min(a, b)),
    (r"\min(a, b, c - d, xy)", Min(a, b, c - d, x * y)),
    (r"\max(a, b)", Max(a, b)),
    (r"\max(a, b, c - d, xy)", Max(a, b, c - d, x * y)),
    (r"\langle x |", Bra('x')),
    (r"| x \rangle", Ket('x')),
    (r"\langle x | y \rangle", InnerProduct(Bra('x'), Ket('y'))),
]

SPACING_RELATED_EXPRESSION_PAIRS = [
    (r"a \, b", _Mul(a, b)),
    (r"a \thinspace b", _Mul(a, b)),
    (r"a \: b", _Mul(a, b)),
    (r"a \medspace b", _Mul(a, b)),
    (r"a \; b", _Mul(a, b)),
    (r"a \thickspace b", _Mul(a, b)),
    (r"a \quad b", _Mul(a, b)),
    (r"a \qquad b", _Mul(a, b)),
    (r"a \! b", _Mul(a, b)),
    (r"a \negthinspace b", _Mul(a, b)),
    (r"a \negmedspace b", _Mul(a, b)),
    (r"a \negthickspace b", _Mul(a, b))
]

UNEVALUATED_BINOMIAL_EXPRESSION_PAIRS = [
    (r"\binom{n}{k}", _binomial(n, k)),
    (r"\tbinom{n}{k}", _binomial(n, k)),
    (r"\dbinom{n}{k}", _binomial(n, k)),
    (r"\binom{n}{0}", _binomial(n, 0)),
    (r"x^\binom{n}{k}", _Pow(x, _binomial(n, k)))
]

EVALUATED_BINOMIAL_EXPRESSION_PAIRS = [
    (r"\binom{n}{k}", binomial(n, k)),
    (r"\tbinom{n}{k}", binomial(n, k)),
    (r"\dbinom{n}{k}", binomial(n, k)),
    (r"\binom{n}{0}", binomial(n, 0)),
    (r"x^\binom{n}{k}", x ** binomial(n, k))
]

MISCELLANEOUS_EXPRESSION_PAIRS = [
    (r"\left(x + y\right) z", _Mul(_Add(x, y), z)),
    (r"\left( x + y\right ) z", _Mul(_Add(x, y), z)),
    (r"\left(  x + y\right ) z", _Mul(_Add(x, y), z)),
]


def test_symbol_expressions():
    expected_failures = {6, 7}
    for i, (latex_str, sympy_expr) in enumerate(SYMBOL_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str


def test_simple_expressions():
    expected_failures = {20}
    for i, (latex_str, sympy_expr) in enumerate(UNEVALUATED_SIMPLE_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

    for i, (latex_str, sympy_expr) in enumerate(EVALUATED_SIMPLE_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str


def test_fraction_expressions():
    for latex_str, sympy_expr in UNEVALUATED_FRACTION_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

    for latex_str, sympy_expr in EVALUATED_FRACTION_EXPRESSION_PAIRS:
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str


def test_relation_expressions():
    for latex_str, sympy_expr in RELATION_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

def test_power_expressions():
    expected_failures = {3}
    for i, (latex_str, sympy_expr) in enumerate(UNEVALUATED_POWER_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

    for i, (latex_str, sympy_expr) in enumerate(EVALUATED_POWER_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str


def test_integral_expressions():
    expected_failures = {14}
    for i, (latex_str, sympy_expr) in enumerate(UNEVALUATED_INTEGRAL_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, i

    for i, (latex_str, sympy_expr) in enumerate(EVALUATED_INTEGRAL_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str


def test_derivative_expressions():
    expected_failures = {3, 4}
    for i, (latex_str, sympy_expr) in enumerate(DERIVATIVE_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

    for i, (latex_str, sympy_expr) in enumerate(DERIVATIVE_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str


def test_trigonometric_expressions():
    expected_failures = {3}
    for i, (latex_str, sympy_expr) in enumerate(TRIGONOMETRIC_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str


def test_limit_expressions():
    for latex_str, sympy_expr in UNEVALUATED_LIMIT_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str


def test_square_root_expressions():
    for latex_str, sympy_expr in UNEVALUATED_SQRT_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

    for latex_str, sympy_expr in EVALUATED_SQRT_EXPRESSION_PAIRS:
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str


def test_factorial_expressions():
    for latex_str, sympy_expr in UNEVALUATED_FACTORIAL_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

    for latex_str, sympy_expr in EVALUATED_FACTORIAL_EXPRESSION_PAIRS:
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str


def test_sum_expressions():
    for latex_str, sympy_expr in UNEVALUATED_SUM_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

    for latex_str, sympy_expr in EVALUATED_SUM_EXPRESSION_PAIRS:
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str


def test_product_expressions():
    for latex_str, sympy_expr in UNEVALUATED_PRODUCT_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

@XFAIL
def test_applied_function_expressions():
    expected_failures = {0, 3, 4}  # 0 is ambiguous, and the others require not-yet-added features
    # not sure why 1, and 2 are failing
    for i, (latex_str, sympy_expr) in enumerate(APPLIED_FUNCTION_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str


def test_common_function_expressions():
    for latex_str, sympy_expr in UNEVALUATED_COMMON_FUNCTION_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

    for latex_str, sympy_expr in EVALUATED_COMMON_FUNCTION_EXPRESSION_PAIRS:
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str

# unhandled bug causing these to fail
@XFAIL
def test_spacing():
    for latex_str, sympy_expr in SPACING_RELATED_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str


def test_binomial_expressions():
    for latex_str, sympy_expr in UNEVALUATED_BINOMIAL_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

    for latex_str, sympy_expr in EVALUATED_BINOMIAL_EXPRESSION_PAIRS:
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str


def test_miscellaneous_expressions():
    for latex_str, sympy_expr in MISCELLANEOUS_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str
