from sympy import MatAdd
from sympy.algebras.quaternion import Quaternion
from sympy.assumptions.ask import Q
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.combinatorics.partitions import Partition
from sympy.concrete.summations import (Sum, summation)
from sympy.core.add import Add
from sympy.core.containers import (Dict, Tuple)
from sympy.core.expr import UnevaluatedExpr, Expr
from sympy.core.function import (Derivative, Function, Lambda, Subs, WildFunction)
from sympy.core.mul import Mul
from sympy.core import (Catalan, EulerGamma, GoldenRatio, TribonacciConstant)
from sympy.core.numbers import (E, Float, I, Integer, Rational, nan, oo, pi, zoo)
from sympy.core.parameters import _exp_is_pow
from sympy.core.power import Pow
from sympy.core.relational import (Eq, Rel, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, Wild, symbols)
from sympy.functions.combinatorial.factorials import (factorial, factorial2, subfactorial)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.delta_functions import Heaviside
from sympy.functions.special.zeta_functions import zeta
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import (Equivalent, false, true, Xor)
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions import Identity
from sympy.matrices.expressions.slice import MatrixSlice
from sympy.matrices import SparseMatrix
from sympy.polys.polytools import factor
from sympy.series.limits import Limit
from sympy.series.order import O
from sympy.sets.sets import (Complement, FiniteSet, Interval, SymmetricDifference)
from sympy.stats import (Covariance, Expectation, Probability, Variance)
from sympy.stats.rv import RandomSymbol
from sympy.external import import_module
from sympy.physics.control.lti import TransferFunction, Series, Parallel, \
    Feedback, TransferFunctionMatrix, MIMOSeries, MIMOParallel, MIMOFeedback
from sympy.physics.units import second, joule
from sympy.polys import (Poly, rootof, RootSum, groebner, ring, field, ZZ, QQ,
    ZZ_I, QQ_I, lex, grlex)
from sympy.geometry import Point, Circle, Polygon, Ellipse, Triangle
from sympy.tensor import NDimArray
from sympy.tensor.array.expressions.array_expressions import ArraySymbol, ArrayElement

from sympy.testing.pytest import raises, warns_deprecated_sympy

from sympy.printing import sstr, sstrrepr, StrPrinter
from sympy.physics.quantum.trace import Tr

x, y, z, w, t = symbols('x,y,z,w,t')
d = Dummy('d')


def test_printmethod():
    class R(Abs):
        def _sympystr(self, printer):
            return "foo(%s)" % printer._print(self.args[0])
    assert sstr(R(x)) == "foo(x)"

    class R(Abs):
        def _sympystr(self, printer):
            return "foo"
    assert sstr(R(x)) == "foo"


def test_Abs():
    assert str(Abs(x)) == "Abs(x)"
    assert str(Abs(Rational(1, 6))) == "1/6"
    assert str(Abs(Rational(-1, 6))) == "1/6"


def test_Add():
    assert str(x + y) == "x + y"
    assert str(x + 1) == "x + 1"
    assert str(x + x**2) == "x**2 + x"
    assert str(Add(0, 1, evaluate=False)) == "0 + 1"
    assert str(Add(0, 0, 1, evaluate=False)) == "0 + 0 + 1"
    assert str(1.0*x) == "1.0*x"
    assert str(5 + x + y + x*y + x**2 + y**2) == "x**2 + x*y + x + y**2 + y + 5"
    assert str(1 + x + x**2/2 + x**3/3) == "x**3/3 + x**2/2 + x + 1"
    assert str(2*x - 7*x**2 + 2 + 3*y) == "-7*x**2 + 2*x + 3*y + 2"
    assert str(x - y) == "x - y"
    assert str(2 - x) == "2 - x"
    assert str(x - 2) == "x - 2"
    assert str(x - y - z - w) == "-w + x - y - z"
    assert str(x - z*y**2*z*w) == "-w*y**2*z**2 + x"
    assert str(x - 1*y*x*y) == "-x*y**2 + x"
    assert str(sin(x).series(x, 0, 15)) == "x - x**3/6 + x**5/120 - x**7/5040 + x**9/362880 - x**11/39916800 + x**13/6227020800 + O(x**15)"
    assert str(Add(Add(-w, x, evaluate=False), Add(-y, z,  evaluate=False),  evaluate=False)) == "(-w + x) + (-y + z)"
    assert str(Add(Add(-x, -y, evaluate=False), -z, evaluate=False)) == "-z + (-x - y)"
    assert str(Add(Add(Add(-x, -y, evaluate=False), -z, evaluate=False), -t, evaluate=False)) == "-t + (-z + (-x - y))"


def test_Catalan():
    assert str(Catalan) == "Catalan"


def test_ComplexInfinity():
    assert str(zoo) == "zoo"


def test_Derivative():
    assert str(Derivative(x, y)) == "Derivative(x, y)"
    assert str(Derivative(x**2, x, evaluate=False)) == "Derivative(x**2, x)"
    assert str(Derivative(
        x**2/y, x, y, evaluate=False)) == "Derivative(x**2/y, x, y)"


def test_dict():
    assert str({1: 1 + x}) == sstr({1: 1 + x}) == "{1: x + 1}"
    assert str({1: x**2, 2: y*x}) in ("{1: x**2, 2: x*y}", "{2: x*y, 1: x**2}")
    assert sstr({1: x**2, 2: y*x}) == "{1: x**2, 2: x*y}"


def test_Dict():
    assert str(Dict({1: 1 + x})) == sstr({1: 1 + x}) == "{1: x + 1}"
    assert str(Dict({1: x**2, 2: y*x})) in (
        "{1: x**2, 2: x*y}", "{2: x*y, 1: x**2}")
    assert sstr(Dict({1: x**2, 2: y*x})) == "{1: x**2, 2: x*y}"


def test_Dummy():
    assert str(d) == "_d"
    assert str(d + x) == "_d + x"


def test_EulerGamma():
    assert str(EulerGamma) == "EulerGamma"


def test_Exp():
    assert str(E) == "E"
    with _exp_is_pow(True):
        assert str(exp(x)) == "E**x"


def test_factorial():
    n = Symbol('n', integer=True)
    assert str(factorial(-2)) == "zoo"
    assert str(factorial(0)) == "1"
    assert str(factorial(7)) == "5040"
    assert str(factorial(n)) == "factorial(n)"
    assert str(factorial(2*n)) == "factorial(2*n)"
    assert str(factorial(factorial(n))) == 'factorial(factorial(n))'
    assert str(factorial(factorial2(n))) == 'factorial(factorial2(n))'
    assert str(factorial2(factorial(n))) == 'factorial2(factorial(n))'
    assert str(factorial2(factorial2(n))) == 'factorial2(factorial2(n))'
    assert str(subfactorial(3)) == "2"
    assert str(subfactorial(n)) == "subfactorial(n)"
    assert str(subfactorial(2*n)) == "subfactorial(2*n)"


def test_Function():
    f = Function('f')
    fx = f(x)
    w = WildFunction('w')
    assert str(f) == "f"
    assert str(fx) == "f(x)"
    assert str(w) == "w_"


def test_Geometry():
    assert sstr(Point(0, 0)) == 'Point2D(0, 0)'
    assert sstr(Circle(Point(0, 0), 3)) == 'Circle(Point2D(0, 0), 3)'
    assert sstr(Ellipse(Point(1, 2), 3, 4)) == 'Ellipse(Point2D(1, 2), 3, 4)'
    assert sstr(Triangle(Point(1, 1), Point(7, 8), Point(0, -1))) == \
        'Triangle(Point2D(1, 1), Point2D(7, 8), Point2D(0, -1))'
    assert sstr(Polygon(Point(5, 6), Point(-2, -3), Point(0, 0), Point(4, 7))) == \
        'Polygon(Point2D(5, 6), Point2D(-2, -3), Point2D(0, 0), Point2D(4, 7))'
    assert sstr(Triangle(Point(0, 0), Point(1, 0), Point(0, 1)), sympy_integers=True) == \
        'Triangle(Point2D(S(0), S(0)), Point2D(S(1), S(0)), Point2D(S(0), S(1)))'
    assert sstr(Ellipse(Point(1, 2), 3, 4), sympy_integers=True) == \
        'Ellipse(Point2D(S(1), S(2)), S(3), S(4))'


def test_GoldenRatio():
    assert str(GoldenRatio) == "GoldenRatio"


def test_Heaviside():
    assert str(Heaviside(x)) == str(Heaviside(x, S.Half)) == "Heaviside(x)"
    assert str(Heaviside(x, 1)) == "Heaviside(x, 1)"


def test_TribonacciConstant():
    assert str(TribonacciConstant) == "TribonacciConstant"


def test_ImaginaryUnit():
    assert str(I) == "I"


def test_Infinity():
    assert str(oo) == "oo"
    assert str(oo*I) == "oo*I"


def test_Integer():
    assert str(Integer(-1)) == "-1"
    assert str(Integer(1)) == "1"
    assert str(Integer(-3)) == "-3"
    assert str(Integer(0)) == "0"
    assert str(Integer(25)) == "25"


def test_Integral():
    assert str(Integral(sin(x), y)) == "Integral(sin(x), y)"
    assert str(Integral(sin(x), (y, 0, 1))) == "Integral(sin(x), (y, 0, 1))"


def test_Interval():
    n = (S.NegativeInfinity, 1, 2, S.Infinity)
    for i in range(len(n)):
        for j in range(i + 1, len(n)):
            for l in (True, False):
                for r in (True, False):
                    ival = Interval(n[i], n[j], l, r)
                    assert S(str(ival)) == ival


def test_AccumBounds():
    a = Symbol('a', real=True)
    assert str(AccumBounds(0, a)) == "AccumBounds(0, a)"
    assert str(AccumBounds(0, 1)) == "AccumBounds(0, 1)"


def test_Lambda():
    assert str(Lambda(d, d**2)) == "Lambda(_d, _d**2)"
    # issue 2908
    assert str(Lambda((), 1)) == "Lambda((), 1)"
    assert str(Lambda((), x)) == "Lambda((), x)"
    assert str(Lambda((x, y), x+y)) == "Lambda((x, y), x + y)"
    assert str(Lambda(((x, y),), x+y)) == "Lambda(((x, y),), x + y)"


def test_Limit():
    assert str(Limit(sin(x)/x, x, y)) == "Limit(sin(x)/x, x, y, dir='+')"
    assert str(Limit(1/x, x, 0)) == "Limit(1/x, x, 0, dir='+')"
    assert str(
        Limit(sin(x)/x, x, y, dir="-")) == "Limit(sin(x)/x, x, y, dir='-')"


def test_list():
    assert str([x]) == sstr([x]) == "[x]"
    assert str([x**2, x*y + 1]) == sstr([x**2, x*y + 1]) == "[x**2, x*y + 1]"
    assert str([x**2, [y + x]]) == sstr([x**2, [y + x]]) == "[x**2, [x + y]]"


def test_Matrix_str():
    M = Matrix([[x**+1, 1], [y, x + y]])
    assert str(M) == "Matrix([[x, 1], [y, x + y]])"
    assert sstr(M) == "Matrix([\n[x,     1],\n[y, x + y]])"
    M = Matrix([[1]])
    assert str(M) == sstr(M) == "Matrix([[1]])"
    M = Matrix([[1, 2]])
    assert str(M) == sstr(M) ==  "Matrix([[1, 2]])"
    M = Matrix()
    assert str(M) == sstr(M) == "Matrix(0, 0, [])"
    M = Matrix(0, 1, lambda i, j: 0)
    assert str(M) == sstr(M) == "Matrix(0, 1, [])"


def test_Mul():
    assert str(x/y) == "x/y"
    assert str(y/x) == "y/x"
    assert str(x/y/z) == "x/(y*z)"
    assert str((x + 1)/(y + 2)) == "(x + 1)/(y + 2)"
    assert str(2*x/3) == '2*x/3'
    assert str(-2*x/3) == '-2*x/3'
    assert str(-1.0*x) == '-1.0*x'
    assert str(1.0*x) == '1.0*x'
    assert str(Mul(0, 1, evaluate=False)) == '0*1'
    assert str(Mul(1, 0, evaluate=False)) == '1*0'
    assert str(Mul(1, 1, evaluate=False)) == '1*1'
    assert str(Mul(1, 1, 1, evaluate=False)) == '1*1*1'
    assert str(Mul(1, 2, evaluate=False)) == '1*2'
    assert str(Mul(1, S.Half, evaluate=False)) == '1*(1/2)'
    assert str(Mul(1, 1, S.Half, evaluate=False)) == '1*1*(1/2)'
    assert str(Mul(1, 1, 2, 3, x, evaluate=False)) == '1*1*2*3*x'
    assert str(Mul(1, -1, evaluate=False)) == '1*(-1)'
    assert str(Mul(-1, 1, evaluate=False)) == '-1*1'
    assert str(Mul(4, 3, 2, 1, 0, y, x, evaluate=False)) == '4*3*2*1*0*y*x'
    assert str(Mul(4, 3, 2, 1+z, 0, y, x, evaluate=False)) == '4*3*2*(z + 1)*0*y*x'
    assert str(Mul(Rational(2, 3), Rational(5, 7), evaluate=False)) == '(2/3)*(5/7)'
    # For issue 14160
    assert str(Mul(-2, x, Pow(Mul(y,y,evaluate=False), -1, evaluate=False),
                                                evaluate=False)) == '-2*x/(y*y)'
    # issue 21537
    assert str(Mul(x, Pow(1/y, -1, evaluate=False), evaluate=False)) == 'x/(1/y)'

    # Issue 24108
    from sympy.core.parameters import evaluate
    with evaluate(False):
        assert str(Mul(Pow(Integer(2), Integer(-1)), Add(Integer(-1), Mul(Integer(-1), Integer(1))))) == "(-1 - 1*1)/2"

    class CustomClass1(Expr):
        is_commutative = True

    class CustomClass2(Expr):
        is_commutative = True
    cc1 = CustomClass1()
    cc2 = CustomClass2()
    assert str(Rational(2)*cc1) == '2*CustomClass1()'
    assert str(cc1*Rational(2)) == '2*CustomClass1()'
    assert str(cc1*Float("1.5")) == '1.5*CustomClass1()'
    assert str(cc2*Rational(2)) == '2*CustomClass2()'
    assert str(cc2*Rational(2)*cc1) == '2*CustomClass1()*CustomClass2()'
    assert str(cc1*Rational(2)*cc2) == '2*CustomClass1()*CustomClass2()'


def test_NaN():
    assert str(nan) == "nan"


def test_NegativeInfinity():
    assert str(-oo) == "-oo"

def test_Order():
    assert str(O(x)) == "O(x)"
    assert str(O(x**2)) == "O(x**2)"
    assert str(O(x*y)) == "O(x*y, x, y)"
    assert str(O(x, x)) == "O(x)"
    assert str(O(x, (x, 0))) == "O(x)"
    assert str(O(x, (x, oo))) == "O(x, (x, oo))"
    assert str(O(x, x, y)) == "O(x, x, y)"
    assert str(O(x, x, y)) == "O(x, x, y)"
    assert str(O(x, (x, oo), (y, oo))) == "O(x, (x, oo), (y, oo))"


def test_Permutation_Cycle():
    from sympy.combinatorics import Permutation, Cycle

    # general principle: economically, canonically show all moved elements
    # and the size of the permutation.

    for p, s in [
        (Cycle(),
        '()'),
        (Cycle(2),
        '(2)'),
        (Cycle(2, 1),
        '(1 2)'),
        (Cycle(1, 2)(5)(6, 7)(10),
        '(1 2)(6 7)(10)'),
        (Cycle(3, 4)(1, 2)(3, 4),
        '(1 2)(4)'),
    ]:
        assert sstr(p) == s

    for p, s in [
        (Permutation([]),
        'Permutation([])'),
        (Permutation([], size=1),
        'Permutation([0])'),
        (Permutation([], size=2),
        'Permutation([0, 1])'),
        (Permutation([], size=10),
        'Permutation([], size=10)'),
        (Permutation([1, 0, 2]),
        'Permutation([1, 0, 2])'),
        (Permutation([1, 0, 2, 3, 4, 5]),
        'Permutation([1, 0], size=6)'),
        (Permutation([1, 0, 2, 3, 4, 5], size=10),
        'Permutation([1, 0], size=10)'),
    ]:
        assert sstr(p, perm_cyclic=False) == s

    for p, s in [
        (Permutation([]),
        '()'),
        (Permutation([], size=1),
        '(0)'),
        (Permutation([], size=2),
        '(1)'),
        (Permutation([], size=10),
        '(9)'),
        (Permutation([1, 0, 2]),
        '(2)(0 1)'),
        (Permutation([1, 0, 2, 3, 4, 5]),
        '(5)(0 1)'),
        (Permutation([1, 0, 2, 3, 4, 5], size=10),
        '(9)(0 1)'),
        (Permutation([0, 1, 3, 2, 4, 5], size=10),
        '(9)(2 3)'),
    ]:
        assert sstr(p) == s


    with warns_deprecated_sympy():
        old_print_cyclic = Permutation.print_cyclic
        Permutation.print_cyclic = False
        assert sstr(Permutation([1, 0, 2])) == 'Permutation([1, 0, 2])'
        Permutation.print_cyclic = old_print_cyclic

def test_Pi():
    assert str(pi) == "pi"


def test_Poly():
    assert str(Poly(0, x)) == "Poly(0, x, domain='ZZ')"
    assert str(Poly(1, x)) == "Poly(1, x, domain='ZZ')"
    assert str(Poly(x, x)) == "Poly(x, x, domain='ZZ')"

    assert str(Poly(2*x + 1, x)) == "Poly(2*x + 1, x, domain='ZZ')"
    assert str(Poly(2*x - 1, x)) == "Poly(2*x - 1, x, domain='ZZ')"

    assert str(Poly(-1, x)) == "Poly(-1, x, domain='ZZ')"
    assert str(Poly(-x, x)) == "Poly(-x, x, domain='ZZ')"

    assert str(Poly(-2*x + 1, x)) == "Poly(-2*x + 1, x, domain='ZZ')"
    assert str(Poly(-2*x - 1, x)) == "Poly(-2*x - 1, x, domain='ZZ')"

    assert str(Poly(x - 1, x)) == "Poly(x - 1, x, domain='ZZ')"
    assert str(Poly(2*x + x**5, x)) == "Poly(x**5 + 2*x, x, domain='ZZ')"

    assert str(Poly(3**(2*x), 3**x)) == "Poly((3**x)**2, 3**x, domain='ZZ')"
    assert str(Poly((x**2)**x)) == "Poly(((x**2)**x), (x**2)**x, domain='ZZ')"

    assert str(Poly((x + y)**3, (x + y), expand=False)
                ) == "Poly((x + y)**3, x + y, domain='ZZ')"
    assert str(Poly((x - 1)**2, (x - 1), expand=False)
                ) == "Poly((x - 1)**2, x - 1, domain='ZZ')"

    assert str(
        Poly(x**2 + 1 + y, x)) == "Poly(x**2 + y + 1, x, domain='ZZ[y]')"
    assert str(
        Poly(x**2 - 1 + y, x)) == "Poly(x**2 + y - 1, x, domain='ZZ[y]')"

    assert str(Poly(x**2 + I*x, x)) == "Poly(x**2 + I*x, x, domain='ZZ_I')"
    assert str(Poly(x**2 - I*x, x)) == "Poly(x**2 - I*x, x, domain='ZZ_I')"

    assert str(Poly(-x*y*z + x*y - 1, x, y, z)
               ) == "Poly(-x*y*z + x*y - 1, x, y, z, domain='ZZ')"
    assert str(Poly(-w*x**21*y**7*z + (1 + w)*z**3 - 2*x*z + 1, x, y, z)) == \
        "Poly(-w*x**21*y**7*z - 2*x*z + (w + 1)*z**3 + 1, x, y, z, domain='ZZ[w]')"

    assert str(Poly(x**2 + 1, x, modulus=2)) == "Poly(x**2 + 1, x, modulus=2)"
    assert str(Poly(2*x**2 + 3*x + 4, x, modulus=17)) == "Poly(2*x**2 + 3*x + 4, x, modulus=17)"


def test_PolyRing():
    assert str(ring("x", ZZ, lex)[0]) == "Polynomial ring in x over ZZ with lex order"
    assert str(ring("x,y", QQ, grlex)[0]) == "Polynomial ring in x, y over QQ with grlex order"
    assert str(ring("x,y,z", ZZ["t"], lex)[0]) == "Polynomial ring in x, y, z over ZZ[t] with lex order"


def test_FracField():
    assert str(field("x", ZZ, lex)[0]) == "Rational function field in x over ZZ with lex order"
    assert str(field("x,y", QQ, grlex)[0]) == "Rational function field in x, y over QQ with grlex order"
    assert str(field("x,y,z", ZZ["t"], lex)[0]) == "Rational function field in x, y, z over ZZ[t] with lex order"


def test_PolyElement():
    Ruv, u,v = ring("u,v", ZZ)
    Rxyz, x,y,z = ring("x,y,z", Ruv)
    Rx_zzi, xz = ring("x", ZZ_I)

    assert str(x - x) == "0"
    assert str(x - 1) == "x - 1"
    assert str(x + 1) == "x + 1"
    assert str(x**2) == "x**2"

    assert str((u**2 + 3*u*v + 1)*x**2*y + u + 1) == "(u**2 + 3*u*v + 1)*x**2*y + u + 1"
    assert str((u**2 + 3*u*v + 1)*x**2*y + (u + 1)*x) == "(u**2 + 3*u*v + 1)*x**2*y + (u + 1)*x"
    assert str((u**2 + 3*u*v + 1)*x**2*y + (u + 1)*x + 1) == "(u**2 + 3*u*v + 1)*x**2*y + (u + 1)*x + 1"
    assert str((-u**2 + 3*u*v - 1)*x**2*y - (u + 1)*x - 1) == "-(u**2 - 3*u*v + 1)*x**2*y - (u + 1)*x - 1"

    assert str(-(v**2 + v + 1)*x + 3*u*v + 1) == "-(v**2 + v + 1)*x + 3*u*v + 1"
    assert str(-(v**2 + v + 1)*x - 3*u*v + 1) == "-(v**2 + v + 1)*x - 3*u*v + 1"

    assert str((1+I)*xz + 2) == "(1 + 1*I)*x + (2 + 0*I)"


def test_FracElement():
    Fuv, u,v = field("u,v", ZZ)
    Fxyzt, x,y,z,t = field("x,y,z,t", Fuv)
    Rx_zzi, xz = field("x", QQ_I)
    i = QQ_I(0, 1)

    assert str(x - x) == "0"
    assert str(x - 1) == "x - 1"
    assert str(x + 1) == "x + 1"

    assert str(x/3) == "x/3"
    assert str(x/z) == "x/z"
    assert str(x*y/z) == "x*y/z"
    assert str(x/(z*t)) == "x/(z*t)"
    assert str(x*y/(z*t)) == "x*y/(z*t)"

    assert str((x - 1)/y) == "(x - 1)/y"
    assert str((x + 1)/y) == "(x + 1)/y"
    assert str((-x - 1)/y) == "(-x - 1)/y"
    assert str((x + 1)/(y*z)) == "(x + 1)/(y*z)"
    assert str(-y/(x + 1)) == "-y/(x + 1)"
    assert str(y*z/(x + 1)) == "y*z/(x + 1)"

    assert str(((u + 1)*x*y + 1)/((v - 1)*z - 1)) == "((u + 1)*x*y + 1)/((v - 1)*z - 1)"
    assert str(((u + 1)*x*y + 1)/((v - 1)*z - t*u*v - 1)) == "((u + 1)*x*y + 1)/((v - 1)*z - u*v*t - 1)"

    assert str((1+i)/xz) == "(1 + 1*I)/x"
    assert str(((1+i)*xz - i)/xz) == "((1 + 1*I)*x + (0 + -1*I))/x"


def test_GaussianInteger():
    assert str(ZZ_I(1, 0)) == "1"
    assert str(ZZ_I(-1, 0)) == "-1"
    assert str(ZZ_I(0, 1)) == "I"
    assert str(ZZ_I(0, -1)) == "-I"
    assert str(ZZ_I(0, 2)) == "2*I"
    assert str(ZZ_I(0, -2)) == "-2*I"
    assert str(ZZ_I(1, 1)) == "1 + I"
    assert str(ZZ_I(-1, -1)) == "-1 - I"
    assert str(ZZ_I(-1, -2)) == "-1 - 2*I"


def test_GaussianRational():
    assert str(QQ_I(1, 0)) == "1"
    assert str(QQ_I(QQ(2, 3), 0)) == "2/3"
    assert str(QQ_I(0, QQ(2, 3))) == "2*I/3"
    assert str(QQ_I(QQ(1, 2), QQ(-2, 3))) == "1/2 - 2*I/3"


def test_Pow():
    assert str(x**-1) == "1/x"
    assert str(x**-2) == "x**(-2)"
    assert str(x**2) == "x**2"
    assert str((x + y)**-1) == "1/(x + y)"
    assert str((x + y)**-2) == "(x + y)**(-2)"
    assert str((x + y)**2) == "(x + y)**2"
    assert str((x + y)**(1 + x)) == "(x + y)**(x + 1)"
    assert str(x**Rational(1, 3)) == "x**(1/3)"
    assert str(1/x**Rational(1, 3)) == "x**(-1/3)"
    assert str(sqrt(sqrt(x))) == "x**(1/4)"
    # not the same as x**-1
    assert str(x**-1.0) == 'x**(-1.0)'
    # see issue #2860
    assert str(Pow(S(2), -1.0, evaluate=False)) == '2**(-1.0)'


def test_sqrt():
    assert str(sqrt(x)) == "sqrt(x)"
    assert str(sqrt(x**2)) == "sqrt(x**2)"
    assert str(1/sqrt(x)) == "1/sqrt(x)"
    assert str(1/sqrt(x**2)) == "1/sqrt(x**2)"
    assert str(y/sqrt(x)) == "y/sqrt(x)"
    assert str(x**0.5) == "x**0.5"
    assert str(1/x**0.5) == "x**(-0.5)"


def test_Rational():
    n1 = Rational(1, 4)
    n2 = Rational(1, 3)
    n3 = Rational(2, 4)
    n4 = Rational(2, -4)
    n5 = Rational(0)
    n7 = Rational(3)
    n8 = Rational(-3)
    assert str(n1*n2) == "1/12"
    assert str(n1*n2) == "1/12"
    assert str(n3) == "1/2"
    assert str(n1*n3) == "1/8"
    assert str(n1 + n3) == "3/4"
    assert str(n1 + n2) == "7/12"
    assert str(n1 + n4) == "-1/4"
    assert str(n4*n4) == "1/4"
    assert str(n4 + n2) == "-1/6"
    assert str(n4 + n5) == "-1/2"
    assert str(n4*n5) == "0"
    assert str(n3 + n4) == "0"
    assert str(n1**n7) == "1/64"
    assert str(n2**n7) == "1/27"
    assert str(n2**n8) == "27"
    assert str(n7**n8) == "1/27"
    assert str(Rational("-25")) == "-25"
    assert str(Rational("1.25")) == "5/4"
    assert str(Rational("-2.6e-2")) == "-13/500"
    assert str(S("25/7")) == "25/7"
    assert str(S("-123/569")) == "-123/569"
    assert str(S("0.1[23]", rational=1)) == "61/495"
    assert str(S("5.1[666]", rational=1)) == "31/6"
    assert str(S("-5.1[666]", rational=1)) == "-31/6"
    assert str(S("0.[9]", rational=1)) == "1"
    assert str(S("-0.[9]", rational=1)) == "-1"

    assert str(sqrt(Rational(1, 4))) == "1/2"
    assert str(sqrt(Rational(1, 36))) == "1/6"

    assert str((123**25) ** Rational(1, 25)) == "123"
    assert str((123**25 + 1)**Rational(1, 25)) != "123"
    assert str((123**25 - 1)**Rational(1, 25)) != "123"
    assert str((123**25 - 1)**Rational(1, 25)) != "122"

    assert str(sqrt(Rational(81, 36))**3) == "27/8"
    assert str(1/sqrt(Rational(81, 36))**3) == "8/27"

    assert str(sqrt(-4)) == str(2*I)
    assert str(2**Rational(1, 10**10)) == "2**(1/10000000000)"

    assert sstr(Rational(2, 3), sympy_integers=True) == "S(2)/3"
    x = Symbol("x")
    assert sstr(x**Rational(2, 3), sympy_integers=True) == "x**(S(2)/3)"
    assert sstr(Eq(x, Rational(2, 3)), sympy_integers=True) == "Eq(x, S(2)/3)"
    assert sstr(Limit(x, x, Rational(7, 2)), sympy_integers=True) == \
        "Limit(x, x, S(7)/2, dir='+')"


def test_Float():
    # NOTE dps is the whole number of decimal digits
    assert str(Float('1.23', dps=1 + 2)) == '1.23'
    assert str(Float('1.23456789', dps=1 + 8)) == '1.23456789'
    assert str(
        Float('1.234567890123456789', dps=1 + 18)) == '1.234567890123456789'
    assert str(pi.evalf(1 + 2)) == '3.14'
    assert str(pi.evalf(1 + 14)) == '3.14159265358979'
    assert str(pi.evalf(1 + 64)) == ('3.141592653589793238462643383279'
                                     '5028841971693993751058209749445923')
    assert str(pi.round(-1)) == '0.0'
    assert str((pi**400 - (pi**400).round(1)).n(2)) == '-0.e+88'
    assert sstr(Float("100"), full_prec=False, min=-2, max=2) == '1.0e+2'
    assert sstr(Float("100"), full_prec=False, min=-2, max=3) == '100.0'
    assert sstr(Float("0.1"), full_prec=False, min=-2, max=3) == '0.1'
    assert sstr(Float("0.099"), min=-2, max=3) == '9.90000000000000e-2'


def test_Relational():
    assert str(Rel(x, y, "<")) == "x < y"
    assert str(Rel(x + y, y, "==")) == "Eq(x + y, y)"
    assert str(Rel(x, y, "!=")) == "Ne(x, y)"
    assert str(Eq(x, 1) | Eq(x, 2)) == "Eq(x, 1) | Eq(x, 2)"
    assert str(Ne(x, 1) & Ne(x, 2)) == "Ne(x, 1) & Ne(x, 2)"


def test_AppliedBinaryRelation():
    assert str(Q.eq(x, y)) == "Q.eq(x, y)"
    assert str(Q.ne(x, y)) == "Q.ne(x, y)"


def test_CRootOf():
    assert str(rootof(x**5 + 2*x - 1, 0)) == "CRootOf(x**5 + 2*x - 1, 0)"


def test_RootSum():
    f = x**5 + 2*x - 1

    assert str(
        RootSum(f, Lambda(z, z), auto=False)) == "RootSum(x**5 + 2*x - 1)"
    assert str(RootSum(f, Lambda(
        z, z**2), auto=False)) == "RootSum(x**5 + 2*x - 1, Lambda(z, z**2))"


def test_GroebnerBasis():
    assert str(groebner(
        [], x, y)) == "GroebnerBasis([], x, y, domain='ZZ', order='lex')"

    F = [x**2 - 3*y - x + 1, y**2 - 2*x + y - 1]

    assert str(groebner(F, order='grlex')) == \
        "GroebnerBasis([x**2 - x - 3*y + 1, y**2 - 2*x + y - 1], x, y, domain='ZZ', order='grlex')"
    assert str(groebner(F, order='lex')) == \
        "GroebnerBasis([2*x - y**2 - y + 1, y**4 + 2*y**3 - 3*y**2 - 16*y + 7], x, y, domain='ZZ', order='lex')"

def test_set():
    assert sstr(set()) == 'set()'
    assert sstr(frozenset()) == 'frozenset()'

    assert sstr({1}) == '{1}'
    assert sstr(frozenset([1])) == 'frozenset({1})'
    assert sstr({1, 2, 3}) == '{1, 2, 3}'
    assert sstr(frozenset([1, 2, 3])) == 'frozenset({1, 2, 3})'

    assert sstr(
        {1, x, x**2, x**3, x**4}) == '{1, x, x**2, x**3, x**4}'
    assert sstr(
        frozenset([1, x, x**2, x**3, x**4])) == 'frozenset({1, x, x**2, x**3, x**4})'


def test_SparseMatrix():
    M = SparseMatrix([[x**+1, 1], [y, x + y]])
    assert str(M) == "Matrix([[x, 1], [y, x + y]])"
    assert sstr(M) == "Matrix([\n[x,     1],\n[y, x + y]])"


def test_Sum():
    assert str(summation(cos(3*z), (z, x, y))) == "Sum(cos(3*z), (z, x, y))"
    assert str(Sum(x*y**2, (x, -2, 2), (y, -5, 5))) == \
        "Sum(x*y**2, (x, -2, 2), (y, -5, 5))"


def test_Symbol():
    assert str(y) == "y"
    assert str(x) == "x"
    e = x
    assert str(e) == "x"


def test_tuple():
    assert str((x,)) == sstr((x,)) == "(x,)"
    assert str((x + y, 1 + x)) == sstr((x + y, 1 + x)) == "(x + y, x + 1)"
    assert str((x + y, (
        1 + x, x**2))) == sstr((x + y, (1 + x, x**2))) == "(x + y, (x + 1, x**2))"


def test_Series_str():
    tf1 = TransferFunction(x*y**2 - z, y**3 - t**3, y)
    tf2 = TransferFunction(x - y, x + y, y)
    tf3 = TransferFunction(t*x**2 - t**w*x + w, t - y, y)
    assert str(Series(tf1, tf2)) == \
        "Series(TransferFunction(x*y**2 - z, -t**3 + y**3, y), TransferFunction(x - y, x + y, y))"
    assert str(Series(tf1, tf2, tf3)) == \
        "Series(TransferFunction(x*y**2 - z, -t**3 + y**3, y), TransferFunction(x - y, x + y, y), TransferFunction(t*x**2 - t**w*x + w, t - y, y))"
    assert str(Series(-tf2, tf1)) == \
        "Series(TransferFunction(-x + y, x + y, y), TransferFunction(x*y**2 - z, -t**3 + y**3, y))"


def test_MIMOSeries_str():
    tf1 = TransferFunction(x*y**2 - z, y**3 - t**3, y)
    tf2 = TransferFunction(x - y, x + y, y)
    tfm_1 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])
    tfm_2 = TransferFunctionMatrix([[tf2, tf1], [tf1, tf2]])
    assert str(MIMOSeries(tfm_1, tfm_2)) == \
        "MIMOSeries(TransferFunctionMatrix(((TransferFunction(x*y**2 - z, -t**3 + y**3, y), TransferFunction(x - y, x + y, y)), "\
            "(TransferFunction(x - y, x + y, y), TransferFunction(x*y**2 - z, -t**3 + y**3, y)))), "\
                "TransferFunctionMatrix(((TransferFunction(x - y, x + y, y), TransferFunction(x*y**2 - z, -t**3 + y**3, y)), "\
                    "(TransferFunction(x*y**2 - z, -t**3 + y**3, y), TransferFunction(x - y, x + y, y)))))"


def test_TransferFunction_str():
    tf1 = TransferFunction(x - 1, x + 1, x)
    assert str(tf1) == "TransferFunction(x - 1, x + 1, x)"
    tf2 = TransferFunction(x + 1, 2 - y, x)
    assert str(tf2) == "TransferFunction(x + 1, 2 - y, x)"
    tf3 = TransferFunction(y, y**2 + 2*y + 3, y)
    assert str(tf3) == "TransferFunction(y, y**2 + 2*y + 3, y)"


def test_Parallel_str():
    tf1 = TransferFunction(x*y**2 - z, y**3 - t**3, y)
    tf2 = TransferFunction(x - y, x + y, y)
    tf3 = TransferFunction(t*x**2 - t**w*x + w, t - y, y)
    assert str(Parallel(tf1, tf2)) == \
        "Parallel(TransferFunction(x*y**2 - z, -t**3 + y**3, y), TransferFunction(x - y, x + y, y))"
    assert str(Parallel(tf1, tf2, tf3)) == \
        "Parallel(TransferFunction(x*y**2 - z, -t**3 + y**3, y), TransferFunction(x - y, x + y, y), TransferFunction(t*x**2 - t**w*x + w, t - y, y))"
    assert str(Parallel(-tf2, tf1)) == \
        "Parallel(TransferFunction(-x + y, x + y, y), TransferFunction(x*y**2 - z, -t**3 + y**3, y))"


def test_MIMOParallel_str():
    tf1 = TransferFunction(x*y**2 - z, y**3 - t**3, y)
    tf2 = TransferFunction(x - y, x + y, y)
    tfm_1 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])
    tfm_2 = TransferFunctionMatrix([[tf2, tf1], [tf1, tf2]])
    assert str(MIMOParallel(tfm_1, tfm_2)) == \
        "MIMOParallel(TransferFunctionMatrix(((TransferFunction(x*y**2 - z, -t**3 + y**3, y), TransferFunction(x - y, x + y, y)), "\
            "(TransferFunction(x - y, x + y, y), TransferFunction(x*y**2 - z, -t**3 + y**3, y)))), "\
                "TransferFunctionMatrix(((TransferFunction(x - y, x + y, y), TransferFunction(x*y**2 - z, -t**3 + y**3, y)), "\
                    "(TransferFunction(x*y**2 - z, -t**3 + y**3, y), TransferFunction(x - y, x + y, y)))))"


def test_Feedback_str():
    tf1 = TransferFunction(x*y**2 - z, y**3 - t**3, y)
    tf2 = TransferFunction(x - y, x + y, y)
    tf3 = TransferFunction(t*x**2 - t**w*x + w, t - y, y)
    assert str(Feedback(tf1*tf2, tf3)) == \
        "Feedback(Series(TransferFunction(x*y**2 - z, -t**3 + y**3, y), TransferFunction(x - y, x + y, y)), " \
        "TransferFunction(t*x**2 - t**w*x + w, t - y, y), -1)"
    assert str(Feedback(tf1, TransferFunction(1, 1, y), 1)) == \
        "Feedback(TransferFunction(x*y**2 - z, -t**3 + y**3, y), TransferFunction(1, 1, y), 1)"


def test_MIMOFeedback_str():
    tf1 = TransferFunction(x**2 - y**3, y - z, x)
    tf2 = TransferFunction(y - x, z + y, x)
    tfm_1 = TransferFunctionMatrix([[tf2, tf1], [tf1, tf2]])
    tfm_2 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])
    assert (str(MIMOFeedback(tfm_1, tfm_2)) \
            == "MIMOFeedback(TransferFunctionMatrix(((TransferFunction(-x + y, y + z, x), TransferFunction(x**2 - y**3, y - z, x))," \
            " (TransferFunction(x**2 - y**3, y - z, x), TransferFunction(-x + y, y + z, x)))), " \
            "TransferFunctionMatrix(((TransferFunction(x**2 - y**3, y - z, x), " \
            "TransferFunction(-x + y, y + z, x)), (TransferFunction(-x + y, y + z, x), TransferFunction(x**2 - y**3, y - z, x)))), -1)")
    assert (str(MIMOFeedback(tfm_1, tfm_2, 1)) \
            == "MIMOFeedback(TransferFunctionMatrix(((TransferFunction(-x + y, y + z, x), TransferFunction(x**2 - y**3, y - z, x)), " \
            "(TransferFunction(x**2 - y**3, y - z, x), TransferFunction(-x + y, y + z, x)))), " \
            "TransferFunctionMatrix(((TransferFunction(x**2 - y**3, y - z, x), TransferFunction(-x + y, y + z, x)), "\
            "(TransferFunction(-x + y, y + z, x), TransferFunction(x**2 - y**3, y - z, x)))), 1)")


def test_TransferFunctionMatrix_str():
    tf1 = TransferFunction(x*y**2 - z, y**3 - t**3, y)
    tf2 = TransferFunction(x - y, x + y, y)
    tf3 = TransferFunction(t*x**2 - t**w*x + w, t - y, y)
    assert str(TransferFunctionMatrix([[tf1], [tf2]])) == \
        "TransferFunctionMatrix(((TransferFunction(x*y**2 - z, -t**3 + y**3, y),), (TransferFunction(x - y, x + y, y),)))"
    assert str(TransferFunctionMatrix([[tf1, tf2], [tf3, tf2]])) == \
        "TransferFunctionMatrix(((TransferFunction(x*y**2 - z, -t**3 + y**3, y), TransferFunction(x - y, x + y, y)), (TransferFunction(t*x**2 - t**w*x + w, t - y, y), TransferFunction(x - y, x + y, y))))"


def test_Quaternion_str_printer():
    q = Quaternion(x, y, z, t)
    assert str(q) == "x + y*i + z*j + t*k"
    q = Quaternion(x,y,z,x*t)
    assert str(q) == "x + y*i + z*j + t*x*k"
    q = Quaternion(x,y,z,x+t)
    assert str(q) == "x + y*i + z*j + (t + x)*k"


def test_Quantity_str():
    assert sstr(second, abbrev=True) == "s"
    assert sstr(joule, abbrev=True) == "J"
    assert str(second) == "second"
    assert str(joule) == "joule"


def test_wild_str():
    # Check expressions containing Wild not causing infinite recursion
    w = Wild('x')
    assert str(w + 1) == 'x_ + 1'
    assert str(exp(2**w) + 5) == 'exp(2**x_) + 5'
    assert str(3*w + 1) == '3*x_ + 1'
    assert str(1/w + 1) == '1 + 1/x_'
    assert str(w**2 + 1) == 'x_**2 + 1'
    assert str(1/(1 - w)) == '1/(1 - x_)'


def test_wild_matchpy():
    from sympy.utilities.matchpy_connector import WildDot, WildPlus, WildStar

    matchpy = import_module("matchpy")

    if matchpy is None:
        return

    wd = WildDot('w_')
    wp = WildPlus('w__')
    ws = WildStar('w___')

    assert str(wd) == 'w_'
    assert str(wp) == 'w__'
    assert str(ws) == 'w___'

    assert str(wp/ws + 2**wd) == '2**w_ + w__/w___'
    assert str(sin(wd)*cos(wp)*sqrt(ws)) == 'sqrt(w___)*sin(w_)*cos(w__)'


def test_zeta():
    assert str(zeta(3)) == "zeta(3)"


def test_issue_3101():
    e = x - y
    a = str(e)
    b = str(e)
    assert a == b


def test_issue_3103():
    e = -2*sqrt(x) - y/sqrt(x)/2
    assert str(e) not in ["(-2)*x**1/2(-1/2)*x**(-1/2)*y",
            "-2*x**1/2(-1/2)*x**(-1/2)*y", "-2*x**1/2-1/2*x**-1/2*w"]
    assert str(e) == "-2*sqrt(x) - y/(2*sqrt(x))"


def test_issue_4021():
    e = Integral(x, x) + 1
    assert str(e) == 'Integral(x, x) + 1'


def test_sstrrepr():
    assert sstr('abc') == 'abc'
    assert sstrrepr('abc') == "'abc'"

    e = ['a', 'b', 'c', x]
    assert sstr(e) == "[a, b, c, x]"
    assert sstrrepr(e) == "['a', 'b', 'c', x]"


def test_infinity():
    assert sstr(oo*I) == "oo*I"


def test_full_prec():
    assert sstr(S("0.3"), full_prec=True) == "0.300000000000000"
    assert sstr(S("0.3"), full_prec="auto") == "0.300000000000000"
    assert sstr(S("0.3"), full_prec=False) == "0.3"
    assert sstr(S("0.3")*x, full_prec=True) in [
        "0.300000000000000*x",
        "x*0.300000000000000"
    ]
    assert sstr(S("0.3")*x, full_prec="auto") in [
        "0.3*x",
        "x*0.3"
    ]
    assert sstr(S("0.3")*x, full_prec=False) in [
        "0.3*x",
        "x*0.3"
    ]


def test_noncommutative():
    A, B, C = symbols('A,B,C', commutative=False)

    assert sstr(A*B*C**-1) == "A*B*C**(-1)"
    assert sstr(C**-1*A*B) == "C**(-1)*A*B"
    assert sstr(A*C**-1*B) == "A*C**(-1)*B"
    assert sstr(sqrt(A)) == "sqrt(A)"
    assert sstr(1/sqrt(A)) == "A**(-1/2)"


def test_empty_printer():
    str_printer = StrPrinter()
    assert str_printer.emptyPrinter("foo") == "foo"
    assert str_printer.emptyPrinter(x*y) == "x*y"
    assert str_printer.emptyPrinter(32) == "32"

def test_decimal_printer():
    dec_printer = StrPrinter(settings={"dps":3})
    f = Function('f')
    assert dec_printer.doprint(f(1.329294)) == "f(1.33)"


def test_settings():
    raises(TypeError, lambda: sstr(S(4), method="garbage"))


def test_RandomDomain():
    from sympy.stats import Normal, Die, Exponential, pspace, where
    X = Normal('x1', 0, 1)
    assert str(where(X > 0)) == "Domain: (0 < x1) & (x1 < oo)"

    D = Die('d1', 6)
    assert str(where(D > 4)) == "Domain: Eq(d1, 5) | Eq(d1, 6)"

    A = Exponential('a', 1)
    B = Exponential('b', 1)
    assert str(pspace(Tuple(A, B)).domain) == "Domain: (0 <= a) & (0 <= b) & (a < oo) & (b < oo)"


def test_FiniteSet():
    assert str(FiniteSet(*range(1, 51))) == (
        '{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,'
        ' 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,'
        ' 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50}'
    )
    assert str(FiniteSet(*range(1, 6))) == '{1, 2, 3, 4, 5}'
    assert str(FiniteSet(*[x*y, x**2])) == '{x**2, x*y}'
    assert str(FiniteSet(FiniteSet(FiniteSet(x, y), 5), FiniteSet(x,y), 5)
               ) == 'FiniteSet(5, FiniteSet(5, {x, y}), {x, y})'


def test_Partition():
    assert str(Partition(FiniteSet(x, y), {z})) == 'Partition({z}, {x, y})'

def test_UniversalSet():
    assert str(S.UniversalSet) == 'UniversalSet'


def test_PrettyPoly():
    F = QQ.frac_field(x, y)
    R = QQ[x, y]
    assert sstr(F.convert(x/(x + y))) == sstr(x/(x + y))
    assert sstr(R.convert(x + y)) == sstr(x + y)


def test_categories():
    from sympy.categories import (Object, NamedMorphism,
        IdentityMorphism, Category)

    A = Object("A")
    B = Object("B")

    f = NamedMorphism(A, B, "f")
    id_A = IdentityMorphism(A)

    K = Category("K")

    assert str(A) == 'Object("A")'
    assert str(f) == 'NamedMorphism(Object("A"), Object("B"), "f")'
    assert str(id_A) == 'IdentityMorphism(Object("A"))'

    assert str(K) == 'Category("K")'


def test_Tr():
    A, B = symbols('A B', commutative=False)
    t = Tr(A*B)
    assert str(t) == 'Tr(A*B)'


def test_issue_6387():
    assert str(factor(-3.0*z + 3)) == '-3.0*(1.0*z - 1.0)'


def test_MatMul_MatAdd():
    X, Y = MatrixSymbol("X", 2, 2), MatrixSymbol("Y", 2, 2)
    assert str(2*(X + Y)) == "2*X + 2*Y"

    assert str(I*X) == "I*X"
    assert str(-I*X) == "-I*X"
    assert str((1 + I)*X) == '(1 + I)*X'
    assert str(-(1 + I)*X) == '(-1 - I)*X'
    assert str(MatAdd(MatAdd(X, Y), MatAdd(X, Y))) == '(X + Y) + (X + Y)'


def test_MatrixSlice():
    n = Symbol('n', integer=True)
    X = MatrixSymbol('X', n, n)
    Y = MatrixSymbol('Y', 10, 10)
    Z = MatrixSymbol('Z', 10, 10)

    assert str(MatrixSlice(X, (None, None, None), (None, None, None))) == 'X[:, :]'
    assert str(X[x:x + 1, y:y + 1]) == 'X[x:x + 1, y:y + 1]'
    assert str(X[x:x + 1:2, y:y + 1:2]) == 'X[x:x + 1:2, y:y + 1:2]'
    assert str(X[:x, y:]) == 'X[:x, y:]'
    assert str(X[:x, y:]) == 'X[:x, y:]'
    assert str(X[x:, :y]) == 'X[x:, :y]'
    assert str(X[x:y, z:w]) == 'X[x:y, z:w]'
    assert str(X[x:y:t, w:t:x]) == 'X[x:y:t, w:t:x]'
    assert str(X[x::y, t::w]) == 'X[x::y, t::w]'
    assert str(X[:x:y, :t:w]) == 'X[:x:y, :t:w]'
    assert str(X[::x, ::y]) == 'X[::x, ::y]'
    assert str(MatrixSlice(X, (0, None, None), (0, None, None))) == 'X[:, :]'
    assert str(MatrixSlice(X, (None, n, None), (None, n, None))) == 'X[:, :]'
    assert str(MatrixSlice(X, (0, n, None), (0, n, None))) == 'X[:, :]'
    assert str(MatrixSlice(X, (0, n, 2), (0, n, 2))) == 'X[::2, ::2]'
    assert str(X[1:2:3, 4:5:6]) == 'X[1:2:3, 4:5:6]'
    assert str(X[1:3:5, 4:6:8]) == 'X[1:3:5, 4:6:8]'
    assert str(X[1:10:2]) == 'X[1:10:2, :]'
    assert str(Y[:5, 1:9:2]) == 'Y[:5, 1:9:2]'
    assert str(Y[:5, 1:10:2]) == 'Y[:5, 1::2]'
    assert str(Y[5, :5:2]) == 'Y[5:6, :5:2]'
    assert str(X[0:1, 0:1]) == 'X[:1, :1]'
    assert str(X[0:1:2, 0:1:2]) == 'X[:1:2, :1:2]'
    assert str((Y + Z)[2:, 2:]) == '(Y + Z)[2:, 2:]'

def test_true_false():
    assert str(true) == repr(true) == sstr(true) == "True"
    assert str(false) == repr(false) == sstr(false) == "False"

def test_Equivalent():
    assert str(Equivalent(y, x)) == "Equivalent(x, y)"

def test_Xor():
    assert str(Xor(y, x, evaluate=False)) == "x ^ y"

def test_Complement():
    assert str(Complement(S.Reals, S.Naturals)) == 'Complement(Reals, Naturals)'

def test_SymmetricDifference():
    assert str(SymmetricDifference(Interval(2, 3), Interval(3, 4),evaluate=False)) == \
           'SymmetricDifference(Interval(2, 3), Interval(3, 4))'


def test_UnevaluatedExpr():
    a, b = symbols("a b")
    expr1 = 2*UnevaluatedExpr(a+b)
    assert str(expr1) == "2*(a + b)"


def test_MatrixElement_printing():
    # test cases for issue #11821
    A = MatrixSymbol("A", 1, 3)
    B = MatrixSymbol("B", 1, 3)
    C = MatrixSymbol("C", 1, 3)

    assert(str(A[0, 0]) == "A[0, 0]")
    assert(str(3 * A[0, 0]) == "3*A[0, 0]")

    F = C[0, 0].subs(C, A - B)
    assert str(F) == "(A - B)[0, 0]"


def test_MatrixSymbol_printing():
    A = MatrixSymbol("A", 3, 3)
    B = MatrixSymbol("B", 3, 3)

    assert str(A - A*B - B) == "A - A*B - B"
    assert str(A*B - (A+B)) == "-A + A*B - B"
    assert str(A**(-1)) == "A**(-1)"
    assert str(A**3) == "A**3"


def test_MatrixExpressions():
    n = Symbol('n', integer=True)
    X = MatrixSymbol('X', n, n)

    assert str(X) == "X"

    # Apply function elementwise (`ElementwiseApplyFunc`):

    expr = (X.T*X).applyfunc(sin)
    assert str(expr) == 'Lambda(_d, sin(_d)).(X.T*X)'

    lamda = Lambda(x, 1/x)
    expr = (n*X).applyfunc(lamda)
    assert str(expr) == 'Lambda(x, 1/x).(n*X)'


def test_Subs_printing():
    assert str(Subs(x, (x,), (1,))) == 'Subs(x, x, 1)'
    assert str(Subs(x + y, (x, y), (1, 2))) == 'Subs(x + y, (x, y), (1, 2))'


def test_issue_15716():
    e = Integral(factorial(x), (x, -oo, oo))
    assert e.as_terms() == ([(e, ((1.0, 0.0), (1,), ()))], [e])


def test_str_special_matrices():
    from sympy.matrices import Identity, ZeroMatrix, OneMatrix
    assert str(Identity(4)) == 'I'
    assert str(ZeroMatrix(2, 2)) == '0'
    assert str(OneMatrix(2, 2)) == '1'


def test_issue_14567():
    assert factorial(Sum(-1, (x, 0, 0))) + y  # doesn't raise an error


def test_issue_21823():
    assert str(Partition([1, 2])) == 'Partition({1, 2})'
    assert str(Partition({1, 2})) == 'Partition({1, 2})'


def test_issue_22689():
    assert str(Mul(Pow(x,-2, evaluate=False), Pow(3,-1,evaluate=False), evaluate=False)) == "1/(x**2*3)"


def test_issue_21119_21460():
    ss = lambda x: str(S(x, evaluate=False))
    assert ss('4/2') == '4/2'
    assert ss('4/-2') == '4/(-2)'
    assert ss('-4/2') == '-4/2'
    assert ss('-4/-2') == '-4/(-2)'
    assert ss('-2*3/-1') == '-2*3/(-1)'
    assert ss('-2*3/-1/2') == '-2*3/(-1*2)'
    assert ss('4/2/1') == '4/(2*1)'
    assert ss('-2/-1/2') == '-2/(-1*2)'
    assert ss('2*3*4**(-2*3)') == '2*3/4**(2*3)'
    assert ss('2*3*1*4**(-2*3)') == '2*3*1/4**(2*3)'


def test_Str():
    from sympy.core.symbol import Str
    assert str(Str('x')) == 'x'
    assert sstrrepr(Str('x')) == "Str('x')"


def test_diffgeom():
    from sympy.diffgeom import Manifold, Patch, CoordSystem, BaseScalarField
    x,y = symbols('x y', real=True)
    m = Manifold('M', 2)
    assert str(m) == "M"
    p = Patch('P', m)
    assert str(p) == "P"
    rect = CoordSystem('rect', p, [x, y])
    assert str(rect) == "rect"
    b = BaseScalarField(rect, 0)
    assert str(b) == "x"

def test_NDimArray():
    assert sstr(NDimArray(1.0), full_prec=True) == '1.00000000000000'
    assert sstr(NDimArray(1.0), full_prec=False) == '1.0'
    assert sstr(NDimArray([1.0, 2.0]), full_prec=True) == '[1.00000000000000, 2.00000000000000]'
    assert sstr(NDimArray([1.0, 2.0]), full_prec=False) == '[1.0, 2.0]'
    assert sstr(NDimArray([], (0,))) == 'ImmutableDenseNDimArray([], (0,))'
    assert sstr(NDimArray([], (0, 0))) == 'ImmutableDenseNDimArray([], (0, 0))'
    assert sstr(NDimArray([], (0, 1))) == 'ImmutableDenseNDimArray([], (0, 1))'
    assert sstr(NDimArray([], (1, 0))) == 'ImmutableDenseNDimArray([], (1, 0))'

def test_Predicate():
    assert sstr(Q.even) == 'Q.even'

def test_AppliedPredicate():
    assert sstr(Q.even(x)) == 'Q.even(x)'

def test_printing_str_array_expressions():
    assert sstr(ArraySymbol("A", (2, 3, 4))) == "A"
    assert sstr(ArrayElement("A", (2, 1/(1-x), 0))) == "A[2, 1/(1 - x), 0]"
    M = MatrixSymbol("M", 3, 3)
    N = MatrixSymbol("N", 3, 3)
    assert sstr(ArrayElement(M*N, [x, 0])) == "(M*N)[x, 0]"

def test_printing_stats():
    # issue 24132
    x = RandomSymbol("x")
    y = RandomSymbol("y")
    z1 = Probability(x > 0)*Identity(2)
    z2 = Expectation(x)*Identity(2)
    z3 = Variance(x)*Identity(2)
    z4 = Covariance(x, y) * Identity(2)

    assert str(z1) == "Probability(x > 0)*I"
    assert str(z2) == "Expectation(x)*I"
    assert str(z3) == "Variance(x)*I"
    assert str(z4) ==  "Covariance(x, y)*I"
    assert z1.is_commutative == False
    assert z2.is_commutative == False
    assert z3.is_commutative == False
    assert z4.is_commutative == False
    assert z2._eval_is_commutative() == False
    assert z3._eval_is_commutative() == False
    assert z4._eval_is_commutative() == False
