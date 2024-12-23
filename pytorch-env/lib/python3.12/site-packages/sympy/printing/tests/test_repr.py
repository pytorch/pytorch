from __future__ import annotations
from typing import Any

from sympy.external.gmpy import GROUND_TYPES
from sympy.testing.pytest import raises, warns_deprecated_sympy
from sympy.assumptions.ask import Q
from sympy.core.function import (Function, WildFunction)
from sympy.core.numbers import (AlgebraicNumber, Float, Integer, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, Wild, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import sin
from sympy.functions.special.delta_functions import Heaviside
from sympy.logic.boolalg import (false, true)
from sympy.matrices.dense import (Matrix, ones)
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.immutable import ImmutableDenseMatrix
from sympy.combinatorics import Cycle, Permutation
from sympy.core.symbol import Str
from sympy.geometry import Point, Ellipse
from sympy.printing import srepr
from sympy.polys import ring, field, ZZ, QQ, lex, grlex, Poly
from sympy.polys.polyclasses import DMP
from sympy.polys.agca.extensions import FiniteExtension

x, y = symbols('x,y')

# eval(srepr(expr)) == expr has to succeed in the right environment. The right
# environment is the scope of "from sympy import *" for most cases.
ENV: dict[str, Any] = {"Str": Str}
exec("from sympy import *", ENV)


def sT(expr, string, import_stmt=None, **kwargs):
    """
    sT := sreprTest

    Tests that srepr delivers the expected string and that
    the condition eval(srepr(expr))==expr holds.
    """
    if import_stmt is None:
        ENV2 = ENV
    else:
        ENV2 = ENV.copy()
        exec(import_stmt, ENV2)

    assert srepr(expr, **kwargs) == string
    assert eval(string, ENV2) == expr


def test_printmethod():
    class R(Abs):
        def _sympyrepr(self, printer):
            return "foo(%s)" % printer._print(self.args[0])
    assert srepr(R(x)) == "foo(Symbol('x'))"


def test_Add():
    sT(x + y, "Add(Symbol('x'), Symbol('y'))")
    assert srepr(x**2 + 1, order='lex') == "Add(Pow(Symbol('x'), Integer(2)), Integer(1))"
    assert srepr(x**2 + 1, order='old') == "Add(Integer(1), Pow(Symbol('x'), Integer(2)))"
    assert srepr(sympify('x + 3 - 2', evaluate=False), order='none') == "Add(Symbol('x'), Integer(3), Mul(Integer(-1), Integer(2)))"


def test_more_than_255_args_issue_10259():
    from sympy.core.add import Add
    from sympy.core.mul import Mul
    for op in (Add, Mul):
        expr = op(*symbols('x:256'))
        assert eval(srepr(expr)) == expr


def test_Function():
    sT(Function("f")(x), "Function('f')(Symbol('x'))")
    # test unapplied Function
    sT(Function('f'), "Function('f')")

    sT(sin(x), "sin(Symbol('x'))")
    sT(sin, "sin")


def test_Heaviside():
    sT(Heaviside(x), "Heaviside(Symbol('x'))")
    sT(Heaviside(x, 1), "Heaviside(Symbol('x'), Integer(1))")


def test_Geometry():
    sT(Point(0, 0), "Point2D(Integer(0), Integer(0))")
    sT(Ellipse(Point(0, 0), 5, 1),
       "Ellipse(Point2D(Integer(0), Integer(0)), Integer(5), Integer(1))")
    # TODO more tests


def test_Singletons():
    sT(S.Catalan, 'Catalan')
    sT(S.ComplexInfinity, 'zoo')
    sT(S.EulerGamma, 'EulerGamma')
    sT(S.Exp1, 'E')
    sT(S.GoldenRatio, 'GoldenRatio')
    sT(S.TribonacciConstant, 'TribonacciConstant')
    sT(S.Half, 'Rational(1, 2)')
    sT(S.ImaginaryUnit, 'I')
    sT(S.Infinity, 'oo')
    sT(S.NaN, 'nan')
    sT(S.NegativeInfinity, '-oo')
    sT(S.NegativeOne, 'Integer(-1)')
    sT(S.One, 'Integer(1)')
    sT(S.Pi, 'pi')
    sT(S.Zero, 'Integer(0)')
    sT(S.Complexes, 'Complexes')
    sT(S.EmptySequence, 'EmptySequence')
    sT(S.EmptySet, 'EmptySet')
    # sT(S.IdentityFunction, 'Lambda(_x, _x)')
    sT(S.Naturals, 'Naturals')
    sT(S.Naturals0, 'Naturals0')
    sT(S.Rationals, 'Rationals')
    sT(S.Reals, 'Reals')
    sT(S.UniversalSet, 'UniversalSet')


def test_Integer():
    sT(Integer(4), "Integer(4)")


def test_list():
    sT([x, Integer(4)], "[Symbol('x'), Integer(4)]")


def test_Matrix():
    for cls, name in [(Matrix, "MutableDenseMatrix"), (ImmutableDenseMatrix, "ImmutableDenseMatrix")]:
        sT(cls([[x**+1, 1], [y, x + y]]),
           "%s([[Symbol('x'), Integer(1)], [Symbol('y'), Add(Symbol('x'), Symbol('y'))]])" % name)

        sT(cls(), "%s([])" % name)

        sT(cls([[x**+1, 1], [y, x + y]]), "%s([[Symbol('x'), Integer(1)], [Symbol('y'), Add(Symbol('x'), Symbol('y'))]])" % name)


def test_empty_Matrix():
    sT(ones(0, 3), "MutableDenseMatrix(0, 3, [])")
    sT(ones(4, 0), "MutableDenseMatrix(4, 0, [])")
    sT(ones(0, 0), "MutableDenseMatrix([])")


def test_Rational():
    sT(Rational(1, 3), "Rational(1, 3)")
    sT(Rational(-1, 3), "Rational(-1, 3)")


def test_Float():
    sT(Float('1.23', dps=3), "Float('1.22998', precision=13)")
    sT(Float('1.23456789', dps=9), "Float('1.23456788994', precision=33)")
    sT(Float('1.234567890123456789', dps=19),
       "Float('1.234567890123456789013', precision=66)")
    sT(Float('0.60038617995049726', dps=15),
       "Float('0.60038617995049726', precision=53)")

    sT(Float('1.23', precision=13), "Float('1.22998', precision=13)")
    sT(Float('1.23456789', precision=33),
       "Float('1.23456788994', precision=33)")
    sT(Float('1.234567890123456789', precision=66),
       "Float('1.234567890123456789013', precision=66)")
    sT(Float('0.60038617995049726', precision=53),
       "Float('0.60038617995049726', precision=53)")

    sT(Float('0.60038617995049726', 15),
       "Float('0.60038617995049726', precision=53)")


def test_Symbol():
    sT(x, "Symbol('x')")
    sT(y, "Symbol('y')")
    sT(Symbol('x', negative=True), "Symbol('x', negative=True)")


def test_Symbol_two_assumptions():
    x = Symbol('x', negative=0, integer=1)
    # order could vary
    s1 = "Symbol('x', integer=True, negative=False)"
    s2 = "Symbol('x', negative=False, integer=True)"
    assert srepr(x) in (s1, s2)
    assert eval(srepr(x), ENV) == x


def test_Symbol_no_special_commutative_treatment():
    sT(Symbol('x'), "Symbol('x')")
    sT(Symbol('x', commutative=False), "Symbol('x', commutative=False)")
    sT(Symbol('x', commutative=0), "Symbol('x', commutative=False)")
    sT(Symbol('x', commutative=True), "Symbol('x', commutative=True)")
    sT(Symbol('x', commutative=1), "Symbol('x', commutative=True)")


def test_Wild():
    sT(Wild('x', even=True), "Wild('x', even=True)")


def test_Dummy():
    d = Dummy('d')
    sT(d, "Dummy('d', dummy_index=%s)" % str(d.dummy_index))


def test_Dummy_assumption():
    d = Dummy('d', nonzero=True)
    assert d == eval(srepr(d))
    s1 = "Dummy('d', dummy_index=%s, nonzero=True)" % str(d.dummy_index)
    s2 = "Dummy('d', nonzero=True, dummy_index=%s)" % str(d.dummy_index)
    assert srepr(d) in (s1, s2)


def test_Dummy_from_Symbol():
    # should not get the full dictionary of assumptions
    n = Symbol('n', integer=True)
    d = n.as_dummy()
    assert srepr(d
        ) == "Dummy('n', dummy_index=%s)" % str(d.dummy_index)


def test_tuple():
    sT((x,), "(Symbol('x'),)")
    sT((x, y), "(Symbol('x'), Symbol('y'))")


def test_WildFunction():
    sT(WildFunction('w'), "WildFunction('w')")


def test_settins():
    raises(TypeError, lambda: srepr(x, method="garbage"))


def test_Mul():
    sT(3*x**3*y, "Mul(Integer(3), Pow(Symbol('x'), Integer(3)), Symbol('y'))")
    assert srepr(3*x**3*y, order='old') == "Mul(Integer(3), Symbol('y'), Pow(Symbol('x'), Integer(3)))"
    assert srepr(sympify('(x+4)*2*x*7', evaluate=False), order='none') == "Mul(Add(Symbol('x'), Integer(4)), Integer(2), Symbol('x'), Integer(7))"


def test_AlgebraicNumber():
    a = AlgebraicNumber(sqrt(2))
    sT(a, "AlgebraicNumber(Pow(Integer(2), Rational(1, 2)), [Integer(1), Integer(0)])")
    a = AlgebraicNumber(root(-2, 3))
    sT(a, "AlgebraicNumber(Pow(Integer(-2), Rational(1, 3)), [Integer(1), Integer(0)])")


def test_PolyRing():
    assert srepr(ring("x", ZZ, lex)[0]) == "PolyRing((Symbol('x'),), ZZ, lex)"
    assert srepr(ring("x,y", QQ, grlex)[0]) == "PolyRing((Symbol('x'), Symbol('y')), QQ, grlex)"
    assert srepr(ring("x,y,z", ZZ["t"], lex)[0]) == "PolyRing((Symbol('x'), Symbol('y'), Symbol('z')), ZZ[t], lex)"


def test_FracField():
    assert srepr(field("x", ZZ, lex)[0]) == "FracField((Symbol('x'),), ZZ, lex)"
    assert srepr(field("x,y", QQ, grlex)[0]) == "FracField((Symbol('x'), Symbol('y')), QQ, grlex)"
    assert srepr(field("x,y,z", ZZ["t"], lex)[0]) == "FracField((Symbol('x'), Symbol('y'), Symbol('z')), ZZ[t], lex)"


def test_PolyElement():
    R, x, y = ring("x,y", ZZ)
    assert srepr(3*x**2*y + 1) == "PolyElement(PolyRing((Symbol('x'), Symbol('y')), ZZ, lex), [((2, 1), 3), ((0, 0), 1)])"


def test_FracElement():
    F, x, y = field("x,y", ZZ)
    assert srepr((3*x**2*y + 1)/(x - y**2)) == "FracElement(FracField((Symbol('x'), Symbol('y')), ZZ, lex), [((2, 1), 3), ((0, 0), 1)], [((1, 0), 1), ((0, 2), -1)])"


def test_FractionField():
    assert srepr(QQ.frac_field(x)) == \
        "FractionField(FracField((Symbol('x'),), QQ, lex))"
    assert srepr(QQ.frac_field(x, y, order=grlex)) == \
        "FractionField(FracField((Symbol('x'), Symbol('y')), QQ, grlex))"


def test_PolynomialRingBase():
    assert srepr(ZZ.old_poly_ring(x)) == \
        "GlobalPolynomialRing(ZZ, Symbol('x'))"
    assert srepr(ZZ[x].old_poly_ring(y)) == \
        "GlobalPolynomialRing(ZZ[x], Symbol('y'))"
    assert srepr(QQ.frac_field(x).old_poly_ring(y)) == \
        "GlobalPolynomialRing(FractionField(FracField((Symbol('x'),), QQ, lex)), Symbol('y'))"


def test_DMP():
    p1 = DMP([1, 2], ZZ)
    p2 = ZZ.old_poly_ring(x)([1, 2])
    if GROUND_TYPES != 'flint':
        assert srepr(p1) == "DMP_Python([1, 2], ZZ)"
        assert srepr(p2) == "DMP_Python([1, 2], ZZ)"
    else:
        assert srepr(p1) == "DUP_Flint([1, 2], ZZ)"
        assert srepr(p2) == "DUP_Flint([1, 2], ZZ)"


def test_FiniteExtension():
    assert srepr(FiniteExtension(Poly(x**2 + 1, x))) == \
        "FiniteExtension(Poly(x**2 + 1, x, domain='ZZ'))"


def test_ExtensionElement():
    A = FiniteExtension(Poly(x**2 + 1, x))
    if GROUND_TYPES != 'flint':
        ans = "ExtElem(DMP_Python([1, 0], ZZ), FiniteExtension(Poly(x**2 + 1, x, domain='ZZ')))"
    else:
        ans = "ExtElem(DUP_Flint([1, 0], ZZ), FiniteExtension(Poly(x**2 + 1, x, domain='ZZ')))"
    assert srepr(A.generator) == ans

def test_BooleanAtom():
    assert srepr(true) == "true"
    assert srepr(false) == "false"


def test_Integers():
    sT(S.Integers, "Integers")


def test_Naturals():
    sT(S.Naturals, "Naturals")


def test_Naturals0():
    sT(S.Naturals0, "Naturals0")


def test_Reals():
    sT(S.Reals, "Reals")


def test_matrix_expressions():
    n = symbols('n', integer=True)
    A = MatrixSymbol("A", n, n)
    B = MatrixSymbol("B", n, n)
    sT(A, "MatrixSymbol(Str('A'), Symbol('n', integer=True), Symbol('n', integer=True))")
    sT(A*B, "MatMul(MatrixSymbol(Str('A'), Symbol('n', integer=True), Symbol('n', integer=True)), MatrixSymbol(Str('B'), Symbol('n', integer=True), Symbol('n', integer=True)))")
    sT(A + B, "MatAdd(MatrixSymbol(Str('A'), Symbol('n', integer=True), Symbol('n', integer=True)), MatrixSymbol(Str('B'), Symbol('n', integer=True), Symbol('n', integer=True)))")


def test_Cycle():
    # FIXME: sT fails because Cycle is not immutable and calling srepr(Cycle(1, 2))
    # adds keys to the Cycle dict (GH-17661)
    #import_stmt = "from sympy.combinatorics import Cycle"
    #sT(Cycle(1, 2), "Cycle(1, 2)", import_stmt)
    assert srepr(Cycle(1, 2)) == "Cycle(1, 2)"


def test_Permutation():
    import_stmt = "from sympy.combinatorics import Permutation"
    sT(Permutation(1, 2)(3, 4), "Permutation([0, 2, 1, 4, 3])", import_stmt, perm_cyclic=False)
    sT(Permutation(1, 2)(3, 4), "Permutation(1, 2)(3, 4)", import_stmt, perm_cyclic=True)

    with warns_deprecated_sympy():
        old_print_cyclic = Permutation.print_cyclic
        Permutation.print_cyclic = False
        sT(Permutation(1, 2)(3, 4), "Permutation([0, 2, 1, 4, 3])", import_stmt)
        Permutation.print_cyclic = old_print_cyclic

def test_dict():
    from sympy.abc import x, y, z
    d = {}
    assert srepr(d) == "{}"
    d = {x: y}
    assert srepr(d) == "{Symbol('x'): Symbol('y')}"
    d = {x: y, y: z}
    assert srepr(d) in (
        "{Symbol('x'): Symbol('y'), Symbol('y'): Symbol('z')}",
        "{Symbol('y'): Symbol('z'), Symbol('x'): Symbol('y')}",
    )
    d = {x: {y: z}}
    assert srepr(d) == "{Symbol('x'): {Symbol('y'): Symbol('z')}}"

def test_set():
    from sympy.abc import x, y
    s = set()
    assert srepr(s) == "set()"
    s = {x, y}
    assert srepr(s) in ("{Symbol('x'), Symbol('y')}", "{Symbol('y'), Symbol('x')}")

def test_Predicate():
    sT(Q.even, "Q.even")

def test_AppliedPredicate():
    sT(Q.even(Symbol('z')), "AppliedPredicate(Q.even, Symbol('z'))")
