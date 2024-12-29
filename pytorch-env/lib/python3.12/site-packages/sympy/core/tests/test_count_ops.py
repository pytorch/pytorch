from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.function import (Derivative, Function, count_ops)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import (Eq, Rel)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import (And, Equivalent, ITE, Implies, Nand,
    Nor, Not, Or, Xor)
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.core.containers import Tuple

x, y, z = symbols('x,y,z')
a, b, c = symbols('a,b,c')

def test_count_ops_non_visual():
    def count(val):
        return count_ops(val, visual=False)
    assert count(x) == 0
    assert count(x) is not S.Zero
    assert count(x + y) == 1
    assert count(x + y) is not S.One
    assert count(x + y*x + 2*y) == 4
    assert count({x + y: x}) == 1
    assert count({x + y: S(2) + x}) is not S.One
    assert count(x < y) == 1
    assert count(Or(x,y)) == 1
    assert count(And(x,y)) == 1
    assert count(Not(x)) == 1
    assert count(Nor(x,y)) == 2
    assert count(Nand(x,y)) == 2
    assert count(Xor(x,y)) == 1
    assert count(Implies(x,y)) == 1
    assert count(Equivalent(x,y)) == 1
    assert count(ITE(x,y,z)) == 1
    assert count(ITE(True,x,y)) == 0


def test_count_ops_visual():
    ADD, MUL, POW, SIN, COS, EXP, AND, D, G, M = symbols(
        'Add Mul Pow sin cos exp And Derivative Integral Sum'.upper())
    DIV, SUB, NEG = symbols('DIV SUB NEG')
    LT, LE, GT, GE, EQ, NE = symbols('LT LE GT GE EQ NE')
    NOT, OR, AND, XOR, IMPLIES, EQUIVALENT, _ITE, BASIC, TUPLE = symbols(
        'Not Or And Xor Implies Equivalent ITE Basic Tuple'.upper())

    def count(val):
        return count_ops(val, visual=True)

    assert count(7) is S.Zero
    assert count(S(7)) is S.Zero
    assert count(-1) == NEG
    assert count(-2) == NEG
    assert count(S(2)/3) == DIV
    assert count(Rational(2, 3)) == DIV
    assert count(pi/3) == DIV
    assert count(-pi/3) == DIV + NEG
    assert count(I - 1) == SUB
    assert count(1 - I) == SUB
    assert count(1 - 2*I) == SUB + MUL

    assert count(x) is S.Zero
    assert count(-x) == NEG
    assert count(-2*x/3) == NEG + DIV + MUL
    assert count(Rational(-2, 3)*x) == NEG + DIV + MUL
    assert count(1/x) == DIV
    assert count(1/(x*y)) == DIV + MUL
    assert count(-1/x) == NEG + DIV
    assert count(-2/x) == NEG + DIV
    assert count(x/y) == DIV
    assert count(-x/y) == NEG + DIV

    assert count(x**2) == POW
    assert count(-x**2) == POW + NEG
    assert count(-2*x**2) == POW + MUL + NEG

    assert count(x + pi/3) == ADD + DIV
    assert count(x + S.One/3) == ADD + DIV
    assert count(x + Rational(1, 3)) == ADD + DIV
    assert count(x + y) == ADD
    assert count(x - y) == SUB
    assert count(y - x) == SUB
    assert count(-1/(x - y)) == DIV + NEG + SUB
    assert count(-1/(y - x)) == DIV + NEG + SUB
    assert count(1 + x**y) == ADD + POW
    assert count(1 + x + y) == 2*ADD
    assert count(1 + x + y + z) == 3*ADD
    assert count(1 + x**y + 2*x*y + y**2) == 3*ADD + 2*POW + 2*MUL
    assert count(2*z + y + x + 1) == 3*ADD + MUL
    assert count(2*z + y**17 + x + 1) == 3*ADD + MUL + POW
    assert count(2*z + y**17 + x + sin(x)) == 3*ADD + POW + MUL + SIN
    assert count(2*z + y**17 + x + sin(x**2)) == 3*ADD + MUL + 2*POW + SIN
    assert count(2*z + y**17 + x + sin(
        x**2) + exp(cos(x))) == 4*ADD + MUL + 2*POW + EXP + COS + SIN

    assert count(Derivative(x, x)) == D
    assert count(Integral(x, x) + 2*x/(1 + x)) == G + DIV + MUL + 2*ADD
    assert count(Sum(x, (x, 1, x + 1)) + 2*x/(1 + x)) == M + DIV + MUL + 3*ADD
    assert count(Basic()) is S.Zero

    assert count({x + 1: sin(x)}) == ADD + SIN
    assert count([x + 1, sin(x) + y, None]) == ADD + SIN + ADD
    assert count({x + 1: sin(x), y: cos(x) + 1}) == SIN + COS + 2*ADD
    assert count({}) is S.Zero
    assert count([x + 1, sin(x)*y, None]) == SIN + ADD + MUL
    assert count([]) is S.Zero

    assert count(Basic()) == 0
    assert count(Basic(Basic(),Basic(x,x+y))) == ADD + 2*BASIC
    assert count(Basic(x, x + y)) == ADD + BASIC
    assert [count(Rel(x, y, op)) for op in '< <= > >= == <> !='.split()
        ] == [LT, LE, GT, GE, EQ, NE, NE]
    assert count(Or(x, y)) == OR
    assert count(And(x, y)) == AND
    assert count(Or(x, Or(y, And(z, a)))) == AND + OR
    assert count(Nor(x, y)) == NOT + OR
    assert count(Nand(x, y)) == NOT + AND
    assert count(Xor(x, y)) == XOR
    assert count(Implies(x, y)) == IMPLIES
    assert count(Equivalent(x, y)) == EQUIVALENT
    assert count(ITE(x, y, z)) == _ITE
    assert count([Or(x, y), And(x, y), Basic(x + y)]
        ) == ADD + AND + BASIC + OR

    assert count(Basic(Tuple(x))) == BASIC + TUPLE
    #It checks that TUPLE is counted as an operation.

    assert count(Eq(x + y, S(2))) == ADD + EQ


def test_issue_9324():
    def count(val):
        return count_ops(val, visual=False)

    M = MatrixSymbol('M', 10, 10)
    assert count(M[0, 0]) == 0
    assert count(2 * M[0, 0] + M[5, 7]) == 2
    P = MatrixSymbol('P', 3, 3)
    Q = MatrixSymbol('Q', 3, 3)
    assert count(P + Q) == 1
    m = Symbol('m', integer=True)
    n = Symbol('n', integer=True)
    M = MatrixSymbol('M', m + n, m * m)
    assert count(M[0, 1]) == 2


def test_issue_21532():
    f = Function('f')
    g = Function('g')
    FUNC_F, FUNC_G = symbols('FUNC_F, FUNC_G')
    assert f(x).count_ops(visual=True) == FUNC_F
    assert g(x).count_ops(visual=True) == FUNC_G
