from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import unchanged
from sympy.core.function import (Function, diff, expand)
from sympy.core.mul import Mul
from sympy.core.mod import Mod
from sympy.core.numbers import (Float, I, Rational, oo, pi, zoo)
from sympy.core.relational import (Eq, Ge, Gt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (Abs, adjoint, arg, conjugate, im, re, transpose)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.piecewise import (Piecewise,
    piecewise_fold, piecewise_exclusive, Undefined, ExprCondPair)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.delta_functions import (DiracDelta, Heaviside)
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.integrals.integrals import (Integral, integrate)
from sympy.logic.boolalg import (And, ITE, Not, Or)
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.printing import srepr
from sympy.sets.contains import Contains
from sympy.sets.sets import Interval
from sympy.solvers.solvers import solve
from sympy.testing.pytest import raises, slow
from sympy.utilities.lambdify import lambdify

a, b, c, d, x, y = symbols('a:d, x, y')
z = symbols('z', nonzero=True)


def test_piecewise1():

    # Test canonicalization
    assert Piecewise((x, x < 1.)).has(1.0)  # doesn't get changed to x < 1
    assert unchanged(Piecewise, ExprCondPair(x, x < 1), ExprCondPair(0, True))
    assert Piecewise((x, x < 1), (0, True)) == Piecewise(ExprCondPair(x, x < 1),
                                                         ExprCondPair(0, True))
    assert Piecewise((x, x < 1), (0, True), (1, True)) == \
        Piecewise((x, x < 1), (0, True))
    assert Piecewise((x, x < 1), (0, False), (-1, 1 > 2)) == \
        Piecewise((x, x < 1))
    assert Piecewise((x, x < 1), (0, x < 1), (0, True)) == \
        Piecewise((x, x < 1), (0, True))
    assert Piecewise((x, x < 1), (0, x < 2), (0, True)) == \
        Piecewise((x, x < 1), (0, True))
    assert Piecewise((x, x < 1), (x, x < 2), (0, True)) == \
        Piecewise((x, Or(x < 1, x < 2)), (0, True))
    assert Piecewise((x, x < 1), (x, x < 2), (x, True)) == x
    assert Piecewise((x, True)) == x
    # Explicitly constructed empty Piecewise not accepted
    raises(TypeError, lambda: Piecewise())
    # False condition is never retained
    assert Piecewise((2*x, x < 0), (x, False)) == \
        Piecewise((2*x, x < 0), (x, False), evaluate=False) == \
        Piecewise((2*x, x < 0))
    assert Piecewise((x, False)) == Undefined
    raises(TypeError, lambda: Piecewise(x))
    assert Piecewise((x, 1)) == x  # 1 and 0 are accepted as True/False
    raises(TypeError, lambda: Piecewise((x, 2)))
    raises(TypeError, lambda: Piecewise((x, x**2)))
    raises(TypeError, lambda: Piecewise(([1], True)))
    assert Piecewise(((1, 2), True)) == Tuple(1, 2)
    cond = (Piecewise((1, x < 0), (2, True)) < y)
    assert Piecewise((1, cond)
        ) == Piecewise((1, ITE(x < 0, y > 1, y > 2)))

    assert Piecewise((1, x > 0), (2, And(x <= 0, x > -1))
        ) == Piecewise((1, x > 0), (2, x > -1))
    assert Piecewise((1, x <= 0), (2, (x < 0) & (x > -1))
        ) == Piecewise((1, x <= 0))

    # test for supporting Contains in Piecewise
    pwise = Piecewise(
        (1, And(x <= 6, x > 1, Contains(x, S.Integers))),
        (0, True))
    assert pwise.subs(x, pi) == 0
    assert pwise.subs(x, 2) == 1
    assert pwise.subs(x, 7) == 0

    # Test subs
    p = Piecewise((-1, x < -1), (x**2, x < 0), (log(x), x >= 0))
    p_x2 = Piecewise((-1, x**2 < -1), (x**4, x**2 < 0), (log(x**2), x**2 >= 0))
    assert p.subs(x, x**2) == p_x2
    assert p.subs(x, -5) == -1
    assert p.subs(x, -1) == 1
    assert p.subs(x, 1) == log(1)

    # More subs tests
    p2 = Piecewise((1, x < pi), (-1, x < 2*pi), (0, x > 2*pi))
    p3 = Piecewise((1, Eq(x, 0)), (1/x, True))
    p4 = Piecewise((1, Eq(x, 0)), (2, 1/x>2))
    assert p2.subs(x, 2) == 1
    assert p2.subs(x, 4) == -1
    assert p2.subs(x, 10) == 0
    assert p3.subs(x, 0.0) == 1
    assert p4.subs(x, 0.0) == 1


    f, g, h = symbols('f,g,h', cls=Function)
    pf = Piecewise((f(x), x < -1), (f(x) + h(x) + 2, x <= 1))
    pg = Piecewise((g(x), x < -1), (g(x) + h(x) + 2, x <= 1))
    assert pg.subs(g, f) == pf

    assert Piecewise((1, Eq(x, 0)), (0, True)).subs(x, 0) == 1
    assert Piecewise((1, Eq(x, 0)), (0, True)).subs(x, 1) == 0
    assert Piecewise((1, Eq(x, y)), (0, True)).subs(x, y) == 1
    assert Piecewise((1, Eq(x, z)), (0, True)).subs(x, z) == 1
    assert Piecewise((1, Eq(exp(x), cos(z))), (0, True)).subs(x, z) == \
        Piecewise((1, Eq(exp(z), cos(z))), (0, True))

    p5 = Piecewise( (0, Eq(cos(x) + y, 0)), (1, True))
    assert p5.subs(y, 0) == Piecewise( (0, Eq(cos(x), 0)), (1, True))

    assert Piecewise((-1, y < 1), (0, x < 0), (1, Eq(x, 0)), (2, True)
        ).subs(x, 1) == Piecewise((-1, y < 1), (2, True))
    assert Piecewise((1, Eq(x**2, -1)), (2, x < 0)).subs(x, I) == 1

    p6 = Piecewise((x, x > 0))
    n = symbols('n', negative=True)
    assert p6.subs(x, n) == Undefined

    # Test evalf
    assert p.evalf() == Piecewise((-1.0, x < -1), (x**2, x < 0), (log(x), True))
    assert p.evalf(subs={x: -2}) == -1.0
    assert p.evalf(subs={x: -1}) == 1.0
    assert p.evalf(subs={x: 1}) == log(1)
    assert p6.evalf(subs={x: -5}) == Undefined

    # Test doit
    f_int = Piecewise((Integral(x, (x, 0, 1)), x < 1))
    assert f_int.doit() == Piecewise( (S.Half, x < 1) )

    # Test differentiation
    f = x
    fp = x*p
    dp = Piecewise((0, x < -1), (2*x, x < 0), (1/x, x >= 0))
    fp_dx = x*dp + p
    assert diff(p, x) == dp
    assert diff(f*p, x) == fp_dx

    # Test simple arithmetic
    assert x*p == fp
    assert x*p + p == p + x*p
    assert p + f == f + p
    assert p + dp == dp + p
    assert p - dp == -(dp - p)

    # Test power
    dp2 = Piecewise((0, x < -1), (4*x**2, x < 0), (1/x**2, x >= 0))
    assert dp**2 == dp2

    # Test _eval_interval
    f1 = x*y + 2
    f2 = x*y**2 + 3
    peval = Piecewise((f1, x < 0), (f2, x > 0))
    peval_interval = f1.subs(
        x, 0) - f1.subs(x, -1) + f2.subs(x, 1) - f2.subs(x, 0)
    assert peval._eval_interval(x, 0, 0) == 0
    assert peval._eval_interval(x, -1, 1) == peval_interval
    peval2 = Piecewise((f1, x < 0), (f2, True))
    assert peval2._eval_interval(x, 0, 0) == 0
    assert peval2._eval_interval(x, 1, -1) == -peval_interval
    assert peval2._eval_interval(x, -1, -2) == f1.subs(x, -2) - f1.subs(x, -1)
    assert peval2._eval_interval(x, -1, 1) == peval_interval
    assert peval2._eval_interval(x, None, 0) == peval2.subs(x, 0)
    assert peval2._eval_interval(x, -1, None) == -peval2.subs(x, -1)

    # Test integration
    assert p.integrate() == Piecewise(
        (-x, x < -1),
        (x**3/3 + Rational(4, 3), x < 0),
        (x*log(x) - x + Rational(4, 3), True))
    p = Piecewise((x, x < 1), (x**2, -1 <= x), (x, 3 < x))
    assert integrate(p, (x, -2, 2)) == Rational(5, 6)
    assert integrate(p, (x, 2, -2)) == Rational(-5, 6)
    p = Piecewise((0, x < 0), (1, x < 1), (0, x < 2), (1, x < 3), (0, True))
    assert integrate(p, (x, -oo, oo)) == 2
    p = Piecewise((x, x < -10), (x**2, x <= -1), (x, 1 < x))
    assert integrate(p, (x, -2, 2)) == Undefined

    # Test commutativity
    assert isinstance(p, Piecewise) and p.is_commutative is True


def test_piecewise_free_symbols():
    f = Piecewise((x, a < 0), (y, True))
    assert f.free_symbols == {x, y, a}


def test_piecewise_integrate1():
    x, y = symbols('x y', real=True)

    f = Piecewise(((x - 2)**2, x >= 0), (1, True))
    assert integrate(f, (x, -2, 2)) == Rational(14, 3)

    g = Piecewise(((x - 5)**5, x >= 4), (f, True))
    assert integrate(g, (x, -2, 2)) == Rational(14, 3)
    assert integrate(g, (x, -2, 5)) == Rational(43, 6)

    assert g == Piecewise(((x - 5)**5, x >= 4), (f, x < 4))

    g = Piecewise(((x - 5)**5, 2 <= x), (f, x < 2))
    assert integrate(g, (x, -2, 2)) == Rational(14, 3)
    assert integrate(g, (x, -2, 5)) == Rational(-701, 6)

    assert g == Piecewise(((x - 5)**5, 2 <= x), (f, True))

    g = Piecewise(((x - 5)**5, 2 <= x), (2*f, True))
    assert integrate(g, (x, -2, 2)) == Rational(28, 3)
    assert integrate(g, (x, -2, 5)) == Rational(-673, 6)


def test_piecewise_integrate1b():
    g = Piecewise((1, x > 0), (0, Eq(x, 0)), (-1, x < 0))
    assert integrate(g, (x, -1, 1)) == 0

    g = Piecewise((1, x - y < 0), (0, True))
    assert integrate(g, (y, -oo, 0)) == -Min(0, x)
    assert g.subs(x, -3).integrate((y, -oo, 0)) == 3
    assert integrate(g, (y, 0, -oo)) == Min(0, x)
    assert integrate(g, (y, 0, oo)) == -Max(0, x) + oo
    assert integrate(g, (y, -oo, 42)) == -Min(42, x) + 42
    assert integrate(g, (y, -oo, oo)) == -x + oo

    g = Piecewise((0, x < 0), (x, x <= 1), (1, True))
    gy1 = g.integrate((x, y, 1))
    g1y = g.integrate((x, 1, y))
    for yy in (-1, S.Half, 2):
        assert g.integrate((x, yy, 1)) == gy1.subs(y, yy)
        assert g.integrate((x, 1, yy)) == g1y.subs(y, yy)
    assert gy1 == Piecewise(
        (-Min(1, Max(0, y))**2/2 + S.Half, y < 1),
        (-y + 1, True))
    assert g1y == Piecewise(
        (Min(1, Max(0, y))**2/2 - S.Half, y < 1),
        (y - 1, True))


@slow
def test_piecewise_integrate1ca():
    y = symbols('y', real=True)
    g = Piecewise(
        (1 - x, Interval(0, 1).contains(x)),
        (1 + x, Interval(-1, 0).contains(x)),
        (0, True)
        )
    gy1 = g.integrate((x, y, 1))
    g1y = g.integrate((x, 1, y))

    assert g.integrate((x, -2, 1)) == gy1.subs(y, -2)
    assert g.integrate((x, 1, -2)) == g1y.subs(y, -2)
    assert g.integrate((x, 0, 1)) == gy1.subs(y, 0)
    assert g.integrate((x, 1, 0)) == g1y.subs(y, 0)
    assert g.integrate((x, 2, 1)) == gy1.subs(y, 2)
    assert g.integrate((x, 1, 2)) == g1y.subs(y, 2)
    assert piecewise_fold(gy1.rewrite(Piecewise)
        ).simplify() == Piecewise(
            (1, y <= -1),
            (-y**2/2 - y + S.Half, y <= 0),
            (y**2/2 - y + S.Half, y < 1),
            (0, True))
    assert piecewise_fold(g1y.rewrite(Piecewise)
        ).simplify() == Piecewise(
            (-1, y <= -1),
            (y**2/2 + y - S.Half, y <= 0),
            (-y**2/2 + y - S.Half, y < 1),
            (0, True))
    assert gy1 == Piecewise(
        (
            -Min(1, Max(-1, y))**2/2 - Min(1, Max(-1, y)) +
            Min(1, Max(0, y))**2 + S.Half, y < 1),
        (0, True)
        )
    assert g1y == Piecewise(
        (
            Min(1, Max(-1, y))**2/2 + Min(1, Max(-1, y)) -
            Min(1, Max(0, y))**2 - S.Half, y < 1),
        (0, True))


@slow
def test_piecewise_integrate1cb():
    y = symbols('y', real=True)
    g = Piecewise(
        (0, Or(x <= -1, x >= 1)),
        (1 - x, x > 0),
        (1 + x, True)
        )
    gy1 = g.integrate((x, y, 1))
    g1y = g.integrate((x, 1, y))

    assert g.integrate((x, -2, 1)) == gy1.subs(y, -2)
    assert g.integrate((x, 1, -2)) == g1y.subs(y, -2)
    assert g.integrate((x, 0, 1)) == gy1.subs(y, 0)
    assert g.integrate((x, 1, 0)) == g1y.subs(y, 0)
    assert g.integrate((x, 2, 1)) == gy1.subs(y, 2)
    assert g.integrate((x, 1, 2)) == g1y.subs(y, 2)

    assert piecewise_fold(gy1.rewrite(Piecewise)
        ).simplify() == Piecewise(
            (1, y <= -1),
            (-y**2/2 - y + S.Half, y <= 0),
            (y**2/2 - y + S.Half, y < 1),
            (0, True))
    assert piecewise_fold(g1y.rewrite(Piecewise)
        ).simplify() == Piecewise(
            (-1, y <= -1),
            (y**2/2 + y - S.Half, y <= 0),
            (-y**2/2 + y - S.Half, y < 1),
            (0, True))

    # g1y and gy1 should simplify if the condition that y < 1
    # is applied, e.g. Min(1, Max(-1, y)) --> Max(-1, y)
    assert gy1 == Piecewise(
        (
            -Min(1, Max(-1, y))**2/2 - Min(1, Max(-1, y)) +
            Min(1, Max(0, y))**2 + S.Half, y < 1),
        (0, True)
        )
    assert g1y == Piecewise(
        (
            Min(1, Max(-1, y))**2/2 + Min(1, Max(-1, y)) -
            Min(1, Max(0, y))**2 - S.Half, y < 1),
        (0, True))


def test_piecewise_integrate2():
    from itertools import permutations
    lim = Tuple(x, c, d)
    p = Piecewise((1, x < a), (2, x > b), (3, True))
    q = p.integrate(lim)
    assert q == Piecewise(
        (-c + 2*d - 2*Min(d, Max(a, c)) + Min(d, Max(a, b, c)), c < d),
        (-2*c + d + 2*Min(c, Max(a, d)) - Min(c, Max(a, b, d)), True))
    for v in permutations((1, 2, 3, 4)):
        r = dict(zip((a, b, c, d), v))
        assert p.subs(r).integrate(lim.subs(r)) == q.subs(r)


def test_meijer_bypass():
    # totally bypass meijerg machinery when dealing
    # with Piecewise in integrate
    assert Piecewise((1, x < 4), (0, True)).integrate((x, oo, 1)) == -3


def test_piecewise_integrate3_inequality_conditions():
    from sympy.utilities.iterables import cartes
    lim = (x, 0, 5)
    # set below includes two pts below range, 2 pts in range,
    # 2 pts above range, and the boundaries
    N = (-2, -1, 0, 1, 2, 5, 6, 7)

    p = Piecewise((1, x > a), (2, x > b), (0, True))
    ans = p.integrate(lim)
    for i, j in cartes(N, repeat=2):
        reps = dict(zip((a, b), (i, j)))
        assert ans.subs(reps) == p.subs(reps).integrate(lim)
    assert ans.subs(a, 4).subs(b, 1) == 0 + 2*3 + 1

    p = Piecewise((1, x > a), (2, x < b), (0, True))
    ans = p.integrate(lim)
    for i, j in cartes(N, repeat=2):
        reps = dict(zip((a, b), (i, j)))
        assert ans.subs(reps) == p.subs(reps).integrate(lim)

    # delete old tests that involved c1 and c2 since those
    # reduce to the above except that a value of 0 was used
    # for two expressions whereas the above uses 3 different
    # values


@slow
def test_piecewise_integrate4_symbolic_conditions():
    a = Symbol('a', real=True)
    b = Symbol('b', real=True)
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    p0 = Piecewise((0, Or(x < a, x > b)), (1, True))
    p1 = Piecewise((0, x < a), (0, x > b), (1, True))
    p2 = Piecewise((0, x > b), (0, x < a), (1, True))
    p3 = Piecewise((0, x < a), (1, x < b), (0, True))
    p4 = Piecewise((0, x > b), (1, x > a), (0, True))
    p5 = Piecewise((1, And(a < x, x < b)), (0, True))

    # check values of a=1, b=3 (and reversed) with values
    # of y of 0, 1, 2, 3, 4
    lim = Tuple(x, -oo, y)
    for p in (p0, p1, p2, p3, p4, p5):
        ans = p.integrate(lim)
        for i in range(5):
            reps = {a:1, b:3, y:i}
            assert ans.subs(reps) == p.subs(reps).integrate(lim.subs(reps))
            reps = {a: 3, b:1, y:i}
            assert ans.subs(reps) == p.subs(reps).integrate(lim.subs(reps))
    lim = Tuple(x, y, oo)
    for p in (p0, p1, p2, p3, p4, p5):
        ans = p.integrate(lim)
        for i in range(5):
            reps = {a:1, b:3, y:i}
            assert ans.subs(reps) == p.subs(reps).integrate(lim.subs(reps))
            reps = {a:3, b:1, y:i}
            assert ans.subs(reps) == p.subs(reps).integrate(lim.subs(reps))

    ans = Piecewise(
        (0, x <= Min(a, b)),
        (x - Min(a, b), x <= b),
        (b - Min(a, b), True))
    for i in (p0, p1, p2, p4):
        assert i.integrate(x) == ans
    assert p3.integrate(x) == Piecewise(
        (0, x < a),
        (-a + x, x <= Max(a, b)),
        (-a + Max(a, b), True))
    assert p5.integrate(x) == Piecewise(
        (0, x <= a),
        (-a + x, x <= Max(a, b)),
        (-a + Max(a, b), True))

    p1 = Piecewise((0, x < a), (S.Half, x > b), (1, True))
    p2 = Piecewise((S.Half, x > b), (0, x < a), (1, True))
    p3 = Piecewise((0, x < a), (1, x < b), (S.Half, True))
    p4 = Piecewise((S.Half, x > b), (1, x > a), (0, True))
    p5 = Piecewise((1, And(a < x, x < b)), (S.Half, x > b), (0, True))

    # check values of a=1, b=3 (and reversed) with values
    # of y of 0, 1, 2, 3, 4
    lim = Tuple(x, -oo, y)
    for p in (p1, p2, p3, p4, p5):
        ans = p.integrate(lim)
        for i in range(5):
            reps = {a:1, b:3, y:i}
            assert ans.subs(reps) == p.subs(reps).integrate(lim.subs(reps))
            reps = {a: 3, b:1, y:i}
            assert ans.subs(reps) == p.subs(reps).integrate(lim.subs(reps))


def test_piecewise_integrate5_independent_conditions():
    p = Piecewise((0, Eq(y, 0)), (x*y, True))
    assert integrate(p, (x, 1, 3)) == Piecewise((0, Eq(y, 0)), (4*y, True))


def test_issue_22917():
    p = (Piecewise((0, ITE((x - y > 1) | (2 * x - 2 * y > 1), False,
                           ITE(x - y > 1, 2 * y - 2 < -1, 2 * x - 2 * y > 1))),
                   (Piecewise((0, ITE(x - y > 1, True, 2 * x - 2 * y > 1)),
                              (2 * Piecewise((0, x - y > 1), (y, True)), True)), True))
         + 2 * Piecewise((1, ITE((x - y > 1) | (2 * x - 2 * y > 1), False,
                                 ITE(x - y > 1, 2 * y - 2 < -1, 2 * x - 2 * y > 1))),
                         (Piecewise((1, ITE(x - y > 1, True, 2 * x - 2 * y > 1)),
                                    (2 * Piecewise((1, x - y > 1), (x, True)), True)), True)))
    assert piecewise_fold(p) == Piecewise((2, (x - y > S.Half) | (x - y > 1)),
                                          (2*y + 4, x - y > 1),
                                          (4*x + 2*y, True))
    assert piecewise_fold(p > 1).rewrite(ITE) == ITE((x - y > S.Half) | (x - y > 1), True,
                                                     ITE(x - y > 1, 2*y + 4 > 1, 4*x + 2*y > 1))


def test_piecewise_simplify():
    p = Piecewise(((x**2 + 1)/x**2, Eq(x*(1 + x) - x**2, 0)),
                  ((-1)**x*(-1), True))
    assert p.simplify() == \
        Piecewise((zoo, Eq(x, 0)), ((-1)**(x + 1), True))
    # simplify when there are Eq in conditions
    assert Piecewise(
        (a, And(Eq(a, 0), Eq(a + b, 0))), (1, True)).simplify(
        ) == Piecewise(
        (0, And(Eq(a, 0), Eq(b, 0))), (1, True))
    assert Piecewise((2*x*factorial(a)/(factorial(y)*factorial(-y + a)),
        Eq(y, 0) & Eq(-y + a, 0)), (2*factorial(a)/(factorial(y)*factorial(-y
        + a)), Eq(y, 0) & Eq(-y + a, 1)), (0, True)).simplify(
        ) == Piecewise(
            (2*x, And(Eq(a, 0), Eq(y, 0))),
            (2, And(Eq(a, 1), Eq(y, 0))),
            (0, True))
    args = (2, And(Eq(x, 2), Ge(y, 0))), (x, True)
    assert Piecewise(*args).simplify() == Piecewise(*args)
    args = (1, Eq(x, 0)), (sin(x)/x, True)
    assert Piecewise(*args).simplify() == Piecewise(*args)
    assert Piecewise((2 + y, And(Eq(x, 2), Eq(y, 0))), (x, True)
        ).simplify() == x
    # check that x or f(x) are recognized as being Symbol-like for lhs
    args = Tuple((1, Eq(x, 0)), (sin(x) + 1 + x, True))
    ans = x + sin(x) + 1
    f = Function('f')
    assert Piecewise(*args).simplify() == ans
    assert Piecewise(*args.subs(x, f(x))).simplify() == ans.subs(x, f(x))

    # issue 18634
    d = Symbol("d", integer=True)
    n = Symbol("n", integer=True)
    t = Symbol("t", positive=True)
    expr = Piecewise((-d + 2*n, Eq(1/t, 1)), (t**(1 - 4*n)*t**(4*n - 1)*(-d + 2*n), True))
    assert expr.simplify() == -d + 2*n

    # issue 22747
    p = Piecewise((0, (t < -2) & (t < -1) & (t < 0)), ((t/2 + 1)*(t +
        1)*(t + 2), (t < -1) & (t < 0)), ((S.Half - t/2)*(1 - t)*(t + 1),
        (t < -2) & (t < -1) & (t < 1)), ((t + 1)*(-t*(t/2 + 1) + (S.Half
        - t/2)*(1 - t)), (t < -2) & (t < -1) & (t < 0) & (t < 1)), ((t +
        1)*((S.Half - t/2)*(1 - t) + (t/2 + 1)*(t + 2)), (t < -1) & (t <
        1)), ((t + 1)*(-t*(t/2 + 1) + (S.Half - t/2)*(1 - t)), (t < -1) &
        (t < 0) & (t < 1)), (0, (t < -2) & (t < -1)), ((t/2 + 1)*(t +
        1)*(t + 2), t < -1), ((t + 1)*(-t*(t/2 + 1) + (S.Half - t/2)*(t +
        1)), (t < 0) & ((t < -2) | (t < 0))), ((S.Half - t/2)*(1 - t)*(t
        + 1), (t < 1) & ((t < -2) | (t < 1))), (0, True)) + Piecewise((0,
        (t < -1) & (t < 0) & (t < 1)), ((1 - t)*(t/2 + S.Half)*(t + 1),
        (t < 0) & (t < 1)), ((1 - t)*(1 - t/2)*(2 - t), (t < -1) & (t <
        0) & (t < 2)), ((1 - t)*((1 - t)*(t/2 + S.Half) + (1 - t/2)*(2 -
        t)), (t < -1) & (t < 0) & (t < 1) & (t < 2)), ((1 - t)*((1 -
        t/2)*(2 - t) + (t/2 + S.Half)*(t + 1)), (t < 0) & (t < 2)), ((1 -
        t)*((1 - t)*(t/2 + S.Half) + (1 - t/2)*(2 - t)), (t < 0) & (t <
        1) & (t < 2)), (0, (t < -1) & (t < 0)), ((1 - t)*(t/2 +
        S.Half)*(t + 1), t < 0), ((1 - t)*(t*(1 - t/2) + (1 - t)*(t/2 +
        S.Half)), (t < 1) & ((t < -1) | (t < 1))), ((1 - t)*(1 - t/2)*(2
        - t), (t < 2) & ((t < -1) | (t < 2))), (0, True))
    assert p.simplify() == Piecewise(
        (0, t < -2), ((t + 1)*(t + 2)**2/2, t < -1), (-3*t**3/2
        - 5*t**2/2 + 1, t < 0), (3*t**3/2 - 5*t**2/2 + 1, t < 1), ((1 -
        t)*(t - 2)**2/2, t < 2), (0, True))

    # coverage
    nan = Undefined
    assert Piecewise((1, x > 3), (2, x < 2), (3, x > 1)).simplify(
        )  == Piecewise((1, x > 3), (2, x < 2), (3, True))
    assert Piecewise((1, x < 2), (2, x < 1), (3, True)).simplify(
        ) == Piecewise((1, x < 2), (3, True))
    assert Piecewise((1, x > 2)).simplify() == Piecewise((1, x > 2),
        (nan, True))
    assert Piecewise((1, (x >= 2) & (x < oo))
        ).simplify() == Piecewise((1, (x >= 2) & (x < oo)), (nan, True))
    assert Piecewise((1, x < 2), (2, (x > 1) & (x < 3)), (3, True)
        ). simplify() == Piecewise((1, x < 2), (2, x < 3), (3, True))
    assert Piecewise((1, x < 2), (2, (x <= 3) & (x > 1)), (3, True)
        ).simplify() == Piecewise((1, x < 2), (2, x <= 3), (3, True))
    assert Piecewise((1, x < 2), (2, (x > 2) & (x < 3)), (3, True)
        ).simplify() == Piecewise((1, x < 2), (2, (x > 2) & (x < 3)),
        (3, True))
    assert Piecewise((1, x < 2), (2, (x >= 1) & (x <= 3)), (3, True)
        ).simplify() == Piecewise((1, x < 2), (2, x <= 3), (3, True))
    assert Piecewise((1, x < 1), (2, (x >= 2) & (x <= 3)), (3, True)
        ).simplify() == Piecewise((1, x < 1), (2, (x >= 2) & (x <= 3)),
        (3, True))
    # https://github.com/sympy/sympy/issues/25603
    assert Piecewise((log(x), (x <= 5) & (x > 3)), (x, True)
        ).simplify() == Piecewise((log(x), (x <= 5) & (x > 3)), (x, True))

    assert Piecewise((1, (x >= 1) & (x < 3)), (2, (x > 2) & (x < 4))
        ).simplify() == Piecewise((1, (x >= 1) & (x < 3)), (
        2, (x >= 3) & (x < 4)), (nan, True))
    assert Piecewise((1, (x >= 1) & (x <= 3)), (2, (x > 2) & (x < 4))
        ).simplify() == Piecewise((1, (x >= 1) & (x <= 3)), (
        2, (x > 3) & (x < 4)), (nan, True))

    # involves a symbolic range so cset.inf fails
    L = Symbol('L', nonnegative=True)
    p = Piecewise((nan, x <= 0), (0, (x >= 0) & (L > x) & (L - x <= 0)),
        (x - L/2, (L > x) & (L - x <= 0)),
        (L/2 - x, (x >= 0) & (L > x)),
        (0, L > x), (nan, True))
    assert p.simplify() == Piecewise(
        (nan, x <= 0), (L/2 - x, L > x), (nan, True))
    assert p.subs(L, y).simplify() == Piecewise(
        (nan, x <= 0), (-x + y/2, x < Max(0, y)), (0, x < y), (nan, True))


def test_piecewise_solve():
    abs2 = Piecewise((-x, x <= 0), (x, x > 0))
    f = abs2.subs(x, x - 2)
    assert solve(f, x) == [2]
    assert solve(f - 1, x) == [1, 3]

    f = Piecewise(((x - 2)**2, x >= 0), (1, True))
    assert solve(f, x) == [2]

    g = Piecewise(((x - 5)**5, x >= 4), (f, True))
    assert solve(g, x) == [2, 5]

    g = Piecewise(((x - 5)**5, x >= 4), (f, x < 4))
    assert solve(g, x) == [2, 5]

    g = Piecewise(((x - 5)**5, x >= 2), (f, x < 2))
    assert solve(g, x) == [5]

    g = Piecewise(((x - 5)**5, x >= 2), (f, True))
    assert solve(g, x) == [5]

    g = Piecewise(((x - 5)**5, x >= 2), (f, True), (10, False))
    assert solve(g, x) == [5]

    g = Piecewise(((x - 5)**5, x >= 2),
                  (-x + 2, x - 2 <= 0), (x - 2, x - 2 > 0))
    assert solve(g, x) == [5]

    # if no symbol is given the piecewise detection must still work
    assert solve(Piecewise((x - 2, x > 2), (2 - x, True)) - 3) == [-1, 5]

    f = Piecewise(((x - 2)**2, x >= 0), (0, True))
    raises(NotImplementedError, lambda: solve(f, x))

    def nona(ans):
        return list(filter(lambda x: x is not S.NaN, ans))
    p = Piecewise((x**2 - 4, x < y), (x - 2, True))
    ans = solve(p, x)
    assert nona([i.subs(y, -2) for i in ans]) == [2]
    assert nona([i.subs(y, 2) for i in ans]) == [-2, 2]
    assert nona([i.subs(y, 3) for i in ans]) == [-2, 2]
    assert ans == [
        Piecewise((-2, y > -2), (S.NaN, True)),
        Piecewise((2, y <= 2), (S.NaN, True)),
        Piecewise((2, y > 2), (S.NaN, True))]

    # issue 6060
    absxm3 = Piecewise(
        (x - 3, 0 <= x - 3),
        (3 - x, 0 > x - 3)
    )
    assert solve(absxm3 - y, x) == [
        Piecewise((-y + 3, -y < 0), (S.NaN, True)),
        Piecewise((y + 3, y >= 0), (S.NaN, True))]
    p = Symbol('p', positive=True)
    assert solve(absxm3 - p, x) == [-p + 3, p + 3]

    # issue 6989
    f = Function('f')
    assert solve(Eq(-f(x), Piecewise((1, x > 0), (0, True))), f(x)) == \
        [Piecewise((-1, x > 0), (0, True))]

    # issue 8587
    f = Piecewise((2*x**2, And(0 < x, x < 1)), (2, True))
    assert solve(f - 1) == [1/sqrt(2)]


def test_piecewise_fold():
    p = Piecewise((x, x < 1), (1, 1 <= x))

    assert piecewise_fold(x*p) == Piecewise((x**2, x < 1), (x, 1 <= x))
    assert piecewise_fold(p + p) == Piecewise((2*x, x < 1), (2, 1 <= x))
    assert piecewise_fold(Piecewise((1, x < 0), (2, True))
                          + Piecewise((10, x < 0), (-10, True))) == \
        Piecewise((11, x < 0), (-8, True))

    p1 = Piecewise((0, x < 0), (x, x <= 1), (0, True))
    p2 = Piecewise((0, x < 0), (1 - x, x <= 1), (0, True))

    p = 4*p1 + 2*p2
    assert integrate(
        piecewise_fold(p), (x, -oo, oo)) == integrate(2*x + 2, (x, 0, 1))

    assert piecewise_fold(
        Piecewise((1, y <= 0), (-Piecewise((2, y >= 0)), True)
        )) == Piecewise((1, y <= 0), (-2, y >= 0))

    assert piecewise_fold(Piecewise((x, ITE(x > 0, y < 1, y > 1)))
        ) == Piecewise((x, ((x <= 0) | (y < 1)) & ((x > 0) | (y > 1))))

    a, b = (Piecewise((2, Eq(x, 0)), (0, True)),
        Piecewise((x, Eq(-x + y, 0)), (1, Eq(-x + y, 1)), (0, True)))
    assert piecewise_fold(Mul(a, b, evaluate=False)
        ) == piecewise_fold(Mul(b, a, evaluate=False))


def test_piecewise_fold_piecewise_in_cond():
    p1 = Piecewise((cos(x), x < 0), (0, True))
    p2 = Piecewise((0, Eq(p1, 0)), (p1 / Abs(p1), True))
    assert p2.subs(x, -pi/2) == 0
    assert p2.subs(x, 1) == 0
    assert p2.subs(x, -pi/4) == 1
    p4 = Piecewise((0, Eq(p1, 0)), (1,True))
    ans = piecewise_fold(p4)
    for i in range(-1, 1):
        assert ans.subs(x, i) == p4.subs(x, i)

    r1 = 1 < Piecewise((1, x < 1), (3, True))
    ans = piecewise_fold(r1)
    for i in range(2):
        assert ans.subs(x, i) == r1.subs(x, i)

    p5 = Piecewise((1, x < 0), (3, True))
    p6 = Piecewise((1, x < 1), (3, True))
    p7 = Piecewise((1, p5 < p6), (0, True))
    ans = piecewise_fold(p7)
    for i in range(-1, 2):
        assert ans.subs(x, i) == p7.subs(x, i)


def test_piecewise_fold_piecewise_in_cond_2():
    p1 = Piecewise((cos(x), x < 0), (0, True))
    p2 = Piecewise((0, Eq(p1, 0)), (1 / p1, True))
    p3 = Piecewise(
        (0, (x >= 0) | Eq(cos(x), 0)),
        (1/cos(x), x < 0),
        (zoo, True))  # redundant b/c all x are already covered
    assert(piecewise_fold(p2) == p3)


def test_piecewise_fold_expand():
    p1 = Piecewise((1, Interval(0, 1, False, True).contains(x)), (0, True))

    p2 = piecewise_fold(expand((1 - x)*p1))
    cond = ((x >= 0) & (x < 1))
    assert piecewise_fold(expand((1 - x)*p1), evaluate=False
        ) == Piecewise((1 - x, cond), (-x, cond), (1, cond), (0, True), evaluate=False)
    assert piecewise_fold(expand((1 - x)*p1), evaluate=None
        ) == Piecewise((1 - x, cond), (0, True))
    assert p2 == Piecewise((1 - x, cond), (0, True))
    assert p2 == expand(piecewise_fold((1 - x)*p1))


def test_piecewise_duplicate():
    p = Piecewise((x, x < -10), (x**2, x <= -1), (x, 1 < x))
    assert p == Piecewise(*p.args)


def test_doit():
    p1 = Piecewise((x, x < 1), (x**2, -1 <= x), (x, 3 < x))
    p2 = Piecewise((x, x < 1), (Integral(2 * x), -1 <= x), (x, 3 < x))
    assert p2.doit() == p1
    assert p2.doit(deep=False) == p2
    # issue 17165
    p1 = Sum(y**x, (x, -1, oo)).doit()
    assert p1.doit() == p1


def test_piecewise_interval():
    p1 = Piecewise((x, Interval(0, 1).contains(x)), (0, True))
    assert p1.subs(x, -0.5) == 0
    assert p1.subs(x, 0.5) == 0.5
    assert p1.diff(x) == Piecewise((1, Interval(0, 1).contains(x)), (0, True))
    assert integrate(p1, x) == Piecewise(
        (0, x <= 0),
        (x**2/2, x <= 1),
        (S.Half, True))


def test_piecewise_exclusive():
    p = Piecewise((0, x < 0), (S.Half, x <= 0), (1, True))
    assert piecewise_exclusive(p) == Piecewise((0, x < 0), (S.Half, Eq(x, 0)),
                                               (1, x > 0), evaluate=False)
    assert piecewise_exclusive(p + 2) == Piecewise((0, x < 0), (S.Half, Eq(x, 0)),
                                               (1, x > 0), evaluate=False) + 2
    assert piecewise_exclusive(Piecewise((1, y <= 0),
                                         (-Piecewise((2, y >= 0)), True))) == \
        Piecewise((1, y <= 0),
                  (-Piecewise((2, y >= 0),
                              (S.NaN, y < 0), evaluate=False), y > 0), evaluate=False)
    assert piecewise_exclusive(Piecewise((1, x > y))) == Piecewise((1, x > y),
                                                                  (S.NaN, x <= y),
                                                                  evaluate=False)
    assert piecewise_exclusive(Piecewise((1, x > y)),
                               skip_nan=True) == Piecewise((1, x > y))

    xr, yr = symbols('xr, yr', real=True)

    p1 = Piecewise((1, xr < 0), (2, True), evaluate=False)
    p1x = Piecewise((1, xr < 0), (2, xr >= 0), evaluate=False)

    p2 = Piecewise((p1, yr < 0), (3, True), evaluate=False)
    p2x = Piecewise((p1, yr < 0), (3, yr >= 0), evaluate=False)
    p2xx = Piecewise((p1x, yr < 0), (3, yr >= 0), evaluate=False)

    assert piecewise_exclusive(p2) == p2xx
    assert piecewise_exclusive(p2, deep=False) == p2x


def test_piecewise_collapse():
    assert Piecewise((x, True)) == x
    a = x < 1
    assert Piecewise((x, a), (x + 1, a)) == Piecewise((x, a))
    assert Piecewise((x, a), (x + 1, a.reversed)) == Piecewise((x, a))
    b = x < 5
    def canonical(i):
        if isinstance(i, Piecewise):
            return Piecewise(*i.args)
        return i
    for args in [
        ((1, a), (Piecewise((2, a), (3, b)), b)),
        ((1, a), (Piecewise((2, a), (3, b.reversed)), b)),
        ((1, a), (Piecewise((2, a), (3, b)), b), (4, True)),
        ((1, a), (Piecewise((2, a), (3, b), (4, True)), b)),
        ((1, a), (Piecewise((2, a), (3, b), (4, True)), b), (5, True))]:
        for i in (0, 2, 10):
            assert canonical(
                Piecewise(*args, evaluate=False).subs(x, i)
                ) == canonical(Piecewise(*args).subs(x, i))
    r1, r2, r3, r4 = symbols('r1:5')
    a = x < r1
    b = x < r2
    c = x < r3
    d = x < r4
    assert Piecewise((1, a), (Piecewise(
        (2, a), (3, b), (4, c)), b), (5, c)
        ) == Piecewise((1, a), (3, b), (5, c))
    assert Piecewise((1, a), (Piecewise(
        (2, a), (3, b), (4, c), (6, True)), c), (5, d)
        ) == Piecewise((1, a), (Piecewise(
        (3, b), (4, c)), c), (5, d))
    assert Piecewise((1, Or(a, d)), (Piecewise(
        (2, d), (3, b), (4, c)), b), (5, c)
        ) == Piecewise((1, Or(a, d)), (Piecewise(
        (2, d), (3, b)), b), (5, c))
    assert Piecewise((1, c), (2, ~c), (3, S.true)
        ) == Piecewise((1, c), (2, S.true))
    assert Piecewise((1, c), (2, And(~c, b)), (3,True)
        ) == Piecewise((1, c), (2, b), (3, True))
    assert Piecewise((1, c), (2, Or(~c, b)), (3,True)
        ).subs(dict(zip((r1, r2, r3, r4, x), (1, 2, 3, 4, 3.5))))  == 2
    assert Piecewise((1, c), (2, ~c)) == Piecewise((1, c), (2, True))


def test_piecewise_lambdify():
    p = Piecewise(
        (x**2, x < 0),
        (x, Interval(0, 1, False, True).contains(x)),
        (2 - x, x >= 1),
        (0, True)
    )

    f = lambdify(x, p)
    assert f(-2.0) == 4.0
    assert f(0.0) == 0.0
    assert f(0.5) == 0.5
    assert f(2.0) == 0.0


def test_piecewise_series():
    from sympy.series.order import O
    p1 = Piecewise((sin(x), x < 0), (cos(x), x > 0))
    p2 = Piecewise((x + O(x**2), x < 0), (1 + O(x**2), x > 0))
    assert p1.nseries(x, n=2) == p2


def test_piecewise_as_leading_term():
    p1 = Piecewise((1/x, x > 1), (0, True))
    p2 = Piecewise((x, x > 1), (0, True))
    p3 = Piecewise((1/x, x > 1), (x, True))
    p4 = Piecewise((x, x > 1), (1/x, True))
    p5 = Piecewise((1/x, x > 1), (x, True))
    p6 = Piecewise((1/x, x < 1), (x, True))
    p7 = Piecewise((x, x < 1), (1/x, True))
    p8 = Piecewise((x, x > 1), (1/x, True))
    assert p1.as_leading_term(x) == 0
    assert p2.as_leading_term(x) == 0
    assert p3.as_leading_term(x) == x
    assert p4.as_leading_term(x) == 1/x
    assert p5.as_leading_term(x) == x
    assert p6.as_leading_term(x) == 1/x
    assert p7.as_leading_term(x) == x
    assert p8.as_leading_term(x) == 1/x


def test_piecewise_complex():
    p1 = Piecewise((2, x < 0), (1, 0 <= x))
    p2 = Piecewise((2*I, x < 0), (I, 0 <= x))
    p3 = Piecewise((I*x, x > 1), (1 + I, True))
    p4 = Piecewise((-I*conjugate(x), x > 1), (1 - I, True))

    assert conjugate(p1) == p1
    assert conjugate(p2) == piecewise_fold(-p2)
    assert conjugate(p3) == p4

    assert p1.is_imaginary is False
    assert p1.is_real is True
    assert p2.is_imaginary is True
    assert p2.is_real is False
    assert p3.is_imaginary is None
    assert p3.is_real is None

    assert p1.as_real_imag() == (p1, 0)
    assert p2.as_real_imag() == (0, -I*p2)


def test_conjugate_transpose():
    A, B = symbols("A B", commutative=False)
    p = Piecewise((A*B**2, x > 0), (A**2*B, True))
    assert p.adjoint() == \
        Piecewise((adjoint(A*B**2), x > 0), (adjoint(A**2*B), True))
    assert p.conjugate() == \
        Piecewise((conjugate(A*B**2), x > 0), (conjugate(A**2*B), True))
    assert p.transpose() == \
        Piecewise((transpose(A*B**2), x > 0), (transpose(A**2*B), True))


def test_piecewise_evaluate():
    assert Piecewise((x, True)) == x
    assert Piecewise((x, True), evaluate=True) == x
    assert Piecewise((1, Eq(1, x))).args == ((1, Eq(x, 1)),)
    assert Piecewise((1, Eq(1, x)), evaluate=False).args == (
        (1, Eq(1, x)),)
    # like the additive and multiplicative identities that
    # cannot be kept in Add/Mul, we also do not keep a single True
    p = Piecewise((x, True), evaluate=False)
    assert p == x


def test_as_expr_set_pairs():
    assert Piecewise((x, x > 0), (-x, x <= 0)).as_expr_set_pairs() == \
        [(x, Interval(0, oo, True, True)), (-x, Interval(-oo, 0))]

    assert Piecewise(((x - 2)**2, x >= 0), (0, True)).as_expr_set_pairs() == \
        [((x - 2)**2, Interval(0, oo)), (0, Interval(-oo, 0, True, True))]


def test_S_srepr_is_identity():
    p = Piecewise((10, Eq(x, 0)), (12, True))
    q = S(srepr(p))
    assert p == q


def test_issue_12587():
    # sort holes into intervals
    p = Piecewise((1, x > 4), (2, Not((x <= 3) & (x > -1))), (3, True))
    assert p.integrate((x, -5, 5)) == 23
    p = Piecewise((1, x > 1), (2, x < y), (3, True))
    lim = x, -3, 3
    ans = p.integrate(lim)
    for i in range(-1, 3):
        assert ans.subs(y, i) == p.subs(y, i).integrate(lim)


def test_issue_11045():
    assert integrate(1/(x*sqrt(x**2 - 1)), (x, 1, 2)) == pi/3

    # handle And with Or arguments
    assert Piecewise((1, And(Or(x < 1, x > 3), x < 2)), (0, True)
        ).integrate((x, 0, 3)) == 1

    # hidden false
    assert Piecewise((1, x > 1), (2, x > x + 1), (3, True)
        ).integrate((x, 0, 3)) == 5
    # targetcond is Eq
    assert Piecewise((1, x > 1), (2, Eq(1, x)), (3, True)
        ).integrate((x, 0, 4)) == 6
    # And has Relational needing to be solved
    assert Piecewise((1, And(2*x > x + 1, x < 2)), (0, True)
        ).integrate((x, 0, 3)) == 1
    # Or has Relational needing to be solved
    assert Piecewise((1, Or(2*x > x + 2, x < 1)), (0, True)
        ).integrate((x, 0, 3)) == 2
    # ignore hidden false (handled in canonicalization)
    assert Piecewise((1, x > 1), (2, x > x + 1), (3, True)
        ).integrate((x, 0, 3)) == 5
    # watch for hidden True Piecewise
    assert Piecewise((2, Eq(1 - x, x*(1/x - 1))), (0, True)
        ).integrate((x, 0, 3)) == 6

    # overlapping conditions of targetcond are recognized and ignored;
    # the condition x > 3 will be pre-empted by the first condition
    assert Piecewise((1, Or(x < 1, x > 2)), (2, x > 3), (3, True)
        ).integrate((x, 0, 4)) == 6

    # convert Ne to Or
    assert Piecewise((1, Ne(x, 0)), (2, True)
        ).integrate((x, -1, 1)) == 2

    # no default but well defined
    assert Piecewise((x, (x > 1) & (x < 3)), (1, (x < 4))
        ).integrate((x, 1, 4)) == 5

    p = Piecewise((x, (x > 1) & (x < 3)), (1, (x < 4)))
    nan = Undefined
    i = p.integrate((x, 1, y))
    assert i == Piecewise(
        (y - 1, y < 1),
        (Min(3, y)**2/2 - Min(3, y) + Min(4, y) - S.Half,
            y <= Min(4, y)),
        (nan, True))
    assert p.integrate((x, 1, -1)) == i.subs(y, -1)
    assert p.integrate((x, 1, 4)) == 5
    assert p.integrate((x, 1, 5)) is nan

    # handle Not
    p = Piecewise((1, x > 1), (2, Not(And(x > 1, x< 3))), (3, True))
    assert p.integrate((x, 0, 3)) == 4

    # handle updating of int_expr when there is overlap
    p = Piecewise(
        (1, And(5 > x, x > 1)),
        (2, Or(x < 3, x > 7)),
        (4, x < 8))
    assert p.integrate((x, 0, 10)) == 20

    # And with Eq arg handling
    assert Piecewise((1, x < 1), (2, And(Eq(x, 3), x > 1))
        ).integrate((x, 0, 3)) is S.NaN
    assert Piecewise((1, x < 1), (2, And(Eq(x, 3), x > 1)), (3, True)
        ).integrate((x, 0, 3)) == 7
    assert Piecewise((1, x < 0), (2, And(Eq(x, 3), x < 1)), (3, True)
        ).integrate((x, -1, 1)) == 4
    # middle condition doesn't matter: it's a zero width interval
    assert Piecewise((1, x < 1), (2, Eq(x, 3) & (y < x)), (3, True)
        ).integrate((x, 0, 3)) == 7


def test_holes():
    nan = Undefined
    assert Piecewise((1, x < 2)).integrate(x) == Piecewise(
        (x, x < 2), (nan, True))
    assert Piecewise((1, And(x > 1, x < 2))).integrate(x) == Piecewise(
        (nan, x < 1), (x, x < 2), (nan, True))
    assert Piecewise((1, And(x > 1, x < 2))).integrate((x, 0, 3)) is nan
    assert Piecewise((1, And(x > 0, x < 4))).integrate((x, 1, 3)) == 2

    # this also tests that the integrate method is used on non-Piecwise
    # arguments in _eval_integral
    A, B = symbols("A B")
    a, b = symbols('a b', real=True)
    assert Piecewise((A, And(x < 0, a < 1)), (B, Or(x < 1, a > 2))
        ).integrate(x) == Piecewise(
        (B*x, (a > 2)),
        (Piecewise((A*x, x < 0), (B*x, x < 1), (nan, True)), a < 1),
        (Piecewise((B*x, x < 1), (nan, True)), True))


def test_issue_11922():
    def f(x):
        return Piecewise((0, x < -1), (1 - x**2, x < 1), (0, True))
    autocorr = lambda k: (
        f(x) * f(x + k)).integrate((x, -1, 1))
    assert autocorr(1.9) > 0
    k = symbols('k')
    good_autocorr = lambda k: (
        (1 - x**2) * f(x + k)).integrate((x, -1, 1))
    a = good_autocorr(k)
    assert a.subs(k, 3) == 0
    k = symbols('k', positive=True)
    a = good_autocorr(k)
    assert a.subs(k, 3) == 0
    assert Piecewise((0, x < 1), (10, (x >= 1))
        ).integrate() == Piecewise((0, x < 1), (10*x - 10, True))


def test_issue_5227():
    f = 0.0032513612725229*Piecewise((0, x < -80.8461538461539),
        (-0.0160799238820171*x + 1.33215984776403, x < 2),
        (Piecewise((0.3, x > 123), (0.7, True)) +
        Piecewise((0.4, x > 2), (0.6, True)), x <=
        123), (-0.00817409766454352*x + 2.10541401273885, x <
        380.571428571429), (0, True))
    i = integrate(f, (x, -oo, oo))
    assert i == Integral(f, (x, -oo, oo)).doit()
    assert str(i) == '1.00195081676351'
    assert Piecewise((1, x - y < 0), (0, True)
        ).integrate(y) == Piecewise((0, y <= x), (-x + y, True))


def test_issue_10137():
    a = Symbol('a', real=True)
    b = Symbol('b', real=True)
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    p0 = Piecewise((0, Or(x < a, x > b)), (1, True))
    p1 = Piecewise((0, Or(a > x, b < x)), (1, True))
    assert integrate(p0, (x, y, oo)) == integrate(p1, (x, y, oo))
    p3 = Piecewise((1, And(0 < x, x < a)), (0, True))
    p4 = Piecewise((1, And(a > x, x > 0)), (0, True))
    ip3 = integrate(p3, x)
    assert ip3 == Piecewise(
        (0, x <= 0),
        (x, x <= Max(0, a)),
        (Max(0, a), True))
    ip4 = integrate(p4, x)
    assert ip4 == ip3
    assert p3.integrate((x, 2, 4)) == Min(4, Max(2, a)) - 2
    assert p4.integrate((x, 2, 4)) == Min(4, Max(2, a)) - 2


def test_stackoverflow_43852159():
    f = lambda x: Piecewise((1, (x >= -1) & (x <= 1)), (0, True))
    Conv = lambda x: integrate(f(x - y)*f(y), (y, -oo, +oo))
    cx = Conv(x)
    assert cx.subs(x, -1.5) == cx.subs(x, 1.5)
    assert cx.subs(x, 3) == 0
    assert piecewise_fold(f(x - y)*f(y)) == Piecewise(
        (1, (y >= -1) & (y <= 1) & (x - y >= -1) & (x - y <= 1)),
        (0, True))


def test_issue_12557():
    '''
    # 3200 seconds to compute the fourier part of issue
    import sympy as sym
    x,y,z,t = sym.symbols('x y z t')
    k = sym.symbols("k", integer=True)
    fourier = sym.fourier_series(sym.cos(k*x)*sym.sqrt(x**2),
                                 (x, -sym.pi, sym.pi))
    assert fourier == FourierSeries(
    sqrt(x**2)*cos(k*x), (x, -pi, pi), (Piecewise((pi**2,
    Eq(k, 0)), (2*(-1)**k/k**2 - 2/k**2, True))/(2*pi),
    SeqFormula(Piecewise((pi**2, (Eq(_n, 0) & Eq(k, 0)) | (Eq(_n, 0) &
    Eq(_n, k) & Eq(k, 0)) | (Eq(_n, 0) & Eq(k, 0) & Eq(_n, -k)) | (Eq(_n,
    0) & Eq(_n, k) & Eq(k, 0) & Eq(_n, -k))), (pi**2/2, Eq(_n, k) | Eq(_n,
    -k) | (Eq(_n, 0) & Eq(_n, k)) | (Eq(_n, k) & Eq(k, 0)) | (Eq(_n, 0) &
    Eq(_n, -k)) | (Eq(_n, k) & Eq(_n, -k)) | (Eq(k, 0) & Eq(_n, -k)) |
    (Eq(_n, 0) & Eq(_n, k) & Eq(_n, -k)) | (Eq(_n, k) & Eq(k, 0) & Eq(_n,
    -k))), ((-1)**k*pi**2*_n**3*sin(pi*_n)/(pi*_n**4 - 2*pi*_n**2*k**2 +
    pi*k**4) - (-1)**k*pi**2*_n**3*sin(pi*_n)/(-pi*_n**4 + 2*pi*_n**2*k**2
    - pi*k**4) + (-1)**k*pi*_n**2*cos(pi*_n)/(pi*_n**4 - 2*pi*_n**2*k**2 +
    pi*k**4) - (-1)**k*pi*_n**2*cos(pi*_n)/(-pi*_n**4 + 2*pi*_n**2*k**2 -
    pi*k**4) - (-1)**k*pi**2*_n*k**2*sin(pi*_n)/(pi*_n**4 -
    2*pi*_n**2*k**2 + pi*k**4) +
    (-1)**k*pi**2*_n*k**2*sin(pi*_n)/(-pi*_n**4 + 2*pi*_n**2*k**2 -
    pi*k**4) + (-1)**k*pi*k**2*cos(pi*_n)/(pi*_n**4 - 2*pi*_n**2*k**2 +
    pi*k**4) - (-1)**k*pi*k**2*cos(pi*_n)/(-pi*_n**4 + 2*pi*_n**2*k**2 -
    pi*k**4) - (2*_n**2 + 2*k**2)/(_n**4 - 2*_n**2*k**2 + k**4),
    True))*cos(_n*x)/pi, (_n, 1, oo)), SeqFormula(0, (_k, 1, oo))))
    '''
    x = symbols("x", real=True)
    k = symbols('k', integer=True, finite=True)
    abs2 = lambda x: Piecewise((-x, x <= 0), (x, x > 0))
    assert integrate(abs2(x), (x, -pi, pi)) == pi**2
    func = cos(k*x)*sqrt(x**2)
    assert integrate(func, (x, -pi, pi)) == Piecewise(
        (2*(-1)**k/k**2 - 2/k**2, Ne(k, 0)), (pi**2, True))

def test_issue_6900():
    from itertools import permutations
    t0, t1, T, t = symbols('t0, t1 T t')
    f = Piecewise((0, t < t0), (x, And(t0 <= t, t < t1)), (0, t >= t1))
    g = f.integrate(t)
    assert g == Piecewise(
        (0, t <= t0),
        (t*x - t0*x, t <= Max(t0, t1)),
        (-t0*x + x*Max(t0, t1), True))
    for i in permutations(range(2)):
        reps = dict(zip((t0,t1), i))
        for tt in range(-1,3):
            assert (g.xreplace(reps).subs(t,tt) ==
                f.xreplace(reps).integrate(t).subs(t,tt))
    lim = Tuple(t, t0, T)
    g = f.integrate(lim)
    ans = Piecewise(
        (-t0*x + x*Min(T, Max(t0, t1)), T > t0),
        (0, True))
    for i in permutations(range(3)):
        reps = dict(zip((t0,t1,T), i))
        tru = f.xreplace(reps).integrate(lim.xreplace(reps))
        assert tru == ans.xreplace(reps)
    assert g == ans


def test_issue_10122():
    assert solve(abs(x) + abs(x - 1) - 1 > 0, x
        ) == Or(And(-oo < x, x < S.Zero), And(S.One < x, x < oo))


def test_issue_4313():
    u = Piecewise((0, x <= 0), (1, x >= a), (x/a, True))
    e = (u - u.subs(x, y))**2/(x - y)**2
    M = Max(0, a)
    assert integrate(e,  x).expand() == Piecewise(
        (Piecewise(
            (0, x <= 0),
            (-y**2/(a**2*x - a**2*y) + x/a**2 - 2*y*log(-y)/a**2 +
                2*y*log(x - y)/a**2 - y/a**2, x <= M),
            (-y**2/(-a**2*y + a**2*M) + 1/(-y + M) -
                1/(x - y) - 2*y*log(-y)/a**2 + 2*y*log(-y +
                M)/a**2 - y/a**2 + M/a**2, True)),
        ((a <= y) & (y <= 0)) | ((y <= 0) & (y > -oo))),
        (Piecewise(
            (-1/(x - y), x <= 0),
            (-a**2/(a**2*x - a**2*y) + 2*a*y/(a**2*x - a**2*y) -
                y**2/(a**2*x - a**2*y) + 2*log(-y)/a - 2*log(x - y)/a +
                2/a + x/a**2 - 2*y*log(-y)/a**2 + 2*y*log(x - y)/a**2 -
                y/a**2, x <= M),
            (-a**2/(-a**2*y + a**2*M) + 2*a*y/(-a**2*y +
                a**2*M) - y**2/(-a**2*y + a**2*M) +
                2*log(-y)/a - 2*log(-y + M)/a + 2/a -
                2*y*log(-y)/a**2 + 2*y*log(-y + M)/a**2 -
                y/a**2 + M/a**2, True)),
        a <= y),
        (Piecewise(
            (-y**2/(a**2*x - a**2*y), x <= 0),
            (x/a**2 + y/a**2, x <= M),
            (a**2/(-a**2*y + a**2*M) -
                a**2/(a**2*x - a**2*y) - 2*a*y/(-a**2*y + a**2*M) +
                2*a*y/(a**2*x - a**2*y) + y**2/(-a**2*y + a**2*M) -
                y**2/(a**2*x - a**2*y) + y/a**2 + M/a**2, True)),
        True))


def test__intervals():
    assert Piecewise((x + 2, Eq(x, 3)))._intervals(x) == (True, [])
    assert Piecewise(
        (1, x > x + 1),
        (Piecewise((1, x < x + 1)), 2*x < 2*x + 1),
        (1, True))._intervals(x) == (True, [(-oo, oo, 1, 1)])
    assert Piecewise((1, Ne(x, I)), (0, True))._intervals(x) == (True,
        [(-oo, oo, 1, 0)])
    assert Piecewise((-cos(x), sin(x) >= 0), (cos(x), True)
        )._intervals(x) == (True,
        [(0, pi, -cos(x), 0), (-oo, oo, cos(x), 1)])
    # the following tests that duplicates are removed and that non-Eq
    # generated zero-width intervals are removed
    assert Piecewise((1, Abs(x**(-2)) > 1), (0, True)
        )._intervals(x) == (True,
        [(-1, 0, 1, 0), (0, 1, 1, 0), (-oo, oo, 0, 1)])


def test_containment():
    a, b, c, d, e = [1, 2, 3, 4, 5]
    p = (Piecewise((d, x > 1), (e, True))*
        Piecewise((a, Abs(x - 1) < 1), (b, Abs(x - 2) < 2), (c, True)))
    assert p.integrate(x).diff(x) == Piecewise(
        (c*e, x <= 0),
        (a*e, x <= 1),
        (a*d, x < 2),  # this is what we want to get right
        (b*d, x < 4),
        (c*d, True))


def test_piecewise_with_DiracDelta():
    d1 = DiracDelta(x - 1)
    assert integrate(d1, (x, -oo, oo)) == 1
    assert integrate(d1, (x, 0, 2)) == 1
    assert Piecewise((d1, Eq(x, 2)), (0, True)).integrate(x) == 0
    assert Piecewise((d1, x < 2), (0, True)).integrate(x) == Piecewise(
        (Heaviside(x - 1), x < 2), (1, True))
    # TODO raise error if function is discontinuous at limit of
    # integration, e.g. integrate(d1, (x, -2, 1)) or Piecewise(
    # (d1, Eq(x, 1)


def test_issue_10258():
    assert Piecewise((0, x < 1), (1, True)).is_zero is None
    assert Piecewise((-1, x < 1), (1, True)).is_zero is False
    a = Symbol('a', zero=True)
    assert Piecewise((0, x < 1), (a, True)).is_zero
    assert Piecewise((1, x < 1), (a, x < 3)).is_zero is None
    a = Symbol('a')
    assert Piecewise((0, x < 1), (a, True)).is_zero is None
    assert Piecewise((0, x < 1), (1, True)).is_nonzero is None
    assert Piecewise((1, x < 1), (2, True)).is_nonzero
    assert Piecewise((0, x < 1), (oo, True)).is_finite is None
    assert Piecewise((0, x < 1), (1, True)).is_finite
    b = Basic()
    assert Piecewise((b, x < 1)).is_finite is None

    # 10258
    c = Piecewise((1, x < 0), (2, True)) < 3
    assert c != True
    assert piecewise_fold(c) == True


def test_issue_10087():
    a, b = Piecewise((x, x > 1), (2, True)), Piecewise((x, x > 3), (3, True))
    m = a*b
    f = piecewise_fold(m)
    for i in (0, 2, 4):
        assert m.subs(x, i) == f.subs(x, i)
    m = a + b
    f = piecewise_fold(m)
    for i in (0, 2, 4):
        assert m.subs(x, i) == f.subs(x, i)


def test_issue_8919():
    c = symbols('c:5')
    x = symbols("x")
    f1 = Piecewise((c[1], x < 1), (c[2], True))
    f2 = Piecewise((c[3], x < Rational(1, 3)), (c[4], True))
    assert integrate(f1*f2, (x, 0, 2)
        ) == c[1]*c[3]/3 + 2*c[1]*c[4]/3 + c[2]*c[4]
    f1 = Piecewise((0, x < 1), (2, True))
    f2 = Piecewise((3, x < 2), (0, True))
    assert integrate(f1*f2, (x, 0, 3)) == 6

    y = symbols("y", positive=True)
    a, b, c, x, z = symbols("a,b,c,x,z", real=True)
    I = Integral(Piecewise(
        (0, (x >= y) | (x < 0) | (b > c)),
        (a, True)), (x, 0, z))
    ans = I.doit()
    assert ans == Piecewise((0, b > c), (a*Min(y, z) - a*Min(0, z), True))
    for cond in (True, False):
        for yy in range(1, 3):
            for zz in range(-yy, 0, yy):
                reps = [(b > c, cond), (y, yy), (z, zz)]
                assert ans.subs(reps) == I.subs(reps).doit()


def test_unevaluated_integrals():
    f = Function('f')
    p = Piecewise((1, Eq(f(x) - 1, 0)), (2, x - 10 < 0), (0, True))
    assert p.integrate(x) == Integral(p, x)
    assert p.integrate((x, 0, 5)) == Integral(p, (x, 0, 5))
    # test it by replacing f(x) with x%2 which will not
    # affect the answer: the integrand is essentially 2 over
    # the domain of integration
    assert Integral(p, (x, 0, 5)).subs(f(x), x%2).n() == 10.0

    # this is a test of using _solve_inequality when
    # solve_univariate_inequality fails
    assert p.integrate(y) == Piecewise(
        (y, Eq(f(x), 1) | ((x < 10) & Eq(f(x), 1))),
        (2*y, (x > -oo) & (x < 10)), (0, True))


def test_conditions_as_alternate_booleans():
    a, b, c = symbols('a:c')
    assert Piecewise((x, Piecewise((y < 1, x > 0), (y > 1, True)))
        ) == Piecewise((x, ITE(x > 0, y < 1, y > 1)))


def test_Piecewise_rewrite_as_ITE():
    a, b, c, d = symbols('a:d')

    def _ITE(*args):
        return Piecewise(*args).rewrite(ITE)

    assert _ITE((a, x < 1), (b, x >= 1)) == ITE(x < 1, a, b)
    assert _ITE((a, x < 1), (b, x < oo)) == ITE(x < 1, a, b)
    assert _ITE((a, x < 1), (b, Or(y < 1, x < oo)), (c, y > 0)
               ) == ITE(x < 1, a, b)
    assert _ITE((a, x < 1), (b, True)) == ITE(x < 1, a, b)
    assert _ITE((a, x < 1), (b, x < 2), (c, True)
               ) == ITE(x < 1, a, ITE(x < 2, b, c))
    assert _ITE((a, x < 1), (b, y < 2), (c, True)
               ) == ITE(x < 1, a, ITE(y < 2, b, c))
    assert _ITE((a, x < 1), (b, x < oo), (c, y < 1)
               ) == ITE(x < 1, a, b)
    assert _ITE((a, x < 1), (c, y < 1), (b, x < oo), (d, True)
               ) == ITE(x < 1, a, ITE(y < 1, c, b))
    assert _ITE((a, x < 0), (b, Or(x < oo, y < 1))
               ) == ITE(x < 0, a, b)
    raises(TypeError, lambda: _ITE((x + 1, x < 1), (x, True)))
    # if `a` in the following were replaced with y then the coverage
    # is complete but something other than as_set would need to be
    # used to detect this
    raises(NotImplementedError, lambda: _ITE((x, x < y), (y, x >= a)))
    raises(ValueError, lambda: _ITE((a, x < 2), (b, x > 3)))


def test_Piecewise_replace_relational_27538():
    x, y = symbols('x, y')
    p1 = Piecewise(
        (0, Eq(x, True)),
        (1, True),
    )
    p2 = p1.xreplace({x: y < 1})
    assert p2.subs(y, 0) == 0
    assert p2.subs(y, 1) == 1


def test_issue_14052():
    assert integrate(abs(sin(x)), (x, 0, 2*pi)) == 4


def test_issue_14240():
    assert piecewise_fold(
        Piecewise((1, a), (2, b), (4, True)) +
        Piecewise((8, a), (16, True))
        ) == Piecewise((9, a), (18, b), (20, True))
    assert piecewise_fold(
        Piecewise((2, a), (3, b), (5, True)) *
        Piecewise((7, a), (11, True))
        ) == Piecewise((14, a), (33, b), (55, True))
    # these will hang if naive folding is used
    assert piecewise_fold(Add(*[
        Piecewise((i, a), (0, True)) for i in range(40)])
        ) == Piecewise((780, a), (0, True))
    assert piecewise_fold(Mul(*[
        Piecewise((i, a), (0, True)) for i in range(1, 41)])
        ) == Piecewise((factorial(40), a), (0, True))


def test_issue_14787():
    x = Symbol('x')
    f = Piecewise((x, x < 1), ((S(58) / 7), True))
    assert str(f.evalf()) == "Piecewise((x, x < 1), (8.28571428571429, True))"

def test_issue_21481():
    b, e = symbols('b e')
    C = Piecewise(
        (2,
        ((b > 1) & (e > 0)) |
        ((b > 0) & (b < 1) & (e < 0)) |
        ((e >= 2) & (b < -1) & Eq(Mod(e, 2), 0)) |
        ((e <= -2) & (b > -1) & (b < 0) & Eq(Mod(e, 2), 0))),
        (S.Half,
        ((b > 1) & (e < 0)) |
        ((b > 0) & (e > 0) & (b < 1)) |
        ((e <= -2) & (b < -1) & Eq(Mod(e, 2), 0)) |
        ((e >= 2) & (b > -1) & (b < 0) & Eq(Mod(e, 2), 0))),
        (-S.Half,
        Eq(Mod(e, 2), 1) &
        (((e <= -1) & (b < -1)) | ((e >= 1) & (b > -1) & (b < 0)))),
        (-2,
        ((e >= 1) & (b < -1) & Eq(Mod(e, 2), 1)) |
        ((e <= -1) & (b > -1) & (b < 0) & Eq(Mod(e, 2), 1)))
    )
    A = Piecewise(
        (1, Eq(b, 1) | Eq(e, 0) | (Eq(b, -1) & Eq(Mod(e, 2), 0))),
        (0, Eq(b, 0) & (e > 0)),
        (-1, Eq(b, -1) & Eq(Mod(e, 2), 1)),
        (C, Eq(im(b), 0) & Eq(im(e), 0))
    )

    B = piecewise_fold(A)
    sa = A.simplify()
    sb = B.simplify()
    v = (-2, -1, -S.Half, 0, S.Half, 1, 2)
    for i in v:
        for j in v:
            r = {b:i, e:j}
            ok = [k.xreplace(r) for k in (A, B, sa, sb)]
            assert len(set(ok)) == 1


def test_issue_8458():
    x, y = symbols('x y')
    # Original issue
    p1 = Piecewise((0, Eq(x, 0)), (sin(x), True))
    assert p1.simplify() == sin(x)
    # Slightly larger variant
    p2 = Piecewise((x, Eq(x, 0)), (4*x + (y-2)**4, Eq(x, 0) & Eq(x+y, 2)), (sin(x), True))
    assert p2.simplify() == sin(x)
    # Test for problem highlighted during review
    p3 = Piecewise((x+1, Eq(x, -1)), (4*x + (y-2)**4, Eq(x, 0) & Eq(x+y, 2)), (sin(x), True))
    assert p3.simplify() == Piecewise((0, Eq(x, -1)), (sin(x), True))


def test_issue_16417():
    z = Symbol('z')
    assert unchanged(Piecewise, (1, Or(Eq(im(z), 0), Gt(re(z), 0))), (2, True))

    x = Symbol('x')
    assert unchanged(Piecewise, (S.Pi, re(x) < 0),
                 (0, Or(re(x) > 0, Ne(im(x), 0))),
                 (S.NaN, True))
    r = Symbol('r', real=True)
    p = Piecewise((S.Pi, re(r) < 0),
                 (0, Or(re(r) > 0, Ne(im(r), 0))),
                 (S.NaN, True))
    assert p == Piecewise((S.Pi, r < 0),
                 (0, r > 0),
                 (S.NaN, True), evaluate=False)
    # Does not work since imaginary != 0...
    #i = Symbol('i', imaginary=True)
    #p = Piecewise((S.Pi, re(i) < 0),
    #              (0, Or(re(i) > 0, Ne(im(i), 0))),
    #              (S.NaN, True))
    #assert p == Piecewise((0, Ne(im(i), 0)),
    #                      (S.NaN, True), evaluate=False)
    i = I*r
    p = Piecewise((S.Pi, re(i) < 0),
                  (0, Or(re(i) > 0, Ne(im(i), 0))),
                  (S.NaN, True))
    assert p == Piecewise((0, Ne(im(i), 0)),
                          (S.NaN, True), evaluate=False)
    assert p == Piecewise((0, Ne(r, 0)),
                          (S.NaN, True), evaluate=False)


def test_eval_rewrite_as_KroneckerDelta():
    x, y, z, n, t, m = symbols('x y z n t m')
    K = KroneckerDelta
    f = lambda p: expand(p.rewrite(K))

    p1 = Piecewise((0, Eq(x, y)), (1, True))
    assert f(p1) == 1 - K(x, y)

    p2 = Piecewise((x, Eq(y,0)), (z, Eq(t,0)), (n, True))
    assert f(p2) == n*K(0, t)*K(0, y) - n*K(0, t) - n*K(0, y) + n + \
           x*K(0, y) - z*K(0, t)*K(0, y) + z*K(0, t)

    p3 = Piecewise((1, Ne(x, y)), (0, True))
    assert f(p3) == 1 - K(x, y)

    p4 = Piecewise((1, Eq(x, 3)), (4, True), (5, True))
    assert f(p4) == 4 - 3*K(3, x)

    p5 = Piecewise((3, Ne(x, 2)), (4, Eq(y, 2)), (5, True))
    assert f(p5) == -K(2, x)*K(2, y) + 2*K(2, x) + 3

    p6 = Piecewise((0, Ne(x, 1) & Ne(y, 4)), (1, True))
    assert f(p6) == -K(1, x)*K(4, y) + K(1, x) + K(4, y)

    p7 = Piecewise((2, Eq(y, 3) & Ne(x, 2)), (1, True))
    assert f(p7) == -K(2, x)*K(3, y) + K(3, y) + 1

    p8 = Piecewise((4, Eq(x, 3) & Ne(y, 2)), (1, True))
    assert f(p8) == -3*K(2, y)*K(3, x) + 3*K(3, x) + 1

    p9 = Piecewise((6, Eq(x, 4) & Eq(y, 1)), (1, True))
    assert f(p9) == 5 * K(1, y) * K(4, x) + 1

    p10 = Piecewise((4, Ne(x, -4) | Ne(y, 1)), (1, True))
    assert f(p10) == -3 * K(-4, x) * K(1, y) + 4

    p11 = Piecewise((1, Eq(y, 2) | Ne(x, -3)), (2, True))
    assert f(p11) == -K(-3, x)*K(2, y) + K(-3, x) + 1

    p12 = Piecewise((-1, Eq(x, 1) | Ne(y, 3)), (1, True))
    assert f(p12) == -2*K(1, x)*K(3, y) + 2*K(3, y) - 1

    p13 = Piecewise((3, Eq(x, 2) | Eq(y, 4)), (1, True))
    assert f(p13) == -2*K(2, x)*K(4, y) + 2*K(2, x) + 2*K(4, y) + 1

    p14 = Piecewise((1, Ne(x, 0) | Ne(y, 1)), (3, True))
    assert f(p14) == 2 * K(0, x) * K(1, y) + 1

    p15 = Piecewise((2, Eq(x, 3) | Ne(y, 2)), (3, Eq(x, 4) & Eq(y, 5)), (1, True))
    assert f(p15) == -2*K(2, y)*K(3, x)*K(4, x)*K(5, y) + K(2, y)*K(3, x) + \
           2*K(2, y)*K(4, x)*K(5, y) - K(2, y) + 2

    p16 = Piecewise((0, Ne(m, n)), (1, True))*Piecewise((0, Ne(n, t)), (1, True))\
          *Piecewise((0, Ne(n, x)), (1, True)) - Piecewise((0, Ne(t, x)), (1, True))
    assert f(p16) == K(m, n)*K(n, t)*K(n, x) - K(t, x)

    p17 = Piecewise((0, Ne(t, x) & (Ne(m, n) | Ne(n, t) | Ne(n, x))),
                    (1, Ne(t, x)), (-1, Ne(m, n) | Ne(n, t) | Ne(n, x)), (0, True))
    assert f(p17) == K(m, n)*K(n, t)*K(n, x) - K(t, x)

    p18 = Piecewise((-4, Eq(y, 1) | (Eq(x, -5) & Eq(x, z))), (4, True))
    assert f(p18) == 8*K(-5, x)*K(1, y)*K(x, z) - 8*K(-5, x)*K(x, z) - 8*K(1, y) + 4

    p19 = Piecewise((0, x > 2), (1, True))
    assert f(p19) == p19

    p20 = Piecewise((0, And(x < 2, x > -5)), (1, True))
    assert f(p20) == p20

    p21 = Piecewise((0, Or(x > 1, x < 0)), (1, True))
    assert f(p21) == p21

    p22 = Piecewise((0, ~((Eq(y, -1) | Ne(x, 0)) & (Ne(x, 1) | Ne(y, -1)))), (1, True))
    assert f(p22) == K(-1, y)*K(0, x) - K(-1, y)*K(1, x) - K(0, x) + 1


@slow
def test_identical_conds_issue():
    from sympy.stats import Uniform, density
    u1 = Uniform('u1', 0, 1)
    u2 = Uniform('u2', 0, 1)
    # Result is quite big, so not really important here (and should ideally be
    # simpler). Should not give an exception though.
    density(u1 + u2)


def test_issue_7370():
    f = Piecewise((1, x <= 2400))
    v = integrate(f, (x, 0, Float("252.4", 30)))
    assert str(v) == '252.400000000000000000000000000'


def test_issue_14933():
    x = Symbol('x')
    y = Symbol('y')

    inp = MatrixSymbol('inp', 1, 1)
    rep_dict = {y: inp[0, 0], x: inp[0, 0]}

    p = Piecewise((1, ITE(y > 0, x < 0, True)))
    assert p.xreplace(rep_dict) == Piecewise((1, ITE(inp[0, 0] > 0, inp[0, 0] < 0, True)))


def test_issue_16715():
    raises(NotImplementedError, lambda: Piecewise((x, x<0), (0, y>1)).as_expr_set_pairs())


def test_issue_20360():
    t, tau = symbols("t tau", real=True)
    n = symbols("n", integer=True)
    lam = pi * (n - S.Half)
    eq = integrate(exp(lam * tau), (tau, 0, t))
    assert eq.simplify() == (2*exp(pi*t*(2*n - 1)/2) - 2)/(pi*(2*n - 1))


def test_piecewise_eval():
    # XXX these tests might need modification if this
    # simplification is moved out of eval and into
    # boolalg or Piecewise simplification functions
    f = lambda x: x.args[0].cond
    # unsimplified
    assert f(Piecewise((x, (x > -oo) & (x < 3)))
        ) == ((x > -oo) & (x < 3))
    assert f(Piecewise((x, (x > -oo) & (x < oo)))
        ) == ((x > -oo) & (x < oo))
    assert f(Piecewise((x, (x > -3) & (x < 3)))
        ) == ((x > -3) & (x < 3))
    assert f(Piecewise((x, (x > -3) & (x < oo)))
        ) == ((x > -3) & (x < oo))
    assert f(Piecewise((x, (x <= 3) & (x > -oo)))
        ) == ((x <= 3) & (x > -oo))
    assert f(Piecewise((x, (x <= 3) & (x > -3)))
        ) == ((x <= 3) & (x > -3))
    assert f(Piecewise((x, (x >= -3) & (x < 3)))
        ) == ((x >= -3) & (x < 3))
    assert f(Piecewise((x, (x >= -3) & (x < oo)))
        ) == ((x >= -3) & (x < oo))
    assert f(Piecewise((x, (x >= -3) & (x <= 3)))
        ) == ((x >= -3) & (x <= 3))
    # could simplify by keeping only the first
    # arg of result
    assert f(Piecewise((x, (x <= oo) & (x > -oo)))
        ) == (x > -oo) & (x <= oo)
    assert f(Piecewise((x, (x <= oo) & (x > -3)))
        ) == (x > -3) & (x <= oo)
    assert f(Piecewise((x, (x >= -oo) & (x < 3)))
        ) == (x < 3) & (x >= -oo)
    assert f(Piecewise((x, (x >= -oo) & (x < oo)))
        ) == (x < oo) & (x >= -oo)
    assert f(Piecewise((x, (x >= -oo) & (x <= 3)))
        ) == (x <= 3) & (x >= -oo)
    assert f(Piecewise((x, (x >= -oo) & (x <= oo)))
        ) == (x <= oo) & (x >= -oo)  # but cannot be True unless x is real
    assert f(Piecewise((x, (x >= -3) & (x <= oo)))
        ) == (x >= -3) & (x <= oo)
    assert f(Piecewise((x, (Abs(arg(a)) <= 1) | (Abs(arg(a)) < 1)))
        ) == (Abs(arg(a)) <= 1) | (Abs(arg(a)) < 1)


def test_issue_22533():
    x = Symbol('x', real=True)
    f = Piecewise((-1 / x, x <= 0), (1 / x, True))
    assert integrate(f, x) == Piecewise((-log(x), x <= 0), (log(x), True))


def test_issue_24072():
    assert Piecewise((1, x > 1), (2, x <= 1), (3, x <= 1)
        ) == Piecewise((1, x > 1), (2, True))


def test_piecewise__eval_is_meromorphic():
    """ Issue 24127: Tests eval_is_meromorphic auxiliary method """
    x = symbols('x', real=True)
    f = Piecewise((1, x < 0), (sqrt(1 - x), True))
    assert f.is_meromorphic(x, I) is None
    assert f.is_meromorphic(x, -1) == True
    assert f.is_meromorphic(x, 0) == None
    assert f.is_meromorphic(x, 1) == False
    assert f.is_meromorphic(x, 2) == True
    assert f.is_meromorphic(x, Symbol('a')) == None
    assert f.is_meromorphic(x, Symbol('a', real=True)) == None
