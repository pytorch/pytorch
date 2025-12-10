"""Tests for solvers of systems of polynomial equations. """
from sympy.polys.domains import  ZZ, QQ_I
from sympy.core.numbers import (I, Integer, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.polyerrors import UnsolvableFactorError
from sympy.polys.polyoptions import Options
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import CRootOf
from sympy.solvers.solvers import solve
from sympy.utilities.iterables import flatten
from sympy.abc import a, b, c, x, y, z
from sympy.polys import PolynomialError
from sympy.solvers.polysys import (solve_poly_system,
                                   solve_triangulated,
                                   solve_biquadratic, SolveFailed,
                                   solve_generic, factor_system_bool,
                                   factor_system_cond, factor_system_poly,
                                   factor_system, _factor_sets, _factor_sets_slow)
from sympy.polys.polytools import parallel_poly_from_expr
from sympy.testing.pytest import raises
from sympy.core.relational import Eq
from sympy.functions.elementary.trigonometric import sin, cos

from sympy.functions.elementary.exponential import exp


def test_solve_poly_system():
    assert solve_poly_system([x - 1], x) == [(S.One,)]

    assert solve_poly_system([y - x, y - x - 1], x, y) is None

    assert solve_poly_system([y - x**2, y + x**2], x, y) == [(S.Zero, S.Zero)]

    assert solve_poly_system([2*x - 3, y*Rational(3, 2) - 2*x, z - 5*y], x, y, z) == \
        [(Rational(3, 2), Integer(2), Integer(10))]

    assert solve_poly_system([x*y - 2*y, 2*y**2 - x**2], x, y) == \
        [(0, 0), (2, -sqrt(2)), (2, sqrt(2))]

    assert solve_poly_system([y - x**2, y + x**2 + 1], x, y) == \
        [(-I*sqrt(S.Half), Rational(-1, 2)), (I*sqrt(S.Half), Rational(-1, 2))]

    f_1 = x**2 + y + z - 1
    f_2 = x + y**2 + z - 1
    f_3 = x + y + z**2 - 1

    a, b = sqrt(2) - 1, -sqrt(2) - 1

    assert solve_poly_system([f_1, f_2, f_3], x, y, z) == \
        [(0, 0, 1), (0, 1, 0), (1, 0, 0), (a, a, a), (b, b, b)]

    solution = [(1, -1), (1, 1)]

    assert solve_poly_system([Poly(x**2 - y**2), Poly(x - 1)]) == solution
    assert solve_poly_system([x**2 - y**2, x - 1], x, y) == solution
    assert solve_poly_system([x**2 - y**2, x - 1]) == solution

    assert solve_poly_system(
        [x + x*y - 3, y + x*y - 4], x, y) == [(-3, -2), (1, 2)]

    raises(NotImplementedError, lambda: solve_poly_system([x**3 - y**3], x, y))
    raises(NotImplementedError, lambda: solve_poly_system(
        [z, -2*x*y**2 + x + y**2*z, y**2*(-z - 4) + 2]))
    raises(PolynomialError, lambda: solve_poly_system([1/x], x))

    raises(NotImplementedError, lambda: solve_poly_system(
          [x-1,], (x, y)))
    raises(NotImplementedError, lambda: solve_poly_system(
          [y-1,], (x, y)))

    # solve_poly_system should ideally construct solutions using
    # CRootOf for the following four tests
    assert solve_poly_system([x**5 - x + 1], [x], strict=False) == []
    raises(UnsolvableFactorError, lambda: solve_poly_system(
        [x**5 - x + 1], [x], strict=True))

    assert solve_poly_system([(x - 1)*(x**5 - x + 1), y**2 - 1], [x, y],
                             strict=False) == [(1, -1), (1, 1)]
    raises(UnsolvableFactorError,
           lambda: solve_poly_system([(x - 1)*(x**5 - x + 1), y**2-1],
                                     [x, y], strict=True))


def test_solve_generic():
    NewOption = Options((x, y), {'domain': 'ZZ'})
    assert solve_generic([x**2 - 2*y**2, y**2 - y + 1], NewOption) == \
           [(-sqrt(-1 - sqrt(3)*I), Rational(1, 2) - sqrt(3)*I/2),
            (sqrt(-1 - sqrt(3)*I), Rational(1, 2) - sqrt(3)*I/2),
            (-sqrt(-1 + sqrt(3)*I), Rational(1, 2) + sqrt(3)*I/2),
            (sqrt(-1 + sqrt(3)*I), Rational(1, 2) + sqrt(3)*I/2)]

    # solve_generic should ideally construct solutions using
    # CRootOf for the following two tests
    assert solve_generic(
        [2*x - y, (y - 1)*(y**5 - y + 1)], NewOption, strict=False) == \
        [(Rational(1, 2), 1)]
    raises(UnsolvableFactorError, lambda: solve_generic(
        [2*x - y, (y - 1)*(y**5 - y + 1)], NewOption, strict=True))


def test_solve_biquadratic():
    x0, y0, x1, y1, r = symbols('x0 y0 x1 y1 r')

    f_1 = (x - 1)**2 + (y - 1)**2 - r**2
    f_2 = (x - 2)**2 + (y - 2)**2 - r**2
    s = sqrt(2*r**2 - 1)
    a = (3 - s)/2
    b = (3 + s)/2
    assert solve_poly_system([f_1, f_2], x, y) == [(a, b), (b, a)]

    f_1 = (x - 1)**2 + (y - 2)**2 - r**2
    f_2 = (x - 1)**2 + (y - 1)**2 - r**2

    assert solve_poly_system([f_1, f_2], x, y) == \
        [(1 - sqrt((2*r - 1)*(2*r + 1))/2, Rational(3, 2)),
         (1 + sqrt((2*r - 1)*(2*r + 1))/2, Rational(3, 2))]

    query = lambda expr: expr.is_Pow and expr.exp is S.Half

    f_1 = (x - 1 )**2 + (y - 2)**2 - r**2
    f_2 = (x - x1)**2 + (y - 1)**2 - r**2

    result = solve_poly_system([f_1, f_2], x, y)

    assert len(result) == 2 and all(len(r) == 2 for r in result)
    assert all(r.count(query) == 1 for r in flatten(result))

    f_1 = (x - x0)**2 + (y - y0)**2 - r**2
    f_2 = (x - x1)**2 + (y - y1)**2 - r**2

    result = solve_poly_system([f_1, f_2], x, y)

    assert len(result) == 2 and all(len(r) == 2 for r in result)
    assert all(len(r.find(query)) == 1 for r in flatten(result))

    s1 = (x*y - y, x**2 - x)
    assert solve(s1) == [{x: 1}, {x: 0, y: 0}]
    s2 = (x*y - x, y**2 - y)
    assert solve(s2) == [{y: 1}, {x: 0, y: 0}]
    gens = (x, y)
    for seq in (s1, s2):
        (f, g), opt = parallel_poly_from_expr(seq, *gens)
        raises(SolveFailed, lambda: solve_biquadratic(f, g, opt))
    seq = (x**2 + y**2 - 2, y**2 - 1)
    (f, g), opt = parallel_poly_from_expr(seq, *gens)
    assert solve_biquadratic(f, g, opt) == [
        (-1, -1), (-1, 1), (1, -1), (1, 1)]
    ans = [(0, -1), (0, 1)]
    seq = (x**2 + y**2 - 1, y**2 - 1)
    (f, g), opt = parallel_poly_from_expr(seq, *gens)
    assert solve_biquadratic(f, g, opt) == ans
    seq = (x**2 + y**2 - 1, x**2 - x + y**2 - 1)
    (f, g), opt = parallel_poly_from_expr(seq, *gens)
    assert solve_biquadratic(f, g, opt) == ans


def test_solve_triangulated():
    f_1 = x**2 + y + z - 1
    f_2 = x + y**2 + z - 1
    f_3 = x + y + z**2 - 1

    a, b = sqrt(2) - 1, -sqrt(2) - 1

    assert solve_triangulated([f_1, f_2, f_3], x, y, z) == \
        [(0, 0, 1), (0, 1, 0), (1, 0, 0)]

    dom = QQ.algebraic_field(sqrt(2))

    assert solve_triangulated([f_1, f_2, f_3], x, y, z, domain=dom) == \
        [(0, 0, 1), (0, 1, 0), (1, 0, 0), (a, a, a), (b, b, b)]

    a, b = CRootOf(z**2 + 2*z - 1, 0), CRootOf(z**2 + 2*z - 1, 1)
    assert solve_triangulated([f_1, f_2, f_3], x, y, z, extension=True) == \
        [(0, 0, 1), (0, 1, 0), (1, 0, 0), (a, a, a), (b, b, b)]


def test_solve_issue_3686():
    roots = solve_poly_system([((x - 5)**2/250000 + (y - Rational(5, 10))**2/250000) - 1, x], x, y)
    assert roots == [(0, S.Half - 15*sqrt(1111)), (0, S.Half + 15*sqrt(1111))]

    roots = solve_poly_system([((x - 5)**2/250000 + (y - 5.0/10)**2/250000) - 1, x], x, y)
    # TODO: does this really have to be so complicated?!
    assert len(roots) == 2
    assert roots[0][0] == 0
    assert roots[0][1].epsilon_eq(-499.474999374969, 1e12)
    assert roots[1][0] == 0
    assert roots[1][1].epsilon_eq(500.474999374969, 1e12)


def test_factor_system():

    assert factor_system([x**2 + 2*x + 1]) ==  [[x + 1]]
    assert factor_system([x**2 + 2*x + 1, y**2 + 2*y + 1]) ==  [[x + 1, y + 1]]
    assert factor_system([x**2 + 1]) ==  [[x**2 + 1]]
    assert factor_system([]) == [[]]

    assert factor_system([x**2 + y**2 + 2*x*y, x**2 - 2], extension=sqrt(2)) == [
        [x + y, x + sqrt(2)],
        [x + y, x - sqrt(2)],
    ]

    assert factor_system([x**2 + 1, y**2 + 1], gaussian=True) == [
        [x + I, y + I],
        [x + I, y - I],
        [x - I, y + I],
        [x - I, y - I],
    ]

    assert factor_system([x**2 + 1, y**2 + 1], domain=QQ_I) == [
        [x + I, y + I],
        [x + I, y - I],
        [x - I, y + I],
        [x - I, y - I],
    ]

    assert factor_system([0]) == [[]]
    assert factor_system([1]) == []
    assert factor_system([0 , x]) == [[x]]
    assert factor_system([1, 0, x]) == []

    assert factor_system([x**4 - 1, y**6 - 1]) == [
        [x**2 + 1, y**2 + y + 1],
        [x**2 + 1, y**2 - y + 1],
        [x**2 + 1, y + 1],
        [x**2 + 1, y - 1],
        [x + 1, y**2 + y + 1],
        [x + 1, y**2 - y + 1],
        [x - 1, y**2 + y + 1],
        [x - 1, y**2 - y + 1],
        [x + 1, y + 1],
        [x + 1, y - 1],
        [x - 1, y + 1],
        [x - 1, y - 1],
    ]

    assert factor_system([(x - 1)*(y - 2), (y - 2)*(z - 3)]) == [
        [x - 1, z - 3],
        [y - 2]
    ]

    assert factor_system([sin(x)**2 + cos(x)**2 - 1, x]) == [
        [x, sin(x)**2 + cos(x)**2 - 1],
    ]

    assert factor_system([sin(x)**2 + cos(x)**2 - 1]) == [
        [sin(x)**2 + cos(x)**2 - 1]
    ]

    assert factor_system([sin(x)**2 + cos(x)**2]) == [
        [sin(x)**2 + cos(x)**2]
    ]

    assert factor_system([a*x, y, a]) == [[y, a]]

    assert factor_system([a*x, y, a], [x, y]) == []

    assert factor_system([a ** 2 * x, y], [x, y]) == [[x, y]]

    assert factor_system([a*x*(x - 1), b*y, c], [x, y]) == []

    assert factor_system([a*x*(x - 1), b*y, c], [x, y, c]) == [
        [x - 1, y, c],
        [x, y, c],
    ]

    assert factor_system([a*x*(x - 1), b*y, c]) == [
        [x - 1, y, c],
        [x, y, c],
        [x - 1, b, c],
        [x, b, c],
        [y, a, c],
        [a, b, c],
    ]

    assert factor_system([x**2 - 2], [y]) == []

    assert factor_system([x**2 - 2], [x]) == [[x**2 - 2]]

    assert factor_system([cos(x)**2 - sin(x)**2, cos(x)**2 + sin(x)**2 - 1]) == [
        [sin(x)**2 + cos(x)**2 - 1, sin(x) + cos(x)],
        [sin(x)**2 + cos(x)**2 - 1, -sin(x) + cos(x)],
    ]

    assert factor_system([(cos(x) + sin(x))**2 - 1, cos(x)**2 - sin(x)**2 - cos(2*x)]) == [
        [sin(x)**2 - cos(x)**2 + cos(2*x), sin(x) + cos(x) + 1],
        [sin(x)**2 - cos(x)**2 + cos(2*x), sin(x) + cos(x) - 1],
    ]

    assert factor_system([(cos(x) + sin(x))*exp(y) - 1, (cos(x) - sin(x))*exp(y) - 1]) == [
        [exp(y)*sin(x) + exp(y)*cos(x) - 1, -exp(y)*sin(x) + exp(y)*cos(x) - 1]
    ]


def test_factor_system_poly():

    px = lambda e: Poly(e, x)
    pxab = lambda e: Poly(e, x, domain=ZZ[a, b])
    pxI = lambda e: Poly(e, x, domain=QQ_I)
    pxyz = lambda e: Poly(e, (x, y, z))

    assert factor_system_poly([px(x**2 - 1), px(x**2 - 4)]) == [
        [px(x + 2), px(x + 1)],
        [px(x + 2), px(x - 1)],
        [px(x + 1), px(x - 2)],
        [px(x - 1), px(x - 2)],
    ]

    assert factor_system_poly([px(x**2 - 1)]) == [[px(x + 1)], [px(x - 1)]]

    assert factor_system_poly([pxyz(x**2*y - y), pxyz(x**2*z - z)]) == [
        [pxyz(x + 1)],
        [pxyz(x - 1)],
        [pxyz(y), pxyz(z)],
    ]

    assert factor_system_poly([px(x**2*(x - 1)**2), px(x*(x - 1))]) == [
        [px(x)],
        [px(x - 1)],
    ]

    assert factor_system_poly([pxyz(x**2 + y*x), pxyz(x**2 + z*x)]) == [
        [pxyz(x + y), pxyz(x + z)],
        [pxyz(x)],
    ]

    assert factor_system_poly([pxab((a - 1)*(x - 2)), pxab((b - 3)*(x - 2))]) == [
        [pxab(x - 2)],
        [pxab(a - 1), pxab(b - 3)],
    ]

    assert factor_system_poly([pxI(x**2 + 1)]) == [[pxI(x + I)], [pxI(x - I)]]

    assert factor_system_poly([]) == [[]]

    assert factor_system_poly([px(1)]) == []
    assert factor_system_poly([px(0), px(x)]) == [[px(x)]]


def test_factor_system_cond():

    assert factor_system_cond([x ** 2 - 1, x ** 2 - 4]) == [
        [x + 2, x + 1],
        [x + 2, x - 1],
        [x + 1, x - 2],
        [x - 1, x - 2],
    ]

    assert factor_system_cond([1]) == []
    assert factor_system_cond([0]) == [[]]
    assert factor_system_cond([1, x]) == []
    assert factor_system_cond([0, x]) == [[x]]
    assert factor_system_cond([]) == [[]]

    assert factor_system_cond([x**2 + y*x]) == [[x + y], [x]]

    assert factor_system_cond([(a - 1)*(x - 2), (b - 3)*(x - 2)], [x]) == [
        [x - 2],
        [a - 1, b - 3],
    ]

    assert factor_system_cond([a * (x - 1), b], [x]) == [[x - 1, b], [a, b]]

    assert factor_system_cond([a*x*(x-1), b*y, c], [x, y]) == [
        [x - 1, y, c],
        [x, y, c],
        [x - 1, b, c],
        [x, b, c],
        [y, a, c],
        [a, b, c],
    ]

    assert factor_system_cond([x*(x-1), y], [x, y]) == [[x - 1, y], [x, y]]

    assert factor_system_cond([a*x, y, a], [x, y]) == [[y, a]]

    assert factor_system_cond([a*x, b*x], [x, y]) == [[x], [a, b]]

    assert factor_system_cond([a*b*x, y], [x, y]) == [[x, y], [y, a*b]]

    assert factor_system_cond([a*b*x, y]) == [[x, y], [y, a], [y, b]]

    assert factor_system_cond([a**2*x, y], [x, y]) == [[x, y], [y, a]]

def test_factor_system_bool():

    eqs = [a*(x - 1)*(y - 1), b*(x - 2)*(y - 1)*(y - 2)]
    assert factor_system_bool(eqs, [x, y]) == (
        Eq(y - 1, 0)
        | (Eq(a, 0) & Eq(b, 0))
        | (Eq(a, 0) & Eq(x - 2, 0))
        | (Eq(a, 0) & Eq(y - 2, 0))
        | (Eq(b, 0) & Eq(x - 1, 0))
        | (Eq(x - 2, 0) & Eq(x - 1, 0))
        | (Eq(x - 1, 0) & Eq(y - 2, 0))
    )

    assert factor_system_bool([x - 1], [x]) == Eq(x - 1, 0)

    assert factor_system_bool([(x - 1)*(x - 2)], [x]) == Eq(x - 2, 0) | Eq(x - 1, 0)

    assert factor_system_bool([], [x]) == True
    assert factor_system_bool([0], [x]) == True
    assert factor_system_bool([1], [x]) == False
    assert factor_system_bool([a], [x]) == Eq(a, 0)

    assert factor_system_bool([a * x, y, a], [x, y]) == Eq(a, 0) & Eq(y, 0)

    assert (factor_system_bool([a*x, b*y*x, a], [x, y]) == (
        Eq(a, 0) & Eq(b, 0))
        | (Eq(a, 0) & Eq(x, 0))
        | (Eq(a, 0) & Eq(y, 0)))

    assert (factor_system_bool([a*x, b*x], [x, y]) == Eq(x, 0) |
            (Eq(a, 0) & Eq(b, 0)))

    assert (factor_system_bool([a*b*x, y], [x, y]) == (
        Eq(x, 0) & Eq(y, 0)) |
        (Eq(y, 0) & Eq(a*b, 0)))

    assert (factor_system_bool([a**2*x, y], [x, y]) == (
        Eq(a, 0) & Eq(y, 0)) |
        (Eq(x, 0) & Eq(y, 0)))

    assert factor_system_bool([a*x*y, b*y*z], [x, y, z]) == (
        Eq(y, 0)
        | (Eq(a, 0) & Eq(b, 0))
        | (Eq(a, 0) & Eq(z, 0))
        | (Eq(b, 0) & Eq(x, 0))
        | (Eq(x, 0) & Eq(z, 0))
    )

    assert factor_system_bool([a*(x - 1), b], [x]) == (
        (Eq(a, 0) & Eq(b, 0))
        | (Eq(x - 1, 0) & Eq(b, 0))
    )


def test_factor_sets():
    #
    from random import randint

    def generate_random_system(n_eqs=3, n_factors=2, max_val=10):
        return [
            [randint(0, max_val) for _ in range(randint(1, n_factors))]
            for _ in range(n_eqs)
        ]

    test_cases = [
        [[1, 2], [1, 3]],
        [[1, 2], [3, 4]],
        [[1], [1, 2], [2]],
    ]

    for case in test_cases:
        assert _factor_sets(case) == _factor_sets_slow(case)

    for _ in range(100):
        system = generate_random_system()
        assert _factor_sets(system) == _factor_sets_slow(system)
