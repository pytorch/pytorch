from sympy.functions import bspline_basis_set, interpolating_spline
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import And
from sympy.sets.sets import Interval
from sympy.testing.pytest import slow

x, y = symbols('x,y')


def test_basic_degree_0():
    d = 0
    knots = range(5)
    splines = bspline_basis_set(d, knots, x)
    for i in range(len(splines)):
        assert splines[i] == Piecewise((1, Interval(i, i + 1).contains(x)),
                                       (0, True))


def test_basic_degree_1():
    d = 1
    knots = range(5)
    splines = bspline_basis_set(d, knots, x)
    assert splines[0] == Piecewise((x, Interval(0, 1).contains(x)),
                                   (2 - x, Interval(1, 2).contains(x)),
                                   (0, True))
    assert splines[1] == Piecewise((-1 + x, Interval(1, 2).contains(x)),
                                   (3 - x, Interval(2, 3).contains(x)),
                                   (0, True))
    assert splines[2] == Piecewise((-2 + x, Interval(2, 3).contains(x)),
                                   (4 - x, Interval(3, 4).contains(x)),
                                   (0, True))


def test_basic_degree_2():
    d = 2
    knots = range(5)
    splines = bspline_basis_set(d, knots, x)
    b0 = Piecewise((x**2/2, Interval(0, 1).contains(x)),
                   (Rational(-3, 2) + 3*x - x**2, Interval(1, 2).contains(x)),
                   (Rational(9, 2) - 3*x + x**2/2, Interval(2, 3).contains(x)),
                   (0, True))
    b1 = Piecewise((S.Half - x + x**2/2, Interval(1, 2).contains(x)),
                   (Rational(-11, 2) + 5*x - x**2, Interval(2, 3).contains(x)),
                   (8 - 4*x + x**2/2, Interval(3, 4).contains(x)),
                   (0, True))
    assert splines[0] == b0
    assert splines[1] == b1


def test_basic_degree_3():
    d = 3
    knots = range(5)
    splines = bspline_basis_set(d, knots, x)
    b0 = Piecewise(
        (x**3/6, Interval(0, 1).contains(x)),
        (Rational(2, 3) - 2*x + 2*x**2 - x**3/2, Interval(1, 2).contains(x)),
        (Rational(-22, 3) + 10*x - 4*x**2 + x**3/2, Interval(2, 3).contains(x)),
        (Rational(32, 3) - 8*x + 2*x**2 - x**3/6, Interval(3, 4).contains(x)),
        (0, True)
    )
    assert splines[0] == b0


def test_repeated_degree_1():
    d = 1
    knots = [0, 0, 1, 2, 2, 3, 4, 4]
    splines = bspline_basis_set(d, knots, x)
    assert splines[0] == Piecewise((1 - x, Interval(0, 1).contains(x)),
                                   (0, True))
    assert splines[1] == Piecewise((x, Interval(0, 1).contains(x)),
                                   (2 - x, Interval(1, 2).contains(x)),
                                   (0, True))
    assert splines[2] == Piecewise((-1 + x, Interval(1, 2).contains(x)),
                                   (0, True))
    assert splines[3] == Piecewise((3 - x, Interval(2, 3).contains(x)),
                                   (0, True))
    assert splines[4] == Piecewise((-2 + x, Interval(2, 3).contains(x)),
                                   (4 - x, Interval(3, 4).contains(x)),
                                   (0, True))
    assert splines[5] == Piecewise((-3 + x, Interval(3, 4).contains(x)),
                                   (0, True))


def test_repeated_degree_2():
    d = 2
    knots = [0, 0, 1, 2, 2, 3, 4, 4]
    splines = bspline_basis_set(d, knots, x)

    assert splines[0] == Piecewise(((-3*x**2/2 + 2*x), And(x <= 1, x >= 0)),
                                   (x**2/2 - 2*x + 2, And(x <= 2, x >= 1)),
                                   (0, True))
    assert splines[1] == Piecewise((x**2/2, And(x <= 1, x >= 0)),
                                   (-3*x**2/2 + 4*x - 2, And(x <= 2, x >= 1)),
                                   (0, True))
    assert splines[2] == Piecewise((x**2 - 2*x + 1, And(x <= 2, x >= 1)),
                                   (x**2 - 6*x + 9, And(x <= 3, x >= 2)),
                                   (0, True))
    assert splines[3] == Piecewise((-3*x**2/2 + 8*x - 10, And(x <= 3, x >= 2)),
                                   (x**2/2 - 4*x + 8, And(x <= 4, x >= 3)),
                                   (0, True))
    assert splines[4] == Piecewise((x**2/2 - 2*x + 2, And(x <= 3, x >= 2)),
                                   (-3*x**2/2 + 10*x - 16, And(x <= 4, x >= 3)),
                                   (0, True))

# Tests for interpolating_spline


def test_10_points_degree_1():
    d = 1
    X = [-5, 2, 3, 4, 7, 9, 10, 30, 31, 34]
    Y = [-10, -2, 2, 4, 7, 6, 20, 45, 19, 25]
    spline = interpolating_spline(d, x, X, Y)

    assert spline == Piecewise((x*Rational(8, 7) - Rational(30, 7), (x >= -5) & (x <= 2)), (4*x - 10, (x >= 2) & (x <= 3)),
                               (2*x - 4, (x >= 3) & (x <= 4)), (x, (x >= 4) & (x <= 7)),
                               (-x/2 + Rational(21, 2), (x >= 7) & (x <= 9)), (14*x - 120, (x >= 9) & (x <= 10)),
                               (x*Rational(5, 4) + Rational(15, 2), (x >= 10) & (x <= 30)), (-26*x + 825, (x >= 30) & (x <= 31)),
                               (2*x - 43, (x >= 31) & (x <= 34)))


def test_3_points_degree_2():
    d = 2
    X = [-3, 10, 19]
    Y = [3, -4, 30]
    spline = interpolating_spline(d, x, X, Y)

    assert spline == Piecewise((505*x**2/2574 - x*Rational(4921, 2574) - Rational(1931, 429), (x >= -3) & (x <= 19)))


def test_5_points_degree_2():
    d = 2
    X = [-3, 2, 4, 5, 10]
    Y = [-1, 2, 5, 10, 14]
    spline = interpolating_spline(d, x, X, Y)

    assert spline == Piecewise((4*x**2/329 + x*Rational(1007, 1645) + Rational(1196, 1645), (x >= -3) & (x <= 3)),
                               (2701*x**2/1645 - x*Rational(15079, 1645) + Rational(5065, 329), (x >= 3) & (x <= Rational(9, 2))),
                               (-1319*x**2/1645 + x*Rational(21101, 1645) - Rational(11216, 329), (x >= Rational(9, 2)) & (x <= 10)))


@slow
def test_6_points_degree_3():
    d = 3
    X = [-1, 0, 2, 3, 9, 12]
    Y = [-4, 3, 3, 7, 9, 20]
    spline = interpolating_spline(d, x, X, Y)

    assert spline == Piecewise((6058*x**3/5301 - 18427*x**2/5301 + x*Rational(12622, 5301) + 3, (x >= -1) & (x <= 2)),
                               (-8327*x**3/5301 + 67883*x**2/5301 - x*Rational(159998, 5301) + Rational(43661, 1767), (x >= 2) & (x <= 3)),
                               (5414*x**3/47709 - 1386*x**2/589 + x*Rational(4267, 279) - Rational(12232, 589), (x >= 3) & (x <= 12)))


def test_issue_19262():
    Delta = symbols('Delta', positive=True)
    knots = [i*Delta for i in range(4)]
    basis = bspline_basis_set(1, knots, x)
    y = symbols('y', nonnegative=True)
    basis2 = bspline_basis_set(1, knots, y)
    assert basis[0].subs(x, y) == basis2[0]
    assert interpolating_spline(1, x,
        [Delta*i for i in [1, 2, 4, 7]], [3, 6, 5, 7]
        )  == Piecewise((3*x/Delta, (Delta <= x) & (x <= 2*Delta)),
        (7 - x/(2*Delta), (x >= 2*Delta) & (x <= 4*Delta)),
        (Rational(7, 3) + 2*x/(3*Delta), (x >= 4*Delta) & (x <= 7*Delta)))
