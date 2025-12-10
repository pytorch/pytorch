from pytest import raises
from sympy import (
    symbols, Expr, Tuple, Integer, cos, solveset, FiniteSet, ImageSet)
from sympy.plotting.utils import (
    _create_ranges, _plot_sympify, extract_solution)
from sympy.physics.mechanics import ReferenceFrame, Vector as MechVector
from sympy.vector import CoordSys3D, Vector


def test_plot_sympify():
    x, y = symbols("x, y")

    # argument is already sympified
    args = x + y
    r = _plot_sympify(args)
    assert r == args

    # one argument needs to be sympified
    args = (x + y, 1)
    r = _plot_sympify(args)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert isinstance(r[0], Expr)
    assert isinstance(r[1], Integer)

    # string and dict should not be sympified
    args = (x + y, (x, 0, 1), "str", 1, {1: 1, 2: 2.0})
    r = _plot_sympify(args)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 5
    assert isinstance(r[0], Expr)
    assert isinstance(r[1], Tuple)
    assert isinstance(r[2], str)
    assert isinstance(r[3], Integer)
    assert isinstance(r[4], dict) and isinstance(r[4][1], int) and isinstance(r[4][2], float)

    # nested arguments containing strings
    args = ((x + y, (y, 0, 1), "a"), (x + 1, (x, 0, 1), "$f_{1}$"))
    r = _plot_sympify(args)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert isinstance(r[0], Tuple)
    assert isinstance(r[0][1], Tuple)
    assert isinstance(r[0][1][1], Integer)
    assert isinstance(r[0][2], str)
    assert isinstance(r[1], Tuple)
    assert isinstance(r[1][1], Tuple)
    assert isinstance(r[1][1][1], Integer)
    assert isinstance(r[1][2], str)

    # vectors from sympy.physics.vectors module are not sympified
    # vectors from sympy.vectors are sympified
    # in both cases, no error should be raised
    R = ReferenceFrame("R")
    v1 = 2 * R.x + R.y
    C = CoordSys3D("C")
    v2 = 2 * C.i + C.j
    args = (v1, v2)
    r = _plot_sympify(args)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert isinstance(v1, MechVector)
    assert isinstance(v2, Vector)


def test_create_ranges():
    x, y = symbols("x, y")

    # user don't provide any range -> return a default range
    r = _create_ranges({x}, [], 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert isinstance(r[0], (Tuple, tuple))
    assert r[0] == (x, -10, 10)

    r = _create_ranges({x, y}, [], 2)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert isinstance(r[0], (Tuple, tuple))
    assert isinstance(r[1], (Tuple, tuple))
    assert r[0] == (x, -10, 10) or (y, -10, 10)
    assert r[1] == (y, -10, 10) or (x, -10, 10)
    assert r[0] != r[1]

    # not enough ranges provided by the user -> create default ranges
    r = _create_ranges(
        {x, y},
        [
            (x, 0, 1),
        ],
        2,
    )
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert isinstance(r[0], (Tuple, tuple))
    assert isinstance(r[1], (Tuple, tuple))
    assert r[0] == (x, 0, 1) or (y, -10, 10)
    assert r[1] == (y, -10, 10) or (x, 0, 1)
    assert r[0] != r[1]

    # too many free symbols
    raises(ValueError, lambda: _create_ranges({x, y}, [], 1))
    raises(ValueError, lambda: _create_ranges({x, y}, [(x, 0, 5), (y, 0, 1)], 1))


def test_extract_solution():
    x = symbols("x")

    sol = solveset(cos(10 * x))
    assert sol.has(ImageSet)
    res = extract_solution(sol)
    assert len(res) == 20
    assert isinstance(res, FiniteSet)

    res = extract_solution(sol, 20)
    assert len(res) == 40
    assert isinstance(res, FiniteSet)
