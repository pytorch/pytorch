#
# test_linsolve.py
#
#    Test the internal implementation of linsolve.
#

from sympy.testing.pytest import raises

from sympy.core.numbers import I
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.abc import x, y, z

from sympy.polys.matrices.linsolve import _linsolve
from sympy.polys.solvers import PolyNonlinearError


def test__linsolve():
    assert _linsolve([], [x]) == {x:x}
    assert _linsolve([S.Zero], [x]) == {x:x}
    assert _linsolve([x-1,x-2], [x]) is None
    assert _linsolve([x-1], [x]) == {x:1}
    assert _linsolve([x-1, y], [x, y]) == {x:1, y:S.Zero}
    assert _linsolve([2*I], [x]) is None
    raises(PolyNonlinearError, lambda: _linsolve([x*(1 + x)], [x]))


def test__linsolve_float():

    # This should give the exact answer:
    eqs = [
        y - x,
        y - 0.0216 * x
    ]
    sol = {x:0.0, y:0.0}
    assert _linsolve(eqs, (x, y)) == sol

    # Other cases should be close to eps

    def all_close(sol1, sol2, eps=1e-15):
        close = lambda a, b: abs(a - b) < eps
        assert sol1.keys() == sol2.keys()
        return all(close(sol1[s], sol2[s]) for s in sol1)

    eqs = [
        0.8*x +         0.8*z + 0.2,
        0.9*x + 0.7*y + 0.2*z + 0.9,
        0.7*x + 0.2*y + 0.2*z + 0.5
    ]
    sol_exact = {x:-29/42, y:-11/21, z:37/84}
    sol_linsolve = _linsolve(eqs, [x,y,z])
    assert all_close(sol_exact, sol_linsolve)

    eqs = [
        0.9*x + 0.3*y + 0.4*z + 0.6,
        0.6*x + 0.9*y + 0.1*z + 0.7,
        0.4*x + 0.6*y + 0.9*z + 0.5
    ]
    sol_exact = {x:-88/175, y:-46/105, z:-1/25}
    sol_linsolve = _linsolve(eqs, [x,y,z])
    assert all_close(sol_exact, sol_linsolve)

    eqs = [
        0.4*x + 0.3*y + 0.6*z + 0.7,
        0.4*x + 0.3*y + 0.9*z + 0.9,
        0.7*x + 0.9*y,
    ]
    sol_exact = {x:-9/5, y:7/5, z:-2/3}
    sol_linsolve = _linsolve(eqs, [x,y,z])
    assert all_close(sol_exact, sol_linsolve)

    eqs = [
        x*(0.7 + 0.6*I) + y*(0.4 + 0.7*I) + z*(0.9 + 0.1*I) + 0.5,
        0.2*I*x + 0.2*I*y + z*(0.9 + 0.2*I) + 0.1,
        x*(0.9 + 0.7*I) + y*(0.9 + 0.7*I) + z*(0.9 + 0.4*I) + 0.4,
    ]
    sol_exact = {
        x:-6157/7995 - 411/5330*I,
        y:8519/15990 + 1784/7995*I,
        z:-34/533 + 107/1599*I,
    }
    sol_linsolve = _linsolve(eqs, [x,y,z])
    assert all_close(sol_exact, sol_linsolve)

    # XXX: This system for x and y over RR(z) is problematic.
    #
    # eqs = [
    #     x*(0.2*z + 0.9) + y*(0.5*z + 0.8) + 0.6,
    #     0.1*x*z + y*(0.1*z + 0.6) + 0.9,
    # ]
    #
    # linsolve(eqs, [x, y])
    # The solution for x comes out as
    #
    #       -3.9e-5*z**2 - 3.6e-5*z - 8.67361737988404e-20
    #  x =  ----------------------------------------------
    #           3.0e-6*z**3 - 1.3e-5*z**2 - 5.4e-5*z
    #
    # The 8e-20 in the numerator should be zero which would allow z to cancel
    # from top and bottom. It should be possible to avoid this somehow because
    # the inverse of the matrix only has a quadratic factor (the determinant)
    # in the denominator.


def test__linsolve_deprecated():
    raises(PolyNonlinearError, lambda:
        _linsolve([Eq(x**2, x**2 + y)], [x, y]))
    raises(PolyNonlinearError, lambda:
        _linsolve([(x + y)**2 - x**2], [x]))
    raises(PolyNonlinearError, lambda:
        _linsolve([Eq((x + y)**2, x**2)], [x]))
