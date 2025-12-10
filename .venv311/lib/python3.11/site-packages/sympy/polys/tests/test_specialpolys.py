"""Tests for functions for generating interesting polynomials. """

from sympy.core.add import Add
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.ntheory.generate import prime
from sympy.polys.domains.integerring import ZZ
from sympy.polys.polytools import Poly
from sympy.utilities.iterables import permute_signs
from sympy.testing.pytest import raises

from sympy.polys.specialpolys import (
    swinnerton_dyer_poly,
    cyclotomic_poly,
    symmetric_poly,
    random_poly,
    interpolating_poly,
    fateman_poly_F_1,
    dmp_fateman_poly_F_1,
    fateman_poly_F_2,
    dmp_fateman_poly_F_2,
    fateman_poly_F_3,
    dmp_fateman_poly_F_3,
)

from sympy.abc import x, y, z


def test_swinnerton_dyer_poly():
    raises(ValueError, lambda: swinnerton_dyer_poly(0, x))

    assert swinnerton_dyer_poly(1, x, polys=True) == Poly(x**2 - 2)

    assert swinnerton_dyer_poly(1, x) == x**2 - 2
    assert swinnerton_dyer_poly(2, x) == x**4 - 10*x**2 + 1
    assert swinnerton_dyer_poly(
        3, x) == x**8 - 40*x**6 + 352*x**4 - 960*x**2 + 576
    # we only need to check that the polys arg works but
    # we may as well test that the roots are correct
    p = [sqrt(prime(i)) for i in range(1, 5)]
    assert str([i.n(3) for i in
        swinnerton_dyer_poly(4, polys=True).all_roots()]
        ) == str(sorted([Add(*i).n(3) for i in permute_signs(p)]))


def test_cyclotomic_poly():
    raises(ValueError, lambda: cyclotomic_poly(0, x))

    assert cyclotomic_poly(1, x, polys=True) == Poly(x - 1)

    assert cyclotomic_poly(1, x) == x - 1
    assert cyclotomic_poly(2, x) == x + 1
    assert cyclotomic_poly(3, x) == x**2 + x + 1
    assert cyclotomic_poly(4, x) == x**2 + 1
    assert cyclotomic_poly(5, x) == x**4 + x**3 + x**2 + x + 1
    assert cyclotomic_poly(6, x) == x**2 - x + 1


def test_symmetric_poly():
    raises(ValueError, lambda: symmetric_poly(-1, x, y, z))
    raises(ValueError, lambda: symmetric_poly(5, x, y, z))

    assert symmetric_poly(1, x, y, z, polys=True) == Poly(x + y + z)
    assert symmetric_poly(1, (x, y, z), polys=True) == Poly(x + y + z)

    assert symmetric_poly(0, x, y, z) == 1
    assert symmetric_poly(1, x, y, z) == x + y + z
    assert symmetric_poly(2, x, y, z) == x*y + x*z + y*z
    assert symmetric_poly(3, x, y, z) == x*y*z


def test_random_poly():
    poly = random_poly(x, 10, -100, 100, polys=False)

    assert Poly(poly).degree() == 10
    assert all(-100 <= coeff <= 100 for coeff in Poly(poly).coeffs()) is True

    poly = random_poly(x, 10, -100, 100, polys=True)

    assert poly.degree() == 10
    assert all(-100 <= coeff <= 100 for coeff in poly.coeffs()) is True


def test_interpolating_poly():
    x0, x1, x2, x3, y0, y1, y2, y3 = symbols('x:4, y:4')

    assert interpolating_poly(0, x) == 0
    assert interpolating_poly(1, x) == y0

    assert interpolating_poly(2, x) == \
        y0*(x - x1)/(x0 - x1) + y1*(x - x0)/(x1 - x0)

    assert interpolating_poly(3, x) == \
        y0*(x - x1)*(x - x2)/((x0 - x1)*(x0 - x2)) + \
        y1*(x - x0)*(x - x2)/((x1 - x0)*(x1 - x2)) + \
        y2*(x - x0)*(x - x1)/((x2 - x0)*(x2 - x1))

    assert interpolating_poly(4, x) == \
        y0*(x - x1)*(x - x2)*(x - x3)/((x0 - x1)*(x0 - x2)*(x0 - x3)) + \
        y1*(x - x0)*(x - x2)*(x - x3)/((x1 - x0)*(x1 - x2)*(x1 - x3)) + \
        y2*(x - x0)*(x - x1)*(x - x3)/((x2 - x0)*(x2 - x1)*(x2 - x3)) + \
        y3*(x - x0)*(x - x1)*(x - x2)/((x3 - x0)*(x3 - x1)*(x3 - x2))

    raises(ValueError, lambda:
        interpolating_poly(2, x, (x, 2), (1, 3)))
    raises(ValueError, lambda:
        interpolating_poly(2, x, (x + y, 2), (1, 3)))
    raises(ValueError, lambda:
        interpolating_poly(2, x + y, (x, 2), (1, 3)))
    raises(ValueError, lambda:
        interpolating_poly(2, 3, (4, 5), (6, 7)))
    raises(ValueError, lambda:
        interpolating_poly(2, 3, (4, 5), (6, 7, 8)))
    assert interpolating_poly(0, x, (1, 2), (3, 4)) == 0
    assert interpolating_poly(1, x, (1, 2), (3, 4)) == 3
    assert interpolating_poly(2, x, (1, 2), (3, 4)) == x + 2


def test_fateman_poly_F_1():
    f, g, h = fateman_poly_F_1(1)
    F, G, H = dmp_fateman_poly_F_1(1, ZZ)

    assert [ t.rep.to_list() for t in [f, g, h] ] == [F, G, H]

    f, g, h = fateman_poly_F_1(3)
    F, G, H = dmp_fateman_poly_F_1(3, ZZ)

    assert [ t.rep.to_list() for t in [f, g, h] ] == [F, G, H]


def test_fateman_poly_F_2():
    f, g, h = fateman_poly_F_2(1)
    F, G, H = dmp_fateman_poly_F_2(1, ZZ)

    assert [ t.rep.to_list() for t in [f, g, h] ] == [F, G, H]

    f, g, h = fateman_poly_F_2(3)
    F, G, H = dmp_fateman_poly_F_2(3, ZZ)

    assert [ t.rep.to_list() for t in [f, g, h] ] == [F, G, H]


def test_fateman_poly_F_3():
    f, g, h = fateman_poly_F_3(1)
    F, G, H = dmp_fateman_poly_F_3(1, ZZ)

    assert [ t.rep.to_list() for t in [f, g, h] ] == [F, G, H]

    f, g, h = fateman_poly_F_3(3)
    F, G, H = dmp_fateman_poly_F_3(3, ZZ)

    assert [ t.rep.to_list() for t in [f, g, h] ] == [F, G, H]
