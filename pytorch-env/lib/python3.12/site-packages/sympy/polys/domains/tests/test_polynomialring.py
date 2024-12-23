"""Tests for the PolynomialRing classes. """

from sympy.polys.domains import QQ, ZZ
from sympy.polys.polyerrors import ExactQuotientFailed, CoercionFailed, NotReversible

from sympy.abc import x, y

from sympy.testing.pytest import raises


def test_build_order():
    R = QQ.old_poly_ring(x, y, order=(("lex", x), ("ilex", y)))
    assert R.order((1, 5)) == ((1,), (-5,))


def test_globalring():
    Qxy = QQ.old_frac_field(x, y)
    R = QQ.old_poly_ring(x, y)
    X = R.convert(x)
    Y = R.convert(y)

    assert x in R
    assert 1/x not in R
    assert 1/(1 + x) not in R
    assert Y in R
    assert X * (Y**2 + 1) == R.convert(x * (y**2 + 1))
    assert X + 1 == R.convert(x + 1)
    raises(ExactQuotientFailed, lambda: X/Y)
    raises(TypeError, lambda: x/Y)
    raises(TypeError, lambda: X/y)
    assert X**2 / X == X

    assert R.from_GlobalPolynomialRing(ZZ.old_poly_ring(x, y).convert(x), ZZ.old_poly_ring(x, y)) == X
    assert R.from_FractionField(Qxy.convert(x), Qxy) == X
    assert R.from_FractionField(Qxy.convert(x/y), Qxy) is None

    assert R._sdm_to_vector(R._vector_to_sdm([X, Y], R.order), 2) == [X, Y]


def test_localring():
    Qxy = QQ.old_frac_field(x, y)
    R = QQ.old_poly_ring(x, y, order="ilex")
    X = R.convert(x)
    Y = R.convert(y)

    assert x in R
    assert 1/x not in R
    assert 1/(1 + x) in R
    assert Y in R
    assert X*(Y**2 + 1)/(1 + X) == R.convert(x*(y**2 + 1)/(1 + x))
    raises(TypeError, lambda: x/Y)
    raises(TypeError, lambda: X/y)
    assert X + 1 == R.convert(x + 1)
    assert X**2 / X == X

    assert R.from_GlobalPolynomialRing(ZZ.old_poly_ring(x, y).convert(x), ZZ.old_poly_ring(x, y)) == X
    assert R.from_FractionField(Qxy.convert(x), Qxy) == X
    raises(CoercionFailed, lambda: R.from_FractionField(Qxy.convert(x/y), Qxy))
    raises(ExactQuotientFailed, lambda: R.exquo(X, Y))
    raises(NotReversible, lambda: R.revert(X))

    assert R._sdm_to_vector(
        R._vector_to_sdm([X/(X + 1), Y/(1 + X*Y)], R.order), 2) == \
        [X*(1 + X*Y), Y*(1 + X)]


def test_conversion():
    L = QQ.old_poly_ring(x, y, order="ilex")
    G = QQ.old_poly_ring(x, y)

    assert L.convert(x) == L.convert(G.convert(x), G)
    assert G.convert(x) == G.convert(L.convert(x), L)
    raises(CoercionFailed, lambda: G.convert(L.convert(1/(1 + x)), L))


def test_units():
    R = QQ.old_poly_ring(x)
    assert R.is_unit(R.convert(1))
    assert R.is_unit(R.convert(2))
    assert not R.is_unit(R.convert(x))
    assert not R.is_unit(R.convert(1 + x))

    R = QQ.old_poly_ring(x, order='ilex')
    assert R.is_unit(R.convert(1))
    assert R.is_unit(R.convert(2))
    assert not R.is_unit(R.convert(x))
    assert R.is_unit(R.convert(1 + x))

    R = ZZ.old_poly_ring(x)
    assert R.is_unit(R.convert(1))
    assert not R.is_unit(R.convert(2))
    assert not R.is_unit(R.convert(x))
    assert not R.is_unit(R.convert(1 + x))
