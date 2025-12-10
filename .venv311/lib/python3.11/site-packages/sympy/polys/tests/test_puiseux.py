#
# Tests for PuiseuxRing and PuiseuxPoly
#

from sympy.testing.pytest import raises

from sympy import ZZ, QQ, ring
from sympy.polys.puiseux import PuiseuxRing, PuiseuxPoly, puiseux_ring

from sympy.abc import x, y


def test_puiseux_ring():
    R, px = puiseux_ring('x', QQ)
    R2, px2 = puiseux_ring([x], QQ)
    assert isinstance(R, PuiseuxRing)
    assert isinstance(px, PuiseuxPoly)
    assert R == R2
    assert px == px2
    assert R == PuiseuxRing('x', QQ)
    assert R == PuiseuxRing([x], QQ)
    assert R != PuiseuxRing('y', QQ)
    assert R != PuiseuxRing('x', ZZ)
    assert R != PuiseuxRing('x, y', QQ)
    assert R != QQ
    assert str(R) == 'PuiseuxRing((x,), QQ)'


def test_puiseux_ring_attributes():
    R1, px1, py1 = ring('x, y', QQ)
    R2, px2, py2 = puiseux_ring('x, y', QQ)
    assert R2.domain == QQ
    assert R2.symbols == (x, y)
    assert R2.gens == (px2, py2)
    assert R2.ngens == 2
    assert R2.poly_ring == R1
    assert R2.zero == PuiseuxPoly(R1.zero, R2)
    assert R2.one == PuiseuxPoly(R1.one, R2)
    assert R2.zero_monom == R1.zero_monom == (0, 0) # type: ignore
    assert R2.monomial_mul((1, 2), (3, 4)) == (4, 6)


def test_puiseux_ring_methods():
    R1, px1, py1 = ring('x, y', QQ)
    R2, px2, py2 = puiseux_ring('x, y', QQ)
    assert R2({(1, 2): 3}) == 3*px2*py2**2
    assert R2(px1) == px2
    assert R2(1) == R2.one
    assert R2(QQ(1,2)) == QQ(1,2)*R2.one
    assert R2.from_poly(px1) == px2
    assert R2.from_poly(px1) != py2
    assert R2.from_dict({(1, 2): QQ(3)}) == 3*px2*py2**2
    assert R2.from_dict({(QQ(1,2), 2): QQ(3)}) == 3*px2**QQ(1,2)*py2**2
    assert R2.from_int(3) == 3*R2.one
    assert R2.domain_new(3) == QQ(3)
    assert QQ.of_type(R2.domain_new(3))
    assert R2.ground_new(3) == 3*R2.one
    assert isinstance(R2.ground_new(3), PuiseuxPoly)
    assert R2.index(px2) == 0
    assert R2.index(py2) == 1


def test_puiseux_poly():
    R1, px1 = ring('x', QQ)
    R2, px2 = puiseux_ring('x', QQ)
    assert PuiseuxPoly(px1, R2) == px2
    assert px2.ring == R2
    assert px2.as_expr() == px1.as_expr() == x
    assert px1 != px2
    assert R2.one == px2**0 == 1
    assert px2 == px1
    assert px2 != 2.0
    assert px2**QQ(1,2) != px1


def test_puiseux_poly_normalization():
    R, x = puiseux_ring('x', QQ)
    assert (x**2 + 1) / x == x + 1/x == R({(1,): 1, (-1,): 1})
    assert (x**QQ(1,6))**2 == x**QQ(1,3) == R({(QQ(1,3),): 1})
    assert (x**QQ(1,6))**(-2) == x**(-QQ(1,3)) == R({(-QQ(1,3),): 1})
    assert (x**QQ(1,6))**QQ(1,2) == x**QQ(1,12) == R({(QQ(1,12),): 1})
    assert (x**QQ(1,6))**6 == x == R({(1,): 1})
    assert x**QQ(1,6) * x**QQ(1,3) == x**QQ(1,2) == R({(QQ(1,2),): 1})
    assert 1/x * x**2 == x == R({(1,): 1})
    assert 1/x**QQ(1,3) * x**QQ(1,3) == 1 == R({(0,): 1})


def test_puiseux_poly_monoms():
    R, x = puiseux_ring('x', QQ)
    assert x.monoms() == [(1,)]
    assert list(x) == [(1,)]
    assert (x**2 + 1).monoms() == [(2,), (0,)]
    assert R({(1,): 1, (-1,): 1}).monoms() == [(1,), (-1,)]
    assert R({(QQ(1,3),): 1}).monoms() == [(QQ(1,3),)]
    assert R({(-QQ(1,3),): 1}).monoms() == [(-QQ(1,3),)]
    p = x**QQ(1,6)
    assert p[(QQ(1,6),)] == 1
    raises(KeyError, lambda: p[(1,)])
    assert p.to_dict() == {(QQ(1,6),): 1}
    assert R(p.to_dict()) == p
    assert PuiseuxPoly.from_dict({(QQ(1,6),): 1}, R) == p


def test_puiseux_poly_repr():
    R, x = puiseux_ring('x', QQ)
    assert repr(x) == 'x'
    assert repr(x**QQ(1,2)) == 'x**(1/2)'
    assert repr(1/x) == 'x**(-1)'
    assert repr(2*x**2 + 1) == '1 + 2*x**2'
    assert repr(R.one) == '1'
    assert repr(2*R.one) == '2'


def test_puiseux_poly_unify():
    R, x = puiseux_ring('x', QQ)
    assert 1/x + x == x + 1/x == R({(1,): 1, (-1,): 1})
    assert repr(1/x + x) == 'x**(-1) + x'
    assert 1/x + 1/x == 2/x == R({(-1,): 2})
    assert repr(1/x + 1/x) == '2*x**(-1)'
    assert x**QQ(1,2) + x**QQ(1,2) == 2*x**QQ(1,2) == R({(QQ(1,2),): 2})
    assert repr(x**QQ(1,2) + x**QQ(1,2)) == '2*x**(1/2)'
    assert x**QQ(1,2) + x**QQ(1,3) == R({(QQ(1,2),): 1, (QQ(1,3),): 1})
    assert repr(x**QQ(1,2) + x**QQ(1,3)) == 'x**(1/3) + x**(1/2)'
    assert x + x**QQ(1,2) == R({(1,): 1, (QQ(1,2),): 1})
    assert repr(x + x**QQ(1,2)) == 'x**(1/2) + x'
    assert 1/x**QQ(1,2) + 1/x**QQ(1,3) == R({(-QQ(1,2),): 1, (-QQ(1,3),): 1})
    assert repr(1/x**QQ(1,2) + 1/x**QQ(1,3)) == 'x**(-1/2) + x**(-1/3)'
    assert 1/x + x**QQ(1,2) == x**QQ(1,2) + 1/x == R({(-1,): 1, (QQ(1,2),): 1})
    assert repr(1/x + x**QQ(1,2)) == 'x**(-1) + x**(1/2)'


def test_puiseux_poly_arit():
    R, x = puiseux_ring('x', QQ)
    R2, y = puiseux_ring('y', QQ)
    p = x**2 + 1
    assert +p == p
    assert -p == -1 - x**2
    assert p + p == 2*p == 2*x**2 + 2
    assert p + 1 == 1 + p == x**2 + 2
    assert p + QQ(1,2) == QQ(1,2) + p == x**2 + QQ(3,2)
    assert p - p == 0
    assert p - 1 == -1 + p == x**2
    assert p - QQ(1,2) == -QQ(1,2) + p == x**2 + QQ(1,2)
    assert 1 - p == -p + 1 == -x**2
    assert QQ(1,2) - p == -p + QQ(1,2) == -x**2 - QQ(1,2)
    assert p * p == x**4 + 2*x**2 + 1
    assert p * 1 == 1 * p == p
    assert 2 * p == p * 2 == 2*x**2 + 2
    assert p * QQ(1,2) == QQ(1,2) * p == QQ(1,2)*x**2 + QQ(1,2)
    assert x**QQ(1,2) * x**QQ(1,2) == x
    raises(ValueError, lambda: x + y)
    raises(ValueError, lambda: x - y)
    raises(ValueError, lambda: x * y)
    raises(TypeError, lambda: x + None)
    raises(TypeError, lambda: x - None)
    raises(TypeError, lambda: x * None)
    raises(TypeError, lambda: None + x)
    raises(TypeError, lambda: None - x)
    raises(TypeError, lambda: None * x)


def test_puiseux_poly_div():
    R, x = puiseux_ring('x', QQ)
    R2, y = puiseux_ring('y', QQ)
    p = x**2 - 1
    assert p / 1 == p
    assert p / QQ(1,2) == 2*p == 2*x**2 - 2
    assert p / x == x - 1/x == R({(1,): 1, (-1,): -1})
    assert 2 / x == 2*x**-1 == R({(-1,): 2})
    assert QQ(1,2) / x == QQ(1,2)*x**-1 == 1/(2*x) == 1/x/2 == R({(-1,): QQ(1,2)})
    raises(ZeroDivisionError, lambda: p / 0)
    raises(ValueError, lambda: (x + 1) / (x + 2))
    raises(ValueError, lambda: (x + 1) / (x + 1))
    raises(ValueError, lambda: x / y)
    raises(TypeError, lambda: x / None)
    raises(TypeError, lambda: None / x)


def test_puiseux_poly_pow():
    R, x = puiseux_ring('x', QQ)
    Rz, xz = puiseux_ring('x', ZZ)
    assert x**0 == 1 == R({(0,): 1})
    assert x**1 == x == R({(1,): 1})
    assert x**2 == x*x == R({(2,): 1})
    assert x**QQ(1,2) == R({(QQ(1,2),): 1})
    assert x**-1 == 1/x == R({(-1,): 1})
    assert x**-QQ(1,2) == 1/x**QQ(1,2) == R({(-QQ(1,2),): 1})
    assert (2*x)**-1 == 1/(2*x) == QQ(1,2)/x == QQ(1,2)*x**-1 == R({(-1,): QQ(1,2)})
    assert 2/x**2 == 2*x**-2 == R({(-2,): 2})
    assert 2/xz**2 == 2*xz**-2 == Rz({(-2,): 2})
    raises(TypeError, lambda: x**None)
    raises(ValueError, lambda: (x + 1)**-1)
    raises(ValueError, lambda: (x + 1)**QQ(1,2))
    raises(ValueError, lambda: (2*x)**QQ(1,2))
    raises(ValueError, lambda: (2*xz)**-1)


def test_puiseux_poly_diff():
    R, x, y = puiseux_ring('x, y', QQ)
    assert (x**2 + 1).diff(x) == 2*x
    assert (x**2 + 1).diff(y) == 0
    assert (x**2 + y**2).diff(x) == 2*x
    assert (x**QQ(1,2) + y**QQ(1,2)).diff(x) == QQ(1,2)*x**-QQ(1,2)
    assert ((x*y)**QQ(1,2)).diff(x) == QQ(1,2)*y**QQ(1,2)*x**-QQ(1,2)
