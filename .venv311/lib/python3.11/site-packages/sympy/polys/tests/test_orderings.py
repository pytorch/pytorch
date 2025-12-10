"""Tests of monomial orderings. """

from sympy.polys.orderings import (
    monomial_key, lex, grlex, grevlex, ilex, igrlex,
    LexOrder, InverseOrder, ProductOrder, build_product_order,
)

from sympy.abc import x, y, z, t
from sympy.core import S
from sympy.testing.pytest import raises

def test_lex_order():
    assert lex((1, 2, 3)) == (1, 2, 3)
    assert str(lex) == 'lex'

    assert lex((1, 2, 3)) == lex((1, 2, 3))

    assert lex((2, 2, 3)) > lex((1, 2, 3))
    assert lex((1, 3, 3)) > lex((1, 2, 3))
    assert lex((1, 2, 4)) > lex((1, 2, 3))

    assert lex((0, 2, 3)) < lex((1, 2, 3))
    assert lex((1, 1, 3)) < lex((1, 2, 3))
    assert lex((1, 2, 2)) < lex((1, 2, 3))

    assert lex.is_global is True
    assert lex == LexOrder()
    assert lex != grlex

def test_grlex_order():
    assert grlex((1, 2, 3)) == (6, (1, 2, 3))
    assert str(grlex) == 'grlex'

    assert grlex((1, 2, 3)) == grlex((1, 2, 3))

    assert grlex((2, 2, 3)) > grlex((1, 2, 3))
    assert grlex((1, 3, 3)) > grlex((1, 2, 3))
    assert grlex((1, 2, 4)) > grlex((1, 2, 3))

    assert grlex((0, 2, 3)) < grlex((1, 2, 3))
    assert grlex((1, 1, 3)) < grlex((1, 2, 3))
    assert grlex((1, 2, 2)) < grlex((1, 2, 3))

    assert grlex((2, 2, 3)) > grlex((1, 2, 4))
    assert grlex((1, 3, 3)) > grlex((1, 2, 4))

    assert grlex((0, 2, 3)) < grlex((1, 2, 2))
    assert grlex((1, 1, 3)) < grlex((1, 2, 2))

    assert grlex((0, 1, 1)) > grlex((0, 0, 2))
    assert grlex((0, 3, 1)) < grlex((2, 2, 1))

    assert grlex.is_global is True

def test_grevlex_order():
    assert grevlex((1, 2, 3)) == (6, (-3, -2, -1))
    assert str(grevlex) == 'grevlex'

    assert grevlex((1, 2, 3)) == grevlex((1, 2, 3))

    assert grevlex((2, 2, 3)) > grevlex((1, 2, 3))
    assert grevlex((1, 3, 3)) > grevlex((1, 2, 3))
    assert grevlex((1, 2, 4)) > grevlex((1, 2, 3))

    assert grevlex((0, 2, 3)) < grevlex((1, 2, 3))
    assert grevlex((1, 1, 3)) < grevlex((1, 2, 3))
    assert grevlex((1, 2, 2)) < grevlex((1, 2, 3))

    assert grevlex((2, 2, 3)) > grevlex((1, 2, 4))
    assert grevlex((1, 3, 3)) > grevlex((1, 2, 4))

    assert grevlex((0, 2, 3)) < grevlex((1, 2, 2))
    assert grevlex((1, 1, 3)) < grevlex((1, 2, 2))

    assert grevlex((0, 1, 1)) > grevlex((0, 0, 2))
    assert grevlex((0, 3, 1)) < grevlex((2, 2, 1))

    assert grevlex.is_global is True

def test_InverseOrder():
    ilex = InverseOrder(lex)
    igrlex = InverseOrder(grlex)

    assert ilex((1, 2, 3)) > ilex((2, 0, 3))
    assert igrlex((1, 2, 3)) < igrlex((0, 2, 3))
    assert str(ilex) == "ilex"
    assert str(igrlex) == "igrlex"
    assert ilex.is_global is False
    assert igrlex.is_global is False
    assert ilex != igrlex
    assert ilex == InverseOrder(LexOrder())

def test_ProductOrder():
    P = ProductOrder((grlex, lambda m: m[:2]), (grlex, lambda m: m[2:]))
    assert P((1, 3, 3, 4, 5)) > P((2, 1, 5, 5, 5))
    assert str(P) == "ProductOrder(grlex, grlex)"
    assert P.is_global is True
    assert ProductOrder((grlex, None), (ilex, None)).is_global is None
    assert ProductOrder((igrlex, None), (ilex, None)).is_global is False

def test_monomial_key():
    assert monomial_key() == lex

    assert monomial_key('lex') == lex
    assert monomial_key('grlex') == grlex
    assert monomial_key('grevlex') == grevlex

    raises(ValueError, lambda: monomial_key('foo'))
    raises(ValueError, lambda: monomial_key(1))

    M = [x, x**2*z**2, x*y, x**2, S.One, y**2, x**3, y, z, x*y**2*z, x**2*y**2]
    assert sorted(M, key=monomial_key('lex', [z, y, x])) == \
        [S.One, x, x**2, x**3, y, x*y, y**2, x**2*y**2, z, x*y**2*z, x**2*z**2]
    assert sorted(M, key=monomial_key('grlex', [z, y, x])) == \
        [S.One, x, y, z, x**2, x*y, y**2, x**3, x**2*y**2, x*y**2*z, x**2*z**2]
    assert sorted(M, key=monomial_key('grevlex', [z, y, x])) == \
        [S.One, x, y, z, x**2, x*y, y**2, x**3, x**2*y**2, x**2*z**2, x*y**2*z]

def test_build_product_order():
    assert build_product_order((("grlex", x, y), ("grlex", z, t)), [x, y, z, t])((4, 5, 6, 7)) == \
        ((9, (4, 5)), (13, (6, 7)))

    assert build_product_order((("grlex", x, y), ("grlex", z, t)), [x, y, z, t]) == \
        build_product_order((("grlex", x, y), ("grlex", z, t)), [x, y, z, t])
