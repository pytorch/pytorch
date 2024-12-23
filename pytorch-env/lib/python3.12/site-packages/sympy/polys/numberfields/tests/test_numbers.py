"""Tests on algebraic numbers. """

from sympy.core.containers import Tuple
from sympy.core.numbers import (AlgebraicNumber, I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.polytools import Poly
from sympy.polys.numberfields.subfield import to_number_field
from sympy.polys.polyclasses import DMP
from sympy.polys.domains import QQ
from sympy.polys.rootoftools import CRootOf
from sympy.abc import x, y


def test_AlgebraicNumber():
    minpoly, root = x**2 - 2, sqrt(2)

    a = AlgebraicNumber(root, gen=x)

    assert a.rep == DMP([QQ(1), QQ(0)], QQ)
    assert a.root == root
    assert a.alias is None
    assert a.minpoly == minpoly
    assert a.is_number

    assert a.is_aliased is False

    assert a.coeffs() == [S.One, S.Zero]
    assert a.native_coeffs() == [QQ(1), QQ(0)]

    a = AlgebraicNumber(root, gen=x, alias='y')

    assert a.rep == DMP([QQ(1), QQ(0)], QQ)
    assert a.root == root
    assert a.alias == Symbol('y')
    assert a.minpoly == minpoly
    assert a.is_number

    assert a.is_aliased is True

    a = AlgebraicNumber(root, gen=x, alias=Symbol('y'))

    assert a.rep == DMP([QQ(1), QQ(0)], QQ)
    assert a.root == root
    assert a.alias == Symbol('y')
    assert a.minpoly == minpoly
    assert a.is_number

    assert a.is_aliased is True

    assert AlgebraicNumber(sqrt(2), []).rep == DMP([], QQ)
    assert AlgebraicNumber(sqrt(2), ()).rep == DMP([], QQ)
    assert AlgebraicNumber(sqrt(2), (0, 0)).rep == DMP([], QQ)

    assert AlgebraicNumber(sqrt(2), [8]).rep == DMP([QQ(8)], QQ)
    assert AlgebraicNumber(sqrt(2), [Rational(8, 3)]).rep == DMP([QQ(8, 3)], QQ)

    assert AlgebraicNumber(sqrt(2), [7, 3]).rep == DMP([QQ(7), QQ(3)], QQ)
    assert AlgebraicNumber(
        sqrt(2), [Rational(7, 9), Rational(3, 2)]).rep == DMP([QQ(7, 9), QQ(3, 2)], QQ)

    assert AlgebraicNumber(sqrt(2), [1, 2, 3]).rep == DMP([QQ(2), QQ(5)], QQ)

    a = AlgebraicNumber(AlgebraicNumber(root, gen=x), [1, 2])

    assert a.rep == DMP([QQ(1), QQ(2)], QQ)
    assert a.root == root
    assert a.alias is None
    assert a.minpoly == minpoly
    assert a.is_number

    assert a.is_aliased is False

    assert a.coeffs() == [S.One, S(2)]
    assert a.native_coeffs() == [QQ(1), QQ(2)]

    a = AlgebraicNumber((minpoly, root), [1, 2])

    assert a.rep == DMP([QQ(1), QQ(2)], QQ)
    assert a.root == root
    assert a.alias is None
    assert a.minpoly == minpoly
    assert a.is_number

    assert a.is_aliased is False

    a = AlgebraicNumber((Poly(minpoly), root), [1, 2])

    assert a.rep == DMP([QQ(1), QQ(2)], QQ)
    assert a.root == root
    assert a.alias is None
    assert a.minpoly == minpoly
    assert a.is_number

    assert a.is_aliased is False

    assert AlgebraicNumber( sqrt(3)).rep == DMP([ QQ(1), QQ(0)], QQ)
    assert AlgebraicNumber(-sqrt(3)).rep == DMP([ QQ(1), QQ(0)], QQ)

    a = AlgebraicNumber(sqrt(2))
    b = AlgebraicNumber(sqrt(2))

    assert a == b

    c = AlgebraicNumber(sqrt(2), gen=x)

    assert a == b
    assert a == c

    a = AlgebraicNumber(sqrt(2), [1, 2])
    b = AlgebraicNumber(sqrt(2), [1, 3])

    assert a != b and a != sqrt(2) + 3

    assert (a == x) is False and (a != x) is True

    a = AlgebraicNumber(sqrt(2), [1, 0])
    b = AlgebraicNumber(sqrt(2), [1, 0], alias=y)

    assert a.as_poly(x) == Poly(x, domain='QQ')
    assert b.as_poly() == Poly(y, domain='QQ')

    assert a.as_expr() == sqrt(2)
    assert a.as_expr(x) == x
    assert b.as_expr() == sqrt(2)
    assert b.as_expr(x) == x

    a = AlgebraicNumber(sqrt(2), [2, 3])
    b = AlgebraicNumber(sqrt(2), [2, 3], alias=y)

    p = a.as_poly()

    assert p == Poly(2*p.gen + 3)

    assert a.as_poly(x) == Poly(2*x + 3, domain='QQ')
    assert b.as_poly() == Poly(2*y + 3, domain='QQ')

    assert a.as_expr() == 2*sqrt(2) + 3
    assert a.as_expr(x) == 2*x + 3
    assert b.as_expr() == 2*sqrt(2) + 3
    assert b.as_expr(x) == 2*x + 3

    a = AlgebraicNumber(sqrt(2))
    b = to_number_field(sqrt(2))
    assert a.args == b.args == (sqrt(2), Tuple(1, 0))
    b = AlgebraicNumber(sqrt(2), alias='alpha')
    assert b.args == (sqrt(2), Tuple(1, 0), Symbol('alpha'))

    a = AlgebraicNumber(sqrt(2), [1, 2, 3])
    assert a.args == (sqrt(2), Tuple(1, 2, 3))

    a = AlgebraicNumber(sqrt(2), [1, 2], "alpha")
    b = AlgebraicNumber(a)
    c = AlgebraicNumber(a, alias="gamma")
    assert a == b
    assert c.alias.name == "gamma"

    a = AlgebraicNumber(sqrt(2) + sqrt(3), [S(1)/2, 0, S(-9)/2, 0])
    b = AlgebraicNumber(a, [1, 0, 0])
    assert b.root == a.root
    assert a.to_root() == sqrt(2)
    assert b.to_root() == 2

    a = AlgebraicNumber(2)
    assert a.is_primitive_element is True


def test_to_algebraic_integer():
    a = AlgebraicNumber(sqrt(3), gen=x).to_algebraic_integer()

    assert a.minpoly == x**2 - 3
    assert a.root == sqrt(3)
    assert a.rep == DMP([QQ(1), QQ(0)], QQ)

    a = AlgebraicNumber(2*sqrt(3), gen=x).to_algebraic_integer()
    assert a.minpoly == x**2 - 12
    assert a.root == 2*sqrt(3)
    assert a.rep == DMP([QQ(1), QQ(0)], QQ)

    a = AlgebraicNumber(sqrt(3)/2, gen=x).to_algebraic_integer()

    assert a.minpoly == x**2 - 12
    assert a.root == 2*sqrt(3)
    assert a.rep == DMP([QQ(1), QQ(0)], QQ)

    a = AlgebraicNumber(sqrt(3)/2, [Rational(7, 19), 3], gen=x).to_algebraic_integer()

    assert a.minpoly == x**2 - 12
    assert a.root == 2*sqrt(3)
    assert a.rep == DMP([QQ(7, 19), QQ(3)], QQ)


def test_AlgebraicNumber_to_root():
    assert AlgebraicNumber(sqrt(2)).to_root() == sqrt(2)

    zeta5_squared = AlgebraicNumber(CRootOf(x**5 - 1, 4), coeffs=[1, 0, 0])
    assert zeta5_squared.to_root() == CRootOf(x**4 + x**3 + x**2 + x + 1, 1)

    zeta3_squared = AlgebraicNumber(CRootOf(x**3 - 1, 2), coeffs=[1, 0, 0])
    assert zeta3_squared.to_root() == -S(1)/2 - sqrt(3)*I/2
    assert zeta3_squared.to_root(radicals=False) == CRootOf(x**2 + x + 1, 0)
