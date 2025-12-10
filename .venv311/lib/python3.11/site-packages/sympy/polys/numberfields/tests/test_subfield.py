"""Tests for the subfield problem and allied problems. """

from sympy.core.numbers import (AlgebraicNumber, I, pi, Rational)
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.external.gmpy import MPQ
from sympy.polys.numberfields.subfield import (
    is_isomorphism_possible,
    field_isomorphism_pslq,
    field_isomorphism,
    primitive_element,
    to_number_field,
)
from sympy.polys.domains import QQ
from sympy.polys.polyerrors import IsomorphismFailed
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import CRootOf
from sympy.testing.pytest import raises

from sympy.abc import x

Q = Rational


def test_field_isomorphism_pslq():
    a = AlgebraicNumber(I)
    b = AlgebraicNumber(I*sqrt(3))

    raises(NotImplementedError, lambda: field_isomorphism_pslq(a, b))

    a = AlgebraicNumber(sqrt(2))
    b = AlgebraicNumber(sqrt(3))
    c = AlgebraicNumber(sqrt(7))
    d = AlgebraicNumber(sqrt(2) + sqrt(3))
    e = AlgebraicNumber(sqrt(2) + sqrt(3) + sqrt(7))

    assert field_isomorphism_pslq(a, a) == [1, 0]
    assert field_isomorphism_pslq(a, b) is None
    assert field_isomorphism_pslq(a, c) is None
    assert field_isomorphism_pslq(a, d) == [Q(1, 2), 0, -Q(9, 2), 0]
    assert field_isomorphism_pslq(
        a, e) == [Q(1, 80), 0, -Q(1, 2), 0, Q(59, 20), 0]

    assert field_isomorphism_pslq(b, a) is None
    assert field_isomorphism_pslq(b, b) == [1, 0]
    assert field_isomorphism_pslq(b, c) is None
    assert field_isomorphism_pslq(b, d) == [-Q(1, 2), 0, Q(11, 2), 0]
    assert field_isomorphism_pslq(b, e) == [-Q(
        3, 640), 0, Q(67, 320), 0, -Q(297, 160), 0, Q(313, 80), 0]

    assert field_isomorphism_pslq(c, a) is None
    assert field_isomorphism_pslq(c, b) is None
    assert field_isomorphism_pslq(c, c) == [1, 0]
    assert field_isomorphism_pslq(c, d) is None
    assert field_isomorphism_pslq(c, e) == [Q(
        3, 640), 0, -Q(71, 320), 0, Q(377, 160), 0, -Q(469, 80), 0]

    assert field_isomorphism_pslq(d, a) is None
    assert field_isomorphism_pslq(d, b) is None
    assert field_isomorphism_pslq(d, c) is None
    assert field_isomorphism_pslq(d, d) == [1, 0]
    assert field_isomorphism_pslq(d, e) == [-Q(
        3, 640), 0, Q(71, 320), 0, -Q(377, 160), 0, Q(549, 80), 0]

    assert field_isomorphism_pslq(e, a) is None
    assert field_isomorphism_pslq(e, b) is None
    assert field_isomorphism_pslq(e, c) is None
    assert field_isomorphism_pslq(e, d) is None
    assert field_isomorphism_pslq(e, e) == [1, 0]

    f = AlgebraicNumber(3*sqrt(2) + 8*sqrt(7) - 5)

    assert field_isomorphism_pslq(
        f, e) == [Q(3, 80), 0, -Q(139, 80), 0, Q(347, 20), 0, -Q(761, 20), -5]


def test_field_isomorphism():
    assert field_isomorphism(3, sqrt(2)) == [3]

    assert field_isomorphism( I*sqrt(3), I*sqrt(3)/2) == [ 2, 0]
    assert field_isomorphism(-I*sqrt(3), I*sqrt(3)/2) == [-2, 0]

    assert field_isomorphism( I*sqrt(3), -I*sqrt(3)/2) == [-2, 0]
    assert field_isomorphism(-I*sqrt(3), -I*sqrt(3)/2) == [ 2, 0]

    assert field_isomorphism( 2*I*sqrt(3)/7, 5*I*sqrt(3)/3) == [ Rational(6, 35), 0]
    assert field_isomorphism(-2*I*sqrt(3)/7, 5*I*sqrt(3)/3) == [Rational(-6, 35), 0]

    assert field_isomorphism( 2*I*sqrt(3)/7, -5*I*sqrt(3)/3) == [Rational(-6, 35), 0]
    assert field_isomorphism(-2*I*sqrt(3)/7, -5*I*sqrt(3)/3) == [ Rational(6, 35), 0]

    assert field_isomorphism(
        2*I*sqrt(3)/7 + 27, 5*I*sqrt(3)/3) == [ Rational(6, 35), 27]
    assert field_isomorphism(
        -2*I*sqrt(3)/7 + 27, 5*I*sqrt(3)/3) == [Rational(-6, 35), 27]

    assert field_isomorphism(
        2*I*sqrt(3)/7 + 27, -5*I*sqrt(3)/3) == [Rational(-6, 35), 27]
    assert field_isomorphism(
        -2*I*sqrt(3)/7 + 27, -5*I*sqrt(3)/3) == [ Rational(6, 35), 27]

    p = AlgebraicNumber( sqrt(2) + sqrt(3))
    q = AlgebraicNumber(-sqrt(2) + sqrt(3))
    r = AlgebraicNumber( sqrt(2) - sqrt(3))
    s = AlgebraicNumber(-sqrt(2) - sqrt(3))

    pos_coeffs = [ S.Half, S.Zero, Rational(-9, 2), S.Zero]
    neg_coeffs = [Rational(-1, 2), S.Zero, Rational(9, 2), S.Zero]

    a = AlgebraicNumber(sqrt(2))

    assert is_isomorphism_possible(a, p) is True
    assert is_isomorphism_possible(a, q) is True
    assert is_isomorphism_possible(a, r) is True
    assert is_isomorphism_possible(a, s) is True

    assert field_isomorphism(a, p, fast=True) == pos_coeffs
    assert field_isomorphism(a, q, fast=True) == neg_coeffs
    assert field_isomorphism(a, r, fast=True) == pos_coeffs
    assert field_isomorphism(a, s, fast=True) == neg_coeffs

    assert field_isomorphism(a, p, fast=False) == pos_coeffs
    assert field_isomorphism(a, q, fast=False) == neg_coeffs
    assert field_isomorphism(a, r, fast=False) == pos_coeffs
    assert field_isomorphism(a, s, fast=False) == neg_coeffs

    a = AlgebraicNumber(-sqrt(2))

    assert is_isomorphism_possible(a, p) is True
    assert is_isomorphism_possible(a, q) is True
    assert is_isomorphism_possible(a, r) is True
    assert is_isomorphism_possible(a, s) is True

    assert field_isomorphism(a, p, fast=True) == neg_coeffs
    assert field_isomorphism(a, q, fast=True) == pos_coeffs
    assert field_isomorphism(a, r, fast=True) == neg_coeffs
    assert field_isomorphism(a, s, fast=True) == pos_coeffs

    assert field_isomorphism(a, p, fast=False) == neg_coeffs
    assert field_isomorphism(a, q, fast=False) == pos_coeffs
    assert field_isomorphism(a, r, fast=False) == neg_coeffs
    assert field_isomorphism(a, s, fast=False) == pos_coeffs

    pos_coeffs = [ S.Half, S.Zero, Rational(-11, 2), S.Zero]
    neg_coeffs = [Rational(-1, 2), S.Zero, Rational(11, 2), S.Zero]

    a = AlgebraicNumber(sqrt(3))

    assert is_isomorphism_possible(a, p) is True
    assert is_isomorphism_possible(a, q) is True
    assert is_isomorphism_possible(a, r) is True
    assert is_isomorphism_possible(a, s) is True

    assert field_isomorphism(a, p, fast=True) == neg_coeffs
    assert field_isomorphism(a, q, fast=True) == neg_coeffs
    assert field_isomorphism(a, r, fast=True) == pos_coeffs
    assert field_isomorphism(a, s, fast=True) == pos_coeffs

    assert field_isomorphism(a, p, fast=False) == neg_coeffs
    assert field_isomorphism(a, q, fast=False) == neg_coeffs
    assert field_isomorphism(a, r, fast=False) == pos_coeffs
    assert field_isomorphism(a, s, fast=False) == pos_coeffs

    a = AlgebraicNumber(-sqrt(3))

    assert is_isomorphism_possible(a, p) is True
    assert is_isomorphism_possible(a, q) is True
    assert is_isomorphism_possible(a, r) is True
    assert is_isomorphism_possible(a, s) is True

    assert field_isomorphism(a, p, fast=True) == pos_coeffs
    assert field_isomorphism(a, q, fast=True) == pos_coeffs
    assert field_isomorphism(a, r, fast=True) == neg_coeffs
    assert field_isomorphism(a, s, fast=True) == neg_coeffs

    assert field_isomorphism(a, p, fast=False) == pos_coeffs
    assert field_isomorphism(a, q, fast=False) == pos_coeffs
    assert field_isomorphism(a, r, fast=False) == neg_coeffs
    assert field_isomorphism(a, s, fast=False) == neg_coeffs

    pos_coeffs = [ Rational(3, 2), S.Zero, Rational(-33, 2), -S(8)]
    neg_coeffs = [Rational(-3, 2), S.Zero, Rational(33, 2), -S(8)]

    a = AlgebraicNumber(3*sqrt(3) - 8)

    assert is_isomorphism_possible(a, p) is True
    assert is_isomorphism_possible(a, q) is True
    assert is_isomorphism_possible(a, r) is True
    assert is_isomorphism_possible(a, s) is True

    assert field_isomorphism(a, p, fast=True) == neg_coeffs
    assert field_isomorphism(a, q, fast=True) == neg_coeffs
    assert field_isomorphism(a, r, fast=True) == pos_coeffs
    assert field_isomorphism(a, s, fast=True) == pos_coeffs

    assert field_isomorphism(a, p, fast=False) == neg_coeffs
    assert field_isomorphism(a, q, fast=False) == neg_coeffs
    assert field_isomorphism(a, r, fast=False) == pos_coeffs
    assert field_isomorphism(a, s, fast=False) == pos_coeffs

    a = AlgebraicNumber(3*sqrt(2) + 2*sqrt(3) + 1)

    pos_1_coeffs = [ S.Half, S.Zero, Rational(-5, 2), S.One]
    neg_5_coeffs = [Rational(-5, 2), S.Zero, Rational(49, 2), S.One]
    pos_5_coeffs = [ Rational(5, 2), S.Zero, Rational(-49, 2), S.One]
    neg_1_coeffs = [Rational(-1, 2), S.Zero, Rational(5, 2), S.One]

    assert is_isomorphism_possible(a, p) is True
    assert is_isomorphism_possible(a, q) is True
    assert is_isomorphism_possible(a, r) is True
    assert is_isomorphism_possible(a, s) is True

    assert field_isomorphism(a, p, fast=True) == pos_1_coeffs
    assert field_isomorphism(a, q, fast=True) == neg_5_coeffs
    assert field_isomorphism(a, r, fast=True) == pos_5_coeffs
    assert field_isomorphism(a, s, fast=True) == neg_1_coeffs

    assert field_isomorphism(a, p, fast=False) == pos_1_coeffs
    assert field_isomorphism(a, q, fast=False) == neg_5_coeffs
    assert field_isomorphism(a, r, fast=False) == pos_5_coeffs
    assert field_isomorphism(a, s, fast=False) == neg_1_coeffs

    a = AlgebraicNumber(sqrt(2))
    b = AlgebraicNumber(sqrt(3))
    c = AlgebraicNumber(sqrt(7))

    assert is_isomorphism_possible(a, b) is True
    assert is_isomorphism_possible(b, a) is True

    assert is_isomorphism_possible(c, p) is False

    assert field_isomorphism(sqrt(2), sqrt(3), fast=True) is None
    assert field_isomorphism(sqrt(3), sqrt(2), fast=True) is None

    assert field_isomorphism(sqrt(2), sqrt(3), fast=False) is None
    assert field_isomorphism(sqrt(3), sqrt(2), fast=False) is None

    a = AlgebraicNumber(sqrt(2))
    b = AlgebraicNumber(2 ** (S(1) / 3))

    assert is_isomorphism_possible(a, b) is False
    assert field_isomorphism(a, b) is None


def test_primitive_element():
    assert primitive_element([sqrt(2)], x) == (x**2 - 2, [1])
    assert primitive_element(
        [sqrt(2), sqrt(3)], x) == (x**4 - 10*x**2 + 1, [1, 1])

    assert primitive_element([sqrt(2)], x, polys=True) == (Poly(x**2 - 2, domain='QQ'), [1])
    assert primitive_element([sqrt(
        2), sqrt(3)], x, polys=True) == (Poly(x**4 - 10*x**2 + 1, domain='QQ'), [1, 1])

    assert primitive_element(
        [sqrt(2)], x, ex=True) == (x**2 - 2, [1], [[1, 0]])
    assert primitive_element([sqrt(2), sqrt(3)], x, ex=True) == \
        (x**4 - 10*x**2 + 1, [1, 1], [[Q(1, 2), 0, -Q(9, 2), 0], [-
         Q(1, 2), 0, Q(11, 2), 0]])

    assert primitive_element(
        [sqrt(2)], x, ex=True, polys=True) == (Poly(x**2 - 2, domain='QQ'), [1], [[1, 0]])
    assert primitive_element([sqrt(2), sqrt(3)], x, ex=True, polys=True) == \
        (Poly(x**4 - 10*x**2 + 1, domain='QQ'), [1, 1], [[Q(1, 2), 0, -Q(9, 2),
         0], [-Q(1, 2), 0, Q(11, 2), 0]])

    assert primitive_element([sqrt(2)], polys=True) == (Poly(x**2 - 2), [1])

    raises(ValueError, lambda: primitive_element([], x, ex=False))
    raises(ValueError, lambda: primitive_element([], x, ex=True))

    # Issue 14117
    a, b = I*sqrt(2*sqrt(2) + 3), I*sqrt(-2*sqrt(2) + 3)
    assert primitive_element([a, b, I], x) == (x**4 + 6*x**2 + 1, [1, 0, 0])

    assert primitive_element([sqrt(2), 0], x) == (x**2 - 2, [1, 0])
    assert primitive_element([0, sqrt(2)], x) == (x**2 - 2, [1, 1])
    assert primitive_element([sqrt(2), 0], x, ex=True) == (x**2 - 2, [1, 0], [[MPQ(1,1), MPQ(0,1)], []])
    assert primitive_element([0, sqrt(2)], x, ex=True) == (x**2 - 2, [1, 1], [[], [MPQ(1,1), MPQ(0,1)]])


def test_to_number_field():
    assert to_number_field(sqrt(2)) == AlgebraicNumber(sqrt(2))
    assert to_number_field(
        [sqrt(2), sqrt(3)]) == AlgebraicNumber(sqrt(2) + sqrt(3))

    a = AlgebraicNumber(sqrt(2) + sqrt(3), [S.Half, S.Zero, Rational(-9, 2), S.Zero])

    assert to_number_field(sqrt(2), sqrt(2) + sqrt(3)) == a
    assert to_number_field(sqrt(2), AlgebraicNumber(sqrt(2) + sqrt(3))) == a

    raises(IsomorphismFailed, lambda: to_number_field(sqrt(2), sqrt(3)))


def test_issue_22561():
    a = to_number_field(sqrt(2), sqrt(2) + sqrt(3))
    b = to_number_field(sqrt(2), sqrt(2) + sqrt(5))
    assert field_isomorphism(a, b) == [1, 0]


def test_issue_22736():
    a = CRootOf(x**4 + x**3 + x**2 + x + 1, -1)
    a._reset()
    b = exp(2*I*pi/5)
    assert field_isomorphism(a, b) == [1, 0]


def test_issue_27798():
    # https://github.com/sympy/sympy/issues/27798
    a, b = CRootOf(49*x**3 - 49*x**2 + 14*x - 1, 2), CRootOf(49*x**3 - 49*x**2 + 14*x - 1, 0)
    assert primitive_element([a, b], polys=True)[0].primitive()[0] == 1
    assert primitive_element([a, b], polys=True, ex=True)[0].primitive()[0] == 1

    f1, f2 = QQ.algebraic_field(a), QQ.algebraic_field(b)
    f3 = f1.unify(f2)
    assert f3.mod.primitive()[0] == 1
    assert Poly(x, x, domain=f1) + Poly(x, x, domain=f2) == Poly(2*x, x, domain=f3)
