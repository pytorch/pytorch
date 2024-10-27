"""Tests for classes defining properties of ground domains, e.g. ZZ, QQ, ZZ[x] ... """

from sympy.external.gmpy import GROUND_TYPES

from sympy.core.numbers import (AlgebraicNumber, E, Float, I, Integer,
    Rational, oo, pi, _illegal)
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.polys.polytools import Poly
from sympy.abc import x, y, z

from sympy.polys.domains import (ZZ, QQ, RR, CC, FF, GF, EX, EXRAW, ZZ_gmpy,
    ZZ_python, QQ_gmpy, QQ_python)
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.gaussiandomains import ZZ_I, QQ_I
from sympy.polys.domains.polynomialring import PolynomialRing
from sympy.polys.domains.realfield import RealField

from sympy.polys.numberfields.subfield import field_isomorphism
from sympy.polys.rings import ring
from sympy.polys.specialpolys import cyclotomic_poly
from sympy.polys.fields import field

from sympy.polys.agca.extensions import FiniteExtension

from sympy.polys.polyerrors import (
    UnificationFailed,
    GeneratorsError,
    CoercionFailed,
    NotInvertible,
    DomainError)

from sympy.testing.pytest import raises, warns_deprecated_sympy

from itertools import product

ALG = QQ.algebraic_field(sqrt(2), sqrt(3))

def unify(K0, K1):
    return K0.unify(K1)

def test_Domain_unify():
    F3 = GF(3)
    F5 = GF(5)

    assert unify(F3, F3) == F3
    raises(UnificationFailed, lambda: unify(F3, ZZ))
    raises(UnificationFailed, lambda: unify(F3, QQ))
    raises(UnificationFailed, lambda: unify(F3, ZZ_I))
    raises(UnificationFailed, lambda: unify(F3, QQ_I))
    raises(UnificationFailed, lambda: unify(F3, ALG))
    raises(UnificationFailed, lambda: unify(F3, RR))
    raises(UnificationFailed, lambda: unify(F3, CC))
    raises(UnificationFailed, lambda: unify(F3, ZZ[x]))
    raises(UnificationFailed, lambda: unify(F3, ZZ.frac_field(x)))
    raises(UnificationFailed, lambda: unify(F3, EX))

    assert unify(F5, F5) == F5
    raises(UnificationFailed, lambda: unify(F5, F3))
    raises(UnificationFailed, lambda: unify(F5, F3[x]))
    raises(UnificationFailed, lambda: unify(F5, F3.frac_field(x)))

    raises(UnificationFailed, lambda: unify(ZZ, F3))
    assert unify(ZZ, ZZ) == ZZ
    assert unify(ZZ, QQ) == QQ
    assert unify(ZZ, ALG) == ALG
    assert unify(ZZ, RR) == RR
    assert unify(ZZ, CC) == CC
    assert unify(ZZ, ZZ[x]) == ZZ[x]
    assert unify(ZZ, ZZ.frac_field(x)) == ZZ.frac_field(x)
    assert unify(ZZ, EX) == EX

    raises(UnificationFailed, lambda: unify(QQ, F3))
    assert unify(QQ, ZZ) == QQ
    assert unify(QQ, QQ) == QQ
    assert unify(QQ, ALG) == ALG
    assert unify(QQ, RR) == RR
    assert unify(QQ, CC) == CC
    assert unify(QQ, ZZ[x]) == QQ[x]
    assert unify(QQ, ZZ.frac_field(x)) == QQ.frac_field(x)
    assert unify(QQ, EX) == EX

    raises(UnificationFailed, lambda: unify(ZZ_I, F3))
    assert unify(ZZ_I, ZZ) == ZZ_I
    assert unify(ZZ_I, ZZ_I) == ZZ_I
    assert unify(ZZ_I, QQ) == QQ_I
    assert unify(ZZ_I, ALG) == QQ.algebraic_field(I, sqrt(2), sqrt(3))
    assert unify(ZZ_I, RR) == CC
    assert unify(ZZ_I, CC) == CC
    assert unify(ZZ_I, ZZ[x]) == ZZ_I[x]
    assert unify(ZZ_I, ZZ_I[x]) == ZZ_I[x]
    assert unify(ZZ_I, ZZ.frac_field(x)) == ZZ_I.frac_field(x)
    assert unify(ZZ_I, ZZ_I.frac_field(x)) == ZZ_I.frac_field(x)
    assert unify(ZZ_I, EX) == EX

    raises(UnificationFailed, lambda: unify(QQ_I, F3))
    assert unify(QQ_I, ZZ) == QQ_I
    assert unify(QQ_I, ZZ_I) == QQ_I
    assert unify(QQ_I, QQ) == QQ_I
    assert unify(QQ_I, ALG) == QQ.algebraic_field(I, sqrt(2), sqrt(3))
    assert unify(QQ_I, RR) == CC
    assert unify(QQ_I, CC) == CC
    assert unify(QQ_I, ZZ[x]) == QQ_I[x]
    assert unify(QQ_I, ZZ_I[x]) == QQ_I[x]
    assert unify(QQ_I, QQ[x]) == QQ_I[x]
    assert unify(QQ_I, QQ_I[x]) == QQ_I[x]
    assert unify(QQ_I, ZZ.frac_field(x)) == QQ_I.frac_field(x)
    assert unify(QQ_I, ZZ_I.frac_field(x)) == QQ_I.frac_field(x)
    assert unify(QQ_I, QQ.frac_field(x)) == QQ_I.frac_field(x)
    assert unify(QQ_I, QQ_I.frac_field(x)) == QQ_I.frac_field(x)
    assert unify(QQ_I, EX) == EX

    raises(UnificationFailed, lambda: unify(RR, F3))
    assert unify(RR, ZZ) == RR
    assert unify(RR, QQ) == RR
    assert unify(RR, ALG) == RR
    assert unify(RR, RR) == RR
    assert unify(RR, CC) == CC
    assert unify(RR, ZZ[x]) == RR[x]
    assert unify(RR, ZZ.frac_field(x)) == RR.frac_field(x)
    assert unify(RR, EX) == EX
    assert RR[x].unify(ZZ.frac_field(y)) == RR.frac_field(x, y)

    raises(UnificationFailed, lambda: unify(CC, F3))
    assert unify(CC, ZZ) == CC
    assert unify(CC, QQ) == CC
    assert unify(CC, ALG) == CC
    assert unify(CC, RR) == CC
    assert unify(CC, CC) == CC
    assert unify(CC, ZZ[x]) == CC[x]
    assert unify(CC, ZZ.frac_field(x)) == CC.frac_field(x)
    assert unify(CC, EX) == EX

    raises(UnificationFailed, lambda: unify(ZZ[x], F3))
    assert unify(ZZ[x], ZZ) == ZZ[x]
    assert unify(ZZ[x], QQ) == QQ[x]
    assert unify(ZZ[x], ALG) == ALG[x]
    assert unify(ZZ[x], RR) == RR[x]
    assert unify(ZZ[x], CC) == CC[x]
    assert unify(ZZ[x], ZZ[x]) == ZZ[x]
    assert unify(ZZ[x], ZZ.frac_field(x)) == ZZ.frac_field(x)
    assert unify(ZZ[x], EX) == EX

    raises(UnificationFailed, lambda: unify(ZZ.frac_field(x), F3))
    assert unify(ZZ.frac_field(x), ZZ) == ZZ.frac_field(x)
    assert unify(ZZ.frac_field(x), QQ) == QQ.frac_field(x)
    assert unify(ZZ.frac_field(x), ALG) == ALG.frac_field(x)
    assert unify(ZZ.frac_field(x), RR) == RR.frac_field(x)
    assert unify(ZZ.frac_field(x), CC) == CC.frac_field(x)
    assert unify(ZZ.frac_field(x), ZZ[x]) == ZZ.frac_field(x)
    assert unify(ZZ.frac_field(x), ZZ.frac_field(x)) == ZZ.frac_field(x)
    assert unify(ZZ.frac_field(x), EX) == EX

    raises(UnificationFailed, lambda: unify(EX, F3))
    assert unify(EX, ZZ) == EX
    assert unify(EX, QQ) == EX
    assert unify(EX, ALG) == EX
    assert unify(EX, RR) == EX
    assert unify(EX, CC) == EX
    assert unify(EX, ZZ[x]) == EX
    assert unify(EX, ZZ.frac_field(x)) == EX
    assert unify(EX, EX) == EX

def test_Domain_unify_composite():
    assert unify(ZZ.poly_ring(x), ZZ) == ZZ.poly_ring(x)
    assert unify(ZZ.poly_ring(x), QQ) == QQ.poly_ring(x)
    assert unify(QQ.poly_ring(x), ZZ) == QQ.poly_ring(x)
    assert unify(QQ.poly_ring(x), QQ) == QQ.poly_ring(x)

    assert unify(ZZ, ZZ.poly_ring(x)) == ZZ.poly_ring(x)
    assert unify(QQ, ZZ.poly_ring(x)) == QQ.poly_ring(x)
    assert unify(ZZ, QQ.poly_ring(x)) == QQ.poly_ring(x)
    assert unify(QQ, QQ.poly_ring(x)) == QQ.poly_ring(x)

    assert unify(ZZ.poly_ring(x, y), ZZ) == ZZ.poly_ring(x, y)
    assert unify(ZZ.poly_ring(x, y), QQ) == QQ.poly_ring(x, y)
    assert unify(QQ.poly_ring(x, y), ZZ) == QQ.poly_ring(x, y)
    assert unify(QQ.poly_ring(x, y), QQ) == QQ.poly_ring(x, y)

    assert unify(ZZ, ZZ.poly_ring(x, y)) == ZZ.poly_ring(x, y)
    assert unify(QQ, ZZ.poly_ring(x, y)) == QQ.poly_ring(x, y)
    assert unify(ZZ, QQ.poly_ring(x, y)) == QQ.poly_ring(x, y)
    assert unify(QQ, QQ.poly_ring(x, y)) == QQ.poly_ring(x, y)

    assert unify(ZZ.frac_field(x), ZZ) == ZZ.frac_field(x)
    assert unify(ZZ.frac_field(x), QQ) == QQ.frac_field(x)
    assert unify(QQ.frac_field(x), ZZ) == QQ.frac_field(x)
    assert unify(QQ.frac_field(x), QQ) == QQ.frac_field(x)

    assert unify(ZZ, ZZ.frac_field(x)) == ZZ.frac_field(x)
    assert unify(QQ, ZZ.frac_field(x)) == QQ.frac_field(x)
    assert unify(ZZ, QQ.frac_field(x)) == QQ.frac_field(x)
    assert unify(QQ, QQ.frac_field(x)) == QQ.frac_field(x)

    assert unify(ZZ.frac_field(x, y), ZZ) == ZZ.frac_field(x, y)
    assert unify(ZZ.frac_field(x, y), QQ) == QQ.frac_field(x, y)
    assert unify(QQ.frac_field(x, y), ZZ) == QQ.frac_field(x, y)
    assert unify(QQ.frac_field(x, y), QQ) == QQ.frac_field(x, y)

    assert unify(ZZ, ZZ.frac_field(x, y)) == ZZ.frac_field(x, y)
    assert unify(QQ, ZZ.frac_field(x, y)) == QQ.frac_field(x, y)
    assert unify(ZZ, QQ.frac_field(x, y)) == QQ.frac_field(x, y)
    assert unify(QQ, QQ.frac_field(x, y)) == QQ.frac_field(x, y)

    assert unify(ZZ.poly_ring(x), ZZ.poly_ring(x)) == ZZ.poly_ring(x)
    assert unify(ZZ.poly_ring(x), QQ.poly_ring(x)) == QQ.poly_ring(x)
    assert unify(QQ.poly_ring(x), ZZ.poly_ring(x)) == QQ.poly_ring(x)
    assert unify(QQ.poly_ring(x), QQ.poly_ring(x)) == QQ.poly_ring(x)

    assert unify(ZZ.poly_ring(x, y), ZZ.poly_ring(x)) == ZZ.poly_ring(x, y)
    assert unify(ZZ.poly_ring(x, y), QQ.poly_ring(x)) == QQ.poly_ring(x, y)
    assert unify(QQ.poly_ring(x, y), ZZ.poly_ring(x)) == QQ.poly_ring(x, y)
    assert unify(QQ.poly_ring(x, y), QQ.poly_ring(x)) == QQ.poly_ring(x, y)

    assert unify(ZZ.poly_ring(x), ZZ.poly_ring(x, y)) == ZZ.poly_ring(x, y)
    assert unify(ZZ.poly_ring(x), QQ.poly_ring(x, y)) == QQ.poly_ring(x, y)
    assert unify(QQ.poly_ring(x), ZZ.poly_ring(x, y)) == QQ.poly_ring(x, y)
    assert unify(QQ.poly_ring(x), QQ.poly_ring(x, y)) == QQ.poly_ring(x, y)

    assert unify(ZZ.poly_ring(x, y), ZZ.poly_ring(x, z)) == ZZ.poly_ring(x, y, z)
    assert unify(ZZ.poly_ring(x, y), QQ.poly_ring(x, z)) == QQ.poly_ring(x, y, z)
    assert unify(QQ.poly_ring(x, y), ZZ.poly_ring(x, z)) == QQ.poly_ring(x, y, z)
    assert unify(QQ.poly_ring(x, y), QQ.poly_ring(x, z)) == QQ.poly_ring(x, y, z)

    assert unify(ZZ.frac_field(x), ZZ.frac_field(x)) == ZZ.frac_field(x)
    assert unify(ZZ.frac_field(x), QQ.frac_field(x)) == QQ.frac_field(x)
    assert unify(QQ.frac_field(x), ZZ.frac_field(x)) == QQ.frac_field(x)
    assert unify(QQ.frac_field(x), QQ.frac_field(x)) == QQ.frac_field(x)

    assert unify(ZZ.frac_field(x, y), ZZ.frac_field(x)) == ZZ.frac_field(x, y)
    assert unify(ZZ.frac_field(x, y), QQ.frac_field(x)) == QQ.frac_field(x, y)
    assert unify(QQ.frac_field(x, y), ZZ.frac_field(x)) == QQ.frac_field(x, y)
    assert unify(QQ.frac_field(x, y), QQ.frac_field(x)) == QQ.frac_field(x, y)

    assert unify(ZZ.frac_field(x), ZZ.frac_field(x, y)) == ZZ.frac_field(x, y)
    assert unify(ZZ.frac_field(x), QQ.frac_field(x, y)) == QQ.frac_field(x, y)
    assert unify(QQ.frac_field(x), ZZ.frac_field(x, y)) == QQ.frac_field(x, y)
    assert unify(QQ.frac_field(x), QQ.frac_field(x, y)) == QQ.frac_field(x, y)

    assert unify(ZZ.frac_field(x, y), ZZ.frac_field(x, z)) == ZZ.frac_field(x, y, z)
    assert unify(ZZ.frac_field(x, y), QQ.frac_field(x, z)) == QQ.frac_field(x, y, z)
    assert unify(QQ.frac_field(x, y), ZZ.frac_field(x, z)) == QQ.frac_field(x, y, z)
    assert unify(QQ.frac_field(x, y), QQ.frac_field(x, z)) == QQ.frac_field(x, y, z)

    assert unify(ZZ.poly_ring(x), ZZ.frac_field(x)) == ZZ.frac_field(x)
    assert unify(ZZ.poly_ring(x), QQ.frac_field(x)) == ZZ.frac_field(x)
    assert unify(QQ.poly_ring(x), ZZ.frac_field(x)) == ZZ.frac_field(x)
    assert unify(QQ.poly_ring(x), QQ.frac_field(x)) == QQ.frac_field(x)

    assert unify(ZZ.poly_ring(x, y), ZZ.frac_field(x)) == ZZ.frac_field(x, y)
    assert unify(ZZ.poly_ring(x, y), QQ.frac_field(x)) == ZZ.frac_field(x, y)
    assert unify(QQ.poly_ring(x, y), ZZ.frac_field(x)) == ZZ.frac_field(x, y)
    assert unify(QQ.poly_ring(x, y), QQ.frac_field(x)) == QQ.frac_field(x, y)

    assert unify(ZZ.poly_ring(x), ZZ.frac_field(x, y)) == ZZ.frac_field(x, y)
    assert unify(ZZ.poly_ring(x), QQ.frac_field(x, y)) == ZZ.frac_field(x, y)
    assert unify(QQ.poly_ring(x), ZZ.frac_field(x, y)) == ZZ.frac_field(x, y)
    assert unify(QQ.poly_ring(x), QQ.frac_field(x, y)) == QQ.frac_field(x, y)

    assert unify(ZZ.poly_ring(x, y), ZZ.frac_field(x, z)) == ZZ.frac_field(x, y, z)
    assert unify(ZZ.poly_ring(x, y), QQ.frac_field(x, z)) == ZZ.frac_field(x, y, z)
    assert unify(QQ.poly_ring(x, y), ZZ.frac_field(x, z)) == ZZ.frac_field(x, y, z)
    assert unify(QQ.poly_ring(x, y), QQ.frac_field(x, z)) == QQ.frac_field(x, y, z)

    assert unify(ZZ.frac_field(x), ZZ.poly_ring(x)) == ZZ.frac_field(x)
    assert unify(ZZ.frac_field(x), QQ.poly_ring(x)) == ZZ.frac_field(x)
    assert unify(QQ.frac_field(x), ZZ.poly_ring(x)) == ZZ.frac_field(x)
    assert unify(QQ.frac_field(x), QQ.poly_ring(x)) == QQ.frac_field(x)

    assert unify(ZZ.frac_field(x, y), ZZ.poly_ring(x)) == ZZ.frac_field(x, y)
    assert unify(ZZ.frac_field(x, y), QQ.poly_ring(x)) == ZZ.frac_field(x, y)
    assert unify(QQ.frac_field(x, y), ZZ.poly_ring(x)) == ZZ.frac_field(x, y)
    assert unify(QQ.frac_field(x, y), QQ.poly_ring(x)) == QQ.frac_field(x, y)

    assert unify(ZZ.frac_field(x), ZZ.poly_ring(x, y)) == ZZ.frac_field(x, y)
    assert unify(ZZ.frac_field(x), QQ.poly_ring(x, y)) == ZZ.frac_field(x, y)
    assert unify(QQ.frac_field(x), ZZ.poly_ring(x, y)) == ZZ.frac_field(x, y)
    assert unify(QQ.frac_field(x), QQ.poly_ring(x, y)) == QQ.frac_field(x, y)

    assert unify(ZZ.frac_field(x, y), ZZ.poly_ring(x, z)) == ZZ.frac_field(x, y, z)
    assert unify(ZZ.frac_field(x, y), QQ.poly_ring(x, z)) == ZZ.frac_field(x, y, z)
    assert unify(QQ.frac_field(x, y), ZZ.poly_ring(x, z)) == ZZ.frac_field(x, y, z)
    assert unify(QQ.frac_field(x, y), QQ.poly_ring(x, z)) == QQ.frac_field(x, y, z)

def test_Domain_unify_algebraic():
    sqrt5 = QQ.algebraic_field(sqrt(5))
    sqrt7 = QQ.algebraic_field(sqrt(7))
    sqrt57 = QQ.algebraic_field(sqrt(5), sqrt(7))

    assert sqrt5.unify(sqrt7) == sqrt57

    assert sqrt5.unify(sqrt5[x, y]) == sqrt5[x, y]
    assert sqrt5[x, y].unify(sqrt5) == sqrt5[x, y]

    assert sqrt5.unify(sqrt5.frac_field(x, y)) == sqrt5.frac_field(x, y)
    assert sqrt5.frac_field(x, y).unify(sqrt5) == sqrt5.frac_field(x, y)

    assert sqrt5.unify(sqrt7[x, y]) == sqrt57[x, y]
    assert sqrt5[x, y].unify(sqrt7) == sqrt57[x, y]

    assert sqrt5.unify(sqrt7.frac_field(x, y)) == sqrt57.frac_field(x, y)
    assert sqrt5.frac_field(x, y).unify(sqrt7) == sqrt57.frac_field(x, y)

def test_Domain_unify_FiniteExtension():
    KxZZ = FiniteExtension(Poly(x**2 - 2, x, domain=ZZ))
    KxQQ = FiniteExtension(Poly(x**2 - 2, x, domain=QQ))
    KxZZy = FiniteExtension(Poly(x**2 - 2, x, domain=ZZ[y]))
    KxQQy = FiniteExtension(Poly(x**2 - 2, x, domain=QQ[y]))

    assert KxZZ.unify(KxZZ) == KxZZ
    assert KxQQ.unify(KxQQ) == KxQQ
    assert KxZZy.unify(KxZZy) == KxZZy
    assert KxQQy.unify(KxQQy) == KxQQy

    assert KxZZ.unify(ZZ) == KxZZ
    assert KxZZ.unify(QQ) == KxQQ
    assert KxQQ.unify(ZZ) == KxQQ
    assert KxQQ.unify(QQ) == KxQQ

    assert KxZZ.unify(ZZ[y]) == KxZZy
    assert KxZZ.unify(QQ[y]) == KxQQy
    assert KxQQ.unify(ZZ[y]) == KxQQy
    assert KxQQ.unify(QQ[y]) == KxQQy

    assert KxZZy.unify(ZZ) == KxZZy
    assert KxZZy.unify(QQ) == KxQQy
    assert KxQQy.unify(ZZ) == KxQQy
    assert KxQQy.unify(QQ) == KxQQy

    assert KxZZy.unify(ZZ[y]) == KxZZy
    assert KxZZy.unify(QQ[y]) == KxQQy
    assert KxQQy.unify(ZZ[y]) == KxQQy
    assert KxQQy.unify(QQ[y]) == KxQQy

    K = FiniteExtension(Poly(x**2 - 2, x, domain=ZZ[y]))
    assert K.unify(ZZ) == K
    assert K.unify(ZZ[x]) == K
    assert K.unify(ZZ[y]) == K
    assert K.unify(ZZ[x, y]) == K

    Kz = FiniteExtension(Poly(x**2 - 2, x, domain=ZZ[y, z]))
    assert K.unify(ZZ[z]) == Kz
    assert K.unify(ZZ[x, z]) == Kz
    assert K.unify(ZZ[y, z]) == Kz
    assert K.unify(ZZ[x, y, z]) == Kz

    Kx = FiniteExtension(Poly(x**2 - 2, x, domain=ZZ))
    Ky = FiniteExtension(Poly(y**2 - 2, y, domain=ZZ))
    Kxy = FiniteExtension(Poly(y**2 - 2, y, domain=Kx))
    assert Kx.unify(Kx) == Kx
    assert Ky.unify(Ky) == Ky
    assert Kx.unify(Ky) == Kxy
    assert Ky.unify(Kx) == Kxy

def test_Domain_unify_with_symbols():
    raises(UnificationFailed, lambda: ZZ[x, y].unify_with_symbols(ZZ, (y, z)))
    raises(UnificationFailed, lambda: ZZ.unify_with_symbols(ZZ[x, y], (y, z)))

def test_Domain__contains__():
    assert (0 in EX) is True
    assert (0 in ZZ) is True
    assert (0 in QQ) is True
    assert (0 in RR) is True
    assert (0 in CC) is True
    assert (0 in ALG) is True
    assert (0 in ZZ[x, y]) is True
    assert (0 in QQ[x, y]) is True
    assert (0 in RR[x, y]) is True

    assert (-7 in EX) is True
    assert (-7 in ZZ) is True
    assert (-7 in QQ) is True
    assert (-7 in RR) is True
    assert (-7 in CC) is True
    assert (-7 in ALG) is True
    assert (-7 in ZZ[x, y]) is True
    assert (-7 in QQ[x, y]) is True
    assert (-7 in RR[x, y]) is True

    assert (17 in EX) is True
    assert (17 in ZZ) is True
    assert (17 in QQ) is True
    assert (17 in RR) is True
    assert (17 in CC) is True
    assert (17 in ALG) is True
    assert (17 in ZZ[x, y]) is True
    assert (17 in QQ[x, y]) is True
    assert (17 in RR[x, y]) is True

    assert (Rational(-1, 7) in EX) is True
    assert (Rational(-1, 7) in ZZ) is False
    assert (Rational(-1, 7) in QQ) is True
    assert (Rational(-1, 7) in RR) is True
    assert (Rational(-1, 7) in CC) is True
    assert (Rational(-1, 7) in ALG) is True
    assert (Rational(-1, 7) in ZZ[x, y]) is False
    assert (Rational(-1, 7) in QQ[x, y]) is True
    assert (Rational(-1, 7) in RR[x, y]) is True

    assert (Rational(3, 5) in EX) is True
    assert (Rational(3, 5) in ZZ) is False
    assert (Rational(3, 5) in QQ) is True
    assert (Rational(3, 5) in RR) is True
    assert (Rational(3, 5) in CC) is True
    assert (Rational(3, 5) in ALG) is True
    assert (Rational(3, 5) in ZZ[x, y]) is False
    assert (Rational(3, 5) in QQ[x, y]) is True
    assert (Rational(3, 5) in RR[x, y]) is True

    assert (3.0 in EX) is True
    assert (3.0 in ZZ) is True
    assert (3.0 in QQ) is True
    assert (3.0 in RR) is True
    assert (3.0 in CC) is True
    assert (3.0 in ALG) is True
    assert (3.0 in ZZ[x, y]) is True
    assert (3.0 in QQ[x, y]) is True
    assert (3.0 in RR[x, y]) is True

    assert (3.14 in EX) is True
    assert (3.14 in ZZ) is False
    assert (3.14 in QQ) is True
    assert (3.14 in RR) is True
    assert (3.14 in CC) is True
    assert (3.14 in ALG) is True
    assert (3.14 in ZZ[x, y]) is False
    assert (3.14 in QQ[x, y]) is True
    assert (3.14 in RR[x, y]) is True

    assert (oo in ALG) is False
    assert (oo in ZZ[x, y]) is False
    assert (oo in QQ[x, y]) is False

    assert (-oo in ZZ) is False
    assert (-oo in QQ) is False
    assert (-oo in ALG) is False
    assert (-oo in ZZ[x, y]) is False
    assert (-oo in QQ[x, y]) is False

    assert (sqrt(7) in EX) is True
    assert (sqrt(7) in ZZ) is False
    assert (sqrt(7) in QQ) is False
    assert (sqrt(7) in RR) is True
    assert (sqrt(7) in CC) is True
    assert (sqrt(7) in ALG) is False
    assert (sqrt(7) in ZZ[x, y]) is False
    assert (sqrt(7) in QQ[x, y]) is False
    assert (sqrt(7) in RR[x, y]) is True

    assert (2*sqrt(3) + 1 in EX) is True
    assert (2*sqrt(3) + 1 in ZZ) is False
    assert (2*sqrt(3) + 1 in QQ) is False
    assert (2*sqrt(3) + 1 in RR) is True
    assert (2*sqrt(3) + 1 in CC) is True
    assert (2*sqrt(3) + 1 in ALG) is True
    assert (2*sqrt(3) + 1 in ZZ[x, y]) is False
    assert (2*sqrt(3) + 1 in QQ[x, y]) is False
    assert (2*sqrt(3) + 1 in RR[x, y]) is True

    assert (sin(1) in EX) is True
    assert (sin(1) in ZZ) is False
    assert (sin(1) in QQ) is False
    assert (sin(1) in RR) is True
    assert (sin(1) in CC) is True
    assert (sin(1) in ALG) is False
    assert (sin(1) in ZZ[x, y]) is False
    assert (sin(1) in QQ[x, y]) is False
    assert (sin(1) in RR[x, y]) is True

    assert (x**2 + 1 in EX) is True
    assert (x**2 + 1 in ZZ) is False
    assert (x**2 + 1 in QQ) is False
    assert (x**2 + 1 in RR) is False
    assert (x**2 + 1 in CC) is False
    assert (x**2 + 1 in ALG) is False
    assert (x**2 + 1 in ZZ[x]) is True
    assert (x**2 + 1 in QQ[x]) is True
    assert (x**2 + 1 in RR[x]) is True
    assert (x**2 + 1 in ZZ[x, y]) is True
    assert (x**2 + 1 in QQ[x, y]) is True
    assert (x**2 + 1 in RR[x, y]) is True

    assert (x**2 + y**2 in EX) is True
    assert (x**2 + y**2 in ZZ) is False
    assert (x**2 + y**2 in QQ) is False
    assert (x**2 + y**2 in RR) is False
    assert (x**2 + y**2 in CC) is False
    assert (x**2 + y**2 in ALG) is False
    assert (x**2 + y**2 in ZZ[x]) is False
    assert (x**2 + y**2 in QQ[x]) is False
    assert (x**2 + y**2 in RR[x]) is False
    assert (x**2 + y**2 in ZZ[x, y]) is True
    assert (x**2 + y**2 in QQ[x, y]) is True
    assert (x**2 + y**2 in RR[x, y]) is True

    assert (Rational(3, 2)*x/(y + 1) - z in QQ[x, y, z]) is False


def test_issue_14433():
    assert (Rational(2, 3)*x in QQ.frac_field(1/x)) is True
    assert (1/x in QQ.frac_field(x)) is True
    assert ((x**2 + y**2) in QQ.frac_field(1/x, 1/y)) is True
    assert ((x + y) in QQ.frac_field(1/x, y)) is True
    assert ((x - y) in QQ.frac_field(x, 1/y)) is True


def test_Domain_get_ring():
    assert ZZ.has_assoc_Ring is True
    assert QQ.has_assoc_Ring is True
    assert ZZ[x].has_assoc_Ring is True
    assert QQ[x].has_assoc_Ring is True
    assert ZZ[x, y].has_assoc_Ring is True
    assert QQ[x, y].has_assoc_Ring is True
    assert ZZ.frac_field(x).has_assoc_Ring is True
    assert QQ.frac_field(x).has_assoc_Ring is True
    assert ZZ.frac_field(x, y).has_assoc_Ring is True
    assert QQ.frac_field(x, y).has_assoc_Ring is True

    assert EX.has_assoc_Ring is False
    assert RR.has_assoc_Ring is False
    assert ALG.has_assoc_Ring is False

    assert ZZ.get_ring() == ZZ
    assert QQ.get_ring() == ZZ
    assert ZZ[x].get_ring() == ZZ[x]
    assert QQ[x].get_ring() == QQ[x]
    assert ZZ[x, y].get_ring() == ZZ[x, y]
    assert QQ[x, y].get_ring() == QQ[x, y]
    assert ZZ.frac_field(x).get_ring() == ZZ[x]
    assert QQ.frac_field(x).get_ring() == QQ[x]
    assert ZZ.frac_field(x, y).get_ring() == ZZ[x, y]
    assert QQ.frac_field(x, y).get_ring() == QQ[x, y]

    assert EX.get_ring() == EX

    assert RR.get_ring() == RR
    # XXX: This should also be like RR
    raises(DomainError, lambda: ALG.get_ring())


def test_Domain_get_field():
    assert EX.has_assoc_Field is True
    assert ZZ.has_assoc_Field is True
    assert QQ.has_assoc_Field is True
    assert RR.has_assoc_Field is True
    assert ALG.has_assoc_Field is True
    assert ZZ[x].has_assoc_Field is True
    assert QQ[x].has_assoc_Field is True
    assert ZZ[x, y].has_assoc_Field is True
    assert QQ[x, y].has_assoc_Field is True

    assert EX.get_field() == EX
    assert ZZ.get_field() == QQ
    assert QQ.get_field() == QQ
    assert RR.get_field() == RR
    assert ALG.get_field() == ALG
    assert ZZ[x].get_field() == ZZ.frac_field(x)
    assert QQ[x].get_field() == QQ.frac_field(x)
    assert ZZ[x, y].get_field() == ZZ.frac_field(x, y)
    assert QQ[x, y].get_field() == QQ.frac_field(x, y)


def test_Domain_set_domain():
    doms = [GF(5), ZZ, QQ, ALG, RR, CC, EX, ZZ[z], QQ[z], RR[z], CC[z], EX[z]]
    for D1 in doms:
        for D2 in doms:
            assert D1[x].set_domain(D2) == D2[x]
            assert D1[x, y].set_domain(D2) == D2[x, y]
            assert D1.frac_field(x).set_domain(D2) == D2.frac_field(x)
            assert D1.frac_field(x, y).set_domain(D2) == D2.frac_field(x, y)
            assert D1.old_poly_ring(x).set_domain(D2) == D2.old_poly_ring(x)
            assert D1.old_poly_ring(x, y).set_domain(D2) == D2.old_poly_ring(x, y)
            assert D1.old_frac_field(x).set_domain(D2) == D2.old_frac_field(x)
            assert D1.old_frac_field(x, y).set_domain(D2) == D2.old_frac_field(x, y)


def test_Domain_is_Exact():
    exact = [GF(5), ZZ, QQ, ALG, EX]
    inexact = [RR, CC]
    for D in exact + inexact:
        for R in D, D[x], D.frac_field(x), D.old_poly_ring(x), D.old_frac_field(x):
            if D in exact:
                assert R.is_Exact is True
            else:
                assert R.is_Exact is False


def test_Domain_get_exact():
    assert EX.get_exact() == EX
    assert ZZ.get_exact() == ZZ
    assert QQ.get_exact() == QQ
    assert RR.get_exact() == QQ
    assert CC.get_exact() == QQ_I
    assert ALG.get_exact() == ALG
    assert ZZ[x].get_exact() == ZZ[x]
    assert QQ[x].get_exact() == QQ[x]
    assert RR[x].get_exact() == QQ[x]
    assert CC[x].get_exact() == QQ_I[x]
    assert ZZ[x, y].get_exact() == ZZ[x, y]
    assert QQ[x, y].get_exact() == QQ[x, y]
    assert RR[x, y].get_exact() == QQ[x, y]
    assert CC[x, y].get_exact() == QQ_I[x, y]
    assert ZZ.frac_field(x).get_exact() == ZZ.frac_field(x)
    assert QQ.frac_field(x).get_exact() == QQ.frac_field(x)
    assert RR.frac_field(x).get_exact() == QQ.frac_field(x)
    assert CC.frac_field(x).get_exact() == QQ_I.frac_field(x)
    assert ZZ.frac_field(x, y).get_exact() == ZZ.frac_field(x, y)
    assert QQ.frac_field(x, y).get_exact() == QQ.frac_field(x, y)
    assert RR.frac_field(x, y).get_exact() == QQ.frac_field(x, y)
    assert CC.frac_field(x, y).get_exact() == QQ_I.frac_field(x, y)
    assert ZZ.old_poly_ring(x).get_exact() == ZZ.old_poly_ring(x)
    assert QQ.old_poly_ring(x).get_exact() == QQ.old_poly_ring(x)
    assert RR.old_poly_ring(x).get_exact() == QQ.old_poly_ring(x)
    assert CC.old_poly_ring(x).get_exact() == QQ_I.old_poly_ring(x)
    assert ZZ.old_poly_ring(x, y).get_exact() == ZZ.old_poly_ring(x, y)
    assert QQ.old_poly_ring(x, y).get_exact() == QQ.old_poly_ring(x, y)
    assert RR.old_poly_ring(x, y).get_exact() == QQ.old_poly_ring(x, y)
    assert CC.old_poly_ring(x, y).get_exact() == QQ_I.old_poly_ring(x, y)
    assert ZZ.old_frac_field(x).get_exact() == ZZ.old_frac_field(x)
    assert QQ.old_frac_field(x).get_exact() == QQ.old_frac_field(x)
    assert RR.old_frac_field(x).get_exact() == QQ.old_frac_field(x)
    assert CC.old_frac_field(x).get_exact() == QQ_I.old_frac_field(x)
    assert ZZ.old_frac_field(x, y).get_exact() == ZZ.old_frac_field(x, y)
    assert QQ.old_frac_field(x, y).get_exact() == QQ.old_frac_field(x, y)
    assert RR.old_frac_field(x, y).get_exact() == QQ.old_frac_field(x, y)
    assert CC.old_frac_field(x, y).get_exact() == QQ_I.old_frac_field(x, y)


def test_Domain_characteristic():
    for F, c in [(FF(3), 3), (FF(5), 5), (FF(7), 7)]:
        for R in F, F[x], F.frac_field(x), F.old_poly_ring(x), F.old_frac_field(x):
            assert R.has_CharacteristicZero is False
            assert R.characteristic() == c
    for D in ZZ, QQ, ZZ_I, QQ_I, ALG:
        for R in D, D[x], D.frac_field(x), D.old_poly_ring(x), D.old_frac_field(x):
            assert R.has_CharacteristicZero is True
            assert R.characteristic() == 0


def test_Domain_is_unit():
    nums = [-2, -1, 0, 1, 2]
    invring = [False, True, False, True, False]
    invfield = [True, True, False, True, True]
    ZZx, QQx, QQxf = ZZ[x], QQ[x], QQ.frac_field(x)
    assert [ZZ.is_unit(ZZ(n)) for n in nums] == invring
    assert [QQ.is_unit(QQ(n)) for n in nums] == invfield
    assert [ZZx.is_unit(ZZx(n)) for n in nums] == invring
    assert [QQx.is_unit(QQx(n)) for n in nums] == invfield
    assert [QQxf.is_unit(QQxf(n)) for n in nums] == invfield
    assert ZZx.is_unit(ZZx(x)) is False
    assert QQx.is_unit(QQx(x)) is False
    assert QQxf.is_unit(QQxf(x)) is True


def test_Domain_convert():

    def check_element(e1, e2, K1, K2, K3):
        assert type(e1) is type(e2), '%s, %s: %s %s -> %s' % (e1, e2, K1, K2, K3)
        assert e1 == e2, '%s, %s: %s %s -> %s' % (e1, e2, K1, K2, K3)

    def check_domains(K1, K2):
        K3 = K1.unify(K2)
        check_element(K3.convert_from(K1.one, K1),  K3.one,  K1, K2, K3)
        check_element(K3.convert_from(K2.one, K2),  K3.one,  K1, K2, K3)
        check_element(K3.convert_from(K1.zero, K1), K3.zero, K1, K2, K3)
        check_element(K3.convert_from(K2.zero, K2), K3.zero, K1, K2, K3)

    def composite_domains(K):
        domains = [
            K,
            K[y], K[z], K[y, z],
            K.frac_field(y), K.frac_field(z), K.frac_field(y, z),
            # XXX: These should be tested and made to work...
            # K.old_poly_ring(y), K.old_frac_field(y),
        ]
        return domains

    QQ2 = QQ.algebraic_field(sqrt(2))
    QQ3 = QQ.algebraic_field(sqrt(3))
    doms = [ZZ, QQ, QQ2, QQ3, QQ_I, ZZ_I, RR, CC]

    for i, K1 in enumerate(doms):
        for K2 in doms[i:]:
            for K3 in composite_domains(K1):
                for K4 in composite_domains(K2):
                    check_domains(K3, K4)

    assert QQ.convert(10e-52) == QQ(1684996666696915, 1684996666696914987166688442938726917102321526408785780068975640576)

    R, xr = ring("x", ZZ)
    assert ZZ.convert(xr - xr) == 0
    assert ZZ.convert(xr - xr, R.to_domain()) == 0

    assert CC.convert(ZZ_I(1, 2)) == CC(1, 2)
    assert CC.convert(QQ_I(1, 2)) == CC(1, 2)

    assert QQ.convert_from(RR(0.5), RR) == QQ(1, 2)
    assert RR.convert_from(QQ(1, 2), QQ) == RR(0.5)
    assert QQ_I.convert_from(CC(0.5, 0.75), CC) == QQ_I(QQ(1, 2), QQ(3, 4))
    assert CC.convert_from(QQ_I(QQ(1, 2), QQ(3, 4)), QQ_I) == CC(0.5, 0.75)

    K1 = QQ.frac_field(x)
    K2 = ZZ.frac_field(x)
    K3 = QQ[x]
    K4 = ZZ[x]
    Ks = [K1, K2, K3, K4]
    for Ka, Kb in product(Ks, Ks):
        assert Ka.convert_from(Kb.from_sympy(x), Kb) == Ka.from_sympy(x)

    assert K2.convert_from(QQ(1, 2), QQ) == K2(QQ(1, 2))


def test_EX_convert():

    elements = [
        (ZZ, ZZ(3)),
        (QQ, QQ(1,2)),
        (ZZ_I, ZZ_I(1,2)),
        (QQ_I, QQ_I(1,2)),
        (RR, RR(3)),
        (CC, CC(1,2)),
        (EX, EX(3)),
        (EXRAW, EXRAW(3)),
        (ALG, ALG.from_sympy(sqrt(2))),
    ]

    for R, e in elements:
        for EE in EX, EXRAW:
            elem = EE.from_sympy(R.to_sympy(e))
            assert EE.convert_from(e, R) == elem
            assert R.convert_from(elem, EE) == e


def test_GlobalPolynomialRing_convert():
    K1 = QQ.old_poly_ring(x)
    K2 = QQ[x]
    assert K1.convert(x) == K1.convert(K2.convert(x), K2)
    assert K2.convert(x) == K2.convert(K1.convert(x), K1)

    K1 = QQ.old_poly_ring(x, y)
    K2 = QQ[x]
    assert K1.convert(x) == K1.convert(K2.convert(x), K2)
    #assert K2.convert(x) == K2.convert(K1.convert(x), K1)

    K1 = ZZ.old_poly_ring(x, y)
    K2 = QQ[x]
    assert K1.convert(x) == K1.convert(K2.convert(x), K2)
    #assert K2.convert(x) == K2.convert(K1.convert(x), K1)


def test_PolynomialRing__init():
    R, = ring("", ZZ)
    assert ZZ.poly_ring() == R.to_domain()


def test_FractionField__init():
    F, = field("", ZZ)
    assert ZZ.frac_field() == F.to_domain()


def test_FractionField_convert():
    K = QQ.frac_field(x)
    assert K.convert(QQ(2, 3), QQ) == K.from_sympy(Rational(2, 3))
    K = QQ.frac_field(x)
    assert K.convert(ZZ(2), ZZ) == K.from_sympy(Integer(2))


def test_inject():
    assert ZZ.inject(x, y, z) == ZZ[x, y, z]
    assert ZZ[x].inject(y, z) == ZZ[x, y, z]
    assert ZZ.frac_field(x).inject(y, z) == ZZ.frac_field(x, y, z)
    raises(GeneratorsError, lambda: ZZ[x].inject(x))


def test_drop():
    assert ZZ.drop(x) == ZZ
    assert ZZ[x].drop(x) == ZZ
    assert ZZ[x, y].drop(x) == ZZ[y]
    assert ZZ.frac_field(x).drop(x) == ZZ
    assert ZZ.frac_field(x, y).drop(x) == ZZ.frac_field(y)
    assert ZZ[x][y].drop(y) == ZZ[x]
    assert ZZ[x][y].drop(x) == ZZ[y]
    assert ZZ.frac_field(x)[y].drop(x) == ZZ[y]
    assert ZZ.frac_field(x)[y].drop(y) == ZZ.frac_field(x)
    Ky = FiniteExtension(Poly(x**2-1, x, domain=ZZ[y]))
    K = FiniteExtension(Poly(x**2-1, x, domain=ZZ))
    assert Ky.drop(y) == K
    raises(GeneratorsError, lambda: Ky.drop(x))


def test_Domain_map():
    seq = ZZ.map([1, 2, 3, 4])

    assert all(ZZ.of_type(elt) for elt in seq)

    seq = ZZ.map([[1, 2, 3, 4]])

    assert all(ZZ.of_type(elt) for elt in seq[0]) and len(seq) == 1


def test_Domain___eq__():
    assert (ZZ[x, y] == ZZ[x, y]) is True
    assert (QQ[x, y] == QQ[x, y]) is True

    assert (ZZ[x, y] == QQ[x, y]) is False
    assert (QQ[x, y] == ZZ[x, y]) is False

    assert (ZZ.frac_field(x, y) == ZZ.frac_field(x, y)) is True
    assert (QQ.frac_field(x, y) == QQ.frac_field(x, y)) is True

    assert (ZZ.frac_field(x, y) == QQ.frac_field(x, y)) is False
    assert (QQ.frac_field(x, y) == ZZ.frac_field(x, y)) is False

    assert RealField()[x] == RR[x]


def test_Domain__algebraic_field():
    alg = ZZ.algebraic_field(sqrt(2))
    assert alg.ext.minpoly == Poly(x**2 - 2)
    assert alg.dom == QQ

    alg = QQ.algebraic_field(sqrt(2))
    assert alg.ext.minpoly == Poly(x**2 - 2)
    assert alg.dom == QQ

    alg = alg.algebraic_field(sqrt(3))
    assert alg.ext.minpoly == Poly(x**4 - 10*x**2 + 1)
    assert alg.dom == QQ


def test_Domain_alg_field_from_poly():
    f = Poly(x**2 - 2)
    g = Poly(x**2 - 3)
    h = Poly(x**4 - 10*x**2 + 1)

    alg = ZZ.alg_field_from_poly(f)
    assert alg.ext.minpoly == f
    assert alg.dom == QQ

    alg = QQ.alg_field_from_poly(f)
    assert alg.ext.minpoly == f
    assert alg.dom == QQ

    alg = alg.alg_field_from_poly(g)
    assert alg.ext.minpoly == h
    assert alg.dom == QQ


def test_Domain_cyclotomic_field():
    K = ZZ.cyclotomic_field(12)
    assert K.ext.minpoly == Poly(cyclotomic_poly(12))
    assert K.dom == QQ

    F = QQ.cyclotomic_field(3)
    assert F.ext.minpoly == Poly(cyclotomic_poly(3))
    assert F.dom == QQ

    E = F.cyclotomic_field(4)
    assert field_isomorphism(E.ext, K.ext) is not None
    assert E.dom == QQ


def test_PolynomialRing_from_FractionField():
    F, x,y = field("x,y", ZZ)
    R, X,Y = ring("x,y", ZZ)

    f = (x**2 + y**2)/(x + 1)
    g = (x**2 + y**2)/4
    h =  x**2 + y**2

    assert R.to_domain().from_FractionField(f, F.to_domain()) is None
    assert R.to_domain().from_FractionField(g, F.to_domain()) == X**2/4 + Y**2/4
    assert R.to_domain().from_FractionField(h, F.to_domain()) == X**2 + Y**2

    F, x,y = field("x,y", QQ)
    R, X,Y = ring("x,y", QQ)

    f = (x**2 + y**2)/(x + 1)
    g = (x**2 + y**2)/4
    h =  x**2 + y**2

    assert R.to_domain().from_FractionField(f, F.to_domain()) is None
    assert R.to_domain().from_FractionField(g, F.to_domain()) == X**2/4 + Y**2/4
    assert R.to_domain().from_FractionField(h, F.to_domain()) == X**2 + Y**2


def test_FractionField_from_PolynomialRing():
    R, x,y = ring("x,y", QQ)
    F, X,Y = field("x,y", ZZ)

    f = 3*x**2 + 5*y**2
    g = x**2/3 + y**2/5

    assert F.to_domain().from_PolynomialRing(f, R.to_domain()) == 3*X**2 + 5*Y**2
    assert F.to_domain().from_PolynomialRing(g, R.to_domain()) == (5*X**2 + 3*Y**2)/15


def test_FF_of_type():
    # XXX: of_type is not very useful here because in the case of ground types
    # = flint all elements are of type nmod.
    assert FF(3).of_type(FF(3)(1)) is True
    assert FF(5).of_type(FF(5)(3)) is True


def test___eq__():
    assert not QQ[x] == ZZ[x]
    assert not QQ.frac_field(x) == ZZ.frac_field(x)


def test_RealField_from_sympy():
    assert RR.convert(S.Zero) == RR.dtype(0)
    assert RR.convert(S(0.0)) == RR.dtype(0.0)
    assert RR.convert(S.One) == RR.dtype(1)
    assert RR.convert(S(1.0)) == RR.dtype(1.0)
    assert RR.convert(sin(1)) == RR.dtype(sin(1).evalf())


def test_not_in_any_domain():
    check = list(_illegal) + [x] + [
        float(i) for i in _illegal[:3]]
    for dom in (ZZ, QQ, RR, CC, EX):
        for i in check:
            if i == x and dom == EX:
                continue
            assert i not in dom, (i, dom)
            raises(CoercionFailed, lambda: dom.convert(i))


def test_ModularInteger():
    F3 = FF(3)

    a = F3(0)
    assert F3.of_type(a) and a == 0
    a = F3(1)
    assert F3.of_type(a) and a == 1
    a = F3(2)
    assert F3.of_type(a) and a == 2
    a = F3(3)
    assert F3.of_type(a) and a == 0
    a = F3(4)
    assert F3.of_type(a) and a == 1

    a = F3(F3(0))
    assert F3.of_type(a) and a == 0
    a = F3(F3(1))
    assert F3.of_type(a) and a == 1
    a = F3(F3(2))
    assert F3.of_type(a) and a == 2
    a = F3(F3(3))
    assert F3.of_type(a) and a == 0
    a = F3(F3(4))
    assert F3.of_type(a) and a == 1

    a = -F3(1)
    assert F3.of_type(a) and a == 2
    a = -F3(2)
    assert F3.of_type(a) and a == 1

    a = 2 + F3(2)
    assert F3.of_type(a) and a == 1
    a = F3(2) + 2
    assert F3.of_type(a) and a == 1
    a = F3(2) + F3(2)
    assert F3.of_type(a) and a == 1
    a = F3(2) + F3(2)
    assert F3.of_type(a) and a == 1

    a = 3 - F3(2)
    assert F3.of_type(a) and a == 1
    a = F3(3) - 2
    assert F3.of_type(a) and a == 1
    a = F3(3) - F3(2)
    assert F3.of_type(a) and a == 1
    a = F3(3) - F3(2)
    assert F3.of_type(a) and a == 1

    a = 2*F3(2)
    assert F3.of_type(a) and a == 1
    a = F3(2)*2
    assert F3.of_type(a) and a == 1
    a = F3(2)*F3(2)
    assert F3.of_type(a) and a == 1
    a = F3(2)*F3(2)
    assert F3.of_type(a) and a == 1

    a = 2/F3(2)
    assert F3.of_type(a) and a == 1
    a = F3(2)/2
    assert F3.of_type(a) and a == 1
    a = F3(2)/F3(2)
    assert F3.of_type(a) and a == 1
    a = F3(2)/F3(2)
    assert F3.of_type(a) and a == 1

    a = F3(2)**0
    assert F3.of_type(a) and a == 1
    a = F3(2)**1
    assert F3.of_type(a) and a == 2
    a = F3(2)**2
    assert F3.of_type(a) and a == 1

    F7 = FF(7)

    a = F7(3)**100000000000
    assert F7.of_type(a) and a == 4
    a = F7(3)**-100000000000
    assert F7.of_type(a) and a == 2

    assert bool(F3(3)) is False
    assert bool(F3(4)) is True

    F5 = FF(5)

    a = F5(1)**(-1)
    assert F5.of_type(a) and a == 1
    a = F5(2)**(-1)
    assert F5.of_type(a) and a == 3
    a = F5(3)**(-1)
    assert F5.of_type(a) and a == 2
    a = F5(4)**(-1)
    assert F5.of_type(a) and a == 4

    if GROUND_TYPES != 'flint':
        # XXX: This gives a core dump with python-flint...
        raises(NotInvertible, lambda: F5(0)**(-1))
        raises(NotInvertible, lambda: F5(5)**(-1))

    raises(ValueError, lambda: FF(0))
    raises(ValueError, lambda: FF(2.1))

    for n1 in range(5):
        for n2 in range(5):
            if GROUND_TYPES != 'flint':
                with warns_deprecated_sympy():
                    assert (F5(n1) < F5(n2)) is (n1 < n2)
                with warns_deprecated_sympy():
                    assert (F5(n1) <= F5(n2)) is (n1 <= n2)
                with warns_deprecated_sympy():
                    assert (F5(n1) > F5(n2)) is (n1 > n2)
                with warns_deprecated_sympy():
                    assert (F5(n1) >= F5(n2)) is (n1 >= n2)
            else:
                raises(TypeError, lambda: F5(n1) < F5(n2))
                raises(TypeError, lambda: F5(n1) <= F5(n2))
                raises(TypeError, lambda: F5(n1) > F5(n2))
                raises(TypeError, lambda: F5(n1) >= F5(n2))

    # https://github.com/sympy/sympy/issues/26789
    assert GF(Integer(5)) == F5
    assert F5(Integer(3)) == F5(3)


def test_QQ_int():
    assert int(QQ(2**2000, 3**1250)) == 455431
    assert int(QQ(2**100, 3)) == 422550200076076467165567735125


def test_RR_double():
    assert RR(3.14) > 1e-50
    assert RR(1e-13) > 1e-50
    assert RR(1e-14) > 1e-50
    assert RR(1e-15) > 1e-50
    assert RR(1e-20) > 1e-50
    assert RR(1e-40) > 1e-50


def test_RR_Float():
    f1 = Float("1.01")
    f2 = Float("1.0000000000000000000001")
    assert f1._prec == 53
    assert f2._prec == 80
    assert RR(f1)-1 > 1e-50
    assert RR(f2)-1 < 1e-50 # RR's precision is lower than f2's

    RR2 = RealField(prec=f2._prec)
    assert RR2(f1)-1 > 1e-50
    assert RR2(f2)-1 > 1e-50 # RR's precision is equal to f2's


def test_CC_double():
    assert CC(3.14).real > 1e-50
    assert CC(1e-13).real > 1e-50
    assert CC(1e-14).real > 1e-50
    assert CC(1e-15).real > 1e-50
    assert CC(1e-20).real > 1e-50
    assert CC(1e-40).real > 1e-50

    assert CC(3.14j).imag > 1e-50
    assert CC(1e-13j).imag > 1e-50
    assert CC(1e-14j).imag > 1e-50
    assert CC(1e-15j).imag > 1e-50
    assert CC(1e-20j).imag > 1e-50
    assert CC(1e-40j).imag > 1e-50


def test_gaussian_domains():
    I = S.ImaginaryUnit
    a, b, c, d = [ZZ_I.convert(x) for x in (5, 2 + I, 3 - I, 5 - 5*I)]
    assert ZZ_I.gcd(a, b) == b
    assert ZZ_I.gcd(a, c) == b
    assert ZZ_I.lcm(a, b) == a
    assert ZZ_I.lcm(a, c) == d
    assert ZZ_I(3, 4) != QQ_I(3, 4)  # XXX is this right or should QQ->ZZ if possible?
    assert ZZ_I(3, 0) != 3           # and should this go to Integer?
    assert QQ_I(S(3)/4, 0) != S(3)/4 # and this to Rational?
    assert ZZ_I(0, 0).quadrant() == 0
    assert ZZ_I(-1, 0).quadrant() == 2

    assert QQ_I.convert(QQ(3, 2)) == QQ_I(QQ(3, 2), QQ(0))
    assert QQ_I.convert(QQ(3, 2), QQ) == QQ_I(QQ(3, 2), QQ(0))

    for G in (QQ_I, ZZ_I):

        q = G(3, 4)
        assert str(q) == '3 + 4*I'
        assert q.parent() == G
        assert q._get_xy(pi) == (None, None)
        assert q._get_xy(2) == (2, 0)
        assert q._get_xy(2*I) == (0, 2)

        assert hash(q) == hash((3, 4))
        assert G(1, 2) == G(1, 2)
        assert G(1, 2) != G(1, 3)
        assert G(3, 0) == G(3)

        assert q + q == G(6, 8)
        assert q - q == G(0, 0)
        assert 3 - q  == -q + 3 == G(0, -4)
        assert 3 + q == q + 3 == G(6, 4)
        assert q * q == G(-7, 24)
        assert 3 * q == q * 3 == G(9, 12)
        assert q ** 0 == G(1, 0)
        assert q ** 1 == q
        assert q ** 2 == q * q == G(-7, 24)
        assert q ** 3 == q * q * q == G(-117, 44)
        assert 1 / q == q ** -1 == QQ_I(S(3)/25, - S(4)/25)
        assert q / 1 == QQ_I(3, 4)
        assert q / 2 == QQ_I(S(3)/2, 2)
        assert q/3 == QQ_I(1, S(4)/3)
        assert 3/q == QQ_I(S(9)/25, -S(12)/25)
        i, r = divmod(q, 2)
        assert 2*i + r == q
        i, r = divmod(2, q)
        assert q*i + r == G(2, 0)

        raises(ZeroDivisionError, lambda: q % 0)
        raises(ZeroDivisionError, lambda: q / 0)
        raises(ZeroDivisionError, lambda: q // 0)
        raises(ZeroDivisionError, lambda: divmod(q, 0))
        raises(ZeroDivisionError, lambda: divmod(q, 0))
        raises(TypeError, lambda: q + x)
        raises(TypeError, lambda: q - x)
        raises(TypeError, lambda: x + q)
        raises(TypeError, lambda: x - q)
        raises(TypeError, lambda: q * x)
        raises(TypeError, lambda: x * q)
        raises(TypeError, lambda: q / x)
        raises(TypeError, lambda: x / q)
        raises(TypeError, lambda: q // x)
        raises(TypeError, lambda: x // q)

        assert G.from_sympy(S(2)) == G(2, 0)
        assert G.to_sympy(G(2, 0)) == S(2)
        raises(CoercionFailed, lambda: G.from_sympy(pi))

        PR = G.inject(x)
        assert isinstance(PR, PolynomialRing)
        assert PR.domain == G
        assert len(PR.gens) == 1 and PR.gens[0].as_expr() == x

        if G is QQ_I:
            AF = G.as_AlgebraicField()
            assert isinstance(AF, AlgebraicField)
            assert AF.domain == QQ
            assert AF.ext.args[0] == I

        for qi in [G(-1, 0), G(1, 0), G(0, -1), G(0, 1)]:
            assert G.is_negative(qi) is False
            assert G.is_positive(qi) is False
            assert G.is_nonnegative(qi) is False
            assert G.is_nonpositive(qi) is False

        domains = [ZZ, QQ, AlgebraicField(QQ, I)]

        # XXX: These domains are all obsolete because ZZ/QQ with MPZ/MPQ
        # already use either gmpy, flint or python depending on the
        # availability of these libraries. We can keep these tests for now but
        # ideally we should remove these alternate domains entirely.
        domains += [ZZ_python(), QQ_python()]
        if GROUND_TYPES == 'gmpy':
            domains += [ZZ_gmpy(), QQ_gmpy()]

        for K in domains:
            assert G.convert(K(2)) == G(2, 0)
            assert G.convert(K(2), K) == G(2, 0)

        for K in ZZ_I, QQ_I:
            assert G.convert(K(1, 1)) == G(1, 1)
            assert G.convert(K(1, 1), K) == G(1, 1)

        if G == ZZ_I:
            assert repr(q) == 'ZZ_I(3, 4)'
            assert q//3 == G(1, 1)
            assert 12//q == G(1, -2)
            assert 12 % q == G(1, 2)
            assert q % 2 == G(-1, 0)
            assert i == G(0, 0)
            assert r == G(2, 0)
            assert G.get_ring() == G
            assert G.get_field() == QQ_I
        else:
            assert repr(q) == 'QQ_I(3, 4)'
            assert G.get_ring() == ZZ_I
            assert G.get_field() == G
            assert q//3 == G(1, S(4)/3)
            assert 12//q == G(S(36)/25, -S(48)/25)
            assert 12 % q == G(0, 0)
            assert q % 2 == G(0, 0)
            assert i == G(S(6)/25, -S(8)/25), (G,i)
            assert r == G(0, 0)
            q2 = G(S(3)/2, S(5)/3)
            assert G.numer(q2) == ZZ_I(9, 10)
            assert G.denom(q2) == ZZ_I(6)


def test_EX_EXRAW():
    assert EXRAW.zero is S.Zero
    assert EXRAW.one is S.One

    assert EX(1) == EX.Expression(1)
    assert EX(1).ex is S.One
    assert EXRAW(1) is S.One

    # EX has cancelling but EXRAW does not
    assert 2*EX((x + y*x)/x) == EX(2 + 2*y) != 2*((x + y*x)/x)
    assert 2*EXRAW((x + y*x)/x) == 2*((x + y*x)/x) != (1 + y)

    assert EXRAW.convert_from(EX(1), EX) is EXRAW.one
    assert EX.convert_from(EXRAW(1), EXRAW) == EX.one

    assert EXRAW.from_sympy(S.One) is S.One
    assert EXRAW.to_sympy(EXRAW.one) is S.One
    raises(CoercionFailed, lambda: EXRAW.from_sympy([]))

    assert EXRAW.get_field() == EXRAW

    assert EXRAW.unify(EX) == EXRAW
    assert EX.unify(EXRAW) == EXRAW


def test_EX_ordering():
    elements = [EX(1), EX(x), EX(3)]
    assert sorted(elements) == [EX(1), EX(3), EX(x)]


def test_canonical_unit():

    for K in [ZZ, QQ, RR]: # CC?
        assert K.canonical_unit(K(2)) == K(1)
        assert K.canonical_unit(K(-2)) == K(-1)

    for K in [ZZ_I, QQ_I]:
        i = K.from_sympy(I)
        assert K.canonical_unit(K(2)) == K(1)
        assert K.canonical_unit(K(2)*i) == -i
        assert K.canonical_unit(-K(2)) == K(-1)
        assert K.canonical_unit(-K(2)*i) == i

    K = ZZ[x]
    assert K.canonical_unit(K(x + 1)) == K(1)
    assert K.canonical_unit(K(-x + 1)) == K(-1)

    K = ZZ_I[x]
    assert K.canonical_unit(K.from_sympy(I*x)) == ZZ_I(0, -1)

    K = ZZ_I.frac_field(x, y)
    i = K.from_sympy(I)
    assert i / i == K.one
    assert (K.one + i)/(i - K.one) == -i


def test_issue_18278():
    assert str(RR(2).parent()) == 'RR'
    assert str(CC(2).parent()) == 'CC'


def test_Domain_is_negative():
    I = S.ImaginaryUnit
    a, b = [CC.convert(x) for x in (2 + I, 5)]
    assert CC.is_negative(a) == False
    assert CC.is_negative(b) == False


def test_Domain_is_positive():
    I = S.ImaginaryUnit
    a, b = [CC.convert(x) for x in (2 + I, 5)]
    assert CC.is_positive(a) == False
    assert CC.is_positive(b) == False


def test_Domain_is_nonnegative():
    I = S.ImaginaryUnit
    a, b = [CC.convert(x) for x in (2 + I, 5)]
    assert CC.is_nonnegative(a) == False
    assert CC.is_nonnegative(b) == False


def test_Domain_is_nonpositive():
    I = S.ImaginaryUnit
    a, b = [CC.convert(x) for x in (2 + I, 5)]
    assert CC.is_nonpositive(a) == False
    assert CC.is_nonpositive(b) == False


def test_exponential_domain():
    K = ZZ[E]
    eK = K.from_sympy(E)
    assert K.from_sympy(exp(3)) == eK ** 3
    assert K.convert(exp(3)) == eK ** 3


def test_AlgebraicField_alias():
    # No default alias:
    k = QQ.algebraic_field(sqrt(2))
    assert k.ext.alias is None

    # For a single extension, its alias is used:
    alpha = AlgebraicNumber(sqrt(2), alias='alpha')
    k = QQ.algebraic_field(alpha)
    assert k.ext.alias.name == 'alpha'

    # Can override the alias of a single extension:
    k = QQ.algebraic_field(alpha, alias='theta')
    assert k.ext.alias.name == 'theta'

    # With multiple extensions, no default alias:
    k = QQ.algebraic_field(sqrt(2), sqrt(3))
    assert k.ext.alias is None

    # With multiple extensions, no default alias, even if one of
    # the extensions has one:
    k = QQ.algebraic_field(alpha, sqrt(3))
    assert k.ext.alias is None

    # With multiple extensions, may set an alias:
    k = QQ.algebraic_field(sqrt(2), sqrt(3), alias='theta')
    assert k.ext.alias.name == 'theta'

    # Alias is passed to constructed field elements:
    k = QQ.algebraic_field(alpha)
    beta = k.to_alg_num(k([1, 2, 3]))
    assert beta.alias is alpha.alias


def test_exsqrt():
    assert ZZ.is_square(ZZ(4)) is True
    assert ZZ.exsqrt(ZZ(4)) == ZZ(2)
    assert ZZ.is_square(ZZ(42)) is False
    assert ZZ.exsqrt(ZZ(42)) is None
    assert ZZ.is_square(ZZ(0)) is True
    assert ZZ.exsqrt(ZZ(0)) == ZZ(0)
    assert ZZ.is_square(ZZ(-1)) is False
    assert ZZ.exsqrt(ZZ(-1)) is None

    assert QQ.is_square(QQ(9, 4)) is True
    assert QQ.exsqrt(QQ(9, 4)) == QQ(3, 2)
    assert QQ.is_square(QQ(18, 8)) is True
    assert QQ.exsqrt(QQ(18, 8)) == QQ(3, 2)
    assert QQ.is_square(QQ(-9, -4)) is True
    assert QQ.exsqrt(QQ(-9, -4)) == QQ(3, 2)
    assert QQ.is_square(QQ(11, 4)) is False
    assert QQ.exsqrt(QQ(11, 4)) is None
    assert QQ.is_square(QQ(9, 5)) is False
    assert QQ.exsqrt(QQ(9, 5)) is None
    assert QQ.is_square(QQ(4)) is True
    assert QQ.exsqrt(QQ(4)) == QQ(2)
    assert QQ.is_square(QQ(0)) is True
    assert QQ.exsqrt(QQ(0)) == QQ(0)
    assert QQ.is_square(QQ(-16, 9)) is False
    assert QQ.exsqrt(QQ(-16, 9)) is None

    assert RR.is_square(RR(6.25)) is True
    assert RR.exsqrt(RR(6.25)) == RR(2.5)
    assert RR.is_square(RR(2)) is True
    assert RR.almosteq(RR.exsqrt(RR(2)), RR(1.4142135623730951), tolerance=1e-15)
    assert RR.is_square(RR(0)) is True
    assert RR.exsqrt(RR(0)) == RR(0)
    assert RR.is_square(RR(-1)) is False
    assert RR.exsqrt(RR(-1)) is None

    assert CC.is_square(CC(2)) is True
    assert CC.almosteq(CC.exsqrt(CC(2)), CC(1.4142135623730951), tolerance=1e-15)
    assert CC.is_square(CC(0)) is True
    assert CC.exsqrt(CC(0)) == CC(0)
    assert CC.is_square(CC(-1)) is True
    assert CC.exsqrt(CC(-1)) == CC(0, 1)
    assert CC.is_square(CC(0, 2)) is True
    assert CC.exsqrt(CC(0, 2)) == CC(1, 1)
    assert CC.is_square(CC(-3, -4)) is True
    assert CC.exsqrt(CC(-3, -4)) == CC(1, -2)

    F2 = FF(2)
    assert F2.is_square(F2(1)) is True
    assert F2.exsqrt(F2(1)) == F2(1)
    assert F2.is_square(F2(0)) is True
    assert F2.exsqrt(F2(0)) == F2(0)

    F7 = FF(7)
    assert F7.is_square(F7(2)) is True
    assert F7.exsqrt(F7(2)) == F7(3)
    assert F7.is_square(F7(3)) is False
    assert F7.exsqrt(F7(3)) is None
    assert F7.is_square(F7(0)) is True
    assert F7.exsqrt(F7(0)) == F7(0)
