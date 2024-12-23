from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.polys import QQ, ZZ
from sympy.polys.polytools import Poly
from sympy.polys.polyerrors import NotInvertible
from sympy.polys.agca.extensions import FiniteExtension
from sympy.polys.domainmatrix import DomainMatrix

from sympy.testing.pytest import raises

from sympy.abc import x, y, t


def test_FiniteExtension():
    # Gaussian integers
    A = FiniteExtension(Poly(x**2 + 1, x))
    assert A.rank == 2
    assert str(A) == 'ZZ[x]/(x**2 + 1)'
    i = A.generator
    assert i.parent() is A

    assert i*i == A(-1)
    raises(TypeError, lambda: i*())

    assert A.basis == (A.one, i)
    assert A(1) == A.one
    assert i**2 == A(-1)
    assert i**2 != -1  # no coercion
    assert (2 + i)*(1 - i) == 3 - i
    assert (1 + i)**8 == A(16)
    assert A(1).inverse() == A(1)
    raises(NotImplementedError, lambda: A(2).inverse())

    # Finite field of order 27
    F = FiniteExtension(Poly(x**3 - x + 1, x, modulus=3))
    assert F.rank == 3
    a = F.generator  # also generates the cyclic group F - {0}
    assert F.basis == (F(1), a, a**2)
    assert a**27 == a
    assert a**26 == F(1)
    assert a**13 == F(-1)
    assert a**9 == a + 1
    assert a**3 == a - 1
    assert a**6 == a**2 + a + 1
    assert F(x**2 + x).inverse() == 1 - a
    assert F(x + 2)**(-1) == F(x + 2).inverse()
    assert a**19 * a**(-19) == F(1)
    assert (a - 1) / (2*a**2 - 1) == a**2 + 1
    assert (a - 1) // (2*a**2 - 1) == a**2 + 1
    assert 2/(a**2 + 1) == a**2 - a + 1
    assert (a**2 + 1)/2 == -a**2 - 1
    raises(NotInvertible, lambda: F(0).inverse())

    # Function field of an elliptic curve
    K = FiniteExtension(Poly(t**2 - x**3 - x + 1, t, field=True))
    assert K.rank == 2
    assert str(K) == 'ZZ(x)[t]/(t**2 - x**3 - x + 1)'
    y = K.generator
    c = 1/(x**3 - x**2 + x - 1)
    assert ((y + x)*(y - x)).inverse() == K(c)
    assert (y + x)*(y - x)*c == K(1)  # explicit inverse of y + x


def test_FiniteExtension_eq_hash():
    # Test eq and hash
    p1 = Poly(x**2 - 2, x, domain=ZZ)
    p2 = Poly(x**2 - 2, x, domain=QQ)
    K1 = FiniteExtension(p1)
    K2 = FiniteExtension(p2)
    assert K1 == FiniteExtension(Poly(x**2 - 2))
    assert K2 != FiniteExtension(Poly(x**2 - 2))
    assert len({K1, K2, FiniteExtension(p1)}) == 2


def test_FiniteExtension_mod():
    # Test mod
    K = FiniteExtension(Poly(x**3 + 1, x, domain=QQ))
    xf = K(x)
    assert (xf**2 - 1) % 1 == K.zero
    assert 1 % (xf**2 - 1) == K.zero
    assert (xf**2 - 1) / (xf - 1) == xf + 1
    assert (xf**2 - 1) // (xf - 1) == xf + 1
    assert (xf**2 - 1) % (xf - 1) == K.zero
    raises(ZeroDivisionError, lambda: (xf**2 - 1) % 0)
    raises(TypeError, lambda: xf % [])
    raises(TypeError, lambda: [] % xf)

    # Test mod over ring
    K = FiniteExtension(Poly(x**3 + 1, x, domain=ZZ))
    xf = K(x)
    assert (xf**2 - 1) % 1 == K.zero
    raises(NotImplementedError, lambda: (xf**2 - 1) % (xf - 1))


def test_FiniteExtension_from_sympy():
    # Test to_sympy/from_sympy
    K = FiniteExtension(Poly(x**3 + 1, x, domain=ZZ))
    xf = K(x)
    assert K.from_sympy(x) == xf
    assert K.to_sympy(xf) == x


def test_FiniteExtension_set_domain():
    KZ = FiniteExtension(Poly(x**2 + 1, x, domain='ZZ'))
    KQ = FiniteExtension(Poly(x**2 + 1, x, domain='QQ'))
    assert KZ.set_domain(QQ) == KQ


def test_FiniteExtension_exquo():
    # Test exquo
    K = FiniteExtension(Poly(x**4 + 1))
    xf = K(x)
    assert K.exquo(xf**2 - 1, xf - 1) == xf + 1


def test_FiniteExtension_convert():
    # Test from_MonogenicFiniteExtension
    K1 = FiniteExtension(Poly(x**2 + 1))
    K2 = QQ[x]
    x1, x2 = K1(x), K2(x)
    assert K1.convert(x2) == x1
    assert K2.convert(x1) == x2

    K = FiniteExtension(Poly(x**2 - 1, domain=QQ))
    assert K.convert_from(QQ(1, 2), QQ) == K.one/2


def test_FiniteExtension_division_ring():
    # Test division in FiniteExtension over a ring
    KQ = FiniteExtension(Poly(x**2 - 1, x, domain=QQ))
    KZ = FiniteExtension(Poly(x**2 - 1, x, domain=ZZ))
    KQt = FiniteExtension(Poly(x**2 - 1, x, domain=QQ[t]))
    KQtf = FiniteExtension(Poly(x**2 - 1, x, domain=QQ.frac_field(t)))
    assert KQ.is_Field is True
    assert KZ.is_Field is False
    assert KQt.is_Field is False
    assert KQtf.is_Field is True
    for K in KQ, KZ, KQt, KQtf:
        xK = K.convert(x)
        assert xK / K.one == xK
        assert xK // K.one == xK
        assert xK % K.one == K.zero
        raises(ZeroDivisionError, lambda: xK / K.zero)
        raises(ZeroDivisionError, lambda: xK // K.zero)
        raises(ZeroDivisionError, lambda: xK % K.zero)
        if K.is_Field:
            assert xK / xK == K.one
            assert xK // xK == K.one
            assert xK % xK == K.zero
        else:
            raises(NotImplementedError, lambda: xK / xK)
            raises(NotImplementedError, lambda: xK // xK)
            raises(NotImplementedError, lambda: xK % xK)


def test_FiniteExtension_Poly():
    K = FiniteExtension(Poly(x**2 - 2))
    p = Poly(x, y, domain=K)
    assert p.domain == K
    assert p.as_expr() == x
    assert (p**2).as_expr() == 2

    K = FiniteExtension(Poly(x**2 - 2, x, domain=QQ))
    K2 = FiniteExtension(Poly(t**2 - 2, t, domain=K))
    assert str(K2) == 'QQ[x]/(x**2 - 2)[t]/(t**2 - 2)'

    eK = K2.convert(x + t)
    assert K2.to_sympy(eK) == x + t
    assert K2.to_sympy(eK ** 2) == 4 + 2*x*t
    p = Poly(x + t, y, domain=K2)
    assert p**2 == Poly(4 + 2*x*t, y, domain=K2)


def test_FiniteExtension_sincos_jacobian():
    # Use FiniteExtensino to compute the Jacobian of a matrix involving sin
    # and cos of different symbols.
    r, p, t = symbols('rho, phi, theta')
    elements = [
        [sin(p)*cos(t), r*cos(p)*cos(t), -r*sin(p)*sin(t)],
        [sin(p)*sin(t), r*cos(p)*sin(t),  r*sin(p)*cos(t)],
        [       cos(p),       -r*sin(p),                0],
    ]

    def make_extension(K):
        K = FiniteExtension(Poly(sin(p)**2+cos(p)**2-1, sin(p), domain=K[cos(p)]))
        K = FiniteExtension(Poly(sin(t)**2+cos(t)**2-1, sin(t), domain=K[cos(t)]))
        return K

    Ksc1 = make_extension(ZZ[r])
    Ksc2 = make_extension(ZZ)[r]

    for K in [Ksc1, Ksc2]:
        elements_K = [[K.convert(e) for e in row] for row in elements]
        J = DomainMatrix(elements_K, (3, 3), K)
        det = J.charpoly()[-1] * (-K.one)**3
        assert det == K.convert(r**2*sin(p))
