from sympy.concrete.summations import Sum
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.quantum.cg import Wigner3j, Wigner6j, Wigner9j, CG, cg_simp
from sympy.functions.special.tensor_functions import KroneckerDelta


def test_cg_simp_add():
    j, m1, m1p, m2, m2p = symbols('j m1 m1p m2 m2p')
    # Test Varshalovich 8.7.1 Eq 1
    a = CG(S.Half, S.Half, 0, 0, S.Half, S.Half)
    b = CG(S.Half, Rational(-1, 2), 0, 0, S.Half, Rational(-1, 2))
    c = CG(1, 1, 0, 0, 1, 1)
    d = CG(1, 0, 0, 0, 1, 0)
    e = CG(1, -1, 0, 0, 1, -1)
    assert cg_simp(a + b) == 2
    assert cg_simp(c + d + e) == 3
    assert cg_simp(a + b + c + d + e) == 5
    assert cg_simp(a + b + c) == 2 + c
    assert cg_simp(2*a + b) == 2 + a
    assert cg_simp(2*c + d + e) == 3 + c
    assert cg_simp(5*a + 5*b) == 10
    assert cg_simp(5*c + 5*d + 5*e) == 15
    assert cg_simp(-a - b) == -2
    assert cg_simp(-c - d - e) == -3
    assert cg_simp(-6*a - 6*b) == -12
    assert cg_simp(-4*c - 4*d - 4*e) == -12
    a = CG(S.Half, S.Half, j, 0, S.Half, S.Half)
    b = CG(S.Half, Rational(-1, 2), j, 0, S.Half, Rational(-1, 2))
    c = CG(1, 1, j, 0, 1, 1)
    d = CG(1, 0, j, 0, 1, 0)
    e = CG(1, -1, j, 0, 1, -1)
    assert cg_simp(a + b) == 2*KroneckerDelta(j, 0)
    assert cg_simp(c + d + e) == 3*KroneckerDelta(j, 0)
    assert cg_simp(a + b + c + d + e) == 5*KroneckerDelta(j, 0)
    assert cg_simp(a + b + c) == 2*KroneckerDelta(j, 0) + c
    assert cg_simp(2*a + b) == 2*KroneckerDelta(j, 0) + a
    assert cg_simp(2*c + d + e) == 3*KroneckerDelta(j, 0) + c
    assert cg_simp(5*a + 5*b) == 10*KroneckerDelta(j, 0)
    assert cg_simp(5*c + 5*d + 5*e) == 15*KroneckerDelta(j, 0)
    assert cg_simp(-a - b) == -2*KroneckerDelta(j, 0)
    assert cg_simp(-c - d - e) == -3*KroneckerDelta(j, 0)
    assert cg_simp(-6*a - 6*b) == -12*KroneckerDelta(j, 0)
    assert cg_simp(-4*c - 4*d - 4*e) == -12*KroneckerDelta(j, 0)
    # Test Varshalovich 8.7.1 Eq 2
    a = CG(S.Half, S.Half, S.Half, Rational(-1, 2), 0, 0)
    b = CG(S.Half, Rational(-1, 2), S.Half, S.Half, 0, 0)
    c = CG(1, 1, 1, -1, 0, 0)
    d = CG(1, 0, 1, 0, 0, 0)
    e = CG(1, -1, 1, 1, 0, 0)
    assert cg_simp(a - b) == sqrt(2)
    assert cg_simp(c - d + e) == sqrt(3)
    assert cg_simp(a - b + c - d + e) == sqrt(2) + sqrt(3)
    assert cg_simp(a - b + c) == sqrt(2) + c
    assert cg_simp(2*a - b) == sqrt(2) + a
    assert cg_simp(2*c - d + e) == sqrt(3) + c
    assert cg_simp(5*a - 5*b) == 5*sqrt(2)
    assert cg_simp(5*c - 5*d + 5*e) == 5*sqrt(3)
    assert cg_simp(-a + b) == -sqrt(2)
    assert cg_simp(-c + d - e) == -sqrt(3)
    assert cg_simp(-6*a + 6*b) == -6*sqrt(2)
    assert cg_simp(-4*c + 4*d - 4*e) == -4*sqrt(3)
    a = CG(S.Half, S.Half, S.Half, Rational(-1, 2), j, 0)
    b = CG(S.Half, Rational(-1, 2), S.Half, S.Half, j, 0)
    c = CG(1, 1, 1, -1, j, 0)
    d = CG(1, 0, 1, 0, j, 0)
    e = CG(1, -1, 1, 1, j, 0)
    assert cg_simp(a - b) == sqrt(2)*KroneckerDelta(j, 0)
    assert cg_simp(c - d + e) == sqrt(3)*KroneckerDelta(j, 0)
    assert cg_simp(a - b + c - d + e) == sqrt(
        2)*KroneckerDelta(j, 0) + sqrt(3)*KroneckerDelta(j, 0)
    assert cg_simp(a - b + c) == sqrt(2)*KroneckerDelta(j, 0) + c
    assert cg_simp(2*a - b) == sqrt(2)*KroneckerDelta(j, 0) + a
    assert cg_simp(2*c - d + e) == sqrt(3)*KroneckerDelta(j, 0) + c
    assert cg_simp(5*a - 5*b) == 5*sqrt(2)*KroneckerDelta(j, 0)
    assert cg_simp(5*c - 5*d + 5*e) == 5*sqrt(3)*KroneckerDelta(j, 0)
    assert cg_simp(-a + b) == -sqrt(2)*KroneckerDelta(j, 0)
    assert cg_simp(-c + d - e) == -sqrt(3)*KroneckerDelta(j, 0)
    assert cg_simp(-6*a + 6*b) == -6*sqrt(2)*KroneckerDelta(j, 0)
    assert cg_simp(-4*c + 4*d - 4*e) == -4*sqrt(3)*KroneckerDelta(j, 0)
    # Test Varshalovich 8.7.2 Eq 9
    # alpha=alphap,beta=betap case
    # numerical
    a = CG(S.Half, S.Half, S.Half, Rational(-1, 2), 1, 0)**2
    b = CG(S.Half, S.Half, S.Half, Rational(-1, 2), 0, 0)**2
    c = CG(1, 0, 1, 1, 1, 1)**2
    d = CG(1, 0, 1, 1, 2, 1)**2
    assert cg_simp(a + b) == 1
    assert cg_simp(c + d) == 1
    assert cg_simp(a + b + c + d) == 2
    assert cg_simp(4*a + 4*b) == 4
    assert cg_simp(4*c + 4*d) == 4
    assert cg_simp(5*a + 3*b) == 3 + 2*a
    assert cg_simp(5*c + 3*d) == 3 + 2*c
    assert cg_simp(-a - b) == -1
    assert cg_simp(-c - d) == -1
    # symbolic
    a = CG(S.Half, m1, S.Half, m2, 1, 1)**2
    b = CG(S.Half, m1, S.Half, m2, 1, 0)**2
    c = CG(S.Half, m1, S.Half, m2, 1, -1)**2
    d = CG(S.Half, m1, S.Half, m2, 0, 0)**2
    assert cg_simp(a + b + c + d) == 1
    assert cg_simp(4*a + 4*b + 4*c + 4*d) == 4
    assert cg_simp(3*a + 5*b + 3*c + 4*d) == 3 + 2*b + d
    assert cg_simp(-a - b - c - d) == -1
    a = CG(1, m1, 1, m2, 2, 2)**2
    b = CG(1, m1, 1, m2, 2, 1)**2
    c = CG(1, m1, 1, m2, 2, 0)**2
    d = CG(1, m1, 1, m2, 2, -1)**2
    e = CG(1, m1, 1, m2, 2, -2)**2
    f = CG(1, m1, 1, m2, 1, 1)**2
    g = CG(1, m1, 1, m2, 1, 0)**2
    h = CG(1, m1, 1, m2, 1, -1)**2
    i = CG(1, m1, 1, m2, 0, 0)**2
    assert cg_simp(a + b + c + d + e + f + g + h + i) == 1
    assert cg_simp(4*(a + b + c + d + e + f + g + h + i)) == 4
    assert cg_simp(a + b + 2*c + d + 4*e + f + g + h + i) == 1 + c + 3*e
    assert cg_simp(-a - b - c - d - e - f - g - h - i) == -1
    # alpha!=alphap or beta!=betap case
    # numerical
    a = CG(S.Half, S(
        1)/2, S.Half, Rational(-1, 2), 1, 0)*CG(S.Half, Rational(-1, 2), S.Half, S.Half, 1, 0)
    b = CG(S.Half, S(
        1)/2, S.Half, Rational(-1, 2), 0, 0)*CG(S.Half, Rational(-1, 2), S.Half, S.Half, 0, 0)
    c = CG(1, 1, 1, 0, 2, 1)*CG(1, 0, 1, 1, 2, 1)
    d = CG(1, 1, 1, 0, 1, 1)*CG(1, 0, 1, 1, 1, 1)
    assert cg_simp(a + b) == 0
    assert cg_simp(c + d) == 0
    # symbolic
    a = CG(S.Half, m1, S.Half, m2, 1, 1)*CG(S.Half, m1p, S.Half, m2p, 1, 1)
    b = CG(S.Half, m1, S.Half, m2, 1, 0)*CG(S.Half, m1p, S.Half, m2p, 1, 0)
    c = CG(S.Half, m1, S.Half, m2, 1, -1)*CG(S.Half, m1p, S.Half, m2p, 1, -1)
    d = CG(S.Half, m1, S.Half, m2, 0, 0)*CG(S.Half, m1p, S.Half, m2p, 0, 0)
    assert cg_simp(a + b + c + d) == KroneckerDelta(m1, m1p)*KroneckerDelta(m2, m2p)
    a = CG(1, m1, 1, m2, 2, 2)*CG(1, m1p, 1, m2p, 2, 2)
    b = CG(1, m1, 1, m2, 2, 1)*CG(1, m1p, 1, m2p, 2, 1)
    c = CG(1, m1, 1, m2, 2, 0)*CG(1, m1p, 1, m2p, 2, 0)
    d = CG(1, m1, 1, m2, 2, -1)*CG(1, m1p, 1, m2p, 2, -1)
    e = CG(1, m1, 1, m2, 2, -2)*CG(1, m1p, 1, m2p, 2, -2)
    f = CG(1, m1, 1, m2, 1, 1)*CG(1, m1p, 1, m2p, 1, 1)
    g = CG(1, m1, 1, m2, 1, 0)*CG(1, m1p, 1, m2p, 1, 0)
    h = CG(1, m1, 1, m2, 1, -1)*CG(1, m1p, 1, m2p, 1, -1)
    i = CG(1, m1, 1, m2, 0, 0)*CG(1, m1p, 1, m2p, 0, 0)
    assert cg_simp(
        a + b + c + d + e + f + g + h + i) == KroneckerDelta(m1, m1p)*KroneckerDelta(m2, m2p)


def test_cg_simp_sum():
    x, a, b, c, cp, alpha, beta, gamma, gammap = symbols(
        'x a b c cp alpha beta gamma gammap')
    # Varshalovich 8.7.1 Eq 1
    assert cg_simp(x * Sum(CG(a, alpha, b, 0, a, alpha), (alpha, -a, a)
                   )) == x*(2*a + 1)*KroneckerDelta(b, 0)
    assert cg_simp(x * Sum(CG(a, alpha, b, 0, a, alpha), (alpha, -a, a)) + CG(1, 0, 1, 0, 1, 0)) == x*(2*a + 1)*KroneckerDelta(b, 0) + CG(1, 0, 1, 0, 1, 0)
    assert cg_simp(2 * Sum(CG(1, alpha, 0, 0, 1, alpha), (alpha, -1, 1))) == 6
    # Varshalovich 8.7.1 Eq 2
    assert cg_simp(x*Sum((-1)**(a - alpha) * CG(a, alpha, a, -alpha, c,
                   0), (alpha, -a, a))) == x*sqrt(2*a + 1)*KroneckerDelta(c, 0)
    assert cg_simp(3*Sum((-1)**(2 - alpha) * CG(
        2, alpha, 2, -alpha, 0, 0), (alpha, -2, 2))) == 3*sqrt(5)
    # Varshalovich 8.7.2 Eq 4
    assert cg_simp(Sum(CG(a, alpha, b, beta, c, gamma)*CG(a, alpha, b, beta, cp, gammap), (alpha, -a, a), (beta, -b, b))) == KroneckerDelta(c, cp)*KroneckerDelta(gamma, gammap)
    assert cg_simp(Sum(CG(a, alpha, b, beta, c, gamma)*CG(a, alpha, b, beta, c, gammap), (alpha, -a, a), (beta, -b, b))) == KroneckerDelta(gamma, gammap)
    assert cg_simp(Sum(CG(a, alpha, b, beta, c, gamma)*CG(a, alpha, b, beta, cp, gamma), (alpha, -a, a), (beta, -b, b))) == KroneckerDelta(c, cp)
    assert cg_simp(Sum(CG(
        a, alpha, b, beta, c, gamma)**2, (alpha, -a, a), (beta, -b, b))) == 1
    assert cg_simp(Sum(CG(2, alpha, 1, beta, 2, gamma)*CG(2, alpha, 1, beta, 2, gammap), (alpha, -2, 2), (beta, -1, 1))) == KroneckerDelta(gamma, gammap)


def test_doit():
    assert Wigner3j(S.Half, Rational(-1, 2), S.Half, S.Half, 0, 0).doit() == -sqrt(2)/2
    assert Wigner3j(1/2,1/2,1/2,1/2,1/2,1/2).doit() == 0
    assert Wigner3j(9/2,9/2,9/2,9/2,9/2,9/2).doit() ==  0
    assert Wigner6j(1, 2, 3, 2, 1, 2).doit() == sqrt(21)/105
    assert Wigner6j(3, 1, 2, 2, 2, 1).doit() == sqrt(21) / 105
    assert Wigner9j(
        2, 1, 1, Rational(3, 2), S.Half, 1, S.Half, S.Half, 0).doit() == sqrt(2)/12
    assert CG(S.Half, S.Half, S.Half, Rational(-1, 2), 1, 0).doit() == sqrt(2)/2
    # J minus M is not integer
    assert Wigner3j(1, -1, S.Half, S.Half, 1, S.Half).doit() == 0
    assert CG(4, -1, S.Half, S.Half, 4, Rational(-1, 2)).doit() == 0
