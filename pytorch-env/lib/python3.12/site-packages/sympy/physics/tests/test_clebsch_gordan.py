from sympy.core.numbers import (I, pi, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.spherical_harmonics import Ynm
from sympy.matrices.dense import Matrix
from sympy.physics.wigner import (clebsch_gordan, wigner_9j, wigner_6j, gaunt,
        real_gaunt, racah, dot_rot_grad_Ynm, wigner_3j, wigner_d_small, wigner_d)
from sympy.testing.pytest import raises

# for test cases, refer : https://en.wikipedia.org/wiki/Table_of_Clebsch%E2%80%93Gordan_coefficients

def test_clebsch_gordan_docs():
    assert clebsch_gordan(Rational(3, 2), S.Half, 2, Rational(3, 2), S.Half, 2) == 1
    assert clebsch_gordan(Rational(3, 2), S.Half, 1, Rational(3, 2), Rational(-1, 2), 1) == sqrt(3)/2
    assert clebsch_gordan(Rational(3, 2), S.Half, 1, Rational(-1, 2), S.Half, 0) == -sqrt(2)/2


def test_clebsch_gordan():
    # Argument order: (j_1, j_2, j, m_1, m_2, m)

    h = S.One
    k = S.Half
    l = Rational(3, 2)
    i = Rational(-1, 2)
    n = Rational(7, 2)
    p = Rational(5, 2)
    assert clebsch_gordan(k, k, 1, k, k, 1) == 1
    assert clebsch_gordan(k, k, 1, k, k, 0) == 0
    assert clebsch_gordan(k, k, 1, i, i, -1) == 1
    assert clebsch_gordan(k, k, 1, k, i, 0) == sqrt(2)/2
    assert clebsch_gordan(k, k, 0, k, i, 0) == sqrt(2)/2
    assert clebsch_gordan(k, k, 1, i, k, 0) == sqrt(2)/2
    assert clebsch_gordan(k, k, 0, i, k, 0) == -sqrt(2)/2
    assert clebsch_gordan(h, k, l, 1, k, l) == 1
    assert clebsch_gordan(h, k, l, 1, i, k) == 1/sqrt(3)
    assert clebsch_gordan(h, k, k, 1, i, k) == sqrt(2)/sqrt(3)
    assert clebsch_gordan(h, k, k, 0, k, k) == -1/sqrt(3)
    assert clebsch_gordan(h, k, l, 0, k, k) == sqrt(2)/sqrt(3)
    assert clebsch_gordan(h, h, S(2), 1, 1, S(2)) == 1
    assert clebsch_gordan(h, h, S(2), 1, 0, 1) == 1/sqrt(2)
    assert clebsch_gordan(h, h, S(2), 0, 1, 1) == 1/sqrt(2)
    assert clebsch_gordan(h, h, 1, 1, 0, 1) == 1/sqrt(2)
    assert clebsch_gordan(h, h, 1, 0, 1, 1) == -1/sqrt(2)
    assert clebsch_gordan(l, l, S(3), l, l, S(3)) == 1
    assert clebsch_gordan(l, l, S(2), l, k, S(2)) == 1/sqrt(2)
    assert clebsch_gordan(l, l, S(3), l, k, S(2)) == 1/sqrt(2)
    assert clebsch_gordan(S(2), S(2), S(4), S(2), S(2), S(4)) == 1
    assert clebsch_gordan(S(2), S(2), S(3), S(2), 1, S(3)) == 1/sqrt(2)
    assert clebsch_gordan(S(2), S(2), S(3), 1, 1, S(2)) == 0
    assert clebsch_gordan(p, h, n, p, 1, n) == 1
    assert clebsch_gordan(p, h, p, p, 0, p) == sqrt(5)/sqrt(7)
    assert clebsch_gordan(p, h, l, k, 1, l) == 1/sqrt(15)


def test_wigner():
    def tn(a, b):
        return (a - b).n(64) < S('1e-64')
    assert tn(wigner_9j(1, 1, 1, 1, 1, 1, 1, 1, 0, prec=64), Rational(1, 18))
    assert wigner_9j(3, 3, 2, 3, 3, 2, 3, 3, 2) == 3221*sqrt(
        70)/(246960*sqrt(105)) - 365/(3528*sqrt(70)*sqrt(105))
    assert wigner_6j(5, 5, 5, 5, 5, 5) == Rational(1, 52)
    assert tn(wigner_6j(8, 8, 8, 8, 8, 8, prec=64), Rational(-12219, 965770))
    # regression test for #8747
    half = S.Half
    assert wigner_9j(0, 0, 0, 0, half, half, 0, half, half) == half
    assert (wigner_9j(3, 5, 4,
                      7 * half, 5 * half, 4,
                      9 * half, 9 * half, 0)
            == -sqrt(Rational(361, 205821000)))
    assert (wigner_9j(1, 4, 3,
                      5 * half, 4, 5 * half,
                      5 * half, 2, 7 * half)
            == -sqrt(Rational(3971, 373403520)))
    assert (wigner_9j(4, 9 * half, 5 * half,
                      2, 4, 4,
                      5, 7 * half, 7 * half)
            == -sqrt(Rational(3481, 5042614500)))


def test_gaunt():
    def tn(a, b):
        return (a - b).n(64) < S('1e-64')
    assert gaunt(1, 0, 1, 1, 0, -1) == -1/(2*sqrt(pi))
    assert isinstance(gaunt(1, 1, 0, -1, 1, 0).args[0], Rational)
    assert isinstance(gaunt(0, 1, 1, 0, -1, 1).args[0], Rational)

    assert tn(gaunt(
        10, 10, 12, 9, 3, -12, prec=64), (Rational(-98, 62031)) * sqrt(6279)/sqrt(pi))
    def gaunt_ref(l1, l2, l3, m1, m2, m3):
        return (
            sqrt((2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1) / (4 * pi)) *
            wigner_3j(l1, l2, l3, 0, 0, 0) *
            wigner_3j(l1, l2, l3, m1, m2, m3)
        )
    threshold = 1e-10
    l_max = 3
    l3_max = 24
    for l1 in range(l_max + 1):
        for l2 in range(l_max + 1):
            for l3 in range(l3_max + 1):
                for m1 in range(-l1, l1 + 1):
                    for m2 in range(-l2, l2 + 1):
                        for m3 in range(-l3, l3 + 1):
                            args = l1, l2, l3, m1, m2, m3
                            g  = gaunt(*args)
                            g0 = gaunt_ref(*args)
                            assert abs(g - g0) < threshold
                            if m1 + m2 + m3 != 0:
                                assert abs(g) < threshold
                            if (l1 + l2 + l3) % 2:
                                assert abs(g) < threshold
    assert gaunt(1, 1, 0, 0, 2, -2) is S.Zero


def test_realgaunt():
    # All non-zero values corresponding to l values from 0 to 2
    for l in range(3):
        for m in range(-l, l+1):
            assert real_gaunt(0, l, l, 0, m, m) == 1/(2*sqrt(pi))
    assert real_gaunt(1, 1, 2, 0, 0, 0) == sqrt(5)/(5*sqrt(pi))
    assert real_gaunt(1, 1, 2, 1, 1, 0) == -sqrt(5)/(10*sqrt(pi))
    assert real_gaunt(2, 2, 2, 0, 0, 0) == sqrt(5)/(7*sqrt(pi))
    assert real_gaunt(2, 2, 2, 0, 2, 2) == -sqrt(5)/(7*sqrt(pi))
    assert real_gaunt(2, 2, 2, -2, -2, 0) == -sqrt(5)/(7*sqrt(pi))
    assert real_gaunt(1, 1, 2, -1, 0, -1) == sqrt(15)/(10*sqrt(pi))
    assert real_gaunt(1, 1, 2, 0, 1, 1) == sqrt(15)/(10*sqrt(pi))
    assert real_gaunt(1, 1, 2, 1, 1, 2) == sqrt(15)/(10*sqrt(pi))
    assert real_gaunt(1, 1, 2, -1, 1, -2) == -sqrt(15)/(10*sqrt(pi))
    assert real_gaunt(1, 1, 2, -1, -1, 2) == -sqrt(15)/(10*sqrt(pi))
    assert real_gaunt(2, 2, 2, 0, 1, 1) == sqrt(5)/(14*sqrt(pi))
    assert real_gaunt(2, 2, 2, 1, 1, 2) == sqrt(15)/(14*sqrt(pi))
    assert real_gaunt(2, 2, 2, -1, -1, 2) == -sqrt(15)/(14*sqrt(pi))

    assert real_gaunt(-2, -2, -2, -2, -2, 0) is S.Zero  # m test
    assert real_gaunt(-2, 1, 0, 1, 1, 1) is S.Zero  # l test
    assert real_gaunt(-2, -1, -2, -1, -1, 0) is S.Zero  # m and l test
    assert real_gaunt(-2, -2, -2, -2, -2, -2) is S.Zero  # m and k test
    assert real_gaunt(-2, -1, -2, -1, -1, -1) is S.Zero  # m, l and k test

    x = symbols('x', integer=True)
    v = [0]*6
    for i in range(len(v)):
        v[i] = x  # non literal ints fail
        raises(ValueError, lambda: real_gaunt(*v))
        v[i] = 0


def test_racah():
    assert racah(3,3,3,3,3,3) == Rational(-1,14)
    assert racah(2,2,2,2,2,2) == Rational(-3,70)
    assert racah(7,8,7,1,7,7, prec=4).is_Float
    assert racah(5.5,7.5,9.5,6.5,8,9) == -719*sqrt(598)/1158924
    assert abs(racah(5.5,7.5,9.5,6.5,8,9, prec=4) - (-0.01517)) < S('1e-4')


def test_dot_rota_grad_SH():
    theta, phi = symbols("theta phi")
    assert dot_rot_grad_Ynm(1, 1, 1, 1, 1, 0) !=  \
        sqrt(30)*Ynm(2, 2, 1, 0)/(10*sqrt(pi))
    assert dot_rot_grad_Ynm(1, 1, 1, 1, 1, 0).doit() ==  \
        sqrt(30)*Ynm(2, 2, 1, 0)/(10*sqrt(pi))
    assert dot_rot_grad_Ynm(1, 5, 1, 1, 1, 2) !=  \
        0
    assert dot_rot_grad_Ynm(1, 5, 1, 1, 1, 2).doit() ==  \
        0
    assert dot_rot_grad_Ynm(3, 3, 3, 3, theta, phi).doit() ==  \
        15*sqrt(3003)*Ynm(6, 6, theta, phi)/(143*sqrt(pi))
    assert dot_rot_grad_Ynm(3, 3, 1, 1, theta, phi).doit() ==  \
        sqrt(3)*Ynm(4, 4, theta, phi)/sqrt(pi)
    assert dot_rot_grad_Ynm(3, 2, 2, 0, theta, phi).doit() ==  \
        3*sqrt(55)*Ynm(5, 2, theta, phi)/(11*sqrt(pi))
    assert dot_rot_grad_Ynm(3, 2, 3, 2, theta, phi).doit().expand() ==  \
        -sqrt(70)*Ynm(4, 4, theta, phi)/(11*sqrt(pi)) + \
        45*sqrt(182)*Ynm(6, 4, theta, phi)/(143*sqrt(pi))


def test_wigner_d():
    half = S(1)/2
    assert wigner_d_small(half, 0) == Matrix([[1, 0], [0, 1]])
    assert wigner_d_small(half, pi/2) == Matrix([[1, 1], [-1, 1]])/sqrt(2)
    assert wigner_d_small(half, pi) == Matrix([[0, 1], [-1, 0]])

    alpha, beta, gamma = symbols("alpha, beta, gamma", real=True)
    D = wigner_d(half, alpha, beta, gamma)
    assert D[0, 0] == exp(I*alpha/2)*exp(I*gamma/2)*cos(beta/2)
    assert D[0, 1] == exp(I*alpha/2)*exp(-I*gamma/2)*sin(beta/2)
    assert D[1, 0] == -exp(-I*alpha/2)*exp(I*gamma/2)*sin(beta/2)
    assert D[1, 1] == exp(-I*alpha/2)*exp(-I*gamma/2)*cos(beta/2)

    # Test Y_{n mi}(g*x)=\sum_{mj}D^n_{mi mj}*Y_{n mj}(x)
    theta, phi = symbols("theta phi", real=True)
    v = Matrix([Ynm(1, mj, theta, phi) for mj in range(1, -2, -1)])
    w = wigner_d(1, -pi/2, pi/2, -pi/2)@v.subs({theta: pi/4, phi: pi})
    w_ = v.subs({theta: pi/2, phi: pi/4})
    assert w.expand(func=True).as_real_imag() == w_.expand(func=True).as_real_imag()
