from sympy.concrete.summations import Sum
from sympy.core.function import expand
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import Matrix
from sympy.abc import alpha, beta, gamma, j, m
from sympy.physics.quantum import hbar, represent, Commutator, InnerProduct
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.cg import CG
from sympy.physics.quantum.spin import (
    Jx, Jy, Jz, Jplus, Jminus, J2,
    JxBra, JyBra, JzBra,
    JxKet, JyKet, JzKet,
    JxKetCoupled, JyKetCoupled, JzKetCoupled,
    couple, uncouple,
    Rotation, WignerD
)

from sympy.testing.pytest import raises, slow

j1, j2, j3, j4, m1, m2, m3, m4 = symbols('j1:5 m1:5')
j12, j13, j24, j34, j123, j134, mi, mi1, mp = symbols(
    'j12 j13 j24 j34 j123 j134 mi mi1 mp')


def test_represent_spin_operators():
    assert represent(Jx) == hbar*Matrix([[0, 1], [1, 0]])/2
    assert represent(
        Jx, j=1) == hbar*sqrt(2)*Matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])/2
    assert represent(Jy) == hbar*I*Matrix([[0, -1], [1, 0]])/2
    assert represent(Jy, j=1) == hbar*I*sqrt(2)*Matrix([[0, -1, 0], [1,
                     0, -1], [0, 1, 0]])/2
    assert represent(Jz) == hbar*Matrix([[1, 0], [0, -1]])/2
    assert represent(
        Jz, j=1) == hbar*Matrix([[1, 0, 0], [0, 0, 0], [0, 0, -1]])


def test_represent_spin_states():
    # Jx basis
    assert represent(JxKet(S.Half, S.Half), basis=Jx) == Matrix([1, 0])
    assert represent(JxKet(S.Half, Rational(-1, 2)), basis=Jx) == Matrix([0, 1])
    assert represent(JxKet(1, 1), basis=Jx) == Matrix([1, 0, 0])
    assert represent(JxKet(1, 0), basis=Jx) == Matrix([0, 1, 0])
    assert represent(JxKet(1, -1), basis=Jx) == Matrix([0, 0, 1])
    assert represent(
        JyKet(S.Half, S.Half), basis=Jx) == Matrix([exp(-I*pi/4), 0])
    assert represent(
        JyKet(S.Half, Rational(-1, 2)), basis=Jx) == Matrix([0, exp(I*pi/4)])
    assert represent(JyKet(1, 1), basis=Jx) == Matrix([-I, 0, 0])
    assert represent(JyKet(1, 0), basis=Jx) == Matrix([0, 1, 0])
    assert represent(JyKet(1, -1), basis=Jx) == Matrix([0, 0, I])
    assert represent(
        JzKet(S.Half, S.Half), basis=Jx) == sqrt(2)*Matrix([-1, 1])/2
    assert represent(
        JzKet(S.Half, Rational(-1, 2)), basis=Jx) == sqrt(2)*Matrix([-1, -1])/2
    assert represent(JzKet(1, 1), basis=Jx) == Matrix([1, -sqrt(2), 1])/2
    assert represent(JzKet(1, 0), basis=Jx) == sqrt(2)*Matrix([1, 0, -1])/2
    assert represent(JzKet(1, -1), basis=Jx) == Matrix([1, sqrt(2), 1])/2
    # Jy basis
    assert represent(
        JxKet(S.Half, S.Half), basis=Jy) == Matrix([exp(I*pi*Rational(-3, 4)), 0])
    assert represent(
        JxKet(S.Half, Rational(-1, 2)), basis=Jy) == Matrix([0, exp(I*pi*Rational(3, 4))])
    assert represent(JxKet(1, 1), basis=Jy) == Matrix([I, 0, 0])
    assert represent(JxKet(1, 0), basis=Jy) == Matrix([0, 1, 0])
    assert represent(JxKet(1, -1), basis=Jy) == Matrix([0, 0, -I])
    assert represent(JyKet(S.Half, S.Half), basis=Jy) == Matrix([1, 0])
    assert represent(JyKet(S.Half, Rational(-1, 2)), basis=Jy) == Matrix([0, 1])
    assert represent(JyKet(1, 1), basis=Jy) == Matrix([1, 0, 0])
    assert represent(JyKet(1, 0), basis=Jy) == Matrix([0, 1, 0])
    assert represent(JyKet(1, -1), basis=Jy) == Matrix([0, 0, 1])
    assert represent(
        JzKet(S.Half, S.Half), basis=Jy) == sqrt(2)*Matrix([-1, I])/2
    assert represent(
        JzKet(S.Half, Rational(-1, 2)), basis=Jy) == sqrt(2)*Matrix([I, -1])/2
    assert represent(JzKet(1, 1), basis=Jy) == Matrix([1, -I*sqrt(2), -1])/2
    assert represent(
        JzKet(1, 0), basis=Jy) == Matrix([-sqrt(2)*I, 0, -sqrt(2)*I])/2
    assert represent(JzKet(1, -1), basis=Jy) == Matrix([-1, -sqrt(2)*I, 1])/2
    # Jz basis
    assert represent(
        JxKet(S.Half, S.Half), basis=Jz) == sqrt(2)*Matrix([1, 1])/2
    assert represent(
        JxKet(S.Half, Rational(-1, 2)), basis=Jz) == sqrt(2)*Matrix([-1, 1])/2
    assert represent(JxKet(1, 1), basis=Jz) == Matrix([1, sqrt(2), 1])/2
    assert represent(JxKet(1, 0), basis=Jz) == sqrt(2)*Matrix([-1, 0, 1])/2
    assert represent(JxKet(1, -1), basis=Jz) == Matrix([1, -sqrt(2), 1])/2
    assert represent(
        JyKet(S.Half, S.Half), basis=Jz) == sqrt(2)*Matrix([-1, -I])/2
    assert represent(
        JyKet(S.Half, Rational(-1, 2)), basis=Jz) == sqrt(2)*Matrix([-I, -1])/2
    assert represent(JyKet(1, 1), basis=Jz) == Matrix([1, sqrt(2)*I, -1])/2
    assert represent(JyKet(1, 0), basis=Jz) == sqrt(2)*Matrix([I, 0, I])/2
    assert represent(JyKet(1, -1), basis=Jz) == Matrix([-1, sqrt(2)*I, 1])/2
    assert represent(JzKet(S.Half, S.Half), basis=Jz) == Matrix([1, 0])
    assert represent(JzKet(S.Half, Rational(-1, 2)), basis=Jz) == Matrix([0, 1])
    assert represent(JzKet(1, 1), basis=Jz) == Matrix([1, 0, 0])
    assert represent(JzKet(1, 0), basis=Jz) == Matrix([0, 1, 0])
    assert represent(JzKet(1, -1), basis=Jz) == Matrix([0, 0, 1])


def test_represent_uncoupled_states():
    # Jx basis
    assert represent(TensorProduct(JxKet(S.Half, S.Half), JxKet(S.Half, S.Half)), basis=Jx) == \
        Matrix([1, 0, 0, 0])
    assert represent(TensorProduct(JxKet(S.Half, S.Half), JxKet(S.Half, Rational(-1, 2))), basis=Jx) == \
        Matrix([0, 1, 0, 0])
    assert represent(TensorProduct(JxKet(S.Half, Rational(-1, 2)), JxKet(S.Half, S.Half)), basis=Jx) == \
        Matrix([0, 0, 1, 0])
    assert represent(TensorProduct(JxKet(S.Half, Rational(-1, 2)), JxKet(S.Half, Rational(-1, 2))), basis=Jx) == \
        Matrix([0, 0, 0, 1])
    assert represent(TensorProduct(JyKet(S.Half, S.Half), JyKet(S.Half, S.Half)), basis=Jx) == \
        Matrix([-I, 0, 0, 0])
    assert represent(TensorProduct(JyKet(S.Half, S.Half), JyKet(S.Half, Rational(-1, 2))), basis=Jx) == \
        Matrix([0, 1, 0, 0])
    assert represent(TensorProduct(JyKet(S.Half, Rational(-1, 2)), JyKet(S.Half, S.Half)), basis=Jx) == \
        Matrix([0, 0, 1, 0])
    assert represent(TensorProduct(JyKet(S.Half, Rational(-1, 2)), JyKet(S.Half, Rational(-1, 2))), basis=Jx) == \
        Matrix([0, 0, 0, I])
    assert represent(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)), basis=Jx) == \
        Matrix([S.Half, Rational(-1, 2), Rational(-1, 2), S.Half])
    assert represent(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), basis=Jx) == \
        Matrix([S.Half, S.Half, Rational(-1, 2), Rational(-1, 2)])
    assert represent(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)), basis=Jx) == \
        Matrix([S.Half, Rational(-1, 2), S.Half, Rational(-1, 2)])
    assert represent(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), basis=Jx) == \
        Matrix([S.Half, S.Half, S.Half, S.Half])
    # Jy basis
    assert represent(TensorProduct(JxKet(S.Half, S.Half), JxKet(S.Half, S.Half)), basis=Jy) == \
        Matrix([I, 0, 0, 0])
    assert represent(TensorProduct(JxKet(S.Half, S.Half), JxKet(S.Half, Rational(-1, 2))), basis=Jy) == \
        Matrix([0, 1, 0, 0])
    assert represent(TensorProduct(JxKet(S.Half, Rational(-1, 2)), JxKet(S.Half, S.Half)), basis=Jy) == \
        Matrix([0, 0, 1, 0])
    assert represent(TensorProduct(JxKet(S.Half, Rational(-1, 2)), JxKet(S.Half, Rational(-1, 2))), basis=Jy) == \
        Matrix([0, 0, 0, -I])
    assert represent(TensorProduct(JyKet(S.Half, S.Half), JyKet(S.Half, S.Half)), basis=Jy) == \
        Matrix([1, 0, 0, 0])
    assert represent(TensorProduct(JyKet(S.Half, S.Half), JyKet(S.Half, Rational(-1, 2))), basis=Jy) == \
        Matrix([0, 1, 0, 0])
    assert represent(TensorProduct(JyKet(S.Half, Rational(-1, 2)), JyKet(S.Half, S.Half)), basis=Jy) == \
        Matrix([0, 0, 1, 0])
    assert represent(TensorProduct(JyKet(S.Half, Rational(-1, 2)), JyKet(S.Half, Rational(-1, 2))), basis=Jy) == \
        Matrix([0, 0, 0, 1])
    assert represent(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)), basis=Jy) == \
        Matrix([S.Half, -I/2, -I/2, Rational(-1, 2)])
    assert represent(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), basis=Jy) == \
        Matrix([-I/2, S.Half, Rational(-1, 2), -I/2])
    assert represent(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)), basis=Jy) == \
        Matrix([-I/2, Rational(-1, 2), S.Half, -I/2])
    assert represent(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), basis=Jy) == \
        Matrix([Rational(-1, 2), -I/2, -I/2, S.Half])
    # Jz basis
    assert represent(TensorProduct(JxKet(S.Half, S.Half), JxKet(S.Half, S.Half)), basis=Jz) == \
        Matrix([S.Half, S.Half, S.Half, S.Half])
    assert represent(TensorProduct(JxKet(S.Half, S.Half), JxKet(S.Half, Rational(-1, 2))), basis=Jz) == \
        Matrix([Rational(-1, 2), S.Half, Rational(-1, 2), S.Half])
    assert represent(TensorProduct(JxKet(S.Half, Rational(-1, 2)), JxKet(S.Half, S.Half)), basis=Jz) == \
        Matrix([Rational(-1, 2), Rational(-1, 2), S.Half, S.Half])
    assert represent(TensorProduct(JxKet(S.Half, Rational(-1, 2)), JxKet(S.Half, Rational(-1, 2))), basis=Jz) == \
        Matrix([S.Half, Rational(-1, 2), Rational(-1, 2), S.Half])
    assert represent(TensorProduct(JyKet(S.Half, S.Half), JyKet(S.Half, S.Half)), basis=Jz) == \
        Matrix([S.Half, I/2, I/2, Rational(-1, 2)])
    assert represent(TensorProduct(JyKet(S.Half, S.Half), JyKet(S.Half, Rational(-1, 2))), basis=Jz) == \
        Matrix([I/2, S.Half, Rational(-1, 2), I/2])
    assert represent(TensorProduct(JyKet(S.Half, Rational(-1, 2)), JyKet(S.Half, S.Half)), basis=Jz) == \
        Matrix([I/2, Rational(-1, 2), S.Half, I/2])
    assert represent(TensorProduct(JyKet(S.Half, Rational(-1, 2)), JyKet(S.Half, Rational(-1, 2))), basis=Jz) == \
        Matrix([Rational(-1, 2), I/2, I/2, S.Half])
    assert represent(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)), basis=Jz) == \
        Matrix([1, 0, 0, 0])
    assert represent(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), basis=Jz) == \
        Matrix([0, 1, 0, 0])
    assert represent(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)), basis=Jz) == \
        Matrix([0, 0, 1, 0])
    assert represent(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), basis=Jz) == \
        Matrix([0, 0, 0, 1])


def test_represent_coupled_states():
    # Jx basis
    assert represent(JxKetCoupled(0, 0, (S.Half, S.Half)), basis=Jx) == \
        Matrix([1, 0, 0, 0])
    assert represent(JxKetCoupled(1, 1, (S.Half, S.Half)), basis=Jx) == \
        Matrix([0, 1, 0, 0])
    assert represent(JxKetCoupled(1, 0, (S.Half, S.Half)), basis=Jx) == \
        Matrix([0, 0, 1, 0])
    assert represent(JxKetCoupled(1, -1, (S.Half, S.Half)), basis=Jx) == \
        Matrix([0, 0, 0, 1])
    assert represent(JyKetCoupled(0, 0, (S.Half, S.Half)), basis=Jx) == \
        Matrix([1, 0, 0, 0])
    assert represent(JyKetCoupled(1, 1, (S.Half, S.Half)), basis=Jx) == \
        Matrix([0, -I, 0, 0])
    assert represent(JyKetCoupled(1, 0, (S.Half, S.Half)), basis=Jx) == \
        Matrix([0, 0, 1, 0])
    assert represent(JyKetCoupled(1, -1, (S.Half, S.Half)), basis=Jx) == \
        Matrix([0, 0, 0, I])
    assert represent(JzKetCoupled(0, 0, (S.Half, S.Half)), basis=Jx) == \
        Matrix([1, 0, 0, 0])
    assert represent(JzKetCoupled(1, 1, (S.Half, S.Half)), basis=Jx) == \
        Matrix([0, S.Half, -sqrt(2)/2, S.Half])
    assert represent(JzKetCoupled(1, 0, (S.Half, S.Half)), basis=Jx) == \
        Matrix([0, sqrt(2)/2, 0, -sqrt(2)/2])
    assert represent(JzKetCoupled(1, -1, (S.Half, S.Half)), basis=Jx) == \
        Matrix([0, S.Half, sqrt(2)/2, S.Half])
    # Jy basis
    assert represent(JxKetCoupled(0, 0, (S.Half, S.Half)), basis=Jy) == \
        Matrix([1, 0, 0, 0])
    assert represent(JxKetCoupled(1, 1, (S.Half, S.Half)), basis=Jy) == \
        Matrix([0, I, 0, 0])
    assert represent(JxKetCoupled(1, 0, (S.Half, S.Half)), basis=Jy) == \
        Matrix([0, 0, 1, 0])
    assert represent(JxKetCoupled(1, -1, (S.Half, S.Half)), basis=Jy) == \
        Matrix([0, 0, 0, -I])
    assert represent(JyKetCoupled(0, 0, (S.Half, S.Half)), basis=Jy) == \
        Matrix([1, 0, 0, 0])
    assert represent(JyKetCoupled(1, 1, (S.Half, S.Half)), basis=Jy) == \
        Matrix([0, 1, 0, 0])
    assert represent(JyKetCoupled(1, 0, (S.Half, S.Half)), basis=Jy) == \
        Matrix([0, 0, 1, 0])
    assert represent(JyKetCoupled(1, -1, (S.Half, S.Half)), basis=Jy) == \
        Matrix([0, 0, 0, 1])
    assert represent(JzKetCoupled(0, 0, (S.Half, S.Half)), basis=Jy) == \
        Matrix([1, 0, 0, 0])
    assert represent(JzKetCoupled(1, 1, (S.Half, S.Half)), basis=Jy) == \
        Matrix([0, S.Half, -I*sqrt(2)/2, Rational(-1, 2)])
    assert represent(JzKetCoupled(1, 0, (S.Half, S.Half)), basis=Jy) == \
        Matrix([0, -I*sqrt(2)/2, 0, -I*sqrt(2)/2])
    assert represent(JzKetCoupled(1, -1, (S.Half, S.Half)), basis=Jy) == \
        Matrix([0, Rational(-1, 2), -I*sqrt(2)/2, S.Half])
    # Jz basis
    assert represent(JxKetCoupled(0, 0, (S.Half, S.Half)), basis=Jz) == \
        Matrix([1, 0, 0, 0])
    assert represent(JxKetCoupled(1, 1, (S.Half, S.Half)), basis=Jz) == \
        Matrix([0, S.Half, sqrt(2)/2, S.Half])
    assert represent(JxKetCoupled(1, 0, (S.Half, S.Half)), basis=Jz) == \
        Matrix([0, -sqrt(2)/2, 0, sqrt(2)/2])
    assert represent(JxKetCoupled(1, -1, (S.Half, S.Half)), basis=Jz) == \
        Matrix([0, S.Half, -sqrt(2)/2, S.Half])
    assert represent(JyKetCoupled(0, 0, (S.Half, S.Half)), basis=Jz) == \
        Matrix([1, 0, 0, 0])
    assert represent(JyKetCoupled(1, 1, (S.Half, S.Half)), basis=Jz) == \
        Matrix([0, S.Half, I*sqrt(2)/2, Rational(-1, 2)])
    assert represent(JyKetCoupled(1, 0, (S.Half, S.Half)), basis=Jz) == \
        Matrix([0, I*sqrt(2)/2, 0, I*sqrt(2)/2])
    assert represent(JyKetCoupled(1, -1, (S.Half, S.Half)), basis=Jz) == \
        Matrix([0, Rational(-1, 2), I*sqrt(2)/2, S.Half])
    assert represent(JzKetCoupled(0, 0, (S.Half, S.Half)), basis=Jz) == \
        Matrix([1, 0, 0, 0])
    assert represent(JzKetCoupled(1, 1, (S.Half, S.Half)), basis=Jz) == \
        Matrix([0, 1, 0, 0])
    assert represent(JzKetCoupled(1, 0, (S.Half, S.Half)), basis=Jz) == \
        Matrix([0, 0, 1, 0])
    assert represent(JzKetCoupled(1, -1, (S.Half, S.Half)), basis=Jz) == \
        Matrix([0, 0, 0, 1])


def test_represent_rotation():
    assert represent(Rotation(0, pi/2, 0)) == \
        Matrix(
            [[WignerD(
                S(
                    1)/2, S(
                        1)/2, S(
                            1)/2, 0, pi/2, 0), WignerD(
                                S.Half, S.Half, Rational(-1, 2), 0, pi/2, 0)],
                [WignerD(S.Half, Rational(-1, 2), S.Half, 0, pi/2, 0), WignerD(S.Half, Rational(-1, 2), Rational(-1, 2), 0, pi/2, 0)]])
    assert represent(Rotation(0, pi/2, 0), doit=True) == \
        Matrix([[sqrt(2)/2, -sqrt(2)/2],
                [sqrt(2)/2, sqrt(2)/2]])


def test_rewrite_same():
    # Rewrite to same basis
    assert JxBra(1, 1).rewrite('Jx') == JxBra(1, 1)
    assert JxBra(j, m).rewrite('Jx') == JxBra(j, m)
    assert JxKet(1, 1).rewrite('Jx') == JxKet(1, 1)
    assert JxKet(j, m).rewrite('Jx') == JxKet(j, m)


def test_rewrite_Bra():
    # Numerical
    assert JxBra(1, 1).rewrite('Jy') == -I*JyBra(1, 1)
    assert JxBra(1, 0).rewrite('Jy') == JyBra(1, 0)
    assert JxBra(1, -1).rewrite('Jy') == I*JyBra(1, -1)
    assert JxBra(1, 1).rewrite(
        'Jz') == JzBra(1, 1)/2 + JzBra(1, 0)/sqrt(2) + JzBra(1, -1)/2
    assert JxBra(
        1, 0).rewrite('Jz') == -sqrt(2)*JzBra(1, 1)/2 + sqrt(2)*JzBra(1, -1)/2
    assert JxBra(1, -1).rewrite(
        'Jz') == JzBra(1, 1)/2 - JzBra(1, 0)/sqrt(2) + JzBra(1, -1)/2
    assert JyBra(1, 1).rewrite('Jx') == I*JxBra(1, 1)
    assert JyBra(1, 0).rewrite('Jx') == JxBra(1, 0)
    assert JyBra(1, -1).rewrite('Jx') == -I*JxBra(1, -1)
    assert JyBra(1, 1).rewrite(
        'Jz') == JzBra(1, 1)/2 - sqrt(2)*I*JzBra(1, 0)/2 - JzBra(1, -1)/2
    assert JyBra(1, 0).rewrite(
        'Jz') == -sqrt(2)*I*JzBra(1, 1)/2 - sqrt(2)*I*JzBra(1, -1)/2
    assert JyBra(1, -1).rewrite(
        'Jz') == -JzBra(1, 1)/2 - sqrt(2)*I*JzBra(1, 0)/2 + JzBra(1, -1)/2
    assert JzBra(1, 1).rewrite(
        'Jx') == JxBra(1, 1)/2 - sqrt(2)*JxBra(1, 0)/2 + JxBra(1, -1)/2
    assert JzBra(
        1, 0).rewrite('Jx') == sqrt(2)*JxBra(1, 1)/2 - sqrt(2)*JxBra(1, -1)/2
    assert JzBra(1, -1).rewrite(
        'Jx') == JxBra(1, 1)/2 + sqrt(2)*JxBra(1, 0)/2 + JxBra(1, -1)/2
    assert JzBra(1, 1).rewrite(
        'Jy') == JyBra(1, 1)/2 + sqrt(2)*I*JyBra(1, 0)/2 - JyBra(1, -1)/2
    assert JzBra(1, 0).rewrite(
        'Jy') == sqrt(2)*I*JyBra(1, 1)/2 + sqrt(2)*I*JyBra(1, -1)/2
    assert JzBra(1, -1).rewrite(
        'Jy') == -JyBra(1, 1)/2 + sqrt(2)*I*JyBra(1, 0)/2 + JyBra(1, -1)/2
    # Symbolic
    assert JxBra(j, m).rewrite('Jy') == Sum(
        WignerD(j, mi, m, pi*Rational(3, 2), 0, 0) * JyBra(j, mi), (mi, -j, j))
    assert JxBra(j, m).rewrite('Jz') == Sum(
        WignerD(j, mi, m, 0, pi/2, 0) * JzBra(j, mi), (mi, -j, j))
    assert JyBra(j, m).rewrite('Jx') == Sum(
        WignerD(j, mi, m, 0, 0, pi/2) * JxBra(j, mi), (mi, -j, j))
    assert JyBra(j, m).rewrite('Jz') == Sum(
        WignerD(j, mi, m, pi*Rational(3, 2), -pi/2, pi/2) * JzBra(j, mi), (mi, -j, j))
    assert JzBra(j, m).rewrite('Jx') == Sum(
        WignerD(j, mi, m, 0, pi*Rational(3, 2), 0) * JxBra(j, mi), (mi, -j, j))
    assert JzBra(j, m).rewrite('Jy') == Sum(
        WignerD(j, mi, m, pi*Rational(3, 2), pi/2, pi/2) * JyBra(j, mi), (mi, -j, j))


def test_rewrite_Ket():
    # Numerical
    assert JxKet(1, 1).rewrite('Jy') == I*JyKet(1, 1)
    assert JxKet(1, 0).rewrite('Jy') == JyKet(1, 0)
    assert JxKet(1, -1).rewrite('Jy') == -I*JyKet(1, -1)
    assert JxKet(1, 1).rewrite(
        'Jz') == JzKet(1, 1)/2 + JzKet(1, 0)/sqrt(2) + JzKet(1, -1)/2
    assert JxKet(
        1, 0).rewrite('Jz') == -sqrt(2)*JzKet(1, 1)/2 + sqrt(2)*JzKet(1, -1)/2
    assert JxKet(1, -1).rewrite(
        'Jz') == JzKet(1, 1)/2 - JzKet(1, 0)/sqrt(2) + JzKet(1, -1)/2
    assert JyKet(1, 1).rewrite('Jx') == -I*JxKet(1, 1)
    assert JyKet(1, 0).rewrite('Jx') == JxKet(1, 0)
    assert JyKet(1, -1).rewrite('Jx') == I*JxKet(1, -1)
    assert JyKet(1, 1).rewrite(
        'Jz') == JzKet(1, 1)/2 + sqrt(2)*I*JzKet(1, 0)/2 - JzKet(1, -1)/2
    assert JyKet(1, 0).rewrite(
        'Jz') == sqrt(2)*I*JzKet(1, 1)/2 + sqrt(2)*I*JzKet(1, -1)/2
    assert JyKet(1, -1).rewrite(
        'Jz') == -JzKet(1, 1)/2 + sqrt(2)*I*JzKet(1, 0)/2 + JzKet(1, -1)/2
    assert JzKet(1, 1).rewrite(
        'Jx') == JxKet(1, 1)/2 - sqrt(2)*JxKet(1, 0)/2 + JxKet(1, -1)/2
    assert JzKet(
        1, 0).rewrite('Jx') == sqrt(2)*JxKet(1, 1)/2 - sqrt(2)*JxKet(1, -1)/2
    assert JzKet(1, -1).rewrite(
        'Jx') == JxKet(1, 1)/2 + sqrt(2)*JxKet(1, 0)/2 + JxKet(1, -1)/2
    assert JzKet(1, 1).rewrite(
        'Jy') == JyKet(1, 1)/2 - sqrt(2)*I*JyKet(1, 0)/2 - JyKet(1, -1)/2
    assert JzKet(1, 0).rewrite(
        'Jy') == -sqrt(2)*I*JyKet(1, 1)/2 - sqrt(2)*I*JyKet(1, -1)/2
    assert JzKet(1, -1).rewrite(
        'Jy') == -JyKet(1, 1)/2 - sqrt(2)*I*JyKet(1, 0)/2 + JyKet(1, -1)/2
    # Symbolic
    assert JxKet(j, m).rewrite('Jy') == Sum(
        WignerD(j, mi, m, pi*Rational(3, 2), 0, 0) * JyKet(j, mi), (mi, -j, j))
    assert JxKet(j, m).rewrite('Jz') == Sum(
        WignerD(j, mi, m, 0, pi/2, 0) * JzKet(j, mi), (mi, -j, j))
    assert JyKet(j, m).rewrite('Jx') == Sum(
        WignerD(j, mi, m, 0, 0, pi/2) * JxKet(j, mi), (mi, -j, j))
    assert JyKet(j, m).rewrite('Jz') == Sum(
        WignerD(j, mi, m, pi*Rational(3, 2), -pi/2, pi/2) * JzKet(j, mi), (mi, -j, j))
    assert JzKet(j, m).rewrite('Jx') == Sum(
        WignerD(j, mi, m, 0, pi*Rational(3, 2), 0) * JxKet(j, mi), (mi, -j, j))
    assert JzKet(j, m).rewrite('Jy') == Sum(
        WignerD(j, mi, m, pi*Rational(3, 2), pi/2, pi/2) * JyKet(j, mi), (mi, -j, j))


def test_rewrite_uncoupled_state():
    # Numerical
    assert TensorProduct(JyKet(1, 1), JxKet(
        1, 1)).rewrite('Jx') == -I*TensorProduct(JxKet(1, 1), JxKet(1, 1))
    assert TensorProduct(JyKet(1, 0), JxKet(
        1, 1)).rewrite('Jx') == TensorProduct(JxKet(1, 0), JxKet(1, 1))
    assert TensorProduct(JyKet(1, -1), JxKet(
        1, 1)).rewrite('Jx') == I*TensorProduct(JxKet(1, -1), JxKet(1, 1))
    assert TensorProduct(JzKet(1, 1), JxKet(1, 1)).rewrite('Jx') == \
        TensorProduct(JxKet(1, -1), JxKet(1, 1))/2 - sqrt(2)*TensorProduct(JxKet(
            1, 0), JxKet(1, 1))/2 + TensorProduct(JxKet(1, 1), JxKet(1, 1))/2
    assert TensorProduct(JzKet(1, 0), JxKet(1, 1)).rewrite('Jx') == \
        -sqrt(2)*TensorProduct(JxKet(1, -1), JxKet(1, 1))/2 + sqrt(
            2)*TensorProduct(JxKet(1, 1), JxKet(1, 1))/2
    assert TensorProduct(JzKet(1, -1), JxKet(1, 1)).rewrite('Jx') == \
        TensorProduct(JxKet(1, -1), JxKet(1, 1))/2 + sqrt(2)*TensorProduct(JxKet(1, 0), JxKet(1, 1))/2 + TensorProduct(JxKet(1, 1), JxKet(1, 1))/2
    assert TensorProduct(JxKet(1, 1), JyKet(
        1, 1)).rewrite('Jy') == I*TensorProduct(JyKet(1, 1), JyKet(1, 1))
    assert TensorProduct(JxKet(1, 0), JyKet(
        1, 1)).rewrite('Jy') == TensorProduct(JyKet(1, 0), JyKet(1, 1))
    assert TensorProduct(JxKet(1, -1), JyKet(
        1, 1)).rewrite('Jy') == -I*TensorProduct(JyKet(1, -1), JyKet(1, 1))
    assert TensorProduct(JzKet(1, 1), JyKet(1, 1)).rewrite('Jy') == \
        -TensorProduct(JyKet(1, -1), JyKet(1, 1))/2 - sqrt(2)*I*TensorProduct(JyKet(1, 0), JyKet(1, 1))/2 + TensorProduct(JyKet(1, 1), JyKet(1, 1))/2
    assert TensorProduct(JzKet(1, 0), JyKet(1, 1)).rewrite('Jy') == \
        -sqrt(2)*I*TensorProduct(JyKet(1, -1), JyKet(
            1, 1))/2 - sqrt(2)*I*TensorProduct(JyKet(1, 1), JyKet(1, 1))/2
    assert TensorProduct(JzKet(1, -1), JyKet(1, 1)).rewrite('Jy') == \
        TensorProduct(JyKet(1, -1), JyKet(1, 1))/2 - sqrt(2)*I*TensorProduct(JyKet(1, 0), JyKet(1, 1))/2 - TensorProduct(JyKet(1, 1), JyKet(1, 1))/2
    assert TensorProduct(JxKet(1, 1), JzKet(1, 1)).rewrite('Jz') == \
        TensorProduct(JzKet(1, -1), JzKet(1, 1))/2 + sqrt(2)*TensorProduct(JzKet(1, 0), JzKet(1, 1))/2 + TensorProduct(JzKet(1, 1), JzKet(1, 1))/2
    assert TensorProduct(JxKet(1, 0), JzKet(1, 1)).rewrite('Jz') == \
        sqrt(2)*TensorProduct(JzKet(1, -1), JzKet(
            1, 1))/2 - sqrt(2)*TensorProduct(JzKet(1, 1), JzKet(1, 1))/2
    assert TensorProduct(JxKet(1, -1), JzKet(1, 1)).rewrite('Jz') == \
        TensorProduct(JzKet(1, -1), JzKet(1, 1))/2 - sqrt(2)*TensorProduct(JzKet(1, 0), JzKet(1, 1))/2 + TensorProduct(JzKet(1, 1), JzKet(1, 1))/2
    assert TensorProduct(JyKet(1, 1), JzKet(1, 1)).rewrite('Jz') == \
        -TensorProduct(JzKet(1, -1), JzKet(1, 1))/2 + sqrt(2)*I*TensorProduct(JzKet(1, 0), JzKet(1, 1))/2 + TensorProduct(JzKet(1, 1), JzKet(1, 1))/2
    assert TensorProduct(JyKet(1, 0), JzKet(1, 1)).rewrite('Jz') == \
        sqrt(2)*I*TensorProduct(JzKet(1, -1), JzKet(
            1, 1))/2 + sqrt(2)*I*TensorProduct(JzKet(1, 1), JzKet(1, 1))/2
    assert TensorProduct(JyKet(1, -1), JzKet(1, 1)).rewrite('Jz') == \
        TensorProduct(JzKet(1, -1), JzKet(1, 1))/2 + sqrt(2)*I*TensorProduct(JzKet(1, 0), JzKet(1, 1))/2 - TensorProduct(JzKet(1, 1), JzKet(1, 1))/2
    # Symbolic
    assert TensorProduct(JyKet(j1, m1), JxKet(j2, m2)).rewrite('Jy') == \
        TensorProduct(JyKet(j1, m1), Sum(
            WignerD(j2, mi, m2, pi*Rational(3, 2), 0, 0) * JyKet(j2, mi), (mi, -j2, j2)))
    assert TensorProduct(JzKet(j1, m1), JxKet(j2, m2)).rewrite('Jz') == \
        TensorProduct(JzKet(j1, m1), Sum(
            WignerD(j2, mi, m2, 0, pi/2, 0) * JzKet(j2, mi), (mi, -j2, j2)))
    assert TensorProduct(JxKet(j1, m1), JyKet(j2, m2)).rewrite('Jx') == \
        TensorProduct(JxKet(j1, m1), Sum(
            WignerD(j2, mi, m2, 0, 0, pi/2) * JxKet(j2, mi), (mi, -j2, j2)))
    assert TensorProduct(JzKet(j1, m1), JyKet(j2, m2)).rewrite('Jz') == \
        TensorProduct(JzKet(j1, m1), Sum(WignerD(
            j2, mi, m2, pi*Rational(3, 2), -pi/2, pi/2) * JzKet(j2, mi), (mi, -j2, j2)))
    assert TensorProduct(JxKet(j1, m1), JzKet(j2, m2)).rewrite('Jx') == \
        TensorProduct(JxKet(j1, m1), Sum(
            WignerD(j2, mi, m2, 0, pi*Rational(3, 2), 0) * JxKet(j2, mi), (mi, -j2, j2)))
    assert TensorProduct(JyKet(j1, m1), JzKet(j2, m2)).rewrite('Jy') == \
        TensorProduct(JyKet(j1, m1), Sum(WignerD(
            j2, mi, m2, pi*Rational(3, 2), pi/2, pi/2) * JyKet(j2, mi), (mi, -j2, j2)))


def test_rewrite_coupled_state():
    # Numerical
    assert JyKetCoupled(0, 0, (S.Half, S.Half)).rewrite('Jx') == \
        JxKetCoupled(0, 0, (S.Half, S.Half))
    assert JyKetCoupled(1, 1, (S.Half, S.Half)).rewrite('Jx') == \
        -I*JxKetCoupled(1, 1, (S.Half, S.Half))
    assert JyKetCoupled(1, 0, (S.Half, S.Half)).rewrite('Jx') == \
        JxKetCoupled(1, 0, (S.Half, S.Half))
    assert JyKetCoupled(1, -1, (S.Half, S.Half)).rewrite('Jx') == \
        I*JxKetCoupled(1, -1, (S.Half, S.Half))
    assert JzKetCoupled(0, 0, (S.Half, S.Half)).rewrite('Jx') == \
        JxKetCoupled(0, 0, (S.Half, S.Half))
    assert JzKetCoupled(1, 1, (S.Half, S.Half)).rewrite('Jx') == \
        JxKetCoupled(1, 1, (S.Half, S.Half))/2 - sqrt(2)*JxKetCoupled(1, 0, (
            S.Half, S.Half))/2 + JxKetCoupled(1, -1, (S.Half, S.Half))/2
    assert JzKetCoupled(1, 0, (S.Half, S.Half)).rewrite('Jx') == \
        sqrt(2)*JxKetCoupled(1, 1, (S(
            1)/2, S.Half))/2 - sqrt(2)*JxKetCoupled(1, -1, (S.Half, S.Half))/2
    assert JzKetCoupled(1, -1, (S.Half, S.Half)).rewrite('Jx') == \
        JxKetCoupled(1, 1, (S.Half, S.Half))/2 + sqrt(2)*JxKetCoupled(1, 0, (
            S.Half, S.Half))/2 + JxKetCoupled(1, -1, (S.Half, S.Half))/2
    assert JxKetCoupled(0, 0, (S.Half, S.Half)).rewrite('Jy') == \
        JyKetCoupled(0, 0, (S.Half, S.Half))
    assert JxKetCoupled(1, 1, (S.Half, S.Half)).rewrite('Jy') == \
        I*JyKetCoupled(1, 1, (S.Half, S.Half))
    assert JxKetCoupled(1, 0, (S.Half, S.Half)).rewrite('Jy') == \
        JyKetCoupled(1, 0, (S.Half, S.Half))
    assert JxKetCoupled(1, -1, (S.Half, S.Half)).rewrite('Jy') == \
        -I*JyKetCoupled(1, -1, (S.Half, S.Half))
    assert JzKetCoupled(0, 0, (S.Half, S.Half)).rewrite('Jy') == \
        JyKetCoupled(0, 0, (S.Half, S.Half))
    assert JzKetCoupled(1, 1, (S.Half, S.Half)).rewrite('Jy') == \
        JyKetCoupled(1, 1, (S.Half, S.Half))/2 - I*sqrt(2)*JyKetCoupled(1, 0, (
            S.Half, S.Half))/2 - JyKetCoupled(1, -1, (S.Half, S.Half))/2
    assert JzKetCoupled(1, 0, (S.Half, S.Half)).rewrite('Jy') == \
        -I*sqrt(2)*JyKetCoupled(1, 1, (S.Half, S.Half))/2 - I*sqrt(
            2)*JyKetCoupled(1, -1, (S.Half, S.Half))/2
    assert JzKetCoupled(1, -1, (S.Half, S.Half)).rewrite('Jy') == \
        -JyKetCoupled(1, 1, (S.Half, S.Half))/2 - I*sqrt(2)*JyKetCoupled(1, 0, (S.Half, S.Half))/2 + JyKetCoupled(1, -1, (S.Half, S.Half))/2
    assert JxKetCoupled(0, 0, (S.Half, S.Half)).rewrite('Jz') == \
        JzKetCoupled(0, 0, (S.Half, S.Half))
    assert JxKetCoupled(1, 1, (S.Half, S.Half)).rewrite('Jz') == \
        JzKetCoupled(1, 1, (S.Half, S.Half))/2 + sqrt(2)*JzKetCoupled(1, 0, (
            S.Half, S.Half))/2 + JzKetCoupled(1, -1, (S.Half, S.Half))/2
    assert JxKetCoupled(1, 0, (S.Half, S.Half)).rewrite('Jz') == \
        -sqrt(2)*JzKetCoupled(1, 1, (S(
            1)/2, S.Half))/2 + sqrt(2)*JzKetCoupled(1, -1, (S.Half, S.Half))/2
    assert JxKetCoupled(1, -1, (S.Half, S.Half)).rewrite('Jz') == \
        JzKetCoupled(1, 1, (S.Half, S.Half))/2 - sqrt(2)*JzKetCoupled(1, 0, (
            S.Half, S.Half))/2 + JzKetCoupled(1, -1, (S.Half, S.Half))/2
    assert JyKetCoupled(0, 0, (S.Half, S.Half)).rewrite('Jz') == \
        JzKetCoupled(0, 0, (S.Half, S.Half))
    assert JyKetCoupled(1, 1, (S.Half, S.Half)).rewrite('Jz') == \
        JzKetCoupled(1, 1, (S.Half, S.Half))/2 + I*sqrt(2)*JzKetCoupled(1, 0, (
            S.Half, S.Half))/2 - JzKetCoupled(1, -1, (S.Half, S.Half))/2
    assert JyKetCoupled(1, 0, (S.Half, S.Half)).rewrite('Jz') == \
        I*sqrt(2)*JzKetCoupled(1, 1, (S.Half, S.Half))/2 + I*sqrt(
            2)*JzKetCoupled(1, -1, (S.Half, S.Half))/2
    assert JyKetCoupled(1, -1, (S.Half, S.Half)).rewrite('Jz') == \
        -JzKetCoupled(1, 1, (S.Half, S.Half))/2 + I*sqrt(2)*JzKetCoupled(1, 0, (S.Half, S.Half))/2 + JzKetCoupled(1, -1, (S.Half, S.Half))/2
    # Symbolic
    assert JyKetCoupled(j, m, (j1, j2)).rewrite('Jx') == \
        Sum(WignerD(j, mi, m, 0, 0, pi/2) * JxKetCoupled(j, mi, (
            j1, j2)), (mi, -j, j))
    assert JzKetCoupled(j, m, (j1, j2)).rewrite('Jx') == \
        Sum(WignerD(j, mi, m, 0, pi*Rational(3, 2), 0) * JxKetCoupled(j, mi, (
            j1, j2)), (mi, -j, j))
    assert JxKetCoupled(j, m, (j1, j2)).rewrite('Jy') == \
        Sum(WignerD(j, mi, m, pi*Rational(3, 2), 0, 0) * JyKetCoupled(j, mi, (
            j1, j2)), (mi, -j, j))
    assert JzKetCoupled(j, m, (j1, j2)).rewrite('Jy') == \
        Sum(WignerD(j, mi, m, pi*Rational(3, 2), pi/2, pi/2) * JyKetCoupled(j,
            mi, (j1, j2)), (mi, -j, j))
    assert JxKetCoupled(j, m, (j1, j2)).rewrite('Jz') == \
        Sum(WignerD(j, mi, m, 0, pi/2, 0) * JzKetCoupled(j, mi, (
            j1, j2)), (mi, -j, j))
    assert JyKetCoupled(j, m, (j1, j2)).rewrite('Jz') == \
        Sum(WignerD(j, mi, m, pi*Rational(3, 2), -pi/2, pi/2) * JzKetCoupled(
            j, mi, (j1, j2)), (mi, -j, j))


def test_innerproducts_of_rewritten_states():
    # Numerical
    assert qapply(JxBra(1, 1)*JxKet(1, 1).rewrite('Jy')).doit() == 1
    assert qapply(JxBra(1, 0)*JxKet(1, 0).rewrite('Jy')).doit() == 1
    assert qapply(JxBra(1, -1)*JxKet(1, -1).rewrite('Jy')).doit() == 1
    assert qapply(JxBra(1, 1)*JxKet(1, 1).rewrite('Jz')).doit() == 1
    assert qapply(JxBra(1, 0)*JxKet(1, 0).rewrite('Jz')).doit() == 1
    assert qapply(JxBra(1, -1)*JxKet(1, -1).rewrite('Jz')).doit() == 1
    assert qapply(JyBra(1, 1)*JyKet(1, 1).rewrite('Jx')).doit() == 1
    assert qapply(JyBra(1, 0)*JyKet(1, 0).rewrite('Jx')).doit() == 1
    assert qapply(JyBra(1, -1)*JyKet(1, -1).rewrite('Jx')).doit() == 1
    assert qapply(JyBra(1, 1)*JyKet(1, 1).rewrite('Jz')).doit() == 1
    assert qapply(JyBra(1, 0)*JyKet(1, 0).rewrite('Jz')).doit() == 1
    assert qapply(JyBra(1, -1)*JyKet(1, -1).rewrite('Jz')).doit() == 1
    assert qapply(JyBra(1, 1)*JyKet(1, 1).rewrite('Jz')).doit() == 1
    assert qapply(JyBra(1, 0)*JyKet(1, 0).rewrite('Jz')).doit() == 1
    assert qapply(JyBra(1, -1)*JyKet(1, -1).rewrite('Jz')).doit() == 1
    assert qapply(JzBra(1, 1)*JzKet(1, 1).rewrite('Jy')).doit() == 1
    assert qapply(JzBra(1, 0)*JzKet(1, 0).rewrite('Jy')).doit() == 1
    assert qapply(JzBra(1, -1)*JzKet(1, -1).rewrite('Jy')).doit() == 1
    assert qapply(JxBra(1, 1)*JxKet(1, 0).rewrite('Jy')).doit() == 0
    assert qapply(JxBra(1, 1)*JxKet(1, -1).rewrite('Jy')) == 0
    assert qapply(JxBra(1, 1)*JxKet(1, 0).rewrite('Jz')).doit() == 0
    assert qapply(JxBra(1, 1)*JxKet(1, -1).rewrite('Jz')) == 0
    assert qapply(JyBra(1, 1)*JyKet(1, 0).rewrite('Jx')).doit() == 0
    assert qapply(JyBra(1, 1)*JyKet(1, -1).rewrite('Jx')) == 0
    assert qapply(JyBra(1, 1)*JyKet(1, 0).rewrite('Jz')).doit() == 0
    assert qapply(JyBra(1, 1)*JyKet(1, -1).rewrite('Jz')) == 0
    assert qapply(JzBra(1, 1)*JzKet(1, 0).rewrite('Jx')).doit() == 0
    assert qapply(JzBra(1, 1)*JzKet(1, -1).rewrite('Jx')) == 0
    assert qapply(JzBra(1, 1)*JzKet(1, 0).rewrite('Jy')).doit() == 0
    assert qapply(JzBra(1, 1)*JzKet(1, -1).rewrite('Jy')) == 0
    assert qapply(JxBra(1, 0)*JxKet(1, 1).rewrite('Jy')) == 0
    assert qapply(JxBra(1, 0)*JxKet(1, -1).rewrite('Jy')) == 0
    assert qapply(JxBra(1, 0)*JxKet(1, 1).rewrite('Jz')) == 0
    assert qapply(JxBra(1, 0)*JxKet(1, -1).rewrite('Jz')) == 0
    assert qapply(JyBra(1, 0)*JyKet(1, 1).rewrite('Jx')) == 0
    assert qapply(JyBra(1, 0)*JyKet(1, -1).rewrite('Jx')) == 0
    assert qapply(JyBra(1, 0)*JyKet(1, 1).rewrite('Jz')) == 0
    assert qapply(JyBra(1, 0)*JyKet(1, -1).rewrite('Jz')) == 0
    assert qapply(JzBra(1, 0)*JzKet(1, 1).rewrite('Jx')) == 0
    assert qapply(JzBra(1, 0)*JzKet(1, -1).rewrite('Jx')) == 0
    assert qapply(JzBra(1, 0)*JzKet(1, 1).rewrite('Jy')) == 0
    assert qapply(JzBra(1, 0)*JzKet(1, -1).rewrite('Jy')) == 0
    assert qapply(JxBra(1, -1)*JxKet(1, 1).rewrite('Jy')) == 0
    assert qapply(JxBra(1, -1)*JxKet(1, 0).rewrite('Jy')).doit() == 0
    assert qapply(JxBra(1, -1)*JxKet(1, 1).rewrite('Jz')) == 0
    assert qapply(JxBra(1, -1)*JxKet(1, 0).rewrite('Jz')).doit() == 0
    assert qapply(JyBra(1, -1)*JyKet(1, 1).rewrite('Jx')) == 0
    assert qapply(JyBra(1, -1)*JyKet(1, 0).rewrite('Jx')).doit() == 0
    assert qapply(JyBra(1, -1)*JyKet(1, 1).rewrite('Jz')) == 0
    assert qapply(JyBra(1, -1)*JyKet(1, 0).rewrite('Jz')).doit() == 0
    assert qapply(JzBra(1, -1)*JzKet(1, 1).rewrite('Jx')) == 0
    assert qapply(JzBra(1, -1)*JzKet(1, 0).rewrite('Jx')).doit() == 0
    assert qapply(JzBra(1, -1)*JzKet(1, 1).rewrite('Jy')) == 0
    assert qapply(JzBra(1, -1)*JzKet(1, 0).rewrite('Jy')).doit() == 0


def test_uncouple_2_coupled_states():
    # j1=1/2, j2=1/2
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple(
            TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple(
            TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple(
            TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple(
            TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) )))
    # j1=1/2, j2=1
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1)) == \
        expand(uncouple(
            couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0)) == \
        expand(uncouple(
            couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1)) == \
        expand(uncouple(
            couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1)) == \
        expand(uncouple(
            couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0)) == \
        expand(uncouple(
            couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1)) == \
        expand(uncouple(
            couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1)) )))
    # j1=1, j2=1
    assert TensorProduct(JzKet(1, 1), JzKet(1, 1)) == \
        expand(uncouple(couple( TensorProduct(JzKet(1, 1), JzKet(1, 1)) )))
    assert TensorProduct(JzKet(1, 1), JzKet(1, 0)) == \
        expand(uncouple(couple( TensorProduct(JzKet(1, 1), JzKet(1, 0)) )))
    assert TensorProduct(JzKet(1, 1), JzKet(1, -1)) == \
        expand(uncouple(couple( TensorProduct(JzKet(1, 1), JzKet(1, -1)) )))
    assert TensorProduct(JzKet(1, 0), JzKet(1, 1)) == \
        expand(uncouple(couple( TensorProduct(JzKet(1, 0), JzKet(1, 1)) )))
    assert TensorProduct(JzKet(1, 0), JzKet(1, 0)) == \
        expand(uncouple(couple( TensorProduct(JzKet(1, 0), JzKet(1, 0)) )))
    assert TensorProduct(JzKet(1, 0), JzKet(1, -1)) == \
        expand(uncouple(couple( TensorProduct(JzKet(1, 0), JzKet(1, -1)) )))
    assert TensorProduct(JzKet(1, -1), JzKet(1, 1)) == \
        expand(uncouple(couple( TensorProduct(JzKet(1, -1), JzKet(1, 1)) )))
    assert TensorProduct(JzKet(1, -1), JzKet(1, 0)) == \
        expand(uncouple(couple( TensorProduct(JzKet(1, -1), JzKet(1, 0)) )))
    assert TensorProduct(JzKet(1, -1), JzKet(1, -1)) == \
        expand(uncouple(couple( TensorProduct(JzKet(1, -1), JzKet(1, -1)) )))


def test_uncouple_3_coupled_states():
    # Default coupling
    # j1=1/2, j2=1/2, j3=1/2
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(
            S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S(
            1)/2, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S(
            1)/2, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S(
            1)/2, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S(
            1)/2, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S(
            1)/2, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S(
            1)/2, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.NegativeOne/
               2), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) )))
    # j1=1/2, j2=1, j3=1/2
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(
            JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, S.Half)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(
            JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(
            JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, S.Half)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(
            JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(
            JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, S.Half)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(
            JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(
            JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, S.Half)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(
            JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(
            JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, S.Half)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(
            JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(
            JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, S.Half)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(
            JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) )))
    # Coupling j1+j3=j13, j13+j2=j
    # j1=1/2, j2=1/2, j3=1/2
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(
            S.Half, S.Half), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(
            S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(
            S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(
            S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(
            S.Half, S.Half), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(
            S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(
            S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(
            S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)) )))
    # j1=1/2, j2=1, j3=1/2
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S(
            1)/2), JzKet(1, 1), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S(
            1)/2), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S(
            1)/2), JzKet(1, 0), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S(
            1)/2), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S(
            1)/2), JzKet(1, -1), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S(
            1)/2), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S(
            -1)/2), JzKet(1, 1), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S(
            -1)/2), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S(
            -1)/2), JzKet(1, 0), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S(
            -1)/2), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S(
            -1)/2), JzKet(1, -1), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.NegativeOne/
               2), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)) )))


@slow
def test_uncouple_4_coupled_states():
    # j1=1/2, j2=1/2, j3=1/2, j4=1/2
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(
            S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S(
            1)/2, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S(
            1)/2, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S(
            1)/2, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S(
            1)/2, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S(
            1)/2, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S(
            1)/2, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(
            S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S(
            1)/2, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S(
            1)/2, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S(
            1)/2, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S(
            1)/2, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S(
            1)/2, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S(
            1)/2, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) )))
    # j1=1/2, j2=1/2, j3=1, j4=1/2
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half),
               JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, S.Half)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half),
               JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half),
               JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, S.Half)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half),
               JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half),
               JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, S.Half)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(
            S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half),
               JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, S.Half)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(
            S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half),
               JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, S.Half)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(
            S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(
            S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, S.Half)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(
            S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)),
               JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, S.Half)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)),
               JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)),
               JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, S.Half)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)),
               JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)),
               JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, S.Half)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(
            S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)),
               JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, S.Half)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(
            S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)),
               JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, S.Half)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(
            S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(
            S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, S.Half)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(
            S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) )))
    # Couple j1+j3=j13, j2+j4=j24, j13+j24=j
    # j1=1/2, j2=1/2, j3=1/2, j4=1/2
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    # j1=1/2, j2=1/2, j3=1, j4=1/2
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))


def test_uncouple_2_coupled_states_numerical():
    # j1=1/2, j2=1/2
    assert uncouple(JzKetCoupled(0, 0, (S.Half, S.Half))) == \
        sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)))/2 - \
        sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half))/2
    assert uncouple(JzKetCoupled(1, 1, (S.Half, S.Half))) == \
        TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half))
    assert uncouple(JzKetCoupled(1, 0, (S.Half, S.Half))) == \
        sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)))/2 + \
        sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half))/2
    assert uncouple(JzKetCoupled(1, -1, (S.Half, S.Half))) == \
        TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)))
    # j1=1, j2=1/2
    assert uncouple(JzKetCoupled(S.Half, S.Half, (1, S.Half))) == \
        -sqrt(3)*TensorProduct(JzKet(1, 0), JzKet(S.Half, S.Half))/3 + \
        sqrt(6)*TensorProduct(JzKet(1, 1), JzKet(S.Half, Rational(-1, 2)))/3
    assert uncouple(JzKetCoupled(S.Half, Rational(-1, 2), (1, S.Half))) == \
        sqrt(3)*TensorProduct(JzKet(1, 0), JzKet(S.Half, Rational(-1, 2)))/3 - \
        sqrt(6)*TensorProduct(JzKet(1, -1), JzKet(S.Half, S.Half))/3
    assert uncouple(JzKetCoupled(Rational(3, 2), Rational(3, 2), (1, S.Half))) == \
        TensorProduct(JzKet(1, 1), JzKet(S.Half, S.Half))
    assert uncouple(JzKetCoupled(Rational(3, 2), S.Half, (1, S.Half))) == \
        sqrt(3)*TensorProduct(JzKet(1, 1), JzKet(S.Half, Rational(-1, 2)))/3 + \
        sqrt(6)*TensorProduct(JzKet(1, 0), JzKet(S.Half, S.Half))/3
    assert uncouple(JzKetCoupled(Rational(3, 2), Rational(-1, 2), (1, S.Half))) == \
        sqrt(6)*TensorProduct(JzKet(1, 0), JzKet(S.Half, Rational(-1, 2)))/3 + \
        sqrt(3)*TensorProduct(JzKet(1, -1), JzKet(S.Half, S.Half))/3
    assert uncouple(JzKetCoupled(Rational(3, 2), Rational(-3, 2), (1, S.Half))) == \
        TensorProduct(JzKet(1, -1), JzKet(S.Half, Rational(-1, 2)))
    # j1=1, j2=1
    assert uncouple(JzKetCoupled(0, 0, (1, 1))) == \
        sqrt(3)*TensorProduct(JzKet(1, 1), JzKet(1, -1))/3 - \
        sqrt(3)*TensorProduct(JzKet(1, 0), JzKet(1, 0))/3 + \
        sqrt(3)*TensorProduct(JzKet(1, -1), JzKet(1, 1))/3
    assert uncouple(JzKetCoupled(1, 1, (1, 1))) == \
        sqrt(2)*TensorProduct(JzKet(1, 1), JzKet(1, 0))/2 - \
        sqrt(2)*TensorProduct(JzKet(1, 0), JzKet(1, 1))/2
    assert uncouple(JzKetCoupled(1, 0, (1, 1))) == \
        sqrt(2)*TensorProduct(JzKet(1, 1), JzKet(1, -1))/2 - \
        sqrt(2)*TensorProduct(JzKet(1, -1), JzKet(1, 1))/2
    assert uncouple(JzKetCoupled(1, -1, (1, 1))) == \
        sqrt(2)*TensorProduct(JzKet(1, 0), JzKet(1, -1))/2 - \
        sqrt(2)*TensorProduct(JzKet(1, -1), JzKet(1, 0))/2
    assert uncouple(JzKetCoupled(2, 2, (1, 1))) == \
        TensorProduct(JzKet(1, 1), JzKet(1, 1))
    assert uncouple(JzKetCoupled(2, 1, (1, 1))) == \
        sqrt(2)*TensorProduct(JzKet(1, 1), JzKet(1, 0))/2 + \
        sqrt(2)*TensorProduct(JzKet(1, 0), JzKet(1, 1))/2
    assert uncouple(JzKetCoupled(2, 0, (1, 1))) == \
        sqrt(6)*TensorProduct(JzKet(1, 1), JzKet(1, -1))/6 + \
        sqrt(6)*TensorProduct(JzKet(1, 0), JzKet(1, 0))/3 + \
        sqrt(6)*TensorProduct(JzKet(1, -1), JzKet(1, 1))/6
    assert uncouple(JzKetCoupled(2, -1, (1, 1))) == \
        sqrt(2)*TensorProduct(JzKet(1, 0), JzKet(1, -1))/2 + \
        sqrt(2)*TensorProduct(JzKet(1, -1), JzKet(1, 0))/2
    assert uncouple(JzKetCoupled(2, -2, (1, 1))) == \
        TensorProduct(JzKet(1, -1), JzKet(1, -1))


def test_uncouple_3_coupled_states_numerical():
    # Default coupling
    # j1=1/2, j2=1/2, j3=1/2
    assert uncouple(JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half))) == \
        TensorProduct(JzKet(
            S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half))
    assert uncouple(JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half))) == \
        sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half))/3 + \
        sqrt(3)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half))/3 + \
        sqrt(3)*TensorProduct(JzKet(
            S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)))/3
    assert uncouple(JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half))) == \
        sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half))/3 + \
        sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)))/3 + \
        sqrt(3)*TensorProduct(JzKet(
            S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)))/3
    assert uncouple(JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half))) == \
        TensorProduct(JzKet(
            S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)))
    # j1=1/2, j2=1/2, j3=1
    assert uncouple(JzKetCoupled(2, 2, (S.Half, S.Half, 1))) == \
        TensorProduct(
            JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1))
    assert uncouple(JzKetCoupled(2, 1, (S.Half, S.Half, 1))) == \
        TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1))/2 + \
        TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1))/2 + \
        sqrt(2)*TensorProduct(
            JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0))/2
    assert uncouple(JzKetCoupled(2, 0, (S.Half, S.Half, 1))) == \
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1))/6 + \
        sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0))/3 + \
        sqrt(3)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0))/3 + \
        sqrt(6)*TensorProduct(
            JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1))/6
    assert uncouple(JzKetCoupled(2, -1, (S.Half, S.Half, 1))) == \
        sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0))/2 + \
        TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1))/2 + \
        TensorProduct(
            JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1))/2
    assert uncouple(JzKetCoupled(2, -2, (S.Half, S.Half, 1))) == \
        TensorProduct(
            JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1))
    assert uncouple(JzKetCoupled(1, 1, (S.Half, S.Half, 1))) == \
        -TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1))/2 - \
        TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1))/2 + \
        sqrt(2)*TensorProduct(
            JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0))/2
    assert uncouple(JzKetCoupled(1, 0, (S.Half, S.Half, 1))) == \
        -sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1))/2 + \
        sqrt(2)*TensorProduct(
            JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1))/2
    assert uncouple(JzKetCoupled(1, -1, (S.Half, S.Half, 1))) == \
        -sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0))/2 + \
        TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1))/2 + \
        TensorProduct(
            JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1))/2
    # j1=1/2, j2=1, j3=1
    assert uncouple(JzKetCoupled(Rational(5, 2), Rational(5, 2), (S.Half, 1, 1))) == \
        TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 1))
    assert uncouple(JzKetCoupled(Rational(5, 2), Rational(3, 2), (S.Half, 1, 1))) == \
        sqrt(5)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 1))/5 + \
        sqrt(10)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 1))/5 + \
        sqrt(10)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1),
             JzKet(1, 0))/5
    assert uncouple(JzKetCoupled(Rational(5, 2), S.Half, (S.Half, 1, 1))) == \
        sqrt(5)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/5 + \
        sqrt(5)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/5 + \
        sqrt(10)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))/10 + \
        sqrt(10)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 0))/5 + \
        sqrt(10)*TensorProduct(
            JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, -1))/10
    assert uncouple(JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, 1, 1))) == \
        sqrt(10)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/10 + \
        sqrt(10)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 0))/5 + \
        sqrt(10)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/10 + \
        sqrt(5)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/5 + \
        sqrt(5)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0),
             JzKet(1, -1))/5
    assert uncouple(JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, 1, 1))) == \
        sqrt(10)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 0))/5 + \
        sqrt(10)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, -1))/5 + \
        sqrt(5)*TensorProduct(
            JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, -1))/5
    assert uncouple(JzKetCoupled(Rational(5, 2), Rational(-5, 2), (S.Half, 1, 1))) == \
        TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, -1))
    assert uncouple(JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, 1, 1))) == \
        -sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 1))/15 - \
        2*sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 1))/15 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1),
             JzKet(1, 0))/5
    assert uncouple(JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1))) == \
        -4*sqrt(5)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/15 + \
        sqrt(5)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/15 - \
        2*sqrt(10)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))/15 + \
        sqrt(10)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 0))/15 + \
        sqrt(10)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1),
             JzKet(1, -1))/5
    assert uncouple(JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1))) == \
        -sqrt(10)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/5 - \
        sqrt(10)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 0))/15 + \
        2*sqrt(10)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/15 - \
        sqrt(5)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/15 + \
        4*sqrt(5)*TensorProduct(
            JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/15
    assert uncouple(JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, 1, 1))) == \
        -sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 0))/5 + \
        2*sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, -1))/15 + \
        sqrt(30)*TensorProduct(
            JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, -1))/15
    assert uncouple(JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1))) == \
        TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/3 - \
        TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/3 + \
        sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))/6 - \
        sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 0))/3 + \
        sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1),
             JzKet(1, -1))/2
    assert uncouple(JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1))) == \
        sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/2 - \
        sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 0))/3 + \
        sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/6 - \
        TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/3 + \
        TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/3
    # j1=1, j2=1, j3=1
    assert uncouple(JzKetCoupled(3, 3, (1, 1, 1))) == \
        TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, 1))
    assert uncouple(JzKetCoupled(3, 2, (1, 1, 1))) == \
        sqrt(3)*TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, 1))/3 + \
        sqrt(3)*TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, 1))/3 + \
        sqrt(3)*TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, 0))/3
    assert uncouple(JzKetCoupled(3, 1, (1, 1, 1))) == \
        sqrt(15)*TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, 1))/15 + \
        2*sqrt(15)*TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, 1))/15 + \
        2*sqrt(15)*TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, 0))/15 + \
        sqrt(15)*TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, 1))/15 + \
        2*sqrt(15)*TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, 0))/15 + \
        sqrt(15)*TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, -1))/15
    assert uncouple(JzKetCoupled(3, 0, (1, 1, 1))) == \
        sqrt(10)*TensorProduct(JzKet(1, -1), JzKet(1, 0), JzKet(1, 1))/10 + \
        sqrt(10)*TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, 0))/10 + \
        sqrt(10)*TensorProduct(JzKet(1, 0), JzKet(1, -1), JzKet(1, 1))/10 + \
        sqrt(10)*TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, 0))/5 + \
        sqrt(10)*TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, -1))/10 + \
        sqrt(10)*TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, 0))/10 + \
        sqrt(10)*TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, -1))/10
    assert uncouple(JzKetCoupled(3, -1, (1, 1, 1))) == \
        sqrt(15)*TensorProduct(JzKet(1, -1), JzKet(1, -1), JzKet(1, 1))/15 + \
        2*sqrt(15)*TensorProduct(JzKet(1, -1), JzKet(1, 0), JzKet(1, 0))/15 + \
        sqrt(15)*TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, -1))/15 + \
        2*sqrt(15)*TensorProduct(JzKet(1, 0), JzKet(1, -1), JzKet(1, 0))/15 + \
        2*sqrt(15)*TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, -1))/15 + \
        sqrt(15)*TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, -1))/15
    assert uncouple(JzKetCoupled(3, -2, (1, 1, 1))) == \
        sqrt(3)*TensorProduct(JzKet(1, -1), JzKet(1, -1), JzKet(1, 0))/3 + \
        sqrt(3)*TensorProduct(JzKet(1, -1), JzKet(1, 0), JzKet(1, -1))/3 + \
        sqrt(3)*TensorProduct(JzKet(1, 0), JzKet(1, -1), JzKet(1, -1))/3
    assert uncouple(JzKetCoupled(3, -3, (1, 1, 1))) == \
        TensorProduct(JzKet(1, -1), JzKet(1, -1), JzKet(1, -1))
    assert uncouple(JzKetCoupled(2, 2, (1, 1, 1))) == \
        -sqrt(6)*TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, 1))/6 - \
        sqrt(6)*TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, 1))/6 + \
        sqrt(6)*TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, 0))/3
    assert uncouple(JzKetCoupled(2, 1, (1, 1, 1))) == \
        -sqrt(3)*TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, 1))/6 - \
        sqrt(3)*TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, 1))/3 + \
        sqrt(3)*TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, 0))/6 - \
        sqrt(3)*TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, 1))/6 + \
        sqrt(3)*TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, 0))/6 + \
        sqrt(3)*TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, -1))/3
    assert uncouple(JzKetCoupled(2, 0, (1, 1, 1))) == \
        -TensorProduct(JzKet(1, -1), JzKet(1, 0), JzKet(1, 1))/2 - \
        TensorProduct(JzKet(1, 0), JzKet(1, -1), JzKet(1, 1))/2 + \
        TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, -1))/2 + \
        TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, -1))/2
    assert uncouple(JzKetCoupled(2, -1, (1, 1, 1))) == \
        -sqrt(3)*TensorProduct(JzKet(1, -1), JzKet(1, -1), JzKet(1, 1))/3 - \
        sqrt(3)*TensorProduct(JzKet(1, -1), JzKet(1, 0), JzKet(1, 0))/6 + \
        sqrt(3)*TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, -1))/6 - \
        sqrt(3)*TensorProduct(JzKet(1, 0), JzKet(1, -1), JzKet(1, 0))/6 + \
        sqrt(3)*TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, -1))/3 + \
        sqrt(3)*TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, -1))/6
    assert uncouple(JzKetCoupled(2, -2, (1, 1, 1))) == \
        -sqrt(6)*TensorProduct(JzKet(1, -1), JzKet(1, -1), JzKet(1, 0))/3 + \
        sqrt(6)*TensorProduct(JzKet(1, -1), JzKet(1, 0), JzKet(1, -1))/6 + \
        sqrt(6)*TensorProduct(JzKet(1, 0), JzKet(1, -1), JzKet(1, -1))/6
    assert uncouple(JzKetCoupled(1, 1, (1, 1, 1))) == \
        sqrt(15)*TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, 1))/30 + \
        sqrt(15)*TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, 1))/15 - \
        sqrt(15)*TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, 0))/10 + \
        sqrt(15)*TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, 1))/30 - \
        sqrt(15)*TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, 0))/10 + \
        sqrt(15)*TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, -1))/5
    assert uncouple(JzKetCoupled(1, 0, (1, 1, 1))) == \
        sqrt(15)*TensorProduct(JzKet(1, -1), JzKet(1, 0), JzKet(1, 1))/10 - \
        sqrt(15)*TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, 0))/15 + \
        sqrt(15)*TensorProduct(JzKet(1, 0), JzKet(1, -1), JzKet(1, 1))/10 - \
        2*sqrt(15)*TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, 0))/15 + \
        sqrt(15)*TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, -1))/10 - \
        sqrt(15)*TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, 0))/15 + \
        sqrt(15)*TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, -1))/10
    assert uncouple(JzKetCoupled(1, -1, (1, 1, 1))) == \
        sqrt(15)*TensorProduct(JzKet(1, -1), JzKet(1, -1), JzKet(1, 1))/5 - \
        sqrt(15)*TensorProduct(JzKet(1, -1), JzKet(1, 0), JzKet(1, 0))/10 + \
        sqrt(15)*TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, -1))/30 - \
        sqrt(15)*TensorProduct(JzKet(1, 0), JzKet(1, -1), JzKet(1, 0))/10 + \
        sqrt(15)*TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, -1))/15 + \
        sqrt(15)*TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, -1))/30
    # Defined j13
    # j1=1/2, j2=1/2, j3=1, j13=1/2
    assert uncouple(JzKetCoupled(1, 1, (S.Half, S.Half, 1), ((1, 3, S.Half), (1, 2, 1)) )) == \
        -sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1))/3 + \
        sqrt(3)*TensorProduct(
            JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0))/3
    assert uncouple(JzKetCoupled(1, 0, (S.Half, S.Half, 1), ((1, 3, S.Half), (1, 2, 1)) )) == \
        -sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1))/3 - \
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0))/6 + \
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0))/6 + \
        sqrt(3)*TensorProduct(
            JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1))/3
    assert uncouple(JzKetCoupled(1, -1, (S.Half, S.Half, 1), ((1, 3, S.Half), (1, 2, 1)) )) == \
        -sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0))/3 + \
        sqrt(6)*TensorProduct(
            JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1))/3
    # j1=1/2, j2=1, j3=1, j13=1/2
    assert uncouple(JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, Rational(3, 2))))) == \
        -sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 1))/3 + \
        sqrt(3)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1),
             JzKet(1, 0))/3
    assert uncouple(JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, Rational(3, 2))))) == \
        -2*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/3 - \
        TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/3 + \
        sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 0))/3 + \
        sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1),
             JzKet(1, -1))/3
    assert uncouple(JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, Rational(3, 2))))) == \
        -sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/3 - \
        sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 0))/3 + \
        TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/3 + \
        2*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/3
    assert uncouple(JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, Rational(3, 2))))) == \
        -sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 0))/3 + \
        sqrt(6)*TensorProduct(
            JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, -1))/3
    # j1=1, j2=1, j3=1, j13=1
    assert uncouple(JzKetCoupled(2, 2, (1, 1, 1), ((1, 3, 1), (1, 2, 2)))) == \
        -sqrt(2)*TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, 1))/2 + \
        sqrt(2)*TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, 0))/2
    assert uncouple(JzKetCoupled(2, 1, (1, 1, 1), ((1, 3, 1), (1, 2, 2)))) == \
        -TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, 1))/2 - \
        TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, 1))/2 + \
        TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, 0))/2 + \
        TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, -1))/2
    assert uncouple(JzKetCoupled(2, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 2)))) == \
        -sqrt(3)*TensorProduct(JzKet(1, -1), JzKet(1, 0), JzKet(1, 1))/3 - \
        sqrt(3)*TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, 0))/6 - \
        sqrt(3)*TensorProduct(JzKet(1, 0), JzKet(1, -1), JzKet(1, 1))/6 + \
        sqrt(3)*TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, -1))/6 + \
        sqrt(3)*TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, 0))/6 + \
        sqrt(3)*TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, -1))/3
    assert uncouple(JzKetCoupled(2, -1, (1, 1, 1), ((1, 3, 1), (1, 2, 2)))) == \
        -TensorProduct(JzKet(1, -1), JzKet(1, -1), JzKet(1, 1))/2 - \
        TensorProduct(JzKet(1, -1), JzKet(1, 0), JzKet(1, 0))/2 + \
        TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, -1))/2 + \
        TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, -1))/2
    assert uncouple(JzKetCoupled(2, -2, (1, 1, 1), ((1, 3, 1), (1, 2, 2)))) == \
        -sqrt(2)*TensorProduct(JzKet(1, -1), JzKet(1, -1), JzKet(1, 0))/2 + \
        sqrt(2)*TensorProduct(JzKet(1, 0), JzKet(1, -1), JzKet(1, -1))/2
    assert uncouple(JzKetCoupled(1, 1, (1, 1, 1), ((1, 3, 1), (1, 2, 1)))) == \
        TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, 1))/2 - \
        TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, 1))/2 + \
        TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, 0))/2 - \
        TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, -1))/2
    assert uncouple(JzKetCoupled(1, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 1)))) == \
        TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, 0))/2 - \
        TensorProduct(JzKet(1, 0), JzKet(1, -1), JzKet(1, 1))/2 - \
        TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, -1))/2 + \
        TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, 0))/2
    assert uncouple(JzKetCoupled(1, -1, (1, 1, 1), ((1, 3, 1), (1, 2, 1)))) == \
        -TensorProduct(JzKet(1, -1), JzKet(1, -1), JzKet(1, 1))/2 + \
        TensorProduct(JzKet(1, -1), JzKet(1, 0), JzKet(1, 0))/2 - \
        TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, -1))/2 + \
        TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, -1))/2


def test_uncouple_4_coupled_states_numerical():
    # j1=1/2, j2=1/2, j3=1, j4=1, default coupling
    assert uncouple(JzKetCoupled(3, 3, (S.Half, S.Half, 1, 1))) == \
        TensorProduct(JzKet(
            S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 1))
    assert uncouple(JzKetCoupled(3, 2, (S.Half, S.Half, 1, 1))) == \
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 1))/6 + \
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 1))/6 + \
        sqrt(3)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 1))/3 + \
        sqrt(3)*TensorProduct(JzKet(S(
            1)/2, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 0))/3
    assert uncouple(JzKetCoupled(3, 1, (S.Half, S.Half, 1, 1))) == \
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 1))/15 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 1))/15 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 0))/15 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/15 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/15 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))/15 + \
        2*sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 0))/15 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half,
             S.Half), JzKet(1, 1), JzKet(1, -1))/15
    assert uncouple(JzKetCoupled(3, 0, (S.Half, S.Half, 1, 1))) == \
        sqrt(10)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/10 + \
        sqrt(10)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/10 + \
        sqrt(5)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))/10 + \
        sqrt(5)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 0))/5 + \
        sqrt(5)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, -1))/10 + \
        sqrt(5)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/10 + \
        sqrt(5)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 0))/5 + \
        sqrt(5)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/10 + \
        sqrt(10)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/10 + \
        sqrt(10)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half,
             S.Half), JzKet(1, 0), JzKet(1, -1))/10
    assert uncouple(JzKetCoupled(3, -1, (S.Half, S.Half, 1, 1))) == \
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/15 + \
        2*sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 0))/15 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/15 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/15 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/15 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 0))/15 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, -1))/15 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half,
             S.Half), JzKet(1, -1), JzKet(1, -1))/15
    assert uncouple(JzKetCoupled(3, -2, (S.Half, S.Half, 1, 1))) == \
        sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 0))/3 + \
        sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, -1))/3 + \
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, -1))/6 + \
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half,
             Rational(-1, 2)), JzKet(1, -1), JzKet(1, -1))/6
    assert uncouple(JzKetCoupled(3, -3, (S.Half, S.Half, 1, 1))) == \
        TensorProduct(JzKet(S.Half, -S(
            1)/2), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, -1))
    assert uncouple(JzKetCoupled(2, 2, (S.Half, S.Half, 1, 1))) == \
        -sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 1))/6 - \
        sqrt(3)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 1))/6 - \
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 1))/6 + \
        sqrt(6)*TensorProduct(JzKet(S(
            1)/2, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 0))/3
    assert uncouple(JzKetCoupled(2, 1, (S.Half, S.Half, 1, 1))) == \
        -sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 1))/6 - \
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 1))/6 + \
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 0))/12 - \
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/6 + \
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/12 - \
        sqrt(3)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))/6 + \
        sqrt(3)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 0))/6 + \
        sqrt(3)*TensorProduct(JzKet(S(
            1)/2, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, -1))/3
    assert uncouple(JzKetCoupled(2, 0, (S.Half, S.Half, 1, 1))) == \
        -TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/2 - \
        sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))/4 + \
        sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, -1))/4 - \
        sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/4 + \
        sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/4 + \
        TensorProduct(JzKet(S(
            1)/2, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/2
    assert uncouple(JzKetCoupled(2, -1, (S.Half, S.Half, 1, 1))) == \
        -sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/3 - \
        sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 0))/6 + \
        sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/6 - \
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/12 + \
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/6 - \
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 0))/12 + \
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, -1))/6 + \
        sqrt(3)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half,
             S.Half), JzKet(1, -1), JzKet(1, -1))/6
    assert uncouple(JzKetCoupled(2, -2, (S.Half, S.Half, 1, 1))) == \
        -sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 0))/3 + \
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, -1))/6 + \
        sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, -1))/6 + \
        sqrt(3)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half,
             Rational(-1, 2)), JzKet(1, -1), JzKet(1, -1))/6
    assert uncouple(JzKetCoupled(1, 1, (S.Half, S.Half, 1, 1))) == \
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 1))/30 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 1))/30 - \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 0))/20 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/30 - \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/20 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))/30 - \
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 0))/10 + \
        sqrt(15)*TensorProduct(JzKet(S(
            1)/2, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, -1))/5
    assert uncouple(JzKetCoupled(1, 0, (S.Half, S.Half, 1, 1))) == \
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/10 - \
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/15 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))/20 - \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 0))/15 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, -1))/20 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/20 - \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 0))/15 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/20 - \
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/15 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half,
             S.Half), JzKet(1, 0), JzKet(1, -1))/10
    assert uncouple(JzKetCoupled(1, -1, (S.Half, S.Half, 1, 1))) == \
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/5 - \
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 0))/10 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/30 - \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/20 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/30 - \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 0))/20 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, -1))/30 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half,
             S.Half), JzKet(1, -1), JzKet(1, -1))/30
    # j1=1/2, j2=1/2, j3=1, j4=1, j12=1, j34=1
    assert uncouple(JzKetCoupled(2, 2, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 1), (1, 3, 2)))) == \
        -sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 1))/2 + \
        sqrt(2)*TensorProduct(JzKet(S(
            1)/2, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 0))/2
    assert uncouple(JzKetCoupled(2, 1, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 1), (1, 3, 2)))) == \
        -sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 1))/4 + \
        sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 0))/4 - \
        sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/4 + \
        sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/4 - \
        TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))/2 + \
        TensorProduct(JzKet(S(
            1)/2, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, -1))/2
    assert uncouple(JzKetCoupled(2, 0, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 1), (1, 3, 2)))) == \
        -sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/6 + \
        sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/6 - \
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))/6 + \
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, -1))/6 - \
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/6 + \
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/6 - \
        sqrt(3)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/6 + \
        sqrt(3)*TensorProduct(JzKet(S(
            1)/2, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/6
    assert uncouple(JzKetCoupled(2, -1, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 1), (1, 3, 2)))) == \
        -TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/2 + \
        TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/2 - \
        sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/4 + \
        sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/4 - \
        sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 0))/4 + \
        sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half,
             Rational(-1, 2)), JzKet(1, 0), JzKet(1, -1))/4
    assert uncouple(JzKetCoupled(2, -2, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 1), (1, 3, 2)))) == \
        -sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 0))/2 + \
        sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half,
             Rational(-1, 2)), JzKet(1, 0), JzKet(1, -1))/2
    assert uncouple(JzKetCoupled(1, 1, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 1), (1, 3, 1)))) == \
        sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 1))/4 - \
        sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 0))/4 + \
        sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/4 - \
        sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/4 - \
        TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))/2 + \
        TensorProduct(JzKet(S(
            1)/2, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, -1))/2
    assert uncouple(JzKetCoupled(1, 0, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 1), (1, 3, 1)))) == \
        TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/2 - \
        TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/2 - \
        TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/2 + \
        TensorProduct(JzKet(S(
            1)/2, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/2
    assert uncouple(JzKetCoupled(1, -1, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 1), (1, 3, 1)))) == \
        TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/2 - \
        TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/2 - \
        sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/4 + \
        sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/4 - \
        sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 0))/4 + \
        sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half,
             Rational(-1, 2)), JzKet(1, 0), JzKet(1, -1))/4
    # j1=1/2, j2=1/2, j3=1, j4=1, j12=1, j34=2
    assert uncouple(JzKetCoupled(3, 3, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 2), (1, 3, 3)))) == \
        TensorProduct(JzKet(
            S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 1))
    assert uncouple(JzKetCoupled(3, 2, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 2), (1, 3, 3)))) == \
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 1))/6 + \
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 1))/6 + \
        sqrt(3)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 1))/3 + \
        sqrt(3)*TensorProduct(JzKet(S(
            1)/2, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 0))/3
    assert uncouple(JzKetCoupled(3, 1, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 2), (1, 3, 3)))) == \
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 1))/15 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 1))/15 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 0))/15 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/15 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/15 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))/15 + \
        2*sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 0))/15 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half,
             S.Half), JzKet(1, 1), JzKet(1, -1))/15
    assert uncouple(JzKetCoupled(3, 0, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 2), (1, 3, 3)))) == \
        sqrt(10)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/10 + \
        sqrt(10)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/10 + \
        sqrt(5)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))/10 + \
        sqrt(5)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 0))/5 + \
        sqrt(5)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, -1))/10 + \
        sqrt(5)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/10 + \
        sqrt(5)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 0))/5 + \
        sqrt(5)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/10 + \
        sqrt(10)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/10 + \
        sqrt(10)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half,
             S.Half), JzKet(1, 0), JzKet(1, -1))/10
    assert uncouple(JzKetCoupled(3, -1, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 2), (1, 3, 3)))) == \
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/15 + \
        2*sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 0))/15 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/15 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/15 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/15 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 0))/15 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, -1))/15 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half,
             S.Half), JzKet(1, -1), JzKet(1, -1))/15
    assert uncouple(JzKetCoupled(3, -2, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 2), (1, 3, 3)))) == \
        sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 0))/3 + \
        sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, -1))/3 + \
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, -1))/6 + \
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half,
             Rational(-1, 2)), JzKet(1, -1), JzKet(1, -1))/6
    assert uncouple(JzKetCoupled(3, -3, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 2), (1, 3, 3)))) == \
        TensorProduct(JzKet(S.Half, -S(
            1)/2), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, -1))
    assert uncouple(JzKetCoupled(2, 2, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 2), (1, 3, 2)))) == \
        -sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 1))/3 - \
        sqrt(3)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 1))/3 + \
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 1))/6 + \
        sqrt(6)*TensorProduct(JzKet(S(
            1)/2, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 0))/6
    assert uncouple(JzKetCoupled(2, 1, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 2), (1, 3, 2)))) == \
        -sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 1))/3 - \
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 1))/12 - \
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 0))/12 - \
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/12 - \
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/12 + \
        sqrt(3)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))/6 + \
        sqrt(3)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 0))/3 + \
        sqrt(3)*TensorProduct(JzKet(S(
            1)/2, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, -1))/6
    assert uncouple(JzKetCoupled(2, 0, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 2), (1, 3, 2)))) == \
        -TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/2 - \
        TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/2 + \
        TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/2 + \
        TensorProduct(JzKet(S(
            1)/2, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/2
    assert uncouple(JzKetCoupled(2, -1, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 2), (1, 3, 2)))) == \
        -sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/6 - \
        sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 0))/3 - \
        sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/6 + \
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/12 + \
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/12 + \
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 0))/12 + \
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, -1))/12 + \
        sqrt(3)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half,
             S.Half), JzKet(1, -1), JzKet(1, -1))/3
    assert uncouple(JzKetCoupled(2, -2, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 2), (1, 3, 2)))) == \
        -sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 0))/6 - \
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, -1))/6 + \
        sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, -1))/3 + \
        sqrt(3)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half,
             Rational(-1, 2)), JzKet(1, -1), JzKet(1, -1))/3
    assert uncouple(JzKetCoupled(1, 1, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 2), (1, 3, 1)))) == \
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 1))/5 - \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 1))/20 - \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 0))/20 - \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/20 - \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/20 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))/30 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 0))/15 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half,
             S.Half), JzKet(1, 1), JzKet(1, -1))/30
    assert uncouple(JzKetCoupled(1, 0, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 2), (1, 3, 1)))) == \
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/10 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/10 - \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))/30 - \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 0))/15 - \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, -1))/30 - \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/30 - \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 0))/15 - \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/30 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/10 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half,
             S.Half), JzKet(1, 0), JzKet(1, -1))/10
    assert uncouple(JzKetCoupled(1, -1, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 2), (1, 3, 1)))) == \
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/30 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 0))/15 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/30 - \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/20 - \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/20 - \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 0))/20 - \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, -1))/20 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half,
             S.Half), JzKet(1, -1), JzKet(1, -1))/5


def test_uncouple_symbolic():
    assert uncouple(JzKetCoupled(j, m, (j1, j2) )) == \
        Sum(CG(j1, m1, j2, m2, j, m) *
            TensorProduct(JzKet(j1, m1), JzKet(j2, m2)),
            (m1, -j1, j1), (m2, -j2, j2))
    assert uncouple(JzKetCoupled(j, m, (j1, j2, j3) )) == \
        Sum(CG(j1, m1, j2, m2, j1 + j2, m1 + m2) * CG(j1 + j2, m1 + m2, j3, m3, j, m) *
            TensorProduct(JzKet(j1, m1), JzKet(j2, m2), JzKet(j3, m3)),
            (m1, -j1, j1), (m2, -j2, j2), (m3, -j3, j3))
    assert uncouple(JzKetCoupled(j, m, (j1, j2, j3), ((1, 3, j13), (1, 2, j)) )) == \
        Sum(CG(j1, m1, j3, m3, j13, m1 + m3) * CG(j13, m1 + m3, j2, m2, j, m) *
            TensorProduct(JzKet(j1, m1), JzKet(j2, m2), JzKet(j3, m3)),
            (m1, -j1, j1), (m2, -j2, j2), (m3, -j3, j3))
    assert uncouple(JzKetCoupled(j, m, (j1, j2, j3, j4) )) == \
        Sum(CG(j1, m1, j2, m2, j1 + j2, m1 + m2) * CG(j1 + j2, m1 + m2, j3, m3, j1 + j2 + j3, m1 + m2 + m3) * CG(j1 + j2 + j3, m1 + m2 + m3, j4, m4, j, m) *
            TensorProduct(
                JzKet(j1, m1), JzKet(j2, m2), JzKet(j3, m3), JzKet(j4, m4)),
            (m1, -j1, j1), (m2, -j2, j2), (m3, -j3, j3), (m4, -j4, j4))
    assert uncouple(JzKetCoupled(j, m, (j1, j2, j3, j4), ((1, 3, j13), (2, 4, j24), (1, 2, j)) )) ==  \
        Sum(CG(j1, m1, j3, m3, j13, m1 + m3) * CG(j2, m2, j4, m4, j24, m2 + m4) * CG(j13, m1 + m3, j24, m2 + m4, j, m) *
            TensorProduct(
                JzKet(j1, m1), JzKet(j2, m2), JzKet(j3, m3), JzKet(j4, m4)),
            (m1, -j1, j1), (m2, -j2, j2), (m3, -j3, j3), (m4, -j4, j4))


def test_couple_2_states():
    # j1=1/2, j2=1/2
    assert JzKetCoupled(0, 0, (S.Half, S.Half)) == \
        expand(couple(uncouple( JzKetCoupled(0, 0, (S.Half, S.Half)) )))
    assert JzKetCoupled(1, 1, (S.Half, S.Half)) == \
        expand(couple(uncouple( JzKetCoupled(1, 1, (S.Half, S.Half)) )))
    assert JzKetCoupled(1, 0, (S.Half, S.Half)) == \
        expand(couple(uncouple( JzKetCoupled(1, 0, (S.Half, S.Half)) )))
    assert JzKetCoupled(1, -1, (S.Half, S.Half)) == \
        expand(couple(uncouple( JzKetCoupled(1, -1, (S.Half, S.Half)) )))
    # j1=1, j2=1/2
    assert JzKetCoupled(S.Half, S.Half, (1, S.Half)) == \
        expand(couple(uncouple( JzKetCoupled(S.Half, S.Half, (1, S.Half)) )))
    assert JzKetCoupled(S.Half, Rational(-1, 2), (1, S.Half)) == \
        expand(couple(uncouple( JzKetCoupled(S.Half, Rational(-1, 2), (1, S.Half)) )))
    assert JzKetCoupled(Rational(3, 2), Rational(3, 2), (1, S.Half)) == \
        expand(couple(uncouple( JzKetCoupled(Rational(3, 2), Rational(3, 2), (1, S.Half)) )))
    assert JzKetCoupled(Rational(3, 2), S.Half, (1, S.Half)) == \
        expand(couple(uncouple( JzKetCoupled(Rational(3, 2), S.Half, (1, S.Half)) )))
    assert JzKetCoupled(Rational(3, 2), Rational(-1, 2), (1, S.Half)) == \
        expand(couple(uncouple( JzKetCoupled(Rational(3, 2), Rational(-1, 2), (1, S.Half)) )))
    assert JzKetCoupled(Rational(3, 2), Rational(-3, 2), (1, S.Half)) == \
        expand(couple(uncouple( JzKetCoupled(Rational(3, 2), Rational(-3, 2), (1, S.Half)) )))
    # j1=1, j2=1
    assert JzKetCoupled(0, 0, (1, 1)) == \
        expand(couple(uncouple( JzKetCoupled(0, 0, (1, 1)) )))
    assert JzKetCoupled(1, 1, (1, 1)) == \
        expand(couple(uncouple( JzKetCoupled(1, 1, (1, 1)) )))
    assert JzKetCoupled(1, 0, (1, 1)) == \
        expand(couple(uncouple( JzKetCoupled(1, 0, (1, 1)) )))
    assert JzKetCoupled(1, -1, (1, 1)) == \
        expand(couple(uncouple( JzKetCoupled(1, -1, (1, 1)) )))
    assert JzKetCoupled(2, 2, (1, 1)) == \
        expand(couple(uncouple( JzKetCoupled(2, 2, (1, 1)) )))
    assert JzKetCoupled(2, 1, (1, 1)) == \
        expand(couple(uncouple( JzKetCoupled(2, 1, (1, 1)) )))
    assert JzKetCoupled(2, 0, (1, 1)) == \
        expand(couple(uncouple( JzKetCoupled(2, 0, (1, 1)) )))
    assert JzKetCoupled(2, -1, (1, 1)) == \
        expand(couple(uncouple( JzKetCoupled(2, -1, (1, 1)) )))
    assert JzKetCoupled(2, -2, (1, 1)) == \
        expand(couple(uncouple( JzKetCoupled(2, -2, (1, 1)) )))
    # j1=1/2, j2=3/2
    assert JzKetCoupled(1, 1, (S.Half, Rational(3, 2))) == \
        expand(couple(uncouple( JzKetCoupled(1, 1, (S.Half, Rational(3, 2))) )))
    assert JzKetCoupled(1, 0, (S.Half, Rational(3, 2))) == \
        expand(couple(uncouple( JzKetCoupled(1, 0, (S.Half, Rational(3, 2))) )))
    assert JzKetCoupled(1, -1, (S.Half, Rational(3, 2))) == \
        expand(couple(uncouple( JzKetCoupled(1, -1, (S.Half, Rational(3, 2))) )))
    assert JzKetCoupled(2, 2, (S.Half, Rational(3, 2))) == \
        expand(couple(uncouple( JzKetCoupled(2, 2, (S.Half, Rational(3, 2))) )))
    assert JzKetCoupled(2, 1, (S.Half, Rational(3, 2))) == \
        expand(couple(uncouple( JzKetCoupled(2, 1, (S.Half, Rational(3, 2))) )))
    assert JzKetCoupled(2, 0, (S.Half, Rational(3, 2))) == \
        expand(couple(uncouple( JzKetCoupled(2, 0, (S.Half, Rational(3, 2))) )))
    assert JzKetCoupled(2, -1, (S.Half, Rational(3, 2))) == \
        expand(couple(uncouple( JzKetCoupled(2, -1, (S.Half, Rational(3, 2))) )))
    assert JzKetCoupled(2, -2, (S.Half, Rational(3, 2))) == \
        expand(couple(uncouple( JzKetCoupled(2, -2, (S.Half, Rational(3, 2))) )))


def test_couple_3_states():
    # Default coupling
    # j1=1/2, j2=1/2, j3=1/2
    assert JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half)) == \
        expand(couple(uncouple(
            JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half)) )))
    assert JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half)) == \
        expand(couple(uncouple(
            JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half)) )))
    assert JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half)) == \
        expand(couple(uncouple(
            JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half)) )))
    assert JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half)) == \
        expand(couple(uncouple(
            JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half)) )))
    assert JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half)) == \
        expand(couple(uncouple(
            JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half)) )))
    assert JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half)) == \
        expand(couple(uncouple(
            JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half)) )))
    # j1=1/2, j2=1/2, j3=1
    assert JzKetCoupled(0, 0, (S.Half, S.Half, 1)) == \
        expand(couple(uncouple( JzKetCoupled(0, 0, (S.Half, S.Half, 1)) )))
    assert JzKetCoupled(1, 1, (S.Half, S.Half, 1)) == \
        expand(couple(uncouple( JzKetCoupled(1, 1, (S.Half, S.Half, 1)) )))
    assert JzKetCoupled(1, 0, (S.Half, S.Half, 1)) == \
        expand(couple(uncouple( JzKetCoupled(1, 0, (S.Half, S.Half, 1)) )))
    assert JzKetCoupled(1, -1, (S.Half, S.Half, 1)) == \
        expand(couple(uncouple( JzKetCoupled(1, -1, (S.Half, S.Half, 1)) )))
    assert JzKetCoupled(2, 2, (S.Half, S.Half, 1)) == \
        expand(couple(uncouple( JzKetCoupled(2, 2, (S.Half, S.Half, 1)) )))
    assert JzKetCoupled(2, 1, (S.Half, S.Half, 1)) == \
        expand(couple(uncouple( JzKetCoupled(2, 1, (S.Half, S.Half, 1)) )))
    assert JzKetCoupled(2, 0, (S.Half, S.Half, 1)) == \
        expand(couple(uncouple( JzKetCoupled(2, 0, (S.Half, S.Half, 1)) )))
    assert JzKetCoupled(2, -1, (S.Half, S.Half, 1)) == \
        expand(couple(uncouple( JzKetCoupled(2, -1, (S.Half, S.Half, 1)) )))
    assert JzKetCoupled(2, -2, (S.Half, S.Half, 1)) == \
        expand(couple(uncouple( JzKetCoupled(2, -2, (S.Half, S.Half, 1)) )))
    # Couple j1+j3=j13, j13+j2=j
    # j1=1/2, j2=1/2, j3=1/2, j13=0
    assert JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half), ((1, 3, 0), (1, 2, S.Half))) == \
        expand(couple(uncouple( JzKetCoupled(S.Half, S.Half, (S.Half, S(
            1)/2, S.Half), ((1, 3, 0), (1, 2, S.Half))) ), ((1, 3), (1, 2)) ))
    assert JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half), ((1, 3, 0), (1, 2, S.Half))) == \
        expand(couple(uncouple( JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S(
            1)/2, S.Half), ((1, 3, 0), (1, 2, S.Half))) ), ((1, 3), (1, 2)) ))
    # j1=1, j2=1/2, j3=1, j13=1
    assert JzKetCoupled(S.Half, S.Half, (1, S.Half, 1), ((1, 3, 1), (1, 2, S.Half))) == \
        expand(couple(uncouple( JzKetCoupled(S.Half, S.Half, (
            1, S.Half, 1), ((1, 3, 1), (1, 2, S.Half))) ), ((1, 3), (1, 2)) ))
    assert JzKetCoupled(S.Half, Rational(-1, 2), (1, S.Half, 1), ((1, 3, 1), (1, 2, S.Half))) == \
        expand(couple(uncouple( JzKetCoupled(S.Half, Rational(-1, 2), (
            1, S.Half, 1), ((1, 3, 1), (1, 2, S.Half))) ), ((1, 3), (1, 2)) ))
    assert JzKetCoupled(Rational(3, 2), Rational(3, 2), (1, S.Half, 1), ((1, 3, 1), (1, 2, Rational(3, 2)))) == \
        expand(couple(uncouple( JzKetCoupled(Rational(3, 2), Rational(3, 2), (
            1, S.Half, 1), ((1, 3, 1), (1, 2, Rational(3, 2)))) ), ((1, 3), (1, 2)) ))
    assert JzKetCoupled(Rational(3, 2), S.Half, (1, S.Half, 1), ((1, 3, 1), (1, 2, Rational(3, 2)))) == \
        expand(couple(uncouple( JzKetCoupled(Rational(3, 2), S.Half, (
            1, S.Half, 1), ((1, 3, 1), (1, 2, Rational(3, 2)))) ), ((1, 3), (1, 2)) ))
    assert JzKetCoupled(Rational(3, 2), Rational(-1, 2), (1, S.Half, 1), ((1, 3, 1), (1, 2, Rational(3, 2)))) == \
        expand(couple(uncouple( JzKetCoupled(Rational(3, 2), Rational(-1, 2), (
            1, S.Half, 1), ((1, 3, 1), (1, 2, Rational(3, 2)))) ), ((1, 3), (1, 2)) ))
    assert JzKetCoupled(Rational(3, 2), Rational(-3, 2), (1, S.Half, 1), ((1, 3, 1), (1, 2, Rational(3, 2)))) == \
        expand(couple(uncouple( JzKetCoupled(Rational(3, 2), Rational(-3, 2), (
            1, S.Half, 1), ((1, 3, 1), (1, 2, Rational(3, 2)))) ), ((1, 3), (1, 2)) ))


def test_couple_4_states():
    # Default coupling
    # j1=1/2, j2=1/2, j3=1/2, j4=1/2
    assert JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half)) == \
        expand(couple(
            uncouple( JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half)) )))
    assert JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half)) == \
        expand(couple(
            uncouple( JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half)) )))
    assert JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half)) == \
        expand(couple(uncouple(
            JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half)) )))
    assert JzKetCoupled(2, 2, (S.Half, S.Half, S.Half, S.Half)) == \
        expand(couple(
            uncouple( JzKetCoupled(2, 2, (S.Half, S.Half, S.Half, S.Half)) )))
    assert JzKetCoupled(2, 1, (S.Half, S.Half, S.Half, S.Half)) == \
        expand(couple(
            uncouple( JzKetCoupled(2, 1, (S.Half, S.Half, S.Half, S.Half)) )))
    assert JzKetCoupled(2, 0, (S.Half, S.Half, S.Half, S.Half)) == \
        expand(couple(
            uncouple( JzKetCoupled(2, 0, (S.Half, S.Half, S.Half, S.Half)) )))
    assert JzKetCoupled(2, -1, (S.Half, S.Half, S.Half, S.Half)) == \
        expand(couple(uncouple(
            JzKetCoupled(2, -1, (S.Half, S.Half, S.Half, S.Half)) )))
    assert JzKetCoupled(2, -2, (S.Half, S.Half, S.Half, S.Half)) == \
        expand(couple(uncouple(
            JzKetCoupled(2, -2, (S.Half, S.Half, S.Half, S.Half)) )))
    # j1=1/2, j2=1/2, j3=1/2, j4=1
    assert JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1)) == \
        expand(couple(uncouple(
            JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1)) )))
    assert JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1)) == \
        expand(couple(uncouple(
            JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1)) )))
    assert JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1)) == \
        expand(couple(uncouple(
            JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1)) )))
    assert JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1)) == \
        expand(couple(uncouple(
            JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1)) )))
    assert JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1)) == \
        expand(couple(uncouple(
            JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1)) )))
    assert JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1)) == \
        expand(couple(uncouple(
            JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1)) )))
    assert JzKetCoupled(Rational(5, 2), Rational(5, 2), (S.Half, S.Half, S.Half, 1)) == \
        expand(couple(uncouple(
            JzKetCoupled(Rational(5, 2), Rational(5, 2), (S.Half, S.Half, S.Half, 1)) )))
    assert JzKetCoupled(Rational(5, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1)) == \
        expand(couple(uncouple(
            JzKetCoupled(Rational(5, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1)) )))
    assert JzKetCoupled(Rational(5, 2), S.Half, (S.Half, S.Half, S.Half, 1)) == \
        expand(couple(uncouple(
            JzKetCoupled(Rational(5, 2), S.Half, (S.Half, S.Half, S.Half, 1)) )))
    assert JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1)) == \
        expand(couple(uncouple(
            JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1)) )))
    assert JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1)) == \
        expand(couple(uncouple(
            JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1)) )))
    assert JzKetCoupled(Rational(5, 2), Rational(-5, 2), (S.Half, S.Half, S.Half, 1)) == \
        expand(couple(uncouple(
            JzKetCoupled(Rational(5, 2), Rational(-5, 2), (S.Half, S.Half, S.Half, 1)) )))
    # Coupling j1+j3=j13, j2+j4=j24, j13+j24=j
    # j1=1/2, j2=1/2, j3=1/2, j4=1/2, j13=1, j24=0
    assert JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 3, 1), (2, 4, 0), (1, 2, 1)) ) == \
        expand(couple(uncouple( JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 3, 1), (2, 4, 0), (1, 2, 1)) ) ), ((1, 3), (2, 4), (1, 2)) ))
    assert JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 3, 1), (2, 4, 0), (1, 2, 1)) ) == \
        expand(couple(uncouple( JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 3, 1), (2, 4, 0), (1, 2, 1)) ) ), ((1, 3), (2, 4), (1, 2)) ))
    assert JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 3, 1), (2, 4, 0), (1, 2, 1)) ) == \
        expand(couple(uncouple( JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 3, 1), (2, 4, 0), (1, 2, 1)) ) ), ((1, 3), (2, 4), (1, 2)) ))
    # j1=1/2, j2=1/2, j3=1/2, j4=1, j13=1, j24=1/2
    assert JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 3, 1), (2, 4, S.Half), (1, 2, S.Half)) ) == \
        expand(couple(uncouple( JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 3, 1), (2, 4, S.Half), (1, 2, S.Half)) )), ((1, 3), (2, 4), (1, 2)) ))
    assert JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 3, 1), (2, 4, S.Half), (1, 2, S.Half)) ) == \
        expand(couple(uncouple( JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 3, 1), (2, 4, S.Half), (1, 2, S.Half)) ) ), ((1, 3), (2, 4), (1, 2)) ))
    assert JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 3, 1), (2, 4, S.Half), (1, 2, Rational(3, 2))) ) == \
        expand(couple(uncouple( JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 3, 1), (2, 4, S.Half), (1, 2, Rational(3, 2))) ) ), ((1, 3), (2, 4), (1, 2)) ))
    assert JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 3, 1), (2, 4, S.Half), (1, 2, Rational(3, 2))) ) == \
        expand(couple(uncouple( JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 3, 1), (2, 4, S.Half), (1, 2, Rational(3, 2))) ) ), ((1, 3), (2, 4), (1, 2)) ))
    assert JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 3, 1), (2, 4, S.Half), (1, 2, Rational(3, 2))) ) == \
        expand(couple(uncouple( JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 3, 1), (2, 4, S.Half), (1, 2, Rational(3, 2))) ) ), ((1, 3), (2, 4), (1, 2)) ))
    assert JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 3, 1), (2, 4, S.Half), (1, 2, Rational(3, 2))) ) == \
        expand(couple(uncouple( JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 3, 1), (2, 4, S.Half), (1, 2, Rational(3, 2))) ) ), ((1, 3), (2, 4), (1, 2)) ))
    # j1=1/2, j2=1, j3=1/2, j4=1, j13=0, j24=1
    assert JzKetCoupled(1, 1, (S.Half, 1, S.Half, 1), ((1, 3, 0), (2, 4, 1), (1, 2, 1)) ) == \
        expand(couple(uncouple( JzKetCoupled(1, 1, (S.Half, 1, S.Half, 1), (
            (1, 3, 0), (2, 4, 1), (1, 2, 1))) ), ((1, 3), (2, 4), (1, 2)) ))
    assert JzKetCoupled(1, 0, (S.Half, 1, S.Half, 1), ((1, 3, 0), (2, 4, 1), (1, 2, 1)) ) == \
        expand(couple(uncouple( JzKetCoupled(1, 0, (S.Half, 1, S.Half, 1), (
            (1, 3, 0), (2, 4, 1), (1, 2, 1))) ), ((1, 3), (2, 4), (1, 2)) ))
    assert JzKetCoupled(1, -1, (S.Half, 1, S.Half, 1), ((1, 3, 0), (2, 4, 1), (1, 2, 1)) ) == \
        expand(couple(uncouple( JzKetCoupled(1, -1, (S.Half, 1, S.Half, 1), (
            (1, 3, 0), (2, 4, 1), (1, 2, 1))) ), ((1, 3), (2, 4), (1, 2)) ))
    # j1=1/2, j2=1, j3=1/2, j4=1, j13=1, j24=1
    assert JzKetCoupled(0, 0, (S.Half, 1, S.Half, 1), ((1, 3, 1), (2, 4, 1), (1, 2, 0)) ) == \
        expand(couple(uncouple( JzKetCoupled(0, 0, (S.Half, 1, S.Half, 1), (
            (1, 3, 1), (2, 4, 1), (1, 2, 0))) ), ((1, 3), (2, 4), (1, 2)) ))
    assert JzKetCoupled(1, 1, (S.Half, 1, S.Half, 1), ((1, 3, 1), (2, 4, 1), (1, 2, 1)) ) == \
        expand(couple(uncouple( JzKetCoupled(1, 1, (S.Half, 1, S.Half, 1), (
            (1, 3, 1), (2, 4, 1), (1, 2, 1))) ), ((1, 3), (2, 4), (1, 2)) ))
    assert JzKetCoupled(1, 0, (S.Half, 1, S.Half, 1), ((1, 3, 1), (2, 4, 1), (1, 2, 1)) ) == \
        expand(couple(uncouple( JzKetCoupled(1, 0, (S.Half, 1, S.Half, 1), (
            (1, 3, 1), (2, 4, 1), (1, 2, 1))) ), ((1, 3), (2, 4), (1, 2)) ))
    assert JzKetCoupled(1, -1, (S.Half, 1, S.Half, 1), ((1, 3, 1), (2, 4, 1), (1, 2, 1)) ) == \
        expand(couple(uncouple( JzKetCoupled(1, -1, (S.Half, 1, S.Half, 1), (
            (1, 3, 1), (2, 4, 1), (1, 2, 1))) ), ((1, 3), (2, 4), (1, 2)) ))
    assert JzKetCoupled(2, 2, (S.Half, 1, S.Half, 1), ((1, 3, 1), (2, 4, 1), (1, 2, 2)) ) == \
        expand(couple(uncouple( JzKetCoupled(2, 2, (S.Half, 1, S.Half, 1), (
            (1, 3, 1), (2, 4, 1), (1, 2, 2))) ), ((1, 3), (2, 4), (1, 2)) ))
    assert JzKetCoupled(2, 1, (S.Half, 1, S.Half, 1), ((1, 3, 1), (2, 4, 1), (1, 2, 2)) ) == \
        expand(couple(uncouple( JzKetCoupled(2, 1, (S.Half, 1, S.Half, 1), (
            (1, 3, 1), (2, 4, 1), (1, 2, 2))) ), ((1, 3), (2, 4), (1, 2)) ))
    assert JzKetCoupled(2, 0, (S.Half, 1, S.Half, 1), ((1, 3, 1), (2, 4, 1), (1, 2, 2)) ) == \
        expand(couple(uncouple( JzKetCoupled(2, 0, (S.Half, 1, S.Half, 1), (
            (1, 3, 1), (2, 4, 1), (1, 2, 2))) ), ((1, 3), (2, 4), (1, 2)) ))
    assert JzKetCoupled(2, -1, (S.Half, 1, S.Half, 1), ((1, 3, 1), (2, 4, 1), (1, 2, 2)) ) == \
        expand(couple(uncouple( JzKetCoupled(2, -1, (S.Half, 1, S.Half, 1), (
            (1, 3, 1), (2, 4, 1), (1, 2, 2))) ), ((1, 3), (2, 4), (1, 2)) ))
    assert JzKetCoupled(2, -2, (S.Half, 1, S.Half, 1), ((1, 3, 1), (2, 4, 1), (1, 2, 2)) ) == \
        expand(couple(uncouple( JzKetCoupled(2, -2, (S.Half, 1, S.Half, 1), (
            (1, 3, 1), (2, 4, 1), (1, 2, 2))) ), ((1, 3), (2, 4), (1, 2)) ))


def test_couple_2_states_numerical():
    # j1=1/2, j2=1/2
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half))) == \
        JzKetCoupled(1, 1, (S.Half, S.Half))
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)))) == \
        sqrt(2)*JzKetCoupled(0, 0, (S(
            1)/2, S.Half))/2 + sqrt(2)*JzKetCoupled(1, 0, (S.Half, S.Half))/2
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half))) == \
        -sqrt(2)*JzKetCoupled(0, 0, (S(
            1)/2, S.Half))/2 + sqrt(2)*JzKetCoupled(1, 0, (S.Half, S.Half))/2
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)))) == \
        JzKetCoupled(1, -1, (S.Half, S.Half))
    # j1=1, j2=1/2
    assert couple(TensorProduct(JzKet(1, 1), JzKet(S.Half, S.Half))) == \
        JzKetCoupled(Rational(3, 2), Rational(3, 2), (1, S.Half))
    assert couple(TensorProduct(JzKet(1, 1), JzKet(S.Half, Rational(-1, 2)))) == \
        sqrt(6)*JzKetCoupled(S.Half, S.Half, (1, S.Half))/3 + sqrt(
            3)*JzKetCoupled(Rational(3, 2), S.Half, (1, S.Half))/3
    assert couple(TensorProduct(JzKet(1, 0), JzKet(S.Half, S.Half))) == \
        -sqrt(3)*JzKetCoupled(S.Half, S.Half, (1, S.Half))/3 + \
        sqrt(6)*JzKetCoupled(Rational(3, 2), S.Half, (1, S.Half))/3
    assert couple(TensorProduct(JzKet(1, 0), JzKet(S.Half, Rational(-1, 2)))) == \
        sqrt(3)*JzKetCoupled(S.Half, Rational(-1, 2), (1, S.Half))/3 + \
        sqrt(6)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (1, S.Half))/3
    assert couple(TensorProduct(JzKet(1, -1), JzKet(S.Half, S.Half))) == \
        -sqrt(6)*JzKetCoupled(S.Half, Rational(-1, 2), (1, S(
            1)/2))/3 + sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (1, S.Half))/3
    assert couple(TensorProduct(JzKet(1, -1), JzKet(S.Half, Rational(-1, 2)))) == \
        JzKetCoupled(Rational(3, 2), Rational(-3, 2), (1, S.Half))
    # j1=1, j2=1
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, 1))) == \
        JzKetCoupled(2, 2, (1, 1))
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, 0))) == \
        sqrt(2)*JzKetCoupled(
            1, 1, (1, 1))/2 + sqrt(2)*JzKetCoupled(2, 1, (1, 1))/2
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, -1))) == \
        sqrt(3)*JzKetCoupled(0, 0, (1, 1))/3 + sqrt(2)*JzKetCoupled(
            1, 0, (1, 1))/2 + sqrt(6)*JzKetCoupled(2, 0, (1, 1))/6
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, 1))) == \
        -sqrt(2)*JzKetCoupled(
            1, 1, (1, 1))/2 + sqrt(2)*JzKetCoupled(2, 1, (1, 1))/2
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, 0))) == \
        -sqrt(3)*JzKetCoupled(
            0, 0, (1, 1))/3 + sqrt(6)*JzKetCoupled(2, 0, (1, 1))/3
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, -1))) == \
        sqrt(2)*JzKetCoupled(
            1, -1, (1, 1))/2 + sqrt(2)*JzKetCoupled(2, -1, (1, 1))/2
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, 1))) == \
        sqrt(3)*JzKetCoupled(0, 0, (1, 1))/3 - sqrt(2)*JzKetCoupled(
            1, 0, (1, 1))/2 + sqrt(6)*JzKetCoupled(2, 0, (1, 1))/6
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, 0))) == \
        -sqrt(2)*JzKetCoupled(
            1, -1, (1, 1))/2 + sqrt(2)*JzKetCoupled(2, -1, (1, 1))/2
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, -1))) == \
        JzKetCoupled(2, -2, (1, 1))
    # j1=3/2, j2=1/2
    assert couple(TensorProduct(JzKet(Rational(3, 2), Rational(3, 2)), JzKet(S.Half, S.Half))) == \
        JzKetCoupled(2, 2, (Rational(3, 2), S.Half))
    assert couple(TensorProduct(JzKet(Rational(3, 2), Rational(3, 2)), JzKet(S.Half, Rational(-1, 2)))) == \
        sqrt(3)*JzKetCoupled(
            1, 1, (Rational(3, 2), S.Half))/2 + JzKetCoupled(2, 1, (Rational(3, 2), S.Half))/2
    assert couple(TensorProduct(JzKet(Rational(3, 2), S.Half), JzKet(S.Half, S.Half))) == \
        -JzKetCoupled(1, 1, (S(
            3)/2, S.Half))/2 + sqrt(3)*JzKetCoupled(2, 1, (Rational(3, 2), S.Half))/2
    assert couple(TensorProduct(JzKet(Rational(3, 2), S.Half), JzKet(S.Half, Rational(-1, 2)))) == \
        sqrt(2)*JzKetCoupled(1, 0, (S(
            3)/2, S.Half))/2 + sqrt(2)*JzKetCoupled(2, 0, (Rational(3, 2), S.Half))/2
    assert couple(TensorProduct(JzKet(Rational(3, 2), Rational(-1, 2)), JzKet(S.Half, S.Half))) == \
        -sqrt(2)*JzKetCoupled(1, 0, (S(
            3)/2, S.Half))/2 + sqrt(2)*JzKetCoupled(2, 0, (Rational(3, 2), S.Half))/2
    assert couple(TensorProduct(JzKet(Rational(3, 2), Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)))) == \
        JzKetCoupled(1, -1, (S(
            3)/2, S.Half))/2 + sqrt(3)*JzKetCoupled(2, -1, (Rational(3, 2), S.Half))/2
    assert couple(TensorProduct(JzKet(Rational(3, 2), Rational(-3, 2)), JzKet(S.Half, S.Half))) == \
        -sqrt(3)*JzKetCoupled(1, -1, (Rational(3, 2), S.Half))/2 + \
        JzKetCoupled(2, -1, (Rational(3, 2), S.Half))/2
    assert couple(TensorProduct(JzKet(Rational(3, 2), Rational(-3, 2)), JzKet(S.Half, Rational(-1, 2)))) == \
        JzKetCoupled(2, -2, (Rational(3, 2), S.Half))


def test_couple_3_states_numerical():
    # Default coupling
    # j1=1/2,j2=1/2,j3=1/2
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half))) == \
        JzKetCoupled(Rational(3, 2), S(
            3)/2, (S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2))) )
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)))) == \
        sqrt(6)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half)) )/3 + \
        sqrt(3)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.One/
             2), ((1, 2, 1), (1, 3, Rational(3, 2))) )/3
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half))) == \
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half), ((1, 2, 0), (1, 3, S.Half)) )/2 - \
        sqrt(6)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half)) )/6 + \
        sqrt(3)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.One/
             2), ((1, 2, 1), (1, 3, Rational(3, 2))) )/3
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)))) == \
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half), ((1, 2, 0), (1, 3, S.Half)) )/2 + \
        sqrt(6)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half)) )/6 + \
        sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.One
             /2), ((1, 2, 1), (1, 3, Rational(3, 2))) )/3
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half))) == \
        -sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half), ((1, 2, 0), (1, 3, S.Half)) )/2 - \
        sqrt(6)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half)) )/6 + \
        sqrt(3)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.One/
             2), ((1, 2, 1), (1, 3, Rational(3, 2))) )/3
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)))) == \
        -sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half), ((1, 2, 0), (1, 3, S.Half)) )/2 + \
        sqrt(6)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half)) )/6 + \
        sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.One
             /2), ((1, 2, 1), (1, 3, Rational(3, 2))) )/3
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half))) == \
        -sqrt(6)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half)) )/3 + \
        sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.One
             /2), ((1, 2, 1), (1, 3, Rational(3, 2))) )/3
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)))) == \
        JzKetCoupled(Rational(3, 2), -S(
            3)/2, (S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2))) )
    # j1=S.Half, j2=S.Half, j3=1
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1))) == \
        JzKetCoupled(2, 2, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 2)) )
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0))) == \
        sqrt(2)*JzKetCoupled(1, 1, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 1)) )/2 + \
        sqrt(2)*JzKetCoupled(
            2, 1, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 2)) )/2
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1))) == \
        sqrt(3)*JzKetCoupled(0, 0, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 0)) )/3 + \
        sqrt(2)*JzKetCoupled(1, 0, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 1)) )/2 + \
        sqrt(6)*JzKetCoupled(
            2, 0, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 2)) )/6
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1))) == \
        sqrt(2)*JzKetCoupled(1, 1, (S.Half, S.Half, 1), ((1, 2, 0), (1, 3, 1)) )/2 - \
        JzKetCoupled(1, 1, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 1)) )/2 + \
        JzKetCoupled(2, 1, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 2)) )/2
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0))) == \
        -sqrt(6)*JzKetCoupled(0, 0, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 0)) )/6 + \
        sqrt(2)*JzKetCoupled(1, 0, (S.Half, S.Half, 1), ((1, 2, 0), (1, 3, 1)) )/2 + \
        sqrt(3)*JzKetCoupled(
            2, 0, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 2)) )/3
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1))) == \
        sqrt(2)*JzKetCoupled(1, -1, (S.Half, S.Half, 1), ((1, 2, 0), (1, 3, 1)) )/2 + \
        JzKetCoupled(1, -1, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 1)) )/2 + \
        JzKetCoupled(2, -1, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 2)) )/2
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1))) == \
        -sqrt(2)*JzKetCoupled(1, 1, (S.Half, S.Half, 1), ((1, 2, 0), (1, 3, 1)) )/2 - \
        JzKetCoupled(1, 1, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 1)) )/2 + \
        JzKetCoupled(2, 1, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 2)) )/2
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0))) == \
        -sqrt(6)*JzKetCoupled(0, 0, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 0)) )/6 - \
        sqrt(2)*JzKetCoupled(1, 0, (S.Half, S.Half, 1), ((1, 2, 0), (1, 3, 1)) )/2 + \
        sqrt(3)*JzKetCoupled(
            2, 0, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 2)) )/3
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1))) == \
        -sqrt(2)*JzKetCoupled(1, -1, (S.Half, S.Half, 1), ((1, 2, 0), (1, 3, 1)) )/2 + \
        JzKetCoupled(1, -1, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 1)) )/2 + \
        JzKetCoupled(2, -1, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 2)) )/2
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1))) == \
        sqrt(3)*JzKetCoupled(0, 0, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 0)) )/3 - \
        sqrt(2)*JzKetCoupled(1, 0, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 1)) )/2 + \
        sqrt(6)*JzKetCoupled(
            2, 0, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 2)) )/6
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0))) == \
        -sqrt(2)*JzKetCoupled(1, -1, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 1)) )/2 + \
        sqrt(2)*JzKetCoupled(
            2, -1, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 2)) )/2
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1))) == \
        JzKetCoupled(2, -2, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 2)) )
    # j1=S.Half, j2=1, j3=1
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 1))) == \
        JzKetCoupled(
            Rational(5, 2), Rational(5, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(5, 2))) )
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 0))) == \
        sqrt(15)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(3, 2))) )/5 + \
        sqrt(10)*JzKetCoupled(S(
            5)/2, Rational(3, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, -1))) == \
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, S.Half)) )/2 + \
        sqrt(10)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(3, 2))) )/5 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, 1, 1), ((1,
             2, Rational(3, 2)), (1, 3, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 1))) == \
        sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, Rational(3, 2))) )/3 - \
        2*sqrt(15)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        sqrt(10)*JzKetCoupled(S(
            5)/2, Rational(3, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 0))) == \
        JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, S.Half)) )/3 - \
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, S.Half)) )/3 + \
        sqrt(2)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, Rational(3, 2))) )/3 + \
        sqrt(10)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        sqrt(10)*JzKetCoupled(S(
            5)/2, S.Half, (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))) == \
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, S.Half)) )/3 + \
        JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, S.Half)) )/3 + \
        JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, Rational(3, 2))) )/3 + \
        4*sqrt(5)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, 1, 1), ((1,
             2, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))) == \
        -2*JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, S.Half)) )/3 + \
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, S.Half)) )/6 + \
        sqrt(2)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, Rational(3, 2))) )/3 - \
        2*sqrt(10)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, 1, 1), ((1,
             2, Rational(3, 2)), (1, 3, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))) == \
        -sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, S.Half)) )/3 - \
        JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, S.Half)) )/3 + \
        2*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, Rational(3, 2))) )/3 - \
        sqrt(5)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, 1, 1), ((1,
             2, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, -1))) == \
        sqrt(6)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, Rational(3, 2))) )/3 + \
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, 1, 1), ((1,
             2, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 1))) == \
        -sqrt(6)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, Rational(3, 2))) )/3 - \
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(S(
            5)/2, Rational(3, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))) == \
        -sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, S.Half)) )/3 - \
        JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, S.Half)) )/3 - \
        2*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, Rational(3, 2))) )/3 + \
        sqrt(5)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(S(
            5)/2, S.Half, (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))) == \
        -2*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, S.Half)) )/3 + \
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, S.Half)) )/6 - \
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, Rational(3, 2))) )/3 + \
        2*sqrt(10)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, 1, 1), ((1,
             2, Rational(3, 2)), (1, 3, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))) == \
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, S.Half)) )/3 + \
        JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, S.Half)) )/3 - \
        JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, Rational(3, 2))) )/3 - \
        4*sqrt(5)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(S(
            5)/2, S.Half, (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 0))) == \
        JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, S.Half)) )/3 - \
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, S.Half)) )/3 - \
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, Rational(3, 2))) )/3 - \
        sqrt(10)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, 1, 1), ((1,
             2, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, -1))) == \
        -sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, Rational(3, 2))) )/3 + \
        2*sqrt(15)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, 1, 1), ((1,
             2, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))) == \
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, S.Half)) )/2 - \
        sqrt(10)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(3, 2))) )/5 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, 1, 1), ((1,
             2, Rational(3, 2)), (1, 3, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 0))) == \
        -sqrt(15)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(3, 2))) )/5 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, 1, 1), ((1,
             2, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, -1))) == \
        JzKetCoupled(S(
            5)/2, Rational(-5, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(5, 2))) )
    #  j1=1, j2=1, j3=1
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, 1))) == \
        JzKetCoupled(3, 3, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, 0))) == \
        sqrt(6)*JzKetCoupled(2, 2, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/3 + \
        sqrt(3)*JzKetCoupled(3, 2, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/3
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, -1))) == \
        sqrt(15)*JzKetCoupled(1, 1, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/5 + \
        sqrt(3)*JzKetCoupled(2, 1, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/3 + \
        sqrt(15)*JzKetCoupled(3, 1, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/15
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, 1))) == \
        sqrt(2)*JzKetCoupled(2, 2, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/2 - \
        sqrt(6)*JzKetCoupled(2, 2, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/6 + \
        sqrt(3)*JzKetCoupled(3, 2, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/3
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, 0))) == \
        JzKetCoupled(1, 1, (1, 1, 1), ((1, 2, 1), (1, 3, 1)) )/2 - \
        sqrt(15)*JzKetCoupled(1, 1, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/10 + \
        JzKetCoupled(2, 1, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/2 + \
        sqrt(3)*JzKetCoupled(2, 1, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/6 + \
        2*sqrt(15)*JzKetCoupled(3, 1, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/15
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, -1))) == \
        sqrt(6)*JzKetCoupled(0, 0, (1, 1, 1), ((1, 2, 1), (1, 3, 0)) )/6 + \
        JzKetCoupled(1, 0, (1, 1, 1), ((1, 2, 1), (1, 3, 1)) )/2 + \
        sqrt(15)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/10 + \
        sqrt(3)*JzKetCoupled(2, 0, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/6 + \
        JzKetCoupled(2, 0, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/2 + \
        sqrt(10)*JzKetCoupled(3, 0, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/10
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, 1))) == \
        sqrt(3)*JzKetCoupled(1, 1, (1, 1, 1), ((1, 2, 0), (1, 3, 1)) )/3 - \
        JzKetCoupled(1, 1, (1, 1, 1), ((1, 2, 1), (1, 3, 1)) )/2 + \
        sqrt(15)*JzKetCoupled(1, 1, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/30 + \
        JzKetCoupled(2, 1, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/2 - \
        sqrt(3)*JzKetCoupled(2, 1, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/6 + \
        sqrt(15)*JzKetCoupled(3, 1, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/15
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, 0))) == \
        -sqrt(6)*JzKetCoupled(0, 0, (1, 1, 1), ((1, 2, 1), (1, 3, 0)) )/6 + \
        sqrt(3)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 2, 0), (1, 3, 1)) )/3 - \
        sqrt(15)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/15 + \
        sqrt(3)*JzKetCoupled(2, 0, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/3 + \
        sqrt(10)*JzKetCoupled(3, 0, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/10
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, -1))) == \
        sqrt(3)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 2, 0), (1, 3, 1)) )/3 + \
        JzKetCoupled(1, -1, (1, 1, 1), ((1, 2, 1), (1, 3, 1)) )/2 + \
        sqrt(15)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/30 + \
        JzKetCoupled(2, -1, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/2 + \
        sqrt(3)*JzKetCoupled(2, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/6 + \
        sqrt(15)*JzKetCoupled(3, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/15
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, 1))) == \
        -sqrt(2)*JzKetCoupled(2, 2, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/2 - \
        sqrt(6)*JzKetCoupled(2, 2, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/6 + \
        sqrt(3)*JzKetCoupled(3, 2, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/3
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, 0))) == \
        -JzKetCoupled(1, 1, (1, 1, 1), ((1, 2, 1), (1, 3, 1)) )/2 - \
        sqrt(15)*JzKetCoupled(1, 1, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/10 - \
        JzKetCoupled(2, 1, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/2 + \
        sqrt(3)*JzKetCoupled(2, 1, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/6 + \
        2*sqrt(15)*JzKetCoupled(3, 1, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/15
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, -1))) == \
        -sqrt(6)*JzKetCoupled(0, 0, (1, 1, 1), ((1, 2, 1), (1, 3, 0)) )/6 - \
        JzKetCoupled(1, 0, (1, 1, 1), ((1, 2, 1), (1, 3, 1)) )/2 + \
        sqrt(15)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/10 - \
        sqrt(3)*JzKetCoupled(2, 0, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/6 + \
        JzKetCoupled(2, 0, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/2 + \
        sqrt(10)*JzKetCoupled(3, 0, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/10
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, 1))) == \
        -sqrt(3)*JzKetCoupled(1, 1, (1, 1, 1), ((1, 2, 0), (1, 3, 1)) )/3 + \
        sqrt(15)*JzKetCoupled(1, 1, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/15 - \
        sqrt(3)*JzKetCoupled(2, 1, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/3 + \
        2*sqrt(15)*JzKetCoupled(3, 1, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/15
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, 0))) == \
        -sqrt(3)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 2, 0), (1, 3, 1)) )/3 - \
        2*sqrt(15)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/15 + \
        sqrt(10)*JzKetCoupled(3, 0, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/5
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, -1))) == \
        -sqrt(3)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 2, 0), (1, 3, 1)) )/3 + \
        sqrt(15)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/15 + \
        sqrt(3)*JzKetCoupled(2, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/3 + \
        2*sqrt(15)*JzKetCoupled(3, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/15
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, -1), JzKet(1, 1))) == \
        sqrt(6)*JzKetCoupled(0, 0, (1, 1, 1), ((1, 2, 1), (1, 3, 0)) )/6 - \
        JzKetCoupled(1, 0, (1, 1, 1), ((1, 2, 1), (1, 3, 1)) )/2 + \
        sqrt(15)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/10 + \
        sqrt(3)*JzKetCoupled(2, 0, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/6 - \
        JzKetCoupled(2, 0, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/2 + \
        sqrt(10)*JzKetCoupled(3, 0, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/10
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, -1), JzKet(1, 0))) == \
        -JzKetCoupled(1, -1, (1, 1, 1), ((1, 2, 1), (1, 3, 1)) )/2 - \
        sqrt(15)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/10 + \
        JzKetCoupled(2, -1, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/2 - \
        sqrt(3)*JzKetCoupled(2, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/6 + \
        2*sqrt(15)*JzKetCoupled(3, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/15
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, -1), JzKet(1, -1))) == \
        sqrt(2)*JzKetCoupled(2, -2, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/2 + \
        sqrt(6)*JzKetCoupled(2, -2, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/6 + \
        sqrt(3)*JzKetCoupled(3, -2, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/3
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, 1))) == \
        sqrt(3)*JzKetCoupled(1, 1, (1, 1, 1), ((1, 2, 0), (1, 3, 1)) )/3 + \
        JzKetCoupled(1, 1, (1, 1, 1), ((1, 2, 1), (1, 3, 1)) )/2 + \
        sqrt(15)*JzKetCoupled(1, 1, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/30 - \
        JzKetCoupled(2, 1, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/2 - \
        sqrt(3)*JzKetCoupled(2, 1, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/6 + \
        sqrt(15)*JzKetCoupled(3, 1, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/15
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, 0))) == \
        sqrt(6)*JzKetCoupled(0, 0, (1, 1, 1), ((1, 2, 1), (1, 3, 0)) )/6 + \
        sqrt(3)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 2, 0), (1, 3, 1)) )/3 - \
        sqrt(15)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/15 - \
        sqrt(3)*JzKetCoupled(2, 0, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/3 + \
        sqrt(10)*JzKetCoupled(3, 0, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/10
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, -1))) == \
        sqrt(3)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 2, 0), (1, 3, 1)) )/3 - \
        JzKetCoupled(1, -1, (1, 1, 1), ((1, 2, 1), (1, 3, 1)) )/2 + \
        sqrt(15)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/30 - \
        JzKetCoupled(2, -1, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/2 + \
        sqrt(3)*JzKetCoupled(2, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/6 + \
        sqrt(15)*JzKetCoupled(3, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/15
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, 0), JzKet(1, 1))) == \
        -sqrt(6)*JzKetCoupled(0, 0, (1, 1, 1), ((1, 2, 1), (1, 3, 0)) )/6 + \
        JzKetCoupled(1, 0, (1, 1, 1), ((1, 2, 1), (1, 3, 1)) )/2 + \
        sqrt(15)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/10 - \
        sqrt(3)*JzKetCoupled(2, 0, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/6 - \
        JzKetCoupled(2, 0, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/2 + \
        sqrt(10)*JzKetCoupled(3, 0, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/10
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, 0), JzKet(1, 0))) == \
        JzKetCoupled(1, -1, (1, 1, 1), ((1, 2, 1), (1, 3, 1)) )/2 - \
        sqrt(15)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/10 - \
        JzKetCoupled(2, -1, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/2 - \
        sqrt(3)*JzKetCoupled(2, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/6 + \
        2*sqrt(15)*JzKetCoupled(3, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/15
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, 0), JzKet(1, -1))) == \
        -sqrt(2)*JzKetCoupled(2, -2, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/2 + \
        sqrt(6)*JzKetCoupled(2, -2, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/6 + \
        sqrt(3)*JzKetCoupled(3, -2, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/3
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, -1), JzKet(1, 1))) == \
        sqrt(15)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/5 - \
        sqrt(3)*JzKetCoupled(2, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/3 + \
        sqrt(15)*JzKetCoupled(3, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/15
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, -1), JzKet(1, 0))) == \
        -sqrt(6)*JzKetCoupled(2, -2, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/3 + \
        sqrt(3)*JzKetCoupled(3, -2, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/3
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, -1), JzKet(1, -1))) == \
        JzKetCoupled(3, -3, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )
    # j1=S.Half, j2=S.Half, j3=Rational(3, 2)
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(Rational(3, 2), Rational(3, 2)))) == \
        JzKetCoupled(Rational(5, 2), S(
            5)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(5, 2))) )
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(Rational(3, 2), S.Half))) == \
        sqrt(10)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(3, 2))) )/5 + \
        sqrt(15)*JzKetCoupled(Rational(5, 2), Rational(3, 2), (S.Half, S.Half, S(3)
             /2), ((1, 2, 1), (1, 3, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(Rational(3, 2), Rational(-1, 2)))) == \
        sqrt(6)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, S.Half)) )/6 + \
        2*sqrt(30)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(3, 2))) )/15 + \
        sqrt(30)*JzKetCoupled(Rational(5, 2), S(
            1)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(Rational(3, 2), Rational(-3, 2)))) == \
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, S.Half)) )/2 + \
        sqrt(10)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(3, 2))) )/5 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), -S(
            1)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(Rational(3, 2), Rational(3, 2)))) == \
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 0), (1, 3, Rational(3, 2))) )/2 - \
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(3, 2))) )/10 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(3, 2), (S.Half, S.Half, S(3)/
             2), ((1, 2, 1), (1, 3, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(Rational(3, 2), S.Half))) == \
        -sqrt(6)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, S.Half)) )/6 + \
        sqrt(2)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 0), (1, 3, Rational(3, 2))) )/2 - \
        sqrt(30)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(3, 2))) )/30 + \
        sqrt(30)*JzKetCoupled(Rational(5, 2), S(
            1)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(Rational(3, 2), Rational(-1, 2)))) == \
        -sqrt(6)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, S.Half)) )/6 + \
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 0), (1, 3, Rational(3, 2))) )/2 + \
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(3, 2))) )/30 + \
        sqrt(30)*JzKetCoupled(Rational(5, 2), -S(
            1)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(Rational(3, 2), Rational(-3, 2)))) == \
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 0), (1, 3, Rational(3, 2))) )/2 + \
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(3, 2))) )/10 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, S.Half, S(3)
             /2), ((1, 2, 1), (1, 3, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(Rational(3, 2), Rational(3, 2)))) == \
        -sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 0), (1, 3, Rational(3, 2))) )/2 - \
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(3, 2))) )/10 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(3, 2), (S.Half, S.Half, S(3)/
             2), ((1, 2, 1), (1, 3, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(Rational(3, 2), S.Half))) == \
        -sqrt(6)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, S.Half)) )/6 - \
        sqrt(2)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 0), (1, 3, Rational(3, 2))) )/2 - \
        sqrt(30)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(3, 2))) )/30 + \
        sqrt(30)*JzKetCoupled(Rational(5, 2), S(
            1)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(Rational(3, 2), Rational(-1, 2)))) == \
        -sqrt(6)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, S.Half)) )/6 - \
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 0), (1, 3, Rational(3, 2))) )/2 + \
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(3, 2))) )/30 + \
        sqrt(30)*JzKetCoupled(Rational(5, 2), -S(
            1)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(Rational(3, 2), Rational(-3, 2)))) == \
        -sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 0), (1, 3, Rational(3, 2))) )/2 + \
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(3, 2))) )/10 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, S.Half, S(3)
             /2), ((1, 2, 1), (1, 3, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(Rational(3, 2), Rational(3, 2)))) == \
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, S.Half)) )/2 - \
        sqrt(10)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(3, 2))) )/5 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), S(
            1)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(Rational(3, 2), S.Half))) == \
        sqrt(6)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, S.Half)) )/6 - \
        2*sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(3, 2))) )/15 + \
        sqrt(30)*JzKetCoupled(Rational(5, 2), -S(
            1)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(Rational(3, 2), Rational(-1, 2)))) == \
        -sqrt(10)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(3, 2))) )/5 + \
        sqrt(15)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, S.Half, S(
            3)/2), ((1, 2, 1), (1, 3, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(Rational(3, 2), Rational(-3, 2)))) == \
        JzKetCoupled(Rational(5, 2), -S(
            5)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(5, 2))) )
    # Couple j1 to j3
    # j1=1/2, j2=1/2, j3=1/2
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)) ) == \
        JzKetCoupled(Rational(3, 2), S(
            3)/2, (S.Half, S.Half, S.Half), ((1, 3, 1), (1, 2, Rational(3, 2))) )
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)) ) == \
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half), ((1, 3, 0), (1, 2, S.Half)) )/2 - \
        sqrt(6)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half), ((1, 3, 1), (1, 2, S.Half)) )/6 + \
        sqrt(3)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.One/
             2), ((1, 3, 1), (1, 2, Rational(3, 2))) )/3
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)) ) == \
        sqrt(6)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half), ((1, 3, 1), (1, 2, S.Half)) )/3 + \
        sqrt(3)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.One/
             2), ((1, 3, 1), (1, 2, Rational(3, 2))) )/3
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)) ) == \
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half), ((1, 3, 0), (1, 2, S.Half)) )/2 + \
        sqrt(6)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half), ((1, 3, 1), (1, 2, S.Half)) )/6 + \
        sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.One
             /2), ((1, 3, 1), (1, 2, Rational(3, 2))) )/3
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)) ) == \
        -sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half), ((1, 3, 0), (1, 2, S.Half)) )/2 - \
        sqrt(6)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half), ((1, 3, 1), (1, 2, S.Half)) )/6 + \
        sqrt(3)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.One/
             2), ((1, 3, 1), (1, 2, Rational(3, 2))) )/3
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)) ) == \
        -sqrt(6)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half), ((1, 3, 1), (1, 2, S.Half)) )/3 + \
        sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.One
             /2), ((1, 3, 1), (1, 2, Rational(3, 2))) )/3
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)) ) == \
        -sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half), ((1, 3, 0), (1, 2, S.Half)) )/2 + \
        sqrt(6)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half), ((1, 3, 1), (1, 2, S.Half)) )/6 + \
        sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.One
             /2), ((1, 3, 1), (1, 2, Rational(3, 2))) )/3
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)) ) == \
        JzKetCoupled(Rational(3, 2), -S(
            3)/2, (S.Half, S.Half, S.Half), ((1, 3, 1), (1, 2, Rational(3, 2))) )
    # j1=1/2, j2=1/2, j3=1
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        JzKetCoupled(2, 2, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 2)) )
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        sqrt(3)*JzKetCoupled(1, 1, (S.Half, S.Half, 1), ((1, 3, S.Half), (1, 2, 1)) )/3 - \
        sqrt(6)*JzKetCoupled(1, 1, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 1)) )/6 + \
        sqrt(2)*JzKetCoupled(
            2, 1, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 2)) )/2
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        -sqrt(3)*JzKetCoupled(0, 0, (S.Half, S.Half, 1), ((1, 3, S.Half), (1, 2, 0)) )/3 + \
        sqrt(3)*JzKetCoupled(1, 0, (S.Half, S.Half, 1), ((1, 3, S.Half), (1, 2, 1)) )/3 - \
        sqrt(6)*JzKetCoupled(1, 0, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 1)) )/6 + \
        sqrt(6)*JzKetCoupled(
            2, 0, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 2)) )/6
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        sqrt(3)*JzKetCoupled(1, 1, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 1)) )/2 + \
        JzKetCoupled(2, 1, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 2)) )/2
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        sqrt(6)*JzKetCoupled(0, 0, (S.Half, S.Half, 1), ((1, 3, S.Half), (1, 2, 0)) )/6 + \
        sqrt(6)*JzKetCoupled(1, 0, (S.Half, S.Half, 1), ((1, 3, S.Half), (1, 2, 1)) )/6 + \
        sqrt(3)*JzKetCoupled(1, 0, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 1)) )/3 + \
        sqrt(3)*JzKetCoupled(
            2, 0, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 2)) )/3
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        sqrt(6)*JzKetCoupled(1, -1, (S.Half, S.Half, 1), ((1, 3, S.Half), (1, 2, 1)) )/3 + \
        sqrt(3)*JzKetCoupled(1, -1, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 1)) )/6 + \
        JzKetCoupled(
            2, -1, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 2)) )/2
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        -sqrt(6)*JzKetCoupled(1, 1, (S.Half, S.Half, 1), ((1, 3, S.Half), (1, 2, 1)) )/3 - \
        sqrt(3)*JzKetCoupled(1, 1, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 1)) )/6 + \
        JzKetCoupled(2, 1, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 2)) )/2
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        sqrt(6)*JzKetCoupled(0, 0, (S.Half, S.Half, 1), ((1, 3, S.Half), (1, 2, 0)) )/6 - \
        sqrt(6)*JzKetCoupled(1, 0, (S.Half, S.Half, 1), ((1, 3, S.Half), (1, 2, 1)) )/6 - \
        sqrt(3)*JzKetCoupled(1, 0, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 1)) )/3 + \
        sqrt(3)*JzKetCoupled(
            2, 0, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 2)) )/3
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        -sqrt(3)*JzKetCoupled(1, -1, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 1)) )/2 + \
        JzKetCoupled(
            2, -1, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 2)) )/2
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        -sqrt(3)*JzKetCoupled(0, 0, (S.Half, S.Half, 1), ((1, 3, S.Half), (1, 2, 0)) )/3 - \
        sqrt(3)*JzKetCoupled(1, 0, (S.Half, S.Half, 1), ((1, 3, S.Half), (1, 2, 1)) )/3 + \
        sqrt(6)*JzKetCoupled(1, 0, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 1)) )/6 + \
        sqrt(6)*JzKetCoupled(
            2, 0, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 2)) )/6
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        -sqrt(3)*JzKetCoupled(1, -1, (S.Half, S.Half, 1), ((1, 3, S.Half), (1, 2, 1)) )/3 + \
        sqrt(6)*JzKetCoupled(1, -1, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 1)) )/6 + \
        sqrt(2)*JzKetCoupled(
            2, -1, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 2)) )/2
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        JzKetCoupled(2, -2, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 2)) )
    # j 1=1/2, j 2=1, j 3=1
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        JzKetCoupled(
            Rational(5, 2), Rational(5, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(5, 2))) )
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, Rational(3, 2))) )/3 - \
        2*sqrt(15)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(3, 2))) )/15 + \
        sqrt(10)*JzKetCoupled(S(
            5)/2, Rational(3, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        -2*JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, S.Half)) )/3 + \
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, S.Half)) )/6 + \
        sqrt(2)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, Rational(3, 2))) )/3 - \
        2*sqrt(10)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(3, 2))) )/15 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, 1, 1), ((1,
             3, Rational(3, 2)), (1, 2, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        sqrt(15)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(3, 2))) )/5 + \
        sqrt(10)*JzKetCoupled(S(
            5)/2, Rational(3, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, S.Half)) )/3 - \
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, S.Half)) )/3 + \
        sqrt(2)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, Rational(3, 2))) )/3 + \
        sqrt(10)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(3, 2))) )/15 + \
        sqrt(10)*JzKetCoupled(S(
            5)/2, S.Half, (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        -sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, S.Half)) )/3 - \
        JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, S.Half)) )/3 + \
        2*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, Rational(3, 2))) )/3 - \
        sqrt(5)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, 1, 1), ((1,
             3, Rational(3, 2)), (1, 2, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, S.Half)) )/2 + \
        sqrt(10)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(3, 2))) )/5 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, 1, 1), ((1,
             3, Rational(3, 2)), (1, 2, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, S.Half)) )/3 + \
        JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, S.Half)) )/3 + \
        JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, Rational(3, 2))) )/3 + \
        4*sqrt(5)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, 1, 1), ((1,
             3, Rational(3, 2)), (1, 2, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        sqrt(6)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, Rational(3, 2))) )/3 + \
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, 1, 1), ((1,
             3, Rational(3, 2)), (1, 2, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        -sqrt(6)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, Rational(3, 2))) )/3 - \
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(S(
            5)/2, Rational(3, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, S.Half)) )/3 + \
        JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, S.Half)) )/3 - \
        JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, Rational(3, 2))) )/3 - \
        4*sqrt(5)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(S(
            5)/2, S.Half, (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, S.Half)) )/2 - \
        sqrt(10)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(3, 2))) )/5 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, 1, 1), ((1,
             3, Rational(3, 2)), (1, 2, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        -sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, S.Half)) )/3 - \
        JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, S.Half)) )/3 - \
        2*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, Rational(3, 2))) )/3 + \
        sqrt(5)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(S(
            5)/2, S.Half, (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, S.Half)) )/3 - \
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, S.Half)) )/3 - \
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, Rational(3, 2))) )/3 - \
        sqrt(10)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(3, 2))) )/15 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, 1, 1), ((1,
             3, Rational(3, 2)), (1, 2, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        -sqrt(15)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(3, 2))) )/5 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, 1, 1), ((1,
             3, Rational(3, 2)), (1, 2, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        -2*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, S.Half)) )/3 + \
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, S.Half)) )/6 - \
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, Rational(3, 2))) )/3 + \
        2*sqrt(10)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(3, 2))) )/15 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, 1, 1), ((1,
             3, Rational(3, 2)), (1, 2, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        -sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, Rational(3, 2))) )/3 + \
        2*sqrt(15)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(3, 2))) )/15 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, 1, 1), ((1,
             3, Rational(3, 2)), (1, 2, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        JzKetCoupled(S(
            5)/2, Rational(-5, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(5, 2))) )
    # j1=1, 1, 1
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        JzKetCoupled(3, 3, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        sqrt(2)*JzKetCoupled(2, 2, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/2 - \
        sqrt(6)*JzKetCoupled(2, 2, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/6 + \
        sqrt(3)*JzKetCoupled(3, 2, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/3
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        sqrt(3)*JzKetCoupled(1, 1, (1, 1, 1), ((1, 3, 0), (1, 2, 1)) )/3 - \
        JzKetCoupled(1, 1, (1, 1, 1), ((1, 3, 1), (1, 2, 1)) )/2 + \
        sqrt(15)*JzKetCoupled(1, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/30 + \
        JzKetCoupled(2, 1, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/2 - \
        sqrt(3)*JzKetCoupled(2, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/6 + \
        sqrt(15)*JzKetCoupled(3, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/15
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        sqrt(6)*JzKetCoupled(2, 2, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/3 + \
        sqrt(3)*JzKetCoupled(3, 2, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/3
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        JzKetCoupled(1, 1, (1, 1, 1), ((1, 3, 1), (1, 2, 1)) )/2 - \
        sqrt(15)*JzKetCoupled(1, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/10 + \
        JzKetCoupled(2, 1, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/2 + \
        sqrt(3)*JzKetCoupled(2, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/6 + \
        2*sqrt(15)*JzKetCoupled(3, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/15
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        -sqrt(6)*JzKetCoupled(0, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 0)) )/6 + \
        sqrt(3)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 3, 0), (1, 2, 1)) )/3 - \
        sqrt(15)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/15 + \
        sqrt(3)*JzKetCoupled(2, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/3 + \
        sqrt(10)*JzKetCoupled(3, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/10
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        sqrt(15)*JzKetCoupled(1, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/5 + \
        sqrt(3)*JzKetCoupled(2, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/3 + \
        sqrt(15)*JzKetCoupled(3, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/15
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        sqrt(6)*JzKetCoupled(0, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 0)) )/6 + \
        JzKetCoupled(1, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 1)) )/2 + \
        sqrt(15)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/10 + \
        sqrt(3)*JzKetCoupled(2, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/6 + \
        JzKetCoupled(2, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/2 + \
        sqrt(10)*JzKetCoupled(3, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/10
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        sqrt(3)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 3, 0), (1, 2, 1)) )/3 + \
        JzKetCoupled(1, -1, (1, 1, 1), ((1, 3, 1), (1, 2, 1)) )/2 + \
        sqrt(15)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/30 + \
        JzKetCoupled(2, -1, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/2 + \
        sqrt(3)*JzKetCoupled(2, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/6 + \
        sqrt(15)*JzKetCoupled(3, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/15
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        -sqrt(2)*JzKetCoupled(2, 2, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/2 - \
        sqrt(6)*JzKetCoupled(2, 2, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/6 + \
        sqrt(3)*JzKetCoupled(3, 2, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/3
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        -sqrt(3)*JzKetCoupled(1, 1, (1, 1, 1), ((1, 3, 0), (1, 2, 1)) )/3 + \
        sqrt(15)*JzKetCoupled(1, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/15 - \
        sqrt(3)*JzKetCoupled(2, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/3 + \
        2*sqrt(15)*JzKetCoupled(3, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/15
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        sqrt(6)*JzKetCoupled(0, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 0)) )/6 - \
        JzKetCoupled(1, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 1)) )/2 + \
        sqrt(15)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/10 + \
        sqrt(3)*JzKetCoupled(2, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/6 - \
        JzKetCoupled(2, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/2 + \
        sqrt(10)*JzKetCoupled(3, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/10
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        -JzKetCoupled(1, 1, (1, 1, 1), ((1, 3, 1), (1, 2, 1)) )/2 - \
        sqrt(15)*JzKetCoupled(1, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/10 - \
        JzKetCoupled(2, 1, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/2 + \
        sqrt(3)*JzKetCoupled(2, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/6 + \
        2*sqrt(15)*JzKetCoupled(3, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/15
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        -sqrt(3)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 3, 0), (1, 2, 1)) )/3 - \
        2*sqrt(15)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/15 + \
        sqrt(10)*JzKetCoupled(3, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/5
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        -JzKetCoupled(1, -1, (1, 1, 1), ((1, 3, 1), (1, 2, 1)) )/2 - \
        sqrt(15)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/10 + \
        JzKetCoupled(2, -1, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/2 - \
        sqrt(3)*JzKetCoupled(2, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/6 + \
        2*sqrt(15)*JzKetCoupled(3, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/15
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, -1), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        -sqrt(6)*JzKetCoupled(0, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 0)) )/6 - \
        JzKetCoupled(1, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 1)) )/2 + \
        sqrt(15)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/10 - \
        sqrt(3)*JzKetCoupled(2, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/6 + \
        JzKetCoupled(2, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/2 + \
        sqrt(10)*JzKetCoupled(3, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/10
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, -1), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        -sqrt(3)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 3, 0), (1, 2, 1)) )/3 + \
        sqrt(15)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/15 + \
        sqrt(3)*JzKetCoupled(2, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/3 + \
        2*sqrt(15)*JzKetCoupled(3, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/15
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, -1), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        sqrt(2)*JzKetCoupled(2, -2, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/2 + \
        sqrt(6)*JzKetCoupled(2, -2, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/6 + \
        sqrt(3)*JzKetCoupled(3, -2, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/3
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        sqrt(3)*JzKetCoupled(1, 1, (1, 1, 1), ((1, 3, 0), (1, 2, 1)) )/3 + \
        JzKetCoupled(1, 1, (1, 1, 1), ((1, 3, 1), (1, 2, 1)) )/2 + \
        sqrt(15)*JzKetCoupled(1, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/30 - \
        JzKetCoupled(2, 1, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/2 - \
        sqrt(3)*JzKetCoupled(2, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/6 + \
        sqrt(15)*JzKetCoupled(3, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/15
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        -sqrt(6)*JzKetCoupled(0, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 0)) )/6 + \
        JzKetCoupled(1, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 1)) )/2 + \
        sqrt(15)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/10 - \
        sqrt(3)*JzKetCoupled(2, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/6 - \
        JzKetCoupled(2, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/2 + \
        sqrt(10)*JzKetCoupled(3, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/10
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        sqrt(15)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/5 - \
        sqrt(3)*JzKetCoupled(2, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/3 + \
        sqrt(15)*JzKetCoupled(3, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/15
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, 0), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        sqrt(6)*JzKetCoupled(0, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 0)) )/6 + \
        sqrt(3)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 3, 0), (1, 2, 1)) )/3 - \
        sqrt(15)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/15 - \
        sqrt(3)*JzKetCoupled(2, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/3 + \
        sqrt(10)*JzKetCoupled(3, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/10
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, 0), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        JzKetCoupled(1, -1, (1, 1, 1), ((1, 3, 1), (1, 2, 1)) )/2 - \
        sqrt(15)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/10 - \
        JzKetCoupled(2, -1, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/2 - \
        sqrt(3)*JzKetCoupled(2, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/6 + \
        2*sqrt(15)*JzKetCoupled(3, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/15
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, 0), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        -sqrt(6)*JzKetCoupled(2, -2, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/3 + \
        sqrt(3)*JzKetCoupled(3, -2, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/3
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, -1), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        sqrt(3)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 3, 0), (1, 2, 1)) )/3 - \
        JzKetCoupled(1, -1, (1, 1, 1), ((1, 3, 1), (1, 2, 1)) )/2 + \
        sqrt(15)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/30 - \
        JzKetCoupled(2, -1, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/2 + \
        sqrt(3)*JzKetCoupled(2, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/6 + \
        sqrt(15)*JzKetCoupled(3, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/15
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, -1), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        -sqrt(2)*JzKetCoupled(2, -2, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/2 + \
        sqrt(6)*JzKetCoupled(2, -2, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/6 + \
        sqrt(3)*JzKetCoupled(3, -2, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/3
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, -1), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        JzKetCoupled(3, -3, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )
    # j1=1/2, j2=1/2, j3=3/2
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(Rational(3, 2), Rational(3, 2))), ((1, 3), (1, 2)) ) == \
        JzKetCoupled(Rational(5, 2), S(
            5)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(5, 2))) )
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(Rational(3, 2), S.Half)), ((1, 3), (1, 2)) ) == \
        JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, Rational(3, 2))) )/2 - \
        sqrt(15)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(3, 2))) )/10 + \
        sqrt(15)*JzKetCoupled(Rational(5, 2), Rational(3, 2), (S.Half, S.Half, S(3)
             /2), ((1, 3, 2), (1, 2, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(Rational(3, 2), Rational(-1, 2))), ((1, 3), (1, 2)) ) == \
        -sqrt(6)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, S.Half)) )/6 + \
        sqrt(3)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, Rational(3, 2))) )/3 - \
        sqrt(5)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(3, 2))) )/5 + \
        sqrt(30)*JzKetCoupled(Rational(5, 2), S(
            1)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(Rational(3, 2), Rational(-3, 2))), ((1, 3), (1, 2)) ) == \
        -sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, S.Half)) )/2 + \
        JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, Rational(3, 2))) )/2 - \
        sqrt(15)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(3, 2))) )/10 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), -S(
            1)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(Rational(3, 2), Rational(3, 2))), ((1, 3), (1, 2)) ) == \
        2*sqrt(5)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(3, 2))) )/5 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(3, 2), (S.Half, S.Half, S(3)/
             2), ((1, 3, 2), (1, 2, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(Rational(3, 2), S.Half)), ((1, 3), (1, 2)) ) == \
        sqrt(6)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, S.Half)) )/6 + \
        sqrt(3)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, Rational(3, 2))) )/6 + \
        3*sqrt(5)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(3, 2))) )/10 + \
        sqrt(30)*JzKetCoupled(Rational(5, 2), S(
            1)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(Rational(3, 2), Rational(-1, 2))), ((1, 3), (1, 2)) ) == \
        sqrt(6)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, S.Half)) )/6 + \
        sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, Rational(3, 2))) )/3 + \
        sqrt(5)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(3, 2))) )/5 + \
        sqrt(30)*JzKetCoupled(Rational(5, 2), -S(
            1)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(Rational(3, 2), Rational(-3, 2))), ((1, 3), (1, 2)) ) == \
        sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, Rational(3, 2))) )/2 + \
        sqrt(5)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(3, 2))) )/10 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, S.Half, S(3)
             /2), ((1, 3, 2), (1, 2, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(Rational(3, 2), Rational(3, 2))), ((1, 3), (1, 2)) ) == \
        -sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, Rational(3, 2))) )/2 - \
        sqrt(5)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(3, 2))) )/10 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(3, 2), (S.Half, S.Half, S(3)/
             2), ((1, 3, 2), (1, 2, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(Rational(3, 2), S.Half)), ((1, 3), (1, 2)) ) == \
        sqrt(6)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, S.Half)) )/6 - \
        sqrt(3)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, Rational(3, 2))) )/3 - \
        sqrt(5)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(3, 2))) )/5 + \
        sqrt(30)*JzKetCoupled(Rational(5, 2), S(
            1)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(Rational(3, 2), Rational(-1, 2))), ((1, 3), (1, 2)) ) == \
        sqrt(6)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, S.Half)) )/6 - \
        sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, Rational(3, 2))) )/6 - \
        3*sqrt(5)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(3, 2))) )/10 + \
        sqrt(30)*JzKetCoupled(Rational(5, 2), -S(
            1)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(Rational(3, 2), Rational(-3, 2))), ((1, 3), (1, 2)) ) == \
        -2*sqrt(5)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(3, 2))) )/5 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, S.Half, S(3)
             /2), ((1, 3, 2), (1, 2, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(Rational(3, 2), Rational(3, 2))), ((1, 3), (1, 2)) ) == \
        -sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, S.Half)) )/2 - \
        JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, Rational(3, 2))) )/2 + \
        sqrt(15)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(3, 2))) )/10 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), S(
            1)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(Rational(3, 2), S.Half)), ((1, 3), (1, 2)) ) == \
        -sqrt(6)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, S.Half)) )/6 - \
        sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, Rational(3, 2))) )/3 + \
        sqrt(5)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(3, 2))) )/5 + \
        sqrt(30)*JzKetCoupled(Rational(5, 2), -S(
            1)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(Rational(3, 2), Rational(-1, 2))), ((1, 3), (1, 2)) ) == \
        -JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, Rational(3, 2))) )/2 + \
        sqrt(15)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(3, 2))) )/10 + \
        sqrt(15)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, S.Half, S(
            3)/2), ((1, 3, 2), (1, 2, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(Rational(3, 2), Rational(-3, 2))), ((1, 3), (1, 2)) ) == \
        JzKetCoupled(Rational(5, 2), -S(
            5)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(5, 2))) )


def test_couple_4_states_numerical():
    # Default coupling
    # j1=1/2, j2=1/2, j3=1/2, j4=1/2
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half))) == \
        JzKetCoupled(2, 2, (S.Half, S(
            1)/2, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 2)) )
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)))) == \
        sqrt(3)*JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 1)) )/2 + \
        JzKetCoupled(2, 1, (S.Half, S(
            1)/2, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 2)) )/2
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half))) == \
        sqrt(6)*JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half), (1, 4, 1)) )/3 - \
        sqrt(3)*JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 1)) )/6 + \
        JzKetCoupled(2, 1, (S.Half, S(
            1)/2, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 2)) )/2
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)))) == \
        sqrt(3)*JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half), (1, 4, 0)) )/3 + \
        sqrt(3)*JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half), (1, 4, 1)) )/3 + \
        sqrt(6)*JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 1)) )/6 + \
        sqrt(6)*JzKetCoupled(2, 0, (S.Half, S(
            1)/2, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 2)) )/6
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half))) == \
        sqrt(2)*JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (1, 3, S.Half), (1, 4, 1)) )/2 - \
        sqrt(6)*JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half), (1, 4, 1)) )/6 - \
        sqrt(3)*JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 1)) )/6 + \
        JzKetCoupled(2, 1, (S.Half, S(
            1)/2, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 2)) )/2
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)),
                        JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)))) == \
        JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half),
                ((1, 2, 0), (1, 3, S.Half), (1, 4, 0)))/2 - \
        sqrt(3)*JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half),
                ((1, 2, 1), (1, 3, S.Half), (1, 4, 0)))/6 + \
        JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half),
                ((1, 2, 0), (1, 3, S.Half), (1, 4, 1)))/2 - \
        sqrt(3)*JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half),
                ((1, 2, 1), (1, 3, S.Half), (1, 4, 1)))/6 + \
        sqrt(6)*JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half),
                ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 1)))/6 + \
        sqrt(6)*JzKetCoupled(2, 0, (S.Half, S.Half, S.Half, S.Half),
                ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 2)))/6
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half))) == \
        -JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (1, 3, S.Half), (1, 4, 0)) )/2 - \
        sqrt(3)*JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half), (1, 4, 0)) )/6 + \
        JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (1, 3, S.Half), (1, 4, 1)) )/2 + \
        sqrt(3)*JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half), (1, 4, 1)) )/6 - \
        sqrt(6)*JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 1)) )/6 + \
        sqrt(6)*JzKetCoupled(2, 0, (S.Half, S(
            1)/2, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 2)) )/6
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)))) == \
        sqrt(2)*JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (1, 3, S.Half), (1, 4, 1)) )/2 + \
        sqrt(6)*JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half), (1, 4, 1)) )/6 + \
        sqrt(3)*JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 1)) )/6 + \
        JzKetCoupled(2, -1, (S.Half, S(
            1)/2, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 2)) )/2
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half))) == \
        -sqrt(2)*JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (1, 3, S.Half), (1, 4, 1)) )/2 - \
        sqrt(6)*JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half), (1, 4, 1)) )/6 - \
        sqrt(3)*JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 1)) )/6 + \
        JzKetCoupled(2, 1, (S.Half, S(
            1)/2, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 2)) )/2
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)))) == \
        -JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (1, 3, S.Half), (1, 4, 0)) )/2 - \
        sqrt(3)*JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half), (1, 4, 0)) )/6 - \
        JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (1, 3, S.Half), (1, 4, 1)) )/2 - \
        sqrt(3)*JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half), (1, 4, 1)) )/6 + \
        sqrt(6)*JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 1)) )/6 + \
        sqrt(6)*JzKetCoupled(2, 0, (S.Half, S(
            1)/2, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 2)) )/6
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half))) == \
        JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (1, 3, S.Half), (1, 4, 0)) )/2 - \
        sqrt(3)*JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half), (1, 4, 0)) )/6 - \
        JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (1, 3, S.Half), (1, 4, 1)) )/2 + \
        sqrt(3)*JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half), (1, 4, 1)) )/6 - \
        sqrt(6)*JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 1)) )/6 + \
        sqrt(6)*JzKetCoupled(2, 0, (S.Half, S(
            1)/2, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 2)) )/6
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)))) == \
        -sqrt(2)*JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (1, 3, S.Half), (1, 4, 1)) )/2 + \
        sqrt(6)*JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half), (1, 4, 1)) )/6 + \
        sqrt(3)*JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 1)) )/6 + \
        JzKetCoupled(2, -1, (S.Half, S(
            1)/2, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 2)) )/2
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half))) == \
        sqrt(3)*JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half), (1, 4, 0)) )/3 - \
        sqrt(3)*JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half), (1, 4, 1)) )/3 - \
        sqrt(6)*JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 1)) )/6 + \
        sqrt(6)*JzKetCoupled(2, 0, (S.Half, S(
            1)/2, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 2)) )/6
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)))) == \
        -sqrt(6)*JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half), (1, 4, 1)) )/3 + \
        sqrt(3)*JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 1)) )/6 + \
        JzKetCoupled(2, -1, (S.Half, S(
            1)/2, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 2)) )/2
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half))) == \
        -sqrt(3)*JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 1)) )/2 + \
        JzKetCoupled(2, -1, (S.Half, S(
            1)/2, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 2)) )/2
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)))) == \
        JzKetCoupled(2, -2, (S.Half, S(
            1)/2, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 2)) )
    # j1=S.Half, S.Half, S.Half, 1
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1))) == \
        JzKetCoupled(Rational(5, 2), Rational(5, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0))) == \
        sqrt(15)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/5 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(3, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1))) == \
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, S.Half)) )/2 + \
        sqrt(10)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/5 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1))) == \
        sqrt(6)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/3 - \
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(3, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0))) == \
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, S.Half)) )/3 - \
        JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, S.Half)) )/3 + \
        2*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/3 + \
        sqrt(5)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1))) == \
        2*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, S.Half)) )/3 + \
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, S.Half)) )/6 + \
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/3 + \
        2*sqrt(10)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/15 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1))) == \
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/2 - \
        sqrt(6)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/6 - \
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(3, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0))) == \
        sqrt(6)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, S.Half)) )/6 - \
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, S.Half)) )/6 - \
        JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, S.Half)) )/3 + \
        sqrt(3)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/3 - \
        JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/3 + \
        sqrt(5)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1))) == \
        sqrt(3)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, S.Half)) )/3 - \
        JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, S.Half)) )/3 + \
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, S.Half)) )/6 + \
        sqrt(6)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/6 - \
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/6 + \
        2*sqrt(10)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/15 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1))) == \
        -sqrt(3)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, S.Half)) )/3 - \
        JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, S.Half)) )/3 + \
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, S.Half)) )/6 + \
        sqrt(6)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/6 + \
        sqrt(2)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/6 - \
        2*sqrt(10)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/15 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0))) == \
        -sqrt(6)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, S.Half)) )/6 - \
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, S.Half)) )/6 - \
        JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, S.Half)) )/3 + \
        sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/3 + \
        JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/3 - \
        sqrt(5)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1))) == \
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/2 + \
        sqrt(6)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/6 + \
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1))) == \
        -sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/2 - \
        sqrt(6)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/6 - \
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(3, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0))) == \
        -sqrt(6)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, S.Half)) )/6 - \
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, S.Half)) )/6 - \
        JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, S.Half)) )/3 - \
        sqrt(3)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/3 - \
        JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/3 + \
        sqrt(5)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1))) == \
        -sqrt(3)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, S.Half)) )/3 - \
        JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, S.Half)) )/3 + \
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, S.Half)) )/6 - \
        sqrt(6)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/6 - \
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/6 + \
        2*sqrt(10)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/15 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1))) == \
        sqrt(3)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, S.Half)) )/3 - \
        JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, S.Half)) )/3 + \
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, S.Half)) )/6 - \
        sqrt(6)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/6 + \
        sqrt(2)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/6 - \
        2*sqrt(10)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/15 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0))) == \
        sqrt(6)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, S.Half)) )/6 - \
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, S.Half)) )/6 - \
        JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, S.Half)) )/3 - \
        sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/3 + \
        JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/3 - \
        sqrt(5)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1))) == \
        -sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/2 + \
        sqrt(6)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/6 + \
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1))) == \
        2*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, S.Half)) )/3 + \
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, S.Half)) )/6 - \
        sqrt(2)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/3 - \
        2*sqrt(10)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/15 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0))) == \
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, S.Half)) )/3 - \
        JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, S.Half)) )/3 - \
        2*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/3 - \
        sqrt(5)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1))) == \
        -sqrt(6)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/3 + \
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1))) == \
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, S.Half)) )/2 - \
        sqrt(10)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/5 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0))) == \
        -sqrt(15)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/5 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1))) == \
        JzKetCoupled(Rational(5, 2), Rational(-5, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )
    # Couple j1 to j2, j3 to j4
    # j1=1/2, j2=1/2, j3=1/2, j4=1/2
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)), ((1, 2), (3, 4), (1, 3)) ) == \
        JzKetCoupled(2, 2, (S(
            1)/2, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 2)) )
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), ((1, 2), (3, 4), (1, 3)) ) == \
        sqrt(2)*JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 0), (1, 3, 1)) )/2 + \
        JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 1)) )/2 + \
        JzKetCoupled(2, 1, (S.Half, S(
            1)/2, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 2)) )/2
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)), ((1, 2), (3, 4), (1, 3)) ) == \
        -sqrt(2)*JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 0), (1, 3, 1)) )/2 + \
        JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 1)) )/2 + \
        JzKetCoupled(2, 1, (S.Half, S(
            1)/2, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 2)) )/2
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), ((1, 2), (3, 4), (1, 3)) ) == \
        sqrt(3)*JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 0)) )/3 + \
        sqrt(2)*JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 1)) )/2 + \
        sqrt(6)*JzKetCoupled(2, 0, (S.Half, S.Half, S.Half, S.One/
             2), ((1, 2, 1), (3, 4, 1), (1, 3, 2)) )/6
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)), ((1, 2), (3, 4), (1, 3)) ) == \
        sqrt(2)*JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (3, 4, 1), (1, 3, 1)) )/2 - \
        JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 1)) )/2 + \
        JzKetCoupled(2, 1, (S.Half, S(
            1)/2, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 2)) )/2
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), ((1, 2), (3, 4), (1, 3)) ) == \
        JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (3, 4, 0), (1, 3, 0)) )/2 - \
        sqrt(3)*JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 0)) )/6 + \
        JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (3, 4, 1), (1, 3, 1)) )/2 + \
        JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 0), (1, 3, 1)) )/2 + \
        sqrt(6)*JzKetCoupled(2, 0, (S.Half, S.Half, S.Half, S.One/
             2), ((1, 2, 1), (3, 4, 1), (1, 3, 2)) )/6
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)), ((1, 2), (3, 4), (1, 3)) ) == \
        -JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (3, 4, 0), (1, 3, 0)) )/2 - \
        sqrt(3)*JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 0)) )/6 + \
        JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (3, 4, 1), (1, 3, 1)) )/2 - \
        JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 0), (1, 3, 1)) )/2 + \
        sqrt(6)*JzKetCoupled(2, 0, (S.Half, S.Half, S.Half, S.One/
             2), ((1, 2, 1), (3, 4, 1), (1, 3, 2)) )/6
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), ((1, 2), (3, 4), (1, 3)) ) == \
        sqrt(2)*JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (3, 4, 1), (1, 3, 1)) )/2 + \
        JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 1)) )/2 + \
        JzKetCoupled(2, -1, (S.Half, S(
            1)/2, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 2)) )/2
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)), ((1, 2), (3, 4), (1, 3)) ) == \
        -sqrt(2)*JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (3, 4, 1), (1, 3, 1)) )/2 - \
        JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 1)) )/2 + \
        JzKetCoupled(2, 1, (S.Half, S(
            1)/2, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 2)) )/2
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), ((1, 2), (3, 4), (1, 3)) ) == \
        -JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (3, 4, 0), (1, 3, 0)) )/2 - \
        sqrt(3)*JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 0)) )/6 - \
        JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (3, 4, 1), (1, 3, 1)) )/2 + \
        JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 0), (1, 3, 1)) )/2 + \
        sqrt(6)*JzKetCoupled(2, 0, (S.Half, S.Half, S.Half, S.One/
             2), ((1, 2, 1), (3, 4, 1), (1, 3, 2)) )/6
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)), ((1, 2), (3, 4), (1, 3)) ) == \
        JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (3, 4, 0), (1, 3, 0)) )/2 - \
        sqrt(3)*JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 0)) )/6 - \
        JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (3, 4, 1), (1, 3, 1)) )/2 - \
        JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 0), (1, 3, 1)) )/2 + \
        sqrt(6)*JzKetCoupled(2, 0, (S.Half, S.Half, S.Half, S.One/
             2), ((1, 2, 1), (3, 4, 1), (1, 3, 2)) )/6
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), ((1, 2), (3, 4), (1, 3)) ) == \
        -sqrt(2)*JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (3, 4, 1), (1, 3, 1)) )/2 + \
        JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 1)) )/2 + \
        JzKetCoupled(2, -1, (S.Half, S(
            1)/2, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 2)) )/2
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)), ((1, 2), (3, 4), (1, 3)) ) == \
        sqrt(3)*JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 0)) )/3 - \
        sqrt(2)*JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 1)) )/2 + \
        sqrt(6)*JzKetCoupled(2, 0, (S.Half, S.Half, S.Half, S.One/
             2), ((1, 2, 1), (3, 4, 1), (1, 3, 2)) )/6
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), ((1, 2), (3, 4), (1, 3)) ) == \
        sqrt(2)*JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 0), (1, 3, 1)) )/2 - \
        JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 1)) )/2 + \
        JzKetCoupled(2, -1, (S.Half, S(
            1)/2, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 2)) )/2
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)), ((1, 2), (3, 4), (1, 3)) ) == \
        -sqrt(2)*JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 0), (1, 3, 1)) )/2 - \
        JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 1)) )/2 + \
        JzKetCoupled(2, -1, (S.Half, S(
            1)/2, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 2)) )/2
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), ((1, 2), (3, 4), (1, 3)) ) == \
        JzKetCoupled(2, -2, (S(
            1)/2, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 2)) )
    # j1=S.Half, S.Half, S.Half, 1
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1)), ((1, 2), (3, 4), (1, 3)) ) == \
        JzKetCoupled(Rational(5, 2), Rational(5, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0)), ((1, 2), (3, 4), (1, 3)) ) == \
        sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, Rational(3, 2))) )/3 + \
        2*sqrt(15)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(3, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1)), ((1, 2), (3, 4), (1, 3)) ) == \
        2*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, S.Half)) )/3 + \
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, S.Half)) )/6 + \
        sqrt(2)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, Rational(3, 2))) )/3 + \
        2*sqrt(10)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1)), ((1, 2), (3, 4), (1, 3)) ) == \
        -sqrt(6)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, Rational(3, 2))) )/3 + \
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(3, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0)), ((1, 2), (3, 4), (1, 3)) ) == \
        -sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, S.Half)) )/3 + \
        JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, S.Half)) )/3 - \
        JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, Rational(3, 2))) )/3 + \
        4*sqrt(5)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1)), ((1, 2), (3, 4), (1, 3)) ) == \
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, S.Half)) )/2 + \
        sqrt(10)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/5 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1)), ((1, 2), (3, 4), (1, 3)) ) == \
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/2 - \
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/10 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(3, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0)), ((1, 2), (3, 4), (1, 3)) ) == \
        sqrt(6)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, S.Half), (1, 3, S.Half)) )/6 - \
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, S.Half)) )/6 - \
        JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, S.Half)) )/3 + \
        sqrt(3)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/3 + \
        JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, Rational(3, 2))) )/3 - \
        sqrt(5)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1)), ((1, 2), (3, 4), (1, 3)) ) == \
        sqrt(3)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, S.Half), (1, 3, S.Half)) )/3 + \
        JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, S.Half)) )/3 - \
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, S.Half)) )/6 + \
        sqrt(6)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/6 + \
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, Rational(3, 2))) )/3 + \
        sqrt(10)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/30 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1)), ((1, 2), (3, 4), (1, 3)) ) == \
        -sqrt(3)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, S.Half), (1, 3, S.Half)) )/3 + \
        JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, S.Half)) )/3 - \
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, S.Half)) )/6 + \
        sqrt(6)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/6 - \
        sqrt(2)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, Rational(3, 2))) )/3 - \
        sqrt(10)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/30 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0)), ((1, 2), (3, 4), (1, 3)) ) == \
        -sqrt(6)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, S.Half), (1, 3, S.Half)) )/6 - \
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, S.Half)) )/6 - \
        JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, S.Half)) )/3 + \
        sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/3 - \
        JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, Rational(3, 2))) )/3 + \
        sqrt(5)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1)), ((1, 2), (3, 4), (1, 3)) ) == \
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/2 + \
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/10 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1)), ((1, 2), (3, 4), (1, 3)) ) == \
        -sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/2 - \
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/10 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(3, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0)), ((1, 2), (3, 4), (1, 3)) ) == \
        -sqrt(6)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, S.Half), (1, 3, S.Half)) )/6 - \
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, S.Half)) )/6 - \
        JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, S.Half)) )/3 - \
        sqrt(3)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/3 + \
        JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, Rational(3, 2))) )/3 - \
        sqrt(5)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1)), ((1, 2), (3, 4), (1, 3)) ) == \
        -sqrt(3)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, S.Half), (1, 3, S.Half)) )/3 + \
        JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, S.Half)) )/3 - \
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, S.Half)) )/6 - \
        sqrt(6)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/6 + \
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, Rational(3, 2))) )/3 + \
        sqrt(10)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/30 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1)), ((1, 2), (3, 4), (1, 3)) ) == \
        sqrt(3)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, S.Half), (1, 3, S.Half)) )/3 + \
        JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, S.Half)) )/3 - \
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, S.Half)) )/6 - \
        sqrt(6)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/6 - \
        sqrt(2)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, Rational(3, 2))) )/3 - \
        sqrt(10)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/30 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0)), ((1, 2), (3, 4), (1, 3)) ) == \
        sqrt(6)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, S.Half), (1, 3, S.Half)) )/6 - \
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, S.Half)) )/6 - \
        JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, S.Half)) )/3 - \
        sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/3 - \
        JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, Rational(3, 2))) )/3 + \
        sqrt(5)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1)), ((1, 2), (3, 4), (1, 3)) ) == \
        -sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/2 + \
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/10 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1)), ((1, 2), (3, 4), (1, 3)) ) == \
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, S.Half)) )/2 - \
        sqrt(10)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/5 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0)), ((1, 2), (3, 4), (1, 3)) ) == \
        -sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, S.Half)) )/3 + \
        JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, S.Half)) )/3 + \
        JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, Rational(3, 2))) )/3 - \
        4*sqrt(5)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1)), ((1, 2), (3, 4), (1, 3)) ) == \
        sqrt(6)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, Rational(3, 2))) )/3 - \
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1)), ((1, 2), (3, 4), (1, 3)) ) == \
        2*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, S.Half)) )/3 + \
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, S.Half)) )/6 - \
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, Rational(3, 2))) )/3 - \
        2*sqrt(10)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0)), ((1, 2), (3, 4), (1, 3)) ) == \
        -sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, Rational(3, 2))) )/3 - \
        2*sqrt(15)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1)), ((1, 2), (3, 4), (1, 3)) ) == \
        JzKetCoupled(Rational(5, 2), Rational(-5, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )


def test_couple_symbolic():
    assert couple(TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == \
        Sum(CG(j1, m1, j2, m2, j, m1 + m2) * JzKetCoupled(j, m1 + m2, (
            j1, j2)), (j, m1 + m2, j1 + j2))
    assert couple(TensorProduct(JzKet(j1, m1), JzKet(j2, m2), JzKet(j3, m3))) == \
        Sum(CG(j1, m1, j2, m2, j12, m1 + m2) * CG(j12, m1 + m2, j3, m3, j, m1 + m2 + m3) *
            JzKetCoupled(j, m1 + m2 + m3, (j1, j2, j3), ((1, 2, j12), (1, 3, j)) ),
            (j12, m1 + m2, j1 + j2), (j, m1 + m2 + m3, j12 + j3))
    assert couple(TensorProduct(JzKet(j1, m1), JzKet(j2, m2), JzKet(j3, m3)), ((1, 3), (1, 2)) ) == \
        Sum(CG(j1, m1, j3, m3, j13, m1 + m3) * CG(j13, m1 + m3, j2, m2, j, m1 + m2 + m3) *
            JzKetCoupled(j, m1 + m2 + m3, (j1, j2, j3), ((1, 3, j13), (1, 2, j)) ),
            (j13, m1 + m3, j1 + j3), (j, m1 + m2 + m3, j13 + j2))
    assert couple(TensorProduct(JzKet(j1, m1), JzKet(j2, m2), JzKet(j3, m3), JzKet(j4, m4))) == \
        Sum(CG(j1, m1, j2, m2, j12, m1 + m2) * CG(j12, m1 + m2, j3, m3, j123, m1 + m2 + m3) * CG(j123, m1 + m2 + m3, j4, m4, j, m1 + m2 + m3 + m4) *
            JzKetCoupled(j, m1 + m2 + m3 + m4, (
                j1, j2, j3, j4), ((1, 2, j12), (1, 3, j123), (1, 4, j)) ),
            (j12, m1 + m2, j1 + j2), (j123, m1 + m2 + m3, j12 + j3), (j, m1 + m2 + m3 + m4, j123 + j4))
    assert couple(TensorProduct(JzKet(j1, m1), JzKet(j2, m2), JzKet(j3, m3), JzKet(j4, m4)), ((1, 2), (3, 4), (1, 3)) ) == \
        Sum(CG(j1, m1, j2, m2, j12, m1 + m2) * CG(j3, m3, j4, m4, j34, m3 + m4) * CG(j12, m1 + m2, j34, m3 + m4, j, m1 + m2 + m3 + m4) *
            JzKetCoupled(j, m1 + m2 + m3 + m4, (
                j1, j2, j3, j4), ((1, 2, j12), (3, 4, j34), (1, 3, j)) ),
            (j12, m1 + m2, j1 + j2), (j34, m3 + m4, j3 + j4), (j, m1 + m2 + m3 + m4, j12 + j34))
    assert couple(TensorProduct(JzKet(j1, m1), JzKet(j2, m2), JzKet(j3, m3), JzKet(j4, m4)), ((1, 3), (1, 4), (1, 2)) ) == \
        Sum(CG(j1, m1, j3, m3, j13, m1 + m3) * CG(j13, m1 + m3, j4, m4, j134, m1 + m3 + m4) * CG(j134, m1 + m3 + m4, j2, m2, j, m1 + m2 + m3 + m4) *
            JzKetCoupled(j, m1 + m2 + m3 + m4, (
                j1, j2, j3, j4), ((1, 3, j13), (1, 4, j134), (1, 2, j)) ),
            (j13, m1 + m3, j1 + j3), (j134, m1 + m3 + m4, j13 + j4), (j, m1 + m2 + m3 + m4, j134 + j2))


def test_innerproduct():
    assert InnerProduct(JzBra(1, 1), JzKet(1, 1)).doit() == 1
    assert InnerProduct(
        JzBra(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))).doit() == 0
    assert InnerProduct(JzBra(j, m), JzKet(j, m)).doit() == 1
    assert InnerProduct(JzBra(1, 0), JyKet(1, 1)).doit() == I/sqrt(2)
    assert InnerProduct(
        JxBra(S.Half, S.Half), JzKet(S.Half, S.Half)).doit() == -sqrt(2)/2
    assert InnerProduct(JyBra(1, 1), JzKet(1, 1)).doit() == S.Half
    assert InnerProduct(JxBra(1, -1), JyKet(1, 1)).doit() == 0


def test_rotation_small_d():
    # Symbolic tests
    # j = 1/2
    assert Rotation.d(S.Half, S.Half, S.Half, beta).doit() == cos(beta/2)
    assert Rotation.d(S.Half, S.Half, Rational(-1, 2), beta).doit() == -sin(beta/2)
    assert Rotation.d(S.Half, Rational(-1, 2), S.Half, beta).doit() == sin(beta/2)
    assert Rotation.d(S.Half, Rational(-1, 2), Rational(-1, 2), beta).doit() == cos(beta/2)
    # j = 1
    assert Rotation.d(1, 1, 1, beta).doit() == (1 + cos(beta))/2
    assert Rotation.d(1, 1, 0, beta).doit() == -sin(beta)/sqrt(2)
    assert Rotation.d(1, 1, -1, beta).doit() == (1 - cos(beta))/2
    assert Rotation.d(1, 0, 1, beta).doit() == sin(beta)/sqrt(2)
    assert Rotation.d(1, 0, 0, beta).doit() == cos(beta)
    assert Rotation.d(1, 0, -1, beta).doit() == -sin(beta)/sqrt(2)
    assert Rotation.d(1, -1, 1, beta).doit() == (1 - cos(beta))/2
    assert Rotation.d(1, -1, 0, beta).doit() == sin(beta)/sqrt(2)
    assert Rotation.d(1, -1, -1, beta).doit() == (1 + cos(beta))/2
    # j = 3/2
    assert Rotation.d(S(
        3)/2, Rational(3, 2), Rational(3, 2), beta).doit() == (3*cos(beta/2) + cos(beta*Rational(3, 2)))/4
    assert Rotation.d(Rational(3, 2), S(
        3)/2, S.Half, beta).doit() == -sqrt(3)*(sin(beta/2) + sin(beta*Rational(3, 2)))/4
    assert Rotation.d(Rational(3, 2), S(
        3)/2, Rational(-1, 2), beta).doit() == sqrt(3)*(cos(beta/2) - cos(beta*Rational(3, 2)))/4
    assert Rotation.d(Rational(3, 2), S(
        3)/2, Rational(-3, 2), beta).doit() == (-3*sin(beta/2) + sin(beta*Rational(3, 2)))/4
    assert Rotation.d(Rational(3, 2), S(
        1)/2, Rational(3, 2), beta).doit() == sqrt(3)*(sin(beta/2) + sin(beta*Rational(3, 2)))/4
    assert Rotation.d(S(
        3)/2, S.Half, S.Half, beta).doit() == (cos(beta/2) + 3*cos(beta*Rational(3, 2)))/4
    assert Rotation.d(S(
        3)/2, S.Half, Rational(-1, 2), beta).doit() == (sin(beta/2) - 3*sin(beta*Rational(3, 2)))/4
    assert Rotation.d(Rational(3, 2), S(
        1)/2, Rational(-3, 2), beta).doit() == sqrt(3)*(cos(beta/2) - cos(beta*Rational(3, 2)))/4
    assert Rotation.d(Rational(3, 2), -S(
        1)/2, Rational(3, 2), beta).doit() == sqrt(3)*(cos(beta/2) - cos(beta*Rational(3, 2)))/4
    assert Rotation.d(Rational(3, 2), -S(
        1)/2, S.Half, beta).doit() == (-sin(beta/2) + 3*sin(beta*Rational(3, 2)))/4
    assert Rotation.d(Rational(3, 2), -S(
        1)/2, Rational(-1, 2), beta).doit() == (cos(beta/2) + 3*cos(beta*Rational(3, 2)))/4
    assert Rotation.d(Rational(3, 2), -S(
        1)/2, Rational(-3, 2), beta).doit() == -sqrt(3)*(sin(beta/2) + sin(beta*Rational(3, 2)))/4
    assert Rotation.d(S(
        3)/2, Rational(-3, 2), Rational(3, 2), beta).doit() == (3*sin(beta/2) - sin(beta*Rational(3, 2)))/4
    assert Rotation.d(Rational(3, 2), -S(
        3)/2, S.Half, beta).doit() == sqrt(3)*(cos(beta/2) - cos(beta*Rational(3, 2)))/4
    assert Rotation.d(Rational(3, 2), -S(
        3)/2, Rational(-1, 2), beta).doit() == sqrt(3)*(sin(beta/2) + sin(beta*Rational(3, 2)))/4
    assert Rotation.d(Rational(3, 2), -S(
        3)/2, Rational(-3, 2), beta).doit() == (3*cos(beta/2) + cos(beta*Rational(3, 2)))/4
    # j = 2
    assert Rotation.d(2, 2, 2, beta).doit() == (3 + 4*cos(beta) + cos(2*beta))/8
    assert Rotation.d(2, 2, 1, beta).doit() == -((cos(beta) + 1)*sin(beta))/2
    assert Rotation.d(2, 2, 0, beta).doit() == sqrt(6)*sin(beta)**2/4
    assert Rotation.d(2, 2, -1, beta).doit() == (cos(beta) - 1)*sin(beta)/2
    assert Rotation.d(2, 2, -2, beta).doit() == (3 - 4*cos(beta) + cos(2*beta))/8
    assert Rotation.d(2, 1, 2, beta).doit() == (cos(beta) + 1)*sin(beta)/2
    assert Rotation.d(2, 1, 1, beta).doit() == (cos(beta) + cos(2*beta))/2
    assert Rotation.d(2, 1, 0, beta).doit() == -sqrt(6)*sin(2*beta)/4
    assert Rotation.d(2, 1, -1, beta).doit() == (cos(beta) - cos(2*beta))/2
    assert Rotation.d(2, 1, -2, beta).doit() == (cos(beta) - 1)*sin(beta)/2
    assert Rotation.d(2, 0, 2, beta).doit() == sqrt(6)*sin(beta)**2/4
    assert Rotation.d(2, 0, 1, beta).doit() == sqrt(6)*sin(2*beta)/4
    assert Rotation.d(2, 0, 0, beta).doit() == (1 + 3*cos(2*beta))/4
    assert Rotation.d(2, 0, -1, beta).doit() == -sqrt(6)*sin(2*beta)/4
    assert Rotation.d(2, 0, -2, beta).doit() == sqrt(6)*sin(beta)**2/4
    assert Rotation.d(2, -1, 2, beta).doit() == (2*sin(beta) - sin(2*beta))/4
    assert Rotation.d(2, -1, 1, beta).doit() == (cos(beta) - cos(2*beta))/2
    assert Rotation.d(2, -1, 0, beta).doit() == sqrt(6)*sin(2*beta)/4
    assert Rotation.d(2, -1, -1, beta).doit() == (cos(beta) + cos(2*beta))/2
    assert Rotation.d(2, -1, -2, beta).doit() == -((cos(beta) + 1)*sin(beta))/2
    assert Rotation.d(2, -2, 2, beta).doit() == (3 - 4*cos(beta) + cos(2*beta))/8
    assert Rotation.d(2, -2, 1, beta).doit() == (2*sin(beta) - sin(2*beta))/4
    assert Rotation.d(2, -2, 0, beta).doit() == sqrt(6)*sin(beta)**2/4
    assert Rotation.d(2, -2, -1, beta).doit() == (cos(beta) + 1)*sin(beta)/2
    assert Rotation.d(2, -2, -2, beta).doit() == (3 + 4*cos(beta) + cos(2*beta))/8
    # Numerical tests
    # j = 1/2
    assert Rotation.d(S.Half, S.Half, S.Half, pi/2).doit() == sqrt(2)/2
    assert Rotation.d(S.Half, S.Half, Rational(-1, 2), pi/2).doit() == -sqrt(2)/2
    assert Rotation.d(S.Half, Rational(-1, 2), S.Half, pi/2).doit() == sqrt(2)/2
    assert Rotation.d(S.Half, Rational(-1, 2), Rational(-1, 2), pi/2).doit() == sqrt(2)/2
    # j = 1
    assert Rotation.d(1, 1, 1, pi/2).doit() == S.Half
    assert Rotation.d(1, 1, 0, pi/2).doit() == -sqrt(2)/2
    assert Rotation.d(1, 1, -1, pi/2).doit() == S.Half
    assert Rotation.d(1, 0, 1, pi/2).doit() == sqrt(2)/2
    assert Rotation.d(1, 0, 0, pi/2).doit() == 0
    assert Rotation.d(1, 0, -1, pi/2).doit() == -sqrt(2)/2
    assert Rotation.d(1, -1, 1, pi/2).doit() == S.Half
    assert Rotation.d(1, -1, 0, pi/2).doit() == sqrt(2)/2
    assert Rotation.d(1, -1, -1, pi/2).doit() == S.Half
    # j = 3/2
    assert Rotation.d(Rational(3, 2), Rational(3, 2), Rational(3, 2), pi/2).doit() == sqrt(2)/4
    assert Rotation.d(Rational(3, 2), Rational(3, 2), S.Half, pi/2).doit() == -sqrt(6)/4
    assert Rotation.d(Rational(3, 2), Rational(3, 2), Rational(-1, 2), pi/2).doit() == sqrt(6)/4
    assert Rotation.d(Rational(3, 2), Rational(3, 2), Rational(-3, 2), pi/2).doit() == -sqrt(2)/4
    assert Rotation.d(Rational(3, 2), S.Half, Rational(3, 2), pi/2).doit() == sqrt(6)/4
    assert Rotation.d(Rational(3, 2), S.Half, S.Half, pi/2).doit() == -sqrt(2)/4
    assert Rotation.d(Rational(3, 2), S.Half, Rational(-1, 2), pi/2).doit() == -sqrt(2)/4
    assert Rotation.d(Rational(3, 2), S.Half, Rational(-3, 2), pi/2).doit() == sqrt(6)/4
    assert Rotation.d(Rational(3, 2), Rational(-1, 2), Rational(3, 2), pi/2).doit() == sqrt(6)/4
    assert Rotation.d(Rational(3, 2), Rational(-1, 2), S.Half, pi/2).doit() == sqrt(2)/4
    assert Rotation.d(Rational(3, 2), Rational(-1, 2), Rational(-1, 2), pi/2).doit() == -sqrt(2)/4
    assert Rotation.d(Rational(3, 2), Rational(-1, 2), Rational(-3, 2), pi/2).doit() == -sqrt(6)/4
    assert Rotation.d(Rational(3, 2), Rational(-3, 2), Rational(3, 2), pi/2).doit() == sqrt(2)/4
    assert Rotation.d(Rational(3, 2), Rational(-3, 2), S.Half, pi/2).doit() == sqrt(6)/4
    assert Rotation.d(Rational(3, 2), Rational(-3, 2), Rational(-1, 2), pi/2).doit() == sqrt(6)/4
    assert Rotation.d(Rational(3, 2), Rational(-3, 2), Rational(-3, 2), pi/2).doit() == sqrt(2)/4
    # j = 2
    assert Rotation.d(2, 2, 2, pi/2).doit() == Rational(1, 4)
    assert Rotation.d(2, 2, 1, pi/2).doit() == Rational(-1, 2)
    assert Rotation.d(2, 2, 0, pi/2).doit() == sqrt(6)/4
    assert Rotation.d(2, 2, -1, pi/2).doit() == Rational(-1, 2)
    assert Rotation.d(2, 2, -2, pi/2).doit() == Rational(1, 4)
    assert Rotation.d(2, 1, 2, pi/2).doit() == S.Half
    assert Rotation.d(2, 1, 1, pi/2).doit() == Rational(-1, 2)
    assert Rotation.d(2, 1, 0, pi/2).doit() == 0
    assert Rotation.d(2, 1, -1, pi/2).doit() == S.Half
    assert Rotation.d(2, 1, -2, pi/2).doit() == Rational(-1, 2)
    assert Rotation.d(2, 0, 2, pi/2).doit() == sqrt(6)/4
    assert Rotation.d(2, 0, 1, pi/2).doit() == 0
    assert Rotation.d(2, 0, 0, pi/2).doit() == Rational(-1, 2)
    assert Rotation.d(2, 0, -1, pi/2).doit() == 0
    assert Rotation.d(2, 0, -2, pi/2).doit() == sqrt(6)/4
    assert Rotation.d(2, -1, 2, pi/2).doit() == S.Half
    assert Rotation.d(2, -1, 1, pi/2).doit() == S.Half
    assert Rotation.d(2, -1, 0, pi/2).doit() == 0
    assert Rotation.d(2, -1, -1, pi/2).doit() == Rational(-1, 2)
    assert Rotation.d(2, -1, -2, pi/2).doit() == Rational(-1, 2)
    assert Rotation.d(2, -2, 2, pi/2).doit() == Rational(1, 4)
    assert Rotation.d(2, -2, 1, pi/2).doit() == S.Half
    assert Rotation.d(2, -2, 0, pi/2).doit() == sqrt(6)/4
    assert Rotation.d(2, -2, -1, pi/2).doit() == S.Half
    assert Rotation.d(2, -2, -2, pi/2).doit() == Rational(1, 4)


def test_rotation_d():
    # Symbolic tests
    # j = 1/2
    assert Rotation.D(S.Half, S.Half, S.Half, alpha, beta, gamma).doit() == \
        cos(beta/2)*exp(-I*alpha/2)*exp(-I*gamma/2)
    assert Rotation.D(S.Half, S.Half, Rational(-1, 2), alpha, beta, gamma).doit() == \
        -sin(beta/2)*exp(-I*alpha/2)*exp(I*gamma/2)
    assert Rotation.D(S.Half, Rational(-1, 2), S.Half, alpha, beta, gamma).doit() == \
        sin(beta/2)*exp(I*alpha/2)*exp(-I*gamma/2)
    assert Rotation.D(S.Half, Rational(-1, 2), Rational(-1, 2), alpha, beta, gamma).doit() == \
        cos(beta/2)*exp(I*alpha/2)*exp(I*gamma/2)
    # j = 1
    assert Rotation.D(1, 1, 1, alpha, beta, gamma).doit() == \
        (1 + cos(beta))/2*exp(-I*alpha)*exp(-I*gamma)
    assert Rotation.D(1, 1, 0, alpha, beta, gamma).doit() == -sin(
        beta)/sqrt(2)*exp(-I*alpha)
    assert Rotation.D(1, 1, -1, alpha, beta, gamma).doit() == \
        (1 - cos(beta))/2*exp(-I*alpha)*exp(I*gamma)
    assert Rotation.D(1, 0, 1, alpha, beta, gamma).doit() == \
        sin(beta)/sqrt(2)*exp(-I*gamma)
    assert Rotation.D(1, 0, 0, alpha, beta, gamma).doit() == cos(beta)
    assert Rotation.D(1, 0, -1, alpha, beta, gamma).doit() == \
        -sin(beta)/sqrt(2)*exp(I*gamma)
    assert Rotation.D(1, -1, 1, alpha, beta, gamma).doit() == \
        (1 - cos(beta))/2*exp(I*alpha)*exp(-I*gamma)
    assert Rotation.D(1, -1, 0, alpha, beta, gamma).doit() == \
        sin(beta)/sqrt(2)*exp(I*alpha)
    assert Rotation.D(1, -1, -1, alpha, beta, gamma).doit() == \
        (1 + cos(beta))/2*exp(I*alpha)*exp(I*gamma)
    # j = 3/2
    assert Rotation.D(Rational(3, 2), Rational(3, 2), Rational(3, 2), alpha, beta, gamma).doit() == \
        (3*cos(beta/2) + cos(beta*Rational(3, 2)))/4*exp(I*alpha*Rational(-3, 2))*exp(I*gamma*Rational(-3, 2))
    assert Rotation.D(Rational(3, 2), Rational(3, 2), S.Half, alpha, beta, gamma).doit() == \
        -sqrt(3)*(sin(beta/2) + sin(beta*Rational(3, 2)))/4*exp(I*alpha*Rational(-3, 2))*exp(-I*gamma/2)
    assert Rotation.D(Rational(3, 2), Rational(3, 2), Rational(-1, 2), alpha, beta, gamma).doit() == \
        sqrt(3)*(cos(beta/2) - cos(beta*Rational(3, 2)))/4*exp(I*alpha*Rational(-3, 2))*exp(I*gamma/2)
    assert Rotation.D(Rational(3, 2), Rational(3, 2), Rational(-3, 2), alpha, beta, gamma).doit() == \
        (-3*sin(beta/2) + sin(beta*Rational(3, 2)))/4*exp(I*alpha*Rational(-3, 2))*exp(I*gamma*Rational(3, 2))
    assert Rotation.D(Rational(3, 2), S.Half, Rational(3, 2), alpha, beta, gamma).doit() == \
        sqrt(3)*(sin(beta/2) + sin(beta*Rational(3, 2)))/4*exp(-I*alpha/2)*exp(I*gamma*Rational(-3, 2))
    assert Rotation.D(Rational(3, 2), S.Half, S.Half, alpha, beta, gamma).doit() == \
        (cos(beta/2) + 3*cos(beta*Rational(3, 2)))/4*exp(-I*alpha/2)*exp(-I*gamma/2)
    assert Rotation.D(Rational(3, 2), S.Half, Rational(-1, 2), alpha, beta, gamma).doit() == \
        (sin(beta/2) - 3*sin(beta*Rational(3, 2)))/4*exp(-I*alpha/2)*exp(I*gamma/2)
    assert Rotation.D(Rational(3, 2), S.Half, Rational(-3, 2), alpha, beta, gamma).doit() == \
        sqrt(3)*(cos(beta/2) - cos(beta*Rational(3, 2)))/4*exp(-I*alpha/2)*exp(I*gamma*Rational(3, 2))
    assert Rotation.D(Rational(3, 2), Rational(-1, 2), Rational(3, 2), alpha, beta, gamma).doit() == \
        sqrt(3)*(cos(beta/2) - cos(beta*Rational(3, 2)))/4*exp(I*alpha/2)*exp(I*gamma*Rational(-3, 2))
    assert Rotation.D(Rational(3, 2), Rational(-1, 2), S.Half, alpha, beta, gamma).doit() == \
        (-sin(beta/2) + 3*sin(beta*Rational(3, 2)))/4*exp(I*alpha/2)*exp(-I*gamma/2)
    assert Rotation.D(Rational(3, 2), Rational(-1, 2), Rational(-1, 2), alpha, beta, gamma).doit() == \
        (cos(beta/2) + 3*cos(beta*Rational(3, 2)))/4*exp(I*alpha/2)*exp(I*gamma/2)
    assert Rotation.D(Rational(3, 2), Rational(-1, 2), Rational(-3, 2), alpha, beta, gamma).doit() == \
        -sqrt(3)*(sin(beta/2) + sin(beta*Rational(3, 2)))/4*exp(I*alpha/2)*exp(I*gamma*Rational(3, 2))
    assert Rotation.D(Rational(3, 2), Rational(-3, 2), Rational(3, 2), alpha, beta, gamma).doit() == \
        (3*sin(beta/2) - sin(beta*Rational(3, 2)))/4*exp(I*alpha*Rational(3, 2))*exp(I*gamma*Rational(-3, 2))
    assert Rotation.D(Rational(3, 2), Rational(-3, 2), S.Half, alpha, beta, gamma).doit() == \
        sqrt(3)*(cos(beta/2) - cos(beta*Rational(3, 2)))/4*exp(I*alpha*Rational(3, 2))*exp(-I*gamma/2)
    assert Rotation.D(Rational(3, 2), Rational(-3, 2), Rational(-1, 2), alpha, beta, gamma).doit() == \
        sqrt(3)*(sin(beta/2) + sin(beta*Rational(3, 2)))/4*exp(I*alpha*Rational(3, 2))*exp(I*gamma/2)
    assert Rotation.D(Rational(3, 2), Rational(-3, 2), Rational(-3, 2), alpha, beta, gamma).doit() == \
        (3*cos(beta/2) + cos(beta*Rational(3, 2)))/4*exp(I*alpha*Rational(3, 2))*exp(I*gamma*Rational(3, 2))
    # j = 2
    assert Rotation.D(2, 2, 2, alpha, beta, gamma).doit() == \
        (3 + 4*cos(beta) + cos(2*beta))/8*exp(-2*I*alpha)*exp(-2*I*gamma)
    assert Rotation.D(2, 2, 1, alpha, beta, gamma).doit() == \
        -((cos(beta) + 1)*exp(-2*I*alpha)*exp(-I*gamma)*sin(beta))/2
    assert Rotation.D(2, 2, 0, alpha, beta, gamma).doit() == \
        sqrt(6)*sin(beta)**2/4*exp(-2*I*alpha)
    assert Rotation.D(2, 2, -1, alpha, beta, gamma).doit() == \
        (cos(beta) - 1)*sin(beta)/2*exp(-2*I*alpha)*exp(I*gamma)
    assert Rotation.D(2, 2, -2, alpha, beta, gamma).doit() == \
        (3 - 4*cos(beta) + cos(2*beta))/8*exp(-2*I*alpha)*exp(2*I*gamma)
    assert Rotation.D(2, 1, 2, alpha, beta, gamma).doit() == \
        (cos(beta) + 1)*sin(beta)/2*exp(-I*alpha)*exp(-2*I*gamma)
    assert Rotation.D(2, 1, 1, alpha, beta, gamma).doit() == \
        (cos(beta) + cos(2*beta))/2*exp(-I*alpha)*exp(-I*gamma)
    assert Rotation.D(2, 1, 0, alpha, beta, gamma).doit() == -sqrt(6)* \
        sin(2*beta)/4*exp(-I*alpha)
    assert Rotation.D(2, 1, -1, alpha, beta, gamma).doit() == \
        (cos(beta) - cos(2*beta))/2*exp(-I*alpha)*exp(I*gamma)
    assert Rotation.D(2, 1, -2, alpha, beta, gamma).doit() == \
        (cos(beta) - 1)*sin(beta)/2*exp(-I*alpha)*exp(2*I*gamma)
    assert Rotation.D(2, 0, 2, alpha, beta, gamma).doit() == \
        sqrt(6)*sin(beta)**2/4*exp(-2*I*gamma)
    assert Rotation.D(2, 0, 1, alpha, beta, gamma).doit() == sqrt(6)* \
        sin(2*beta)/4*exp(-I*gamma)
    assert Rotation.D(
        2, 0, 0, alpha, beta, gamma).doit() == (1 + 3*cos(2*beta))/4
    assert Rotation.D(2, 0, -1, alpha, beta, gamma).doit() == -sqrt(6)* \
        sin(2*beta)/4*exp(I*gamma)
    assert Rotation.D(2, 0, -2, alpha, beta, gamma).doit() == \
        sqrt(6)*sin(beta)**2/4*exp(2*I*gamma)
    assert Rotation.D(2, -1, 2, alpha, beta, gamma).doit() == \
        (2*sin(beta) - sin(2*beta))/4*exp(I*alpha)*exp(-2*I*gamma)
    assert Rotation.D(2, -1, 1, alpha, beta, gamma).doit() == \
        (cos(beta) - cos(2*beta))/2*exp(I*alpha)*exp(-I*gamma)
    assert Rotation.D(2, -1, 0, alpha, beta, gamma).doit() == sqrt(6)* \
        sin(2*beta)/4*exp(I*alpha)
    assert Rotation.D(2, -1, -1, alpha, beta, gamma).doit() == \
        (cos(beta) + cos(2*beta))/2*exp(I*alpha)*exp(I*gamma)
    assert Rotation.D(2, -1, -2, alpha, beta, gamma).doit() == \
        -((cos(beta) + 1)*sin(beta))/2*exp(I*alpha)*exp(2*I*gamma)
    assert Rotation.D(2, -2, 2, alpha, beta, gamma).doit() == \
        (3 - 4*cos(beta) + cos(2*beta))/8*exp(2*I*alpha)*exp(-2*I*gamma)
    assert Rotation.D(2, -2, 1, alpha, beta, gamma).doit() == \
        (2*sin(beta) - sin(2*beta))/4*exp(2*I*alpha)*exp(-I*gamma)
    assert Rotation.D(2, -2, 0, alpha, beta, gamma).doit() == \
        sqrt(6)*sin(beta)**2/4*exp(2*I*alpha)
    assert Rotation.D(2, -2, -1, alpha, beta, gamma).doit() == \
        (cos(beta) + 1)*sin(beta)/2*exp(2*I*alpha)*exp(I*gamma)
    assert Rotation.D(2, -2, -2, alpha, beta, gamma).doit() == \
        (3 + 4*cos(beta) + cos(2*beta))/8*exp(2*I*alpha)*exp(2*I*gamma)
    # Numerical tests
    # j = 1/2
    assert Rotation.D(
        S.Half, S.Half, S.Half, pi/2, pi/2, pi/2).doit() == -I*sqrt(2)/2
    assert Rotation.D(
        S.Half, S.Half, Rational(-1, 2), pi/2, pi/2, pi/2).doit() == -sqrt(2)/2
    assert Rotation.D(
        S.Half, Rational(-1, 2), S.Half, pi/2, pi/2, pi/2).doit() == sqrt(2)/2
    assert Rotation.D(
        S.Half, Rational(-1, 2), Rational(-1, 2), pi/2, pi/2, pi/2).doit() == I*sqrt(2)/2
    # j = 1
    assert Rotation.D(1, 1, 1, pi/2, pi/2, pi/2).doit() == Rational(-1, 2)
    assert Rotation.D(1, 1, 0, pi/2, pi/2, pi/2).doit() == I*sqrt(2)/2
    assert Rotation.D(1, 1, -1, pi/2, pi/2, pi/2).doit() == S.Half
    assert Rotation.D(1, 0, 1, pi/2, pi/2, pi/2).doit() == -I*sqrt(2)/2
    assert Rotation.D(1, 0, 0, pi/2, pi/2, pi/2).doit() == 0
    assert Rotation.D(1, 0, -1, pi/2, pi/2, pi/2).doit() == -I*sqrt(2)/2
    assert Rotation.D(1, -1, 1, pi/2, pi/2, pi/2).doit() == S.Half
    assert Rotation.D(1, -1, 0, pi/2, pi/2, pi/2).doit() == I*sqrt(2)/2
    assert Rotation.D(1, -1, -1, pi/2, pi/2, pi/2).doit() == Rational(-1, 2)
    # j = 3/2
    assert Rotation.D(
        Rational(3, 2), Rational(3, 2), Rational(3, 2), pi/2, pi/2, pi/2).doit() == I*sqrt(2)/4
    assert Rotation.D(
        Rational(3, 2), Rational(3, 2), S.Half, pi/2, pi/2, pi/2).doit() == sqrt(6)/4
    assert Rotation.D(
        Rational(3, 2), Rational(3, 2), Rational(-1, 2), pi/2, pi/2, pi/2).doit() == -I*sqrt(6)/4
    assert Rotation.D(
        Rational(3, 2), Rational(3, 2), Rational(-3, 2), pi/2, pi/2, pi/2).doit() == -sqrt(2)/4
    assert Rotation.D(
        Rational(3, 2), S.Half, Rational(3, 2), pi/2, pi/2, pi/2).doit() == -sqrt(6)/4
    assert Rotation.D(
        Rational(3, 2), S.Half, S.Half, pi/2, pi/2, pi/2).doit() == I*sqrt(2)/4
    assert Rotation.D(
        Rational(3, 2), S.Half, Rational(-1, 2), pi/2, pi/2, pi/2).doit() == -sqrt(2)/4
    assert Rotation.D(
        Rational(3, 2), S.Half, Rational(-3, 2), pi/2, pi/2, pi/2).doit() == I*sqrt(6)/4
    assert Rotation.D(
        Rational(3, 2), Rational(-1, 2), Rational(3, 2), pi/2, pi/2, pi/2).doit() == -I*sqrt(6)/4
    assert Rotation.D(
        Rational(3, 2), Rational(-1, 2), S.Half, pi/2, pi/2, pi/2).doit() == sqrt(2)/4
    assert Rotation.D(
        Rational(3, 2), Rational(-1, 2), Rational(-1, 2), pi/2, pi/2, pi/2).doit() == -I*sqrt(2)/4
    assert Rotation.D(
        Rational(3, 2), Rational(-1, 2), Rational(-3, 2), pi/2, pi/2, pi/2).doit() == sqrt(6)/4
    assert Rotation.D(
        Rational(3, 2), Rational(-3, 2), Rational(3, 2), pi/2, pi/2, pi/2).doit() == sqrt(2)/4
    assert Rotation.D(
        Rational(3, 2), Rational(-3, 2), S.Half, pi/2, pi/2, pi/2).doit() == I*sqrt(6)/4
    assert Rotation.D(
        Rational(3, 2), Rational(-3, 2), Rational(-1, 2), pi/2, pi/2, pi/2).doit() == -sqrt(6)/4
    assert Rotation.D(
        Rational(3, 2), Rational(-3, 2), Rational(-3, 2), pi/2, pi/2, pi/2).doit() == -I*sqrt(2)/4
    # j = 2
    assert Rotation.D(2, 2, 2, pi/2, pi/2, pi/2).doit() == Rational(1, 4)
    assert Rotation.D(2, 2, 1, pi/2, pi/2, pi/2).doit() == -I/2
    assert Rotation.D(2, 2, 0, pi/2, pi/2, pi/2).doit() == -sqrt(6)/4
    assert Rotation.D(2, 2, -1, pi/2, pi/2, pi/2).doit() == I/2
    assert Rotation.D(2, 2, -2, pi/2, pi/2, pi/2).doit() == Rational(1, 4)
    assert Rotation.D(2, 1, 2, pi/2, pi/2, pi/2).doit() == I/2
    assert Rotation.D(2, 1, 1, pi/2, pi/2, pi/2).doit() == S.Half
    assert Rotation.D(2, 1, 0, pi/2, pi/2, pi/2).doit() == 0
    assert Rotation.D(2, 1, -1, pi/2, pi/2, pi/2).doit() == S.Half
    assert Rotation.D(2, 1, -2, pi/2, pi/2, pi/2).doit() == -I/2
    assert Rotation.D(2, 0, 2, pi/2, pi/2, pi/2).doit() == -sqrt(6)/4
    assert Rotation.D(2, 0, 1, pi/2, pi/2, pi/2).doit() == 0
    assert Rotation.D(2, 0, 0, pi/2, pi/2, pi/2).doit() == Rational(-1, 2)
    assert Rotation.D(2, 0, -1, pi/2, pi/2, pi/2).doit() == 0
    assert Rotation.D(2, 0, -2, pi/2, pi/2, pi/2).doit() == -sqrt(6)/4
    assert Rotation.D(2, -1, 2, pi/2, pi/2, pi/2).doit() == -I/2
    assert Rotation.D(2, -1, 1, pi/2, pi/2, pi/2).doit() == S.Half
    assert Rotation.D(2, -1, 0, pi/2, pi/2, pi/2).doit() == 0
    assert Rotation.D(2, -1, -1, pi/2, pi/2, pi/2).doit() == S.Half
    assert Rotation.D(2, -1, -2, pi/2, pi/2, pi/2).doit() == I/2
    assert Rotation.D(2, -2, 2, pi/2, pi/2, pi/2).doit() == Rational(1, 4)
    assert Rotation.D(2, -2, 1, pi/2, pi/2, pi/2).doit() == I/2
    assert Rotation.D(2, -2, 0, pi/2, pi/2, pi/2).doit() == -sqrt(6)/4
    assert Rotation.D(2, -2, -1, pi/2, pi/2, pi/2).doit() == -I/2
    assert Rotation.D(2, -2, -2, pi/2, pi/2, pi/2).doit() == Rational(1, 4)


def test_wignerd():
    assert Rotation.D(
        j, m, mp, alpha, beta, gamma) == WignerD(j, m, mp, alpha, beta, gamma)
    assert Rotation.d(j, m, mp, beta) == WignerD(j, m, mp, 0, beta, 0)

def test_wignerD():
    i,j=symbols('i j')
    assert Rotation.D(1, 1, 1, 0, 0, 0) == WignerD(1, 1, 1, 0, 0, 0)
    assert Rotation.D(1, 1, 2, 0, 0, 0) == WignerD(1, 1, 2, 0, 0, 0)
    assert Rotation.D(1, i**2 - j**2, i**2 - j**2, 0, 0, 0) == WignerD(1, i**2 - j**2, i**2 - j**2, 0, 0, 0)
    assert Rotation.D(1, i, i, 0, 0, 0) == WignerD(1, i, i, 0, 0, 0)
    assert Rotation.D(1, i, i+1, 0, 0, 0) == WignerD(1, i, i+1, 0, 0, 0)
    assert Rotation.D(1, 0, 0, 0, 0, 0) == WignerD(1, 0, 0, 0, 0, 0)

def test_jplus():
    assert Commutator(Jplus, Jminus).doit() == 2*hbar*Jz
    assert Jplus.matrix_element(1, 1, 1, 1) == 0
    assert Jplus.rewrite('xyz') == Jx + I*Jy
    # Normal operators, normal states
    # Numerical
    assert qapply(Jplus*JxKet(1, 1)) == \
        -hbar*sqrt(2)*JxKet(1, 0)/2 + hbar*JxKet(1, 1)
    assert qapply(Jplus*JyKet(1, 1)) == \
        hbar*sqrt(2)*JyKet(1, 0)/2 + I*hbar*JyKet(1, 1)
    assert qapply(Jplus*JzKet(1, 1)) == 0
    # Symbolic
    assert qapply(Jplus*JxKet(j, m)) == \
        Sum(hbar * sqrt(-mi**2 - mi + j**2 + j) * WignerD(j, mi, m, 0, pi/2, 0) *
        Sum(WignerD(j, mi1, mi + 1, 0, pi*Rational(3, 2), 0) * JxKet(j, mi1),
        (mi1, -j, j)), (mi, -j, j))
    assert qapply(Jplus*JyKet(j, m)) == \
        Sum(hbar * sqrt(j**2 + j - mi**2 - mi) * WignerD(j, mi, m, pi*Rational(3, 2), -pi/2, pi/2) *
        Sum(WignerD(j, mi1, mi + 1, pi*Rational(3, 2), pi/2, pi/2) * JyKet(j, mi1),
        (mi1, -j, j)), (mi, -j, j))
    assert qapply(Jplus*JzKet(j, m)) == \
        hbar*sqrt(j**2 + j - m**2 - m)*JzKet(j, m + 1)
    # Normal operators, coupled states
    # Numerical
    assert qapply(Jplus*JxKetCoupled(1, 1, (1, 1))) == -hbar*sqrt(2) * \
        JxKetCoupled(1, 0, (1, 1))/2 + hbar*JxKetCoupled(1, 1, (1, 1))
    assert qapply(Jplus*JyKetCoupled(1, 1, (1, 1))) == hbar*sqrt(2) * \
        JyKetCoupled(1, 0, (1, 1))/2 + I*hbar*JyKetCoupled(1, 1, (1, 1))
    assert qapply(Jplus*JzKet(1, 1)) == 0
    # Symbolic
    assert qapply(Jplus*JxKetCoupled(j, m, (j1, j2))) == \
        Sum(hbar * sqrt(-mi**2 - mi + j**2 + j) * WignerD(j, mi, m, 0, pi/2, 0) *
        Sum(
            WignerD(
                j, mi1, mi + 1, 0, pi*Rational(3, 2), 0) * JxKetCoupled(j, mi1, (j1, j2)),
        (mi1, -j, j)), (mi, -j, j))
    assert qapply(Jplus*JyKetCoupled(j, m, (j1, j2))) == \
        Sum(hbar * sqrt(j**2 + j - mi**2 - mi) * WignerD(j, mi, m, pi*Rational(3, 2), -pi/2, pi/2) *
        Sum(
            WignerD(j, mi1, mi + 1, pi*Rational(3, 2), pi/2, pi/2) *
            JyKetCoupled(j, mi1, (j1, j2)),
        (mi1, -j, j)), (mi, -j, j))
    assert qapply(Jplus*JzKetCoupled(j, m, (j1, j2))) == \
        hbar*sqrt(j**2 + j - m**2 - m)*JzKetCoupled(j, m + 1, (j1, j2))
    # Uncoupled operators, uncoupled states
    # Numerical
    assert qapply(TensorProduct(Jplus, 1)*TensorProduct(JxKet(1, 1), JxKet(1, -1))) == \
        -hbar*sqrt(2)*TensorProduct(JxKet(1, 0), JxKet(1, -1))/2 + \
        hbar*TensorProduct(JxKet(1, 1), JxKet(1, -1))
    assert qapply(TensorProduct(1, Jplus)*TensorProduct(JxKet(1, 1), JxKet(1, -1))) == \
        -hbar*TensorProduct(JxKet(1, 1), JxKet(1, -1)) + \
        hbar*sqrt(2)*TensorProduct(JxKet(1, 1), JxKet(1, 0))/2
    assert qapply(TensorProduct(Jplus, 1)*TensorProduct(JyKet(1, 1), JyKet(1, -1))) == \
        hbar*sqrt(2)*TensorProduct(JyKet(1, 0), JyKet(1, -1))/2 + \
        hbar*I*TensorProduct(JyKet(1, 1), JyKet(1, -1))
    assert qapply(TensorProduct(1, Jplus)*TensorProduct(JyKet(1, 1), JyKet(1, -1))) == \
        -hbar*I*TensorProduct(JyKet(1, 1), JyKet(1, -1)) + \
        hbar*sqrt(2)*TensorProduct(JyKet(1, 1), JyKet(1, 0))/2
    assert qapply(
        TensorProduct(Jplus, 1)*TensorProduct(JzKet(1, 1), JzKet(1, -1))) == 0
    assert qapply(TensorProduct(1, Jplus)*TensorProduct(JzKet(1, 1), JzKet(1, -1))) == \
        hbar*sqrt(2)*TensorProduct(JzKet(1, 1), JzKet(1, 0))
    # Symbolic
    assert qapply(TensorProduct(Jplus, 1)*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == \
        TensorProduct(Sum(hbar * sqrt(-mi**2 - mi + j1**2 + j1) * WignerD(j1, mi, m1, 0, pi/2, 0) *
        Sum(WignerD(j1, mi1, mi + 1, 0, pi*Rational(3, 2), 0) * JxKet(j1, mi1),
        (mi1, -j1, j1)), (mi, -j1, j1)), JxKet(j2, m2))
    assert qapply(TensorProduct(1, Jplus)*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == \
        TensorProduct(JxKet(j1, m1), Sum(hbar * sqrt(-mi**2 - mi + j2**2 + j2) * WignerD(j2, mi, m2, 0, pi/2, 0) *
        Sum(WignerD(j2, mi1, mi + 1, 0, pi*Rational(3, 2), 0) * JxKet(j2, mi1),
        (mi1, -j2, j2)), (mi, -j2, j2)))
    assert qapply(TensorProduct(Jplus, 1)*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == \
        TensorProduct(Sum(hbar * sqrt(j1**2 + j1 - mi**2 - mi) * WignerD(j1, mi, m1, pi*Rational(3, 2), -pi/2, pi/2) *
        Sum(WignerD(j1, mi1, mi + 1, pi*Rational(3, 2), pi/2, pi/2) * JyKet(j1, mi1),
        (mi1, -j1, j1)), (mi, -j1, j1)), JyKet(j2, m2))
    assert qapply(TensorProduct(1, Jplus)*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == \
        TensorProduct(JyKet(j1, m1), Sum(hbar * sqrt(j2**2 + j2 - mi**2 - mi) * WignerD(j2, mi, m2, pi*Rational(3, 2), -pi/2, pi/2) *
        Sum(WignerD(j2, mi1, mi + 1, pi*Rational(3, 2), pi/2, pi/2) * JyKet(j2, mi1),
        (mi1, -j2, j2)), (mi, -j2, j2)))
    assert qapply(TensorProduct(Jplus, 1)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == \
        hbar*sqrt(
            j1**2 + j1 - m1**2 - m1)*TensorProduct(JzKet(j1, m1 + 1), JzKet(j2, m2))
    assert qapply(TensorProduct(1, Jplus)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == \
        hbar*sqrt(
            j2**2 + j2 - m2**2 - m2)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2 + 1))


def test_jminus():
    assert qapply(Jminus*JzKet(1, -1)) == 0
    assert Jminus.matrix_element(1, 0, 1, 1) == sqrt(2)*hbar
    assert Jminus.rewrite('xyz') == Jx - I*Jy
    # Normal operators, normal states
    # Numerical
    assert qapply(Jminus*JxKet(1, 1)) == \
        hbar*sqrt(2)*JxKet(1, 0)/2 + hbar*JxKet(1, 1)
    assert qapply(Jminus*JyKet(1, 1)) == \
        hbar*sqrt(2)*JyKet(1, 0)/2 - hbar*I*JyKet(1, 1)
    assert qapply(Jminus*JzKet(1, 1)) == sqrt(2)*hbar*JzKet(1, 0)
    # Symbolic
    assert qapply(Jminus*JxKet(j, m)) == \
        Sum(hbar*sqrt(j**2 + j - mi**2 + mi)*WignerD(j, mi, m, 0, pi/2, 0) *
        Sum(WignerD(j, mi1, mi - 1, 0, pi*Rational(3, 2), 0)*JxKet(j, mi1),
        (mi1, -j, j)), (mi, -j, j))
    assert qapply(Jminus*JyKet(j, m)) == \
        Sum(hbar*sqrt(j**2 + j - mi**2 + mi)*WignerD(j, mi, m, pi*Rational(3, 2), -pi/2, pi/2) *
        Sum(WignerD(j, mi1, mi - 1, pi*Rational(3, 2), pi/2, pi/2)*JyKet(j, mi1),
        (mi1, -j, j)), (mi, -j, j))
    assert qapply(Jminus*JzKet(j, m)) == \
        hbar*sqrt(j**2 + j - m**2 + m)*JzKet(j, m - 1)
    # Normal operators, coupled states
    # Numerical
    assert qapply(Jminus*JxKetCoupled(1, 1, (1, 1))) == \
        hbar*sqrt(2)*JxKetCoupled(1, 0, (1, 1))/2 + \
        hbar*JxKetCoupled(1, 1, (1, 1))
    assert qapply(Jminus*JyKetCoupled(1, 1, (1, 1))) == \
        hbar*sqrt(2)*JyKetCoupled(1, 0, (1, 1))/2 - \
        hbar*I*JyKetCoupled(1, 1, (1, 1))
    assert qapply(Jminus*JzKetCoupled(1, 1, (1, 1))) == \
        sqrt(2)*hbar*JzKetCoupled(1, 0, (1, 1))
    # Symbolic
    assert qapply(Jminus*JxKetCoupled(j, m, (j1, j2))) == \
        Sum(hbar*sqrt(j**2 + j - mi**2 + mi)*WignerD(j, mi, m, 0, pi/2, 0) *
        Sum(WignerD(j, mi1, mi - 1, 0, pi*Rational(3, 2), 0)*JxKetCoupled(j, mi1, (j1, j2)),
        (mi1, -j, j)), (mi, -j, j))
    assert qapply(Jminus*JyKetCoupled(j, m, (j1, j2))) == \
        Sum(hbar*sqrt(j**2 + j - mi**2 + mi)*WignerD(j, mi, m, pi*Rational(3, 2), -pi/2, pi/2) *
        Sum(
            WignerD(j, mi1, mi - 1, pi*Rational(3, 2), pi/2, pi/2)*
            JyKetCoupled(j, mi1, (j1, j2)),
        (mi1, -j, j)), (mi, -j, j))
    assert qapply(Jminus*JzKetCoupled(j, m, (j1, j2))) == \
        hbar*sqrt(j**2 + j - m**2 + m)*JzKetCoupled(j, m - 1, (j1, j2))
    # Uncoupled operators, uncoupled states
    # Numerical
    assert qapply(TensorProduct(Jminus, 1)*TensorProduct(JxKet(1, 1), JxKet(1, -1))) == \
        hbar*sqrt(2)*TensorProduct(JxKet(1, 0), JxKet(1, -1))/2 + \
        hbar*TensorProduct(JxKet(1, 1), JxKet(1, -1))
    assert qapply(TensorProduct(1, Jminus)*TensorProduct(JxKet(1, 1), JxKet(1, -1))) == \
        -hbar*TensorProduct(JxKet(1, 1), JxKet(1, -1)) - \
        hbar*sqrt(2)*TensorProduct(JxKet(1, 1), JxKet(1, 0))/2
    assert qapply(TensorProduct(Jminus, 1)*TensorProduct(JyKet(1, 1), JyKet(1, -1))) == \
        hbar*sqrt(2)*TensorProduct(JyKet(1, 0), JyKet(1, -1))/2 - \
        hbar*I*TensorProduct(JyKet(1, 1), JyKet(1, -1))
    assert qapply(TensorProduct(1, Jminus)*TensorProduct(JyKet(1, 1), JyKet(1, -1))) == \
        hbar*I*TensorProduct(JyKet(1, 1), JyKet(1, -1)) + \
        hbar*sqrt(2)*TensorProduct(JyKet(1, 1), JyKet(1, 0))/2
    assert qapply(TensorProduct(Jminus, 1)*TensorProduct(JzKet(1, 1), JzKet(1, -1))) == \
        sqrt(2)*hbar*TensorProduct(JzKet(1, 0), JzKet(1, -1))
    assert qapply(TensorProduct(
        1, Jminus)*TensorProduct(JzKet(1, 1), JzKet(1, -1))) == 0
    # Symbolic
    assert qapply(TensorProduct(Jminus, 1)*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == \
        TensorProduct(Sum(hbar*sqrt(j1**2 + j1 - mi**2 + mi)*WignerD(j1, mi, m1, 0, pi/2, 0) *
        Sum(WignerD(j1, mi1, mi - 1, 0, pi*Rational(3, 2), 0)*JxKet(j1, mi1),
        (mi1, -j1, j1)), (mi, -j1, j1)), JxKet(j2, m2))
    assert qapply(TensorProduct(1, Jminus)*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == \
        TensorProduct(JxKet(j1, m1), Sum(hbar*sqrt(j2**2 + j2 - mi**2 + mi)*WignerD(j2, mi, m2, 0, pi/2, 0) *
        Sum(WignerD(j2, mi1, mi - 1, 0, pi*Rational(3, 2), 0)*JxKet(j2, mi1),
        (mi1, -j2, j2)), (mi, -j2, j2)))
    assert qapply(TensorProduct(Jminus, 1)*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == \
        TensorProduct(Sum(hbar*sqrt(j1**2 + j1 - mi**2 + mi)*WignerD(j1, mi, m1, pi*Rational(3, 2), -pi/2, pi/2) *
        Sum(WignerD(j1, mi1, mi - 1, pi*Rational(3, 2), pi/2, pi/2)*JyKet(j1, mi1),
        (mi1, -j1, j1)), (mi, -j1, j1)), JyKet(j2, m2))
    assert qapply(TensorProduct(1, Jminus)*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == \
        TensorProduct(JyKet(j1, m1), Sum(hbar*sqrt(j2**2 + j2 - mi**2 + mi)*WignerD(j2, mi, m2, pi*Rational(3, 2), -pi/2, pi/2) *
        Sum(WignerD(j2, mi1, mi - 1, pi*Rational(3, 2), pi/2, pi/2)*JyKet(j2, mi1),
        (mi1, -j2, j2)), (mi, -j2, j2)))
    assert qapply(TensorProduct(Jminus, 1)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == \
        hbar*sqrt(
            j1**2 + j1 - m1**2 + m1)*TensorProduct(JzKet(j1, m1 - 1), JzKet(j2, m2))
    assert qapply(TensorProduct(1, Jminus)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == \
        hbar*sqrt(
            j2**2 + j2 - m2**2 + m2)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2 - 1))


def test_j2():
    assert Commutator(J2, Jz).doit() == 0
    assert J2.matrix_element(1, 1, 1, 1) == 2*hbar**2
    # Normal operators, normal states
    # Numerical
    assert qapply(J2*JxKet(1, 1)) == 2*hbar**2*JxKet(1, 1)
    assert qapply(J2*JyKet(1, 1)) == 2*hbar**2*JyKet(1, 1)
    assert qapply(J2*JzKet(1, 1)) == 2*hbar**2*JzKet(1, 1)
    # Symbolic
    assert qapply(J2*JxKet(j, m)) == \
        hbar**2*j**2*JxKet(j, m) + hbar**2*j*JxKet(j, m)
    assert qapply(J2*JyKet(j, m)) == \
        hbar**2*j**2*JyKet(j, m) + hbar**2*j*JyKet(j, m)
    assert qapply(J2*JzKet(j, m)) == \
        hbar**2*j**2*JzKet(j, m) + hbar**2*j*JzKet(j, m)
    # Normal operators, coupled states
    # Numerical
    assert qapply(J2*JxKetCoupled(1, 1, (1, 1))) == \
        2*hbar**2*JxKetCoupled(1, 1, (1, 1))
    assert qapply(J2*JyKetCoupled(1, 1, (1, 1))) == \
        2*hbar**2*JyKetCoupled(1, 1, (1, 1))
    assert qapply(J2*JzKetCoupled(1, 1, (1, 1))) == \
        2*hbar**2*JzKetCoupled(1, 1, (1, 1))
    # Symbolic
    assert qapply(J2*JxKetCoupled(j, m, (j1, j2))) == \
        hbar**2*j**2*JxKetCoupled(j, m, (j1, j2)) + \
        hbar**2*j*JxKetCoupled(j, m, (j1, j2))
    assert qapply(J2*JyKetCoupled(j, m, (j1, j2))) == \
        hbar**2*j**2*JyKetCoupled(j, m, (j1, j2)) + \
        hbar**2*j*JyKetCoupled(j, m, (j1, j2))
    assert qapply(J2*JzKetCoupled(j, m, (j1, j2))) == \
        hbar**2*j**2*JzKetCoupled(j, m, (j1, j2)) + \
        hbar**2*j*JzKetCoupled(j, m, (j1, j2))
    # Uncoupled operators, uncoupled states
    # Numerical
    assert qapply(TensorProduct(J2, 1)*TensorProduct(JxKet(1, 1), JxKet(1, -1))) == \
        2*hbar**2*TensorProduct(JxKet(1, 1), JxKet(1, -1))
    assert qapply(TensorProduct(1, J2)*TensorProduct(JxKet(1, 1), JxKet(1, -1))) == \
        2*hbar**2*TensorProduct(JxKet(1, 1), JxKet(1, -1))
    assert qapply(TensorProduct(J2, 1)*TensorProduct(JyKet(1, 1), JyKet(1, -1))) == \
        2*hbar**2*TensorProduct(JyKet(1, 1), JyKet(1, -1))
    assert qapply(TensorProduct(1, J2)*TensorProduct(JyKet(1, 1), JyKet(1, -1))) == \
        2*hbar**2*TensorProduct(JyKet(1, 1), JyKet(1, -1))
    assert qapply(TensorProduct(J2, 1)*TensorProduct(JzKet(1, 1), JzKet(1, -1))) == \
        2*hbar**2*TensorProduct(JzKet(1, 1), JzKet(1, -1))
    assert qapply(TensorProduct(1, J2)*TensorProduct(JzKet(1, 1), JzKet(1, -1))) == \
        2*hbar**2*TensorProduct(JzKet(1, 1), JzKet(1, -1))
    # Symbolic
    assert qapply(TensorProduct(J2, 1)*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == \
        hbar**2*j1**2*TensorProduct(JxKet(j1, m1), JxKet(j2, m2)) + \
        hbar**2*j1*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))
    assert qapply(TensorProduct(1, J2)*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == \
        hbar**2*j2**2*TensorProduct(JxKet(j1, m1), JxKet(j2, m2)) + \
        hbar**2*j2*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))
    assert qapply(TensorProduct(J2, 1)*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == \
        hbar**2*j1**2*TensorProduct(JyKet(j1, m1), JyKet(j2, m2)) + \
        hbar**2*j1*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))
    assert qapply(TensorProduct(1, J2)*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == \
        hbar**2*j2**2*TensorProduct(JyKet(j1, m1), JyKet(j2, m2)) + \
        hbar**2*j2*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))
    assert qapply(TensorProduct(J2, 1)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == \
        hbar**2*j1**2*TensorProduct(JzKet(j1, m1), JzKet(j2, m2)) + \
        hbar**2*j1*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))
    assert qapply(TensorProduct(1, J2)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == \
        hbar**2*j2**2*TensorProduct(JzKet(j1, m1), JzKet(j2, m2)) + \
        hbar**2*j2*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))


def test_jx():
    assert Commutator(Jx, Jz).doit() == -I*hbar*Jy
    assert Jx.rewrite('plusminus') == (Jminus + Jplus)/2
    assert represent(Jx, basis=Jz, j=1) == (
        represent(Jplus, basis=Jz, j=1) + represent(Jminus, basis=Jz, j=1))/2
    # Normal operators, normal states
    # Numerical
    assert qapply(Jx*JxKet(1, 1)) == hbar*JxKet(1, 1)
    assert qapply(Jx*JyKet(1, 1)) == hbar*JyKet(1, 1)
    assert qapply(Jx*JzKet(1, 1)) == sqrt(2)*hbar*JzKet(1, 0)/2
    # Symbolic
    assert qapply(Jx*JxKet(j, m)) == hbar*m*JxKet(j, m)
    assert qapply(Jx*JyKet(j, m)) == \
        Sum(hbar*mi*WignerD(j, mi, m, 0, 0, pi/2)*Sum(WignerD(j,
            mi1, mi, pi*Rational(3, 2), 0, 0)*JyKet(j, mi1), (mi1, -j, j)), (mi, -j, j))
    assert qapply(Jx*JzKet(j, m)) == \
        hbar*sqrt(j**2 + j - m**2 - m)*JzKet(j, m + 1)/2 + hbar*sqrt(j**2 +
                  j - m**2 + m)*JzKet(j, m - 1)/2
    # Normal operators, coupled states
    # Numerical
    assert qapply(Jx*JxKetCoupled(1, 1, (1, 1))) == \
        hbar*JxKetCoupled(1, 1, (1, 1))
    assert qapply(Jx*JyKetCoupled(1, 1, (1, 1))) == \
        hbar*JyKetCoupled(1, 1, (1, 1))
    assert qapply(Jx*JzKetCoupled(1, 1, (1, 1))) == \
        sqrt(2)*hbar*JzKetCoupled(1, 0, (1, 1))/2
    # Symbolic
    assert qapply(Jx*JxKetCoupled(j, m, (j1, j2))) == \
        hbar*m*JxKetCoupled(j, m, (j1, j2))
    assert qapply(Jx*JyKetCoupled(j, m, (j1, j2))) == \
        Sum(hbar*mi*WignerD(j, mi, m, 0, 0, pi/2)*Sum(WignerD(j, mi1, mi, pi*Rational(3, 2), 0, 0)*JyKetCoupled(j, mi1, (j1, j2)), (mi1, -j, j)), (mi, -j, j))
    assert qapply(Jx*JzKetCoupled(j, m, (j1, j2))) == \
        hbar*sqrt(j**2 + j - m**2 - m)*JzKetCoupled(j, m + 1, (j1, j2))/2 + \
        hbar*sqrt(j**2 + j - m**2 + m)*JzKetCoupled(j, m - 1, (j1, j2))/2
    # Normal operators, uncoupled states
    # Numerical
    assert qapply(Jx*TensorProduct(JxKet(1, 1), JxKet(1, 1))) == \
        2*hbar*TensorProduct(JxKet(1, 1), JxKet(1, 1))
    assert qapply(Jx*TensorProduct(JyKet(1, 1), JyKet(1, 1))) == \
        hbar*TensorProduct(JyKet(1, 1), JyKet(1, 1)) + \
        hbar*TensorProduct(JyKet(1, 1), JyKet(1, 1))
    assert qapply(Jx*TensorProduct(JzKet(1, 1), JzKet(1, 1))) == \
        sqrt(2)*hbar*TensorProduct(JzKet(1, 1), JzKet(1, 0))/2 + \
        sqrt(2)*hbar*TensorProduct(JzKet(1, 0), JzKet(1, 1))/2
    assert qapply(Jx*TensorProduct(JxKet(1, 1), JxKet(1, -1))) == 0
    # Symbolic
    assert qapply(Jx*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == \
        hbar*m1*TensorProduct(JxKet(j1, m1), JxKet(j2, m2)) + \
        hbar*m2*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))
    assert qapply(Jx*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == \
        TensorProduct(Sum(hbar*mi*WignerD(j1, mi, m1, 0, 0, pi/2)*Sum(WignerD(j1, mi1, mi, pi*Rational(3, 2), 0, 0)*JyKet(j1, mi1), (mi1, -j1, j1)), (mi, -j1, j1)), JyKet(j2, m2)) + \
        TensorProduct(JyKet(j1, m1), Sum(hbar*mi*WignerD(j2, mi, m2, 0, 0, pi/2)*Sum(WignerD(j2, mi1, mi, pi*Rational(3, 2), 0, 0)*JyKet(j2, mi1), (mi1, -j2, j2)), (mi, -j2, j2)))
    assert qapply(Jx*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == \
        hbar*sqrt(j1**2 + j1 - m1**2 - m1)*TensorProduct(JzKet(j1, m1 + 1), JzKet(j2, m2))/2 + \
        hbar*sqrt(j1**2 + j1 - m1**2 + m1)*TensorProduct(JzKet(j1, m1 - 1), JzKet(j2, m2))/2 + \
        hbar*sqrt(j2**2 + j2 - m2**2 - m2)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2 + 1))/2 + \
        hbar*sqrt(
            j2**2 + j2 - m2**2 + m2)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2 - 1))/2
    # Uncoupled operators, uncoupled states
    # Numerical
    assert qapply(TensorProduct(Jx, 1)*TensorProduct(JxKet(1, 1), JxKet(1, -1))) == \
        hbar*TensorProduct(JxKet(1, 1), JxKet(1, -1))
    assert qapply(TensorProduct(1, Jx)*TensorProduct(JxKet(1, 1), JxKet(1, -1))) == \
        -hbar*TensorProduct(JxKet(1, 1), JxKet(1, -1))
    assert qapply(TensorProduct(Jx, 1)*TensorProduct(JyKet(1, 1), JyKet(1, -1))) == \
        hbar*TensorProduct(JyKet(1, 1), JyKet(1, -1))
    assert qapply(TensorProduct(1, Jx)*TensorProduct(JyKet(1, 1), JyKet(1, -1))) == \
        -hbar*TensorProduct(JyKet(1, 1), JyKet(1, -1))
    assert qapply(TensorProduct(Jx, 1)*TensorProduct(JzKet(1, 1), JzKet(1, -1))) == \
        hbar*sqrt(2)*TensorProduct(JzKet(1, 0), JzKet(1, -1))/2
    assert qapply(TensorProduct(1, Jx)*TensorProduct(JzKet(1, 1), JzKet(1, -1))) == \
        hbar*sqrt(2)*TensorProduct(JzKet(1, 1), JzKet(1, 0))/2
    # Symbolic
    assert qapply(TensorProduct(Jx, 1)*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == \
        hbar*m1*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))
    assert qapply(TensorProduct(1, Jx)*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == \
        hbar*m2*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))
    assert qapply(TensorProduct(Jx, 1)*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == \
        TensorProduct(Sum(hbar*mi*WignerD(j1, mi, m1, 0, 0, pi/2) * Sum(WignerD(j1, mi1, mi, pi*Rational(3, 2), 0, 0)*JyKet(j1, mi1), (mi1, -j1, j1)), (mi, -j1, j1)), JyKet(j2, m2))
    assert qapply(TensorProduct(1, Jx)*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == \
        TensorProduct(JyKet(j1, m1), Sum(hbar*mi*WignerD(j2, mi, m2, 0, 0, pi/2) * Sum(WignerD(j2, mi1, mi, pi*Rational(3, 2), 0, 0)*JyKet(j2, mi1), (mi1, -j2, j2)), (mi, -j2, j2)))
    assert qapply(TensorProduct(Jx, 1)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == \
        hbar*sqrt(j1**2 + j1 - m1**2 - m1)*TensorProduct(JzKet(j1, m1 + 1), JzKet(j2, m2))/2 + \
        hbar*sqrt(
            j1**2 + j1 - m1**2 + m1)*TensorProduct(JzKet(j1, m1 - 1), JzKet(j2, m2))/2
    assert qapply(TensorProduct(1, Jx)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == \
        hbar*sqrt(j2**2 + j2 - m2**2 - m2)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2 + 1))/2 + \
        hbar*sqrt(
            j2**2 + j2 - m2**2 + m2)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2 - 1))/2


def test_jy():
    assert Commutator(Jy, Jz).doit() == I*hbar*Jx
    assert Jy.rewrite('plusminus') == (Jplus - Jminus)/(2*I)
    assert represent(Jy, basis=Jz) == (
        represent(Jplus, basis=Jz) - represent(Jminus, basis=Jz))/(2*I)
    # Normal operators, normal states
    # Numerical
    assert qapply(Jy*JxKet(1, 1)) == hbar*JxKet(1, 1)
    assert qapply(Jy*JyKet(1, 1)) == hbar*JyKet(1, 1)
    assert qapply(Jy*JzKet(1, 1)) == sqrt(2)*hbar*I*JzKet(1, 0)/2
    # Symbolic
    assert qapply(Jy*JxKet(j, m)) == \
        Sum(hbar*mi*WignerD(j, mi, m, pi*Rational(3, 2), 0, 0)*Sum(WignerD(
            j, mi1, mi, 0, 0, pi/2)*JxKet(j, mi1), (mi1, -j, j)), (mi, -j, j))
    assert qapply(Jy*JyKet(j, m)) == hbar*m*JyKet(j, m)
    assert qapply(Jy*JzKet(j, m)) == \
        -hbar*I*sqrt(j**2 + j - m**2 - m)*JzKet(
            j, m + 1)/2 + hbar*I*sqrt(j**2 + j - m**2 + m)*JzKet(j, m - 1)/2
    # Normal operators, coupled states
    # Numerical
    assert qapply(Jy*JxKetCoupled(1, 1, (1, 1))) == \
        hbar*JxKetCoupled(1, 1, (1, 1))
    assert qapply(Jy*JyKetCoupled(1, 1, (1, 1))) == \
        hbar*JyKetCoupled(1, 1, (1, 1))
    assert qapply(Jy*JzKetCoupled(1, 1, (1, 1))) == \
        sqrt(2)*hbar*I*JzKetCoupled(1, 0, (1, 1))/2
    # Symbolic
    assert qapply(Jy*JxKetCoupled(j, m, (j1, j2))) == \
        Sum(hbar*mi*WignerD(j, mi, m, pi*Rational(3, 2), 0, 0)*Sum(WignerD(j, mi1, mi, 0, 0, pi/2)*JxKetCoupled(j, mi1, (j1, j2)), (mi1, -j, j)), (mi, -j, j))
    assert qapply(Jy*JyKetCoupled(j, m, (j1, j2))) == \
        hbar*m*JyKetCoupled(j, m, (j1, j2))
    assert qapply(Jy*JzKetCoupled(j, m, (j1, j2))) == \
        -hbar*I*sqrt(j**2 + j - m**2 - m)*JzKetCoupled(j, m + 1, (j1, j2))/2 + \
        hbar*I*sqrt(j**2 + j - m**2 + m)*JzKetCoupled(j, m - 1, (j1, j2))/2
    # Normal operators, uncoupled states
    # Numerical
    assert qapply(Jy*TensorProduct(JxKet(1, 1), JxKet(1, 1))) == \
        hbar*TensorProduct(JxKet(1, 1), JxKet(1, 1)) + \
        hbar*TensorProduct(JxKet(1, 1), JxKet(1, 1))
    assert qapply(Jy*TensorProduct(JyKet(1, 1), JyKet(1, 1))) == \
        2*hbar*TensorProduct(JyKet(1, 1), JyKet(1, 1))
    assert qapply(Jy*TensorProduct(JzKet(1, 1), JzKet(1, 1))) == \
        sqrt(2)*hbar*I*TensorProduct(JzKet(1, 1), JzKet(1, 0))/2 + \
        sqrt(2)*hbar*I*TensorProduct(JzKet(1, 0), JzKet(1, 1))/2
    assert qapply(Jy*TensorProduct(JyKet(1, 1), JyKet(1, -1))) == 0
    # Symbolic
    assert qapply(Jy*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == \
        TensorProduct(JxKet(j1, m1), Sum(hbar*mi*WignerD(j2, mi, m2, pi*Rational(3, 2), 0, 0)*Sum(WignerD(j2, mi1, mi, 0, 0, pi/2)*JxKet(j2, mi1), (mi1, -j2, j2)), (mi, -j2, j2))) + \
        TensorProduct(Sum(hbar*mi*WignerD(j1, mi, m1, pi*Rational(3, 2), 0, 0)*Sum(WignerD(j1, mi1, mi, 0, 0, pi/2)*JxKet(j1, mi1), (mi1, -j1, j1)), (mi, -j1, j1)), JxKet(j2, m2))
    assert qapply(Jy*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == \
        hbar*m1*TensorProduct(JyKet(j1, m1), JyKet(
            j2, m2)) + hbar*m2*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))
    assert qapply(Jy*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == \
        -hbar*I*sqrt(j1**2 + j1 - m1**2 - m1)*TensorProduct(JzKet(j1, m1 + 1), JzKet(j2, m2))/2 + \
        hbar*I*sqrt(j1**2 + j1 - m1**2 + m1)*TensorProduct(JzKet(j1, m1 - 1), JzKet(j2, m2))/2 + \
        -hbar*I*sqrt(j2**2 + j2 - m2**2 - m2)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2 + 1))/2 + \
        hbar*I*sqrt(
            j2**2 + j2 - m2**2 + m2)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2 - 1))/2
    # Uncoupled operators, uncoupled states
    # Numerical
    assert qapply(TensorProduct(Jy, 1)*TensorProduct(JxKet(1, 1), JxKet(1, -1))) == \
        hbar*TensorProduct(JxKet(1, 1), JxKet(1, -1))
    assert qapply(TensorProduct(1, Jy)*TensorProduct(JxKet(1, 1), JxKet(1, -1))) == \
        -hbar*TensorProduct(JxKet(1, 1), JxKet(1, -1))
    assert qapply(TensorProduct(Jy, 1)*TensorProduct(JyKet(1, 1), JyKet(1, -1))) == \
        hbar*TensorProduct(JyKet(1, 1), JyKet(1, -1))
    assert qapply(TensorProduct(1, Jy)*TensorProduct(JyKet(1, 1), JyKet(1, -1))) == \
        -hbar*TensorProduct(JyKet(1, 1), JyKet(1, -1))
    assert qapply(TensorProduct(Jy, 1)*TensorProduct(JzKet(1, 1), JzKet(1, -1))) == \
        hbar*sqrt(2)*I*TensorProduct(JzKet(1, 0), JzKet(1, -1))/2
    assert qapply(TensorProduct(1, Jy)*TensorProduct(JzKet(1, 1), JzKet(1, -1))) == \
        -hbar*sqrt(2)*I*TensorProduct(JzKet(1, 1), JzKet(1, 0))/2
    # Symbolic
    assert qapply(TensorProduct(Jy, 1)*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == \
        TensorProduct(Sum(hbar*mi*WignerD(j1, mi, m1, pi*Rational(3, 2), 0, 0) * Sum(WignerD(j1, mi1, mi, 0, 0, pi/2)*JxKet(j1, mi1), (mi1, -j1, j1)), (mi, -j1, j1)), JxKet(j2, m2))
    assert qapply(TensorProduct(1, Jy)*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == \
        TensorProduct(JxKet(j1, m1), Sum(hbar*mi*WignerD(j2, mi, m2, pi*Rational(3, 2), 0, 0) * Sum(WignerD(j2, mi1, mi, 0, 0, pi/2)*JxKet(j2, mi1), (mi1, -j2, j2)), (mi, -j2, j2)))
    assert qapply(TensorProduct(Jy, 1)*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == \
        hbar*m1*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))
    assert qapply(TensorProduct(1, Jy)*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == \
        hbar*m2*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))
    assert qapply(TensorProduct(Jy, 1)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == \
        -hbar*I*sqrt(j1**2 + j1 - m1**2 - m1)*TensorProduct(JzKet(j1, m1 + 1), JzKet(j2, m2))/2 + \
        hbar*I*sqrt(
            j1**2 + j1 - m1**2 + m1)*TensorProduct(JzKet(j1, m1 - 1), JzKet(j2, m2))/2
    assert qapply(TensorProduct(1, Jy)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == \
        -hbar*I*sqrt(j2**2 + j2 - m2**2 - m2)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2 + 1))/2 + \
        hbar*I*sqrt(
            j2**2 + j2 - m2**2 + m2)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2 - 1))/2


def test_jz():
    assert Commutator(Jz, Jminus).doit() == -hbar*Jminus
    # Normal operators, normal states
    # Numerical
    assert qapply(Jz*JxKet(1, 1)) == -sqrt(2)*hbar*JxKet(1, 0)/2
    assert qapply(Jz*JyKet(1, 1)) == -sqrt(2)*hbar*I*JyKet(1, 0)/2
    assert qapply(Jz*JzKet(2, 1)) == hbar*JzKet(2, 1)
    # Symbolic
    assert qapply(Jz*JxKet(j, m)) == \
        Sum(hbar*mi*WignerD(j, mi, m, 0, pi/2, 0)*Sum(WignerD(j,
            mi1, mi, 0, pi*Rational(3, 2), 0)*JxKet(j, mi1), (mi1, -j, j)), (mi, -j, j))
    assert qapply(Jz*JyKet(j, m)) == \
        Sum(hbar*mi*WignerD(j, mi, m, pi*Rational(3, 2), -pi/2, pi/2)*Sum(WignerD(j, mi1,
            mi, pi*Rational(3, 2), pi/2, pi/2)*JyKet(j, mi1), (mi1, -j, j)), (mi, -j, j))
    assert qapply(Jz*JzKet(j, m)) == hbar*m*JzKet(j, m)
    # Normal operators, coupled states
    # Numerical
    assert qapply(Jz*JxKetCoupled(1, 1, (1, 1))) == \
        -sqrt(2)*hbar*JxKetCoupled(1, 0, (1, 1))/2
    assert qapply(Jz*JyKetCoupled(1, 1, (1, 1))) == \
        -sqrt(2)*hbar*I*JyKetCoupled(1, 0, (1, 1))/2
    assert qapply(Jz*JzKetCoupled(1, 1, (1, 1))) == \
        hbar*JzKetCoupled(1, 1, (1, 1))
    # Symbolic
    assert qapply(Jz*JxKetCoupled(j, m, (j1, j2))) == \
        Sum(hbar*mi*WignerD(j, mi, m, 0, pi/2, 0)*Sum(WignerD(j, mi1, mi, 0, pi*Rational(3, 2), 0)*JxKetCoupled(j, mi1, (j1, j2)), (mi1, -j, j)), (mi, -j, j))
    assert qapply(Jz*JyKetCoupled(j, m, (j1, j2))) == \
        Sum(hbar*mi*WignerD(j, mi, m, pi*Rational(3, 2), -pi/2, pi/2)*Sum(WignerD(j, mi1, mi, pi*Rational(3, 2), pi/2, pi/2)*JyKetCoupled(j, mi1, (j1, j2)), (mi1, -j, j)), (mi, -j, j))
    assert qapply(Jz*JzKetCoupled(j, m, (j1, j2))) == \
        hbar*m*JzKetCoupled(j, m, (j1, j2))
    # Normal operators, uncoupled states
    # Numerical
    assert qapply(Jz*TensorProduct(JxKet(1, 1), JxKet(1, 1))) == \
        -sqrt(2)*hbar*TensorProduct(JxKet(1, 1), JxKet(1, 0))/2 - \
        sqrt(2)*hbar*TensorProduct(JxKet(1, 0), JxKet(1, 1))/2
    assert qapply(Jz*TensorProduct(JyKet(1, 1), JyKet(1, 1))) == \
        -sqrt(2)*hbar*I*TensorProduct(JyKet(1, 1), JyKet(1, 0))/2 - \
        sqrt(2)*hbar*I*TensorProduct(JyKet(1, 0), JyKet(1, 1))/2
    assert qapply(Jz*TensorProduct(JzKet(1, 1), JzKet(1, 1))) == \
        2*hbar*TensorProduct(JzKet(1, 1), JzKet(1, 1))
    assert qapply(Jz*TensorProduct(JzKet(1, 1), JzKet(1, -1))) == 0
    # Symbolic
    assert qapply(Jz*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == \
        TensorProduct(JxKet(j1, m1), Sum(hbar*mi*WignerD(j2, mi, m2, 0, pi/2, 0)*Sum(WignerD(j2, mi1, mi, 0, pi*Rational(3, 2), 0)*JxKet(j2, mi1), (mi1, -j2, j2)), (mi, -j2, j2))) + \
        TensorProduct(Sum(hbar*mi*WignerD(j1, mi, m1, 0, pi/2, 0)*Sum(WignerD(j1, mi1, mi, 0, pi*Rational(3, 2), 0)*JxKet(j1, mi1), (mi1, -j1, j1)), (mi, -j1, j1)), JxKet(j2, m2))
    assert qapply(Jz*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == \
        TensorProduct(JyKet(j1, m1), Sum(hbar*mi*WignerD(j2, mi, m2, pi*Rational(3, 2), -pi/2, pi/2)*Sum(WignerD(j2, mi1, mi, pi*Rational(3, 2), pi/2, pi/2)*JyKet(j2, mi1), (mi1, -j2, j2)), (mi, -j2, j2))) + \
        TensorProduct(Sum(hbar*mi*WignerD(j1, mi, m1, pi*Rational(3, 2), -pi/2, pi/2)*Sum(WignerD(j1, mi1, mi, pi*Rational(3, 2), pi/2, pi/2)*JyKet(j1, mi1), (mi1, -j1, j1)), (mi, -j1, j1)), JyKet(j2, m2))
    assert qapply(Jz*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == \
        hbar*m1*TensorProduct(JzKet(j1, m1), JzKet(
            j2, m2)) + hbar*m2*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))
    # Uncoupled Operators
    # Numerical
    assert qapply(TensorProduct(Jz, 1)*TensorProduct(JxKet(1, 1), JxKet(1, -1))) == \
        -sqrt(2)*hbar*TensorProduct(JxKet(1, 0), JxKet(1, -1))/2
    assert qapply(TensorProduct(1, Jz)*TensorProduct(JxKet(1, 1), JxKet(1, -1))) == \
        -sqrt(2)*hbar*TensorProduct(JxKet(1, 1), JxKet(1, 0))/2
    assert qapply(TensorProduct(Jz, 1)*TensorProduct(JyKet(1, 1), JyKet(1, -1))) == \
        -sqrt(2)*I*hbar*TensorProduct(JyKet(1, 0), JyKet(1, -1))/2
    assert qapply(TensorProduct(1, Jz)*TensorProduct(JyKet(1, 1), JyKet(1, -1))) == \
        sqrt(2)*I*hbar*TensorProduct(JyKet(1, 1), JyKet(1, 0))/2
    assert qapply(TensorProduct(Jz, 1)*TensorProduct(JzKet(1, 1), JzKet(1, -1))) == \
        hbar*TensorProduct(JzKet(1, 1), JzKet(1, -1))
    assert qapply(TensorProduct(1, Jz)*TensorProduct(JzKet(1, 1), JzKet(1, -1))) == \
        -hbar*TensorProduct(JzKet(1, 1), JzKet(1, -1))
    # Symbolic
    assert qapply(TensorProduct(Jz, 1)*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == \
        TensorProduct(Sum(hbar*mi*WignerD(j1, mi, m1, 0, pi/2, 0)*Sum(WignerD(j1, mi1, mi, 0, pi*Rational(3, 2), 0)*JxKet(j1, mi1), (mi1, -j1, j1)), (mi, -j1, j1)), JxKet(j2, m2))
    assert qapply(TensorProduct(1, Jz)*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == \
        TensorProduct(JxKet(j1, m1), Sum(hbar*mi*WignerD(j2, mi, m2, 0, pi/2, 0)*Sum(WignerD(j2, mi1, mi, 0, pi*Rational(3, 2), 0)*JxKet(j2, mi1), (mi1, -j2, j2)), (mi, -j2, j2)))
    assert qapply(TensorProduct(Jz, 1)*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == \
        TensorProduct(Sum(hbar*mi*WignerD(j1, mi, m1, pi*Rational(3, 2), -pi/2, pi/2)*Sum(WignerD(j1, mi1, mi, pi*Rational(3, 2), pi/2, pi/2)*JyKet(j1, mi1), (mi1, -j1, j1)), (mi, -j1, j1)), JyKet(j2, m2))
    assert qapply(TensorProduct(1, Jz)*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == \
        TensorProduct(JyKet(j1, m1), Sum(hbar*mi*WignerD(j2, mi, m2, pi*Rational(3, 2), -pi/2, pi/2)*Sum(WignerD(j2, mi1, mi, pi*Rational(3, 2), pi/2, pi/2)*JyKet(j2, mi1), (mi1, -j2, j2)), (mi, -j2, j2)))
    assert qapply(TensorProduct(Jz, 1)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == \
        hbar*m1*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))
    assert qapply(TensorProduct(1, Jz)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == \
        hbar*m2*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))


def test_rotation():
    a, b, g = symbols('a b g')
    j, m = symbols('j m')
    #Uncoupled
    answ = [JxKet(1,-1)/2 - sqrt(2)*JxKet(1,0)/2 + JxKet(1,1)/2 ,
       JyKet(1,-1)/2 - sqrt(2)*JyKet(1,0)/2 + JyKet(1,1)/2 ,
       JzKet(1,-1)/2 - sqrt(2)*JzKet(1,0)/2 + JzKet(1,1)/2]
    fun = [state(1, 1) for state in (JxKet, JyKet, JzKet)]
    for state in fun:
        got = qapply(Rotation(0, pi/2, 0)*state)
        assert got in answ
        answ.remove(got)
    assert not answ
    arg = Rotation(a, b, g)*fun[0]
    assert qapply(arg) == (-exp(-I*a)*exp(I*g)*cos(b)*JxKet(1,-1)/2 +
        exp(-I*a)*exp(I*g)*JxKet(1,-1)/2 - sqrt(2)*exp(-I*a)*sin(b)*JxKet(1,0)/2 +
        exp(-I*a)*exp(-I*g)*cos(b)*JxKet(1,1)/2 + exp(-I*a)*exp(-I*g)*JxKet(1,1)/2)
    #dummy effective
    assert str(qapply(Rotation(a, b, g)*JzKet(j, m), dummy=False)) == str(
        qapply(Rotation(a, b, g)*JzKet(j, m), dummy=True)).replace('_','')
    #Coupled
    ans = [JxKetCoupled(1,-1,(1,1))/2 - sqrt(2)*JxKetCoupled(1,0,(1,1))/2 +
       JxKetCoupled(1,1,(1,1))/2 ,
       JyKetCoupled(1,-1,(1,1))/2 - sqrt(2)*JyKetCoupled(1,0,(1,1))/2 +
       JyKetCoupled(1,1,(1,1))/2 ,
       JzKetCoupled(1,-1,(1,1))/2 - sqrt(2)*JzKetCoupled(1,0,(1,1))/2 +
       JzKetCoupled(1,1,(1,1))/2]
    fun = [state(1, 1, (1,1)) for state in (JxKetCoupled, JyKetCoupled, JzKetCoupled)]
    for state in fun:
        got = qapply(Rotation(0, pi/2, 0)*state)
        assert got in ans
        ans.remove(got)
    assert not ans
    arg = Rotation(a, b, g)*fun[0]
    assert qapply(arg) == (
        -exp(-I*a)*exp(I*g)*cos(b)*JxKetCoupled(1,-1,(1,1))/2 +
        exp(-I*a)*exp(I*g)*JxKetCoupled(1,-1,(1,1))/2 -
        sqrt(2)*exp(-I*a)*sin(b)*JxKetCoupled(1,0,(1,1))/2 +
        exp(-I*a)*exp(-I*g)*cos(b)*JxKetCoupled(1,1,(1,1))/2 +
        exp(-I*a)*exp(-I*g)*JxKetCoupled(1,1,(1,1))/2)
    #dummy effective
    assert str(qapply(Rotation(a,b,g)*JzKetCoupled(j,m,(j1,j2)), dummy=False)) == str(
        qapply(Rotation(a,b,g)*JzKetCoupled(j,m,(j1,j2)), dummy=True)).replace('_','')


def test_jzket():
    j, m = symbols('j m')
    # j not integer or half integer
    raises(ValueError, lambda: JzKet(Rational(2, 3), Rational(-1, 3)))
    raises(ValueError, lambda: JzKet(Rational(2, 3), m))
    # j < 0
    raises(ValueError, lambda: JzKet(-1, 1))
    raises(ValueError, lambda: JzKet(-1, m))
    # m not integer or half integer
    raises(ValueError, lambda: JzKet(j, Rational(-1, 3)))
    # abs(m) > j
    raises(ValueError, lambda: JzKet(1, 2))
    raises(ValueError, lambda: JzKet(1, -2))
    # j-m not integer
    raises(ValueError, lambda: JzKet(1, S.Half))


def test_jzketcoupled():
    j, m = symbols('j m')
    # j not integer or half integer
    raises(ValueError, lambda: JzKetCoupled(Rational(2, 3), Rational(-1, 3), (1,)))
    raises(ValueError, lambda: JzKetCoupled(Rational(2, 3), m, (1,)))
    # j < 0
    raises(ValueError, lambda: JzKetCoupled(-1, 1, (1,)))
    raises(ValueError, lambda: JzKetCoupled(-1, m, (1,)))
    # m not integer or half integer
    raises(ValueError, lambda: JzKetCoupled(j, Rational(-1, 3), (1,)))
    # abs(m) > j
    raises(ValueError, lambda: JzKetCoupled(1, 2, (1,)))
    raises(ValueError, lambda: JzKetCoupled(1, -2, (1,)))
    # j-m not integer
    raises(ValueError, lambda: JzKetCoupled(1, S.Half, (1,)))
    # checks types on coupling scheme
    raises(TypeError, lambda: JzKetCoupled(1, 1, 1))
    raises(TypeError, lambda: JzKetCoupled(1, 1, (1,), 1))
    raises(TypeError, lambda: JzKetCoupled(1, 1, (1, 1), (1,)))
    raises(TypeError, lambda: JzKetCoupled(1, 1, (1, 1, 1), (1, 2, 1),
           (1, 3, 1)))
    # checks length of coupling terms
    raises(ValueError, lambda: JzKetCoupled(1, 1, (1,), ((1, 2, 1),)))
    raises(ValueError, lambda: JzKetCoupled(1, 1, (1, 1), ((1, 2),)))
    # all jn are integer or half-integer
    raises(ValueError, lambda: JzKetCoupled(1, 1, (Rational(1, 3), Rational(2, 3))))
    # indices in coupling scheme must be integers
    raises(ValueError, lambda: JzKetCoupled(1, 1, (1, 1), ((S.Half, 1, 2),) ))
    raises(ValueError, lambda: JzKetCoupled(1, 1, (1, 1), ((1, S.Half, 2),) ))
    # indices out of range
    raises(ValueError, lambda: JzKetCoupled(1, 1, (1, 1), ((0, 2, 1),) ))
    raises(ValueError, lambda: JzKetCoupled(1, 1, (1, 1), ((3, 2, 1),) ))
    raises(ValueError, lambda: JzKetCoupled(1, 1, (1, 1), ((1, 0, 1),) ))
    raises(ValueError, lambda: JzKetCoupled(1, 1, (1, 1), ((1, 3, 1),) ))
    # all j values in coupling scheme must by integer or half-integer
    raises(ValueError, lambda: JzKetCoupled(1, 1, (1, 1, 1), ((1, 2, S(
        4)/3), (1, 3, 1)) ))
    # each coupling must satisfy |j1-j2| <= j3 <= j1+j2
    raises(ValueError, lambda: JzKetCoupled(1, 1, (1, 5)))
    raises(ValueError, lambda: JzKetCoupled(5, 1, (1, 1)))
    # final j of coupling must be j of the state
    raises(ValueError, lambda: JzKetCoupled(1, 1, (1, 1), ((1, 2, 2),) ))
