"""Tests for cartesian.py"""

from sympy.core.numbers import (I, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.delta_functions import DiracDelta
from sympy.sets.sets import Interval
from sympy.testing.pytest import XFAIL

from sympy.physics.quantum import qapply, represent, L2, Dagger
from sympy.physics.quantum import Commutator, hbar
from sympy.physics.quantum.cartesian import (
    XOp, YOp, ZOp, PxOp, X, Y, Z, Px, XKet, XBra, PxKet, PxBra,
    PositionKet3D, PositionBra3D
)
from sympy.physics.quantum.operator import DifferentialOperator

x, y, z, x_1, x_2, x_3, y_1, z_1 = symbols('x,y,z,x_1,x_2,x_3,y_1,z_1')
px, py, px_1, px_2 = symbols('px py px_1 px_2')


def test_x():
    assert X.hilbert_space == L2(Interval(S.NegativeInfinity, S.Infinity))
    assert Commutator(X, Px).doit() == I*hbar
    assert qapply(X*XKet(x)) == x*XKet(x)
    assert XKet(x).dual_class() == XBra
    assert XBra(x).dual_class() == XKet
    assert (Dagger(XKet(y))*XKet(x)).doit() == DiracDelta(x - y)
    assert (PxBra(px)*XKet(x)).doit() == \
        exp(-I*x*px/hbar)/sqrt(2*pi*hbar)
    assert represent(XKet(x)) == DiracDelta(x - x_1)
    assert represent(XBra(x)) == DiracDelta(-x + x_1)
    assert XBra(x).position == x
    assert represent(XOp()*XKet()) == x*DiracDelta(x - x_2)
    assert represent(XBra("y")*XKet()) == DiracDelta(x - y)
    assert represent(
        XKet()*XBra()) == DiracDelta(x - x_2) * DiracDelta(x_1 - x)

    rep_p = represent(XOp(), basis=PxOp)
    assert rep_p == hbar*I*DiracDelta(px_1 - px_2)*DifferentialOperator(px_1)
    assert rep_p == represent(XOp(), basis=PxOp())
    assert rep_p == represent(XOp(), basis=PxKet)
    assert rep_p == represent(XOp(), basis=PxKet())

    assert represent(XOp()*PxKet(), basis=PxKet) == \
        hbar*I*DiracDelta(px - px_2)*DifferentialOperator(px)


@XFAIL
def _text_x_broken():
    # represent has some broken logic that is relying in particular
    # forms of input, rather than a full and proper handling of
    # all valid quantum expressions. Marking this test as XFAIL until
    # we can refactor represent.
    assert represent(XOp()*XKet()*XBra('y')) == \
        x*DiracDelta(x - x_3)*DiracDelta(x_1 - y)


def test_p():
    assert Px.hilbert_space == L2(Interval(S.NegativeInfinity, S.Infinity))
    assert qapply(Px*PxKet(px)) == px*PxKet(px)
    assert PxKet(px).dual_class() == PxBra
    assert PxBra(x).dual_class() == PxKet
    assert (Dagger(PxKet(py))*PxKet(px)).doit() == DiracDelta(px - py)
    assert (XBra(x)*PxKet(px)).doit() == \
        exp(I*x*px/hbar)/sqrt(2*pi*hbar)
    assert represent(PxKet(px)) == DiracDelta(px - px_1)

    rep_x = represent(PxOp(), basis=XOp)
    assert rep_x == -hbar*I*DiracDelta(x_1 - x_2)*DifferentialOperator(x_1)
    assert rep_x == represent(PxOp(), basis=XOp())
    assert rep_x == represent(PxOp(), basis=XKet)
    assert rep_x == represent(PxOp(), basis=XKet())

    assert represent(PxOp()*XKet(), basis=XKet) == \
        -hbar*I*DiracDelta(x - x_2)*DifferentialOperator(x)
    assert represent(XBra("y")*PxOp()*XKet(), basis=XKet) == \
        -hbar*I*DiracDelta(x - y)*DifferentialOperator(x)


def test_3dpos():
    assert Y.hilbert_space == L2(Interval(S.NegativeInfinity, S.Infinity))
    assert Z.hilbert_space == L2(Interval(S.NegativeInfinity, S.Infinity))

    test_ket = PositionKet3D(x, y, z)
    assert qapply(X*test_ket) == x*test_ket
    assert qapply(Y*test_ket) == y*test_ket
    assert qapply(Z*test_ket) == z*test_ket
    assert qapply(X*Y*test_ket) == x*y*test_ket
    assert qapply(X*Y*Z*test_ket) == x*y*z*test_ket
    assert qapply(Y*Z*test_ket) == y*z*test_ket

    assert PositionKet3D() == test_ket
    assert YOp() == Y
    assert ZOp() == Z

    assert PositionKet3D.dual_class() == PositionBra3D
    assert PositionBra3D.dual_class() == PositionKet3D

    other_ket = PositionKet3D(x_1, y_1, z_1)
    assert (Dagger(other_ket)*test_ket).doit() == \
        DiracDelta(x - x_1)*DiracDelta(y - y_1)*DiracDelta(z - z_1)

    assert test_ket.position_x == x
    assert test_ket.position_y == y
    assert test_ket.position_z == z
    assert other_ket.position_x == x_1
    assert other_ket.position_y == y_1
    assert other_ket.position_z == z_1

    # TODO: Add tests for representations
