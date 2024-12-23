"""Tests for piab.py"""

from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.sets.sets import Interval
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.quantum import L2, qapply, hbar, represent
from sympy.physics.quantum.piab import PIABHamiltonian, PIABKet, PIABBra, m, L

i, j, n, x = symbols('i j n x')


def test_H():
    assert PIABHamiltonian('H').hilbert_space == \
        L2(Interval(S.NegativeInfinity, S.Infinity))
    assert qapply(PIABHamiltonian('H')*PIABKet(n)) == \
        (n**2*pi**2*hbar**2)/(2*m*L**2)*PIABKet(n)


def test_states():
    assert PIABKet(n).dual_class() == PIABBra
    assert PIABKet(n).hilbert_space == \
        L2(Interval(S.NegativeInfinity, S.Infinity))
    assert represent(PIABKet(n)) == sqrt(2/L)*sin(n*pi*x/L)
    assert (PIABBra(i)*PIABKet(j)).doit() == KroneckerDelta(i, j)
    assert PIABBra(n).dual_class() == PIABKet
