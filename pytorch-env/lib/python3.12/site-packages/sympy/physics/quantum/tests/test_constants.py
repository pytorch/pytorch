from sympy.core.numbers import Float

from sympy.physics.quantum.constants import hbar


def test_hbar():
    assert hbar.is_commutative is True
    assert hbar.is_real is True
    assert hbar.is_positive is True
    assert hbar.is_negative is False
    assert hbar.is_irrational is True

    assert hbar.evalf() == Float(1.05457162e-34)
