"""Kinds for Operators, Bras, and Kets.

This module defines kinds for operators, bras, and kets. These are useful
in various places in ``sympy.physics.quantum`` as you often want to know
what the kind is of a compound expression. For example, if you multiply
an operator, bra, or ket by a number, you get back another operator, bra,
or ket - even though if you did an ``isinstance`` check you would find that
you have a ``Mul`` instead. The kind system is meant to give you a quick
way of determining how a compound expression behaves in terms of lower
level kinds.

The resolution calculation of kinds for compound expressions can be found
either in container classes or in functions that are registered with
kind dispatchers.
"""

from sympy.core.mul import Mul
from sympy.core.kind import Kind, _NumberKind


__all__ = [
    '_KetKind',
    'KetKind',
    '_BraKind',
    'BraKind',
    '_OperatorKind',
    'OperatorKind',
]


class _KetKind(Kind):
    """A kind for quantum kets."""

    def __new__(cls):
        obj = super().__new__(cls)
        return obj

    def __repr__(self):
        return "KetKind"

# Create an instance as many situations need this.
KetKind = _KetKind()


class _BraKind(Kind):
    """A kind for quantum bras."""

    def __new__(cls):
        obj = super().__new__(cls)
        return obj

    def __repr__(self):
        return "BraKind"

# Create an instance as many situations need this.
BraKind = _BraKind()


from sympy.core.kind import Kind

class _OperatorKind(Kind):
    """A kind for quantum operators."""

    def __new__(cls):
        obj = super().__new__(cls)
        return obj

    def __repr__(self):
        return "OperatorKind"

# Create an instance as many situations need this.
OperatorKind = _OperatorKind()


#-----------------------------------------------------------------------------
# Kind resolution.
#-----------------------------------------------------------------------------

# Note: We can't currently add kind dispatchers for the following combinations
#       as the Mul._kind_dispatcher is set to commutative and will also
#       register the opposite order, which isn't correct for these pairs:
#
# 1. (_OperatorKind, _KetKind)
# 2. (_BraKind, _OperatorKind)
# 3. (_BraKind, _KetKind)


@Mul._kind_dispatcher.register(_NumberKind, _KetKind)
def _mul_number_ket_kind(lhs, rhs):
    """Perform the kind calculation of NumberKind*KetKind -> KetKind."""
    return KetKind


@Mul._kind_dispatcher.register(_NumberKind, _BraKind)
def _mul_number_bra_kind(lhs, rhs):
    """Perform the kind calculation of NumberKind*BraKind -> BraKind."""
    return BraKind


@Mul._kind_dispatcher.register(_NumberKind, _OperatorKind)
def _mul_operator_kind(lhs, rhs):
    """Perform the kind calculation of NumberKind*OperatorKind -> OperatorKind."""
    return OperatorKind
