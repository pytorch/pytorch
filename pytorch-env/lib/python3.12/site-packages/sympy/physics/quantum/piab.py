"""1D quantum particle in a box."""

from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.sets.sets import Interval

from sympy.physics.quantum.operator import HermitianOperator
from sympy.physics.quantum.state import Ket, Bra
from sympy.physics.quantum.constants import hbar
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.quantum.hilbert import L2

m = Symbol('m')
L = Symbol('L')


__all__ = [
    'PIABHamiltonian',
    'PIABKet',
    'PIABBra'
]


class PIABHamiltonian(HermitianOperator):
    """Particle in a box Hamiltonian operator."""

    @classmethod
    def _eval_hilbert_space(cls, label):
        return L2(Interval(S.NegativeInfinity, S.Infinity))

    def _apply_operator_PIABKet(self, ket, **options):
        n = ket.label[0]
        return (n**2*pi**2*hbar**2)/(2*m*L**2)*ket


class PIABKet(Ket):
    """Particle in a box eigenket."""

    @classmethod
    def _eval_hilbert_space(cls, args):
        return L2(Interval(S.NegativeInfinity, S.Infinity))

    @classmethod
    def dual_class(self):
        return PIABBra

    def _represent_default_basis(self, **options):
        return self._represent_XOp(None, **options)

    def _represent_XOp(self, basis, **options):
        x = Symbol('x')
        n = Symbol('n')
        subs_info = options.get('subs', {})
        return sqrt(2/L)*sin(n*pi*x/L).subs(subs_info)

    def _eval_innerproduct_PIABBra(self, bra):
        return KroneckerDelta(bra.label[0], self.label[0])


class PIABBra(Bra):
    """Particle in a box eigenbra."""

    @classmethod
    def _eval_hilbert_space(cls, label):
        return L2(Interval(S.NegativeInfinity, S.Infinity))

    @classmethod
    def dual_class(self):
        return PIABKet
