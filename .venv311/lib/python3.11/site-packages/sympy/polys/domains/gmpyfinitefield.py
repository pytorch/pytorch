"""Implementation of :class:`GMPYFiniteField` class. """


from sympy.polys.domains.finitefield import FiniteField
from sympy.polys.domains.gmpyintegerring import GMPYIntegerRing

from sympy.utilities import public

@public
class GMPYFiniteField(FiniteField):
    """Finite field based on GMPY integers. """

    alias = 'FF_gmpy'

    def __init__(self, mod, symmetric=True):
        super().__init__(mod, GMPYIntegerRing(), symmetric)
