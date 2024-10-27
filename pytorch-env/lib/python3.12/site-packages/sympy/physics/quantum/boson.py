"""Bosonic quantum operators."""

from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.quantum import Operator
from sympy.physics.quantum import HilbertSpace, FockSpace, Ket, Bra, IdentityOperator
from sympy.functions.special.tensor_functions import KroneckerDelta


__all__ = [
    'BosonOp',
    'BosonFockKet',
    'BosonFockBra',
    'BosonCoherentKet',
    'BosonCoherentBra'
]


class BosonOp(Operator):
    """A bosonic operator that satisfies [a, Dagger(a)] == 1.

    Parameters
    ==========

    name : str
        A string that labels the bosonic mode.

    annihilation : bool
        A bool that indicates if the bosonic operator is an annihilation (True,
        default value) or creation operator (False)

    Examples
    ========

    >>> from sympy.physics.quantum import Dagger, Commutator
    >>> from sympy.physics.quantum.boson import BosonOp
    >>> a = BosonOp("a")
    >>> Commutator(a, Dagger(a)).doit()
    1
    """

    @property
    def name(self):
        return self.args[0]

    @property
    def is_annihilation(self):
        return bool(self.args[1])

    @classmethod
    def default_args(self):
        return ("a", True)

    def __new__(cls, *args, **hints):
        if not len(args) in [1, 2]:
            raise ValueError('1 or 2 parameters expected, got %s' % args)

        if len(args) == 1:
            args = (args[0], S.One)

        if len(args) == 2:
            args = (args[0], Integer(args[1]))

        return Operator.__new__(cls, *args)

    def _eval_commutator_BosonOp(self, other, **hints):
        if self.name == other.name:
            # [a^\dagger, a] = -1
            if not self.is_annihilation and other.is_annihilation:
                return S.NegativeOne

        elif 'independent' in hints and hints['independent']:
            # [a, b] = 0
            return S.Zero

        return None

    def _eval_commutator_FermionOp(self, other, **hints):
        return S.Zero

    def _eval_anticommutator_BosonOp(self, other, **hints):
        if 'independent' in hints and hints['independent']:
            # {a, b} = 2 * a * b, because [a, b] = 0
            return 2 * self * other

        return None

    def _eval_adjoint(self):
        return BosonOp(str(self.name), not self.is_annihilation)

    def __mul__(self, other):

        if other == IdentityOperator(2):
            return self

        if isinstance(other, Mul):
            args1 = tuple(arg for arg in other.args if arg.is_commutative)
            args2 = tuple(arg for arg in other.args if not arg.is_commutative)
            x = self
            for y in args2:
                x = x * y
            return Mul(*args1) * x

        return Mul(self, other)

    def _print_contents_latex(self, printer, *args):
        if self.is_annihilation:
            return r'{%s}' % str(self.name)
        else:
            return r'{{%s}^\dagger}' % str(self.name)

    def _print_contents(self, printer, *args):
        if self.is_annihilation:
            return r'%s' % str(self.name)
        else:
            return r'Dagger(%s)' % str(self.name)

    def _print_contents_pretty(self, printer, *args):
        from sympy.printing.pretty.stringpict import prettyForm
        pform = printer._print(self.args[0], *args)
        if self.is_annihilation:
            return pform
        else:
            return pform**prettyForm('\N{DAGGER}')


class BosonFockKet(Ket):
    """Fock state ket for a bosonic mode.

    Parameters
    ==========

    n : Number
        The Fock state number.

    """

    def __new__(cls, n):
        return Ket.__new__(cls, n)

    @property
    def n(self):
        return self.label[0]

    @classmethod
    def dual_class(self):
        return BosonFockBra

    @classmethod
    def _eval_hilbert_space(cls, label):
        return FockSpace()

    def _eval_innerproduct_BosonFockBra(self, bra, **hints):
        return KroneckerDelta(self.n, bra.n)

    def _apply_from_right_to_BosonOp(self, op, **options):
        if op.is_annihilation:
            return sqrt(self.n) * BosonFockKet(self.n - 1)
        else:
            return sqrt(self.n + 1) * BosonFockKet(self.n + 1)


class BosonFockBra(Bra):
    """Fock state bra for a bosonic mode.

    Parameters
    ==========

    n : Number
        The Fock state number.

    """

    def __new__(cls, n):
        return Bra.__new__(cls, n)

    @property
    def n(self):
        return self.label[0]

    @classmethod
    def dual_class(self):
        return BosonFockKet

    @classmethod
    def _eval_hilbert_space(cls, label):
        return FockSpace()


class BosonCoherentKet(Ket):
    """Coherent state ket for a bosonic mode.

    Parameters
    ==========

    alpha : Number, Symbol
        The complex amplitude of the coherent state.

    """

    def __new__(cls, alpha):
        return Ket.__new__(cls, alpha)

    @property
    def alpha(self):
        return self.label[0]

    @classmethod
    def dual_class(self):
        return BosonCoherentBra

    @classmethod
    def _eval_hilbert_space(cls, label):
        return HilbertSpace()

    def _eval_innerproduct_BosonCoherentBra(self, bra, **hints):
        if self.alpha == bra.alpha:
            return S.One
        else:
            return exp(-(abs(self.alpha)**2 + abs(bra.alpha)**2 - 2 * conjugate(bra.alpha) * self.alpha)/2)

    def _apply_from_right_to_BosonOp(self, op, **options):
        if op.is_annihilation:
            return self.alpha * self
        else:
            return None


class BosonCoherentBra(Bra):
    """Coherent state bra for a bosonic mode.

    Parameters
    ==========

    alpha : Number, Symbol
        The complex amplitude of the coherent state.

    """

    def __new__(cls, alpha):
        return Bra.__new__(cls, alpha)

    @property
    def alpha(self):
        return self.label[0]

    @classmethod
    def dual_class(self):
        return BosonCoherentKet

    def _apply_operator_BosonOp(self, op, **options):
        if not op.is_annihilation:
            return self.alpha * self
        else:
            return None
