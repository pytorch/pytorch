"""Simple Harmonic Oscillator 1-Dimension"""

from sympy.core.numbers import (I, Integer)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum.operator import Operator
from sympy.physics.quantum.state import Bra, Ket, State
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.cartesian import X, Px
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.quantum.hilbert import ComplexSpace
from sympy.physics.quantum.matrixutils import matrix_zeros

#------------------------------------------------------------------------------

class SHOOp(Operator):
    """A base class for the SHO Operators.

    We are limiting the number of arguments to be 1.

    """

    @classmethod
    def _eval_args(cls, args):
        args = QExpr._eval_args(args)
        if len(args) == 1:
            return args
        else:
            raise ValueError("Too many arguments")

    @classmethod
    def _eval_hilbert_space(cls, label):
        return ComplexSpace(S.Infinity)

class RaisingOp(SHOOp):
    """The Raising Operator or a^dagger.

    When a^dagger acts on a state it raises the state up by one. Taking
    the adjoint of a^dagger returns 'a', the Lowering Operator. a^dagger
    can be rewritten in terms of position and momentum. We can represent
    a^dagger as a matrix, which will be its default basis.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the
        operator.

    Examples
    ========

    Create a Raising Operator and rewrite it in terms of position and
    momentum, and show that taking its adjoint returns 'a':

        >>> from sympy.physics.quantum.sho1d import RaisingOp
        >>> from sympy.physics.quantum import Dagger

        >>> ad = RaisingOp('a')
        >>> ad.rewrite('xp').doit()
        sqrt(2)*(m*omega*X - I*Px)/(2*sqrt(hbar)*sqrt(m*omega))

        >>> Dagger(ad)
        a

    Taking the commutator of a^dagger with other Operators:

        >>> from sympy.physics.quantum import Commutator
        >>> from sympy.physics.quantum.sho1d import RaisingOp, LoweringOp
        >>> from sympy.physics.quantum.sho1d import NumberOp

        >>> ad = RaisingOp('a')
        >>> a = LoweringOp('a')
        >>> N = NumberOp('N')
        >>> Commutator(ad, a).doit()
        -1
        >>> Commutator(ad, N).doit()
        -RaisingOp(a)

    Apply a^dagger to a state:

        >>> from sympy.physics.quantum import qapply
        >>> from sympy.physics.quantum.sho1d import RaisingOp, SHOKet

        >>> ad = RaisingOp('a')
        >>> k = SHOKet('k')
        >>> qapply(ad*k)
        sqrt(k + 1)*|k + 1>

    Matrix Representation

        >>> from sympy.physics.quantum.sho1d import RaisingOp
        >>> from sympy.physics.quantum.represent import represent
        >>> ad = RaisingOp('a')
        >>> represent(ad, basis=N, ndim=4, format='sympy')
        Matrix([
        [0,       0,       0, 0],
        [1,       0,       0, 0],
        [0, sqrt(2),       0, 0],
        [0,       0, sqrt(3), 0]])

    """

    def _eval_rewrite_as_xp(self, *args, **kwargs):
        return (S.One/sqrt(Integer(2)*hbar*m*omega))*(
            S.NegativeOne*I*Px + m*omega*X)

    def _eval_adjoint(self):
        return LoweringOp(*self.args)

    def _eval_commutator_LoweringOp(self, other):
        return S.NegativeOne

    def _eval_commutator_NumberOp(self, other):
        return S.NegativeOne*self

    def _apply_operator_SHOKet(self, ket, **options):
        temp = ket.n + S.One
        return sqrt(temp)*SHOKet(temp)

    def _represent_default_basis(self, **options):
        return self._represent_NumberOp(None, **options)

    def _represent_XOp(self, basis, **options):
        # This logic is good but the underlying position
        # representation logic is broken.
        # temp = self.rewrite('xp').doit()
        # result = represent(temp, basis=X)
        # return result
        raise NotImplementedError('Position representation is not implemented')

    def _represent_NumberOp(self, basis, **options):
        ndim_info = options.get('ndim', 4)
        format = options.get('format','sympy')
        matrix = matrix_zeros(ndim_info, ndim_info, **options)
        for i in range(ndim_info - 1):
            value = sqrt(i + 1)
            if format == 'scipy.sparse':
                value = float(value)
            matrix[i + 1, i] = value
        if format == 'scipy.sparse':
            matrix = matrix.tocsr()
        return matrix

    #--------------------------------------------------------------------------
    # Printing Methods
    #--------------------------------------------------------------------------

    def _print_contents(self, printer, *args):
        arg0 = printer._print(self.args[0], *args)
        return '%s(%s)' % (self.__class__.__name__, arg0)

    def _print_contents_pretty(self, printer, *args):
        from sympy.printing.pretty.stringpict import prettyForm
        pform = printer._print(self.args[0], *args)
        pform = pform**prettyForm('\N{DAGGER}')
        return pform

    def _print_contents_latex(self, printer, *args):
        arg = printer._print(self.args[0])
        return '%s^{\\dagger}' % arg

class LoweringOp(SHOOp):
    """The Lowering Operator or 'a'.

    When 'a' acts on a state it lowers the state up by one. Taking
    the adjoint of 'a' returns a^dagger, the Raising Operator. 'a'
    can be rewritten in terms of position and momentum. We can
    represent 'a' as a matrix, which will be its default basis.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the
        operator.

    Examples
    ========

    Create a Lowering Operator and rewrite it in terms of position and
    momentum, and show that taking its adjoint returns a^dagger:

        >>> from sympy.physics.quantum.sho1d import LoweringOp
        >>> from sympy.physics.quantum import Dagger

        >>> a = LoweringOp('a')
        >>> a.rewrite('xp').doit()
        sqrt(2)*(m*omega*X + I*Px)/(2*sqrt(hbar)*sqrt(m*omega))

        >>> Dagger(a)
        RaisingOp(a)

    Taking the commutator of 'a' with other Operators:

        >>> from sympy.physics.quantum import Commutator
        >>> from sympy.physics.quantum.sho1d import LoweringOp, RaisingOp
        >>> from sympy.physics.quantum.sho1d import NumberOp

        >>> a = LoweringOp('a')
        >>> ad = RaisingOp('a')
        >>> N = NumberOp('N')
        >>> Commutator(a, ad).doit()
        1
        >>> Commutator(a, N).doit()
        a

    Apply 'a' to a state:

        >>> from sympy.physics.quantum import qapply
        >>> from sympy.physics.quantum.sho1d import LoweringOp, SHOKet

        >>> a = LoweringOp('a')
        >>> k = SHOKet('k')
        >>> qapply(a*k)
        sqrt(k)*|k - 1>

    Taking 'a' of the lowest state will return 0:

        >>> from sympy.physics.quantum import qapply
        >>> from sympy.physics.quantum.sho1d import LoweringOp, SHOKet

        >>> a = LoweringOp('a')
        >>> k = SHOKet(0)
        >>> qapply(a*k)
        0

    Matrix Representation

        >>> from sympy.physics.quantum.sho1d import LoweringOp
        >>> from sympy.physics.quantum.represent import represent
        >>> a = LoweringOp('a')
        >>> represent(a, basis=N, ndim=4, format='sympy')
        Matrix([
        [0, 1,       0,       0],
        [0, 0, sqrt(2),       0],
        [0, 0,       0, sqrt(3)],
        [0, 0,       0,       0]])

    """

    def _eval_rewrite_as_xp(self, *args, **kwargs):
        return (S.One/sqrt(Integer(2)*hbar*m*omega))*(
            I*Px + m*omega*X)

    def _eval_adjoint(self):
        return RaisingOp(*self.args)

    def _eval_commutator_RaisingOp(self, other):
        return S.One

    def _eval_commutator_NumberOp(self, other):
        return self

    def _apply_operator_SHOKet(self, ket, **options):
        temp = ket.n - Integer(1)
        if ket.n is S.Zero:
            return S.Zero
        else:
            return sqrt(ket.n)*SHOKet(temp)

    def _represent_default_basis(self, **options):
        return self._represent_NumberOp(None, **options)

    def _represent_XOp(self, basis, **options):
        # This logic is good but the underlying position
        # representation logic is broken.
        # temp = self.rewrite('xp').doit()
        # result = represent(temp, basis=X)
        # return result
        raise NotImplementedError('Position representation is not implemented')

    def _represent_NumberOp(self, basis, **options):
        ndim_info = options.get('ndim', 4)
        format = options.get('format', 'sympy')
        matrix = matrix_zeros(ndim_info, ndim_info, **options)
        for i in range(ndim_info - 1):
            value = sqrt(i + 1)
            if format == 'scipy.sparse':
                value = float(value)
            matrix[i,i + 1] = value
        if format == 'scipy.sparse':
            matrix = matrix.tocsr()
        return matrix


class NumberOp(SHOOp):
    """The Number Operator is simply a^dagger*a

    It is often useful to write a^dagger*a as simply the Number Operator
    because the Number Operator commutes with the Hamiltonian. And can be
    expressed using the Number Operator. Also the Number Operator can be
    applied to states. We can represent the Number Operator as a matrix,
    which will be its default basis.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the
        operator.

    Examples
    ========

    Create a Number Operator and rewrite it in terms of the ladder
    operators, position and momentum operators, and Hamiltonian:

        >>> from sympy.physics.quantum.sho1d import NumberOp

        >>> N = NumberOp('N')
        >>> N.rewrite('a').doit()
        RaisingOp(a)*a
        >>> N.rewrite('xp').doit()
        -1/2 + (m**2*omega**2*X**2 + Px**2)/(2*hbar*m*omega)
        >>> N.rewrite('H').doit()
        -1/2 + H/(hbar*omega)

    Take the Commutator of the Number Operator with other Operators:

        >>> from sympy.physics.quantum import Commutator
        >>> from sympy.physics.quantum.sho1d import NumberOp, Hamiltonian
        >>> from sympy.physics.quantum.sho1d import RaisingOp, LoweringOp

        >>> N = NumberOp('N')
        >>> H = Hamiltonian('H')
        >>> ad = RaisingOp('a')
        >>> a = LoweringOp('a')
        >>> Commutator(N,H).doit()
        0
        >>> Commutator(N,ad).doit()
        RaisingOp(a)
        >>> Commutator(N,a).doit()
        -a

    Apply the Number Operator to a state:

        >>> from sympy.physics.quantum import qapply
        >>> from sympy.physics.quantum.sho1d import NumberOp, SHOKet

        >>> N = NumberOp('N')
        >>> k = SHOKet('k')
        >>> qapply(N*k)
        k*|k>

    Matrix Representation

        >>> from sympy.physics.quantum.sho1d import NumberOp
        >>> from sympy.physics.quantum.represent import represent
        >>> N = NumberOp('N')
        >>> represent(N, basis=N, ndim=4, format='sympy')
        Matrix([
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 3]])

    """

    def _eval_rewrite_as_a(self, *args, **kwargs):
        return ad*a

    def _eval_rewrite_as_xp(self, *args, **kwargs):
        return (S.One/(Integer(2)*m*hbar*omega))*(Px**2 + (
            m*omega*X)**2) - S.Half

    def _eval_rewrite_as_H(self, *args, **kwargs):
        return H/(hbar*omega) - S.Half

    def _apply_operator_SHOKet(self, ket, **options):
        return ket.n*ket

    def _eval_commutator_Hamiltonian(self, other):
        return S.Zero

    def _eval_commutator_RaisingOp(self, other):
        return other

    def _eval_commutator_LoweringOp(self, other):
        return S.NegativeOne*other

    def _represent_default_basis(self, **options):
        return self._represent_NumberOp(None, **options)

    def _represent_XOp(self, basis, **options):
        # This logic is good but the underlying position
        # representation logic is broken.
        # temp = self.rewrite('xp').doit()
        # result = represent(temp, basis=X)
        # return result
        raise NotImplementedError('Position representation is not implemented')

    def _represent_NumberOp(self, basis, **options):
        ndim_info = options.get('ndim', 4)
        format = options.get('format', 'sympy')
        matrix = matrix_zeros(ndim_info, ndim_info, **options)
        for i in range(ndim_info):
            value = i
            if format == 'scipy.sparse':
                value = float(value)
            matrix[i,i] = value
        if format == 'scipy.sparse':
            matrix = matrix.tocsr()
        return matrix


class Hamiltonian(SHOOp):
    """The Hamiltonian Operator.

    The Hamiltonian is used to solve the time-independent Schrodinger
    equation. The Hamiltonian can be expressed using the ladder operators,
    as well as by position and momentum. We can represent the Hamiltonian
    Operator as a matrix, which will be its default basis.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the
        operator.

    Examples
    ========

    Create a Hamiltonian Operator and rewrite it in terms of the ladder
    operators, position and momentum, and the Number Operator:

        >>> from sympy.physics.quantum.sho1d import Hamiltonian

        >>> H = Hamiltonian('H')
        >>> H.rewrite('a').doit()
        hbar*omega*(1/2 + RaisingOp(a)*a)
        >>> H.rewrite('xp').doit()
        (m**2*omega**2*X**2 + Px**2)/(2*m)
        >>> H.rewrite('N').doit()
        hbar*omega*(1/2 + N)

    Take the Commutator of the Hamiltonian and the Number Operator:

        >>> from sympy.physics.quantum import Commutator
        >>> from sympy.physics.quantum.sho1d import Hamiltonian, NumberOp

        >>> H = Hamiltonian('H')
        >>> N = NumberOp('N')
        >>> Commutator(H,N).doit()
        0

    Apply the Hamiltonian Operator to a state:

        >>> from sympy.physics.quantum import qapply
        >>> from sympy.physics.quantum.sho1d import Hamiltonian, SHOKet

        >>> H = Hamiltonian('H')
        >>> k = SHOKet('k')
        >>> qapply(H*k)
        hbar*k*omega*|k> + hbar*omega*|k>/2

    Matrix Representation

        >>> from sympy.physics.quantum.sho1d import Hamiltonian
        >>> from sympy.physics.quantum.represent import represent

        >>> H = Hamiltonian('H')
        >>> represent(H, basis=N, ndim=4, format='sympy')
        Matrix([
        [hbar*omega/2,              0,              0,              0],
        [           0, 3*hbar*omega/2,              0,              0],
        [           0,              0, 5*hbar*omega/2,              0],
        [           0,              0,              0, 7*hbar*omega/2]])

    """

    def _eval_rewrite_as_a(self, *args, **kwargs):
        return hbar*omega*(ad*a + S.Half)

    def _eval_rewrite_as_xp(self, *args, **kwargs):
        return (S.One/(Integer(2)*m))*(Px**2 + (m*omega*X)**2)

    def _eval_rewrite_as_N(self, *args, **kwargs):
        return hbar*omega*(N + S.Half)

    def _apply_operator_SHOKet(self, ket, **options):
        return (hbar*omega*(ket.n + S.Half))*ket

    def _eval_commutator_NumberOp(self, other):
        return S.Zero

    def _represent_default_basis(self, **options):
        return self._represent_NumberOp(None, **options)

    def _represent_XOp(self, basis, **options):
        # This logic is good but the underlying position
        # representation logic is broken.
        # temp = self.rewrite('xp').doit()
        # result = represent(temp, basis=X)
        # return result
        raise NotImplementedError('Position representation is not implemented')

    def _represent_NumberOp(self, basis, **options):
        ndim_info = options.get('ndim', 4)
        format = options.get('format', 'sympy')
        matrix = matrix_zeros(ndim_info, ndim_info, **options)
        for i in range(ndim_info):
            value = i + S.Half
            if format == 'scipy.sparse':
                value = float(value)
            matrix[i,i] = value
        if format == 'scipy.sparse':
            matrix = matrix.tocsr()
        return hbar*omega*matrix

#------------------------------------------------------------------------------

class SHOState(State):
    """State class for SHO states"""

    @classmethod
    def _eval_hilbert_space(cls, label):
        return ComplexSpace(S.Infinity)

    @property
    def n(self):
        return self.args[0]


class SHOKet(SHOState, Ket):
    """1D eigenket.

    Inherits from SHOState and Ket.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the ket
        This is usually its quantum numbers or its symbol.

    Examples
    ========

    Ket's know about their associated bra:

        >>> from sympy.physics.quantum.sho1d import SHOKet

        >>> k = SHOKet('k')
        >>> k.dual
        <k|
        >>> k.dual_class()
        <class 'sympy.physics.quantum.sho1d.SHOBra'>

    Take the Inner Product with a bra:

        >>> from sympy.physics.quantum import InnerProduct
        >>> from sympy.physics.quantum.sho1d import SHOKet, SHOBra

        >>> k = SHOKet('k')
        >>> b = SHOBra('b')
        >>> InnerProduct(b,k).doit()
        KroneckerDelta(b, k)

    Vector representation of a numerical state ket:

        >>> from sympy.physics.quantum.sho1d import SHOKet, NumberOp
        >>> from sympy.physics.quantum.represent import represent

        >>> k = SHOKet(3)
        >>> N = NumberOp('N')
        >>> represent(k, basis=N, ndim=4)
        Matrix([
        [0],
        [0],
        [0],
        [1]])

    """

    @classmethod
    def dual_class(self):
        return SHOBra

    def _eval_innerproduct_SHOBra(self, bra, **hints):
        result = KroneckerDelta(self.n, bra.n)
        return result

    def _represent_default_basis(self, **options):
        return self._represent_NumberOp(None, **options)

    def _represent_NumberOp(self, basis, **options):
        ndim_info = options.get('ndim', 4)
        format = options.get('format', 'sympy')
        options['spmatrix'] = 'lil'
        vector = matrix_zeros(ndim_info, 1, **options)
        if isinstance(self.n, Integer):
            if self.n >= ndim_info:
                return ValueError("N-Dimension too small")
            if format == 'scipy.sparse':
                vector[int(self.n), 0] = 1.0
                vector = vector.tocsr()
            elif format == 'numpy':
                vector[int(self.n), 0] = 1.0
            else:
                vector[self.n, 0] = S.One
            return vector
        else:
            return ValueError("Not Numerical State")


class SHOBra(SHOState, Bra):
    """A time-independent Bra in SHO.

    Inherits from SHOState and Bra.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the ket
        This is usually its quantum numbers or its symbol.

    Examples
    ========

    Bra's know about their associated ket:

        >>> from sympy.physics.quantum.sho1d import SHOBra

        >>> b = SHOBra('b')
        >>> b.dual
        |b>
        >>> b.dual_class()
        <class 'sympy.physics.quantum.sho1d.SHOKet'>

    Vector representation of a numerical state bra:

        >>> from sympy.physics.quantum.sho1d import SHOBra, NumberOp
        >>> from sympy.physics.quantum.represent import represent

        >>> b = SHOBra(3)
        >>> N = NumberOp('N')
        >>> represent(b, basis=N, ndim=4)
        Matrix([[0, 0, 0, 1]])

    """

    @classmethod
    def dual_class(self):
        return SHOKet

    def _represent_default_basis(self, **options):
        return self._represent_NumberOp(None, **options)

    def _represent_NumberOp(self, basis, **options):
        ndim_info = options.get('ndim', 4)
        format = options.get('format', 'sympy')
        options['spmatrix'] = 'lil'
        vector = matrix_zeros(1, ndim_info, **options)
        if isinstance(self.n, Integer):
            if self.n >= ndim_info:
                return ValueError("N-Dimension too small")
            if format == 'scipy.sparse':
                vector[0, int(self.n)] = 1.0
                vector = vector.tocsr()
            elif format == 'numpy':
                vector[0, int(self.n)] = 1.0
            else:
                vector[0, self.n] = S.One
            return vector
        else:
            return ValueError("Not Numerical State")


ad = RaisingOp('a')
a = LoweringOp('a')
H = Hamiltonian('H')
N = NumberOp('N')
omega = Symbol('omega')
m = Symbol('m')
