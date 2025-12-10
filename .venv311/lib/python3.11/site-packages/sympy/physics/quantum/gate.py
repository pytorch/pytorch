"""An implementation of gates that act on qubits.

Gates are unitary operators that act on the space of qubits.

Medium Term Todo:

* Optimize Gate._apply_operators_Qubit to remove the creation of many
  intermediate Qubit objects.
* Add commutation relationships to all operators and use this in gate_sort.
* Fix gate_sort and gate_simp.
* Get multi-target UGates plotting properly.
* Get UGate to work with either sympy/numpy matrices and output either
  format. This should also use the matrix slots.
"""

from itertools import chain
import random

from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer)
from sympy.core.power import Pow
from sympy.core.numbers import Number
from sympy.core.singleton import S as _S
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import _sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.printing.pretty.stringpict import prettyForm, stringPict

from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.qexpr import QuantumError
from sympy.physics.quantum.hilbert import ComplexSpace
from sympy.physics.quantum.operator import (UnitaryOperator, Operator,
                                            HermitianOperator)
from sympy.physics.quantum.matrixutils import matrix_tensor_product, matrix_eye
from sympy.physics.quantum.matrixcache import matrix_cache

from sympy.matrices.matrixbase import MatrixBase

from sympy.utilities.iterables import is_sequence

__all__ = [
    'Gate',
    'CGate',
    'UGate',
    'OneQubitGate',
    'TwoQubitGate',
    'IdentityGate',
    'HadamardGate',
    'XGate',
    'YGate',
    'ZGate',
    'TGate',
    'PhaseGate',
    'SwapGate',
    'CNotGate',
    # Aliased gate names
    'CNOT',
    'SWAP',
    'H',
    'X',
    'Y',
    'Z',
    'T',
    'S',
    'Phase',
    'normalized',
    'gate_sort',
    'gate_simp',
    'random_circuit',
    'CPHASE',
    'CGateS',
]

#-----------------------------------------------------------------------------
# Gate Super-Classes
#-----------------------------------------------------------------------------

_normalized = True


def _max(*args, **kwargs):
    if "key" not in kwargs:
        kwargs["key"] = default_sort_key
    return max(*args, **kwargs)


def _min(*args, **kwargs):
    if "key" not in kwargs:
        kwargs["key"] = default_sort_key
    return min(*args, **kwargs)


def normalized(normalize):
    r"""Set flag controlling normalization of Hadamard gates by `1/\sqrt{2}`.

    This is a global setting that can be used to simplify the look of various
    expressions, by leaving off the leading `1/\sqrt{2}` of the Hadamard gate.

    Parameters
    ----------
    normalize : bool
        Should the Hadamard gate include the `1/\sqrt{2}` normalization factor?
        When True, the Hadamard gate will have the `1/\sqrt{2}`. When False, the
        Hadamard gate will not have this factor.
    """
    global _normalized
    _normalized = normalize


def _validate_targets_controls(tandc):
    tandc = list(tandc)
    # Check for integers
    for bit in tandc:
        if not bit.is_Integer and not bit.is_Symbol:
            raise TypeError('Integer expected, got: %r' % tandc[bit])
    # Detect duplicates
    if len(set(tandc)) != len(tandc):
        raise QuantumError(
            'Target/control qubits in a gate cannot be duplicated'
        )


class Gate(UnitaryOperator):
    """Non-controlled unitary gate operator that acts on qubits.

    This is a general abstract gate that needs to be subclassed to do anything
    useful.

    Parameters
    ----------
    label : tuple, int
        A list of the target qubits (as ints) that the gate will apply to.

    Examples
    ========


    """

    _label_separator = ','

    gate_name = 'G'
    gate_name_latex = 'G'

    #-------------------------------------------------------------------------
    # Initialization/creation
    #-------------------------------------------------------------------------

    @classmethod
    def _eval_args(cls, args):
        args = Tuple(*UnitaryOperator._eval_args(args))
        _validate_targets_controls(args)
        return args

    @classmethod
    def _eval_hilbert_space(cls, args):
        """This returns the smallest possible Hilbert space."""
        return ComplexSpace(2)**(_max(args) + 1)

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def nqubits(self):
        """The total number of qubits this gate acts on.

        For controlled gate subclasses this includes both target and control
        qubits, so that, for examples the CNOT gate acts on 2 qubits.
        """
        return len(self.targets)

    @property
    def min_qubits(self):
        """The minimum number of qubits this gate needs to act on."""
        return _max(self.targets) + 1

    @property
    def targets(self):
        """A tuple of target qubits."""
        return self.label

    @property
    def gate_name_plot(self):
        return r'$%s$' % self.gate_name_latex

    #-------------------------------------------------------------------------
    # Gate methods
    #-------------------------------------------------------------------------

    def get_target_matrix(self, format='sympy'):
        """The matrix representation of the target part of the gate.

        Parameters
        ----------
        format : str
            The format string ('sympy','numpy', etc.)
        """
        raise NotImplementedError(
            'get_target_matrix is not implemented in Gate.')

    #-------------------------------------------------------------------------
    # Apply
    #-------------------------------------------------------------------------

    def _apply_operator_IntQubit(self, qubits, **options):
        """Redirect an apply from IntQubit to Qubit"""
        return self._apply_operator_Qubit(qubits, **options)

    def _apply_operator_Qubit(self, qubits, **options):
        """Apply this gate to a Qubit."""

        # Check number of qubits this gate acts on.
        if qubits.nqubits < self.min_qubits:
            raise QuantumError(
                'Gate needs a minimum of %r qubits to act on, got: %r' %
                (self.min_qubits, qubits.nqubits)
            )

        # If the controls are not met, just return
        if isinstance(self, CGate):
            if not self.eval_controls(qubits):
                return qubits

        targets = self.targets
        target_matrix = self.get_target_matrix(format='sympy')

        # Find which column of the target matrix this applies to.
        column_index = 0
        n = 1
        for target in targets:
            column_index += n*qubits[target]
            n = n << 1
        column = target_matrix[:, int(column_index)]

        # Now apply each column element to the qubit.
        result = 0
        for index in range(column.rows):
            # TODO: This can be optimized to reduce the number of Qubit
            # creations. We should simply manipulate the raw list of qubit
            # values and then build the new Qubit object once.
            # Make a copy of the incoming qubits.
            new_qubit = qubits.__class__(*qubits.args)
            # Flip the bits that need to be flipped.
            for bit, target in enumerate(targets):
                if new_qubit[target] != (index >> bit) & 1:
                    new_qubit = new_qubit.flip(target)
            # The value in that row and column times the flipped-bit qubit
            # is the result for that part.
            result += column[index]*new_qubit
        return result

    #-------------------------------------------------------------------------
    # Represent
    #-------------------------------------------------------------------------

    def _represent_default_basis(self, **options):
        return self._represent_ZGate(None, **options)

    def _represent_ZGate(self, basis, **options):
        format = options.get('format', 'sympy')
        nqubits = options.get('nqubits', 0)
        if nqubits == 0:
            raise QuantumError(
                'The number of qubits must be given as nqubits.')

        # Make sure we have enough qubits for the gate.
        if nqubits < self.min_qubits:
            raise QuantumError(
                'The number of qubits %r is too small for the gate.' % nqubits
            )

        target_matrix = self.get_target_matrix(format)
        targets = self.targets
        if isinstance(self, CGate):
            controls = self.controls
        else:
            controls = []
        m = represent_zbasis(
            controls, targets, target_matrix, nqubits, format
        )
        return m

    #-------------------------------------------------------------------------
    # Print methods
    #-------------------------------------------------------------------------

    def _sympystr(self, printer, *args):
        label = self._print_label(printer, *args)
        return '%s(%s)' % (self.gate_name, label)

    def _pretty(self, printer, *args):
        a = stringPict(self.gate_name)
        b = self._print_label_pretty(printer, *args)
        return self._print_subscript_pretty(a, b)

    def _latex(self, printer, *args):
        label = self._print_label(printer, *args)
        return '%s_{%s}' % (self.gate_name_latex, label)

    def plot_gate(self, axes, gate_idx, gate_grid, wire_grid):
        raise NotImplementedError('plot_gate is not implemented.')


class CGate(Gate):
    """A general unitary gate with control qubits.

    A general control gate applies a target gate to a set of targets if all
    of the control qubits have a particular values (set by
    ``CGate.control_value``).

    Parameters
    ----------
    label : tuple
        The label in this case has the form (controls, gate), where controls
        is a tuple/list of control qubits (as ints) and gate is a ``Gate``
        instance that is the target operator.

    Examples
    ========

    """

    gate_name = 'C'
    gate_name_latex = 'C'

    # The values this class controls for.
    control_value = _S.One

    simplify_cgate = False

    #-------------------------------------------------------------------------
    # Initialization
    #-------------------------------------------------------------------------

    @classmethod
    def _eval_args(cls, args):
        # _eval_args has the right logic for the controls argument.
        controls = args[0]
        gate = args[1]
        if not is_sequence(controls):
            controls = (controls,)
        controls = UnitaryOperator._eval_args(controls)
        _validate_targets_controls(chain(controls, gate.targets))
        return (Tuple(*controls), gate)

    @classmethod
    def _eval_hilbert_space(cls, args):
        """This returns the smallest possible Hilbert space."""
        return ComplexSpace(2)**_max(_max(args[0]) + 1, args[1].min_qubits)

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def nqubits(self):
        """The total number of qubits this gate acts on.

        For controlled gate subclasses this includes both target and control
        qubits, so that, for examples the CNOT gate acts on 2 qubits.
        """
        return len(self.targets) + len(self.controls)

    @property
    def min_qubits(self):
        """The minimum number of qubits this gate needs to act on."""
        return _max(_max(self.controls), _max(self.targets)) + 1

    @property
    def targets(self):
        """A tuple of target qubits."""
        return self.gate.targets

    @property
    def controls(self):
        """A tuple of control qubits."""
        return tuple(self.label[0])

    @property
    def gate(self):
        """The non-controlled gate that will be applied to the targets."""
        return self.label[1]

    #-------------------------------------------------------------------------
    # Gate methods
    #-------------------------------------------------------------------------

    def get_target_matrix(self, format='sympy'):
        return self.gate.get_target_matrix(format)

    def eval_controls(self, qubit):
        """Return True/False to indicate if the controls are satisfied."""
        return all(qubit[bit] == self.control_value for bit in self.controls)

    def decompose(self, **options):
        """Decompose the controlled gate into CNOT and single qubits gates."""
        if len(self.controls) == 1:
            c = self.controls[0]
            t = self.gate.targets[0]
            if isinstance(self.gate, YGate):
                g1 = PhaseGate(t)
                g2 = CNotGate(c, t)
                g3 = PhaseGate(t)
                g4 = ZGate(t)
                return g1*g2*g3*g4
            if isinstance(self.gate, ZGate):
                g1 = HadamardGate(t)
                g2 = CNotGate(c, t)
                g3 = HadamardGate(t)
                return g1*g2*g3
        else:
            return self

    #-------------------------------------------------------------------------
    # Print methods
    #-------------------------------------------------------------------------

    def _print_label(self, printer, *args):
        controls = self._print_sequence(self.controls, ',', printer, *args)
        gate = printer._print(self.gate, *args)
        return '(%s),%s' % (controls, gate)

    def _pretty(self, printer, *args):
        controls = self._print_sequence_pretty(
            self.controls, ',', printer, *args)
        gate = printer._print(self.gate)
        gate_name = stringPict(self.gate_name)
        first = self._print_subscript_pretty(gate_name, controls)
        gate = self._print_parens_pretty(gate)
        final = prettyForm(*first.right(gate))
        return final

    def _latex(self, printer, *args):
        controls = self._print_sequence(self.controls, ',', printer, *args)
        gate = printer._print(self.gate, *args)
        return r'%s_{%s}{\left(%s\right)}' % \
            (self.gate_name_latex, controls, gate)

    def plot_gate(self, circ_plot, gate_idx):
        """
        Plot the controlled gate. If *simplify_cgate* is true, simplify
        C-X and C-Z gates into their more familiar forms.
        """
        min_wire = int(_min(chain(self.controls, self.targets)))
        max_wire = int(_max(chain(self.controls, self.targets)))
        circ_plot.control_line(gate_idx, min_wire, max_wire)
        for c in self.controls:
            circ_plot.control_point(gate_idx, int(c))
        if self.simplify_cgate:
            if self.gate.gate_name == 'X':
                self.gate.plot_gate_plus(circ_plot, gate_idx)
            elif self.gate.gate_name == 'Z':
                circ_plot.control_point(gate_idx, self.targets[0])
            else:
                self.gate.plot_gate(circ_plot, gate_idx)
        else:
            self.gate.plot_gate(circ_plot, gate_idx)

    #-------------------------------------------------------------------------
    # Miscellaneous
    #-------------------------------------------------------------------------

    def _eval_dagger(self):
        if isinstance(self.gate, HermitianOperator):
            return self
        else:
            return Gate._eval_dagger(self)

    def _eval_inverse(self):
        if isinstance(self.gate, HermitianOperator):
            return self
        else:
            return Gate._eval_inverse(self)

    def _eval_power(self, exp):
        if isinstance(self.gate, HermitianOperator):
            if exp == -1:
                return Gate._eval_power(self, exp)
            elif abs(exp) % 2 == 0:
                return self*(Gate._eval_inverse(self))
            else:
                return self
        else:
            return Gate._eval_power(self, exp)

class CGateS(CGate):
    """Version of CGate that allows gate simplifications.
    I.e. cnot looks like an oplus, cphase has dots, etc.
    """
    simplify_cgate=True


class UGate(Gate):
    """General gate specified by a set of targets and a target matrix.

    Parameters
    ----------
    label : tuple
        A tuple of the form (targets, U), where targets is a tuple of the
        target qubits and U is a unitary matrix with dimension of
        len(targets).
    """
    gate_name = 'U'
    gate_name_latex = 'U'

    #-------------------------------------------------------------------------
    # Initialization
    #-------------------------------------------------------------------------

    @classmethod
    def _eval_args(cls, args):
        targets = args[0]
        if not is_sequence(targets):
            targets = (targets,)
        targets = Gate._eval_args(targets)
        _validate_targets_controls(targets)
        mat = args[1]
        if not isinstance(mat, MatrixBase):
            raise TypeError('Matrix expected, got: %r' % mat)
        #make sure this matrix is of a Basic type
        mat = _sympify(mat)
        dim = 2**len(targets)
        if not all(dim == shape for shape in mat.shape):
            raise IndexError(
                'Number of targets must match the matrix size: %r %r' %
                (targets, mat)
            )
        return (targets, mat)

    @classmethod
    def _eval_hilbert_space(cls, args):
        """This returns the smallest possible Hilbert space."""
        return ComplexSpace(2)**(_max(args[0]) + 1)

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def targets(self):
        """A tuple of target qubits."""
        return tuple(self.label[0])

    #-------------------------------------------------------------------------
    # Gate methods
    #-------------------------------------------------------------------------

    def get_target_matrix(self, format='sympy'):
        """The matrix rep. of the target part of the gate.

        Parameters
        ----------
        format : str
            The format string ('sympy','numpy', etc.)
        """
        return self.label[1]

    #-------------------------------------------------------------------------
    # Print methods
    #-------------------------------------------------------------------------
    def _pretty(self, printer, *args):
        targets = self._print_sequence_pretty(
            self.targets, ',', printer, *args)
        gate_name = stringPict(self.gate_name)
        return self._print_subscript_pretty(gate_name, targets)

    def _latex(self, printer, *args):
        targets = self._print_sequence(self.targets, ',', printer, *args)
        return r'%s_{%s}' % (self.gate_name_latex, targets)

    def plot_gate(self, circ_plot, gate_idx):
        circ_plot.one_qubit_box(
            self.gate_name_plot,
            gate_idx, int(self.targets[0])
        )


class OneQubitGate(Gate):
    """A single qubit unitary gate base class."""

    nqubits = _S.One

    def plot_gate(self, circ_plot, gate_idx):
        circ_plot.one_qubit_box(
            self.gate_name_plot,
            gate_idx, int(self.targets[0])
        )

    def _eval_commutator(self, other, **hints):
        if isinstance(other, OneQubitGate):
            if self.targets != other.targets or self.__class__ == other.__class__:
                return _S.Zero
        return Operator._eval_commutator(self, other, **hints)

    def _eval_anticommutator(self, other, **hints):
        if isinstance(other, OneQubitGate):
            if self.targets != other.targets or self.__class__ == other.__class__:
                return Integer(2)*self*other
        return Operator._eval_anticommutator(self, other, **hints)


class TwoQubitGate(Gate):
    """A two qubit unitary gate base class."""

    nqubits = Integer(2)

#-----------------------------------------------------------------------------
# Single Qubit Gates
#-----------------------------------------------------------------------------


class IdentityGate(OneQubitGate):
    """The single qubit identity gate.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    """
    is_hermitian = True
    gate_name = '1'
    gate_name_latex = '1'

    # Short cut version of gate._apply_operator_Qubit
    def _apply_operator_Qubit(self, qubits, **options):
        # Check number of qubits this gate acts on (see gate._apply_operator_Qubit)
        if qubits.nqubits < self.min_qubits:
            raise QuantumError(
                'Gate needs a minimum of %r qubits to act on, got: %r' %
                (self.min_qubits, qubits.nqubits)
            )
        return qubits # no computation required for IdentityGate

    def get_target_matrix(self, format='sympy'):
        return matrix_cache.get_matrix('eye2', format)

    def _eval_commutator(self, other, **hints):
        return _S.Zero

    def _eval_anticommutator(self, other, **hints):
        return Integer(2)*other


class HadamardGate(HermitianOperator, OneQubitGate):
    """The single qubit Hadamard gate.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.physics.quantum.qubit import Qubit
    >>> from sympy.physics.quantum.gate import HadamardGate
    >>> from sympy.physics.quantum.qapply import qapply
    >>> qapply(HadamardGate(0)*Qubit('1'))
    sqrt(2)*|0>/2 - sqrt(2)*|1>/2
    >>> # Hadamard on bell state, applied on 2 qubits.
    >>> psi = 1/sqrt(2)*(Qubit('00')+Qubit('11'))
    >>> qapply(HadamardGate(0)*HadamardGate(1)*psi)
    sqrt(2)*|00>/2 + sqrt(2)*|11>/2

    """
    gate_name = 'H'
    gate_name_latex = 'H'

    def get_target_matrix(self, format='sympy'):
        if _normalized:
            return matrix_cache.get_matrix('H', format)
        else:
            return matrix_cache.get_matrix('Hsqrt2', format)

    def _eval_commutator_XGate(self, other, **hints):
        return I*sqrt(2)*YGate(self.targets[0])

    def _eval_commutator_YGate(self, other, **hints):
        return I*sqrt(2)*(ZGate(self.targets[0]) - XGate(self.targets[0]))

    def _eval_commutator_ZGate(self, other, **hints):
        return -I*sqrt(2)*YGate(self.targets[0])

    def _eval_anticommutator_XGate(self, other, **hints):
        return sqrt(2)*IdentityGate(self.targets[0])

    def _eval_anticommutator_YGate(self, other, **hints):
        return _S.Zero

    def _eval_anticommutator_ZGate(self, other, **hints):
        return sqrt(2)*IdentityGate(self.targets[0])


class XGate(HermitianOperator, OneQubitGate):
    """The single qubit X, or NOT, gate.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    """
    gate_name = 'X'
    gate_name_latex = 'X'

    def get_target_matrix(self, format='sympy'):
        return matrix_cache.get_matrix('X', format)

    def plot_gate(self, circ_plot, gate_idx):
        OneQubitGate.plot_gate(self,circ_plot,gate_idx)

    def plot_gate_plus(self, circ_plot, gate_idx):
        circ_plot.not_point(
            gate_idx, int(self.label[0])
        )

    def _eval_commutator_YGate(self, other, **hints):
        return Integer(2)*I*ZGate(self.targets[0])

    def _eval_anticommutator_XGate(self, other, **hints):
        return Integer(2)*IdentityGate(self.targets[0])

    def _eval_anticommutator_YGate(self, other, **hints):
        return _S.Zero

    def _eval_anticommutator_ZGate(self, other, **hints):
        return _S.Zero


class YGate(HermitianOperator, OneQubitGate):
    """The single qubit Y gate.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    """
    gate_name = 'Y'
    gate_name_latex = 'Y'

    def get_target_matrix(self, format='sympy'):
        return matrix_cache.get_matrix('Y', format)

    def _eval_commutator_ZGate(self, other, **hints):
        return Integer(2)*I*XGate(self.targets[0])

    def _eval_anticommutator_YGate(self, other, **hints):
        return Integer(2)*IdentityGate(self.targets[0])

    def _eval_anticommutator_ZGate(self, other, **hints):
        return _S.Zero


class ZGate(HermitianOperator, OneQubitGate):
    """The single qubit Z gate.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    """
    gate_name = 'Z'
    gate_name_latex = 'Z'

    def get_target_matrix(self, format='sympy'):
        return matrix_cache.get_matrix('Z', format)

    def _eval_commutator_XGate(self, other, **hints):
        return Integer(2)*I*YGate(self.targets[0])

    def _eval_anticommutator_YGate(self, other, **hints):
        return _S.Zero


class PhaseGate(OneQubitGate):
    """The single qubit phase, or S, gate.

    This gate rotates the phase of the state by pi/2 if the state is ``|1>`` and
    does nothing if the state is ``|0>``.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    """
    is_hermitian =  False
    gate_name = 'S'
    gate_name_latex = 'S'

    def get_target_matrix(self, format='sympy'):
        return matrix_cache.get_matrix('S', format)

    def _eval_commutator_ZGate(self, other, **hints):
        return _S.Zero

    def _eval_commutator_TGate(self, other, **hints):
        return _S.Zero


class TGate(OneQubitGate):
    """The single qubit pi/8 gate.

    This gate rotates the phase of the state by pi/4 if the state is ``|1>`` and
    does nothing if the state is ``|0>``.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    """
    is_hermitian = False
    gate_name = 'T'
    gate_name_latex = 'T'

    def get_target_matrix(self, format='sympy'):
        return matrix_cache.get_matrix('T', format)

    def _eval_commutator_ZGate(self, other, **hints):
        return _S.Zero

    def _eval_commutator_PhaseGate(self, other, **hints):
        return _S.Zero


# Aliases for gate names.
H = HadamardGate
X = XGate
Y = YGate
Z = ZGate
T = TGate
Phase = S = PhaseGate


#-----------------------------------------------------------------------------
# 2 Qubit Gates
#-----------------------------------------------------------------------------


class CNotGate(HermitianOperator, CGate, TwoQubitGate):
    """Two qubit controlled-NOT.

    This gate performs the NOT or X gate on the target qubit if the control
    qubits all have the value 1.

    Parameters
    ----------
    label : tuple
        A tuple of the form (control, target).

    Examples
    ========

    >>> from sympy.physics.quantum.gate import CNOT
    >>> from sympy.physics.quantum.qapply import qapply
    >>> from sympy.physics.quantum.qubit import Qubit
    >>> c = CNOT(1,0)
    >>> qapply(c*Qubit('10')) # note that qubits are indexed from right to left
    |11>

    """
    gate_name = 'CNOT'
    gate_name_latex = r'\text{CNOT}'
    simplify_cgate = True

    #-------------------------------------------------------------------------
    # Initialization
    #-------------------------------------------------------------------------

    @classmethod
    def _eval_args(cls, args):
        args = Gate._eval_args(args)
        return args

    @classmethod
    def _eval_hilbert_space(cls, args):
        """This returns the smallest possible Hilbert space."""
        return ComplexSpace(2)**(_max(args) + 1)

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def min_qubits(self):
        """The minimum number of qubits this gate needs to act on."""
        return _max(self.label) + 1

    @property
    def targets(self):
        """A tuple of target qubits."""
        return (self.label[1],)

    @property
    def controls(self):
        """A tuple of control qubits."""
        return (self.label[0],)

    @property
    def gate(self):
        """The non-controlled gate that will be applied to the targets."""
        return XGate(self.label[1])

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    # The default printing of Gate works better than those of CGate, so we
    # go around the overridden methods in CGate.

    def _print_label(self, printer, *args):
        return Gate._print_label(self, printer, *args)

    def _pretty(self, printer, *args):
        return Gate._pretty(self, printer, *args)

    def _latex(self, printer, *args):
        return Gate._latex(self, printer, *args)

    #-------------------------------------------------------------------------
    # Commutator/AntiCommutator
    #-------------------------------------------------------------------------

    def _eval_commutator_ZGate(self, other, **hints):
        """[CNOT(i, j), Z(i)] == 0."""
        if self.controls[0] == other.targets[0]:
            return _S.Zero
        else:
            raise NotImplementedError('Commutator not implemented: %r' % other)

    def _eval_commutator_TGate(self, other, **hints):
        """[CNOT(i, j), T(i)] == 0."""
        return self._eval_commutator_ZGate(other, **hints)

    def _eval_commutator_PhaseGate(self, other, **hints):
        """[CNOT(i, j), S(i)] == 0."""
        return self._eval_commutator_ZGate(other, **hints)

    def _eval_commutator_XGate(self, other, **hints):
        """[CNOT(i, j), X(j)] == 0."""
        if self.targets[0] == other.targets[0]:
            return _S.Zero
        else:
            raise NotImplementedError('Commutator not implemented: %r' % other)

    def _eval_commutator_CNotGate(self, other, **hints):
        """[CNOT(i, j), CNOT(i,k)] == 0."""
        if self.controls[0] == other.controls[0]:
            return _S.Zero
        else:
            raise NotImplementedError('Commutator not implemented: %r' % other)


class SwapGate(TwoQubitGate):
    """Two qubit SWAP gate.

    This gate swap the values of the two qubits.

    Parameters
    ----------
    label : tuple
        A tuple of the form (target1, target2).

    Examples
    ========

    """
    is_hermitian = True
    gate_name = 'SWAP'
    gate_name_latex = r'\text{SWAP}'

    def get_target_matrix(self, format='sympy'):
        return matrix_cache.get_matrix('SWAP', format)

    def decompose(self, **options):
        """Decompose the SWAP gate into CNOT gates."""
        i, j = self.targets[0], self.targets[1]
        g1 = CNotGate(i, j)
        g2 = CNotGate(j, i)
        return g1*g2*g1

    def plot_gate(self, circ_plot, gate_idx):
        min_wire = int(_min(self.targets))
        max_wire = int(_max(self.targets))
        circ_plot.control_line(gate_idx, min_wire, max_wire)
        circ_plot.swap_point(gate_idx, min_wire)
        circ_plot.swap_point(gate_idx, max_wire)

    def _represent_ZGate(self, basis, **options):
        """Represent the SWAP gate in the computational basis.

        The following representation is used to compute this:

        SWAP = |1><1|x|1><1| + |0><0|x|0><0| + |1><0|x|0><1| + |0><1|x|1><0|
        """
        format = options.get('format', 'sympy')
        targets = [int(t) for t in self.targets]
        min_target = _min(targets)
        max_target = _max(targets)
        nqubits = options.get('nqubits', self.min_qubits)

        op01 = matrix_cache.get_matrix('op01', format)
        op10 = matrix_cache.get_matrix('op10', format)
        op11 = matrix_cache.get_matrix('op11', format)
        op00 = matrix_cache.get_matrix('op00', format)
        eye2 = matrix_cache.get_matrix('eye2', format)

        result = None
        for i, j in ((op01, op10), (op10, op01), (op00, op00), (op11, op11)):
            product = nqubits*[eye2]
            product[nqubits - min_target - 1] = i
            product[nqubits - max_target - 1] = j
            new_result = matrix_tensor_product(*product)
            if result is None:
                result = new_result
            else:
                result = result + new_result

        return result


# Aliases for gate names.
CNOT = CNotGate
SWAP = SwapGate
def CPHASE(a,b): return CGateS((a,),Z(b))


#-----------------------------------------------------------------------------
# Represent
#-----------------------------------------------------------------------------


def represent_zbasis(controls, targets, target_matrix, nqubits, format='sympy'):
    """Represent a gate with controls, targets and target_matrix.

    This function does the low-level work of representing gates as matrices
    in the standard computational basis (ZGate). Currently, we support two
    main cases:

    1. One target qubit and no control qubits.
    2. One target qubits and multiple control qubits.

    For the base of multiple controls, we use the following expression [1]:

    1_{2**n} + (|1><1|)^{(n-1)} x (target-matrix - 1_{2})

    Parameters
    ----------
    controls : list, tuple
        A sequence of control qubits.
    targets : list, tuple
        A sequence of target qubits.
    target_matrix : sympy.Matrix, numpy.matrix, scipy.sparse
        The matrix form of the transformation to be performed on the target
        qubits.  The format of this matrix must match that passed into
        the `format` argument.
    nqubits : int
        The total number of qubits used for the representation.
    format : str
        The format of the final matrix ('sympy', 'numpy', 'scipy.sparse').

    Examples
    ========

    References
    ----------
    [1] http://www.johnlapeyre.com/qinf/qinf_html/node6.html.
    """
    controls = [int(x) for x in controls]
    targets = [int(x) for x in targets]
    nqubits = int(nqubits)

    # This checks for the format as well.
    op11 = matrix_cache.get_matrix('op11', format)
    eye2 = matrix_cache.get_matrix('eye2', format)

    # Plain single qubit case
    if len(controls) == 0 and len(targets) == 1:
        product = []
        bit = targets[0]
        # Fill product with [I1,Gate,I2] such that the unitaries,
        # I, cause the gate to be applied to the correct Qubit
        if bit != nqubits - 1:
            product.append(matrix_eye(2**(nqubits - bit - 1), format=format))
        product.append(target_matrix)
        if bit != 0:
            product.append(matrix_eye(2**bit, format=format))
        return matrix_tensor_product(*product)

    # Single target, multiple controls.
    elif len(targets) == 1 and len(controls) >= 1:
        target = targets[0]

        # Build the non-trivial part.
        product2 = []
        for i in range(nqubits):
            product2.append(matrix_eye(2, format=format))
        for control in controls:
            product2[nqubits - 1 - control] = op11
        product2[nqubits - 1 - target] = target_matrix - eye2

        return matrix_eye(2**nqubits, format=format) + \
            matrix_tensor_product(*product2)

    # Multi-target, multi-control is not yet implemented.
    else:
        raise NotImplementedError(
            'The representation of multi-target, multi-control gates '
            'is not implemented.'
        )


#-----------------------------------------------------------------------------
# Gate manipulation functions.
#-----------------------------------------------------------------------------


def gate_simp(circuit):
    """Simplifies gates symbolically

    It first sorts gates using gate_sort. It then applies basic
    simplification rules to the circuit, e.g., XGate**2 = Identity
    """

    # Bubble sort out gates that commute.
    circuit = gate_sort(circuit)

    # Do simplifications by subing a simplification into the first element
    # which can be simplified. We recursively call gate_simp with new circuit
    # as input more simplifications exist.
    if isinstance(circuit, Add):
        return sum(gate_simp(t) for t in circuit.args)
    elif isinstance(circuit, Mul):
        circuit_args = circuit.args
    elif isinstance(circuit, Pow):
        b, e = circuit.as_base_exp()
        circuit_args = (gate_simp(b)**e,)
    else:
        return circuit

    # Iterate through each element in circuit, simplify if possible.
    for i in range(len(circuit_args)):
        # H,X,Y or Z squared is 1.
        # T**2 = S, S**2 = Z
        if isinstance(circuit_args[i], Pow):
            if isinstance(circuit_args[i].base,
                (HadamardGate, XGate, YGate, ZGate)) \
                    and isinstance(circuit_args[i].exp, Number):
                # Build a new circuit taking replacing the
                # H,X,Y,Z squared with one.
                newargs = (circuit_args[:i] +
                          (circuit_args[i].base**(circuit_args[i].exp % 2),) +
                           circuit_args[i + 1:])
                # Recursively simplify the new circuit.
                circuit = gate_simp(Mul(*newargs))
                break
            elif isinstance(circuit_args[i].base, PhaseGate):
                # Build a new circuit taking old circuit but splicing
                # in simplification.
                newargs = circuit_args[:i]
                # Replace PhaseGate**2 with ZGate.
                newargs = newargs + (ZGate(circuit_args[i].base.args[0])**
                (Integer(circuit_args[i].exp/2)), circuit_args[i].base**
                (circuit_args[i].exp % 2))
                # Append the last elements.
                newargs = newargs + circuit_args[i + 1:]
                # Recursively simplify the new circuit.
                circuit = gate_simp(Mul(*newargs))
                break
            elif isinstance(circuit_args[i].base, TGate):
                # Build a new circuit taking all the old elements.
                newargs = circuit_args[:i]

                # Put an Phasegate in place of any TGate**2.
                newargs = newargs + (PhaseGate(circuit_args[i].base.args[0])**
                Integer(circuit_args[i].exp/2), circuit_args[i].base**
                    (circuit_args[i].exp % 2))

                # Append the last elements.
                newargs = newargs + circuit_args[i + 1:]
                # Recursively simplify the new circuit.
                circuit = gate_simp(Mul(*newargs))
                break
    return circuit


def gate_sort(circuit):
    """Sorts the gates while keeping track of commutation relations

    This function uses a bubble sort to rearrange the order of gate
    application. Keeps track of Quantum computations special commutation
    relations (e.g. things that apply to the same Qubit do not commute with
    each other)

    circuit is the Mul of gates that are to be sorted.
    """
    # Make sure we have an Add or Mul.
    if isinstance(circuit, Add):
        return sum(gate_sort(t) for t in circuit.args)
    if isinstance(circuit, Pow):
        return gate_sort(circuit.base)**circuit.exp
    elif isinstance(circuit, Gate):
        return circuit
    if not isinstance(circuit, Mul):
        return circuit

    changes = True
    while changes:
        changes = False
        circ_array = circuit.args
        for i in range(len(circ_array) - 1):
            # Go through each element and switch ones that are in wrong order
            if isinstance(circ_array[i], (Gate, Pow)) and \
                    isinstance(circ_array[i + 1], (Gate, Pow)):
                # If we have a Pow object, look at only the base
                first_base, first_exp = circ_array[i].as_base_exp()
                second_base, second_exp = circ_array[i + 1].as_base_exp()

                # Use SymPy's hash based sorting. This is not mathematical
                # sorting, but is rather based on comparing hashes of objects.
                # See Basic.compare for details.
                if first_base.compare(second_base) > 0:
                    if Commutator(first_base, second_base).doit() == 0:
                        new_args = (circuit.args[:i] + (circuit.args[i + 1],) +
                                   (circuit.args[i],) + circuit.args[i + 2:])
                        circuit = Mul(*new_args)
                        changes = True
                        break
                    if AntiCommutator(first_base, second_base).doit() == 0:
                        new_args = (circuit.args[:i] + (circuit.args[i + 1],) +
                                   (circuit.args[i],) + circuit.args[i + 2:])
                        sign = _S.NegativeOne**(first_exp*second_exp)
                        circuit = sign*Mul(*new_args)
                        changes = True
                        break
    return circuit


#-----------------------------------------------------------------------------
# Utility functions
#-----------------------------------------------------------------------------


def random_circuit(ngates, nqubits, gate_space=(X, Y, Z, S, T, H, CNOT, SWAP)):
    """Return a random circuit of ngates and nqubits.

    This uses an equally weighted sample of (X, Y, Z, S, T, H, CNOT, SWAP)
    gates.

    Parameters
    ----------
    ngates : int
        The number of gates in the circuit.
    nqubits : int
        The number of qubits in the circuit.
    gate_space : tuple
        A tuple of the gate classes that will be used in the circuit.
        Repeating gate classes multiple times in this tuple will increase
        the frequency they appear in the random circuit.
    """
    qubit_space = range(nqubits)
    result = []
    for i in range(ngates):
        g = random.choice(gate_space)
        if g == CNotGate or g == SwapGate:
            qubits = random.sample(qubit_space, 2)
            g = g(*qubits)
        else:
            qubit = random.choice(qubit_space)
            g = g(qubit)
        result.append(g)
    return Mul(*result)


def zx_basis_transform(self, format='sympy'):
    """Transformation matrix from Z to X basis."""
    return matrix_cache.get_matrix('ZX', format)


def zy_basis_transform(self, format='sympy'):
    """Transformation matrix from Z to Y basis."""
    return matrix_cache.get_matrix('ZY', format)
