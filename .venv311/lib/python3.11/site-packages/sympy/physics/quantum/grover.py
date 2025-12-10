"""Grover's algorithm and helper functions.

Todo:

* W gate construction (or perhaps -W gate based on Mermin's book)
* Generalize the algorithm for an unknown function that returns 1 on multiple
  qubit states, not just one.
* Implement _represent_ZGate in OracleGate
"""

from sympy.core.numbers import pi
from sympy.core.sympify import sympify
from sympy.core.basic import Atom
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import eye
from sympy.core.numbers import NegativeOne
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qexpr import QuantumError
from sympy.physics.quantum.hilbert import ComplexSpace
from sympy.physics.quantum.operator import UnitaryOperator
from sympy.physics.quantum.gate import Gate
from sympy.physics.quantum.qubit import IntQubit

__all__ = [
    'OracleGate',
    'WGate',
    'superposition_basis',
    'grover_iteration',
    'apply_grover'
]


def superposition_basis(nqubits):
    """Creates an equal superposition of the computational basis.

    Parameters
    ==========

    nqubits : int
        The number of qubits.

    Returns
    =======

    state : Qubit
        An equal superposition of the computational basis with nqubits.

    Examples
    ========

    Create an equal superposition of 2 qubits::

        >>> from sympy.physics.quantum.grover import superposition_basis
        >>> superposition_basis(2)
        |0>/2 + |1>/2 + |2>/2 + |3>/2
    """

    amp = 1/sqrt(2**nqubits)
    return sum(amp*IntQubit(n, nqubits=nqubits) for n in range(2**nqubits))

class OracleGateFunction(Atom):
    """Wrapper for python functions used in `OracleGate`s"""

    def __new__(cls, function):
        if not callable(function):
            raise TypeError('Callable expected, got: %r' % function)
        obj = Atom.__new__(cls)
        obj.function = function
        return obj

    def _hashable_content(self):
        return type(self), self.function

    def __call__(self, *args):
        return self.function(*args)


class OracleGate(Gate):
    """A black box gate.

    The gate marks the desired qubits of an unknown function by flipping
    the sign of the qubits.  The unknown function returns true when it
    finds its desired qubits and false otherwise.

    Parameters
    ==========

    qubits : int
        Number of qubits.

    oracle : callable
        A callable function that returns a boolean on a computational basis.

    Examples
    ========

    Apply an Oracle gate that flips the sign of ``|2>`` on different qubits::

        >>> from sympy.physics.quantum.qubit import IntQubit
        >>> from sympy.physics.quantum.qapply import qapply
        >>> from sympy.physics.quantum.grover import OracleGate
        >>> f = lambda qubits: qubits == IntQubit(2)
        >>> v = OracleGate(2, f)
        >>> qapply(v*IntQubit(2))
        -|2>
        >>> qapply(v*IntQubit(3))
        |3>
    """

    gate_name = 'V'
    gate_name_latex = 'V'

    #-------------------------------------------------------------------------
    # Initialization/creation
    #-------------------------------------------------------------------------

    @classmethod
    def _eval_args(cls, args):
        if len(args) != 2:
            raise QuantumError(
                'Insufficient/excessive arguments to Oracle.  Please ' +
                'supply the number of qubits and an unknown function.'
            )
        sub_args = (args[0],)
        sub_args = UnitaryOperator._eval_args(sub_args)
        if not sub_args[0].is_Integer:
            raise TypeError('Integer expected, got: %r' % sub_args[0])

        function = args[1]
        if not isinstance(function, OracleGateFunction):
            function = OracleGateFunction(function)

        return (sub_args[0], function)

    @classmethod
    def _eval_hilbert_space(cls, args):
        """This returns the smallest possible Hilbert space."""
        return ComplexSpace(2)**args[0]

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def search_function(self):
        """The unknown function that helps find the sought after qubits."""
        return self.label[1]

    @property
    def targets(self):
        """A tuple of target qubits."""
        return sympify(tuple(range(self.args[0])))

    #-------------------------------------------------------------------------
    # Apply
    #-------------------------------------------------------------------------

    def _apply_operator_Qubit(self, qubits, **options):
        """Apply this operator to a Qubit subclass.

        Parameters
        ==========

        qubits : Qubit
            The qubit subclass to apply this operator to.

        Returns
        =======

        state : Expr
            The resulting quantum state.
        """
        if qubits.nqubits != self.nqubits:
            raise QuantumError(
                'OracleGate operates on %r qubits, got: %r'
                % (self.nqubits, qubits.nqubits)
            )
        # If function returns 1 on qubits
            # return the negative of the qubits (flip the sign)
        if self.search_function(qubits):
            return -qubits
        else:
            return qubits

    #-------------------------------------------------------------------------
    # Represent
    #-------------------------------------------------------------------------

    def _represent_ZGate(self, basis, **options):
        """
        Represent the OracleGate in the computational basis.
        """
        nbasis = 2**self.nqubits  # compute it only once
        matrixOracle = eye(nbasis)
        # Flip the sign given the output of the oracle function
        for i in range(nbasis):
            if self.search_function(IntQubit(i, nqubits=self.nqubits)):
                matrixOracle[i, i] = NegativeOne()
        return matrixOracle


class WGate(Gate):
    """General n qubit W Gate in Grover's algorithm.

    The gate performs the operation ``2|phi><phi| - 1`` on some qubits.
    ``|phi> = (tensor product of n Hadamards)*(|0> with n qubits)``

    Parameters
    ==========

    nqubits : int
        The number of qubits to operate on

    """

    gate_name = 'W'
    gate_name_latex = 'W'

    @classmethod
    def _eval_args(cls, args):
        if len(args) != 1:
            raise QuantumError(
                'Insufficient/excessive arguments to W gate.  Please ' +
                'supply the number of qubits to operate on.'
            )
        args = UnitaryOperator._eval_args(args)
        if not args[0].is_Integer:
            raise TypeError('Integer expected, got: %r' % args[0])
        return args

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def targets(self):
        return sympify(tuple(reversed(range(self.args[0]))))

    #-------------------------------------------------------------------------
    # Apply
    #-------------------------------------------------------------------------

    def _apply_operator_Qubit(self, qubits, **options):
        """
        qubits: a set of qubits (Qubit)
        Returns: quantum object (quantum expression - QExpr)
        """
        if qubits.nqubits != self.nqubits:
            raise QuantumError(
                'WGate operates on %r qubits, got: %r'
                % (self.nqubits, qubits.nqubits)
            )

        # See 'Quantum Computer Science' by David Mermin p.92 -> W|a> result
        # Return (2/(sqrt(2^n)))|phi> - |a> where |a> is the current basis
        # state and phi is the superposition of basis states (see function
        # create_computational_basis above)
        basis_states = superposition_basis(self.nqubits)
        change_to_basis = (2/sqrt(2**self.nqubits))*basis_states
        return change_to_basis - qubits


def grover_iteration(qstate, oracle):
    """Applies one application of the Oracle and W Gate, WV.

    Parameters
    ==========

    qstate : Qubit
        A superposition of qubits.
    oracle : OracleGate
        The black box operator that flips the sign of the desired basis qubits.

    Returns
    =======

    Qubit : The qubits after applying the Oracle and W gate.

    Examples
    ========

    Perform one iteration of grover's algorithm to see a phase change::

        >>> from sympy.physics.quantum.qapply import qapply
        >>> from sympy.physics.quantum.qubit import IntQubit
        >>> from sympy.physics.quantum.grover import OracleGate
        >>> from sympy.physics.quantum.grover import superposition_basis
        >>> from sympy.physics.quantum.grover import grover_iteration
        >>> numqubits = 2
        >>> basis_states = superposition_basis(numqubits)
        >>> f = lambda qubits: qubits == IntQubit(2)
        >>> v = OracleGate(numqubits, f)
        >>> qapply(grover_iteration(basis_states, v))
        |2>

    """
    wgate = WGate(oracle.nqubits)
    return wgate*oracle*qstate


def apply_grover(oracle, nqubits, iterations=None):
    """Applies grover's algorithm.

    Parameters
    ==========

    oracle : callable
        The unknown callable function that returns true when applied to the
        desired qubits and false otherwise.

    Returns
    =======

    state : Expr
        The resulting state after Grover's algorithm has been iterated.

    Examples
    ========

    Apply grover's algorithm to an even superposition of 2 qubits::

        >>> from sympy.physics.quantum.qapply import qapply
        >>> from sympy.physics.quantum.qubit import IntQubit
        >>> from sympy.physics.quantum.grover import apply_grover
        >>> f = lambda qubits: qubits == IntQubit(2)
        >>> qapply(apply_grover(f, 2))
        |2>

    """
    if nqubits <= 0:
        raise QuantumError(
            'Grover\'s algorithm needs nqubits > 0, received %r qubits'
            % nqubits
        )
    if iterations is None:
        iterations = floor(sqrt(2**nqubits)*(pi/4))

    v = OracleGate(nqubits, oracle)
    iterated = superposition_basis(nqubits)
    for iter in range(iterations):
        iterated = grover_iteration(iterated, v)
        iterated = qapply(iterated)

    return iterated
