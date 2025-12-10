from collections import deque
from sympy.core.random import randint

from sympy.external import import_module
from sympy.core.basic import Basic
from sympy.core.mul import Mul
from sympy.core.numbers import Number, equal_valued
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.dagger import Dagger

__all__ = [
    # Public interfaces
    'generate_gate_rules',
    'generate_equivalent_ids',
    'GateIdentity',
    'bfs_identity_search',
    'random_identity_search',

    # "Private" functions
    'is_scalar_sparse_matrix',
    'is_scalar_nonsparse_matrix',
    'is_degenerate',
    'is_reducible',
]

np = import_module('numpy')
scipy = import_module('scipy', import_kwargs={'fromlist': ['sparse']})


def is_scalar_sparse_matrix(circuit, nqubits, identity_only, eps=1e-11):
    """Checks if a given scipy.sparse matrix is a scalar matrix.

    A scalar matrix is such that B = bI, where B is the scalar
    matrix, b is some scalar multiple, and I is the identity
    matrix.  A scalar matrix would have only the element b along
    it's main diagonal and zeroes elsewhere.

    Parameters
    ==========

    circuit : Gate tuple
        Sequence of quantum gates representing a quantum circuit
    nqubits : int
        Number of qubits in the circuit
    identity_only : bool
        Check for only identity matrices
    eps : number
        The tolerance value for zeroing out elements in the matrix.
        Values in the range [-eps, +eps] will be changed to a zero.
    """

    if not np or not scipy:
        pass

    matrix = represent(Mul(*circuit), nqubits=nqubits,
                       format='scipy.sparse')

    # In some cases, represent returns a 1D scalar value in place
    # of a multi-dimensional scalar matrix
    if (isinstance(matrix, int)):
        return matrix == 1 if identity_only else True

    # If represent returns a matrix, check if the matrix is diagonal
    # and if every item along the diagonal is the same
    else:
        # Due to floating pointing operations, must zero out
        # elements that are "very" small in the dense matrix
        # See parameter for default value.

        # Get the ndarray version of the dense matrix
        dense_matrix = matrix.todense().getA()
        # Since complex values can't be compared, must split
        # the matrix into real and imaginary components
        # Find the real values in between -eps and eps
        bool_real = np.logical_and(dense_matrix.real > -eps,
                                   dense_matrix.real < eps)
        # Find the imaginary values between -eps and eps
        bool_imag = np.logical_and(dense_matrix.imag > -eps,
                                   dense_matrix.imag < eps)
        # Replaces values between -eps and eps with 0
        corrected_real = np.where(bool_real, 0.0, dense_matrix.real)
        corrected_imag = np.where(bool_imag, 0.0, dense_matrix.imag)
        # Convert the matrix with real values into imaginary values
        corrected_imag = corrected_imag * complex(1j)
        # Recombine the real and imaginary components
        corrected_dense = corrected_real + corrected_imag

        # Check if it's diagonal
        row_indices = corrected_dense.nonzero()[0]
        col_indices = corrected_dense.nonzero()[1]
        # Check if the rows indices and columns indices are the same
        # If they match, then matrix only contains elements along diagonal
        bool_indices = row_indices == col_indices
        is_diagonal = bool_indices.all()

        first_element = corrected_dense[0][0]
        # If the first element is a zero, then can't rescale matrix
        # and definitely not diagonal
        if (first_element == 0.0 + 0.0j):
            return False

        # The dimensions of the dense matrix should still
        # be 2^nqubits if there are elements all along the
        # the main diagonal
        trace_of_corrected = (corrected_dense/first_element).trace()
        expected_trace = pow(2, nqubits)
        has_correct_trace = trace_of_corrected == expected_trace

        # If only looking for identity matrices
        # first element must be a 1
        real_is_one = abs(first_element.real - 1.0) < eps
        imag_is_zero = abs(first_element.imag) < eps
        is_one = real_is_one and imag_is_zero
        is_identity = is_one if identity_only else True
        return bool(is_diagonal and has_correct_trace and is_identity)


def is_scalar_nonsparse_matrix(circuit, nqubits, identity_only, eps=None):
    """Checks if a given circuit, in matrix form, is equivalent to
    a scalar value.

    Parameters
    ==========

    circuit : Gate tuple
        Sequence of quantum gates representing a quantum circuit
    nqubits : int
        Number of qubits in the circuit
    identity_only : bool
        Check for only identity matrices
    eps : number
        This argument is ignored. It is just for signature compatibility with
        is_scalar_sparse_matrix.

    Note: Used in situations when is_scalar_sparse_matrix has bugs
    """

    matrix = represent(Mul(*circuit), nqubits=nqubits)

    # In some cases, represent returns a 1D scalar value in place
    # of a multi-dimensional scalar matrix
    if (isinstance(matrix, Number)):
        return matrix == 1 if identity_only else True

    # If represent returns a matrix, check if the matrix is diagonal
    # and if every item along the diagonal is the same
    else:
        # Added up the diagonal elements
        matrix_trace = matrix.trace()
        # Divide the trace by the first element in the matrix
        # if matrix is not required to be the identity matrix
        adjusted_matrix_trace = (matrix_trace/matrix[0]
                                 if not identity_only
                                 else matrix_trace)

        is_identity = equal_valued(matrix[0], 1) if identity_only else True

        has_correct_trace = adjusted_matrix_trace == pow(2, nqubits)

        # The matrix is scalar if it's diagonal and the adjusted trace
        # value is equal to 2^nqubits
        return bool(
            matrix.is_diagonal() and has_correct_trace and is_identity)

if np and scipy:
    is_scalar_matrix = is_scalar_sparse_matrix
else:
    is_scalar_matrix = is_scalar_nonsparse_matrix


def _get_min_qubits(a_gate):
    if isinstance(a_gate, Pow):
        return a_gate.base.min_qubits
    else:
        return a_gate.min_qubits


def ll_op(left, right):
    """Perform a LL operation.

    A LL operation multiplies both left and right circuits
    with the dagger of the left circuit's leftmost gate, and
    the dagger is multiplied on the left side of both circuits.

    If a LL is possible, it returns the new gate rule as a
    2-tuple (LHS, RHS), where LHS is the left circuit and
    and RHS is the right circuit of the new rule.
    If a LL is not possible, None is returned.

    Parameters
    ==========

    left : Gate tuple
        The left circuit of a gate rule expression.
    right : Gate tuple
        The right circuit of a gate rule expression.

    Examples
    ========

    Generate a new gate rule using a LL operation:

    >>> from sympy.physics.quantum.identitysearch import ll_op
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> ll_op((x, y, z), ())
    ((Y(0), Z(0)), (X(0),))

    >>> ll_op((y, z), (x,))
    ((Z(0),), (Y(0), X(0)))
    """

    if (len(left) > 0):
        ll_gate = left[0]
        ll_gate_is_unitary = is_scalar_matrix(
            (Dagger(ll_gate), ll_gate), _get_min_qubits(ll_gate), True)

    if (len(left) > 0 and ll_gate_is_unitary):
        # Get the new left side w/o the leftmost gate
        new_left = left[1:len(left)]
        # Add the leftmost gate to the left position on the right side
        new_right = (Dagger(ll_gate),) + right
        # Return the new gate rule
        return (new_left, new_right)

    return None


def lr_op(left, right):
    """Perform a LR operation.

    A LR operation multiplies both left and right circuits
    with the dagger of the left circuit's rightmost gate, and
    the dagger is multiplied on the right side of both circuits.

    If a LR is possible, it returns the new gate rule as a
    2-tuple (LHS, RHS), where LHS is the left circuit and
    and RHS is the right circuit of the new rule.
    If a LR is not possible, None is returned.

    Parameters
    ==========

    left : Gate tuple
        The left circuit of a gate rule expression.
    right : Gate tuple
        The right circuit of a gate rule expression.

    Examples
    ========

    Generate a new gate rule using a LR operation:

    >>> from sympy.physics.quantum.identitysearch import lr_op
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> lr_op((x, y, z), ())
    ((X(0), Y(0)), (Z(0),))

    >>> lr_op((x, y), (z,))
    ((X(0),), (Z(0), Y(0)))
    """

    if (len(left) > 0):
        lr_gate = left[len(left) - 1]
        lr_gate_is_unitary = is_scalar_matrix(
            (Dagger(lr_gate), lr_gate), _get_min_qubits(lr_gate), True)

    if (len(left) > 0 and lr_gate_is_unitary):
        # Get the new left side w/o the rightmost gate
        new_left = left[0:len(left) - 1]
        # Add the rightmost gate to the right position on the right side
        new_right = right + (Dagger(lr_gate),)
        # Return the new gate rule
        return (new_left, new_right)

    return None


def rl_op(left, right):
    """Perform a RL operation.

    A RL operation multiplies both left and right circuits
    with the dagger of the right circuit's leftmost gate, and
    the dagger is multiplied on the left side of both circuits.

    If a RL is possible, it returns the new gate rule as a
    2-tuple (LHS, RHS), where LHS is the left circuit and
    and RHS is the right circuit of the new rule.
    If a RL is not possible, None is returned.

    Parameters
    ==========

    left : Gate tuple
        The left circuit of a gate rule expression.
    right : Gate tuple
        The right circuit of a gate rule expression.

    Examples
    ========

    Generate a new gate rule using a RL operation:

    >>> from sympy.physics.quantum.identitysearch import rl_op
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> rl_op((x,), (y, z))
    ((Y(0), X(0)), (Z(0),))

    >>> rl_op((x, y), (z,))
    ((Z(0), X(0), Y(0)), ())
    """

    if (len(right) > 0):
        rl_gate = right[0]
        rl_gate_is_unitary = is_scalar_matrix(
            (Dagger(rl_gate), rl_gate), _get_min_qubits(rl_gate), True)

    if (len(right) > 0 and rl_gate_is_unitary):
        # Get the new right side w/o the leftmost gate
        new_right = right[1:len(right)]
        # Add the leftmost gate to the left position on the left side
        new_left = (Dagger(rl_gate),) + left
        # Return the new gate rule
        return (new_left, new_right)

    return None


def rr_op(left, right):
    """Perform a RR operation.

    A RR operation multiplies both left and right circuits
    with the dagger of the right circuit's rightmost gate, and
    the dagger is multiplied on the right side of both circuits.

    If a RR is possible, it returns the new gate rule as a
    2-tuple (LHS, RHS), where LHS is the left circuit and
    and RHS is the right circuit of the new rule.
    If a RR is not possible, None is returned.

    Parameters
    ==========

    left : Gate tuple
        The left circuit of a gate rule expression.
    right : Gate tuple
        The right circuit of a gate rule expression.

    Examples
    ========

    Generate a new gate rule using a RR operation:

    >>> from sympy.physics.quantum.identitysearch import rr_op
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> rr_op((x, y), (z,))
    ((X(0), Y(0), Z(0)), ())

    >>> rr_op((x,), (y, z))
    ((X(0), Z(0)), (Y(0),))
    """

    if (len(right) > 0):
        rr_gate = right[len(right) - 1]
        rr_gate_is_unitary = is_scalar_matrix(
            (Dagger(rr_gate), rr_gate), _get_min_qubits(rr_gate), True)

    if (len(right) > 0 and rr_gate_is_unitary):
        # Get the new right side w/o the rightmost gate
        new_right = right[0:len(right) - 1]
        # Add the rightmost gate to the right position on the right side
        new_left = left + (Dagger(rr_gate),)
        # Return the new gate rule
        return (new_left, new_right)

    return None


def generate_gate_rules(gate_seq, return_as_muls=False):
    """Returns a set of gate rules.  Each gate rules is represented
    as a 2-tuple of tuples or Muls.  An empty tuple represents an arbitrary
    scalar value.

    This function uses the four operations (LL, LR, RL, RR)
    to generate the gate rules.

    A gate rule is an expression such as ABC = D or AB = CD, where
    A, B, C, and D are gates.  Each value on either side of the
    equal sign represents a circuit.  The four operations allow
    one to find a set of equivalent circuits from a gate identity.
    The letters denoting the operation tell the user what
    activities to perform on each expression.  The first letter
    indicates which side of the equal sign to focus on.  The
    second letter indicates which gate to focus on given the
    side.  Once this information is determined, the inverse
    of the gate is multiplied on both circuits to create a new
    gate rule.

    For example, given the identity, ABCD = 1, a LL operation
    means look at the left value and multiply both left sides by the
    inverse of the leftmost gate A.  If A is Hermitian, the inverse
    of A is still A.  The resulting new rule is BCD = A.

    The following is a summary of the four operations.  Assume
    that in the examples, all gates are Hermitian.

        LL : left circuit, left multiply
             ABCD = E -> AABCD = AE -> BCD = AE
        LR : left circuit, right multiply
             ABCD = E -> ABCDD = ED -> ABC = ED
        RL : right circuit, left multiply
             ABC = ED -> EABC = EED -> EABC = D
        RR : right circuit, right multiply
             AB = CD -> ABD = CDD -> ABD = C

    The number of gate rules generated is n*(n+1), where n
    is the number of gates in the sequence (unproven).

    Parameters
    ==========

    gate_seq : Gate tuple, Mul, or Number
        A variable length tuple or Mul of Gates whose product is equal to
        a scalar matrix
    return_as_muls : bool
        True to return a set of Muls; False to return a set of tuples

    Examples
    ========

    Find the gate rules of the current circuit using tuples:

    >>> from sympy.physics.quantum.identitysearch import generate_gate_rules
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> generate_gate_rules((x, x))
    {((X(0),), (X(0),)), ((X(0), X(0)), ())}

    >>> generate_gate_rules((x, y, z))
    {((), (X(0), Z(0), Y(0))), ((), (Y(0), X(0), Z(0))),
     ((), (Z(0), Y(0), X(0))), ((X(0),), (Z(0), Y(0))),
     ((Y(0),), (X(0), Z(0))), ((Z(0),), (Y(0), X(0))),
     ((X(0), Y(0)), (Z(0),)), ((Y(0), Z(0)), (X(0),)),
     ((Z(0), X(0)), (Y(0),)), ((X(0), Y(0), Z(0)), ()),
     ((Y(0), Z(0), X(0)), ()), ((Z(0), X(0), Y(0)), ())}

    Find the gate rules of the current circuit using Muls:

    >>> generate_gate_rules(x*x, return_as_muls=True)
    {(1, 1)}

    >>> generate_gate_rules(x*y*z, return_as_muls=True)
    {(1, X(0)*Z(0)*Y(0)), (1, Y(0)*X(0)*Z(0)),
     (1, Z(0)*Y(0)*X(0)), (X(0)*Y(0), Z(0)),
     (Y(0)*Z(0), X(0)), (Z(0)*X(0), Y(0)),
     (X(0)*Y(0)*Z(0), 1), (Y(0)*Z(0)*X(0), 1),
     (Z(0)*X(0)*Y(0), 1), (X(0), Z(0)*Y(0)),
     (Y(0), X(0)*Z(0)), (Z(0), Y(0)*X(0))}
    """

    if isinstance(gate_seq, Number):
        if return_as_muls:
            return {(S.One, S.One)}
        else:
            return {((), ())}

    elif isinstance(gate_seq, Mul):
        gate_seq = gate_seq.args

    # Each item in queue is a 3-tuple:
    #     i)   first item is the left side of an equality
    #    ii)   second item is the right side of an equality
    #   iii)   third item is the number of operations performed
    # The argument, gate_seq, will start on the left side, and
    # the right side will be empty, implying the presence of an
    # identity.
    queue = deque()
    # A set of gate rules
    rules = set()
    # Maximum number of operations to perform
    max_ops = len(gate_seq)

    def process_new_rule(new_rule, ops):
        if new_rule is not None:
            new_left, new_right = new_rule

            if new_rule not in rules and (new_right, new_left) not in rules:
                rules.add(new_rule)
            # If haven't reached the max limit on operations
            if ops + 1 < max_ops:
                queue.append(new_rule + (ops + 1,))

    queue.append((gate_seq, (), 0))
    rules.add((gate_seq, ()))

    while len(queue) > 0:
        left, right, ops = queue.popleft()

        # Do a LL
        new_rule = ll_op(left, right)
        process_new_rule(new_rule, ops)
        # Do a LR
        new_rule = lr_op(left, right)
        process_new_rule(new_rule, ops)
        # Do a RL
        new_rule = rl_op(left, right)
        process_new_rule(new_rule, ops)
        # Do a RR
        new_rule = rr_op(left, right)
        process_new_rule(new_rule, ops)

    if return_as_muls:
        # Convert each rule as tuples into a rule as muls
        mul_rules = set()
        for rule in rules:
            left, right = rule
            mul_rules.add((Mul(*left), Mul(*right)))

        rules = mul_rules

    return rules


def generate_equivalent_ids(gate_seq, return_as_muls=False):
    """Returns a set of equivalent gate identities.

    A gate identity is a quantum circuit such that the product
    of the gates in the circuit is equal to a scalar value.
    For example, XYZ = i, where X, Y, Z are the Pauli gates and
    i is the imaginary value, is considered a gate identity.

    This function uses the four operations (LL, LR, RL, RR)
    to generate the gate rules and, subsequently, to locate equivalent
    gate identities.

    Note that all equivalent identities are reachable in n operations
    from the starting gate identity, where n is the number of gates
    in the sequence.

    The max number of gate identities is 2n, where n is the number
    of gates in the sequence (unproven).

    Parameters
    ==========

    gate_seq : Gate tuple, Mul, or Number
        A variable length tuple or Mul of Gates whose product is equal to
        a scalar matrix.
    return_as_muls: bool
        True to return as Muls; False to return as tuples

    Examples
    ========

    Find equivalent gate identities from the current circuit with tuples:

    >>> from sympy.physics.quantum.identitysearch import generate_equivalent_ids
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> generate_equivalent_ids((x, x))
    {(X(0), X(0))}

    >>> generate_equivalent_ids((x, y, z))
    {(X(0), Y(0), Z(0)), (X(0), Z(0), Y(0)), (Y(0), X(0), Z(0)),
     (Y(0), Z(0), X(0)), (Z(0), X(0), Y(0)), (Z(0), Y(0), X(0))}

    Find equivalent gate identities from the current circuit with Muls:

    >>> generate_equivalent_ids(x*x, return_as_muls=True)
    {1}

    >>> generate_equivalent_ids(x*y*z, return_as_muls=True)
    {X(0)*Y(0)*Z(0), X(0)*Z(0)*Y(0), Y(0)*X(0)*Z(0),
     Y(0)*Z(0)*X(0), Z(0)*X(0)*Y(0), Z(0)*Y(0)*X(0)}
    """

    if isinstance(gate_seq, Number):
        return {S.One}
    elif isinstance(gate_seq, Mul):
        gate_seq = gate_seq.args

    # Filter through the gate rules and keep the rules
    # with an empty tuple either on the left or right side

    # A set of equivalent gate identities
    eq_ids = set()

    gate_rules = generate_gate_rules(gate_seq)
    for rule in gate_rules:
        l, r = rule
        if l == ():
            eq_ids.add(r)
        elif r == ():
            eq_ids.add(l)

    if return_as_muls:
        convert_to_mul = lambda id_seq: Mul(*id_seq)
        eq_ids = set(map(convert_to_mul, eq_ids))

    return eq_ids


class GateIdentity(Basic):
    """Wrapper class for circuits that reduce to a scalar value.

    A gate identity is a quantum circuit such that the product
    of the gates in the circuit is equal to a scalar value.
    For example, XYZ = i, where X, Y, Z are the Pauli gates and
    i is the imaginary value, is considered a gate identity.

    Parameters
    ==========

    args : Gate tuple
        A variable length tuple of Gates that form an identity.

    Examples
    ========

    Create a GateIdentity and look at its attributes:

    >>> from sympy.physics.quantum.identitysearch import GateIdentity
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> an_identity = GateIdentity(x, y, z)
    >>> an_identity.circuit
    X(0)*Y(0)*Z(0)

    >>> an_identity.equivalent_ids
    {(X(0), Y(0), Z(0)), (X(0), Z(0), Y(0)), (Y(0), X(0), Z(0)),
     (Y(0), Z(0), X(0)), (Z(0), X(0), Y(0)), (Z(0), Y(0), X(0))}
    """

    def __new__(cls, *args):
        # args should be a tuple - a variable length argument list
        obj = Basic.__new__(cls, *args)
        obj._circuit = Mul(*args)
        obj._rules = generate_gate_rules(args)
        obj._eq_ids = generate_equivalent_ids(args)

        return obj

    @property
    def circuit(self):
        return self._circuit

    @property
    def gate_rules(self):
        return self._rules

    @property
    def equivalent_ids(self):
        return self._eq_ids

    @property
    def sequence(self):
        return self.args

    def __str__(self):
        """Returns the string of gates in a tuple."""
        return str(self.circuit)


def is_degenerate(identity_set, gate_identity):
    """Checks if a gate identity is a permutation of another identity.

    Parameters
    ==========

    identity_set : set
        A Python set with GateIdentity objects.
    gate_identity : GateIdentity
        The GateIdentity to check for existence in the set.

    Examples
    ========

    Check if the identity is a permutation of another identity:

    >>> from sympy.physics.quantum.identitysearch import (
    ...     GateIdentity, is_degenerate)
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> an_identity = GateIdentity(x, y, z)
    >>> id_set = {an_identity}
    >>> another_id = (y, z, x)
    >>> is_degenerate(id_set, another_id)
    True

    >>> another_id = (x, x)
    >>> is_degenerate(id_set, another_id)
    False
    """

    # For now, just iteratively go through the set and check if the current
    # gate_identity is a permutation of an identity in the set
    for an_id in identity_set:
        if (gate_identity in an_id.equivalent_ids):
            return True
    return False


def is_reducible(circuit, nqubits, begin, end):
    """Determines if a circuit is reducible by checking
    if its subcircuits are scalar values.

    Parameters
    ==========

    circuit : Gate tuple
        A tuple of Gates representing a circuit.  The circuit to check
        if a gate identity is contained in a subcircuit.
    nqubits : int
        The number of qubits the circuit operates on.
    begin : int
        The leftmost gate in the circuit to include in a subcircuit.
    end : int
        The rightmost gate in the circuit to include in a subcircuit.

    Examples
    ========

    Check if the circuit can be reduced:

    >>> from sympy.physics.quantum.identitysearch import is_reducible
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> is_reducible((x, y, z), 1, 0, 3)
    True

    Check if an interval in the circuit can be reduced:

    >>> is_reducible((x, y, z), 1, 1, 3)
    False

    >>> is_reducible((x, y, y), 1, 1, 3)
    True
    """

    current_circuit = ()
    # Start from the gate at "end" and go down to almost the gate at "begin"
    for ndx in reversed(range(begin, end)):
        next_gate = circuit[ndx]
        current_circuit = (next_gate,) + current_circuit

        # If a circuit as a matrix is equivalent to a scalar value
        if (is_scalar_matrix(current_circuit, nqubits, False)):
            return True

    return False


def bfs_identity_search(gate_list, nqubits, max_depth=None,
       identity_only=False):
    """Constructs a set of gate identities from the list of possible gates.

    Performs a breadth first search over the space of gate identities.
    This allows the finding of the shortest gate identities first.

    Parameters
    ==========

    gate_list : list, Gate
        A list of Gates from which to search for gate identities.
    nqubits : int
        The number of qubits the quantum circuit operates on.
    max_depth : int
        The longest quantum circuit to construct from gate_list.
    identity_only : bool
        True to search for gate identities that reduce to identity;
        False to search for gate identities that reduce to a scalar.

    Examples
    ========

    Find a list of gate identities:

    >>> from sympy.physics.quantum.identitysearch import bfs_identity_search
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> bfs_identity_search([x], 1, max_depth=2)
    {GateIdentity(X(0), X(0))}

    >>> bfs_identity_search([x, y, z], 1)
    {GateIdentity(X(0), X(0)), GateIdentity(Y(0), Y(0)),
     GateIdentity(Z(0), Z(0)), GateIdentity(X(0), Y(0), Z(0))}

    Find a list of identities that only equal to 1:

    >>> bfs_identity_search([x, y, z], 1, identity_only=True)
    {GateIdentity(X(0), X(0)), GateIdentity(Y(0), Y(0)),
     GateIdentity(Z(0), Z(0))}
    """

    if max_depth is None or max_depth <= 0:
        max_depth = len(gate_list)

    id_only = identity_only

    # Start with an empty sequence (implicitly contains an IdentityGate)
    queue = deque([()])

    # Create an empty set of gate identities
    ids = set()

    # Begin searching for gate identities in given space.
    while (len(queue) > 0):
        current_circuit = queue.popleft()

        for next_gate in gate_list:
            new_circuit = current_circuit + (next_gate,)

            # Determines if a (strict) subcircuit is a scalar matrix
            circuit_reducible = is_reducible(new_circuit, nqubits,
                                             1, len(new_circuit))

            # In many cases when the matrix is a scalar value,
            # the evaluated matrix will actually be an integer
            if (is_scalar_matrix(new_circuit, nqubits, id_only) and
                not is_degenerate(ids, new_circuit) and
                    not circuit_reducible):
                ids.add(GateIdentity(*new_circuit))

            elif (len(new_circuit) < max_depth and
                  not circuit_reducible):
                queue.append(new_circuit)

    return ids


def random_identity_search(gate_list, numgates, nqubits):
    """Randomly selects numgates from gate_list and checks if it is
    a gate identity.

    If the circuit is a gate identity, the circuit is returned;
    Otherwise, None is returned.
    """

    gate_size = len(gate_list)
    circuit = ()

    for i in range(numgates):
        next_gate = gate_list[randint(0, gate_size - 1)]
        circuit = circuit + (next_gate,)

    is_scalar = is_scalar_matrix(circuit, nqubits, False)

    return circuit if is_scalar else None
