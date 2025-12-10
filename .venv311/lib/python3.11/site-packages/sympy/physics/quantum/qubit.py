"""Qubits for quantum computing.

Todo:
* Finish implementing measurement logic. This should include POVM.
* Update docstrings.
* Update tests.
"""


import math

from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.exponential import log
from sympy.core.basic import _sympify
from sympy.external.gmpy import SYMPY_INTS
from sympy.matrices import Matrix, zeros
from sympy.printing.pretty.stringpict import prettyForm

from sympy.physics.quantum.hilbert import ComplexSpace
from sympy.physics.quantum.state import Ket, Bra, State

from sympy.physics.quantum.qexpr import QuantumError
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.matrixutils import (
    numpy_ndarray, scipy_sparse_matrix
)
from mpmath.libmp.libintmath import bitcount

__all__ = [
    'Qubit',
    'QubitBra',
    'IntQubit',
    'IntQubitBra',
    'qubit_to_matrix',
    'matrix_to_qubit',
    'matrix_to_density',
    'measure_all',
    'measure_partial',
    'measure_partial_oneshot',
    'measure_all_oneshot'
]

#-----------------------------------------------------------------------------
# Qubit Classes
#-----------------------------------------------------------------------------


class QubitState(State):
    """Base class for Qubit and QubitBra."""

    #-------------------------------------------------------------------------
    # Initialization/creation
    #-------------------------------------------------------------------------

    @classmethod
    def _eval_args(cls, args):
        # If we are passed a QubitState or subclass, we just take its qubit
        # values directly.
        if len(args) == 1 and isinstance(args[0], QubitState):
            return args[0].qubit_values

        # Turn strings into tuple of strings
        if len(args) == 1 and isinstance(args[0], str):
            args = tuple( S.Zero if qb == "0" else S.One for qb in args[0])
        else:
            args = tuple( S.Zero if qb == "0" else S.One if qb == "1" else qb for qb in args)
        args = tuple(_sympify(arg) for arg in args)

        # Validate input (must have 0 or 1 input)
        for element in args:
            if element not in (S.Zero, S.One):
                raise ValueError(
                    "Qubit values must be 0 or 1, got: %r" % element)
        return args

    @classmethod
    def _eval_hilbert_space(cls, args):
        return ComplexSpace(2)**len(args)

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def dimension(self):
        """The number of Qubits in the state."""
        return len(self.qubit_values)

    @property
    def nqubits(self):
        return self.dimension

    @property
    def qubit_values(self):
        """Returns the values of the qubits as a tuple."""
        return self.label

    #-------------------------------------------------------------------------
    # Special methods
    #-------------------------------------------------------------------------

    def __len__(self):
        return self.dimension

    def __getitem__(self, bit):
        return self.qubit_values[int(self.dimension - bit - 1)]

    #-------------------------------------------------------------------------
    # Utility methods
    #-------------------------------------------------------------------------

    def flip(self, *bits):
        """Flip the bit(s) given."""
        newargs = list(self.qubit_values)
        for i in bits:
            bit = int(self.dimension - i - 1)
            if newargs[bit] == 1:
                newargs[bit] = 0
            else:
                newargs[bit] = 1
        return self.__class__(*tuple(newargs))


class Qubit(QubitState, Ket):
    """A multi-qubit ket in the computational (z) basis.

    We use the normal convention that the least significant qubit is on the
    right, so ``|00001>`` has a 1 in the least significant qubit.

    Parameters
    ==========

    values : list, str
        The qubit values as a list of ints ([0,0,0,1,1,]) or a string ('011').

    Examples
    ========

    Create a qubit in a couple of different ways and look at their attributes:

        >>> from sympy.physics.quantum.qubit import Qubit
        >>> Qubit(0,0,0)
        |000>
        >>> q = Qubit('0101')
        >>> q
        |0101>

        >>> q.nqubits
        4
        >>> len(q)
        4
        >>> q.dimension
        4
        >>> q.qubit_values
        (0, 1, 0, 1)

    We can flip the value of an individual qubit:

        >>> q.flip(1)
        |0111>

    We can take the dagger of a Qubit to get a bra:

        >>> from sympy.physics.quantum.dagger import Dagger
        >>> Dagger(q)
        <0101|
        >>> type(Dagger(q))
        <class 'sympy.physics.quantum.qubit.QubitBra'>

    Inner products work as expected:

        >>> ip = Dagger(q)*q
        >>> ip
        <0101|0101>
        >>> ip.doit()
        1
    """

    @classmethod
    def dual_class(self):
        return QubitBra

    def _eval_innerproduct_QubitBra(self, bra, **hints):
        if self.label == bra.label:
            return S.One
        else:
            return S.Zero

    def _represent_default_basis(self, **options):
        return self._represent_ZGate(None, **options)

    def _represent_ZGate(self, basis, **options):
        """Represent this qubits in the computational basis (ZGate).
        """
        _format = options.get('format', 'sympy')
        n = 1
        definite_state = 0
        for it in reversed(self.qubit_values):
            definite_state += n*it
            n = n*2
        result = [0]*(2**self.dimension)
        result[int(definite_state)] = 1
        if _format == 'sympy':
            return Matrix(result)
        elif _format == 'numpy':
            import numpy as np
            return np.array(result, dtype='complex').transpose()
        elif _format == 'scipy.sparse':
            from scipy import sparse
            return sparse.csr_matrix(result, dtype='complex').transpose()

    def _eval_trace(self, bra, **kwargs):
        indices = kwargs.get('indices', [])

        #sort index list to begin trace from most-significant
        #qubit
        sorted_idx = list(indices)
        if len(sorted_idx) == 0:
            sorted_idx = list(range(0, self.nqubits))
        sorted_idx.sort()

        #trace out for each of index
        new_mat = self*bra
        for i in range(len(sorted_idx) - 1, -1, -1):
            # start from tracing out from leftmost qubit
            new_mat = self._reduced_density(new_mat, int(sorted_idx[i]))

        if (len(sorted_idx) == self.nqubits):
            #in case full trace was requested
            return new_mat[0]
        else:
            return matrix_to_density(new_mat)

    def _reduced_density(self, matrix, qubit, **options):
        """Compute the reduced density matrix by tracing out one qubit.
           The qubit argument should be of type Python int, since it is used
           in bit operations
        """
        def find_index_that_is_projected(j, k, qubit):
            bit_mask = 2**qubit - 1
            return ((j >> qubit) << (1 + qubit)) + (j & bit_mask) + (k << qubit)

        old_matrix = represent(matrix, **options)
        old_size = old_matrix.cols
        #we expect the old_size to be even
        new_size = old_size//2
        new_matrix = Matrix().zeros(new_size)

        for i in range(new_size):
            for j in range(new_size):
                for k in range(2):
                    col = find_index_that_is_projected(j, k, qubit)
                    row = find_index_that_is_projected(i, k, qubit)
                    new_matrix[i, j] += old_matrix[row, col]

        return new_matrix


class QubitBra(QubitState, Bra):
    """A multi-qubit bra in the computational (z) basis.

    We use the normal convention that the least significant qubit is on the
    right, so ``|00001>`` has a 1 in the least significant qubit.

    Parameters
    ==========

    values : list, str
        The qubit values as a list of ints ([0,0,0,1,1,]) or a string ('011').

    See also
    ========

    Qubit: Examples using qubits

    """
    @classmethod
    def dual_class(self):
        return Qubit


class IntQubitState(QubitState):
    """A base class for qubits that work with binary representations."""

    @classmethod
    def _eval_args(cls, args, nqubits=None):
        # The case of a QubitState instance
        if len(args) == 1 and isinstance(args[0], QubitState):
            return QubitState._eval_args(args)
        # otherwise, args should be integer
        elif not all(isinstance(a, (int, Integer)) for a in args):
            raise ValueError('values must be integers, got (%s)' % (tuple(type(a) for a in args),))
        # use nqubits if specified
        if nqubits is not None:
            if not isinstance(nqubits, (int, Integer)):
                raise ValueError('nqubits must be an integer, got (%s)' % type(nqubits))
            if len(args) != 1:
                raise ValueError(
                    'too many positional arguments (%s). should be (number, nqubits=n)' % (args,))
            return cls._eval_args_with_nqubits(args[0], nqubits)
        # For a single argument, we construct the binary representation of
        # that integer with the minimal number of bits.
        if len(args) == 1 and args[0] > 1:
            #rvalues is the minimum number of bits needed to express the number
            rvalues = reversed(range(bitcount(abs(args[0]))))
            qubit_values = [(args[0] >> i) & 1 for i in rvalues]
            return QubitState._eval_args(qubit_values)
        # For two numbers, the second number is the number of bits
        # on which it is expressed, so IntQubit(0,5) == |00000>.
        elif len(args) == 2 and args[1] > 1:
            return cls._eval_args_with_nqubits(args[0], args[1])
        else:
            return QubitState._eval_args(args)

    @classmethod
    def _eval_args_with_nqubits(cls, number, nqubits):
        need = bitcount(abs(number))
        if nqubits < need:
            raise ValueError(
                'cannot represent %s with %s bits' % (number, nqubits))
        qubit_values = [(number >> i) & 1 for i in reversed(range(nqubits))]
        return QubitState._eval_args(qubit_values)

    def as_int(self):
        """Return the numerical value of the qubit."""
        number = 0
        n = 1
        for i in reversed(self.qubit_values):
            number += n*i
            n = n << 1
        return number

    def _print_label(self, printer, *args):
        return str(self.as_int())

    def _print_label_pretty(self, printer, *args):
        label = self._print_label(printer, *args)
        return prettyForm(label)

    _print_label_repr = _print_label
    _print_label_latex = _print_label


class IntQubit(IntQubitState, Qubit):
    """A qubit ket that store integers as binary numbers in qubit values.

    The differences between this class and ``Qubit`` are:

    * The form of the constructor.
    * The qubit values are printed as their corresponding integer, rather
      than the raw qubit values. The internal storage format of the qubit
      values in the same as ``Qubit``.

    Parameters
    ==========

    values : int, tuple
        If a single argument, the integer we want to represent in the qubit
        values. This integer will be represented using the fewest possible
        number of qubits.
        If a pair of integers and the second value is more than one, the first
        integer gives the integer to represent in binary form and the second
        integer gives the number of qubits to use.
        List of zeros and ones is also accepted to generate qubit by bit pattern.

    nqubits : int
        The integer that represents the number of qubits.
        This number should be passed with keyword ``nqubits=N``.
        You can use this in order to avoid ambiguity of Qubit-style tuple of bits.
        Please see the example below for more details.

    Examples
    ========

    Create a qubit for the integer 5:

        >>> from sympy.physics.quantum.qubit import IntQubit
        >>> from sympy.physics.quantum.qubit import Qubit
        >>> q = IntQubit(5)
        >>> q
        |5>

    We can also create an ``IntQubit`` by passing a ``Qubit`` instance.

        >>> q = IntQubit(Qubit('101'))
        >>> q
        |5>
        >>> q.as_int()
        5
        >>> q.nqubits
        3
        >>> q.qubit_values
        (1, 0, 1)

    We can go back to the regular qubit form.

        >>> Qubit(q)
        |101>

    Please note that ``IntQubit`` also accepts a ``Qubit``-style list of bits.
    So, the code below yields qubits 3, not a single bit ``1``.

        >>> IntQubit(1, 1)
        |3>

    To avoid ambiguity, use ``nqubits`` parameter.
    Use of this keyword is recommended especially when you provide the values by variables.

        >>> IntQubit(1, nqubits=1)
        |1>
        >>> a = 1
        >>> IntQubit(a, nqubits=1)
        |1>
    """
    @classmethod
    def dual_class(self):
        return IntQubitBra

    def _eval_innerproduct_IntQubitBra(self, bra, **hints):
        return Qubit._eval_innerproduct_QubitBra(self, bra)

class IntQubitBra(IntQubitState, QubitBra):
    """A qubit bra that store integers as binary numbers in qubit values."""

    @classmethod
    def dual_class(self):
        return IntQubit


#-----------------------------------------------------------------------------
# Qubit <---> Matrix conversion functions
#-----------------------------------------------------------------------------


def matrix_to_qubit(matrix):
    """Convert from the matrix repr. to a sum of Qubit objects.

    Parameters
    ----------
    matrix : Matrix, numpy.matrix, scipy.sparse
        The matrix to build the Qubit representation of. This works with
        SymPy matrices, numpy matrices and scipy.sparse sparse matrices.

    Examples
    ========

    Represent a state and then go back to its qubit form:

        >>> from sympy.physics.quantum.qubit import matrix_to_qubit, Qubit
        >>> from sympy.physics.quantum.represent import represent
        >>> q = Qubit('01')
        >>> matrix_to_qubit(represent(q))
        |01>
    """
    # Determine the format based on the type of the input matrix
    format = 'sympy'
    if isinstance(matrix, numpy_ndarray):
        format = 'numpy'
    if isinstance(matrix, scipy_sparse_matrix):
        format = 'scipy.sparse'

    # Make sure it is of correct dimensions for a Qubit-matrix representation.
    # This logic should work with sympy, numpy or scipy.sparse matrices.
    if matrix.shape[0] == 1:
        mlistlen = matrix.shape[1]
        nqubits = log(mlistlen, 2)
        ket = False
        cls = QubitBra
    elif matrix.shape[1] == 1:
        mlistlen = matrix.shape[0]
        nqubits = log(mlistlen, 2)
        ket = True
        cls = Qubit
    else:
        raise QuantumError(
            'Matrix must be a row/column vector, got %r' % matrix
        )
    if not isinstance(nqubits, Integer):
        raise QuantumError('Matrix must be a row/column vector of size '
                           '2**nqubits, got: %r' % matrix)
    # Go through each item in matrix, if element is non-zero, make it into a
    # Qubit item times the element.
    result = 0
    for i in range(mlistlen):
        if ket:
            element = matrix[i, 0]
        else:
            element = matrix[0, i]
        if format in ('numpy', 'scipy.sparse'):
            element = complex(element)
        if element:
            # Form Qubit array; 0 in bit-locations where i is 0, 1 in
            # bit-locations where i is 1
            qubit_array = [int(i & (1 << x) != 0) for x in range(nqubits)]
            qubit_array.reverse()
            result = result + element*cls(*qubit_array)

    # If SymPy simplified by pulling out a constant coefficient, undo that.
    if isinstance(result, (Mul, Add, Pow)):
        result = result.expand()

    return result


def matrix_to_density(mat):
    """
    Works by finding the eigenvectors and eigenvalues of the matrix.
    We know we can decompose rho by doing:
    sum(EigenVal*|Eigenvect><Eigenvect|)
    """
    from sympy.physics.quantum.density import Density
    eigen = mat.eigenvects()
    args = [[matrix_to_qubit(Matrix(
        [vector, ])), x[0]] for x in eigen for vector in x[2] if x[0] != 0]
    if (len(args) == 0):
        return S.Zero
    else:
        return Density(*args)


def qubit_to_matrix(qubit, format='sympy'):
    """Converts an Add/Mul of Qubit objects into it's matrix representation

    This function is the inverse of ``matrix_to_qubit`` and is a shorthand
    for ``represent(qubit)``.
    """
    return represent(qubit, format=format)


#-----------------------------------------------------------------------------
# Measurement
#-----------------------------------------------------------------------------


def measure_all(qubit, format='sympy', normalize=True):
    """Perform an ensemble measurement of all qubits.

    Parameters
    ==========

    qubit : Qubit, Add
        The qubit to measure. This can be any Qubit or a linear combination
        of them.
    format : str
        The format of the intermediate matrices to use. Possible values are
        ('sympy','numpy','scipy.sparse'). Currently only 'sympy' is
        implemented.

    Returns
    =======

    result : list
        A list that consists of primitive states and their probabilities.

    Examples
    ========

        >>> from sympy.physics.quantum.qubit import Qubit, measure_all
        >>> from sympy.physics.quantum.gate import H
        >>> from sympy.physics.quantum.qapply import qapply

        >>> c = H(0)*H(1)*Qubit('00')
        >>> c
        H(0)*H(1)*|00>
        >>> q = qapply(c)
        >>> measure_all(q)
        [(|00>, 1/4), (|01>, 1/4), (|10>, 1/4), (|11>, 1/4)]
    """
    m = qubit_to_matrix(qubit, format)

    if format == 'sympy':
        results = []

        if normalize:
            m = m.normalized()

        size = max(m.shape)  # Max of shape to account for bra or ket
        nqubits = int(math.log(size)/math.log(2))
        for i in range(size):
            if m[i]:
                results.append(
                    (Qubit(IntQubit(i, nqubits=nqubits)), m[i]*conjugate(m[i]))
                )
        return results
    else:
        raise NotImplementedError(
            "This function cannot handle non-SymPy matrix formats yet"
        )


def measure_partial(qubit, bits, format='sympy', normalize=True):
    """Perform a partial ensemble measure on the specified qubits.

    Parameters
    ==========

    qubits : Qubit
        The qubit to measure.  This can be any Qubit or a linear combination
        of them.
    bits : tuple
        The qubits to measure.
    format : str
        The format of the intermediate matrices to use. Possible values are
        ('sympy','numpy','scipy.sparse'). Currently only 'sympy' is
        implemented.

    Returns
    =======

    result : list
        A list that consists of primitive states and their probabilities.

    Examples
    ========

        >>> from sympy.physics.quantum.qubit import Qubit, measure_partial
        >>> from sympy.physics.quantum.gate import H
        >>> from sympy.physics.quantum.qapply import qapply

        >>> c = H(0)*H(1)*Qubit('00')
        >>> c
        H(0)*H(1)*|00>
        >>> q = qapply(c)
        >>> measure_partial(q, (0,))
        [(sqrt(2)*|00>/2 + sqrt(2)*|10>/2, 1/2), (sqrt(2)*|01>/2 + sqrt(2)*|11>/2, 1/2)]
    """
    m = qubit_to_matrix(qubit, format)

    if isinstance(bits, (SYMPY_INTS, Integer)):
        bits = (int(bits),)

    if format == 'sympy':
        if normalize:
            m = m.normalized()

        possible_outcomes = _get_possible_outcomes(m, bits)

        # Form output from function.
        output = []
        for outcome in possible_outcomes:
            # Calculate probability of finding the specified bits with
            # given values.
            prob_of_outcome = 0
            prob_of_outcome += (outcome.H*outcome)[0]

            # If the output has a chance, append it to output with found
            # probability.
            if prob_of_outcome != 0:
                if normalize:
                    next_matrix = matrix_to_qubit(outcome.normalized())
                else:
                    next_matrix = matrix_to_qubit(outcome)

                output.append((
                    next_matrix,
                    prob_of_outcome
                ))

        return output
    else:
        raise NotImplementedError(
            "This function cannot handle non-SymPy matrix formats yet"
        )


def measure_partial_oneshot(qubit, bits, format='sympy'):
    """Perform a partial oneshot measurement on the specified qubits.

    A oneshot measurement is equivalent to performing a measurement on a
    quantum system. This type of measurement does not return the probabilities
    like an ensemble measurement does, but rather returns *one* of the
    possible resulting states. The exact state that is returned is determined
    by picking a state randomly according to the ensemble probabilities.

    Parameters
    ----------
    qubits : Qubit
        The qubit to measure.  This can be any Qubit or a linear combination
        of them.
    bits : tuple
        The qubits to measure.
    format : str
        The format of the intermediate matrices to use. Possible values are
        ('sympy','numpy','scipy.sparse'). Currently only 'sympy' is
        implemented.

    Returns
    -------
    result : Qubit
        The qubit that the system collapsed to upon measurement.
    """
    import random
    m = qubit_to_matrix(qubit, format)

    if format == 'sympy':
        m = m.normalized()
        possible_outcomes = _get_possible_outcomes(m, bits)

        # Form output from function
        random_number = random.random()
        total_prob = 0
        for outcome in possible_outcomes:
            # Calculate probability of finding the specified bits
            # with given values
            total_prob += (outcome.H*outcome)[0]
            if total_prob >= random_number:
                return matrix_to_qubit(outcome.normalized())
    else:
        raise NotImplementedError(
            "This function cannot handle non-SymPy matrix formats yet"
        )


def _get_possible_outcomes(m, bits):
    """Get the possible states that can be produced in a measurement.

    Parameters
    ----------
    m : Matrix
        The matrix representing the state of the system.
    bits : tuple, list
        Which bits will be measured.

    Returns
    -------
    result : list
        The list of possible states which can occur given this measurement.
        These are un-normalized so we can derive the probability of finding
        this state by taking the inner product with itself
    """

    # This is filled with loads of dirty binary tricks...You have been warned

    size = max(m.shape)  # Max of shape to account for bra or ket
    nqubits = int(math.log2(size) + .1)  # Number of qubits possible

    # Make the output states and put in output_matrices, nothing in them now.
    # Each state will represent a possible outcome of the measurement
    # Thus, output_matrices[0] is the matrix which we get when all measured
    # bits return 0. and output_matrices[1] is the matrix for only the 0th
    # bit being true
    output_matrices = []
    for i in range(1 << len(bits)):
        output_matrices.append(zeros(2**nqubits, 1))

    # Bitmasks will help sort how to determine possible outcomes.
    # When the bit mask is and-ed with a matrix-index,
    # it will determine which state that index belongs to
    bit_masks = []
    for bit in bits:
        bit_masks.append(1 << bit)

    # Make possible outcome states
    for i in range(2**nqubits):
        trueness = 0  # This tells us to which output_matrix this value belongs
        # Find trueness
        for j in range(len(bit_masks)):
            if i & bit_masks[j]:
                trueness += j + 1
        # Put the value in the correct output matrix
        output_matrices[trueness][i] = m[i]
    return output_matrices


def measure_all_oneshot(qubit, format='sympy'):
    """Perform a oneshot ensemble measurement on all qubits.

    A oneshot measurement is equivalent to performing a measurement on a
    quantum system. This type of measurement does not return the probabilities
    like an ensemble measurement does, but rather returns *one* of the
    possible resulting states. The exact state that is returned is determined
    by picking a state randomly according to the ensemble probabilities.

    Parameters
    ----------
    qubits : Qubit
        The qubit to measure.  This can be any Qubit or a linear combination
        of them.
    format : str
        The format of the intermediate matrices to use. Possible values are
        ('sympy','numpy','scipy.sparse'). Currently only 'sympy' is
        implemented.

    Returns
    -------
    result : Qubit
        The qubit that the system collapsed to upon measurement.
    """
    import random
    m = qubit_to_matrix(qubit)

    if format == 'sympy':
        m = m.normalized()
        random_number = random.random()
        total = 0
        result = 0
        for i in m:
            total += i*i.conjugate()
            if total > random_number:
                break
            result += 1
        return Qubit(IntQubit(result, nqubits=int(math.log2(max(m.shape)) + .1)))
    else:
        raise NotImplementedError(
            "This function cannot handle non-SymPy matrix formats yet"
        )
