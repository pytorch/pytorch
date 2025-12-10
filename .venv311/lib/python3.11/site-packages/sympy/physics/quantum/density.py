from itertools import product

from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import expand
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import log
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.operator import HermitianOperator
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.matrixutils import numpy_ndarray, scipy_sparse_matrix, to_numpy
from sympy.physics.quantum.trace import Tr


class Density(HermitianOperator):
    """Density operator for representing mixed states.

    TODO: Density operator support for Qubits

    Parameters
    ==========

    values : tuples/lists
    Each tuple/list should be of form (state, prob) or [state,prob]

    Examples
    ========

    Create a density operator with 2 states represented by Kets.

    >>> from sympy.physics.quantum.state import Ket
    >>> from sympy.physics.quantum.density import Density
    >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
    >>> d
    Density((|0>, 0.5),(|1>, 0.5))

    """
    @classmethod
    def _eval_args(cls, args):
        # call this to qsympify the args
        args = super()._eval_args(args)

        for arg in args:
            # Check if arg is a tuple
            if not (isinstance(arg, Tuple) and len(arg) == 2):
                raise ValueError("Each argument should be of form [state,prob]"
                                 " or ( state, prob )")

        return args

    def states(self):
        """Return list of all states.

        Examples
        ========

        >>> from sympy.physics.quantum.state import Ket
        >>> from sympy.physics.quantum.density import Density
        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
        >>> d.states()
        (|0>, |1>)

        """
        return Tuple(*[arg[0] for arg in self.args])

    def probs(self):
        """Return list of all probabilities.

        Examples
        ========

        >>> from sympy.physics.quantum.state import Ket
        >>> from sympy.physics.quantum.density import Density
        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
        >>> d.probs()
        (0.5, 0.5)

        """
        return Tuple(*[arg[1] for arg in self.args])

    def get_state(self, index):
        """Return specific state by index.

        Parameters
        ==========

        index : index of state to be returned

        Examples
        ========

        >>> from sympy.physics.quantum.state import Ket
        >>> from sympy.physics.quantum.density import Density
        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
        >>> d.states()[1]
        |1>

        """
        state = self.args[index][0]
        return state

    def get_prob(self, index):
        """Return probability of specific state by index.

        Parameters
        ===========

        index : index of states whose probability is returned.

        Examples
        ========

        >>> from sympy.physics.quantum.state import Ket
        >>> from sympy.physics.quantum.density import Density
        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
        >>> d.probs()[1]
        0.500000000000000

        """
        prob = self.args[index][1]
        return prob

    def apply_op(self, op):
        """op will operate on each individual state.

        Parameters
        ==========

        op : Operator

        Examples
        ========

        >>> from sympy.physics.quantum.state import Ket
        >>> from sympy.physics.quantum.density import Density
        >>> from sympy.physics.quantum.operator import Operator
        >>> A = Operator('A')
        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
        >>> d.apply_op(A)
        Density((A*|0>, 0.5),(A*|1>, 0.5))

        """
        new_args = [(op*state, prob) for (state, prob) in self.args]
        return Density(*new_args)

    def doit(self, **hints):
        """Expand the density operator into an outer product format.

        Examples
        ========

        >>> from sympy.physics.quantum.state import Ket
        >>> from sympy.physics.quantum.density import Density
        >>> from sympy.physics.quantum.operator import Operator
        >>> A = Operator('A')
        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
        >>> d.doit()
        0.5*|0><0| + 0.5*|1><1|

        """

        terms = []
        for (state, prob) in self.args:
            state = state.expand()  # needed to break up (a+b)*c
            if (isinstance(state, Add)):
                for arg in product(state.args, repeat=2):
                    terms.append(prob*self._generate_outer_prod(arg[0],
                                                                arg[1]))
            else:
                terms.append(prob*self._generate_outer_prod(state, state))

        return Add(*terms)

    def _generate_outer_prod(self, arg1, arg2):
        c_part1, nc_part1 = arg1.args_cnc()
        c_part2, nc_part2 = arg2.args_cnc()

        if (len(nc_part1) == 0 or len(nc_part2) == 0):
            raise ValueError('Atleast one-pair of'
                             ' Non-commutative instance required'
                             ' for outer product.')

        # We were able to remove some tensor product simplifications that
        # used to be here as those transformations are not automatically
        # applied by transforms.py.
        op = Mul(*nc_part1)*Dagger(Mul(*nc_part2))

        return Mul(*c_part1)*Mul(*c_part2) * op

    def _represent(self, **options):
        return represent(self.doit(), **options)

    def _print_operator_name_latex(self, printer, *args):
        return r'\rho'

    def _print_operator_name_pretty(self, printer, *args):
        return prettyForm('\N{GREEK SMALL LETTER RHO}')

    def _eval_trace(self, **kwargs):
        indices = kwargs.get('indices', [])
        return Tr(self.doit(), indices).doit()

    def entropy(self):
        """ Compute the entropy of a density matrix.

        Refer to density.entropy() method  for examples.
        """
        return entropy(self)


def entropy(density):
    """Compute the entropy of a matrix/density object.

    This computes -Tr(density*ln(density)) using the eigenvalue decomposition
    of density, which is given as either a Density instance or a matrix
    (numpy.ndarray, sympy.Matrix or scipy.sparse).

    Parameters
    ==========

    density : density matrix of type Density, SymPy matrix,
    scipy.sparse or numpy.ndarray

    Examples
    ========

    >>> from sympy.physics.quantum.density import Density, entropy
    >>> from sympy.physics.quantum.spin import JzKet
    >>> from sympy import S
    >>> up = JzKet(S(1)/2,S(1)/2)
    >>> down = JzKet(S(1)/2,-S(1)/2)
    >>> d = Density((up,S(1)/2),(down,S(1)/2))
    >>> entropy(d)
    log(2)/2

    """
    if isinstance(density, Density):
        density = represent(density)  # represent in Matrix

    if isinstance(density, scipy_sparse_matrix):
        density = to_numpy(density)

    if isinstance(density, Matrix):
        eigvals = density.eigenvals().keys()
        return expand(-sum(e*log(e) for e in eigvals))
    elif isinstance(density, numpy_ndarray):
        import numpy as np
        eigvals = np.linalg.eigvals(density)
        return -np.sum(eigvals*np.log(eigvals))
    else:
        raise ValueError(
            "numpy.ndarray, scipy.sparse or SymPy matrix expected")


def fidelity(state1, state2):
    """ Computes the fidelity [1]_ between two quantum states

    The arguments provided to this function should be a square matrix or a
    Density object. If it is a square matrix, it is assumed to be diagonalizable.

    Parameters
    ==========

    state1, state2 : a density matrix or Matrix


    Examples
    ========

    >>> from sympy import S, sqrt
    >>> from sympy.physics.quantum.dagger import Dagger
    >>> from sympy.physics.quantum.spin import JzKet
    >>> from sympy.physics.quantum.density import fidelity
    >>> from sympy.physics.quantum.represent import represent
    >>>
    >>> up = JzKet(S(1)/2,S(1)/2)
    >>> down = JzKet(S(1)/2,-S(1)/2)
    >>> amp = 1/sqrt(2)
    >>> updown = (amp*up) + (amp*down)
    >>>
    >>> # represent turns Kets into matrices
    >>> up_dm = represent(up*Dagger(up))
    >>> down_dm = represent(down*Dagger(down))
    >>> updown_dm = represent(updown*Dagger(updown))
    >>>
    >>> fidelity(up_dm, up_dm)
    1
    >>> fidelity(up_dm, down_dm) #orthogonal states
    0
    >>> fidelity(up_dm, updown_dm).evalf().round(3)
    0.707

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fidelity_of_quantum_states

    """
    state1 = represent(state1) if isinstance(state1, Density) else state1
    state2 = represent(state2) if isinstance(state2, Density) else state2

    if not isinstance(state1, Matrix) or not isinstance(state2, Matrix):
        raise ValueError("state1 and state2 must be of type Density or Matrix "
                         "received type=%s for state1 and type=%s for state2" %
                         (type(state1), type(state2)))

    if state1.shape != state2.shape and state1.is_square:
        raise ValueError("The dimensions of both args should be equal and the "
                         "matrix obtained should be a square matrix")

    sqrt_state1 = state1**S.Half
    return Tr((sqrt_state1*state2*sqrt_state1)**S.Half).doit()
