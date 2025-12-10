from __future__ import annotations
import random
import itertools
from typing import Sequence as tSequence
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import (Function, Lambda)
from sympy.core.mul import Mul
from sympy.core.intfunc import igcd
from sympy.core.numbers import (Integer, Rational, oo, pi)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.gamma_functions import gamma
from sympy.logic.boolalg import (And, Not, Or)
from sympy.matrices.exceptions import NonSquareMatrixError
from sympy.matrices.dense import (Matrix, eye, ones, zeros)
from sympy.matrices.expressions.blockmatrix import BlockMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import Identity
from sympy.matrices.immutable import ImmutableMatrix
from sympy.sets.conditionset import ConditionSet
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Range
from sympy.sets.sets import (FiniteSet, Intersection, Interval, Set, Union)
from sympy.solvers.solveset import linsolve
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import strongly_connected_components
from sympy.stats.joint_rv import JointDistribution
from sympy.stats.joint_rv_types import JointDistributionHandmade
from sympy.stats.rv import (RandomIndexedSymbol, random_symbols, RandomSymbol,
                            _symbol_converter, _value_check, pspace, given,
                           dependent, is_random, sample_iter, Distribution,
                           Density)
from sympy.stats.stochastic_process import StochasticPSpace
from sympy.stats.symbolic_probability import Probability, Expectation
from sympy.stats.frv_types import Bernoulli, BernoulliDistribution, FiniteRV
from sympy.stats.drv_types import Poisson, PoissonDistribution
from sympy.stats.crv_types import Normal, NormalDistribution, Gamma, GammaDistribution
from sympy.core.sympify import _sympify, sympify

EmptySet = S.EmptySet

__all__ = [
    'StochasticProcess',
    'DiscreteTimeStochasticProcess',
    'DiscreteMarkovChain',
    'TransitionMatrixOf',
    'StochasticStateSpaceOf',
    'GeneratorMatrixOf',
    'ContinuousMarkovChain',
    'BernoulliProcess',
    'PoissonProcess',
    'WienerProcess',
    'GammaProcess'
]


@is_random.register(Indexed)
def _(x):
    return is_random(x.base)

@is_random.register(RandomIndexedSymbol)  # type: ignore
def _(x):
    return True

def _set_converter(itr):
    """
    Helper function for converting list/tuple/set to Set.
    If parameter is not an instance of list/tuple/set then
    no operation is performed.

    Returns
    =======

    Set
        The argument converted to Set.


    Raises
    ======

    TypeError
        If the argument is not an instance of list/tuple/set.
    """
    if isinstance(itr, (list, tuple, set)):
        itr = FiniteSet(*itr)
    if not isinstance(itr, Set):
        raise TypeError("%s is not an instance of list/tuple/set."%(itr))
    return itr

def _state_converter(itr: tSequence) -> Tuple | Range:
    """
    Helper function for converting list/tuple/set/Range/Tuple/FiniteSet
    to tuple/Range.
    """
    itr_ret: Tuple | Range

    if isinstance(itr, (Tuple, set, FiniteSet)):
        itr_ret = Tuple(*(sympify(i) if isinstance(i, str) else i for i in itr))

    elif isinstance(itr, (list, tuple)):
        # check if states are unique
        if len(set(itr)) != len(itr):
            raise ValueError('The state space must have unique elements.')
        itr_ret = Tuple(*(sympify(i) if isinstance(i, str) else i for i in itr))

    elif isinstance(itr, Range):
        # the only ordered set in SymPy I know of
        # try to convert to tuple
        try:
            itr_ret = Tuple(*(sympify(i) if isinstance(i, str) else i for i in itr))
        except (TypeError, ValueError):
            itr_ret = itr

    else:
        raise TypeError("%s is not an instance of list/tuple/set/Range/Tuple/FiniteSet." % (itr))
    return itr_ret

def _sym_sympify(arg):
    """
    Converts an arbitrary expression to a type that can be used inside SymPy.
    As generally strings are unwise to use in the expressions,
    it returns the Symbol of argument if the string type argument is passed.

    Parameters
    =========

    arg: The parameter to be converted to be used in SymPy.

    Returns
    =======

    The converted parameter.

    """
    if isinstance(arg, str):
        return Symbol(arg)
    else:
        return _sympify(arg)

def _matrix_checks(matrix):
    if not isinstance(matrix, (Matrix, MatrixSymbol, ImmutableMatrix)):
        raise TypeError("Transition probabilities either should "
                            "be a Matrix or a MatrixSymbol.")
    if matrix.shape[0] != matrix.shape[1]:
        raise NonSquareMatrixError("%s is not a square matrix"%(matrix))
    if isinstance(matrix, Matrix):
        matrix = ImmutableMatrix(matrix.tolist())
    return matrix

class StochasticProcess(Basic):
    """
    Base class for all the stochastic processes whether
    discrete or continuous.

    Parameters
    ==========

    sym: Symbol or str
    state_space: Set
        The state space of the stochastic process, by default S.Reals.
        For discrete sets it is zero indexed.

    See Also
    ========

    DiscreteTimeStochasticProcess
    """

    index_set = S.Reals

    def __new__(cls, sym, state_space=S.Reals, **kwargs):
        sym = _symbol_converter(sym)
        state_space = _set_converter(state_space)
        return Basic.__new__(cls, sym, state_space)

    @property
    def symbol(self):
        return self.args[0]

    @property
    def state_space(self) -> FiniteSet | Range:
        if not isinstance(self.args[1], (FiniteSet, Range)):
            assert isinstance(self.args[1], Tuple)
            return FiniteSet(*self.args[1])
        return self.args[1]

    def _deprecation_warn_distribution(self):
        sympy_deprecation_warning(
            """
            Calling the distribution method with a RandomIndexedSymbol
            argument, like X.distribution(X(t)) is deprecated. Instead, call
            distribution() with the given timestamp, like

            X.distribution(t)
            """,
            deprecated_since_version="1.7.1",
            active_deprecations_target="deprecated-distribution-randomindexedsymbol",
            stacklevel=4,
        )

    def distribution(self, key=None):
        if key is None:
            self._deprecation_warn_distribution()
        return Distribution()

    def density(self, x):
        return Density()

    def __call__(self, time):
        """
        Overridden in ContinuousTimeStochasticProcess.
        """
        raise NotImplementedError("Use [] for indexing discrete time stochastic process.")

    def __getitem__(self, time):
        """
        Overridden in DiscreteTimeStochasticProcess.
        """
        raise NotImplementedError("Use () for indexing continuous time stochastic process.")

    def probability(self, condition):
        raise NotImplementedError()

    def joint_distribution(self, *args):
        """
        Computes the joint distribution of the random indexed variables.

        Parameters
        ==========

        args: iterable
            The finite list of random indexed variables/the key of a stochastic
            process whose joint distribution has to be computed.

        Returns
        =======

        JointDistribution
            The joint distribution of the list of random indexed variables.
            An unevaluated object is returned if it is not possible to
            compute the joint distribution.

        Raises
        ======

        ValueError: When the arguments passed are not of type RandomIndexSymbol
        or Number.
        """
        args = list(args)
        for i, arg in enumerate(args):
            if S(arg).is_Number:
                if self.index_set.is_subset(S.Integers):
                    args[i] = self.__getitem__(arg)
                else:
                    args[i] = self.__call__(arg)
            elif not isinstance(arg, RandomIndexedSymbol):
                raise ValueError("Expected a RandomIndexedSymbol or "
                                "key not  %s"%(type(arg)))

        if args[0].pspace.distribution == Distribution():
            return JointDistribution(*args)
        density = Lambda(tuple(args),
                         expr=Mul.fromiter(arg.pspace.process.density(arg) for arg in args))
        return JointDistributionHandmade(density)

    def expectation(self, condition, given_condition):
        raise NotImplementedError("Abstract method for expectation queries.")

    def sample(self):
        raise NotImplementedError("Abstract method for sampling queries.")

class DiscreteTimeStochasticProcess(StochasticProcess):
    """
    Base class for all discrete stochastic processes.
    """
    def __getitem__(self, time):
        """
        For indexing discrete time stochastic processes.

        Returns
        =======

        RandomIndexedSymbol
        """
        time = sympify(time)
        if not time.is_symbol and time not in self.index_set:
            raise IndexError("%s is not in the index set of %s"%(time, self.symbol))
        idx_obj = Indexed(self.symbol, time)
        pspace_obj = StochasticPSpace(self.symbol, self, self.distribution(time))
        return RandomIndexedSymbol(idx_obj, pspace_obj)

class ContinuousTimeStochasticProcess(StochasticProcess):
    """
    Base class for all continuous time stochastic process.
    """
    def __call__(self, time):
        """
        For indexing continuous time stochastic processes.

        Returns
        =======

        RandomIndexedSymbol
        """
        time = sympify(time)
        if not time.is_symbol and time not in self.index_set:
            raise IndexError("%s is not in the index set of %s"%(time, self.symbol))
        func_obj = Function(self.symbol)(time)
        pspace_obj = StochasticPSpace(self.symbol, self, self.distribution(time))
        return RandomIndexedSymbol(func_obj, pspace_obj)

class TransitionMatrixOf(Boolean):
    """
    Assumes that the matrix is the transition matrix
    of the process.
    """

    def __new__(cls, process, matrix):
        if not isinstance(process, DiscreteMarkovChain):
            raise ValueError("Currently only DiscreteMarkovChain "
                                "support TransitionMatrixOf.")
        matrix = _matrix_checks(matrix)
        return Basic.__new__(cls, process, matrix)

    process = property(lambda self: self.args[0])
    matrix = property(lambda self: self.args[1])

class GeneratorMatrixOf(TransitionMatrixOf):
    """
    Assumes that the matrix is the generator matrix
    of the process.
    """

    def __new__(cls, process, matrix):
        if not isinstance(process, ContinuousMarkovChain):
            raise ValueError("Currently only ContinuousMarkovChain "
                                "support GeneratorMatrixOf.")
        matrix = _matrix_checks(matrix)
        return Basic.__new__(cls, process, matrix)

class StochasticStateSpaceOf(Boolean):

    def __new__(cls, process, state_space):
        if not isinstance(process, (DiscreteMarkovChain, ContinuousMarkovChain)):
            raise ValueError("Currently only DiscreteMarkovChain and ContinuousMarkovChain "
                                "support StochasticStateSpaceOf.")
        state_space = _state_converter(state_space)
        if isinstance(state_space, Range):
            ss_size = ceiling((state_space.stop - state_space.start) / state_space.step)
        else:
            ss_size = len(state_space)
        state_index = Range(ss_size)
        return Basic.__new__(cls, process, state_index)

    process = property(lambda self: self.args[0])
    state_index = property(lambda self: self.args[1])

class MarkovProcess(StochasticProcess):
    """
    Contains methods that handle queries
    common to Markov processes.
    """

    @property
    def number_of_states(self) -> Integer | Symbol:
        """
        The number of states in the Markov Chain.
        """
        return _sympify(self.args[2].shape[0]) # type: ignore

    @property
    def _state_index(self):
        """
        Returns state index as Range.
        """
        return self.args[1]

    @classmethod
    def _sanity_checks(cls, state_space, trans_probs):
        # Try to never have None as state_space or trans_probs.
        # This helps a lot if we get it done at the start.
        if (state_space is None) and (trans_probs is None):
            _n = Dummy('n', integer=True, nonnegative=True)
            state_space = _state_converter(Range(_n))
            trans_probs = _matrix_checks(MatrixSymbol('_T', _n, _n))

        elif state_space is None:
            trans_probs = _matrix_checks(trans_probs)
            state_space = _state_converter(Range(trans_probs.shape[0]))

        elif trans_probs is None:
            state_space = _state_converter(state_space)
            if isinstance(state_space, Range):
                _n = ceiling((state_space.stop - state_space.start) / state_space.step)
            else:
                _n = len(state_space)
            trans_probs = MatrixSymbol('_T', _n, _n)

        else:
            state_space = _state_converter(state_space)
            trans_probs = _matrix_checks(trans_probs)
            # Range object doesn't want to give a symbolic size
            # so we do it ourselves.
            if isinstance(state_space, Range):
                ss_size = ceiling((state_space.stop - state_space.start) / state_space.step)
            else:
                ss_size = len(state_space)
            if ss_size != trans_probs.shape[0]:
                raise ValueError('The size of the state space and the number of '
                                 'rows of the transition matrix must be the same.')

        return state_space, trans_probs

    def _extract_information(self, given_condition):
        """
        Helper function to extract information, like,
        transition matrix/generator matrix, state space, etc.
        """
        if isinstance(self, DiscreteMarkovChain):
            trans_probs = self.transition_probabilities
            state_index = self._state_index
        elif isinstance(self, ContinuousMarkovChain):
            trans_probs = self.generator_matrix
            state_index = self._state_index
        if isinstance(given_condition, And):
            gcs = given_condition.args
            given_condition = S.true
            for gc in gcs:
                if isinstance(gc, TransitionMatrixOf):
                    trans_probs = gc.matrix
                if isinstance(gc, StochasticStateSpaceOf):
                    state_index = gc.state_index
                if isinstance(gc, Relational):
                    given_condition = given_condition & gc
        if isinstance(given_condition, TransitionMatrixOf):
            trans_probs = given_condition.matrix
            given_condition = S.true
        if isinstance(given_condition, StochasticStateSpaceOf):
            state_index = given_condition.state_index
            given_condition = S.true
        return trans_probs, state_index, given_condition

    def _check_trans_probs(self, trans_probs, row_sum=1):
        """
        Helper function for checking the validity of transition
        probabilities.
        """
        if not isinstance(trans_probs, MatrixSymbol):
            rows = trans_probs.tolist()
            for row in rows:
                if (sum(row) - row_sum) != 0:
                    raise ValueError("Values in a row must sum to %s. "
                    "If you are using Float or floats then please use Rational."%(row_sum))

    def _work_out_state_index(self, state_index, given_condition, trans_probs):
        """
        Helper function to extract state space if there
        is a random symbol in the given condition.
        """
        # if given condition is None, then there is no need to work out
        # state_space from random variables
        if given_condition != None:
            rand_var = list(given_condition.atoms(RandomSymbol) -
                        given_condition.atoms(RandomIndexedSymbol))
            if len(rand_var) == 1:
                state_index = rand_var[0].pspace.set

        # `not None` is `True`. So the old test fails for symbolic sizes.
        # Need to build the statement differently.
        sym_cond = not self.number_of_states.is_Integer
        cond1 = not sym_cond and len(state_index) != trans_probs.shape[0]
        if cond1:
            raise ValueError("state space is not compatible with the transition probabilities.")
        if not isinstance(trans_probs.shape[0], Symbol):
            state_index = FiniteSet(*range(trans_probs.shape[0]))
        return state_index

    @cacheit
    def _preprocess(self, given_condition, evaluate):
        """
        Helper function for pre-processing the information.
        """
        is_insufficient = False

        if not evaluate: # avoid pre-processing if the result is not to be evaluated
            return (True, None, None, None)

        # extracting transition matrix and state space
        trans_probs, state_index, given_condition = self._extract_information(given_condition)

        # given_condition does not have sufficient information
        # for computations
        if trans_probs is None or \
            given_condition is None:
            is_insufficient = True
        else:
            # checking transition probabilities
            if isinstance(self, DiscreteMarkovChain):
                self._check_trans_probs(trans_probs, row_sum=1)
            elif isinstance(self, ContinuousMarkovChain):
                self._check_trans_probs(trans_probs, row_sum=0)

            # working out state space
            state_index = self._work_out_state_index(state_index, given_condition, trans_probs)

        return is_insufficient, trans_probs, state_index, given_condition

    def replace_with_index(self, condition):
        if isinstance(condition, Relational):
            lhs, rhs = condition.lhs, condition.rhs
            if not isinstance(lhs, RandomIndexedSymbol):
                lhs, rhs = rhs, lhs
            condition = type(condition)(self.index_of.get(lhs, lhs),
                                        self.index_of.get(rhs, rhs))
        return condition

    def probability(self, condition, given_condition=None, evaluate=True, **kwargs):
        """
        Handles probability queries for Markov process.

        Parameters
        ==========

        condition: Relational
        given_condition: Relational/And

        Returns
        =======
        Probability
            If the information is not sufficient.
        Expr
            In all other cases.

        Note
        ====
        Any information passed at the time of query overrides
        any information passed at the time of object creation like
        transition probabilities, state space.
        Pass the transition matrix using TransitionMatrixOf,
        generator matrix using GeneratorMatrixOf and state space
        using StochasticStateSpaceOf in given_condition using & or And.
        """
        check, mat, state_index, new_given_condition = \
            self._preprocess(given_condition, evaluate)

        rv = list(condition.atoms(RandomIndexedSymbol))
        symbolic = False
        for sym in rv:
            if sym.key.is_symbol:
                symbolic = True
                break

        if check:
            return Probability(condition, new_given_condition)

        if isinstance(self, ContinuousMarkovChain):
            trans_probs = self.transition_probabilities(mat)
        elif isinstance(self, DiscreteMarkovChain):
            trans_probs = mat
        condition = self.replace_with_index(condition)
        given_condition = self.replace_with_index(given_condition)
        new_given_condition = self.replace_with_index(new_given_condition)

        if isinstance(condition, Relational):
            if isinstance(new_given_condition, And):
                gcs = new_given_condition.args
            else:
                gcs = (new_given_condition, )
            min_key_rv = list(new_given_condition.atoms(RandomIndexedSymbol))

            if len(min_key_rv):
                min_key_rv = min_key_rv[0]
                for r in rv:
                    if min_key_rv.key.is_symbol or r.key.is_symbol:
                        continue
                    if min_key_rv.key > r.key:
                        return Probability(condition)
            else:
                min_key_rv = None
                return Probability(condition)

            if symbolic:
                return self._symbolic_probability(condition, new_given_condition, rv, min_key_rv)

            if len(rv) > 1:
                rv[0] = condition.lhs
                rv[1] = condition.rhs
                if rv[0].key < rv[1].key:
                        rv[0], rv[1] = rv[1], rv[0]
                        if isinstance(condition, Gt):
                            condition = Lt(condition.lhs, condition.rhs)
                        elif isinstance(condition, Lt):
                            condition = Gt(condition.lhs, condition.rhs)
                        elif isinstance(condition, Ge):
                            condition = Le(condition.lhs, condition.rhs)
                        elif isinstance(condition, Le):
                            condition = Ge(condition.lhs, condition.rhs)
                s = Rational(0, 1)
                n = len(self.state_space)

                if isinstance(condition, (Eq, Ne)):
                    for i in range(0, n):
                        s += self.probability(Eq(rv[0], i), Eq(rv[1], i)) * self.probability(Eq(rv[1], i), new_given_condition)
                    return s if isinstance(condition, Eq) else 1 - s
                else:
                    upper = 0
                    greater = False
                    if isinstance(condition, (Ge, Lt)):
                        upper = 1
                    if isinstance(condition, (Ge, Gt)):
                        greater = True

                    for i in range(0, n):
                        if i <= n//2:
                            for j in range(0, i + upper):
                                s += self.probability(Eq(rv[0], i), Eq(rv[1], j)) * self.probability(Eq(rv[1], j), new_given_condition)
                        else:
                            s += self.probability(Eq(rv[0], i), new_given_condition)
                            for j in range(i + upper, n):
                                s -= self.probability(Eq(rv[0], i), Eq(rv[1], j)) * self.probability(Eq(rv[1], j), new_given_condition)
                    return s if greater else 1 - s

            rv = rv[0]
            states = condition.as_set()
            prob, gstate = {}, None
            for gc in gcs:
                if gc.has(min_key_rv):
                    if gc.has(Probability):
                        p, gp = (gc.rhs, gc.lhs) if isinstance(gc.lhs, Probability) \
                                    else (gc.lhs, gc.rhs)
                        gr = gp.args[0]
                        gset = Intersection(gr.as_set(), state_index)
                        gstate = list(gset)[0]
                        prob[gset] = p
                    else:
                        _, gstate = (gc.lhs.key, gc.rhs) if isinstance(gc.lhs, RandomIndexedSymbol) \
                                    else (gc.rhs.key, gc.lhs)

            if not all(k in self.index_set for k in (rv.key, min_key_rv.key)):
                raise IndexError("The timestamps of the process are not in it's index set.")
            states = Intersection(states, state_index) if not isinstance(self.number_of_states, Symbol) else states
            for state in Union(states, FiniteSet(gstate)):
                if not state.is_Integer or Ge(state, mat.shape[0]) is True:
                    raise IndexError("No information is available for (%s, %s) in "
                        "transition probabilities of shape, (%s, %s). "
                        "State space is zero indexed."
                        %(gstate, state, mat.shape[0], mat.shape[1]))
            if prob:
                gstates = Union(*prob.keys())
                if len(gstates) == 1:
                    gstate = list(gstates)[0]
                    gprob = list(prob.values())[0]
                    prob[gstates] = gprob
                elif len(gstates) == len(state_index) - 1:
                    gstate = list(state_index - gstates)[0]
                    gprob = S.One - sum(prob.values())
                    prob[state_index - gstates] = gprob
                else:
                    raise ValueError("Conflicting information.")
            else:
                gprob = S.One

            if min_key_rv == rv:
                return sum(prob[FiniteSet(state)] for state in states)
            if isinstance(self, ContinuousMarkovChain):
                return gprob * sum(trans_probs(rv.key - min_key_rv.key).__getitem__((gstate, state))
                                    for state in states)
            if isinstance(self, DiscreteMarkovChain):
                return gprob * sum((trans_probs**(rv.key - min_key_rv.key)).__getitem__((gstate, state))
                                    for state in states)

        if isinstance(condition, Not):
            expr = condition.args[0]
            return S.One - self.probability(expr, given_condition, evaluate, **kwargs)

        if isinstance(condition, And):
            compute_later, state2cond, conds = [], {}, condition.args
            for expr in conds:
                if isinstance(expr, Relational):
                    ris = list(expr.atoms(RandomIndexedSymbol))[0]
                    if state2cond.get(ris, None) is None:
                        state2cond[ris] = S.true
                    state2cond[ris] &= expr
                else:
                    compute_later.append(expr)
            ris = []
            for ri in state2cond:
                ris.append(ri)
                cset = Intersection(state2cond[ri].as_set(), state_index)
                if len(cset) == 0:
                    return S.Zero
                state2cond[ri] = cset.as_relational(ri)
            sorted_ris = sorted(ris, key=lambda ri: ri.key)
            prod = self.probability(state2cond[sorted_ris[0]], given_condition, evaluate, **kwargs)
            for i in range(1, len(sorted_ris)):
                ri, prev_ri = sorted_ris[i], sorted_ris[i-1]
                if not isinstance(state2cond[ri], Eq):
                    raise ValueError("The process is in multiple states at %s, unable to determine the probability."%(ri))
                mat_of = TransitionMatrixOf(self, mat) if isinstance(self, DiscreteMarkovChain) else GeneratorMatrixOf(self, mat)
                prod *= self.probability(state2cond[ri], state2cond[prev_ri]
                                 & mat_of
                                 & StochasticStateSpaceOf(self, state_index),
                                 evaluate, **kwargs)
            for expr in compute_later:
                prod *= self.probability(expr, given_condition, evaluate, **kwargs)
            return prod

        if isinstance(condition, Or):
            return sum(self.probability(expr, given_condition, evaluate, **kwargs)
                        for expr in condition.args)

        raise NotImplementedError("Mechanism for handling (%s, %s) queries hasn't been "
                                "implemented yet."%(condition, given_condition))

    def _symbolic_probability(self, condition, new_given_condition, rv, min_key_rv):
        #Function to calculate probability for queries with symbols
        if isinstance(condition, Relational):
            curr_state = new_given_condition.rhs if isinstance(new_given_condition.lhs, RandomIndexedSymbol) \
                    else new_given_condition.lhs
            next_state = condition.rhs if isinstance(condition.lhs, RandomIndexedSymbol) \
                else condition.lhs

            if isinstance(condition, (Eq, Ne)):
                if isinstance(self, DiscreteMarkovChain):
                    P = self.transition_probabilities**(rv[0].key - min_key_rv.key)
                else:
                    P = exp(self.generator_matrix*(rv[0].key - min_key_rv.key))
                prob = P[curr_state, next_state] if isinstance(condition, Eq) else 1 - P[curr_state, next_state]
                return Piecewise((prob, rv[0].key > min_key_rv.key), (Probability(condition), True))
            else:
                upper = 1
                greater = False
                if isinstance(condition, (Ge, Lt)):
                    upper = 0
                if isinstance(condition, (Ge, Gt)):
                    greater = True
                k = Dummy('k')
                condition = Eq(condition.lhs, k) if isinstance(condition.lhs, RandomIndexedSymbol)\
                    else Eq(condition.rhs, k)
                total = Sum(self.probability(condition, new_given_condition), (k, next_state + upper, self.state_space._sup))
                return Piecewise((total, rv[0].key > min_key_rv.key), (Probability(condition), True)) if greater\
                    else Piecewise((1 - total, rv[0].key > min_key_rv.key), (Probability(condition), True))
        else:
            return Probability(condition, new_given_condition)

    def expectation(self, expr, condition=None, evaluate=True, **kwargs):
        """
        Handles expectation queries for markov process.

        Parameters
        ==========

        expr: RandomIndexedSymbol, Relational, Logic
            Condition for which expectation has to be computed. Must
            contain a RandomIndexedSymbol of the process.
        condition: Relational, Logic
            The given conditions under which computations should be done.

        Returns
        =======

        Expectation
            Unevaluated object if computations cannot be done due to
            insufficient information.
        Expr
            In all other cases when the computations are successful.

        Note
        ====

        Any information passed at the time of query overrides
        any information passed at the time of object creation like
        transition probabilities, state space.

        Pass the transition matrix using TransitionMatrixOf,
        generator matrix using GeneratorMatrixOf and state space
        using StochasticStateSpaceOf in given_condition using & or And.
        """

        check, mat, state_index, condition = \
            self._preprocess(condition, evaluate)

        if check:
            return Expectation(expr, condition)

        rvs = random_symbols(expr)
        if isinstance(expr, Expr) and isinstance(condition, Eq) \
            and len(rvs) == 1:
            # handle queries similar to E(f(X[i]), Eq(X[i-m], <some-state>))
            condition=self.replace_with_index(condition)
            state_index=self.replace_with_index(state_index)
            rv = list(rvs)[0]
            lhsg, rhsg = condition.lhs, condition.rhs
            if not isinstance(lhsg, RandomIndexedSymbol):
                lhsg, rhsg = (rhsg, lhsg)
            if rhsg not in state_index:
                raise ValueError("%s state is not in the state space."%(rhsg))
            if rv.key < lhsg.key:
                raise ValueError("Incorrect given condition is given, expectation "
                    "time %s < time %s"%(rv.key, rv.key))
            mat_of = TransitionMatrixOf(self, mat) if isinstance(self, DiscreteMarkovChain) else GeneratorMatrixOf(self, mat)
            cond = condition & mat_of & \
                    StochasticStateSpaceOf(self, state_index)
            func = lambda s: self.probability(Eq(rv, s), cond) * expr.subs(rv, self._state_index[s])
            return sum(func(s) for s in state_index)

        raise NotImplementedError("Mechanism for handling (%s, %s) queries hasn't been "
                                "implemented yet."%(expr, condition))

class DiscreteMarkovChain(DiscreteTimeStochasticProcess, MarkovProcess):
    """
    Represents a finite discrete time-homogeneous Markov chain.

    This type of Markov Chain can be uniquely characterised by
    its (ordered) state space and its one-step transition probability
    matrix.

    Parameters
    ==========

    sym:
        The name given to the Markov Chain
    state_space:
        Optional, by default, Range(n)
    trans_probs:
        Optional, by default, MatrixSymbol('_T', n, n)

    Examples
    ========

    >>> from sympy.stats import DiscreteMarkovChain, TransitionMatrixOf, P, E
    >>> from sympy import Matrix, MatrixSymbol, Eq, symbols
    >>> T = Matrix([[0.5, 0.2, 0.3],[0.2, 0.5, 0.3],[0.2, 0.3, 0.5]])
    >>> Y = DiscreteMarkovChain("Y", [0, 1, 2], T)
    >>> YS = DiscreteMarkovChain("Y")

    >>> Y.state_space
    {0, 1, 2}
    >>> Y.transition_probabilities
    Matrix([
    [0.5, 0.2, 0.3],
    [0.2, 0.5, 0.3],
    [0.2, 0.3, 0.5]])
    >>> TS = MatrixSymbol('T', 3, 3)
    >>> P(Eq(YS[3], 2), Eq(YS[1], 1) & TransitionMatrixOf(YS, TS))
    T[0, 2]*T[1, 0] + T[1, 1]*T[1, 2] + T[1, 2]*T[2, 2]
    >>> P(Eq(Y[3], 2), Eq(Y[1], 1)).round(2)
    0.36

    Probabilities will be calculated based on indexes rather
    than state names. For example, with the Sunny-Cloudy-Rainy
    model with string state names:

    >>> from sympy.core.symbol import Str
    >>> Y = DiscreteMarkovChain("Y", [Str('Sunny'), Str('Cloudy'), Str('Rainy')], T)
    >>> P(Eq(Y[3], 2), Eq(Y[1], 1)).round(2)
    0.36

    This gives the same answer as the ``[0, 1, 2]`` state space.
    Currently, there is no support for state names within probability
    and expectation statements. Here is a work-around using ``Str``:

    >>> P(Eq(Str('Rainy'), Y[3]), Eq(Y[1], Str('Cloudy'))).round(2)
    0.36

    Symbol state names can also be used:

    >>> sunny, cloudy, rainy = symbols('Sunny, Cloudy, Rainy')
    >>> Y = DiscreteMarkovChain("Y", [sunny, cloudy, rainy], T)
    >>> P(Eq(Y[3], rainy), Eq(Y[1], cloudy)).round(2)
    0.36

    Expectations will be calculated as follows:

    >>> E(Y[3], Eq(Y[1], cloudy))
    0.38*Cloudy + 0.36*Rainy + 0.26*Sunny

    Probability of expressions with multiple RandomIndexedSymbols
    can also be calculated provided there is only 1 RandomIndexedSymbol
    in the given condition. It is always better to use Rational instead
    of floating point numbers for the probabilities in the
    transition matrix to avoid errors.

    >>> from sympy import Gt, Le, Rational
    >>> T = Matrix([[Rational(5, 10), Rational(3, 10), Rational(2, 10)], [Rational(2, 10), Rational(7, 10), Rational(1, 10)], [Rational(3, 10), Rational(3, 10), Rational(4, 10)]])
    >>> Y = DiscreteMarkovChain("Y", [0, 1, 2], T)
    >>> P(Eq(Y[3], Y[1]), Eq(Y[0], 0)).round(3)
    0.409
    >>> P(Gt(Y[3], Y[1]), Eq(Y[0], 0)).round(2)
    0.36
    >>> P(Le(Y[15], Y[10]), Eq(Y[8], 2)).round(7)
    0.6963328

    Symbolic probability queries are also supported

    >>> a, b, c, d = symbols('a b c d')
    >>> T = Matrix([[Rational(1, 10), Rational(4, 10), Rational(5, 10)], [Rational(3, 10), Rational(4, 10), Rational(3, 10)], [Rational(7, 10), Rational(2, 10), Rational(1, 10)]])
    >>> Y = DiscreteMarkovChain("Y", [0, 1, 2], T)
    >>> query = P(Eq(Y[a], b), Eq(Y[c], d))
    >>> query.subs({a:10, b:2, c:5, d:1}).round(4)
    0.3096
    >>> P(Eq(Y[10], 2), Eq(Y[5], 1)).evalf().round(4)
    0.3096
    >>> query_gt = P(Gt(Y[a], b), Eq(Y[c], d))
    >>> query_gt.subs({a:21, b:0, c:5, d:0}).evalf().round(5)
    0.64705
    >>> P(Gt(Y[21], 0), Eq(Y[5], 0)).round(5)
    0.64705

    There is limited support for arbitrarily sized states:

    >>> n = symbols('n', nonnegative=True, integer=True)
    >>> T = MatrixSymbol('T', n, n)
    >>> Y = DiscreteMarkovChain("Y", trans_probs=T)
    >>> Y.state_space
    Range(0, n, 1)
    >>> query = P(Eq(Y[a], b), Eq(Y[c], d))
    >>> query.subs({a:10, b:2, c:5, d:1})
    (T**5)[1, 2]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Markov_chain#Discrete-time_Markov_chain
    .. [2] https://web.archive.org/web/20201230182007/https://www.dartmouth.edu/~chance/teaching_aids/books_articles/probability_book/Chapter11.pdf
    """
    index_set = S.Naturals0

    def __new__(cls, sym, state_space=None, trans_probs=None):
        sym = _symbol_converter(sym)

        state_space, trans_probs = MarkovProcess._sanity_checks(state_space, trans_probs)

        obj = Basic.__new__(cls, sym, state_space, trans_probs) # type: ignore
        indices = {}
        if isinstance(obj.number_of_states, Integer):
            for index, state in enumerate(obj._state_index):
                indices[state] = index
        obj.index_of = indices
        return obj

    @property
    def transition_probabilities(self):
        """
        Transition probabilities of discrete Markov chain,
        either an instance of Matrix or MatrixSymbol.
        """
        return self.args[2]

    def communication_classes(self) -> list[tuple[list[Basic], Boolean, Integer]]:
        """
        Returns the list of communication classes that partition
        the states of the markov chain.

        A communication class is defined to be a set of states
        such that every state in that set is reachable from
        every other state in that set. Due to its properties
        this forms a class in the mathematical sense.
        Communication classes are also known as recurrence
        classes.

        Returns
        =======

        classes
            The ``classes`` are a list of tuples. Each
            tuple represents a single communication class
            with its properties. The first element in the
            tuple is the list of states in the class, the
            second element is whether the class is recurrent
            and the third element is the period of the
            communication class.

        Examples
        ========

        >>> from sympy.stats import DiscreteMarkovChain
        >>> from sympy import Matrix
        >>> T = Matrix([[0, 1, 0],
        ...             [1, 0, 0],
        ...             [1, 0, 0]])
        >>> X = DiscreteMarkovChain('X', [1, 2, 3], T)
        >>> classes = X.communication_classes()
        >>> for states, is_recurrent, period in classes:
        ...     states, is_recurrent, period
        ([1, 2], True, 2)
        ([3], False, 1)

        From this we can see that states ``1`` and ``2``
        communicate, are recurrent and have a period
        of 2. We can also see state ``3`` is transient
        with a period of 1.

        Notes
        =====

        The algorithm used is of order ``O(n**2)`` where
        ``n`` is the number of states in the markov chain.
        It uses Tarjan's algorithm to find the classes
        themselves and then it uses a breadth-first search
        algorithm to find each class's periodicity.
        Most of the algorithm's components approach ``O(n)``
        as the matrix becomes more and more sparse.

        References
        ==========

        .. [1] https://web.archive.org/web/20220207032113/https://www.columbia.edu/~ww2040/4701Sum07/4701-06-Notes-MCII.pdf
        .. [2] https://cecas.clemson.edu/~shierd/Shier/markov.pdf
        .. [3] https://www.proquest.com/openview/4adc6a51d8371be5b0e4c7dff287fc70/1?pq-origsite=gscholar&cbl=2026366&diss=y
        .. [4] https://www.mathworks.com/help/econ/dtmc.classify.html
        """
        n = self.number_of_states
        T = self.transition_probabilities

        if isinstance(T, MatrixSymbol):
            raise NotImplementedError("Cannot perform the operation with a symbolic matrix.")

        # begin Tarjan's algorithm
        V = Range(n)
        # don't use state names. Rather use state
        # indexes since we use them for matrix
        # indexing here and later onward
        E = [(i, j) for i in V for j in V if T[i, j] != 0]
        classes = strongly_connected_components((V, E))
        # end Tarjan's algorithm

        recurrence = []
        periods = []
        for class_ in classes:
            # begin recurrent check (similar to self._check_trans_probs())
            submatrix = T[class_, class_]  # get the submatrix with those states
            is_recurrent = S.true
            rows = submatrix.tolist()
            for row in rows:
                if (sum(row) - 1) != 0:
                    is_recurrent = S.false
                    break
            recurrence.append(is_recurrent)
            # end recurrent check

            # begin breadth-first search
            non_tree_edge_values: set[int] = set()
            visited = {class_[0]}
            newly_visited = {class_[0]}
            level = {class_[0]: 0}
            current_level = 0
            done = False  # imitate a do-while loop
            while not done:  # runs at most len(class_) times
                done = len(visited) == len(class_)
                current_level += 1

                # this loop and the while loop above run a combined len(class_) number of times.
                # so this triple nested loop runs through each of the n states once.
                for i in newly_visited:

                    # the loop below runs len(class_) number of times
                    # complexity is around about O(n * avg(len(class_)))
                    newly_visited = {j for j in class_ if T[i, j] != 0}

                    new_tree_edges = newly_visited.difference(visited)
                    for j in new_tree_edges:
                        level[j] = current_level

                    new_non_tree_edges = newly_visited.intersection(visited)
                    new_non_tree_edge_values = {level[i]-level[j]+1 for j in new_non_tree_edges}

                    non_tree_edge_values = non_tree_edge_values.union(new_non_tree_edge_values)
                    visited = visited.union(new_tree_edges)

            # igcd needs at least 2 arguments
            positive_ntev = {val_e for val_e in non_tree_edge_values if val_e > 0}
            if len(positive_ntev) == 0:
                periods.append(len(class_))
            elif len(positive_ntev) == 1:
                periods.append(positive_ntev.pop())
            else:
                periods.append(igcd(*positive_ntev))
            # end breadth-first search

        # convert back to the user's state names
        classes = [[_sympify(self._state_index[i]) for i in class_] for class_ in classes]
        return list(zip(classes, recurrence, map(Integer,periods)))

    def fundamental_matrix(self):
        """
        Each entry fundamental matrix can be interpreted as
        the expected number of times the chains is in state j
        if it started in state i.

        References
        ==========

        .. [1] https://lips.cs.princeton.edu/the-fundamental-matrix-of-a-finite-markov-chain/

        """
        _, _, _, Q = self.decompose()

        if Q.shape[0] > 0:  # if non-ergodic
            I = eye(Q.shape[0])
            if (I - Q).det() == 0:
                raise ValueError("The fundamental matrix doesn't exist.")
            return (I - Q).inv().as_immutable()
        else:  # if ergodic
            P = self.transition_probabilities
            I = eye(P.shape[0])
            w = self.fixed_row_vector()
            W = Matrix([list(w) for i in range(0, P.shape[0])])
            if (I - P + W).det() == 0:
                raise ValueError("The fundamental matrix doesn't exist.")
            return (I - P + W).inv().as_immutable()

    def absorbing_probabilities(self):
        """
        Computes the absorbing probabilities, i.e.
        the ij-th entry of the matrix denotes the
        probability of Markov chain being absorbed
        in state j starting from state i.
        """
        _, _, R, _ = self.decompose()
        N = self.fundamental_matrix()
        if R is None or N is None:
            return None
        return N*R

    def absorbing_probabilites(self):
        sympy_deprecation_warning(
            """
            DiscreteMarkovChain.absorbing_probabilites() is deprecated. Use
            absorbing_probabilities() instead (note the spelling difference).
            """,
            deprecated_since_version="1.7",
            active_deprecations_target="deprecated-absorbing_probabilites",
        )
        return self.absorbing_probabilities()

    def is_regular(self):
        tuples = self.communication_classes()
        if len(tuples) == 0:
            return S.false  # not defined for a 0x0 matrix
        classes, _, periods = list(zip(*tuples))
        return And(len(classes) == 1, periods[0] == 1)

    def is_ergodic(self):
        tuples = self.communication_classes()
        if len(tuples) == 0:
            return S.false  # not defined for a 0x0 matrix
        classes, _, _ = list(zip(*tuples))
        return S(len(classes) == 1)

    def is_absorbing_state(self, state):
        trans_probs = self.transition_probabilities
        if isinstance(trans_probs, ImmutableMatrix) and \
            state < trans_probs.shape[0]:
            return S(trans_probs[state, state]) is S.One

    def is_absorbing_chain(self):
        states, A, B, C = self.decompose()
        r = A.shape[0]
        return And(r > 0, A == Identity(r).as_explicit())

    def stationary_distribution(self, condition_set=False) -> ImmutableMatrix | ConditionSet | Lambda:
        r"""
        The stationary distribution is any row vector, p, that solves p = pP,
        is row stochastic and each element in p must be nonnegative.
        That means in matrix form: :math:`(P-I)^T p^T = 0` and
        :math:`(1, \dots, 1) p = 1`
        where ``P`` is the one-step transition matrix.

        All time-homogeneous Markov Chains with a finite state space
        have at least one stationary distribution. In addition, if
        a finite time-homogeneous Markov Chain is irreducible, the
        stationary distribution is unique.

        Parameters
        ==========

        condition_set : bool
            If the chain has a symbolic size or transition matrix,
            it will return a ``Lambda`` if ``False`` and return a
            ``ConditionSet`` if ``True``.

        Examples
        ========

        >>> from sympy.stats import DiscreteMarkovChain
        >>> from sympy import Matrix, S

        An irreducible Markov Chain

        >>> T = Matrix([[S(1)/2, S(1)/2, 0],
        ...             [S(4)/5, S(1)/5, 0],
        ...             [1, 0, 0]])
        >>> X = DiscreteMarkovChain('X', trans_probs=T)
        >>> X.stationary_distribution()
        Matrix([[8/13, 5/13, 0]])

        A reducible Markov Chain

        >>> T = Matrix([[S(1)/2, S(1)/2, 0],
        ...             [S(4)/5, S(1)/5, 0],
        ...             [0, 0, 1]])
        >>> X = DiscreteMarkovChain('X', trans_probs=T)
        >>> X.stationary_distribution()
        Matrix([[8/13 - 8*tau0/13, 5/13 - 5*tau0/13, tau0]])

        >>> Y = DiscreteMarkovChain('Y')
        >>> Y.stationary_distribution()
        Lambda((wm, _T), Eq(wm*_T, wm))

        >>> Y.stationary_distribution(condition_set=True)
        ConditionSet(wm, Eq(wm*_T, wm))

        References
        ==========

        .. [1] https://www.probabilitycourse.com/chapter11/11_2_6_stationary_and_limiting_distributions.php
        .. [2] https://web.archive.org/web/20210508104430/https://galton.uchicago.edu/~yibi/teaching/stat317/2014/Lectures/Lecture4_6up.pdf

        See Also
        ========

        sympy.stats.DiscreteMarkovChain.limiting_distribution
        """
        trans_probs = self.transition_probabilities
        n = self.number_of_states

        if n == 0:
            return ImmutableMatrix(Matrix([[]]))

        # symbolic matrix version
        if isinstance(trans_probs, MatrixSymbol):
            wm = MatrixSymbol('wm', 1, n)
            if condition_set:
                return ConditionSet(wm, Eq(wm * trans_probs, wm))
            else:
                return Lambda((wm, trans_probs), Eq(wm * trans_probs, wm))

        # numeric matrix version
        a = Matrix(trans_probs - Identity(n)).T
        a[0, 0:n] = ones(1, n) # type: ignore
        b = zeros(n, 1)
        b[0, 0] = 1

        soln = list(linsolve((a, b)))[0]
        return ImmutableMatrix([soln])

    def fixed_row_vector(self):
        """
        A wrapper for ``stationary_distribution()``.
        """
        return self.stationary_distribution()

    @property
    def limiting_distribution(self):
        """
        The fixed row vector is the limiting
        distribution of a discrete Markov chain.
        """
        return self.fixed_row_vector()

    def decompose(self) -> tuple[list[Basic], ImmutableMatrix, ImmutableMatrix, ImmutableMatrix]:
        """
        Decomposes the transition matrix into submatrices with
        special properties.

        The transition matrix can be decomposed into 4 submatrices:
        - A - the submatrix from recurrent states to recurrent states.
        - B - the submatrix from transient to recurrent states.
        - C - the submatrix from transient to transient states.
        - O - the submatrix of zeros for recurrent to transient states.

        Returns
        =======

        states, A, B, C
            ``states`` - a list of state names with the first being
            the recurrent states and the last being
            the transient states in the order
            of the row names of A and then the row names of C.
            ``A`` - the submatrix from recurrent states to recurrent states.
            ``B`` - the submatrix from transient to recurrent states.
            ``C`` - the submatrix from transient to transient states.

        Examples
        ========

        >>> from sympy.stats import DiscreteMarkovChain
        >>> from sympy import Matrix, S

        One can decompose this chain for example:

        >>> T = Matrix([[S(1)/2, S(1)/2, 0,      0,      0],
        ...             [S(2)/5, S(1)/5, S(2)/5, 0,      0],
        ...             [0,      0,      1,      0,      0],
        ...             [0,      0,      S(1)/2, S(1)/2, 0],
        ...             [S(1)/2, 0,      0,      0, S(1)/2]])
        >>> X = DiscreteMarkovChain('X', trans_probs=T)
        >>> states, A, B, C = X.decompose()
        >>> states
        [2, 0, 1, 3, 4]

        >>> A   # recurrent to recurrent
        Matrix([[1]])

        >>> B  # transient to recurrent
        Matrix([
        [  0],
        [2/5],
        [1/2],
        [  0]])

        >>> C  # transient to transient
        Matrix([
        [1/2, 1/2,   0,   0],
        [2/5, 1/5,   0,   0],
        [  0,   0, 1/2,   0],
        [1/2,   0,   0, 1/2]])

        This means that state 2 is the only absorbing state
        (since A is a 1x1 matrix). B is a 4x1 matrix since
        the 4 remaining transient states all merge into recurrent
        state 2. And C is the 4x4 matrix that shows how the
        transient states 0, 1, 3, 4 all interact.

        See Also
        ========

        sympy.stats.DiscreteMarkovChain.communication_classes
        sympy.stats.DiscreteMarkovChain.canonical_form

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Absorbing_Markov_chain
        .. [2] https://people.brandeis.edu/~igusa/Math56aS08/Math56a_S08_notes015.pdf
        """
        trans_probs = self.transition_probabilities

        classes = self.communication_classes()
        r_states = []
        t_states = []

        for states, recurrent, period in classes:
            if recurrent:
                r_states += states
            else:
                t_states += states

        states = r_states + t_states
        indexes = [self.index_of[state] for state in states] # type: ignore

        A = Matrix(len(r_states), len(r_states),
                   lambda i, j: trans_probs[indexes[i], indexes[j]])

        B = Matrix(len(t_states), len(r_states),
                   lambda i, j: trans_probs[indexes[len(r_states) + i], indexes[j]])

        C = Matrix(len(t_states), len(t_states),
                   lambda i, j: trans_probs[indexes[len(r_states) + i], indexes[len(r_states) + j]])

        return states, A.as_immutable(), B.as_immutable(), C.as_immutable()

    def canonical_form(self) -> tuple[list[Basic], ImmutableMatrix]:
        """
        Reorders the one-step transition matrix
        so that recurrent states appear first and transient
        states appear last. Other representations include inserting
        transient states first and recurrent states last.

        Returns
        =======

        states, P_new
            ``states`` is the list that describes the order of the
            new states in the matrix
            so that the ith element in ``states`` is the state of the
            ith row of A.
            ``P_new`` is the new transition matrix in canonical form.

        Examples
        ========

        >>> from sympy.stats import DiscreteMarkovChain
        >>> from sympy import Matrix, S

        You can convert your chain into canonical form:

        >>> T = Matrix([[S(1)/2, S(1)/2, 0,      0,      0],
        ...             [S(2)/5, S(1)/5, S(2)/5, 0,      0],
        ...             [0,      0,      1,      0,      0],
        ...             [0,      0,      S(1)/2, S(1)/2, 0],
        ...             [S(1)/2, 0,      0,      0, S(1)/2]])
        >>> X = DiscreteMarkovChain('X', list(range(1, 6)), trans_probs=T)
        >>> states, new_matrix = X.canonical_form()
        >>> states
        [3, 1, 2, 4, 5]

        >>> new_matrix
        Matrix([
        [  1,   0,   0,   0,   0],
        [  0, 1/2, 1/2,   0,   0],
        [2/5, 2/5, 1/5,   0,   0],
        [1/2,   0,   0, 1/2,   0],
        [  0, 1/2,   0,   0, 1/2]])

        The new states are [3, 1, 2, 4, 5] and you can
        create a new chain with this and its canonical
        form will remain the same (since it is already
        in canonical form).

        >>> X = DiscreteMarkovChain('X', states, new_matrix)
        >>> states, new_matrix = X.canonical_form()
        >>> states
        [3, 1, 2, 4, 5]

        >>> new_matrix
        Matrix([
        [  1,   0,   0,   0,   0],
        [  0, 1/2, 1/2,   0,   0],
        [2/5, 2/5, 1/5,   0,   0],
        [1/2,   0,   0, 1/2,   0],
        [  0, 1/2,   0,   0, 1/2]])

        This is not limited to absorbing chains:

        >>> T = Matrix([[0, 5,  5, 0,  0],
        ...             [0, 0,  0, 10, 0],
        ...             [5, 0,  5, 0,  0],
        ...             [0, 10, 0, 0,  0],
        ...             [0, 3,  0, 3,  4]])/10
        >>> X = DiscreteMarkovChain('X', trans_probs=T)
        >>> states, new_matrix = X.canonical_form()
        >>> states
        [1, 3, 0, 2, 4]

        >>> new_matrix
        Matrix([
        [   0,    1,   0,   0,   0],
        [   1,    0,   0,   0,   0],
        [ 1/2,    0,   0, 1/2,   0],
        [   0,    0, 1/2, 1/2,   0],
        [3/10, 3/10,   0,   0, 2/5]])

        See Also
        ========

        sympy.stats.DiscreteMarkovChain.communication_classes
        sympy.stats.DiscreteMarkovChain.decompose

        References
        ==========

        .. [1] https://onlinelibrary.wiley.com/doi/pdf/10.1002/9780470316887.app1
        .. [2] http://www.columbia.edu/~ww2040/6711F12/lect1023big.pdf
        """
        states, A, B, C = self.decompose()
        O = zeros(A.shape[0], C.shape[1])
        return states, BlockMatrix([[A, O], [B, C]]).as_explicit()

    def sample(self):
        """
        Returns
        =======

        sample: iterator object
            iterator object containing the sample

        """
        if not isinstance(self.transition_probabilities, (Matrix, ImmutableMatrix)):
            raise ValueError("Transition Matrix must be provided for sampling")
        Tlist = self.transition_probabilities.tolist()
        samps = [random.choice(list(self.state_space))]
        yield samps[0]
        time = 1
        densities = {}
        for state in self.state_space:
            states = list(self.state_space)
            densities[state] = {states[i]: Tlist[state][i]
                        for i in range(len(states))}
        while time < S.Infinity:
            samps.append((next(sample_iter(FiniteRV("_", densities[samps[time - 1]])))))
            yield samps[time]
            time += 1

class ContinuousMarkovChain(ContinuousTimeStochasticProcess, MarkovProcess):
    """
    Represents continuous time Markov chain.

    Parameters
    ==========

    sym : Symbol/str
    state_space : Set
        Optional, by default, S.Reals
    gen_mat : Matrix/ImmutableMatrix/MatrixSymbol
        Optional, by default, None

    Examples
    ========

    >>> from sympy.stats import ContinuousMarkovChain, P
    >>> from sympy import Matrix, S, Eq, Gt
    >>> G = Matrix([[-S(1), S(1)], [S(1), -S(1)]])
    >>> C = ContinuousMarkovChain('C', state_space=[0, 1], gen_mat=G)
    >>> C.limiting_distribution()
    Matrix([[1/2, 1/2]])
    >>> C.state_space
    {0, 1}
    >>> C.generator_matrix
    Matrix([
    [-1,  1],
    [ 1, -1]])

    Probability queries are supported

    >>> P(Eq(C(1.96), 0), Eq(C(0.78), 1)).round(5)
    0.45279
    >>> P(Gt(C(1.7), 0), Eq(C(0.82), 1)).round(5)
    0.58602

    Probability of expressions with multiple RandomIndexedSymbols
    can also be calculated provided there is only 1 RandomIndexedSymbol
    in the given condition. It is always better to use Rational instead
    of floating point numbers for the probabilities in the
    generator matrix to avoid errors.

    >>> from sympy import Gt, Le, Rational
    >>> G = Matrix([[-S(1), Rational(1, 10), Rational(9, 10)], [Rational(2, 5), -S(1), Rational(3, 5)], [Rational(1, 2), Rational(1, 2), -S(1)]])
    >>> C = ContinuousMarkovChain('C', state_space=[0, 1, 2], gen_mat=G)
    >>> P(Eq(C(3.92), C(1.75)), Eq(C(0.46), 0)).round(5)
    0.37933
    >>> P(Gt(C(3.92), C(1.75)), Eq(C(0.46), 0)).round(5)
    0.34211
    >>> P(Le(C(1.57), C(3.14)), Eq(C(1.22), 1)).round(4)
    0.7143

    Symbolic probability queries are also supported

    >>> from sympy import symbols
    >>> a,b,c,d = symbols('a b c d')
    >>> G = Matrix([[-S(1), Rational(1, 10), Rational(9, 10)], [Rational(2, 5), -S(1), Rational(3, 5)], [Rational(1, 2), Rational(1, 2), -S(1)]])
    >>> C = ContinuousMarkovChain('C', state_space=[0, 1, 2], gen_mat=G)
    >>> query = P(Eq(C(a), b), Eq(C(c), d))
    >>> query.subs({a:3.65, b:2, c:1.78, d:1}).evalf().round(10)
    0.4002723175
    >>> P(Eq(C(3.65), 2), Eq(C(1.78), 1)).round(10)
    0.4002723175
    >>> query_gt = P(Gt(C(a), b), Eq(C(c), d))
    >>> query_gt.subs({a:43.2, b:0, c:3.29, d:2}).evalf().round(10)
    0.6832579186
    >>> P(Gt(C(43.2), 0), Eq(C(3.29), 2)).round(10)
    0.6832579186

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Markov_chain#Continuous-time_Markov_chain
    .. [2] https://u.math.biu.ac.il/~amirgi/CTMCnotes.pdf
    """
    index_set = S.Reals

    def __new__(cls, sym, state_space=None, gen_mat=None):
        sym = _symbol_converter(sym)
        state_space, gen_mat = MarkovProcess._sanity_checks(state_space, gen_mat)
        obj = Basic.__new__(cls, sym, state_space, gen_mat)
        indices = {}
        if isinstance(obj.number_of_states, Integer):
            for index, state in enumerate(obj.state_space):
                indices[state] = index
        obj.index_of = indices
        return obj

    @property
    def generator_matrix(self):
        return self.args[2]

    @cacheit
    def transition_probabilities(self, gen_mat=None):
        t = Dummy('t')
        if isinstance(gen_mat, (Matrix, ImmutableMatrix)) and \
                gen_mat.is_diagonalizable():
            # for faster computation use diagonalized generator matrix
            Q, D = gen_mat.diagonalize()
            return Lambda(t, Q*exp(t*D)*Q.inv())
        if gen_mat != None:
            return Lambda(t, exp(t*gen_mat))

    def limiting_distribution(self):
        gen_mat = self.generator_matrix
        if gen_mat is None:
            return None
        if isinstance(gen_mat, MatrixSymbol):
            wm = MatrixSymbol('wm', 1, gen_mat.shape[0])
            return Lambda((wm, gen_mat), Eq(wm*gen_mat, wm))
        w = IndexedBase('w')
        wi = [w[i] for i in range(gen_mat.shape[0])]
        wm = Matrix([wi])
        eqs = (wm*gen_mat).tolist()[0]
        eqs.append(sum(wi) - 1)
        soln = list(linsolve(eqs, wi))[0]
        return ImmutableMatrix([soln])


class BernoulliProcess(DiscreteTimeStochasticProcess):
    """
    The Bernoulli process consists of repeated
    independent Bernoulli process trials with the same parameter `p`.
    It's assumed that the probability `p` applies to every
    trial and that the outcomes of each trial
    are independent of all the rest. Therefore Bernoulli Process
    is Discrete State and Discrete Time Stochastic Process.

    Parameters
    ==========

    sym : Symbol/str
    success : Integer/str
            The event which is considered to be success. Default: 1.
    failure: Integer/str
            The event which is considered to be failure. Default: 0.
    p : Real Number between 0 and 1
            Represents the probability of getting success.

    Examples
    ========

    >>> from sympy.stats import BernoulliProcess, P, E
    >>> from sympy import Eq, Gt
    >>> B = BernoulliProcess("B", p=0.7, success=1, failure=0)
    >>> B.state_space
    {0, 1}
    >>> B.p.round(2)
    0.70
    >>> B.success
    1
    >>> B.failure
    0
    >>> X = B[1] + B[2] + B[3]
    >>> P(Eq(X, 0)).round(2)
    0.03
    >>> P(Eq(X, 2)).round(2)
    0.44
    >>> P(Eq(X, 4)).round(2)
    0
    >>> P(Gt(X, 1)).round(2)
    0.78
    >>> P(Eq(B[1], 0) & Eq(B[2], 1) & Eq(B[3], 0) & Eq(B[4], 1)).round(2)
    0.04
    >>> B.joint_distribution(B[1], B[2])
    JointDistributionHandmade(Lambda((B[1], B[2]), Piecewise((0.7, Eq(B[1], 1)),
    (0.3, Eq(B[1], 0)), (0, True))*Piecewise((0.7, Eq(B[2], 1)), (0.3, Eq(B[2], 0)),
    (0, True))))
    >>> E(2*B[1] + B[2]).round(2)
    2.10
    >>> P(B[1] < 1).round(2)
    0.30

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bernoulli_process
    .. [2] https://mathcs.clarku.edu/~djoyce/ma217/bernoulli.pdf

    """

    index_set = S.Naturals0

    def __new__(cls, sym, p, success=1, failure=0):
        _value_check(p >= 0 and p <= 1, 'Value of p must be between 0 and 1.')
        sym = _symbol_converter(sym)
        p = _sympify(p)
        success = _sym_sympify(success)
        failure = _sym_sympify(failure)
        return Basic.__new__(cls, sym, p, success, failure)

    @property
    def symbol(self):
        return self.args[0]

    @property
    def p(self):
        return self.args[1]

    @property
    def success(self):
        return self.args[2]

    @property
    def failure(self):
        return self.args[3]

    @property
    def state_space(self):
        return _set_converter([self.success, self.failure])

    def distribution(self, key=None):
        if key is None:
            self._deprecation_warn_distribution()
            return BernoulliDistribution(self.p)
        return BernoulliDistribution(self.p, self.success, self.failure)

    def simple_rv(self, rv):
        return Bernoulli(rv.name, p=self.p,
                succ=self.success, fail=self.failure)

    def expectation(self, expr, condition=None, evaluate=True, **kwargs):
        """
        Computes expectation.

        Parameters
        ==========

        expr : RandomIndexedSymbol, Relational, Logic
            Condition for which expectation has to be computed. Must
            contain a RandomIndexedSymbol of the process.
        condition : Relational, Logic
            The given conditions under which computations should be done.

        Returns
        =======

        Expectation of the RandomIndexedSymbol.

        """

        return _SubstituteRV._expectation(expr, condition, evaluate, **kwargs)

    def probability(self, condition, given_condition=None, evaluate=True, **kwargs):
        """
        Computes probability.

        Parameters
        ==========

        condition : Relational
                Condition for which probability has to be computed. Must
                contain a RandomIndexedSymbol of the process.
        given_condition : Relational, Logic
                The given conditions under which computations should be done.

        Returns
        =======

        Probability of the condition.

        """

        return _SubstituteRV._probability(condition, given_condition, evaluate, **kwargs)

    def density(self, x):
        return Piecewise((self.p, Eq(x, self.success)),
                         (1 - self.p, Eq(x, self.failure)),
                         (S.Zero, True))

class _SubstituteRV:
    """
    Internal class to handle the queries of expectation and probability
    by substitution.
    """

    @staticmethod
    def _rvindexed_subs(expr, condition=None):
        """
        Substitutes the RandomIndexedSymbol with the RandomSymbol with
        same name, distribution and probability as RandomIndexedSymbol.

        Parameters
        ==========

        expr: RandomIndexedSymbol, Relational, Logic
            Condition for which expectation has to be computed. Must
            contain a RandomIndexedSymbol of the process.
        condition: Relational, Logic
            The given conditions under which computations should be done.

        """

        rvs_expr = random_symbols(expr)
        if len(rvs_expr) != 0:
            swapdict_expr = {}
            for rv in rvs_expr:
                if isinstance(rv, RandomIndexedSymbol):
                    newrv = rv.pspace.process.simple_rv(rv) # substitute with equivalent simple rv
                    swapdict_expr[rv] = newrv
            expr = expr.subs(swapdict_expr)
        rvs_cond = random_symbols(condition)
        if len(rvs_cond)!=0:
            swapdict_cond = {}
            for rv in rvs_cond:
                if isinstance(rv, RandomIndexedSymbol):
                    newrv = rv.pspace.process.simple_rv(rv)
                    swapdict_cond[rv] = newrv
            condition = condition.subs(swapdict_cond)
        return expr, condition

    @classmethod
    def _expectation(self, expr, condition=None, evaluate=True, **kwargs):
        """
        Internal method for computing expectation of indexed RV.

        Parameters
        ==========

        expr: RandomIndexedSymbol, Relational, Logic
            Condition for which expectation has to be computed. Must
            contain a RandomIndexedSymbol of the process.
        condition: Relational, Logic
            The given conditions under which computations should be done.

        Returns
        =======

        Expectation of the RandomIndexedSymbol.

        """
        new_expr, new_condition = self._rvindexed_subs(expr, condition)

        if not is_random(new_expr):
            return new_expr
        new_pspace = pspace(new_expr)
        if new_condition is not None:
            new_expr = given(new_expr, new_condition)
        if new_expr.is_Add:  # As E is Linear
            return Add(*[new_pspace.compute_expectation(
                        expr=arg, evaluate=evaluate, **kwargs)
                        for arg in new_expr.args])
        return new_pspace.compute_expectation(
                new_expr, evaluate=evaluate, **kwargs)

    @classmethod
    def _probability(self, condition, given_condition=None, evaluate=True, **kwargs):
        """
        Internal method for computing probability of indexed RV

        Parameters
        ==========

        condition: Relational
                Condition for which probability has to be computed. Must
                contain a RandomIndexedSymbol of the process.
        given_condition: Relational/And
                The given conditions under which computations should be done.

        Returns
        =======

        Probability of the condition.

        """
        new_condition, new_givencondition = self._rvindexed_subs(condition, given_condition)

        if isinstance(new_givencondition, RandomSymbol):
            condrv = random_symbols(new_condition)
            if len(condrv) == 1 and condrv[0] == new_givencondition:
                return BernoulliDistribution(self._probability(new_condition), 0, 1)

            if any(dependent(rv, new_givencondition) for rv in condrv):
                return Probability(new_condition, new_givencondition)
            else:
                return self._probability(new_condition)

        if new_givencondition is not None and \
                not isinstance(new_givencondition, (Relational, Boolean)):
            raise ValueError("%s is not a relational or combination of relationals"
                    % (new_givencondition))
        if new_givencondition == False or new_condition == False:
            return S.Zero
        if new_condition == True:
            return S.One
        if not isinstance(new_condition, (Relational, Boolean)):
            raise ValueError("%s is not a relational or combination of relationals"
                    % (new_condition))

        if new_givencondition is not None:  # If there is a condition
        # Recompute on new conditional expr
            return self._probability(given(new_condition, new_givencondition, **kwargs), **kwargs)
        result = pspace(new_condition).probability(new_condition, **kwargs)
        if evaluate and hasattr(result, 'doit'):
            return result.doit()
        else:
            return result

def get_timerv_swaps(expr, condition):
    """
    Finds the appropriate interval for each time stamp in expr by parsing
    the given condition and returns intervals for each timestamp and
    dictionary that maps variable time-stamped Random Indexed Symbol to its
    corresponding Random Indexed variable with fixed time stamp.

    Parameters
    ==========

    expr: SymPy Expression
        Expression containing Random Indexed Symbols with variable time stamps
    condition: Relational/Boolean Expression
        Expression containing time bounds of variable time stamps in expr

    Examples
    ========

    >>> from sympy.stats.stochastic_process_types import get_timerv_swaps, PoissonProcess
    >>> from sympy import symbols, Contains, Interval
    >>> x, t, d = symbols('x t d', positive=True)
    >>> X = PoissonProcess("X", 3)
    >>> get_timerv_swaps(x*X(t), Contains(t, Interval.Lopen(0, 1)))
    ([Interval.Lopen(0, 1)], {X(t): X(1)})
    >>> get_timerv_swaps((X(t)**2 + X(d)**2), Contains(t, Interval.Lopen(0, 1))
    ... & Contains(d, Interval.Ropen(1, 4))) # doctest: +SKIP
    ([Interval.Ropen(1, 4), Interval.Lopen(0, 1)], {X(d): X(3), X(t): X(1)})

    Returns
    =======

    intervals: list
        List of Intervals/FiniteSet on which each time stamp is defined
    rv_swap: dict
        Dictionary mapping variable time Random Indexed Symbol to constant time
        Random Indexed Variable

    """

    if not isinstance(condition, (Relational, Boolean)):
        raise ValueError("%s is not a relational or combination of relationals"
            % (condition))
    expr_syms = list(expr.atoms(RandomIndexedSymbol))
    if isinstance(condition, (And, Or)):
        given_cond_args = condition.args
    else: # single condition
        given_cond_args = (condition, )
    rv_swap = {}
    intervals = []
    for expr_sym in expr_syms:
        for arg in given_cond_args:
            if arg.has(expr_sym.key) and isinstance(expr_sym.key, Symbol):
                intv = _set_converter(arg.args[1])
                diff_key = intv._sup - intv._inf
                if diff_key == oo:
                    raise ValueError("%s should have finite bounds" % str(expr_sym.name))
                elif diff_key == S.Zero: # has singleton set
                    diff_key = intv._sup
                rv_swap[expr_sym] = expr_sym.subs({expr_sym.key: diff_key})
                intervals.append(intv)
    return intervals, rv_swap


class CountingProcess(ContinuousTimeStochasticProcess):
    """
    This class handles the common methods of the Counting Processes
    such as Poisson, Wiener and Gamma Processes
    """
    index_set = _set_converter(Interval(0, oo))

    @property
    def symbol(self):
        return self.args[0]

    def expectation(self, expr, condition=None, evaluate=True, **kwargs):
        """
        Computes expectation

        Parameters
        ==========

        expr: RandomIndexedSymbol, Relational, Logic
            Condition for which expectation has to be computed. Must
            contain a RandomIndexedSymbol of the process.
        condition: Relational, Boolean
            The given conditions under which computations should be done, i.e,
            the intervals on which each variable time stamp in expr is defined

        Returns
        =======

        Expectation of the given expr

        """
        if condition is not None:
            intervals, rv_swap = get_timerv_swaps(expr, condition)
             # they are independent when they have non-overlapping intervals
            if len(intervals) == 1 or all(Intersection(*intv_comb) == EmptySet
                for intv_comb in itertools.combinations(intervals, 2)):
                if expr.is_Add:
                    return Add.fromiter(self.expectation(arg, condition)
                            for arg in expr.args)
                expr = expr.subs(rv_swap)
            else:
                return Expectation(expr, condition)

        return _SubstituteRV._expectation(expr, evaluate=evaluate, **kwargs)

    def _solve_argwith_tworvs(self, arg):
        if arg.args[0].key >= arg.args[1].key or isinstance(arg, Eq):
            diff_key = abs(arg.args[0].key - arg.args[1].key)
            rv = arg.args[0]
            arg = arg.__class__(rv.pspace.process(diff_key), 0)
        else:
            diff_key = arg.args[1].key - arg.args[0].key
            rv = arg.args[1]
            arg = arg.__class__(rv.pspace.process(diff_key), 0)
        return arg

    def _solve_numerical(self, condition, given_condition=None):
        if isinstance(condition, And):
            args_list = list(condition.args)
        else:
            args_list = [condition]
        if given_condition is not None:
            if isinstance(given_condition, And):
                args_list.extend(list(given_condition.args))
            else:
                args_list.extend([given_condition])
        # sort the args based on timestamp to get the independent increments in
        # each segment using all the condition args as well as given_condition args
        args_list = sorted(args_list, key=lambda x: x.args[0].key)
        result = []
        cond_args = list(condition.args) if isinstance(condition, And) else [condition]
        if args_list[0] in cond_args and not (is_random(args_list[0].args[0])
                        and is_random(args_list[0].args[1])):
            result.append(_SubstituteRV._probability(args_list[0]))

        if is_random(args_list[0].args[0]) and is_random(args_list[0].args[1]):
            arg = self._solve_argwith_tworvs(args_list[0])
            result.append(_SubstituteRV._probability(arg))

        for i in range(len(args_list) - 1):
            curr, nex = args_list[i], args_list[i + 1]
            diff_key = nex.args[0].key - curr.args[0].key
            working_set = curr.args[0].pspace.process.state_space
            if curr.args[1] > nex.args[1]: #impossible condition so return 0
                result.append(0)
                break
            if isinstance(curr, Eq):
                working_set = Intersection(working_set, Interval.Lopen(curr.args[1], oo))
            else:
                working_set = Intersection(working_set, curr.as_set())
            if isinstance(nex, Eq):
                working_set = Intersection(working_set, Interval(-oo, nex.args[1]))
            else:
                working_set = Intersection(working_set, nex.as_set())
            if working_set == EmptySet:
                rv = Eq(curr.args[0].pspace.process(diff_key), 0)
                result.append(_SubstituteRV._probability(rv))
            else:
                if working_set.is_finite_set:
                    if isinstance(curr, Eq) and isinstance(nex, Eq):
                        rv = Eq(curr.args[0].pspace.process(diff_key), len(working_set))
                        result.append(_SubstituteRV._probability(rv))
                    elif isinstance(curr, Eq) ^ isinstance(nex, Eq):
                        result.append(Add.fromiter(_SubstituteRV._probability(Eq(
                        curr.args[0].pspace.process(diff_key), x))
                                for x in range(len(working_set))))
                    else:
                        n = len(working_set)
                        result.append(Add.fromiter((n - x)*_SubstituteRV._probability(Eq(
                        curr.args[0].pspace.process(diff_key), x)) for x in range(n)))
                else:
                    result.append(_SubstituteRV._probability(
                    curr.args[0].pspace.process(diff_key) <= working_set._sup - working_set._inf))
        return Mul.fromiter(result)


    def probability(self, condition, given_condition=None, evaluate=True, **kwargs):
        """
        Computes probability.

        Parameters
        ==========

        condition: Relational
            Condition for which probability has to be computed. Must
            contain a RandomIndexedSymbol of the process.
        given_condition: Relational, Boolean
            The given conditions under which computations should be done, i.e,
            the intervals on which each variable time stamp in expr is defined

        Returns
        =======

        Probability of the condition

        """
        check_numeric = True
        if isinstance(condition, (And, Or)):
            cond_args = condition.args
        else:
            cond_args = (condition, )
        # check that condition args are numeric or not
        if not all(arg.args[0].key.is_number for arg in cond_args):
            check_numeric = False
        if given_condition is not None:
            check_given_numeric = True
            if isinstance(given_condition, (And, Or)):
                given_cond_args = given_condition.args
            else:
                given_cond_args = (given_condition, )
            # check that given condition args are numeric or not
            if given_condition.has(Contains):
                check_given_numeric = False
            # Handle numerical queries
            if check_numeric and check_given_numeric:
                res = []
                if isinstance(condition, Or):
                    res.append(Add.fromiter(self._solve_numerical(arg, given_condition)
                            for arg in condition.args))
                if isinstance(given_condition, Or):
                    res.append(Add.fromiter(self._solve_numerical(condition, arg)
                            for arg in given_condition.args))
                if res:
                    return Add.fromiter(res)
                return self._solve_numerical(condition, given_condition)

            # No numeric queries, go by Contains?... then check that all the
            # given condition are in form of `Contains`
            if not all(arg.has(Contains) for arg in given_cond_args):
                raise ValueError("If given condition is passed with `Contains`, then "
                "please pass the evaluated condition with its corresponding information "
                "in terms of intervals of each time stamp to be passed in given condition.")

            intervals, rv_swap = get_timerv_swaps(condition, given_condition)
            # they are independent when they have non-overlapping intervals
            if len(intervals) == 1 or all(Intersection(*intv_comb) == EmptySet
                for intv_comb in itertools.combinations(intervals, 2)):
                if isinstance(condition, And):
                    return Mul.fromiter(self.probability(arg, given_condition)
                            for arg in condition.args)
                elif isinstance(condition, Or):
                    return Add.fromiter(self.probability(arg, given_condition)
                            for arg in condition.args)
                condition = condition.subs(rv_swap)
            else:
                return Probability(condition, given_condition)
        if check_numeric:
            return self._solve_numerical(condition)
        return _SubstituteRV._probability(condition, evaluate=evaluate, **kwargs)

class PoissonProcess(CountingProcess):
    """
    The Poisson process is a counting process. It is usually used in scenarios
    where we are counting the occurrences of certain events that appear
    to happen at a certain rate, but completely at random.

    Parameters
    ==========

    sym : Symbol/str
    lamda : Positive number
        Rate of the process, ``lambda > 0``

    Examples
    ========

    >>> from sympy.stats import PoissonProcess, P, E
    >>> from sympy import symbols, Eq, Ne, Contains, Interval
    >>> X = PoissonProcess("X", lamda=3)
    >>> X.state_space
    Naturals0
    >>> X.lamda
    3
    >>> t1, t2 = symbols('t1 t2', positive=True)
    >>> P(X(t1) < 4)
    (9*t1**3/2 + 9*t1**2/2 + 3*t1 + 1)*exp(-3*t1)
    >>> P(Eq(X(t1), 2) | Ne(X(t1), 4), Contains(t1, Interval.Ropen(2, 4)))
    1 - 36*exp(-6)
    >>> P(Eq(X(t1), 2) & Eq(X(t2), 3), Contains(t1, Interval.Lopen(0, 2))
    ... & Contains(t2, Interval.Lopen(2, 4)))
    648*exp(-12)
    >>> E(X(t1))
    3*t1
    >>> E(X(t1)**2 + 2*X(t2),  Contains(t1, Interval.Lopen(0, 1))
    ... & Contains(t2, Interval.Lopen(1, 2)))
    18
    >>> P(X(3) < 1, Eq(X(1), 0))
    exp(-6)
    >>> P(Eq(X(4), 3), Eq(X(2), 3))
    exp(-6)
    >>> P(X(2) <= 3, X(1) > 1)
    5*exp(-3)

    Merging two Poisson Processes

    >>> Y = PoissonProcess("Y", lamda=4)
    >>> Z = X + Y
    >>> Z.lamda
    7

    Splitting a Poisson Process into two independent Poisson Processes

    >>> N, M = Z.split(l1=2, l2=5)
    >>> N.lamda, M.lamda
    (2, 5)

    References
    ==========

    .. [1] https://www.probabilitycourse.com/chapter11/11_0_0_intro.php
    .. [2] https://en.wikipedia.org/wiki/Poisson_point_process

    """

    def __new__(cls, sym, lamda):
        _value_check(lamda > 0, 'lamda should be a positive number.')
        sym = _symbol_converter(sym)
        lamda = _sympify(lamda)
        return Basic.__new__(cls, sym, lamda)

    @property
    def lamda(self):
        return self.args[1]

    @property
    def state_space(self):
        return S.Naturals0

    def distribution(self, key):
        if isinstance(key, RandomIndexedSymbol):
            self._deprecation_warn_distribution()
            return PoissonDistribution(self.lamda*key.key)
        return PoissonDistribution(self.lamda*key)

    def density(self, x):
        return (self.lamda*x.key)**x / factorial(x) * exp(-(self.lamda*x.key))

    def simple_rv(self, rv):
        return Poisson(rv.name, lamda=self.lamda*rv.key)

    def __add__(self, other):
        if not isinstance(other, PoissonProcess):
            raise ValueError("Only instances of Poisson Process can be merged")
        return PoissonProcess(Dummy(self.symbol.name + other.symbol.name),
                self.lamda + other.lamda)

    def split(self, l1, l2):
        if _sympify(l1 + l2) != self.lamda:
            raise ValueError("Sum of l1 and l2 should be %s" % str(self.lamda))
        return PoissonProcess(Dummy("l1"), l1), PoissonProcess(Dummy("l2"), l2)

class WienerProcess(CountingProcess):
    """
    The Wiener process is a real valued continuous-time stochastic process.
    In physics it is used to study Brownian motion and it is often also called
    Brownian motion due to its historical connection with physical process of the
    same name originally observed by Scottish botanist Robert Brown.

    Parameters
    ==========

    sym : Symbol/str

    Examples
    ========

    >>> from sympy.stats import WienerProcess, P, E
    >>> from sympy import symbols, Contains, Interval
    >>> X = WienerProcess("X")
    >>> X.state_space
    Reals
    >>> t1, t2 = symbols('t1 t2', positive=True)
    >>> P(X(t1) < 7).simplify()
    erf(7*sqrt(2)/(2*sqrt(t1)))/2 + 1/2
    >>> P((X(t1) > 2) | (X(t1) < 4), Contains(t1, Interval.Ropen(2, 4))).simplify()
    -erf(1)/2 + erf(2)/2 + 1
    >>> E(X(t1))
    0
    >>> E(X(t1) + 2*X(t2),  Contains(t1, Interval.Lopen(0, 1))
    ... & Contains(t2, Interval.Lopen(1, 2)))
    0

    References
    ==========

    .. [1] https://www.probabilitycourse.com/chapter11/11_4_0_brownian_motion_wiener_process.php
    .. [2] https://en.wikipedia.org/wiki/Wiener_process

    """
    def __new__(cls, sym):
        sym = _symbol_converter(sym)
        return Basic.__new__(cls, sym)

    @property
    def state_space(self):
        return S.Reals

    def distribution(self, key):
        if isinstance(key, RandomIndexedSymbol):
            self._deprecation_warn_distribution()
            return NormalDistribution(0, sqrt(key.key))
        return NormalDistribution(0, sqrt(key))

    def density(self, x):
        return exp(-x**2/(2*x.key)) / (sqrt(2*pi)*sqrt(x.key))

    def simple_rv(self, rv):
        return Normal(rv.name, 0, sqrt(rv.key))


class GammaProcess(CountingProcess):
    r"""
    A Gamma process is a random process with independent gamma distributed
    increments. It is a pure-jump increasing Levy process.

    Parameters
    ==========

    sym : Symbol/str
    lamda : Positive number
        Jump size of the process, ``lamda > 0``
    gamma : Positive number
        Rate of jump arrivals, `\gamma > 0`

    Examples
    ========

    >>> from sympy.stats import GammaProcess, E, P, variance
    >>> from sympy import symbols, Contains, Interval, Not
    >>> t, d, x, l, g = symbols('t d x l g', positive=True)
    >>> X = GammaProcess("X", l, g)
    >>> E(X(t))
    g*t/l
    >>> variance(X(t)).simplify()
    g*t/l**2
    >>> X = GammaProcess('X', 1, 2)
    >>> P(X(t) < 1).simplify()
    lowergamma(2*t, 1)/gamma(2*t)
    >>> P(Not((X(t) < 5) & (X(d) > 3)), Contains(t, Interval.Ropen(2, 4)) &
    ... Contains(d, Interval.Lopen(7, 8))).simplify()
    -4*exp(-3) + 472*exp(-8)/3 + 1
    >>> E(X(2) + x*E(X(5)))
    10*x + 4

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gamma_process

    """
    def __new__(cls, sym, lamda, gamma):
        _value_check(lamda > 0, 'lamda should be a positive number')
        _value_check(gamma > 0, 'gamma should be a positive number')
        sym = _symbol_converter(sym)
        gamma = _sympify(gamma)
        lamda = _sympify(lamda)
        return Basic.__new__(cls, sym, lamda, gamma)

    @property
    def lamda(self):
        return self.args[1]

    @property
    def gamma(self):
        return self.args[2]

    @property
    def state_space(self):
        return _set_converter(Interval(0, oo))

    def distribution(self, key):
        if isinstance(key, RandomIndexedSymbol):
            self._deprecation_warn_distribution()
            return GammaDistribution(self.gamma*key.key, 1/self.lamda)
        return GammaDistribution(self.gamma*key, 1/self.lamda)

    def density(self, x):
        k = self.gamma*x.key
        theta = 1/self.lamda
        return x**(k - 1) * exp(-x/theta) / (gamma(k)*theta**k)

    def simple_rv(self, rv):
        return Gamma(rv.name, self.gamma*rv.key, 1/self.lamda)
