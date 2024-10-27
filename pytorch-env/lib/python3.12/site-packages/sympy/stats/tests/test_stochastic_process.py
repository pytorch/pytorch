from sympy.concrete.summations import Sum
from sympy.core.containers import Tuple
from sympy.core.function import Lambda
from sympy.core.numbers import (Float, Rational, oo, pi)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.error_functions import erf
from sympy.functions.special.gamma_functions import (gamma, lowergamma)
from sympy.logic.boolalg import (And, Not)
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.immutable import ImmutableMatrix
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Range
from sympy.sets.sets import (FiniteSet, Interval)
from sympy.stats import (DiscreteMarkovChain, P, TransitionMatrixOf, E,
                         StochasticStateSpaceOf, variance, ContinuousMarkovChain,
                         BernoulliProcess, PoissonProcess, WienerProcess,
                         GammaProcess, sample_stochastic_process)
from sympy.stats.joint_rv import JointDistribution
from sympy.stats.joint_rv_types import JointDistributionHandmade
from sympy.stats.rv import RandomIndexedSymbol
from sympy.stats.symbolic_probability import Probability, Expectation
from sympy.testing.pytest import (raises, skip, ignore_warnings,
                                  warns_deprecated_sympy)
from sympy.external import import_module
from sympy.stats.frv_types import BernoulliDistribution
from sympy.stats.drv_types import PoissonDistribution
from sympy.stats.crv_types import NormalDistribution, GammaDistribution
from sympy.core.symbol import Str


def test_DiscreteMarkovChain():

    # pass only the name
    X = DiscreteMarkovChain("X")
    assert isinstance(X.state_space, Range)
    assert X.index_set == S.Naturals0
    assert isinstance(X.transition_probabilities, MatrixSymbol)
    t = symbols('t', positive=True, integer=True)
    assert isinstance(X[t], RandomIndexedSymbol)
    assert E(X[0]) == Expectation(X[0])
    raises(TypeError, lambda: DiscreteMarkovChain(1))
    raises(NotImplementedError, lambda: X(t))
    raises(NotImplementedError, lambda: X.communication_classes())
    raises(NotImplementedError, lambda: X.canonical_form())
    raises(NotImplementedError, lambda: X.decompose())

    nz = Symbol('n', integer=True)
    TZ = MatrixSymbol('M', nz, nz)
    SZ = Range(nz)
    YZ = DiscreteMarkovChain('Y', SZ, TZ)
    assert P(Eq(YZ[2], 1), Eq(YZ[1], 0)) == TZ[0, 1]

    raises(ValueError, lambda: sample_stochastic_process(t))
    raises(ValueError, lambda: next(sample_stochastic_process(X)))
    # pass name and state_space
    # any hashable object should be a valid state
    # states should be valid as a tuple/set/list/Tuple/Range
    sym, rainy, cloudy, sunny = symbols('a Rainy Cloudy Sunny', real=True)
    state_spaces = [(1, 2, 3), [Str('Hello'), sym, DiscreteMarkovChain("Y", (1,2,3))],
                    Tuple(S(1), exp(sym), Str('World'), sympify=False), Range(-1, 5, 2),
                    [rainy, cloudy, sunny]]
    chains = [DiscreteMarkovChain("Y", state_space) for state_space in state_spaces]

    for i, Y in enumerate(chains):
        assert isinstance(Y.transition_probabilities, MatrixSymbol)
        assert Y.state_space == state_spaces[i] or Y.state_space == FiniteSet(*state_spaces[i])
        assert Y.number_of_states == 3

        with ignore_warnings(UserWarning):  # TODO: Restore tests once warnings are removed
            assert P(Eq(Y[2], 1), Eq(Y[0], 2), evaluate=False) == Probability(Eq(Y[2], 1), Eq(Y[0], 2))
        assert E(Y[0]) == Expectation(Y[0])

        raises(ValueError, lambda: next(sample_stochastic_process(Y)))

    raises(TypeError, lambda: DiscreteMarkovChain("Y", {1: 1}))
    Y = DiscreteMarkovChain("Y", Range(1, t, 2))
    assert Y.number_of_states == ceiling((t-1)/2)

    # pass name and transition_probabilities
    chains = [DiscreteMarkovChain("Y", trans_probs=Matrix([[]])),
              DiscreteMarkovChain("Y", trans_probs=Matrix([[0, 1], [1, 0]])),
              DiscreteMarkovChain("Y", trans_probs=Matrix([[pi, 1-pi], [sym, 1-sym]]))]
    for Z in chains:
        assert Z.number_of_states == Z.transition_probabilities.shape[0]
        assert isinstance(Z.transition_probabilities, ImmutableMatrix)

    # pass name, state_space and transition_probabilities
    T = Matrix([[0.5, 0.2, 0.3],[0.2, 0.5, 0.3],[0.2, 0.3, 0.5]])
    TS = MatrixSymbol('T', 3, 3)
    Y = DiscreteMarkovChain("Y", [0, 1, 2], T)
    YS = DiscreteMarkovChain("Y", ['One', 'Two', 3], TS)
    assert Y.joint_distribution(1, Y[2], 3) == JointDistribution(Y[1], Y[2], Y[3])
    raises(ValueError, lambda: Y.joint_distribution(Y[1].symbol, Y[2].symbol))
    assert P(Eq(Y[3], 2), Eq(Y[1], 1)).round(2) == Float(0.36, 2)
    assert (P(Eq(YS[3], 2), Eq(YS[1], 1)) -
            (TS[0, 2]*TS[1, 0] + TS[1, 1]*TS[1, 2] + TS[1, 2]*TS[2, 2])).simplify() == 0
    assert P(Eq(YS[1], 1), Eq(YS[2], 2)) == Probability(Eq(YS[1], 1))
    assert P(Eq(YS[3], 3), Eq(YS[1], 1)) == TS[0, 2]*TS[1, 0] + TS[1, 1]*TS[1, 2] + TS[1, 2]*TS[2, 2]
    TO = Matrix([[0.25, 0.75, 0],[0, 0.25, 0.75],[0.75, 0, 0.25]])
    assert P(Eq(Y[3], 2), Eq(Y[1], 1) & TransitionMatrixOf(Y, TO)).round(3) == Float(0.375, 3)
    with ignore_warnings(UserWarning): ### TODO: Restore tests once warnings are removed
        assert E(Y[3], evaluate=False) == Expectation(Y[3])
        assert E(Y[3], Eq(Y[2], 1)).round(2) == Float(1.1, 3)
    TSO = MatrixSymbol('T', 4, 4)
    raises(ValueError, lambda: str(P(Eq(YS[3], 2), Eq(YS[1], 1) & TransitionMatrixOf(YS, TSO))))
    raises(TypeError, lambda: DiscreteMarkovChain("Z", [0, 1, 2], symbols('M')))
    raises(ValueError, lambda: DiscreteMarkovChain("Z", [0, 1, 2], MatrixSymbol('T', 3, 4)))
    raises(ValueError, lambda: E(Y[3], Eq(Y[2], 6)))
    raises(ValueError, lambda: E(Y[2], Eq(Y[3], 1)))


    # extended tests for probability queries
    TO1 = Matrix([[Rational(1, 4), Rational(3, 4), 0],[Rational(1, 3), Rational(1, 3), Rational(1, 3)],[0, Rational(1, 4), Rational(3, 4)]])
    assert P(And(Eq(Y[2], 1), Eq(Y[1], 1), Eq(Y[0], 0)),
            Eq(Probability(Eq(Y[0], 0)), Rational(1, 4)) & TransitionMatrixOf(Y, TO1)) == Rational(1, 16)
    assert P(And(Eq(Y[2], 1), Eq(Y[1], 1), Eq(Y[0], 0)), TransitionMatrixOf(Y, TO1)) == \
            Probability(Eq(Y[0], 0))/4
    assert P(Lt(X[1], 2) & Gt(X[1], 0), Eq(X[0], 2) &
        StochasticStateSpaceOf(X, [0, 1, 2]) & TransitionMatrixOf(X, TO1)) == Rational(1, 4)
    assert P(Lt(X[1], 2) & Gt(X[1], 0), Eq(X[0], 2) &
             StochasticStateSpaceOf(X, [S(0), '0', 1]) & TransitionMatrixOf(X, TO1)) == Rational(1, 4)
    assert P(Ne(X[1], 2) & Ne(X[1], 1), Eq(X[0], 2) &
        StochasticStateSpaceOf(X, [0, 1, 2]) & TransitionMatrixOf(X, TO1)) is S.Zero
    assert P(Ne(X[1], 2) & Ne(X[1], 1), Eq(X[0], 2) &
             StochasticStateSpaceOf(X, [S(0), '0', 1]) & TransitionMatrixOf(X, TO1)) is S.Zero
    assert P(And(Eq(Y[2], 1), Eq(Y[1], 1), Eq(Y[0], 0)), Eq(Y[1], 1)) == 0.1*Probability(Eq(Y[0], 0))

    # testing properties of Markov chain
    TO2 = Matrix([[S.One, 0, 0],[Rational(1, 3), Rational(1, 3), Rational(1, 3)],[0, Rational(1, 4), Rational(3, 4)]])
    TO3 = Matrix([[Rational(1, 4), Rational(3, 4), 0],[Rational(1, 3), Rational(1, 3), Rational(1, 3)], [0, Rational(1, 4), Rational(3, 4)]])
    Y2 = DiscreteMarkovChain('Y', trans_probs=TO2)
    Y3 = DiscreteMarkovChain('Y', trans_probs=TO3)
    assert Y3.fundamental_matrix() == ImmutableMatrix([[176, 81, -132], [36, 141, -52], [-44, -39, 208]])/125
    assert Y2.is_absorbing_chain() == True
    assert Y3.is_absorbing_chain() == False
    assert Y2.canonical_form() == ([0, 1, 2], TO2)
    assert Y3.canonical_form() == ([0, 1, 2], TO3)
    assert Y2.decompose() == ([0, 1, 2], TO2[0:1, 0:1], TO2[1:3, 0:1], TO2[1:3, 1:3])
    assert Y3.decompose() == ([0, 1, 2], TO3, Matrix(0, 3, []), Matrix(0, 0, []))
    TO4 = Matrix([[Rational(1, 5), Rational(2, 5), Rational(2, 5)], [Rational(1, 10), S.Half, Rational(2, 5)], [Rational(3, 5), Rational(3, 10), Rational(1, 10)]])
    Y4 = DiscreteMarkovChain('Y', trans_probs=TO4)
    w = ImmutableMatrix([[Rational(11, 39), Rational(16, 39), Rational(4, 13)]])
    assert Y4.limiting_distribution == w
    assert Y4.is_regular() == True
    assert Y4.is_ergodic() == True
    TS1 = MatrixSymbol('T', 3, 3)
    Y5 = DiscreteMarkovChain('Y', trans_probs=TS1)
    assert Y5.limiting_distribution(w, TO4).doit() == True
    assert Y5.stationary_distribution(condition_set=True).subs(TS1, TO4).contains(w).doit() == S.true
    TO6 = Matrix([[S.One, 0, 0, 0, 0],[S.Half, 0, S.Half, 0, 0],[0, S.Half, 0, S.Half, 0], [0, 0, S.Half, 0, S.Half], [0, 0, 0, 0, 1]])
    Y6 = DiscreteMarkovChain('Y', trans_probs=TO6)
    assert Y6.fundamental_matrix() == ImmutableMatrix([[Rational(3, 2), S.One, S.Half], [S.One, S(2), S.One], [S.Half, S.One, Rational(3, 2)]])
    assert Y6.absorbing_probabilities() == ImmutableMatrix([[Rational(3, 4), Rational(1, 4)], [S.Half, S.Half], [Rational(1, 4), Rational(3, 4)]])
    with warns_deprecated_sympy():
        Y6.absorbing_probabilites()
    TO7 = Matrix([[Rational(1, 2), Rational(1, 4), Rational(1, 4)], [Rational(1, 2), 0, Rational(1, 2)], [Rational(1, 4), Rational(1, 4), Rational(1, 2)]])
    Y7 = DiscreteMarkovChain('Y', trans_probs=TO7)
    assert Y7.is_absorbing_chain() == False
    assert Y7.fundamental_matrix() == ImmutableMatrix([[Rational(86, 75), Rational(1, 25), Rational(-14, 75)],
                                                            [Rational(2, 25), Rational(21, 25), Rational(2, 25)],
                                                            [Rational(-14, 75), Rational(1, 25), Rational(86, 75)]])

    # test for zero-sized matrix functionality
    X = DiscreteMarkovChain('X', trans_probs=Matrix([[]]))
    assert X.number_of_states == 0
    assert X.stationary_distribution() == Matrix([[]])
    assert X.communication_classes() == []
    assert X.canonical_form() == ([], Matrix([[]]))
    assert X.decompose() == ([], Matrix([[]]), Matrix([[]]), Matrix([[]]))
    assert X.is_regular() == False
    assert X.is_ergodic() == False

    # test communication_class
    # see https://drive.google.com/drive/folders/1HbxLlwwn2b3U8Lj7eb_ASIUb5vYaNIjg?usp=sharing
    # tutorial 2.pdf
    TO7 = Matrix([[0, 5, 5, 0, 0],
                  [0, 0, 0, 10, 0],
                  [5, 0, 5, 0, 0],
                  [0, 10, 0, 0, 0],
                  [0, 3, 0, 3, 4]])/10
    Y7 = DiscreteMarkovChain('Y', trans_probs=TO7)
    tuples = Y7.communication_classes()
    classes, recurrence, periods = list(zip(*tuples))
    assert classes == ([1, 3], [0, 2], [4])
    assert recurrence == (True, False, False)
    assert periods == (2, 1, 1)

    TO8 = Matrix([[0, 0, 0, 10, 0, 0],
                  [5, 0, 5, 0, 0, 0],
                  [0, 4, 0, 0, 0, 6],
                  [10, 0, 0, 0, 0, 0],
                  [0, 10, 0, 0, 0, 0],
                  [0, 0, 0, 5, 5, 0]])/10
    Y8 = DiscreteMarkovChain('Y', trans_probs=TO8)
    tuples = Y8.communication_classes()
    classes, recurrence, periods = list(zip(*tuples))
    assert classes == ([0, 3], [1, 2, 5, 4])
    assert recurrence == (True, False)
    assert periods == (2, 2)

    TO9 = Matrix([[2, 0, 0, 3, 0, 0, 3, 2, 0, 0],
                  [0, 10, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 2, 2, 0, 0, 0, 0, 0, 3, 3],
                  [0, 0, 0, 3, 0, 0, 6, 1, 0, 0],
                  [0, 0, 0, 0, 5, 5, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 10, 0, 0, 0, 0],
                  [4, 0, 0, 5, 0, 0, 1, 0, 0, 0],
                  [2, 0, 0, 4, 0, 0, 2, 2, 0, 0],
                  [3, 0, 1, 0, 0, 0, 0, 0, 4, 2],
                  [0, 0, 4, 0, 0, 0, 0, 0, 3, 3]])/10
    Y9 = DiscreteMarkovChain('Y', trans_probs=TO9)
    tuples = Y9.communication_classes()
    classes, recurrence, periods = list(zip(*tuples))
    assert classes == ([0, 3, 6, 7], [1], [2, 8, 9], [5], [4])
    assert recurrence == (True, True, False, True, False)
    assert periods == (1, 1, 1, 1, 1)

    # test canonical form
    # see https://web.archive.org/web/20201230182007/https://www.dartmouth.edu/~chance/teaching_aids/books_articles/probability_book/Chapter11.pdf
    # example 11.13
    T = Matrix([[1, 0, 0, 0, 0],
                [S(1) / 2, 0, S(1) / 2, 0, 0],
                [0, S(1) / 2, 0, S(1) / 2, 0],
                [0, 0, S(1) / 2, 0, S(1) / 2],
                [0, 0, 0, 0, S(1)]])
    DW = DiscreteMarkovChain('DW', [0, 1, 2, 3, 4], T)
    states, A, B, C = DW.decompose()
    assert states == [0, 4, 1, 2, 3]
    assert A == Matrix([[1, 0], [0, 1]])
    assert B == Matrix([[S(1)/2, 0], [0, 0], [0, S(1)/2]])
    assert C == Matrix([[0, S(1)/2, 0], [S(1)/2, 0, S(1)/2], [0, S(1)/2, 0]])
    states, new_matrix = DW.canonical_form()
    assert states == [0, 4, 1, 2, 3]
    assert new_matrix == Matrix([[1, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 0],
                                 [S(1)/2, 0, 0, S(1)/2, 0],
                                 [0, 0, S(1)/2, 0, S(1)/2],
                                 [0, S(1)/2, 0, S(1)/2, 0]])

    # test regular and ergodic
    # https://web.archive.org/web/20201230182007/https://www.dartmouth.edu/~chance/teaching_aids/books_articles/probability_book/Chapter11.pdf
    T = Matrix([[0, 4, 0, 0, 0],
                [1, 0, 3, 0, 0],
                [0, 2, 0, 2, 0],
                [0, 0, 3, 0, 1],
                [0, 0, 0, 4, 0]])/4
    X = DiscreteMarkovChain('X', trans_probs=T)
    assert not X.is_regular()
    assert X.is_ergodic()
    T = Matrix([[0, 1], [1, 0]])
    X = DiscreteMarkovChain('X', trans_probs=T)
    assert not X.is_regular()
    assert X.is_ergodic()
    # http://www.math.wisc.edu/~valko/courses/331/MC2.pdf
    T = Matrix([[2, 1, 1],
                [2, 0, 2],
                [1, 1, 2]])/4
    X = DiscreteMarkovChain('X', trans_probs=T)
    assert X.is_regular()
    assert X.is_ergodic()
    # https://docs.ufpr.br/~lucambio/CE222/1S2014/Kemeny-Snell1976.pdf
    T = Matrix([[1, 1], [1, 1]])/2
    X = DiscreteMarkovChain('X', trans_probs=T)
    assert X.is_regular()
    assert X.is_ergodic()

    # test is_absorbing_chain
    T = Matrix([[0, 1, 0],
                [1, 0, 0],
                [0, 0, 1]])
    X = DiscreteMarkovChain('X', trans_probs=T)
    assert not X.is_absorbing_chain()
    # https://en.wikipedia.org/wiki/Absorbing_Markov_chain
    T = Matrix([[1, 1, 0, 0],
                [0, 1, 1, 0],
                [1, 0, 0, 1],
                [0, 0, 0, 2]])/2
    X = DiscreteMarkovChain('X', trans_probs=T)
    assert X.is_absorbing_chain()
    T = Matrix([[2, 0, 0, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 0, 1],
                [0, 0, 0, 0, 2]])/2
    X = DiscreteMarkovChain('X', trans_probs=T)
    assert X.is_absorbing_chain()

    # test custom state space
    Y10 = DiscreteMarkovChain('Y', [1, 2, 3], TO2)
    tuples = Y10.communication_classes()
    classes, recurrence, periods = list(zip(*tuples))
    assert classes == ([1], [2, 3])
    assert recurrence == (True, False)
    assert periods == (1, 1)
    assert Y10.canonical_form() == ([1, 2, 3], TO2)
    assert Y10.decompose() == ([1, 2, 3], TO2[0:1, 0:1], TO2[1:3, 0:1], TO2[1:3, 1:3])

    # testing miscellaneous queries
    T = Matrix([[S.Half, Rational(1, 4), Rational(1, 4)],
                [Rational(1, 3), 0, Rational(2, 3)],
                [S.Half, S.Half, 0]])
    X = DiscreteMarkovChain('X', [0, 1, 2], T)
    assert P(Eq(X[1], 2) & Eq(X[2], 1) & Eq(X[3], 0),
    Eq(P(Eq(X[1], 0)), Rational(1, 4)) & Eq(P(Eq(X[1], 1)), Rational(1, 4))) == Rational(1, 12)
    assert P(Eq(X[2], 1) | Eq(X[2], 2), Eq(X[1], 1)) == Rational(2, 3)
    assert P(Eq(X[2], 1) & Eq(X[2], 2), Eq(X[1], 1)) is S.Zero
    assert P(Ne(X[2], 2), Eq(X[1], 1)) == Rational(1, 3)
    assert E(X[1]**2, Eq(X[0], 1)) == Rational(8, 3)
    assert variance(X[1], Eq(X[0], 1)) == Rational(8, 9)
    raises(ValueError, lambda: E(X[1], Eq(X[2], 1)))
    raises(ValueError, lambda: DiscreteMarkovChain('X', [0, 1], T))

    # testing miscellaneous queries with different state space
    X = DiscreteMarkovChain('X', ['A', 'B', 'C'], T)
    assert P(Eq(X[1], 2) & Eq(X[2], 1) & Eq(X[3], 0),
    Eq(P(Eq(X[1], 0)), Rational(1, 4)) & Eq(P(Eq(X[1], 1)), Rational(1, 4))) == Rational(1, 12)
    assert P(Eq(X[2], 1) | Eq(X[2], 2), Eq(X[1], 1)) == Rational(2, 3)
    assert P(Eq(X[2], 1) & Eq(X[2], 2), Eq(X[1], 1)) is S.Zero
    assert P(Ne(X[2], 2), Eq(X[1], 1)) == Rational(1, 3)
    a = X.state_space.args[0]
    c = X.state_space.args[2]
    assert (E(X[1] ** 2, Eq(X[0], 1)) - (a**2/3 + 2*c**2/3)).simplify() == 0
    assert (variance(X[1], Eq(X[0], 1)) - (2*(-a/3 + c/3)**2/3 + (2*a/3 - 2*c/3)**2/3)).simplify() == 0
    raises(ValueError, lambda: E(X[1], Eq(X[2], 1)))

    #testing queries with multiple RandomIndexedSymbols
    T = Matrix([[Rational(5, 10), Rational(3, 10), Rational(2, 10)], [Rational(2, 10), Rational(7, 10), Rational(1, 10)], [Rational(3, 10), Rational(3, 10), Rational(4, 10)]])
    Y = DiscreteMarkovChain("Y", [0, 1, 2], T)
    assert P(Eq(Y[7], Y[5]), Eq(Y[2], 0)).round(5) == Float(0.44428, 5)
    assert P(Gt(Y[3], Y[1]), Eq(Y[0], 0)).round(2) == Float(0.36, 2)
    assert P(Le(Y[5], Y[10]), Eq(Y[4], 2)).round(6) == Float(0.583120, 6)
    assert Float(P(Eq(Y[10], Y[5]), Eq(Y[4], 1)), 14) == Float(1 - P(Ne(Y[10], Y[5]), Eq(Y[4], 1)), 14)
    assert Float(P(Gt(Y[8], Y[9]), Eq(Y[3], 2)), 14) == Float(1 - P(Le(Y[8], Y[9]), Eq(Y[3], 2)), 14)
    assert Float(P(Lt(Y[1], Y[4]), Eq(Y[0], 0)), 14) == Float(1 - P(Ge(Y[1], Y[4]), Eq(Y[0], 0)), 14)
    assert P(Eq(Y[5], Y[10]), Eq(Y[2], 1)) == P(Eq(Y[10], Y[5]), Eq(Y[2], 1))
    assert P(Gt(Y[1], Y[2]), Eq(Y[0], 1)) == P(Lt(Y[2], Y[1]), Eq(Y[0], 1))
    assert P(Ge(Y[7], Y[6]), Eq(Y[4], 1)) == P(Le(Y[6], Y[7]), Eq(Y[4], 1))

    #test symbolic queries
    a, b, c, d = symbols('a b c d')
    T = Matrix([[Rational(1, 10), Rational(4, 10), Rational(5, 10)], [Rational(3, 10), Rational(4, 10), Rational(3, 10)], [Rational(7, 10), Rational(2, 10), Rational(1, 10)]])
    Y = DiscreteMarkovChain("Y", [0, 1, 2], T)
    query = P(Eq(Y[a], b), Eq(Y[c], d))
    assert query.subs({a:10, b:2, c:5, d:1}).evalf().round(4) == P(Eq(Y[10], 2), Eq(Y[5], 1)).round(4)
    assert query.subs({a:15, b:0, c:10, d:1}).evalf().round(4) == P(Eq(Y[15], 0), Eq(Y[10], 1)).round(4)
    query_gt = P(Gt(Y[a], b), Eq(Y[c], d))
    query_le = P(Le(Y[a], b), Eq(Y[c], d))
    assert query_gt.subs({a:5, b:2, c:1, d:0}).evalf() + query_le.subs({a:5, b:2, c:1, d:0}).evalf() == 1.0
    query_ge = P(Ge(Y[a], b), Eq(Y[c], d))
    query_lt = P(Lt(Y[a], b), Eq(Y[c], d))
    assert query_ge.subs({a:4, b:1, c:0, d:2}).evalf() + query_lt.subs({a:4, b:1, c:0, d:2}).evalf() == 1.0

    #test issue 20078
    assert (2*Y[1] + 3*Y[1]).simplify() == 5*Y[1]
    assert (2*Y[1] - 3*Y[1]).simplify() == -Y[1]
    assert (2*(0.25*Y[1])).simplify() == 0.5*Y[1]
    assert ((2*Y[1]) * (0.25*Y[1])).simplify() == 0.5*Y[1]**2
    assert (Y[1]**2 + Y[1]**3).simplify() == (Y[1] + 1)*Y[1]**2

def test_sample_stochastic_process():
    if not import_module('scipy'):
        skip('SciPy Not installed. Skip sampling tests')
    import random
    random.seed(0)
    numpy = import_module('numpy')
    if numpy:
        numpy.random.seed(0) # scipy uses numpy to sample so to set its seed
    T = Matrix([[0.5, 0.2, 0.3],[0.2, 0.5, 0.3],[0.2, 0.3, 0.5]])
    Y = DiscreteMarkovChain("Y", [0, 1, 2], T)
    for samps in range(10):
        assert next(sample_stochastic_process(Y)) in Y.state_space
    Z = DiscreteMarkovChain("Z", ['1', 1, 0], T)
    for samps in range(10):
        assert next(sample_stochastic_process(Z)) in Z.state_space

    T = Matrix([[S.Half, Rational(1, 4), Rational(1, 4)],
                [Rational(1, 3), 0, Rational(2, 3)],
                [S.Half, S.Half, 0]])
    X = DiscreteMarkovChain('X', [0, 1, 2], T)
    for samps in range(10):
        assert next(sample_stochastic_process(X)) in X.state_space
    W = DiscreteMarkovChain('W', [1, pi, oo], T)
    for samps in range(10):
        assert next(sample_stochastic_process(W)) in W.state_space


def test_ContinuousMarkovChain():
    T1 = Matrix([[S(-2), S(2), S.Zero],
                 [S.Zero, S.NegativeOne, S.One],
                 [Rational(3, 2), Rational(3, 2), S(-3)]])
    C1 = ContinuousMarkovChain('C', [0, 1, 2], T1)
    assert C1.limiting_distribution() == ImmutableMatrix([[Rational(3, 19), Rational(12, 19), Rational(4, 19)]])

    T2 = Matrix([[-S.One, S.One, S.Zero], [S.One, -S.One, S.Zero], [S.Zero, S.One, -S.One]])
    C2 = ContinuousMarkovChain('C', [0, 1, 2], T2)
    A, t = C2.generator_matrix, symbols('t', positive=True)
    assert C2.transition_probabilities(A)(t) == Matrix([[S.Half + exp(-2*t)/2, S.Half - exp(-2*t)/2, 0],
                                                       [S.Half - exp(-2*t)/2, S.Half + exp(-2*t)/2, 0],
                                                       [S.Half - exp(-t) + exp(-2*t)/2, S.Half - exp(-2*t)/2, exp(-t)]])
    with ignore_warnings(UserWarning): ### TODO: Restore tests once warnings are removed
        assert P(Eq(C2(1), 1), Eq(C2(0), 1), evaluate=False) == Probability(Eq(C2(1), 1), Eq(C2(0), 1))
    assert P(Eq(C2(1), 1), Eq(C2(0), 1)) == exp(-2)/2 + S.Half
    assert P(Eq(C2(1), 0) & Eq(C2(2), 1) & Eq(C2(3), 1),
                Eq(P(Eq(C2(1), 0)), S.Half)) == (Rational(1, 4) - exp(-2)/4)*(exp(-2)/2 + S.Half)
    assert P(Not(Eq(C2(1), 0) & Eq(C2(2), 1) & Eq(C2(3), 2)) |
                (Eq(C2(1), 0) & Eq(C2(2), 1) & Eq(C2(3), 2)),
                Eq(P(Eq(C2(1), 0)), Rational(1, 4)) & Eq(P(Eq(C2(1), 1)), Rational(1, 4))) is S.One
    assert E(C2(Rational(3, 2)), Eq(C2(0), 2)) == -exp(-3)/2 + 2*exp(Rational(-3, 2)) + S.Half
    assert variance(C2(Rational(3, 2)), Eq(C2(0), 1)) == ((S.Half - exp(-3)/2)**2*(exp(-3)/2 + S.Half)
                                                    + (Rational(-1, 2) - exp(-3)/2)**2*(S.Half - exp(-3)/2))
    raises(KeyError, lambda: P(Eq(C2(1), 0), Eq(P(Eq(C2(1), 1)), S.Half)))
    assert P(Eq(C2(1), 0), Eq(P(Eq(C2(5), 1)), S.Half)) == Probability(Eq(C2(1), 0))
    TS1 = MatrixSymbol('G', 3, 3)
    CS1 = ContinuousMarkovChain('C', [0, 1, 2], TS1)
    A = CS1.generator_matrix
    assert CS1.transition_probabilities(A)(t) == exp(t*A)

    C3 = ContinuousMarkovChain('C', [Symbol('0'), Symbol('1'), Symbol('2')], T2)
    assert P(Eq(C3(1), 1), Eq(C3(0), 1)) == exp(-2)/2 + S.Half
    assert P(Eq(C3(1), Symbol('1')), Eq(C3(0), Symbol('1'))) == exp(-2)/2 + S.Half

    #test probability queries
    G = Matrix([[-S(1), Rational(1, 10), Rational(9, 10)], [Rational(2, 5), -S(1), Rational(3, 5)], [Rational(1, 2), Rational(1, 2), -S(1)]])
    C = ContinuousMarkovChain('C', state_space=[0, 1, 2], gen_mat=G)
    assert P(Eq(C(7.385), C(3.19)), Eq(C(0.862), 0)).round(5) == Float(0.35469, 5)
    assert P(Gt(C(98.715), C(19.807)), Eq(C(11.314), 2)).round(5) == Float(0.32452, 5)
    assert P(Le(C(5.9), C(10.112)), Eq(C(4), 1)).round(6) == Float(0.675214, 6)
    assert Float(P(Eq(C(7.32), C(2.91)), Eq(C(2.63), 1)), 14) == Float(1 - P(Ne(C(7.32), C(2.91)), Eq(C(2.63), 1)), 14)
    assert Float(P(Gt(C(3.36), C(1.101)), Eq(C(0.8), 2)), 14) == Float(1 - P(Le(C(3.36), C(1.101)), Eq(C(0.8), 2)), 14)
    assert Float(P(Lt(C(4.9), C(2.79)), Eq(C(1.61), 0)), 14) == Float(1 - P(Ge(C(4.9), C(2.79)), Eq(C(1.61), 0)), 14)
    assert P(Eq(C(5.243), C(10.912)), Eq(C(2.174), 1)) == P(Eq(C(10.912), C(5.243)), Eq(C(2.174), 1))
    assert P(Gt(C(2.344), C(9.9)), Eq(C(1.102), 1)) == P(Lt(C(9.9), C(2.344)), Eq(C(1.102), 1))
    assert P(Ge(C(7.87), C(1.008)), Eq(C(0.153), 1)) == P(Le(C(1.008), C(7.87)), Eq(C(0.153), 1))

    #test symbolic queries
    a, b, c, d = symbols('a b c d')
    query = P(Eq(C(a), b), Eq(C(c), d))
    assert query.subs({a:3.65, b:2, c:1.78, d:1}).evalf().round(10) == P(Eq(C(3.65), 2), Eq(C(1.78), 1)).round(10)
    query_gt = P(Gt(C(a), b), Eq(C(c), d))
    query_le = P(Le(C(a), b), Eq(C(c), d))
    assert query_gt.subs({a:13.2, b:0, c:3.29, d:2}).evalf() + query_le.subs({a:13.2, b:0, c:3.29, d:2}).evalf() == 1.0
    query_ge = P(Ge(C(a), b), Eq(C(c), d))
    query_lt = P(Lt(C(a), b), Eq(C(c), d))
    assert query_ge.subs({a:7.43, b:1, c:1.45, d:0}).evalf() + query_lt.subs({a:7.43, b:1, c:1.45, d:0}).evalf() == 1.0

    #test issue 20078
    assert (2*C(1) + 3*C(1)).simplify() == 5*C(1)
    assert (2*C(1) - 3*C(1)).simplify() == -C(1)
    assert (2*(0.25*C(1))).simplify() == 0.5*C(1)
    assert (2*C(1) * 0.25*C(1)).simplify() == 0.5*C(1)**2
    assert (C(1)**2 + C(1)**3).simplify() == (C(1) + 1)*C(1)**2

def test_BernoulliProcess():

    B = BernoulliProcess("B", p=0.6, success=1, failure=0)
    assert B.state_space == FiniteSet(0, 1)
    assert B.index_set == S.Naturals0
    assert B.success == 1
    assert B.failure == 0

    X = BernoulliProcess("X", p=Rational(1,3), success='H', failure='T')
    assert X.state_space == FiniteSet('H', 'T')
    H, T = symbols("H,T")
    assert E(X[1]+X[2]*X[3]) == H**2/9 + 4*H*T/9 + H/3 + 4*T**2/9 + 2*T/3

    t, x = symbols('t, x', positive=True, integer=True)
    assert isinstance(B[t], RandomIndexedSymbol)

    raises(ValueError, lambda: BernoulliProcess("X", p=1.1, success=1, failure=0))
    raises(NotImplementedError, lambda: B(t))

    raises(IndexError, lambda: B[-3])
    assert B.joint_distribution(B[3], B[9]) == JointDistributionHandmade(Lambda((B[3], B[9]),
                Piecewise((0.6, Eq(B[3], 1)), (0.4, Eq(B[3], 0)), (0, True))
                *Piecewise((0.6, Eq(B[9], 1)), (0.4, Eq(B[9], 0)), (0, True))))

    assert B.joint_distribution(2, B[4]) == JointDistributionHandmade(Lambda((B[2], B[4]),
                Piecewise((0.6, Eq(B[2], 1)), (0.4, Eq(B[2], 0)), (0, True))
                *Piecewise((0.6, Eq(B[4], 1)), (0.4, Eq(B[4], 0)), (0, True))))

    # Test for the sum distribution of Bernoulli Process RVs
    Y = B[1] + B[2] + B[3]
    assert P(Eq(Y, 0)).round(2) == Float(0.06, 1)
    assert P(Eq(Y, 2)).round(2) == Float(0.43, 2)
    assert P(Eq(Y, 4)).round(2) == 0
    assert P(Gt(Y, 1)).round(2) == Float(0.65, 2)
    # Test for independency of each Random Indexed variable
    assert P(Eq(B[1], 0) & Eq(B[2], 1) & Eq(B[3], 0) & Eq(B[4], 1)).round(2) == Float(0.06, 1)

    assert E(2 * B[1] + B[2]).round(2) == Float(1.80, 3)
    assert E(2 * B[1] + B[2] + 5).round(2) == Float(6.80, 3)
    assert E(B[2] * B[4] + B[10]).round(2) == Float(0.96, 2)
    assert E(B[2] > 0, Eq(B[1],1) & Eq(B[2],1)).round(2) == Float(0.60,2)
    assert E(B[1]) == 0.6
    assert P(B[1] > 0).round(2) == Float(0.60, 2)
    assert P(B[1] < 1).round(2) == Float(0.40, 2)
    assert P(B[1] > 0, B[2] <= 1).round(2) == Float(0.60, 2)
    assert P(B[12] * B[5] > 0).round(2) == Float(0.36, 2)
    assert P(B[12] * B[5] > 0, B[4] < 1).round(2) == Float(0.36, 2)
    assert P(Eq(B[2], 1), B[2] > 0) == 1.0
    assert P(Eq(B[5], 3)) == 0
    assert P(Eq(B[1], 1), B[1] < 0) == 0
    assert P(B[2] > 0, Eq(B[2], 1)) == 1
    assert P(B[2] < 0, Eq(B[2], 1)) == 0
    assert P(B[2] > 0, B[2]==7) == 0
    assert P(B[5] > 0, B[5]) == BernoulliDistribution(0.6, 0, 1)
    raises(ValueError, lambda: P(3))
    raises(ValueError, lambda: P(B[3] > 0, 3))

    # test issue 19456
    expr = Sum(B[t], (t, 0, 4))
    expr2 = Sum(B[t], (t, 1, 3))
    expr3 = Sum(B[t]**2, (t, 1, 3))
    assert expr.doit() == B[0] + B[1] + B[2] + B[3] + B[4]
    assert expr2.doit() == Y
    assert expr3.doit() == B[1]**2 + B[2]**2 + B[3]**2
    assert B[2*t].free_symbols == {B[2*t], t}
    assert B[4].free_symbols == {B[4]}
    assert B[x*t].free_symbols == {B[x*t], x, t}

    #test issue 20078
    assert (2*B[t] + 3*B[t]).simplify() == 5*B[t]
    assert (2*B[t] - 3*B[t]).simplify() == -B[t]
    assert (2*(0.25*B[t])).simplify() == 0.5*B[t]
    assert (2*B[t] * 0.25*B[t]).simplify() == 0.5*B[t]**2
    assert (B[t]**2 + B[t]**3).simplify() == (B[t] + 1)*B[t]**2

def test_PoissonProcess():
    X = PoissonProcess("X", 3)
    assert X.state_space == S.Naturals0
    assert X.index_set == Interval(0, oo)
    assert X.lamda == 3

    t, d, x, y = symbols('t d x y', positive=True)
    assert isinstance(X(t), RandomIndexedSymbol)
    assert X.distribution(t) == PoissonDistribution(3*t)
    with warns_deprecated_sympy():
        X.distribution(X(t))
    raises(ValueError, lambda: PoissonProcess("X", -1))
    raises(NotImplementedError, lambda: X[t])
    raises(IndexError, lambda: X(-5))

    assert X.joint_distribution(X(2), X(3)) == JointDistributionHandmade(Lambda((X(2), X(3)),
                6**X(2)*9**X(3)*exp(-15)/(factorial(X(2))*factorial(X(3)))))

    assert X.joint_distribution(4, 6) == JointDistributionHandmade(Lambda((X(4), X(6)),
                12**X(4)*18**X(6)*exp(-30)/(factorial(X(4))*factorial(X(6)))))

    assert P(X(t) < 1) == exp(-3*t)
    assert P(Eq(X(t), 0), Contains(t, Interval.Lopen(3, 5))) == exp(-6) # exp(-2*lamda)
    res = P(Eq(X(t), 1), Contains(t, Interval.Lopen(3, 4)))
    assert res == 3*exp(-3)

    # Equivalent to P(Eq(X(t), 1))**4 because of non-overlapping intervals
    assert P(Eq(X(t), 1) & Eq(X(d), 1) & Eq(X(x), 1) & Eq(X(y), 1), Contains(t, Interval.Lopen(0, 1))
    & Contains(d, Interval.Lopen(1, 2)) & Contains(x, Interval.Lopen(2, 3))
    & Contains(y, Interval.Lopen(3, 4))) == res**4

    # Return Probability because of overlapping intervals
    assert P(Eq(X(t), 2) & Eq(X(d), 3), Contains(t, Interval.Lopen(0, 2))
    & Contains(d, Interval.Ropen(2, 4))) == \
                Probability(Eq(X(d), 3) & Eq(X(t), 2), Contains(t, Interval.Lopen(0, 2))
                & Contains(d, Interval.Ropen(2, 4)))

    raises(ValueError, lambda: P(Eq(X(t), 2) & Eq(X(d), 3),
    Contains(t, Interval.Lopen(0, 4)) & Contains(d, Interval.Lopen(3, oo)))) # no bound on d
    assert P(Eq(X(3), 2)) == 81*exp(-9)/2
    assert P(Eq(X(t), 2), Contains(t, Interval.Lopen(0, 5))) == 225*exp(-15)/2

    # Check that probability works correctly by adding it to 1
    res1 = P(X(t) <= 3, Contains(t, Interval.Lopen(0, 5)))
    res2 = P(X(t) > 3, Contains(t, Interval.Lopen(0, 5)))
    assert res1 == 691*exp(-15)
    assert (res1 + res2).simplify() == 1

    # Check Not and  Or
    assert P(Not(Eq(X(t), 2) & (X(d) > 3)), Contains(t, Interval.Ropen(2, 4)) & \
            Contains(d, Interval.Lopen(7, 8))).simplify() == -18*exp(-6) + 234*exp(-9) + 1
    assert P(Eq(X(t), 2) | Ne(X(t), 4), Contains(t, Interval.Ropen(2, 4))) == 1 - 36*exp(-6)
    raises(ValueError, lambda: P(X(t) > 2, X(t) + X(d)))
    assert E(X(t)) == 3*t  # property of the distribution at a given timestamp
    assert E(X(t)**2 + X(d)*2 + X(y)**3, Contains(t, Interval.Lopen(0, 1))
        & Contains(d, Interval.Lopen(1, 2)) & Contains(y, Interval.Ropen(3, 4))) == 75
    assert E(X(t)**2, Contains(t, Interval.Lopen(0, 1))) == 12
    assert E(x*(X(t) + X(d))*(X(t)**2+X(d)**2), Contains(t, Interval.Lopen(0, 1))
    & Contains(d, Interval.Ropen(1, 2))) == \
            Expectation(x*(X(d) + X(t))*(X(d)**2 + X(t)**2), Contains(t, Interval.Lopen(0, 1))
            & Contains(d, Interval.Ropen(1, 2)))

    # Value Error because of infinite time bound
    raises(ValueError, lambda: E(X(t)**3, Contains(t, Interval.Lopen(1, oo))))

    # Equivalent to E(X(t)**2) - E(X(d)**2) == E(X(1)**2) - E(X(1)**2) == 0
    assert E((X(t) + X(d))*(X(t) - X(d)), Contains(t, Interval.Lopen(0, 1))
        & Contains(d, Interval.Lopen(1, 2))) == 0
    assert E(X(2) + x*E(X(5))) == 15*x + 6
    assert E(x*X(1) + y) == 3*x + y
    assert P(Eq(X(1), 2) & Eq(X(t), 3), Contains(t, Interval.Lopen(1, 2))) == 81*exp(-6)/4
    Y = PoissonProcess("Y", 6)
    Z = X + Y
    assert Z.lamda == X.lamda + Y.lamda == 9
    raises(ValueError, lambda: X + 5) # should be added be only PoissonProcess instance
    N, M = Z.split(4, 5)
    assert N.lamda == 4
    assert M.lamda == 5
    raises(ValueError, lambda: Z.split(3, 2)) # 2+3 != 9

    raises(ValueError, lambda :P(Eq(X(t), 0), Contains(t, Interval.Lopen(1, 3)) & Eq(X(1), 0)))
    # check if it handles queries with two random variables in one args
    res1 = P(Eq(N(3), N(5)))
    assert res1 == P(Eq(N(t), 0), Contains(t, Interval(3, 5)))
    res2 = P(N(3) > N(1))
    assert res2 == P((N(t) > 0), Contains(t, Interval(1, 3)))
    assert P(N(3) < N(1)) == 0 # condition is not possible
    res3 = P(N(3) <= N(1)) # holds only for Eq(N(3), N(1))
    assert res3 == P(Eq(N(t), 0), Contains(t, Interval(1, 3)))

    # tests from https://www.probabilitycourse.com/chapter11/11_1_2_basic_concepts_of_the_poisson_process.php
    X = PoissonProcess('X', 10) # 11.1
    assert P(Eq(X(S(1)/3), 3) & Eq(X(1), 10)) == exp(-10)*Rational(8000000000, 11160261)
    assert P(Eq(X(1), 1), Eq(X(S(1)/3), 3)) == 0
    assert P(Eq(X(1), 10), Eq(X(S(1)/3), 3)) == P(Eq(X(S(2)/3), 7))

    X = PoissonProcess('X', 2) # 11.2
    assert P(X(S(1)/2) < 1) == exp(-1)
    assert P(X(3) < 1, Eq(X(1), 0)) == exp(-4)
    assert P(Eq(X(4), 3), Eq(X(2), 3)) == exp(-4)

    X = PoissonProcess('X', 3)
    assert P(Eq(X(2), 5) & Eq(X(1), 2)) == Rational(81, 4)*exp(-6)

    # check few properties
    assert P(X(2) <= 3, X(1)>=1) == 3*P(Eq(X(1), 0)) + 2*P(Eq(X(1), 1)) + P(Eq(X(1), 2))
    assert P(X(2) <= 3, X(1) > 1) == 2*P(Eq(X(1), 0)) + 1*P(Eq(X(1), 1))
    assert P(Eq(X(2), 5) & Eq(X(1), 2)) == P(Eq(X(1), 3))*P(Eq(X(1), 2))
    assert P(Eq(X(3), 4), Eq(X(1), 3)) == P(Eq(X(2), 1))

    #test issue 20078
    assert (2*X(t) + 3*X(t)).simplify() == 5*X(t)
    assert (2*X(t) - 3*X(t)).simplify() == -X(t)
    assert (2*(0.25*X(t))).simplify() == 0.5*X(t)
    assert (2*X(t) * 0.25*X(t)).simplify() == 0.5*X(t)**2
    assert (X(t)**2 + X(t)**3).simplify() == (X(t) + 1)*X(t)**2

def test_WienerProcess():
    X = WienerProcess("X")
    assert X.state_space == S.Reals
    assert X.index_set == Interval(0, oo)

    t, d, x, y = symbols('t d x y', positive=True)
    assert isinstance(X(t), RandomIndexedSymbol)
    assert X.distribution(t) == NormalDistribution(0, sqrt(t))
    with warns_deprecated_sympy():
        X.distribution(X(t))
    raises(ValueError, lambda: PoissonProcess("X", -1))
    raises(NotImplementedError, lambda: X[t])
    raises(IndexError, lambda: X(-2))

    assert X.joint_distribution(X(2), X(3)) == JointDistributionHandmade(
        Lambda((X(2), X(3)), sqrt(6)*exp(-X(2)**2/4)*exp(-X(3)**2/6)/(12*pi)))
    assert X.joint_distribution(4, 6) == JointDistributionHandmade(
        Lambda((X(4), X(6)), sqrt(6)*exp(-X(4)**2/8)*exp(-X(6)**2/12)/(24*pi)))

    assert P(X(t) < 3).simplify() == erf(3*sqrt(2)/(2*sqrt(t)))/2 + S(1)/2
    assert P(X(t) > 2, Contains(t, Interval.Lopen(3, 7))).simplify() == S(1)/2 -\
                erf(sqrt(2)/2)/2

    # Equivalent to P(X(1)>1)**4
    assert P((X(t) > 4) & (X(d) > 3) & (X(x) > 2) & (X(y) > 1),
        Contains(t, Interval.Lopen(0, 1)) & Contains(d, Interval.Lopen(1, 2))
        & Contains(x, Interval.Lopen(2, 3)) & Contains(y, Interval.Lopen(3, 4))).simplify() ==\
        (1 - erf(sqrt(2)/2))*(1 - erf(sqrt(2)))*(1 - erf(3*sqrt(2)/2))*(1 - erf(2*sqrt(2)))/16

    # Contains an overlapping interval so, return Probability
    assert P((X(t)< 2) & (X(d)> 3), Contains(t, Interval.Lopen(0, 2))
        & Contains(d, Interval.Ropen(2, 4))) == Probability((X(d) > 3) & (X(t) < 2),
        Contains(d, Interval.Ropen(2, 4)) & Contains(t, Interval.Lopen(0, 2)))

    assert str(P(Not((X(t) < 5) & (X(d) > 3)), Contains(t, Interval.Ropen(2, 4)) &
        Contains(d, Interval.Lopen(7, 8))).simplify()) == \
                '-(1 - erf(3*sqrt(2)/2))*(2 - erfc(5/2))/4 + 1'
    # Distribution has mean 0 at each timestamp
    assert E(X(t)) == 0
    assert E(x*(X(t) + X(d))*(X(t)**2+X(d)**2), Contains(t, Interval.Lopen(0, 1))
    & Contains(d, Interval.Ropen(1, 2))) == Expectation(x*(X(d) + X(t))*(X(d)**2 + X(t)**2),
    Contains(d, Interval.Ropen(1, 2)) & Contains(t, Interval.Lopen(0, 1)))
    assert E(X(t) + x*E(X(3))) == 0

    #test issue 20078
    assert (2*X(t) + 3*X(t)).simplify() == 5*X(t)
    assert (2*X(t) - 3*X(t)).simplify() == -X(t)
    assert (2*(0.25*X(t))).simplify() == 0.5*X(t)
    assert (2*X(t) * 0.25*X(t)).simplify() == 0.5*X(t)**2
    assert (X(t)**2 + X(t)**3).simplify() == (X(t) + 1)*X(t)**2


def test_GammaProcess_symbolic():
    t, d, x, y, g, l = symbols('t d x y g l', positive=True)
    X = GammaProcess("X", l, g)

    raises(NotImplementedError, lambda: X[t])
    raises(IndexError, lambda: X(-1))
    assert isinstance(X(t), RandomIndexedSymbol)
    assert X.state_space == Interval(0, oo)
    assert X.distribution(t) == GammaDistribution(g*t, 1/l)
    with warns_deprecated_sympy():
        X.distribution(X(t))
    assert X.joint_distribution(5, X(3)) == JointDistributionHandmade(Lambda(
        (X(5), X(3)), l**(8*g)*exp(-l*X(3))*exp(-l*X(5))*X(3)**(3*g - 1)*X(5)**(5*g
        - 1)/(gamma(3*g)*gamma(5*g))))
    # property of the gamma process at any given timestamp
    assert E(X(t)) == g*t/l
    assert variance(X(t)).simplify() == g*t/l**2

    # Equivalent to E(2*X(1)) + E(X(1)**2) + E(X(1)**3), where E(X(1)) == g/l
    assert E(X(t)**2 + X(d)*2 + X(y)**3, Contains(t, Interval.Lopen(0, 1))
        & Contains(d, Interval.Lopen(1, 2)) & Contains(y, Interval.Ropen(3, 4))) == \
            2*g/l + (g**2 + g)/l**2 + (g**3 + 3*g**2 + 2*g)/l**3

    assert P(X(t) > 3, Contains(t, Interval.Lopen(3, 4))).simplify() == \
                                1 - lowergamma(g, 3*l)/gamma(g) # equivalent to P(X(1)>3)


    #test issue 20078
    assert (2*X(t) + 3*X(t)).simplify() == 5*X(t)
    assert (2*X(t) - 3*X(t)).simplify() == -X(t)
    assert (2*(0.25*X(t))).simplify() == 0.5*X(t)
    assert (2*X(t) * 0.25*X(t)).simplify() == 0.5*X(t)**2
    assert (X(t)**2 + X(t)**3).simplify() == (X(t) + 1)*X(t)**2
def test_GammaProcess_numeric():
    t, d, x, y = symbols('t d x y', positive=True)
    X = GammaProcess("X", 1, 2)
    assert X.state_space == Interval(0, oo)
    assert X.index_set == Interval(0, oo)
    assert X.lamda == 1
    assert X.gamma == 2

    raises(ValueError, lambda: GammaProcess("X", -1, 2))
    raises(ValueError, lambda: GammaProcess("X", 0, -2))
    raises(ValueError, lambda: GammaProcess("X", -1, -2))

    # all are independent because of non-overlapping intervals
    assert P((X(t) > 4) & (X(d) > 3) & (X(x) > 2) & (X(y) > 1), Contains(t,
        Interval.Lopen(0, 1)) & Contains(d, Interval.Lopen(1, 2)) & Contains(x,
        Interval.Lopen(2, 3)) & Contains(y, Interval.Lopen(3, 4))).simplify() == \
                                                            120*exp(-10)

    # Check working with Not and Or
    assert P(Not((X(t) < 5) & (X(d) > 3)), Contains(t, Interval.Ropen(2, 4)) &
        Contains(d, Interval.Lopen(7, 8))).simplify() == -4*exp(-3) + 472*exp(-8)/3 + 1
    assert P((X(t) > 2) | (X(t) < 4), Contains(t, Interval.Ropen(1, 4))).simplify() == \
                                            -643*exp(-4)/15 + 109*exp(-2)/15 + 1

    assert E(X(t)) == 2*t # E(X(t)) == gamma*t/l
    assert E(X(2) + x*E(X(5))) == 10*x + 4
