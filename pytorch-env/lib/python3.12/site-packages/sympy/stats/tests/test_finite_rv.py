from sympy.concrete.summations import Sum
from sympy.core.containers import (Dict, Tuple)
from sympy.core.function import Function
from sympy.core.numbers import (I, Rational, nan)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.combinatorial.numbers import harmonic
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import cos
from sympy.functions.special.beta_functions import beta
from sympy.logic.boolalg import (And, Or)
from sympy.polys.polytools import cancel
from sympy.sets.sets import FiniteSet
from sympy.simplify.simplify import simplify
from sympy.matrices import Matrix
from sympy.stats import (DiscreteUniform, Die, Bernoulli, Coin, Binomial, BetaBinomial,
                         Hypergeometric, Rademacher, IdealSoliton, RobustSoliton, P, E, variance,
                         covariance, skewness, density, where, FiniteRV, pspace, cdf,
                         correlation, moment, cmoment, smoment, characteristic_function,
                         moment_generating_function, quantile,  kurtosis, median, coskewness)
from sympy.stats.frv_types import DieDistribution, BinomialDistribution, \
    HypergeometricDistribution
from sympy.stats.rv import Density
from sympy.testing.pytest import raises


def BayesTest(A, B):
    assert P(A, B) == P(And(A, B)) / P(B)
    assert P(A, B) == P(B, A) * P(A) / P(B)


def test_discreteuniform():
    # Symbolic
    a, b, c, t = symbols('a b c t')
    X = DiscreteUniform('X', [a, b, c])

    assert E(X) == (a + b + c)/3
    assert simplify(variance(X)
                    - ((a**2 + b**2 + c**2)/3 - (a/3 + b/3 + c/3)**2)) == 0
    assert P(Eq(X, a)) == P(Eq(X, b)) == P(Eq(X, c)) == S('1/3')

    Y = DiscreteUniform('Y', range(-5, 5))

    # Numeric
    assert E(Y) == S('-1/2')
    assert variance(Y) == S('33/4')
    assert median(Y) == FiniteSet(-1, 0)

    for x in range(-5, 5):
        assert P(Eq(Y, x)) == S('1/10')
        assert P(Y <= x) == S(x + 6)/10
        assert P(Y >= x) == S(5 - x)/10

    assert dict(density(Die('D', 6)).items()) == \
           dict(density(DiscreteUniform('U', range(1, 7))).items())

    assert characteristic_function(X)(t) == exp(I*a*t)/3 + exp(I*b*t)/3 + exp(I*c*t)/3
    assert moment_generating_function(X)(t) == exp(a*t)/3 + exp(b*t)/3 + exp(c*t)/3
    # issue 18611
    raises(ValueError, lambda: DiscreteUniform('Z', [a, a, a, b, b, c]))

def test_dice():
    # TODO: Make iid method!
    X, Y, Z = Die('X', 6), Die('Y', 6), Die('Z', 6)
    a, b, t, p = symbols('a b t p')

    assert E(X) == 3 + S.Half
    assert variance(X) == Rational(35, 12)
    assert E(X + Y) == 7
    assert E(X + X) == 7
    assert E(a*X + b) == a*E(X) + b
    assert variance(X + Y) == variance(X) + variance(Y) == cmoment(X + Y, 2)
    assert variance(X + X) == 4 * variance(X) == cmoment(X + X, 2)
    assert cmoment(X, 0) == 1
    assert cmoment(4*X, 3) == 64*cmoment(X, 3)
    assert covariance(X, Y) is S.Zero
    assert covariance(X, X + Y) == variance(X)
    assert density(Eq(cos(X*S.Pi), 1))[True] == S.Half
    assert correlation(X, Y) == 0
    assert correlation(X, Y) == correlation(Y, X)
    assert smoment(X + Y, 3) == skewness(X + Y)
    assert smoment(X + Y, 4) == kurtosis(X + Y)
    assert smoment(X, 0) == 1
    assert P(X > 3) == S.Half
    assert P(2*X > 6) == S.Half
    assert P(X > Y) == Rational(5, 12)
    assert P(Eq(X, Y)) == P(Eq(X, 1))

    assert E(X, X > 3) == 5 == moment(X, 1, 0, X > 3)
    assert E(X, Y > 3) == E(X) == moment(X, 1, 0, Y > 3)
    assert E(X + Y, Eq(X, Y)) == E(2*X)
    assert moment(X, 0) == 1
    assert moment(5*X, 2) == 25*moment(X, 2)
    assert quantile(X)(p) == Piecewise((nan, (p > 1) | (p < 0)),\
        (S.One, p <= Rational(1, 6)), (S(2), p <= Rational(1, 3)), (S(3), p <= S.Half),\
        (S(4), p <= Rational(2, 3)), (S(5), p <= Rational(5, 6)), (S(6), p <= 1))

    assert P(X > 3, X > 3) is S.One
    assert P(X > Y, Eq(Y, 6)) is S.Zero
    assert P(Eq(X + Y, 12)) == Rational(1, 36)
    assert P(Eq(X + Y, 12), Eq(X, 6)) == Rational(1, 6)

    assert density(X + Y) == density(Y + Z) != density(X + X)
    d = density(2*X + Y**Z)
    assert d[S(22)] == Rational(1, 108) and d[S(4100)] == Rational(1, 216) and S(3130) not in d

    assert pspace(X).domain.as_boolean() == Or(
        *[Eq(X.symbol, i) for i in [1, 2, 3, 4, 5, 6]])

    assert where(X > 3).set == FiniteSet(4, 5, 6)

    assert characteristic_function(X)(t) == exp(6*I*t)/6 + exp(5*I*t)/6 + exp(4*I*t)/6 + exp(3*I*t)/6 + exp(2*I*t)/6 + exp(I*t)/6
    assert moment_generating_function(X)(t) == exp(6*t)/6 + exp(5*t)/6 + exp(4*t)/6 + exp(3*t)/6 + exp(2*t)/6 + exp(t)/6
    assert median(X) == FiniteSet(3, 4)
    D = Die('D', 7)
    assert median(D) == FiniteSet(4)
    # Bayes test for die
    BayesTest(X > 3, X + Y < 5)
    BayesTest(Eq(X - Y, Z), Z > Y)
    BayesTest(X > 3, X > 2)

    # arg test for die
    raises(ValueError, lambda: Die('X', -1))  # issue 8105: negative sides.
    raises(ValueError, lambda: Die('X', 0))
    raises(ValueError, lambda: Die('X', 1.5))  # issue 8103: non integer sides.

    # symbolic test for die
    n, k = symbols('n, k', positive=True)
    D = Die('D', n)
    dens = density(D).dict
    assert dens == Density(DieDistribution(n))
    assert set(dens.subs(n, 4).doit().keys()) == {1, 2, 3, 4}
    assert set(dens.subs(n, 4).doit().values()) == {Rational(1, 4)}
    k = Dummy('k', integer=True)
    assert E(D).dummy_eq(
        Sum(Piecewise((k/n, k <= n), (0, True)), (k, 1, n)))
    assert variance(D).subs(n, 6).doit() == Rational(35, 12)

    ki = Dummy('ki')
    cumuf = cdf(D)(k)
    assert cumuf.dummy_eq(
    Sum(Piecewise((1/n, (ki >= 1) & (ki <= n)), (0, True)), (ki, 1, k)))
    assert cumuf.subs({n: 6, k: 2}).doit() == Rational(1, 3)

    t = Dummy('t')
    cf = characteristic_function(D)(t)
    assert cf.dummy_eq(
    Sum(Piecewise((exp(ki*I*t)/n, (ki >= 1) & (ki <= n)), (0, True)), (ki, 1, n)))
    assert cf.subs(n, 3).doit() == exp(3*I*t)/3 + exp(2*I*t)/3 + exp(I*t)/3
    mgf = moment_generating_function(D)(t)
    assert mgf.dummy_eq(
    Sum(Piecewise((exp(ki*t)/n, (ki >= 1) & (ki <= n)), (0, True)), (ki, 1, n)))
    assert mgf.subs(n, 3).doit() == exp(3*t)/3 + exp(2*t)/3 + exp(t)/3

def test_given():
    X = Die('X', 6)
    assert density(X, X > 5) == {S(6): S.One}
    assert where(X > 2, X > 5).as_boolean() == Eq(X.symbol, 6)


def test_domains():
    X, Y = Die('x', 6), Die('y', 6)
    x, y = X.symbol, Y.symbol
    # Domains
    d = where(X > Y)
    assert d.condition == (x > y)
    d = where(And(X > Y, Y > 3))
    assert d.as_boolean() == Or(And(Eq(x, 5), Eq(y, 4)), And(Eq(x, 6),
        Eq(y, 5)), And(Eq(x, 6), Eq(y, 4)))
    assert len(d.elements) == 3

    assert len(pspace(X + Y).domain.elements) == 36

    Z = Die('x', 4)

    raises(ValueError, lambda: P(X > Z))  # Two domains with same internal symbol

    assert pspace(X + Y).domain.set == FiniteSet(1, 2, 3, 4, 5, 6)**2

    assert where(X > 3).set == FiniteSet(4, 5, 6)
    assert X.pspace.domain.dict == FiniteSet(
        *[Dict({X.symbol: i}) for i in range(1, 7)])

    assert where(X > Y).dict == FiniteSet(*[Dict({X.symbol: i, Y.symbol: j})
            for i in range(1, 7) for j in range(1, 7) if i > j])

def test_bernoulli():
    p, a, b, t = symbols('p a b t')
    X = Bernoulli('B', p, a, b)

    assert E(X) == a*p + b*(-p + 1)
    assert density(X)[a] == p
    assert density(X)[b] == 1 - p
    assert characteristic_function(X)(t) == p * exp(I * a * t) + (-p + 1) * exp(I * b * t)
    assert moment_generating_function(X)(t) == p * exp(a * t) + (-p + 1) * exp(b * t)

    X = Bernoulli('B', p, 1, 0)
    z = Symbol("z")

    assert E(X) == p
    assert simplify(variance(X)) == p*(1 - p)
    assert E(a*X + b) == a*E(X) + b
    assert simplify(variance(a*X + b)) == simplify(a**2 * variance(X))
    assert quantile(X)(z) == Piecewise((nan, (z > 1) | (z < 0)), (0, z <= 1 - p), (1, z <= 1))
    Y = Bernoulli('Y', Rational(1, 2))
    assert median(Y) == FiniteSet(0, 1)
    Z = Bernoulli('Z', Rational(2, 3))
    assert median(Z) == FiniteSet(1)
    raises(ValueError, lambda: Bernoulli('B', 1.5))
    raises(ValueError, lambda: Bernoulli('B', -0.5))

    #issue 8248
    assert X.pspace.compute_expectation(1) == 1

    p = Rational(1, 5)
    X = Binomial('X', 5, p)
    Y = Binomial('Y', 7, 2*p)
    Z = Binomial('Z', 9, 3*p)
    assert coskewness(Y + Z, X + Y, X + Z).simplify() == 0
    assert coskewness(Y + 2*X + Z, X + 2*Y + Z, X + 2*Z + Y).simplify() == \
                        sqrt(1529)*Rational(12, 16819)
    assert coskewness(Y + 2*X + Z, X + 2*Y + Z, X + 2*Z + Y, X < 2).simplify() \
                        == -sqrt(357451121)*Rational(2812, 4646864573)

def test_cdf():
    D = Die('D', 6)
    o = S.One

    assert cdf(
        D) == sympify({1: o/6, 2: o/3, 3: o/2, 4: 2*o/3, 5: 5*o/6, 6: o})


def test_coins():
    C, D = Coin('C'), Coin('D')
    H, T = symbols('H, T')
    assert P(Eq(C, D)) == S.Half
    assert density(Tuple(C, D)) == {(H, H): Rational(1, 4), (H, T): Rational(1, 4),
            (T, H): Rational(1, 4), (T, T): Rational(1, 4)}
    assert dict(density(C).items()) == {H: S.Half, T: S.Half}

    F = Coin('F', Rational(1, 10))
    assert P(Eq(F, H)) == Rational(1, 10)

    d = pspace(C).domain

    assert d.as_boolean() == Or(Eq(C.symbol, H), Eq(C.symbol, T))

    raises(ValueError, lambda: P(C > D))  # Can't intelligently compare H to T

def test_binomial_verify_parameters():
    raises(ValueError, lambda: Binomial('b', .2, .5))
    raises(ValueError, lambda: Binomial('b', 3, 1.5))

def test_binomial_numeric():
    nvals = range(5)
    pvals = [0, Rational(1, 4), S.Half, Rational(3, 4), 1]

    for n in nvals:
        for p in pvals:
            X = Binomial('X', n, p)
            assert E(X) == n*p
            assert variance(X) == n*p*(1 - p)
            if n > 0 and 0 < p < 1:
                assert skewness(X) == (1 - 2*p)/sqrt(n*p*(1 - p))
                assert kurtosis(X) == 3 + (1 - 6*p*(1 - p))/(n*p*(1 - p))
            for k in range(n + 1):
                assert P(Eq(X, k)) == binomial(n, k)*p**k*(1 - p)**(n - k)

def test_binomial_quantile():
    X = Binomial('X', 50, S.Half)
    assert quantile(X)(0.95) == S(31)
    assert median(X) == FiniteSet(25)

    X = Binomial('X', 5, S.Half)
    p = Symbol("p", positive=True)
    assert quantile(X)(p) == Piecewise((nan, p > S.One), (S.Zero, p <= Rational(1, 32)),\
        (S.One, p <= Rational(3, 16)), (S(2), p <= S.Half), (S(3), p <= Rational(13, 16)),\
        (S(4), p <= Rational(31, 32)), (S(5), p <= S.One))
    assert median(X) == FiniteSet(2, 3)


def test_binomial_symbolic():
    n = 2
    p = symbols('p', positive=True)
    X = Binomial('X', n, p)
    t = Symbol('t')

    assert simplify(E(X)) == n*p == simplify(moment(X, 1))
    assert simplify(variance(X)) == n*p*(1 - p) == simplify(cmoment(X, 2))
    assert cancel(skewness(X) - (1 - 2*p)/sqrt(n*p*(1 - p))) == 0
    assert cancel((kurtosis(X)) - (3 + (1 - 6*p*(1 - p))/(n*p*(1 - p)))) == 0
    assert characteristic_function(X)(t) == p ** 2 * exp(2 * I * t) + 2 * p * (-p + 1) * exp(I * t) + (-p + 1) ** 2
    assert moment_generating_function(X)(t) == p ** 2 * exp(2 * t) + 2 * p * (-p + 1) * exp(t) + (-p + 1) ** 2

    # Test ability to change success/failure winnings
    H, T = symbols('H T')
    Y = Binomial('Y', n, p, succ=H, fail=T)
    assert simplify(E(Y) - (n*(H*p + T*(1 - p)))) == 0

    # test symbolic dimensions
    n = symbols('n')
    B = Binomial('B', n, p)
    raises(NotImplementedError, lambda: P(B > 2))
    assert density(B).dict == Density(BinomialDistribution(n, p, 1, 0))
    assert set(density(B).dict.subs(n, 4).doit().keys()) == \
    {S.Zero, S.One, S(2), S(3), S(4)}
    assert set(density(B).dict.subs(n, 4).doit().values()) == \
    {(1 - p)**4, 4*p*(1 - p)**3, 6*p**2*(1 - p)**2, 4*p**3*(1 - p), p**4}
    k = Dummy('k', integer=True)
    assert E(B > 2).dummy_eq(
        Sum(Piecewise((k*p**k*(1 - p)**(-k + n)*binomial(n, k), (k >= 0)
        & (k <= n) & (k > 2)), (0, True)), (k, 0, n)))

def test_beta_binomial():
    # verify parameters
    raises(ValueError, lambda: BetaBinomial('b', .2, 1, 2))
    raises(ValueError, lambda: BetaBinomial('b', 2, -1, 2))
    raises(ValueError, lambda: BetaBinomial('b', 2, 1, -2))
    assert BetaBinomial('b', 2, 1, 1)

    # test numeric values
    nvals = range(1,5)
    alphavals = [Rational(1, 4), S.Half, Rational(3, 4), 1, 10]
    betavals = [Rational(1, 4), S.Half, Rational(3, 4), 1, 10]

    for n in nvals:
        for a in alphavals:
            for b in betavals:
                X = BetaBinomial('X', n, a, b)
                assert E(X) == moment(X, 1)
                assert variance(X) == cmoment(X, 2)

    # test symbolic
    n, a, b = symbols('a b n')
    assert BetaBinomial('x', n, a, b)
    n = 2 # Because we're using for loops, can't do symbolic n
    a, b = symbols('a b', positive=True)
    X = BetaBinomial('X', n, a, b)
    t = Symbol('t')

    assert E(X).expand() == moment(X, 1).expand()
    assert variance(X).expand() == cmoment(X, 2).expand()
    assert skewness(X) == smoment(X, 3)
    assert characteristic_function(X)(t) == exp(2*I*t)*beta(a + 2, b)/beta(a, b) +\
         2*exp(I*t)*beta(a + 1, b + 1)/beta(a, b) + beta(a, b + 2)/beta(a, b)
    assert moment_generating_function(X)(t) == exp(2*t)*beta(a + 2, b)/beta(a, b) +\
         2*exp(t)*beta(a + 1, b + 1)/beta(a, b) + beta(a, b + 2)/beta(a, b)

def test_hypergeometric_numeric():
    for N in range(1, 5):
        for m in range(0, N + 1):
            for n in range(1, N + 1):
                X = Hypergeometric('X', N, m, n)
                N, m, n = map(sympify, (N, m, n))
                assert sum(density(X).values()) == 1
                assert E(X) == n * m / N
                if N > 1:
                    assert variance(X) == n*(m/N)*(N - m)/N*(N - n)/(N - 1)
                # Only test for skewness when defined
                if N > 2 and 0 < m < N and n < N:
                    assert skewness(X) == simplify((N - 2*m)*sqrt(N - 1)*(N - 2*n)
                        / (sqrt(n*m*(N - m)*(N - n))*(N - 2)))

def test_hypergeometric_symbolic():
    N, m, n = symbols('N, m, n')
    H = Hypergeometric('H', N, m, n)
    dens = density(H).dict
    expec = E(H > 2)
    assert dens == Density(HypergeometricDistribution(N, m, n))
    assert dens.subs(N, 5).doit() == Density(HypergeometricDistribution(5, m, n))
    assert set(dens.subs({N: 3, m: 2, n: 1}).doit().keys()) == {S.Zero, S.One}
    assert set(dens.subs({N: 3, m: 2, n: 1}).doit().values()) == {Rational(1, 3), Rational(2, 3)}
    k = Dummy('k', integer=True)
    assert expec.dummy_eq(
        Sum(Piecewise((k*binomial(m, k)*binomial(N - m, -k + n)
        /binomial(N, n), k > 2), (0, True)), (k, 0, n)))

def test_rademacher():
    X = Rademacher('X')
    t = Symbol('t')

    assert E(X) == 0
    assert variance(X) == 1
    assert density(X)[-1] == S.Half
    assert density(X)[1] == S.Half
    assert characteristic_function(X)(t) == exp(I*t)/2 + exp(-I*t)/2
    assert moment_generating_function(X)(t) == exp(t) / 2 + exp(-t) / 2

def test_ideal_soliton():
    raises(ValueError, lambda : IdealSoliton('sol', -12))
    raises(ValueError, lambda : IdealSoliton('sol', 13.2))
    raises(ValueError, lambda : IdealSoliton('sol', 0))
    f = Function('f')
    raises(ValueError, lambda : density(IdealSoliton('sol', 10)).pmf(f))

    k = Symbol('k', integer=True, positive=True)
    x = Symbol('x', integer=True, positive=True)
    t = Symbol('t')
    sol = IdealSoliton('sol', k)
    assert density(sol).low == S.One
    assert density(sol).high == k
    assert density(sol).dict == Density(density(sol))
    assert density(sol).pmf(x) == Piecewise((1/k, Eq(x, 1)), (1/(x*(x - 1)), k >= x), (0, True))

    k_vals = [5, 20, 50, 100, 1000]
    for i in k_vals:
        assert E(sol.subs(k, i)) == harmonic(i) == moment(sol.subs(k, i), 1)
        assert variance(sol.subs(k, i)) == (i - 1) + harmonic(i) - harmonic(i)**2 == cmoment(sol.subs(k, i),2)
        assert skewness(sol.subs(k, i)) == smoment(sol.subs(k, i), 3)
        assert kurtosis(sol.subs(k, i)) == smoment(sol.subs(k, i), 4)

    assert exp(I*t)/10 + Sum(exp(I*t*x)/(x*x - x), (x, 2, k)).subs(k, 10).doit() == characteristic_function(sol.subs(k, 10))(t)
    assert exp(t)/10 + Sum(exp(t*x)/(x*x - x), (x, 2, k)).subs(k, 10).doit() == moment_generating_function(sol.subs(k, 10))(t)

def test_robust_soliton():
    raises(ValueError, lambda : RobustSoliton('robSol', -12, 0.1, 0.02))
    raises(ValueError, lambda : RobustSoliton('robSol', 13, 1.89, 0.1))
    raises(ValueError, lambda : RobustSoliton('robSol', 15, 0.6, -2.31))
    f = Function('f')
    raises(ValueError, lambda : density(RobustSoliton('robSol', 15, 0.6, 0.1)).pmf(f))

    k = Symbol('k', integer=True, positive=True)
    delta = Symbol('delta', positive=True)
    c = Symbol('c', positive=True)
    robSol = RobustSoliton('robSol', k, delta, c)
    assert density(robSol).low == 1
    assert density(robSol).high == k

    k_vals = [10, 20, 50]
    delta_vals = [0.2, 0.4, 0.6]
    c_vals = [0.01, 0.03, 0.05]
    for x in k_vals:
        for y in delta_vals:
            for z in c_vals:
                assert E(robSol.subs({k: x, delta: y, c: z})) == moment(robSol.subs({k: x, delta: y, c: z}), 1)
                assert variance(robSol.subs({k: x, delta: y, c: z})) == cmoment(robSol.subs({k: x, delta: y, c: z}), 2)
                assert skewness(robSol.subs({k: x, delta: y, c: z})) == smoment(robSol.subs({k: x, delta: y, c: z}), 3)
                assert kurtosis(robSol.subs({k: x, delta: y, c: z})) == smoment(robSol.subs({k: x, delta: y, c: z}), 4)

def test_FiniteRV():
    F = FiniteRV('F', {1: S.Half, 2: Rational(1, 4), 3: Rational(1, 4)}, check=True)
    p = Symbol("p", positive=True)

    assert dict(density(F).items()) == {S.One: S.Half, S(2): Rational(1, 4), S(3): Rational(1, 4)}
    assert P(F >= 2) == S.Half
    assert quantile(F)(p) == Piecewise((nan, p > S.One), (S.One, p <= S.Half),\
        (S(2), p <= Rational(3, 4)),(S(3), True))

    assert pspace(F).domain.as_boolean() == Or(
        *[Eq(F.symbol, i) for i in [1, 2, 3]])

    assert F.pspace.domain.set == FiniteSet(1, 2, 3)
    raises(ValueError, lambda: FiniteRV('F', {1: S.Half, 2: S.Half, 3: S.Half}, check=True))
    raises(ValueError, lambda: FiniteRV('F', {1: S.Half, 2: Rational(-1, 2), 3: S.One}, check=True))
    raises(ValueError, lambda: FiniteRV('F', {1: S.One, 2: Rational(3, 2), 3: S.Zero,\
        4: Rational(-1, 2), 5: Rational(-3, 4), 6: Rational(-1, 4)}, check=True))

    # purposeful invalid pmf but it should not raise since check=False
    # see test_drv_types.test_ContinuousRV for explanation
    X = FiniteRV('X', {1: 1, 2: 2})
    assert E(X) == 5
    assert P(X <= 2) + P(X > 2) != 1

def test_density_call():
    from sympy.abc import p
    x = Bernoulli('x', p)
    d = density(x)
    assert d(0) == 1 - p
    assert d(S.Zero) == 1 - p
    assert d(5) == 0

    assert 0 in d
    assert 5 not in d
    assert d(S.Zero) == d[S.Zero]


def test_DieDistribution():
    from sympy.abc import x
    X = DieDistribution(6)
    assert X.pmf(S.Half) is S.Zero
    assert X.pmf(x).subs({x: 1}).doit() == Rational(1, 6)
    assert X.pmf(x).subs({x: 7}).doit() == 0
    assert X.pmf(x).subs({x: -1}).doit() == 0
    assert X.pmf(x).subs({x: Rational(1, 3)}).doit() == 0
    raises(ValueError, lambda: X.pmf(Matrix([0, 0])))
    raises(ValueError, lambda: X.pmf(x**2 - 1))

def test_FinitePSpace():
    X = Die('X', 6)
    space = pspace(X)
    assert space.density == DieDistribution(6)

def test_symbolic_conditions():
    B = Bernoulli('B', Rational(1, 4))
    D = Die('D', 4)
    b, n = symbols('b, n')
    Y = P(Eq(B, b))
    Z = E(D > n)
    assert Y == \
    Piecewise((Rational(1, 4), Eq(b, 1)), (0, True)) + \
    Piecewise((Rational(3, 4), Eq(b, 0)), (0, True))
    assert Z == \
    Piecewise((Rational(1, 4), n < 1), (0, True)) + Piecewise((S.Half, n < 2), (0, True)) + \
    Piecewise((Rational(3, 4), n < 3), (0, True)) + Piecewise((S.One, n < 4), (0, True))
