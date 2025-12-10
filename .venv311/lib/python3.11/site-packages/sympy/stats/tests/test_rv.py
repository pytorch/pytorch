from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.function import Lambda
from sympy.core.numbers import (Rational, nan, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.combinatorial.factorials import (FallingFactorial, binomial)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.delta_functions import DiracDelta
from sympy.integrals.integrals import integrate
from sympy.logic.boolalg import (And, Or)
from sympy.matrices.dense import Matrix
from sympy.sets.sets import Interval
from sympy.tensor.indexed import Indexed
from sympy.stats import (Die, Normal, Exponential, FiniteRV, P, E, H, variance,
        density, given, independent, dependent, where, pspace, GaussianUnitaryEnsemble,
        random_symbols, sample, Geometric, factorial_moment, Binomial, Hypergeometric,
        DiscreteUniform, Poisson, characteristic_function, moment_generating_function,
        BernoulliProcess, Variance, Expectation, Probability, Covariance, covariance, cmoment,
        moment, median)
from sympy.stats.rv import (IndependentProductPSpace, rs_swap, Density, NamedArgsMixin,
        RandomSymbol, sample_iter, PSpace, is_random, RandomIndexedSymbol, RandomMatrixSymbol)
from sympy.testing.pytest import raises, skip, XFAIL, warns_deprecated_sympy
from sympy.external import import_module
from sympy.core.numbers import comp
from sympy.stats.frv_types import BernoulliDistribution
from sympy.core.symbol import Dummy
from sympy.functions.elementary.piecewise import Piecewise

def test_where():
    X, Y = Die('X'), Die('Y')
    Z = Normal('Z', 0, 1)

    assert where(Z**2 <= 1).set == Interval(-1, 1)
    assert where(Z**2 <= 1).as_boolean() == Interval(-1, 1).as_relational(Z.symbol)
    assert where(And(X > Y, Y > 4)).as_boolean() == And(
        Eq(X.symbol, 6), Eq(Y.symbol, 5))

    assert len(where(X < 3).set) == 2
    assert 1 in where(X < 3).set

    X, Y = Normal('X', 0, 1), Normal('Y', 0, 1)
    assert where(And(X**2 <= 1, X >= 0)).set == Interval(0, 1)
    XX = given(X, And(X**2 <= 1, X >= 0))
    assert XX.pspace.domain.set == Interval(0, 1)
    assert XX.pspace.domain.as_boolean() == \
        And(0 <= X.symbol, X.symbol**2 <= 1, -oo < X.symbol, X.symbol < oo)

    with raises(TypeError):
        XX = given(X, X + 3)


def test_random_symbols():
    X, Y = Normal('X', 0, 1), Normal('Y', 0, 1)

    assert set(random_symbols(2*X + 1)) == {X}
    assert set(random_symbols(2*X + Y)) == {X, Y}
    assert set(random_symbols(2*X + Y.symbol)) == {X}
    assert set(random_symbols(2)) == set()


def test_characteristic_function():
    #  Imports I from sympy
    from sympy.core.numbers import I
    X = Normal('X',0,1)
    Y = DiscreteUniform('Y', [1,2,7])
    Z = Poisson('Z', 2)
    t = symbols('_t')
    P = Lambda(t, exp(-t**2/2))
    Q = Lambda(t, exp(7*t*I)/3 + exp(2*t*I)/3 + exp(t*I)/3)
    R = Lambda(t, exp(2 * exp(t*I) - 2))


    assert characteristic_function(X).dummy_eq(P)
    assert characteristic_function(Y).dummy_eq(Q)
    assert characteristic_function(Z).dummy_eq(R)


def test_moment_generating_function():

    X = Normal('X',0,1)
    Y = DiscreteUniform('Y', [1,2,7])
    Z = Poisson('Z', 2)
    t = symbols('_t')
    P = Lambda(t, exp(t**2/2))
    Q = Lambda(t, (exp(7*t)/3 + exp(2*t)/3 + exp(t)/3))
    R = Lambda(t, exp(2 * exp(t) - 2))


    assert moment_generating_function(X).dummy_eq(P)
    assert moment_generating_function(Y).dummy_eq(Q)
    assert moment_generating_function(Z).dummy_eq(R)

def test_sample_iter():

    X = Normal('X',0,1)
    Y = DiscreteUniform('Y', [1, 2, 7])
    Z = Poisson('Z', 2)

    scipy = import_module('scipy')
    if not scipy:
        skip('Scipy is not installed. Abort tests')
    expr = X**2 + 3
    iterator = sample_iter(expr)

    expr2 = Y**2 + 5*Y + 4
    iterator2 = sample_iter(expr2)

    expr3 = Z**3 + 4
    iterator3 = sample_iter(expr3)

    def is_iterator(obj):
        if (
            hasattr(obj, '__iter__') and
            (hasattr(obj, 'next') or
            hasattr(obj, '__next__')) and
            callable(obj.__iter__) and
            obj.__iter__() is obj
           ):
            return True
        else:
            return False
    assert is_iterator(iterator)
    assert is_iterator(iterator2)
    assert is_iterator(iterator3)

def test_pspace():
    X, Y = Normal('X', 0, 1), Normal('Y', 0, 1)
    x = Symbol('x')

    raises(ValueError, lambda: pspace(5 + 3))
    raises(ValueError, lambda: pspace(x < 1))
    assert pspace(X) == X.pspace
    assert pspace(2*X + 1) == X.pspace
    assert pspace(2*X + Y) == IndependentProductPSpace(Y.pspace, X.pspace)

def test_rs_swap():
    X = Normal('x', 0, 1)
    Y = Exponential('y', 1)

    XX = Normal('x', 0, 2)
    YY = Normal('y', 0, 3)

    expr = 2*X + Y
    assert expr.subs(rs_swap((X, Y), (YY, XX))) == 2*XX + YY


def test_RandomSymbol():

    X = Normal('x', 0, 1)
    Y = Normal('x', 0, 2)
    assert X.symbol == Y.symbol
    assert X != Y

    assert X.name == X.symbol.name

    X = Normal('lambda', 0, 1) # make sure we can use protected terms
    X = Normal('Lambda', 0, 1) # make sure we can use SymPy terms


def test_RandomSymbol_diff():
    X = Normal('x', 0, 1)
    assert (2*X).diff(X)


def test_random_symbol_no_pspace():
    x = RandomSymbol(Symbol('x'))
    assert x.pspace == PSpace()

def test_overlap():
    X = Normal('x', 0, 1)
    Y = Normal('x', 0, 2)

    raises(ValueError, lambda: P(X > Y))


def test_IndependentProductPSpace():
    X = Normal('X', 0, 1)
    Y = Normal('Y', 0, 1)
    px = X.pspace
    py = Y.pspace
    assert pspace(X + Y) == IndependentProductPSpace(px, py)
    assert pspace(X + Y) == IndependentProductPSpace(py, px)


def test_E():
    assert E(5) == 5


def test_H():
    X = Normal('X', 0, 1)
    D = Die('D', sides = 4)
    G = Geometric('G', 0.5)
    assert H(X, X > 0) == -log(2)/2 + S.Half + log(pi)/2
    assert H(D, D > 2) == log(2)
    assert comp(H(G).evalf().round(2), 1.39)


def test_Sample():
    X = Die('X', 6)
    Y = Normal('Y', 0, 1)
    z = Symbol('z', integer=True)

    scipy = import_module('scipy')
    if not scipy:
        skip('Scipy is not installed. Abort tests')
    assert sample(X) in [1, 2, 3, 4, 5, 6]
    assert isinstance(sample(X + Y), float)

    assert P(X + Y > 0, Y < 0, numsamples=10).is_number
    assert E(X + Y, numsamples=10).is_number
    assert E(X**2 + Y, numsamples=10).is_number
    assert E((X + Y)**2, numsamples=10).is_number
    assert variance(X + Y, numsamples=10).is_number

    raises(TypeError, lambda: P(Y > z, numsamples=5))

    assert P(sin(Y) <= 1, numsamples=10) == 1.0
    assert P(sin(Y) <= 1, cos(Y) < 1, numsamples=10) == 1.0

    assert all(i in range(1, 7) for i in density(X, numsamples=10))
    assert all(i in range(4, 7) for i in density(X, X>3, numsamples=10))

    numpy = import_module('numpy')
    if not numpy:
        skip('Numpy is not installed. Abort tests')
    #Test Issue #21563: Output of sample must be a float or array
    assert isinstance(sample(X), (numpy.int32, numpy.int64))
    assert isinstance(sample(Y), numpy.float64)
    assert isinstance(sample(X, size=2), numpy.ndarray)

    with warns_deprecated_sympy():
        sample(X, numsamples=2)

@XFAIL
def test_samplingE():
    scipy = import_module('scipy')
    if not scipy:
        skip('Scipy is not installed. Abort tests')
    Y = Normal('Y', 0, 1)
    z = Symbol('z', integer=True)
    assert E(Sum(1/z**Y, (z, 1, oo)), Y > 2, numsamples=3).is_number


def test_given():
    X = Normal('X', 0, 1)
    Y = Normal('Y', 0, 1)
    A = given(X, True)
    B = given(X, Y > 2)

    assert X == A == B


def test_factorial_moment():
    X = Poisson('X', 2)
    Y = Binomial('Y', 2, S.Half)
    Z = Hypergeometric('Z', 4, 2, 2)
    assert factorial_moment(X, 2) == 4
    assert factorial_moment(Y, 2) == S.Half
    assert factorial_moment(Z, 2) == Rational(1, 3)

    x, y, z, l = symbols('x y z l')
    Y = Binomial('Y', 2, y)
    Z = Hypergeometric('Z', 10, 2, 3)
    assert factorial_moment(Y, l) == y**2*FallingFactorial(
        2, l) + 2*y*(1 - y)*FallingFactorial(1, l) + (1 - y)**2*\
            FallingFactorial(0, l)
    assert factorial_moment(Z, l) == 7*FallingFactorial(0, l)/\
        15 + 7*FallingFactorial(1, l)/15 + FallingFactorial(2, l)/15


def test_dependence():
    X, Y = Die('X'), Die('Y')
    assert independent(X, 2*Y)
    assert not dependent(X, 2*Y)

    X, Y = Normal('X', 0, 1), Normal('Y', 0, 1)
    assert independent(X, Y)
    assert dependent(X, 2*X)

    # Create a dependency
    XX, YY = given(Tuple(X, Y), Eq(X + Y, 3))
    assert dependent(XX, YY)

def test_dependent_finite():
    X, Y = Die('X'), Die('Y')
    # Dependence testing requires symbolic conditions which currently break
    # finite random variables
    assert dependent(X, Y + X)

    XX, YY = given(Tuple(X, Y), X + Y > 5)  # Create a dependency
    assert dependent(XX, YY)


def test_normality():
    X, Y = Normal('X', 0, 1), Normal('Y', 0, 1)
    x = Symbol('x', real=True)
    z = Symbol('z', real=True)
    dens = density(X - Y, Eq(X + Y, z))

    assert integrate(dens(x), (x, -oo, oo)) == 1


def test_Density():
    X = Die('X', 6)
    d = Density(X)
    assert d.doit() == density(X)

def test_NamedArgsMixin():
    class Foo(Basic, NamedArgsMixin):
        _argnames = 'foo', 'bar'

    a = Foo(S(1), S(2))

    assert a.foo == 1
    assert a.bar == 2

    raises(AttributeError, lambda: a.baz)

    class Bar(Basic, NamedArgsMixin):
        pass

    raises(AttributeError, lambda: Bar(S(1), S(2)).foo)

def test_density_constant():
    assert density(3)(2) == 0
    assert density(3)(3) == DiracDelta(0)

def test_cmoment_constant():
    assert variance(3) == 0
    assert cmoment(3, 3) == 0
    assert cmoment(3, 4) == 0
    x = Symbol('x')
    assert variance(x) == 0
    assert cmoment(x, 15) == 0
    assert cmoment(x, 0) == 1

def test_moment_constant():
    assert moment(3, 0) == 1
    assert moment(3, 1) == 3
    assert moment(3, 2) == 9
    x = Symbol('x')
    assert moment(x, 2) == x**2

def test_median_constant():
    assert median(3) == 3
    x = Symbol('x')
    assert median(x) == x

def test_real():
    x = Normal('x', 0, 1)
    assert x.is_real


def test_issue_10052():
    X = Exponential('X', 3)
    assert P(X < oo) == 1
    assert P(X > oo) == 0
    assert P(X < 2, X > oo) == 0
    assert P(X < oo, X > oo) == 0
    assert P(X < oo, X > 2) == 1
    assert P(X < 3, X == 2) == 0
    raises(ValueError, lambda: P(1))
    raises(ValueError, lambda: P(X < 1, 2))

def test_issue_11934():
    density = {0: .5, 1: .5}
    X = FiniteRV('X', density)
    assert E(X) == 0.5
    assert P( X>= 2) == 0

def test_issue_8129():
    X = Exponential('X', 4)
    assert P(X >= X) == 1
    assert P(X > X) == 0
    assert P(X > X+1) == 0

def test_issue_12237():
    X = Normal('X', 0, 1)
    Y = Normal('Y', 0, 1)
    U = P(X > 0, X)
    V = P(Y < 0, X)
    W = P(X + Y > 0, X)
    assert W == P(X + Y > 0, X)
    assert U == BernoulliDistribution(S.Half, S.Zero, S.One)
    assert V == S.Half

def test_is_random():
    X = Normal('X', 0, 1)
    Y = Normal('Y', 0, 1)
    a, b = symbols('a, b')
    G = GaussianUnitaryEnsemble('U', 2)
    B = BernoulliProcess('B', 0.9)
    assert not is_random(a)
    assert not is_random(a + b)
    assert not is_random(a * b)
    assert not is_random(Matrix([a**2, b**2]))
    assert is_random(X)
    assert is_random(X**2 + Y)
    assert is_random(Y + b**2)
    assert is_random(Y > 5)
    assert is_random(B[3] < 1)
    assert is_random(G)
    assert is_random(X * Y * B[1])
    assert is_random(Matrix([[X, B[2]], [G, Y]]))
    assert is_random(Eq(X, 4))

def test_issue_12283():
    x = symbols('x')
    X = RandomSymbol(x)
    Y = RandomSymbol('Y')
    Z = RandomMatrixSymbol('Z', 2, 1)
    W = RandomMatrixSymbol('W', 2, 1)
    RI = RandomIndexedSymbol(Indexed('RI', 3))
    assert pspace(Z) == PSpace()
    assert pspace(RI) == PSpace()
    assert pspace(X) == PSpace()
    assert E(X) == Expectation(X)
    assert P(Y > 3) == Probability(Y > 3)
    assert variance(X) == Variance(X)
    assert variance(RI) == Variance(RI)
    assert covariance(X, Y) == Covariance(X, Y)
    assert covariance(W, Z) == Covariance(W, Z)

def test_issue_6810():
    X = Die('X', 6)
    Y = Normal('Y', 0, 1)
    assert P(Eq(X, 2)) == S(1)/6
    assert P(Eq(Y, 0)) == 0
    assert P(Or(X > 2, X < 3)) == 1
    assert P(And(X > 3, X > 2)) == S(1)/2

def test_issue_20286():
    n, p = symbols('n p')
    B = Binomial('B', n, p)
    k = Dummy('k', integer = True)
    eq = Sum(Piecewise((-p**k*(1 - p)**(-k + n)*log(p**k*(1 - p)**(-k + n)*binomial(n, k))*binomial(n, k), (k >= 0) & (k <= n)), (nan, True)), (k, 0, n))
    assert eq.dummy_eq(H(B))
