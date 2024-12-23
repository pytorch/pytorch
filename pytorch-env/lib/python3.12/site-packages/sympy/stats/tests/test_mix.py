from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import (Integer, oo, pi)
from sympy.core.power import Pow
from sympy.core.relational import (Eq, Ne)
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.delta_functions import DiracDelta
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import Integral
from sympy.simplify.simplify import simplify
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.functions.elementary.piecewise import ExprCondPair
from sympy.stats import (Poisson, Beta, Exponential, P,
                        Multinomial, MultivariateBeta)
from sympy.stats.crv_types import Normal
from sympy.stats.drv_types import PoissonDistribution
from sympy.stats.compound_rv import CompoundPSpace, CompoundDistribution
from sympy.stats.joint_rv import MarginalDistribution
from sympy.stats.rv import pspace, density
from sympy.testing.pytest import ignore_warnings

def test_density():
    x = Symbol('x')
    l = Symbol('l', positive=True)
    rate = Beta(l, 2, 3)
    X = Poisson(x, rate)
    assert isinstance(pspace(X), CompoundPSpace)
    assert density(X, Eq(rate, rate.symbol)) == PoissonDistribution(l)
    N1 = Normal('N1', 0, 1)
    N2 = Normal('N2', N1, 2)
    assert density(N2)(0).doit() == sqrt(10)/(10*sqrt(pi))
    assert simplify(density(N2, Eq(N1, 1))(x)) == \
        sqrt(2)*exp(-(x - 1)**2/8)/(4*sqrt(pi))
    assert simplify(density(N2)(x)) == sqrt(10)*exp(-x**2/10)/(10*sqrt(pi))

def test_MarginalDistribution():
    a1, p1, p2 = symbols('a1 p1 p2', positive=True)
    C = Multinomial('C', 2, p1, p2)
    B = MultivariateBeta('B', a1, C[0])
    MGR = MarginalDistribution(B, (C[0],))
    mgrc = Mul(Symbol('B'), Piecewise(ExprCondPair(Mul(Integer(2),
    Pow(Symbol('p1', positive=True), Indexed(IndexedBase(Symbol('C')),
    Integer(0))), Pow(Symbol('p2', positive=True),
    Indexed(IndexedBase(Symbol('C')), Integer(1))),
    Pow(factorial(Indexed(IndexedBase(Symbol('C')), Integer(0))), Integer(-1)),
    Pow(factorial(Indexed(IndexedBase(Symbol('C')), Integer(1))), Integer(-1))),
    Eq(Add(Indexed(IndexedBase(Symbol('C')), Integer(0)),
    Indexed(IndexedBase(Symbol('C')), Integer(1))), Integer(2))),
    ExprCondPair(Integer(0), True)), Pow(gamma(Symbol('a1', positive=True)),
    Integer(-1)), gamma(Add(Symbol('a1', positive=True),
    Indexed(IndexedBase(Symbol('C')), Integer(0)))),
    Pow(gamma(Indexed(IndexedBase(Symbol('C')), Integer(0))), Integer(-1)),
    Pow(Indexed(IndexedBase(Symbol('B')), Integer(0)),
    Add(Symbol('a1', positive=True), Integer(-1))),
    Pow(Indexed(IndexedBase(Symbol('B')), Integer(1)),
    Add(Indexed(IndexedBase(Symbol('C')), Integer(0)), Integer(-1))))
    assert MGR(C) == mgrc

def test_compound_distribution():
    Y = Poisson('Y', 1)
    Z = Poisson('Z', Y)
    assert isinstance(pspace(Z), CompoundPSpace)
    assert isinstance(pspace(Z).distribution, CompoundDistribution)
    assert Z.pspace.distribution.pdf(1).doit() == exp(-2)*exp(exp(-1))

def test_mix_expression():
    Y, E = Poisson('Y', 1), Exponential('E', 1)
    k = Dummy('k')
    expr1 = Integral(Sum(exp(-1)*Integral(exp(-k)*DiracDelta(k - 2), (k, 0, oo)
    )/factorial(k), (k, 0, oo)), (k, -oo, 0))
    expr2 = Integral(Sum(exp(-1)*Integral(exp(-k)*DiracDelta(k - 2), (k, 0, oo)
    )/factorial(k), (k, 0, oo)), (k, 0, oo))
    assert P(Eq(Y + E, 1)) == 0
    assert P(Ne(Y + E, 2)) == 1
    with ignore_warnings(UserWarning): ### TODO: Restore tests once warnings are removed
        assert P(E + Y < 2, evaluate=False).rewrite(Integral).dummy_eq(expr1)
        assert P(E + Y > 2, evaluate=False).rewrite(Integral).dummy_eq(expr2)
