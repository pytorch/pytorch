from sympy.concrete.summations import Sum
from sympy.core.mul import Mul
from sympy.core.numbers import (oo, pi)
from sympy.core.relational import Eq
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.integrals.integrals import Integral
from sympy.core.expr import unchanged
from sympy.stats import (Normal, Poisson, variance, Covariance, Variance,
                         Probability, Expectation, Moment, CentralMoment)
from sympy.stats.rv import probability, expectation


def test_literal_probability():
    X = Normal('X', 2, 3)
    Y = Normal('Y', 3, 4)
    Z = Poisson('Z', 4)
    W = Poisson('W', 3)
    x = symbols('x', real=True)
    y, w, z = symbols('y, w, z')

    assert Probability(X > 0).evaluate_integral() == probability(X > 0)
    assert Probability(X > x).evaluate_integral() == probability(X > x)
    assert Probability(X > 0).rewrite(Integral).doit() == probability(X > 0)
    assert Probability(X > x).rewrite(Integral).doit() == probability(X > x)

    assert Expectation(X).evaluate_integral() == expectation(X)
    assert Expectation(X).rewrite(Integral).doit() == expectation(X)
    assert Expectation(X**2).evaluate_integral() == expectation(X**2)
    assert Expectation(x*X).args == (x*X,)
    assert Expectation(x*X).expand() == x*Expectation(X)
    assert Expectation(2*X + 3*Y + z*X*Y).expand() == 2*Expectation(X) + 3*Expectation(Y) + z*Expectation(X*Y)
    assert Expectation(2*X + 3*Y + z*X*Y).args == (2*X + 3*Y + z*X*Y,)
    assert Expectation(sin(X)) == Expectation(sin(X)).expand()
    assert Expectation(2*x*sin(X)*Y + y*X**2 + z*X*Y).expand() == 2*x*Expectation(sin(X)*Y) \
                            + y*Expectation(X**2) + z*Expectation(X*Y)
    assert Expectation(X + Y).expand() ==  Expectation(X) + Expectation(Y)
    assert Expectation((X + Y)*(X - Y)).expand() == Expectation(X**2) - Expectation(Y**2)
    assert Expectation((X + Y)*(X - Y)).expand().doit() == -12
    assert Expectation(X + Y, evaluate=True).doit() == 5
    assert Expectation(X + Expectation(Y)).doit() == 5
    assert Expectation(X + Expectation(Y)).doit(deep=False) == 2 + Expectation(Expectation(Y))
    assert Expectation(X + Expectation(Y + Expectation(2*X))).doit(deep=False) == 2 \
                                + Expectation(Expectation(Y + Expectation(2*X)))
    assert Expectation(X + Expectation(Y + Expectation(2*X))).doit() == 9
    assert Expectation(Expectation(2*X)).doit() == 4
    assert Expectation(Expectation(2*X)).doit(deep=False) == Expectation(2*X)
    assert Expectation(4*Expectation(2*X)).doit(deep=False) == 4*Expectation(2*X)
    assert Expectation((X + Y)**3).expand() == 3*Expectation(X*Y**2) +\
                3*Expectation(X**2*Y) + Expectation(X**3) + Expectation(Y**3)
    assert Expectation((X - Y)**3).expand() == 3*Expectation(X*Y**2) -\
                3*Expectation(X**2*Y) + Expectation(X**3) - Expectation(Y**3)
    assert Expectation((X - Y)**2).expand() == -2*Expectation(X*Y) +\
                Expectation(X**2) + Expectation(Y**2)

    assert Variance(w).args == (w,)
    assert Variance(w).expand() == 0
    assert Variance(X).evaluate_integral() == Variance(X).rewrite(Integral).doit() == variance(X)
    assert Variance(X + z).args == (X + z,)
    assert Variance(X + z).expand() == Variance(X)
    assert Variance(X*Y).args == (Mul(X, Y),)
    assert type(Variance(X*Y)) == Variance
    assert Variance(z*X).expand() == z**2*Variance(X)
    assert Variance(X + Y).expand() == Variance(X) + Variance(Y) + 2*Covariance(X, Y)
    assert Variance(X + Y + Z + W).expand() == (Variance(X) + Variance(Y) + Variance(Z) + Variance(W) +
                                       2 * Covariance(X, Y) + 2 * Covariance(X, Z) + 2 * Covariance(X, W) +
                                       2 * Covariance(Y, Z) + 2 * Covariance(Y, W) + 2 * Covariance(W, Z))
    assert Variance(X**2).evaluate_integral() == variance(X**2)
    assert unchanged(Variance, X**2)
    assert Variance(x*X**2).expand() == x**2*Variance(X**2)
    assert Variance(sin(X)).args == (sin(X),)
    assert Variance(sin(X)).expand() == Variance(sin(X))
    assert Variance(x*sin(X)).expand() == x**2*Variance(sin(X))

    assert Covariance(w, z).args == (w, z)
    assert Covariance(w, z).expand() == 0
    assert Covariance(X, w).expand() == 0
    assert Covariance(w, X).expand() == 0
    assert Covariance(X, Y).args == (X, Y)
    assert type(Covariance(X, Y)) == Covariance
    assert Covariance(z*X + 3, Y).expand() == z*Covariance(X, Y)
    assert Covariance(X, X).args == (X, X)
    assert Covariance(X, X).expand() == Variance(X)
    assert Covariance(z*X + 3, w*Y + 4).expand() == w*z*Covariance(X,Y)
    assert Covariance(X, Y) == Covariance(Y, X)
    assert Covariance(X + Y, Z + W).expand() == Covariance(W, X) + Covariance(W, Y) + Covariance(X, Z) + Covariance(Y, Z)
    assert Covariance(x*X + y*Y, z*Z + w*W).expand() == (x*w*Covariance(W, X) + w*y*Covariance(W, Y) +
                                                x*z*Covariance(X, Z) + y*z*Covariance(Y, Z))
    assert Covariance(x*X**2 + y*sin(Y), z*Y*Z**2 + w*W).expand() == (w*x*Covariance(W, X**2) + w*y*Covariance(sin(Y), W) +
                                                        x*z*Covariance(Y*Z**2, X**2) + y*z*Covariance(Y*Z**2, sin(Y)))
    assert Covariance(X, X**2).expand() == Covariance(X, X**2)
    assert Covariance(X, sin(X)).expand() == Covariance(sin(X), X)
    assert Covariance(X**2, sin(X)*Y).expand() == Covariance(sin(X)*Y, X**2)
    assert Covariance(w, X).evaluate_integral() == 0


def test_probability_rewrite():
    X = Normal('X', 2, 3)
    Y = Normal('Y', 3, 4)
    Z = Poisson('Z', 4)
    W = Poisson('W', 3)
    x, y, w, z = symbols('x, y, w, z')

    assert Variance(w).rewrite(Expectation) == 0
    assert Variance(X).rewrite(Expectation) == Expectation(X ** 2) - Expectation(X) ** 2
    assert Variance(X, condition=Y).rewrite(Expectation) == Expectation(X ** 2, Y) - Expectation(X, Y) ** 2
    assert Variance(X, Y) != Expectation(X**2) - Expectation(X)**2
    assert Variance(X + z).rewrite(Expectation) == Expectation((X + z) ** 2) - Expectation(X + z) ** 2
    assert Variance(X * Y).rewrite(Expectation) == Expectation(X ** 2 * Y ** 2) - Expectation(X * Y) ** 2

    assert Covariance(w, X).rewrite(Expectation) == -w*Expectation(X) + Expectation(w*X)
    assert Covariance(X, Y).rewrite(Expectation) == Expectation(X*Y) - Expectation(X)*Expectation(Y)
    assert Covariance(X, Y, condition=W).rewrite(Expectation) == Expectation(X * Y, W) - Expectation(X, W) * Expectation(Y, W)

    w, x, z = symbols("W, x, z")
    px = Probability(Eq(X, x))
    pz = Probability(Eq(Z, z))

    assert Expectation(X).rewrite(Probability) == Integral(x*px, (x, -oo, oo))
    assert Expectation(Z).rewrite(Probability) == Sum(z*pz, (z, 0, oo))
    assert Variance(X).rewrite(Probability) == Integral(x**2*px, (x, -oo, oo)) - Integral(x*px, (x, -oo, oo))**2
    assert Variance(Z).rewrite(Probability) == Sum(z**2*pz, (z, 0, oo)) - Sum(z*pz, (z, 0, oo))**2
    assert Covariance(w, X).rewrite(Probability) == \
           -w*Integral(x*Probability(Eq(X, x)), (x, -oo, oo)) + Integral(w*x*Probability(Eq(X, x)), (x, -oo, oo))

    # To test rewrite as sum function
    assert Variance(X).rewrite(Sum) == Variance(X).rewrite(Integral)
    assert Expectation(X).rewrite(Sum) == Expectation(X).rewrite(Integral)

    assert Covariance(w, X).rewrite(Sum) == 0

    assert Covariance(w, X).rewrite(Integral) == 0

    assert Variance(X, condition=Y).rewrite(Probability) == Integral(x**2*Probability(Eq(X, x), Y), (x, -oo, oo)) - \
                                                            Integral(x*Probability(Eq(X, x), Y), (x, -oo, oo))**2


def test_symbolic_Moment():
    mu = symbols('mu', real=True)
    sigma = symbols('sigma', positive=True)
    x = symbols('x')
    X = Normal('X', mu, sigma)
    M = Moment(X, 4, 2)
    assert M.rewrite(Expectation) == Expectation((X - 2)**4)
    assert M.rewrite(Probability) == Integral((x - 2)**4*Probability(Eq(X, x)),
                                    (x, -oo, oo))
    k = Dummy('k')
    expri = Integral(sqrt(2)*(k - 2)**4*exp(-(k - \
                mu)**2/(2*sigma**2))/(2*sqrt(pi)*sigma), (k, -oo, oo))
    assert M.rewrite(Integral).dummy_eq(expri)
    assert M.doit() == (mu**4 - 8*mu**3 + 6*mu**2*sigma**2 + \
                24*mu**2 - 24*mu*sigma**2 - 32*mu + 3*sigma**4 + 24*sigma**2 + 16)
    M = Moment(2, 5)
    assert M.doit() == 2**5


def test_symbolic_CentralMoment():
    mu = symbols('mu', real=True)
    sigma = symbols('sigma', positive=True)
    x = symbols('x')
    X = Normal('X', mu, sigma)
    CM = CentralMoment(X, 6)
    assert CM.rewrite(Expectation) == Expectation((X - Expectation(X))**6)
    assert CM.rewrite(Probability) == Integral((x - Integral(x*Probability(True),
                    (x, -oo, oo)))**6*Probability(Eq(X, x)), (x, -oo, oo))
    k = Dummy('k')
    expri = Integral(sqrt(2)*(k - Integral(sqrt(2)*k*exp(-(k - \
        mu)**2/(2*sigma**2))/(2*sqrt(pi)*sigma), (k, -oo, oo)))**6*exp(-(k - \
        mu)**2/(2*sigma**2))/(2*sqrt(pi)*sigma), (k, -oo, oo))
    assert CM.rewrite(Integral).dummy_eq(expri)
    assert CM.doit().simplify() == 15*sigma**6
    CM = Moment(5, 5)
    assert CM.doit() == 5**5
