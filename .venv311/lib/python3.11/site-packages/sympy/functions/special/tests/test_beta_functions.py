from sympy.core.function import (diff, expand_func)
from sympy.core.numbers import I, Rational, pi
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.combinatorial.numbers import catalan
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.beta_functions import (beta, betainc, betainc_regularized)
from sympy.functions.special.gamma_functions import gamma, polygamma
from sympy.functions.special.hyper import hyper
from sympy.integrals.integrals import Integral
from sympy.core.function import ArgumentIndexError
from sympy.core.expr import unchanged
from sympy.testing.pytest import raises


def test_beta():
    x, y = symbols('x y')
    t = Dummy('t')

    assert unchanged(beta, x, y)
    assert unchanged(beta, x, x)

    assert beta(5, -3).is_real == True
    assert beta(3, y).is_real is None

    assert expand_func(beta(x, y)) == gamma(x)*gamma(y)/gamma(x + y)
    assert expand_func(beta(x, y) - beta(y, x)) == 0  # Symmetric
    assert expand_func(beta(x, y)) == expand_func(beta(x, y + 1) + beta(x + 1, y)).simplify()

    assert diff(beta(x, y), x) == beta(x, y)*(polygamma(0, x) - polygamma(0, x + y))
    assert diff(beta(x, y), y) == beta(x, y)*(polygamma(0, y) - polygamma(0, x + y))

    assert conjugate(beta(x, y)) == beta(conjugate(x), conjugate(y))

    raises(ArgumentIndexError, lambda: beta(x, y).fdiff(3))

    assert beta(x, y).rewrite(gamma) == gamma(x)*gamma(y)/gamma(x + y)
    assert beta(x).rewrite(gamma) == gamma(x)**2/gamma(2*x)
    assert beta(x, y).rewrite(Integral).dummy_eq(Integral(t**(x - 1) * (1 - t)**(y - 1), (t, 0, 1)))
    assert beta(Rational(-19, 10), Rational(-1, 10)) == S.Zero
    assert beta(Rational(-19, 10), Rational(-9, 10)) == \
        800*2**(S(4)/5)*sqrt(pi)*gamma(S.One/10)/(171*gamma(-S(7)/5))
    assert beta(Rational(19, 10), Rational(29, 10)) == 100/(551*catalan(Rational(19, 10)))
    assert beta(1, 0) == S.ComplexInfinity
    assert beta(0, 1) == S.ComplexInfinity
    assert beta(2, 3) == S.One/12
    assert unchanged(beta, x, x + 1)
    assert unchanged(beta, x, 1)
    assert unchanged(beta, 1, y)
    assert beta(x, x + 1).doit() == 1/(x*(x+1)*catalan(x))
    assert beta(1, y).doit() == 1/y
    assert beta(x, 1).doit() == 1/x
    assert beta(Rational(-19, 10), Rational(-1, 10), evaluate=False).doit() == S.Zero
    assert beta(2) == beta(2, 2)
    assert beta(x, evaluate=False) != beta(x, x)
    assert beta(x, evaluate=False).doit() == beta(x, x)


def test_betainc():
    a, b, x1, x2 = symbols('a b x1 x2')

    assert unchanged(betainc, a, b, x1, x2)
    assert unchanged(betainc, a, b, 0, x1)

    assert betainc(1, 2, 0, -5).is_real == True
    assert betainc(1, 2, 0, x2).is_real is None
    assert conjugate(betainc(I, 2, 3 - I, 1 + 4*I)) == betainc(-I, 2, 3 + I, 1 - 4*I)

    assert betainc(a, b, 0, 1).rewrite(Integral).dummy_eq(beta(a, b).rewrite(Integral))
    assert betainc(1, 2, 0, x2).rewrite(hyper) == x2*hyper((1, -1), (2,), x2)

    assert betainc(1, 2, 3, 3).evalf() == 0


def test_betainc_regularized():
    a, b, x1, x2 = symbols('a b x1 x2')

    assert unchanged(betainc_regularized, a, b, x1, x2)
    assert unchanged(betainc_regularized, a, b, 0, x1)

    assert betainc_regularized(3, 5, 0, -1).is_real == True
    assert betainc_regularized(3, 5, 0, x2).is_real is None
    assert conjugate(betainc_regularized(3*I, 1, 2 + I, 1 + 2*I)) == betainc_regularized(-3*I, 1, 2 - I, 1 - 2*I)

    assert betainc_regularized(a, b, 0, 1).rewrite(Integral) == 1
    assert betainc_regularized(1, 2, x1, x2).rewrite(hyper) == 2*x2*hyper((1, -1), (2,), x2) - 2*x1*hyper((1, -1), (2,), x1)

    assert betainc_regularized(4, 1, 5, 5).evalf() == 0
