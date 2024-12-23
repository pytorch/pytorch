from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.combinatorial.factorials import (RisingFactorial, factorial)
from sympy.functions.elementary.complexes import polar_lift
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.bessel import besselk
from sympy.functions.special.gamma_functions import gamma
from sympy.matrices.dense import eye
from sympy.matrices.expressions.determinant import Determinant
from sympy.sets.fancysets import Range
from sympy.sets.sets import (Interval, ProductSet)
from sympy.simplify.simplify import simplify
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.core.numbers import comp
from sympy.integrals.integrals import integrate
from sympy.matrices import Matrix, MatrixSymbol
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.stats import density, median, marginal_distribution, Normal, Laplace, E, sample
from sympy.stats.joint_rv_types import (JointRV, MultivariateNormalDistribution,
                JointDistributionHandmade, MultivariateT, NormalGamma,
                GeneralizedMultivariateLogGammaOmega as GMVLGO, MultivariateBeta,
                GeneralizedMultivariateLogGamma as GMVLG, MultivariateEwens,
                Multinomial, NegativeMultinomial, MultivariateNormal,
                MultivariateLaplace)
from sympy.testing.pytest import raises, XFAIL, skip, slow
from sympy.external import import_module

from sympy.abc import x, y



def test_Normal():
    m = Normal('A', [1, 2], [[1, 0], [0, 1]])
    A = MultivariateNormal('A', [1, 2], [[1, 0], [0, 1]])
    assert m == A
    assert density(m)(1, 2) == 1/(2*pi)
    assert m.pspace.distribution.set == ProductSet(S.Reals, S.Reals)
    raises (ValueError, lambda:m[2])
    n = Normal('B', [1, 2, 3], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    p = Normal('C',  Matrix([1, 2]), Matrix([[1, 0], [0, 1]]))
    assert density(m)(x, y) == density(p)(x, y)
    assert marginal_distribution(n, 0, 1)(1, 2) == 1/(2*pi)
    raises(ValueError, lambda: marginal_distribution(m))
    assert integrate(density(m)(x, y), (x, -oo, oo), (y, -oo, oo)).evalf() == 1.0
    N = Normal('N', [1, 2], [[x, 0], [0, y]])
    assert density(N)(0, 0) == exp(-((4*x + y)/(2*x*y)))/(2*pi*sqrt(x*y))

    raises (ValueError, lambda: Normal('M', [1, 2], [[1, 1], [1, -1]]))
    # symbolic
    n = symbols('n', integer=True, positive=True)
    mu = MatrixSymbol('mu', n, 1)
    sigma = MatrixSymbol('sigma', n, n)
    X = Normal('X', mu, sigma)
    assert density(X) == MultivariateNormalDistribution(mu, sigma)
    raises (NotImplementedError, lambda: median(m))
    # Below tests should work after issue #17267 is resolved
    # assert E(X) == mu
    # assert variance(X) == sigma

    # test symbolic multivariate normal densities
    n = 3

    Sg = MatrixSymbol('Sg', n, n)
    mu = MatrixSymbol('mu', n, 1)
    obs = MatrixSymbol('obs', n, 1)

    X = MultivariateNormal('X', mu, Sg)
    density_X = density(X)

    eval_a = density_X(obs).subs({Sg: eye(3),
        mu: Matrix([0, 0, 0]), obs: Matrix([0, 0, 0])}).doit()
    eval_b = density_X(0, 0, 0).subs({Sg: eye(3), mu: Matrix([0, 0, 0])}).doit()

    assert eval_a == sqrt(2)/(4*pi**Rational(3/2))
    assert eval_b == sqrt(2)/(4*pi**Rational(3/2))

    n = symbols('n', integer=True, positive=True)

    Sg = MatrixSymbol('Sg', n, n)
    mu = MatrixSymbol('mu', n, 1)
    obs = MatrixSymbol('obs', n, 1)

    X = MultivariateNormal('X', mu, Sg)
    density_X_at_obs = density(X)(obs)

    expected_density = MatrixElement(
        exp((S(1)/2) * (mu.T - obs.T) * Sg**(-1) * (-mu + obs)) / \
        sqrt((2*pi)**n * Determinant(Sg)), 0, 0)

    assert density_X_at_obs == expected_density


def test_MultivariateTDist():
    t1 = MultivariateT('T', [0, 0], [[1, 0], [0, 1]], 2)
    assert(density(t1))(1, 1) == 1/(8*pi)
    assert t1.pspace.distribution.set == ProductSet(S.Reals, S.Reals)
    assert integrate(density(t1)(x, y), (x, -oo, oo), \
        (y, -oo, oo)).evalf() == 1.0
    raises(ValueError, lambda: MultivariateT('T', [1, 2], [[1, 1], [1, -1]], 1))
    t2 = MultivariateT('t2', [1, 2], [[x, 0], [0, y]], 1)
    assert density(t2)(1, 2) == 1/(2*pi*sqrt(x*y))


def test_multivariate_laplace():
    raises(ValueError, lambda: Laplace('T', [1, 2], [[1, 2], [2, 1]]))
    L = Laplace('L', [1, 0], [[1, 0], [0, 1]])
    L2 = MultivariateLaplace('L2', [1, 0], [[1, 0], [0, 1]])
    assert density(L)(2, 3) == exp(2)*besselk(0, sqrt(39))/pi
    L1 = Laplace('L1', [1, 2], [[x, 0], [0, y]])
    assert density(L1)(0, 1) == \
        exp(2/y)*besselk(0, sqrt((2 + 4/y + 1/x)/y))/(pi*sqrt(x*y))
    assert L.pspace.distribution.set == ProductSet(S.Reals, S.Reals)
    assert L.pspace.distribution == L2.pspace.distribution


def test_NormalGamma():
    ng = NormalGamma('G', 1, 2, 3, 4)
    assert density(ng)(1, 1) == 32*exp(-4)/sqrt(pi)
    assert ng.pspace.distribution.set == ProductSet(S.Reals, Interval(0, oo))
    raises(ValueError, lambda:NormalGamma('G', 1, 2, 3, -1))
    assert marginal_distribution(ng, 0)(1) == \
        3*sqrt(10)*gamma(Rational(7, 4))/(10*sqrt(pi)*gamma(Rational(5, 4)))
    assert marginal_distribution(ng, y)(1) == exp(Rational(-1, 4))/128
    assert marginal_distribution(ng,[0,1])(x) == x**2*exp(-x/4)/128


def test_GeneralizedMultivariateLogGammaDistribution():
    h = S.Half
    omega = Matrix([[1, h, h, h],
                     [h, 1, h, h],
                     [h, h, 1, h],
                     [h, h, h, 1]])
    v, l, mu = (4, [1, 2, 3, 4], [1, 2, 3, 4])
    y_1, y_2, y_3, y_4 = symbols('y_1:5', real=True)
    delta = symbols('d', positive=True)
    G = GMVLGO('G', omega, v, l, mu)
    Gd = GMVLG('Gd', delta, v, l, mu)
    dend = ("d**4*Sum(4*24**(-n - 4)*(1 - d)**n*exp((n + 4)*(y_1 + 2*y_2 + 3*y_3 "
            "+ 4*y_4) - exp(y_1) - exp(2*y_2)/2 - exp(3*y_3)/3 - exp(4*y_4)/4)/"
            "(gamma(n + 1)*gamma(n + 4)**3), (n, 0, oo))")
    assert str(density(Gd)(y_1, y_2, y_3, y_4)) == dend
    den = ("5*2**(2/3)*5**(1/3)*Sum(4*24**(-n - 4)*(-2**(2/3)*5**(1/3)/4 + 1)**n*"
          "exp((n + 4)*(y_1 + 2*y_2 + 3*y_3 + 4*y_4) - exp(y_1) - exp(2*y_2)/2 - "
          "exp(3*y_3)/3 - exp(4*y_4)/4)/(gamma(n + 1)*gamma(n + 4)**3), (n, 0, oo))/64")
    assert str(density(G)(y_1, y_2, y_3, y_4)) == den
    marg = ("5*2**(2/3)*5**(1/3)*exp(4*y_1)*exp(-exp(y_1))*Integral(exp(-exp(4*G[3])"
            "/4)*exp(16*G[3])*Integral(exp(-exp(3*G[2])/3)*exp(12*G[2])*Integral(exp("
            "-exp(2*G[1])/2)*exp(8*G[1])*Sum((-1/4)**n*(-4 + 2**(2/3)*5**(1/3"
            "))**n*exp(n*y_1)*exp(2*n*G[1])*exp(3*n*G[2])*exp(4*n*G[3])/(24**n*gamma(n + 1)"
            "*gamma(n + 4)**3), (n, 0, oo)), (G[1], -oo, oo)), (G[2], -oo, oo)), (G[3]"
            ", -oo, oo))/5308416")
    assert str(marginal_distribution(G, G[0])(y_1)) == marg
    omega_f1 = Matrix([[1, h, h]])
    omega_f2 = Matrix([[1, h, h, h],
                     [h, 1, 2, h],
                     [h, h, 1, h],
                     [h, h, h, 1]])
    omega_f3 = Matrix([[6, h, h, h],
                     [h, 1, 2, h],
                     [h, h, 1, h],
                     [h, h, h, 1]])
    v_f = symbols("v_f", positive=False, real=True)
    l_f = [1, 2, v_f, 4]
    m_f = [v_f, 2, 3, 4]
    omega_f4 = Matrix([[1, h, h, h, h],
                     [h, 1, h, h, h],
                     [h, h, 1, h, h],
                     [h, h, h, 1, h],
                     [h, h, h, h, 1]])
    l_f1 = [1, 2, 3, 4, 5]
    omega_f5 = Matrix([[1]])
    mu_f5 = l_f5 = [1]

    raises(ValueError, lambda: GMVLGO('G', omega_f1, v, l, mu))
    raises(ValueError, lambda: GMVLGO('G', omega_f2, v, l, mu))
    raises(ValueError, lambda: GMVLGO('G', omega_f3, v, l, mu))
    raises(ValueError, lambda: GMVLGO('G', omega, v_f, l, mu))
    raises(ValueError, lambda: GMVLGO('G', omega, v, l_f, mu))
    raises(ValueError, lambda: GMVLGO('G', omega, v, l, m_f))
    raises(ValueError, lambda: GMVLGO('G', omega_f4, v, l, mu))
    raises(ValueError, lambda: GMVLGO('G', omega, v, l_f1, mu))
    raises(ValueError, lambda: GMVLGO('G', omega_f5, v, l_f5, mu_f5))
    raises(ValueError, lambda: GMVLG('G', Rational(3, 2), v, l, mu))


def test_MultivariateBeta():
    a1, a2 = symbols('a1, a2', positive=True)
    a1_f, a2_f = symbols('a1, a2', positive=False, real=True)
    mb = MultivariateBeta('B', [a1, a2])
    mb_c = MultivariateBeta('C', a1, a2)
    assert density(mb)(1, 2) == S(2)**(a2 - 1)*gamma(a1 + a2)/\
                                (gamma(a1)*gamma(a2))
    assert marginal_distribution(mb_c, 0)(3) == S(3)**(a1 - 1)*gamma(a1 + a2)/\
                                                (a2*gamma(a1)*gamma(a2))
    raises(ValueError, lambda: MultivariateBeta('b1', [a1_f, a2]))
    raises(ValueError, lambda: MultivariateBeta('b2', [a1, a2_f]))
    raises(ValueError, lambda: MultivariateBeta('b3', [0, 0]))
    raises(ValueError, lambda: MultivariateBeta('b4', [a1_f, a2_f]))
    assert mb.pspace.distribution.set == ProductSet(Interval(0, 1), Interval(0, 1))


def test_MultivariateEwens():
    n, theta, i = symbols('n theta i', positive=True)

    # tests for integer dimensions
    theta_f = symbols('t_f', negative=True)
    a = symbols('a_1:4', positive = True, integer = True)
    ed = MultivariateEwens('E', 3, theta)
    assert density(ed)(a[0], a[1], a[2]) == Piecewise((6*2**(-a[1])*3**(-a[2])*
                                            theta**a[0]*theta**a[1]*theta**a[2]/
                                            (theta*(theta + 1)*(theta + 2)*
                                            factorial(a[0])*factorial(a[1])*
                                            factorial(a[2])), Eq(a[0] + 2*a[1] +
                                            3*a[2], 3)), (0, True))
    assert marginal_distribution(ed, ed[1])(a[1]) == Piecewise((6*2**(-a[1])*
                                                    theta**a[1]/((theta + 1)*
                                                    (theta + 2)*factorial(a[1])),
                                                    Eq(2*a[1] + 1, 3)), (0, True))
    raises(ValueError, lambda: MultivariateEwens('e1', 5, theta_f))
    assert ed.pspace.distribution.set == ProductSet(Range(0, 4, 1),
                                            Range(0, 2, 1), Range(0, 2, 1))

    # tests for symbolic dimensions
    eds = MultivariateEwens('E', n, theta)
    a = IndexedBase('a')
    j, k = symbols('j, k')
    den = Piecewise((factorial(n)*Product(theta**a[j]*(j + 1)**(-a[j])/
           factorial(a[j]), (j, 0, n - 1))/RisingFactorial(theta, n),
            Eq(n, Sum((k + 1)*a[k], (k, 0, n - 1)))), (0, True))
    assert density(eds)(a).dummy_eq(den)


def test_Multinomial():
    n, x1, x2, x3, x4 = symbols('n, x1, x2, x3, x4', nonnegative=True, integer=True)
    p1, p2, p3, p4 = symbols('p1, p2, p3, p4', positive=True)
    p1_f, n_f = symbols('p1_f, n_f', negative=True)
    M = Multinomial('M', n, [p1, p2, p3, p4])
    C = Multinomial('C', 3, p1, p2, p3)
    f = factorial
    assert density(M)(x1, x2, x3, x4) == Piecewise((p1**x1*p2**x2*p3**x3*p4**x4*
                                            f(n)/(f(x1)*f(x2)*f(x3)*f(x4)),
                                            Eq(n, x1 + x2 + x3 + x4)), (0, True))
    assert marginal_distribution(C, C[0])(x1).subs(x1, 1) ==\
                                                            3*p1*p2**2 +\
                                                            6*p1*p2*p3 +\
                                                            3*p1*p3**2
    raises(ValueError, lambda: Multinomial('b1', 5, [p1, p2, p3, p1_f]))
    raises(ValueError, lambda: Multinomial('b2', n_f, [p1, p2, p3, p4]))
    raises(ValueError, lambda: Multinomial('b3', n, 0.5, 0.4, 0.3, 0.1))


def test_NegativeMultinomial():
    k0, x1, x2, x3, x4 = symbols('k0, x1, x2, x3, x4', nonnegative=True, integer=True)
    p1, p2, p3, p4 = symbols('p1, p2, p3, p4', positive=True)
    p1_f = symbols('p1_f', negative=True)
    N = NegativeMultinomial('N', 4, [p1, p2, p3, p4])
    C = NegativeMultinomial('C', 4, 0.1, 0.2, 0.3)
    g = gamma
    f = factorial
    assert simplify(density(N)(x1, x2, x3, x4) -
            p1**x1*p2**x2*p3**x3*p4**x4*(-p1 - p2 - p3 - p4 + 1)**4*g(x1 + x2 +
            x3 + x4 + 4)/(6*f(x1)*f(x2)*f(x3)*f(x4))) is S.Zero
    assert comp(marginal_distribution(C, C[0])(1).evalf(), 0.33, .01)
    raises(ValueError, lambda: NegativeMultinomial('b1', 5, [p1, p2, p3, p1_f]))
    raises(ValueError, lambda: NegativeMultinomial('b2', k0, 0.5, 0.4, 0.3, 0.4))
    assert N.pspace.distribution.set == ProductSet(Range(0, oo, 1),
                    Range(0, oo, 1), Range(0, oo, 1), Range(0, oo, 1))


@slow
def test_JointPSpace_marginal_distribution():
    T = MultivariateT('T', [0, 0], [[1, 0], [0, 1]], 2)
    got = marginal_distribution(T, T[1])(x)
    ans = sqrt(2)*(x**2/2 + 1)/(4*polar_lift(x**2/2 + 1)**(S(5)/2))
    assert got == ans, got
    assert integrate(marginal_distribution(T, 1)(x), (x, -oo, oo)) == 1

    t = MultivariateT('T', [0, 0, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 3)
    assert comp(marginal_distribution(t, 0)(1).evalf(), 0.2, .01)


def test_JointRV():
    x1, x2 = (Indexed('x', i) for i in (1, 2))
    pdf = exp(-x1**2/2 + x1 - x2**2/2 - S.Half)/(2*pi)
    X = JointRV('x', pdf)
    assert density(X)(1, 2) == exp(-2)/(2*pi)
    assert isinstance(X.pspace.distribution, JointDistributionHandmade)
    assert marginal_distribution(X, 0)(2) == sqrt(2)*exp(Rational(-1, 2))/(2*sqrt(pi))


def test_expectation():
    m = Normal('A', [x, y], [[1, 0], [0, 1]])
    assert simplify(E(m[1])) == y


@XFAIL
def test_joint_vector_expectation():
    m = Normal('A', [x, y], [[1, 0], [0, 1]])
    assert E(m) == (x, y)


def test_sample_numpy():
    distribs_numpy = [
        MultivariateNormal("M", [3, 4], [[2, 1], [1, 2]]),
        MultivariateBeta("B", [0.4, 5, 15, 50, 203]),
        Multinomial("N", 50, [0.3, 0.2, 0.1, 0.25, 0.15])
    ]
    size = 3
    numpy = import_module('numpy')
    if not numpy:
        skip('Numpy is not installed. Abort tests for _sample_numpy.')
    else:
        for X in distribs_numpy:
            samps = sample(X, size=size, library='numpy')
            for sam in samps:
                assert tuple(sam) in X.pspace.distribution.set
        N_c = NegativeMultinomial('N', 3, 0.1, 0.1, 0.1)
        raises(NotImplementedError, lambda: sample(N_c, library='numpy'))


def test_sample_scipy():
    distribs_scipy = [
        MultivariateNormal("M", [0, 0], [[0.1, 0.025], [0.025, 0.1]]),
        MultivariateBeta("B", [0.4, 5, 15]),
        Multinomial("N", 8, [0.3, 0.2, 0.1, 0.4])
    ]

    size = 3
    scipy = import_module('scipy')
    if not scipy:
        skip('Scipy not installed. Abort tests for _sample_scipy.')
    else:
        for X in distribs_scipy:
            samps = sample(X, size=size)
            samps2 = sample(X, size=(2, 2))
            for sam in samps:
                assert tuple(sam) in X.pspace.distribution.set
            for i in range(2):
                for j in range(2):
                    assert tuple(samps2[i][j]) in X.pspace.distribution.set
        N_c = NegativeMultinomial('N', 3, 0.1, 0.1, 0.1)
        raises(NotImplementedError, lambda: sample(N_c))


def test_sample_pymc():
    distribs_pymc = [
        MultivariateNormal("M", [5, 2], [[1, 0], [0, 1]]),
        MultivariateBeta("B", [0.4, 5, 15]),
        Multinomial("N", 4, [0.3, 0.2, 0.1, 0.4])
    ]
    size = 3
    pymc = import_module('pymc')
    if not pymc:
        skip('PyMC is not installed. Abort tests for _sample_pymc.')
    else:
        for X in distribs_pymc:
            samps = sample(X, size=size, library='pymc')
            for sam in samps:
                assert tuple(sam.flatten()) in X.pspace.distribution.set
        N_c = NegativeMultinomial('N', 3, 0.1, 0.1, 0.1)
        raises(NotImplementedError, lambda: sample(N_c, library='pymc'))


def test_sample_seed():
    x1, x2 = (Indexed('x', i) for i in (1, 2))
    pdf = exp(-x1**2/2 + x1 - x2**2/2 - S.Half)/(2*pi)
    X = JointRV('x', pdf)

    libraries = ['scipy', 'numpy', 'pymc']
    for lib in libraries:
        try:
            imported_lib = import_module(lib)
            if imported_lib:
                s0, s1, s2 = [], [], []
                s0 = sample(X, size=10, library=lib, seed=0)
                s1 = sample(X, size=10, library=lib, seed=0)
                s2 = sample(X, size=10, library=lib, seed=1)
                assert all(s0 == s1)
                assert all(s1 != s2)
        except NotImplementedError:
            continue

#
# XXX: This fails for pymc. Previously the test appeared to pass but that is
# just because the library argument was not passed so the test always used
# scipy.
#
def test_issue_21057():
    m = Normal("x", [0, 0], [[0, 0], [0, 0]])
    n = MultivariateNormal("x", [0, 0], [[0, 0], [0, 0]])
    p = Normal("x", [0, 0], [[0, 0], [0, 1]])
    assert m == n
    libraries = ('scipy', 'numpy')  # , 'pymc')  # <-- pymc fails
    for library in libraries:
        try:
            imported_lib = import_module(library)
            if imported_lib:
                s1 = sample(m, size=8, library=library)
                s2 = sample(n, size=8, library=library)
                s3 = sample(p, size=8, library=library)
                assert tuple(s1.flatten()) == tuple(s2.flatten())
                for s in s3:
                    assert tuple(s.flatten()) in p.pspace.distribution.set
        except NotImplementedError:
            continue


#
# When this passes the pymc part can be uncommented in test_issue_21057 above
# and this can be deleted.
#
@XFAIL
def test_issue_21057_pymc():
    m = Normal("x", [0, 0], [[0, 0], [0, 0]])
    n = MultivariateNormal("x", [0, 0], [[0, 0], [0, 0]])
    p = Normal("x", [0, 0], [[0, 0], [0, 1]])
    assert m == n
    libraries = ('pymc',)
    for library in libraries:
        try:
            imported_lib = import_module(library)
            if imported_lib:
                s1 = sample(m, size=8, library=library)
                s2 = sample(n, size=8, library=library)
                s3 = sample(p, size=8, library=library)
                assert tuple(s1.flatten()) == tuple(s2.flatten())
                for s in s3:
                    assert tuple(s.flatten()) in p.pspace.distribution.set
        except NotImplementedError:
            continue
