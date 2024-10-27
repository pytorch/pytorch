from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.function import Lambda
from sympy.core.mul import Mul
from sympy.core.numbers import (Integer, Rational, pi)
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (rf, factorial)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.bessel import besselk
from sympy.functions.special.gamma_functions import gamma
from sympy.matrices.dense import (Matrix, ones)
from sympy.sets.fancysets import Range
from sympy.sets.sets import (Intersection, Interval)
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.matrices import ImmutableMatrix, MatrixSymbol
from sympy.matrices.expressions.determinant import det
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.stats.joint_rv import JointDistribution, JointPSpace, MarginalDistribution
from sympy.stats.rv import _value_check, random_symbols

__all__ = ['JointRV',
'MultivariateNormal',
'MultivariateLaplace',
'Dirichlet',
'GeneralizedMultivariateLogGamma',
'GeneralizedMultivariateLogGammaOmega',
'Multinomial',
'MultivariateBeta',
'MultivariateEwens',
'MultivariateT',
'NegativeMultinomial',
'NormalGamma'
]

def multivariate_rv(cls, sym, *args):
    args = list(map(sympify, args))
    dist = cls(*args)
    args = dist.args
    dist.check(*args)
    return JointPSpace(sym, dist).value


def marginal_distribution(rv, *indices):
    """
    Marginal distribution function of a joint random variable.

    Parameters
    ==========

    rv : A random variable with a joint probability distribution.
    indices : Component indices or the indexed random symbol
        for which the joint distribution is to be calculated

    Returns
    =======

    A Lambda expression in `sym`.

    Examples
    ========

    >>> from sympy.stats import MultivariateNormal, marginal_distribution
    >>> m = MultivariateNormal('X', [1, 2], [[2, 1], [1, 2]])
    >>> marginal_distribution(m, m[0])(1)
    1/(2*sqrt(pi))

    """
    indices = list(indices)
    for i in range(len(indices)):
        if isinstance(indices[i], Indexed):
            indices[i] = indices[i].args[1]
    prob_space = rv.pspace
    if not indices:
        raise ValueError(
            "At least one component for marginal density is needed.")
    if hasattr(prob_space.distribution, '_marginal_distribution'):
        return prob_space.distribution._marginal_distribution(indices, rv.symbol)
    return prob_space.marginal_distribution(*indices)


class JointDistributionHandmade(JointDistribution):

    _argnames = ('pdf',)
    is_Continuous = True

    @property
    def set(self):
        return self.args[1]


def JointRV(symbol, pdf, _set=None):
    """
    Create a Joint Random Variable where each of its component is continuous,
    given the following:

    Parameters
    ==========

    symbol : Symbol
        Represents name of the random variable.
    pdf : A PDF in terms of indexed symbols of the symbol given
        as the first argument

    NOTE
    ====

    As of now, the set for each component for a ``JointRV`` is
    equal to the set of all integers, which cannot be changed.

    Examples
    ========

    >>> from sympy import exp, pi, Indexed, S
    >>> from sympy.stats import density, JointRV
    >>> x1, x2 = (Indexed('x', i) for i in (1, 2))
    >>> pdf = exp(-x1**2/2 + x1 - x2**2/2 - S(1)/2)/(2*pi)
    >>> N1 = JointRV('x', pdf) #Multivariate Normal distribution
    >>> density(N1)(1, 2)
    exp(-2)/(2*pi)

    Returns
    =======

    RandomSymbol

    """
    #TODO: Add support for sets provided by the user
    symbol = sympify(symbol)
    syms = [i for i in pdf.free_symbols if isinstance(i, Indexed)
        and i.base == IndexedBase(symbol)]
    syms = tuple(sorted(syms, key = lambda index: index.args[1]))
    _set = S.Reals**len(syms)
    pdf = Lambda(syms, pdf)
    dist = JointDistributionHandmade(pdf, _set)
    jrv = JointPSpace(symbol, dist).value
    rvs = random_symbols(pdf)
    if len(rvs) != 0:
        dist = MarginalDistribution(dist, (jrv,))
        return JointPSpace(symbol, dist).value
    return jrv

#-------------------------------------------------------------------------------
# Multivariate Normal distribution ---------------------------------------------

class MultivariateNormalDistribution(JointDistribution):
    _argnames = ('mu', 'sigma')

    is_Continuous=True

    @property
    def set(self):
        k = self.mu.shape[0]
        return S.Reals**k

    @staticmethod
    def check(mu, sigma):
        _value_check(mu.shape[0] == sigma.shape[0],
            "Size of the mean vector and covariance matrix are incorrect.")
        #check if covariance matrix is positive semi definite or not.
        if not isinstance(sigma, MatrixSymbol):
            _value_check(sigma.is_positive_semidefinite,
            "The covariance matrix must be positive semi definite. ")

    def pdf(self, *args):
        mu, sigma = self.mu, self.sigma
        k = mu.shape[0]
        if len(args) == 1 and args[0].is_Matrix:
            args = args[0]
        else:
            args = ImmutableMatrix(args)
        x = args - mu
        density = S.One/sqrt((2*pi)**(k)*det(sigma))*exp(
            Rational(-1, 2)*x.transpose()*(sigma.inv()*x))
        return MatrixElement(density, 0, 0)

    def _marginal_distribution(self, indices, sym):
        sym = ImmutableMatrix([Indexed(sym, i) for i in indices])
        _mu, _sigma = self.mu, self.sigma
        k = self.mu.shape[0]
        for i in range(k):
            if i not in indices:
                _mu = _mu.row_del(i)
                _sigma = _sigma.col_del(i)
                _sigma = _sigma.row_del(i)
        return Lambda(tuple(sym), S.One/sqrt((2*pi)**(len(_mu))*det(_sigma))*exp(
            Rational(-1, 2)*(_mu - sym).transpose()*(_sigma.inv()*\
                (_mu - sym)))[0])

def MultivariateNormal(name, mu, sigma):
    r"""
    Creates a continuous random variable with Multivariate Normal
    Distribution.

    The density of the multivariate normal distribution can be found at [1].

    Parameters
    ==========

    mu : List representing the mean or the mean vector
    sigma : Positive semidefinite square matrix
        Represents covariance Matrix.
        If `\sigma` is noninvertible then only sampling is supported currently

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import MultivariateNormal, density, marginal_distribution
    >>> from sympy import symbols, MatrixSymbol
    >>> X = MultivariateNormal('X', [3, 4], [[2, 1], [1, 2]])
    >>> y, z = symbols('y z')
    >>> density(X)(y, z)
    sqrt(3)*exp(-y**2/3 + y*z/3 + 2*y/3 - z**2/3 + 5*z/3 - 13/3)/(6*pi)
    >>> density(X)(1, 2)
    sqrt(3)*exp(-4/3)/(6*pi)
    >>> marginal_distribution(X, X[1])(y)
    exp(-(y - 4)**2/4)/(2*sqrt(pi))
    >>> marginal_distribution(X, X[0])(y)
    exp(-(y - 3)**2/4)/(2*sqrt(pi))

    The example below shows that it is also possible to use
    symbolic parameters to define the MultivariateNormal class.

    >>> n = symbols('n', integer=True, positive=True)
    >>> Sg = MatrixSymbol('Sg', n, n)
    >>> mu = MatrixSymbol('mu', n, 1)
    >>> obs = MatrixSymbol('obs', n, 1)
    >>> X = MultivariateNormal('X', mu, Sg)

    The density of a multivariate normal can be
    calculated using a matrix argument, as shown below.

    >>> density(X)(obs)
    (exp(((1/2)*mu.T - (1/2)*obs.T)*Sg**(-1)*(-mu + obs))/sqrt((2*pi)**n*Determinant(Sg)))[0, 0]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Multivariate_normal_distribution

    """
    return multivariate_rv(MultivariateNormalDistribution, name, mu, sigma)

#-------------------------------------------------------------------------------
# Multivariate Laplace distribution --------------------------------------------

class MultivariateLaplaceDistribution(JointDistribution):
    _argnames = ('mu', 'sigma')
    is_Continuous=True

    @property
    def set(self):
        k = self.mu.shape[0]
        return S.Reals**k

    @staticmethod
    def check(mu, sigma):
        _value_check(mu.shape[0] == sigma.shape[0],
                     "Size of the mean vector and covariance matrix are incorrect.")
        # check if covariance matrix is positive definite or not.
        if not isinstance(sigma, MatrixSymbol):
            _value_check(sigma.is_positive_definite,
                         "The covariance matrix must be positive definite. ")

    def pdf(self, *args):
        mu, sigma = self.mu, self.sigma
        mu_T = mu.transpose()
        k = S(mu.shape[0])
        sigma_inv = sigma.inv()
        args = ImmutableMatrix(args)
        args_T = args.transpose()
        x = (mu_T*sigma_inv*mu)[0]
        y = (args_T*sigma_inv*args)[0]
        v = 1 - k/2
        return (2 * (y/(2 + x))**(v/2) * besselk(v, sqrt((2 + x)*y)) *
                exp((args_T * sigma_inv * mu)[0]) /
                ((2 * pi)**(k/2) * sqrt(det(sigma))))


def MultivariateLaplace(name, mu, sigma):
    """
    Creates a continuous random variable with Multivariate Laplace
    Distribution.

    The density of the multivariate Laplace distribution can be found at [1].

    Parameters
    ==========

    mu : List representing the mean or the mean vector
    sigma : Positive definite square matrix
        Represents covariance Matrix

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import MultivariateLaplace, density
    >>> from sympy import symbols
    >>> y, z = symbols('y z')
    >>> X = MultivariateLaplace('X', [2, 4], [[3, 1], [1, 3]])
    >>> density(X)(y, z)
    sqrt(2)*exp(y/4 + 5*z/4)*besselk(0, sqrt(15*y*(3*y/8 - z/8)/2 + 15*z*(-y/8 + 3*z/8)/2))/(4*pi)
    >>> density(X)(1, 2)
    sqrt(2)*exp(11/4)*besselk(0, sqrt(165)/4)/(4*pi)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Multivariate_Laplace_distribution

    """
    return multivariate_rv(MultivariateLaplaceDistribution, name, mu, sigma)

#-------------------------------------------------------------------------------
# Multivariate StudentT distribution -------------------------------------------

class MultivariateTDistribution(JointDistribution):
    _argnames = ('mu', 'shape_mat', 'dof')
    is_Continuous=True

    @property
    def set(self):
        k = self.mu.shape[0]
        return S.Reals**k

    @staticmethod
    def check(mu, sigma, v):
        _value_check(mu.shape[0] == sigma.shape[0],
                     "Size of the location vector and shape matrix are incorrect.")
        # check if covariance matrix is positive definite or not.
        if not isinstance(sigma, MatrixSymbol):
            _value_check(sigma.is_positive_definite,
                         "The shape matrix must be positive definite. ")

    def pdf(self, *args):
        mu, sigma = self.mu, self.shape_mat
        v = S(self.dof)
        k = S(mu.shape[0])
        sigma_inv = sigma.inv()
        args = ImmutableMatrix(args)
        x = args - mu
        return gamma((k + v)/2)/(gamma(v/2)*(v*pi)**(k/2)*sqrt(det(sigma)))\
        *(1 + 1/v*(x.transpose()*sigma_inv*x)[0])**((-v - k)/2)

def MultivariateT(syms, mu, sigma, v):
    """
    Creates a joint random variable with multivariate T-distribution.

    Parameters
    ==========

    syms : A symbol/str
        For identifying the random variable.
    mu : A list/matrix
        Representing the location vector
    sigma : The shape matrix for the distribution

    Examples
    ========

    >>> from sympy.stats import density, MultivariateT
    >>> from sympy import Symbol

    >>> x = Symbol("x")
    >>> X = MultivariateT("x", [1, 1], [[1, 0], [0, 1]], 2)

    >>> density(X)(1, 2)
    2/(9*pi)

    Returns
    =======

    RandomSymbol

    """
    return multivariate_rv(MultivariateTDistribution, syms, mu, sigma, v)


#-------------------------------------------------------------------------------
# Multivariate Normal Gamma distribution ---------------------------------------

class NormalGammaDistribution(JointDistribution):

    _argnames = ('mu', 'lamda', 'alpha', 'beta')
    is_Continuous=True

    @staticmethod
    def check(mu, lamda, alpha, beta):
        _value_check(mu.is_real, "Location must be real.")
        _value_check(lamda > 0, "Lambda must be positive")
        _value_check(alpha > 0, "alpha must be positive")
        _value_check(beta > 0, "beta must be positive")

    @property
    def set(self):
        return S.Reals*Interval(0, S.Infinity)

    def pdf(self, x, tau):
        beta, alpha, lamda = self.beta, self.alpha, self.lamda
        mu = self.mu

        return beta**alpha*sqrt(lamda)/(gamma(alpha)*sqrt(2*pi))*\
        tau**(alpha - S.Half)*exp(-1*beta*tau)*\
        exp(-1*(lamda*tau*(x - mu)**2)/S(2))

    def _marginal_distribution(self, indices, *sym):
        if len(indices) == 2:
            return self.pdf(*sym)
        if indices[0] == 0:
            #For marginal over `x`, return non-standardized Student-T's
            #distribution
            x = sym[0]
            v, mu, sigma = self.alpha - S.Half, self.mu, \
                S(self.beta)/(self.lamda * self.alpha)
            return Lambda(sym, gamma((v + 1)/2)/(gamma(v/2)*sqrt(pi*v)*sigma)*\
                (1 + 1/v*((x - mu)/sigma)**2)**((-v -1)/2))
        #For marginal over `tau`, return Gamma distribution as per construction
        from sympy.stats.crv_types import GammaDistribution
        return Lambda(sym, GammaDistribution(self.alpha, self.beta)(sym[0]))

def NormalGamma(sym, mu, lamda, alpha, beta):
    """
    Creates a bivariate joint random variable with multivariate Normal gamma
    distribution.

    Parameters
    ==========

    sym : A symbol/str
        For identifying the random variable.
    mu : A real number
        The mean of the normal distribution
    lamda : A positive integer
        Parameter of joint distribution
    alpha : A positive integer
        Parameter of joint distribution
    beta : A positive integer
        Parameter of joint distribution

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density, NormalGamma
    >>> from sympy import symbols

    >>> X = NormalGamma('x', 0, 1, 2, 3)
    >>> y, z = symbols('y z')

    >>> density(X)(y, z)
    9*sqrt(2)*z**(3/2)*exp(-3*z)*exp(-y**2*z/2)/(2*sqrt(pi))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Normal-gamma_distribution

    """
    return multivariate_rv(NormalGammaDistribution, sym, mu, lamda, alpha, beta)

#-------------------------------------------------------------------------------
# Multivariate Beta/Dirichlet distribution -------------------------------------

class MultivariateBetaDistribution(JointDistribution):

    _argnames = ('alpha',)
    is_Continuous = True

    @staticmethod
    def check(alpha):
        _value_check(len(alpha) >= 2, "At least two categories should be passed.")
        for a_k in alpha:
            _value_check((a_k > 0) != False, "Each concentration parameter"
                                            " should be positive.")

    @property
    def set(self):
        k = len(self.alpha)
        return Interval(0, 1)**k

    def pdf(self, *syms):
        alpha = self.alpha
        B = Mul.fromiter(map(gamma, alpha))/gamma(Add(*alpha))
        return Mul.fromiter(sym**(a_k - 1) for a_k, sym in zip(alpha, syms))/B

def MultivariateBeta(syms, *alpha):
    """
    Creates a continuous random variable with Dirichlet/Multivariate Beta
    Distribution.

    The density of the Dirichlet distribution can be found at [1].

    Parameters
    ==========

    alpha : Positive real numbers
        Signifies concentration numbers.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density, MultivariateBeta, marginal_distribution
    >>> from sympy import Symbol
    >>> a1 = Symbol('a1', positive=True)
    >>> a2 = Symbol('a2', positive=True)
    >>> B = MultivariateBeta('B', [a1, a2])
    >>> C = MultivariateBeta('C', a1, a2)
    >>> x = Symbol('x')
    >>> y = Symbol('y')
    >>> density(B)(x, y)
    x**(a1 - 1)*y**(a2 - 1)*gamma(a1 + a2)/(gamma(a1)*gamma(a2))
    >>> marginal_distribution(C, C[0])(x)
    x**(a1 - 1)*gamma(a1 + a2)/(a2*gamma(a1)*gamma(a2))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Dirichlet_distribution
    .. [2] https://mathworld.wolfram.com/DirichletDistribution.html

    """
    if not isinstance(alpha[0], list):
        alpha = (list(alpha),)
    return multivariate_rv(MultivariateBetaDistribution, syms, alpha[0])

Dirichlet = MultivariateBeta

#-------------------------------------------------------------------------------
# Multivariate Ewens distribution ----------------------------------------------

class MultivariateEwensDistribution(JointDistribution):

    _argnames = ('n', 'theta')
    is_Discrete = True
    is_Continuous = False

    @staticmethod
    def check(n, theta):
        _value_check((n > 0),
                        "sample size should be positive integer.")
        _value_check(theta.is_positive, "mutation rate should be positive.")

    @property
    def set(self):
        if not isinstance(self.n, Integer):
            i = Symbol('i', integer=True, positive=True)
            return Product(Intersection(S.Naturals0, Interval(0, self.n//i)),
                                    (i, 1, self.n))
        prod_set = Range(0, self.n + 1)
        for i in range(2, self.n + 1):
            prod_set *= Range(0, self.n//i + 1)
        return prod_set.flatten()

    def pdf(self, *syms):
        n, theta = self.n, self.theta
        condi = isinstance(self.n, Integer)
        if not (isinstance(syms[0], IndexedBase) or condi):
            raise ValueError("Please use IndexedBase object for syms as "
                                "the dimension is symbolic")
        term_1 = factorial(n)/rf(theta, n)
        if condi:
            term_2 = Mul.fromiter(theta**syms[j]/((j+1)**syms[j]*factorial(syms[j]))
                                    for j in range(n))
            cond = Eq(sum((k + 1)*syms[k] for k in range(n)), n)
            return Piecewise((term_1 * term_2, cond), (0, True))
        syms = syms[0]
        j, k = symbols('j, k', positive=True, integer=True)
        term_2 = Product(theta**syms[j]/((j+1)**syms[j]*factorial(syms[j])),
                            (j, 0, n - 1))
        cond = Eq(Sum((k + 1)*syms[k], (k, 0, n - 1)), n)
        return Piecewise((term_1 * term_2, cond), (0, True))


def MultivariateEwens(syms, n, theta):
    """
    Creates a discrete random variable with Multivariate Ewens
    Distribution.

    The density of the said distribution can be found at [1].

    Parameters
    ==========

    n : Positive integer
        Size of the sample or the integer whose partitions are considered
    theta : Positive real number
        Denotes Mutation rate

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density, marginal_distribution, MultivariateEwens
    >>> from sympy import Symbol
    >>> a1 = Symbol('a1', positive=True)
    >>> a2 = Symbol('a2', positive=True)
    >>> ed = MultivariateEwens('E', 2, 1)
    >>> density(ed)(a1, a2)
    Piecewise((1/(2**a2*factorial(a1)*factorial(a2)), Eq(a1 + 2*a2, 2)), (0, True))
    >>> marginal_distribution(ed, ed[0])(a1)
    Piecewise((1/factorial(a1), Eq(a1, 2)), (0, True))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Ewens%27s_sampling_formula
    .. [2] https://www.jstor.org/stable/24780825
    """
    return multivariate_rv(MultivariateEwensDistribution, syms, n, theta)

#-------------------------------------------------------------------------------
# Generalized Multivariate Log Gamma distribution ------------------------------

class GeneralizedMultivariateLogGammaDistribution(JointDistribution):

    _argnames = ('delta', 'v', 'lamda', 'mu')
    is_Continuous=True

    def check(self, delta, v, l, mu):
        _value_check((delta >= 0, delta <= 1), "delta must be in range [0, 1].")
        _value_check((v > 0), "v must be positive")
        for lk in l:
            _value_check((lk > 0), "lamda must be a positive vector.")
        for muk in mu:
            _value_check((muk > 0), "mu must be a positive vector.")
        _value_check(len(l) > 1,"the distribution should have at least"
                                " two random variables.")

    @property
    def set(self):
        return S.Reals**len(self.lamda)

    def pdf(self, *y):
        d, v, l, mu = self.delta, self.v, self.lamda, self.mu
        n = Symbol('n', negative=False, integer=True)
        k = len(l)
        sterm1 = Pow((1 - d), n)/\
                ((gamma(v + n)**(k - 1))*gamma(v)*gamma(n + 1))
        sterm2 = Mul.fromiter(mui*li**(-v - n) for mui, li in zip(mu, l))
        term1 = sterm1 * sterm2
        sterm3 = (v + n) * sum(mui * yi for mui, yi in zip(mu, y))
        sterm4 = sum(exp(mui * yi)/li for (mui, yi, li) in zip(mu, y, l))
        term2 = exp(sterm3 - sterm4)
        return Pow(d, v) * Sum(term1 * term2, (n, 0, S.Infinity))

def GeneralizedMultivariateLogGamma(syms, delta, v, lamda, mu):
    """
    Creates a joint random variable with generalized multivariate log gamma
    distribution.

    The joint pdf can be found at [1].

    Parameters
    ==========

    syms : list/tuple/set of symbols for identifying each component
    delta : A constant in range $[0, 1]$
    v : Positive real number
    lamda : List of positive real numbers
    mu : List of positive real numbers

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density
    >>> from sympy.stats.joint_rv_types import GeneralizedMultivariateLogGamma
    >>> from sympy import symbols, S
    >>> v = 1
    >>> l, mu = [1, 1, 1], [1, 1, 1]
    >>> d = S.Half
    >>> y = symbols('y_1:4', positive=True)
    >>> Gd = GeneralizedMultivariateLogGamma('G', d, v, l, mu)
    >>> density(Gd)(y[0], y[1], y[2])
    Sum(exp((n + 1)*(y_1 + y_2 + y_3) - exp(y_1) - exp(y_2) -
    exp(y_3))/(2**n*gamma(n + 1)**3), (n, 0, oo))/2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Generalized_multivariate_log-gamma_distribution
    .. [2] https://www.researchgate.net/publication/234137346_On_a_multivariate_log-gamma_distribution_and_the_use_of_the_distribution_in_the_Bayesian_analysis

    Note
    ====

    If the GeneralizedMultivariateLogGamma is too long to type use,

    >>> from sympy.stats.joint_rv_types import GeneralizedMultivariateLogGamma as GMVLG
    >>> Gd = GMVLG('G', d, v, l, mu)

    If you want to pass the matrix omega instead of the constant delta, then use
    ``GeneralizedMultivariateLogGammaOmega``.

    """
    return multivariate_rv(GeneralizedMultivariateLogGammaDistribution,
                            syms, delta, v, lamda, mu)

def GeneralizedMultivariateLogGammaOmega(syms, omega, v, lamda, mu):
    """
    Extends GeneralizedMultivariateLogGamma.

    Parameters
    ==========

    syms : list/tuple/set of symbols
        For identifying each component
    omega : A square matrix
           Every element of square matrix must be absolute value of
           square root of correlation coefficient
    v : Positive real number
    lamda : List of positive real numbers
    mu : List of positive real numbers

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density
    >>> from sympy.stats.joint_rv_types import GeneralizedMultivariateLogGammaOmega
    >>> from sympy import Matrix, symbols, S
    >>> omega = Matrix([[1, S.Half, S.Half], [S.Half, 1, S.Half], [S.Half, S.Half, 1]])
    >>> v = 1
    >>> l, mu = [1, 1, 1], [1, 1, 1]
    >>> G = GeneralizedMultivariateLogGammaOmega('G', omega, v, l, mu)
    >>> y = symbols('y_1:4', positive=True)
    >>> density(G)(y[0], y[1], y[2])
    sqrt(2)*Sum((1 - sqrt(2)/2)**n*exp((n + 1)*(y_1 + y_2 + y_3) - exp(y_1) -
    exp(y_2) - exp(y_3))/gamma(n + 1)**3, (n, 0, oo))/2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Generalized_multivariate_log-gamma_distribution
    .. [2] https://www.researchgate.net/publication/234137346_On_a_multivariate_log-gamma_distribution_and_the_use_of_the_distribution_in_the_Bayesian_analysis

    Notes
    =====

    If the GeneralizedMultivariateLogGammaOmega is too long to type use,

    >>> from sympy.stats.joint_rv_types import GeneralizedMultivariateLogGammaOmega as GMVLGO
    >>> G = GMVLGO('G', omega, v, l, mu)

    """
    _value_check((omega.is_square, isinstance(omega, Matrix)), "omega must be a"
                                                            " square matrix")
    for val in omega.values():
        _value_check((val >= 0, val <= 1),
            "all values in matrix must be between 0 and 1(both inclusive).")
    _value_check(omega.diagonal().equals(ones(1, omega.shape[0])),
                    "all the elements of diagonal should be 1.")
    _value_check((omega.shape[0] == len(lamda), len(lamda) == len(mu)),
                    "lamda, mu should be of same length and omega should "
                    " be of shape (length of lamda, length of mu)")
    _value_check(len(lamda) > 1,"the distribution should have at least"
                            " two random variables.")
    delta = Pow(Rational(omega.det()), Rational(1, len(lamda) - 1))
    return GeneralizedMultivariateLogGamma(syms, delta, v, lamda, mu)


#-------------------------------------------------------------------------------
# Multinomial distribution -----------------------------------------------------

class MultinomialDistribution(JointDistribution):

    _argnames = ('n', 'p')
    is_Continuous=False
    is_Discrete = True

    @staticmethod
    def check(n, p):
        _value_check(n > 0,
                        "number of trials must be a positive integer")
        for p_k in p:
            _value_check((p_k >= 0, p_k <= 1),
                        "probability must be in range [0, 1]")
        _value_check(Eq(sum(p), 1),
                        "probabilities must sum to 1")

    @property
    def set(self):
        return Intersection(S.Naturals0, Interval(0, self.n))**len(self.p)

    def pdf(self, *x):
        n, p = self.n, self.p
        term_1 = factorial(n)/Mul.fromiter(factorial(x_k) for x_k in x)
        term_2 = Mul.fromiter(p_k**x_k for p_k, x_k in zip(p, x))
        return Piecewise((term_1 * term_2, Eq(sum(x), n)), (0, True))

def Multinomial(syms, n, *p):
    """
    Creates a discrete random variable with Multinomial Distribution.

    The density of the said distribution can be found at [1].

    Parameters
    ==========

    n : Positive integer
        Represents number of trials
    p : List of event probabilities
        Must be in the range of $[0, 1]$.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density, Multinomial, marginal_distribution
    >>> from sympy import symbols
    >>> x1, x2, x3 = symbols('x1, x2, x3', nonnegative=True, integer=True)
    >>> p1, p2, p3 = symbols('p1, p2, p3', positive=True)
    >>> M = Multinomial('M', 3, p1, p2, p3)
    >>> density(M)(x1, x2, x3)
    Piecewise((6*p1**x1*p2**x2*p3**x3/(factorial(x1)*factorial(x2)*factorial(x3)),
    Eq(x1 + x2 + x3, 3)), (0, True))
    >>> marginal_distribution(M, M[0])(x1).subs(x1, 1)
    3*p1*p2**2 + 6*p1*p2*p3 + 3*p1*p3**2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Multinomial_distribution
    .. [2] https://mathworld.wolfram.com/MultinomialDistribution.html

    """
    if not isinstance(p[0], list):
        p = (list(p), )
    return multivariate_rv(MultinomialDistribution, syms, n, p[0])

#-------------------------------------------------------------------------------
# Negative Multinomial Distribution --------------------------------------------

class NegativeMultinomialDistribution(JointDistribution):

    _argnames = ('k0', 'p')
    is_Continuous=False
    is_Discrete = True

    @staticmethod
    def check(k0, p):
        _value_check(k0 > 0,
                        "number of failures must be a positive integer")
        for p_k in p:
            _value_check((p_k >= 0, p_k <= 1),
                        "probability must be in range [0, 1].")
        _value_check(sum(p) <= 1,
                        "success probabilities must not be greater than 1.")

    @property
    def set(self):
        return Range(0, S.Infinity)**len(self.p)

    def pdf(self, *k):
        k0, p = self.k0, self.p
        term_1 = (gamma(k0 + sum(k))*(1 - sum(p))**k0)/gamma(k0)
        term_2 = Mul.fromiter(pi**ki/factorial(ki) for pi, ki in zip(p, k))
        return term_1 * term_2

def NegativeMultinomial(syms, k0, *p):
    """
    Creates a discrete random variable with Negative Multinomial Distribution.

    The density of the said distribution can be found at [1].

    Parameters
    ==========

    k0 : positive integer
        Represents number of failures before the experiment is stopped
    p : List of event probabilities
        Must be in the range of $[0, 1]$

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density, NegativeMultinomial, marginal_distribution
    >>> from sympy import symbols
    >>> x1, x2, x3 = symbols('x1, x2, x3', nonnegative=True, integer=True)
    >>> p1, p2, p3 = symbols('p1, p2, p3', positive=True)
    >>> N = NegativeMultinomial('M', 3, p1, p2, p3)
    >>> N_c = NegativeMultinomial('M', 3, 0.1, 0.1, 0.1)
    >>> density(N)(x1, x2, x3)
    p1**x1*p2**x2*p3**x3*(-p1 - p2 - p3 + 1)**3*gamma(x1 + x2 +
    x3 + 3)/(2*factorial(x1)*factorial(x2)*factorial(x3))
    >>> marginal_distribution(N_c, N_c[0])(1).evalf().round(2)
    0.25


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Negative_multinomial_distribution
    .. [2] https://mathworld.wolfram.com/NegativeBinomialDistribution.html

    """
    if not isinstance(p[0], list):
        p = (list(p), )
    return multivariate_rv(NegativeMultinomialDistribution, syms, k0, p[0])
