from math import prod

from sympy.core.basic import Basic
from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.special.gamma_functions import multigamma
from sympy.core.sympify import sympify, _sympify
from sympy.matrices import (ImmutableMatrix, Inverse, Trace, Determinant,
                            MatrixSymbol, MatrixBase, Transpose, MatrixSet,
                            matrix2numpy)
from sympy.stats.rv import (_value_check, RandomMatrixSymbol, NamedArgsMixin, PSpace,
                            _symbol_converter, MatrixDomain, Distribution)
from sympy.external import import_module


################################################################################
#------------------------Matrix Probability Space------------------------------#
################################################################################
class MatrixPSpace(PSpace):
    """
    Represents probability space for
    Matrix Distributions.
    """
    def __new__(cls, sym, distribution, dim_n, dim_m):
        sym = _symbol_converter(sym)
        dim_n, dim_m = _sympify(dim_n), _sympify(dim_m)
        if not (dim_n.is_integer and dim_m.is_integer):
            raise ValueError("Dimensions should be integers")
        return Basic.__new__(cls, sym, distribution, dim_n, dim_m)

    distribution = property(lambda self: self.args[1])
    symbol = property(lambda self: self.args[0])

    @property
    def domain(self):
        return MatrixDomain(self.symbol, self.distribution.set)

    @property
    def value(self):
        return RandomMatrixSymbol(self.symbol, self.args[2], self.args[3], self)

    @property
    def values(self):
        return {self.value}

    def compute_density(self, expr, *args):
        rms = expr.atoms(RandomMatrixSymbol)
        if len(rms) > 1 or (not isinstance(expr, RandomMatrixSymbol)):
            raise NotImplementedError("Currently, no algorithm has been "
                    "implemented to handle general expressions containing "
                    "multiple matrix distributions.")
        return self.distribution.pdf(expr)

    def sample(self, size=(), library='scipy', seed=None):
        """
        Internal sample method

        Returns dictionary mapping RandomMatrixSymbol to realization value.
        """
        return {self.value: self.distribution.sample(size, library=library, seed=seed)}


def rv(symbol, cls, args):
    args = list(map(sympify, args))
    dist = cls(*args)
    dist.check(*args)
    dim = dist.dimension
    pspace = MatrixPSpace(symbol, dist, dim[0], dim[1])
    return pspace.value


class SampleMatrixScipy:
    """Returns the sample from scipy of the given distribution"""
    def __new__(cls, dist, size, seed=None):
        return cls._sample_scipy(dist, size, seed)

    @classmethod
    def _sample_scipy(cls, dist, size, seed):
        """Sample from SciPy."""

        from scipy import stats as scipy_stats
        import numpy
        scipy_rv_map = {
            'WishartDistribution': lambda dist, size, rand_state: scipy_stats.wishart.rvs(
                df=int(dist.n), scale=matrix2numpy(dist.scale_matrix, float), size=size),
            'MatrixNormalDistribution': lambda dist, size, rand_state: scipy_stats.matrix_normal.rvs(
                mean=matrix2numpy(dist.location_matrix, float),
                rowcov=matrix2numpy(dist.scale_matrix_1, float),
                colcov=matrix2numpy(dist.scale_matrix_2, float), size=size, random_state=rand_state)
        }

        sample_shape = {
            'WishartDistribution': lambda dist: dist.scale_matrix.shape,
            'MatrixNormalDistribution' : lambda dist: dist.location_matrix.shape
        }

        dist_list = scipy_rv_map.keys()

        if dist.__class__.__name__ not in dist_list:
            return None

        if seed is None or isinstance(seed, int):
            rand_state = numpy.random.default_rng(seed=seed)
        else:
            rand_state = seed
        samp = scipy_rv_map[dist.__class__.__name__](dist, prod(size), rand_state)
        return samp.reshape(size + sample_shape[dist.__class__.__name__](dist))


class SampleMatrixNumpy:
    """Returns the sample from numpy of the given distribution"""

    ### TODO: Add tests after adding matrix distributions in numpy_rv_map
    def __new__(cls, dist, size, seed=None):
        return cls._sample_numpy(dist, size, seed)

    @classmethod
    def _sample_numpy(cls, dist, size, seed):
        """Sample from NumPy."""

        numpy_rv_map = {
        }

        sample_shape = {
        }

        dist_list = numpy_rv_map.keys()

        if dist.__class__.__name__ not in dist_list:
            return None

        import numpy
        if seed is None or isinstance(seed, int):
            rand_state = numpy.random.default_rng(seed=seed)
        else:
            rand_state = seed
        samp = numpy_rv_map[dist.__class__.__name__](dist, prod(size), rand_state)
        return samp.reshape(size + sample_shape[dist.__class__.__name__](dist))


class SampleMatrixPymc:
    """Returns the sample from pymc of the given distribution"""

    def __new__(cls, dist, size, seed=None):
        return cls._sample_pymc(dist, size, seed)

    @classmethod
    def _sample_pymc(cls, dist, size, seed):
        """Sample from PyMC."""

        try:
            import pymc
        except ImportError:
            import pymc3 as pymc
        pymc_rv_map = {
            'MatrixNormalDistribution': lambda dist: pymc.MatrixNormal('X',
                mu=matrix2numpy(dist.location_matrix, float),
                rowcov=matrix2numpy(dist.scale_matrix_1, float),
                colcov=matrix2numpy(dist.scale_matrix_2, float),
                shape=dist.location_matrix.shape),
            'WishartDistribution': lambda dist: pymc.WishartBartlett('X',
                nu=int(dist.n), S=matrix2numpy(dist.scale_matrix, float))
        }

        sample_shape = {
            'WishartDistribution': lambda dist: dist.scale_matrix.shape,
            'MatrixNormalDistribution' : lambda dist: dist.location_matrix.shape
        }

        dist_list = pymc_rv_map.keys()

        if dist.__class__.__name__ not in dist_list:
            return None
        import logging
        logging.getLogger("pymc").setLevel(logging.ERROR)
        with pymc.Model():
            pymc_rv_map[dist.__class__.__name__](dist)
            samps = pymc.sample(draws=prod(size), chains=1, progressbar=False, random_seed=seed, return_inferencedata=False, compute_convergence_checks=False)['X']
        return samps.reshape(size + sample_shape[dist.__class__.__name__](dist))

_get_sample_class_matrixrv = {
    'scipy': SampleMatrixScipy,
    'pymc3': SampleMatrixPymc,
    'pymc': SampleMatrixPymc,
    'numpy': SampleMatrixNumpy
}

################################################################################
#-------------------------Matrix Distribution----------------------------------#
################################################################################

class MatrixDistribution(Distribution, NamedArgsMixin):
    """
    Abstract class for Matrix Distribution.
    """
    def __new__(cls, *args):
        args = [ImmutableMatrix(arg) if isinstance(arg, list)
                else _sympify(arg) for arg in args]
        return Basic.__new__(cls, *args)

    @staticmethod
    def check(*args):
        pass

    def __call__(self, expr):
        if isinstance(expr, list):
            expr = ImmutableMatrix(expr)
        return self.pdf(expr)

    def sample(self, size=(), library='scipy', seed=None):
        """
        Internal sample method

        Returns dictionary mapping RandomSymbol to realization value.
        """

        libraries = ['scipy', 'numpy', 'pymc3', 'pymc']
        if library not in libraries:
            raise NotImplementedError("Sampling from %s is not supported yet."
                                        % str(library))
        if not import_module(library):
            raise ValueError("Failed to import %s" % library)

        samps = _get_sample_class_matrixrv[library](self, size, seed)

        if samps is not None:
            return samps
        raise NotImplementedError(
                "Sampling for %s is not currently implemented from %s"
                % (self.__class__.__name__, library)
                )

################################################################################
#------------------------Matrix Distribution Types-----------------------------#
################################################################################

#-------------------------------------------------------------------------------
# Matrix Gamma distribution ----------------------------------------------------

class MatrixGammaDistribution(MatrixDistribution):

    _argnames = ('alpha', 'beta', 'scale_matrix')

    @staticmethod
    def check(alpha, beta, scale_matrix):
        if not isinstance(scale_matrix, MatrixSymbol):
            _value_check(scale_matrix.is_positive_definite, "The shape "
                "matrix must be positive definite.")
        _value_check(scale_matrix.is_square, "Should "
        "be square matrix")
        _value_check(alpha.is_positive, "Shape parameter should be positive.")
        _value_check(beta.is_positive, "Scale parameter should be positive.")

    @property
    def set(self):
        k = self.scale_matrix.shape[0]
        return MatrixSet(k, k, S.Reals)

    @property
    def dimension(self):
        return self.scale_matrix.shape

    def pdf(self, x):
        alpha, beta, scale_matrix = self.alpha, self.beta, self.scale_matrix
        p = scale_matrix.shape[0]
        if isinstance(x, list):
            x = ImmutableMatrix(x)
        if not isinstance(x, (MatrixBase, MatrixSymbol)):
            raise ValueError("%s should be an isinstance of Matrix "
                    "or MatrixSymbol" % str(x))
        sigma_inv_x = - Inverse(scale_matrix)*x / beta
        term1 = exp(Trace(sigma_inv_x))/((beta**(p*alpha)) * multigamma(alpha, p))
        term2 = (Determinant(scale_matrix))**(-alpha)
        term3 = (Determinant(x))**(alpha - S(p + 1)/2)
        return term1 * term2 * term3

def MatrixGamma(symbol, alpha, beta, scale_matrix):
    """
    Creates a random variable with Matrix Gamma Distribution.

    The density of the said distribution can be found at [1].

    Parameters
    ==========

    alpha: Positive Real number
        Shape Parameter
    beta: Positive Real number
        Scale Parameter
    scale_matrix: Positive definite real square matrix
        Scale Matrix

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density, MatrixGamma
    >>> from sympy import MatrixSymbol, symbols
    >>> a, b = symbols('a b', positive=True)
    >>> M = MatrixGamma('M', a, b, [[2, 1], [1, 2]])
    >>> X = MatrixSymbol('X', 2, 2)
    >>> density(M)(X).doit()
    exp(Trace(Matrix([
    [-2/3,  1/3],
    [ 1/3, -2/3]])*X)/b)*Determinant(X)**(a - 3/2)/(3**a*sqrt(pi)*b**(2*a)*gamma(a)*gamma(a - 1/2))
    >>> density(M)([[1, 0], [0, 1]]).doit()
    exp(-4/(3*b))/(3**a*sqrt(pi)*b**(2*a)*gamma(a)*gamma(a - 1/2))


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Matrix_gamma_distribution

    """
    if isinstance(scale_matrix, list):
        scale_matrix = ImmutableMatrix(scale_matrix)
    return rv(symbol, MatrixGammaDistribution, (alpha, beta, scale_matrix))

#-------------------------------------------------------------------------------
# Wishart Distribution ---------------------------------------------------------

class WishartDistribution(MatrixDistribution):

    _argnames = ('n', 'scale_matrix')

    @staticmethod
    def check(n, scale_matrix):
        if not isinstance(scale_matrix, MatrixSymbol):
            _value_check(scale_matrix.is_positive_definite, "The shape "
                "matrix must be positive definite.")
        _value_check(scale_matrix.is_square, "Should "
        "be square matrix")
        _value_check(n.is_positive, "Shape parameter should be positive.")

    @property
    def set(self):
        k = self.scale_matrix.shape[0]
        return MatrixSet(k, k, S.Reals)

    @property
    def dimension(self):
        return self.scale_matrix.shape

    def pdf(self, x):
        n, scale_matrix = self.n, self.scale_matrix
        p = scale_matrix.shape[0]
        if isinstance(x, list):
            x = ImmutableMatrix(x)
        if not isinstance(x, (MatrixBase, MatrixSymbol)):
            raise ValueError("%s should be an isinstance of Matrix "
                    "or MatrixSymbol" % str(x))
        sigma_inv_x = - Inverse(scale_matrix)*x / S(2)
        term1 = exp(Trace(sigma_inv_x))/((2**(p*n/S(2))) * multigamma(n/S(2), p))
        term2 = (Determinant(scale_matrix))**(-n/S(2))
        term3 = (Determinant(x))**(S(n - p - 1)/2)
        return term1 * term2 * term3

def Wishart(symbol, n, scale_matrix):
    """
    Creates a random variable with Wishart Distribution.

    The density of the said distribution can be found at [1].

    Parameters
    ==========

    n: Positive Real number
        Represents degrees of freedom
    scale_matrix: Positive definite real square matrix
        Scale Matrix

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density, Wishart
    >>> from sympy import MatrixSymbol, symbols
    >>> n = symbols('n', positive=True)
    >>> W = Wishart('W', n, [[2, 1], [1, 2]])
    >>> X = MatrixSymbol('X', 2, 2)
    >>> density(W)(X).doit()
    exp(Trace(Matrix([
    [-1/3,  1/6],
    [ 1/6, -1/3]])*X))*Determinant(X)**(n/2 - 3/2)/(2**n*3**(n/2)*sqrt(pi)*gamma(n/2)*gamma(n/2 - 1/2))
    >>> density(W)([[1, 0], [0, 1]]).doit()
    exp(-2/3)/(2**n*3**(n/2)*sqrt(pi)*gamma(n/2)*gamma(n/2 - 1/2))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Wishart_distribution

    """
    if isinstance(scale_matrix, list):
        scale_matrix = ImmutableMatrix(scale_matrix)
    return rv(symbol, WishartDistribution, (n, scale_matrix))

#-------------------------------------------------------------------------------
# Matrix Normal distribution ---------------------------------------------------

class MatrixNormalDistribution(MatrixDistribution):

    _argnames = ('location_matrix', 'scale_matrix_1', 'scale_matrix_2')

    @staticmethod
    def check(location_matrix, scale_matrix_1, scale_matrix_2):
        if not isinstance(scale_matrix_1, MatrixSymbol):
            _value_check(scale_matrix_1.is_positive_definite, "The shape "
                "matrix must be positive definite.")
        if not isinstance(scale_matrix_2, MatrixSymbol):
            _value_check(scale_matrix_2.is_positive_definite, "The shape "
                "matrix must be positive definite.")
        _value_check(scale_matrix_1.is_square, "Scale matrix 1 should be "
        "be square matrix")
        _value_check(scale_matrix_2.is_square, "Scale matrix 2 should be "
        "be square matrix")
        n = location_matrix.shape[0]
        p = location_matrix.shape[1]
        _value_check(scale_matrix_1.shape[0] == n, "Scale matrix 1 should be"
        " of shape %s x %s"% (str(n), str(n)))
        _value_check(scale_matrix_2.shape[0] == p, "Scale matrix 2 should be"
        " of shape %s x %s"% (str(p), str(p)))

    @property
    def set(self):
        n, p = self.location_matrix.shape
        return MatrixSet(n, p, S.Reals)

    @property
    def dimension(self):
        return self.location_matrix.shape

    def pdf(self, x):
        M, U, V = self.location_matrix, self.scale_matrix_1, self.scale_matrix_2
        n, p = M.shape
        if isinstance(x, list):
            x = ImmutableMatrix(x)
        if not isinstance(x, (MatrixBase, MatrixSymbol)):
            raise ValueError("%s should be an isinstance of Matrix "
                    "or MatrixSymbol" % str(x))
        term1 = Inverse(V)*Transpose(x - M)*Inverse(U)*(x - M)
        num = exp(-Trace(term1)/S(2))
        den = (2*pi)**(S(n*p)/2) * Determinant(U)**(S(p)/2) * Determinant(V)**(S(n)/2)
        return num/den

def MatrixNormal(symbol, location_matrix, scale_matrix_1, scale_matrix_2):
    """
    Creates a random variable with Matrix Normal Distribution.

    The density of the said distribution can be found at [1].

    Parameters
    ==========

    location_matrix: Real ``n x p`` matrix
        Represents degrees of freedom
    scale_matrix_1: Positive definite matrix
        Scale Matrix of shape ``n x n``
    scale_matrix_2: Positive definite matrix
        Scale Matrix of shape ``p x p``

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy import MatrixSymbol
    >>> from sympy.stats import density, MatrixNormal
    >>> M = MatrixNormal('M', [[1, 2]], [1], [[1, 0], [0, 1]])
    >>> X = MatrixSymbol('X', 1, 2)
    >>> density(M)(X).doit()
    exp(-Trace((Matrix([
    [-1],
    [-2]]) + X.T)*(Matrix([[-1, -2]]) + X))/2)/(2*pi)
    >>> density(M)([[3, 4]]).doit()
    exp(-4)/(2*pi)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Matrix_normal_distribution

    """
    if isinstance(location_matrix, list):
        location_matrix = ImmutableMatrix(location_matrix)
    if isinstance(scale_matrix_1, list):
        scale_matrix_1 = ImmutableMatrix(scale_matrix_1)
    if isinstance(scale_matrix_2, list):
        scale_matrix_2 = ImmutableMatrix(scale_matrix_2)
    args = (location_matrix, scale_matrix_1, scale_matrix_2)
    return rv(symbol, MatrixNormalDistribution, args)

#-------------------------------------------------------------------------------
# Matrix Student's T distribution ---------------------------------------------------

class MatrixStudentTDistribution(MatrixDistribution):

    _argnames = ('nu', 'location_matrix', 'scale_matrix_1', 'scale_matrix_2')

    @staticmethod
    def check(nu, location_matrix, scale_matrix_1, scale_matrix_2):
        if not isinstance(scale_matrix_1, MatrixSymbol):
            _value_check(scale_matrix_1.is_positive_definite != False, "The shape "
                                                              "matrix must be positive definite.")
        if not isinstance(scale_matrix_2, MatrixSymbol):
            _value_check(scale_matrix_2.is_positive_definite != False, "The shape "
                                                              "matrix must be positive definite.")
        _value_check(scale_matrix_1.is_square != False, "Scale matrix 1 should be "
                                               "be square matrix")
        _value_check(scale_matrix_2.is_square != False, "Scale matrix 2 should be "
                                               "be square matrix")
        n = location_matrix.shape[0]
        p = location_matrix.shape[1]
        _value_check(scale_matrix_1.shape[0] == p, "Scale matrix 1 should be"
                                                   " of shape %s x %s" % (str(p), str(p)))
        _value_check(scale_matrix_2.shape[0] == n, "Scale matrix 2 should be"
                                                   " of shape %s x %s" % (str(n), str(n)))
        _value_check(nu.is_positive != False, "Degrees of freedom must be positive")

    @property
    def set(self):
        n, p = self.location_matrix.shape
        return MatrixSet(n, p, S.Reals)

    @property
    def dimension(self):
        return self.location_matrix.shape

    def pdf(self, x):
        from sympy.matrices.dense import eye
        if isinstance(x, list):
            x = ImmutableMatrix(x)
        if not isinstance(x, (MatrixBase, MatrixSymbol)):
            raise ValueError("%s should be an isinstance of Matrix "
                             "or MatrixSymbol" % str(x))
        nu, M, Omega, Sigma = self.nu, self.location_matrix, self.scale_matrix_1, self.scale_matrix_2
        n, p = M.shape

        K = multigamma((nu + n + p - 1)/2, p) * Determinant(Omega)**(-n/2) * Determinant(Sigma)**(-p/2) \
            / ((pi)**(n*p/2) * multigamma((nu + p - 1)/2, p))
        return K * (Determinant(eye(n) + Inverse(Sigma)*(x - M)*Inverse(Omega)*Transpose(x - M))) \
               **(-(nu + n + p -1)/2)



def MatrixStudentT(symbol, nu, location_matrix, scale_matrix_1, scale_matrix_2):
    """
    Creates a random variable with Matrix Gamma Distribution.

    The density of the said distribution can be found at [1].

    Parameters
    ==========

    nu: Positive Real number
        degrees of freedom
    location_matrix: Positive definite real square matrix
        Location Matrix of shape ``n x p``
    scale_matrix_1: Positive definite real square matrix
        Scale Matrix of shape ``p x p``
    scale_matrix_2: Positive definite real square matrix
        Scale Matrix of shape ``n x n``

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy import MatrixSymbol,symbols
    >>> from sympy.stats import density, MatrixStudentT
    >>> v = symbols('v',positive=True)
    >>> M = MatrixStudentT('M', v, [[1, 2]], [[1, 0], [0, 1]], [1])
    >>> X = MatrixSymbol('X', 1, 2)
    >>> density(M)(X)
    gamma(v/2 + 1)*Determinant((Matrix([[-1, -2]]) + X)*(Matrix([
    [-1],
    [-2]]) + X.T) + Matrix([[1]]))**(-v/2 - 1)/(pi**1.0*gamma(v/2)*Determinant(Matrix([[1]]))**1.0*Determinant(Matrix([
    [1, 0],
    [0, 1]]))**0.5)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Matrix_t-distribution

    """
    if isinstance(location_matrix, list):
        location_matrix = ImmutableMatrix(location_matrix)
    if isinstance(scale_matrix_1, list):
        scale_matrix_1 = ImmutableMatrix(scale_matrix_1)
    if isinstance(scale_matrix_2, list):
        scale_matrix_2 = ImmutableMatrix(scale_matrix_2)
    args = (nu, location_matrix, scale_matrix_1, scale_matrix_2)
    return rv(symbol, MatrixStudentTDistribution, args)
