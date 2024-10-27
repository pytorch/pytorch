from sympy.concrete.products import Product
from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.gamma_functions import gamma
from sympy.matrices import Determinant, Matrix, Trace, MatrixSymbol, MatrixSet
from sympy.stats import density, sample
from sympy.stats.matrix_distributions import (MatrixGammaDistribution,
                MatrixGamma, MatrixPSpace, Wishart, MatrixNormal, MatrixStudentT)
from sympy.testing.pytest import raises, skip
from sympy.external import import_module


def test_MatrixPSpace():
    M = MatrixGammaDistribution(1, 2, [[2, 1], [1, 2]])
    MP = MatrixPSpace('M', M, 2, 2)
    assert MP.distribution == M
    raises(ValueError, lambda: MatrixPSpace('M', M, 1.2, 2))

def test_MatrixGamma():
    M = MatrixGamma('M', 1, 2, [[1, 0], [0, 1]])
    assert M.pspace.distribution.set == MatrixSet(2, 2, S.Reals)
    assert isinstance(density(M), MatrixGammaDistribution)
    X = MatrixSymbol('X', 2, 2)
    num = exp(Trace(Matrix([[-S(1)/2, 0], [0, -S(1)/2]])*X))
    assert density(M)(X).doit() == num/(4*pi*sqrt(Determinant(X)))
    assert density(M)([[2, 1], [1, 2]]).doit() == sqrt(3)*exp(-2)/(12*pi)
    X = MatrixSymbol('X', 1, 2)
    Y = MatrixSymbol('Y', 1, 2)
    assert density(M)([X, Y]).doit() == exp(-X[0, 0]/2 - Y[0, 1]/2)/(4*pi*sqrt(
                                X[0, 0]*Y[0, 1] - X[0, 1]*Y[0, 0]))
    # symbolic
    a, b = symbols('a b', positive=True)
    d = symbols('d', positive=True, integer=True)
    Y = MatrixSymbol('Y', d, d)
    Z = MatrixSymbol('Z', 2, 2)
    SM = MatrixSymbol('SM', d, d)
    M2 = MatrixGamma('M2', a, b, SM)
    M3 = MatrixGamma('M3', 2, 3, [[2, 1], [1, 2]])
    k = Dummy('k')
    exprd = pi**(-d*(d - 1)/4)*b**(-a*d)*exp(Trace((-1/b)*SM**(-1)*Y)
        )*Determinant(SM)**(-a)*Determinant(Y)**(a - d/2 - S(1)/2)/Product(
        gamma(-k/2 + a + S(1)/2), (k, 1, d))
    assert density(M2)(Y).dummy_eq(exprd)
    raises(NotImplementedError, lambda: density(M3 + M)(Z))
    raises(ValueError, lambda: density(M)(1))
    raises(ValueError, lambda: MatrixGamma('M', -1, 2, [[1, 0], [0, 1]]))
    raises(ValueError, lambda: MatrixGamma('M', -1, -2, [[1, 0], [0, 1]]))
    raises(ValueError, lambda: MatrixGamma('M', -1, 2, [[1, 0], [2, 1]]))
    raises(ValueError, lambda: MatrixGamma('M', -1, 2, [[1, 0], [0]]))

def test_Wishart():
    W = Wishart('W', 5, [[1, 0], [0, 1]])
    assert W.pspace.distribution.set == MatrixSet(2, 2, S.Reals)
    X = MatrixSymbol('X', 2, 2)
    term1 = exp(Trace(Matrix([[-S(1)/2, 0], [0, -S(1)/2]])*X))
    assert density(W)(X).doit() == term1 * Determinant(X)/(24*pi)
    assert density(W)([[2, 1], [1, 2]]).doit() == exp(-2)/(8*pi)
    n = symbols('n', positive=True)
    d = symbols('d', positive=True, integer=True)
    Y = MatrixSymbol('Y', d, d)
    SM = MatrixSymbol('SM', d, d)
    W = Wishart('W', n, SM)
    k = Dummy('k')
    exprd = 2**(-d*n/2)*pi**(-d*(d - 1)/4)*exp(Trace(-(S(1)/2)*SM**(-1)*Y)
    )*Determinant(SM)**(-n/2)*Determinant(Y)**(
    -d/2 + n/2 - S(1)/2)/Product(gamma(-k/2 + n/2 + S(1)/2), (k, 1, d))
    assert density(W)(Y).dummy_eq(exprd)
    raises(ValueError, lambda: density(W)(1))
    raises(ValueError, lambda: Wishart('W', -1, [[1, 0], [0, 1]]))
    raises(ValueError, lambda: Wishart('W', -1, [[1, 0], [2, 1]]))
    raises(ValueError, lambda: Wishart('W',  2, [[1, 0], [0]]))

def test_MatrixNormal():
    M = MatrixNormal('M', [[5, 6]], [4], [[2, 1], [1, 2]])
    assert M.pspace.distribution.set == MatrixSet(1, 2, S.Reals)
    X = MatrixSymbol('X', 1, 2)
    term1 = exp(-Trace(Matrix([[ S(2)/3, -S(1)/3], [-S(1)/3, S(2)/3]])*(
            Matrix([[-5], [-6]]) + X.T)*Matrix([[S(1)/4]])*(Matrix([[-5, -6]]) + X))/2)
    assert density(M)(X).doit() == (sqrt(3)) * term1/(24*pi)
    assert density(M)([[7, 8]]).doit() == sqrt(3)*exp(-S(1)/3)/(24*pi)
    d, n = symbols('d n', positive=True, integer=True)
    SM2 = MatrixSymbol('SM2', d, d)
    SM1 = MatrixSymbol('SM1', n, n)
    LM = MatrixSymbol('LM', n, d)
    Y = MatrixSymbol('Y', n, d)
    M = MatrixNormal('M', LM, SM1, SM2)
    exprd = (2*pi)**(-d*n/2)*exp(-Trace(SM2**(-1)*(-LM.T + Y.T)*SM1**(-1)*(-LM + Y)
        )/2)*Determinant(SM1)**(-d/2)*Determinant(SM2)**(-n/2)
    assert density(M)(Y).doit() == exprd
    raises(ValueError, lambda: density(M)(1))
    raises(ValueError, lambda: MatrixNormal('M', [1, 2], [[1, 0], [0, 1]], [[1, 0], [2, 1]]))
    raises(ValueError, lambda: MatrixNormal('M', [1, 2], [[1, 0], [2, 1]], [[1, 0], [0, 1]]))
    raises(ValueError, lambda: MatrixNormal('M', [1, 2], [[1, 0], [0, 1]], [[1, 0], [0, 1]]))
    raises(ValueError, lambda: MatrixNormal('M', [1, 2], [[1, 0], [2]], [[1, 0], [0, 1]]))
    raises(ValueError, lambda: MatrixNormal('M', [1, 2], [[1, 0], [2, 1]], [[1, 0], [0]]))
    raises(ValueError, lambda: MatrixNormal('M', [[1, 2]], [[1, 0], [0, 1]], [[1, 0]]))
    raises(ValueError, lambda: MatrixNormal('M', [[1, 2]], [1], [[1, 0]]))

def test_MatrixStudentT():
    M = MatrixStudentT('M', 2, [[5, 6]], [[2, 1], [1, 2]], [4])
    assert M.pspace.distribution.set == MatrixSet(1, 2, S.Reals)
    X = MatrixSymbol('X', 1, 2)
    D = pi ** (-1.0) * Determinant(Matrix([[4]])) ** (-1.0) * Determinant(Matrix([[2, 1], [1, 2]])) \
        ** (-0.5) / Determinant(Matrix([[S(1) / 4]]) * (Matrix([[-5, -6]]) + X)
                                * Matrix([[S(2) / 3, -S(1) / 3], [-S(1) / 3, S(2) / 3]]) * (
                                        Matrix([[-5], [-6]]) + X.T) + Matrix([[1]])) ** 2
    assert density(M)(X) == D

    v = symbols('v', positive=True)
    n, p = 1, 2
    Omega = MatrixSymbol('Omega', p, p)
    Sigma = MatrixSymbol('Sigma', n, n)
    Location = MatrixSymbol('Location', n, p)
    Y = MatrixSymbol('Y', n, p)
    M = MatrixStudentT('M', v, Location, Omega, Sigma)

    exprd = gamma(v/2 + 1)*Determinant(Matrix([[1]]) + Sigma**(-1)*(-Location + Y)*Omega**(-1)*(-Location.T + Y.T))**(-v/2 - 1) / \
            (pi*gamma(v/2)*sqrt(Determinant(Omega))*Determinant(Sigma))

    assert density(M)(Y) == exprd
    raises(ValueError, lambda: density(M)(1))
    raises(ValueError, lambda: MatrixStudentT('M', 1, [1, 2], [[1, 0], [0, 1]], [[1, 0], [2, 1]]))
    raises(ValueError, lambda: MatrixStudentT('M', 1, [1, 2], [[1, 0], [2, 1]], [[1, 0], [0, 1]]))
    raises(ValueError, lambda: MatrixStudentT('M', 1, [1, 2], [[1, 0], [0, 1]], [[1, 0], [0, 1]]))
    raises(ValueError, lambda: MatrixStudentT('M', 1, [1, 2], [[1, 0], [2]], [[1, 0], [0, 1]]))
    raises(ValueError, lambda: MatrixStudentT('M', 1, [1, 2], [[1, 0], [2, 1]], [[1], [2]]))
    raises(ValueError, lambda: MatrixStudentT('M', 1, [[1, 2]], [[1, 0], [0, 1]], [[1, 0]]))
    raises(ValueError, lambda: MatrixStudentT('M', 1, [[1, 2]], [1], [[1, 0]]))
    raises(ValueError, lambda: MatrixStudentT('M', -1, [1, 2], [[1, 0], [0, 1]], [4]))

def test_sample_scipy():
    distribs_scipy = [
        MatrixNormal('M', [[5, 6]], [4], [[2, 1], [1, 2]]),
        Wishart('W', 5, [[1, 0], [0, 1]])
    ]

    size = 5
    scipy = import_module('scipy')
    if not scipy:
        skip('Scipy not installed. Abort tests for _sample_scipy.')
    else:
        for X in distribs_scipy:
            samps = sample(X, size=size)
            for sam in samps:
                assert Matrix(sam) in X.pspace.distribution.set
        M = MatrixGamma('M', 1, 2, [[1, 0], [0, 1]])
        raises(NotImplementedError, lambda: sample(M, size=3))

def test_sample_pymc():
    distribs_pymc = [
        MatrixNormal('M', [[5, 6], [3, 4]], [[1, 0], [0, 1]], [[2, 1], [1, 2]]),
        Wishart('W', 7, [[2, 1], [1, 2]])
    ]
    size = 3
    pymc = import_module('pymc')
    if not pymc:
        skip('PyMC is not installed. Abort tests for _sample_pymc.')
    else:
        for X in distribs_pymc:
            samps = sample(X, size=size, library='pymc')
            for sam in samps:
                assert Matrix(sam) in X.pspace.distribution.set
        M = MatrixGamma('M', 1, 2, [[1, 0], [0, 1]])
        raises(NotImplementedError, lambda: sample(M, size=3))

def test_sample_seed():
    X = MatrixNormal('M', [[5, 6], [3, 4]], [[1, 0], [0, 1]], [[2, 1], [1, 2]])

    libraries = ['scipy', 'numpy', 'pymc']
    for lib in libraries:
        try:
            imported_lib = import_module(lib)
            if imported_lib:
                s0, s1, s2 = [], [], []
                s0 = sample(X, size=10, library=lib, seed=0)
                s1 = sample(X, size=10, library=lib, seed=0)
                s2 = sample(X, size=10, library=lib, seed=1)
                for i in range(10):
                    assert (s0[i] == s1[i]).all()
                    assert (s1[i] != s2[i]).all()

        except NotImplementedError:
            continue
