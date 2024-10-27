from sympy.stats import Expectation, Normal, Variance, Covariance
from sympy.testing.pytest import raises
from sympy.core.symbol import symbols
from sympy.matrices.exceptions import ShapeError
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import ZeroMatrix
from sympy.stats.rv import RandomMatrixSymbol
from sympy.stats.symbolic_multivariate_probability import (ExpectationMatrix,
                            VarianceMatrix, CrossCovarianceMatrix)

j, k = symbols("j,k")

A = MatrixSymbol("A", k, k)
B = MatrixSymbol("B", k, k)
C = MatrixSymbol("C", k, k)
D = MatrixSymbol("D", k, k)

a = MatrixSymbol("a", k, 1)
b = MatrixSymbol("b", k, 1)

A2 = MatrixSymbol("A2", 2, 2)
B2 = MatrixSymbol("B2", 2, 2)

X = RandomMatrixSymbol("X", k, 1)
Y = RandomMatrixSymbol("Y", k, 1)
Z = RandomMatrixSymbol("Z", k, 1)
W = RandomMatrixSymbol("W", k, 1)

R = RandomMatrixSymbol("R", k, k)

X2 = RandomMatrixSymbol("X2", 2, 1)

normal = Normal("normal", 0, 1)

m1 = Matrix([
    [1, j*Normal("normal2", 2, 1)],
    [normal, 0]
])

def test_multivariate_expectation():
    expr = Expectation(a)
    assert expr == Expectation(a) == ExpectationMatrix(a)
    assert expr.expand() == a

    expr = Expectation(X)
    assert expr == Expectation(X) == ExpectationMatrix(X)
    assert expr.shape == (k, 1)
    assert expr.rows == k
    assert expr.cols == 1
    assert isinstance(expr, ExpectationMatrix)

    expr = Expectation(A*X + b)
    assert expr == ExpectationMatrix(A*X + b)
    assert expr.expand() == A*ExpectationMatrix(X) + b
    assert isinstance(expr, ExpectationMatrix)
    assert expr.shape == (k, 1)

    expr = Expectation(m1*X2)
    assert expr.expand() == expr

    expr = Expectation(A2*m1*B2*X2)
    assert expr.args[0].args == (A2, m1, B2, X2)
    assert expr.expand() == A2*ExpectationMatrix(m1*B2*X2)

    expr = Expectation((X + Y)*(X - Y).T)
    assert expr.expand() == ExpectationMatrix(X*X.T) - ExpectationMatrix(X*Y.T) +\
                ExpectationMatrix(Y*X.T) - ExpectationMatrix(Y*Y.T)

    expr = Expectation(A*X + B*Y)
    assert expr.expand() == A*ExpectationMatrix(X) + B*ExpectationMatrix(Y)

    assert Expectation(m1).doit() == Matrix([[1, 2*j], [0, 0]])

    x1 = Matrix([
    [Normal('N11', 11, 1), Normal('N12', 12, 1)],
    [Normal('N21', 21, 1), Normal('N22', 22, 1)]
    ])
    x2 = Matrix([
    [Normal('M11', 1, 1), Normal('M12', 2, 1)],
    [Normal('M21', 3, 1), Normal('M22', 4, 1)]
    ])

    assert Expectation(Expectation(x1 + x2)).doit(deep=False) == ExpectationMatrix(x1 + x2)
    assert Expectation(Expectation(x1 + x2)).doit() == Matrix([[12, 14], [24, 26]])


def test_multivariate_variance():
    raises(ShapeError, lambda: Variance(A))

    expr = Variance(a)
    assert expr == Variance(a) == VarianceMatrix(a)
    assert expr.expand() == ZeroMatrix(k, k)
    expr = Variance(a.T)
    assert expr == Variance(a.T) == VarianceMatrix(a.T)
    assert expr.expand() == ZeroMatrix(k, k)

    expr = Variance(X)
    assert expr == Variance(X) == VarianceMatrix(X)
    assert expr.shape == (k, k)
    assert expr.rows == k
    assert expr.cols == k
    assert isinstance(expr, VarianceMatrix)

    expr = Variance(A*X)
    assert expr == VarianceMatrix(A*X)
    assert expr.expand() == A*VarianceMatrix(X)*A.T
    assert isinstance(expr, VarianceMatrix)
    assert expr.shape == (k, k)

    expr = Variance(A*B*X)
    assert expr.expand() == A*B*VarianceMatrix(X)*B.T*A.T

    expr = Variance(m1*X2)
    assert expr.expand() == expr

    expr = Variance(A2*m1*B2*X2)
    assert expr.args[0].args == (A2, m1, B2, X2)
    assert expr.expand() == expr

    expr = Variance(A*X + B*Y)
    assert expr.expand() == 2*A*CrossCovarianceMatrix(X, Y)*B.T +\
                    A*VarianceMatrix(X)*A.T + B*VarianceMatrix(Y)*B.T

def test_multivariate_crosscovariance():
    raises(ShapeError, lambda: Covariance(X, Y.T))
    raises(ShapeError, lambda: Covariance(X, A))


    expr = Covariance(a.T, b.T)
    assert expr.shape == (1, 1)
    assert expr.expand() == ZeroMatrix(1, 1)

    expr = Covariance(a, b)
    assert expr == Covariance(a, b) == CrossCovarianceMatrix(a, b)
    assert expr.expand() == ZeroMatrix(k, k)
    assert expr.shape == (k, k)
    assert expr.rows == k
    assert expr.cols == k
    assert isinstance(expr, CrossCovarianceMatrix)

    expr = Covariance(A*X + a, b)
    assert expr.expand() == ZeroMatrix(k, k)

    expr = Covariance(X, Y)
    assert isinstance(expr, CrossCovarianceMatrix)
    assert expr.expand() == expr

    expr = Covariance(X, X)
    assert isinstance(expr, CrossCovarianceMatrix)
    assert expr.expand() == VarianceMatrix(X)

    expr = Covariance(X + Y, Z)
    assert isinstance(expr, CrossCovarianceMatrix)
    assert expr.expand() == CrossCovarianceMatrix(X, Z) + CrossCovarianceMatrix(Y, Z)

    expr = Covariance(A*X, Y)
    assert isinstance(expr, CrossCovarianceMatrix)
    assert expr.expand() == A*CrossCovarianceMatrix(X, Y)

    expr = Covariance(X, B*Y)
    assert isinstance(expr, CrossCovarianceMatrix)
    assert expr.expand() == CrossCovarianceMatrix(X, Y)*B.T

    expr = Covariance(A*X + a, B.T*Y + b)
    assert isinstance(expr, CrossCovarianceMatrix)
    assert expr.expand() == A*CrossCovarianceMatrix(X, Y)*B

    expr = Covariance(A*X + B*Y + a, C.T*Z + D.T*W + b)
    assert isinstance(expr, CrossCovarianceMatrix)
    assert expr.expand() == A*CrossCovarianceMatrix(X, W)*D + A*CrossCovarianceMatrix(X, Z)*C \
        + B*CrossCovarianceMatrix(Y, W)*D + B*CrossCovarianceMatrix(Y, Z)*C
