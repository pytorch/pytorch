from sympy.matrices.dense import Matrix, eye
from sympy.matrices.exceptions import ShapeError
from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.special import Identity, OneMatrix, ZeroMatrix
from sympy.core import symbols
from sympy.testing.pytest import raises, warns_deprecated_sympy

from sympy.matrices import MatrixSymbol
from sympy.matrices.expressions import (HadamardProduct, hadamard_product, HadamardPower, hadamard_power)

n, m, k = symbols('n,m,k')
Z = MatrixSymbol('Z', n, n)
A = MatrixSymbol('A', n, m)
B = MatrixSymbol('B', n, m)
C = MatrixSymbol('C', m, k)


def test_HadamardProduct():
    assert HadamardProduct(A, B, A).shape == A.shape

    raises(TypeError,  lambda: HadamardProduct(A, n))
    raises(TypeError,  lambda: HadamardProduct(A, 1))

    assert HadamardProduct(A, 2*B, -A)[1, 1] == \
            -2 * A[1, 1] * B[1, 1] * A[1, 1]

    mix = HadamardProduct(Z*A, B)*C
    assert mix.shape == (n, k)

    assert set(HadamardProduct(A, B, A).T.args) == {A.T, A.T, B.T}


def test_HadamardProduct_isnt_commutative():
    assert HadamardProduct(A, B) != HadamardProduct(B, A)


def test_mixed_indexing():
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)
    Z = MatrixSymbol('Z', 2, 2)

    assert (X*HadamardProduct(Y, Z))[0, 0] == \
            X[0, 0]*Y[0, 0]*Z[0, 0] + X[0, 1]*Y[1, 0]*Z[1, 0]


def test_canonicalize():
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)
    with warns_deprecated_sympy():
        expr = HadamardProduct(X, check=False)
    assert isinstance(expr, HadamardProduct)
    expr2 = expr.doit() # unpack is called
    assert isinstance(expr2, MatrixSymbol)
    Z = ZeroMatrix(2, 2)
    U = OneMatrix(2, 2)
    assert HadamardProduct(Z, X).doit() == Z
    assert HadamardProduct(U, X, X, U).doit() == HadamardPower(X, 2)
    assert HadamardProduct(X, U, Y).doit() == HadamardProduct(X, Y)
    assert HadamardProduct(X, Z, U, Y).doit() == Z


def test_hadamard():
    m, n, p = symbols('m, n, p', integer=True)
    A = MatrixSymbol('A', m, n)
    B = MatrixSymbol('B', m, n)
    X = MatrixSymbol('X', m, m)
    I = Identity(m)

    raises(TypeError, lambda: hadamard_product())
    assert hadamard_product(A) == A
    assert isinstance(hadamard_product(A, B), HadamardProduct)
    assert hadamard_product(A, B).doit() == hadamard_product(A, B)
    assert hadamard_product(X, I) == HadamardProduct(I, X)
    assert isinstance(hadamard_product(X, I), HadamardProduct)

    a = MatrixSymbol("a", k, 1)
    expr = MatAdd(ZeroMatrix(k, 1), OneMatrix(k, 1))
    expr = HadamardProduct(expr, a)
    assert expr.doit() == a

    raises(ValueError, lambda: HadamardProduct())


def test_hadamard_product_with_explicit_mat():
    A = MatrixSymbol("A", 3, 3).as_explicit()
    B = MatrixSymbol("B", 3, 3).as_explicit()
    X = MatrixSymbol("X", 3, 3)
    expr = hadamard_product(A, B)
    ret = Matrix([i*j for i, j in zip(A, B)]).reshape(3, 3)
    assert expr == ret
    expr = hadamard_product(A, X, B)
    assert expr == HadamardProduct(ret, X)
    expr = hadamard_product(eye(3), A)
    assert expr == Matrix([[A[0, 0], 0, 0], [0, A[1, 1], 0], [0, 0, A[2, 2]]])
    expr = hadamard_product(eye(3), eye(3))
    assert expr == eye(3)


def test_hadamard_power():
    m, n, p = symbols('m, n, p', integer=True)
    A = MatrixSymbol('A', m, n)

    assert hadamard_power(A, 1) == A
    assert isinstance(hadamard_power(A, 2), HadamardPower)
    assert hadamard_power(A, n).T == hadamard_power(A.T, n)
    assert hadamard_power(A, n)[0, 0] == A[0, 0]**n
    assert hadamard_power(m, n) == m**n
    raises(ValueError, lambda: hadamard_power(A, A))


def test_hadamard_power_explicit():
    A = MatrixSymbol('A', 2, 2)
    B = MatrixSymbol('B', 2, 2)
    a, b = symbols('a b')

    assert HadamardPower(a, b) == a**b

    assert HadamardPower(a, B).as_explicit() == \
        Matrix([
            [a**B[0, 0], a**B[0, 1]],
            [a**B[1, 0], a**B[1, 1]]])

    assert HadamardPower(A, b).as_explicit() == \
        Matrix([
            [A[0, 0]**b, A[0, 1]**b],
            [A[1, 0]**b, A[1, 1]**b]])

    assert HadamardPower(A, B).as_explicit() == \
        Matrix([
            [A[0, 0]**B[0, 0], A[0, 1]**B[0, 1]],
            [A[1, 0]**B[1, 0], A[1, 1]**B[1, 1]]])


def test_shape_error():
    A = MatrixSymbol('A', 2, 3)
    B = MatrixSymbol('B', 3, 3)
    raises(ShapeError, lambda: HadamardProduct(A, B))
    raises(ShapeError, lambda: HadamardPower(A, B))
    A = MatrixSymbol('A', 3, 2)
    raises(ShapeError, lambda: HadamardProduct(A, B))
    raises(ShapeError, lambda: HadamardPower(A, B))
