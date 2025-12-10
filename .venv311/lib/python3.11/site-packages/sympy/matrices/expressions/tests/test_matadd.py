from sympy.matrices.expressions import MatrixSymbol, MatAdd, MatPow, MatMul
from sympy.matrices.expressions.special import GenericZeroMatrix, ZeroMatrix
from sympy.matrices.exceptions import ShapeError
from sympy.matrices import eye, ImmutableMatrix
from sympy.core import Add, Basic, S
from sympy.core.add import add
from sympy.testing.pytest import XFAIL, raises

X = MatrixSymbol('X', 2, 2)
Y = MatrixSymbol('Y', 2, 2)

def test_evaluate():
    assert MatAdd(X, X, evaluate=True) == add(X, X, evaluate=True) == MatAdd(X, X).doit()

def test_sort_key():
    assert MatAdd(Y, X).doit().args == add(Y, X).doit().args == (X, Y)


def test_matadd_sympify():
    assert isinstance(MatAdd(eye(1), eye(1)).args[0], Basic)
    assert isinstance(add(eye(1), eye(1)).args[0], Basic)


def test_matadd_of_matrices():
    assert MatAdd(eye(2), 4*eye(2), eye(2)).doit() == ImmutableMatrix(6*eye(2))
    assert add(eye(2), 4*eye(2), eye(2)).doit() == ImmutableMatrix(6*eye(2))


def test_doit_args():
    A = ImmutableMatrix([[1, 2], [3, 4]])
    B = ImmutableMatrix([[2, 3], [4, 5]])
    assert MatAdd(A, MatPow(B, 2)).doit() == A + B**2
    assert MatAdd(A, MatMul(A, B)).doit() == A + A*B
    assert (MatAdd(A, X, MatMul(A, B), Y, MatAdd(2*A, B)).doit() ==
    add(A, X, MatMul(A, B), Y, add(2*A, B)).doit() ==
    MatAdd(3*A + A*B + B, X, Y))


def test_generic_identity():
    assert MatAdd.identity == GenericZeroMatrix()
    assert MatAdd.identity != S.Zero


def test_zero_matrix_add():
    assert Add(ZeroMatrix(2, 2), ZeroMatrix(2, 2)) == ZeroMatrix(2, 2)

@XFAIL
def test_matrix_Add_with_scalar():
    raises(TypeError, lambda: Add(0, ZeroMatrix(2, 2)))


def test_shape_error():
    A = MatrixSymbol('A', 2, 3)
    B = MatrixSymbol('B', 3, 3)
    raises(ShapeError, lambda: MatAdd(A, B))

    A = MatrixSymbol('A', 3, 2)
    raises(ShapeError, lambda: MatAdd(A, B))
