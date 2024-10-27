from sympy.core import Lambda, S, symbols
from sympy.concrete import Sum
from sympy.functions import adjoint, conjugate, transpose
from sympy.matrices import eye, Matrix, ShapeError, ImmutableMatrix
from sympy.matrices.expressions import (
    Adjoint, Identity, FunctionMatrix, MatrixExpr, MatrixSymbol, Trace,
    ZeroMatrix, trace, MatPow, MatAdd, MatMul
)
from sympy.matrices.expressions.special import OneMatrix
from sympy.testing.pytest import raises
from sympy.abc import i


n = symbols('n', integer=True)
A = MatrixSymbol('A', n, n)
B = MatrixSymbol('B', n, n)
C = MatrixSymbol('C', 3, 4)


def test_Trace():
    assert isinstance(Trace(A), Trace)
    assert not isinstance(Trace(A), MatrixExpr)
    raises(ShapeError, lambda: Trace(C))
    assert trace(eye(3)) == 3
    assert trace(Matrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9])) == 15

    assert adjoint(Trace(A)) == trace(Adjoint(A))
    assert conjugate(Trace(A)) == trace(Adjoint(A))
    assert transpose(Trace(A)) == Trace(A)

    _ = A / Trace(A)  # Make sure this is possible

    # Some easy simplifications
    assert trace(Identity(5)) == 5
    assert trace(ZeroMatrix(5, 5)) == 0
    assert trace(OneMatrix(1, 1)) == 1
    assert trace(OneMatrix(2, 2)) == 2
    assert trace(OneMatrix(n, n)) == n
    assert trace(2*A*B) == 2*Trace(A*B)
    assert trace(A.T) == trace(A)

    i, j = symbols('i j')
    F = FunctionMatrix(3, 3, Lambda((i, j), i + j))
    assert trace(F) == (0 + 0) + (1 + 1) + (2 + 2)

    raises(TypeError, lambda: Trace(S.One))

    assert Trace(A).arg is A

    assert str(trace(A)) == str(Trace(A).doit())

    assert Trace(A).is_commutative is True

def test_Trace_A_plus_B():
    assert trace(A + B) == Trace(A) + Trace(B)
    assert Trace(A + B).arg == MatAdd(A, B)
    assert Trace(A + B).doit() == Trace(A) + Trace(B)


def test_Trace_MatAdd_doit():
    # See issue #9028
    X = ImmutableMatrix([[1, 2, 3]]*3)
    Y = MatrixSymbol('Y', 3, 3)
    q = MatAdd(X, 2*X, Y, -3*Y)
    assert Trace(q).arg == q
    assert Trace(q).doit() == 18 - 2*Trace(Y)


def test_Trace_MatPow_doit():
    X = Matrix([[1, 2], [3, 4]])
    assert Trace(X).doit() == 5
    q = MatPow(X, 2)
    assert Trace(q).arg == q
    assert Trace(q).doit() == 29


def test_Trace_MutableMatrix_plus():
    # See issue #9043
    X = Matrix([[1, 2], [3, 4]])
    assert Trace(X) + Trace(X) == 2*Trace(X)


def test_Trace_doit_deep_False():
    X = Matrix([[1, 2], [3, 4]])
    q = MatPow(X, 2)
    assert Trace(q).doit(deep=False).arg == q
    q = MatAdd(X, 2*X)
    assert Trace(q).doit(deep=False).arg == q
    q = MatMul(X, 2*X)
    assert Trace(q).doit(deep=False).arg == q


def test_trace_constant_factor():
    # Issue 9052: gave 2*Trace(MatMul(A)) instead of 2*Trace(A)
    assert trace(2*A) == 2*Trace(A)
    X = ImmutableMatrix([[1, 2], [3, 4]])
    assert trace(MatMul(2, X)) == 10


def test_trace_rewrite():
    assert trace(A).rewrite(Sum) == Sum(A[i, i], (i, 0, n - 1))
    assert trace(eye(3)).rewrite(Sum) == 3


def test_trace_normalize():
    assert Trace(B*A) != Trace(A*B)
    assert Trace(B*A)._normalize() == Trace(A*B)
    assert Trace(B*A.T)._normalize() == Trace(A*B.T)


def test_trace_as_explicit():
    raises(ValueError, lambda: Trace(A).as_explicit())

    X = MatrixSymbol("X", 3, 3)
    assert Trace(X).as_explicit() == X[0, 0] + X[1, 1] + X[2, 2]
    assert Trace(eye(3)).as_explicit() == 3
