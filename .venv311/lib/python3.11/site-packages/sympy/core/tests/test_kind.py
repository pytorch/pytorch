from sympy.core.add import Add
from sympy.core.kind import NumberKind, UndefinedKind
from sympy.core.mul import Mul
from sympy.core.numbers import pi, zoo, I, AlgebraicNumber
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.integrals.integrals import Integral
from sympy.core.function import Derivative
from sympy.matrices import (Matrix, SparseMatrix, ImmutableMatrix,
    ImmutableSparseMatrix, MatrixSymbol, MatrixKind, MatMul)

comm_x = Symbol('x')
noncomm_x = Symbol('x', commutative=False)

def test_NumberKind():
    assert S.One.kind is NumberKind
    assert pi.kind is NumberKind
    assert S.NaN.kind is NumberKind
    assert zoo.kind is NumberKind
    assert I.kind is NumberKind
    assert AlgebraicNumber(1).kind is NumberKind

def test_Add_kind():
    assert Add(2, 3, evaluate=False).kind is NumberKind
    assert Add(2,comm_x).kind is NumberKind
    assert Add(2,noncomm_x).kind is UndefinedKind

def test_mul_kind():
    assert Mul(2,comm_x, evaluate=False).kind is NumberKind
    assert Mul(2,3, evaluate=False).kind is NumberKind
    assert Mul(noncomm_x,2, evaluate=False).kind is UndefinedKind
    assert Mul(2,noncomm_x, evaluate=False).kind is UndefinedKind

def test_Symbol_kind():
    assert comm_x.kind is NumberKind
    assert noncomm_x.kind is UndefinedKind

def test_Integral_kind():
    A = MatrixSymbol('A', 2,2)
    assert Integral(comm_x, comm_x).kind is NumberKind
    assert Integral(A, comm_x).kind is MatrixKind(NumberKind)

def test_Derivative_kind():
    A = MatrixSymbol('A', 2,2)
    assert Derivative(comm_x, comm_x).kind is NumberKind
    assert Derivative(A, comm_x).kind is MatrixKind(NumberKind)

def test_Matrix_kind():
    classes = (Matrix, SparseMatrix, ImmutableMatrix, ImmutableSparseMatrix)
    for cls in classes:
        m = cls.zeros(3, 2)
        assert m.kind is MatrixKind(NumberKind)

def test_MatMul_kind():
    M = Matrix([[1,2],[3,4]])
    assert MatMul(2, M).kind is MatrixKind(NumberKind)
    assert MatMul(comm_x, M).kind is MatrixKind(NumberKind)
