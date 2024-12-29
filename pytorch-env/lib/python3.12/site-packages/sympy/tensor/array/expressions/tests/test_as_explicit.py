from sympy.core.symbol import Symbol
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.tensor.array.arrayop import (permutedims, tensorcontraction, tensordiagonal, tensorproduct)
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
from sympy.tensor.array.expressions.array_expressions import ZeroArray, OneArray, ArraySymbol, \
    ArrayTensorProduct, PermuteDims, ArrayDiagonal, ArrayContraction, ArrayAdd
from sympy.testing.pytest import raises


def test_array_as_explicit_call():

    assert ZeroArray(3, 2, 4).as_explicit() == ImmutableDenseNDimArray.zeros(3, 2, 4)
    assert OneArray(3, 2, 4).as_explicit() == ImmutableDenseNDimArray([1 for i in range(3*2*4)]).reshape(3, 2, 4)

    k = Symbol("k")
    X = ArraySymbol("X", (k, 3, 2))
    raises(ValueError, lambda: X.as_explicit())
    raises(ValueError, lambda: ZeroArray(k, 2, 3).as_explicit())
    raises(ValueError, lambda: OneArray(2, k, 2).as_explicit())

    A = ArraySymbol("A", (3, 3))
    B = ArraySymbol("B", (3, 3))

    texpr = tensorproduct(A, B)
    assert isinstance(texpr, ArrayTensorProduct)
    assert texpr.as_explicit() == tensorproduct(A.as_explicit(), B.as_explicit())

    texpr = tensorcontraction(A, (0, 1))
    assert isinstance(texpr, ArrayContraction)
    assert texpr.as_explicit() == A[0, 0] + A[1, 1] + A[2, 2]

    texpr = tensordiagonal(A, (0, 1))
    assert isinstance(texpr, ArrayDiagonal)
    assert texpr.as_explicit() == ImmutableDenseNDimArray([A[0, 0], A[1, 1], A[2, 2]])

    texpr = permutedims(A, [1, 0])
    assert isinstance(texpr, PermuteDims)
    assert texpr.as_explicit() == permutedims(A.as_explicit(), [1, 0])


def test_array_as_explicit_matrix_symbol():

    A = MatrixSymbol("A", 3, 3)
    B = MatrixSymbol("B", 3, 3)

    texpr = tensorproduct(A, B)
    assert isinstance(texpr, ArrayTensorProduct)
    assert texpr.as_explicit() == tensorproduct(A.as_explicit(), B.as_explicit())

    texpr = tensorcontraction(A, (0, 1))
    assert isinstance(texpr, ArrayContraction)
    assert texpr.as_explicit() == A[0, 0] + A[1, 1] + A[2, 2]

    texpr = tensordiagonal(A, (0, 1))
    assert isinstance(texpr, ArrayDiagonal)
    assert texpr.as_explicit() == ImmutableDenseNDimArray([A[0, 0], A[1, 1], A[2, 2]])

    texpr = permutedims(A, [1, 0])
    assert isinstance(texpr, PermuteDims)
    assert texpr.as_explicit() == permutedims(A.as_explicit(), [1, 0])

    expr = ArrayAdd(ArrayTensorProduct(A, B), ArrayTensorProduct(B, A))
    assert expr.as_explicit() == expr.args[0].as_explicit() + expr.args[1].as_explicit()
