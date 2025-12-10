from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import Identity
from sympy.matrices.expressions.applyfunc import ElementwiseApplyFunction
from sympy.tensor.array.expressions.array_expressions import ArraySymbol, ArrayTensorProduct, \
    PermuteDims, ArrayDiagonal, ArrayElementwiseApplyFunc, ArrayContraction, _permute_dims, Reshape
from sympy.tensor.array.expressions.arrayexpr_derivatives import array_derive

k = symbols("k")

I = Identity(k)
X = MatrixSymbol("X", k, k)
x = MatrixSymbol("x", k, 1)

A = MatrixSymbol("A", k, k)
B = MatrixSymbol("B", k, k)
C = MatrixSymbol("C", k, k)
D = MatrixSymbol("D", k, k)

A1 = ArraySymbol("A", (3, 2, k))


def test_arrayexpr_derivatives1():

    res = array_derive(X, X)
    assert res == PermuteDims(ArrayTensorProduct(I, I), [0, 2, 1, 3])

    cg = ArrayTensorProduct(A, X, B)
    res = array_derive(cg, X)
    assert res == _permute_dims(
        ArrayTensorProduct(I, A, I, B),
        [0, 4, 2, 3, 1, 5, 6, 7])

    cg = ArrayContraction(X, (0, 1))
    res = array_derive(cg, X)
    assert res == ArrayContraction(ArrayTensorProduct(I, I), (1, 3))

    cg = ArrayDiagonal(X, (0, 1))
    res = array_derive(cg, X)
    assert res == ArrayDiagonal(ArrayTensorProduct(I, I), (1, 3))

    cg = ElementwiseApplyFunction(sin, X)
    res = array_derive(cg, X)
    assert res.dummy_eq(ArrayDiagonal(
        ArrayTensorProduct(
            ElementwiseApplyFunction(cos, X),
            I,
            I
        ), (0, 3), (1, 5)))

    cg = ArrayElementwiseApplyFunc(sin, X)
    res = array_derive(cg, X)
    assert res.dummy_eq(ArrayDiagonal(
        ArrayTensorProduct(
            I,
            I,
            ArrayElementwiseApplyFunc(cos, X)
        ), (1, 4), (3, 5)))

    res = array_derive(A1, A1)
    assert res == PermuteDims(
        ArrayTensorProduct(Identity(3), Identity(2), Identity(k)),
        [0, 2, 4, 1, 3, 5]
    )

    cg = ArrayElementwiseApplyFunc(sin, A1)
    res = array_derive(cg, A1)
    assert res.dummy_eq(ArrayDiagonal(
        ArrayTensorProduct(
            Identity(3), Identity(2), Identity(k),
            ArrayElementwiseApplyFunc(cos, A1)
        ), (1, 6), (3, 7), (5, 8)
    ))

    cg = Reshape(A, (k**2,))
    res = array_derive(cg, A)
    assert res == Reshape(PermuteDims(ArrayTensorProduct(I, I), [0, 2, 1, 3]), (k, k, k**2))
