from sympy import Sum, Dummy, sin
from sympy.tensor.array.expressions import ArraySymbol, ArrayTensorProduct, ArrayContraction, PermuteDims, \
    ArrayDiagonal, ArrayAdd, OneArray, ZeroArray, convert_indexed_to_array, ArrayElementwiseApplyFunc, Reshape
from sympy.tensor.array.expressions.from_array_to_indexed import convert_array_to_indexed

from sympy.abc import i, j, k, l, m, n, o


def test_convert_array_to_indexed_main():
    A = ArraySymbol("A", (3, 3, 3))
    B = ArraySymbol("B", (3, 3))
    C = ArraySymbol("C", (3, 3))

    d_ = Dummy("d_")

    assert convert_array_to_indexed(A, [i, j, k]) == A[i, j, k]

    expr = ArrayTensorProduct(A, B, C)
    conv = convert_array_to_indexed(expr, [i,j,k,l,m,n,o])
    assert conv == A[i,j,k]*B[l,m]*C[n,o]
    assert convert_indexed_to_array(conv, [i,j,k,l,m,n,o]) == expr

    expr = ArrayContraction(A, (0, 2))
    assert convert_array_to_indexed(expr, [i]).dummy_eq(Sum(A[d_, i, d_], (d_, 0, 2)))

    expr = ArrayDiagonal(A, (0, 2))
    assert convert_array_to_indexed(expr, [i, j]) == A[j, i, j]

    expr = PermuteDims(A, [1, 2, 0])
    conv = convert_array_to_indexed(expr, [i, j, k])
    assert conv == A[k, i, j]
    assert convert_indexed_to_array(conv, [i, j, k]) == expr

    expr = ArrayAdd(B, C, PermuteDims(C, [1, 0]))
    conv = convert_array_to_indexed(expr, [i, j])
    assert conv == B[i, j] + C[i, j] + C[j, i]
    assert convert_indexed_to_array(conv, [i, j]) == expr

    expr = ArrayElementwiseApplyFunc(sin, A)
    conv = convert_array_to_indexed(expr, [i, j, k])
    assert conv == sin(A[i, j, k])
    assert convert_indexed_to_array(conv, [i, j, k]).dummy_eq(expr)

    assert convert_array_to_indexed(OneArray(3, 3), [i, j]) == 1
    assert convert_array_to_indexed(ZeroArray(3, 3), [i, j]) == 0

    expr = Reshape(A, (27,))
    assert convert_array_to_indexed(expr, [i]) == A[i // 9, i // 3 % 3, i % 3]

    X = ArraySymbol("X", (2, 3, 4, 5, 6))
    expr = Reshape(X, (2*3*4*5*6,))
    assert convert_array_to_indexed(expr, [i]) == X[i // 360, i // 120 % 3, i // 30 % 4, i // 6 % 5, i % 6]

    expr = Reshape(X, (4, 9, 2, 2, 5))
    one_index = 180*i + 20*j + 10*k + 5*l + m
    expected = X[one_index // (3*4*5*6), one_index // (4*5*6) % 3, one_index // (5*6) % 4, one_index // 6 % 5, one_index % 6]
    assert convert_array_to_indexed(expr, [i, j, k, l, m]) == expected

    X = ArraySymbol("X", (2*3*5,))
    expr = Reshape(X, (2, 3, 5))
    assert convert_array_to_indexed(expr, [i, j, k]) == X[15*i + 5*j + k]
