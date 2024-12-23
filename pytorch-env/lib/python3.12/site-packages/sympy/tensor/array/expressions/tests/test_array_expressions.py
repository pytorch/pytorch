import random

from sympy import tensordiagonal, eye, KroneckerDelta, Array
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.expressions.diagonal import DiagMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import ZeroMatrix
from sympy.tensor.array.arrayop import (permutedims, tensorcontraction, tensorproduct)
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
from sympy.combinatorics import Permutation
from sympy.tensor.array.expressions.array_expressions import ZeroArray, OneArray, ArraySymbol, ArrayElement, \
    PermuteDims, ArrayContraction, ArrayTensorProduct, ArrayDiagonal, \
    ArrayAdd, nest_permutation, ArrayElementwiseApplyFunc, _EditArrayContraction, _ArgE, _array_tensor_product, \
    _array_contraction, _array_diagonal, _array_add, _permute_dims, Reshape
from sympy.testing.pytest import raises

i, j, k, l, m, n = symbols("i j k l m n")


M = ArraySymbol("M", (k, k))
N = ArraySymbol("N", (k, k))
P = ArraySymbol("P", (k, k))
Q = ArraySymbol("Q", (k, k))

A = ArraySymbol("A", (k, k))
B = ArraySymbol("B", (k, k))
C = ArraySymbol("C", (k, k))
D = ArraySymbol("D", (k, k))

X = ArraySymbol("X", (k, k))
Y = ArraySymbol("Y", (k, k))

a = ArraySymbol("a", (k, 1))
b = ArraySymbol("b", (k, 1))
c = ArraySymbol("c", (k, 1))
d = ArraySymbol("d", (k, 1))


def test_array_symbol_and_element():
    A = ArraySymbol("A", (2,))
    A0 = ArrayElement(A, (0,))
    A1 = ArrayElement(A, (1,))
    assert A[0] == A0
    assert A[1] != A0
    assert A.as_explicit() == ImmutableDenseNDimArray([A0, A1])

    A2 = tensorproduct(A, A)
    assert A2.shape == (2, 2)
    # TODO: not yet supported:
    # assert A2.as_explicit() == Array([[A[0]*A[0], A[1]*A[0]], [A[0]*A[1], A[1]*A[1]]])
    A3 = tensorcontraction(A2, (0, 1))
    assert A3.shape == ()
    # TODO: not yet supported:
    # assert A3.as_explicit() == Array([])

    A = ArraySymbol("A", (2, 3, 4))
    Ae = A.as_explicit()
    assert Ae == ImmutableDenseNDimArray(
        [[[ArrayElement(A, (i, j, k)) for k in range(4)] for j in range(3)] for i in range(2)])

    p = _permute_dims(A, Permutation(0, 2, 1))
    assert isinstance(p, PermuteDims)

    A = ArraySymbol("A", (2,))
    raises(IndexError, lambda: A[()])
    raises(IndexError, lambda: A[0, 1])
    raises(ValueError, lambda: A[-1])
    raises(ValueError, lambda: A[2])

    O = OneArray(3, 4)
    Z = ZeroArray(m, n)

    raises(IndexError, lambda: O[()])
    raises(IndexError, lambda: O[1, 2, 3])
    raises(ValueError, lambda: O[3, 0])
    raises(ValueError, lambda: O[0, 4])

    assert O[1, 2] == 1
    assert Z[1, 2] == 0


def test_zero_array():
    assert ZeroArray() == 0
    assert ZeroArray().is_Integer

    za = ZeroArray(3, 2, 4)
    assert za.shape == (3, 2, 4)
    za_e = za.as_explicit()
    assert za_e.shape == (3, 2, 4)

    m, n, k = symbols("m n k")
    za = ZeroArray(m, n, k, 2)
    assert za.shape == (m, n, k, 2)
    raises(ValueError, lambda: za.as_explicit())


def test_one_array():
    assert OneArray() == 1
    assert OneArray().is_Integer

    oa = OneArray(3, 2, 4)
    assert oa.shape == (3, 2, 4)
    oa_e = oa.as_explicit()
    assert oa_e.shape == (3, 2, 4)

    m, n, k = symbols("m n k")
    oa = OneArray(m, n, k, 2)
    assert oa.shape == (m, n, k, 2)
    raises(ValueError, lambda: oa.as_explicit())


def test_arrayexpr_contraction_construction():

    cg = _array_contraction(A)
    assert cg == A

    cg = _array_contraction(_array_tensor_product(A, B), (1, 0))
    assert cg == _array_contraction(_array_tensor_product(A, B), (0, 1))

    cg = _array_contraction(_array_tensor_product(M, N), (0, 1))
    indtup = cg._get_contraction_tuples()
    assert indtup == [[(0, 0), (0, 1)]]
    assert cg._contraction_tuples_to_contraction_indices(cg.expr, indtup) == [(0, 1)]

    cg = _array_contraction(_array_tensor_product(M, N), (1, 2))
    indtup = cg._get_contraction_tuples()
    assert indtup == [[(0, 1), (1, 0)]]
    assert cg._contraction_tuples_to_contraction_indices(cg.expr, indtup) == [(1, 2)]

    cg = _array_contraction(_array_tensor_product(M, M, N), (1, 4), (2, 5))
    indtup = cg._get_contraction_tuples()
    assert indtup == [[(0, 0), (1, 1)], [(0, 1), (2, 0)]]
    assert cg._contraction_tuples_to_contraction_indices(cg.expr, indtup) == [(0, 3), (1, 4)]

    # Test removal of trivial contraction:
    assert _array_contraction(a, (1,)) == a
    assert _array_contraction(
        _array_tensor_product(a, b), (0, 2), (1,), (3,)) == _array_contraction(
        _array_tensor_product(a, b), (0, 2))


def test_arrayexpr_array_flatten():

    # Flatten nested ArrayTensorProduct objects:
    expr1 = _array_tensor_product(M, N)
    expr2 = _array_tensor_product(P, Q)
    expr = _array_tensor_product(expr1, expr2)
    assert expr == _array_tensor_product(M, N, P, Q)
    assert expr.args == (M, N, P, Q)

    # Flatten mixed ArrayTensorProduct and ArrayContraction objects:
    cg1 = _array_contraction(expr1, (1, 2))
    cg2 = _array_contraction(expr2, (0, 3))

    expr = _array_tensor_product(cg1, cg2)
    assert expr == _array_contraction(_array_tensor_product(M, N, P, Q), (1, 2), (4, 7))

    expr = _array_tensor_product(M, cg1)
    assert expr == _array_contraction(_array_tensor_product(M, M, N), (3, 4))

    # Flatten nested ArrayContraction objects:
    cgnested = _array_contraction(cg1, (0, 1))
    assert cgnested == _array_contraction(_array_tensor_product(M, N), (0, 3), (1, 2))

    cgnested = _array_contraction(_array_tensor_product(cg1, cg2), (0, 3))
    assert cgnested == _array_contraction(_array_tensor_product(M, N, P, Q), (0, 6), (1, 2), (4, 7))

    cg3 = _array_contraction(_array_tensor_product(M, N, P, Q), (1, 3), (2, 4))
    cgnested = _array_contraction(cg3, (0, 1))
    assert cgnested == _array_contraction(_array_tensor_product(M, N, P, Q), (0, 5), (1, 3), (2, 4))

    cgnested = _array_contraction(cg3, (0, 3), (1, 2))
    assert cgnested == _array_contraction(_array_tensor_product(M, N, P, Q), (0, 7), (1, 3), (2, 4), (5, 6))

    cg4 = _array_contraction(_array_tensor_product(M, N, P, Q), (1, 5), (3, 7))
    cgnested = _array_contraction(cg4, (0, 1))
    assert cgnested == _array_contraction(_array_tensor_product(M, N, P, Q), (0, 2), (1, 5), (3, 7))

    cgnested = _array_contraction(cg4, (0, 1), (2, 3))
    assert cgnested == _array_contraction(_array_tensor_product(M, N, P, Q), (0, 2), (1, 5), (3, 7), (4, 6))

    cg = _array_diagonal(cg4)
    assert cg == cg4
    assert isinstance(cg, type(cg4))

    # Flatten nested ArrayDiagonal objects:
    cg1 = _array_diagonal(expr1, (1, 2))
    cg2 = _array_diagonal(expr2, (0, 3))
    cg3 = _array_diagonal(_array_tensor_product(M, N, P, Q), (1, 3), (2, 4))
    cg4 = _array_diagonal(_array_tensor_product(M, N, P, Q), (1, 5), (3, 7))

    cgnested = _array_diagonal(cg1, (0, 1))
    assert cgnested == _array_diagonal(_array_tensor_product(M, N), (1, 2), (0, 3))

    cgnested = _array_diagonal(cg3, (1, 2))
    assert cgnested == _array_diagonal(_array_tensor_product(M, N, P, Q), (1, 3), (2, 4), (5, 6))

    cgnested = _array_diagonal(cg4, (1, 2))
    assert cgnested == _array_diagonal(_array_tensor_product(M, N, P, Q), (1, 5), (3, 7), (2, 4))

    cg = _array_add(M, N)
    cg2 = _array_add(cg, P)
    assert isinstance(cg2, ArrayAdd)
    assert cg2.args == (M, N, P)
    assert cg2.shape == (k, k)

    expr = _array_tensor_product(_array_diagonal(X, (0, 1)), _array_diagonal(A, (0, 1)))
    assert expr == _array_diagonal(_array_tensor_product(X, A), (0, 1), (2, 3))

    expr1 = _array_diagonal(_array_tensor_product(X, A), (1, 2))
    expr2 = _array_tensor_product(expr1, a)
    assert expr2 == _permute_dims(_array_diagonal(_array_tensor_product(X, A, a), (1, 2)), [0, 1, 4, 2, 3])

    expr1 = _array_contraction(_array_tensor_product(X, A), (1, 2))
    expr2 = _array_tensor_product(expr1, a)
    assert isinstance(expr2, ArrayContraction)
    assert isinstance(expr2.expr, ArrayTensorProduct)

    cg = _array_tensor_product(_array_diagonal(_array_tensor_product(A, X, Y), (0, 3), (1, 5)), a, b)
    assert cg == _permute_dims(_array_diagonal(_array_tensor_product(A, X, Y, a, b), (0, 3), (1, 5)), [0, 1, 6, 7, 2, 3, 4, 5])


def test_arrayexpr_array_diagonal():
    cg = _array_diagonal(M, (1, 0))
    assert cg == _array_diagonal(M, (0, 1))

    cg = _array_diagonal(_array_tensor_product(M, N, P), (4, 1), (2, 0))
    assert cg == _array_diagonal(_array_tensor_product(M, N, P), (1, 4), (0, 2))

    cg = _array_diagonal(_array_tensor_product(M, N), (1, 2), (3,), allow_trivial_diags=True)
    assert cg == _permute_dims(_array_diagonal(_array_tensor_product(M, N), (1, 2)), [0, 2, 1])

    Ax = ArraySymbol("Ax", shape=(1, 2, 3, 4, 3, 5, 6, 2, 7))
    cg = _array_diagonal(Ax, (1, 7), (3,), (2, 4), (6,), allow_trivial_diags=True)
    assert cg == _permute_dims(_array_diagonal(Ax, (1, 7), (2, 4)), [0, 2, 4, 5, 1, 6, 3])

    cg = _array_diagonal(M, (0,), allow_trivial_diags=True)
    assert cg == _permute_dims(M, [1, 0])

    raises(ValueError, lambda: _array_diagonal(M, (0, 0)))


def test_arrayexpr_array_shape():
    expr = _array_tensor_product(M, N, P, Q)
    assert expr.shape == (k, k, k, k, k, k, k, k)
    Z = MatrixSymbol("Z", m, n)
    expr = _array_tensor_product(M, Z)
    assert expr.shape == (k, k, m, n)
    expr2 = _array_contraction(expr, (0, 1))
    assert expr2.shape == (m, n)
    expr2 = _array_diagonal(expr, (0, 1))
    assert expr2.shape == (m, n, k)
    exprp = _permute_dims(expr, [2, 1, 3, 0])
    assert exprp.shape == (m, k, n, k)
    expr3 = _array_tensor_product(N, Z)
    expr2 = _array_add(expr, expr3)
    assert expr2.shape == (k, k, m, n)

    # Contraction along axes with discordant dimensions:
    raises(ValueError, lambda: _array_contraction(expr, (1, 2)))
    # Also diagonal needs the same dimensions:
    raises(ValueError, lambda: _array_diagonal(expr, (1, 2)))
    # Diagonal requires at least to axes to compute the diagonal:
    raises(ValueError, lambda: _array_diagonal(expr, (1,)))


def test_arrayexpr_permutedims_sink():

    cg = _permute_dims(_array_tensor_product(M, N), [0, 1, 3, 2], nest_permutation=False)
    sunk = nest_permutation(cg)
    assert sunk == _array_tensor_product(M, _permute_dims(N, [1, 0]))

    cg = _permute_dims(_array_tensor_product(M, N), [1, 0, 3, 2], nest_permutation=False)
    sunk = nest_permutation(cg)
    assert sunk == _array_tensor_product(_permute_dims(M, [1, 0]), _permute_dims(N, [1, 0]))

    cg = _permute_dims(_array_tensor_product(M, N), [3, 2, 1, 0], nest_permutation=False)
    sunk = nest_permutation(cg)
    assert sunk == _array_tensor_product(_permute_dims(N, [1, 0]), _permute_dims(M, [1, 0]))

    cg = _permute_dims(_array_contraction(_array_tensor_product(M, N), (1, 2)), [1, 0], nest_permutation=False)
    sunk = nest_permutation(cg)
    assert sunk == _array_contraction(_permute_dims(_array_tensor_product(M, N), [[0, 3]]), (1, 2))

    cg = _permute_dims(_array_tensor_product(M, N), [1, 0, 3, 2], nest_permutation=False)
    sunk = nest_permutation(cg)
    assert sunk == _array_tensor_product(_permute_dims(M, [1, 0]), _permute_dims(N, [1, 0]))

    cg = _permute_dims(_array_contraction(_array_tensor_product(M, N, P), (1, 2), (3, 4)), [1, 0], nest_permutation=False)
    sunk = nest_permutation(cg)
    assert sunk == _array_contraction(_permute_dims(_array_tensor_product(M, N, P), [[0, 5]]), (1, 2), (3, 4))


def test_arrayexpr_push_indices_up_and_down():

    indices = list(range(12))

    contr_diag_indices = [(0, 6), (2, 8)]
    assert ArrayContraction._push_indices_down(contr_diag_indices, indices) == (1, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15)
    assert ArrayContraction._push_indices_up(contr_diag_indices, indices) == (None, 0, None, 1, 2, 3, None, 4, None, 5, 6, 7)

    assert ArrayDiagonal._push_indices_down(contr_diag_indices, indices, 10) == (1, 3, 4, 5, 7, 9, (0, 6), (2, 8), None, None, None, None)
    assert ArrayDiagonal._push_indices_up(contr_diag_indices, indices, 10) == (6, 0, 7, 1, 2, 3, 6, 4, 7, 5, None, None)

    contr_diag_indices = [(1, 2), (7, 8)]
    assert ArrayContraction._push_indices_down(contr_diag_indices, indices) == (0, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15)
    assert ArrayContraction._push_indices_up(contr_diag_indices, indices) == (0, None, None, 1, 2, 3, 4, None, None, 5, 6, 7)

    assert ArrayDiagonal._push_indices_down(contr_diag_indices, indices, 10) == (0, 3, 4, 5, 6, 9, (1, 2), (7, 8), None, None, None, None)
    assert ArrayDiagonal._push_indices_up(contr_diag_indices, indices, 10) == (0, 6, 6, 1, 2, 3, 4, 7, 7, 5, None, None)


def test_arrayexpr_split_multiple_contractions():
    a = MatrixSymbol("a", k, 1)
    b = MatrixSymbol("b", k, 1)
    A = MatrixSymbol("A", k, k)
    B = MatrixSymbol("B", k, k)
    C = MatrixSymbol("C", k, k)
    X = MatrixSymbol("X", k, k)

    cg = _array_contraction(_array_tensor_product(A.T, a, b, b.T, (A*X*b).applyfunc(cos)), (1, 2, 8), (5, 6, 9))
    expected = _array_contraction(_array_tensor_product(A.T, DiagMatrix(a), OneArray(1), b, b.T, (A*X*b).applyfunc(cos)), (1, 3), (2, 9), (6, 7, 10))
    assert cg.split_multiple_contractions().dummy_eq(expected)

    # Check no overlap of lines:

    cg = _array_contraction(_array_tensor_product(A, a, C, a, B), (1, 2, 4), (5, 6, 8), (3, 7))
    assert cg.split_multiple_contractions() == cg

    cg = _array_contraction(_array_tensor_product(a, b, A), (0, 2, 4), (1, 3))
    assert cg.split_multiple_contractions() == cg


def test_arrayexpr_nested_permutations():

    cg = _permute_dims(_permute_dims(M, (1, 0)), (1, 0))
    assert cg == M

    times = 3
    plist1 = [list(range(6)) for i in range(times)]
    plist2 = [list(range(6)) for i in range(times)]

    for i in range(times):
        random.shuffle(plist1[i])
        random.shuffle(plist2[i])

    plist1.append([2, 5, 4, 1, 0, 3])
    plist2.append([3, 5, 0, 4, 1, 2])

    plist1.append([2, 5, 4, 0, 3, 1])
    plist2.append([3, 0, 5, 1, 2, 4])

    plist1.append([5, 4, 2, 0, 3, 1])
    plist2.append([4, 5, 0, 2, 3, 1])

    Me = M.subs(k, 3).as_explicit()
    Ne = N.subs(k, 3).as_explicit()
    Pe = P.subs(k, 3).as_explicit()
    cge = tensorproduct(Me, Ne, Pe)

    for permutation_array1, permutation_array2 in zip(plist1, plist2):
        p1 = Permutation(permutation_array1)
        p2 = Permutation(permutation_array2)

        cg = _permute_dims(
            _permute_dims(
                _array_tensor_product(M, N, P),
                p1),
            p2
        )
        result = _permute_dims(
            _array_tensor_product(M, N, P),
            p2*p1
        )
        assert cg == result

        # Check that `permutedims` behaves the same way with explicit-component arrays:
        result1 = _permute_dims(_permute_dims(cge, p1), p2)
        result2 = _permute_dims(cge, p2*p1)
        assert result1 == result2


def test_arrayexpr_contraction_permutation_mix():

    Me = M.subs(k, 3).as_explicit()
    Ne = N.subs(k, 3).as_explicit()

    cg1 = _array_contraction(PermuteDims(_array_tensor_product(M, N), Permutation([0, 2, 1, 3])), (2, 3))
    cg2 = _array_contraction(_array_tensor_product(M, N), (1, 3))
    assert cg1 == cg2
    cge1 = tensorcontraction(permutedims(tensorproduct(Me, Ne), Permutation([0, 2, 1, 3])), (2, 3))
    cge2 = tensorcontraction(tensorproduct(Me, Ne), (1, 3))
    assert cge1 == cge2

    cg1 = _permute_dims(_array_tensor_product(M, N), Permutation([0, 1, 3, 2]))
    cg2 = _array_tensor_product(M, _permute_dims(N, Permutation([1, 0])))
    assert cg1 == cg2

    cg1 = _array_contraction(
        _permute_dims(
            _array_tensor_product(M, N, P, Q), Permutation([0, 2, 3, 1, 4, 5, 7, 6])),
        (1, 2), (3, 5)
    )
    cg2 = _array_contraction(
        _array_tensor_product(M, N, P, _permute_dims(Q, Permutation([1, 0]))),
        (1, 5), (2, 3)
    )
    assert cg1 == cg2

    cg1 = _array_contraction(
        _permute_dims(
            _array_tensor_product(M, N, P, Q), Permutation([1, 0, 4, 6, 2, 7, 5, 3])),
        (0, 1), (2, 6), (3, 7)
    )
    cg2 = _permute_dims(
        _array_contraction(
            _array_tensor_product(M, P, Q, N),
            (0, 1), (2, 3), (4, 7)),
        [1, 0]
    )
    assert cg1 == cg2

    cg1 = _array_contraction(
        _permute_dims(
            _array_tensor_product(M, N, P, Q), Permutation([1, 0, 4, 6, 7, 2, 5, 3])),
        (0, 1), (2, 6), (3, 7)
    )
    cg2 = _permute_dims(
        _array_contraction(
            _array_tensor_product(_permute_dims(M, [1, 0]), N, P, Q),
            (0, 1), (3, 6), (4, 5)
        ),
        Permutation([1, 0])
    )
    assert cg1 == cg2


def test_arrayexpr_permute_tensor_product():
    cg1 = _permute_dims(_array_tensor_product(M, N, P, Q), Permutation([2, 3, 1, 0, 5, 4, 6, 7]))
    cg2 = _array_tensor_product(N, _permute_dims(M, [1, 0]),
                                    _permute_dims(P, [1, 0]), Q)
    assert cg1 == cg2

    # TODO: reverse operation starting with `PermuteDims` and getting down to `bb`...
    cg1 = _permute_dims(_array_tensor_product(M, N, P, Q), Permutation([2, 3, 4, 5, 0, 1, 6, 7]))
    cg2 = _array_tensor_product(N, P, M, Q)
    assert cg1 == cg2

    cg1 = _permute_dims(_array_tensor_product(M, N, P, Q), Permutation([2, 3, 4, 6, 5, 7, 0, 1]))
    assert cg1.expr == _array_tensor_product(N, P, Q, M)
    assert cg1.permutation == Permutation([0, 1, 2, 4, 3, 5, 6, 7])

    cg1 = _array_contraction(
        _permute_dims(
            _array_tensor_product(N, Q, Q, M),
            [2, 1, 5, 4, 0, 3, 6, 7]),
        [1, 2, 6])
    cg2 = _permute_dims(_array_contraction(_array_tensor_product(Q, Q, N, M), (3, 5, 6)), [0, 2, 3, 1, 4])
    assert cg1 == cg2

    cg1 = _array_contraction(
        _array_contraction(
            _array_contraction(
                _array_contraction(
                    _permute_dims(
                        _array_tensor_product(N, Q, Q, M),
                        [2, 1, 5, 4, 0, 3, 6, 7]),
                    [1, 2, 6]),
                [1, 3, 4]),
            [1]),
        [0])
    cg2 = _array_contraction(_array_tensor_product(M, N, Q, Q), (0, 3, 5), (1, 4, 7), (2,), (6,))
    assert cg1 == cg2


def test_arrayexpr_canonicalize_diagonal__permute_dims():
    tp = _array_tensor_product(M, Q, N, P)
    expr = _array_diagonal(
        _permute_dims(tp, [0, 1, 2, 4, 7, 6, 3, 5]), (2, 4, 5), (6, 7),
        (0, 3))
    result = _array_diagonal(tp, (2, 6, 7), (3, 5), (0, 4))
    assert expr == result

    tp = _array_tensor_product(M, N, P, Q)
    expr = _array_diagonal(_permute_dims(tp, [0, 5, 2, 4, 1, 6, 3, 7]), (1, 2, 6), (3, 4))
    result = _array_diagonal(_array_tensor_product(M, P, N, Q), (3, 4, 5), (1, 2))
    assert expr == result


def test_arrayexpr_canonicalize_diagonal_contraction():
    tp = _array_tensor_product(M, N, P, Q)
    expr = _array_contraction(_array_diagonal(tp, (1, 3, 4)), (0, 3))
    result = _array_diagonal(_array_contraction(_array_tensor_product(M, N, P, Q), (0, 6)), (0, 2, 3))
    assert expr == result

    expr = _array_contraction(_array_diagonal(tp, (0, 1, 2, 3, 7)), (1, 2, 3))
    result = _array_contraction(_array_tensor_product(M, N, P, Q), (0, 1, 2, 3, 5, 6, 7))
    assert expr == result

    expr = _array_contraction(_array_diagonal(tp, (0, 2, 6, 7)), (1, 2, 3))
    result = _array_diagonal(_array_contraction(tp, (3, 4, 5)), (0, 2, 3, 4))
    assert expr == result

    td = _array_diagonal(_array_tensor_product(M, N, P, Q), (0, 3))
    expr = _array_contraction(td, (2, 1), (0, 4, 6, 5, 3))
    result = _array_contraction(_array_tensor_product(M, N, P, Q), (0, 1, 3, 5, 6, 7), (2, 4))
    assert expr == result


def test_arrayexpr_array_wrong_permutation_size():
    cg = _array_tensor_product(M, N)
    raises(ValueError, lambda: _permute_dims(cg, [1, 0]))
    raises(ValueError, lambda: _permute_dims(cg, [1, 0, 2, 3, 5, 4]))


def test_arrayexpr_nested_array_elementwise_add():
    cg = _array_contraction(_array_add(
        _array_tensor_product(M, N),
        _array_tensor_product(N, M)
    ), (1, 2))
    result = _array_add(
        _array_contraction(_array_tensor_product(M, N), (1, 2)),
        _array_contraction(_array_tensor_product(N, M), (1, 2))
    )
    assert cg == result

    cg = _array_diagonal(_array_add(
        _array_tensor_product(M, N),
        _array_tensor_product(N, M)
    ), (1, 2))
    result = _array_add(
        _array_diagonal(_array_tensor_product(M, N), (1, 2)),
        _array_diagonal(_array_tensor_product(N, M), (1, 2))
    )
    assert cg == result


def test_arrayexpr_array_expr_zero_array():
    za1 = ZeroArray(k, l, m, n)
    zm1 = ZeroMatrix(m, n)

    za2 = ZeroArray(k, m, m, n)
    zm2 = ZeroMatrix(m, m)
    zm3 = ZeroMatrix(k, k)

    assert _array_tensor_product(M, N, za1) == ZeroArray(k, k, k, k, k, l, m, n)
    assert _array_tensor_product(M, N, zm1) == ZeroArray(k, k, k, k, m, n)

    assert _array_contraction(za1, (3,)) == ZeroArray(k, l, m)
    assert _array_contraction(zm1, (1,)) == ZeroArray(m)
    assert _array_contraction(za2, (1, 2)) == ZeroArray(k, n)
    assert _array_contraction(zm2, (0, 1)) == 0

    assert _array_diagonal(za2, (1, 2)) == ZeroArray(k, n, m)
    assert _array_diagonal(zm2, (0, 1)) == ZeroArray(m)

    assert _permute_dims(za1, [2, 1, 3, 0]) == ZeroArray(m, l, n, k)
    assert _permute_dims(zm1, [1, 0]) == ZeroArray(n, m)

    assert _array_add(za1) == za1
    assert _array_add(zm1) == ZeroArray(m, n)
    tp1 = _array_tensor_product(MatrixSymbol("A", k, l), MatrixSymbol("B", m, n))
    assert _array_add(tp1, za1) == tp1
    tp2 = _array_tensor_product(MatrixSymbol("C", k, l), MatrixSymbol("D", m, n))
    assert _array_add(tp1, za1, tp2) == _array_add(tp1, tp2)
    assert _array_add(M, zm3) == M
    assert _array_add(M, N, zm3) == _array_add(M, N)


def test_arrayexpr_array_expr_applyfunc():

    A = ArraySymbol("A", (3, k, 2))
    aaf = ArrayElementwiseApplyFunc(sin, A)
    assert aaf.shape == (3, k, 2)


def test_edit_array_contraction():
    cg = _array_contraction(_array_tensor_product(A, B, C, D), (1, 2, 5))
    ecg = _EditArrayContraction(cg)
    assert ecg.to_array_contraction() == cg

    ecg.args_with_ind[1], ecg.args_with_ind[2] = ecg.args_with_ind[2], ecg.args_with_ind[1]
    assert ecg.to_array_contraction() == _array_contraction(_array_tensor_product(A, C, B, D), (1, 3, 4))

    ci = ecg.get_new_contraction_index()
    new_arg = _ArgE(X)
    new_arg.indices = [ci, ci]
    ecg.args_with_ind.insert(2, new_arg)
    assert ecg.to_array_contraction() == _array_contraction(_array_tensor_product(A, C, X, B, D), (1, 3, 6), (4, 5))

    assert ecg.get_contraction_indices() == [[1, 3, 6], [4, 5]]
    assert [[tuple(j) for j in i] for i in ecg.get_contraction_indices_to_ind_rel_pos()] == [[(0, 1), (1, 1), (3, 0)], [(2, 0), (2, 1)]]
    assert [list(i) for i in ecg.get_mapping_for_index(0)] == [[0, 1], [1, 1], [3, 0]]
    assert [list(i) for i in ecg.get_mapping_for_index(1)] == [[2, 0], [2, 1]]
    raises(ValueError, lambda: ecg.get_mapping_for_index(2))

    ecg.args_with_ind.pop(1)
    assert ecg.to_array_contraction() == _array_contraction(_array_tensor_product(A, X, B, D), (1, 4), (2, 3))

    ecg.args_with_ind[0].indices[1] = ecg.args_with_ind[1].indices[0]
    ecg.args_with_ind[1].indices[1] = ecg.args_with_ind[2].indices[0]
    assert ecg.to_array_contraction() == _array_contraction(_array_tensor_product(A, X, B, D), (1, 2), (3, 4))

    ecg.insert_after(ecg.args_with_ind[1], _ArgE(C))
    assert ecg.to_array_contraction() == _array_contraction(_array_tensor_product(A, X, C, B, D), (1, 2), (3, 6))


def test_array_expressions_no_canonicalization():

    tp = _array_tensor_product(M, N, P)

    # ArrayTensorProduct:

    expr = ArrayTensorProduct(tp, N)
    assert str(expr) == "ArrayTensorProduct(ArrayTensorProduct(M, N, P), N)"
    assert expr.doit() == ArrayTensorProduct(M, N, P, N)

    expr = ArrayTensorProduct(ArrayContraction(M, (0, 1)), N)
    assert str(expr) == "ArrayTensorProduct(ArrayContraction(M, (0, 1)), N)"
    assert expr.doit() == ArrayContraction(ArrayTensorProduct(M, N), (0, 1))

    expr = ArrayTensorProduct(ArrayDiagonal(M, (0, 1)), N)
    assert str(expr) == "ArrayTensorProduct(ArrayDiagonal(M, (0, 1)), N)"
    assert expr.doit() == PermuteDims(ArrayDiagonal(ArrayTensorProduct(M, N), (0, 1)), [2, 0, 1])

    expr = ArrayTensorProduct(PermuteDims(M, [1, 0]), N)
    assert str(expr) == "ArrayTensorProduct(PermuteDims(M, (0 1)), N)"
    assert expr.doit() == PermuteDims(ArrayTensorProduct(M, N), [1, 0, 2, 3])

    # ArrayContraction:

    expr = ArrayContraction(_array_contraction(tp, (0, 2)), (0, 1))
    assert isinstance(expr, ArrayContraction)
    assert isinstance(expr.expr, ArrayContraction)
    assert str(expr) == "ArrayContraction(ArrayContraction(ArrayTensorProduct(M, N, P), (0, 2)), (0, 1))"
    assert expr.doit() == ArrayContraction(tp, (0, 2), (1, 3))

    expr = ArrayContraction(ArrayContraction(ArrayContraction(tp, (0, 1)), (0, 1)), (0, 1))
    assert expr.doit() == ArrayContraction(tp, (0, 1), (2, 3), (4, 5))
    # assert expr._canonicalize() == ArrayContraction(ArrayContraction(tp, (0, 1)), (0, 1), (2, 3))

    expr = ArrayContraction(ArrayDiagonal(tp, (0, 1)), (0, 1))
    assert str(expr) == "ArrayContraction(ArrayDiagonal(ArrayTensorProduct(M, N, P), (0, 1)), (0, 1))"
    assert expr.doit() == ArrayDiagonal(ArrayContraction(ArrayTensorProduct(N, M, P), (0, 1)), (0, 1))

    expr = ArrayContraction(PermuteDims(M, [1, 0]), (0, 1))
    assert str(expr) == "ArrayContraction(PermuteDims(M, (0 1)), (0, 1))"
    assert expr.doit() == ArrayContraction(M, (0, 1))

    # ArrayDiagonal:

    expr = ArrayDiagonal(ArrayDiagonal(tp, (0, 2)), (0, 1))
    assert str(expr) == "ArrayDiagonal(ArrayDiagonal(ArrayTensorProduct(M, N, P), (0, 2)), (0, 1))"
    assert expr.doit() == ArrayDiagonal(tp, (0, 2), (1, 3))

    expr = ArrayDiagonal(ArrayDiagonal(ArrayDiagonal(tp, (0, 1)), (0, 1)), (0, 1))
    assert expr.doit() == ArrayDiagonal(tp, (0, 1), (2, 3), (4, 5))
    assert expr._canonicalize() == expr.doit()

    expr = ArrayDiagonal(ArrayContraction(tp, (0, 1)), (0, 1))
    assert str(expr) == "ArrayDiagonal(ArrayContraction(ArrayTensorProduct(M, N, P), (0, 1)), (0, 1))"
    assert expr.doit() == expr

    expr = ArrayDiagonal(PermuteDims(M, [1, 0]), (0, 1))
    assert str(expr) == "ArrayDiagonal(PermuteDims(M, (0 1)), (0, 1))"
    assert expr.doit() == ArrayDiagonal(M, (0, 1))

    # ArrayAdd:

    expr = ArrayAdd(M)
    assert isinstance(expr, ArrayAdd)
    assert expr.doit() == M

    expr = ArrayAdd(ArrayAdd(M, N), P)
    assert str(expr) == "ArrayAdd(ArrayAdd(M, N), P)"
    assert expr.doit() == ArrayAdd(M, N, P)

    expr = ArrayAdd(M, ArrayAdd(N, ArrayAdd(P, M)))
    assert expr.doit() == ArrayAdd(M, N, P, M)
    assert expr._canonicalize() == ArrayAdd(M, N, ArrayAdd(P, M))

    expr = ArrayAdd(M, ZeroArray(k, k), N)
    assert str(expr) == "ArrayAdd(M, ZeroArray(k, k), N)"
    assert expr.doit() == ArrayAdd(M, N)

    # PermuteDims:

    expr = PermuteDims(PermuteDims(M, [1, 0]), [1, 0])
    assert str(expr) == "PermuteDims(PermuteDims(M, (0 1)), (0 1))"
    assert expr.doit() == M

    expr = PermuteDims(PermuteDims(PermuteDims(M, [1, 0]), [1, 0]), [1, 0])
    assert expr.doit() == PermuteDims(M, [1, 0])
    assert expr._canonicalize() == expr.doit()

    # Reshape

    expr = Reshape(A, (k**2,))
    assert expr.shape == (k**2,)
    assert isinstance(expr, Reshape)


def test_array_expr_construction_with_functions():

    tp = tensorproduct(M, N)
    assert tp == ArrayTensorProduct(M, N)

    expr = tensorproduct(A, eye(2))
    assert expr == ArrayTensorProduct(A, eye(2))

    # Contraction:

    expr = tensorcontraction(M, (0, 1))
    assert expr == ArrayContraction(M, (0, 1))

    expr = tensorcontraction(tp, (1, 2))
    assert expr == ArrayContraction(tp, (1, 2))

    expr = tensorcontraction(tensorcontraction(tp, (1, 2)), (0, 1))
    assert expr == ArrayContraction(tp, (0, 3), (1, 2))

    # Diagonalization:

    expr = tensordiagonal(M, (0, 1))
    assert expr == ArrayDiagonal(M, (0, 1))

    expr = tensordiagonal(tensordiagonal(tp, (0, 1)), (0, 1))
    assert expr == ArrayDiagonal(tp, (0, 1), (2, 3))

    # Permutation of dimensions:

    expr = permutedims(M, [1, 0])
    assert expr == PermuteDims(M, [1, 0])

    expr = permutedims(PermuteDims(tp, [1, 0, 2, 3]), [0, 1, 3, 2])
    assert expr == PermuteDims(tp, [1, 0, 3, 2])

    expr = PermuteDims(tp, index_order_new=["a", "b", "c", "d"], index_order_old=["d", "c", "b", "a"])
    assert expr == PermuteDims(tp, [3, 2, 1, 0])

    arr = Array(range(32)).reshape(2, 2, 2, 2, 2)
    expr = PermuteDims(arr, index_order_new=["a", "b", "c", "d", "e"], index_order_old=['b', 'e', 'a', 'd', 'c'])
    assert expr == PermuteDims(arr, [2, 0, 4, 3, 1])
    assert expr.as_explicit() == permutedims(arr, index_order_new=["a", "b", "c", "d", "e"], index_order_old=['b', 'e', 'a', 'd', 'c'])


def test_array_element_expressions():
    # Check commutative property:
    assert M[0, 0]*N[0, 0] == N[0, 0]*M[0, 0]

    # Check derivatives:
    assert M[0, 0].diff(M[0, 0]) == 1
    assert M[0, 0].diff(M[1, 0]) == 0
    assert M[0, 0].diff(N[0, 0]) == 0
    assert M[0, 1].diff(M[i, j]) == KroneckerDelta(i, 0)*KroneckerDelta(j, 1)
    assert M[0, 1].diff(N[i, j]) == 0

    K4 = ArraySymbol("K4", shape=(k, k, k, k))

    assert K4[i, j, k, l].diff(K4[1, 2, 3, 4]) == (
        KroneckerDelta(i, 1)*KroneckerDelta(j, 2)*KroneckerDelta(k, 3)*KroneckerDelta(l, 4)
    )


def test_array_expr_reshape():

    A = MatrixSymbol("A", 2, 2)
    B = ArraySymbol("B", (2, 2, 2))
    C = Array([1, 2, 3, 4])

    expr = Reshape(A, (4,))
    assert expr.expr == A
    assert expr.shape == (4,)
    assert expr.as_explicit() == Array([A[0, 0], A[0, 1], A[1, 0], A[1, 1]])

    expr = Reshape(B, (2, 4))
    assert expr.expr == B
    assert expr.shape == (2, 4)
    ee = expr.as_explicit()
    assert isinstance(ee, ImmutableDenseNDimArray)
    assert ee.shape == (2, 4)
    assert ee == Array([[B[0, 0, 0], B[0, 0, 1], B[0, 1, 0], B[0, 1, 1]], [B[1, 0, 0], B[1, 0, 1], B[1, 1, 0], B[1, 1, 1]]])

    expr = Reshape(A, (k, 2))
    assert expr.shape == (k, 2)

    raises(ValueError, lambda: Reshape(A, (2, 3)))
    raises(ValueError, lambda: Reshape(A, (3,)))

    expr = Reshape(C, (2, 2))
    assert expr.expr == C
    assert expr.shape == (2, 2)
    assert expr.doit() == Array([[1, 2], [3, 4]])


def test_array_expr_as_explicit_with_explicit_component_arrays():
    # Test if .as_explicit() works with explicit-component arrays
    # nested in array expressions:
    from sympy.abc import x, y, z, t
    A = Array([[x, y], [z, t]])
    assert ArrayTensorProduct(A, A).as_explicit() == tensorproduct(A, A)
    assert ArrayDiagonal(A, (0, 1)).as_explicit() == tensordiagonal(A, (0, 1))
    assert ArrayContraction(A, (0, 1)).as_explicit() == tensorcontraction(A, (0, 1))
    assert ArrayAdd(A, A).as_explicit() == A + A
    assert ArrayElementwiseApplyFunc(sin, A).as_explicit() == A.applyfunc(sin)
    assert PermuteDims(A, [1, 0]).as_explicit() == permutedims(A, [1, 0])
    assert Reshape(A, [4]).as_explicit() == A.reshape(4)
