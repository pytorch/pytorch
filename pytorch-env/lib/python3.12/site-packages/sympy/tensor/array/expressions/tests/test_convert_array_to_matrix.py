from sympy import Lambda, S, Dummy, KroneckerProduct
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.matrices.expressions.hadamard import HadamardProduct, HadamardPower
from sympy.matrices.expressions.special import (Identity, OneMatrix, ZeroMatrix)
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
from sympy.tensor.array.expressions.from_array_to_matrix import _support_function_tp1_recognize, \
    _array_diag2contr_diagmatrix, convert_array_to_matrix, _remove_trivial_dims, _array2matrix, \
    _combine_removed, identify_removable_identity_matrices, _array_contraction_to_diagonal_multiple_identity
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.combinatorics import Permutation
from sympy.matrices.expressions.diagonal import DiagMatrix, DiagonalMatrix
from sympy.matrices import Trace, MatMul, Transpose
from sympy.tensor.array.expressions.array_expressions import ZeroArray, OneArray, \
    ArrayElement, ArraySymbol, ArrayElementwiseApplyFunc, _array_tensor_product, _array_contraction, \
    _array_diagonal, _permute_dims, PermuteDims, ArrayAdd, ArrayDiagonal, ArrayContraction, ArrayTensorProduct
from sympy.testing.pytest import raises


i, j, k, l, m, n = symbols("i j k l m n")

I = Identity(k)
I1 = Identity(1)

M = MatrixSymbol("M", k, k)
N = MatrixSymbol("N", k, k)
P = MatrixSymbol("P", k, k)
Q = MatrixSymbol("Q", k, k)

A = MatrixSymbol("A", k, k)
B = MatrixSymbol("B", k, k)
C = MatrixSymbol("C", k, k)
D = MatrixSymbol("D", k, k)

X = MatrixSymbol("X", k, k)
Y = MatrixSymbol("Y", k, k)

a = MatrixSymbol("a", k, 1)
b = MatrixSymbol("b", k, 1)
c = MatrixSymbol("c", k, 1)
d = MatrixSymbol("d", k, 1)

x = MatrixSymbol("x", k, 1)
y = MatrixSymbol("y", k, 1)


def test_arrayexpr_convert_array_to_matrix():

    cg = _array_contraction(_array_tensor_product(M), (0, 1))
    assert convert_array_to_matrix(cg) == Trace(M)

    cg = _array_contraction(_array_tensor_product(M, N), (0, 1), (2, 3))
    assert convert_array_to_matrix(cg) == Trace(M) * Trace(N)

    cg = _array_contraction(_array_tensor_product(M, N), (0, 3), (1, 2))
    assert convert_array_to_matrix(cg) == Trace(M * N)

    cg = _array_contraction(_array_tensor_product(M, N), (0, 2), (1, 3))
    assert convert_array_to_matrix(cg) == Trace(M * N.T)

    cg = convert_matrix_to_array(M * N * P)
    assert convert_array_to_matrix(cg) == M * N * P

    cg = convert_matrix_to_array(M * N.T * P)
    assert convert_array_to_matrix(cg) == M * N.T * P

    cg = _array_contraction(_array_tensor_product(M,N,P,Q), (1, 2), (5, 6))
    assert convert_array_to_matrix(cg) == _array_tensor_product(M * N, P * Q)

    cg = _array_contraction(_array_tensor_product(-2, M, N), (1, 2))
    assert convert_array_to_matrix(cg) == -2 * M * N

    a = MatrixSymbol("a", k, 1)
    b = MatrixSymbol("b", k, 1)
    c = MatrixSymbol("c", k, 1)
    cg = PermuteDims(
        _array_contraction(
            _array_tensor_product(
                a,
                ArrayAdd(
                    _array_tensor_product(b, c),
                    _array_tensor_product(c, b),
                )
            ), (2, 4)), [0, 1, 3, 2])
    assert convert_array_to_matrix(cg) == a * (b.T * c + c.T * b)

    za = ZeroArray(m, n)
    assert convert_array_to_matrix(za) == ZeroMatrix(m, n)

    cg = _array_tensor_product(3, M)
    assert convert_array_to_matrix(cg) == 3 * M

    # Partial conversion to matrix multiplication:
    expr = _array_contraction(_array_tensor_product(M, N, P, Q), (0, 2), (1, 4, 6))
    assert convert_array_to_matrix(expr) == _array_contraction(_array_tensor_product(M.T*N, P, Q), (0, 2, 4))

    x = MatrixSymbol("x", k, 1)
    cg = PermuteDims(
        _array_contraction(_array_tensor_product(OneArray(1), x, OneArray(1), DiagMatrix(Identity(1))),
                                (0, 5)), Permutation(1, 2, 3))
    assert convert_array_to_matrix(cg) == x

    expr = ArrayAdd(M, PermuteDims(M, [1, 0]))
    assert convert_array_to_matrix(expr) == M + Transpose(M)


def test_arrayexpr_convert_array_to_matrix2():
    cg = _array_contraction(_array_tensor_product(M, N), (1, 3))
    assert convert_array_to_matrix(cg) == M * N.T

    cg = PermuteDims(_array_tensor_product(M, N), Permutation([0, 1, 3, 2]))
    assert convert_array_to_matrix(cg) == _array_tensor_product(M, N.T)

    cg = _array_tensor_product(M, PermuteDims(N, Permutation([1, 0])))
    assert convert_array_to_matrix(cg) == _array_tensor_product(M, N.T)

    cg = _array_contraction(
        PermuteDims(
            _array_tensor_product(M, N, P, Q), Permutation([0, 2, 3, 1, 4, 5, 7, 6])),
        (1, 2), (3, 5)
    )
    assert convert_array_to_matrix(cg) == _array_tensor_product(M * P.T * Trace(N), Q.T)

    cg = _array_contraction(
        _array_tensor_product(M, N, P, PermuteDims(Q, Permutation([1, 0]))),
        (1, 5), (2, 3)
    )
    assert convert_array_to_matrix(cg) == _array_tensor_product(M * P.T * Trace(N), Q.T)

    cg = _array_tensor_product(M, PermuteDims(N, [1, 0]))
    assert convert_array_to_matrix(cg) == _array_tensor_product(M, N.T)

    cg = _array_tensor_product(PermuteDims(M, [1, 0]), PermuteDims(N, [1, 0]))
    assert convert_array_to_matrix(cg) == _array_tensor_product(M.T, N.T)

    cg = _array_tensor_product(PermuteDims(N, [1, 0]), PermuteDims(M, [1, 0]))
    assert convert_array_to_matrix(cg) == _array_tensor_product(N.T, M.T)

    cg = _array_contraction(M, (0,), (1,))
    assert convert_array_to_matrix(cg) == OneMatrix(1, k)*M*OneMatrix(k, 1)

    cg = _array_contraction(x, (0,), (1,))
    assert convert_array_to_matrix(cg) == OneMatrix(1, k)*x

    Xm = MatrixSymbol("Xm", m, n)
    cg = _array_contraction(Xm, (0,), (1,))
    assert convert_array_to_matrix(cg) == OneMatrix(1, m)*Xm*OneMatrix(n, 1)


def test_arrayexpr_convert_array_to_diagonalized_vector():

    # Check matrix recognition over trivial dimensions:

    cg = _array_tensor_product(a, b)
    assert convert_array_to_matrix(cg) == a * b.T

    cg = _array_tensor_product(I1, a, b)
    assert convert_array_to_matrix(cg) == a * b.T

    # Recognize trace inside a tensor product:

    cg = _array_contraction(_array_tensor_product(A, B, C), (0, 3), (1, 2))
    assert convert_array_to_matrix(cg) == Trace(A * B) * C

    # Transform diagonal operator to contraction:

    cg = _array_diagonal(_array_tensor_product(A, a), (1, 2))
    assert _array_diag2contr_diagmatrix(cg) == _array_contraction(_array_tensor_product(A, OneArray(1), DiagMatrix(a)), (1, 3))
    assert convert_array_to_matrix(cg) == A * DiagMatrix(a)

    cg = _array_diagonal(_array_tensor_product(a, b), (0, 2))
    assert _array_diag2contr_diagmatrix(cg) == _permute_dims(
        _array_contraction(_array_tensor_product(DiagMatrix(a), OneArray(1), b), (0, 3)), [1, 2, 0]
    )
    assert convert_array_to_matrix(cg) == b.T * DiagMatrix(a)

    cg = _array_diagonal(_array_tensor_product(A, a), (0, 2))
    assert _array_diag2contr_diagmatrix(cg) == _array_contraction(_array_tensor_product(A, OneArray(1), DiagMatrix(a)), (0, 3))
    assert convert_array_to_matrix(cg) == A.T * DiagMatrix(a)

    cg = _array_diagonal(_array_tensor_product(I, x, I1), (0, 2), (3, 5))
    assert _array_diag2contr_diagmatrix(cg) == _array_contraction(_array_tensor_product(I, OneArray(1), I1, DiagMatrix(x)), (0, 5))
    assert convert_array_to_matrix(cg) == DiagMatrix(x)

    cg = _array_diagonal(_array_tensor_product(I, x, A, B), (1, 2), (5, 6))
    assert _array_diag2contr_diagmatrix(cg) == _array_diagonal(_array_contraction(_array_tensor_product(I, OneArray(1), A, B, DiagMatrix(x)), (1, 7)), (5, 6))
    # TODO: this is returning a wrong result:
    # convert_array_to_matrix(cg)

    cg = _array_diagonal(_array_tensor_product(I1, a, b), (1, 3, 5))
    assert convert_array_to_matrix(cg) == a*b.T

    cg = _array_diagonal(_array_tensor_product(I1, a, b), (1, 3))
    assert _array_diag2contr_diagmatrix(cg) == _array_contraction(_array_tensor_product(OneArray(1), a, b, I1), (2, 6))
    assert convert_array_to_matrix(cg) == a*b.T

    cg = _array_diagonal(_array_tensor_product(x, I1), (1, 2))
    assert isinstance(cg, ArrayDiagonal)
    assert cg.diagonal_indices == ((1, 2),)
    assert convert_array_to_matrix(cg) == x

    cg = _array_diagonal(_array_tensor_product(x, I), (0, 2))
    assert _array_diag2contr_diagmatrix(cg) == _array_contraction(_array_tensor_product(OneArray(1), I, DiagMatrix(x)), (1, 3))
    assert convert_array_to_matrix(cg).doit() == DiagMatrix(x)

    raises(ValueError, lambda: _array_diagonal(x, (1,)))

    # Ignore identity matrices with contractions:

    cg = _array_contraction(_array_tensor_product(I, A, I, I), (0, 2), (1, 3), (5, 7))
    assert cg.split_multiple_contractions() == cg
    assert convert_array_to_matrix(cg) == Trace(A) * I

    cg = _array_contraction(_array_tensor_product(Trace(A) * I, I, I), (1, 5), (3, 4))
    assert cg.split_multiple_contractions() == cg
    assert convert_array_to_matrix(cg).doit() == Trace(A) * I

    # Add DiagMatrix when required:

    cg = _array_contraction(_array_tensor_product(A, a), (1, 2))
    assert cg.split_multiple_contractions() == cg
    assert convert_array_to_matrix(cg) == A * a

    cg = _array_contraction(_array_tensor_product(A, a, B), (1, 2, 4))
    assert cg.split_multiple_contractions() == _array_contraction(_array_tensor_product(A, DiagMatrix(a), OneArray(1), B), (1, 2), (3, 5))
    assert convert_array_to_matrix(cg) == A * DiagMatrix(a) * B

    cg = _array_contraction(_array_tensor_product(A, a, B), (0, 2, 4))
    assert cg.split_multiple_contractions() == _array_contraction(_array_tensor_product(A, DiagMatrix(a), OneArray(1), B), (0, 2), (3, 5))
    assert convert_array_to_matrix(cg) == A.T * DiagMatrix(a) * B

    cg = _array_contraction(_array_tensor_product(A, a, b, a.T, B), (0, 2, 4, 7, 9))
    assert cg.split_multiple_contractions() == _array_contraction(_array_tensor_product(A, DiagMatrix(a), OneArray(1),
                                                DiagMatrix(b), OneArray(1), DiagMatrix(a), OneArray(1), B),
                                               (0, 2), (3, 5), (6, 9), (8, 12))
    assert convert_array_to_matrix(cg) == A.T * DiagMatrix(a) * DiagMatrix(b) * DiagMatrix(a) * B.T

    cg = _array_contraction(_array_tensor_product(I1, I1, I1), (1, 2, 4))
    assert cg.split_multiple_contractions() == _array_contraction(_array_tensor_product(I1, I1, OneArray(1), I1), (1, 2), (3, 5))
    assert convert_array_to_matrix(cg) == 1

    cg = _array_contraction(_array_tensor_product(I, I, I, I, A), (1, 2, 8), (5, 6, 9))
    assert convert_array_to_matrix(cg.split_multiple_contractions()).doit() == A

    cg = _array_contraction(_array_tensor_product(A, a, C, a, B), (1, 2, 4), (5, 6, 8))
    expected = _array_contraction(_array_tensor_product(A, DiagMatrix(a), OneArray(1), C, DiagMatrix(a), OneArray(1), B), (1, 3), (2, 5), (6, 7), (8, 10))
    assert cg.split_multiple_contractions() == expected
    assert convert_array_to_matrix(cg) == A * DiagMatrix(a) * C * DiagMatrix(a) * B

    cg = _array_contraction(_array_tensor_product(a, I1, b, I1, (a.T*b).applyfunc(cos)), (1, 2, 8), (5, 6, 9))
    expected = _array_contraction(_array_tensor_product(a, I1, OneArray(1), b, I1, OneArray(1), (a.T*b).applyfunc(cos)),
                                (1, 3), (2, 10), (6, 8), (7, 11))
    assert cg.split_multiple_contractions().dummy_eq(expected)
    assert convert_array_to_matrix(cg).doit().dummy_eq(MatMul(a, (a.T * b).applyfunc(cos), b.T))


def test_arrayexpr_convert_array_contraction_tp_additions():
    a = ArrayAdd(
        _array_tensor_product(M, N),
        _array_tensor_product(N, M)
    )
    tp = _array_tensor_product(P, a, Q)
    expr = _array_contraction(tp, (3, 4))
    expected = _array_tensor_product(
        P,
        ArrayAdd(
            _array_contraction(_array_tensor_product(M, N), (1, 2)),
            _array_contraction(_array_tensor_product(N, M), (1, 2)),
        ),
        Q
    )
    assert expr == expected
    assert convert_array_to_matrix(expr) == _array_tensor_product(P, M * N + N * M, Q)

    expr = _array_contraction(tp, (1, 2), (3, 4), (5, 6))
    result = _array_contraction(
        _array_tensor_product(
            P,
            ArrayAdd(
                _array_contraction(_array_tensor_product(M, N), (1, 2)),
                _array_contraction(_array_tensor_product(N, M), (1, 2)),
            ),
            Q
        ), (1, 2), (3, 4))
    assert expr == result
    assert convert_array_to_matrix(expr) == P * (M * N + N * M) * Q


def test_arrayexpr_convert_array_to_implicit_matmul():
    # Trivial dimensions are suppressed, so the result can be expressed in matrix form:

    cg = _array_tensor_product(a, b)
    assert convert_array_to_matrix(cg) == a * b.T

    cg = _array_tensor_product(a, b, I)
    assert convert_array_to_matrix(cg) == _array_tensor_product(a*b.T, I)

    cg = _array_tensor_product(I, a, b)
    assert convert_array_to_matrix(cg) == _array_tensor_product(I, a*b.T)

    cg = _array_tensor_product(a, I, b)
    assert convert_array_to_matrix(cg) == _array_tensor_product(a, I, b)

    cg = _array_contraction(_array_tensor_product(I, I), (1, 2))
    assert convert_array_to_matrix(cg) == I

    cg = PermuteDims(_array_tensor_product(I, Identity(1)), [0, 2, 1, 3])
    assert convert_array_to_matrix(cg) == I


def test_arrayexpr_convert_array_to_matrix_remove_trivial_dims():

    # Tensor Product:
    assert _remove_trivial_dims(_array_tensor_product(a, b)) == (a * b.T, [1, 3])
    assert _remove_trivial_dims(_array_tensor_product(a.T, b)) == (a * b.T, [0, 3])
    assert _remove_trivial_dims(_array_tensor_product(a, b.T)) == (a * b.T, [1, 2])
    assert _remove_trivial_dims(_array_tensor_product(a.T, b.T)) == (a * b.T, [0, 2])

    assert _remove_trivial_dims(_array_tensor_product(I, a.T, b.T)) == (_array_tensor_product(I, a * b.T), [2, 4])
    assert _remove_trivial_dims(_array_tensor_product(a.T, I, b.T)) == (_array_tensor_product(a.T, I, b.T), [])

    assert _remove_trivial_dims(_array_tensor_product(a, I)) == (_array_tensor_product(a, I), [])
    assert _remove_trivial_dims(_array_tensor_product(I, a)) == (_array_tensor_product(I, a), [])

    assert _remove_trivial_dims(_array_tensor_product(a.T, b.T, c, d)) == (
        _array_tensor_product(a * b.T, c * d.T), [0, 2, 5, 7])
    assert _remove_trivial_dims(_array_tensor_product(a.T, I, b.T, c, d, I)) == (
        _array_tensor_product(a.T, I, b*c.T, d, I), [4, 7])

    # Addition:

    cg = ArrayAdd(_array_tensor_product(a, b), _array_tensor_product(c, d))
    assert _remove_trivial_dims(cg) == (a * b.T + c * d.T, [1, 3])

    # Permute Dims:

    cg = PermuteDims(_array_tensor_product(a, b), Permutation(3)(1, 2))
    assert _remove_trivial_dims(cg) == (a * b.T, [2, 3])

    cg = PermuteDims(_array_tensor_product(a, I, b), Permutation(5)(1, 2, 3, 4))
    assert _remove_trivial_dims(cg) == (cg, [])

    cg = PermuteDims(_array_tensor_product(I, b, a), Permutation(5)(1, 2, 4, 5, 3))
    assert _remove_trivial_dims(cg) == (PermuteDims(_array_tensor_product(I, b * a.T), [0, 2, 3, 1]), [4, 5])

    # Diagonal:

    cg = _array_diagonal(_array_tensor_product(M, a), (1, 2))
    assert _remove_trivial_dims(cg) == (cg, [])

    # Contraction:

    cg = _array_contraction(_array_tensor_product(M, a), (1, 2))
    assert _remove_trivial_dims(cg) == (cg, [])

    # A few more cases to test the removal and shift of nested removed axes
    # with array contractions and array diagonals:
    tp = _array_tensor_product(
        OneMatrix(1, 1),
        M,
        x,
        OneMatrix(1, 1),
        Identity(1),
    )

    expr = _array_contraction(tp, (1, 8))
    rexpr, removed = _remove_trivial_dims(expr)
    assert removed == [0, 5, 6, 7]

    expr = _array_contraction(tp, (1, 8), (3, 4))
    rexpr, removed = _remove_trivial_dims(expr)
    assert removed == [0, 3, 4, 5]

    expr = _array_diagonal(tp, (1, 8))
    rexpr, removed = _remove_trivial_dims(expr)
    assert removed == [0, 5, 6, 7, 8]

    expr = _array_diagonal(tp, (1, 8), (3, 4))
    rexpr, removed = _remove_trivial_dims(expr)
    assert removed == [0, 3, 4, 5, 6]

    expr = _array_diagonal(_array_contraction(_array_tensor_product(A, x, I, I1), (1, 2, 5)), (1, 4))
    rexpr, removed = _remove_trivial_dims(expr)
    assert removed == [2, 3]

    cg = _array_diagonal(_array_tensor_product(PermuteDims(_array_tensor_product(x, I1), Permutation(1, 2, 3)), (x.T*x).applyfunc(sqrt)), (2, 4), (3, 5))
    rexpr, removed = _remove_trivial_dims(cg)
    assert removed == [1, 2]

    # Contractions with identity matrices need to be followed by a permutation
    # in order
    cg = _array_contraction(_array_tensor_product(A, B, C, M, I), (1, 8))
    ret, removed = _remove_trivial_dims(cg)
    assert ret == PermuteDims(_array_tensor_product(A, B, C, M), [0, 2, 3, 4, 5, 6, 7, 1])
    assert removed == []

    cg = _array_contraction(_array_tensor_product(A, B, C, M, I), (1, 8), (3, 4))
    ret, removed = _remove_trivial_dims(cg)
    assert ret == PermuteDims(_array_contraction(_array_tensor_product(A, B, C, M), (3, 4)), [0, 2, 3, 4, 5, 1])
    assert removed == []

    # Trivial matrices are sometimes inserted into MatMul expressions:

    cg = _array_tensor_product(b*b.T, a.T*a)
    ret, removed = _remove_trivial_dims(cg)
    assert ret == b*a.T*a*b.T
    assert removed == [2, 3]

    Xs = ArraySymbol("X", (3, 2, k))
    cg = _array_tensor_product(M, Xs, b.T*c, a*a.T, b*b.T, c.T*d)
    ret, removed = _remove_trivial_dims(cg)
    assert ret == _array_tensor_product(M, Xs, a*b.T*c*c.T*d*a.T, b*b.T)
    assert removed == [5, 6, 11, 12]

    cg = _array_diagonal(_array_tensor_product(I, I1, x), (1, 4), (3, 5))
    assert _remove_trivial_dims(cg) == (PermuteDims(_array_diagonal(_array_tensor_product(I, x), (1, 2)), Permutation(1, 2)), [1])

    expr = _array_diagonal(_array_tensor_product(x, I, y), (0, 2))
    assert _remove_trivial_dims(expr) == (PermuteDims(_array_tensor_product(DiagMatrix(x), y), [1, 2, 3, 0]), [0])

    expr = _array_diagonal(_array_tensor_product(x, I, y), (0, 2), (3, 4))
    assert _remove_trivial_dims(expr) == (expr, [])


def test_arrayexpr_convert_array_to_matrix_diag2contraction_diagmatrix():
    cg = _array_diagonal(_array_tensor_product(M, a), (1, 2))
    res = _array_diag2contr_diagmatrix(cg)
    assert res.shape == cg.shape
    assert res == _array_contraction(_array_tensor_product(M, OneArray(1), DiagMatrix(a)), (1, 3))

    raises(ValueError, lambda: _array_diagonal(_array_tensor_product(a, M), (1, 2)))

    cg = _array_diagonal(_array_tensor_product(a.T, M), (1, 2))
    res = _array_diag2contr_diagmatrix(cg)
    assert res.shape == cg.shape
    assert res == _array_contraction(_array_tensor_product(OneArray(1), M, DiagMatrix(a.T)), (1, 4))

    cg = _array_diagonal(_array_tensor_product(a.T, M, N, b.T), (1, 2), (4, 7))
    res = _array_diag2contr_diagmatrix(cg)
    assert res.shape == cg.shape
    assert res == _array_contraction(
        _array_tensor_product(OneArray(1), M, N, OneArray(1), DiagMatrix(a.T), DiagMatrix(b.T)), (1, 7), (3, 9))

    cg = _array_diagonal(_array_tensor_product(a, M, N, b.T), (0, 2), (4, 7))
    res = _array_diag2contr_diagmatrix(cg)
    assert res.shape == cg.shape
    assert res == _array_contraction(
        _array_tensor_product(OneArray(1), M, N, OneArray(1), DiagMatrix(a), DiagMatrix(b.T)), (1, 6), (3, 9))

    cg = _array_diagonal(_array_tensor_product(a, M, N, b.T), (0, 4), (3, 7))
    res = _array_diag2contr_diagmatrix(cg)
    assert res.shape == cg.shape
    assert res == _array_contraction(
        _array_tensor_product(OneArray(1), M, N, OneArray(1), DiagMatrix(a), DiagMatrix(b.T)), (3, 6), (2, 9))

    I1 = Identity(1)
    x = MatrixSymbol("x", k, 1)
    A = MatrixSymbol("A", k, k)
    cg = _array_diagonal(_array_tensor_product(x, A.T, I1), (0, 2))
    assert _array_diag2contr_diagmatrix(cg).shape == cg.shape
    assert _array2matrix(cg).shape == cg.shape


def test_arrayexpr_convert_array_to_matrix_support_function():

    assert _support_function_tp1_recognize([], [2 * k]) == 2 * k

    assert _support_function_tp1_recognize([(1, 2)], [A, 2 * k, B, 3]) == 6 * k * A * B

    assert _support_function_tp1_recognize([(0, 3), (1, 2)], [A, B]) == Trace(A * B)

    assert _support_function_tp1_recognize([(1, 2)], [A, B]) == A * B
    assert _support_function_tp1_recognize([(0, 2)], [A, B]) == A.T * B
    assert _support_function_tp1_recognize([(1, 3)], [A, B]) == A * B.T
    assert _support_function_tp1_recognize([(0, 3)], [A, B]) == A.T * B.T

    assert _support_function_tp1_recognize([(1, 2), (5, 6)], [A, B, C, D]) == _array_tensor_product(A * B, C * D)
    assert _support_function_tp1_recognize([(1, 4), (3, 6)], [A, B, C, D]) == PermuteDims(
        _array_tensor_product(A * C, B * D), [0, 2, 1, 3])

    assert _support_function_tp1_recognize([(0, 3), (1, 4)], [A, B, C]) == B * A * C

    assert _support_function_tp1_recognize([(9, 10), (1, 2), (5, 6), (3, 4), (7, 8)],
                                           [X, Y, A, B, C, D]) == X * Y * A * B * C * D

    assert _support_function_tp1_recognize([(9, 10), (1, 2), (5, 6), (3, 4)],
                                           [X, Y, A, B, C, D]) == _array_tensor_product(X * Y * A * B, C * D)

    assert _support_function_tp1_recognize([(1, 7), (3, 8), (4, 11)], [X, Y, A, B, C, D]) == PermuteDims(
        _array_tensor_product(X * B.T, Y * C, A.T * D.T), [0, 2, 4, 1, 3, 5]
    )

    assert _support_function_tp1_recognize([(0, 1), (3, 6), (5, 8)], [X, A, B, C, D]) == PermuteDims(
        _array_tensor_product(Trace(X) * A * C, B * D), [0, 2, 1, 3])

    assert _support_function_tp1_recognize([(1, 2), (3, 4), (5, 6), (7, 8)], [A, A, B, C, D]) == A ** 2 * B * C * D
    assert _support_function_tp1_recognize([(1, 2), (3, 4), (5, 6), (7, 8)], [X, A, B, C, D]) == X * A * B * C * D

    assert _support_function_tp1_recognize([(1, 6), (3, 8), (5, 10)], [X, Y, A, B, C, D]) == PermuteDims(
        _array_tensor_product(X * B, Y * C, A * D), [0, 2, 4, 1, 3, 5]
    )

    assert _support_function_tp1_recognize([(1, 4), (3, 6)], [A, B, C, D]) == PermuteDims(
        _array_tensor_product(A * C, B * D), [0, 2, 1, 3])

    assert _support_function_tp1_recognize([(0, 4), (1, 7), (2, 5), (3, 8)], [X, A, B, C, D]) == C*X.T*B*A*D

    assert _support_function_tp1_recognize([(0, 4), (1, 7), (2, 5), (3, 8)], [X, A, B, C, D]) == C*X.T*B*A*D


def test_convert_array_to_hadamard_products():

    expr = HadamardProduct(M, N)
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert ret == expr

    expr = HadamardProduct(M, N)*P
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert ret == expr

    expr = Q*HadamardProduct(M, N)*P
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert ret == expr

    expr = Q*HadamardProduct(M, N.T)*P
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert ret == expr

    expr = HadamardProduct(M, N)*HadamardProduct(Q, P)
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert expr == ret

    expr = P.T*HadamardProduct(M, N)*HadamardProduct(Q, P)
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert expr == ret

    # ArrayDiagonal should be converted
    cg = _array_diagonal(_array_tensor_product(M, N, Q), (1, 3), (0, 2, 4))
    ret = convert_array_to_matrix(cg)
    expected = PermuteDims(_array_diagonal(_array_tensor_product(HadamardProduct(M.T, N.T), Q), (1, 2)), [1, 0, 2])
    assert expected == ret

    # Special case that should return the same expression:
    cg = _array_diagonal(_array_tensor_product(HadamardProduct(M, N), Q), (0, 2))
    ret = convert_array_to_matrix(cg)
    assert ret == cg

    # Hadamard products with traces:

    expr = Trace(HadamardProduct(M, N))
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert ret == Trace(HadamardProduct(M.T, N.T))

    expr = Trace(A*HadamardProduct(M, N))
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert ret == Trace(HadamardProduct(M, N)*A)

    expr = Trace(HadamardProduct(A, M)*N)
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert ret == Trace(HadamardProduct(M.T, N)*A)

    # These should not be converted into Hadamard products:

    cg = _array_diagonal(_array_tensor_product(M, N), (0, 1, 2, 3))
    ret = convert_array_to_matrix(cg)
    assert ret == cg

    cg = _array_diagonal(_array_tensor_product(A), (0, 1))
    ret = convert_array_to_matrix(cg)
    assert ret == cg

    cg = _array_diagonal(_array_tensor_product(M, N, P), (0, 2, 4), (1, 3, 5))
    assert convert_array_to_matrix(cg) == HadamardProduct(M, N, P)

    cg = _array_diagonal(_array_tensor_product(M, N, P), (0, 3, 4), (1, 2, 5))
    assert convert_array_to_matrix(cg) == HadamardProduct(M, P, N.T)

    cg = _array_diagonal(_array_tensor_product(I, I1, x), (1, 4), (3, 5))
    assert convert_array_to_matrix(cg) == DiagMatrix(x)


def test_identify_removable_identity_matrices():

    D = DiagonalMatrix(MatrixSymbol("D", k, k))

    cg = _array_contraction(_array_tensor_product(A, B, I), (1, 2, 4, 5))
    expected = _array_contraction(_array_tensor_product(A, B), (1, 2))
    assert identify_removable_identity_matrices(cg) == expected

    cg = _array_contraction(_array_tensor_product(A, B, C, I), (1, 3, 5, 6, 7))
    expected = _array_contraction(_array_tensor_product(A, B, C), (1, 3, 5))
    assert identify_removable_identity_matrices(cg) == expected

    # Tests with diagonal matrices:

    cg = _array_contraction(_array_tensor_product(A, B, D), (1, 2, 4, 5))
    ret = identify_removable_identity_matrices(cg)
    expected = _array_contraction(_array_tensor_product(A, B, D), (1, 4), (2, 5))
    assert ret == expected

    cg = _array_contraction(_array_tensor_product(A, B, D, M, N), (1, 2, 4, 5, 6, 8))
    ret = identify_removable_identity_matrices(cg)
    assert ret == cg


def test_combine_removed():

    assert _combine_removed(6, [0, 1, 2], [0, 1, 2]) == [0, 1, 2, 3, 4, 5]
    assert _combine_removed(8, [2, 5], [1, 3, 4]) == [1, 2, 4, 5, 6]
    assert _combine_removed(8, [7], []) == [7]


def test_array_contraction_to_diagonal_multiple_identities():

    expr = _array_contraction(_array_tensor_product(A, B, I, C), (1, 2, 4), (5, 6))
    assert _array_contraction_to_diagonal_multiple_identity(expr) == (expr, [])
    assert convert_array_to_matrix(expr) == _array_contraction(_array_tensor_product(A, B, C), (1, 2, 4))

    expr = _array_contraction(_array_tensor_product(A, I, I), (1, 2, 4))
    assert _array_contraction_to_diagonal_multiple_identity(expr) == (A, [2])
    assert convert_array_to_matrix(expr) == A

    expr = _array_contraction(_array_tensor_product(A, I, I, B), (1, 2, 4), (3, 6))
    assert _array_contraction_to_diagonal_multiple_identity(expr) == (expr, [])

    expr = _array_contraction(_array_tensor_product(A, I, I, B), (1, 2, 3, 4, 6))
    assert _array_contraction_to_diagonal_multiple_identity(expr) == (expr, [])


def test_convert_array_element_to_matrix():

    expr = ArrayElement(M, (i, j))
    assert convert_array_to_matrix(expr) == MatrixElement(M, i, j)

    expr = ArrayElement(_array_contraction(_array_tensor_product(M, N), (1, 3)), (i, j))
    assert convert_array_to_matrix(expr) == MatrixElement(M*N.T, i, j)

    expr = ArrayElement(_array_tensor_product(M, N), (i, j, m, n))
    assert convert_array_to_matrix(expr) == expr


def test_convert_array_elementwise_function_to_matrix():

    d = Dummy("d")

    expr = ArrayElementwiseApplyFunc(Lambda(d, sin(d)), x.T*y)
    assert convert_array_to_matrix(expr) == sin(x.T*y)

    expr = ArrayElementwiseApplyFunc(Lambda(d, d**2), x.T*y)
    assert convert_array_to_matrix(expr) == (x.T*y)**2

    expr = ArrayElementwiseApplyFunc(Lambda(d, sin(d)), x)
    assert convert_array_to_matrix(expr).dummy_eq(x.applyfunc(sin))

    expr = ArrayElementwiseApplyFunc(Lambda(d, 1 / (2 * sqrt(d))), x)
    assert convert_array_to_matrix(expr) == S.Half * HadamardPower(x, -S.Half)


def test_array2matrix():
    # See issue https://github.com/sympy/sympy/pull/22877
    expr = PermuteDims(ArrayContraction(ArrayTensorProduct(x, I, I1, x), (0, 3), (1, 7)), Permutation(2, 3))
    expected = PermuteDims(ArrayTensorProduct(x*x.T, I1), Permutation(3)(1, 2))
    assert _array2matrix(expr) == expected


def test_recognize_broadcasting():
    expr = ArrayTensorProduct(x.T*x, A)
    assert _remove_trivial_dims(expr) == (KroneckerProduct(x.T*x, A), [0, 1])

    expr = ArrayTensorProduct(A, x.T*x)
    assert _remove_trivial_dims(expr) == (KroneckerProduct(A, x.T*x), [2, 3])

    expr = ArrayTensorProduct(A, B, x.T*x, C)
    assert _remove_trivial_dims(expr) == (ArrayTensorProduct(A, KroneckerProduct(B, x.T*x), C), [4, 5])

    # Always prefer matrix multiplication to Kronecker product, if possible:
    expr = ArrayTensorProduct(a, b, x.T*x)
    assert _remove_trivial_dims(expr) == (a*x.T*x*b.T, [1, 3, 4, 5])
