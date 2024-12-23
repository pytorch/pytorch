from copy import copy

from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
from sympy.core.containers import Dict
from sympy.core.function import diff
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.matrices import SparseMatrix
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.matrices import Matrix
from sympy.tensor.array.sparse_ndim_array import ImmutableSparseNDimArray
from sympy.testing.pytest import raises


def test_ndim_array_initiation():
    arr_with_no_elements = ImmutableDenseNDimArray([], shape=(0,))
    assert len(arr_with_no_elements) == 0
    assert arr_with_no_elements.rank() == 1

    raises(ValueError, lambda: ImmutableDenseNDimArray([0], shape=(0,)))
    raises(ValueError, lambda: ImmutableDenseNDimArray([1, 2, 3], shape=(0,)))
    raises(ValueError, lambda: ImmutableDenseNDimArray([], shape=()))

    raises(ValueError, lambda: ImmutableSparseNDimArray([0], shape=(0,)))
    raises(ValueError, lambda: ImmutableSparseNDimArray([1, 2, 3], shape=(0,)))
    raises(ValueError, lambda: ImmutableSparseNDimArray([], shape=()))

    arr_with_one_element = ImmutableDenseNDimArray([23])
    assert len(arr_with_one_element) == 1
    assert arr_with_one_element[0] == 23
    assert arr_with_one_element[:] == ImmutableDenseNDimArray([23])
    assert arr_with_one_element.rank() == 1

    arr_with_symbol_element = ImmutableDenseNDimArray([Symbol('x')])
    assert len(arr_with_symbol_element) == 1
    assert arr_with_symbol_element[0] == Symbol('x')
    assert arr_with_symbol_element[:] == ImmutableDenseNDimArray([Symbol('x')])
    assert arr_with_symbol_element.rank() == 1

    number5 = 5
    vector = ImmutableDenseNDimArray.zeros(number5)
    assert len(vector) == number5
    assert vector.shape == (number5,)
    assert vector.rank() == 1

    vector = ImmutableSparseNDimArray.zeros(number5)
    assert len(vector) == number5
    assert vector.shape == (number5,)
    assert vector._sparse_array == Dict()
    assert vector.rank() == 1

    n_dim_array = ImmutableDenseNDimArray(range(3**4), (3, 3, 3, 3,))
    assert len(n_dim_array) == 3 * 3 * 3 * 3
    assert n_dim_array.shape == (3, 3, 3, 3)
    assert n_dim_array.rank() == 4

    array_shape = (3, 3, 3, 3)
    sparse_array = ImmutableSparseNDimArray.zeros(*array_shape)
    assert len(sparse_array._sparse_array) == 0
    assert len(sparse_array) == 3 * 3 * 3 * 3
    assert n_dim_array.shape == array_shape
    assert n_dim_array.rank() == 4

    one_dim_array = ImmutableDenseNDimArray([2, 3, 1])
    assert len(one_dim_array) == 3
    assert one_dim_array.shape == (3,)
    assert one_dim_array.rank() == 1
    assert one_dim_array.tolist() == [2, 3, 1]

    shape = (3, 3)
    array_with_many_args = ImmutableSparseNDimArray.zeros(*shape)
    assert len(array_with_many_args) == 3 * 3
    assert array_with_many_args.shape == shape
    assert array_with_many_args[0, 0] == 0
    assert array_with_many_args.rank() == 2

    shape = (int(3), int(3))
    array_with_long_shape = ImmutableSparseNDimArray.zeros(*shape)
    assert len(array_with_long_shape) == 3 * 3
    assert array_with_long_shape.shape == shape
    assert array_with_long_shape[int(0), int(0)] == 0
    assert array_with_long_shape.rank() == 2

    vector_with_long_shape = ImmutableDenseNDimArray(range(5), int(5))
    assert len(vector_with_long_shape) == 5
    assert vector_with_long_shape.shape == (int(5),)
    assert vector_with_long_shape.rank() == 1
    raises(ValueError, lambda: vector_with_long_shape[int(5)])

    from sympy.abc import x
    for ArrayType in [ImmutableDenseNDimArray, ImmutableSparseNDimArray]:
        rank_zero_array = ArrayType(x)
        assert len(rank_zero_array) == 1
        assert rank_zero_array.shape == ()
        assert rank_zero_array.rank() == 0
        assert rank_zero_array[()] == x
        raises(ValueError, lambda: rank_zero_array[0])


def test_reshape():
    array = ImmutableDenseNDimArray(range(50), 50)
    assert array.shape == (50,)
    assert array.rank() == 1

    array = array.reshape(5, 5, 2)
    assert array.shape == (5, 5, 2)
    assert array.rank() == 3
    assert len(array) == 50


def test_getitem():
    for ArrayType in [ImmutableDenseNDimArray, ImmutableSparseNDimArray]:
        array = ArrayType(range(24)).reshape(2, 3, 4)
        assert array.tolist() == [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]
        assert array[0] == ArrayType([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
        assert array[0, 0] == ArrayType([0, 1, 2, 3])
        value = 0
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    assert array[i, j, k] == value
                    value += 1

    raises(ValueError, lambda: array[3, 4, 5])
    raises(ValueError, lambda: array[3, 4, 5, 6])
    raises(ValueError, lambda: array[3, 4, 5, 3:4])


def test_iterator():
    array = ImmutableDenseNDimArray(range(4), (2, 2))
    assert array[0] == ImmutableDenseNDimArray([0, 1])
    assert array[1] == ImmutableDenseNDimArray([2, 3])

    array = array.reshape(4)
    j = 0
    for i in array:
        assert i == j
        j += 1


def test_sparse():
    sparse_array = ImmutableSparseNDimArray([0, 0, 0, 1], (2, 2))
    assert len(sparse_array) == 2 * 2
    # dictionary where all data is, only non-zero entries are actually stored:
    assert len(sparse_array._sparse_array) == 1

    assert sparse_array.tolist() == [[0, 0], [0, 1]]

    for i, j in zip(sparse_array, [[0, 0], [0, 1]]):
        assert i == ImmutableSparseNDimArray(j)

    def sparse_assignment():
        sparse_array[0, 0] = 123

    assert len(sparse_array._sparse_array) == 1
    raises(TypeError, sparse_assignment)
    assert len(sparse_array._sparse_array) == 1
    assert sparse_array[0, 0] == 0
    assert sparse_array/0 == ImmutableSparseNDimArray([[S.NaN, S.NaN], [S.NaN, S.ComplexInfinity]], (2, 2))

    # test for large scale sparse array
    # equality test
    assert ImmutableSparseNDimArray.zeros(100000, 200000) == ImmutableSparseNDimArray.zeros(100000, 200000)

    # __mul__ and __rmul__
    a = ImmutableSparseNDimArray({200001: 1}, (100000, 200000))
    assert a * 3 == ImmutableSparseNDimArray({200001: 3}, (100000, 200000))
    assert 3 * a == ImmutableSparseNDimArray({200001: 3}, (100000, 200000))
    assert a * 0 == ImmutableSparseNDimArray({}, (100000, 200000))
    assert 0 * a == ImmutableSparseNDimArray({}, (100000, 200000))

    # __truediv__
    assert a/3 == ImmutableSparseNDimArray({200001: Rational(1, 3)}, (100000, 200000))

    # __neg__
    assert -a == ImmutableSparseNDimArray({200001: -1}, (100000, 200000))


def test_calculation():

    a = ImmutableDenseNDimArray([1]*9, (3, 3))
    b = ImmutableDenseNDimArray([9]*9, (3, 3))

    c = a + b
    for i in c:
        assert i == ImmutableDenseNDimArray([10, 10, 10])

    assert c == ImmutableDenseNDimArray([10]*9, (3, 3))
    assert c == ImmutableSparseNDimArray([10]*9, (3, 3))

    c = b - a
    for i in c:
        assert i == ImmutableDenseNDimArray([8, 8, 8])

    assert c == ImmutableDenseNDimArray([8]*9, (3, 3))
    assert c == ImmutableSparseNDimArray([8]*9, (3, 3))


def test_ndim_array_converting():
    dense_array = ImmutableDenseNDimArray([1, 2, 3, 4], (2, 2))
    alist = dense_array.tolist()

    assert alist == [[1, 2], [3, 4]]

    matrix = dense_array.tomatrix()
    assert (isinstance(matrix, Matrix))

    for i in range(len(dense_array)):
        assert dense_array[dense_array._get_tuple_index(i)] == matrix[i]
    assert matrix.shape == dense_array.shape

    assert ImmutableDenseNDimArray(matrix) == dense_array
    assert ImmutableDenseNDimArray(matrix.as_immutable()) == dense_array
    assert ImmutableDenseNDimArray(matrix.as_mutable()) == dense_array

    sparse_array = ImmutableSparseNDimArray([1, 2, 3, 4], (2, 2))
    alist = sparse_array.tolist()

    assert alist == [[1, 2], [3, 4]]

    matrix = sparse_array.tomatrix()
    assert(isinstance(matrix, SparseMatrix))

    for i in range(len(sparse_array)):
        assert sparse_array[sparse_array._get_tuple_index(i)] == matrix[i]
    assert matrix.shape == sparse_array.shape

    assert ImmutableSparseNDimArray(matrix) == sparse_array
    assert ImmutableSparseNDimArray(matrix.as_immutable()) == sparse_array
    assert ImmutableSparseNDimArray(matrix.as_mutable()) == sparse_array


def test_converting_functions():
    arr_list = [1, 2, 3, 4]
    arr_matrix = Matrix(((1, 2), (3, 4)))

    # list
    arr_ndim_array = ImmutableDenseNDimArray(arr_list, (2, 2))
    assert (isinstance(arr_ndim_array, ImmutableDenseNDimArray))
    assert arr_matrix.tolist() == arr_ndim_array.tolist()

    # Matrix
    arr_ndim_array = ImmutableDenseNDimArray(arr_matrix)
    assert (isinstance(arr_ndim_array, ImmutableDenseNDimArray))
    assert arr_matrix.tolist() == arr_ndim_array.tolist()
    assert arr_matrix.shape == arr_ndim_array.shape


def test_equality():
    first_list = [1, 2, 3, 4]
    second_list = [1, 2, 3, 4]
    third_list = [4, 3, 2, 1]
    assert first_list == second_list
    assert first_list != third_list

    first_ndim_array = ImmutableDenseNDimArray(first_list, (2, 2))
    second_ndim_array = ImmutableDenseNDimArray(second_list, (2, 2))
    fourth_ndim_array = ImmutableDenseNDimArray(first_list, (2, 2))

    assert first_ndim_array == second_ndim_array

    def assignment_attempt(a):
        a[0, 0] = 0

    raises(TypeError, lambda: assignment_attempt(second_ndim_array))
    assert first_ndim_array == second_ndim_array
    assert first_ndim_array == fourth_ndim_array


def test_arithmetic():
    a = ImmutableDenseNDimArray([3 for i in range(9)], (3, 3))
    b = ImmutableDenseNDimArray([7 for i in range(9)], (3, 3))

    c1 = a + b
    c2 = b + a
    assert c1 == c2

    d1 = a - b
    d2 = b - a
    assert d1 == d2 * (-1)

    e1 = a * 5
    e2 = 5 * a
    e3 = copy(a)
    e3 *= 5
    assert e1 == e2 == e3

    f1 = a / 5
    f2 = copy(a)
    f2 /= 5
    assert f1 == f2
    assert f1[0, 0] == f1[0, 1] == f1[0, 2] == f1[1, 0] == f1[1, 1] == \
    f1[1, 2] == f1[2, 0] == f1[2, 1] == f1[2, 2] == Rational(3, 5)

    assert type(a) == type(b) == type(c1) == type(c2) == type(d1) == type(d2) \
        == type(e1) == type(e2) == type(e3) == type(f1)

    z0 = -a
    assert z0 == ImmutableDenseNDimArray([-3 for i in range(9)], (3, 3))


def test_higher_dimenions():
    m3 = ImmutableDenseNDimArray(range(10, 34), (2, 3, 4))

    assert m3.tolist() == [[[10, 11, 12, 13],
            [14, 15, 16, 17],
            [18, 19, 20, 21]],

           [[22, 23, 24, 25],
            [26, 27, 28, 29],
            [30, 31, 32, 33]]]

    assert m3._get_tuple_index(0) == (0, 0, 0)
    assert m3._get_tuple_index(1) == (0, 0, 1)
    assert m3._get_tuple_index(4) == (0, 1, 0)
    assert m3._get_tuple_index(12) == (1, 0, 0)

    assert str(m3) == '[[[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]], [[22, 23, 24, 25], [26, 27, 28, 29], [30, 31, 32, 33]]]'

    m3_rebuilt = ImmutableDenseNDimArray([[[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]], [[22, 23, 24, 25], [26, 27, 28, 29], [30, 31, 32, 33]]])
    assert m3 == m3_rebuilt

    m3_other = ImmutableDenseNDimArray([[[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]], [[22, 23, 24, 25], [26, 27, 28, 29], [30, 31, 32, 33]]], (2, 3, 4))

    assert m3 == m3_other


def test_rebuild_immutable_arrays():
    sparr = ImmutableSparseNDimArray(range(10, 34), (2, 3, 4))
    densarr = ImmutableDenseNDimArray(range(10, 34), (2, 3, 4))

    assert sparr == sparr.func(*sparr.args)
    assert densarr == densarr.func(*densarr.args)


def test_slices():
    md = ImmutableDenseNDimArray(range(10, 34), (2, 3, 4))

    assert md[:] == ImmutableDenseNDimArray(range(10, 34), (2, 3, 4))
    assert md[:, :, 0].tomatrix() == Matrix([[10, 14, 18], [22, 26, 30]])
    assert md[0, 1:2, :].tomatrix() == Matrix([[14, 15, 16, 17]])
    assert md[0, 1:3, :].tomatrix() == Matrix([[14, 15, 16, 17], [18, 19, 20, 21]])
    assert md[:, :, :] == md

    sd = ImmutableSparseNDimArray(range(10, 34), (2, 3, 4))
    assert sd == ImmutableSparseNDimArray(md)

    assert sd[:] == ImmutableSparseNDimArray(range(10, 34), (2, 3, 4))
    assert sd[:, :, 0].tomatrix() == Matrix([[10, 14, 18], [22, 26, 30]])
    assert sd[0, 1:2, :].tomatrix() == Matrix([[14, 15, 16, 17]])
    assert sd[0, 1:3, :].tomatrix() == Matrix([[14, 15, 16, 17], [18, 19, 20, 21]])
    assert sd[:, :, :] == sd


def test_diff_and_applyfunc():
    from sympy.abc import x, y, z
    md = ImmutableDenseNDimArray([[x, y], [x*z, x*y*z]])
    assert md.diff(x) == ImmutableDenseNDimArray([[1, 0], [z, y*z]])
    assert diff(md, x) == ImmutableDenseNDimArray([[1, 0], [z, y*z]])

    sd = ImmutableSparseNDimArray(md)
    assert sd == ImmutableSparseNDimArray([x, y, x*z, x*y*z], (2, 2))
    assert sd.diff(x) == ImmutableSparseNDimArray([[1, 0], [z, y*z]])
    assert diff(sd, x) == ImmutableSparseNDimArray([[1, 0], [z, y*z]])

    mdn = md.applyfunc(lambda x: x*3)
    assert mdn == ImmutableDenseNDimArray([[3*x, 3*y], [3*x*z, 3*x*y*z]])
    assert md != mdn

    sdn = sd.applyfunc(lambda x: x/2)
    assert sdn == ImmutableSparseNDimArray([[x/2, y/2], [x*z/2, x*y*z/2]])
    assert sd != sdn

    sdp = sd.applyfunc(lambda x: x+1)
    assert sdp == ImmutableSparseNDimArray([[x + 1, y + 1], [x*z + 1, x*y*z + 1]])
    assert sd != sdp


def test_op_priority():
    from sympy.abc import x
    md = ImmutableDenseNDimArray([1, 2, 3])
    e1 = (1+x)*md
    e2 = md*(1+x)
    assert e1 == ImmutableDenseNDimArray([1+x, 2+2*x, 3+3*x])
    assert e1 == e2

    sd = ImmutableSparseNDimArray([1, 2, 3])
    e3 = (1+x)*sd
    e4 = sd*(1+x)
    assert e3 == ImmutableDenseNDimArray([1+x, 2+2*x, 3+3*x])
    assert e3 == e4


def test_symbolic_indexing():
    x, y, z, w = symbols("x y z w")
    M = ImmutableDenseNDimArray([[x, y], [z, w]])
    i, j = symbols("i, j")
    Mij = M[i, j]
    assert isinstance(Mij, Indexed)
    Ms = ImmutableSparseNDimArray([[2, 3*x], [4, 5]])
    msij = Ms[i, j]
    assert isinstance(msij, Indexed)
    for oi, oj in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        assert Mij.subs({i: oi, j: oj}) == M[oi, oj]
        assert msij.subs({i: oi, j: oj}) == Ms[oi, oj]
    A = IndexedBase("A", (0, 2))
    assert A[0, 0].subs(A, M) == x
    assert A[i, j].subs(A, M) == M[i, j]
    assert M[i, j].subs(M, A) == A[i, j]

    assert isinstance(M[3 * i - 2, j], Indexed)
    assert M[3 * i - 2, j].subs({i: 1, j: 0}) == M[1, 0]
    assert isinstance(M[i, 0], Indexed)
    assert M[i, 0].subs(i, 0) == M[0, 0]
    assert M[0, i].subs(i, 1) == M[0, 1]

    assert M[i, j].diff(x) == ImmutableDenseNDimArray([[1, 0], [0, 0]])[i, j]
    assert Ms[i, j].diff(x) == ImmutableSparseNDimArray([[0, 3], [0, 0]])[i, j]

    Mo = ImmutableDenseNDimArray([1, 2, 3])
    assert Mo[i].subs(i, 1) == 2
    Mos = ImmutableSparseNDimArray([1, 2, 3])
    assert Mos[i].subs(i, 1) == 2

    raises(ValueError, lambda: M[i, 2])
    raises(ValueError, lambda: M[i, -1])
    raises(ValueError, lambda: M[2, i])
    raises(ValueError, lambda: M[-1, i])

    raises(ValueError, lambda: Ms[i, 2])
    raises(ValueError, lambda: Ms[i, -1])
    raises(ValueError, lambda: Ms[2, i])
    raises(ValueError, lambda: Ms[-1, i])


def test_issue_12665():
    # Testing Python 3 hash of immutable arrays:
    arr = ImmutableDenseNDimArray([1, 2, 3])
    # This should NOT raise an exception:
    hash(arr)


def test_zeros_without_shape():
    arr = ImmutableDenseNDimArray.zeros()
    assert arr == ImmutableDenseNDimArray(0)

def test_issue_21870():
    a0 = ImmutableDenseNDimArray(0)
    assert a0.rank() == 0
    a1 = ImmutableDenseNDimArray(a0)
    assert a1.rank() == 0
