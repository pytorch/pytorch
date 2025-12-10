from copy import copy

from sympy.tensor.array.dense_ndim_array import MutableDenseNDimArray
from sympy.core.function import diff
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.matrices import SparseMatrix
from sympy.matrices import Matrix
from sympy.tensor.array.sparse_ndim_array import MutableSparseNDimArray
from sympy.testing.pytest import raises


def test_ndim_array_initiation():
    arr_with_one_element = MutableDenseNDimArray([23])
    assert len(arr_with_one_element) == 1
    assert arr_with_one_element[0] == 23
    assert arr_with_one_element.rank() == 1
    raises(ValueError, lambda: arr_with_one_element[1])

    arr_with_symbol_element = MutableDenseNDimArray([Symbol('x')])
    assert len(arr_with_symbol_element) == 1
    assert arr_with_symbol_element[0] == Symbol('x')
    assert arr_with_symbol_element.rank() == 1

    number5 = 5
    vector = MutableDenseNDimArray.zeros(number5)
    assert len(vector) == number5
    assert vector.shape == (number5,)
    assert vector.rank() == 1
    raises(ValueError, lambda: arr_with_one_element[5])

    vector = MutableSparseNDimArray.zeros(number5)
    assert len(vector) == number5
    assert vector.shape == (number5,)
    assert vector._sparse_array == {}
    assert vector.rank() == 1

    n_dim_array = MutableDenseNDimArray(range(3**4), (3, 3, 3, 3,))
    assert len(n_dim_array) == 3 * 3 * 3 * 3
    assert n_dim_array.shape == (3, 3, 3, 3)
    assert n_dim_array.rank() == 4
    raises(ValueError, lambda: n_dim_array[0, 0, 0, 3])
    raises(ValueError, lambda: n_dim_array[3, 0, 0, 0])
    raises(ValueError, lambda: n_dim_array[3**4])

    array_shape = (3, 3, 3, 3)
    sparse_array = MutableSparseNDimArray.zeros(*array_shape)
    assert len(sparse_array._sparse_array) == 0
    assert len(sparse_array) == 3 * 3 * 3 * 3
    assert n_dim_array.shape == array_shape
    assert n_dim_array.rank() == 4

    one_dim_array = MutableDenseNDimArray([2, 3, 1])
    assert len(one_dim_array) == 3
    assert one_dim_array.shape == (3,)
    assert one_dim_array.rank() == 1
    assert one_dim_array.tolist() == [2, 3, 1]

    shape = (3, 3)
    array_with_many_args = MutableSparseNDimArray.zeros(*shape)
    assert len(array_with_many_args) == 3 * 3
    assert array_with_many_args.shape == shape
    assert array_with_many_args[0, 0] == 0
    assert array_with_many_args.rank() == 2

    shape = (int(3), int(3))
    array_with_long_shape = MutableSparseNDimArray.zeros(*shape)
    assert len(array_with_long_shape) == 3 * 3
    assert array_with_long_shape.shape == shape
    assert array_with_long_shape[int(0), int(0)] == 0
    assert array_with_long_shape.rank() == 2

    vector_with_long_shape = MutableDenseNDimArray(range(5), int(5))
    assert len(vector_with_long_shape) == 5
    assert vector_with_long_shape.shape == (int(5),)
    assert vector_with_long_shape.rank() == 1
    raises(ValueError, lambda: vector_with_long_shape[int(5)])

    from sympy.abc import x
    for ArrayType in [MutableDenseNDimArray, MutableSparseNDimArray]:
        rank_zero_array = ArrayType(x)
        assert len(rank_zero_array) == 1
        assert rank_zero_array.shape == ()
        assert rank_zero_array.rank() == 0
        assert rank_zero_array[()] == x
        raises(ValueError, lambda: rank_zero_array[0])

def test_sympify():
    from sympy.abc import x, y, z, t
    arr = MutableDenseNDimArray([[x, y], [1, z*t]])
    arr_other = sympify(arr)
    assert arr_other.shape == (2, 2)
    assert arr_other == arr


def test_reshape():
    array = MutableDenseNDimArray(range(50), 50)
    assert array.shape == (50,)
    assert array.rank() == 1

    array = array.reshape(5, 5, 2)
    assert array.shape == (5, 5, 2)
    assert array.rank() == 3
    assert len(array) == 50


def test_iterator():
    array = MutableDenseNDimArray(range(4), (2, 2))
    assert array[0] == MutableDenseNDimArray([0, 1])
    assert array[1] == MutableDenseNDimArray([2, 3])

    array = array.reshape(4)
    j = 0
    for i in array:
        assert i == j
        j += 1


def test_getitem():
    for ArrayType in [MutableDenseNDimArray, MutableSparseNDimArray]:
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


def test_sparse():
    sparse_array = MutableSparseNDimArray([0, 0, 0, 1], (2, 2))
    assert len(sparse_array) == 2 * 2
    # dictionary where all data is, only non-zero entries are actually stored:
    assert len(sparse_array._sparse_array) == 1

    assert sparse_array.tolist() == [[0, 0], [0, 1]]

    for i, j in zip(sparse_array, [[0, 0], [0, 1]]):
        assert i == MutableSparseNDimArray(j)

    sparse_array[0, 0] = 123
    assert len(sparse_array._sparse_array) == 2
    assert sparse_array[0, 0] == 123
    assert sparse_array/0 == MutableSparseNDimArray([[S.ComplexInfinity, S.NaN], [S.NaN, S.ComplexInfinity]], (2, 2))

    # when element in sparse array become zero it will disappear from
    # dictionary
    sparse_array[0, 0] = 0
    assert len(sparse_array._sparse_array) == 1
    sparse_array[1, 1] = 0
    assert len(sparse_array._sparse_array) == 0
    assert sparse_array[0, 0] == 0

    # test for large scale sparse array
    # equality test
    a = MutableSparseNDimArray.zeros(100000, 200000)
    b = MutableSparseNDimArray.zeros(100000, 200000)
    assert a == b
    a[1, 1] = 1
    b[1, 1] = 2
    assert a != b

    # __mul__ and __rmul__
    assert a * 3 == MutableSparseNDimArray({200001: 3}, (100000, 200000))
    assert 3 * a == MutableSparseNDimArray({200001: 3}, (100000, 200000))
    assert a * 0 == MutableSparseNDimArray({}, (100000, 200000))
    assert 0 * a == MutableSparseNDimArray({}, (100000, 200000))

    # __truediv__
    assert a/3 == MutableSparseNDimArray({200001: Rational(1, 3)}, (100000, 200000))

    # __neg__
    assert -a == MutableSparseNDimArray({200001: -1}, (100000, 200000))


def test_calculation():

    a = MutableDenseNDimArray([1]*9, (3, 3))
    b = MutableDenseNDimArray([9]*9, (3, 3))

    c = a + b
    for i in c:
        assert i == MutableDenseNDimArray([10, 10, 10])

    assert c == MutableDenseNDimArray([10]*9, (3, 3))
    assert c == MutableSparseNDimArray([10]*9, (3, 3))

    c = b - a
    for i in c:
        assert i == MutableSparseNDimArray([8, 8, 8])

    assert c == MutableDenseNDimArray([8]*9, (3, 3))
    assert c == MutableSparseNDimArray([8]*9, (3, 3))


def test_ndim_array_converting():
    dense_array = MutableDenseNDimArray([1, 2, 3, 4], (2, 2))
    alist = dense_array.tolist()

    assert alist == [[1, 2], [3, 4]]

    matrix = dense_array.tomatrix()
    assert (isinstance(matrix, Matrix))

    for i in range(len(dense_array)):
        assert dense_array[dense_array._get_tuple_index(i)] == matrix[i]
    assert matrix.shape == dense_array.shape

    assert MutableDenseNDimArray(matrix) == dense_array
    assert MutableDenseNDimArray(matrix.as_immutable()) == dense_array
    assert MutableDenseNDimArray(matrix.as_mutable()) == dense_array

    sparse_array = MutableSparseNDimArray([1, 2, 3, 4], (2, 2))
    alist = sparse_array.tolist()

    assert alist == [[1, 2], [3, 4]]

    matrix = sparse_array.tomatrix()
    assert(isinstance(matrix, SparseMatrix))

    for i in range(len(sparse_array)):
        assert sparse_array[sparse_array._get_tuple_index(i)] == matrix[i]
    assert matrix.shape == sparse_array.shape

    assert MutableSparseNDimArray(matrix) == sparse_array
    assert MutableSparseNDimArray(matrix.as_immutable()) == sparse_array
    assert MutableSparseNDimArray(matrix.as_mutable()) == sparse_array


def test_converting_functions():
    arr_list = [1, 2, 3, 4]
    arr_matrix = Matrix(((1, 2), (3, 4)))

    # list
    arr_ndim_array = MutableDenseNDimArray(arr_list, (2, 2))
    assert (isinstance(arr_ndim_array, MutableDenseNDimArray))
    assert arr_matrix.tolist() == arr_ndim_array.tolist()

    # Matrix
    arr_ndim_array = MutableDenseNDimArray(arr_matrix)
    assert (isinstance(arr_ndim_array, MutableDenseNDimArray))
    assert arr_matrix.tolist() == arr_ndim_array.tolist()
    assert arr_matrix.shape == arr_ndim_array.shape


def test_equality():
    first_list = [1, 2, 3, 4]
    second_list = [1, 2, 3, 4]
    third_list = [4, 3, 2, 1]
    assert first_list == second_list
    assert first_list != third_list

    first_ndim_array = MutableDenseNDimArray(first_list, (2, 2))
    second_ndim_array = MutableDenseNDimArray(second_list, (2, 2))
    third_ndim_array = MutableDenseNDimArray(third_list, (2, 2))
    fourth_ndim_array = MutableDenseNDimArray(first_list, (2, 2))

    assert first_ndim_array == second_ndim_array
    second_ndim_array[0, 0] = 0
    assert first_ndim_array != second_ndim_array
    assert first_ndim_array != third_ndim_array
    assert first_ndim_array == fourth_ndim_array


def test_arithmetic():
    a = MutableDenseNDimArray([3 for i in range(9)], (3, 3))
    b = MutableDenseNDimArray([7 for i in range(9)], (3, 3))

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
    assert z0 == MutableDenseNDimArray([-3 for i in range(9)], (3, 3))


def test_higher_dimenions():
    m3 = MutableDenseNDimArray(range(10, 34), (2, 3, 4))

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

    m3_rebuilt = MutableDenseNDimArray([[[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]], [[22, 23, 24, 25], [26, 27, 28, 29], [30, 31, 32, 33]]])
    assert m3 == m3_rebuilt

    m3_other = MutableDenseNDimArray([[[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]], [[22, 23, 24, 25], [26, 27, 28, 29], [30, 31, 32, 33]]], (2, 3, 4))

    assert m3 == m3_other


def test_slices():
    md = MutableDenseNDimArray(range(10, 34), (2, 3, 4))

    assert md[:] == MutableDenseNDimArray(range(10, 34), (2, 3, 4))
    assert md[:, :, 0].tomatrix() == Matrix([[10, 14, 18], [22, 26, 30]])
    assert md[0, 1:2, :].tomatrix() == Matrix([[14, 15, 16, 17]])
    assert md[0, 1:3, :].tomatrix() == Matrix([[14, 15, 16, 17], [18, 19, 20, 21]])
    assert md[:, :, :] == md

    sd = MutableSparseNDimArray(range(10, 34), (2, 3, 4))
    assert sd == MutableSparseNDimArray(md)

    assert sd[:] == MutableSparseNDimArray(range(10, 34), (2, 3, 4))
    assert sd[:, :, 0].tomatrix() == Matrix([[10, 14, 18], [22, 26, 30]])
    assert sd[0, 1:2, :].tomatrix() == Matrix([[14, 15, 16, 17]])
    assert sd[0, 1:3, :].tomatrix() == Matrix([[14, 15, 16, 17], [18, 19, 20, 21]])
    assert sd[:, :, :] == sd


def test_slices_assign():
    a = MutableDenseNDimArray(range(12), shape=(4, 3))
    b = MutableSparseNDimArray(range(12), shape=(4, 3))

    for i in [a, b]:
        assert i.tolist() == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        i[0, :] = [2, 2, 2]
        assert i.tolist() == [[2, 2, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        i[0, 1:] = [8, 8]
        assert i.tolist() == [[2, 8, 8], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        i[1:3, 1] = [20, 44]
        assert i.tolist() == [[2, 8, 8], [3, 20, 5], [6, 44, 8], [9, 10, 11]]


def test_diff():
    from sympy.abc import x, y, z
    md = MutableDenseNDimArray([[x, y], [x*z, x*y*z]])
    assert md.diff(x) == MutableDenseNDimArray([[1, 0], [z, y*z]])
    assert diff(md, x) == MutableDenseNDimArray([[1, 0], [z, y*z]])

    sd = MutableSparseNDimArray(md)
    assert sd == MutableSparseNDimArray([x, y, x*z, x*y*z], (2, 2))
    assert sd.diff(x) == MutableSparseNDimArray([[1, 0], [z, y*z]])
    assert diff(sd, x) == MutableSparseNDimArray([[1, 0], [z, y*z]])
