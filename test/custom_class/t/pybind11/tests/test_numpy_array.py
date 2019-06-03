import pytest
from pybind11_tests import numpy_array as m

pytestmark = pytest.requires_numpy

with pytest.suppress(ImportError):
    import numpy as np


@pytest.fixture(scope='function')
def arr():
    return np.array([[1, 2, 3], [4, 5, 6]], '=u2')


def test_array_attributes():
    a = np.array(0, 'f8')
    assert m.ndim(a) == 0
    assert all(m.shape(a) == [])
    assert all(m.strides(a) == [])
    with pytest.raises(IndexError) as excinfo:
        m.shape(a, 0)
    assert str(excinfo.value) == 'invalid axis: 0 (ndim = 0)'
    with pytest.raises(IndexError) as excinfo:
        m.strides(a, 0)
    assert str(excinfo.value) == 'invalid axis: 0 (ndim = 0)'
    assert m.writeable(a)
    assert m.size(a) == 1
    assert m.itemsize(a) == 8
    assert m.nbytes(a) == 8
    assert m.owndata(a)

    a = np.array([[1, 2, 3], [4, 5, 6]], 'u2').view()
    a.flags.writeable = False
    assert m.ndim(a) == 2
    assert all(m.shape(a) == [2, 3])
    assert m.shape(a, 0) == 2
    assert m.shape(a, 1) == 3
    assert all(m.strides(a) == [6, 2])
    assert m.strides(a, 0) == 6
    assert m.strides(a, 1) == 2
    with pytest.raises(IndexError) as excinfo:
        m.shape(a, 2)
    assert str(excinfo.value) == 'invalid axis: 2 (ndim = 2)'
    with pytest.raises(IndexError) as excinfo:
        m.strides(a, 2)
    assert str(excinfo.value) == 'invalid axis: 2 (ndim = 2)'
    assert not m.writeable(a)
    assert m.size(a) == 6
    assert m.itemsize(a) == 2
    assert m.nbytes(a) == 12
    assert not m.owndata(a)


@pytest.mark.parametrize('args, ret', [([], 0), ([0], 0), ([1], 3), ([0, 1], 1), ([1, 2], 5)])
def test_index_offset(arr, args, ret):
    assert m.index_at(arr, *args) == ret
    assert m.index_at_t(arr, *args) == ret
    assert m.offset_at(arr, *args) == ret * arr.dtype.itemsize
    assert m.offset_at_t(arr, *args) == ret * arr.dtype.itemsize


def test_dim_check_fail(arr):
    for func in (m.index_at, m.index_at_t, m.offset_at, m.offset_at_t, m.data, m.data_t,
                 m.mutate_data, m.mutate_data_t):
        with pytest.raises(IndexError) as excinfo:
            func(arr, 1, 2, 3)
        assert str(excinfo.value) == 'too many indices for an array: 3 (ndim = 2)'


@pytest.mark.parametrize('args, ret',
                         [([], [1, 2, 3, 4, 5, 6]),
                          ([1], [4, 5, 6]),
                          ([0, 1], [2, 3, 4, 5, 6]),
                          ([1, 2], [6])])
def test_data(arr, args, ret):
    from sys import byteorder
    assert all(m.data_t(arr, *args) == ret)
    assert all(m.data(arr, *args)[(0 if byteorder == 'little' else 1)::2] == ret)
    assert all(m.data(arr, *args)[(1 if byteorder == 'little' else 0)::2] == 0)


@pytest.mark.parametrize('dim', [0, 1, 3])
def test_at_fail(arr, dim):
    for func in m.at_t, m.mutate_at_t:
        with pytest.raises(IndexError) as excinfo:
            func(arr, *([0] * dim))
        assert str(excinfo.value) == 'index dimension mismatch: {} (ndim = 2)'.format(dim)


def test_at(arr):
    assert m.at_t(arr, 0, 2) == 3
    assert m.at_t(arr, 1, 0) == 4

    assert all(m.mutate_at_t(arr, 0, 2).ravel() == [1, 2, 4, 4, 5, 6])
    assert all(m.mutate_at_t(arr, 1, 0).ravel() == [1, 2, 4, 5, 5, 6])


def test_mutate_readonly(arr):
    arr.flags.writeable = False
    for func, args in (m.mutate_data, ()), (m.mutate_data_t, ()), (m.mutate_at_t, (0, 0)):
        with pytest.raises(ValueError) as excinfo:
            func(arr, *args)
        assert str(excinfo.value) == 'array is not writeable'


def test_mutate_data(arr):
    assert all(m.mutate_data(arr).ravel() == [2, 4, 6, 8, 10, 12])
    assert all(m.mutate_data(arr).ravel() == [4, 8, 12, 16, 20, 24])
    assert all(m.mutate_data(arr, 1).ravel() == [4, 8, 12, 32, 40, 48])
    assert all(m.mutate_data(arr, 0, 1).ravel() == [4, 16, 24, 64, 80, 96])
    assert all(m.mutate_data(arr, 1, 2).ravel() == [4, 16, 24, 64, 80, 192])

    assert all(m.mutate_data_t(arr).ravel() == [5, 17, 25, 65, 81, 193])
    assert all(m.mutate_data_t(arr).ravel() == [6, 18, 26, 66, 82, 194])
    assert all(m.mutate_data_t(arr, 1).ravel() == [6, 18, 26, 67, 83, 195])
    assert all(m.mutate_data_t(arr, 0, 1).ravel() == [6, 19, 27, 68, 84, 196])
    assert all(m.mutate_data_t(arr, 1, 2).ravel() == [6, 19, 27, 68, 84, 197])


def test_bounds_check(arr):
    for func in (m.index_at, m.index_at_t, m.data, m.data_t,
                 m.mutate_data, m.mutate_data_t, m.at_t, m.mutate_at_t):
        with pytest.raises(IndexError) as excinfo:
            func(arr, 2, 0)
        assert str(excinfo.value) == 'index 2 is out of bounds for axis 0 with size 2'
        with pytest.raises(IndexError) as excinfo:
            func(arr, 0, 4)
        assert str(excinfo.value) == 'index 4 is out of bounds for axis 1 with size 3'


def test_make_c_f_array():
    assert m.make_c_array().flags.c_contiguous
    assert not m.make_c_array().flags.f_contiguous
    assert m.make_f_array().flags.f_contiguous
    assert not m.make_f_array().flags.c_contiguous


def test_make_empty_shaped_array():
    m.make_empty_shaped_array()


def test_wrap():
    def assert_references(a, b, base=None):
        from distutils.version import LooseVersion
        if base is None:
            base = a
        assert a is not b
        assert a.__array_interface__['data'][0] == b.__array_interface__['data'][0]
        assert a.shape == b.shape
        assert a.strides == b.strides
        assert a.flags.c_contiguous == b.flags.c_contiguous
        assert a.flags.f_contiguous == b.flags.f_contiguous
        assert a.flags.writeable == b.flags.writeable
        assert a.flags.aligned == b.flags.aligned
        if LooseVersion(np.__version__) >= LooseVersion("1.14.0"):
            assert a.flags.writebackifcopy == b.flags.writebackifcopy
        else:
            assert a.flags.updateifcopy == b.flags.updateifcopy
        assert np.all(a == b)
        assert not b.flags.owndata
        assert b.base is base
        if a.flags.writeable and a.ndim == 2:
            a[0, 0] = 1234
            assert b[0, 0] == 1234

    a1 = np.array([1, 2], dtype=np.int16)
    assert a1.flags.owndata and a1.base is None
    a2 = m.wrap(a1)
    assert_references(a1, a2)

    a1 = np.array([[1, 2], [3, 4]], dtype=np.float32, order='F')
    assert a1.flags.owndata and a1.base is None
    a2 = m.wrap(a1)
    assert_references(a1, a2)

    a1 = np.array([[1, 2], [3, 4]], dtype=np.float32, order='C')
    a1.flags.writeable = False
    a2 = m.wrap(a1)
    assert_references(a1, a2)

    a1 = np.random.random((4, 4, 4))
    a2 = m.wrap(a1)
    assert_references(a1, a2)

    a1t = a1.transpose()
    a2 = m.wrap(a1t)
    assert_references(a1t, a2, a1)

    a1d = a1.diagonal()
    a2 = m.wrap(a1d)
    assert_references(a1d, a2, a1)

    a1m = a1[::-1, ::-1, ::-1]
    a2 = m.wrap(a1m)
    assert_references(a1m, a2, a1)


def test_numpy_view(capture):
    with capture:
        ac = m.ArrayClass()
        ac_view_1 = ac.numpy_view()
        ac_view_2 = ac.numpy_view()
        assert np.all(ac_view_1 == np.array([1, 2], dtype=np.int32))
        del ac
        pytest.gc_collect()
    assert capture == """
        ArrayClass()
        ArrayClass::numpy_view()
        ArrayClass::numpy_view()
    """
    ac_view_1[0] = 4
    ac_view_1[1] = 3
    assert ac_view_2[0] == 4
    assert ac_view_2[1] == 3
    with capture:
        del ac_view_1
        del ac_view_2
        pytest.gc_collect()
        pytest.gc_collect()
    assert capture == """
        ~ArrayClass()
    """


@pytest.unsupported_on_pypy
def test_cast_numpy_int64_to_uint64():
    m.function_taking_uint64(123)
    m.function_taking_uint64(np.uint64(123))


def test_isinstance():
    assert m.isinstance_untyped(np.array([1, 2, 3]), "not an array")
    assert m.isinstance_typed(np.array([1.0, 2.0, 3.0]))


def test_constructors():
    defaults = m.default_constructors()
    for a in defaults.values():
        assert a.size == 0
    assert defaults["array"].dtype == np.array([]).dtype
    assert defaults["array_t<int32>"].dtype == np.int32
    assert defaults["array_t<double>"].dtype == np.float64

    results = m.converting_constructors([1, 2, 3])
    for a in results.values():
        np.testing.assert_array_equal(a, [1, 2, 3])
    assert results["array"].dtype == np.int_
    assert results["array_t<int32>"].dtype == np.int32
    assert results["array_t<double>"].dtype == np.float64


def test_overload_resolution(msg):
    # Exact overload matches:
    assert m.overloaded(np.array([1], dtype='float64')) == 'double'
    assert m.overloaded(np.array([1], dtype='float32')) == 'float'
    assert m.overloaded(np.array([1], dtype='ushort')) == 'unsigned short'
    assert m.overloaded(np.array([1], dtype='intc')) == 'int'
    assert m.overloaded(np.array([1], dtype='longlong')) == 'long long'
    assert m.overloaded(np.array([1], dtype='complex')) == 'double complex'
    assert m.overloaded(np.array([1], dtype='csingle')) == 'float complex'

    # No exact match, should call first convertible version:
    assert m.overloaded(np.array([1], dtype='uint8')) == 'double'

    with pytest.raises(TypeError) as excinfo:
        m.overloaded("not an array")
    assert msg(excinfo.value) == """
        overloaded(): incompatible function arguments. The following argument types are supported:
            1. (arg0: numpy.ndarray[float64]) -> str
            2. (arg0: numpy.ndarray[float32]) -> str
            3. (arg0: numpy.ndarray[int32]) -> str
            4. (arg0: numpy.ndarray[uint16]) -> str
            5. (arg0: numpy.ndarray[int64]) -> str
            6. (arg0: numpy.ndarray[complex128]) -> str
            7. (arg0: numpy.ndarray[complex64]) -> str

        Invoked with: 'not an array'
    """

    assert m.overloaded2(np.array([1], dtype='float64')) == 'double'
    assert m.overloaded2(np.array([1], dtype='float32')) == 'float'
    assert m.overloaded2(np.array([1], dtype='complex64')) == 'float complex'
    assert m.overloaded2(np.array([1], dtype='complex128')) == 'double complex'
    assert m.overloaded2(np.array([1], dtype='float32')) == 'float'

    assert m.overloaded3(np.array([1], dtype='float64')) == 'double'
    assert m.overloaded3(np.array([1], dtype='intc')) == 'int'
    expected_exc = """
        overloaded3(): incompatible function arguments. The following argument types are supported:
            1. (arg0: numpy.ndarray[int32]) -> str
            2. (arg0: numpy.ndarray[float64]) -> str

        Invoked with: """

    with pytest.raises(TypeError) as excinfo:
        m.overloaded3(np.array([1], dtype='uintc'))
    assert msg(excinfo.value) == expected_exc + repr(np.array([1], dtype='uint32'))
    with pytest.raises(TypeError) as excinfo:
        m.overloaded3(np.array([1], dtype='float32'))
    assert msg(excinfo.value) == expected_exc + repr(np.array([1.], dtype='float32'))
    with pytest.raises(TypeError) as excinfo:
        m.overloaded3(np.array([1], dtype='complex'))
    assert msg(excinfo.value) == expected_exc + repr(np.array([1. + 0.j]))

    # Exact matches:
    assert m.overloaded4(np.array([1], dtype='double')) == 'double'
    assert m.overloaded4(np.array([1], dtype='longlong')) == 'long long'
    # Non-exact matches requiring conversion.  Since float to integer isn't a
    # save conversion, it should go to the double overload, but short can go to
    # either (and so should end up on the first-registered, the long long).
    assert m.overloaded4(np.array([1], dtype='float32')) == 'double'
    assert m.overloaded4(np.array([1], dtype='short')) == 'long long'

    assert m.overloaded5(np.array([1], dtype='double')) == 'double'
    assert m.overloaded5(np.array([1], dtype='uintc')) == 'unsigned int'
    assert m.overloaded5(np.array([1], dtype='float32')) == 'unsigned int'


def test_greedy_string_overload():
    """Tests fix for #685 - ndarray shouldn't go to std::string overload"""

    assert m.issue685("abc") == "string"
    assert m.issue685(np.array([97, 98, 99], dtype='b')) == "array"
    assert m.issue685(123) == "other"


def test_array_unchecked_fixed_dims(msg):
    z1 = np.array([[1, 2], [3, 4]], dtype='float64')
    m.proxy_add2(z1, 10)
    assert np.all(z1 == [[11, 12], [13, 14]])

    with pytest.raises(ValueError) as excinfo:
        m.proxy_add2(np.array([1., 2, 3]), 5.0)
    assert msg(excinfo.value) == "array has incorrect number of dimensions: 1; expected 2"

    expect_c = np.ndarray(shape=(3, 3, 3), buffer=np.array(range(3, 30)), dtype='int')
    assert np.all(m.proxy_init3(3.0) == expect_c)
    expect_f = np.transpose(expect_c)
    assert np.all(m.proxy_init3F(3.0) == expect_f)

    assert m.proxy_squared_L2_norm(np.array(range(6))) == 55
    assert m.proxy_squared_L2_norm(np.array(range(6), dtype="float64")) == 55

    assert m.proxy_auxiliaries2(z1) == [11, 11, True, 2, 8, 2, 2, 4, 32]
    assert m.proxy_auxiliaries2(z1) == m.array_auxiliaries2(z1)


def test_array_unchecked_dyn_dims(msg):
    z1 = np.array([[1, 2], [3, 4]], dtype='float64')
    m.proxy_add2_dyn(z1, 10)
    assert np.all(z1 == [[11, 12], [13, 14]])

    expect_c = np.ndarray(shape=(3, 3, 3), buffer=np.array(range(3, 30)), dtype='int')
    assert np.all(m.proxy_init3_dyn(3.0) == expect_c)

    assert m.proxy_auxiliaries2_dyn(z1) == [11, 11, True, 2, 8, 2, 2, 4, 32]
    assert m.proxy_auxiliaries2_dyn(z1) == m.array_auxiliaries2(z1)


def test_array_failure():
    with pytest.raises(ValueError) as excinfo:
        m.array_fail_test()
    assert str(excinfo.value) == 'cannot create a pybind11::array from a nullptr'

    with pytest.raises(ValueError) as excinfo:
        m.array_t_fail_test()
    assert str(excinfo.value) == 'cannot create a pybind11::array_t from a nullptr'

    with pytest.raises(ValueError) as excinfo:
        m.array_fail_test_negative_size()
    assert str(excinfo.value) == 'negative dimensions are not allowed'


def test_initializer_list():
    assert m.array_initializer_list1().shape == (1,)
    assert m.array_initializer_list2().shape == (1, 2)
    assert m.array_initializer_list3().shape == (1, 2, 3)
    assert m.array_initializer_list4().shape == (1, 2, 3, 4)


def test_array_resize(msg):
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='float64')
    m.array_reshape2(a)
    assert(a.size == 9)
    assert(np.all(a == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    # total size change should succced with refcheck off
    m.array_resize3(a, 4, False)
    assert(a.size == 64)
    # ... and fail with refcheck on
    try:
        m.array_resize3(a, 3, True)
    except ValueError as e:
        assert(str(e).startswith("cannot resize an array"))
    # transposed array doesn't own data
    b = a.transpose()
    try:
        m.array_resize3(b, 3, False)
    except ValueError as e:
        assert(str(e).startswith("cannot resize this array: it does not own its data"))
    # ... but reshape should be fine
    m.array_reshape2(b)
    assert(b.shape == (8, 8))


@pytest.unsupported_on_pypy
def test_array_create_and_resize(msg):
    a = m.create_and_resize(2)
    assert(a.size == 4)
    assert(np.all(a == 42.))


@pytest.unsupported_on_py2
def test_index_using_ellipsis():
    a = m.index_using_ellipsis(np.zeros((5, 6, 7)))
    assert a.shape == (6,)
