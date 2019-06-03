import pytest
from pybind11_tests import numpy_vectorize as m

pytestmark = pytest.requires_numpy

with pytest.suppress(ImportError):
    import numpy as np


def test_vectorize(capture):
    assert np.isclose(m.vectorized_func3(np.array(3 + 7j)), [6 + 14j])

    for f in [m.vectorized_func, m.vectorized_func2]:
        with capture:
            assert np.isclose(f(1, 2, 3), 6)
        assert capture == "my_func(x:int=1, y:float=2, z:float=3)"
        with capture:
            assert np.isclose(f(np.array(1), np.array(2), 3), 6)
        assert capture == "my_func(x:int=1, y:float=2, z:float=3)"
        with capture:
            assert np.allclose(f(np.array([1, 3]), np.array([2, 4]), 3), [6, 36])
        assert capture == """
            my_func(x:int=1, y:float=2, z:float=3)
            my_func(x:int=3, y:float=4, z:float=3)
        """
        with capture:
            a = np.array([[1, 2], [3, 4]], order='F')
            b = np.array([[10, 20], [30, 40]], order='F')
            c = 3
            result = f(a, b, c)
            assert np.allclose(result, a * b * c)
            assert result.flags.f_contiguous
        # All inputs are F order and full or singletons, so we the result is in col-major order:
        assert capture == """
            my_func(x:int=1, y:float=10, z:float=3)
            my_func(x:int=3, y:float=30, z:float=3)
            my_func(x:int=2, y:float=20, z:float=3)
            my_func(x:int=4, y:float=40, z:float=3)
        """
        with capture:
            a, b, c = np.array([[1, 3, 5], [7, 9, 11]]), np.array([[2, 4, 6], [8, 10, 12]]), 3
            assert np.allclose(f(a, b, c), a * b * c)
        assert capture == """
            my_func(x:int=1, y:float=2, z:float=3)
            my_func(x:int=3, y:float=4, z:float=3)
            my_func(x:int=5, y:float=6, z:float=3)
            my_func(x:int=7, y:float=8, z:float=3)
            my_func(x:int=9, y:float=10, z:float=3)
            my_func(x:int=11, y:float=12, z:float=3)
        """
        with capture:
            a, b, c = np.array([[1, 2, 3], [4, 5, 6]]), np.array([2, 3, 4]), 2
            assert np.allclose(f(a, b, c), a * b * c)
        assert capture == """
            my_func(x:int=1, y:float=2, z:float=2)
            my_func(x:int=2, y:float=3, z:float=2)
            my_func(x:int=3, y:float=4, z:float=2)
            my_func(x:int=4, y:float=2, z:float=2)
            my_func(x:int=5, y:float=3, z:float=2)
            my_func(x:int=6, y:float=4, z:float=2)
        """
        with capture:
            a, b, c = np.array([[1, 2, 3], [4, 5, 6]]), np.array([[2], [3]]), 2
            assert np.allclose(f(a, b, c), a * b * c)
        assert capture == """
            my_func(x:int=1, y:float=2, z:float=2)
            my_func(x:int=2, y:float=2, z:float=2)
            my_func(x:int=3, y:float=2, z:float=2)
            my_func(x:int=4, y:float=3, z:float=2)
            my_func(x:int=5, y:float=3, z:float=2)
            my_func(x:int=6, y:float=3, z:float=2)
        """
        with capture:
            a, b, c = np.array([[1, 2, 3], [4, 5, 6]], order='F'), np.array([[2], [3]]), 2
            assert np.allclose(f(a, b, c), a * b * c)
        assert capture == """
            my_func(x:int=1, y:float=2, z:float=2)
            my_func(x:int=2, y:float=2, z:float=2)
            my_func(x:int=3, y:float=2, z:float=2)
            my_func(x:int=4, y:float=3, z:float=2)
            my_func(x:int=5, y:float=3, z:float=2)
            my_func(x:int=6, y:float=3, z:float=2)
        """
        with capture:
            a, b, c = np.array([[1, 2, 3], [4, 5, 6]])[::, ::2], np.array([[2], [3]]), 2
            assert np.allclose(f(a, b, c), a * b * c)
        assert capture == """
            my_func(x:int=1, y:float=2, z:float=2)
            my_func(x:int=3, y:float=2, z:float=2)
            my_func(x:int=4, y:float=3, z:float=2)
            my_func(x:int=6, y:float=3, z:float=2)
        """
        with capture:
            a, b, c = np.array([[1, 2, 3], [4, 5, 6]], order='F')[::, ::2], np.array([[2], [3]]), 2
            assert np.allclose(f(a, b, c), a * b * c)
        assert capture == """
            my_func(x:int=1, y:float=2, z:float=2)
            my_func(x:int=3, y:float=2, z:float=2)
            my_func(x:int=4, y:float=3, z:float=2)
            my_func(x:int=6, y:float=3, z:float=2)
        """


def test_type_selection():
    assert m.selective_func(np.array([1], dtype=np.int32)) == "Int branch taken."
    assert m.selective_func(np.array([1.0], dtype=np.float32)) == "Float branch taken."
    assert m.selective_func(np.array([1.0j], dtype=np.complex64)) == "Complex float branch taken."


def test_docs(doc):
    assert doc(m.vectorized_func) == """
        vectorized_func(arg0: numpy.ndarray[int32], arg1: numpy.ndarray[float32], arg2: numpy.ndarray[float64]) -> object
    """  # noqa: E501 line too long


def test_trivial_broadcasting():
    trivial, vectorized_is_trivial = m.trivial, m.vectorized_is_trivial

    assert vectorized_is_trivial(1, 2, 3) == trivial.c_trivial
    assert vectorized_is_trivial(np.array(1), np.array(2), 3) == trivial.c_trivial
    assert vectorized_is_trivial(np.array([1, 3]), np.array([2, 4]), 3) == trivial.c_trivial
    assert trivial.c_trivial == vectorized_is_trivial(
        np.array([[1, 3, 5], [7, 9, 11]]), np.array([[2, 4, 6], [8, 10, 12]]), 3)
    assert vectorized_is_trivial(
        np.array([[1, 2, 3], [4, 5, 6]]), np.array([2, 3, 4]), 2) == trivial.non_trivial
    assert vectorized_is_trivial(
        np.array([[1, 2, 3], [4, 5, 6]]), np.array([[2], [3]]), 2) == trivial.non_trivial
    z1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype='int32')
    z2 = np.array(z1, dtype='float32')
    z3 = np.array(z1, dtype='float64')
    assert vectorized_is_trivial(z1, z2, z3) == trivial.c_trivial
    assert vectorized_is_trivial(1, z2, z3) == trivial.c_trivial
    assert vectorized_is_trivial(z1, 1, z3) == trivial.c_trivial
    assert vectorized_is_trivial(z1, z2, 1) == trivial.c_trivial
    assert vectorized_is_trivial(z1[::2, ::2], 1, 1) == trivial.non_trivial
    assert vectorized_is_trivial(1, 1, z1[::2, ::2]) == trivial.c_trivial
    assert vectorized_is_trivial(1, 1, z3[::2, ::2]) == trivial.non_trivial
    assert vectorized_is_trivial(z1, 1, z3[1::4, 1::4]) == trivial.c_trivial

    y1 = np.array(z1, order='F')
    y2 = np.array(y1)
    y3 = np.array(y1)
    assert vectorized_is_trivial(y1, y2, y3) == trivial.f_trivial
    assert vectorized_is_trivial(y1, 1, 1) == trivial.f_trivial
    assert vectorized_is_trivial(1, y2, 1) == trivial.f_trivial
    assert vectorized_is_trivial(1, 1, y3) == trivial.f_trivial
    assert vectorized_is_trivial(y1, z2, 1) == trivial.non_trivial
    assert vectorized_is_trivial(z1[1::4, 1::4], y2, 1) == trivial.f_trivial
    assert vectorized_is_trivial(y1[1::4, 1::4], z2, 1) == trivial.c_trivial

    assert m.vectorized_func(z1, z2, z3).flags.c_contiguous
    assert m.vectorized_func(y1, y2, y3).flags.f_contiguous
    assert m.vectorized_func(z1, 1, 1).flags.c_contiguous
    assert m.vectorized_func(1, y2, 1).flags.f_contiguous
    assert m.vectorized_func(z1[1::4, 1::4], y2, 1).flags.f_contiguous
    assert m.vectorized_func(y1[1::4, 1::4], z2, 1).flags.c_contiguous


def test_passthrough_arguments(doc):
    assert doc(m.vec_passthrough) == (
        "vec_passthrough(" + ", ".join([
            "arg0: float",
            "arg1: numpy.ndarray[float64]",
            "arg2: numpy.ndarray[float64]",
            "arg3: numpy.ndarray[int32]",
            "arg4: int",
            "arg5: m.numpy_vectorize.NonPODClass",
            "arg6: numpy.ndarray[float64]"]) + ") -> object")

    b = np.array([[10, 20, 30]], dtype='float64')
    c = np.array([100, 200])  # NOT a vectorized argument
    d = np.array([[1000], [2000], [3000]], dtype='int')
    g = np.array([[1000000, 2000000, 3000000]], dtype='int')  # requires casting
    assert np.all(
        m.vec_passthrough(1, b, c, d, 10000, m.NonPODClass(100000), g) ==
        np.array([[1111111, 2111121, 3111131],
                  [1112111, 2112121, 3112131],
                  [1113111, 2113121, 3113131]]))


def test_method_vectorization():
    o = m.VectorizeTestClass(3)
    x = np.array([1, 2], dtype='int')
    y = np.array([[10], [20]], dtype='float32')
    assert np.all(o.method(x, y) == [[14, 15], [24, 25]])


def test_array_collapse():
    assert not isinstance(m.vectorized_func(1, 2, 3), np.ndarray)
    assert not isinstance(m.vectorized_func(np.array(1), 2, 3), np.ndarray)
    z = m.vectorized_func([1], 2, 3)
    assert isinstance(z, np.ndarray)
    assert z.shape == (1, )
    z = m.vectorized_func(1, [[[2]]], 3)
    assert isinstance(z, np.ndarray)
    assert z.shape == (1, 1, 1)
