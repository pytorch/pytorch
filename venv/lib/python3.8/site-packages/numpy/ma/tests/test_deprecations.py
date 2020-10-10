"""Test deprecation and future warnings.

"""
import numpy as np
from numpy.testing import assert_warns
from numpy.ma.testutils import assert_equal
from numpy.ma.core import MaskedArrayFutureWarning

class TestArgsort:
    """ gh-8701 """
    def _test_base(self, argsort, cls):
        arr_0d = np.array(1).view(cls)
        argsort(arr_0d)

        arr_1d = np.array([1, 2, 3]).view(cls)
        argsort(arr_1d)

        # argsort has a bad default for >1d arrays
        arr_2d = np.array([[1, 2], [3, 4]]).view(cls)
        result = assert_warns(
            np.ma.core.MaskedArrayFutureWarning, argsort, arr_2d)
        assert_equal(result, argsort(arr_2d, axis=None))

        # should be no warnings for explicitly specifying it
        argsort(arr_2d, axis=None)
        argsort(arr_2d, axis=-1)

    def test_function_ndarray(self):
        return self._test_base(np.ma.argsort, np.ndarray)

    def test_function_maskedarray(self):
        return self._test_base(np.ma.argsort, np.ma.MaskedArray)

    def test_method(self):
        return self._test_base(np.ma.MaskedArray.argsort, np.ma.MaskedArray)


class TestMinimumMaximum:
    def test_minimum(self):
        assert_warns(DeprecationWarning, np.ma.minimum, np.ma.array([1, 2]))

    def test_maximum(self):
        assert_warns(DeprecationWarning, np.ma.maximum, np.ma.array([1, 2]))

    def test_axis_default(self):
        # NumPy 1.13, 2017-05-06

        data1d = np.ma.arange(6)
        data2d = data1d.reshape(2, 3)

        ma_min = np.ma.minimum.reduce
        ma_max = np.ma.maximum.reduce

        # check that the default axis is still None, but warns on 2d arrays
        result = assert_warns(MaskedArrayFutureWarning, ma_max, data2d)
        assert_equal(result, ma_max(data2d, axis=None))

        result = assert_warns(MaskedArrayFutureWarning, ma_min, data2d)
        assert_equal(result, ma_min(data2d, axis=None))

        # no warnings on 1d, as both new and old defaults are equivalent
        result = ma_min(data1d)
        assert_equal(result, ma_min(data1d, axis=None))
        assert_equal(result, ma_min(data1d, axis=0))

        result = ma_max(data1d)
        assert_equal(result, ma_max(data1d, axis=None))
        assert_equal(result, ma_max(data1d, axis=0))
