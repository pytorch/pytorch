# Owner(s): ["module: dynamo"]

from unittest import expectedFailure as xfail

import pytest

import torch._numpy as np
from pytest import raises as assert_raises
from torch._numpy.testing import assert_array_equal, assert_equal

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestArange(TestCase):
    def test_infinite(self):
        assert_raises(
            (RuntimeError, ValueError), np.arange, 0, np.inf
        )  # "size exceeded",

    def test_nan_step(self):
        assert_raises(
            (RuntimeError, ValueError), np.arange, 0, 1, np.nan
        )  # "cannot compute length",

    def test_zero_step(self):
        assert_raises(ZeroDivisionError, np.arange, 0, 10, 0)
        assert_raises(ZeroDivisionError, np.arange, 0.0, 10.0, 0.0)

        # empty range
        assert_raises(ZeroDivisionError, np.arange, 0, 0, 0)
        assert_raises(ZeroDivisionError, np.arange, 0.0, 0.0, 0.0)

    def test_require_range(self):
        assert_raises(TypeError, np.arange)
        assert_raises(TypeError, np.arange, step=3)
        assert_raises(TypeError, np.arange, dtype="int64")

    @xfail  # (reason="XXX: arange(start=0, stop, step=1)")
    def test_require_range_2(self):
        assert_raises(TypeError, np.arange, start=4)

    def test_start_stop_kwarg(self):
        keyword_stop = np.arange(stop=3)
        keyword_zerotostop = np.arange(start=0, stop=3)
        keyword_start_stop = np.arange(start=3, stop=9)

        assert len(keyword_stop) == 3
        assert len(keyword_zerotostop) == 3
        assert len(keyword_start_stop) == 6
        assert_array_equal(keyword_stop, keyword_zerotostop)

    @xfail  # (reason="XXX: arange(..., dtype=bool)")
    def test_arange_booleans(self):
        # Arange makes some sense for booleans and works up to length 2.
        # But it is weird since `arange(2, 4, dtype=bool)` works.
        # Arguably, much or all of this could be deprecated/removed.
        res = np.arange(False, dtype=bool)
        assert_array_equal(res, np.array([], dtype="bool"))

        res = np.arange(True, dtype="bool")
        assert_array_equal(res, [False])

        res = np.arange(2, dtype="bool")
        assert_array_equal(res, [False, True])

        # This case is especially weird, but drops out without special case:
        res = np.arange(6, 8, dtype="bool")
        assert_array_equal(res, [True, True])

        with pytest.raises(TypeError):
            np.arange(3, dtype="bool")

    @parametrize("which", [0, 1, 2])
    def test_error_paths_and_promotion(self, which):
        args = [0, 10, 2]  # start, stop, and step
        args[which] = np.float64(2.0)  # should ensure float64 output
        assert np.arange(*args).dtype == np.float64

        # Cover stranger error path, test only to achieve code coverage!
        args[which] = [None, []]
        with pytest.raises((ValueError, RuntimeError)):
            # Fails discovering start dtype
            np.arange(*args)


class TestAppend(TestCase):
    # tests taken from np.append docstring
    def test_basic(self):
        result = np.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]])
        assert_equal(result, np.arange(1, 10, dtype=int))

        # When `axis` is specified, `values` must have the correct shape.
        result = np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0)
        assert_equal(result, np.arange(1, 10, dtype=int).reshape((3, 3)))

        with pytest.raises((RuntimeError, ValueError)):
            np.append([[1, 2, 3], [4, 5, 6]], [7, 8, 9], axis=0)


if __name__ == "__main__":
    run_tests()
