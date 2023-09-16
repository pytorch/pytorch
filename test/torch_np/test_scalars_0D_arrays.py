# Owner(s): ["module: dynamo"]

"""
Basic tests to assert and illustrate the  behavior around the decision to use 0D
arrays in place of array scalars.

Extensive tests of this sort of functionality is in numpy_tests/core/*scalar*

Also test the isscalar function (which is deliberately a bit more lax).
"""
import pytest

import torch._numpy as np
from torch._numpy.testing import assert_equal


@pytest.mark.parametrize(
    "value", [np.int64(42), np.array(42), np.asarray(42), np.asarray(np.int64(42))]
)
class TestArrayScalars:
    def test_array_scalar_basic(self, value):
        assert value.ndim == 0
        assert value.shape == ()
        assert value.size == 1
        assert value.dtype == np.dtype("int64")

    def test_conversion_to_int(self, value):
        py_scalar = int(value)
        assert py_scalar == 42
        assert isinstance(py_scalar, int)
        assert not isinstance(value, int)

    def test_decay_to_py_scalar(self, value):
        # NumPy distinguishes array scalars and 0D arrays. For instance
        # `scalar * list` is equivalent to `int(scalar) * list`, but
        # `0D array * list` is equivalent to `0D array * np.asarray(list)`.
        # Our scalars follow 0D array behavior (because they are 0D arrays)
        lst = [1, 2, 3]

        product = value * lst
        assert isinstance(product, np.ndarray)
        assert product.shape == (3,)
        assert_equal(product, [42, 42 * 2, 42 * 3])

        # repeat with right-mulitply
        product = lst * value
        assert isinstance(product, np.ndarray)
        assert product.shape == (3,)
        assert_equal(product, [42, 42 * 2, 42 * 3])


def test_scalar_comparisons():
    scalar = np.int64(42)
    arr = np.array(42)

    assert arr == scalar
    assert arr >= scalar
    assert arr <= scalar

    assert scalar == 42
    assert arr == 42


class TestIsScalar:
    #
    # np.isscalar(...) checks that its argument is a numeric object with exactly one element.
    #
    # This differs from NumPy which also requires that shape == ().
    #
    scalars = [
        42,
        int(42.0),
        np.float32(42),
        np.array(42),
        [42],
        [[42]],
        np.array([42]),
        np.array([[42]]),
    ]

    import math

    not_scalars = [
        int,
        np.float32,
        "s",
        "string",
        (),
        [],
        math.sin,
        np,
        np.transpose,
        [1, 2],
        np.asarray([1, 2]),
        np.float32([1, 2]),
    ]

    @pytest.mark.parametrize("value", scalars)
    def test_is_scalar(self, value):
        assert np.isscalar(value)

    @pytest.mark.parametrize("value", not_scalars)
    def test_is_not_scalar(self, value):
        assert not np.isscalar(value)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
