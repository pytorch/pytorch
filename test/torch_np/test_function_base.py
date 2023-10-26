# Owner(s): ["module: dynamo"]


import pytest

import torch._numpy as np
from torch._numpy.testing import assert_equal

from torch.testing._internal.common_utils import run_tests, TestCase


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
