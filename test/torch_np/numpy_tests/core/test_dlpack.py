# Owner(s): ["module: dynamo"]

import functools
import sys
import unittest
from unittest import skipIf as skipif

import numpy
import pytest

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xpassIfTorchDynamo_np,
)


if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy.testing import assert_array_equal
else:
    import torch._numpy as np
    from torch._numpy.testing import assert_array_equal


skip = functools.partial(skipif, True)


IS_PYPY = False


@skipif(numpy.__version__ < "1.24", reason="numpy.dlpack is new in numpy 1.23")
@instantiate_parametrized_tests
class TestDLPack(TestCase):
    @xpassIfTorchDynamo_np  # (reason="pytorch seems to handle refcounts differently")
    @skipif(IS_PYPY, reason="PyPy can't get refcounts.")
    def test_dunder_dlpack_refcount(self):
        x = np.arange(5)
        y = x.__dlpack__()
        assert sys.getrefcount(x) == 3
        del y
        assert sys.getrefcount(x) == 2

    @unittest.expectedFailure
    @skipIfTorchDynamo("I can't figure out how to get __dlpack__ into trace_rules.py")
    def test_dunder_dlpack_stream(self):
        x = np.arange(5)
        x.__dlpack__(stream=None)

        with pytest.raises(RuntimeError):
            x.__dlpack__(stream=1)

    @xpassIfTorchDynamo_np  # (reason="pytorch seems to handle refcounts differently")
    @skipif(IS_PYPY, reason="PyPy can't get refcounts.")
    def test_from_dlpack_refcount(self):
        x = np.arange(5)
        y = np.from_dlpack(x)
        assert sys.getrefcount(x) == 3
        del y
        assert sys.getrefcount(x) == 2

    @parametrize(
        "dtype",
        [
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.float16,
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
        ],
    )
    def test_dtype_passthrough(self, dtype):
        x = np.arange(5, dtype=dtype)
        y = np.from_dlpack(x)

        assert y.dtype == x.dtype
        assert_array_equal(x, y)

    def test_non_contiguous(self):
        x = np.arange(25).reshape((5, 5))

        y1 = x[0]
        assert_array_equal(y1, np.from_dlpack(y1))

        y2 = x[:, 0]
        assert_array_equal(y2, np.from_dlpack(y2))

        y3 = x[1, :]
        assert_array_equal(y3, np.from_dlpack(y3))

        y4 = x[1]
        assert_array_equal(y4, np.from_dlpack(y4))

        y5 = np.diagonal(x).copy()
        assert_array_equal(y5, np.from_dlpack(y5))

    @parametrize("ndim", range(33))
    def test_higher_dims(self, ndim):
        shape = (1,) * ndim
        x = np.zeros(shape, dtype=np.float64)

        assert shape == np.from_dlpack(x).shape

    def test_dlpack_device(self):
        x = np.arange(5)
        assert x.__dlpack_device__() == (1, 0)
        y = np.from_dlpack(x)
        assert y.__dlpack_device__() == (1, 0)
        z = y[::2]
        assert z.__dlpack_device__() == (1, 0)

    def dlpack_deleter_exception(self):
        x = np.arange(5)
        _ = x.__dlpack__()
        raise RuntimeError

    def test_dlpack_destructor_exception(self):
        with pytest.raises(RuntimeError):
            self.dlpack_deleter_exception()

    @skip(reason="no readonly arrays in pytorch")
    def test_readonly(self):
        x = np.arange(5)
        x.flags.writeable = False
        with pytest.raises(BufferError):
            x.__dlpack__()

    def test_ndim0(self):
        x = np.array(1.0)
        y = np.from_dlpack(x)
        assert_array_equal(x, y)

    def test_from_torch(self):
        t = torch.arange(4)
        a = np.from_dlpack(t)
        assert_array_equal(a, np.asarray(t))

    def test_to_torch(self):
        a = np.arange(4)
        t = torch.from_dlpack(a)
        assert_array_equal(np.asarray(t), a)


if __name__ == "__main__":
    run_tests()
