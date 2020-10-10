import os
import pytest

import numpy as np
from numpy.testing import assert_raises, assert_equal

from . import util


def _path(*a):
    return os.path.join(*((os.path.dirname(__file__),) + a))


class TestParameters(util.F2PyTest):
    # Check that intent(in out) translates as intent(inout)
    sources = [_path('src', 'parameter', 'constant_real.f90'),
               _path('src', 'parameter', 'constant_integer.f90'),
               _path('src', 'parameter', 'constant_both.f90'),
               _path('src', 'parameter', 'constant_compound.f90'),
               _path('src', 'parameter', 'constant_non_compound.f90'),
    ]

    @pytest.mark.slow
    def test_constant_real_single(self):
        # non-contiguous should raise error
        x = np.arange(6, dtype=np.float32)[::2]
        assert_raises(ValueError, self.module.foo_single, x)

        # check values with contiguous array
        x = np.arange(3, dtype=np.float32)
        self.module.foo_single(x)
        assert_equal(x, [0 + 1 + 2*3, 1, 2])

    @pytest.mark.slow
    def test_constant_real_double(self):
        # non-contiguous should raise error
        x = np.arange(6, dtype=np.float64)[::2]
        assert_raises(ValueError, self.module.foo_double, x)

        # check values with contiguous array
        x = np.arange(3, dtype=np.float64)
        self.module.foo_double(x)
        assert_equal(x, [0 + 1 + 2*3, 1, 2])

    @pytest.mark.slow
    def test_constant_compound_int(self):
        # non-contiguous should raise error
        x = np.arange(6, dtype=np.int32)[::2]
        assert_raises(ValueError, self.module.foo_compound_int, x)

        # check values with contiguous array
        x = np.arange(3, dtype=np.int32)
        self.module.foo_compound_int(x)
        assert_equal(x, [0 + 1 + 2*6, 1, 2])

    @pytest.mark.slow
    def test_constant_non_compound_int(self):
        # check values
        x = np.arange(4, dtype=np.int32)
        self.module.foo_non_compound_int(x)
        assert_equal(x, [0 + 1 + 2 + 3*4, 1, 2, 3])

    @pytest.mark.slow
    def test_constant_integer_int(self):
        # non-contiguous should raise error
        x = np.arange(6, dtype=np.int32)[::2]
        assert_raises(ValueError, self.module.foo_int, x)

        # check values with contiguous array
        x = np.arange(3, dtype=np.int32)
        self.module.foo_int(x)
        assert_equal(x, [0 + 1 + 2*3, 1, 2])

    @pytest.mark.slow
    def test_constant_integer_long(self):
        # non-contiguous should raise error
        x = np.arange(6, dtype=np.int64)[::2]
        assert_raises(ValueError, self.module.foo_long, x)

        # check values with contiguous array
        x = np.arange(3, dtype=np.int64)
        self.module.foo_long(x)
        assert_equal(x, [0 + 1 + 2*3, 1, 2])

    @pytest.mark.slow
    def test_constant_both(self):
        # non-contiguous should raise error
        x = np.arange(6, dtype=np.float64)[::2]
        assert_raises(ValueError, self.module.foo, x)

        # check values with contiguous array
        x = np.arange(3, dtype=np.float64)
        self.module.foo(x)
        assert_equal(x, [0 + 1*3*3 + 2*3*3, 1*3, 2*3])

    @pytest.mark.slow
    def test_constant_no(self):
        # non-contiguous should raise error
        x = np.arange(6, dtype=np.float64)[::2]
        assert_raises(ValueError, self.module.foo_no, x)

        # check values with contiguous array
        x = np.arange(3, dtype=np.float64)
        self.module.foo_no(x)
        assert_equal(x, [0 + 1*3*3 + 2*3*3, 1*3, 2*3])

    @pytest.mark.slow
    def test_constant_sum(self):
        # non-contiguous should raise error
        x = np.arange(6, dtype=np.float64)[::2]
        assert_raises(ValueError, self.module.foo_sum, x)

        # check values with contiguous array
        x = np.arange(3, dtype=np.float64)
        self.module.foo_sum(x)
        assert_equal(x, [0 + 1*3*3 + 2*3*3, 1*3, 2*3])
