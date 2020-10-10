import os
import sys
import pytest

import numpy as np
from . import util

from numpy.testing import assert_array_equal

def _path(*a):
    return os.path.join(*((os.path.dirname(__file__),) + a))

class TestCommonBlock(util.F2PyTest):
    sources = [_path('src', 'common', 'block.f')]

    @pytest.mark.skipif(sys.platform=='win32',
                        reason='Fails with MinGW64 Gfortran (Issue #9673)')
    def test_common_block(self):
        self.module.initcb()
        assert_array_equal(self.module.block.long_bn,
                           np.array(1.0, dtype=np.float64))
        assert_array_equal(self.module.block.string_bn,
                           np.array('2', dtype='|S1'))
        assert_array_equal(self.module.block.ok,
                           np.array(3, dtype=np.int32))
