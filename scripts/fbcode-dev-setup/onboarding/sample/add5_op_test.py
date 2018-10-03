from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
from hypothesis import given
import hypothesis.strategies as st
import numpy as np
import unittest


class TestWhere(hu.HypothesisTestCase):

    @given(N=st.integers(min_value=1, max_value=10),
           C=st.integers(min_value=1, max_value=10),
           # NB: use hg.gcs when built with CUDA
           **hu.gcs_cpu_only)
    def test_where(self, N, C, gc, dc):
        # set th seed
        np.random.seed(101)
        # TODO: test double, int and int64
        data = np.random.rand(N, C).astype(np.float32)
        op = core.CreateOperator("Add5", ["data"], ["output"])

        # device check
        self.assertDeviceChecks(dc, op, [data], [0])

        # gradient check
        self.assertGradientChecks(gc, op, [data], 0, [0])

        # reference check
        def ref_add5(input):
            return [input + 5]
        self.assertReferenceChecks(gc, op, [data], ref_add5)


if __name__ == "__main__":
    unittest.main()
