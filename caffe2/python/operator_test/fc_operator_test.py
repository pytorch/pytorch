from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np


class TestFcOperator(hu.HypothesisTestCase):

    @given(n=st.integers(1, 5), m=st.integers(1, 5),
           k=st.integers(1, 5), **hu.gcs)
    def test_fc(self, n, m, k, gc, dc):
        X = np.random.rand(m, k).astype(np.float32) - 0.5
        W = np.random.rand(n, k).astype(np.float32) - 0.5
        b = np.random.rand(n).astype(np.float32) - 0.5

        def fc_op(X, W, b):
            return [np.dot(X, W.transpose()) + b]

        op = core.CreateOperator(
            'FC',
            ['X', 'W', 'b'],
            'out'
        )

        # Check against numpy reference
        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X, W, b],
            reference=fc_op,
        )
        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [X, W, b], [0])
        # Gradient check wrt X
        self.assertGradientChecks(gc, op, [X, W, b], 0, [0])
        # Gradient check wrt W
        self.assertGradientChecks(gc, op, [X, W, b], 1, [0])
        # Gradient check wrt b
        self.assertGradientChecks(gc, op, [X, W, b], 2, [0])


if __name__ == "__main__":
    import unittest
    unittest.main()
