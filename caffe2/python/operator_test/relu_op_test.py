from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.mkl_test_util as mu
import numpy as np

import unittest


class TestRelu(hu.HypothesisTestCase):

    @given(X=hu.tensor(), **mu.gcs)
    def test_relu(self, X, gc, dc):
        op = core.CreateOperator("Relu", ["X"], ["Y"])
        # go away from the origin point to avoid kink problems
        X += 0.02 * np.sign(X)
        X[X == 0.0] += 0.02
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0])


if __name__ == "__main__":
    unittest.main()
