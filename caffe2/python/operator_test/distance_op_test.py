from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
from hypothesis import given


class DistanceTest(hu.HypothesisTestCase):
    @given(inputs=hu.tensors(n=2,
                             min_dim=1,
                             max_dim=2,
                             dtype=np.float32),
           **hu.gcs)
    def test_dot_product(self, inputs, gc, dc):
        X, Y = inputs
        op = core.CreateOperator(
            'DotProduct',
            ['X', 'Y'],
            ['DOT'],
        )

        def dot_ref(X, Y):
            return ([np.dot(x, y) for x, y in zip(X, Y)],)

        # Check against numpy dot reference
        self.assertReferenceChecks(gc, op, [X, Y], dot_ref)
        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [X, Y], [0])
        # Gradient check wrt X
        self.assertGradientChecks(gc, op, [X, Y], 0, [0])
        # Gradient check wrt Y
        self.assertGradientChecks(gc, op, [X, Y], 1, [0])
