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
                             max_dim=4,
                             dtype=np.float32),
           **hu.gcs)
    def test_L1_distance(self, inputs, gc, dc):
        X, Y = inputs
        # avoid kinks by moving away from 0
        X += 0.02 * np.sign(X - Y)
        X[(X - Y) == 0.0] += 0.02

        self.ws.create_blob("X").feed(X)
        self.ws.create_blob("Y").feed(Y)
        op = core.CreateOperator(
            'L1Distance',
            ['X', 'Y'],
            ['l1_dist'],
        )
        self.ws.run(op)

        np.testing.assert_allclose(self.ws.blobs[("l1_dist")].fetch(),
                                     np.linalg.norm((X - Y).flatten(), ord=1),
                                    rtol=1e-4, atol=1e-4)

        self.assertDeviceChecks(dc, op, [X, Y], [0])
        # Gradient check wrt X
        self.assertGradientChecks(gc, op, [X, Y], 0, [0], stepsize=1e-2, threshold=1e-2)
        # Gradient check wrt Y
        self.assertGradientChecks(gc, op, [X, Y], 1, [0], stepsize=1e-2, threshold=1e-2)

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
