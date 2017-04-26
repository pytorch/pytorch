from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given
import hypothesis.strategies as st
import caffe2.python.hypothesis_test_util as hu
import numpy as np

import unittest


class TestScaleGradient(hu.HypothesisTestCase):

    @given(X=hu.tensor(min_dim=0, max_dim=3),
           scale=st.floats(min_value=-100, max_value=100),
           **hu.gcs_cpu_only)
    def test_scale_gradient(self, X, scale, gc, dc):
        if isinstance(X, float):
            X = np.array(X, dtype=np.float32)
        op = core.CreateOperator(
            "ScaleGradient", ["X"], ["X"],
            scale=float(scale),
        )

        def pass_through(x):
            return [x]

        def grad_ref(grad, o, x):
            return [x[0] * float(scale)]

        self.assertReferenceChecks(
            gc, op, [X], pass_through,
            output_to_grad='X',
            grad_reference=grad_ref,
        )
        self.assertDeviceChecks(dc, op, [X], [0])


if __name__ == "__main__":
    unittest.main()
