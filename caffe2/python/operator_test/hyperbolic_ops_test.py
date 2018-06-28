from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np


class TestHyperbolicOps(hu.HypothesisTestCase):
    def _test_hyperbolic_op(self, op_name, np_ref, X, in_place, gc, dc):
        op = core.CreateOperator(op_name, ["X"], ["X"] if in_place else ["Y"])

        def ref(X):
            return [np_ref(X)]

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=ref,
        )
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0])

    @given(X=hu.tensor(dtype=np.float32), in_place=st.booleans(), **hu.gcs)
    def test_tanh(self, X, in_place, gc, dc):
        self._test_hyperbolic_op("Tanh", np.tanh, X, in_place, gc, dc)
