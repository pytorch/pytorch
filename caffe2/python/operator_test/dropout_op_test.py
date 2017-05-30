from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from hypothesis import assume, given
import hypothesis.strategies as st
import numpy as np

from caffe2.proto import caffe2_pb2
from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu


class TestDropout(hu.HypothesisTestCase):

    @given(X=hu.tensor(), in_place=st.booleans(), **hu.gcs)
    def test_dropout_ratio0(self, X, in_place, gc, dc):

        # TODO: enable this path when the op is fixed
        if in_place:
            # Skip if trying in-place on GPU
            assume(gc.device_type != caffe2_pb2.CUDA)
            # If in-place on CPU, don't compare with GPU
            dc = dc[:1]

        op = core.CreateOperator("Dropout", ["X"],
                                 ["X" if in_place else "Y", "mask"],
                                 ratio=0.0)
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0])
        def reference_dropout_ratio0(x):
            return x, np.ones(x.shape, dtype=np.bool)
        self.assertReferenceChecks(gc, op, [X], reference_dropout_ratio0)
