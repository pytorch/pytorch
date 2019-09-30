from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import hypothesis.strategies as st
from hypothesis import given
import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.ideep_test_util as mu

@unittest.skipIf(not workspace.C.use_mkldnn, "No MKLDNN support.")
class TransposeTest(hu.HypothesisTestCase):
    @given(
        X=hu.tensor(min_dim=1, max_dim=5, dtype=np.float32), use_axes=st.booleans(), **mu.gcs)
    def test_transpose(self, X, use_axes, gc, dc):
        ndim = len(X.shape)
        axes = np.arange(ndim)
        np.random.shuffle(axes)

        if use_axes:
            op = core.CreateOperator(
                "Transpose", ["X"], ["Y"], axes=axes, device_option=gc)
        else:
            op = core.CreateOperator(
                "Transpose", ["X"], ["Y"], device_option=gc)

        def transpose_ref(X):
            if use_axes:
                return [np.transpose(X, axes=axes)]
            else:
                return [np.transpose(X)]

        self.assertReferenceChecks(gc, op, [X], transpose_ref)
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0])


if __name__ == "__main__":
    unittest.main()
