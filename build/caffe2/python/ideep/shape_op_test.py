from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import hypothesis.strategies as st
from hypothesis import given, settings
import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.ideep_test_util as mu


@unittest.skipIf(not workspace.C.use_mkldnn, "No MKLDNN support.")
class ShapeTest(hu.HypothesisTestCase):
    @given(n=st.integers(1, 128),
           c=st.integers(1, 128),
           h=st.integers(1, 128),
           w=st.integers(1, 128),
           **mu.gcs)
    def test_shape(self, n, c, h, w, gc, dc):
        op0 = core.CreateOperator(
            "Shape",
            ["X0"],
            ["Y0"],
            device_option=dc[0]
        )
        op1 = core.CreateOperator(
            "Shape",
            ["X1"],
            ["Y1"],
            device_option=dc[1]
        )
        X = np.random.rand(n, c, h, w).astype(np.float32) - 0.5
        workspace.FeedBlob('X0', X, dc[0])
        workspace.FeedBlob('X1', X, dc[1])
        workspace.RunOperatorOnce(op0)
        workspace.RunOperatorOnce(op1)
        Y0 = workspace.FetchBlob('Y0')
        Y1 = workspace.FetchBlob('Y1')

        if not np.allclose(Y0, Y1, atol=0, rtol=0):
            print(Y1.flatten())
            print(Y0.flatten())
            print(np.max(np.abs(Y1 - Y0)))
            self.assertTrue(False)

    @given(n=st.integers(1, 128),
           c=st.integers(1, 128),
           h=st.integers(1, 128),
           w=st.integers(1, 128),
           axes=st.lists(st.integers(0, 3), min_size=1, max_size=3),
           **mu.gcs)
    def test_shape_with_axes(self, n, c, h, w, axes, gc, dc):
        axes = list(set(axes)).sort()
        op0 = core.CreateOperator(
            "Shape",
            ["X0"],
            ["Y0"],
            axes = axes,
            device_option=dc[0]
        )
        op1 = core.CreateOperator(
            "Shape",
            ["X1"],
            ["Y1"],
            axes = axes,
            device_option=dc[1]
        )
        X = np.random.rand(n, c, h, w).astype(np.float32) - 0.5
        workspace.FeedBlob('X0', X, dc[0])
        workspace.FeedBlob('X1', X, dc[1])
        workspace.RunOperatorOnce(op0)
        workspace.RunOperatorOnce(op1)
        Y0 = workspace.FetchBlob('Y0')
        Y1 = workspace.FetchBlob('Y1')

        if not np.allclose(Y0, Y1, atol=0, rtol=0):
            print(Y1.flatten())
            print(Y0.flatten())
            print(np.max(np.abs(Y1 - Y0)))
            self.assertTrue(False)


if __name__ == "__main__":
    unittest.main()
