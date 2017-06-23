from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import hypothesis.strategies as st
import unittest
import caffe2.python.hypothesis_test_util as hu
from caffe2.python import core
from hypothesis import given


class TestResize(hu.HypothesisTestCase):
    @given(width_scale=st.floats(0.2, 4.0) | st.just(2.0),
           height_scale=st.floats(0.2, 4.0) | st.just(2.0),
           size_w=st.integers(16, 128),
           size_h=st.integers(16, 128),
           input_channels=st.integers(1, 4),
           batch_size=st.integers(1, 4),
           **hu.gcs)
    def test_nearest(self, width_scale, height_scale, size_w, size_h,
                     input_channels, batch_size,
                     gc, dc):
        op = core.CreateOperator(
            "ResizeNearest",
            ["X"],
            ["Y"],
            width_scale=width_scale,
            height_scale=height_scale,
        )
        X = np.random.rand(
            batch_size, input_channels, size_h, size_w).astype(np.float32)

        """
        This reference check is disabled because PIL's nearest neighbor
        resizing works differently than torch's SpatialUpSamplingNearest,
        which is the behavior we really care about matching
        def ref(X):
            from scipy import misc
            N = X.shape[0]
            C = X.shape[1]
            Y_h = int(size_h * height_scale)
            Y_w = int(size_w * width_scale)
            Y = np.zeros((N, C, Y_h, Y_w)).astype(np.float32)
            for n in range(N):
                for c in range(C):
                    X_ = X[n][c]
                    assert len(X_.shape) == 2
                    Y_ = misc.imresize(X_, (Y_h, Y_w), 'nearest', 'F')
                    Y[n][c] = Y_
            return (Y,)
        self.assertReferenceChecks(gc, op, [X], ref)
        """

        self.assertDeviceChecks(dc, op, [X], [0])


if __name__ == "__main__":
    unittest.main()
