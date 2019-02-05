from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace
import torch
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np

class TorchIntegration(hu.HypothesisTestCase):
    @given(
        H=st.integers(min_value=50, max_value=100),
        W=st.integers(min_value=50, max_value=100),
        C=st.integers(min_value=1, max_value=3),
        num_rois=st.integers(min_value=1, max_value=10),
        pooled_size=st.sampled_from([7, 14])
        )
    def test_roi_align(self, H, W, C, num_rois, pooled_size):
        X = np.random.randn(1, C, H, W).astype(np.float32)
        R = np.zeros((num_rois, 5)).astype(np.float32)
        for i in range(num_rois):
            x = np.random.uniform(1, W - 1)
            y = np.random.uniform(1, H - 1)
            w = np.random.uniform(1, min(x, W - x))
            h = np.random.uniform(1, min(y, H - y))
            R[i] = [0, x, y, w, h]

        def roialign_ref(X, R):
            ref_op = core.CreateOperator(
                "RoIAlign",
                ["X_ref", "R_ref"],
                ["Y_ref"],
                order="NCHW",
                spatial_scale=1.0,
                pooled_h=pooled_size,
                pooled_w=pooled_size,
                sampling_ratio=0,
            )
            workspace.FeedBlob("X_ref", X)
            workspace.FeedBlob("R_ref", R)
            workspace.RunOperatorOnce(ref_op)
            return workspace.FetchBlob("Y_ref")
        Y_ref = torch.tensor(roialign_ref(X, R))
        Y = torch.ops._caffe2.RoIAlign(
                torch.tensor(X), torch.tensor(R),
                "NCHW", 1., pooled_size, pooled_size, 0)
        torch.testing.assert_allclose(Y, Y_ref)

