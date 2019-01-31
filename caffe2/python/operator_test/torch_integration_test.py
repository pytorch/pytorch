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

    @given(
        A=st.integers(min_value=4, max_value=4),
        H=st.integers(min_value=10, max_value=10),
        W=st.integers(min_value=8, max_value=8),
        img_count=st.integers(min_value=3, max_value=3),
        )
    def test_generate_proposals(self, A, H, W, img_count):
        scores = np.ones((img_count, A, H, W)).astype(np.float32)
        bbox_deltas = np.linspace(0, 10, num=img_count*4*A*H*W).reshape(
                (img_count, 4*A, H, W)).astype(np.float32)
        im_info = np.ones((img_count, 3)).astype(np.float32) / 10
        anchors = np.ones((A, 4)).astype(np.float32)

        def generate_proposals_ref():
            ref_op = core.CreateOperator(
                "GenerateProposals",
                ["scores", "bbox_deltas", "im_info", "anchors"],
                ["rois", "rois_probs"],
                spatial_scale=2.0,
            )
            workspace.FeedBlob("scores", scores)
            workspace.FeedBlob("bbox_deltas", bbox_deltas)
            workspace.FeedBlob("im_info", im_info)
            workspace.FeedBlob("anchors", anchors)
            workspace.RunOperatorOnce(ref_op)
            return workspace.FetchBlob("rois"), workspace.FetchBlob("rois_probs")

        rois, rois_probs = generate_proposals_ref()
        rois = torch.tensor(rois)
        rois_probs = torch.tensor(rois_probs)
        a, b = torch.ops._caffe2.GenerateProposals(
                torch.tensor(scores), torch.tensor(bbox_deltas),
                torch.tensor(im_info), torch.tensor(anchors),
                2.0, 6000, 300, 0.7, 16, False, True, -90, 90, 1.0)
        torch.testing.assert_allclose(rois, a)
        torch.testing.assert_allclose(rois_probs, b)

