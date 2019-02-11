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

def generate_rois(roi_counts, im_dims):
    assert len(roi_counts) == len(im_dims)
    all_rois = []
    for i, num_rois in enumerate(roi_counts):
        if num_rois == 0:
            continue
        # [batch_idx, x1, y1, x2, y2]
        rois = np.random.uniform(0, im_dims[i], size=(roi_counts[i], 5)).astype(
            np.float32
        )
        rois[:, 0] = i  # batch_idx
        # Swap (x1, x2) if x1 > x2
        rois[:, 1], rois[:, 3] = (
            np.minimum(rois[:, 1], rois[:, 3]),
            np.maximum(rois[:, 1], rois[:, 3]),
        )
        # Swap (y1, y2) if y1 > y2
        rois[:, 2], rois[:, 4] = (
            np.minimum(rois[:, 2], rois[:, 4]),
            np.maximum(rois[:, 2], rois[:, 4]),
        )
        all_rois.append(rois)
    if len(all_rois) > 0:
        return np.vstack(all_rois)
    return np.empty((0, 5)).astype(np.float32)

def generate_rois_rotated(roi_counts, im_dims):
    rois = generate_rois(roi_counts, im_dims)
    # [batch_id, ctr_x, ctr_y, w, h, angle]
    rotated_rois = np.empty((rois.shape[0], 6)).astype(np.float32)
    rotated_rois[:, 0] = rois[:, 0]  # batch_id
    rotated_rois[:, 1] = (rois[:, 1] + rois[:, 3]) / 2.  # ctr_x = (x1 + x2) / 2
    rotated_rois[:, 2] = (rois[:, 2] + rois[:, 4]) / 2.  # ctr_y = (y1 + y2) / 2
    rotated_rois[:, 3] = rois[:, 3] - rois[:, 1] + 1.0  # w = x2 - x1 + 1
    rotated_rois[:, 4] = rois[:, 4] - rois[:, 2] + 1.0  # h = y2 - y1 + 1
    rotated_rois[:, 5] = np.random.uniform(-90.0, 90.0)  # angle in degrees
    return rotated_rois

class TorchIntegration(hu.HypothesisTestCase):

    @given(
        roi_counts=st.lists(st.integers(0, 5), min_size=1, max_size=10),
        num_classes=st.integers(1, 10),
        rotated=st.booleans(),
        angle_bound_on=st.booleans(),
        clip_angle_thresh=st.sampled_from([-1.0, 1.0]),
        **hu.gcs_cpu_only
    )
    def test_bbox_transform(self,
        roi_counts,
        num_classes,
        rotated,
        angle_bound_on,
        clip_angle_thresh,
        gc,
        dc,
            ):
        """
        Test with rois for multiple images in a batch
        """
        batch_size = len(roi_counts)
        total_rois = sum(roi_counts)
        im_dims = np.random.randint(100, 600, batch_size)
        rois = (
            generate_rois_rotated(roi_counts, im_dims)
            if rotated
            else generate_rois(roi_counts, im_dims)
        )
        box_dim = 5 if rotated else 4
        deltas = np.random.randn(total_rois, box_dim * num_classes).astype(np.float32)
        im_info = np.zeros((batch_size, 3)).astype(np.float32)
        im_info[:, 0] = im_dims
        im_info[:, 1] = im_dims
        im_info[:, 2] = 1.0

        def bbox_transform_ref():
            ref_op = core.CreateOperator(
                "BBoxTransform",
                ["rois", "deltas", "im_info"],
                ["box_out"],
                apply_scale=False,
                correct_transform_coords=True,
                rotated=rotated,
                angle_bound_on=angle_bound_on,
                clip_angle_thresh=clip_angle_thresh,
            )
            workspace.FeedBlob("rois", rois)
            workspace.FeedBlob("deltas", deltas)
            workspace.FeedBlob("im_info", im_info)
            workspace.RunOperatorOnce(ref_op)
            return workspace.FetchBlob("box_out")

        box_out = torch.tensor(bbox_transform_ref())
        a, b = torch.ops._caffe2.BBoxTransform(
                torch.tensor(rois), torch.tensor(deltas),
                torch.tensor(im_info),
                [1.0, 1.0, 1.0, 1.0],
                False, True, rotated, angle_bound_on,
                -90, 90, clip_angle_thresh)

        torch.testing.assert_allclose(box_out, a)
