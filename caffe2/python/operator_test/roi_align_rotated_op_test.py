from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
import copy


class RoIAlignRotatedOp(hu.HypothesisTestCase):
    def bbox_xywh_to_xyxy(self, boxes):
        """
        Convert from [center_x center_y w h] format to [x1 y1 x2 y2].
        """
        w, h = boxes[:, 2], boxes[:, 3]
        boxes[:, 0] -= w / 2.0  # x1 = center_x - width/2
        boxes[:, 1] -= h / 2.0  # y1 = center_y - height/2
        boxes[:, 2] = boxes[:, 0] + w  # x2 = x1 + width
        boxes[:, 3] = boxes[:, 1] + h  # y2 = y1 + height
        return boxes

    @given(
        H=st.integers(min_value=100, max_value=200),
        W=st.integers(min_value=100, max_value=200),
        C=st.integers(min_value=1, max_value=5),
        num_rois=st.integers(min_value=0, max_value=10),
        pooled_size=st.sampled_from([7, 14]),
        order=st.sampled_from(["NCHW", "NHWC"]),
        **hu.gcs_cpu_only
    )
    def test_horizontal_rois(self, H, W, C, num_rois, pooled_size, order, gc, dc):
        """
        Test that results match with RoIAlign when angle=0.
        """
        if order == "NCHW":
            X = np.random.randn(1, C, H, W).astype(np.float32)
        else:
            X = np.random.randn(1, H, W, C).astype(np.float32)
        R = np.zeros((num_rois, 6)).astype(np.float32)
        angle = 0.0
        for i in range(num_rois):
            x = np.random.uniform(1, W - 1)
            y = np.random.uniform(1, H - 1)
            w = np.random.uniform(1, min(x, W - x))
            h = np.random.uniform(1, min(y, H - y))
            R[i] = [0, x, y, w, h, angle]

        op = core.CreateOperator(
            "RoIAlignRotated",
            ["X", "R"],
            ["Y"],
            pooled_h=pooled_size,
            pooled_w=pooled_size,
            order=order,
            sampling_ratio=0,
        )

        def roialign_ref(X, R):
            # Remove angle and convert from [center_x center_y w h]
            # to [x1 y1 x2 y2] format.
            R_ref = copy.deepcopy(R[:, 0:5])
            R_ref[:, 1:5] = self.bbox_xywh_to_xyxy(R_ref[:, 1:5])

            ref_op = core.CreateOperator(
                "RoIAlign",
                ["X_ref", "R_ref"],
                ["Y_ref"],
                pooled_h=pooled_size,
                pooled_w=pooled_size,
                order=order,
                sampling_ratio=0,
            )
            workspace.FeedBlob("X_ref", X)
            workspace.FeedBlob("R_ref", R_ref)
            workspace.RunOperatorOnce(ref_op)
            return [workspace.FetchBlob("Y_ref")]

        self.assertReferenceChecks(
            device_option=gc, op=op, inputs=[X, R], reference=roialign_ref
        )

    @given(
        H=st.integers(min_value=100, max_value=200),
        W=st.integers(min_value=100, max_value=200),
        C=st.integers(min_value=1, max_value=5),
        num_rois=st.integers(min_value=0, max_value=10),
        pooled_size=st.sampled_from([7, 14]),
        order=st.sampled_from(["NCHW", "NHWC"]),
        angle=st.sampled_from([-270, -180, -90, 90, 180, 270]),
        **hu.gcs_cpu_only
    )
    def test_simple_rotations(
        self, H, W, C, num_rois, pooled_size, order, angle, gc, dc
    ):
        """
        Test with right-angled rotations that don't need interpolation.
        """
        if order == "NCHW":
            X = np.random.randn(1, C, H, W).astype(np.float32)
        else:
            X = np.random.randn(1, H, W, C).astype(np.float32)
        R = np.zeros((num_rois, 6)).astype(np.float32)
        for i in range(num_rois):
            x = np.random.uniform(1, W - 1)
            y = np.random.uniform(1, H - 1)
            w = np.random.uniform(1, min(x, W - x, y, H - y))
            h = np.random.uniform(1, min(x, W - x, y, H - y))
            R[i] = [0, x, y, w, h, angle]

        op = core.CreateOperator(
            "RoIAlignRotated",
            ["X", "R"],
            ["Y"],
            pooled_h=pooled_size,
            pooled_w=pooled_size,
            order=order,
            sampling_ratio=0,
        )

        def roialign_ref(X, R):
            # `angle` denotes counter-clockwise rotation. Rotate the input
            # feature map in the opposite (clockwise) direction and perform
            # standard RoIAlign. We assume all RoIs have the same angle.
            norm_angle = (angle + 360) % 360
            axes = (2, 3) if order == "NCHW" else (1, 2)
            X_ref = np.rot90(X, k=-norm_angle / 90, axes=axes)

            # Rotate RoIs clockwise wrt the center of the input feature
            # map to make them horizontal and convert from
            # [center_x center_y w h] to [x1 y1 x2 y2] format.
            roi_x, roi_y = R[:, 1], R[:, 2]
            if norm_angle == 90:
                new_roi_x = H - roi_y - 1
                new_roi_y = roi_x
            elif norm_angle == 180:
                new_roi_x = W - roi_x - 1
                new_roi_y = H - roi_y - 1
            elif norm_angle == 270:
                new_roi_x = roi_y
                new_roi_y = W - roi_x - 1
            else:
                raise NotImplementedError
            R_ref = copy.deepcopy(R[:, 0:5])
            R_ref[:, 1], R_ref[:, 2] = new_roi_x, new_roi_y
            R_ref[:, 1:5] = self.bbox_xywh_to_xyxy(R_ref[:, 1:5])

            ref_op = core.CreateOperator(
                "RoIAlign",
                ["X_ref", "R_ref"],
                ["Y_ref"],
                pooled_h=pooled_size,
                pooled_w=pooled_size,
                order=order,
                sampling_ratio=0,
            )
            workspace.FeedBlob("X_ref", X_ref)
            workspace.FeedBlob("R_ref", R_ref)
            workspace.RunOperatorOnce(ref_op)
            return [workspace.FetchBlob("Y_ref")]

        self.assertReferenceChecks(
            device_option=gc, op=op, inputs=[X, R], reference=roialign_ref
        )
