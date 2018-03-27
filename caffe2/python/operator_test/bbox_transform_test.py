from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np


# Reference implementation from detectron/lib/utils/boxes.py
def bbox_transform(boxes, deltas, weights=(1.0, 1.0, 1.0, 1.0)):
    """Forward transform that maps proposal boxes to predicted ground-truth
    boxes using bounding-box regression deltas. See bbox_transform_inv for a
    description of the weights argument.
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = deltas[:, 0::4] / wx
    dy = deltas[:, 1::4] / wy
    dw = deltas[:, 2::4] / ww
    dh = deltas[:, 3::4] / wh

    # Prevent sending too large values into np.exp()
    BBOX_XFORM_CLIP = np.log(1000. / 16.)
    dw = np.minimum(dw, BBOX_XFORM_CLIP)
    dh = np.minimum(dh, BBOX_XFORM_CLIP)

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
    # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

    return pred_boxes


# Reference implementation from detectron/lib/utils/boxes.py
def clip_tiled_boxes(boxes, im_shape):
    """Clip boxes to image boundaries. im_shape is [height, width] and boxes
    has shape (N, 4 * num_tiled_boxes)."""
    assert boxes.shape[1] % 4 == 0, \
        'boxes.shape[1] is {:d}, but must be divisible by 4.'.format(
        boxes.shape[1]
    )
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def generate_rois(roi_counts, im_dims):
    assert len(roi_counts) == len(im_dims)
    all_rois = []
    for i, num_rois in enumerate(roi_counts):
        if num_rois == 0:
            continue
        # [batch_idx, x1, y1, x2, y2]
        rois = np.random.uniform(
            0, im_dims[i], size=(roi_counts[i], 5)
        ).astype(np.float32)
        rois[:, 0] = i  # batch_idx
        # Swap (x1, x2) if x1 > x2
        rois[:, 1], rois[:, 3] = np.minimum(rois[:, 1], rois[:, 3]), \
                np.maximum(rois[:, 1], rois[:, 3])
        # Swap (y1, y2) if y1 > y2
        rois[:, 2], rois[:, 4] = np.minimum(rois[:, 2], rois[:, 4]), \
                np.maximum(rois[:, 2], rois[:, 4])
        all_rois.append(rois)
    if len(all_rois) > 0:
        return np.vstack(all_rois)
    return np.empty((0, 5)).astype(np.float32)


class TestBBoxTransformOp(hu.HypothesisTestCase):
    @given(
        num_rois=st.integers(1, 10),
        num_classes=st.integers(1, 10),
        im_dim=st.integers(100, 600),
        skip_batch_id=st.booleans(),
        **hu.gcs_cpu_only
    )
    def test_bbox_transform(
        self, num_rois, num_classes, im_dim, skip_batch_id, gc, dc
    ):
        """
        Test with all rois belonging to a single image per run.
        """
        rois = generate_rois([num_rois], [im_dim])
        if skip_batch_id:
            rois = rois[:, 1:5]
        deltas = np.random.randn(num_rois, 4 * num_classes).astype(np.float32)
        im_info = np.array([im_dim, im_dim,
                            1.0]).astype(np.float32).reshape(1, 3)

        def bbox_transform_ref(rois, deltas, im_info):
            boxes = rois if rois.shape[1] == 4 else rois[:, 1:5]
            box_out = bbox_transform(boxes, deltas)
            im_shape = im_info[0, 0:2]
            box_out = clip_tiled_boxes(box_out, im_shape)
            return [box_out]

        op = core.CreateOperator(
            "BBoxTransform",
            ["rois", "deltas", "im_info"],
            ["box_out"],
            apply_scale=False,
            correct_transform_coords=True,
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[rois, deltas, im_info],
            reference=bbox_transform_ref,
        )

    @given(
        roi_counts=st.lists(st.integers(0, 5), min_size=1, max_size=10),
        num_classes=st.integers(1, 10),
        **hu.gcs_cpu_only
    )
    def test_bbox_transform_batch(self, roi_counts, num_classes, gc, dc):
        """
        Test with rois for multiple images in a batch
        """
        batch_size = len(roi_counts)
        total_rois = sum(roi_counts)
        im_dims = np.random.randint(100, 600, batch_size)
        rois = generate_rois(roi_counts, im_dims)
        deltas = np.random.randn(total_rois, 4 * num_classes).astype(np.float32)
        im_info = np.zeros((batch_size, 3)).astype(np.float32)
        im_info[:, 0] = im_dims
        im_info[:, 1] = im_dims
        im_info[:, 2] = 1.0

        def bbox_transform_ref(rois, deltas, im_info):
            box_out = []
            offset = 0
            for i, num_rois in enumerate(roi_counts):
                if num_rois == 0:
                    continue
                cur_boxes = rois[offset:offset + num_rois, 1:5]
                cur_deltas = deltas[offset:offset + num_rois]
                cur_box_out = bbox_transform(cur_boxes, cur_deltas)
                im_shape = im_info[i, 0:2]
                cur_box_out = clip_tiled_boxes(cur_box_out, im_shape)
                box_out.append(cur_box_out)
                offset += num_rois

            if len(box_out) > 0:
                box_out = np.vstack(box_out)
            else:
                box_out = np.empty(deltas.shape).astype(np.float32)
            return [box_out, roi_counts]

        op = core.CreateOperator(
            "BBoxTransform",
            ["rois", "deltas", "im_info"],
            ["box_out", "roi_batch_splits"],
            apply_scale=False,
            correct_transform_coords=True,
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[rois, deltas, im_info],
            reference=bbox_transform_ref,
        )
