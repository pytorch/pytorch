




from caffe2.python import core
from hypothesis import given, settings
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
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
    assert (
        boxes.shape[1] % 4 == 0
    ), "boxes.shape[1] is {:d}, but must be divisible by 4.".format(
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


def bbox_transform_rotated(
    boxes,
    deltas,
    weights=(1.0, 1.0, 1.0, 1.0),
    angle_bound_on=True,
    angle_bound_lo=-90,
    angle_bound_hi=90,
):
    """
    Similar to bbox_transform but for rotated boxes with angle info.
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    ctr_x = boxes[:, 0]
    ctr_y = boxes[:, 1]
    widths = boxes[:, 2]
    heights = boxes[:, 3]
    angles = boxes[:, 4]

    wx, wy, ww, wh = weights
    dx = deltas[:, 0::5] / wx
    dy = deltas[:, 1::5] / wy
    dw = deltas[:, 2::5] / ww
    dh = deltas[:, 3::5] / wh
    da = deltas[:, 4::5] * 180.0 / np.pi

    # Prevent sending too large values into np.exp()
    BBOX_XFORM_CLIP = np.log(1000. / 16.)
    dw = np.minimum(dw, BBOX_XFORM_CLIP)
    dh = np.minimum(dh, BBOX_XFORM_CLIP)

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    pred_boxes[:, 0::5] = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_boxes[:, 1::5] = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_boxes[:, 2::5] = np.exp(dw) * widths[:, np.newaxis]
    pred_boxes[:, 3::5] = np.exp(dh) * heights[:, np.newaxis]

    pred_angle = da + angles[:, np.newaxis]
    if angle_bound_on:
        period = angle_bound_hi - angle_bound_lo
        assert period % 180 == 0
        pred_angle[np.where(pred_angle < angle_bound_lo)] += period
        pred_angle[np.where(pred_angle > angle_bound_hi)] -= period
    pred_boxes[:, 4::5] = pred_angle

    return pred_boxes


def clip_tiled_boxes_rotated(boxes, im_shape, angle_thresh=1.0):
    """
    Similar to clip_tiled_boxes but for rotated boxes with angle info.
    Only clips almost horizontal boxes within angle_thresh. The rest are
    left unchanged.
    """
    assert (
        boxes.shape[1] % 5 == 0
    ), "boxes.shape[1] is {:d}, but must be divisible by 5.".format(
        boxes.shape[1]
    )

    (H, W) = im_shape[:2]

    # Filter boxes that are almost upright within angle_thresh tolerance
    idx = np.where(np.abs(boxes[:, 4::5]) <= angle_thresh)
    idx5 = idx[1] * 5
    # convert to (x1, y1, x2, y2)
    x1 = boxes[idx[0], idx5] - (boxes[idx[0], idx5 + 2] - 1) / 2.0
    y1 = boxes[idx[0], idx5 + 1] - (boxes[idx[0], idx5 + 3] - 1) / 2.0
    x2 = boxes[idx[0], idx5] + (boxes[idx[0], idx5 + 2] - 1) / 2.0
    y2 = boxes[idx[0], idx5 + 1] + (boxes[idx[0], idx5 + 3] - 1) / 2.0
    # clip
    x1 = np.maximum(np.minimum(x1, W - 1), 0)
    y1 = np.maximum(np.minimum(y1, H - 1), 0)
    x2 = np.maximum(np.minimum(x2, W - 1), 0)
    y2 = np.maximum(np.minimum(y2, H - 1), 0)
    # convert back to (xc, yc, w, h)
    boxes[idx[0], idx5] = (x1 + x2) / 2.0
    boxes[idx[0], idx5 + 1] = (y1 + y2) / 2.0
    boxes[idx[0], idx5 + 2] = x2 - x1 + 1
    boxes[idx[0], idx5 + 3] = y2 - y1 + 1

    return boxes


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


class TestBBoxTransformOp(serial.SerializedTestCase):
    @given(
        num_rois=st.integers(1, 10),
        num_classes=st.integers(1, 10),
        im_dim=st.integers(100, 600),
        skip_batch_id=st.booleans(),
        rotated=st.booleans(),
        angle_bound_on=st.booleans(),
        clip_angle_thresh=st.sampled_from([-1.0, 1.0]),
        **hu.gcs_cpu_only
    )
    @settings(deadline=1000)
    def test_bbox_transform(
        self,
        num_rois,
        num_classes,
        im_dim,
        skip_batch_id,
        rotated,
        angle_bound_on,
        clip_angle_thresh,
        gc,
        dc,
    ):
        """
        Test with all rois belonging to a single image per run.
        """
        rois = (
            generate_rois_rotated([num_rois], [im_dim])
            if rotated
            else generate_rois([num_rois], [im_dim])
        )
        box_dim = 5 if rotated else 4
        if skip_batch_id:
            rois = rois[:, 1:]
        deltas = np.random.randn(num_rois, box_dim * num_classes).astype(np.float32)
        im_info = np.array([im_dim, im_dim, 1.0]).astype(np.float32).reshape(1, 3)

        def bbox_transform_ref(rois, deltas, im_info):
            boxes = rois if rois.shape[1] == box_dim else rois[:, 1:]
            im_shape = im_info[0, 0:2]
            if rotated:
                box_out = bbox_transform_rotated(
                    boxes, deltas, angle_bound_on=angle_bound_on
                )
                box_out = clip_tiled_boxes_rotated(
                    box_out, im_shape, angle_thresh=clip_angle_thresh
                )
            else:
                box_out = bbox_transform(boxes, deltas)
                box_out = clip_tiled_boxes(box_out, im_shape)
            return [box_out]

        op = core.CreateOperator(
            "BBoxTransform",
            ["rois", "deltas", "im_info"],
            ["box_out"],
            apply_scale=False,
            correct_transform_coords=True,
            rotated=rotated,
            angle_bound_on=angle_bound_on,
            clip_angle_thresh=clip_angle_thresh,
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
        rotated=st.booleans(),
        angle_bound_on=st.booleans(),
        clip_angle_thresh=st.sampled_from([-1.0, 1.0]),
        **hu.gcs_cpu_only
    )
    @settings(deadline=1000)
    def test_bbox_transform_batch(
        self,
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

        def bbox_transform_ref(rois, deltas, im_info):
            box_out = []
            offset = 0
            for i, num_rois in enumerate(roi_counts):
                if num_rois == 0:
                    continue
                cur_boxes = rois[offset : offset + num_rois, 1:]
                cur_deltas = deltas[offset : offset + num_rois]
                im_shape = im_info[i, 0:2]
                if rotated:
                    cur_box_out = bbox_transform_rotated(
                        cur_boxes, cur_deltas, angle_bound_on=angle_bound_on
                    )
                    cur_box_out = clip_tiled_boxes_rotated(
                        cur_box_out, im_shape, angle_thresh=clip_angle_thresh
                    )
                else:
                    cur_box_out = bbox_transform(cur_boxes, cur_deltas)
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
            rotated=rotated,
            angle_bound_on=angle_bound_on,
            clip_angle_thresh=clip_angle_thresh,
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[rois, deltas, im_info],
            reference=bbox_transform_ref,
        )
