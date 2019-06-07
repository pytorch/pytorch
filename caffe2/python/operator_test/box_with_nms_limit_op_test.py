from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
from hypothesis import given
import hypothesis.strategies as st
import unittest
import numpy as np


def get_op(input_len, output_len, args):
    input_names = ['in_scores', 'in_boxes', 'in_batch_splits']
    assert input_len <= len(input_names)
    input_names = input_names[:input_len]

    out_names = ['scores', 'boxes', 'classes', 'batch_splits', 'keeps', 'keeps_size']
    assert output_len <= len(out_names)
    out_names = out_names[:output_len]

    op = core.CreateOperator(
        'BoxWithNMSLimit',
        input_names,
        out_names,
        **args)

    return op


HU_CONFIG = {
    'gc': hu.gcs_cpu_only['gc'],
}


def gen_boxes(count, center):
    len = 10
    len_half = len / 2.0
    ret = np.tile(
        np.array(
            [center[0] - len_half, center[1] - len_half,
            center[0] + len_half, center[1] + len_half]
        ).astype(np.float32),
        (count, 1)
    )
    return ret


def gen_multiple_boxes(centers, scores, count, num_classes):
    ret_box = None
    ret_scores = None
    for cc, ss in zip(centers, scores):
        box = gen_boxes(count, cc)
        ret_box = np.vstack((ret_box, box)) if ret_box is not None else box
        cur_sc = np.ones((count, 1), dtype=np.float32) * ss
        ret_scores = np.vstack((ret_scores, cur_sc)) \
            if ret_scores is not None else cur_sc
    ret_box = np.tile(ret_box, (1, num_classes))
    ret_scores = np.tile(ret_scores, (1, num_classes))
    assert ret_box.shape == (len(centers) * count, 4 * num_classes)
    assert ret_scores.shape == (len(centers) * count, num_classes)
    return ret_box, ret_scores


class TestBoxWithNMSLimitOp(serial.SerializedTestCase):
    @serial.given(**HU_CONFIG)
    def test_simple(self, gc):
        in_centers = [(0, 0), (20, 20), (50, 50)]
        in_scores = [0.9, 0.8, 0.6]
        boxes, scores = gen_multiple_boxes(in_centers, in_scores, 10, 2)

        gt_boxes, gt_scores = gen_multiple_boxes(in_centers, in_scores, 1, 1)
        gt_classes = np.ones(gt_boxes.shape[0], dtype=np.float32)

        op = get_op(2, 3, {"score_thresh": 0.5, "nms": 0.9})

        def ref(*args, **kwargs):
            return (gt_scores.flatten(), gt_boxes, gt_classes)

        self.assertReferenceChecks(gc, op, [scores, boxes], ref)

    @given(**HU_CONFIG)
    def test_score_thresh(self, gc):
        in_centers = [(0, 0), (20, 20), (50, 50)]
        in_scores = [0.7, 0.85, 0.6]
        boxes, scores = gen_multiple_boxes(in_centers, in_scores, 10, 2)

        gt_centers = [(20, 20)]
        gt_scores = [0.85]
        gt_boxes, gt_scores = gen_multiple_boxes(gt_centers, gt_scores, 1, 1)
        gt_classes = np.ones(gt_boxes.shape[0], dtype=np.float32)

        op = get_op(2, 3, {"score_thresh": 0.8, "nms": 0.9})

        def ref(*args, **kwargs):
            return (gt_scores.flatten(), gt_boxes, gt_classes)

        self.assertReferenceChecks(gc, op, [scores, boxes], ref)

    @given(det_per_im=st.integers(1, 3), **HU_CONFIG)
    def test_detections_per_im(self, det_per_im, gc):
        in_centers = [(0, 0), (20, 20), (50, 50)]
        in_scores = [0.7, 0.85, 0.6]
        boxes, scores = gen_multiple_boxes(in_centers, in_scores, 10, 2)

        gt_centers = [(20, 20), (0, 0), (50, 50)][:det_per_im]
        gt_scores = [0.85, 0.7, 0.6][:det_per_im]
        gt_boxes, gt_scores = gen_multiple_boxes(gt_centers, gt_scores, 1, 1)
        gt_classes = np.ones(gt_boxes.shape[0], dtype=np.float32)

        op = get_op(
            2, 3,
            {"score_thresh": 0.5, "nms": 0.9, "detections_per_im": det_per_im}
        )

        def ref(*args, **kwargs):
            return (gt_scores.flatten(), gt_boxes, gt_classes)

        self.assertReferenceChecks(gc, op, [scores, boxes], ref)

    @given(
        num_classes=st.integers(2, 10),
        det_per_im=st.integers(1, 4),
        cls_agnostic_bbox_reg=st.booleans(),
        input_boxes_include_bg_cls=st.booleans(),
        output_classes_include_bg_cls=st.booleans(),
        **HU_CONFIG
    )
    def test_multiclass(
        self,
        num_classes,
        det_per_im,
        cls_agnostic_bbox_reg,
        input_boxes_include_bg_cls,
        output_classes_include_bg_cls,
        gc
    ):
        in_centers = [(0, 0), (20, 20), (50, 50)]
        in_scores = [0.7, 0.85, 0.6]
        boxes, scores = gen_multiple_boxes(in_centers, in_scores, 10, num_classes)

        if not input_boxes_include_bg_cls:
            # remove backgound class
            boxes = boxes[:, 4:]
        if cls_agnostic_bbox_reg:
            # only leave one class
            boxes = boxes[:, :4]
        # randomize un-used scores for background class
        scores_bg_class_id = 0 if input_boxes_include_bg_cls else -1
        scores[:, scores_bg_class_id] = np.random.rand(scores.shape[0]).astype(np.float32)

        gt_centers = [(20, 20), (0, 0), (50, 50)][:det_per_im]
        gt_scores = [0.85, 0.7, 0.6][:det_per_im]
        gt_boxes, gt_scores = gen_multiple_boxes(gt_centers, gt_scores, 1, 1)
        # [1, 1, 1, 2, 2, 2, 3, 3, 3, ...]
        gt_classes = np.tile(
            np.array(range(1, num_classes), dtype=np.float32),
            (gt_boxes.shape[0], 1)).T.flatten()
        if not output_classes_include_bg_cls:
            # remove backgound class
            gt_classes -= 1
        gt_boxes = np.tile(gt_boxes, (num_classes - 1, 1))
        gt_scores = np.tile(gt_scores, (num_classes - 1, 1)).flatten()

        op = get_op(
            2, 3,
            {
                "score_thresh": 0.5,
                "nms": 0.9,
                "detections_per_im": (num_classes - 1) * det_per_im,
                "cls_agnostic_bbox_reg": cls_agnostic_bbox_reg,
                "input_boxes_include_bg_cls": input_boxes_include_bg_cls,
                "output_classes_include_bg_cls": output_classes_include_bg_cls
            }
        )

        def ref(*args, **kwargs):
            return (gt_scores, gt_boxes, gt_classes)

        self.assertReferenceChecks(gc, op, [scores, boxes], ref)

    @given(det_per_im=st.integers(1, 3), **HU_CONFIG)
    def test_detections_per_im_same_thresh(self, det_per_im, gc):
        in_centers = [(0, 0), (20, 20), (50, 50)]
        in_scores = [0.7, 0.7, 0.7]
        boxes, scores = gen_multiple_boxes(in_centers, in_scores, 10, 2)

        gt_centers = [(20, 20), (0, 0), (50, 50)][:det_per_im]
        gt_scores = [0.7, 0.7, 0.7][:det_per_im]
        gt_boxes, gt_scores = gen_multiple_boxes(gt_centers, gt_scores, 1, 1)
        gt_classes = np.ones(gt_boxes.shape[0], dtype=np.float32)

        op = get_op(
            2, 3,
            {"score_thresh": 0.5, "nms": 0.9, "detections_per_im": det_per_im}
        )

        # boxes output could be in any order
        def verify(inputs, outputs):
            # check scores
            np.testing.assert_allclose(
                outputs[0], gt_scores.flatten(), atol=1e-4, rtol=1e-4,
            )
            # check classes
            np.testing.assert_allclose(
                outputs[2], gt_classes, atol=1e-4, rtol=1e-4,
            )
            self.assertEqual(outputs[1].shape, gt_boxes.shape)

        self.assertValidationChecks(gc, op, [scores, boxes], verify, as_kwargs=False)

    @given(num_classes=st.integers(2, 10), **HU_CONFIG)
    def test_detections_per_im_same_thresh_multiclass(self, num_classes, gc):
        in_centers = [(0, 0), (20, 20), (50, 50)]
        in_scores = [0.6, 0.7, 0.7]
        boxes, scores = gen_multiple_boxes(in_centers, in_scores, 10, num_classes)

        det_per_im = 1
        gt_centers = [(20, 20), (50, 50)]
        gt_scores = [0.7, 0.7]
        gt_boxes, gt_scores = gen_multiple_boxes(gt_centers, gt_scores, 1, 1)

        op = get_op(
            2, 3,
            {"score_thresh": 0.5, "nms": 0.9, "detections_per_im": det_per_im}
        )

        # boxes output could be in any order
        def verify(inputs, outputs):
            # check scores
            self.assertEqual(outputs[0].shape, (1,))
            self.assertEqual(outputs[0][0], gt_scores[0])

            # check boxes
            self.assertTrue(
                np.allclose(outputs[1], gt_boxes[0, :], atol=1e-4, rtol=1e-4) or
                np.allclose(outputs[1], gt_boxes[1, :], atol=1e-4, rtol=1e-4)
            )

            # check class
            self.assertNotEqual(outputs[2][0], 0)

        self.assertValidationChecks(gc, op, [scores, boxes], verify, as_kwargs=False)


if __name__ == "__main__":
    unittest.main()
