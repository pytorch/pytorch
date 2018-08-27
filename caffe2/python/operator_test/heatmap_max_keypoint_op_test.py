from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import unittest
from scipy import interpolate
import sys

import caffe2.python.hypothesis_test_util as hu
from caffe2.python import core, utils
from caffe2.proto import caffe2_pb2

import caffe2.python.operator_test.detectron_keypoints as keypoint_utils

NUM_TEST_ROI = 14
NUM_KEYPOINTS = 19
HEATMAP_SIZE = 56


def heatmap_FAIR_keypoint_ref(maps, rois):
    return [keypoint_utils.heatmaps_to_keypoints(maps, rois)]


def heatmap_approx_keypoint_ref(maps, rois):
    return [keypoint_utils.approx_heatmap_keypoint(maps, rois)]


class TestHeatmapMaxKeypointOp(hu.HypothesisTestCase):
    def setUp(self):
        np.random.seed(0)

        # initial coordinates and interpolate HEATMAP_SIZE from it
        HEATMAP_SMALL_SIZE = 4
        bboxes_in = 500 * np.random.rand(NUM_TEST_ROI, 4).astype(np.float32)
        # only bbox with smaller first coordiantes
        for i in range(NUM_TEST_ROI):
            if bboxes_in[i][0] > bboxes_in[i][2]:
                tmp = bboxes_in[i][2]
                bboxes_in[i][2] = bboxes_in[i][0]
                bboxes_in[i][0] = tmp
            if bboxes_in[i][1] > bboxes_in[i][3]:
                tmp = bboxes_in[i][3]
                bboxes_in[i][3] = bboxes_in[i][1]
                bboxes_in[i][1] = tmp

        # initial randomized coordiantes for heatmaps and expand it with interpolation
        init = np.random.rand(
            NUM_TEST_ROI,
            NUM_KEYPOINTS,
            HEATMAP_SMALL_SIZE,
            HEATMAP_SMALL_SIZE).astype(np.float32)
        heatmaps_in = np.zeros((NUM_TEST_ROI, NUM_KEYPOINTS,
            HEATMAP_SIZE, HEATMAP_SIZE)).astype(np.float32)
        for roi in range(NUM_TEST_ROI):
            for keyp in range(NUM_KEYPOINTS):
                f = interpolate.interp2d(
                    np.arange(0, 1, 1.0 / HEATMAP_SMALL_SIZE),
                    np.arange(0, 1, 1.0 / HEATMAP_SMALL_SIZE),
                    init[roi][keyp],
                    kind='cubic')
                heatmaps_in[roi][keyp] = f(
                    np.arange(0, 1, 1.0 / HEATMAP_SIZE),
                    np.arange(0, 1, 1.0 / HEATMAP_SIZE))

        self.heatmaps_in = heatmaps_in
        self.bboxes_in = bboxes_in

        self.op = core.CreateOperator(
            'HeatmapMaxKeypoint',
            ['heatmaps_in', 'bboxes_in'],
            ['keypoints_out'],
            arg=[
                utils.MakeArgument("should_output_softmax", True),
            ],
            device_option=caffe2_pb2.DeviceOption())

    @unittest.skipIf('cv2' not in sys.modules, 'python-opencv is not installed')
    def test_close_to_FAIR(self):
        # 10 pixel error in scale of 500px bbox
        self.assertReferenceChecks(
            device_option=caffe2_pb2.DeviceOption(),
            op=self.op,
            inputs=[self.heatmaps_in, self.bboxes_in],
            reference=heatmap_FAIR_keypoint_ref,
            threshold=10,
        )

    def test_approx_heatmap_keypoint(self):
        # C++/Python implementation should be bit-wise equal
        self.assertReferenceChecks(
            device_option=caffe2_pb2.DeviceOption(),
            op=self.op,
            inputs=[self.heatmaps_in, self.bboxes_in],
            reference=heatmap_approx_keypoint_ref,
        )

    def test_special_cases(self):
        example_bboxes = np.array([[0, 0, 100, 100]]).astype(np.float32)
        heatmap_tests = []
        # special case #1
        heatmap_tests.append(np.array([
            [0.14722, 0.807823, 0.447052],
            [0.652919, 0.850923, -0.225462],
            [0.805912, 0.75778, -0.563371],
        ]).astype(np.float32).reshape((1, 1, 3, 3)))
        # special case #2
        heatmap_tests.append(np.array([
            [3.19541, 3.69551, 3.87579],
            [3.63094, 3.89978, 3.67606],
            [3.78555, 3.87291, 3.28083],
        ]).astype(np.float32).reshape((1, 1, 3, 3)))

        for heatmap_test in heatmap_tests:
            self.assertReferenceChecks(
                device_option=caffe2_pb2.DeviceOption(),
                op=self.op,
                inputs=[heatmap_test, example_bboxes],
                reference=heatmap_approx_keypoint_ref,
            )


if __name__ == "__main__":
    unittest.main()
