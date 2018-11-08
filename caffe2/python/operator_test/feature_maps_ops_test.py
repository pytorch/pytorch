from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core, workspace, dyndep
from caffe2.python.test_util import TestCase
import numpy as np


class TestFeatureMapsOps(TestCase):

    def test_merge_single_scalar_feature_tensors(self):
        op = core.CreateOperator(
            "MergeSingleScalarFeatureTensors",
            [
                "in1", "in1_presence",
                "in2", "in2_presence",
            ],
            [
                "out_lengths", "out_keys", "out_values",
            ],
            feature_ids=[11, 12]
        )

        # Input 1.
        workspace.FeedBlob(
            "in1",
            np.array([11.1, 0.0], dtype=np.float)
        )
        workspace.FeedBlob(
            "in1_presence",
            np.array([True, False], dtype=np.bool)
        )
        # Input 2.
        workspace.FeedBlob(
            "in2",
            np.array([12.1, 12.2], dtype=np.float)
        )
        workspace.FeedBlob(
            "in2_presence",
            np.array([True, True], dtype=np.bool)
        )

        workspace.RunOperatorOnce(op)

        np.testing.assert_array_equal(
            workspace.FetchBlob("out_lengths"),
            np.array([2, 1], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            workspace.FetchBlob("out_keys"),
            np.array([11, 12, 12], dtype=np.int64)
        )
        np.testing.assert_array_equal(
            workspace.FetchBlob("out_values"),
            np.array([11.1, 12.1, 12.2], dtype=np.float)
        )

    def test_merge_single_scalar_feature_tensors_gradient(self):
        op = core.CreateOperator(
            "MergeSingleScalarFeatureTensorsGradient",
            [
                "in1_presence",
                "in2_presence",
                "in3_presence",
                "out_values_grad",
            ],
            [
                "in1_grad", "in2_grad", "in3_grad",
            ],
        )

        # Inputs 1, 2 & 3.
        workspace.FeedBlob(
            "in1_presence",
            np.array([True, False], dtype=np.bool)
        )
        workspace.FeedBlob(
            "in2_presence",
            np.array([True, True], dtype=np.bool)
        )
        workspace.FeedBlob(
            "in3_presence",
            np.array([False, True], dtype=np.bool)
        )
        # Input 4.
        workspace.FeedBlob(
            "out_values_grad",
            np.array([0.1, 1.1, 1.2, 2.3], dtype=np.float)
        )

        workspace.RunOperatorOnce(op)

        np.testing.assert_array_equal(
            workspace.FetchBlob("in1_grad"),
            np.array([0.1, 0], dtype=np.float)
        )
        np.testing.assert_array_equal(
            workspace.FetchBlob("in2_grad"),
            np.array([1.1, 1.2], dtype=np.float)
        )
        np.testing.assert_array_equal(
            workspace.FetchBlob("in3_grad"),
            np.array([0, 2.3], dtype=np.float)
        )

    def test_merge_single_scalar_feature_tensors_gradient_with_strings(self):
        op = core.CreateOperator(
            "MergeSingleScalarFeatureTensorsGradient",
            [
                "in1_presence",
                "in2_presence",
                "in3_presence",
                "out_values_grad",
            ],
            [
                "in1_grad", "in2_grad", "in3_grad",
            ],
        )

        # Inputs 1, 2 & 3.
        workspace.FeedBlob(
            "in1_presence",
            np.array([True, False], dtype=np.bool)
        )
        workspace.FeedBlob(
            "in2_presence",
            np.array([True, True], dtype=np.bool)
        )
        workspace.FeedBlob(
            "in3_presence",
            np.array([False, True], dtype=np.bool)
        )
        # Input 4.
        workspace.FeedBlob(
            "out_values_grad",
            np.array(["0.1", "1.1", "1.2", "2.3"], dtype=np.unicode_)
        )

        workspace.RunOperatorOnce(op)

        np.testing.assert_array_equal(
            workspace.FetchBlob("in1_grad"),
            np.array(["0.1", ""], dtype=np.bytes_)
        )
        np.testing.assert_array_equal(
            workspace.FetchBlob("in2_grad"),
            np.array(["1.1", "1.2"], dtype=np.bytes_)
        )
        np.testing.assert_array_equal(
            workspace.FetchBlob("in3_grad"),
            np.array(["", "2.3"], dtype=np.bytes_)
        )

    def test_merge_single_list_feature_tensors(self):
        op = core.CreateOperator(
            "MergeSingleListFeatureTensors",
            [
                "in1_lengths", "in1_values", "in1_presence",
                "in2_lengths", "in2_values", "in2_presence",
            ],
            [
                "out_lengths", "out_keys", "out_values_lengths",
                "out_values_values",
            ],
            feature_ids=[11, 12]
        )

        # Input 1.
        workspace.FeedBlob(
            "in1_lengths",
            np.array([2, 0], dtype=np.int32)
        )
        workspace.FeedBlob(
            "in1_values",
            np.array([11.1, 11.2], dtype=np.float)
        )
        workspace.FeedBlob(
            "in1_presence",
            np.array([True, False], dtype=np.bool)
        )
        # Input 2.
        workspace.FeedBlob(
            "in2_lengths",
            np.array([2, 2], dtype=np.int32)
        )
        workspace.FeedBlob(
            "in2_values",
            np.array([12.1, 12.2, 12.3, 12.4], dtype=np.float)
        )
        workspace.FeedBlob(
            "in2_presence",
            np.array([True, True], dtype=np.bool)
        )

        workspace.RunOperatorOnce(op)

        np.testing.assert_array_equal(
            workspace.FetchBlob("out_lengths"),
            np.array([2, 1], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            workspace.FetchBlob("out_keys"),
            np.array([11, 12, 12], dtype=np.int64)
        )
        np.testing.assert_array_equal(
            workspace.FetchBlob("out_values_lengths"),
            np.array([2, 2, 2], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            workspace.FetchBlob("out_values_values"),
            np.array([11.1, 11.2, 12.1, 12.2, 12.3, 12.4], dtype=np.float)
        )

    def test_merge_single_list_feature_tensors_gradient(self):
        self._test_merge_single_list_or_map_feature_tensors_gradient(
            "MergeSingleListFeatureTensorsGradient"
        )

    def test_merge_single_map_feature_tensors_gradient(self):
        self._test_merge_single_list_or_map_feature_tensors_gradient(
            "MergeSingleMapFeatureTensorsGradient"
        )

    def _test_merge_single_list_or_map_feature_tensors_gradient(self, op_name):
        op = core.CreateOperator(
            op_name,
            [
                "in1_lengths", "in1_presence",
                "in2_lengths", "in2_presence",
                "out_values_values_grad",
            ],
            [
                "in1_values_grad",
                "in2_values_grad",
            ],
        )

        # Input 1.
        workspace.FeedBlob(
            "in1_lengths",
            np.array([2, 0], dtype=np.int32)
        )
        workspace.FeedBlob(
            "in1_presence",
            np.array([True, False], dtype=np.bool)
        )
        # Input 2.
        workspace.FeedBlob(
            "in2_lengths",
            np.array([2, 2], dtype=np.int32)
        )
        workspace.FeedBlob(
            "in2_presence",
            np.array([True, True], dtype=np.bool)
        )
        workspace.FeedBlob(
            "out_values_values_grad",
            np.array([11.1, 11.2, 12.1, 12.2, 12.3, 12.4], dtype=np.float)
        )

        workspace.RunOperatorOnce(op)

        np.testing.assert_array_equal(
            workspace.FetchBlob("in1_values_grad"),
            np.array([11.1, 11.2], dtype=np.float)
        )
        np.testing.assert_array_equal(
            workspace.FetchBlob("in2_values_grad"),
            np.array([12.1, 12.2, 12.3, 12.4], dtype=np.float)
        )

    def test_merge_single_map_feature_tensors(self):
        op = core.CreateOperator(
            "MergeSingleMapFeatureTensors",
            [
                "in1_lengths", "in1_keys", "in1_values", "in1_presence",
                "in2_lengths", "in2_keys", "in2_values", "in2_presence",
            ],
            [
                "out_lengths", "out_keys", "out_values_lengths",
                "out_values_keys", "out_values_values",
            ],
            feature_ids=[11, 12]
        )

        # Input 1.
        workspace.FeedBlob(
            "in1_lengths",
            np.array([2, 0], dtype=np.int32)
        )
        workspace.FeedBlob(
            "in1_keys",
            np.array([111, 112], dtype=np.int64)
        )
        workspace.FeedBlob(
            "in1_values",
            np.array([11.1, 11.2], dtype=np.float)
        )
        workspace.FeedBlob(
            "in1_presence",
            np.array([True, False], dtype=np.bool)
        )
        # Input 2.
        workspace.FeedBlob(
            "in2_lengths",
            np.array([2, 2], dtype=np.int32)
        )
        workspace.FeedBlob(
            "in2_keys",
            np.array([121, 122, 123, 124], dtype=np.int64)
        )
        workspace.FeedBlob(
            "in2_values",
            np.array([12.1, 12.2, 12.3, 12.4], dtype=np.float)
        )
        workspace.FeedBlob(
            "in2_presence",
            np.array([True, True], dtype=np.bool)
        )

        workspace.RunOperatorOnce(op)

        np.testing.assert_array_equal(
            workspace.FetchBlob("out_lengths"),
            np.array([2, 1], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            workspace.FetchBlob("out_keys"),
            np.array([11, 12, 12], dtype=np.int64)
        )
        np.testing.assert_array_equal(
            workspace.FetchBlob("out_values_lengths"),
            np.array([2, 2, 2], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            workspace.FetchBlob("out_values_keys"),
            np.array([111, 112, 121, 122, 123, 124], dtype=np.int64)
        )
        np.testing.assert_array_equal(
            workspace.FetchBlob("out_values_values"),
            np.array([11.1, 11.2, 12.1, 12.2, 12.3, 12.4], dtype=np.float)
        )

    def test_merge_multi_scalar_feature_tensors(self):
        op = core.CreateOperator(
            "MergeMultiScalarFeatureTensors",
            [
                "in1_lengths", "in1_keys", "in1_values",
                "in2_lengths", "in2_keys", "in2_values",
            ],
            [
                "out_lengths", "out_keys", "out_values",
            ]
        )

        # Input 1.
        workspace.FeedBlob(
            "in1_lengths",
            np.array([1, 2], dtype=np.int32)
        )
        workspace.FeedBlob(
            "in1_keys",
            np.array([11, 12, 13], dtype=np.int64)
        )
        workspace.FeedBlob(
            "in1_values",
            np.array([11.0, 12.0, 13.0], dtype=np.float)
        )
        # Input 2.
        workspace.FeedBlob(
            "in2_lengths",
            np.array([2, 1], dtype=np.int32)
        )
        workspace.FeedBlob(
            "in2_keys",
            np.array([14, 15, 16], dtype=np.int64)
        )
        workspace.FeedBlob(
            "in2_values",
            np.array([14.0, 15.0, 16.0], dtype=np.float)
        )

        workspace.RunOperatorOnce(op)

        np.testing.assert_array_equal(
            workspace.FetchBlob("out_lengths"),
            np.array([3, 3], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            workspace.FetchBlob("out_keys"),
            np.array([11, 14, 15, 12, 13, 16], dtype=np.int64)
        )
        np.testing.assert_array_equal(
            workspace.FetchBlob("out_values"),
            np.array([11.0, 14.0, 15.0, 12.0, 13.0, 16.0], dtype=np.float)
        )

    def test_merge_multi_scalar_feature_tensors_gradient(self):
        op = core.CreateOperator(
            "MergeMultiScalarFeatureTensorsGradient",
            [
                "in1_lengths",
                "in2_lengths",
                "out_values_grad"
            ],
            [
                "in1_values_grad",
                "in2_values_grad",
            ]
        )

        # Input 1.
        workspace.FeedBlob(
            "in1_lengths",
            np.array([1, 2, 0], dtype=np.int32)
        )
        # Input 2.
        workspace.FeedBlob(
            "in2_lengths",
            np.array([2, 1, 1], dtype=np.int32)
        )
        # Grad input.
        workspace.FeedBlob(
            "out_values_grad",
            np.array([11.0, 14.0, 15.0, 12.0, 13.0, 16.0, 17.0], dtype=np.float)
        )

        workspace.RunOperatorOnce(op)

        np.testing.assert_array_equal(
            workspace.FetchBlob("in1_values_grad"),
            np.array([11.0, 12.0, 13.0], dtype=np.float)
        )
        np.testing.assert_array_equal(
            workspace.FetchBlob("in2_values_grad"),
            np.array([14.0, 15.0, 16.0, 17.0], dtype=np.float)
        )

    def test_merge_multi_list_feature_tensors(self):
        op = core.CreateOperator(
            "MergeMultiListFeatureTensors",
            [
                "in1_lengths", "in1_keys", "in1_values_lengths",
                "in1_values_values",
                "in2_lengths", "in2_keys", "in2_values_lengths",
                "in2_values_values",
            ],
            [
                "out_lengths", "out_keys", "out_values_lengths",
                "out_values_values"
            ]
        )

        # Input 1.
        workspace.FeedBlob(
            "in1_lengths",
            np.array([1, 2], dtype=np.int32)
        )
        workspace.FeedBlob(
            "in1_keys",
            np.array([11, 12, 13], dtype=np.int64)
        )
        workspace.FeedBlob(
            "in1_values_lengths",
            np.array([2, 2, 2], dtype=np.int32)
        )
        workspace.FeedBlob(
            "in1_values_values",
            np.array([11.1, 11.2, 12.1, 12.2, 13.1, 13.2], dtype=np.float)
        )
        # Input 2.
        workspace.FeedBlob(
            "in2_lengths",
            np.array([2, 1], dtype=np.int32)
        )
        workspace.FeedBlob(
            "in2_keys",
            np.array([14, 15, 16], dtype=np.int64)
        )
        workspace.FeedBlob(
            "in2_values_lengths",
            np.array([2, 2, 2], dtype=np.int32)
        )
        workspace.FeedBlob(
            "in2_values_values",
            np.array([14.1, 14.2, 15.1, 15.2, 16.1, 16.2], dtype=np.float)
        )

        workspace.RunOperatorOnce(op)

        np.testing.assert_array_equal(
            workspace.FetchBlob("out_lengths"),
            np.array([3, 3], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            workspace.FetchBlob("out_keys"),
            np.array([11, 14, 15, 12, 13, 16], dtype=np.int64)
        )
        np.testing.assert_array_equal(
            workspace.FetchBlob("out_values_lengths"),
            np.array([2, 2, 2, 2, 2, 2], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            workspace.FetchBlob("out_values_values"),
            np.array(
                [
                    11.1, 11.2, 14.1, 14.2, 15.1, 15.2, 12.1, 12.2, 13.1, 13.2,
                    16.1, 16.2
                ],
                dtype=np.float
            )
        )

    def test_merge_multi_map_feature_tensors(self):
        op = core.CreateOperator(
            "MergeMultiMapFeatureTensors",
            [
                "in1_lengths", "in1_keys", "in1_values_lengths",
                "in1_values_keys", "in1_values_values",
                "in2_lengths", "in2_keys", "in2_values_lengths",
                "in2_values_keys", "in2_values_values",
            ],
            [
                "out_lengths", "out_keys", "out_values_lengths",
                "out_values_keys", "out_values_values"
            ]
        )

        # Input 1.
        workspace.FeedBlob(
            "in1_lengths",
            np.array([1, 2], dtype=np.int32)
        )
        workspace.FeedBlob(
            "in1_keys",
            np.array([11, 12, 13], dtype=np.int64)
        )
        workspace.FeedBlob(
            "in1_values_lengths",
            np.array([2, 2, 2], dtype=np.int32)
        )
        workspace.FeedBlob(
            "in1_values_keys",
            np.array([111, 112, 121, 122, 131, 132], dtype=np.int64)
        )
        workspace.FeedBlob(
            "in1_values_values",
            np.array([11.1, 11.2, 12.1, 12.2, 13.1, 13.2], dtype=np.float)
        )
        # Input 2.
        workspace.FeedBlob(
            "in2_lengths",
            np.array([2, 1], dtype=np.int32)
        )
        workspace.FeedBlob(
            "in2_keys",
            np.array([14, 15, 16], dtype=np.int64)
        )
        workspace.FeedBlob(
            "in2_values_lengths",
            np.array([2, 2, 2], dtype=np.int32)
        )
        workspace.FeedBlob(
            "in2_values_keys",
            np.array([141, 142, 151, 152, 161, 162], dtype=np.int64)
        )
        workspace.FeedBlob(
            "in2_values_values",
            np.array([14.1, 14.2, 15.1, 15.2, 16.1, 16.2], dtype=np.float)
        )

        workspace.RunOperatorOnce(op)

        np.testing.assert_array_equal(
            workspace.FetchBlob("out_lengths"),
            np.array([3, 3], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            workspace.FetchBlob("out_keys"),
            np.array([11, 14, 15, 12, 13, 16], dtype=np.int64)
        )
        np.testing.assert_array_equal(
            workspace.FetchBlob("out_values_lengths"),
            np.array([2, 2, 2, 2, 2, 2], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            workspace.FetchBlob("out_values_keys"),
            np.array(
                [111, 112, 141, 142, 151, 152, 121, 122, 131, 132, 161, 162],
                dtype=np.int64
            )
        )
        np.testing.assert_array_equal(
            workspace.FetchBlob("out_values_values"),
            np.array(
                [
                    11.1, 11.2, 14.1, 14.2, 15.1, 15.2, 12.1, 12.2, 13.1, 13.2,
                    16.1, 16.2
                ],
                dtype=np.float
            )
        )

    def test_merge_multi_list_feature_tensors_gradient(self):
        self._test_merge_multi_list_or_map_feature_tensors_gradient(
            "MergeMultiListFeatureTensorsGradient"
        )

    def test_merge_multi_map_feature_tensors_gradient(self):
        self._test_merge_multi_list_or_map_feature_tensors_gradient(
            "MergeMultiMapFeatureTensorsGradient"
        )

    def _test_merge_multi_list_or_map_feature_tensors_gradient(self, op_name):
        op = core.CreateOperator(
            op_name,
            [
                "in1_lengths", "in1_values_lengths",
                "in2_lengths", "in2_values_lengths",
                "out_values_values_grad"
            ],
            [
                "in1_values_values_grad",
                "in2_values_values_grad",
            ]
        )

        # Input 1.
        workspace.FeedBlob(
            "in1_lengths",
            np.array([1, 2], dtype=np.int32)
        )
        workspace.FeedBlob(
            "in1_values_lengths",
            np.array([2, 2, 2], dtype=np.int32)
        )
        # Input 2.
        workspace.FeedBlob(
            "in2_lengths",
            np.array([2, 1], dtype=np.int32)
        )
        workspace.FeedBlob(
            "in2_values_lengths",
            np.array([2, 2, 2], dtype=np.int32)
        )
        # Grad Input.
        workspace.FeedBlob(
            "out_values_values_grad",
            np.array(
                [
                    11.1, 11.2, 14.1, 14.2, 15.1, 15.2, 12.1, 12.2, 13.1, 13.2,
                    16.1, 16.2
                ],
                dtype=np.float
            )
        )

        workspace.RunOperatorOnce(op)

        np.testing.assert_array_equal(
            workspace.FetchBlob("in1_values_values_grad"),
            np.array([11.1, 11.2, 12.1, 12.2, 13.1, 13.2], dtype=np.float)
        )
        np.testing.assert_array_equal(
            workspace.FetchBlob("in2_values_values_grad"),
            np.array([14.1, 14.2, 15.1, 15.2, 16.1, 16.2], dtype=np.float)
        )
