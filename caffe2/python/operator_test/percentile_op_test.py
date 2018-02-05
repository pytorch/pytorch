from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace, dyndep
import caffe2.python.hypothesis_test_util as hu
import numpy as np

class TestPercentileOp(hu.HypothesisTestCase):
    def _test_percentile_op(
        self,
        original_inp,
        value_to_pct_map,
        dist_lengths,
        expected_values
    ):
        op = core.CreateOperator(
            'Percentile',
            ['original_values', 'value_to_pct_map', 'dist_lengths'],
            ['percentile_values']
        )
        workspace.FeedBlob('original_values', np.array(original_inp, dtype=np.float32))
        workspace.FeedBlob(
            'value_to_pct_map', np.array(value_to_pct_map, dtype=np.float32))
        workspace.FeedBlob('dist_lengths', np.array(dist_lengths, dtype=np.int32))
        workspace.RunOperatorOnce(op)
        np.testing.assert_array_almost_equal(
            workspace.FetchBlob('percentile_values'),
            np.array(expected_values),
            decimal=5
        )

    def test_percentile_op_with_only_one_dist(self):
        self._test_percentile_op(
            original_inp=[[5]],
            value_to_pct_map=[[5, 0.4]],
            dist_lengths=[1],
            expected_values=[[0.4]]
        )

    def test_percentile_op_with_all_elements_in_map(self):
        self._test_percentile_op(
            original_inp=[[3, 4], [10, 4]],
            value_to_pct_map=[[3, 0.3], [4, 0.6], [10, 0.8], [4, 0.5], [5, 0.6]],
            dist_lengths=[3, 2],
            expected_values=[[0.3, 0.5], [0.8, 0.5]],
        )

    def test_percentile_op_with_same_value(self):
        self._test_percentile_op(
            original_inp=[[1, 1], [1, 2]],
            value_to_pct_map=[[1, 0.1], [4, 0.4], [2, 0.5]],
            dist_lengths=[2, 1],
            expected_values=[[0.1, 0.0], [0.1, 0.5]]
        )

    def test_percentile_op_with_elements_bigger_than_map_range(self):
        self._test_percentile_op(
            original_inp=[[1, 5], [3, 4]],
            value_to_pct_map=[[1, 0.1], [4, 0.4], [2, 0.1], [3, 0.3]],
            dist_lengths=[2, 2],
            expected_values=[[0.1, 1.], [0.3, 1.0]]
        )

    def test_percentile_op_with_elements_smaller_than_map_range(self):
        self._test_percentile_op(
            original_inp=[[1], [5], [6]],
            value_to_pct_map=[[2, 0.2], [5, 0.5], [7, 0.5]],
            dist_lengths=[3],
            expected_values=[[0.0], [0.5], [0.5]]
        )

    def test_percentile_op_with_interpolation(self):
        self._test_percentile_op(
            original_inp=[[3, 2, 5], [6, 7, 8]],
            value_to_pct_map=[[1, 0.1], [4, 0.7], [4.5, 0.8],
                              [6, 0.5], [8, 0.9],
                              [8, 0.6]],
            dist_lengths=[3, 2, 1],
            expected_values=[[0.5, 0.0, 0.0], [1.0, 0.7, 0.6]]
        )

    def test_percentile_op_with_large_sample_size_per_dist(self):
        self._test_percentile_op(
            original_inp=[[3, 1], [5, 7]],
            value_to_pct_map=[[3, 0.5], [4, 0.6], [5, 0.7],
                              [1, 0.2], [2, 0.3], [5, 0.8]],
            dist_lengths=[3, 3],
            expected_values=[[0.5, 0.2], [0.7, 1.0]]
        )


if __name__ == "__main__":
    import unittest
    unittest.main()
