from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import numpy as np


class TestArgMaxOp(hu.HypothesisTestCase):
    def _test_op(
        self,
        op_name,
        original_inp,
        expected_values,
    ):
        op = core.CreateOperator(
            op_name,
            ['input'],
            ['output'],
        )
        workspace.FeedBlob('input', np.array(original_inp, dtype=np.float32))
        workspace.RunOperatorOnce(op)
        np.testing.assert_array_equal(
            workspace.FetchBlob('output'),
            np.array(expected_values),
        )

    def test_rowwise_argmax_op_with_large_input(self):
        X = [[1, 2, 3, 100, 3, 2, 1, 1.5, 1, 1, 1, 1, 1, 1.0],
            [1, 2, 3, 1, 3, 2, 1, 1.5, 1, 100, 1, 1, 1, 1.0],
            [1, 2, 3, 1, 3, 2, 1, 1.5, 1, 1, 100, 1, 1, 1.0],
            [1, 2, 3, 1, 3, 2, 1, 1.5, 1, 1, 1, 100, 1, 1.0],
            [1, 2, 3, 1, 3, 2, 1, 1.5, 1, 1, 1, 1, 100, 1.0],
            [100, 2, 3, 1, 3, 2, 1, 1.5, 1, 1, 1, 1, 1, 1.0],
            [1, 2, 3, 100, 3, 2, 1, 1.5, 1, 1, 1, 1, 1, 1.0],
            [1, 2, 3, 1, 3, 2, 100, 1.5, 1, 1, 1, 1, 1, 1.0]]

        self._test_op(
            op_name='RowWiseArgMax',
            original_inp=X,
            expected_values=[[3], [9], [10], [11], [12], [0], [3], [6]],
        )

    def test_rowwise_argmax_op_with_small_input(self):
        X = [[4.2, 6, 3.1],
             [10, 20, 40.4],
             [100.01, 25, 3]]

        self._test_op(
            op_name='RowWiseArgMax',
            original_inp=X,
            expected_values=[[1], [2], [0]],
        )

    def test_rowwise_argmax_with_duplicate_values(self):
        X = [[2, 2], [3, 3]]
        self._test_op(
            op_name='RowWiseArgMax',
            original_inp=X,
            expected_values=[[0], [0]],
        )

    def test_rowwise_argmax_with_1x1_tensor(self):
        X = [[1]]
        self._test_op(
            op_name='RowWiseArgMax',
            original_inp=X,
            expected_values=[[0]],
        )


if __name__ == "__main__":
    import unittest
    unittest.main()
