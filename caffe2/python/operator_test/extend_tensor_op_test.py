from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace
from caffe2.python.test_util import TestCase
import numpy as np


class TestExtendTensorOp(TestCase):
    def test_extend_tensor(self):
        # Tensor of size 6 holding info about elements 0 to 5
        old_tensor = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
        workspace.FeedBlob('old_tensor', old_tensor)

        indices = np.array([0, 5, 8, 2, 3, 7], dtype=np.int32)
        workspace.FeedBlob('indices', indices)

        new_tensor_expected = np.array([1, 2, 3, 4, 5, 6, 0, 0, 0],
                                       dtype=np.int32)

        extend_tensor_op = core.CreateOperator(
            'ExtendTensor',
            ['old_tensor', 'indices'],
            ['old_tensor'])

        workspace.RunOperatorOnce(extend_tensor_op)
        new_tensor_observed = workspace.FetchBlob('old_tensor')

        np.testing.assert_array_equal(new_tensor_expected, new_tensor_observed)

    def test_counting(self):
        # Tensor of size 6 holding counts of elements with indices 0 to 5
        counts = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
        workspace.FeedBlob('counts', counts)

        # Indices of new words to be counted
        indices = np.array([0, 5, 8, 2, 3, 7, 7], dtype=np.int32)
        workspace.FeedBlob('indices', indices)

        # Extend the 'counts' tensor if necessary (if new words are seen)
        extend_tensor_op = core.CreateOperator(
            'ExtendTensor',
            ['counts', 'indices'],
            ['counts'])
        workspace.RunOperatorOnce(extend_tensor_op)

        ones_counts = np.array([1], dtype=np.float32)
        ones_indices = np.array(
            [1 for i in range(len(indices))], dtype=np.float32)
        one = np.array([1], dtype=np.float32)
        workspace.FeedBlob('ones_counts', ones_counts)
        workspace.FeedBlob('ones_indices', ones_indices)
        workspace.FeedBlob('one', one)

        ins = ['counts', 'ones_counts', 'indices', 'ones_indices', 'one']
        op = core.CreateOperator('ScatterWeightedSum', ins, ['counts'])
        workspace.RunOperatorOnce(op)

        new_tensor_expected = np.array([2, 2, 4, 5, 5, 7, 0, 2, 1],
                                       dtype=np.float32)
        new_tensor_observed = workspace.FetchBlob('counts')

        np.testing.assert_array_equal(new_tensor_expected, new_tensor_observed)

if __name__ == "__main__":
    import unittest
    unittest.main()
