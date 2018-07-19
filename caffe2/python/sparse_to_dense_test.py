from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core, workspace
from caffe2.python.test_util import TestCase

import numpy as np


class TestSparseToDense(TestCase):
    def test_sparse_to_dense(self):
        op = core.CreateOperator(
            'SparseToDense',
            ['indices', 'values'],
            ['output'])
        workspace.FeedBlob(
            'indices',
            np.array([2, 4, 999, 2], dtype=np.int32))
        workspace.FeedBlob(
            'values',
            np.array([1, 2, 6, 7], dtype=np.int32))

        workspace.RunOperatorOnce(op)
        output = workspace.FetchBlob('output')
        print(output)

        expected = np.zeros(1000, dtype=np.int32)
        expected[2] = 1 + 7
        expected[4] = 2
        expected[999] = 6

        self.assertEqual(output.shape, expected.shape)
        np.testing.assert_array_equal(output, expected)

    def test_sparse_to_dense_invalid_inputs(self):
        op = core.CreateOperator(
            'SparseToDense',
            ['indices', 'values'],
            ['output'])
        workspace.FeedBlob(
            'indices',
            np.array([2, 4, 999, 2], dtype=np.int32))
        workspace.FeedBlob(
            'values',
            np.array([1, 2, 6], dtype=np.int32))

        with self.assertRaises(RuntimeError):
            workspace.RunOperatorOnce(op)

    def test_sparse_to_dense_with_data_to_infer_dim(self):
        op = core.CreateOperator(
            'SparseToDense',
            ['indices', 'values', 'data_to_infer_dim'],
            ['output'])
        workspace.FeedBlob(
            'indices',
            np.array([2, 4, 999, 2], dtype=np.int32))
        workspace.FeedBlob(
            'values',
            np.array([1, 2, 6, 7], dtype=np.int32))
        workspace.FeedBlob(
            'data_to_infer_dim',
            np.array(np.zeros(1500, ), dtype=np.int32))

        workspace.RunOperatorOnce(op)
        output = workspace.FetchBlob('output')
        print(output)

        expected = np.zeros(1500, dtype=np.int32)
        expected[2] = 1 + 7
        expected[4] = 2
        expected[999] = 6

        self.assertEqual(output.shape, expected.shape)
        np.testing.assert_array_equal(output, expected)
