from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core, workspace
from caffe2.python.test_util import TestCase

import numpy as np


class TestSparseToDenseMask(TestCase):

    def test_sparse_to_dense_mask_float(self):
        op = core.CreateOperator(
            'SparseToDenseMask',
            ['indices', 'values', 'default', 'lengths'],
            ['output'],
            mask=[999999999, 2, 6])
        workspace.FeedBlob(
            'indices',
            np.array([2, 4, 6, 1, 2, 999999999, 2], dtype=np.int32))
        workspace.FeedBlob(
            'values',
            np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float))
        workspace.FeedBlob('default', np.array(-1, dtype=np.float))
        workspace.FeedBlob('lengths', np.array([3, 4], dtype=np.int32))
        workspace.RunOperatorOnce(op)
        output = workspace.FetchBlob('output')
        expected = np.array([[-1, 1, 3], [6, 7, -1]], dtype=np.float)
        self.assertEqual(output.shape, expected.shape)
        np.testing.assert_array_equal(output, expected)

    def test_sparse_to_dense_mask_invalid_inputs(self):
        op = core.CreateOperator(
            'SparseToDenseMask',
            ['indices', 'values', 'default', 'lengths'],
            ['output'],
            mask=[999999999, 2],
            max_skipped_indices=3)
        workspace.FeedBlob(
            'indices',
            np.array([2000000000000, 999999999, 2, 3, 4, 5], dtype=np.int32))
        workspace.FeedBlob(
            'values',
            np.array([1, 2, 3, 4, 5, 6], dtype=np.float))
        workspace.FeedBlob('default', np.array(-1, dtype=np.float))
        workspace.FeedBlob('lengths', np.array([6], dtype=np.int32))
        try:
            workspace.RunOperatorOnce(op)
        except RuntimeError:
            self.fail("Exception raised with only one negative index")

        # 3 invalid inputs should throw.
        workspace.FeedBlob(
            'indices',
            np.array([-1, 1, 2, 3, 4, 5], dtype=np.int32))
        with self.assertRaises(RuntimeError):
            workspace.RunOperatorMultiple(op, 3)

    def test_sparse_to_dense_mask_subtensor(self):
        op = core.CreateOperator(
            'SparseToDenseMask',
            ['indices', 'values', 'default', 'lengths'],
            ['output'],
            mask=[999999999, 2, 888, 6])
        workspace.FeedBlob(
            'indices',
            np.array([2, 4, 6, 999999999, 2], dtype=np.int64))
        workspace.FeedBlob(
            'values',
            np.array([[[1, -1]], [[2, -2]], [[3, -3]], [[4, -4]], [[5, -5]]],
                     dtype=np.float))
        workspace.FeedBlob('default', np.array([[-1, 0]], dtype=np.float))
        workspace.FeedBlob('lengths', np.array([2, 3], dtype=np.int32))
        workspace.RunOperatorOnce(op)
        output = workspace.FetchBlob('output')
        expected = np.array([
            [[[-1, 0]], [[1, -1]], [[-1, 0]], [[-1, 0]]],
            [[[4, -4]], [[5, -5]], [[-1, 0]], [[3, -3]]]], dtype=np.float)
        self.assertEqual(output.shape, expected.shape)
        np.testing.assert_array_equal(output, expected)

    def test_sparse_to_dense_mask_string(self):
        op = core.CreateOperator(
            'SparseToDenseMask',
            ['indices', 'values', 'default', 'lengths'],
            ['output'],
            mask=[999999999, 2, 6])
        workspace.FeedBlob(
            'indices',
            np.array([2, 4, 6, 1, 2, 999999999, 2], dtype=np.int32))
        workspace.FeedBlob(
            'values',
            np.array(['1', '2', '3', '4', '5', '6', '7'], dtype='S'))
        workspace.FeedBlob('default', np.array('-1', dtype='S'))
        workspace.FeedBlob('lengths', np.array([3, 4], dtype=np.int32))
        workspace.RunOperatorOnce(op)
        output = workspace.FetchBlob('output')
        expected =\
            np.array([['-1', '1', '3'], ['6', '7', '-1']], dtype='S')
        self.assertEqual(output.shape, expected.shape)
        np.testing.assert_array_equal(output, expected)

    def test_sparse_to_dense_mask_empty_lengths(self):
        op = core.CreateOperator(
            'SparseToDenseMask',
            ['indices', 'values', 'default'],
            ['output'],
            mask=[1, 2, 6])
        workspace.FeedBlob('indices', np.array([2, 4, 6], dtype=np.int32))
        workspace.FeedBlob('values', np.array([1, 2, 3], dtype=np.float))
        workspace.FeedBlob('default', np.array(-1, dtype=np.float))
        workspace.RunOperatorOnce(op)
        output = workspace.FetchBlob('output')
        expected = np.array([-1, 1, 3], dtype=np.float)
        self.assertEqual(output.shape, expected.shape)
        np.testing.assert_array_equal(output, expected)

    def test_sparse_to_dense_mask_no_lengths(self):
        op = core.CreateOperator(
            'SparseToDenseMask',
            ['indices', 'values', 'default'],
            ['output'],
            mask=[1, 2, 6])
        workspace.FeedBlob('indices', np.array([2, 4, 6], dtype=np.int32))
        workspace.FeedBlob('values', np.array([1, 2, 3], dtype=np.float))
        workspace.FeedBlob('default', np.array(-1, dtype=np.float))
        workspace.RunOperatorOnce(op)
        output = workspace.FetchBlob('output')
        expected = np.array([-1, 1, 3], dtype=np.float)
        self.assertEqual(output.shape, expected.shape)
        np.testing.assert_array_equal(output, expected)

    def test_sparse_to_dense_mask_presence_mask(self):
        op = core.CreateOperator(
            'SparseToDenseMask',
            ['indices', 'values', 'default', 'lengths'],
            ['output', 'presence_mask'],
            mask=[11, 12],
            return_presence_mask=True)
        workspace.FeedBlob('indices', np.array([11, 12, 13], dtype=np.int32))
        workspace.FeedBlob('values', np.array([11, 12, 13], dtype=np.float))
        workspace.FeedBlob('default', np.array(-1, dtype=np.float))
        workspace.FeedBlob('lengths', np.array([1, 2], dtype=np.int32))

        workspace.RunOperatorOnce(op)

        output = workspace.FetchBlob('output')
        presence_mask = workspace.FetchBlob('presence_mask')
        expected_output = np.array([[11, -1], [-1, 12]], dtype=np.float)
        expected_presence_mask = np.array(
            [[True, False], [False, True]],
            dtype=np.bool)
        self.assertEqual(output.shape, expected_output.shape)
        np.testing.assert_array_equal(output, expected_output)
        self.assertEqual(presence_mask.shape, expected_presence_mask.shape)
        np.testing.assert_array_equal(presence_mask, expected_presence_mask)
