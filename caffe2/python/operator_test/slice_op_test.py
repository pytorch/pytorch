from __future__ import absolute_import, division, print_function, unicode_literals

import random
import unittest

import numpy as np
import six
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
from hypothesis import assume, given


class TestSliceOp(unittest.TestCase):
    def _test_slice(self):
        X = np.random.randn(20, 10, 5).astype(np.float32)

        starts = np.array([1], dtype=np.int64)
        ends = np.array([7], dtype=np.int64)
        axes = np.array([1], dtype=np.int64)
        Y = X[:, 1:7]

        op = core.CreateOperator(
            "Slice", ["X"], ["Y"], starts=starts, ends=ends, axes=axes
        )

        workspace.FeedBlob("X", X)
        workspace.RunOperatorOnce(op)
        Z = workspace.FetchBlob("Y")
        np.testing.assert_array_equal(Y, Z)
        self.assertEqual(Y.tobytes(), Z.tobytes())

    def test_slice_null_axes(self):
        X = np.random.randn(20, 10, 5).astype(np.float32)

        starts = np.array([5, 0, 0], dtype=np.int64)
        ends = np.array([16, 10, 5], dtype=np.int64)
        Y = X[5:16, :]

        op = core.CreateOperator("Slice", ["X"], ["Y"], starts=starts, ends=ends)

        workspace.FeedBlob("X", X)
        workspace.RunOperatorOnce(op)
        Z = workspace.FetchBlob("Y")
        np.testing.assert_array_equal(Y, Z)
        self.assertEqual(Y.tobytes(), Z.tobytes())

    def test_slice_null_axes_partial(self):
        X = np.random.randn(20, 10, 5).astype(np.float32)

        starts = np.array([5, 0], dtype=np.int64)
        ends = np.array([16, 10], dtype=np.int64)
        Y = X[5:16, :]

        op = core.CreateOperator("Slice", ["X"], ["Y"], starts=starts, ends=ends)

        workspace.FeedBlob("X", X)
        workspace.RunOperatorOnce(op)
        Z = workspace.FetchBlob("Y")
        np.testing.assert_array_equal(Y, Z)
        self.assertEqual(Y.tobytes(), Z.tobytes())

    def test_slice_null_axes_out_of_range(self):
        X = np.random.randn(20, 10, 5).astype(np.float32)

        starts = np.array([0, 3, 0], dtype=np.int64)
        ends = np.array([20, 4, 1000], dtype=np.int64)
        Y = X[:, 3:4, 0:1000]

        op = core.CreateOperator("Slice", ["X"], ["Y"], starts=starts, ends=ends)

        workspace.FeedBlob("X", X)
        workspace.RunOperatorOnce(op)
        Z = workspace.FetchBlob("Y")
        np.testing.assert_array_equal(Y, Z)
        self.assertEqual(Y.tobytes(), Z.tobytes())

    def test_slice_null_axes_partial_out_of_range(self):
        X = np.random.randn(20, 10, 5).astype(np.float32)

        starts = np.array([0, 6], dtype=np.int64)
        ends = np.array([100, 9], dtype=np.int64)
        Y = X[:, 6:9, :]

        op = core.CreateOperator("Slice", ["X"], ["Y"], starts=starts, ends=ends)

        workspace.FeedBlob("X", X)
        workspace.RunOperatorOnce(op)
        Z = workspace.FetchBlob("Y")
        np.testing.assert_array_equal(Y, Z)
        self.assertEqual(Y.tobytes(), Z.tobytes())

    def _test_slice_out_of_range(self):
        X = np.random.randn(20, 10, 5).astype(np.float32)

        starts = np.array([1], dtype=np.int64)
        ends = np.array([1000], dtype=np.int64)
        axes = np.array([1], dtype=np.int64)
        Y = X[:, 1:1000]

        op = core.CreateOperator(
            "Slice", ["X"], ["Y"], starts=starts, ends=ends, axes=axes
        )

        workspace.FeedBlob("X", X)
        workspace.RunOperatorOnce(op)
        Z = workspace.FetchBlob("Y")
        np.testing.assert_array_equal(Y, Z)
        self.assertEqual(Y.tobytes(), Z.tobytes())

    def _test_slice_neg_axes(self):
        X = np.random.randn(20, 10, 5).astype(np.float32)
        starts = np.array([0, 0, 3], dtype=np.int64)
        ends = np.array([20, 10, 4], dtype=np.int64)
        axes = np.array([0, -2, -1], dtype=np.int64)
        Y = X[:, :, 3:4]

        op = core.CreateOperator(
            "Slice", ["X"], ["Y"], starts=starts, ends=ends, axes=axes
        )

        workspace.FeedBlob("X", X)
        workspace.RunOperatorOnce(op)
        Z = workspace.FetchBlob("Y")
        np.testing.assert_array_equal(Y, Z)
        self.assertEqual(Y.tobytes(), Z.tobytes())

    def _test_slice_unordered_axes(self):
        X = np.random.randn(20, 10, 5, 33).astype(np.float32)
        starts = np.array([0, 0, 0], dtype=np.int64)
        ends = np.array([20, 5, 10], dtype=np.int64)
        axes = np.array([0, -1, 1], dtype=np.int64)
        Y = X[:, :, :, 0:5]

        op = core.CreateOperator(
            "Slice", ["X"], ["Y"], starts=starts, ends=ends, axes=axes
        )

        workspace.FeedBlob("X", X)
        workspace.RunOperatorOnce(op)
        Z = workspace.FetchBlob("Y")
        np.testing.assert_array_equal(Y, Z)
        self.assertEqual(Y.tobytes(), Z.tobytes())
