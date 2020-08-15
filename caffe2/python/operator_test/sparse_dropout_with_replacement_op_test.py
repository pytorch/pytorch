from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import numpy as np


class SparseDropoutWithReplacementTest(hu.HypothesisTestCase):
    @given(**hu.gcs_cpu_only)
    def test_no_dropout(self, gc, dc):
        X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.int64)
        Lengths = np.array([2, 2, 2, 2, 2]).astype(np.int32)
        replacement_value = -1
        self.ws.create_blob("X").feed(X)
        self.ws.create_blob("Lengths").feed(Lengths)
        sparse_dropout_op = core.CreateOperator(
            "SparseDropoutWithReplacement", ["X", "Lengths"], ["Y", "LY"],
            ratio=0.0, replacement_value=replacement_value)
        self.ws.run(sparse_dropout_op)
        Y = self.ws.blobs["Y"].fetch()
        OutputLengths = self.ws.blobs["LY"].fetch()
        self.assertListEqual(X.tolist(), Y.tolist(),
                             "Values should stay unchanged")
        self.assertListEqual(Lengths.tolist(), OutputLengths.tolist(),
                             "Lengths should stay unchanged.")

    @given(**hu.gcs_cpu_only)
    def test_all_dropout(self, gc, dc):
        X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.int64)
        Lengths = np.array([2, 2, 2, 2, 2]).astype(np.int32)
        replacement_value = -1
        self.ws.create_blob("X").feed(X)
        self.ws.create_blob("Lengths").feed(Lengths)
        sparse_dropout_op = core.CreateOperator(
            "SparseDropoutWithReplacement", ["X", "Lengths"], ["Y", "LY"],
            ratio=1.0, replacement_value=replacement_value)
        self.ws.run(sparse_dropout_op)
        y = self.ws.blobs["Y"].fetch()
        lengths = self.ws.blobs["LY"].fetch()
        for elem in y:
            self.assertEqual(elem, replacement_value, "Expected all \
                negative elements when dropout ratio is 1.")
        for length in lengths:
            self.assertEqual(length, 1)
        self.assertEqual(sum(lengths), len(y))

    @given(**hu.gcs_cpu_only)
    def test_all_dropout_empty_input(self, gc, dc):
        X = np.array([]).astype(np.int64)
        Lengths = np.array([0]).astype(np.int32)
        replacement_value = -1
        self.ws.create_blob("X").feed(X)
        self.ws.create_blob("Lengths").feed(Lengths)
        sparse_dropout_op = core.CreateOperator(
            "SparseDropoutWithReplacement", ["X", "Lengths"], ["Y", "LY"],
            ratio=1.0, replacement_value=replacement_value)
        self.ws.run(sparse_dropout_op)
        y = self.ws.blobs["Y"].fetch()
        lengths = self.ws.blobs["LY"].fetch()
        self.assertEqual(len(y), 1, "Expected single dropout value")
        self.assertEqual(len(lengths), 1, "Expected single element \
            in lengths array")
        self.assertEqual(lengths[0], 1, "Expected 1 as sole length")
        self.assertEqual(sum(lengths), len(y))
