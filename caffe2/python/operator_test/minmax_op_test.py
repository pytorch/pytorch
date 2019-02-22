from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
from hypothesis import given
import hypothesis.strategies as st
import numpy as np

import unittest


class TestMax(serial.SerializedTestCase):

    @given(**hu.gcs_cpu_only)
    def test_max_single_inp(self, gc, dc):

        op = core.CreateOperator("Max", ["X"], ["Y"])

        X = np.array(np.random.rand(4, 4), dtype=np.float32)

        self.assertDeviceChecks(dc, op, [X], [0])

        def elementwise_max(X):
            return [X]

        self.assertReferenceChecks(gc, op, [X], elementwise_max)

    @given(**hu.gcs_cpu_only)
    def test_max_two_inps(self, gc, dc):

        op = core.CreateOperator("Max", ["X", "Y"], ["Z"])

        X = np.array(np.random.rand(3, 2), dtype=np.float32)
        Y = np.array(np.random.rand(), dtype=np.float32)

        self.assertDeviceChecks(dc, op, [X, Y], [0])

        def elementwise_max(X, Y):
            return [np.maximum(X, Y)]

        self.assertReferenceChecks(gc, op, [X, Y], elementwise_max)

    @given(**hu.gcs_cpu_only)
    def test_max_three_inps(self, gc, dc):

        op = core.CreateOperator("Max", ["X", "Y", "Z"], ["W"])

        X = np.array(np.random.rand(3, 2), dtype=np.float32)
        Y = np.array(np.random.rand(3, 2), dtype=np.float32)
        Z = np.array(np.random.rand(1, 2), dtype=np.float32)

        self.assertDeviceChecks(dc, op, [X, Y, Z], [0])

        def elementwise_max(X, Y, Z):
            return [np.maximum(np.maximum(X, Y), Z)]

        self.assertReferenceChecks(gc, op, [X, Y, Z], elementwise_max)

if __name__ == "__main__":
    unittest.main()
