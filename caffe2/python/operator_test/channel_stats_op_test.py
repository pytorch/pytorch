from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
from hypothesis import assume, given
import hypothesis.strategies as st
import numpy as np
import unittest


class TestChannelStats(serial.SerializedTestCase):
    @serial.given(
        size=st.integers(7, 10),
        inputChannels=st.integers(1, 10),
        batchSize=st.integers(1, 3),
        **hu.gcs
    )
    def testChannelStats(self, size, inputChannels, batchSize, gc, dc):

        op = core.CreateOperator(
            "ChannelStats",
            ["X"],
            ["sum", "sumsq"],
        )

        def referenceChannelStatsTest(X):
            sums = np.sum(X, axis=(0, 2, 3), keepdims=False)
            sumsq = np.zeros(inputChannels)
            sumsq = np.sum(X**2, axis=(0, 2, 3), keepdims=False)
            return sums, sumsq

        X = np.random.rand(batchSize, inputChannels, size, size)\
                .astype(np.float32) - 0.5
        self.assertReferenceChecks(gc, op, [X], referenceChannelStatsTest)


if __name__ == "__main__":
    unittest.main()
