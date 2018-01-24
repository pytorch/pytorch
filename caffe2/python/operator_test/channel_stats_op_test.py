# Copyright (c) 2016-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from hypothesis import assume, given
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
from caffe2.proto import caffe2_pb2
import unittest


class TestChannelStats(hu.HypothesisTestCase):
    @given(
        size=st.integers(7, 10),
        inputChannels=st.integers(1, 10),
        batchSize=st.integers(1, 3),
        **hu.gcs
    )
    def testChannelStats(self, size, inputChannels, batchSize, gc, dc):

        assume(gc.device_type != caffe2_pb2.CUDA)
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
