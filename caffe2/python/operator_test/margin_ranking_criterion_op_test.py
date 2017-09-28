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

from hypothesis import given
import hypothesis.strategies as st
import numpy as np

from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu


class TestMarginRankingCriterion(hu.HypothesisTestCase):
    @given(N=st.integers(min_value=10, max_value=20),
           seed=st.integers(min_value=0, max_value=65535),
           margin=st.floats(min_value=-0.5, max_value=0.5),
           **hu.gcs)
    def test_margin_ranking_criterion(self, N, seed, margin, gc, dc):
        np.random.seed(seed)
        X1 = np.random.randn(N).astype(np.float32)
        X2 = np.random.randn(N).astype(np.float32)
        Y = np.random.choice([-1, 1], size=N).astype(np.int32)
        op = core.CreateOperator(
            "MarginRankingCriterion", ["X1", "X2", "Y"], ["loss"],
            margin=margin)

        def ref_cec(X1, X2, Y):
            result = np.maximum(-Y * (X1 - X2) + margin, 0)
            return (result, )

        inputs = [X1, X2, Y]
        # This checks the op implementation against a reference function in
        # python.
        self.assertReferenceChecks(gc, op, inputs, ref_cec)
        # This checks the op implementation over multiple device options (e.g.
        # CPU and CUDA). [0] means that the 0-th output is checked.
        self.assertDeviceChecks(dc, op, inputs, [0])

        # Make singular points less sensitive
        X1[np.abs(margin - Y * (X1 - X2)) < 0.1] += 0.1
        X2[np.abs(margin - Y * (X1 - X2)) < 0.1] -= 0.1

        # Check dX1
        self.assertGradientChecks(gc, op, inputs, 0, [0])
        # Check dX2
        self.assertGradientChecks(gc, op, inputs, 1, [0])

if __name__ == "__main__":
    import unittest
    unittest.main()
