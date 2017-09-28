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

from caffe2.python import core
from hypothesis import given
import hypothesis.strategies as st
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.mkl_test_util as mu
import numpy as np

import unittest


class TestRelu(hu.HypothesisTestCase):

    @given(X=hu.tensor(),
           engine=st.sampled_from(["", "CUDNN"]),
           **mu.gcs)
    def test_relu(self, X, gc, dc, engine):
        op = core.CreateOperator("Relu", ["X"], ["Y"], engine=engine)
        # go away from the origin point to avoid kink problems
        X += 0.02 * np.sign(X)
        X[X == 0.0] += 0.02
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0])


if __name__ == "__main__":
    unittest.main()
