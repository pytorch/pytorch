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
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np


class TestElementwiseLinearOp(hu.HypothesisTestCase):

    @given(n=st.integers(2, 100), d=st.integers(2, 10), **hu.gcs)
    # @given(n=st.integers(2, 50), d=st.integers(2, 50), **hu.gcs_cpu_only)
    def test(self, n, d, gc, dc):
        X = np.random.rand(n, d).astype(np.float32)
        a = np.random.rand(d).astype(np.float32)
        b = np.random.rand(d).astype(np.float32)

        def ref_op(X, a, b):
            d = a.shape[0]
            return [np.multiply(X, a.reshape(1, d)) + b.reshape(1, d)]

        op = core.CreateOperator(
            "ElementwiseLinear",
            ["X", "a", "b"],
            ["Y"]
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X, a, b],
            reference=ref_op,
        )

        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [X, a, b], [0])
        # Gradient check wrt X
        self.assertGradientChecks(gc, op, [X, a, b], 0, [0])
        # Gradient check wrt a
        self.assertGradientChecks(gc, op, [X, a, b], 1, [0])
        # # Gradient check wrt b
        self.assertGradientChecks(gc, op, [X, a, b], 2, [0])
