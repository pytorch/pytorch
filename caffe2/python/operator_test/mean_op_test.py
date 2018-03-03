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
import numpy as np

import unittest


class TestMean(hu.HypothesisTestCase):
    @given(
        k=st.integers(1, 5),
        n=st.integers(1, 10),
        m=st.integers(1, 10),
        in_place=st.booleans(),
        seed=st.integers(0, 2**32 - 1),
        **hu.gcs
    )
    def test_mean(self, k, n, m, in_place, seed, gc, dc):
        np.random.seed(seed)
        input_names = []
        input_vars = []

        for i in range(k):
            X_name = 'X' + str(i)
            input_names.append(X_name)
            var = np.random.randn(n, m).astype(np.float32)
            input_vars.append(var)

        def mean_ref(*args):
            return [np.mean(args, axis=0)]

        op = core.CreateOperator(
            "Mean",
            input_names,
            ['Y' if not in_place else 'X0'],
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=input_vars,
            reference=mean_ref,
        )

        self.assertGradientChecks(
            device_option=gc,
            op=op,
            inputs=input_vars,
            outputs_to_check=0,
            outputs_with_grads=[0],
        )

        self.assertDeviceChecks(dc, op, input_vars, [0])


if __name__ == "__main__":
    unittest.main()

