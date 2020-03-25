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
import caffe2.python.hypothesis_test_util as hu
import functools
from hypothesis import given
import hypothesis.strategies as st
import numpy as np

class TestWeightScale(hu.HypothesisTestCase):
    @given(inputs=hu.tensors(n=1),
           ITER=st.integers(min_value=0, max_value=100),
           stepsize=st.integers(min_value=20, max_value=50),
           upper_bound_iter=st.integers(min_value=5, max_value=100),
           scale=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
           **hu.gcs)
    def test_weight_scale(self, inputs, ITER, stepsize, upper_bound_iter, scale, gc, dc):
        ITER = np.array([ITER], dtype=np.int64)
        op = core.CreateOperator(
            "WeightScale", ["w", "iter"], ["nw"], stepsize=stepsize, upper_bound_iter=upper_bound_iter, scale=scale)

        def ref_weight_scale(w, iter, stepsize, upper_bound_iter, scale):
            iter = iter + 1
            return [w * scale if iter % stepsize == 0 and iter < upper_bound_iter else w]

        self.assertReferenceChecks(
            gc,
            op,
            [inputs[0], ITER],
            functools.partial(ref_weight_scale, stepsize=stepsize, upper_bound_iter=upper_bound_iter, scale=scale))
