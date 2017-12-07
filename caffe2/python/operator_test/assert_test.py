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

import numpy as np
from hypothesis import given
import hypothesis.strategies as st
from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu


class TestAssert(hu.HypothesisTestCase):
    @given(
        dtype=st.sampled_from(['bool_', 'int32', 'int64']),
        shape=st.lists(elements=st.integers(1, 10), min_size=1, max_size=4),
        **hu.gcs)
    def test_assert(self, dtype, shape, gc, dc):
        test_tensor = np.random.rand(*shape).astype(np.dtype(dtype))

        op = core.CreateOperator('Assert', ['X'], [])

        def assert_ref(X):
            return []

        try:
            self.assertReferenceChecks(gc, op, [test_tensor], assert_ref)
        except Exception:
            assert(not np.all(test_tensor))
