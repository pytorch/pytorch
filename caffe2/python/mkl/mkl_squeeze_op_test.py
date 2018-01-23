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

import unittest
import hypothesis.strategies as st
from hypothesis import given
import numpy as np
from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.mkl_test_util as mu


@unittest.skipIf(
    not workspace.C.has_mkldnn, "Skipping as we do not have mkldnn."
)
class MKLSqueezeTest(hu.HypothesisTestCase):
    @given(
        squeeze_dims=st.lists(st.integers(0, 3), min_size=1, max_size=3),
        inplace=st.booleans(),
        **mu.gcs
    )
    def test_mkl_squeeze(self, squeeze_dims, inplace, gc, dc):
        shape = [
            1 if dim in squeeze_dims else np.random.randint(1, 5)
            for dim in range(4)
        ]
        X = np.random.rand(*shape).astype(np.float32)
        op = core.CreateOperator(
            "Squeeze", "X", "X" if inplace else "Y", dims=squeeze_dims
        )
        self.assertDeviceChecks(dc, op, [X], [0])


if __name__ == "__main__":
    unittest.main()
