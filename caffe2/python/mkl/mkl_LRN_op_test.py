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
from hypothesis import given, settings
import numpy as np
from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.mkl_test_util as mu


@unittest.skipIf(not workspace.C.has_mkldnn,
                 "Skipping as we do not have mkldnn.")


class MKLLRNTest(hu.HypothesisTestCase):
    @given(input_channels=st.integers(1, 3),
           batch_size=st.integers(1, 3),
           im_size=st.integers(1, 10),
           order=st.sampled_from(["NCHW"]),
           **mu.gcs)

    def test_mkl_LRN(self, input_channels,
                            batch_size, im_size, order,
                             gc, dc):
        op = core.CreateOperator(
            "LRN",
            ["X"],
            ["Y", "Y_scale"],
            size=5,
            alpha=0.001,
            beta=0.75,
            bias=2.0,
            order=order,
        )
        X = np.random.rand(
            batch_size, input_channels, im_size, im_size).astype(np.float32)

        self.assertDeviceChecks(dc, op, [X], [0])


if __name__ == "__main__":
    import unittest
    unittest.main()
