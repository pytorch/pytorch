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
import hypothesis.strategies as st
import unittest
import caffe2.python.hypothesis_test_util as hu
from caffe2.python import core
from hypothesis import given


class TestPad(hu.HypothesisTestCase):
    @given(pad_t=st.integers(-5, 0),
           pad_l=st.integers(-5, 0),
           pad_b=st.integers(-5, 0),
           pad_r=st.integers(-5, 0),
           mode=st.sampled_from(["constant", "reflect", "edge"]),
           size_w=st.integers(16, 128),
           size_h=st.integers(16, 128),
           size_c=st.integers(1, 4),
           size_n=st.integers(1, 4),
           **hu.gcs)
    def test_crop(self,
                  pad_t, pad_l, pad_b, pad_r,
                  mode,
                  size_w, size_h, size_c, size_n,
                  gc, dc):
        op = core.CreateOperator(
            "PadImage",
            ["X"],
            ["Y"],
            pad_t=pad_t,
            pad_l=pad_l,
            pad_b=pad_b,
            pad_r=pad_r,
        )
        X = np.random.rand(
            size_n, size_c, size_h, size_w).astype(np.float32)

        def ref(X):
            return (X[:, :, -pad_t:pad_b or None, -pad_l:pad_r or None],)

        self.assertReferenceChecks(gc, op, [X], ref)
        self.assertDeviceChecks(dc, op, [X], [0])


if __name__ == "__main__":
    unittest.main()
