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

import numpy as np

from hypothesis import given
import hypothesis.strategies as st

from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu


class TestClip(hu.HypothesisTestCase):
    @given(X=hu.tensor(),
           min_=st.floats(min_value=-1, max_value=0),
           max_=st.floats(min_value=0, max_value=1),
           inplace=st.booleans(),
           **hu.gcs)
    def test_clip(self, X, min_, max_, inplace, gc, dc):
        # go away from the origin point to avoid kink problems

        X[np.abs(X - min_) < 0.05] += 0.1
        X[np.abs(X - max_) < 0.05] += 0.1

        def clip_ref(X):
            X = X.clip(min_, max_)
            return (X,)

        op = core.CreateOperator(
            "Clip",
            ["X"], ["Y" if not inplace else "X"],
            min=min_,
            max=max_)
        self.assertReferenceChecks(gc, op, [X], clip_ref)
        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [X], [0])
        # Gradient check wrt X
        self.assertGradientChecks(gc, op, [X], 0, [0])


if __name__ == "__main__":
    import unittest
    unittest.main()
