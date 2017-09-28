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

import functools

import numpy as np
from hypothesis import given
import hypothesis.strategies as st
from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu


class TestNormalizeOp(hu.HypothesisTestCase):

    @given(X=hu.tensor(min_dim=1,
                       max_dim=5,
                       elements=st.floats(min_value=0.5, max_value=1.0)),
           **hu.gcs)
    def test_normalize(self, X, gc, dc):
        def ref_normalize(X, axis):
            x_normed = X / (
                np.sqrt((X**2).sum(axis=axis, keepdims=True)) + np.finfo(X.dtype).tiny)
            return (x_normed,)

        for axis in range(-X.ndim, X.ndim):
            op = core.CreateOperator("Normalize", "X", "Y", axis=axis)
            self.assertReferenceChecks(
                gc,
                op,
                [X],
                functools.partial(ref_normalize, axis=axis))
            self.assertDeviceChecks(dc, op, [X], [0])
            self.assertGradientChecks(gc, op, [X], 0, [0])

    @given(X=hu.tensor(min_dim=1,
                       max_dim=5,
                       elements=st.floats(min_value=0.5, max_value=1.0)),
           **hu.gcs)
    def test_normalize_L1(self, X, gc, dc):
        def ref(X, axis):
            norm = abs(X).sum(axis=axis, keepdims=True)
            return (X / norm,)

        for axis in range(-X.ndim, X.ndim):
            print('axis: ', axis)
            op = core.CreateOperator("NormalizeL1", "X", "Y", axis=axis)
            self.assertReferenceChecks(
                gc,
                op,
                [X],
                functools.partial(ref, axis=axis))
            self.assertDeviceChecks(dc, op, [X], [0])
