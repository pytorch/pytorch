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






import numpy as np

from hypothesis import given
import hypothesis.strategies as st

from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu


class TestTTContraction(hu.HypothesisTestCase):
    @given(D=st.integers(min_value=5, max_value=20),
           K=st.integers(min_value=5, max_value=20),
           M=st.integers(min_value=5, max_value=20),
           N=st.integers(min_value=5, max_value=20),
           **hu.gcs)
    def test_tt_contraction(self, D, K, M, N, gc, dc):
        A = np.random.rand(K, M).astype(np.float32)
        B = np.random.rand(D, K, N).astype(np.float32)

        workspace.FeedBlob('A', A)
        workspace.FeedBlob('B', B)

        op = core.CreateOperator(
            'TTContraction',
            ['A', 'B'],
            ['C'],
            K=K,
            M=M,
            N=N)
        workspace.RunOperatorOnce(op)

        def tt_contraction_ref(A_, B_):
            return ((A_[:, :, np.newaxis] * B_[:, :, np.newaxis, :])
                    .sum(axis=1).flatten()),

        # Check against numpy reference
        self.assertReferenceChecks(gc, op, [A, B], tt_contraction_ref)
        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [A, B], [0])
        # Gradient check wrt A
        self.assertGradientChecks(gc, op, [A, B], 0, [0])
        # Gradient check wrt B
        self.assertGradientChecks(gc, op, [A, B], 1, [0])
