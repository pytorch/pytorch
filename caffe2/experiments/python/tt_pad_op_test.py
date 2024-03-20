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


class TestTTPad(hu.HypothesisTestCase):
    @given(K=st.integers(min_value=2, max_value=10),
           M=st.integers(min_value=10, max_value=20),
           N=st.integers(min_value=10, max_value=20),
           **hu.gcs)
    def test_tt_pad(self, K, M, N, gc, dc):
        op = core.CreateOperator(
            'TTPad',
            ['A'],
            ['A', 'dim0'],
            scale=(K))

        A = np.random.rand(M, N).astype(np.float32)
        workspace.FeedBlob('A', A)
        workspace.RunOperatorOnce(op)

        def tt_pad_ref(A_):
            M_ = A_.shape[0]
            if M_ % K == 0:
                new_dim0 = M_
            else:
                new_dim0 = (M_ // K + 1) * K
            return (np.vstack((A_, np.zeros((new_dim0 - M_, A_.shape[1])))),
                    np.array([A.shape[0]]))

        # Check against numpy reference
        self.assertReferenceChecks(gc, op, [A], tt_pad_ref)
        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [A], [0])
        # Gradient check wrt A
        self.assertGradientChecks(gc, op, [A], 0, [0])
