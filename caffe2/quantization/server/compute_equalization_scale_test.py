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

from __future__ import absolute_import, division, print_function, unicode_literals

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, workspace
from hypothesis import given, settings


class TestComputeEqualizationScaleOp(hu.HypothesisTestCase):
    @settings(max_examples=10)
    @given(
        m=st.integers(1, 50),
        n=st.integers(1, 50),
        k=st.integers(1, 50),
        rnd_seed=st.integers(1, 5),
        **hu.gcs_cpu_only
    )
    def test_compute_equalization_scale(self, m, n, k, rnd_seed, gc, dc):
        np.random.seed(rnd_seed)
        W = np.random.rand(n, k).astype(np.float32) - 0.5
        X = np.random.rand(m, k).astype(np.float32) - 0.5

        def ref_compute_equalization_scale(X, W):
            S = np.ones([X.shape[1]])
            S_INV = np.ones([X.shape[1]])
            for j in range(W.shape[1]):
                WcolMax = np.absolute(W[:, j]).max()
                XcolMax = np.absolute(X[:, j]).max()
                if WcolMax and XcolMax:
                    S[j] = np.sqrt(WcolMax / XcolMax)
                    S_INV[j] = 1 / S[j]
            return S, S_INV

        net = core.Net("test")

        ComputeEqualizationScaleOp = core.CreateOperator(
            "ComputeEqualizationScale", ["X", "W"], ["S", "S_INV"]
        )
        net.Proto().op.extend([ComputeEqualizationScaleOp])

        self.ws.create_blob("X").feed(X, device_option=gc)
        self.ws.create_blob("W").feed(W, device_option=gc)
        self.ws.run(net)

        S = self.ws.blobs["S"].fetch()
        S_INV = self.ws.blobs["S_INV"].fetch()
        S_ref, S_INV_ref = ref_compute_equalization_scale(X, W)
        np.testing.assert_allclose(S, S_ref, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(S_INV, S_INV_ref, atol=1e-3, rtol=1e-3)

    def test_compute_equalization_scale_shape_inference(self):
        X = np.array([[1, 2], [2, 4], [6, 7]]).astype(np.float32)
        W = np.array([[2, 3], [5, 4], [8, 2]]).astype(np.float32)
        ComputeEqualizationScaleOp = core.CreateOperator(
            "ComputeEqualizationScale", ["X", "W"], ["S", "S_INV"]
        )
        workspace.FeedBlob("X", X)
        workspace.FeedBlob("W", W)

        net = core.Net("test_shape_inference")
        net.Proto().op.extend([ComputeEqualizationScaleOp])
        shapes, types = workspace.InferShapesAndTypes(
            [net],
            blob_dimensions={"X": X.shape, "W": W.shape},
            blob_types={"X": core.DataType.FLOAT, "W": core.DataType.FLOAT},
        )
        assert (
            "S" in shapes and "S" in types and "S_INV" in shapes and "S_INV" in types
        ), "Failed to infer the shape or type of output"
        self.assertEqual(shapes["S"], [1, 2])
        self.assertEqual(shapes["S_INV"], [1, 2])
        self.assertEqual(types["S"], core.DataType.FLOAT)
        self.assertEqual(types["S_INV"], core.DataType.FLOAT)
