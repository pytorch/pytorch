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



import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, workspace
from caffe2.quantization.server import dnnlowp_pybind11
from hypothesis import given, settings


class TestInt8GenQuantParamsOperator(hu.HypothesisTestCase):
    @settings(max_examples=20, deadline=None)
    @given(
        n=st.integers(10, 100),
        m=st.integers(1, 128),
        k=st.integers(64, 1024),
        quantization_kind=st.sampled_from(
            [
                "MIN_MAX_QUANTIZATION",
                "L2_MIN_QUANTIZATION_APPROX",
                "L2_MIN_QUANTIZATION",
                "P99_QUANTIZATION"
            ]
        ),
        preserve_sparsity=st.booleans(),
        rnd_seed=st.integers(1, 5),
        **hu.gcs_cpu_only
    )
    def test_int8_gen_quant_params_op(
        self, n, m, k, quantization_kind, preserve_sparsity, rnd_seed, gc, dc
    ):
        assert n > 0, "Zero samples in the input data"
        X_min = 0 if preserve_sparsity else -77
        X_max = X_min + 255
        np.random.seed(rnd_seed)
        X = np.round(np.random.rand(n, m, k) * (X_max - X_min) + X_min).astype(
            np.float32
        )
        # Calculate X_qparam
        hist, bin_edges = np.histogram(X.flatten(), bins=2048)
        X_qparam = dnnlowp_pybind11.ChooseStaticQuantizationParams(
            np.min(X), np.max(X), hist, preserve_sparsity, 8, quantization_kind
        )

        # Build a net to generate X's qparam using the Int8GenQuantParams op
        workspace.FeedBlob("X", X, device_option=gc)
        dnnlowp_pybind11.CreateInt8QuantSchemeBlob(
            "quant_scheme", quantization_kind, preserve_sparsity
        )
        assert workspace.HasBlob(
            "quant_scheme"
        ), "Failed to create the quant_scheme blob in current workspace"

        gen_quant_params_net = core.Net("gen_quant_params")
        gen_quant_params_op = core.CreateOperator(
            "Int8GenQuantParams",
            ["X", "quant_scheme"],
            ["quant_param"],
            device_option=gc,
        )
        gen_quant_params_net.Proto().op.extend([gen_quant_params_op])
        assert workspace.RunNetOnce(
            gen_quant_params_net
        ), "Failed to run the gen_quant_params net"
        scale, zero_point = dnnlowp_pybind11.ObserveInt8QuantParamsBlob("quant_param")
        shapes, types = workspace.InferShapesAndTypes(
            [gen_quant_params_net],
            blob_dimensions={"X": [n, m, k], "quant_scheme": [1]},
            blob_types={"X": core.DataType.FLOAT, "quant_scheme": core.DataType.STRING}
        )
        self.assertEqual(shapes["quant_param"], [1])
        self.assertEqual(types["quant_param"], core.DataType.FLOAT)

        np.testing.assert_equal(scale, X_qparam.scale)
        np.testing.assert_equal(zero_point, X_qparam.zero_point)
