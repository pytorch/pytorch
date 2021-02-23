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
from caffe2.python import core, workspace
from hypothesis import given
from caffe2.quantization.server import dnnlowp_pybind11

class TestInt8QuantSchemeBlobFillOperator(hu.HypothesisTestCase):
    @given(
        **hu.gcs_cpu_only
    )
    def test_int8_quant_scheme_blob_fill_op(
        self, gc, dc
    ):
        # Build a net to generate qscheme blob using the Int8QuantSchemeBlobFill op
        gen_quant_scheme_net = core.Net("gen_quant_scheme")
        gen_quant_scheme_op = core.CreateOperator(
            "Int8QuantSchemeBlobFill",
            [],
            ["quant_scheme"],
            quantization_kind="MIN_MAX_QUANTIZATION",
            preserve_sparsity=False,
            device_option=gc,
        )
        gen_quant_scheme_net.Proto().op.extend([gen_quant_scheme_op])
        assert workspace.RunNetOnce(
            gen_quant_scheme_net
        ), "Failed to run the gen_quant_scheme net"
        quantization_kind, preserve_sparsity = dnnlowp_pybind11.ObserveInt8QuantSchemeBlob("quant_scheme")
        assert quantization_kind == "MIN_MAX_QUANTIZATION"
        assert not preserve_sparsity
