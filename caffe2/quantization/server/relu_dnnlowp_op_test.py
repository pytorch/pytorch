from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import caffe2.python.hypothesis_test_util as hu
from caffe2.python import core, dyndep
from hypothesis import given
import hypothesis.strategies as st
import collections
from dnnlowp_test_utils import check_quantized_results_close

dyndep.InitOpsLibrary("//caffe2/caffe2/quantization/server:dnnlowp_ops")


class DNNLowPReluOpTest(hu.HypothesisTestCase):
    @given(size=st.integers(1024, 2048),
           **hu.gcs_cpu_only)
    def test_dnnlowp_relu(self, size, gc, dc):
        min_ = -10.
        max_ = 10.
        scale = (max_ - min_) / 255
        zero_point = int(np.round(-min_ / scale))
        X = (np.random.rand(size) * (max_ - min_) + min_).astype(np.float32)

        Output = collections.namedtuple("Output", ["Y", "op_type", "engine"])
        outputs = []

        op_engine_list = [
            ("Relu", ""),
            ("Relu", "DNNLOWP"),
            ("Int8Relu", "DNNLOWP"),
        ]

        for op_type, engine in op_engine_list:
            net = core.Net("test_net")

            if engine == "DNNLOWP":
                quantize = core.CreateOperator(
                    "Quantize",
                    ["X"],
                    ["X_q"],
                    engine=engine,
                    device_option=gc,
                    Y_scale=scale,
                    Y_zero_point=zero_point,
                )
                net.Proto().op.extend([quantize])

            relu = core.CreateOperator(
                op_type,
                ["X_q" if engine == "DNNLOWP" else "X"],
                ["Y_q" if engine == "DNNLOWP" else "Y"],
                engine=engine,
                device_option=gc,
            )
            net.Proto().op.extend([relu])

            if engine == "DNNLOWP":
                dequantize = core.CreateOperator(
                    "Dequantize",
                    ["Y_q"],
                    ["Y"],
                    engine=engine,
                    device_option=gc,
                )
                net.Proto().op.extend([dequantize])

            self.ws.create_blob("X").feed(X, device_option=gc)
            self.ws.run(net)
            outputs.append(Output(
                Y=self.ws.blobs["Y"].fetch(), op_type=op_type, engine=engine))

        # Y = max(0, X) so the only error is quantization of inputs
        check_quantized_results_close(outputs, ref=X)
