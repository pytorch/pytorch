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

dyndep.InitOpsLibrary("//caffe2/caffe2/quantization/server:dnnlowp_ops")


class DNNLowPTanhOpTest(hu.HypothesisTestCase):
    @given(size=st.integers(1024, 2048),
           **hu.gcs_cpu_only)
    def test_dnnlowp_tanh(self, size, gc, dc):

        X = (np.random.rand(size) * 10 - 5).astype(np.float32)

        Output = collections.namedtuple("Output", ["Y", "op_type", "engine"])
        outputs = []

        op_engine_list = [
            ("Tanh", ""),
            ("Tanh", "DNNLOWP"),
            ("Int8Tanh", "DNNLOWP"),
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
                    followed_by="Tanh",
                )
                net.Proto().op.extend([quantize])

            tanh = core.CreateOperator(
                op_type,
                ["X_q" if engine == "DNNLOWP" else "X"],
                ["Y_q" if engine == "DNNLOWP" else "Y"],
                engine=engine,
                device_option=gc,
            )
            net.Proto().op.extend([tanh])

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

        for o in outputs:
            np.testing.assert_allclose(o.Y, outputs[0].Y, atol=0.02, rtol=0)
