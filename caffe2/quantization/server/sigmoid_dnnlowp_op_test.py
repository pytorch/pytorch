from __future__ import absolute_import, division, print_function, unicode_literals

import collections

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, dyndep, workspace
from hypothesis import given


dyndep.InitOpsLibrary("//caffe2/caffe2/quantization/server:dnnlowp_ops")
workspace.GlobalInit(["caffe2", "--caffe2_omp_num_threads=11"])


class DNNLowPSigmoidOpTest(hu.HypothesisTestCase):
    @given(size=st.integers(1024, 2048), is_empty=st.booleans(), **hu.gcs_cpu_only)
    def test_dnnlowp_sigmoid(self, size, is_empty, gc, dc):
        if is_empty:
            size = 0

        X = (np.random.rand(size) * 20 - 10).astype(np.float32)

        Output = collections.namedtuple("Output", ["Y", "op_type", "engine"])
        outputs = []

        op_engine_list = [
            ("Sigmoid", ""),
            ("Sigmoid", "DNNLOWP"),
            ("Int8Sigmoid", "DNNLOWP"),
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
                    followed_by="Sigmoid",
                )
                net.Proto().op.extend([quantize])

            sigmoid = core.CreateOperator(
                op_type,
                ["X_q" if engine == "DNNLOWP" else "X"],
                ["Y_q" if engine == "DNNLOWP" else "Y"],
                engine=engine,
                device_option=gc,
            )
            net.Proto().op.extend([sigmoid])

            if engine == "DNNLOWP":
                dequantize = core.CreateOperator(
                    "Dequantize", ["Y_q"], ["Y"], engine=engine, device_option=gc
                )
                net.Proto().op.extend([dequantize])

            self.ws.create_blob("X").feed(X, device_option=gc)
            self.ws.run(net)
            outputs.append(
                Output(Y=self.ws.blobs["Y"].fetch(), op_type=op_type, engine=engine)
            )

        for o in outputs:
            np.testing.assert_allclose(o.Y, outputs[0].Y, atol=0.01, rtol=0)
