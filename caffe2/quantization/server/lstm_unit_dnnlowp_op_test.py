from __future__ import absolute_import, division, print_function, unicode_literals

import collections

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, dyndep, workspace
from hypothesis import given


dyndep.InitOpsLibrary("//caffe2/caffe2/quantization/server:dnnlowp_ops")
workspace.GlobalInit(["caffe2", "--caffe2_omp_num_threads=11"])


class DNNLowPLSTMUnitOpTest(hu.HypothesisTestCase):
    @given(
        N=st.integers(4, 64),
        D=st.integers(4, 64),
        forget_bias=st.integers(0, 4),
        **hu.gcs_cpu_only
    )
    def test_dnnlowp_lstm_unit(self, N, D, forget_bias, gc, dc):

        # X has scale 1, so exactly represented after quantization
        H_in = np.clip(np.random.randn(1, N, D), -1, 1).astype(np.float32)
        C_in = np.clip(np.random.randn(1, N, D), -1, 1).astype(np.float32)
        G = np.clip(np.random.randn(1, N, 4 * D), -1, 1).astype(np.float32)
        seq_lengths = np.round(np.random.rand(N)).astype(np.int32)
        # seq_lengths.fill(0)
        t = np.array([5]).astype(np.int32)

        Output = collections.namedtuple("Output", ["H_out", "C_out", "engine"])
        outputs = []

        engine_list = ["", "DNNLOWP"]
        for engine in engine_list:
            net = core.Net("test_net")

            if engine == "DNNLOWP":
                quantize_H_in = core.CreateOperator(
                    "Quantize", ["H_in"], ["H_in_q"], engine=engine, device_option=gc
                )
                quantize_C_in = core.CreateOperator(
                    "Quantize", ["C_in"], ["C_in_q"], engine=engine, device_option=gc
                )
                quantize_G = core.CreateOperator(
                    "Quantize", ["G"], ["G_q"], engine=engine, device_option=gc
                )
                net.Proto().op.extend([quantize_H_in, quantize_C_in, quantize_G])

            lstm = core.CreateOperator(
                "LSTMUnit",
                [
                    "H_in_q" if engine == "DNNLOWP" else "H_in",
                    "C_in_q" if engine == "DNNLOWP" else "C_in",
                    "G_q" if engine == "DNNLOWP" else "G",
                    "seq_lengths",
                    "t",
                ],
                [
                    "H_out_q" if engine == "DNNLOWP" else "H_out",
                    "C_out_q" if engine == "DNNLOWP" else "C_out",
                ],
                engine=engine,
                device_option=gc,
                axis=0,
            )
            net.Proto().op.extend([lstm])

            if engine == "DNNLOWP":
                dequantize_H_out = core.CreateOperator(
                    "Dequantize",
                    ["H_out_q"],
                    ["H_out"],
                    engine=engine,
                    device_option=gc,
                )
                dequantize_C_out = core.CreateOperator(
                    "Dequantize",
                    ["C_out_q"],
                    ["C_out"],
                    engine=engine,
                    device_option=gc,
                )
                net.Proto().op.extend([dequantize_H_out, dequantize_C_out])

            self.ws.create_blob("H_in").feed(H_in, device_option=gc)
            self.ws.create_blob("C_in").feed(C_in, device_option=gc)
            self.ws.create_blob("G").feed(G, device_option=gc)
            self.ws.create_blob("seq_lengths").feed(seq_lengths, device_option=gc)
            self.ws.create_blob("t").feed(t, device_option=gc)
            self.ws.run(net)
            outputs.append(
                Output(
                    H_out=self.ws.blobs["H_out"].fetch(),
                    C_out=self.ws.blobs["C_out"].fetch(),
                    engine=engine,
                )
            )

        for o in outputs:
            np.testing.assert_allclose(o.C_out, outputs[0].C_out, atol=0.1, rtol=0.2)
            np.testing.assert_allclose(o.H_out, outputs[0].H_out, atol=0.1, rtol=0.2)
