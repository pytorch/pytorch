from __future__ import absolute_import, division, print_function, unicode_literals

import collections

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, dyndep, workspace
from dnnlowp_test_utils import check_quantized_results_close
from hypothesis import given


dyndep.InitOpsLibrary("//caffe2/caffe2/quantization/server:dnnlowp_ops")
workspace.GlobalInit(["caffe2", "--caffe2_omp_num_threads=11"])


class DNNLowPElementwiseLinearOpTest(hu.HypothesisTestCase):
    @given(
        N=st.integers(32, 256),
        D=st.integers(32, 256),
        in_quantized=st.booleans(),
        out_quantized=st.booleans(),
        **hu.gcs_cpu_only
    )
    def test_dnnlowp_elementwise_linear_int(
        self, N, D, in_quantized, out_quantized, gc, dc
    ):
        # All inputs have scale 1, so exactly represented after quantization
        min_ = -100
        max_ = min_ + 255
        X = np.round(np.random.rand(N, D) * (max_ - min_) + min_)
        X = X.astype(np.float32)
        X[0, 0] = min_
        X[0, 1] = max_

        a = np.round(np.random.rand(D) * 255 - 128).astype(np.float32)
        a[0] = -128
        a[1] = 127

        b = np.round(np.random.rand(D) * 255 - 128).astype(np.float32)
        b[0] = -128
        b[1] = 127

        Output = collections.namedtuple("Output", ["Y", "op_type", "engine"])
        outputs = []

        op_engine_list = [
            ("ElementwiseLinear", ""),
            ("ElementwiseLinear", "DNNLOWP"),
            ("Int8ElementwiseLinear", "DNNLOWP"),
        ]

        for op_type, engine in op_engine_list:
            net = core.Net("test_net")

            do_quantize = "DNNLOWP" in engine and in_quantized
            do_dequantize = "DNNLOWP" in engine and out_quantized

            if do_quantize:
                quantize = core.CreateOperator(
                    "Quantize", ["X"], ["X_q"], engine=engine, device_option=gc
                )
                net.Proto().op.extend([quantize])

            eltwise_linear = core.CreateOperator(
                op_type,
                ["X_q" if do_quantize else "X", "a", "b"],
                ["Y_q" if do_dequantize else "Y"],
                dequantize_output=not do_dequantize,
                engine=engine,
                device_option=gc,
            )
            net.Proto().op.extend([eltwise_linear])

            if do_dequantize:
                dequantize = core.CreateOperator(
                    "Dequantize", ["Y_q"], ["Y"], engine=engine, device_option=gc
                )
                net.Proto().op.extend([dequantize])

            self.ws.create_blob("X").feed(X, device_option=gc)
            self.ws.create_blob("a").feed(a, device_option=gc)
            self.ws.create_blob("b").feed(b, device_option=gc)
            self.ws.run(net)
            outputs.append(
                Output(Y=self.ws.blobs["Y"].fetch(), op_type=op_type, engine=engine)
            )

        check_quantized_results_close(outputs)
