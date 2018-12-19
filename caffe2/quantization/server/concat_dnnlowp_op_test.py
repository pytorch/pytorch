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


class DNNLowPConcatOpTest(hu.HypothesisTestCase):
    @given(
        dim1=st.integers(128, 256),
        dim2=st.integers(128, 256),
        in_quantized=st.booleans(),
        out_quantized=st.booleans(),
        **hu.gcs_cpu_only
    )
    def test_dnnlowp_concat_int(self, dim1, dim2, in_quantized, out_quantized, gc, dc):

        # X has scale 1, so exactly represented after quantization
        min_ = -100
        max_ = min_ + 255
        X = np.round(np.random.rand(dim1, dim2) * (max_ - min_) + min_)
        X = X.astype(np.float32)
        X[0, 0] = min_
        X[0, 1] = max_

        # Y has scale 1/2, so exactly represented after quantization
        Y = np.round(np.random.rand(dim1, dim2) * 255 / 2 - 64)
        Y = Y.astype(np.float32)
        Y[0, 0] = -64
        Y[0, 1] = 127.0 / 2

        Output = collections.namedtuple("Output", ["Z", "op_type", "engine"])
        outputs = []

        op_engine_list = [
            ("Concat", ""),
            ("Concat", "DNNLOWP"),
            ("Int8Concat", "DNNLOWP"),
        ]

        for op_type, engine in op_engine_list:
            net = core.Net("test_net")

            do_quantize = "DNNLOWP" in engine and in_quantized
            do_dequantize = "DNNLOWP" in engine and out_quantized

            if do_quantize:
                quantize_x = core.CreateOperator(
                    "Quantize", ["X"], ["X_q"], engine=engine, device_option=gc
                )
                quantize_y = core.CreateOperator(
                    "Quantize", ["Y"], ["Y_q"], engine=engine, device_option=gc
                )
                net.Proto().op.extend([quantize_x, quantize_y])

            concat = core.CreateOperator(
                op_type,
                ["X_q", "Y_q"] if do_quantize else ["X", "Y"],
                ["Z_q" if do_dequantize else "Z", "split"],
                dequantize_output=not do_dequantize,
                engine=engine,
                device_option=gc,
                axis=0,
            )
            net.Proto().op.extend([concat])

            if do_dequantize:
                dequantize = core.CreateOperator(
                    "Dequantize", ["Z_q"], ["Z"], engine=engine, device_option=gc
                )
                net.Proto().op.extend([dequantize])

            self.ws.create_blob("X").feed(X, device_option=gc)
            self.ws.create_blob("Y").feed(Y, device_option=gc)
            self.ws.create_blob("split")
            self.ws.run(net)
            outputs.append(
                Output(Z=self.ws.blobs["Z"].fetch(), op_type=op_type, engine=engine)
            )

        check_quantized_results_close(outputs)
