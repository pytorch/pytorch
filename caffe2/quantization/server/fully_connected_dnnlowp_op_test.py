from __future__ import absolute_import, division, print_function, unicode_literals

import collections

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, dyndep
from caffe2.python.fb import hardcode_scale_zp
from caffe2.quantization.server import utils as dnnlowp_utils
from dnnlowp_test_utils import (
    avoid_vpmaddubsw_overflow_fc,
    check_quantized_results_close,
)
from hypothesis import given

dyndep.InitOpsLibrary("//caffe2/caffe2/quantization/server:dnnlowp_ops")


class DNNLowPFullyConnectedOpTest(hu.HypothesisTestCase):
    # correctness test with no quantization error in inputs
    @given(
        input_channels=st.sampled_from([3, 4, 5, 8, 16, 32]),
        output_channels=st.integers(2, 16),
        batch_size=st.integers(1, 16),
        in_quantized=st.booleans(),
        out_quantized=st.booleans(),
        weight_quantized=st.booleans(),
        **hu.gcs_cpu_only
    )
    def test_dnnlowp_fully_connected_int(
        self,
        input_channels,
        output_channels,
        batch_size,
        in_quantized,
        out_quantized,
        weight_quantized,
        gc,
        dc,
    ):
        # X and W have scale 1, so exactly represented after quantization
        X_min = -77
        X_max = X_min + 255
        X = np.round(
            np.random.rand(batch_size, input_channels) * (X_max - X_min) + X_min
        )
        X = X.astype(np.float32)
        # input channels 0 and 1 are all X_min to avoid overflow from vpmaddubsw
        # when multiplied with W_min and W_max
        X[:, 0] = X_min
        X[0, 1] = X_max

        W_min = -100
        W_max = W_min + 255
        W = np.round(
            np.random.rand(output_channels, input_channels) * (W_max - W_min) + W_min
        )
        W = W.astype(np.float32)
        W[0, 0] = W_min
        W[1, 0] = W_max

        # Make sure we won't have overflows from vpmaddubsw instruction used in
        # fbgemm
        avoid_vpmaddubsw_overflow_fc(
            batch_size,
            input_channels,
            output_channels,
            X,
            X_min,
            X_max,
            W,
            W_min,
            W_max,
        )

        b = np.random.randn(output_channels).astype(np.float32)

        Output = collections.namedtuple("Output", ["Y", "op_type", "engine"])
        outputs = []

        op_engine_list = [
            ("FC", ""),
            ("FC", "DNNLOWP"),
            ("FC", "DNNLOWP_16"),
            ("Int8FC", "DNNLOWP"),
        ]

        for op_type, engine in op_engine_list:
            net = core.Net("test_net")

            do_quantize = "DNNLOWP" in engine and in_quantized
            do_dequantize = "DNNLOWP" in engine and out_quantized
            do_quantize_weight = (
                engine == "DNNLOWP" and weight_quantized and len(outputs) > 0
            )

            if do_quantize:
                quantize = core.CreateOperator(
                    "Quantize", ["X"], ["X_q"], engine=engine, device_option=gc
                )
                net.Proto().op.extend([quantize])

            if do_quantize_weight:
                int8_given_tensor_fill, w_q_param = dnnlowp_utils.create_int8_given_tensor_fill(
                    W, "W_q"
                )
                net.Proto().op.extend([int8_given_tensor_fill])

                # Bias
                x_q_param = hardcode_scale_zp.choose_quantization_params(
                    X.min(), X.max()
                )
                int8_bias_tensor_fill = dnnlowp_utils.create_int8_bias_tensor_fill(
                    b, "b_q", x_q_param, w_q_param
                )
                net.Proto().op.extend([int8_bias_tensor_fill])

            fc = core.CreateOperator(
                op_type,
                [
                    "X_q" if do_quantize else "X",
                    "W_q" if do_quantize_weight else "W",
                    "b_q" if do_quantize_weight else "b",
                ],
                ["Y_q" if do_dequantize else "Y"],
                dequantize_output=not do_dequantize,
                engine=engine,
                device_option=gc,
            )
            if do_quantize_weight:
                # When quantized weight is provided, we can't rescale the
                # output dynamically by looking at the range of output of each
                # batch, so here we provide the range of output observed from
                # fp32 reference implementation
                dnnlowp_utils.add_quantization_param_args(fc, outputs[0][0])
            net.Proto().op.extend([fc])

            if do_dequantize:
                dequantize = core.CreateOperator(
                    "Dequantize", ["Y_q"], ["Y"], engine=engine, device_option=gc
                )
                net.Proto().op.extend([dequantize])

            self.ws.create_blob("X").feed(X, device_option=gc)
            self.ws.create_blob("W").feed(W, device_option=gc)
            self.ws.create_blob("b").feed(b, device_option=gc)
            self.ws.run(net)
            outputs.append(
                Output(Y=self.ws.blobs["Y"].fetch(), op_type=op_type, engine=engine)
            )

        check_quantized_results_close(outputs)
