from __future__ import absolute_import, division, print_function, unicode_literals

import collections

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, dyndep
from dnnlowp_test_utils import (
    avoid_vpmaddubsw_overflow_fc,
    check_quantized_results_close,
)
from hypothesis import given


dyndep.InitOpsLibrary("//caffe2/caffe2/quantization/server:dnnlowp_ops")


class RowWiseDNNLowPFullyConnectedOpTest(hu.HypothesisTestCase):
    # correctness test with no quantization error in inputs
    @given(
        input_channels=st.sampled_from([3, 4, 5, 8, 16, 32]),
        output_channels=st.integers(2, 16),
        batch_size=st.integers(1, 16),
        in_quantized=st.booleans(),
        out_quantized=st.booleans(),
        **hu.gcs_cpu_only
    )
    def test_rowwise_dnnlowp_fully_connected_int(
        self,
        input_channels,
        output_channels,
        batch_size,
        in_quantized,
        out_quantized,
        gc,
        dc,
    ):
        print("@given M ", batch_size, " K ", input_channels, " N ", output_channels)
        print("@given in_quantized ", in_quantized, " out_quantized ", out_quantized)

        # X has scale 1, so exactly represented after quantization
        X_min = -77
        X_max = X_min + 255
        X = np.round(
            np.random.rand(batch_size, input_channels) * (X_max - X_min) + X_min
        )
        X = X.astype(np.float32)
        # input channels 0 and 1 are all X_min to avoid overflow from vpmaddubsw
        # when multiplied with W_min and W_max
        X[:, 0:2] = X_min
        X[0, 2] = X_max

        # Each row of W has scale 1 but with different offset, so row-wise
        # quantization shouldn't have any input quantization error.
        W = np.zeros((output_channels, input_channels))
        W = W.astype(np.float32)
        for i in range(output_channels):
            W_min = -100 + i
            W_max = W_min + 255
            W[i, :] = np.round(np.random.rand(input_channels) * (W_max - W_min) + W_min)
            W[i, 0] = W_min
            W[i, 1] = W_max

            # Make sure we won't have overflows from vpmaddubsw instruction used in
            # fbgemm
            avoid_vpmaddubsw_overflow_fc(
                batch_size,
                input_channels,
                1,
                X,
                X_min,
                X_max,
                W[i : i + 1,],
                W_min,
                W_max,
            )

        b = np.random.randn(output_channels).astype(np.float32)

        Output = collections.namedtuple("Output", ["Y", "op_type", "engine"])
        outputs = []

        op_engine_list = [
            ("FC", ""),
            ("FC", "DNNLOWP_ROWWISE"),
            ("FC", "DNNLOWP_ROWWISE_16"),
            ("Int8FC", "DNNLOWP_ROWWISE"),
            ("Int8FCRowWise", "DNNLOWP"),
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

            fc = core.CreateOperator(
                op_type,
                ["X_q" if do_quantize else "X", "W", "b"],
                ["Y_q" if do_dequantize else "Y"],
                dequantize_output=not do_dequantize,
                engine=engine,
                device_option=gc,
            )
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
