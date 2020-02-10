from __future__ import absolute_import, division, print_function, unicode_literals

import collections

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, dyndep, workspace
from caffe2.quantization.server import utils as dnnlowp_utils
from dnnlowp_test_utils import (
    avoid_vpmaddubsw_overflow_fc,
    check_quantized_results_close,
    run_conv_or_fc,
)
from hypothesis import given


dyndep.InitOpsLibrary("//caffe2/caffe2/quantization/server:dnnlowp_ops")
workspace.GlobalInit(["caffe2", "--caffe2_omp_num_threads=11"])


class RowWiseDNNLowPFullyConnectedOpTest(hu.HypothesisTestCase):
    # correctness test with no quantization error in inputs
    @given(
        input_channels=st.sampled_from([3, 4, 5, 8, 16, 32]),
        output_channels=st.integers(2, 16),
        batch_size=st.integers(0, 16),
        in_quantized=st.booleans(),
        out_quantized=st.booleans(),
        prepack_weight=st.booleans(),
        **hu.gcs_cpu_only
    )
    def test_rowwise_dnnlowp_fully_connected_int(
        self,
        input_channels,
        output_channels,
        batch_size,
        in_quantized,
        out_quantized,
        prepack_weight,
        gc,
        dc,
    ):
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
        if batch_size != 0:
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

            if i % 2 == 0:
                W[i, :] = (W[i, :] - W_min) * 2 + W_min

        b = np.random.randn(output_channels).astype(np.float32)

        Output = collections.namedtuple("Output", ["Y", "op_type", "engine"])
        outputs = []

        op_engine_list = [
            ("FC", ""),
            ("FC", "DNNLOWP_ROWWISE"),
            ("FC", "DNNLOWP_ROWWISE_16"),
            ("Int8FC", "DNNLOWP_ROWWISE"),
        ]

        for op_type, engine in op_engine_list:
            init_net = core.Net("test_init_net")
            net = core.Net("test_net")

            do_quantize = "DNNLOWP" in engine and in_quantized
            do_dequantize = "DNNLOWP" in engine and out_quantized
            do_prepack_weight = engine == "DNNLOWP_ROWWISE" and prepack_weight

            if do_quantize:
                quantize = core.CreateOperator(
                    "Quantize", ["X"], ["X_q"], engine=engine, device_option=gc
                )
                net.Proto().op.extend([quantize])

            X_min = 0 if X.size == 0 else X.min()
            X_max = 0 if X.size == 0 else X.max()
            x_q_param = dnnlowp_utils.choose_quantization_params(X_min, X_max)

            if do_prepack_weight:
                inputs = ["W"]
                if do_dequantize:
                    inputs += ["b"]
                pack = core.CreateOperator(
                    "Int8FCPackWeight",
                    inputs,
                    ["W_packed"],
                    in_scale=x_q_param.scale,
                    engine=engine,
                )
                init_net.Proto().op.extend([pack])

            fc = core.CreateOperator(
                op_type,
                [
                    "X_q" if do_quantize else "X",
                    "W_packed" if do_prepack_weight else "W",
                    "b",
                ],
                ["Y_q" if do_dequantize else "Y"],
                dequantize_output=not do_dequantize,
                engine=engine,
                device_option=gc,
            )
            if do_prepack_weight:
                # When pre-packed quantized weight is provided, we can't rescale
                # the output dynamically by looking at the range of output of
                # each batch, so here we provide the range of output observed
                # from fp32 reference implementation
                dnnlowp_utils.add_quantization_param_args(fc, outputs[0][0])
            net.Proto().op.extend([fc])

            if do_dequantize:
                dequantize = core.CreateOperator(
                    "Dequantize", ["Y_q"], ["Y"], engine=engine, device_option=gc
                )
                net.Proto().op.extend([dequantize])

            run_conv_or_fc(
                self, init_net, net, X, W, b, op_type, engine, None, gc, outputs
            )

        check_quantized_results_close(outputs)
