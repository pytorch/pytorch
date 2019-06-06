from __future__ import absolute_import, division, print_function, unicode_literals

import collections

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, dyndep, utils, workspace
from caffe2.quantization.server import utils as dnnlowp_utils
from dnnlowp_test_utils import check_quantized_results_close
from hypothesis import given


dyndep.InitOpsLibrary("//caffe2/caffe2/quantization/server:dnnlowp_ops")
workspace.GlobalInit(["caffe2", "--caffe2_omp_num_threads=11"])


class DNNLowPOpGroupNormTest(hu.HypothesisTestCase):
    @given(
        N=st.integers(1, 4),
        G=st.integers(2, 4),
        K=st.integers(2, 12),
        H=st.integers(4, 16),
        W=st.integers(4, 16),
        order=st.sampled_from(["NCHW", "NHWC"]),
        in_quantized=st.booleans(),
        out_quantized=st.booleans(),
        weight_quantized=st.booleans(),
        **hu.gcs_cpu_only
    )
    def test_dnnlowp_group_norm(
        self,
        N,
        G,
        K,
        H,
        W,
        order,
        in_quantized,
        out_quantized,
        weight_quantized,
        gc,
        dc,
    ):
        C = G * K

        X = np.random.rand(N, C, H, W).astype(np.float32) * 5.0 - 1.0
        if order == "NHWC":
            X = utils.NCHW2NHWC(X)
        gamma = np.random.rand(C).astype(np.float32) * 2.0 - 1.0
        beta = np.random.randn(C).astype(np.float32) - 0.5

        Output = collections.namedtuple("Output", ["Y", "op_type", "engine"])
        outputs = []

        op_engine_list = [
            ("GroupNorm", ""),
            ("GroupNorm", "DNNLOWP"),
            ("Int8GroupNorm", "DNNLOWP"),
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
                int8_given_tensor_fill, gamma_q_param = dnnlowp_utils.create_int8_given_tensor_fill(
                    gamma, "gamma_q"
                )
                net.Proto().op.extend([int8_given_tensor_fill])

                X_q_param = dnnlowp_utils.choose_quantization_params(X.min(), X.max())
                int8_bias_tensor_fill = dnnlowp_utils.create_int8_bias_tensor_fill(
                    beta, "beta_q", X_q_param, gamma_q_param
                )
                net.Proto().op.extend([int8_bias_tensor_fill])

            group_norm = core.CreateOperator(
                op_type,
                [
                    "X_q" if do_quantize else "X",
                    "gamma_q" if do_quantize_weight else "gamma",
                    "beta_q" if do_quantize_weight else "beta",
                ],
                ["Y_q" if do_dequantize else "Y"],
                dequantize_output=0 if do_dequantize else 1,
                group=G,
                order=order,
                is_test=True,
                engine=engine,
                device_option=gc,
            )

            if do_quantize_weight:
                # When quantized weight is provided, we can't rescale the
                # output dynamically by looking at the range of output of each
                # batch, so here we provide the range of output observed from
                # fp32 reference implementation
                dnnlowp_utils.add_quantization_param_args(group_norm, outputs[0][0])

            net.Proto().op.extend([group_norm])

            if do_dequantize:
                dequantize = core.CreateOperator(
                    "Dequantize", ["Y_q"], ["Y"], engine=engine, device_option=gc
                )
                net.Proto().op.extend([dequantize])

            self.ws.create_blob("X").feed(X, device_option=gc)
            self.ws.create_blob("gamma").feed(gamma, device_option=gc)
            self.ws.create_blob("beta").feed(beta, device_option=gc)
            self.ws.run(net)
            outputs.append(
                Output(Y=self.ws.blobs["Y"].fetch(), op_type=op_type, engine=engine)
            )

        check_quantized_results_close(outputs, atol_scale=2.0)
