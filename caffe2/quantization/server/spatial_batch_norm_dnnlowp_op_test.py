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


class DNNLowPOpSpatialBNTest(hu.HypothesisTestCase):
    # correctness test with no quantization error in inputs
    @given(
        size=st.integers(10, 16),
        input_channels=st.integers(2, 16),
        output_channels=st.integers(2, 16),
        batch_size=st.integers(1, 3),
        order=st.sampled_from(["NCHW", "NHWC"]),
        in_quantized=st.booleans(),
        out_quantized=st.booleans(),
        **hu.gcs_cpu_only
    )
    def test_dnnlowp_spatial_bn_int(
        self,
        size,
        input_channels,
        output_channels,
        batch_size,
        order,
        in_quantized,
        out_quantized,
        gc,
        dc,
    ):
        X_min = -77
        X_max = X_min + 255
        X = np.round(np.random.rand(batch_size, size, size, input_channels)).astype(
            np.float32
        )
        X[0, 0, 0, 0] = X_min
        X[0, 0, 0, 1] = X_max

        epsilon = np.abs(np.random.rand())
        scale = np.random.rand(input_channels).astype(np.float32)
        bias = np.random.rand(input_channels).astype(np.float32)
        mean = np.random.rand(input_channels).astype(np.float32)
        var = np.random.rand(input_channels).astype(np.float32)

        if order == "NCHW":
            X = utils.NHWC2NCHW(X)

        Output = collections.namedtuple("Output", ["Y", "op_type", "engine"])
        outputs = []

        op_engine_list = [
            ("SpatialBN", ""),
            ("SpatialBN", "DNNLOWP"),
            ("Int8SpatialBN", "DNNLOWP"),
        ]

        for op_type, engine in op_engine_list:
            net = core.Net("test_net")

            do_quantize = "DNNLOWP" in engine and in_quantized
            do_dequantize = "DNNLOWP" in engine and out_quantized

            if do_quantize:
                quantize = core.CreateOperator(
                    "Quantize", ["X"], ["X_q"], engine=engine
                )
                net.Proto().op.extend([quantize])

            bn = core.CreateOperator(
                op_type,
                ["X_q" if do_quantize else "X", "scale", "bias", "mean", "var"],
                ["Y_q" if do_dequantize else "Y"],
                is_test=True,
                epsilon=epsilon,
                order=order,
                engine=engine,
                dequantize_output=not do_dequantize,
            )
            net.Proto().op.extend([bn])
            if "DNNLOWP" in engine:
                dnnlowp_utils.add_quantization_param_args(bn, outputs[0][0])

            if do_dequantize:
                dequantize = core.CreateOperator(
                    "Dequantize", ["Y_q"], ["Y"], engine=engine
                )
                net.Proto().op.extend([dequantize])

            self.ws.create_blob("X").feed(X, device_option=gc)
            self.ws.create_blob("scale").feed(scale, device_option=gc)
            self.ws.create_blob("bias").feed(bias, device_option=gc)
            self.ws.create_blob("mean").feed(mean, device_option=gc)
            self.ws.create_blob("var").feed(var, device_option=gc)
            self.ws.run(net)
            outputs.append(
                Output(Y=self.ws.blobs["Y"].fetch(), op_type=op_type, engine=engine)
            )

        check_quantized_results_close(outputs)
