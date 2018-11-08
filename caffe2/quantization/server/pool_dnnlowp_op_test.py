from __future__ import absolute_import, division, print_function, unicode_literals

import collections

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, dyndep
from dnnlowp_test_utils import check_quantized_results_close
from hypothesis import assume, given


dyndep.InitOpsLibrary("//caffe2/caffe2/quantization/server:dnnlowp_ops")


class DNNLowPOpPoolTest(hu.HypothesisTestCase):
    @given(
        stride=st.integers(1, 3),
        pad=st.integers(0, 3),
        kernel=st.integers(1, 5),
        size=st.integers(1, 20),
        input_channels=st.integers(1, 3),
        batch_size=st.integers(1, 3),
        order=st.sampled_from({"NCHW", "NHWC"}),
        in_quantized=st.booleans(),
        **hu.gcs_cpu_only
    )
    def test_dnnlowp_max_pool(
        self,
        stride,
        pad,
        kernel,
        size,
        input_channels,
        batch_size,
        order,
        in_quantized,
        gc,
        dc,
    ):
        assume(kernel <= size)
        assume(pad < kernel)

        C = input_channels
        N = batch_size
        H = W = size

        min_ = -10
        max_ = 20
        if order == "NCHW":
            X = np.round(np.random.rand(N, C, H, W) * (max_ - min_) + min_)
        elif order == "NHWC":
            X = np.round(np.random.rand(N, H, W, C) * (max_ - min_) + min_)
        X = X.astype(np.float32)
        Output = collections.namedtuple("Output", ["Y", "op_type", "engine"])
        outputs = []

        op_engine_list = [
            ("MaxPool", ""),
            ("MaxPool", "DNNLOWP"),
            ("Int8MaxPool", "DNNLOWP"),
        ]

        for op_type, engine in op_engine_list:
            net = core.Net("test_net")

            do_quantize = "DNNLOWP" in engine and in_quantized

            if do_quantize:
                quantize = core.CreateOperator(
                    "Quantize", ["X"], ["X_q"], engine=engine, device_option=gc
                )
                net.Proto().op.extend([quantize])

            max_pool = core.CreateOperator(
                op_type,
                ["X_q" if do_quantize else "X"],
                ["Y_q" if engine == "DNNLOWP" else "Y"],
                stride=stride,
                kernel=kernel,
                pad=pad,
                order=order,
                engine=engine,
                device_option=gc,
            )
            net.Proto().op.extend([max_pool])

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

        # Y_i = max(X_j) so the only error is in quantization of inputs
        check_quantized_results_close(outputs, ref=X)

    @given(
        ndim=st.integers(2, 3),
        stride=st.integers(1, 1),
        pad=st.integers(0, 0),
        kernel=st.integers(1, 5),
        size=st.integers(2, 2),
        input_channels=st.integers(1, 1),
        batch_size=st.integers(2, 2),
        order=st.sampled_from({"NCHW", "NHWC"}),
        in_quantized=st.booleans(),
        **hu.gcs_cpu_only
    )
    def test_dnnlowp_average_pool(
        self,
        ndim,
        stride,
        pad,
        kernel,
        size,
        input_channels,
        batch_size,
        order,
        in_quantized,
        gc,
        dc,
    ):
        kernel = 2  # Only kernel size 2 is supported
        assume(kernel <= size)
        assume(pad < kernel)

        C = input_channels
        N = batch_size

        strides = (stride,) * ndim
        pads = (pad,) * (ndim * 2)
        kernels = (kernel,) * ndim
        sizes = (size,) * ndim

        # X has scale 1, so no input quantization error
        min_ = -100
        max_ = min_ + 255
        if order == "NCHW":
            X = np.round(np.random.rand(*((N, C) + sizes)) * (max_ - min_) + min_)
            X = X.astype(np.float32)
            X[(0,) * (ndim + 2)] = min_
            X[(0,) * (ndim + 1) + (1,)] = max_
        elif order == "NHWC":
            X = np.round(np.random.rand(*((N,) + sizes + (C,))) * (max_ - min_) + min_)
            X = X.astype(np.float32)
            X[(0,) * (ndim + 2)] = min_
            X[(0, 1,) + (0,) * ndim] = max_

        Output = collections.namedtuple("Output", ["Y", "op_type", "engine"])
        outputs = []

        op_engine_list = [
            ("AveragePool", ""),
            ("AveragePool", "DNNLOWP"),
            ("Int8AveragePool", "DNNLOWP"),
        ]

        for op_type, engine in op_engine_list:
            net = core.Net("test_net")

            do_quantize = "DNNLOWP" in engine and in_quantized

            if do_quantize:
                quantize = core.CreateOperator(
                    "Quantize", ["X"], ["X_q"], engine=engine, device_option=gc
                )
                net.Proto().op.extend([quantize])

            max_pool = core.CreateOperator(
                op_type,
                ["X_q" if do_quantize else "X"],
                ["Y_q" if engine == "DNNLOWP" else "Y"],
                strides=strides,
                kernels=kernels,
                pads=pads,
                order=order,
                engine=engine,
                device_option=gc,
            )
            net.Proto().op.extend([max_pool])

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

        check_quantized_results_close(outputs)
