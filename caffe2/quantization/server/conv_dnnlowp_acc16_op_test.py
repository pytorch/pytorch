from __future__ import absolute_import, division, print_function, unicode_literals

import collections

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, dyndep
from caffe2.python.fb import hardcode_scale_zp
from caffe2.quantization.server import utils as dnnlowp_utils
from dnnlowp_test_utils import (
    check_quantized_results_close,
    generate_conv_inputs,
    nchw2nhwc,
    nhwc2nchw,
)
from hypothesis import assume, given


dyndep.InitOpsLibrary("//caffe2/caffe2/quantization/server:dnnlowp_ops")


class DNNLowPOpConvAcc16OpTest(hu.HypothesisTestCase):
    # correctness test with no quantization error in inputs
    @given(
        stride=st.integers(1, 2),
        pad=st.integers(0, 2),
        kernel=st.integers(1, 5),
        dilation=st.integers(1, 2),
        size=st.integers(10, 16),
        group=st.integers(1, 4),
        input_channels_per_group=st.sampled_from([2, 3, 4, 5, 8, 16, 32]),
        output_channels_per_group=st.integers(2, 16),
        batch_size=st.integers(1, 3),
        order=st.sampled_from(["NCHW", "NHWC"]),
        in_quantized=st.booleans(),
        out_quantized=st.booleans(),
        weight_quantized=st.booleans(),
        share_col_buffer=st.booleans(),
        preserve_activation_sparsity=st.booleans(),
        preserve_weight_sparsity=st.booleans(),
        **hu.gcs_cpu_only
    )
    def test_dnnlowp_conv_acc16_int(
        self,
        stride,
        pad,
        kernel,
        dilation,
        size,
        group,
        input_channels_per_group,
        output_channels_per_group,
        batch_size,
        order,
        in_quantized,
        out_quantized,
        weight_quantized,
        share_col_buffer,
        preserve_activation_sparsity,
        preserve_weight_sparsity,
        gc,
        dc,
    ):
        if group > 1:
            dilation = 1
        assume(size >= dilation * (kernel - 1) + 1)

        input_channels = input_channels_per_group * group
        output_channels = output_channels_per_group * group

        # X and W have scale 1, so exactly represented after quantization
        # This was made sure by having at least one 0 and one 255 for unsigned
        # 8-bit tensors, and at least one -128 and one 127 for signed 8-bit
        # tensors.
        # Since fbgemm_acc16 accumulates to 16-bit, To avoid overflow, we use
        # small numbers except for those 0, 255, -128, and 127, for this test
        # We also make sure 255, -128, or 127 are not multiplied together by
        # putting them in different input channels and the corresponding input
        # channel in other matrix is 0.
        # For example, we put 255 in input channel 1 in X, so we make the
        # corresponding input channel in W all zeros.
        X_min = 0 if preserve_activation_sparsity else -77
        X_max = X_min + 255
        X = np.random.rand(batch_size, size, size, input_channels) * 4 + X_min
        X = np.round(X).astype(np.float32)
        X[..., 0] = X_min
        X[0, 0, 0, 1] = X_max

        if preserve_weight_sparsity:
            W_min = -128
            W_max = 100
        else:
            W_min = -100
            W_max = W_min + 255
        W = (
            np.random.rand(output_channels, kernel, kernel, input_channels_per_group)
            * 4
            - 2
            + W_min
            + 128
        )
        W = np.round(W).astype(np.float32)
        W[0, 0, 0, 0] = W_min
        W[1, 0, 0, 0] = W_max
        W[..., 1] = W_min + 128  # "zeros"

        if order == "NCHW":
            X = nhwc2nchw(X)
            W = nhwc2nchw(W)

        # No input quantization error in bias
        b = np.round(np.random.randn(output_channels)).astype(np.float32)

        Output = collections.namedtuple("Output", ["Y", "op_type", "engine", "order"])
        outputs = []

        op_engine_list = [
            ("Conv", ""),
            ("Conv", "DNNLOWP_ACC16"),
            ("Int8Conv", "DNNLOWP_ACC16"),
        ]

        for op_type, engine in op_engine_list:
            net = core.Net("test_net")

            do_quantize = "DNNLOWP" in engine and in_quantized
            do_dequantize = "DNNLOWP" in engine and out_quantized
            do_quantize_weight = (
                "DNNLOWP" in engine and weight_quantized and len(outputs) > 0
            )

            if do_quantize:
                quantize = core.CreateOperator(
                    "Quantize",
                    ["X"],
                    ["X_q"],
                    preserve_activation_sparsity=preserve_activation_sparsity,
                    engine="DNNLOWP",
                    device_option=gc,
                )
                net.Proto().op.extend([quantize])

            if do_quantize_weight:
                int8_given_tensor_fill, w_q_param = dnnlowp_utils.create_int8_given_tensor_fill(
                    W, "W_q", preserve_weight_sparsity
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

            conv = core.CreateOperator(
                op_type,
                [
                    "X_q" if do_quantize else "X",
                    "W_q" if do_quantize_weight else "W",
                    "b_q" if do_quantize_weight else "b",
                ],
                ["Y_q" if do_dequantize else "Y"],
                stride=stride,
                kernel=kernel,
                dilation=dilation,
                pad=pad,
                order=order,
                dequantize_output=not do_dequantize,
                shared_buffer=(1 if share_col_buffer else 0),
                preserve_activation_sparsity=preserve_activation_sparsity,
                preserve_weight_sparsity=preserve_weight_sparsity,
                engine=engine,
                group=group,
                device_option=gc,
            )
            if do_dequantize or do_quantize_weight:
                # When quantized weight is provided, we can't rescale the
                # output dynamically by looking at the range of output of each
                # batch, so here we provide the range of output observed from
                # fp32 reference implementation
                dnnlowp_utils.add_quantization_param_args(
                    conv, outputs[0][0], preserve_activation_sparsity
                )
            net.Proto().op.extend([conv])

            if do_dequantize:
                dequantize = core.CreateOperator(
                    "Dequantize", ["Y_q"], ["Y"], engine="DNNLOWP", device_option=gc
                )
                net.Proto().op.extend([dequantize])

            self.ws.create_blob("X").feed(X, device_option=gc)
            self.ws.create_blob("W").feed(W, device_option=gc)
            self.ws.create_blob("b").feed(b, device_option=gc)
            self.ws.run(net)
            Y = self.ws.blobs["Y"].fetch()
            outputs.append(Output(Y=Y, op_type=op_type, engine=engine, order=order))

        check_quantized_results_close(outputs, symmetric=preserve_activation_sparsity)

    @given(
        stride=st.integers(1, 2),
        pad=st.integers(0, 2),
        kernel=st.integers(1, 5),
        dilation=st.integers(1, 2),
        size=st.integers(10, 16),
        group=st.integers(1, 4),
        input_channels_per_group=st.sampled_from([2, 3, 4, 5, 8, 16, 32]),
        output_channels_per_group=st.integers(2, 16),
        batch_size=st.integers(1, 3),
        order=st.sampled_from(["NHWC"]),
        in_quantized=st.booleans(),
        out_quantized=st.booleans(),
        weight_quantized=st.booleans(),
        nbits_in_non_outlier=st.sampled_from((0, 6)),
        share_col_buffer=st.booleans(),
        preserve_activation_sparsity=st.booleans(),
        preserve_weight_sparsity=st.booleans(),
        **hu.gcs_cpu_only
    )
    def test_dnnlowp_conv_acc16_outlier(
        self,
        stride,
        pad,
        kernel,
        dilation,
        size,
        group,
        input_channels_per_group,
        output_channels_per_group,
        batch_size,
        order,
        in_quantized,
        out_quantized,
        weight_quantized,
        nbits_in_non_outlier,
        share_col_buffer,
        preserve_activation_sparsity,
        preserve_weight_sparsity,
        gc,
        dc,
    ):
        if group > 1:
            dilation = 1
        assume(size >= dilation * (kernel - 1) + 1)

        input_channels = input_channels_per_group * group
        output_channels = output_channels_per_group * group

        if nbits_in_non_outlier == 0:
            X, W, b = generate_conv_inputs(
                stride,
                pad,
                kernel,
                dilation,
                size,
                group,
                input_channels_per_group,
                output_channels_per_group,
                batch_size,
                order,
                preserve_activation_sparsity=preserve_activation_sparsity,
                preserve_weight_sparsity=preserve_weight_sparsity,
            )
        else:
            X_min = 0 if preserve_activation_sparsity else -77
            X_max = X_min + 255
            X = np.random.rand(batch_size, size, size, input_channels) * 4 + X_min
            X = np.round(X).astype(np.float32)
            X[..., 0] = X_min
            X[0, 0, 0, 1] = X_max

            if preserve_weight_sparsity:
                W_min = -128
                W_max = 100
            else:
                W_min = -100
                W_max = W_min + 255
            W = (
                np.random.rand(
                    output_channels, kernel, kernel, input_channels_per_group
                )
                * 4
                - 2
                + W_min
                + 128
            )
            W = np.round(W).astype(np.float32)
            W[0, 0, 0, 0] = W_min
            W[1, 0, 0, 0] = W_max
            W[..., 1] = W_min + 128

            if order == "NCHW":
                X = nhwc2nchw(X)
                W = nhwc2nchw(W)

            # No input quantization error in bias
            b = np.round(np.random.randn(output_channels)).astype(np.float32)

        Output = collections.namedtuple("Output", ["Y", "op_type", "engine", "order"])
        outputs = []

        op_engine_list = [
            ("Conv", ""),
            ("Conv", "DNNLOWP_ACC16"),
            ("Int8Conv", "DNNLOWP_ACC16"),
        ]

        for op_type, engine in op_engine_list:
            net = core.Net("test_net")

            do_quantize = "DNNLOWP" in engine and in_quantized
            do_dequantize = "DNNLOWP" in engine and out_quantized
            do_quantize_weight = "DNNLOWP" in engine and weight_quantized

            if do_quantize:
                quantize = core.CreateOperator(
                    "Quantize",
                    ["X"],
                    ["X_q"],
                    preserve_activation_sparsity=preserve_activation_sparsity,
                    engine="DNNLOWP",
                    device_option=gc,
                )
                net.Proto().op.extend([quantize])

            if do_quantize_weight:
                int8_given_tensor_fill, w_q_param = dnnlowp_utils.create_int8_given_tensor_fill(
                    W, "W_q", preserve_weight_sparsity
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

            conv = core.CreateOperator(
                op_type,
                [
                    "X_q" if do_quantize else "X",
                    "W_q" if do_quantize_weight else "W",
                    "b_q" if do_quantize_weight else "b",
                ],
                ["Y_q" if do_dequantize else "Y"],
                stride=stride,
                kernel=kernel,
                dilation=dilation,
                pad=pad,
                order=order,
                dequantize_output=not do_dequantize,
                nbits_in_non_outlier=nbits_in_non_outlier,
                shared_buffer=(1 if share_col_buffer else 0),
                preserve_activation_sparsity=preserve_activation_sparsity,
                preserve_weight_sparsity=preserve_weight_sparsity,
                engine=engine,
                group=group,
                device_option=gc,
            )
            if do_dequantize or do_quantize_weight:
                # When quantized weight is provided, we can't rescale the
                # output dynamically by looking at the range of output of each
                # batch, so here we provide the range of output observed from
                # fp32 reference implementation
                dnnlowp_utils.add_quantization_param_args(
                    conv, outputs[0][0], preserve_activation_sparsity
                )
            net.Proto().op.extend([conv])

            if do_dequantize:
                dequantize = core.CreateOperator(
                    "Dequantize", ["Y_q"], ["Y"], engine="DNNLOWP", device_option=gc
                )
                net.Proto().op.extend([dequantize])

            self.ws.create_blob("X").feed(X, device_option=gc)
            self.ws.create_blob("W").feed(W, device_option=gc)
            self.ws.create_blob("b").feed(b, device_option=gc)
            self.ws.run(net)
            Y = self.ws.blobs["Y"].fetch()
            outputs.append(Output(Y=Y, op_type=op_type, engine=engine, order=order))

        check_quantized_results_close(outputs, symmetric=preserve_activation_sparsity)
