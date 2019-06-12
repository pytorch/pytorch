from __future__ import absolute_import, division, print_function, unicode_literals

import collections

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
from caffe2.python import core, dyndep
from caffe2.quantization.server import utils as dnnlowp_utils
from dnnlowp_test_utils import (
    check_quantized_results_close,
    generate_conv_inputs,
    generate_convnd_inputs,
    nchw2nhwc,
    nhwc2nchw,
)
from hypothesis import given

dyndep.InitOpsLibrary("//caffe2/caffe2/quantization/server:dnnlowp_ops")


class DNNLowPOpConvDepthWiseTest(hu.HypothesisTestCase):
    @given(
        stride=st.integers(1, 2),
        size=st.integers(10, 16),
        # depthwise 3x3 fast path only works for a multiple of 8
        group=st.sampled_from([8, 32, 40]),
        batch_size=st.integers(1, 3),
        share_col_buffer=st.booleans(),
        preserve_activation_sparsity=st.booleans(),
        preserve_weight_sparsity=st.booleans(),
        relu=st.booleans(),
        **hu.gcs_cpu_only
    )
    def test_dnnlowp_depthwise_3x3_conv(
        self,
        stride,
        size,
        group,
        batch_size,
        share_col_buffer,
        preserve_activation_sparsity,
        preserve_weight_sparsity,
        relu,
        gc,
        dc,
    ):
        pad = 1
        kernel = 3
        dilation = 1
        input_channels_per_group = 1
        output_channels_per_group = 1
        order = "NHWC"

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

        Output = collections.namedtuple("Output", ["Y", "op_type", "engine", "order"])
        outputs = []

        if relu:
            op_engine_list = [("Conv", ""), ("ConvRelu", "DNNLOWP"), ("Int8ConvRelu", "DNNLOWP")]
        else:
            op_engine_list = [("Conv", ""), ("Conv", "DNNLOWP"), ("Int8Conv", "DNNLOWP")]

        for op_type, engine in op_engine_list:
            net = core.Net("test_net")

            do_quantize = "DNNLOWP" in engine
            do_dequantize = "DNNLOWP" in engine

            preserve_activation_sparsity_int = 1 if preserve_activation_sparsity else 0
            preserve_weight_sparsity_int = 1 if preserve_weight_sparsity else 0

            if do_quantize:
                quantize = core.CreateOperator(
                    "Quantize",
                    ["X"],
                    ["X_q"],
                    preserve_activation_sparsity=preserve_activation_sparsity_int,
                    engine=engine,
                    device_option=gc,
                )
                net.Proto().op.extend([quantize])

            conv = core.CreateOperator(
                op_type,
                ["X_q" if do_quantize else "X", "W", "b"],
                ["Y_q" if do_dequantize else "Y"],
                stride=stride,
                kernel=kernel,
                dilation=dilation,
                pad=pad,
                order=order,
                shared_buffer=(1 if share_col_buffer else 0),
                preserve_activation_sparsity=preserve_activation_sparsity_int,
                preserve_weight_sparsity=preserve_weight_sparsity_int,
                engine=engine,
                group=group,
                device_option=gc,
            )
            if do_dequantize:
                dnnlowp_utils.add_quantization_param_args(
                    conv, outputs[0][0], preserve_activation_sparsity
                )
            net.Proto().op.extend([conv])

            if do_dequantize:
                dequantize = core.CreateOperator(
                    "Dequantize", ["Y_q"], ["Y"], engine=engine, device_option=gc
                )
                net.Proto().op.extend([dequantize])
            elif relu:
                relu_op = core.CreateOperator(
                    "Relu", ["Y"], ["Y"], engine=engine, device_option=gc
                )
                net.Proto().op.extend([relu_op])

            self.ws.create_blob("X").feed(X, device_option=gc)
            self.ws.create_blob("W").feed(W, device_option=gc)
            self.ws.create_blob("b").feed(b, device_option=gc)
            self.ws.run(net)
            Y = self.ws.blobs["Y"].fetch()
            outputs.append(Output(Y=Y, op_type=op_type, engine=engine, order=order))

        check_quantized_results_close(outputs, symmetric=preserve_activation_sparsity)

    @given(
        stride=st.integers(1, 2),
        size=st.integers(10, 16),
        # depthwise 3x3 fast path only works for a multiple of 8
        group=st.sampled_from([8, 32, 40]),
        batch_size=st.integers(1, 3),
        fuse_relu=st.booleans(),
        share_col_buffer=st.booleans(),
        preserve_activation_sparsity=st.booleans(),
        preserve_weight_sparsity=st.booleans(),
        **hu.gcs_cpu_only
    )
    def test_dnnlowp_depthwise_3x3x3_conv(
        self,
        stride,
        size,
        group,
        batch_size,
        fuse_relu,
        share_col_buffer,
        preserve_activation_sparsity,
        preserve_weight_sparsity,
        gc,
        dc,
    ):
        pad = 1
        kernel = 3
        dilation = 1
        input_channels_per_group = 1
        output_channels_per_group = 1
        order = "NHWC"

        X, W, b = generate_convnd_inputs(
            (stride,) * 3,
            (pad,) * 3,
            (kernel,) * 3,
            (dilation,) * 3,
            (size,) * 3,
            group,
            input_channels_per_group,
            output_channels_per_group,
            batch_size,
            order,
            preserve_activation_sparsity=preserve_activation_sparsity,
            preserve_weight_sparsity=preserve_weight_sparsity,
        )

        Output = collections.namedtuple("Output", ["Y", "op_type", "engine", "order"])
        outputs = []

        op = "ConvRelu" if fuse_relu else "Conv"
        op_engine_list = [(op, ""), (op, "DNNLOWP"), ("Int8" + op, "DNNLOWP")]

        for op_type, engine in op_engine_list:
            net = core.Net("test_net")

            fall_back_to_NCHW = "DNNLOWP" not in engine

            if fall_back_to_NCHW:
                X_nchw = nhwc2nchw(X)
                W_nchw = nhwc2nchw(W)

            do_quantize = "DNNLOWP" in engine
            do_dequantize = "DNNLOWP" in engine

            preserve_activation_sparsity_int = 1 if preserve_activation_sparsity else 0
            preserve_weight_sparsity_int = 1 if preserve_weight_sparsity else 0

            if do_quantize:
                quantize = core.CreateOperator(
                    "Quantize",
                    ["X"],
                    ["X_q"],
                    preserve_activation_sparsity=preserve_activation_sparsity_int,
                    engine=engine,
                    device_option=gc,
                )
                net.Proto().op.extend([quantize])

            conv = core.CreateOperator(
                op_type,
                ["X_q" if do_quantize else "X", "W", "b"],
                ["Y_q" if do_dequantize else "Y"],
                strides=[stride] * 3,
                kernels=[kernel] * 3,
                dilations=[dilation] * 3,
                pads=[pad] * (3 * 2),
                order="NCHW" if fall_back_to_NCHW else order,
                shared_buffer=(1 if share_col_buffer else 0),
                preserve_activation_sparsity=preserve_activation_sparsity_int,
                preserve_weight_sparsity=preserve_weight_sparsity_int,
                engine=engine,
                group=group,
                device_option=gc,
            )
            if do_dequantize:
                dnnlowp_utils.add_quantization_param_args(
                    conv, outputs[0][0], preserve_activation_sparsity
                )
            net.Proto().op.extend([conv])

            if do_dequantize:
                dequantize = core.CreateOperator(
                    "Dequantize", ["Y_q"], ["Y"], engine=engine, device_option=gc
                )
                net.Proto().op.extend([dequantize])

            self.ws.create_blob("X").feed(
                X_nchw if fall_back_to_NCHW else X, device_option=gc
            )
            self.ws.create_blob("W").feed(
                W_nchw if fall_back_to_NCHW else W, device_option=gc
            )
            self.ws.create_blob("b").feed(b, device_option=gc)
            self.ws.run(net)
            Y = self.ws.blobs["Y"].fetch()
            if fall_back_to_NCHW:
                Y = nchw2nhwc(Y)
            outputs.append(Output(Y=Y, op_type=op_type, engine=engine, order=order))

        check_quantized_results_close(outputs, symmetric=preserve_activation_sparsity)
