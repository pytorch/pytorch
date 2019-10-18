from __future__ import absolute_import, division, print_function, unicode_literals

import collections

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
from caffe2.python import core, dyndep, workspace
from caffe2.quantization.server import utils as dnnlowp_utils
from dnnlowp_test_utils import (
    check_quantized_results_close,
    generate_conv_inputs,
    generate_convnd_inputs,
    run_conv_or_fc,
)
from hypothesis import given


dyndep.InitOpsLibrary("//caffe2/caffe2/quantization/server:dnnlowp_ops")
workspace.GlobalInit(["caffe2", "--caffe2_omp_num_threads=11"])


class DNNLowPOpConvDepthWiseTest(hu.HypothesisTestCase):
    @given(
        stride=st.integers(1, 2),
        size=st.integers(10, 16),
        # depthwise 3x3 fast path only works for a multiple of 8
        group=st.sampled_from([8, 24, 32]),
        batch_size=st.integers(0, 3),
        prepack_weight=st.booleans(),
        share_col_buffer=st.booleans(),
        preserve_activation_sparsity=st.booleans(),
        preserve_weight_sparsity=st.booleans(),
        quantize_groupwise=st.booleans(),
        relu=st.booleans(),
        **hu.gcs_cpu_only
    )
    def test_dnnlowp_depthwise_3x3_conv(
        self,
        stride,
        size,
        group,
        batch_size,
        prepack_weight,
        share_col_buffer,
        preserve_activation_sparsity,
        preserve_weight_sparsity,
        quantize_groupwise,
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
            groupwise_quantization=quantize_groupwise,
            preserve_activation_sparsity=preserve_activation_sparsity,
            preserve_weight_sparsity=preserve_weight_sparsity,
        )

        Output = collections.namedtuple("Output", ["Y", "op_type", "engine", "order"])
        outputs = []

        if relu:
            op_engine_list = [
                ("Conv", ""),
                ("ConvRelu", "DNNLOWP"),
                ("Int8ConvRelu", "DNNLOWP"),
            ]
        else:
            op_engine_list = [
                ("Conv", ""),
                ("Conv", "DNNLOWP"),
                ("Int8Conv", "DNNLOWP"),
            ]

        for op_type, engine in op_engine_list:
            init_net = core.Net("test_init_net")
            net = core.Net("test_net")

            do_quantize = "DNNLOWP" in engine
            do_dequantize = "DNNLOWP" in engine
            do_prepack_weight = engine == "DNNLOWP" and prepack_weight

            if do_quantize:
                quantize = core.CreateOperator(
                    "Quantize",
                    ["X"],
                    ["X_q"],
                    preserve_activation_sparsity=preserve_activation_sparsity,
                    engine=engine,
                    device_option=gc,
                )
                net.Proto().op.extend([quantize])

            if do_prepack_weight:
                X_min = 0 if X.size == 0 else X.min()
                X_max = 0 if X.size == 0 else X.max()
                x_q_param = dnnlowp_utils.choose_quantization_params(
                    X_min, X_max, preserve_activation_sparsity
                )
                inputs = ["W"]
                if do_dequantize:
                    inputs += ["b"]
                pack = core.CreateOperator(
                    "Int8ConvPackWeight",
                    inputs,
                    ["W_packed"],
                    group=group,
                    quantize_groupwise=quantize_groupwise,
                    preserve_weight_sparsity=preserve_weight_sparsity,
                    in_scale=x_q_param.scale,
                    engine=engine,
                )
                init_net.Proto().op.extend([pack])

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
                preserve_activation_sparsity=preserve_activation_sparsity,
                preserve_weight_sparsity=preserve_weight_sparsity,
                engine=engine,
                group=group,
                quantize_groupwise=quantize_groupwise,
                device_option=gc,
            )
            if do_dequantize or do_prepack_weight:
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

            run_conv_or_fc(
                self, init_net, net, X, W, b, op_type, engine, order, gc, outputs
            )

        check_quantized_results_close(outputs, symmetric=preserve_activation_sparsity)

    @given(
        stride_0=st.integers(1, 2),
        stride_1=st.integers(1, 2),
        stride_2=st.integers(1, 2),
        size=st.integers(5, 12),
        # depthwise 3x3x3 fast path only works for a multiple of 8
        group=st.sampled_from([8, 24, 32]),
        batch_size=st.integers(0, 2),
        prepack_weight=st.booleans(),
        fuse_relu=st.booleans(),
        share_col_buffer=st.booleans(),
        preserve_activation_sparsity=st.booleans(),
        preserve_weight_sparsity=st.booleans(),
        quantize_groupwise=st.just(True),
        **hu.gcs_cpu_only
    )
    def test_dnnlowp_depthwise_3x3x3_conv(
        self,
        stride_0,
        stride_1,
        stride_2,
        size,
        group,
        batch_size,
        prepack_weight,
        fuse_relu,
        share_col_buffer,
        preserve_activation_sparsity,
        preserve_weight_sparsity,
        quantize_groupwise,
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
            (stride_0, stride_1, stride_2),
            (pad,) * 3,
            (kernel,) * 3,
            (dilation,) * 3,
            (size,) * 3,
            group,
            input_channels_per_group,
            output_channels_per_group,
            batch_size,
            order,
            groupwise_quantization=quantize_groupwise,
            preserve_activation_sparsity=preserve_activation_sparsity,
            preserve_weight_sparsity=preserve_weight_sparsity,
        )

        Output = collections.namedtuple("Output", ["Y", "op_type", "engine", "order"])
        outputs = []

        op = "ConvRelu" if fuse_relu else "Conv"
        op_engine_list = [(op, ""), (op, "DNNLOWP"), ("Int8" + op, "DNNLOWP")]

        for op_type, engine in op_engine_list:
            init_net = core.Net("test_init_net")
            net = core.Net("test_net")

            do_quantize = "DNNLOWP" in engine
            do_dequantize = "DNNLOWP" in engine
            do_prepack_weight = engine == "DNNLOWP" and prepack_weight

            if do_quantize:
                quantize = core.CreateOperator(
                    "Quantize",
                    ["X"],
                    ["X_q"],
                    preserve_activation_sparsity=preserve_activation_sparsity,
                    engine=engine,
                    device_option=gc,
                )
                net.Proto().op.extend([quantize])

            if do_prepack_weight:
                X_min = 0 if X.size == 0 else X.min()
                X_max = 0 if X.size == 0 else X.max()
                x_q_param = dnnlowp_utils.choose_quantization_params(
                    X_min, X_max, preserve_activation_sparsity
                )
                inputs = ["W"]
                if do_dequantize:
                    inputs += ["b"]
                pack = core.CreateOperator(
                    "Int8ConvPackWeight",
                    inputs,
                    ["W_packed"],
                    group=group,
                    quantize_groupwise=quantize_groupwise,
                    preserve_weight_sparsity=preserve_weight_sparsity,
                    in_scale=x_q_param.scale,
                    engine=engine,
                )
                init_net.Proto().op.extend([pack])

            conv = core.CreateOperator(
                op_type,
                ["X_q" if do_quantize else "X", "W", "b"],
                ["Y_q" if do_dequantize else "Y"],
                strides=[stride_0, stride_1, stride_2],
                kernels=[kernel] * 3,
                dilations=[dilation] * 3,
                pads=[pad] * (3 * 2),
                order=order,
                shared_buffer=(1 if share_col_buffer else 0),
                preserve_activation_sparsity=preserve_activation_sparsity,
                preserve_weight_sparsity=preserve_weight_sparsity,
                engine=engine,
                group=group,
                quantize_groupwise=quantize_groupwise,
                device_option=gc,
            )
            if do_dequantize or do_prepack_weight:
                dnnlowp_utils.add_quantization_param_args(
                    conv, outputs[0][0], preserve_activation_sparsity
                )
            net.Proto().op.extend([conv])

            if do_dequantize:
                dequantize = core.CreateOperator(
                    "Dequantize", ["Y_q"], ["Y"], engine=engine, device_option=gc
                )
                net.Proto().op.extend([dequantize])

            run_conv_or_fc(
                self, init_net, net, X, W, b, op_type, engine, order, gc, outputs
            )

        check_quantized_results_close(outputs, symmetric=preserve_activation_sparsity)
