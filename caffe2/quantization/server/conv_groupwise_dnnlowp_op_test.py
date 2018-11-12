from __future__ import absolute_import, division, print_function, unicode_literals

import collections

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
from caffe2.python import core, dyndep
from caffe2.quantization.server import utils as dnnlowp_utils
from dnnlowp_test_utils import (
    check_quantized_results_close,
    generate_conv_inputs,
    nchw2nhwc,
    nhwc2nchw,
)
from hypothesis import given


dyndep.InitOpsLibrary("//caffe2/caffe2/quantization/server:dnnlowp_ops")


class GroupWiseDNNLowPOpConvTest(hu.HypothesisTestCase):
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
        preserve_activation_sparsity=st.booleans(),
        preserve_weight_sparsity=st.booleans(),
        **hu.gcs_cpu_only
    )
    def test_groupwise_dnnlowp_conv_int(
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
        preserve_activation_sparsity,
        preserve_weight_sparsity,
        gc,
        dc,
    ):
        if group > 1:
            dilation = 1

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
            groupwise_quantization=True,
            preserve_activation_sparsity=preserve_activation_sparsity,
            preserve_weight_sparsity=preserve_weight_sparsity,
        )

        Output = collections.namedtuple("Output", ["Y", "op_type", "engine", "order"])
        outputs = []

        op_engine_list = [
            ("Conv", ""),
            ("Conv", "DNNLOWP"),
            ("Conv", "DNNLOWP_16"),
            ("Int8Conv", "DNNLOWP"),
        ]

        for op_type, engine in op_engine_list:
            net = core.Net("test_net")

            do_quantize = "DNNLOWP" in engine and in_quantized
            do_dequantize = "DNNLOWP" in engine and out_quantized

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

            conv = core.CreateOperator(
                op_type,
                ["X_q" if do_quantize else "X", "W", "b"],
                ["Y_q" if do_dequantize else "Y"],
                stride=stride,
                kernel=kernel,
                dilation=dilation,
                pad=pad,
                order=order,
                dequantize_output=not do_dequantize,
                preserve_activation_sparsity=preserve_activation_sparsity,
                preserve_weight_sparsity=preserve_weight_sparsity,
                engine=engine,
                group=group,
                quantize_groupwise=1,
                device_option=gc,
            )
            if do_dequantize:
                # groupwise quantization only works with static quantization
                # so we need to set quantization parameters
                dnnlowp_utils.add_quantization_param_args(
                    conv, outputs[0][0], preserve_activation_sparsity
                )
            net.Proto().op.extend([conv])

            if do_dequantize:
                dequantize = core.CreateOperator(
                    "Dequantize",
                    ["Y_q"],
                    ["Y"],
                    preserve_activation_sparsity=preserve_activation_sparsity,
                    engine=engine,
                    device_option=gc,
                )
                net.Proto().op.extend([dequantize])

            self.ws.create_blob("X").feed(X, device_option=gc)
            self.ws.create_blob("W").feed(W, device_option=gc)
            self.ws.create_blob("b").feed(b, device_option=gc)
            self.ws.run(net)
            Y = self.ws.blobs["Y"].fetch()
            outputs.append(Output(Y=Y, op_type=op_type, engine=engine, order=order))

        check_quantized_results_close(outputs, symmetric=preserve_activation_sparsity)

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
        **hu.gcs_cpu_only
    )
    def test_groupwise_dnnlowp_conv_relu_int(
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
        gc,
        dc,
    ):
        if group > 1:
            dilation = 1

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
            True,  # group-wise
        )

        Output = collections.namedtuple("Output", ["Y", "op_type", "engine", "order"])
        outputs = []

        op_engine_list = [
            ("Conv", ""),
            ("ConvRelu", "DNNLOWP"),
            ("ConvRelu", "DNNLOWP_16"),
            ("Int8ConvRelu", "DNNLOWP"),
        ]

        for op_type, engine in op_engine_list:
            net = core.Net("test_net")

            if "DNNLOWP" in engine:
                quantize = core.CreateOperator(
                    "Quantize", ["X"], ["X_q"], engine=engine, device_option=gc
                )
                net.Proto().op.extend([quantize])

                conv = core.CreateOperator(
                    op_type,
                    ["X_q", "W", "b"],
                    ["Y_q"],
                    stride=stride,
                    kernel=kernel,
                    dilation=dilation,
                    pad=pad,
                    order=order,
                    engine=engine,
                    group=group,
                    quantize_groupwise=1,
                    device_option=gc,
                )
                # groupwise quantization only works with static quantization
                # so we need to set quantization parameters
                dnnlowp_utils.add_quantization_param_args(conv, outputs[0][0])
                net.Proto().op.extend([conv])

                dequantize = core.CreateOperator(
                    "Dequantize", ["Y_q"], ["Y"], engine=engine, device_option=gc
                )
                net.Proto().op.extend([dequantize])
            else:
                conv = core.CreateOperator(
                    op_type,
                    ["X", "W", "b"],
                    ["Y"],
                    stride=stride,
                    kernel=kernel,
                    dilation=dilation,
                    pad=pad,
                    order=order,
                    engine=engine,
                    group=group,
                    device_option=gc,
                )
                net.Proto().op.extend([conv])

                relu = core.CreateOperator(
                    "Relu", ["Y"], ["Y"], engine=engine, device_option=gc
                )
                net.Proto().op.extend([relu])

            self.ws.create_blob("X").feed(X, device_option=gc)
            self.ws.create_blob("W").feed(W, device_option=gc)
            self.ws.create_blob("b").feed(b, device_option=gc)
            self.ws.run(net)
            Y = self.ws.blobs["Y"].fetch()
            outputs.append(Output(Y=Y, op_type=op_type, engine=engine, order=order))

        check_quantized_results_close(outputs)
