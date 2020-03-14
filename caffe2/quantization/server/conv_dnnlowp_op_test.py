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
from hypothesis import assume, given


dyndep.InitOpsLibrary("//caffe2/caffe2/quantization/server:dnnlowp_ops")
workspace.GlobalInit(["caffe2", "--caffe2_omp_num_threads=11"])


class DNNLowPOpConvTest(hu.HypothesisTestCase):
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
        batch_size=st.integers(0, 3),
        order=st.sampled_from(["NCHW", "NHWC"]),
        weight_quantized=st.booleans(),
        prepack_weight=st.booleans(),
        share_col_buffer=st.booleans(),
        preserve_activation_sparsity=st.booleans(),
        preserve_weight_sparsity=st.booleans(),
        **hu.gcs_cpu_only
    )
    def test_dnnlowp_conv_int(
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
        weight_quantized,
        prepack_weight,
        share_col_buffer,
        preserve_activation_sparsity,
        preserve_weight_sparsity,
        gc,
        dc,
    ):
        assume(group == 1 or dilation == 1)
        assume((not prepack_weight) or order == "NHWC")

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

        op_engine_list = [
            ("Conv", ""),
            ("Conv", "DNNLOWP"),
            ("Conv", "DNNLOWP_16"),
            ("Int8Conv", "DNNLOWP"),
        ]

        for op_type, engine in op_engine_list:
            init_net = core.Net("test_init_net")
            net = core.Net("test_net")

            do_quantize = "DNNLOWP" in engine
            do_dequantize = "DNNLOWP" in engine
            # If output scale/zp aren't set, it gets computed from ref fp32 op
            # in DNNLOWP, which isn't possible when we quantize input weights.
            # Make sure atleast one output is collected to compute output
            # scale/zp.
            do_quantize_weight = (
                engine == "DNNLOWP" and weight_quantized and len(outputs) > 0
            )
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

            X_min = 0 if X.size == 0 else X.min()
            X_max = 0 if X.size == 0 else X.max()
            x_q_param = dnnlowp_utils.choose_quantization_params(
                X_min, X_max, preserve_activation_sparsity
            )
            if do_quantize_weight:
                int8_given_tensor_fill, w_q_param = dnnlowp_utils.create_int8_given_tensor_fill(
                    W, "W_q", preserve_weight_sparsity
                )
                init_net.Proto().op.extend([int8_given_tensor_fill])

                # Bias
                int8_bias_tensor_fill = dnnlowp_utils.create_int8_bias_tensor_fill(
                    b, "b_q", x_q_param, w_q_param
                )
                init_net.Proto().op.extend([int8_bias_tensor_fill])

            if do_prepack_weight:
                inputs = ["W_q" if do_quantize_weight else "W"]
                if do_dequantize:
                    inputs += ["b_q" if do_quantize_weight else "b"]
                pack = core.CreateOperator(
                    "Int8ConvPackWeight",
                    inputs,
                    ["W_packed"],
                    stride=stride,
                    kernel=kernel,
                    dilation=dilation,
                    pad=pad,
                    preserve_weight_sparsity=preserve_weight_sparsity,
                    engine=engine,
                    group=group,
                    in_scale=x_q_param.scale,
                )
                init_net.Proto().op.extend([pack])

            conv = core.CreateOperator(
                op_type,
                [
                    "X_q" if do_quantize else "X",
                    "W_packed"
                    if do_prepack_weight
                    else ("W_q" if do_quantize_weight else "W"),
                    "b_q" if do_quantize_weight else "b",
                ],
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
                device_option=gc,
            )
            if do_quantize_weight or do_prepack_weight:
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
                    "Dequantize", ["Y_q"], ["Y"], engine=engine, device_option=gc
                )
                net.Proto().op.extend([dequantize])

            run_conv_or_fc(
                self, init_net, net, X, W, b, op_type, engine, order, gc, outputs
            )

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
        batch_size=st.integers(0, 3),
        order=st.sampled_from(["NCHW", "NHWC"]),
        share_col_buffer=st.booleans(),
        **hu.gcs_cpu_only
    )
    def test_dnnlowp_conv_relu_int(
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
        share_col_buffer,
        gc,
        dc,
    ):
        assume(group == 1 or dilation == 1)

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
                    shared_buffer=(1 if share_col_buffer else 0),
                    group=group,
                    device_option=gc,
                )
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
                    shared_buffer=(1 if share_col_buffer else 0),
                    engine=engine,
                    group=group,
                    device_option=gc,
                )
                net.Proto().op.extend([conv])

                relu = core.CreateOperator(
                    "Relu", ["Y"], ["Y"], engine=engine, device_option=gc
                )
                net.Proto().op.extend([relu])

            run_conv_or_fc(
                self, None, net, X, W, b, op_type, engine, order, gc, outputs
            )

        check_quantized_results_close(outputs)

    def _test_dnnlowp_nd_int(
        self,
        stride,
        pad,
        kernels,
        dilation,
        size,
        group,
        input_channels_per_group,
        output_channels_per_group,
        batch_size,
        order,
        prepack_weight,
        gc,
        dc,
    ):
        assume(group == 1 or dilation == 1)
        assume((not prepack_weight) or order == "NHWC")
        ndim = len(kernels)

        X, W, b = generate_convnd_inputs(
            (stride,) * ndim,
            (pad,) * ndim,
            kernels,
            (dilation,) * ndim,
            (size,) * ndim,
            group,
            input_channels_per_group,
            output_channels_per_group,
            batch_size,
            order,
        )

        Output = collections.namedtuple("Output", ["Y", "op_type", "engine", "order"])
        outputs = []

        op_engine_list = [("Conv", ""), ("Conv", "DNNLOWP_16"), ("Int8Conv", "DNNLOWP")]

        for op_type, engine in op_engine_list:
            init_net = core.Net("test_init_net")
            net = core.Net("test_net")

            do_quantize = "DNNLOWP" in engine
            do_dequantize = "DNNLOWP" in engine
            # If output scale/zp aren't set, it gets computed from ref fp32 op
            # in DNNLOWP, which isn't possible when we quantize input weights.
            # Make sure atleast one output is collected to compute output
            # scale/zp.
            do_quantize_weight = engine == "DNNLOWP" and len(outputs) > 0
            do_prepack_weight = engine == "DNNLOWP" and prepack_weight

            if do_quantize:
                quantize = core.CreateOperator(
                    "Quantize", ["X"], ["X_q"], engine=engine, device_option=gc
                )
                net.Proto().op.extend([quantize])

            X_min = 0 if X.size == 0 else X.min()
            X_max = 0 if X.size == 0 else X.max()
            x_q_param = dnnlowp_utils.choose_quantization_params(X_min, X_max)
            if do_quantize_weight:
                int8_given_tensor_fill, w_q_param = dnnlowp_utils.create_int8_given_tensor_fill(
                    W, "W_q"
                )
                init_net.Proto().op.extend([int8_given_tensor_fill])

                # Bias
                int8_bias_tensor_fill = dnnlowp_utils.create_int8_bias_tensor_fill(
                    b, "b_q", x_q_param, w_q_param
                )
                init_net.Proto().op.extend([int8_bias_tensor_fill])

            if do_prepack_weight:
                inputs = ["W_q" if do_quantize_weight else "W"]
                if do_dequantize:
                    inputs += ["b_q" if do_quantize_weight else "b"]
                pack = core.CreateOperator(
                    "Int8ConvPackWeight",
                    inputs,
                    ["W_packed"],
                    strides=[stride] * ndim,
                    kernels=kernels,
                    dilations=[dilation] * ndim,
                    pads=[pad] * (ndim * 2),
                    engine=engine,
                    group=group,
                    in_scale=x_q_param.scale,
                )
                init_net.Proto().op.extend([pack])

            conv = core.CreateOperator(
                op_type,
                [
                    "X_q" if do_quantize else "X",
                    "W_packed"
                    if do_prepack_weight
                    else ("W_q" if do_quantize_weight else "W"),
                    "b_q" if do_quantize_weight else "b",
                ],
                ["Y_q" if do_dequantize else "Y"],
                strides=[stride] * ndim,
                kernels=kernels,
                dilations=[dilation] * ndim,
                pads=[pad] * (ndim * 2),
                order=order,
                dequantize_output=not do_dequantize,
                engine=engine,
                group=group,
                device_option=gc,
            )
            if do_quantize_weight or do_prepack_weight:
                # When quantized weight is provided, we can't rescale the
                # output dynamically by looking at the range of output of each
                # batch, so here we provide the range of output observed from
                # fp32 reference implementation
                dnnlowp_utils.add_quantization_param_args(conv, outputs[0][0])
            net.Proto().op.extend([conv])

            if do_dequantize:
                dequantize = core.CreateOperator(
                    "Dequantize", ["Y_q"], ["Y"], engine=engine, device_option=gc
                )
                net.Proto().op.extend([dequantize])

            run_conv_or_fc(
                self, init_net, net, X, W, b, op_type, engine, order, gc, outputs
            )

        check_quantized_results_close(outputs)

    @given(
        stride=st.integers(1, 2),
        pad=st.integers(0, 2),
        temporal_kernels=st.sampled_from([1, 5]),
        spatial_kernels=st.sampled_from([1, 3]),
        dilation=st.integers(1, 1),
        size=st.sampled_from([5, 8]),
        group=st.integers(1, 2),
        input_channels_per_group=st.sampled_from([2, 3]),
        output_channels_per_group=st.sampled_from([2, 3]),
        batch_size=st.integers(0, 2),
        order=st.sampled_from(["NCHW", "NHWC"]),
        prepack_weight=st.booleans(),
        **hu.gcs_cpu_only
    )
    def test_dnnlowp_conv3d_int(
        self,
        stride,
        pad,
        temporal_kernels,
        spatial_kernels,
        dilation,
        size,
        group,
        input_channels_per_group,
        output_channels_per_group,
        batch_size,
        order,
        prepack_weight,
        gc,
        dc,
    ):
        self._test_dnnlowp_nd_int(
            stride,
            pad,
            (temporal_kernels,) + (spatial_kernels,) * 2,
            dilation,
            size,
            group,
            input_channels_per_group,
            output_channels_per_group,
            batch_size,
            order,
            prepack_weight,
            gc,
            dc,
        )

    @given(
        stride=st.integers(1, 2),
        pad=st.integers(0, 2),
        kernels=st.sampled_from([1, 3]),
        dilation=st.integers(1, 1),
        size=st.sampled_from([5, 8]),
        group=st.integers(1, 2),
        input_channels_per_group=st.sampled_from([2, 3]),
        output_channels_per_group=st.sampled_from([2, 3]),
        batch_size=st.integers(0, 2),
        order=st.sampled_from(["NCHW", "NHWC"]),
        prepack_weight=st.booleans(),
        **hu.gcs_cpu_only
    )
    def test_dnnlowp_conv1d_int(
        self,
        stride,
        pad,
        kernels,
        dilation,
        size,
        group,
        input_channels_per_group,
        output_channels_per_group,
        batch_size,
        order,
        prepack_weight,
        gc,
        dc,
    ):
        self._test_dnnlowp_nd_int(
            stride,
            pad,
            (kernels,),
            dilation,
            size,
            group,
            input_channels_per_group,
            output_channels_per_group,
            batch_size,
            order,
            prepack_weight,
            gc,
            dc,
        )
