from __future__ import absolute_import, division, print_function, unicode_literals

import collections

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, dyndep, utils, workspace
from caffe2.quantization.server import utils as dnnlowp_utils
from dnnlowp_test_utils import (
    check_quantized_results_close,
    run_conv_or_fc,
)
from hypothesis import assume, given


dyndep.InitOpsLibrary("//caffe2/caffe2/quantization/server:dnnlowp_ops")
workspace.GlobalInit(
    [
        "caffe2",
        "--caffe2_omp_num_threads=11",
        # Increase this threshold to test acc16 with randomly generated data
        "--caffe2_dnnlowp_acc16_density_threshold=0.5",
    ]
)


class GroupWiseDNNLowPOpConvAcc16OpTest(hu.HypothesisTestCase):
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
        share_col_buffer=st.booleans(),
        preserve_activation_sparsity=st.booleans(),
        preserve_weight_sparsity=st.booleans(),
        **hu.gcs_cpu_only
    )
    def test_groupwise_dnnlowp_conv_acc16_int(
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
        preserve_activation_sparsity,
        preserve_weight_sparsity,
        gc,
        dc,
    ):
        assume(group == 1 or dilation == 1)
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
        W[..., 1] = W_min + 128  # "zeros"
        for g in range(group):
            W[g * output_channels_per_group, 0, 0, 0] = W_min
            W[g * output_channels_per_group + 1, 0, 0, 0] = W_max
            if not preserve_weight_sparsity:
                W[
                    g * output_channels_per_group : (g + 1) * output_channels_per_group,
                ] += g

        if order == "NCHW":
            X = utils.NHWC2NCHW(X)
            W = utils.NHWC2NCHW(W)

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

            do_quantize = "DNNLOWP" in engine
            do_dequantize = "DNNLOWP" in engine

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
                    "Dequantize", ["Y_q"], ["Y"], engine="DNNLOWP", device_option=gc
                )
                net.Proto().op.extend([dequantize])

            run_conv_or_fc(
                self, None, net, X, W, b, op_type, engine, order, gc, outputs
            )

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
        prepack_weight=st.booleans(),
        nbits_in_non_outlier=st.sampled_from((0, 1, 6, 8)),
        share_col_buffer=st.booleans(),
        **hu.gcs_cpu_only
    )
    def test_groupwise_dnnlowp_conv_acc16_outlier(
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
        prepack_weight,
        nbits_in_non_outlier,
        share_col_buffer,
        gc,
        dc,
    ):
        assume(group == 1 or dilation == 1)
        assume(size >= dilation * (kernel - 1) + 1)

        input_channels = input_channels_per_group * group
        output_channels = output_channels_per_group * group

        X_min = -77
        X_max = X_min + 255
        X = np.random.rand(batch_size, size, size, input_channels) * 4 + X_min
        X = np.round(X).astype(np.float32)
        X[..., 0] = X_min
        X[0, 0, 0, 1] = X_max

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
        W[..., 1] = W_min + 128  # "zeros"
        for g in range(group):
            W[g * output_channels_per_group, 0, 0, 0] = W_min
            W[g * output_channels_per_group + 1, 0, 0, 0] = W_max
            W[g * output_channels_per_group : (g + 1) * output_channels_per_group,] += g

        if order == "NCHW":
            X = utils.NHWC2NCHW(X)
            W = utils.NHWC2NCHW(W)

        b = np.round(np.random.randn(output_channels)).astype(np.float32)

        Output = collections.namedtuple("Output", ["Y", "op_type", "engine", "order"])
        outputs = []

        op_engine_list = [
            ("Conv", ""),
            ("Conv", "DNNLOWP_ACC16"),
            ("Int8Conv", "DNNLOWP_ACC16"),
        ]

        for op_type, engine in op_engine_list:
            init_net = core.Net("test_init_net")
            net = core.Net("test_net")

            do_quantize = "DNNLOWP" in engine
            do_dequantize = "DNNLOWP" in engine
            do_prepack_weight = "DNNLOWP" in engine and prepack_weight

            if do_quantize:
                quantize = core.CreateOperator(
                    "Quantize", ["X"], ["X_q"], engine="DNNLOWP", device_option=gc
                )
                net.Proto().op.extend([quantize])

            if do_prepack_weight:
                x_q_param = dnnlowp_utils.choose_quantization_params(X.min(), X.max())
                inputs = ["W"]
                if do_dequantize:
                    inputs += ["b"]
                pack = core.CreateOperator(
                    "Int8ConvPackWeight",
                    inputs,
                    ["W_packed"],
                    group=group,
                    quantize_groupwise=1,
                    nbits_in_non_outlier=nbits_in_non_outlier,
                    in_scale=x_q_param.scale,
                    engine=engine,
                )
                init_net.Proto().op.extend([pack])

            conv = core.CreateOperator(
                op_type,
                [
                    "X_q" if do_quantize else "X",
                    "W_packed" if do_prepack_weight else "W",
                    "b",
                ],
                ["Y_q" if do_dequantize else "Y"],
                stride=stride,
                kernel=kernel,
                dilation=dilation,
                pad=pad,
                order=order,
                nbits_in_non_outlier=nbits_in_non_outlier,
                shared_buffer=(1 if share_col_buffer else 0),
                engine=engine,
                group=group,
                quantize_groupwise=1,
                device_option=gc,
            )
            if do_dequantize or do_prepack_weight:
                # groupwise quantization only works with static quantization
                # so we need to set quantization parameters
                dnnlowp_utils.add_quantization_param_args(conv, outputs[0][0])
            net.Proto().op.extend([conv])

            if do_dequantize:
                dequantize = core.CreateOperator(
                    "Dequantize", ["Y_q"], ["Y"], engine="DNNLOWP", device_option=gc
                )
                net.Proto().op.extend([dequantize])

            run_conv_or_fc(
                self, init_net, net, X, W, b, op_type, engine, order, gc, outputs
            )

        check_quantized_results_close(outputs)
