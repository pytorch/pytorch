from __future__ import absolute_import, division, print_function, unicode_literals

import collections
from tempfile import NamedTemporaryFile

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


class DNNLowPFullyConnectedOpTest(hu.HypothesisTestCase):
    # correctness test with no quantization error in inputs
    @given(
        input_channels=st.sampled_from([3, 4, 5, 8, 16, 32]),
        output_channels=st.integers(2, 16),
        batch_size=st.integers(0, 16),
        in_quantized=st.booleans(),
        out_quantized=st.booleans(),
        weight_quantized=st.booleans(),
        prepack_weight=st.booleans(),
        preserve_activation_sparsity=st.booleans(),
        preserve_weight_sparsity=st.booleans(),
        fuse_relu=st.booleans(),
        **hu.gcs_cpu_only
    )
    def test_dnnlowp_fully_connected_int(
        self,
        input_channels,
        output_channels,
        batch_size,
        in_quantized,
        out_quantized,
        weight_quantized,
        prepack_weight,
        preserve_activation_sparsity,
        preserve_weight_sparsity,
        fuse_relu,
        gc,
        dc,
    ):
        # X and W have scale 1, so exactly represented after quantization
        X_min = 0 if preserve_activation_sparsity else -77
        X_max = X_min + 255
        X = np.round(
            np.random.rand(batch_size, input_channels) * (X_max - X_min) + X_min
        )
        X = X.astype(np.float32)
        # input channels 0 and 1 are all X_min to avoid overflow from vpmaddubsw
        # when multiplied with W_min and W_max
        X[:, 0] = X_min
        if batch_size != 0:
            X[0, 1] = X_max

        if preserve_weight_sparsity:
            W_min = -128
            W_max = 100
        else:
            W_min = -100
            W_max = W_min + 255
        W = np.round(
            np.random.rand(output_channels, input_channels) * (W_max - W_min) + W_min
        )
        W = W.astype(np.float32)
        W[0, 0] = W_min
        W[1, 0] = W_max

        # Make sure we won't have overflows from vpmaddubsw instruction used in
        # fbgemm
        avoid_vpmaddubsw_overflow_fc(
            batch_size,
            input_channels,
            output_channels,
            X,
            X_min,
            X_max,
            W,
            W_min,
            W_max,
        )

        b = np.random.randn(output_channels).astype(np.float32)

        Output = collections.namedtuple("Output", ["Y", "op_type", "engine"])
        outputs = []

        op_engine_list = [("FC", "")]
        if fuse_relu:
            op_engine_list += [("Int8FCRelu", "DNNLOWP")]
        else:
            op_engine_list += [
                ("FC", "DNNLOWP"),
                ("FC", "DNNLOWP_16"),
                ("Int8FC", "DNNLOWP"),
            ]

        for op_type, engine in op_engine_list:
            init_net = core.Net("test_init_net")
            net = core.Net("test_net")

            do_quantize = "DNNLOWP" in engine and in_quantized
            do_dequantize = "DNNLOWP" in engine and out_quantized
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
                    "Int8FCPackWeight",
                    inputs,
                    ["W_packed"],
                    preserve_weight_sparsity=preserve_weight_sparsity,
                    in_scale=x_q_param.scale,
                    engine=engine,
                )
                init_net.Proto().op.extend([pack])

            fc = core.CreateOperator(
                op_type,
                [
                    "X_q" if do_quantize else "X",
                    "W_packed"
                    if do_prepack_weight
                    else ("W_q" if do_quantize_weight else "W"),
                    "b_q" if do_quantize_weight else "b",
                ],
                ["Y_q" if do_dequantize else "Y"],
                dequantize_output=not do_dequantize,
                preserve_activation_sparsity=preserve_activation_sparsity,
                preserve_weight_sparsity=preserve_weight_sparsity,
                engine=engine,
                device_option=gc,
            )
            if do_quantize_weight or do_prepack_weight:
                # When quantized weight is provided, we can't rescale the
                # output dynamically by looking at the range of output of each
                # batch, so here we provide the range of output observed from
                # fp32 reference implementation
                dnnlowp_utils.add_quantization_param_args(
                    fc, outputs[0][0], preserve_activation_sparsity
                )
            net.Proto().op.extend([fc])
            if fuse_relu and "DNNLOWP" not in engine:
                net.Relu(["Y"], "Y")

            if do_dequantize:
                dequantize = core.CreateOperator(
                    "Dequantize", ["Y_q"], ["Y"], engine=engine, device_option=gc
                )
                net.Proto().op.extend([dequantize])

            run_conv_or_fc(
                self, init_net, net, X, W, b, op_type, engine, None, gc, outputs
            )

        check_quantized_results_close(outputs, symmetric=preserve_activation_sparsity)

    @given(
        input_channels=st.sampled_from([3, 4, 5, 8, 16, 32]),
        output_channels=st.integers(2, 16),
        batch_size=st.integers(1, 16),
        preserve_activation_sparsity=st.booleans(),
        preserve_weight_sparsity=st.booleans(),
        **hu.gcs_cpu_only
    )
    def test_serializer_for_dnnlowp_fully_connected_op(
        self,
        input_channels,
        output_channels,
        batch_size,
        preserve_activation_sparsity,
        preserve_weight_sparsity,
        gc,
        dc,
    ):
        X_min = 0 if preserve_activation_sparsity else -77
        X_max = X_min + 255
        X = np.round(np.random.rand(batch_size, input_channels) * (X_max - X_min) + X_min)
        X = X.astype(np.float32)
        X[:, 0] = X_min
        X[0, 1] = X_max

        if preserve_weight_sparsity:
            W_min = -128
            W_max = 100
        else:
            W_min = -100
            W_max = W_min + 255
        W = np.round(
            np.random.rand(output_channels, input_channels) * (W_max - W_min) + W_min
        )
        W = W.astype(np.float32)
        W[0, 0] = W_min
        W[1, 0] = W_max

        b = np.random.randn(output_channels).astype(np.float32)

        workspace.FeedBlob("X", X, device_option=gc)
        workspace.FeedBlob("W", W, device_option=gc)
        workspace.FeedBlob("b", b, device_option=gc)

        x_q_param = dnnlowp_utils.choose_quantization_params(
            X_min, X_max, preserve_activation_sparsity
        )
        pack = core.CreateOperator(
            "Int8FCPackWeight",
            ["W"],
            ["W_packed"],
            in_scale=x_q_param.scale,
            preserve_weight_sparsity=preserve_weight_sparsity,
            save_unpacked_weights=True,
            engine="DNNLOWP",
            device_option=gc,
        )
        pack_net = core.Net("pack_net")
        pack_net.Proto().op.extend([pack])
        workspace.RunNetOnce(pack_net)

        # Save packed weights
        f = NamedTemporaryFile(delete=True)
        save_net = core.Net("save_net")
        save_net.Save("W_packed", [], db=f.name, db_type="minidb", absolute_path=True)
        workspace.RunNetOnce(save_net)
        # put garbage into W_packed
        workspace.FeedBlob("W_packed", np.random.rand(0, 0).astype(np.float32))

        # Load prepacked weights
        load_net = core.Net("load_net")
        load_net.Load([], "W_packed", db=f.name, db_type="minidb", absolute_path=True)
        workspace.RunNetOnce(load_net)

        # Choose output qparams based on ground output
        def fc_op(X, W, b):
            return [np.dot(X, W.T) + b]

        ground_output = fc_op(X, W, b)
        y_q_param = dnnlowp_utils.choose_quantization_params(
            np.min(ground_output), np.max(ground_output), preserve_activation_sparsity
        )
        w_q_param = dnnlowp_utils.choose_quantization_params(
            W_min, W_max, preserve_weight_sparsity
        )

        simple_fc_net = core.Net("fc")
        # Generate X_q
        simple_fc_net.Int8Quantize(
            ["X"],
            ["X_q"],
            Y_scale=x_q_param.scale,
            Y_zero_point=x_q_param.zero_point,
            preserve_activation_sparsity=preserve_activation_sparsity,
            engine="DNNLOWP",
            device_option=gc,
        )
        # Generate W_q
        simple_fc_net.Int8Quantize(
            ["W"],
            ["W_q"],
            Y_scale=w_q_param.scale,
            Y_zero_point=w_q_param.zero_point,
            preserve_activation_sparsity=preserve_weight_sparsity,
            engine="DNNLOWP",
            device_option=gc,
        )

        # Int8FC using prepacked weights
        simple_fc_net.Int8FC(
            ["X_q", "W_packed", "b"],
            "Y_q",
            engine="DNNLOWP",
            Y_scale=y_q_param.scale,
            Y_zero_point=y_q_param.zero_point,
            device_option=gc,
        )
        # Int8FC using newly packed weights
        new_pack_op = core.CreateOperator(
            "Int8FCPackWeight",
            ["W"],
            ["W_packed_new"],
            in_scale=x_q_param.scale,
            preserve_weight_sparsity=preserve_weight_sparsity,
            save_unpacked_weights=True,
            engine="DNNLOWP",
            device_option=gc,
        )
        simple_fc_net.Proto().op.extend([new_pack_op])

        simple_fc_net.Int8FC(
            ["X_q", "W_packed_new", "b"],
            ["Y_q_1"],
            engine="DNNLOWP",
            Y_scale=y_q_param.scale,
            Y_zero_point=y_q_param.zero_point,
            device_option=gc,
        )
        # Int8FC using unpacked weights
        simple_fc_net.Int8FC(
            ["X_q", "W_q", "b"],
            "Y_q_2",
            engine="DNNLOWP",
            Y_scale=y_q_param.scale,
            Y_zero_point=y_q_param.zero_point,
            device_option=gc,
        )
        # Int8FC using floating point inputs
        simple_fc_net.Int8FC(
            ["X", "W", "b"],
            "Y_q_3",
            engine="DNNLOWP",
            Y_scale=y_q_param.scale,
            Y_zero_point=y_q_param.zero_point,
            preserve_weight_sparsity=preserve_weight_sparsity,
            preserve_activation_sparsity=preserve_activation_sparsity,
            device_option=gc,
        )

        # Dequantize outputs
        simple_fc_net.Int8Dequantize(["Y_q"], ["Y"], engine="DNNLOWP")
        simple_fc_net.Int8Dequantize(["Y_q_1"], ["Y_1"], engine="DNNLOWP")
        simple_fc_net.Int8Dequantize(["Y_q_2"], ["Y_2"], engine="DNNLOWP")
        simple_fc_net.Int8Dequantize(["Y_q_3"], ["Y_3"], engine="DNNLOWP")

        workspace.RunNetOnce(simple_fc_net)

        outputs = []
        Output = collections.namedtuple("Output", ["Y", "op_type", "engine"])
        outputs.append(Output(Y=workspace.FetchBlob("Y"), op_type="FC", engine="DNNLOWP"))
        outputs.append(Output(Y=workspace.FetchBlob("Y_1"), op_type="FC", engine="DNNLOWP"))
        outputs.append(Output(Y=workspace.FetchBlob("Y_2"), op_type="FC", engine="DNNLOWP"))
        outputs.append(Output(Y=workspace.FetchBlob("Y_3"), op_type="FC", engine="DNNLOWP"))
        check_quantized_results_close(outputs, symmetric=preserve_activation_sparsity)
