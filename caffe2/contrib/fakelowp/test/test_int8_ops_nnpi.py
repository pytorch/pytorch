import caffe2.python.fakelowp.init_shared_libs  # noqa
import numpy as np
from caffe2.python import core, workspace
from caffe2.python.onnx.onnxifi import onnxifi_caffe2_net
from hypothesis import given, strategies as st, settings
from caffe2.python.fakelowp.test_utils import print_test_debug_info
import caffe2.python.serialized_test.serialized_test_util as serial
import datetime

core.GlobalInit(["caffe2",
                 "--caffe2_log_level=-3",
                 "--glow_global_fp16=1",
                 "--glow_clip_quant_range_to_fp16=1",
                 "--glow_global_fp16_constants=1"
                 ])


class Int8OpsTest(serial.SerializedTestCase):
    def _get_scale_zp(self, tensor):
        tensor_max = np.max(tensor)
        tensor_min = min(0, np.min(tensor))
        scale = np.float32(np.float16((tensor_max - tensor_min) / 255.0))
        if scale < 1e-6:
            scale = 1e-6
        zero_point = 0 - tensor_min / scale
        zero_point = int(round(np.clip(zero_point, 0, 255.0)))
        return (scale, zero_point)

    @given(
        n=st.integers(2, 1024),
        rand_seed=st.integers(0, 65534),
        non_zero_offset=st.booleans()
    )
    @settings(deadline=datetime.timedelta(seconds=50))
    def test_int8_quantize(self, n, rand_seed, non_zero_offset):
        print("n={}, rand_seed={}".format(n, rand_seed))
        np.random.seed(rand_seed)
        workspace.ResetWorkspace()

        if non_zero_offset:
            X_fp32 = np.random.uniform(-1, 1, size=(n, n)).astype(np.float16) \
                .astype(np.float32)
        else:
            X_fp32 = np.random.rand(n, n).astype(np.float16).astype(np.float32)

        W_fp32 = np.identity(n, dtype=np.float32)
        b_fp32 = np.zeros((n,), dtype=np.float32)

        X_scale, X_zero_point = self._get_scale_zp(X_fp32)

        workspace.FeedBlob("X", X_fp32)
        workspace.FeedBlob("W", W_fp32)
        workspace.FeedBlob("b", b_fp32)

        workspace.RunOperatorOnce(
            core.CreateOperator(
                "Int8FCPackWeight",
                ["W"],
                ["W_int8"],
                engine="DNNLOWP",
                save_unpacked_weights=True,
                in_scale=X_scale,
            )
        )

        ref_net = core.Net("net")
        ref_net.Int8QuantizeNNPI(
            ["X"],
            ["X_int8"],
            Y_scale=X_scale,
            Y_zero_point=X_zero_point
        )
        ref_net.Int8FCFakeAcc32NNPI(
            ["X_int8", "W_int8", "b"],
            ["Y_int8"],
            Y_scale=X_scale,
            Y_zero_point=X_zero_point,
        )
        ref_net.Int8DequantizeNNPI(
            ["Y_int8"],
            ["Y"]
        )
        ref_net.Proto().external_output.append("Y")

        # run ref_net
        workspace.RunNetOnce(ref_net)
        Y_fbgemm = workspace.FetchBlob("Y")

        # run onnxifi net
        ref_net.Proto().op[0].type = "Int8Quantize"
        ref_net.Proto().op[1].type = "Int8FC"
        ref_net.Proto().op[2].type = "Int8Dequantize"
        net_onnxified = onnxifi_caffe2_net(
            ref_net.Proto(),
            {},
            debug=True,
            adjust_batch=False,
            use_onnx=False,
            weight_names=["W_int8", "b"],
        )
        num_onnxified_ops = sum(
            1 if o.type == "Onnxifi" else 0 for o in net_onnxified.op
        )
        np.testing.assert_equal(num_onnxified_ops, 1)
        workspace.CreateNet(net_onnxified)
        workspace.RunNet(net_onnxified.name)
        Y_glow = workspace.FetchBlob("Y")

        if not np.allclose(Y_glow, Y_fbgemm):
            diff_Y = np.abs(Y_glow - Y_fbgemm)
            print_test_debug_info(
                "int8_fc",
                {
                    "seed": rand_seed,
                    "n": n,
                    "X": X_fp32,
                    "W": W_fp32,
                    "b": b_fp32,
                    "Y_fbgemm": Y_fbgemm,
                    "Y_glow": Y_glow,
                    "diff": diff_Y,
                    "maxdiff": diff_Y.max(axis=1),
                },
            )
            assert 0

    @given(
        n=st.integers(1, 1024),
        m=st.integers(1, 1024),
        k=st.integers(1, 1024),
        f=st.integers(1, 1),  # TODO: figure a safe number to increase
        rand_seed=st.integers(0, 65534),
        quantize_bias=st.sampled_from([False]),
    )
    @settings(deadline=datetime.timedelta(seconds=50))
    def test_int8_fc(
        self, n, m, k, rand_seed, quantize_bias, f
    ):
        print(
            f"n={n}, m={m}, k={k}, rand_seed={rand_seed}, quantize_bias={quantize_bias}"
        )
        np.random.seed(rand_seed)
        workspace.ResetWorkspace()

        ff = float(f)
        X_fp32 = np.random.uniform(-ff, ff, size=(m, k)).astype(np.float32)
        W_fp32 = np.random.uniform(-ff, ff, size=(n, k)).astype(np.float32)
        b_fp32 = np.random.uniform(-ff, ff, size=(n)).astype(np.float32)

        X_scale, X_zero_point = self._get_scale_zp(X_fp32)
        Y_fp32 = np.dot(X_fp32, W_fp32.T) + b_fp32
        Y_scale, Y_zero_point = self._get_scale_zp(Y_fp32)

        workspace.FeedBlob("X", X_fp32)
        workspace.FeedBlob("W", W_fp32)
        workspace.FeedBlob("b", b_fp32)

        workspace.RunOperatorOnce(
            core.CreateOperator(
                "Int8FCPackWeight",
                ["W", "b"] if quantize_bias else ["W"],
                ["W_int8", "b_int32"] if quantize_bias else ["W_int8"],
                engine="DNNLOWP",
                save_unpacked_weights=True,
                in_scale=X_scale,
            )
        )

        ref_net = core.Net("net")
        ref_net.Int8QuantizeNNPI(
            ["X"],
            ["X_int8"],
            Y_scale=X_scale,
            Y_zero_point=X_zero_point
        )
        ref_net.Int8FCFakeAcc32NNPI(
            ["X_int8", "W_int8", "b_int32" if quantize_bias else "b"],
            ["Y_int8"],
            Y_scale=Y_scale,
            Y_zero_point=Y_zero_point,
        )
        ref_net.Int8DequantizeNNPI(
            ["Y_int8"],
            ["Y"]
        )
        ref_net.Proto().external_output.append("Y")

        # run ref_net
        workspace.RunNetOnce(ref_net)
        Y_fbgemm = workspace.FetchBlob("Y")

        # run onnxifi net
        ref_net.Proto().op[0].type = "Int8Quantize"
        ref_net.Proto().op[1].type = "Int8FC"
        ref_net.Proto().op[2].type = "Int8Dequantize"
        net_onnxified = onnxifi_caffe2_net(
            ref_net.Proto(),
            {},
            debug=True,
            adjust_batch=False,
            use_onnx=False,
            weight_names=["W_int8", "b_int32"] if quantize_bias else ["W_int8", "b"],
        )
        num_onnxified_ops = sum(
            1 if o.type == "Onnxifi" else 0 for o in net_onnxified.op
        )
        np.testing.assert_equal(num_onnxified_ops, 1)
        workspace.CreateNet(net_onnxified)
        workspace.RunNet(net_onnxified.name)
        Y_glow = workspace.FetchBlob("Y")

        if not np.allclose(Y_glow, Y_fbgemm):
            diff_Y = np.abs(Y_glow - Y_fbgemm)
            print_test_debug_info(
                "int8_fc",
                {
                    "seed": rand_seed,
                    "n": n,
                    "m": m,
                    "k": k,
                    "X": X_fp32,
                    "W": W_fp32,
                    "b": b_fp32,
                    "Y_fbgemm": Y_fbgemm,
                    "Y_glow": Y_glow,
                    "diff": diff_Y,
                    "maxdiff": diff_Y.max(axis=1),
                },
            )
            assert 0

    @given(
        n=st.integers(1, 4),
        rand_seed=st.integers(0, 65534)
    )
    @settings(deadline=datetime.timedelta(seconds=10))
    def test_int8_small_input(self, n, rand_seed):
        print("n={}, rand_seed={}".format(n, rand_seed))
        np.random.seed(rand_seed)
        workspace.ResetWorkspace()

        X_fp32 = np.random.uniform(0.01, 0.03, size=(n, n)).astype(np.float32)
        W_fp32 = np.identity(n, dtype=np.float32)
        b_fp32 = np.zeros((n,), dtype=np.float32)

        X_scale, X_zero_point = self._get_scale_zp(X_fp32)

        workspace.FeedBlob("X", X_fp32)
        workspace.FeedBlob("W", W_fp32)
        workspace.FeedBlob("b", b_fp32)

        workspace.RunOperatorOnce(
            core.CreateOperator(
                "Int8FCPackWeight",
                ["W"],
                ["W_int8"],
                engine="DNNLOWP",
                save_unpacked_weights=True,
                in_scale=X_scale,
            )
        )

        ref_net = core.Net("net")
        ref_net.Int8QuantizeNNPI(
            ["X"],
            ["X_int8"],
            Y_scale=X_scale,
            Y_zero_point=X_zero_point
        )
        ref_net.Int8FCFakeAcc32NNPI(
            ["X_int8", "W_int8", "b"],
            ["Y_int8"],
            Y_scale=X_scale,
            Y_zero_point=X_zero_point,
        )
        ref_net.Int8DequantizeNNPI(
            ["Y_int8"],
            ["Y"]
        )
        ref_net.Proto().external_output.append("Y")

        # run ref_net
        workspace.RunNetOnce(ref_net)
        Y_fbgemm = workspace.FetchBlob("Y")

        # run onnxifi net
        ref_net.Proto().op[0].type = "Int8Quantize"
        ref_net.Proto().op[1].type = "Int8FC"
        ref_net.Proto().op[2].type = "Int8Dequantize"
        net_onnxified = onnxifi_caffe2_net(
            ref_net.Proto(),
            {},
            debug=True,
            adjust_batch=False,
            use_onnx=False,
            weight_names=["W_int8", "b"],
        )
        num_onnxified_ops = sum(
            1 if o.type == "Onnxifi" else 0 for o in net_onnxified.op
        )
        np.testing.assert_equal(num_onnxified_ops, 1)
        workspace.CreateNet(net_onnxified)
        workspace.RunNet(net_onnxified.name)
        Y_glow = workspace.FetchBlob("Y")

        if not np.allclose(Y_glow, Y_fbgemm):
            diff_Y = np.abs(Y_glow - Y_fbgemm)
            print_test_debug_info(
                "int8_fc",
                {
                    "seed": rand_seed,
                    "n": n,
                    "X": X_fp32,
                    "W": W_fp32,
                    "b": b_fp32,
                    "Y_fbgemm": Y_fbgemm,
                    "Y_glow": Y_glow,
                    "diff": diff_Y,
                    "maxdiff": diff_Y.max(axis=1),
                },
            )
            assert 0
