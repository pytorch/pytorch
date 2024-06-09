# Must happen before importing caffe2.python.*
import caffe2.python.fakelowp.init_shared_libs  # noqa
import datetime
import numpy as np
from hypothesis import given, settings, example
from hypothesis import strategies as st
from caffe2.python import core, workspace
from caffe2.python.onnx.onnxifi import onnxifi_caffe2_net
from caffe2.python.fakelowp.test_utils import print_test_debug_info
import caffe2.python.serialized_test.serialized_test_util as serial

# Test that parallel chunks behave the same way as the serial one

workspace.GlobalInit(
    [
        "caffe2",
        "--glow_global_fp16=1",
        "--glow_global_fused_scale_offset_fp16=1",
        "--glow_global_force_sls_fp16_accum=1",
        "--glow_nnpi_num_parallel_chunks=2",
        "--glow_use_dag_optimizer=false",
        "--glow_dump_graph=true",
    ]
)

class Fusions(serial.SerializedTestCase):
    def _get_scale_zp(self, tensor):
        tensor_max = np.max(tensor)
        tensor_min = min(0, np.min(tensor))
        scale = np.float32(np.float16((tensor_max - tensor_min) / 255.0))
        if scale < 1e-6:
            scale = np.float32(1e-6)
        zero_point = 0 - tensor_min / scale
        zero_point = int(round(np.clip(zero_point, 0, 255.0)))
        return (scale, zero_point)

    @given(
        scale=st.floats(1e-4, 1e2),
        zp=st.integers(-128, 128),
        rand_seed=st.integers(0, 65534),
        m=st.integers(32, 64),
        k=st.integers(1000, 6000),
        n=st.integers(200, 600),
    )
    # @example(m=64, k=5423, n=553, scale=1e-3, zp=120, rand_seed=1)
    @settings(deadline=datetime.timedelta(seconds=1000), max_examples=1)
    def test_ParallelFC(self, m, k, n, scale, zp, rand_seed):
        np.random.seed(rand_seed)
        workspace.ResetWorkspace()

        # Y = W_T * X + b
        X_fp32 = np.random.uniform(-1, 1, size=(m, k)).astype(np.float16) \
            .astype(np.float32)

        W_fp32 = np.random.uniform(-1, 1, size=(n, k)).astype(np.float32)
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
        ref_net.Int8Relu(
            ["Y_int8"],
            ["Y_relu"],
            Y_zero_point=X_zero_point,
            Y_scale=X_scale,
        )
        ref_net.Int8DequantizeNNPI(
            ["Y_relu"],
            ["Y"]
        )
        ref_net.Proto().external_output.append("Y")

        # run ref_net
        workspace.RunNetOnce(ref_net)
        Y_fbgemm = workspace.FetchBlob("Y")

        # run onnxifi net
        ref_net.Proto().op[0].type = "Int8Quantize"
        ref_net.Proto().op[1].type = "Int8FC"
        ref_net.Proto().op[2].type = "Int8Relu"
        ref_net.Proto().op[3].type = "Int8Dequantize"
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
        print(net_onnxified)
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
