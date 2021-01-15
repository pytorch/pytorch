import numpy as np
import caffe2.python.fakelowp.init_shared_libs  # noqa
from caffe2.python import core, workspace
from caffe2.python.onnx.onnxifi import onnxifi_caffe2_net
from caffe2.python.fakelowp.test_utils import print_test_debug_info
import caffe2.python.serialized_test.serialized_test_util as serial
import datetime
from hypothesis import settings

core.GlobalInit(["caffe2", "--caffe2_log_level=-3", "--glow_global_fp16=1"])

class DeqSwishQuantTest(serial.SerializedTestCase):
    def _get_scale_zp(self, tensor):
        tensor_max = np.max(tensor)
        tensor_min = min(0, np.min(tensor))
        scale = np.float32(np.float16((tensor_max - tensor_min) / 255.))
        zero_point = -tensor_min / scale
        zero_point = int(round(np.clip(zero_point, 0, 255.0)))
        return (scale, zero_point)

    def _sigmoid(self, x):
        return 1. / (1. + np.exp(np.float32(-x)))

    def _swish(self, x):
        return np.float32(x) * self._sigmoid(x)

    @settings(deadline=datetime.timedelta(seconds=10))
    def test_swish_int8(self):
        np.random.seed(0)
        workspace.ResetWorkspace()
        n = 256

        X_fp32 = np.linspace(-20.5, 8., num=n).astype(np.float32).reshape(1, n)
        Y_fp32 = self._swish(X_fp32)
        X_scale, X_zero_point = self._get_scale_zp(X_fp32)
        Y_scale, Y_zero_point = self._get_scale_zp(Y_fp32)
        W_fp32 = np.identity(n, dtype=np.float32)
        b_fp32 = np.zeros((n,), dtype=np.float32)

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

        ref_net1 = core.Net("net")
        ref_net1.Int8QuantizeNNPI(
            ["X"],
            ["X_int8"],
            Y_scale=X_scale,
            Y_zero_point=X_zero_point
        )
        ref_net1.Int8FCFakeAcc32NNPI(
            ["X_int8", "W_int8", "b"],
            ["U_int8"],
            Y_scale=X_scale,
            Y_zero_point=X_zero_point,
        )
        ref_net1.SwishFakeInt8NNPI(
            ["U_int8"],
            ["Y"],
            X_scale=X_scale,
            X_zero_point=X_zero_point,
            Y_scale=Y_scale,
            Y_zero_point=Y_zero_point
        )
        ref_net1.Proto().external_output.append("Y")

        ref_net = core.Net("net")
        ref_net.Int8QuantizeNNPI(
            ["X"],
            ["X_int8"],
            Y_scale=X_scale,
            Y_zero_point=X_zero_point
        )
        ref_net.Int8FCFakeAcc32NNPI(
            ["X_int8", "W_int8", "b"],
            ["U_int8"],
            Y_scale=X_scale,
            Y_zero_point=X_zero_point,
        )
        ref_net.Int8DequantizeNNPI(
            ["U_int8"],
            ["U_fp16"],
            UsingOneOverScale=False
        )
        ref_net.SwishFakeFp16NNPI(
            ["U_fp16"],
            ["Y_fp16"]
        )
        ref_net.Int8QuantizeNNPI(
            ["Y_fp16"],
            ["Y"],
            Y_scale=Y_scale,
            Y_zero_point=Y_zero_point
        )
        ref_net.Proto().external_output.append("Y")

        # run ref_net
        workspace.RunNetOnce(ref_net1)
        Y_fbgemm = workspace.FetchInt8Blob("Y")

        # run onnxifi net
        ref_net.Proto().op[0].type = "Int8Quantize"
        ref_net.Proto().op[1].type = "Int8FC"
        ref_net.Proto().op[2].type = "Int8Dequantize"
        ref_net.Proto().op[3].type = "Swish"
        ref_net.Proto().op[4].type = "Int8Quantize"
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
        # TODO: add an assertion to check the optimized net
        # fused Dequantize->Swish->Quantize to QuantizedSwish
        workspace.CreateNet(net_onnxified)
        workspace.RunNet(net_onnxified.name)
        Y_glow = workspace.FetchInt8Blob("Y")
        U_int8 = workspace.FetchInt8Blob("U_int8")

        diff_Y = np.abs(Y_glow.data - Y_fbgemm.data)

        num_mismatches = np.count_nonzero(diff_Y)
        max_diff = np.max(diff_Y)
        if max_diff > 0 or Y_glow.scale != Y_fbgemm.scale or \
           Y_glow.zero_point != Y_fbgemm.zero_point:
            print_test_debug_info(
                "QuantizedSwish",
                {
                    "X": X_fp32,
                    "X_scale": X_scale,
                    "X_zero_point": X_zero_point,
                    "Y_scale": Y_scale,
                    "Y_zero_point": Y_zero_point,
                    "U_int8": U_int8,
                    "Y_fbgemm": Y_fbgemm,
                    "Y_glow": Y_glow,
                    "diff": diff_Y,
                    "max_diff": max_diff,
                    "num_mismatches": num_mismatches,
                },
            )
            assert 0
