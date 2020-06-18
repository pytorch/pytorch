from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import caffe2.python.fakelowp.init_shared_libs  # noqa
import numpy as np
from caffe2.python import core, workspace
from caffe2.python.onnx.onnxifi import onnxifi_caffe2_net
from hypothesis import given, note, strategies as st, settings
from caffe2.python.fakelowp.test_utils import print_test_debug_info
import caffe2.python.serialized_test.serialized_test_util as serial

core.GlobalInit(["caffe2", "--caffe2_log_level=-3", "--glow_global_fp16=1"])


class Int8OpsTest(serial.SerializedTestCase):
    def _get_scale_zp(self, tensor):
        tensor_max = np.max(tensor)
        tensor_min = min(0, np.min(tensor))
        scale = np.float32(np.float16((tensor_max - tensor_min) / 255.0))
        zero_point = 0 - tensor_min / scale
        zero_point = int(round(np.clip(zero_point, 0, 255.0)))
        return (scale, zero_point)

    @settings(max_examples=1)
    @given(
        n=st.integers(2, 1024), m=st.integers(2, 1024), rand_seed=st.integers(0, 65534)
    )
    def _test_int8_quantize(self, n, m, rand_seed):
        note("n={}, m={}, rand_seed={}".format(n, m, rand_seed))
        np.random.seed(rand_seed)
        X_fp16 = np.random.rand(n, m).astype(np.float16)
        X_fp32 = X_fp16.astype(np.float32)
        scale, zero_point = self._get_scale_zp(X_fp32)

        print("X scale zp", scale, zero_point)
        ref_net = core.Net("net")
        ref_net.Int8QuantizeNNPI(
            ["X"], ["X_int8"], Y_scale=scale, Y_zero_point=zero_point
        )
        ref_net.Int8DequantizeNNPI(["X_int8"], ["Y"])
        ref_net.Proto().external_output.extend(["X_int8"])

        # run ref net
        workspace.ResetWorkspace()
        workspace.FeedBlob("X", X_fp32)
        workspace.RunNetOnce(ref_net)


        X_int8 = workspace.FetchInt8Blob("X_int8")
        print("after running ", X_int8)
        Y_fbgemm = workspace.FetchBlob("Y")

        # run onnxifi net
        workspace.ResetWorkspace()
        workspace.FeedBlob("X", X_fp32)
        ref_net.Proto().op[0].type = "Int8Quantize"
        ref_net.Proto().op[1].type = "Int8Dequantize"

        net_onnxified = onnxifi_caffe2_net(
            ref_net.Proto(),
            {},
            debug=True,
            adjust_batch=False,
            use_onnx=False,
            weight_names=[],
        )
        num_onnxified_ops = sum(
            1 if o.type == "Onnxifi" else 0 for o in net_onnxified.op
        )

        # np.testing.assert_equal(num_onnxified_ops, 1)

        workspace.CreateNet(net_onnxified)
        workspace.RunNet(net_onnxified.name)
        X_int8_glow = workspace.FetchInt8Blob("X_int8")
        Y_glow = workspace.FetchBlob("Y")
        np.testing.assert_allclose(Y_fbgemm, Y_glow)

    @given(
        n=st.integers(1, 1024),
        m=st.integers(1, 1024),
        k=st.integers(1, 1024),
        rand_seed=st.integers(0, 65534),
        quantize_bias=st.sampled_from([False]),
    )
    def test_int8_fc(
        self, n, m, k, rand_seed, quantize_bias
    ):  # Int8FCFakeAcc32NNPI only supports quantize_bias=True
        print(
            "n={}, m={}, k={}, rand_seed={}, quantize_bias={}".format(
                n, m, k, rand_seed, quantize_bias
            )
        )
        np.random.seed(rand_seed)
        workspace.ResetWorkspace()

        X_fp32 = np.random.rand(m, k).astype(np.float16).astype(np.float32)
        W_fp32 = np.random.rand(n, k).astype(np.float32)
        b_fp32 = np.random.rand(n).astype(np.float16).astype(np.float32)

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
            ["X"], ["X_int8"], Y_scale=X_scale, Y_zero_point=X_zero_point
        )
        ref_net.Int8FCFakeAcc32NNPI(
            ["X_int8", "W_int8", "b_int32" if quantize_bias else "b"],
            ["Y_int8"],
            Y_scale=Y_scale,
            Y_zero_point=Y_zero_point,
        )
        ref_net.Int8DequantizeNNPI(["Y_int8"], ["Y"])
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
        np.testing.assert_allclose(num_onnxified_ops, 1)
        workspace.CreateNet(net_onnxified)
        workspace.RunNet(net_onnxified.name)
        Y_glow = workspace.FetchBlob("Y")

        diff = Y_fbgemm - Y_glow
        if np.count_nonzero(diff) * 10 > diff.size:
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
                    "Y_fbgemm": Y_fbgemm.shape,
                    "Y_glow": Y_glow.shape,
                    "diff": diff,
                    "maxdiff": diff.max(axis=1),
                },
            )
            assert 0
