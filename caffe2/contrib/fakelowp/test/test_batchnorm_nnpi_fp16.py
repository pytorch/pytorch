import numpy as np
import unittest

import caffe2.python.fakelowp.init_shared_libs  # noqa
from hypothesis import given, settings
from hypothesis import strategies as st
from caffe2.proto import caffe2_pb2
from caffe2.python import core
from caffe2.python import workspace
from caffe2.python.onnx.onnxifi import onnxifi_caffe2_net
from caffe2.python.fakelowp.test_utils import print_test_debug_info
import caffe2.python.serialized_test.serialized_test_util as serial
import datetime

core.GlobalInit(["caffe2", "--glow_global_fp16=1",
                 "--glow_global_fused_scale_offset_fp16=1",
                 "--glow_global_force_sls_fp16_accum=1"])

GLOW_LOWERED_BATCHNORM = False


def reference_spatialbn_test16(X, scale, bias, mean, var, epsilon, order):
    X = X.astype(np.float16)
    scale = scale.astype(np.float16)
    bias = bias.astype(np.float16)
    mean = mean.astype(np.float16)
    # var = var.astype(np.float16)
    assert(order == "NCHW")

    scale = scale[np.newaxis, :, np.newaxis, np.newaxis]
    bias = bias[np.newaxis, :, np.newaxis, np.newaxis]
    mean = mean[np.newaxis, :, np.newaxis, np.newaxis]
    var = var[np.newaxis, :, np.newaxis, np.newaxis]
    Y = ((X - mean) * (scale / np.sqrt(var + epsilon).astype(np.float16))) + bias
    return Y.astype(np.float32)


# Test the lowered BN op
class BatchnormTest(serial.SerializedTestCase):
    # TODO: using hypothesis seed, sweep dimensions
    @given(seed=st.integers(0, 65535),
           size=st.integers(2, 30),
           input_channels=st.integers(2, 40),
           batch_size=st.integers(2, 20))
    @settings(deadline=datetime.timedelta(seconds=10))
    def test_bn(self, seed, size, input_channels, batch_size):
        workspace.ResetWorkspace()
        np.random.seed(seed)

        order = "NCHW"
        epsilon = 1e-3

        pred_net = caffe2_pb2.NetDef()
        pred_net.name = "pred"
        pred_net.external_input.extend(["X", "scale", "bias", "mean", "var"])
        pred_net.external_output.append("Y")
        pred_net.op.add().CopyFrom(
            core.CreateOperator(
                "SpatialBN",
                ["X", "scale", "bias", "mean", "var"],
                ["Y"],
                order=order,
                is_test=True,
                epsilon=epsilon
            )
        )

        if GLOW_LOWERED_BATCHNORM:
            refopname = "SpatialBNFakeLoweredFp16NNPI"
        else:
            refopname = "SpatialBNFakeFp16NNPI"

        pred_net_ref = caffe2_pb2.NetDef()
        pred_net_ref.name = "pred"
        pred_net_ref.external_input.extend(["X", "scale", "bias", "mean", "var"])
        pred_net_ref.external_output.append("X")
        pred_net_ref.op.add().CopyFrom(
            core.CreateOperator(
                refopname,
                ["X", "scale", "bias", "mean", "var"],
                ["Y"],
                order=order,
                is_test=True,
                epsilon=epsilon
            )
        )

        scale = np.random.rand(input_channels).astype(np.float32) + 0.5
        bias = np.random.rand(input_channels).astype(np.float32) - 0.5
        mean = np.random.randn(input_channels).astype(np.float32)
        var = np.random.rand(input_channels).astype(np.float32) + 0.5
        X = np.random.rand(
            batch_size, input_channels, size, size).astype(np.float32) - 0.5

        workspace.FeedBlob("scale", scale)
        workspace.FeedBlob("bias", bias)
        workspace.FeedBlob("mean", mean)
        workspace.FeedBlob("var", var)

        # Use for reference to debug
        # Y_np = reference_spatialbn_test16(X, scale, bias, mean, var, epsilon, order)

        pred_net_onnxified = onnxifi_caffe2_net(
            pred_net,
            {"X": [batch_size, input_channels, size, size],
             "scale": [input_channels],
             "bias": [input_channels],
             "mean": [input_channels],
             "var": [input_channels]},
            debug=True,
            adjust_batch=False,
            use_onnx=False
        )
        num_onnxified_ops = sum(
            1 if o.type == "Onnxifi" else 0 for o in pred_net_onnxified.op)
        np.testing.assert_equal(num_onnxified_ops, 1)

        workspace.FeedBlob("X", X)

        workspace.CreateNet(pred_net_onnxified)
        workspace.CreateNet(pred_net_ref)

        workspace.RunNet(pred_net_ref.name)
        Y_c2 = workspace.FetchBlob("Y")

        workspace.RunNet(pred_net_onnxified.name)
        Y_glow = workspace.FetchBlob("Y")

        if not np.allclose(Y_glow.astype(np.float16), Y_c2.astype(np.float16)):
            diff = np.abs(Y_glow - Y_c2).astype(np.float16)
            print_test_debug_info(
                "bn",
                {
                    "seed": seed,
                    "scale": scale,
                    "bias": bias,
                    "mean": mean,
                    "var": var,
                    "Y_np": Y_c2,
                    "Y_glow": Y_glow,
                    "diff": diff,
                    "rowwise_diff": np.max(np.abs(diff), -1)})
            assert(0)
