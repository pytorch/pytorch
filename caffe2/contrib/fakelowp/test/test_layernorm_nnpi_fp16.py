from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import caffe2.python.fakelowp.init_shared_libs  # noqa
import time
from caffe2.proto import caffe2_pb2
from caffe2.python import core
from caffe2.python import workspace
from caffe2.python.onnx.onnxifi import onnxifi_caffe2_net
from caffe2.python.onnx.tests.test_utils import TestCase
from caffe2.python.fakelowp.test_utils import print_test_debug_info

core.GlobalInit(["caffe2",
                 "--glow_global_fp16=1",
                 "--glow_global_fused_scale_offset_fp16=1",
                 "--glow_global_force_sls_fp16_accum=1"])

GLOW_LOWERED_BATCHNORM = False


# Test the lowered LayerNorm op
class LayerNorm(TestCase):
    def _test_layernorm(self):
        size = 3
        input_channels = 2
        batch_size = 4
        seed = int(time.time())
        np.random.seed(seed)

        epsilon = 1e-3

        pred_net = caffe2_pb2.NetDef()
        pred_net.name = "pred"
        pred_net.external_input.extend(["X"])
        pred_net.external_output.extend(["Y", "mean", "rstd"])
        pred_net.op.add().CopyFrom(
            core.CreateOperator(
                "LayerNorm",
                ["X"],
                ["Y", "mean", "rstd"],
                # axis=-1,
                epsilon=epsilon
            )
        )

        pred_net_ref = caffe2_pb2.NetDef()
        pred_net_ref.name = "pred"
        pred_net_ref.external_input.extend(["X"])
        pred_net_ref.external_output.extend(["Y", "mean", "rstd"])
        pred_net_ref.op.add().CopyFrom(
            core.CreateOperator(
                "LayerNormFakeFP16",
                ["X"],
                ["Y", "mean", "rstd"],
                # axis=-1,
                epsilon=epsilon
            )
        )

        X = np.random.rand(
            batch_size, input_channels, size, size).astype(np.float32) - 0.5

        pred_net_onnxified = onnxifi_caffe2_net(
            pred_net,
            {"X": [batch_size, input_channels, size, size]},
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
        mean_c2 = workspace.FetchBlob("mean")
        std_c2 = workspace.FetchBlob("rstd")

        workspace.RunNet(pred_net_onnxified.name)
        Y_glow = workspace.FetchBlob("Y")
        mean_glow = workspace.FetchBlob("mean")
        std_glow = workspace.FetchBlob("rstd")

        if not np.allclose(Y_glow.astype(np.float16), Y_c2.astype(np.float16)):
            diff_Y = np.abs(Y_glow - Y_c2).astype(np.float16)
            diff_std = np.abs(std_glow - std_c2).astype(np.float16)
            diff_mean = np.abs(mean_glow - mean_c2).astype(np.float16)
            print_test_debug_info(
                "layernorm",
                {
                    "seed": seed,
                    "X": X,
                    "Y_glow": Y_glow,
                    "Y_c2": Y_c2,
                    "Y": diff_Y,
                    "mean": diff_mean,
                    "std": diff_std,
                }
            )
            assert(0)
