from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import caffe2.python.fakelowp.init_shared_libs  # noqa
from caffe2.proto import caffe2_pb2
from caffe2.python import core
from caffe2.python import workspace
from caffe2.python.onnx.onnxifi import onnxifi_caffe2_net
from caffe2.python.fakelowp.test_utils import print_test_debug_info
from hypothesis import given, settings
from hypothesis import strategies as st
import caffe2.python.serialized_test.serialized_test_util as serial

core.GlobalInit(["caffe2",
                 "--glow_global_fp16=1",
                 "--glow_global_fused_scale_offset_fp16=1",
                 "--glow_global_force_sls_fp16_accum=1"])

GLOW_LOWERED_BATCHNORM = False


# Test the lowered LayerNorm op
class LayerNorm(serial.SerializedTestCase):

    @given(seed=st.integers(0, 65535))
    @settings(max_examples=10)
    def test_layernorm(self, seed):
        np.random.seed(seed)
        # Reset the workspace
        size = 4
        input_channels = 4
        batch_size = 1
        axis = 1
        epsilon = 1e-4
        workspace.ResetWorkspace()

        dims = np.array(([batch_size, input_channels, size, size]))
        X = np.random.uniform(size=dims).astype(np.float32) - 0.5
        gamma = np.random.randn(*X.shape[axis:]).astype(np.float32)
        beta = np.random.randn(*X.shape[axis:]).astype(np.float32)

        pred_net = caffe2_pb2.NetDef()
        pred_net.name = "pred"
        pred_net.external_input.extend(["X", "gamma", "beta"])
        pred_net.external_output.extend(["Y", "mean", "rstd"])
        pred_net.op.add().CopyFrom(
            core.CreateOperator(
                "LayerNorm",
                ["X", "gamma", "beta"],
                ["Y", "mean", "rstd"],
                axis=1,
                epsilon=epsilon,
                elementwise_affine=True
            )
        )

        pred_net_ref = caffe2_pb2.NetDef()
        pred_net_ref.name = "pred_ref"
        pred_net_ref.external_input.extend(["X", "gamma", "beta"])
        pred_net_ref.external_output.extend(["Y", "mean", "rstd"])
        pred_net_ref.op.add().CopyFrom(
            core.CreateOperator(
                "LayerNormFakeFP16NNPI",
                ["X", "gamma", "beta"],
                ["Y", "mean", "rstd"],
                axis=1,
                epsilon=epsilon,
                elementwise_affine=True
            )
        )

        shape_hits = {"X": X.shape, "gamma": gamma.shape, "beta": beta.shape}
        pred_net_onnxified = onnxifi_caffe2_net(
            pred_net,
            shape_hits,
            debug=True,
            adjust_batch=True,
            use_onnx=False
        )
        num_onnxified_ops = sum(
            1 if o.type == "Onnxifi" else 0 for o in pred_net_onnxified.op)
        np.testing.assert_equal(num_onnxified_ops, 1)

        workspace.FeedBlob("X", X)
        workspace.FeedBlob("gamma", gamma)
        workspace.FeedBlob("beta", beta)

        workspace.CreateNet(pred_net_ref)
        workspace.CreateNet(pred_net_onnxified)

        workspace.RunNet(pred_net_ref.name)
        Y_c2 = workspace.FetchBlob("Y")

        workspace.RunNet(pred_net_onnxified.name)
        Y_glow = workspace.FetchBlob("Y")

        if not np.allclose(Y_glow.astype(np.float16), Y_c2.astype(np.float16)):
            diff_Y = np.abs(Y_glow - Y_c2).astype(np.float16)
            print_test_debug_info(
                "layernorm",
                {
                    "seed": seed,
                    "size": size,
                    "input_channels": input_channels,
                    "batch_size": batch_size,
                    "epsilon": epsilon,
                    "axis": axis,
                    "X": X,
                    "Y_glow": Y_glow,
                    "Y_c2": Y_c2,
                    "diff_Y": diff_Y,
                }
            )
            assert(0)
