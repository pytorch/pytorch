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
import datetime

core.GlobalInit(["caffe2",
                 "--glow_global_fp16=1",
                 "--glow_global_fused_scale_offset_fp16=1",
                 "--glow_global_force_sls_fp16_accum=1"])

GLOW_LOWERED_BATCHNORM = False


# Test the lowered LayerNorm op
class LayerNorm(serial.SerializedTestCase):

    @given(seed=st.integers(0, 65535),
           batch_size=st.integers(min_value=1, max_value=50),
           size=st.integers(min_value=2, max_value=128),
           epsilon=st.floats(min_value=1e-4, max_value=1e-3),
           elementwise_affine=st.booleans())
    @settings(deadline=datetime.timedelta(seconds=10))
    def test_layernorm(self, seed, batch_size, size, epsilon, elementwise_affine):
        np.random.seed(seed)
        # Reset the workspace
        workspace.ResetWorkspace()
        axis = 1

        dims = np.array(([batch_size, size]))
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
                ["X", "gamma", "beta"] if elementwise_affine else ["X"],
                ["Y", "mean", "rstd"],
                axis=axis,
                epsilon=epsilon,
                elementwise_affine=elementwise_affine
            )
        )

        pred_net_ref = caffe2_pb2.NetDef()
        pred_net_ref.name = "pred_ref"
        pred_net_ref.external_input.extend(["X", "gamma", "beta"])
        pred_net_ref.external_output.extend(["Y", "mean", "rstd"])
        pred_net_ref.op.add().CopyFrom(
            core.CreateOperator(
                "LayerNormFakeFP16NNPI",
                ["X", "gamma", "beta"] if elementwise_affine else ["X"],
                ["Y", "mean", "rstd"],
                axis=axis,
                epsilon=epsilon,
                elementwise_affine=elementwise_affine
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

        dims1 = np.array(([1, *dims]))
        X_glow = X.reshape(dims1)
        workspace.FeedBlob("X", X_glow)

        workspace.RunNet(pred_net_onnxified.name)
        Y_glow = workspace.FetchBlob("Y")

        if not np.allclose(Y_glow, Y_c2):
            diff_Y = np.abs(Y_glow - Y_c2)
            print_test_debug_info(
                "layernorm",
                {
                    "seed": seed,
                    "size": size,
                    "batch_size": batch_size,
                    "epsilon": epsilon,
                    "gamma": gamma,
                    "beta": beta,
                    "elementwise_affine": elementwise_affine,
                    "X": X,
                    "Y_glow": Y_glow,
                    "Y_c2": Y_c2,
                    "diff_Y": diff_Y,
                }
            )
            assert(0)

    def _get_scale_zp(self, tensor):
        tensor_max = np.max(tensor)
        tensor_min = min(0, np.min(tensor))
        scale = np.float32(np.float16((tensor_max - tensor_min) / 255.0))
        if scale < 1e-6:
            scale = np.float32(1e-6)
        zero_point = 0 - tensor_min / scale
        zero_point = int(round(np.clip(zero_point, 0, 255.0)))
        return (scale, zero_point)

    def _layernorm_transform(self, X):
        mean = np.mean(X, axis=1)
        mean_exp = np.outer(mean, np.ones(X.shape[1]))
        std = np.std(X, axis=1)
        std_exp = np.outer(std, np.ones(X.shape[1]))
        Y = (X - mean_exp) / std_exp
        return Y

    @given(seed=st.integers(0, 65535),
           batch_size=st.integers(min_value=1, max_value=50),
           size=st.integers(min_value=2, max_value=128),
           epsilon=st.floats(min_value=1e-4, max_value=1e-3),
           elementwise_affine=st.booleans())
    @settings(deadline=datetime.timedelta(seconds=10))
    # re-enable when T74553975 gets fixed
    def test_fused_ln_quantize(self, seed, batch_size, size, epsilon, elementwise_affine):
        np.random.seed(seed)

        # Reset the workspace
        workspace.ResetWorkspace()
        axis = 1

        dims = np.array(([batch_size, size]))
        X = np.random.uniform(size=dims).astype(np.float32) - 0.5
        gamma = np.random.randn(*X.shape[axis:]).astype(np.float32)
        beta = np.random.randn(*X.shape[axis:]).astype(np.float32)

        Y = self._layernorm_transform(X)
        scale, zp = self._get_scale_zp(Y)

        pred_net = caffe2_pb2.NetDef()
        pred_net.name = "pred"
        pred_net.external_input.extend(["X", "gamma", "beta"])
        pred_net.external_output.extend(["Y_q"])
        pred_net.op.add().CopyFrom(
            core.CreateOperator(
                "LayerNorm",
                ["X", "gamma", "beta"] if elementwise_affine else ["X"],
                ["Y", "mean", "rstd"],
                axis=axis,
                epsilon=epsilon,
                elementwise_affine=elementwise_affine
            )
        )
        pred_net.op.add().CopyFrom(
            core.CreateOperator(
                "Int8Quantize", ["Y"], ["Y_q"], Y_scale=scale, Y_zero_point=zp
            )
        )

        print(pred_net)
        pred_net_ref = caffe2_pb2.NetDef()
        pred_net_ref.name = "pred_ref"
        pred_net_ref.external_input.extend(["X", "gamma", "beta"])
        pred_net_ref.external_output.extend(["Y_q"])
        pred_net_ref.op.add().CopyFrom(
            core.CreateOperator(
                "LayerNormInt8QuantizeFakeNNPI",
                ["X", "gamma", "beta"] if elementwise_affine else ["X"],
                ["Y_q", "mean", "rstd"],
                axis=axis,
                epsilon=epsilon,
                elementwise_affine=elementwise_affine,
                Y_scale=scale, Y_zero_point=zp
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
        Y_c2 = workspace.FetchInt8Blob("Y_q")

        workspace.RunNet(pred_net_onnxified.name)
        Y_glow = workspace.FetchInt8Blob("Y_q")

        if not np.allclose(Y_glow.data, Y_c2.data) or \
           Y_glow.scale != Y_c2.scale or Y_glow.zero_point != Y_c2.zero_point:
            diff_Y = np.abs(Y_glow.data.astype(np.float32) - Y_c2.data.astype(np.float32))
            print_test_debug_info(
                "layernorm",
                {
                    "seed": seed,
                    "size": size,
                    "batch_size": batch_size,
                    "epsilon": epsilon,
                    "gamma": gamma,
                    "beta": beta,
                    "elementwise_affine": elementwise_affine,
                    "X": X,
                    "Y_glow": Y_glow,
                    "Y_c2": Y_c2,
                    "diff_Y": diff_Y,
                }
            )
            assert(0)
