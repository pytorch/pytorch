# Must happen before importing caffe2.python.*
import caffe2.python.fakelowp.init_shared_libs  # noqa
import datetime
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
from caffe2.python.onnx.onnxifi import onnxifi_caffe2_net
from caffe2.python.fakelowp.test_utils import print_test_debug_info
import caffe2.python.serialized_test.serialized_test_util as serial

workspace.GlobalInit(
    [
        "caffe2",
        "--glow_global_fp16=1",
        "--glow_global_fused_scale_offset_fp16=1",
        "--glow_global_force_sls_fp16_accum=1",
    ]
)

class Fusions(serial.SerializedTestCase):
    @given(
        scale=st.floats(1e-4, 1e2),
        zp=st.integers(-128, 128),
        size=st.integers(1, 100000),
        rand_seed=st.integers(0, 65534),
    )
    @settings(deadline=datetime.timedelta(seconds=10))
    def test_tanhquantize(self, scale, zp, size, rand_seed):
        np.random.seed(rand_seed)

        workspace.ResetWorkspace()

        pred_net = caffe2_pb2.NetDef()
        pred_net.name = "ref"
        pred_net.external_input.append("X")
        pred_net.external_output.append("Y_q")

        pred_net.op.add().CopyFrom(
            core.CreateOperator(
                "Tanh", ["X"], ["Y"]
            )
        )

        pred_net.op.add().CopyFrom(
            core.CreateOperator(
                "Int8Quantize", ["Y"], ["Y_q"], Y_scale=scale, Y_zero_point=zp
            )
        )

        X = np.linspace(-1, 1, size).astype(np.float16).astype(np.float32)

        pred_net_onnxified = onnxifi_caffe2_net(
            pred_net,
            {"X": X.shape},
            debug=True,
            adjust_batch=False,
            use_onnx=False,
        )
        num_onnxified_ops = sum(
            1 if o.type == "Onnxifi" else 0 for o in pred_net_onnxified.op
        )
        np.testing.assert_equal(num_onnxified_ops, 1)
        workspace.FeedBlob("X", X)
        workspace.CreateNet(pred_net_onnxified)
        workspace.RunNet(pred_net_onnxified.name)
        Y_glow = workspace.FetchInt8Blob("Y_q")

        ref_net = caffe2_pb2.NetDef()
        ref_net.name = "ref"
        ref_net.external_input.append("X")
        ref_net.external_output.append("Y_q")

        ref_net.op.add().CopyFrom(
            core.CreateOperator(
                "TanhQuantFakeFp16NNPI", ["X"], ["Y_q"], Y_scale=scale, Y_zero_point=zp
            )
        )

        workspace.CreateNet(ref_net)
        workspace.RunNet(ref_net.name)
        Y_ref = workspace.FetchInt8Blob("Y_q")

        if not np.array_equal(Y_ref.data, Y_glow.data) or \
           not Y_ref.scale == Y_glow.scale or \
           not Y_ref.zero_point == Y_glow.zero_point:
            print_test_debug_info(
                "tanhfusion",
                {
                    "scale": scale,
                    "zp": zp,
                    "input": X,
                    "ideal nonquant": np.tanh(X),
                    "Y_glow": Y_glow,
                    "Y_c2": Y_ref,
                }
            )
            assert(0)
