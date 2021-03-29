# Must happen before importing caffe2.python.*
import caffe2.python.fakelowp.init_shared_libs  # noqa
import datetime
import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
from caffe2.python.onnx.onnxifi import onnxifi_caffe2_net
import caffe2.python.serialized_test.serialized_test_util as serial
from hypothesis import settings

workspace.GlobalInit(
    [
        "caffe2",
        "--glow_global_fp16=0",
        "--glow_global_fused_scale_offset_fp16=0",
        "--glow_global_force_sls_fp16_accum=0",
    ]
)

class QuantTest(serial.SerializedTestCase):
    @settings(deadline=datetime.timedelta(seconds=10))
    def test_dequantize(self):
        pred_net = caffe2_pb2.NetDef()
        pred_net.name = "pred"
        pred_net.external_input.append("X")
        pred_net.external_output.append("Y")
        x_scale = 0.10000000149011612
        pred_net.op.add().CopyFrom(
            core.CreateOperator(
                "Int8Quantize", ["X"], ["I"], Y_scale=x_scale, Y_zero_point=0
            )
        )
        pred_net.op.add().CopyFrom(
            core.CreateOperator(
                "Int8Dequantize", ["I"], ["Y"],
            )
        )
        print(pred_net)
        X = np.asarray([[1, 0], [0, 1]]).astype(np.float32)
        workspace.FeedBlob("X", X)
        workspace.CreateNet(pred_net)
        workspace.RunNet(pred_net.name)
        Y_ref = workspace.FetchBlob("Y")
        workspace.ResetWorkspace()
        pred_net_onnxified = onnxifi_caffe2_net(
            pred_net,
            {"X": [5, 2]},
            debug=True,
            adjust_batch=True,
            block_list=[0],
            use_onnx=False,
        )
        num_onnxified_ops = sum(
            1 if o.type == "Onnxifi" else 0 for o in pred_net_onnxified.op
        )
        np.testing.assert_equal(len(pred_net_onnxified.op), 2)
        np.testing.assert_equal(num_onnxified_ops, 1)
        workspace.FeedBlob("X", X)
        workspace.CreateNet(pred_net_onnxified)
        workspace.RunNet(pred_net_onnxified.name)
        Y_glow = workspace.FetchBlob("Y")
        np.testing.assert_equal(Y_ref, Y_glow)

    @settings(deadline=datetime.timedelta(seconds=20))
    def test_quantize(self):
        pred_net = caffe2_pb2.NetDef()
        pred_net.name = "pred"
        pred_net.external_input.append("X")
        pred_net.external_output.append("Y")
        x_scale = 0.10000000149011612
        pred_net.op.add().CopyFrom(
            core.CreateOperator(
                "Int8Quantize", ["X"], ["Y"], Y_scale=x_scale, Y_zero_point=0
            )
        )
        print(pred_net)
        X = np.asarray([[1, 0], [0, 1]]).astype(np.float32)
        workspace.FeedBlob("X", X)
        workspace.RunNetOnce(pred_net)
        Y_ref = workspace.FetchInt8Blob("Y")
        workspace.ResetWorkspace()
        pred_net_onnxified = onnxifi_caffe2_net(
            pred_net,
            {"X": [2, 2]},
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
        Y_glow = workspace.FetchInt8Blob("Y")
        np.testing.assert_equal(Y_ref.data, Y_glow.data)
