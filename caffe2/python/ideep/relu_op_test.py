




import unittest
import hypothesis.strategies as st
from hypothesis import given, settings
import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.ideep_test_util as mu

@unittest.skipIf(not workspace.C.use_mkldnn, "No MKLDNN support.")
class ReluTest(hu.HypothesisTestCase):
    @given(X=hu.tensor(),
           inplace=st.booleans(),
           **mu.gcs)
    @settings(deadline=1000)
    def test_relu(self, X, inplace, gc, dc):
        op = core.CreateOperator(
            "Relu",
            ["X"],
            ["Y"] if not inplace else ["X"],
        )
        # go away from the origin point to avoid kink problems
        X += 0.02 * np.sign(X)
        X[X == 0.0] += 0.02

        self.assertDeviceChecks(dc, op, [X], [0])

        self.assertGradientChecks(gc, op, [X], 0, [0])

    @given(size=st.integers(7, 9),
           input_channels=st.integers(1, 3),
           batch_size=st.integers(1, 3),
           inplace=st.booleans(),
           **mu.gcs_cpu_ideep)
    @settings(max_examples=10, deadline=None)
    def test_int8_relu(self, size, input_channels, batch_size, inplace, gc, dc):
        relu_fp32 = core.CreateOperator(
            "Relu",
            ["X"],
            ["Y"] if not inplace else ["X"],
            device_option=dc[0]
        )

        X = np.random.rand(
            batch_size, input_channels, size, size).astype(np.float32) - 0.5
        # go away from the origin point to avoid kink problems
        X += 0.02 * np.sign(X)
        X[X == 0.0] += 0.02

        if X.min() >=0:
            scale = np.absolute(X).max() / 0xFF
            zero_point = 0
        else:
            scale = np.absolute(X).max() / 0x7F
            zero_point = 128

        old_ws_name = workspace.CurrentWorkspace()
        workspace.SwitchWorkspace("_device_check_", True)

        workspace.FeedBlob("X", X, dc[0])
        workspace.RunOperatorOnce(relu_fp32)
        Y = workspace.FetchBlob("X" if inplace else "Y")

        workspace.ResetWorkspace()

        sw2nhwc = core.CreateOperator(
            "NCHW2NHWC",
            ["Xi"],
            ["Xi_nhwc"],
            device_option=dc[1]
        )

        quantize = core.CreateOperator(
            "Int8Quantize",
            ["Xi_nhwc"],
            ["Xi_quantized"],
            engine="DNNLOWP",
            device_option=dc[1],
            Y_zero_point=zero_point,
            Y_scale=scale,
        )

        relu = core.CreateOperator(
            "Int8Relu",
            ["Xi_quantized"],
            ["Y_quantized"] if not inplace else ["Xi_quantized"],
            engine="DNNLOWP",
            device_option=dc[1],
        )

        dequantize = core.CreateOperator(
            "Int8Dequantize",
            ["Y_quantized"] if not inplace else ["Xi_quantized"],
            ["Y_nhwc"],
            engine="DNNLOWP",
            device_option=dc[1],
        )

        sw2nchw = core.CreateOperator(
            "NHWC2NCHW",
            ["Y_nhwc"],
            ["Y_out"],
            device_option=dc[1]
        )

        net = caffe2_pb2.NetDef()
        net.op.extend([sw2nhwc, quantize, relu, dequantize, sw2nchw])

        workspace.FeedBlob("Xi", X, dc[1])
        workspace.RunNetOnce(net)
        Y_out = workspace.FetchBlob("Y_out")

        MSE = np.square(np.subtract(Y, Y_out)).mean()
        if MSE > 0.005:
            print(Y.flatten())
            print(Y_out.flatten())
            print(np.max(np.abs(Y_out - Y)))
            print("MSE", MSE)
            self.assertTrue(False)

        workspace.SwitchWorkspace(old_ws_name)

if __name__ == "__main__":
    unittest.main()
