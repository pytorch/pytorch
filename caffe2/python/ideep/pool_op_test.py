




import unittest
import hypothesis.strategies as st
from hypothesis import assume, given, settings
import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.ideep_test_util as mu

@unittest.skipIf(not workspace.C.use_mkldnn, "No MKLDNN support.")
class PoolTest(hu.HypothesisTestCase):
    @given(stride=st.integers(1, 3),
           pad=st.integers(0, 3),
           kernel=st.integers(3, 5),
           size=st.integers(7, 9),
           input_channels=st.integers(1, 3),
           batch_size=st.integers(1, 3),
           method=st.sampled_from(["MaxPool", "AveragePool"]),
           **mu.gcs)
    @settings(deadline=10000)
    def test_pooling(self, stride, pad, kernel, size,
                         input_channels, batch_size,
                         method, gc, dc):
        assume(pad < kernel)
        op = core.CreateOperator(
            method,
            ["X"],
            ["Y"],
            stride=stride,
            pad=pad,
            kernel=kernel,
            device_option=dc[0],
        )
        X = np.random.rand(
            batch_size, input_channels, size, size
        ).astype(np.float32)

        self.assertDeviceChecks(dc, op, [X], [0])

        if 'MaxPool' not in method:
            self.assertGradientChecks(gc, op, [X], 0, [0])

    @given(stride=st.integers(1, 3),
           pad=st.integers(0, 3),
           kernel=st.integers(3, 5),
           size=st.integers(7, 9),
           input_channels=st.integers(1, 3),
           batch_size=st.integers(1, 3),
           method=st.sampled_from(["MaxPool", "AveragePool"]),
           **mu.gcs_cpu_ideep)
    def test_int8_pooling(self, stride, pad, kernel, size,
                         input_channels, batch_size,
                         method, gc, dc):
        assume(pad < kernel)
        pool_fp32 = core.CreateOperator(
            method,
            ["X"],
            ["Y"],
            stride=stride,
            pad=pad,
            kernel=kernel,
            device_option=dc[0]
        )
        X = np.random.rand(
            batch_size, input_channels, size, size).astype(np.float32)

        if X.min() >=0:
            scale = np.absolute(X).max() / 0xFF
            zero_point = 0
        else:
            scale = np.absolute(X).max() / 0x7F
            zero_point = 128

        old_ws_name = workspace.CurrentWorkspace()
        workspace.SwitchWorkspace("_device_check_", True)

        workspace.FeedBlob("X", X, dc[0])
        workspace.RunOperatorOnce(pool_fp32)
        Y = workspace.FetchBlob("Y")

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

        pool = core.CreateOperator(
            "Int8{}".format(method),
            ["Xi_quantized"],
            ["Y_quantized"],
            stride=stride,
            pad=pad,
            kernel=kernel,
            engine="DNNLOWP",
            device_option=dc[1],
        )

        dequantize = core.CreateOperator(
            "Int8Dequantize",
            ["Y_quantized"],
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
        net.op.extend([sw2nhwc, quantize, pool, dequantize, sw2nchw])

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
