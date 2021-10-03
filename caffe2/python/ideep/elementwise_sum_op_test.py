




import unittest
import hypothesis.strategies as st
from hypothesis import given
import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.ideep_test_util as mu


@unittest.skipIf(not workspace.C.use_mkldnn, "No MKLDNN support.")
class ElementwiseSumTest(hu.HypothesisTestCase):
    @given(size=st.integers(7, 9),
           input_channels=st.integers(1, 3),
           batch_size=st.integers(1, 3),
           inputs=st.integers(2, 7),
           inplace=st.booleans(),
           **mu.gcs)
    def test_elementwise_sum(self,
                                 size,
                                 input_channels,
                                 batch_size,
                                 inputs,
                                 inplace,
                                 gc,
                                 dc):
        op = core.CreateOperator(
            "Sum",
            ["X_{}".format(i) for i in range(inputs)],
            ["X_0" if inplace else "Y"],
        )
        Xs = [np.random.rand(batch_size, input_channels, size, size).astype(
            np.float32) for _ in range(inputs)]
        self.assertDeviceChecks(dc, op, Xs, [0])


    @given(size=st.integers(7, 9),
           input_channels=st.integers(1, 3),
           batch_size=st.integers(1, 3),
           inputs=st.integers(2, 7),
           inplace=st.booleans(),
           **mu.gcs_cpu_ideep)
    def test_elementwise_sum_fallback(self,
                                      size,
                                      input_channels,
                                      batch_size,
                                      inputs,
                                      inplace,
                                      gc,
                                      dc):
        op = core.CreateOperator(
            "Sum",
            ["X_{}".format(i) for i in range(inputs)],
            ["X_0" if inplace else "Y"],
            device_option=dc[1]
        )
        Xs = [np.random.rand(batch_size, input_channels, size, size).astype(
            np.float32) for _ in range(inputs)]

        sum_val = Xs[0]
        workspace.FeedBlob("X_0", Xs[0], dc[0])
        for i, x in enumerate(Xs):
            if i == 0: continue
            sum_val += x
            workspace.FeedBlob("X_{}".format(i), x, dc[1])

        workspace.RunOperatorOnce(op)
        Y = workspace.FetchBlob("X_0" if inplace else "Y")

        if not np.allclose(sum_val, Y, atol=0.01, rtol=0.01):
            print(Y.flatten())
            print(sum_val.flatten())
            print(np.max(np.abs(Y - sum_val)))
            self.assertTrue(False)


    @given(size=st.integers(7, 9),
           input_channels=st.integers(1, 3),
           batch_size=st.integers(1, 3),
           inputs=st.integers(2, 7),
           inplace=st.booleans(),
           **mu.gcs_cpu_ideep)
    def test_int8_elementwise_sum(self,
                                 size,
                                 input_channels,
                                 batch_size,
                                 inputs,
                                 inplace,
                                 gc,
                                 dc):
        sum_fp32 = core.CreateOperator(
            "Sum",
            ["X_{}".format(i) for i in range(inputs)],
            ["X_0" if inplace else "Y"],
        )
        Xs = [np.random.rand(batch_size, input_channels, size, size).astype(
            np.float32) for _ in range(inputs)]

        old_ws_name = workspace.CurrentWorkspace()
        workspace.SwitchWorkspace("_device_check_", True)

        Xi_scales = []
        Xi_zero_points = []
        for i, X in enumerate(Xs):
            workspace.FeedBlob("X_{}".format(i), X, dc[0])
            if X.min() >= 0:
                Xi_scales.append(np.absolute(X).max() / 0xFF)
                Xi_zero_points.append(0)
            else:
                Xi_scales.append(np.absolute(X).max() / 0x7F)
                Xi_zero_points.append(128)

        workspace.RunOperatorOnce(sum_fp32)
        Y = workspace.FetchBlob("X_0" if inplace else "Y")

        if Y.min() >= 0:
            Y_scale = np.absolute(Y).max() / 0xFF
            Y_zero_point = 0
        else:
            Y_scale = np.absolute(Y).max() / 0x7F
            Y_zero_point = 128

        workspace.ResetWorkspace()

        net = caffe2_pb2.NetDef()
        for i, Xi in enumerate(Xs):
            workspace.FeedBlob("Xi_{}".format(i), Xi, dc[1])
            sw2nhwc = core.CreateOperator(
                "NCHW2NHWC",
                ["Xi_{}".format(i)],
                ["Xi_{}_nhwc".format(i)],
                device_option=dc[1]
            )
            quantize = core.CreateOperator(
                "Int8Quantize",
                ["Xi_{}_nhwc".format(i)],
                ["Xi_{}_quantized".format(i)],
                engine="DNNLOWP",
                device_option=dc[1],
                Y_zero_point=Xi_zero_points[i],
                Y_scale=Xi_scales[i],
            )
            net.op.extend([sw2nhwc, quantize])

        sum = core.CreateOperator(
            "Int8Sum",
            ["Xi_{}_quantized".format(i) for i in range(inputs)],
            ["Xi_0_quantized" if inplace else "Y_quantized"],
            engine="DNNLOWP",
            device_option=dc[1],
            Y_zero_point=Y_zero_point,
            Y_scale=Y_scale,
        )

        dequantize = core.CreateOperator(
            "Int8Dequantize",
            ["Xi_0_quantized" if inplace else "Y_quantized"],
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

        net.op.extend([sum, dequantize, sw2nchw])
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
