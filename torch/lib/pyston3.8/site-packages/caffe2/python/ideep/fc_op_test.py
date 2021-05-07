




import unittest
from functools import reduce
import hypothesis.strategies as st
from hypothesis import given, settings
import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.ideep_test_util as mu


@unittest.skipIf(not workspace.C.use_mkldnn, "No MKLDNN support.")
class FcTest(hu.HypothesisTestCase):
    @given(n=st.integers(1, 5), m=st.integers(1, 5),
           k=st.integers(1, 5), **mu.gcs)
    @settings(deadline=1000)
    def test_fc_2_dims(self, n, m, k, gc, dc):
        X = np.random.rand(m, k).astype(np.float32) - 0.5
        W = np.random.rand(n, k).astype(np.float32) - 0.5
        b = np.random.rand(n).astype(np.float32) - 0.5

        op = core.CreateOperator(
            'FC',
            ['X', 'W', 'b'],
            ["Y"]
        )

        self.assertDeviceChecks(dc, op, [X, W, b], [0])

        for i in range(3):
            self.assertGradientChecks(gc, op, [X, W, b], i, [0])

    @given(n=st.integers(1, 5),
           m=st.integers(1, 5),
           c=st.integers(1, 5),
           h=st.integers(1, 5),
           w=st.integers(1, 5),
           axis=st.integers(1, 3),
           **mu.gcs)
    def test_fc_with_axis(self, n, m, c, h, w, axis, gc, dc):
        X = np.random.rand(n, c, h, w).astype(np.float32) - 0.5
        k = reduce((lambda x, y: x * y), [n, c, h, w][axis - 4:])
        nn = reduce((lambda x, y: x * y), [n, c, h, w][:axis])
        W = np.random.rand(m, k).astype(np.float32) - 0.5
        b = np.random.rand(m).astype(np.float32) - 0.5
        dY = np.random.rand(nn, m).astype(np.float32) - 0.5

        op0 = core.CreateOperator(
            'FC',
            ['X', 'W', 'b'],
            ["Y"],
            axis=axis,
            device_option=dc[0]
        )

        op0_bw = core.CreateOperator(
            'FCGradient',
            ['X', 'W', 'dY'],
            ["dW", "db"],
            axis=axis,
            device_option=dc[0]
        )

        workspace.ResetWorkspace()
        workspace.FeedBlob('X', X, dc[0])
        workspace.FeedBlob('W', W, dc[0])
        workspace.FeedBlob('b', b, dc[0])
        workspace.RunOperatorOnce(op0)
        Y0 = workspace.FetchBlob('Y')

        workspace.FeedBlob('dY', dY, dc[0])
        workspace.RunOperatorOnce(op0_bw)
        dW0 = workspace.FetchBlob('dW')
        db0 = workspace.FetchBlob('db')

        op1 = core.CreateOperator(
            'FC',
            ['X', 'W', 'b'],
            ["Y"],
            axis=axis,
            device_option=dc[1]
        )

        op1_bw = core.CreateOperator(
            'FCGradient',
            ['X', 'W', 'dY'],
            ["dW", "db"],
            axis=axis,
            device_option=dc[1]
        )

        workspace.SwitchWorkspace("_device_check_", True)
        workspace.FeedBlob('X', X, dc[1])
        workspace.FeedBlob('W', W, dc[1])
        workspace.FeedBlob('b', b, dc[1])
        workspace.RunOperatorOnce(op1)
        Y1 = workspace.FetchBlob('Y')

        workspace.FeedBlob('dY', dY, dc[1])
        workspace.RunOperatorOnce(op1_bw)
        dW1 = workspace.FetchBlob('dW')
        db1 = workspace.FetchBlob('db')

        Y0 = Y0.flatten()
        Y1 = Y1.flatten()
        if not np.allclose(Y0, Y1, atol=0.01, rtol=0.01):
            print(Y1)
            print(Y0)
            print(np.max(np.abs(Y1 - Y0)))
            self.assertTrue(False)

        dW0 = dW0.flatten()
        dW1 = dW1.flatten()
        if not np.allclose(dW0, dW1, atol=0.01, rtol=0.01):
            print(dW1)
            print(dW0)
            print(np.max(np.abs(dW1 - dW0)))
            self.assertTrue(False)

        db0 = db0.flatten()
        db1 = db1.flatten()
        if not np.allclose(db0, db1, atol=0.01, rtol=0.01):
            print(db1)
            print(db0)
            print(np.max(np.abs(db1 - db0)))
            self.assertTrue(False)

    @given(n=st.integers(1, 5),
           o=st.integers(1, 5),
           i=st.integers(1, 5),
           h=st.integers(1, 5),
           w=st.integers(1, 5),
           axis_w=st.integers(1, 3),
           **mu.gcs)
    @settings(deadline=1000)
    def test_fc_with_axis_w(self, n, o, i, h, w, axis_w, gc, dc):
        W = np.random.rand(o, i, h, w).astype(np.float32) - 0.5
        k = reduce((lambda x, y: x * y), [o, i, h, w][axis_w - 4:])
        m = reduce((lambda x, y: x * y), [o, i, h, w][:axis_w])
        X = np.random.rand(n, k).astype(np.float32) - 0.5
        b = np.random.rand(m).astype(np.float32) - 0.5
        dY = np.random.rand(n, m).astype(np.float32) - 0.5

        op0 = core.CreateOperator(
            'FC',
            ['X', 'W', 'b'],
            ["Y"],
            axis_w=axis_w,
            device_option=dc[0]
        )

        op0_bw = core.CreateOperator(
            'FCGradient',
            ['X', 'W', 'dY'],
            ["dW", "db"],
            axis_w=axis_w,
            device_option=dc[0]
        )

        workspace.ResetWorkspace()
        workspace.FeedBlob('X', X, dc[0])
        workspace.FeedBlob('W', W, dc[0])
        workspace.FeedBlob('b', b, dc[0])
        workspace.RunOperatorOnce(op0)
        Y0 = workspace.FetchBlob('Y')

        workspace.FeedBlob('dY', dY, dc[0])
        workspace.RunOperatorOnce(op0_bw)
        dW0 = workspace.FetchBlob('dW')
        db0 = workspace.FetchBlob('db')

        op1 = core.CreateOperator(
            'FC',
            ['X', 'W', 'b'],
            ["Y"],
            axis_w=axis_w,
            device_option=dc[1]
        )

        op1_bw = core.CreateOperator(
            'FCGradient',
            ['X', 'W', 'dY'],
            ["dW", "db"],
            axis_w=axis_w,
            device_option=dc[1]
        )

        workspace.SwitchWorkspace("_device_check_", True)
        workspace.FeedBlob('X', X, dc[1])
        workspace.FeedBlob('W', W, dc[1])
        workspace.FeedBlob('b', b, dc[1])
        workspace.RunOperatorOnce(op1)
        Y1 = workspace.FetchBlob('Y')

        workspace.FeedBlob('dY', dY, dc[1])
        workspace.RunOperatorOnce(op1_bw)
        dW1 = workspace.FetchBlob('dW')
        db1 = workspace.FetchBlob('db')

        Y0 = Y0.flatten()
        Y1 = Y1.flatten()
        if not np.allclose(Y0, Y1, atol=0.01, rtol=0.01):
            print(Y1)
            print(Y0)
            print(np.max(np.abs(Y1 - Y0)))
            self.assertTrue(False)

        dW0 = dW0.flatten()
        dW1 = dW1.flatten()
        if not np.allclose(dW0, dW1, atol=0.01, rtol=0.01):
            print(dW1)
            print(dW0)
            print(np.max(np.abs(dW1 - dW0)))
            self.assertTrue(False)

        db0 = db0.flatten()
        db1 = db1.flatten()
        if not np.allclose(db0, db1, atol=0.01, rtol=0.01):
            print(db1)
            print(db0)
            print(np.max(np.abs(db1 - db0)))
            self.assertTrue(False)

    @given(n=st.integers(1, 5), m=st.integers(1, 5),
           k=st.integers(1, 5), **mu.gcs)
    @settings(deadline=10000)
    def test_fc_4_dims_src(self, n, m, k, gc, dc):
        X = np.random.rand(m, k, m, m).astype(np.float32) - 0.5
        W = np.random.rand(n, k * m * m).astype(np.float32) - 0.5
        b = np.random.rand(n).astype(np.float32) - 0.5

        op = core.CreateOperator(
            'FC',
            ['X', 'W', 'b'],
            ["Y"]
        )

        self.assertDeviceChecks(dc, op, [X, W, b], [0])

        for i in range(3):
            self.assertGradientChecks(gc, op, [X, W, b], i, [0])

    @given(n=st.integers(1, 5), m=st.integers(1, 5),
           k=st.integers(1, 5), **mu.gcs)
    @settings(deadline=10000)
    def test_fc_4_dims(self, n, m, k, gc, dc):
        X = np.random.rand(m, k, m, m).astype(np.float32) - 0.5
        W = np.random.rand(n, k, m, m).astype(np.float32) - 0.5
        b = np.random.rand(n).astype(np.float32) - 0.5

        op = core.CreateOperator(
            'FC',
            ['X', 'W', 'b'],
            ["Y"]
        )

        self.assertDeviceChecks(dc, op, [X, W, b], [0])

        for i in range(3):
            self.assertGradientChecks(gc, op, [X, W, b], i, [0])

    @given(n=st.integers(2, 5), m=st.integers(2, 5),
           k=st.integers(2, 5), **mu.gcs_cpu_ideep)
    def test_int8_fc_4_dims(self, n, m, k, gc, dc):
        X = np.random.rand(m, k, m, m).astype(np.float32) - 0.5
        w = np.random.rand(n, k, m, m).astype(np.float32) - 0.5
        b = np.random.rand(n).astype(np.float32) - 0.5

        fc_fp32 = core.CreateOperator(
            'FC',
            ['X', 'w', 'b'],
            ["Y"]
        )

        old_ws_name = workspace.CurrentWorkspace()
        workspace.SwitchWorkspace("_device_check_", True)

        workspace.FeedBlob('X', X, dc[0])
        workspace.FeedBlob('w', w, dc[0])
        workspace.FeedBlob('b', b, dc[0])
        workspace.RunOperatorOnce(fc_fp32)
        Y = workspace.FetchBlob('Y')

        workspace.ResetWorkspace()

        Y_absmax = np.array([np.absolute(Y).max()]).astype(np.float32)
        if Y.min() >= 0:
            Y_scale = Y_absmax / 0xFF
            Y_zero_point = 0
        else:
            Y_scale = Y_absmax / 0x7F
            Y_zero_point = 128

        X_absmax = np.array([np.absolute(X).max()]).astype(np.float32)
        if X.min() >= 0:
            X_scale = X_absmax / 0xFF
            X_zero_point = 0
        else:
            X_scale = X_absmax / 0x7F
            X_zero_point = 128

        w_absmax = np.array([np.absolute(w[i, ...]).max() for i in range(w.shape[0])]).astype(np.float32)
        w_scale = w_absmax / 0x7F
        w_zero_point = 128
        w = np.transpose(w, (0, 2, 3, 1)).astype(np.float32)
        w_bytes = np.rint([w[i, ...] / w_scale[i] for i in range(w.shape[0])]).astype(np.int8) + w_zero_point

        w_filler = core.CreateOperator(
            "Int8GivenTensorFill",
            [], ["wi"],
            shape=w.shape,
            values=w_bytes.astype(np.uint8).tobytes(),
            Y_zero_point=w_zero_point,
            Y_scales=w_scale,
            device_option=dc[1],
        )

        b_scale = w_scale * X_scale
        b_zero_point = 0
        b_bytes = np.rint([b[i] / b_scale[i] for i in range(b.shape[0])]).astype(np.int32)
        b_filler = core.CreateOperator(
            "Int8GivenIntTensorFill",
            [], ["bi"],
            shape=b.shape,
            values=b_bytes,
            Y_zero_point=b_zero_point,
            Y_scales=b_scale,
            device_option=dc[1],
        )

        sw2nhwc = core.CreateOperator(
            "NCHW2NHWC",
            ["Xi"],
            ["Xi_nhwc"],
            device_option=dc[1]
        )

        quantize_X = core.CreateOperator(
            "Int8Quantize",
            ["Xi_nhwc"],
            ["Xi_quantized"],
            engine="DNNLOWP",
            device_option=dc[1],
            Y_zero_point=X_zero_point,
            Y_scale=X_scale[0],
        )

        fc = core.CreateOperator(
            'Int8FC',
            ['Xi_quantized', 'wi', 'bi'],
            ["Y_out"],
            engine="DNNLOWP",
            device_option=dc[1],
            Y_zero_point=Y_zero_point,
            Y_scale=Y_scale[0],
        )

        net = caffe2_pb2.NetDef()
        net.op.extend([w_filler, b_filler, sw2nhwc, quantize_X, fc])

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
