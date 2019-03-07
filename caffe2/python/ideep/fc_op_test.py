from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
from functools import reduce
import hypothesis.strategies as st
from hypothesis import given, settings
import numpy as np
from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.ideep_test_util as mu


@unittest.skipIf(not workspace.C.use_mkldnn, "No MKLDNN support.")
class FcTest(hu.HypothesisTestCase):
    @given(n=st.integers(1, 5), m=st.integers(1, 5),
           k=st.integers(1, 5), **mu.gcs)
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


if __name__ == "__main__":
    unittest.main()
