from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np


class TestIntegralImageOps(serial.SerializedTestCase):
    @serial.given(batch_size=st.integers(1, 3),
           height=st.integers(7, 10),
           width=st.integers(7, 10),
           channels=st.integers(1, 8),
           **hu.gcs)
    def test_integral_image_ops(self, batch_size, height, width, channels, gc, dc):
        N = batch_size
        C = channels
        H = height
        W = width

        im = np.random.rand(N, C, H, W).astype(np.float32)
        op = core.CreateOperator("IntegralImage",
                                 ["im"], ["y"])

        def integral_image(im):
            y = np.random.rand(N, C, H + 1, W + 1).astype(np.float32)
            for i1 in range(N):
                for i2 in range(C):
                    for i3 in range(W + 1):
                        y[i1, i2, 0, i3] = 0
                    for i3 in range(H + 1):
                        y[i1, i2, i3, 0] = 0
                    for i3 in range(1, H + 1):
                        for i4 in range(1, W + 1):
                            y[i1, i2, i3, i4] = im[i1, i2, i3 - 1, i4 - 1] + \
                                y[i1, i2, i3 - 1, i4] + \
                                y[i1, i2, i3, i4 - 1] - \
                                y[i1, i2, i3 - 1, i4 - 1]

            return [y]

        self.assertDeviceChecks(dc, op, [im], [0])
        self.assertReferenceChecks(gc, op, [im], integral_image)

    @serial.given(batch_size=st.integers(1, 3),
           height=st.integers(7, 10),
           width=st.integers(7, 10),
           channels=st.integers(1, 8),
           **hu.gcs)
    def test_integral_image_gradient_ops(self, batch_size,
    height, width, channels, gc, dc):
        N = batch_size
        C = channels
        H = height
        W = width

        X = np.random.rand(N, C, H, W).astype(np.float32)
        dY = np.random.rand(N, C, H + 1, W + 1).astype(np.float32)
        op = core.CreateOperator(
            "IntegralImageGradient",
            ["X", "dY"],
            ["dX"])

        def integral_image_gradient(X, dY):
            dX = np.random.rand(N, C, H, W).astype(np.float32)
            dX1 = np.random.rand(N, C, H + 1, W).astype(np.float32)
            #H+1,W+1=>H+1, W
            for i1 in range(N):
                for i2 in range(C):
                    for i3 in range(H + 1):
                        dX1[i1, i2, i3, 0] = dY[i1, i2, i3, 0]
                        for i4 in range(1, W):
                            dX1[i1, i2, i3, i4] = dY[i1, i2, i3, i4] + \
                                dX1[i1, i2, i3, i4 - 1]

            #H+1,W=>H,W
            for i1 in range(N):
                for i2 in range(C):
                    for i3 in range(W):
                        dX[i1, i2, 0, i3] = dX1[i1, i2, 0, i3]
                        for i4 in range(1, H):
                            dX[i1, i2, i4, i3] = dX1[i1, i2, i4, i3] + \
                                dX[i1, i2, i4 - 1, i3]
            return [dX]

        self.assertDeviceChecks(dc, op, [X, dY], [0])
        self.assertReferenceChecks(gc, op, [X, dY], integral_image_gradient)
