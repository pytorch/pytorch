



from caffe2.python import core, workspace
from hypothesis import given, settings

import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st

import numpy as np
import unittest


class TestTransposeOp(serial.SerializedTestCase):
    @given(
        X=hu.tensor(dtype=np.float32), use_axes=st.booleans(), **hu.gcs)
    @settings(deadline=None, max_examples=50)
    def test_transpose(self, X, use_axes, gc, dc):
        ndim = len(X.shape)
        axes = np.arange(ndim)
        np.random.shuffle(axes)

        if (use_axes):
            op = core.CreateOperator(
                "Transpose", ["X"], ["Y"], axes=axes, device_option=gc)
        else:
            op = core.CreateOperator(
                "Transpose", ["X"], ["Y"], device_option=gc)

        def transpose_ref(X):
            if use_axes:
                return [np.transpose(X, axes=axes)]
            else:
                return [np.transpose(X)]

        self.assertReferenceChecks(gc, op, [X], transpose_ref)
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0])

    @given(M=st.integers(10, 200), N=st.integers(10, 200), **hu.gcs)
    @settings(max_examples=10, deadline=None)
    def test_transpose_large_matrix(self, M, N, gc, dc):
        op = core.CreateOperator("Transpose", ["X"], ["Y"], device_option=gc)
        X = np.random.rand(M, N).astype(np.float32) - 0.5

        def transpose_ref(X):
            return [np.transpose(X)]

        self.assertReferenceChecks(gc, op, [X], transpose_ref)
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0])


    @unittest.skipIf(not workspace.has_cuda_support, "no cuda support")
    @given(X=hu.tensor(dtype=np.float32), use_axes=st.booleans(),
           **hu.gcs_cuda_only)
    def test_transpose_cudnn(self, X, use_axes, gc, dc):
        ndim = len(X.shape)
        axes = np.arange(ndim)
        np.random.shuffle(axes)

        if (use_axes):
            op = core.CreateOperator(
                "Transpose", ["X"], ["Y"], axes=axes, engine="CUDNN",
                device_option=hu.cuda_do)
        else:
            op = core.CreateOperator(
                "Transpose", ["X"], ["Y"], engine="CUDNN",
                device_option=hu.cuda_do)

        def transpose_ref(X):
            if use_axes:
                return [np.transpose(X, axes=axes)]
            else:
                return [np.transpose(X)]

        self.assertReferenceChecks(hu.gpu_do, op, [X], transpose_ref)
        self.assertGradientChecks(hu.gpu_do, op, [X], 0, [0])


if __name__ == "__main__":
    unittest.main()
