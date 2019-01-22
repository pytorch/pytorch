from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect

import numpy as np

from hypothesis import assume, given, settings
import hypothesis.strategies as st

from caffe2.proto import caffe2_pb2
from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial


class TestMatMul(serial.SerializedTestCase):
    @serial.given(
        M=st.integers(min_value=1, max_value=10),
        K=st.integers(min_value=1, max_value=10),
        N=st.integers(min_value=1, max_value=10),
        trans_a=st.booleans(),
        trans_b=st.booleans(),
        **hu.gcs
    )
    def test_matmul(self, M, K, N, trans_a, trans_b, gc, dc):
        X = np.random.rand(M, K).astype(np.float32) - 0.5
        if trans_a:
            X = X.transpose()

        Y = np.random.rand(K, N).astype(np.float32) - 0.5
        if trans_b:
            Y = Y.transpose()

        op = core.CreateOperator(
            'MatMul', ['X', 'Y'], 'out', trans_a=trans_a, trans_b=trans_b
        )

        def matmul_ref(X, Y, trans_a, trans_b):
            XX = X.transpose() if trans_a else X
            YY = Y.transpose() if trans_b else Y
            return (XX.dot(YY), )

        # Check against numpy reference
        self.assertReferenceChecks(gc, op, [X, Y, trans_a, trans_b], matmul_ref)
        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [X, Y], [0])
        # Gradient check wrt X
        self.assertGradientChecks(gc, op, [X, Y], 0, [0])
        # Gradient check wrt Y
        self.assertGradientChecks(gc, op, [X, Y], 1, [0])

    @given(
        M=st.integers(min_value=1, max_value=10),
        K=st.integers(min_value=1, max_value=10),
        N=st.integers(min_value=1, max_value=10),
        axis_a=st.sampled_from([-3, -2, -1, 1, 2, 3]),
        axis_b=st.sampled_from([-3, -2, -1, 1, 2, 3]),
        trans_a=st.booleans(),
        trans_b=st.booleans(),
        **hu.gcs
    )
    def test_matmul_axis(
        self, M, K, N, axis_a, axis_b, trans_a, trans_b, gc, dc
    ):
        X = np.random.rand(M, K).astype(np.float32) - 0.5
        if trans_a:
            X = X.transpose()
        shape_x = [X.shape[0], 1, 1, 1]
        shape_x[axis_a] = X.shape[1]
        X = X.reshape(*shape_x)

        Y = np.random.rand(K, N).astype(np.float32) - 0.5
        if trans_b:
            Y = Y.transpose()
        shape_y = [Y.shape[0], 1, 1, 1]
        shape_y[axis_b] = Y.shape[1]
        Y = Y.reshape(*shape_y)
        op = core.CreateOperator(
            'MatMul', ['X', 'Y'],
            'out',
            axis_a=axis_a,
            axis_b=axis_b,
            trans_a=trans_a,
            trans_b=trans_b
        )

        def size_to_dim(X, axis):
            dim = 1
            for i in range(axis):
                dim *= X.shape[i]
            return dim

        def size_from_dim(X, axis):
            dim = 1
            for i in range(axis, X.ndim):
                dim *= X.shape[i]
            return dim

        def reshape(X, axis):
            dim_0, dim_1 = size_to_dim(X, axis), size_from_dim(X, axis)
            return X.reshape(dim_0, dim_1)

        def canonical_axis(axis, ndim):
            return ndim + axis if axis < 0 else axis

        def matmul_ref(X, Y, axis_a, axis_b, trans_a, trans_b):
            can_axis_a = canonical_axis(axis_a, X.ndim)
            can_axis_b = canonical_axis(axis_b, Y.ndim)
            X, Y = reshape(X, can_axis_a), reshape(Y, can_axis_b)
            XX = X.transpose() if trans_a else X
            YY = Y.transpose() if trans_b else Y
            return (XX.dot(YY), )

        # Check against numpy reference
        self.assertReferenceChecks(
            gc, op, [X, Y, axis_a, axis_b, trans_a, trans_b], matmul_ref
        )
        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [X, Y], [0])
        # Gradient check wrt X
        self.assertGradientChecks(gc, op, [X, Y], 0, [0])
        # Gradient check wrt Y
        self.assertGradientChecks(gc, op, [X, Y], 1, [0])


class TestBatchMatMul(serial.SerializedTestCase):
    @settings(max_examples=30)
    @given(
        C=st.integers(min_value=0, max_value=3),  # number of batch dims
        M=st.integers(min_value=1, max_value=10),
        K=st.integers(min_value=1, max_value=10),
        N=st.integers(min_value=1, max_value=10),
        trans_a=st.booleans(),
        trans_b=st.booleans(),
        dtype=st.sampled_from([np.float32, np.float16]),
        **hu.gcs
    )
    def test_batch_matmul(self, C, M, K, N, trans_a, trans_b, dtype, gc, dc):
        if dtype == np.float16:
            # fp16 is only supported with CUDA/HIP
            assume(core.IsGPUDeviceType(gc.device_type))
            dc = [d for d in dc if core.IsGPUDeviceType(d.device_type)]

        batch_dims = np.random.randint(
            low=1,
            high=3,
            size=C,
            dtype=np.int64).tolist()
        X = np.random.rand(*(batch_dims + [M, K])).astype(dtype) - 0.5
        if trans_a:
            X = X.swapaxes(-1, -2)
        Y = np.random.rand(*(batch_dims + [K, N])).astype(dtype) - 0.5
        if trans_b:
            Y = Y.swapaxes(-1, -2)

        op = core.CreateOperator(
            'BatchMatMul', ['X', 'Y'], 'out', trans_a=trans_a, trans_b=trans_b
        )

        def matmul_ref(X, Y, trans_a, trans_b, dtype):
            XX = (X.swapaxes(-1, -2) if trans_a else X).astype(np.float32)
            YY = (Y.swapaxes(-1, -2) if trans_b else Y).astype(np.float32)
            return (np.matmul(XX, YY).astype(dtype),)

        # relaxing the "threshold" for fp16 to 150x of the default
        def relax_fp16_check(check_func, *args, **kwargs):
            # inspect the default "threshold" value in check_func
            argspec = inspect.getargspec(check_func)
            threshold = argspec.defaults[
                argspec.args.index('threshold') -
                (len(argspec.args) - len(argspec.defaults))]

            if dtype == np.float16:
                threshold = 150 * threshold
            check_func(*args, threshold=threshold, **kwargs)

        # Check against numpy reference
        relax_fp16_check(self.assertReferenceChecks, gc, op, [X, Y, trans_a, trans_b, dtype], matmul_ref)
        # Check over multiple devices
        relax_fp16_check(self.assertDeviceChecks, dc, op, [X, Y], [0])
        # Gradient check wrt X
        relax_fp16_check(self.assertGradientChecks, gc, op, [X, Y], 0, [0])
        # Gradient check wrt Y
        relax_fp16_check(self.assertGradientChecks, gc, op, [X, Y], 1, [0])

    def _test_batch_matmul_with_broadcast_common(
        self,
        X,
        Y,
        dtype,
        gc,
        dc,
        trans_a=None,
        trans_b=None,
    ):
        if trans_a is not None and trans_b is not None:
            op = core.CreateOperator(
                'BatchMatMul', ['X', 'Y'], 'out', trans_a=trans_a, trans_b=trans_b, broadcast=1
            )
        else:
            op = core.CreateOperator(
                'BatchMatMul', ['X', 'Y'], 'out', broadcast=1
            )

        def matmul_ref(X, Y, trans_a, trans_b, dtype):
            XX = (X.swapaxes(-1, -2) if trans_a else X).astype(np.float32)
            YY = (Y.swapaxes(-1, -2) if trans_b else Y).astype(np.float32)
            return (np.matmul(XX, YY).astype(dtype),)

        # Check against numpy reference
        self.assertReferenceChecks(gc, op, [X, Y, trans_a, trans_b, dtype], matmul_ref)
        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [X, Y], [0])

    @serial.given(
        C_1=st.integers(min_value=0, max_value=3),  # number of batch dims
        C_2=st.integers(min_value=0, max_value=3),
        M=st.integers(min_value=1, max_value=10),
        K=st.integers(min_value=1, max_value=10),
        N=st.integers(min_value=1, max_value=10),
        trans_a=st.booleans(),
        trans_b=st.booleans(),
        **hu.gcs
    )
    def test_numpy_batch_matmul(self, C_1, C_2, M, K, N, trans_a, trans_b, gc, dc):
        np.set_printoptions(threshold=np.nan)
        dtype = np.float32
        batch_dims = np.random.randint(
            low=0,
            high=3,
            size=max(C_1, C_2),
            dtype=np.int64).tolist()
        lbd = len(batch_dims)
        X = np.random.rand(*(batch_dims[lbd - C_1:] + [M, K])).astype(dtype) - 0.5
        if trans_a:
            X = X.swapaxes(-1, -2)
        Y = np.random.rand(*(batch_dims[lbd - C_2:] + [K, N])).astype(dtype) - 0.5
        if trans_b:
            Y = Y.swapaxes(-1, -2)

        self._test_batch_matmul_with_broadcast_common(X, Y, dtype, gc, dc, trans_a, trans_b)

    @settings(max_examples=30)
    @given(
        K=st.integers(min_value=1, max_value=10),
        **hu.gcs
    )
    def test_numpy_batch_matmul_1d(self, K, gc, dc):
        np.set_printoptions(threshold=np.nan)
        dtype = np.float32
        X = np.random.rand(K).astype(dtype) - 0.5
        # TODO: test trans_a and trans_b
        Y = np.random.rand(K).astype(dtype) - 0.5

        self._test_batch_matmul_with_broadcast_common(X, Y, dtype, gc, dc)

    @settings(max_examples=30)
    @given(
        K=st.integers(min_value=1, max_value=10),
        N=st.integers(min_value=1, max_value=10),
        **hu.gcs
    )
    def test_numpy_batch_matmul_1d_2d(self, K, N, gc, dc):
        np.set_printoptions(threshold=np.nan)
        dtype = np.float32
        X = np.random.rand(K).astype(dtype) - 0.5
        # TODO: test trans_a and trans_b
        Y = np.random.rand(*[K, N]).astype(dtype) - 0.5

        self._test_batch_matmul_with_broadcast_common(X, Y, dtype, gc, dc)

    @settings(max_examples=30)
    @given(
        M=st.integers(min_value=1, max_value=10),
        K=st.integers(min_value=1, max_value=10),
        **hu.gcs
    )
    def test_numpy_batch_matmul_2d_1d(self, M, K, gc, dc):
        np.set_printoptions(threshold=np.nan)
        dtype = np.float32
        X = np.random.rand(*[M, K]).astype(dtype) - 0.5
        # TODO: test trans_a and trans_b
        Y = np.random.rand(K).astype(dtype) - 0.5

        self._test_batch_matmul_with_broadcast_common(X, Y, dtype, gc, dc)


if __name__ == "__main__":
    import unittest
    unittest.main()
