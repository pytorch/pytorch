from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace
from hypothesis import given

import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np
import itertools as it
import unittest


class TestReduceOps(serial.SerializedTestCase):
    def run_reduce_op_test_impl(
            self, op_name, X, axes, keepdims, ref_func, gc, dc):
        if axes is None:
            op = core.CreateOperator(
                op_name,
                ["X"],
                ["Y"],
                keepdims=keepdims,
            )
        else:
            op = core.CreateOperator(
                op_name,
                ["X"],
                ["Y"],
                axes=axes,
                keepdims=keepdims,
            )

        def ref(X):
            return [ref_func(
                X, axis=None if axes is None else tuple(axes),
                keepdims=keepdims)]

        self.assertReferenceChecks(gc, op, [X], ref)
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0])

    def run_reduce_op_test(
            self, op_name, X, keepdims, num_axes, ref_func, gc, dc):
        self.run_reduce_op_test_impl(
            op_name, X, None, keepdims, ref_func, gc, dc)

        num_dims = len(X.shape)
        if num_dims < num_axes:
            self.run_reduce_op_test_impl(
                op_name, X, range(num_dims), keepdims, ref_func, gc, dc)
        else:
            for axes in it.combinations(range(num_dims), num_axes):
                self.run_reduce_op_test_impl(
                    op_name, X, axes, keepdims, ref_func, gc, dc)

    @serial.given(
        X=hu.tensor(max_dim=3, dtype=np.float32), keepdims=st.booleans(),
        num_axes=st.integers(1, 3), **hu.gcs)
    def test_reduce_min(self, X, keepdims, num_axes, gc, dc):
        X_dims = X.shape
        X_size = X.size
        X = np.arange(X_size, dtype=np.float32)
        np.random.shuffle(X)
        X = X.reshape(X_dims)
        self.run_reduce_op_test(
            "ReduceMin", X, keepdims, num_axes, np.min, gc, dc)

    @serial.given(
        X=hu.tensor(max_dim=3, dtype=np.float32), keepdims=st.booleans(),
        num_axes=st.integers(1, 3), **hu.gcs)
    def test_reduce_max(self, X, keepdims, num_axes, gc, dc):
        X_dims = X.shape
        X_size = X.size
        X = np.arange(X_size, dtype=np.float32)
        np.random.shuffle(X)
        X = X.reshape(X_dims)
        self.run_reduce_op_test(
            "ReduceMax", X, keepdims, num_axes, np.max, gc, dc)

    @given(n=st.integers(0, 5), m=st.integers(0, 5), k=st.integers(0, 5),
           t=st.integers(0, 5), keepdims=st.booleans(),
           num_axes=st.integers(1, 3), **hu.gcs)
    def test_reduce_sum(self, n, m, k, t, keepdims, num_axes, gc, dc):
        X = np.random.randn(n, m, k, t).astype(np.float32)
        self.run_reduce_op_test(
            "ReduceSum", X, keepdims, num_axes, np.sum, gc, dc)

    @serial.given(X=hu.tensor(dtype=np.float32), keepdims=st.booleans(),
           num_axes=st.integers(1, 4), **hu.gcs)
    def test_reduce_mean(self, X, keepdims, num_axes, gc, dc):
        self.run_reduce_op_test(
            "ReduceMean", X, keepdims, num_axes, np.mean, gc, dc)

    @given(n=st.integers(1, 3), m=st.integers(1, 3), k=st.integers(1, 3),
           keepdims=st.booleans(), num_axes=st.integers(1, 3), **hu.gcs_cpu_only)
    def test_reduce_l1(self, n, m, k, keepdims, num_axes, gc, dc):
        X = np.arange(n * m * k, dtype=np.float32) - 0.5
        np.random.shuffle(X)
        X = X.reshape((m, n, k))
        self.run_reduce_op_test(
            "ReduceL1", X, keepdims, num_axes, getNorm(1), gc, dc)

    @serial.given(n=st.integers(1, 5), m=st.integers(1, 5), k=st.integers(1, 5),
           keepdims=st.booleans(), num_axes=st.integers(1, 3), **hu.gcs_cpu_only)
    def test_reduce_l2(self, n, m, k, keepdims, num_axes, gc, dc):
        X = np.random.randn(n, m, k).astype(np.float32)
        self.run_reduce_op_test(
            "ReduceL2", X, keepdims, num_axes, getNorm(2), gc, dc)


def getNorm(p):
    if p == 1:
        def norm(X, axis, keepdims):
            return np.sum(np.abs(X), axis=axis, keepdims=keepdims)
    elif p == 2:
        def norm(X, axis, keepdims):
            return np.sqrt(np.sum(np.power(X, 2), axis=axis, keepdims=keepdims))
    else:
        raise RuntimeError("Only L1 and L2 norms supported")
    return norm


class TestReduceFrontReductions(serial.SerializedTestCase):
    def grad_variant_input_test(self, grad_op_name, X, ref, num_reduce_dim):
        workspace.ResetWorkspace()

        Y = np.array(ref(X)[0]).astype(np.float32)
        dY = np.array(np.random.rand(*Y.shape)).astype(np.float32)
        shape = np.array(X.shape).astype(np.int64)

        workspace.FeedBlob("X", X)
        workspace.FeedBlob("dY", dY)
        workspace.FeedBlob("shape", shape)

        grad_op = core.CreateOperator(
            grad_op_name, ["dY", "X"], ["dX"], num_reduce_dim=num_reduce_dim)

        grad_op1 = core.CreateOperator(
            grad_op_name, ["dY", "shape"], ["dX1"],
            num_reduce_dim=num_reduce_dim)

        workspace.RunOperatorOnce(grad_op)
        workspace.RunOperatorOnce(grad_op1)

        dX = workspace.FetchBlob("dX")
        dX1 = workspace.FetchBlob("dX1")
        np.testing.assert_array_equal(dX, dX1)

    def max_op_test(
            self, op_name, num_reduce_dim, gc, dc, in_data, in_names, ref_max):

        op = core.CreateOperator(
            op_name,
            in_names,
            ["outputs"],
            num_reduce_dim=num_reduce_dim
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=in_data,
            reference=ref_max,
        )

        # Skip gradient check because it is too unreliable with max.
        # Just check CPU and CUDA have same results
        Y = np.array(ref_max(*in_data)[0]).astype(np.float32)
        dY = np.array(np.random.rand(*Y.shape)).astype(np.float32)
        if len(in_data) == 2:
            grad_in_names = ["dY", in_names[0], "Y", in_names[1]]
            grad_in_data = [dY, in_data[0], Y, in_data[1]]
        else:
            grad_in_names = ["dY", in_names[0], "Y"]
            grad_in_data = [dY, in_data[0], Y]

        grad_op = core.CreateOperator(
            op_name + "Gradient",
            grad_in_names,
            ["dX"],
            num_reduce_dim=num_reduce_dim
        )
        self.assertDeviceChecks(dc, grad_op, grad_in_data, [0])

    def reduce_op_test(self, op_name, op_ref, in_data, in_names,
                       num_reduce_dims, device):
        op = core.CreateOperator(
            op_name,
            in_names,
            ["outputs"],
            num_reduce_dim=num_reduce_dims
        )

        self.assertReferenceChecks(
            device_option=device,
            op=op,
            inputs=in_data,
            reference=op_ref
        )

        self.assertGradientChecks(
            device, op, in_data, 0, [0], stepsize=1e-2, threshold=1e-2)

    @serial.given(num_reduce_dim=st.integers(0, 4), **hu.gcs)
    def test_reduce_front_sum(self, num_reduce_dim, gc, dc):
        X = np.random.rand(7, 4, 3, 5).astype(np.float32)

        def ref_sum(X):
            return [np.sum(X, axis=(tuple(range(num_reduce_dim))))]

        self.reduce_op_test(
            "ReduceFrontSum", ref_sum, [X], ["input"], num_reduce_dim, gc)
        self.grad_variant_input_test(
            "ReduceFrontSumGradient", X, ref_sum, num_reduce_dim)

    @given(num_reduce_dim=st.integers(0, 4), seed=st.integers(0, 4), **hu.gcs)
    def test_reduce_front_sum_empty_batch(self, num_reduce_dim, seed, gc, dc):
        np.random.seed(seed)
        X = np.random.rand(0, 4, 3, 5).astype(np.float32)

        def ref_sum(X):
            return [np.sum(X, axis=(tuple(range(num_reduce_dim))))]

        self.reduce_op_test(
            "ReduceFrontSum", ref_sum, [X], ["input"], num_reduce_dim, gc)
        self.grad_variant_input_test(
            "ReduceFrontSumGradient", X, ref_sum, num_reduce_dim)

        # test the second iteration
        not_empty_X = np.random.rand(2, 4, 3, 5).astype(np.float32)
        net = core.Net('test')
        with core.DeviceScope(gc):
            net.ReduceFrontSum(
                ['X'], ['output'],
                num_reduce_dim=num_reduce_dim
            )
            workspace.CreateNet(net)

            workspace.FeedBlob('X', not_empty_X)
            workspace.RunNet(workspace.GetNetName(net))
            output = workspace.FetchBlob('output')
            np.testing.assert_allclose(
                output, ref_sum(not_empty_X)[0], atol=1e-3)

            workspace.FeedBlob('X', X)
            workspace.RunNet(workspace.GetNetName(net))
            output = workspace.FetchBlob('output')
            np.testing.assert_allclose(output, ref_sum(X)[0], atol=1e-3)

    @given(**hu.gcs)
    def test_reduce_front_sum_with_length(self, dc, gc):
        num_reduce_dim = 1
        X = np.random.rand(2, 3, 4, 5).astype(np.float32)
        batch_size = int(np.prod([2, 3, 4, 5][num_reduce_dim:]))
        d = 120 // batch_size
        lengths = np.random.randint(1, d, size=batch_size).astype(np.int32)

        def ref_sum(X, lengths):
            Y = X.reshape(d, lengths.size)
            rv = np.zeros((lengths.size, 1)).astype(np.float32)
            for ii in range(lengths.size):
                rv[ii] = np.sum(Y[:lengths[ii], ii])
            return [rv.reshape((2, 3, 4, 5)[num_reduce_dim:])]

        self.reduce_op_test(
            "ReduceFrontSum", ref_sum, [X, lengths], ["input", "lengths"],
            num_reduce_dim, gc)

    @serial.given(num_reduce_dim=st.integers(0, 4), **hu.gcs)
    def test_reduce_front_mean(self, num_reduce_dim, gc, dc):
        X = np.random.rand(6, 7, 8, 2).astype(np.float32)

        def ref_mean(X):
            return [np.mean(X, axis=(tuple(range(num_reduce_dim))))]

        self.reduce_op_test(
            "ReduceFrontMean", ref_mean, [X], ["input"], num_reduce_dim, gc)
        self.grad_variant_input_test(
            "ReduceFrontMeanGradient", X, ref_mean, num_reduce_dim)

    @given(**hu.gcs)
    def test_reduce_front_mean_with_length(self, dc, gc):
        num_reduce_dim = 1
        X = np.random.rand(2, 3, 4, 5).astype(np.float32)
        batch_size = int(np.prod([2, 3, 4, 5][num_reduce_dim:]))
        d = 120 // batch_size
        lengths = np.random.randint(1, d, size=batch_size).astype(np.int32)

        def ref_mean(X, lengths):
            Y = X.reshape(d, lengths.size)
            rv = np.zeros((lengths.size, 1)).astype(np.float32)
            for ii in range(lengths.size):
                rv[ii] = np.mean(Y[:lengths[ii], ii])
            return [rv.reshape((2, 3, 4, 5)[num_reduce_dim:])]

        self.reduce_op_test(
            "ReduceFrontMean", ref_mean, [X, lengths], ["input", "lengths"],
            num_reduce_dim, gc)

    @serial.given(num_reduce_dim=st.integers(0, 4), **hu.gcs)
    def test_reduce_front_max(self, num_reduce_dim, gc, dc):
        X = np.random.rand(6, 7, 8, 2).astype(np.float32)

        def ref_frontmax(X):
            return [np.max(X, axis=(tuple(range(num_reduce_dim))))]

        self.max_op_test(
            "ReduceFrontMax", num_reduce_dim, gc, dc, [X], ["X"], ref_frontmax)

    @given(**hu.gcs)
    def test_reduce_front_max_with_length(self, dc, gc):
        num_reduce_dim = 1
        X = np.random.rand(2, 3, 4, 5).astype(np.float32)
        batch_size = int(np.prod([2, 3, 4, 5][num_reduce_dim:]))
        d = 120 // batch_size
        lengths = np.random.randint(1, d, size=batch_size).astype(np.int32)

        def ref_max(X, lengths):
            Y = X.reshape(d, lengths.size)
            rv = np.zeros((lengths.size, 1)).astype(np.float32)
            for ii in range(lengths.size):
                rv[ii] = np.max(Y[:lengths[ii], ii])
            return [rv.reshape((2, 3, 4, 5)[num_reduce_dim:])]

        self.max_op_test(
            "ReduceFrontMax", num_reduce_dim, gc, dc, [X, lengths],
            ["X", "lengths"], ref_max)

    @serial.given(num_reduce_dim=st.integers(0, 4), **hu.gcs)
    def test_reduce_back_max(self, num_reduce_dim, gc, dc):
        X = np.random.rand(6, 7, 8, 2).astype(np.float32)

        def ref_backmax(X):
            return [np.max(X, axis=(0, 1, 2, 3)[4 - num_reduce_dim:])]

        self.max_op_test(
            "ReduceBackMax", num_reduce_dim, gc, dc, [X], ["X"], ref_backmax)

    @given(**hu.gcs)
    def test_reduce_back_max_with_length(self, gc, dc):
        num_reduce_dim = 1
        X = np.random.rand(2, 3, 4, 5).astype(np.float32)
        batch_size = int(np.prod([2, 3, 4, 5][:4 - num_reduce_dim]))
        d = 120 // batch_size
        lengths = np.random.randint(1, d, size=batch_size).astype(np.int32)

        def ref_max(X, lengths):
            Y = X.reshape(lengths.size, d)
            rv = np.zeros((lengths.size, 1)).astype(np.float32)
            for ii in range(lengths.size):
                rv[ii] = np.max(Y[ii, :lengths[ii]])
            return [rv.reshape((2, 3, 4, 5)[:4 - num_reduce_dim])]

        self.max_op_test(
            "ReduceBackMax", num_reduce_dim, gc, dc, [X, lengths],
            ["X", "lengths"], ref_max)

    @given(**hu.gcs)
    def test_reduce_back_sum(self, dc, gc):
        num_reduce_dim = 1
        X = np.random.rand(6, 7, 8, 2).astype(np.float32)

        def ref_sum(X):
            return [np.sum(X, axis=(0, 1, 2, 3)[4 - num_reduce_dim:])]

        self.reduce_op_test(
            "ReduceBackSum", ref_sum, [X], ["input"], num_reduce_dim, gc)
        self.grad_variant_input_test(
            "ReduceBackSumGradient", X, ref_sum, num_reduce_dim)

    @given(**hu.gcs)
    def test_reduce_back_sum_with_length(self, dc, gc):
        num_reduce_dim = 1
        X = np.random.rand(2, 3, 4, 5).astype(np.float32)
        batch_size = int(np.prod([2, 3, 4, 5][:4 - num_reduce_dim]))
        d = 120 // batch_size
        lengths = np.random.randint(1, d, size=batch_size).astype(np.int32)

        def ref_sum(X, lengths):
            Y = X.reshape(lengths.size, d)
            rv = np.zeros((lengths.size, 1)).astype(np.float32)
            for ii in range(lengths.size):
                rv[ii] = np.sum(Y[ii, :lengths[ii]])
            return [rv.reshape((2, 3, 4, 5)[:4 - num_reduce_dim])]

        self.reduce_op_test(
            "ReduceBackSum", ref_sum, [X, lengths], ["input", "lengths"],
            num_reduce_dim, gc)

    @serial.given(num_reduce_dim=st.integers(0, 4), **hu.gcs)
    def test_reduce_back_mean(self, num_reduce_dim, dc, gc):
        X = np.random.rand(6, 7, 8, 2).astype(np.float32)

        def ref_mean(X):
            return [np.mean(X, axis=(0, 1, 2, 3)[4 - num_reduce_dim:])]

        self.reduce_op_test(
            "ReduceBackMean", ref_mean, [X], ["input"], num_reduce_dim, gc)
        self.grad_variant_input_test(
            "ReduceBackMeanGradient", X, ref_mean, num_reduce_dim)

    @given(**hu.gcs)
    def test_reduce_back_mean_with_length(self, dc, gc):
        num_reduce_dim = 1
        X = np.random.rand(2, 3, 4, 5).astype(np.float32)
        batch_size = int(np.prod([2, 3, 4, 5][:4 - num_reduce_dim]))
        d = 120 // batch_size
        lengths = np.random.randint(1, d, size=batch_size).astype(np.int32)

        def ref_mean(X, lengths):
            Y = X.reshape(lengths.size, d)
            rv = np.zeros((lengths.size, 1)).astype(np.float32)
            for ii in range(lengths.size):
                rv[ii] = np.mean(Y[ii, :lengths[ii]])
            return [rv.reshape((2, 3, 4, 5)[:4 - num_reduce_dim])]

        self.reduce_op_test(
            "ReduceBackMean", ref_mean, [X, lengths], ["input", "lengths"],
            num_reduce_dim, gc)
