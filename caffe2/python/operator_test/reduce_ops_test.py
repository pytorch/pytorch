from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace
from hypothesis import given

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
import functools
import itertools as it


class TestReduceOps(hu.HypothesisTestCase):
    @given(
        d0=st.integers(1, 5),
        d1=st.integers(1, 5),
        d2=st.integers(1, 5),
        d3=st.integers(1, 5),
        keepdims=st.integers(0, 1),
        seed=st.integers(0, 2**32 - 1),
        **hu.gcs_cpu_only)
    def test_reduce_sum_mean(self, d0, d1, d2, d3, keepdims, seed, gc, dc):
        def reduce_mean_ref(data, axis, keepdims):
            return [np.mean(data, axis=axis, keepdims=keepdims)]

        def reduce_sum_ref(data, axis, keepdims):
            return [np.sum(data, axis=axis, keepdims=keepdims)]

        def reduce_op_test(op_name, op_ref, data, axes, keepdims, device):
            op = core.CreateOperator(
                op_name,
                ["data"],
                ["Y"],
                axes=axes,
                keepdims=keepdims,
            )

            self.assertReferenceChecks(device, op, [data],
                                       functools.partial(
                                           op_ref,
                                           axis=axes,
                                           keepdims=keepdims))

        np.random.seed(seed)
        for axes in it.combinations(range(4), 2):
            data = np.random.randn(d0, d1, d2, d3).astype(np.float32)

            reduce_op_test("ReduceMean", reduce_mean_ref, data, axes, keepdims,
                           gc)

            reduce_op_test("ReduceSum", reduce_sum_ref, data, axes, keepdims,
                           gc)

        for axes in it.combinations(range(3), 2):
            data = np.random.randn(d0, d1, d2).astype(np.float32)

            reduce_op_test("ReduceMean", reduce_mean_ref, data, axes, keepdims,
                           gc)

            reduce_op_test("ReduceSum", reduce_sum_ref, data, axes, keepdims,
                           gc)

        for axes in it.combinations(range(2), 2):
            data = np.random.randn(d0, d1).astype(np.float32)

            reduce_op_test("ReduceMean", reduce_mean_ref, data, axes, keepdims,
                           gc)

            reduce_op_test("ReduceSum", reduce_sum_ref, data, axes, keepdims,
                           gc)

        for axes in it.combinations(range(1), 1):
            data = np.random.randn(d0).astype(np.float32)

            reduce_op_test("ReduceMean", reduce_mean_ref, data, axes, keepdims,
                           gc)

            reduce_op_test("ReduceSum", reduce_sum_ref, data, axes, keepdims,
                           gc)


class TestReduceFrontReductions(hu.HypothesisTestCase):
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

    def max_op_test(self, op_name, num_reduce_dim, gc, dc, in_data, in_names, ref_max):

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

    @given(num_reduce_dim=st.integers(0, 4), **hu.gcs)
    def test_reduce_front_sum(self, num_reduce_dim, gc, dc):
        X = np.random.rand(7, 4, 3, 5).astype(np.float32)

        def ref_sum(X):
            return [np.sum(X, axis=(tuple(range(num_reduce_dim))))]

        self.reduce_op_test(
            "ReduceFrontSum", ref_sum, [X], ["input"], num_reduce_dim, gc)
        self.grad_variant_input_test(
            "ReduceFrontSumGradient", X, ref_sum, num_reduce_dim)

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

    @given(num_reduce_dim=st.integers(0, 4), **hu.gcs)
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

    @given(num_reduce_dim=st.integers(0, 4), **hu.gcs)
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

    @given(num_reduce_dim=st.integers(0, 4), **hu.gcs)
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

    @given(num_reduce_dim=st.integers(0, 4), **hu.gcs)
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

