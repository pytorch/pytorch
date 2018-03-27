from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import hypothesis.strategies as st
import numpy as np
import random

from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu


class TestTopK(hu.HypothesisTestCase):

    def top_k_ref(self, X, k, flatten_indices, axis=-1):
        in_dims = X.shape
        out_dims = list(in_dims)
        out_dims[axis] = k
        out_dims = tuple(out_dims)
        if axis == -1:
            axis = len(in_dims) - 1
        prev_dims = 1
        next_dims = 1
        for i in range(axis):
            prev_dims *= in_dims[i]
        for i in range(axis + 1, len(in_dims)):
            next_dims *= in_dims[i]
        n = in_dims[axis]
        X_flat = X.reshape((prev_dims, n, next_dims))

        values_ref = np.ndarray(
            shape=(prev_dims, k, next_dims), dtype=np.float32)
        indices_ref = np.ndarray(
            shape=(prev_dims, k, next_dims), dtype=np.int64)
        flatten_indices_ref = np.ndarray(
            shape=(prev_dims, k, next_dims), dtype=np.int64)
        for i in range(prev_dims):
            for j in range(next_dims):
                kv = []
                for x in range(n):
                    val = X_flat[i, x, j]
                    y = x * next_dims + i * in_dims[axis] * next_dims + j
                    kv.append((val, x, y))
                cnt = 0
                for val, x, y in sorted(
                        kv, key=lambda x: (x[0], -x[1]), reverse=True):
                    values_ref[i, cnt, j] = val
                    indices_ref[i, cnt, j] = x
                    flatten_indices_ref[i, cnt, j] = y
                    cnt += 1
                    if cnt >= k:
                        break

        values_ref = values_ref.reshape(out_dims)
        indices_ref = indices_ref.reshape(out_dims)
        flatten_indices_ref = flatten_indices_ref.flatten()

        if flatten_indices:
            return (values_ref, indices_ref, flatten_indices_ref)
        else:
            return (values_ref, indices_ref)

    @given(X=hu.tensor(), flatten_indices=st.booleans(), **hu.gcs)
    def test_top_k(self, X, flatten_indices, gc, dc):
        X = X.astype(dtype=np.float32)
        k = random.randint(1, X.shape[-1])
        output_list = ["Values", "Indices"]
        if flatten_indices:
            output_list.append("FlattenIndices")
        op = core.CreateOperator("TopK", ["X"], output_list,
                                 k=k, device_option=gc)

        def bind_ref(X_loc):
            return self.top_k_ref(X_loc, k, flatten_indices)

        self.assertReferenceChecks(gc, op, [X], bind_ref)
        self.assertDeviceChecks(dc, op, [X], [0])

    @given(bs=st.integers(1, 3), n=st.integers(1, 1), k=st.integers(1, 1),
           flatten_indices=st.booleans(), **hu.gcs)
    def test_top_k_1(self, bs, n, k, flatten_indices, gc, dc):
        X = np.random.rand(bs, n).astype(dtype=np.float32)
        output_list = ["Values", "Indices"]
        if flatten_indices:
            output_list.append("FlattenIndices")
        op = core.CreateOperator("TopK", ["X"], output_list,
                                 k=k, device_option=gc)

        def bind_ref(X_loc):
            return self.top_k_ref(X_loc, k, flatten_indices)

        self.assertReferenceChecks(gc, op, [X], bind_ref)
        self.assertDeviceChecks(dc, op, [X], [0])

    @given(bs=st.integers(1, 3), n=st.integers(1, 10000), k=st.integers(1, 1),
           flatten_indices=st.booleans(), **hu.gcs)
    def test_top_k_2(self, bs, n, k, flatten_indices, gc, dc):
        X = np.random.rand(bs, n).astype(dtype=np.float32)

        output_list = ["Values", "Indices"]
        if flatten_indices:
            output_list.append("FlattenIndices")
        op = core.CreateOperator("TopK", ["X"], output_list,
                                 k=k, device_option=gc)

        def bind_ref(X_loc):
            return self.top_k_ref(X_loc, k, flatten_indices)

        self.assertReferenceChecks(gc, op, [X], bind_ref)
        self.assertDeviceChecks(dc, op, [X], [0])

    @given(bs=st.integers(1, 3), n=st.integers(1, 10000),
           k=st.integers(1, 1024), flatten_indices=st.booleans(), **hu.gcs)
    def test_top_k_3(self, bs, n, k, flatten_indices, gc, dc):
        X = np.random.rand(bs, n).astype(dtype=np.float32)
        k = min(k, n)

        output_list = ["Values", "Indices"]
        if flatten_indices:
            output_list.append("FlattenIndices")
        op = core.CreateOperator("TopK", ["X"], output_list,
                                 k=k, device_option=gc)

        def bind_ref(X_loc):
            return self.top_k_ref(X_loc, k, flatten_indices)

        self.assertReferenceChecks(gc, op, [X], bind_ref)
        self.assertDeviceChecks(dc, op, [X], [0])

    @given(bs=st.integers(1, 3), n=st.integers(100, 10000),
           flatten_indices=st.booleans(), **hu.gcs)
    def test_top_k_4(self, bs, n, flatten_indices, gc, dc):
        k = np.random.randint(n // 3, 3 * n // 4)
        X = np.random.rand(bs, n).astype(dtype=np.float32)

        output_list = ["Values", "Indices"]
        if flatten_indices:
            output_list.append("FlattenIndices")
        op = core.CreateOperator("TopK", ["X"], output_list,
                                 k=k, device_option=gc)

        def bind_ref(X_loc):
            return self.top_k_ref(X_loc, k, flatten_indices)

        self.assertReferenceChecks(gc, op, [X], bind_ref)
        self.assertDeviceChecks(dc, op, [X], [0])

    @given(bs=st.integers(1, 3), n=st.integers(1, 1024),
           flatten_indices=st.booleans(), **hu.gcs)
    def test_top_k_5(self, bs, n, flatten_indices, gc, dc):
        k = n
        X = np.random.rand(bs, n).astype(dtype=np.float32)

        output_list = ["Values", "Indices"]
        if flatten_indices:
            output_list.append("FlattenIndices")
        op = core.CreateOperator("TopK", ["X"], output_list,
                                 k=k, device_option=gc)

        def bind_ref(X_loc):
            return self.top_k_ref(X_loc, k, flatten_indices)

        self.assertReferenceChecks(gc, op, [X], bind_ref)
        self.assertDeviceChecks(dc, op, [X], [0])

    @given(bs=st.integers(1, 3), n=st.integers(1, 5000),
           flatten_indices=st.booleans(), **hu.gcs)
    def test_top_k_6(self, bs, n, flatten_indices, gc, dc):
        k = n
        X = np.random.rand(bs, n).astype(dtype=np.float32)

        output_list = ["Values", "Indices"]
        if flatten_indices:
            output_list.append("FlattenIndices")
        op = core.CreateOperator("TopK", ["X"], output_list,
                                 k=k, device_option=gc)

        def bind_ref(X_loc):
            return self.top_k_ref(X_loc, k, flatten_indices)

        self.assertReferenceChecks(gc, op, [X], bind_ref)
        self.assertDeviceChecks(dc, op, [X], [0])

    @given(X=hu.tensor(dtype=np.float32), k=st.integers(1, 5),
           axis=st.integers(-1, 5), flatten_indices=st.booleans(),
           **hu.gcs)
    def test_top_k_axis(self, X, k, axis, flatten_indices, gc, dc):
        dims = X.shape
        if axis >= len(dims):
            axis %= len(dims)
        if k > dims[axis]:
            k = (k - 1) % dims[axis] + 1
        output_list = ["Values", "Indices"]
        if flatten_indices:
            output_list.append("FlattenIndices")
        op = core.CreateOperator(
            "TopK", ["X"], output_list, k=k, axis=axis, device_option=gc)

        def bind_ref(X_loc):
            return self.top_k_ref(X_loc, k, flatten_indices, axis)

        self.assertReferenceChecks(gc, op, [X], bind_ref)
        self.assertDeviceChecks(dc, op, [X], [0])

    @given(X=hu.tensor(dtype=np.float32), k=st.integers(1, 5),
           axis=st.integers(-1, 5), **hu.gcs)
    def test_top_k_grad(self, X, k, axis, gc, dc):
        dims = X.shape
        if axis >= len(dims):
            axis %= len(dims)
        if k > dims[axis]:
            k = (k - 1) % dims[axis] + 1

        # this try to make sure adding stepsize (0.05)
        # will not change TopK selections at all
        # since dims max_value = 5 as defined in
        # caffe2/caffe2/python/hypothesis_test_util.py
        input_axis = len(dims) - 1 if axis == -1 else axis
        prev_dims = 1
        next_dims = 1
        for i in range(input_axis):
            prev_dims *= dims[i]
        for i in range(input_axis + 1, len(dims)):
            next_dims *= dims[i]

        X_flat = X.reshape((prev_dims, dims[input_axis], next_dims))
        for i in range(prev_dims):
            for j in range(next_dims):
                X_flat[i, :, j] = np.arange(
                    dims[axis], dtype=np.float32) / np.float32(dims[axis])
                np.random.shuffle(X_flat[i, :, j])
        X = X_flat.reshape(dims)

        op = core.CreateOperator(
            "TopK", ["X"], ["Values", "Indices"], k=k, axis=axis,
            device_option=gc)
        self.assertGradientChecks(gc, op, [X], 0, [0])
