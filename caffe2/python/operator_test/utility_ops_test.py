from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
import random


class TestUtilityOps(hu.HypothesisTestCase):

    @given(X=hu.tensor(), neg=st.booleans(), **hu.gcs)
    def test_slice(self, X, neg, gc, dc):
        X = X.astype(dtype=np.float32)
        dim = random.randint(0, X.ndim - 1)
        slice_start = random.randint(0, X.shape[dim] - 1)
        slice_end = random.randint(slice_start, X.shape[dim] - 1)
        starts = np.array([0] * X.ndim).astype(np.int32)
        ends = np.array([-1] * X.ndim).astype(np.int32)
        starts[dim] = slice_start
        ends[dim] = slice_end

        op = core.CreateOperator(
            "Slice", ["X", "starts", "ends"], ["Y"], device_option=gc
        )

        def slice_ref(X, starts, ends):
            slc = [slice(None)] * X.ndim
            slc[dim] = slice(slice_start, slice_end)
            return [X[slc]]

        self.assertReferenceChecks(gc, op, [X, starts, ends], slice_ref)

        self.assertDeviceChecks(dc, op, [X, starts, ends], [0])

    @given(dtype=st.sampled_from([np.float32, np.int32, np.int64]),
           ndims=st.integers(min_value=1, max_value=5),
           seed=st.integers(min_value=0, max_value=65536),
           null_axes=st.booleans(),
           engine=st.sampled_from(['CUDNN', None]),
           **hu.gcs)
    def test_transpose(self, dtype, ndims, seed, null_axes, engine, gc, dc):
        dims = (np.random.rand(ndims) * 16 + 1).astype(np.int32)
        X = (np.random.rand(*dims) * 16).astype(dtype)

        if null_axes:
            axes = None
            op = core.CreateOperator(
                "Transpose",
                ["input"], ["output"],
                engine=engine)
        else:
            np.random.seed(int(seed))
            axes = [int(v) for v in list(np.random.permutation(X.ndim))]
            op = core.CreateOperator(
                "Transpose",
                ["input"], ["output"],
                axes=axes,
                engine=engine)

        def transpose_ref(x, axes):
            return (np.transpose(x, axes),)

        self.assertReferenceChecks(gc, op, [X, axes],
                                   transpose_ref)

    @given(m=st.integers(5, 10), n=st.integers(5, 10),
           o=st.integers(5, 10), nans=st.booleans(), **hu.gcs)
    def test_nan_check(self, m, n, o, nans, gc, dc):
        other = np.array([1, 2, 3]).astype(np.float32)
        X = np.random.rand(m, n, o).astype(np.float32)
        if nans:
            x_nan = np.random.randint(0, m)
            y_nan = np.random.randint(0, n)
            z_nan = np.random.randint(0, o)
            X[x_nan, y_nan, z_nan] = float('NaN')

        # print('nans: {}'.format(nans))
        # print(X)

        def nan_reference(X, Y):
            if not np.isnan(X).any():
                return [X]
            else:
                return [np.array([])]

        op = core.CreateOperator(
            "NanCheck",
            ["X", "other"],
            ["Y"]
        )

        try:
            self.assertReferenceChecks(
                device_option=gc,
                op=op,
                inputs=[X, other],
                reference=nan_reference,
            )
            if nans:
                self.assertTrue(False, "Did not fail when presented with NaN!")
        except RuntimeError:
            self.assertTrue(nans, "No NaNs but failed")

        try:
            self.assertGradientChecks(
                device_option=gc,
                op=op,
                inputs=[X],
                outputs_to_check=0,
                outputs_with_grads=[0],
            )
            if nans:
                self.assertTrue(False, "Did not fail when gradient had NaN!")
        except RuntimeError:
            pass

    @given(n=st.integers(4, 5), m=st.integers(6, 7),
           d=st.integers(2, 3), **hu.gcs)
    def test_elementwise_max(self, n, m, d, gc, dc):
        X = np.random.rand(n, m, d).astype(np.float32)
        Y = np.random.rand(n, m, d).astype(np.float32)
        Z = np.random.rand(n, m, d).astype(np.float32)

        def max_op(X, Y, Z):
            return [np.maximum(np.maximum(X, Y), Z)]

        op = core.CreateOperator(
            "Max",
            ["X", "Y", "Z"],
            ["mx"]
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X, Y, Z],
            reference=max_op,
        )

    @given(n=st.integers(4, 5), m=st.integers(6, 7),
           d=st.integers(2, 3), **hu.gcs)
    def test_elementwise_max_grad(self, n, m, d, gc, dc):
        go = np.random.rand(n, m, d).astype(np.float32)
        X = np.random.rand(n, m, d).astype(np.float32)
        Y = np.random.rand(n, m, d).astype(np.float32)
        Z = np.random.rand(n, m, d).astype(np.float32)
        mx = np.maximum(np.maximum(X, Y), Z)

        def max_grad_op(mx, go, X, Y, Z):
            def mx_grad(a):
                return go * (mx == a)

            return [mx_grad(a) for a in [X, Y, Z]]

        op = core.CreateOperator(
            "MaxGradient",
            ["mx", "go", "X", "Y", "Z"],
            ["gX", "gY", "gZ"]
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[mx, go, X, Y, Z],
            reference=max_grad_op,
        )

    @given(
        inputs=hu.lengths_tensor().flatmap(
            lambda pair: st.tuples(
                st.just(pair[0]),
                st.just(pair[1]),
                hu.dims(max_value=len(pair[1])),
            )
        ).flatmap(
            lambda tup: st.tuples(
                st.just(tup[0]),
                st.just(tup[1]),
                hu.arrays(
                    tup[2], dtype=np.int32,
                    elements=st.integers(
                        min_value=0, max_value=len(tup[1]) - 1)),
            )
        ),
        **hu.gcs_cpu_only)
    def test_lengths_gather(self, inputs, gc, dc):
        items = inputs[0]
        lengths = inputs[1]
        indices = inputs[2]

        def lengths_gather_op(items, lengths, indices):
            ends = np.cumsum(lengths)
            return [np.concatenate(
                list(items[ends[i] - lengths[i]:ends[i]] for i in indices))]

        op = core.CreateOperator(
            "LengthsGather",
            ["items", "lengths", "indices"],
            ["output"]
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[items, lengths, indices],
            reference=lengths_gather_op,
        )

    @given(**hu.gcs)
    def test_size_op(self, gc, dc):
        X = np.array([[1, 2], [3, 4]]).astype(np.float32)

        def size_op(tensor):
            return [np.prod(tensor.shape)]

        op = core.CreateOperator(
            "Size",
            ["X"],
            ["output"]
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=size_op,
        )

    def test_alias_op(self):
        """ Don't use hypothesis because there are only 2 cases to check"""
        for size in [0, 5]:
            X = np.arange(size).astype(np.float32)
            workspace.FeedBlob('X', X)

            op = core.CreateOperator(
                "Alias",
                ["X"],
                ["Y"]
            )
            workspace.RunOperatorOnce(op)
            Y = workspace.FetchBlob('Y')
            np.testing.assert_array_equal(X, Y)
