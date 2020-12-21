




from caffe2.python import core, workspace
from hypothesis import assume, given, settings
from caffe2.proto import caffe2_pb2
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np
import random
import six


class TestUtilityOps(serial.SerializedTestCase):

    @given(X=hu.tensor(), args=st.booleans(), **hu.gcs)
    @settings(deadline=10000)
    def test_slice(self, X, args, gc, dc):
        X = X.astype(dtype=np.float32)
        dim = random.randint(0, X.ndim - 1)
        slice_start = random.randint(0, X.shape[dim] - 1)
        slice_end = random.randint(slice_start, X.shape[dim] - 1)
        starts = np.array([0] * X.ndim).astype(np.int32)
        ends = np.array([-1] * X.ndim).astype(np.int32)
        starts[dim] = slice_start
        ends[dim] = slice_end

        if args:
            op = core.CreateOperator(
                "Slice", ["X"], ["Y"], starts=starts, ends=ends, device_option=gc
            )

            def slice_ref(X):
                slc = [slice(None)] * X.ndim
                slc[dim] = slice(slice_start, slice_end)
                return [X[slc]]
            inputs = [X]
        else:
            op = core.CreateOperator(
                "Slice", ["X", "starts", "ends"], ["Y"], device_option=gc
            )

            def slice_ref(x, starts, ends):
                slc = [slice(None)] * x.ndim
                slc[dim] = slice(slice_start, slice_end)
                return [x[slc]]
            inputs = [X, starts, ends]

        self.assertReferenceChecks(gc, op, inputs, slice_ref)
        self.assertDeviceChecks(dc, op, inputs, [0])
        self.assertGradientChecks(
            device_option=gc,
            op=op,
            inputs=inputs,
            outputs_to_check=0,
            outputs_with_grads=[0],
        )

    @given(ndims=st.integers(min_value=1, max_value=10), **hu.gcs)
    @settings(deadline=10000)
    def test_resize_like(self, ndims, gc, dc):
        X = np.zeros((ndims * 2, ))
        Y = np.zeros((ndims, 2))

        op = core.CreateOperator(
            "ResizeLike", ["X", "Y"], ["Z"],
        )

        def resize_like(X, Y):
            return [X.reshape(Y.shape)]

        self.assertDeviceChecks(dc, op, [X, Y], [0])
        self.assertReferenceChecks(gc, op, [X, Y], resize_like, ensure_outputs_are_inferred=True)

    @given(dtype=st.sampled_from([np.float32, np.int32]),
           ndims=st.integers(min_value=1, max_value=5),
           seed=st.integers(min_value=0, max_value=65536),
           null_axes=st.booleans(),
           engine=st.sampled_from(['CUDNN', None]),
           **hu.gcs)
    @settings(deadline=10000)
    def test_transpose(self, dtype, ndims, seed, null_axes, engine, gc, dc):
        if (gc.device_type == caffe2_pb2.CUDA and engine == "CUDNN"):
            # cudnn 5.1 does not support int.
            assume(workspace.GetCuDNNVersion() >= 6000 or dtype != np.int32)

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
    @settings(deadline=10000)
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

    @serial.given(n=st.integers(4, 5), m=st.integers(6, 7),
           d=st.integers(2, 3), **hu.gcs)
    def test_elementwise_max(self, n, m, d, gc, dc):
        X = np.random.rand(n, m, d).astype(np.float32)
        Y = np.random.rand(n, m, d).astype(np.float32)
        Z = np.random.rand(n, m, d).astype(np.float32)
        inputs = [X, Y, Z]

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
            inputs=inputs,
            reference=max_op,
        )
        self.assertDeviceChecks(dc, op, inputs, [0])

    @given(n=st.integers(4, 5), m=st.integers(6, 7),
           d=st.integers(2, 3), **hu.gcs)
    @settings(deadline=10000)
    def test_elementwise_max_grad(self, n, m, d, gc, dc):
        go = np.random.rand(n, m, d).astype(np.float32)
        X = np.random.rand(n, m, d).astype(np.float32)
        Y = np.random.rand(n, m, d).astype(np.float32)
        Z = np.random.rand(n, m, d).astype(np.float32)
        mx = np.maximum(np.maximum(X, Y), Z)
        inputs = [mx, go, X, Y, Z]

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
            inputs=inputs,
            reference=max_grad_op,
        )
        self.assertDeviceChecks(dc, op, inputs, [0, 1, 2])

    @serial.given(n=st.integers(4, 5), m=st.integers(6, 7),
           d=st.integers(2, 3), **hu.gcs)
    def test_elementwise_min(self, n, m, d, gc, dc):
        X = np.random.rand(n, m, d).astype(np.float32)
        Y = np.random.rand(n, m, d).astype(np.float32)
        Z = np.random.rand(n, m, d).astype(np.float32)
        inputs = [X, Y, Z]

        def min_op(X, Y, Z):
            return [np.minimum(np.minimum(X, Y), Z)]

        op = core.CreateOperator(
            "Min",
            ["X", "Y", "Z"],
            ["mx"]
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=inputs,
            reference=min_op,
        )
        self.assertDeviceChecks(dc, op, inputs, [0])

    @given(n=st.integers(4, 5), m=st.integers(6, 7),
           d=st.integers(2, 3), **hu.gcs)
    @settings(deadline=10000)
    def test_elementwise_min_grad(self, n, m, d, gc, dc):
        go = np.random.rand(n, m, d).astype(np.float32)
        X = np.random.rand(n, m, d).astype(np.float32)
        Y = np.random.rand(n, m, d).astype(np.float32)
        Z = np.random.rand(n, m, d).astype(np.float32)
        mx = np.minimum(np.minimum(X, Y), Z)
        inputs = [mx, go, X, Y, Z]

        def min_grad_op(mx, go, X, Y, Z):
            def mx_grad(a):
                return go * (mx == a)

            return [mx_grad(a) for a in [X, Y, Z]]

        op = core.CreateOperator(
            "MinGradient",
            ["mx", "go", "X", "Y", "Z"],
            ["gX", "gY", "gZ"]
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=inputs,
            reference=min_grad_op,
        )
        self.assertDeviceChecks(dc, op, inputs, [0, 1, 2])

    @given(
        n=st.integers(1, 8), m=st.integers(1, 10), d=st.integers(1, 4),
        in_place=st.booleans(), engine=st.sampled_from(["", "CUDNN"]),
        seed=st.integers(min_value=0, max_value=65535),
        dtype=st.sampled_from([np.int32, np.int64, np.float32]),
        **hu.gcs)
    @settings(deadline=10000)
    def test_sum(
            self, n, m, d, in_place, engine, seed, dtype, gc, dc):
        input_names = []
        input_vars = []
        np.random.seed(seed)
        for i in range(m):
            X_name = 'X' + str(i)
            input_names.extend([X_name])
            var = np.random.rand(n, d).astype(dtype)
            vars()[X_name] = var
            input_vars.append(var)

        def sum_op_ref(*args):
            res = np.zeros((n, d))
            for i in range(m):
                res = res + args[i]
            return (res, )

        op = core.CreateOperator(
            "Sum",
            input_names,
            [input_names[0]] if in_place else ['Y'],
            engine=engine,
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=input_vars,
            reference=sum_op_ref,
        )
        self.assertDeviceChecks(dc, op, input_vars, [0])

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
    @settings(deadline=1000)
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

    @given(
        inputs=hu.lengths_tensor(),
        **hu.gcs_cpu_only)
    @settings(deadline=1000)
    def test_lengths_to_ranges(self, inputs, gc, dc):
        _, lengths = inputs

        def lengths_to_ranges_op(lengths):
            return [
                [[x, y] for x, y in zip(np.cumsum(np.append([0], lengths)),
                                        lengths)]
            ]

        op = core.CreateOperator(
            "LengthsToRanges",
            ["lengths"],
            ["output"]
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[lengths],
            reference=lengths_to_ranges_op,
        )

        # Test shape inference logic
        net = core.Net("test_shape_inference")

        workspace.FeedBlob("lengths", lengths)
        output = net.LengthsToRanges(
            ["lengths"],
            ["output"]
        )
        (shapes, types) = workspace.InferShapesAndTypes([net])
        workspace.RunNetOnce(net)
        self.assertEqual(shapes[output], list(workspace.blobs[output].shape))
        self.assertEqual(shapes[output], list(lengths.shape) + [2])
        self.assertEqual(types[output], core.DataType.INT32)

    @given(**hu.gcs)
    @settings(deadline=None, max_examples=50)
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

    @given(**hu.gcs)
    @settings(deadline=10000)
    def test_range(self, gc, dc):
        names = [
            ('stop_',),
            ('start_', 'stop_'),
            ('start_', 'stop_', 'step_'),
        ]
        # Most random values aren't great here, so use a fixed set instead of
        # hypothesis.
        for inputs in (
            (10,),
            (np.float32(10.0),),
            (0,),
            (0, 0),
            (10., 5.0, -1.),
            (2, 10000),
            (2, 10000, 20000),
            (2, 10000, -1),
        ):
            inputs = [np.array(v) for v in inputs]
            op = core.CreateOperator(
                "Range",
                names[len(inputs) - 1],
                ["Y"]
            )

            self.assertReferenceChecks(
                device_option=gc,
                op=op,
                inputs=inputs,
                reference=lambda *x: [np.arange(*x)],
            )
            self.assertDeviceChecks(dc, op, inputs, [0])

        inputs = (np.array(0), np.array(10), np.array(0))
        op = core.CreateOperator(
            "Range",
            names[len(inputs) - 1],
            ["Y"]
        )
        with six.assertRaisesRegex(self, RuntimeError, 'Step size cannot be 0'):
            self.assertReferenceChecks(
                device_option=gc,
                op=op,
                inputs=inputs,
                reference=lambda *x: [np.arange(*x)],
            )
