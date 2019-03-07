from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial

from hypothesis import given
import hypothesis.strategies as st
import numpy as np


def _fill_diagonal(shape, value):
    result = np.zeros(shape)
    np.fill_diagonal(result, value)
    return (result,)


class TestFillerOperator(serial.SerializedTestCase):

    @given(**hu.gcs)
    def test_shape_error(self, gc, dc):
        op = core.CreateOperator(
            'GaussianFill',
            [],
            'out',
            shape=32,  # illegal parameter
            mean=0.0,
            std=1.0,
        )
        exception = False
        try:
            workspace.RunOperatorOnce(op)
        except Exception:
            exception = True
        self.assertTrue(exception, "Did not throw exception on illegal shape")

        op = core.CreateOperator(
            'ConstantFill',
            [],
            'out',
            shape=[],  # scalar
            value=2.0,
        )
        exception = False
        self.assertTrue(workspace.RunOperatorOnce(op))
        self.assertEqual(workspace.FetchBlob('out'), [2.0])

    @given(**hu.gcs)
    def test_int64_shape(self, gc, dc):
        large_dim = 2 ** 31 + 1
        net = core.Net("test_shape_net")
        net.UniformFill(
            [],
            'out',
            shape=[0, large_dim],
            min=0.0,
            max=1.0,
        )
        self.assertTrue(workspace.CreateNet(net))
        self.assertTrue(workspace.RunNet(net.Name()))
        self.assertEqual(workspace.blobs['out'].shape, (0, large_dim))

    @given(
        shape=hu.dims().flatmap(
            lambda dims: hu.arrays(
                [dims], dtype=np.int64,
                elements=st.integers(min_value=0, max_value=20)
            )
        ),
        a=st.integers(min_value=0, max_value=100),
        b=st.integers(min_value=0, max_value=100),
        **hu.gcs
    )
    def test_uniform_int_fill_op_blob_input(self, shape, a, b, gc, dc):
        net = core.Net('test_net')

        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
            shape_blob = net.Const(shape, dtype=np.int64)
        a_blob = net.Const(a, dtype=np.int32)
        b_blob = net.Const(b, dtype=np.int32)
        uniform_fill = net.UniformIntFill([shape_blob, a_blob, b_blob],
                                          1, input_as_shape=1)

        workspace.RunNetOnce(net)

        blob_out = workspace.FetchBlob(uniform_fill)
        if b < a:
            new_shape = shape[:]
            new_shape[0] = 0
            np.testing.assert_array_equal(new_shape, blob_out.shape)
        else:
            np.testing.assert_array_equal(shape, blob_out.shape)
            self.assertTrue((blob_out >= a).all())
            self.assertTrue((blob_out <= b).all())

    @given(
        **hu.gcs
    )
    def test_uniform_fill_using_arg(self, gc, dc):
        net = core.Net('test_net')
        shape = [2**3, 5]
        # uncomment this to test filling large blob
        # shape = [2**30, 5]
        min_v = -100
        max_v = 100
        output_blob = net.UniformIntFill(
            [],
            ['output_blob'],
            shape=shape,
            min=min_v,
            max=max_v,
        )

        workspace.RunNetOnce(net)
        output_data = workspace.FetchBlob(output_blob)

        np.testing.assert_array_equal(shape, output_data.shape)
        min_data = np.min(output_data)
        max_data = np.max(output_data)

        self.assertGreaterEqual(min_data, min_v)
        self.assertLessEqual(max_data, max_v)

        self.assertNotEqual(min_data, max_data)

    @serial.given(
        shape=st.sampled_from(
            [
                [3, 3],
                [5, 5, 5],
                [7, 7, 7, 7],
            ]
        ),
        **hu.gcs
    )
    def test_diagonal_fill_op_float(self, shape, gc, dc):
        value = 2.5
        op = core.CreateOperator(
            'DiagonalFill',
            [],
            'out',
            shape=shape,  # scalar
            value=value,
        )

        for device_option in dc:
            op.device_option.CopyFrom(device_option)
            # Check against numpy reference
            self.assertReferenceChecks(gc, op, [shape, value], _fill_diagonal)

    @given(**hu.gcs)
    def test_diagonal_fill_op_int(self, gc, dc):
        value = 2
        shape = [3, 3]
        op = core.CreateOperator(
            'DiagonalFill',
            [],
            'out',
            shape=shape,
            dtype=core.DataType.INT32,
            value=value,
        )

        # Check against numpy reference
        self.assertReferenceChecks(gc, op, [shape, value], _fill_diagonal)

    @serial.given(lengths=st.lists(st.integers(min_value=0, max_value=10),
                                   min_size=0,
                                   max_size=10),
           **hu.gcs)
    def test_lengths_range_fill(self, lengths, gc, dc):
        op = core.CreateOperator(
            "LengthsRangeFill",
            ["lengths"],
            ["increasing_seq"])

        def _len_range_fill(lengths):
            sids = []
            for _, l in enumerate(lengths):
                sids.extend(list(range(l)))
            return (np.array(sids, dtype=np.int32), )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[np.array(lengths, dtype=np.int32)],
            reference=_len_range_fill)

    @given(**hu.gcs)
    def test_gaussian_fill_op(self, gc, dc):
        op = core.CreateOperator(
            'GaussianFill',
            [],
            'out',
            shape=[17, 3, 3],  # sample odd dimensions
            mean=0.0,
            std=1.0,
        )

        for device_option in dc:
            op.device_option.CopyFrom(device_option)
            assert workspace.RunOperatorOnce(op), "GaussianFill op did not run "
            "successfully"

            blob_out = workspace.FetchBlob('out')
            assert np.count_nonzero(blob_out) > 0, "All generated elements are "
            "zeros. Is the random generator functioning correctly?"

    @given(**hu.gcs)
    def test_msra_fill_op(self, gc, dc):
        op = core.CreateOperator(
            'MSRAFill',
            [],
            'out',
            shape=[15, 5, 3],  # sample odd dimensions
        )
        for device_option in dc:
            op.device_option.CopyFrom(device_option)
            assert workspace.RunOperatorOnce(op), "MSRAFill op did not run "
            "successfully"

            blob_out = workspace.FetchBlob('out')
            assert np.count_nonzero(blob_out) > 0, "All generated elements are "
            "zeros. Is the random generator functioning correctly?"


if __name__ == "__main__":
    import unittest
    unittest.main()
