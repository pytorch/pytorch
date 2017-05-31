from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import hypothesis.strategies as st

from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu

import numpy as np


class TestFillerOperator(hu.HypothesisTestCase):

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
