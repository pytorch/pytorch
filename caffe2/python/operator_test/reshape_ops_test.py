from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import six
from numpy.testing import assert_array_equal

from caffe2.python import core, workspace
from caffe2.python.test_util import TestCase
from caffe2.proto import caffe2_pb2


class TestLengthsToShapeOps(TestCase):
    def test_lengths_to_shape_ops(self):
        workspace.FeedBlob('l', np.array([200, 200, 200], dtype=np.int32))
        workspace.RunOperatorOnce(core.CreateOperator(
            'LengthsToShape', ['l'], ['s']))
        workspace.FeedBlob('res', np.array([3, 200], dtype=np.int32))
        assert_array_equal(workspace.FetchBlob('s'), workspace.FetchBlob('res'))

    def test_reshape_ops(self):
        workspace.FeedBlob('res', np.array([[0, 0, 0, 0]], dtype=np.float32))
        workspace.FeedBlob('shape', np.array([1, 4], dtype=np.int32))
        workspace.FeedBlob('input', np.zeros((2, 2), dtype=np.float32))
        workspace.RunOperatorOnce(core.CreateOperator(
            'Reshape', ['input', 'shape'], ['output', 'old_shape']))
        assert_array_equal(workspace.FetchBlob('output'),
                           workspace.FetchBlob('res'))

    def test_basic_reshape(self):
        _test_reshape(old_shape=(4, 2, 1), new_shape=(2, 4))
        _test_reshape(old_shape=(4, 2, 1), new_shape=(2, 4), arg_shape=False)

    def test_missing_dim(self):
        _test_reshape(old_shape=(4, 2, 1), new_shape=(-1, 8))
        _test_reshape(old_shape=(4, 2, 1), new_shape=(-1, 8), arg_shape=False)

    def test_in_place(self):
        _test_reshape(old_shape=(4, 2, 1), new_shape=(-1, 8), in_place=True)
        _test_reshape(old_shape=(4, 2, 1), new_shape=(-1, 8),
                     in_place=True, arg_shape=False)

    def test_zero_dim(self):
        _test_reshape(old_shape=(4, 2, 1), new_shape=(0, 0, 0),
                     expected_shape=(4, 2, 1))
        _test_reshape(old_shape=(4, 2, 1), new_shape=(0, 0, 0),
                     expected_shape=(4, 2, 1), arg_shape=False)
        _test_reshape(old_shape=(4, 2, 1), new_shape=(0, 2, 1),
                     expected_shape=(4, 2, 1))
        _test_reshape(old_shape=(4, 2, 1), new_shape=(0, 2, 1),
                     expected_shape=(4, 2, 1), arg_shape=False)
        _test_reshape(old_shape=(0, 0), new_shape=(0, 0, 0),
                     expected_shape=(0, 0, 0), arg_shape=False)

    def test_zero_dim_and_missing_dim(self):
        _test_reshape(old_shape=(4, 2, 1), new_shape=(0, -1, 0),
                     expected_shape=(4, 2, 1))
        _test_reshape(old_shape=(4, 2, 1), new_shape=(0, -1, 0),
                     expected_shape=(4, 2, 1), arg_shape=False)
        _test_reshape(old_shape=(4, 3, 2), new_shape=(-1, 0),
                     expected_shape=(8, 3))
        _test_reshape(old_shape=(4, 3, 2), new_shape=(-1, 0),
                     expected_shape=(8, 3), arg_shape=False)

        with six.assertRaisesRegex(self, RuntimeError, "size is zero"):
            _test_reshape(old_shape=(2, 0), new_shape=(-1, 0),
                          expected_shape=(2, 0), arg_shape=False)

    def test_backprop(self):
        old_shape = (4, 2, 1)
        new_shape = (1, 8)
        X = np.random.rand(*old_shape).astype(np.float32)
        Y = np.random.rand(*new_shape).astype(np.float32)

        net = core.Net('net')

        net.GivenTensorFill([], 'X', shape=old_shape, values=X.flatten())
        net.GivenTensorFill([], 'Y', shape=new_shape, values=Y.flatten())

        net.Reshape(['X'], ['X_out', 'old_shape'], shape=new_shape)
        net.DotProduct(['X_out', 'Y'], 'Z')
        net.AddGradientOperators(['Z'])

        workspace.RunNetOnce(net)

        Z = workspace.FetchBlob('Z')
        X_grad = workspace.FetchBlob('X_grad')

        # Check forward computation
        np.testing.assert_allclose(
            Z.squeeze(), X.reshape(new_shape).dot(Y.T).squeeze(), rtol=1e-5)

        # Check the shape of the gradient
        np.testing.assert_array_equal(X_grad.shape, X.shape)

        # Check the gradient
        np.testing.assert_allclose(X_grad, Y.reshape(old_shape), rtol=1e-5)

    def test_input_shape_changes(self):
        workspace.FeedBlob(
            'input_blob',
            np.array(np.random.rand(10, 20, 10), dtype=np.float32))
        net = core.Net('mynet')
        z, _ = net.Reshape('input_blob',
                           ['z_reshape', 'dummy_size'],
                           shape=(-1, 10))
        workspace.CreateNet(net)
        workspace.RunNet(net)
        workspace.FeedBlob(
            'input_blob',
            np.array(np.random.rand(10, 40, 10), dtype=np.float32))
        workspace.RunNet(net)


def _test_reshape(old_shape, new_shape, expected_shape=None, arg_shape=True,
                 in_place=False):
    devices = [core.DeviceOption(caffe2_pb2.CPU, 0)]
    if workspace.NumGpuDevices() > 0:
        devices.append(core.DeviceOption(workspace.GpuDeviceType, 0))

    for device_opt in devices:
        with core.DeviceScope(device_opt):
            if expected_shape is None:
                expected_shape = new_shape
            X = np.random.rand(*old_shape).astype(np.float32)

            blob_in = 'X'
            blob_out = blob_in if in_place else blob_in + '_out'

            if arg_shape:
                op = core.CreateOperator('Reshape',
                                         [blob_in],
                                         [blob_out, 'old_shape'],
                                         shape=new_shape)
            else:
                op = core.CreateOperator('Reshape',
                                         [blob_in, 'new_shape'],
                                         [blob_out, 'old_shape'])
                workspace.FeedBlob('new_shape', np.asarray(new_shape))

            workspace.FeedBlob(blob_in, X)
            workspace.RunOperatorOnce(op)

            Y = workspace.FetchBlob(blob_out)
            np.testing.assert_allclose(Y, X.reshape(expected_shape))

if __name__ == "__main__":
    import unittest
    unittest.main()
