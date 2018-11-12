from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from hypothesis import given, assume
import hypothesis.strategies as st

from caffe2.python import core, model_helper, brew
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial

import unittest
import os


class TestInstanceNorm(serial.SerializedTestCase):

    def _get_inputs(self, N, C, H, W, order):
        if order == 'NCHW':
            input_data = np.random.rand(N, C, H, W).astype(np.float32)
        elif order == 'NHWC':
            # Allocate in the same order as NCHW and transpose to make sure
            # the inputs are identical on freshly-seeded calls.
            input_data = np.random.rand(N, C, H, W).astype(np.float32)
            input_data = np.transpose(input_data, axes=(0, 2, 3, 1))
        else:
            raise Exception('unknown order type ({})'.format(order))

        scale_data = np.random.rand(C).astype(np.float32)
        bias_data = np.random.rand(C).astype(np.float32)
        return input_data, scale_data, bias_data

    def _get_op(self, device_option, store_mean, store_inv_stdev, epsilon,
                order, inplace=False):
        outputs = ['output' if not inplace else "input"]
        if store_mean or store_inv_stdev:
            outputs += ['mean']
        if store_inv_stdev:
            outputs += ['inv_stdev']
        op = core.CreateOperator(
            'InstanceNorm',
            ['input', 'scale', 'bias'],
            outputs,
            order=order,
            epsilon=epsilon,
            device_option=device_option)
        return op

    def _feed_inputs(self, input_blobs, device_option):
        names = ['input', 'scale', 'bias']
        for name, blob in zip(names, input_blobs):
            self.ws.create_blob(name).feed(blob, device_option=device_option)

    @given(gc=hu.gcs['gc'],
           dc=hu.gcs['dc'],
           N=st.integers(2, 3),
           C=st.integers(2, 3),
           H=st.integers(2, 3),
           W=st.integers(2, 3),
           order=st.sampled_from(['NCHW', 'NHWC']),
           epsilon=st.floats(1e-6, 1e-4),
           store_mean=st.booleans(),
           seed=st.integers(0, 1000),
           store_inv_stdev=st.booleans())
    def test_instance_norm_gradients(
            self, gc, dc, N, C, H, W, order, store_mean, store_inv_stdev,
            epsilon, seed):
        np.random.seed(seed)

        # force store_inv_stdev if store_mean to match existing forward pass
        # implementation
        store_inv_stdev |= store_mean

        op = self._get_op(
            device_option=gc,
            store_mean=store_mean,
            store_inv_stdev=store_inv_stdev,
            epsilon=epsilon,
            order=order)
        input_blobs = self._get_inputs(N, C, H, W, order)

        output_indices = [0]
        # if store_inv_stdev is turned on, store_mean must also be forced on
        if store_mean or store_inv_stdev:
            output_indices += [1]
        if store_inv_stdev:
            output_indices += [2]
        self.assertDeviceChecks(dc, op, input_blobs, output_indices)
        # The gradient only flows from output #0 since the other two only
        # store the temporary mean and inv_stdev buffers.
        # Check dl/dinput
        self.assertGradientChecks(gc, op, input_blobs, 0, [0], stepsize=0.005,
                                  threshold=0.01)
        # Check dl/dscale
        self.assertGradientChecks(gc, op, input_blobs, 1, [0])
        # Check dl/dbias
        self.assertGradientChecks(gc, op, input_blobs, 2, [0])

    @given(gc=hu.gcs['gc'],
           dc=hu.gcs['dc'],
           N=st.integers(2, 10),
           C=st.integers(3, 10),
           H=st.integers(5, 10),
           W=st.integers(7, 10),
           seed=st.integers(0, 1000),
           epsilon=st.floats(1e-6, 1e-4),
           store_mean=st.booleans(),
           store_inv_stdev=st.booleans())
    def test_instance_norm_layout(self, gc, dc, N, C, H, W, store_mean,
                                  store_inv_stdev, epsilon, seed):
        # force store_inv_stdev if store_mean to match existing forward pass
        # implementation
        store_inv_stdev |= store_mean

        outputs = {}
        for order in ('NCHW', 'NHWC'):
            np.random.seed(seed)
            input_blobs = self._get_inputs(N, C, H, W, order)
            self._feed_inputs(input_blobs, device_option=gc)
            op = self._get_op(
                device_option=gc,
                store_mean=store_mean,
                store_inv_stdev=store_inv_stdev,
                epsilon=epsilon,
                order=order)
            self.ws.run(op)
            outputs[order] = self.ws.blobs['output'].fetch()
        np.testing.assert_allclose(
            outputs['NCHW'],
            outputs['NHWC'].transpose((0, 3, 1, 2)),
            atol=1e-4,
            rtol=1e-4)

    @serial.given(gc=hu.gcs['gc'],
           dc=hu.gcs['dc'],
           N=st.integers(2, 10),
           C=st.integers(3, 10),
           H=st.integers(5, 10),
           W=st.integers(7, 10),
           order=st.sampled_from(['NCHW', 'NHWC']),
           epsilon=st.floats(1e-6, 1e-4),
           store_mean=st.booleans(),
           seed=st.integers(0, 1000),
           store_inv_stdev=st.booleans(),
           inplace=st.booleans())
    def test_instance_norm_reference_check(
            self, gc, dc, N, C, H, W, order, store_mean, store_inv_stdev,
            epsilon, seed, inplace):
        np.random.seed(seed)

        # force store_inv_stdev if store_mean to match existing forward pass
        # implementation
        store_inv_stdev |= store_mean
        if order != "NCHW":
            assume(not inplace)

        inputs = self._get_inputs(N, C, H, W, order)
        op = self._get_op(
            device_option=gc,
            store_mean=store_mean,
            store_inv_stdev=store_inv_stdev,
            epsilon=epsilon,
            order=order,
            inplace=inplace)

        def ref(input_blob, scale_blob, bias_blob):
            if order == 'NHWC':
                input_blob = np.transpose(input_blob, axes=(0, 3, 1, 2))

            mean_blob = input_blob.reshape((N, C, -1)).mean(axis=2)
            inv_stdev_blob = 1.0 / \
                np.sqrt(input_blob.reshape((N, C, -1)).var(axis=2) + epsilon)
            # _bc indicates blobs that are reshaped for broadcast
            scale_bc = scale_blob[np.newaxis, :, np.newaxis, np.newaxis]
            mean_bc = mean_blob[:, :, np.newaxis, np.newaxis]
            inv_stdev_bc = inv_stdev_blob[:, :, np.newaxis, np.newaxis]
            bias_bc = bias_blob[np.newaxis, :, np.newaxis, np.newaxis]
            normalized_blob = scale_bc * (input_blob - mean_bc) * inv_stdev_bc \
                + bias_bc

            if order == 'NHWC':
                normalized_blob = np.transpose(
                    normalized_blob, axes=(0, 2, 3, 1))

            if not store_mean and not store_inv_stdev:
                return normalized_blob,
            elif not store_inv_stdev:
                return normalized_blob, mean_blob
            else:
                return normalized_blob, mean_blob, inv_stdev_blob

        self.assertReferenceChecks(gc, op, inputs, ref)

    @given(gc=hu.gcs['gc'],
           dc=hu.gcs['dc'],
           N=st.integers(2, 10),
           C=st.integers(3, 10),
           H=st.integers(5, 10),
           W=st.integers(7, 10),
           order=st.sampled_from(['NCHW', 'NHWC']),
           epsilon=st.floats(1e-6, 1e-4),
           store_mean=st.booleans(),
           seed=st.integers(0, 1000),
           store_inv_stdev=st.booleans())
    def test_instance_norm_device_check(
            self, gc, dc, N, C, H, W, order, store_mean, store_inv_stdev,
            epsilon, seed):
        np.random.seed(seed)

        # force store_inv_stdev if store_mean to match existing forward pass
        # implementation
        store_inv_stdev |= store_mean

        inputs = self._get_inputs(N, C, H, W, order)
        op = self._get_op(
            device_option=gc,
            store_mean=store_mean,
            store_inv_stdev=store_inv_stdev,
            epsilon=epsilon,
            order=order)

        self.assertDeviceChecks(dc, op, inputs, [0])

    @given(is_test=st.booleans(),
           N=st.integers(2, 10),
           C=st.integers(3, 10),
           H=st.integers(5, 10),
           W=st.integers(7, 10),
           order=st.sampled_from(['NCHW', 'NHWC']),
           epsilon=st.floats(1e-6, 1e-4),
           seed=st.integers(0, 1000))
    def test_instance_norm_model_helper(
            self, N, C, H, W, order, epsilon, seed, is_test):
        np.random.seed(seed)
        model = model_helper.ModelHelper(name="test_model")
        brew.instance_norm(
            model,
            'input',
            'output',
            C,
            epsilon=epsilon,
            order=order,
            is_test=is_test)

        input_blob = np.random.rand(N, C, H, W).astype(np.float32)
        if order == 'NHWC':
            input_blob = np.transpose(input_blob, axes=(0, 2, 3, 1))

        self.ws.create_blob('input').feed(input_blob)

        self.ws.create_net(model.param_init_net).run()
        self.ws.create_net(model.net).run()

        if is_test:
            scale = self.ws.blobs['output_s'].fetch()
            assert scale is not None
            assert scale.shape == (C, )
            bias = self.ws.blobs['output_b'].fetch()
            assert bias is not None
            assert bias.shape == (C, )

        output_blob = self.ws.blobs['output'].fetch()
        if order == 'NHWC':
            output_blob = np.transpose(output_blob, axes=(0, 3, 1, 2))

        assert output_blob.shape == (N, C, H, W)


if __name__ == '__main__':
    import unittest
    unittest.main()
