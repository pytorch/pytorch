from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import hypothesis.strategies as st
from hypothesis import given
import numpy as np
from caffe2.python import core, workspace, model_helper
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.ideep_test_util as mu


@unittest.skipIf(not workspace.C.use_mkldnn, "No MKLDNN support.")
class LeakyReluTest(hu.HypothesisTestCase):
    def _get_inputs(self, N, C, H, W, order):
        input_data = np.random.rand(N, C, H, W).astype(np.float32) - 0.5

        # default step size is 0.05
        input_data[np.logical_and(
            input_data >= 0, input_data <= 0.051)] = 0.051
        input_data[np.logical_and(
            input_data <= 0, input_data >= -0.051)] = -0.051

        return input_data,

    def _get_op(self, device_option, alpha, order, inplace=False):
        outputs = ['output' if not inplace else "input"]
        op = core.CreateOperator(
            'LeakyRelu',
            ['input'],
            outputs,
            alpha=alpha,
            device_option=device_option)
        return op

    def _feed_inputs(self, input_blobs, device_option):
        names = ['input', 'scale', 'bias']
        for name, blob in zip(names, input_blobs):
            self.ws.create_blob(name).feed(blob, device_option=device_option)

    @given(N=st.integers(2, 3),
           C=st.integers(2, 3),
           H=st.integers(2, 3),
           W=st.integers(2, 3),
           alpha=st.floats(0, 1),
           seed=st.integers(0, 1000),
           **mu.gcs)
    def test_leaky_relu_gradients(self, gc, dc, N, C, H, W, alpha, seed):
        np.random.seed(seed)

        op = self._get_op(
            device_option=gc,
            alpha=alpha,
            order='NCHW')
        input_blobs = self._get_inputs(N, C, H, W, "NCHW")

        self.assertDeviceChecks(dc, op, input_blobs, [0])
        self.assertGradientChecks(gc, op, input_blobs, 0, [0])

    @given(N=st.integers(2, 10),
           C=st.integers(3, 10),
           H=st.integers(5, 10),
           W=st.integers(7, 10),
           alpha=st.floats(0, 1),
           seed=st.integers(0, 1000))
    def test_leaky_relu_model_helper_helper(self, N, C, H, W, alpha, seed):
        np.random.seed(seed)
        order = 'NCHW'
        arg_scope = {'order': order}
        model = model_helper.ModelHelper(name="test_model", arg_scope=arg_scope)
        model.LeakyRelu(
            'input',
            'output',
            alpha=alpha)

        input_blob = np.random.rand(N, C, H, W).astype(np.float32)

        self.ws.create_blob('input').feed(input_blob)

        self.ws.create_net(model.param_init_net).run()
        self.ws.create_net(model.net).run()

        output_blob = self.ws.blobs['output'].fetch()

        assert output_blob.shape == (N, C, H, W)


if __name__ == "__main__":
    unittest.main()
