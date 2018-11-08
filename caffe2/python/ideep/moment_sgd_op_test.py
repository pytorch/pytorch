from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import hypothesis.strategies as st
import unittest
import caffe2.python.hypothesis_test_util as hu
from caffe2.python import core, workspace
from hypothesis import given
import caffe2.python.ideep_test_util as mu


@unittest.skipIf(not workspace.C.use_mkldnn, "No MKLDNN support.")
class TestMomentumSGDUpdateOps(hu.HypothesisTestCase):
    @given(n=st.integers(4, 8), nesterov=st.booleans(),
           **mu.gcs)
    def test_MomentumSGDUpdate(self, n, nesterov, gc, dc):
        param = np.random.rand(n).astype(np.float32)
        grad = np.random.rand(n).astype(np.float32)
        lr = np.random.rand(1).astype(np.float32)
        param_momentum = np.random.rand(n).astype(np.float32)
        momentum = 0.9
        op = core.CreateOperator(
            "MomentumSGDUpdate",
            ["grad", "param_momentum", "lr", "param"],
            ["grad", "param_momentum", "param"],
            momentum=momentum,
            nesterov=int(nesterov),
        )
        # Iter lives on the CPU
        input_device_options = {'lr': hu.cpu_do}

        self.assertDeviceChecks(
            dc,
            op,
            [grad, param_momentum, lr, param],
            [0],
            input_device_options=input_device_options,
            threshold=0.001)

        op_noparam = core.CreateOperator(
            "MomentumSGD",
            ["grad", "param_momentum", "lr"],
            ["grad", "param_momentum"],
            momentum=momentum,
            nesterov=int(nesterov),
        )

        self.assertDeviceChecks(
            dc,
            op_noparam,
            [grad, param_momentum, lr],
            [0],
            input_device_options=input_device_options,
            threshold=0.001)


if __name__ == "__main__":
    unittest.main()
