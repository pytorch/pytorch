from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np

import unittest


class TestMomentumSGD(hu.HypothesisTestCase):

    @given(n=st.integers(4, 8), **hu.gcs)
    def test_momentum_sgd(self, n, gc, dc):
        param = np.random.rand(n).astype(np.float32)
        grad = np.random.rand(n).astype(np.float32)
        lr = np.random.rand(1).astype(np.float32)
        param_momentum = np.random.rand(n).astype(np.float32)
        momentum = 0.9

        def momentum_sgd(grad, param_momentum, lr, param=None):
            adjgrad = lr * grad + momentum * param_momentum
            if param is None:
                return [adjgrad, adjgrad]
            else:
                paramup = param - adjgrad
                return [adjgrad, adjgrad, paramup]

        op = core.CreateOperator(
            "MomentumSGDUpdate",
            ["grad", "param_momentum", "lr", "param"],
            ["grad", "param_momentum", "param"],
            momentum=momentum,
            nesterov=0,
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[grad, param_momentum, lr, param],
            reference=momentum_sgd
        )

        op_noparam = core.CreateOperator(
            "MomentumSGD",
            ["grad", "param_momentum", "lr"],
            ["grad", "param_momentum"],
            momentum=momentum,
            nesterov=0,
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op_noparam,
            inputs=[grad, param_momentum, lr],
            reference=momentum_sgd
        )


if __name__ == "__main__":
    unittest.main()
