from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu

from hypothesis import given
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

    @given(inputs=hu.tensors(n=3),
           momentum=st.floats(min_value=0.1, max_value=0.9),
           nesterov=st.booleans(),
           lr=st.floats(min_value=0.1, max_value=0.9),
           **hu.gcs_cpu_only)
    def test_sparse_momentum_sgd(
            self, inputs, momentum, nesterov, lr, gc, dc):
        w, grad, m = inputs
        indices = np.arange(m.shape[0])
        indices = indices[indices % 2 == 0]

        grad = grad[indices]
        m = np.abs(m)
        lr = np.asarray([lr], dtype=np.float32)

        op = core.CreateOperator(
            "SparseMomentumSGDUpdate",
            ["grad", "m", "lr", "param", "indices"],
            ["adjusted_grad", "m", "param"],
            momentum=momentum,
            nesterov=int(nesterov),
            device_option=gc)

        # Reference
        def momentum_sgd(grad, m, lr):
            lr = lr[0]
            if not nesterov:
                adjusted_gradient = lr * grad + momentum * m
                return (adjusted_gradient, adjusted_gradient)
            else:
                m_new = momentum * m + lr * grad
                return ((1 + momentum) * m_new - momentum * m, m_new)

        def sparse(grad, m, lr, param, i):
            grad_new, m_new = momentum_sgd(grad, m[i], lr)
            m[i] = m_new
            param[i] -= grad_new
            return (grad_new, m, param)

        self.assertReferenceChecks(gc, op, [grad, m, lr, w, indices], sparse)


if __name__ == "__main__":
    unittest.main()
