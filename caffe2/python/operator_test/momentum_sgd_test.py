from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu

import hypothesis
from hypothesis import given
import hypothesis.strategies as st
import numpy as np
import unittest


class TestMomentumSGD(hu.HypothesisTestCase):
    @given(n=st.integers(4, 8), nesterov=st.booleans(), **hu.gcs)
    def test_momentum_sgd(self, n, nesterov, gc, dc):
        param = np.random.rand(n).astype(np.float32)
        grad = np.random.rand(n).astype(np.float32)
        lr = np.random.rand(1).astype(np.float32)
        param_momentum = np.random.rand(n).astype(np.float32)
        momentum = 0.9

        def momentum_sgd(grad, param_momentum, lr, param=None):
            if not nesterov:
                adjusted_gradient = lr * grad + momentum * param_momentum
                if param is None:
                    return [adjusted_gradient, adjusted_gradient]
                else:
                    paramup = param - adjusted_gradient
                    return [adjusted_gradient, adjusted_gradient, paramup]
            else:
                m_new = momentum * param_momentum + lr * grad
                grad_new = (1 + momentum) * m_new - momentum * param_momentum
                if param is None:
                    return [grad_new, m_new]
                else:
                    paramup = param - grad_new
                    return [grad_new, m_new, paramup]

        op = core.CreateOperator(
            "MomentumSGDUpdate",
            ["grad", "param_momentum", "lr", "param"],
            ["grad", "param_momentum", "param"],
            momentum=momentum,
            nesterov=int(nesterov),
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
            nesterov=int(nesterov),
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op_noparam,
            inputs=[grad, param_momentum, lr],
            reference=momentum_sgd
        )

    @given(
        inputs=hu.tensors(n=3),
        momentum=st.floats(min_value=0.1, max_value=0.9),
        nesterov=st.booleans(),
        lr=st.floats(min_value=0.1, max_value=0.9),
        data_strategy=st.data(),
        **hu.gcs
    )
    def test_sparse_momentum_sgd(
        self, inputs, momentum, nesterov, lr, data_strategy, gc, dc
    ):
        w, grad, m = inputs

        # Create an indexing array containing values which index into grad
        indices = data_strategy.draw(
            hu.tensor(
                max_dim=1,
                min_value=1,
                max_value=grad.shape[0],
                dtype=np.int64,
                elements=st.sampled_from(np.arange(grad.shape[0])),
            ),
        )

        # Verify that the generated indices are unique
        hypothesis.assume(
            np.array_equal(
                np.unique(indices.flatten()),
                np.sort(indices.flatten())))

        # Sparsify grad
        grad = grad[indices]

        # Make momentum >= 0
        m = np.abs(m)

        # Convert lr to a numpy array
        lr = np.asarray([lr], dtype=np.float32)

        op = core.CreateOperator(
            "SparseMomentumSGDUpdate", ["grad", "m", "lr", "param", "indices"],
            ["adjusted_grad", "m", "param"],
            momentum=momentum,
            nesterov=int(nesterov),
            device_option=gc
        )

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

        self.assertReferenceChecks(
            gc,
            op,
            [grad, m, lr, w, indices],
            sparse)

    @given(n=st.integers(4, 8), nesterov=st.booleans(), **hu.gcs_gpu_only)
    @unittest.skipIf(not workspace.has_gpu_support, "No gpu support.")
    def test_fp16momentum_sgd(self, n, nesterov, gc, dc):
        gpuvers = workspace.GetDeviceProperties(0)["major"]
        if gpuvers < 6:
            print("No FP16 support because major version {} < 6".format(gpuvers))
            return

        param = np.random.rand(n).astype(np.float16)
        grad = np.random.rand(n).astype(np.float16)
        lr = np.random.rand(1).astype(np.float32)
        param_momentum = np.random.rand(n).astype(np.float16)
        momentum = 0.9
        nesterov = True

        def momentum_sgd(grad, param_momentum, lr, param=None):
            if not nesterov:
                adjusted_gradient = lr * grad + momentum * param_momentum
                paramup = param - adjusted_gradient
                return [adjusted_gradient, adjusted_gradient, paramup]
            else:
                m_new = momentum * param_momentum + lr * grad
                grad_new = (1 + momentum) * m_new - momentum * param_momentum
                paramup = param - grad_new
                return [grad_new, m_new, paramup]

        op = core.CreateOperator(
            "FP16MomentumSGDUpdate",
            ["grad", "param_momentum", "lr", "param"],
            ["grad", "param_momentum", "param"],
            momentum=momentum,
            nesterov=int(nesterov),
            weight_decay=0.0,
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[grad, param_momentum, lr, param],
            reference=momentum_sgd
        )


if __name__ == "__main__":
    unittest.main()
