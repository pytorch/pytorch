




import functools

from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st
import numpy as np

from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu


class TestStorm(hu.HypothesisTestCase):
    @given(inputs=hu.tensors(n=3),
           grad_sq_sum=st.floats(min_value=0.01, max_value=0.99,
                                 allow_nan=False, allow_infinity=False),
           lr=st.floats(min_value=0.01, max_value=1.0,
                        allow_nan=False, allow_infinity=False),
           momentum=st.floats(min_value=0.1, max_value=100.0,
                              allow_nan=False, allow_infinity=False),
           beta=st.floats(min_value=0.1, max_value=10.0,
                          allow_nan=False, allow_infinity=False),
           **hu.gcs_cpu_only)
    def test_storm_dense(self, inputs, grad_sq_sum, lr, momentum, beta, gc, dc):
        param, moment, grad = inputs
        grad_sq_sum = np.array([grad_sq_sum], dtype=np.float32)
        lr = np.array([lr], dtype=np.float32)

        op = core.CreateOperator(
            "Storm",
            ["param", "moment", "grad_sq_sum", "grad", "lr"],
            ["param", "moment", "grad_sq_sum"],
            momentum=momentum,
            beta=beta,
            device_option=gc
        )

        def ref_dense(param, moment, grad_sq_sum, grad, lr, momentum, beta):
            grad_sq_sum_out = grad_sq_sum + np.sum(grad * grad)
            nlr = lr * np.power(beta + grad_sq_sum_out, -1.0 / 3.0)
            alpha = momentum * np.square(nlr)
            moment_out = grad + (1 - alpha) * (moment - grad)
            param_out = param + nlr * moment_out

            return (param_out.astype(np.float32), moment_out.astype(np.float32),
                    grad_sq_sum_out.astype(np.float32))

        self.assertReferenceChecks(
            gc, op,
            [param, moment, grad_sq_sum, grad, lr],
            functools.partial(ref_dense, momentum=momentum, beta=beta)
        )

    # Suppress filter_too_much health check.
    # Likely caused by `assume` call falling through too often.
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    @given(inputs=hu.tensors(n=3),
           grad_sq_sum=st.floats(min_value=0.01, max_value=0.99,
                                 allow_nan=False, allow_infinity=False),
           lr=st.floats(min_value=0.01, max_value=1.0,
                        allow_nan=False, allow_infinity=False),
           momentum=st.floats(min_value=0.1, max_value=100.0,
                              allow_nan=False, allow_infinity=False),
           beta=st.floats(min_value=0.1, max_value=10.0,
                          allow_nan=False, allow_infinity=False),
           **hu.gcs_cpu_only)
    def test_storm_sparse(self, inputs, grad_sq_sum, lr,
                          momentum, beta, gc, dc):
        param, moment, grad = inputs
        grad_sq_sum = np.array([grad_sq_sum], dtype=np.float32)
        lr = np.array([lr], dtype=np.float32)

        # Create an indexing array containing values that are lists of indices,
        # which index into grad
        indices = np.random.choice(np.arange(grad.shape[0]),
                                   size=np.random.randint(grad.shape[0]),
                                   replace=False)

        # Sparsify grad
        grad = grad[indices]

        op = core.CreateOperator(
            "SparseStorm",
            ["param", "moment", "grad_sq_sum", "grad", "indices", "lr"],
            ["param", "moment", "grad_sq_sum"],
            momentum=momentum,
            beta=beta,
            device_option=gc)

        def ref_sparse(param, moment, grad_sq_sum, grad, indices,
                       lr, momentum, beta):
            param_out = np.copy(param)
            moment_out = np.copy(moment)
            grad_sq_sum_out = np.copy(grad_sq_sum)

            grad_sq_sum_out = grad_sq_sum + np.sum(grad * grad)
            nlr = lr * np.power(beta + grad_sq_sum_out, -1.0 / 3.0)
            alpha = momentum * np.square(nlr)
            for i, index in enumerate(indices):
                gi = grad[i]
                moment_out[index] = gi + (1 - alpha) * (moment[index] - gi)
                param_out[index] = param[index] + nlr * moment_out[index]

            return (param_out.astype(np.float32), moment_out.astype(np.float32),
                    grad_sq_sum_out.astype(np.float32))

        self.assertReferenceChecks(
            gc, op,
            [param, moment, grad_sq_sum, grad, indices, lr],
            functools.partial(ref_sparse, momentum=momentum, beta=beta)
        )

    @given(inputs=hu.tensors(n=2),
           grad_sq_sum=st.floats(min_value=0.01, max_value=0.99,
                                 allow_nan=False, allow_infinity=False),
           lr=st.floats(min_value=0.01, max_value=1.0,
                        allow_nan=False, allow_infinity=False),
           momentum=st.floats(min_value=0.1, max_value=100.0,
                              allow_nan=False, allow_infinity=False),
           beta=st.floats(min_value=0.1, max_value=10.0,
                          allow_nan=False, allow_infinity=False),
           data_strategy=st.data(),
           **hu.gcs_cpu_only)
    def test_storm_sparse_empty(self, inputs, grad_sq_sum, lr, momentum,
                                beta, data_strategy, gc, dc):
        param, moment = inputs
        grad_sq_sum = np.array([grad_sq_sum], dtype=np.float32)
        lr = np.array([lr], dtype=np.float32)

        grad = np.empty(shape=(0,) + param.shape[1:], dtype=np.float32)
        indices = np.empty(shape=(0,), dtype=np.int64)

        op = core.CreateOperator(
            "SparseStorm",
            ["param", "moment", "grad_sq_sum", "grad", "indices", "lr"],
            ["param", "moment", "grad_sq_sum"],
            momentum=momentum,
            beta=beta,
            device_option=gc)

        def ref_sparse_empty(param, moment, grad_sq_sum, grad, indices,
                             lr, momentum, beta):
            param_out = np.copy(param)
            moment_out = np.copy(moment)
            grad_sq_sum_out = np.copy(grad_sq_sum)

            return (param_out.astype(np.float32), moment_out.astype(np.float32),
                    grad_sq_sum_out.astype(np.float32))

        self.assertReferenceChecks(
            gc, op,
            [param, moment, grad_sq_sum, grad, indices, lr],
            functools.partial(ref_sparse_empty, momentum=momentum, beta=beta)
        )
