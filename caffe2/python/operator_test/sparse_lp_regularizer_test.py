




import hypothesis
from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st
import numpy as np

from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu


class TestSparseLpNorm(hu.HypothesisTestCase):

    @staticmethod
    def ref_lpnorm(param_in, p, reg_lambda):
        """Reference function that should be matched by the Caffe2 operator."""
        if p == 2.0:
            return param_in * (1 - reg_lambda)
        if p == 1.0:
            reg_term = np.ones_like(param_in) * reg_lambda * np.sign(param_in)
            param_out = param_in - reg_term
            param_out[np.abs(param_in) <= reg_lambda] = 0.
            return param_out
        raise ValueError

    # Suppress filter_too_much health check.
    # Likely caused by `assume` call falling through too often.
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    @given(inputs=hu.tensors(n=1, min_dim=2, max_dim=2),
           p=st.integers(min_value=1, max_value=2),
           reg_lambda=st.floats(min_value=1e-4, max_value=1e-1),
           data_strategy=st.data(),
           **hu.gcs_cpu_only)
    def test_sparse_lpnorm(self, inputs, p, reg_lambda, data_strategy, gc, dc):

        param, = inputs
        param += 0.02 * np.sign(param)
        param[param == 0.0] += 0.02

        # Create an indexing array containing values that are lists of indices,
        # which index into param
        indices = data_strategy.draw(
            hu.tensor(dtype=np.int64, min_dim=1, max_dim=1,
                      elements=st.sampled_from(np.arange(param.shape[0]))),
        )
        hypothesis.note('indices.shape: %s' % str(indices.shape))

        # For now, the indices must be unique
        hypothesis.assume(np.array_equal(np.unique(indices.flatten()),
                                         np.sort(indices.flatten())))

        op = core.CreateOperator(
            "SparseLpRegularizer",
            ["param", "indices"],
            ["param"],
            p=float(p),
            reg_lambda=reg_lambda,
        )

        def ref_sparse_lp_regularizer(param, indices, grad=None):
            param_out = np.copy(param)
            for _, index in enumerate(indices):
                param_out[index] = self.ref_lpnorm(
                    param[index],
                    p=p,
                    reg_lambda=reg_lambda,
                )
            return (param_out,)

        self.assertReferenceChecks(
            gc, op, [param, indices],
            ref_sparse_lp_regularizer
        )
