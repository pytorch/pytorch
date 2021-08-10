import caffe2.python.hypothesis_test_util as hu
import hypothesis
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core
from hypothesis import HealthCheck, given, settings


class TestSparseNormalize(hu.HypothesisTestCase):
    @staticmethod
    def ref_normalize(param_in, use_max_norm, norm):
        param_norm = np.linalg.norm(param_in) + 1e-12
        if (use_max_norm and param_norm > norm) or not use_max_norm:
            param_in = param_in * norm / param_norm
        return param_in

    # Suppress filter_too_much health check.
    # Likely caused by `assume` call falling through too often.
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    @given(
        inputs=hu.tensors(n=2, min_dim=2, max_dim=2),
        use_max_norm=st.booleans(),
        norm=st.floats(min_value=1.0, max_value=4.0),
        data_strategy=st.data(),
        use_fp16=st.booleans(),
        **hu.gcs_cpu_only
    )
    def test_sparse_normalize(
        self, inputs, use_max_norm, norm, data_strategy, use_fp16, gc, dc
    ):
        param, grad = inputs
        param += 0.02 * np.sign(param)
        param[param == 0.0] += 0.02

        if use_fp16:
            param = param.astype(np.float16)
            grad = grad.astype(np.float16)

        # Create an indexing array containing values that are lists of indices,
        # which index into param
        indices = data_strategy.draw(
            hu.tensor(
                dtype=np.int64,
                min_dim=1,
                max_dim=1,
                elements=st.sampled_from(np.arange(param.shape[0])),
            )
        )
        hypothesis.note("indices.shape: %s" % str(indices.shape))

        # For now, the indices must be unique
        hypothesis.assume(
            np.array_equal(np.unique(indices.flatten()), np.sort(indices.flatten()))
        )

        op1 = core.CreateOperator(
            "Float16SparseNormalize" if use_fp16 else "SparseNormalize",
            ["param", "indices"],
            ["param"],
            use_max_norm=use_max_norm,
            norm=norm,
        )

        # Sparsify grad
        grad = grad[indices]

        op2 = core.CreateOperator(
            "Float16SparseNormalize" if use_fp16 else "SparseNormalize",
            ["param", "indices", "grad"],
            ["param"],
            use_max_norm=use_max_norm,
            norm=norm,
        )

        def ref_sparse_normalize(param, indices, grad=None):
            param_out = np.copy(param)
            for _, index in enumerate(indices):
                param_out[index] = self.ref_normalize(param[index], use_max_norm, norm)
            return (param_out,)

        # self.assertDeviceChecks(dc, op, [param, indices], [0])
        self.assertReferenceChecks(
            gc,
            op1,
            [param, indices],
            ref_sparse_normalize,
            threshold=1e-2 if use_fp16 else 1e-4,
        )

        self.assertReferenceChecks(
            gc,
            op2,
            [param, indices, grad],
            ref_sparse_normalize,
            threshold=1e-2 if use_fp16 else 1e-4,
        )
