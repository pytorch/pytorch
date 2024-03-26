

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
from caffe2.python import core, workspace
from hypothesis import given


class TestEnsureClipped(hu.HypothesisTestCase):
    @given(
        X=hu.arrays(dims=[5, 10], elements=hu.floats(min_value=-1.0, max_value=1.0)),
        in_place=st.booleans(),
        sparse=st.booleans(),
        indices=hu.arrays(dims=[5], elements=st.booleans()),
        **hu.gcs_cpu_only
    )
    def test_ensure_clipped(self, X, in_place, sparse, indices, gc, dc):
        if (not in_place) and sparse:
            return
        param = X.astype(np.float32)
        m, n = param.shape
        indices = np.array(np.nonzero(indices)[0], dtype=np.int64)
        grad = np.random.rand(len(indices), n)
        workspace.FeedBlob("indices", indices)
        workspace.FeedBlob("grad", grad)
        workspace.FeedBlob("param", param)
        input = ["param", "indices", "grad"] if sparse else ["param"]
        output = "param" if in_place else "output"
        op = core.CreateOperator("EnsureClipped", input, output, min=0.0)
        workspace.RunOperatorOnce(op)

        def ref():
            return (
                np.array(
                    [np.clip(X[i], 0, None) if i in indices else X[i] for i in range(m)]
                )
                if sparse
                else np.clip(X, 0, None)
            )

        npt.assert_allclose(workspace.blobs[output], ref(), rtol=1e-3)
