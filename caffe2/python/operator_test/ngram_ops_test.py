from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import hypothesis.strategies as st

from caffe2.python import core, workspace
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu

import numpy as np


class TestNGramOps(hu.HypothesisTestCase):
    @given(
        seed=st.integers(0, 2**32 - 1),
        N=st.integers(min_value=10, max_value=100),
        D=st.integers(min_value=2, max_value=10),
        out_of_vcb=st.floats(min_value=0, max_value=0.5),
        max_categorical_limit=st.integers(min_value=5, max_value=20),
        max_in_vcb_val=st.integers(min_value=1000, max_value=10000),
        **hu.gcs_cpu_only
    )
    def test_ngram_from_categorical_op(
        self,
        seed,
        N,
        D,
        out_of_vcb,
        max_categorical_limit,
        max_in_vcb_val,
        gc,
        dc,
    ):
        np.random.seed(seed)
        col_num = max(int(D / 2), 1)
        col_ids = np.random.choice(D, col_num, False).astype(np.int32)
        categorical_limits = np.random.randint(
            2, high=max_categorical_limit, size=col_num
        ).astype(np.int32)
        vcb = [
            np.random.choice(max_in_vcb_val, x, False)
            for x in categorical_limits
        ]
        vals = np.array([x for l in vcb for x in l], dtype=np.int32)

        # Enforce round(floats) to be negative.
        floats = np.random.rand(N, D).astype(np.float32) - 2
        expected_output = []
        for i in range(N):
            val = 0
            for (k, j) in enumerate(col_ids):
                base = np.prod(categorical_limits[:k])
                r = np.random.randint(categorical_limits[k])
                p = np.random.rand()
                if p > out_of_vcb:
                    val += base * r
                    floats[i][j] = vcb[k][r]
            expected_output.append(val)
        expected_output = np.array(expected_output, dtype=np.int32)

        workspace.ResetWorkspace()
        workspace.FeedBlob('floats', floats)
        op = core.CreateOperator(
            "NGramFromCategorical",
            ['floats'],
            ['output'],
            col_ids=col_ids,
            categorical_limits=categorical_limits,
            vals=vals,
        )
        workspace.RunOperatorOnce(op)
        output = workspace.blobs['output']
        np.testing.assert_array_equal(output, expected_output)
