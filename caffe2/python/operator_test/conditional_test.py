from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from hypothesis import given
import hypothesis.strategies as st
from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu


class TestConditionalOp(hu.HypothesisTestCase):
    @given(rows_num=st.integers(1, 10000), **hu.gcs_cpu_only)
    def test_conditional(self, rows_num, gc, dc):
        op = core.CreateOperator(
            "Conditional", ["condition", "data_t", "data_f"], "output"
        )
        data_t = np.random.random((rows_num, 10, 20)).astype(np.float32)
        data_f = np.random.random((rows_num, 10, 20)).astype(np.float32)
        condition = np.random.choice(a=[True, False], size=rows_num)

        def ref(condition, data_t, data_f):
            output = [
                data_t[i] if condition[i] else data_f[i]
                for i in range(rows_num)
            ]
            return (output,)

        self.assertReferenceChecks(gc, op, [condition, data_t, data_f], ref)
