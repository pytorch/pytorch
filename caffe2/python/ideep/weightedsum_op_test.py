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
class TestWeightedSumOp(hu.HypothesisTestCase):
    @given(n=st.integers(5, 8), m=st.integers(1, 1),
           d=st.integers(2, 4), grad_on_w=st.booleans(),
           **mu.gcs_ideep_only)
    def test_weighted_sum(self, n, m, d, grad_on_w, gc, dc):
        input_names = []
        input_vars = []
        for i in range(m):
            X_name = 'X' + str(i)
            w_name = 'w' + str(i)
            input_names.extend([X_name, w_name])
            var = np.random.rand(n, d).astype(np.float32)
            vars()[X_name] = var
            input_vars.append(var)
            var = np.random.rand(1).astype(np.float32)
            vars()[w_name] = var
            input_vars.append(var)

        def weighted_sum_op_ref(*args):
            res = np.zeros((n, d))
            for i in range(m):
                res = res + args[2 * i + 1] * args[2 * i]

            return (res, )

        op = core.CreateOperator(
            "WeightedSum",
            input_names,
            ['Y'],
            grad_on_w=grad_on_w,
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=input_vars,
            reference=weighted_sum_op_ref,
        )


if __name__ == "__main__":
    unittest.main()
