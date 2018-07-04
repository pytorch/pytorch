from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np


class TestWeightedSumOp(hu.HypothesisTestCase):

    @given(n=st.integers(5, 8), m=st.integers(1, 1),
           d=st.integers(2, 4), grad_on_w=st.booleans(),
           **hu.gcs_cpu_only)
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

        output_to_check_grad = range(2 * m) if grad_on_w else range(0, 2 * m, 2)
        for i in output_to_check_grad:
            self.assertGradientChecks(
                device_option=gc,
                op=op,
                inputs=input_vars,
                outputs_to_check=i,
                outputs_with_grads=[0],
            )
