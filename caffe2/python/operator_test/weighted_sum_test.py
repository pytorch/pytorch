




from caffe2.python import core
from hypothesis import given, settings

import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np


class TestWeightedSumOp(serial.SerializedTestCase):

    @given(
        n=st.integers(1, 8), m=st.integers(1, 10), d=st.integers(1, 4),
        in_place=st.booleans(), engine=st.sampled_from(["", "CUDNN"]),
        seed=st.integers(min_value=0, max_value=65535),
        **hu.gcs)
    @settings(deadline=10000)
    def test_weighted_sum(
            self, n, m, d, in_place, engine, seed, gc, dc):
        input_names = []
        input_vars = []
        np.random.seed(seed)
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
            [input_names[0]] if in_place else ['Y'],
            engine=engine,
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=input_vars,
            reference=weighted_sum_op_ref,
        )
        self.assertDeviceChecks(dc, op, input_vars, [0])

    @given(n=st.integers(1, 8), m=st.integers(1, 10), d=st.integers(1, 4),
           grad_on_w=st.booleans(),
           seed=st.integers(min_value=0, max_value=65535), **hu.gcs_cpu_only)
    @settings(deadline=10000)
    def test_weighted_sum_grad(
            self, n, m, d, grad_on_w, seed, gc, dc):
        input_names = []
        input_vars = []
        np.random.seed(seed)
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

        op = core.CreateOperator(
            "WeightedSum",
            input_names,
            ['Y'],
            grad_on_w=grad_on_w,
        )

        output_to_check_grad = (
            range(2 * m) if grad_on_w else range(0, 2 * m, 2))
        for i in output_to_check_grad:
            self.assertGradientChecks(
                device_option=gc,
                op=op,
                inputs=input_vars,
                outputs_to_check=i,
                outputs_with_grads=[0],
            )


if __name__ == "__main__":
    serial.testWithArgs()
