from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial

from hypothesis import given
import hypothesis.strategies as st
import numpy as np
import unittest


class TestMean(serial.SerializedTestCase):
    @serial.given(
        k=st.integers(1, 5),
        n=st.integers(1, 10),
        m=st.integers(1, 10),
        in_place=st.booleans(),
        seed=st.integers(0, 2**32 - 1),
        **hu.gcs
    )
    def test_mean(self, k, n, m, in_place, seed, gc, dc):
        np.random.seed(seed)
        input_names = []
        input_vars = []

        for i in range(k):
            X_name = 'X' + str(i)
            input_names.append(X_name)
            var = np.random.randn(n, m).astype(np.float32)
            input_vars.append(var)

        def mean_ref(*args):
            return [np.mean(args, axis=0)]

        op = core.CreateOperator(
            "Mean",
            input_names,
            ['Y' if not in_place else 'X0'],
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=input_vars,
            reference=mean_ref,
        )

        self.assertGradientChecks(
            device_option=gc,
            op=op,
            inputs=input_vars,
            outputs_to_check=0,
            outputs_with_grads=[0],
        )

        self.assertDeviceChecks(dc, op, input_vars, [0])

    @serial.given(
        in_place=st.booleans(),
        seed=st.integers(0, 2**32 - 1),
        **hu.gcs
    )
    def test_mean_broadcast(self, in_place, seed, gc, dc):
        np.random.seed(seed)

        X = np.random.randn(2, 3, 4, 5).astype(np.float32)
        Y = np.random.randn(1, 4, 5).astype(np.float32)
        Z = np.random.randn(2, 3, 1, 1).astype(np.float32)
        W = np.random.randn(1, 6).astype(np.float32)
        U = np.random.randn(4, 5).astype(np.float32)

        def mean_ref(*args):
            return [np.mean(args, axis=0)]

        # Check that non-valid broadcast is not allowed
        op_bad = core.CreateOperator(
            "Mean",
            ['X', 'W'],
            ['out'],
        )
        self.assertRunOpRaises(
            gc,
            op_bad,
            inputs=[X, W],
            exception=RuntimeError,
        )

        # Check that non-valid in-place broadcast is not allowed
        op_in_place_bad = core.CreateOperator(
            "Mean",
            ['Y', 'X'],
            ['Y'],
        )
        self.assertRunOpRaises(
            gc,
            op_in_place_bad,
            inputs=[Y, X],
            exception=RuntimeError
        )
        op_in_place_bad_3_inputs = core.CreateOperator(
            "Mean",
            ['Y', 'U', 'X'],
            ['Y'],
        )
        self.assertRunOpRaises(
            gc,
            op_in_place_bad_3_inputs,
            inputs=[Y, U, X],
            exception=RuntimeError
        )

        # Check normal operation
        if in_place:
            op = core.CreateOperator(
                "Mean",
                ['X', 'Y', 'Z'],
                ['X'],
            )
            input_vars = [X, Y, Z]
        else:
            op = core.CreateOperator(
                "Mean",
                ['Y', 'Z', 'X'],
                ['out'],
            )
            input_vars = [Y, Z, X]
        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=input_vars,
            reference=mean_ref,
        )
        for i in range(len(input_vars)):
            self.assertGradientChecks(gc, op, input_vars, i, [0])

        self.assertDeviceChecks(dc, op, input_vars, [0])

        # Check size zero case
        A = np.ones((1,1)).astype(np.float32)
        B = np.ones(0).astype(np.float32)

        op0 = core.CreateOperator(
            "Mean",
            ['A', 'B'],
            ['C'],
        )
        self.assertReferenceChecks(
            device_option=gc,
            op=op0,
            inputs=[A, B],
            reference=mean_ref,
        )
        self.assertGradientChecks(gc, op0, [A, B], 0, [0])


if __name__ == "__main__":
    unittest.main()
