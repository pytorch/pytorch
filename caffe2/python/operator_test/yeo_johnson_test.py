#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.





from caffe2.python import core
from hypothesis import given, settings

import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np


# The reference implementation is susceptible to numerical cancellation when
# *lambda1* is small and *data* is near one. We leave it up to the caller to
# truncate lambda to zero or bound data away from one. Unfortunately, the C++
# implementation may be using higher precision than the python version, which
# could cause this test to fail. We bound inputs away from the critical values.
# (Note that a tolerance of 1e-6 on _either_ parameter is typically sufficient
# to avoid catastrophic cancellation when the other is far from zero/one.)
TOLERANCE = 1e-3


@st.composite
def _inputs(draw):
    N = draw(st.integers(min_value=0, max_value=5))
    D = draw(st.integers(min_value=1, max_value=5))
    # N, D, data, lambda1, lambda2
    return (
        N,
        D,
        draw(st.lists(
            min_size=N * D,
            max_size=N * D,
            elements=st.one_of(
                st.floats(min_value=-10, max_value=1 - TOLERANCE),
                st.floats(min_value=1 + TOLERANCE, max_value=10))
        )),
        draw(st.lists(
            elements=st.one_of(
                st.floats(min_value=-2, max_value=-TOLERANCE),
                st.floats(min_value=TOLERANCE, max_value=2)),
            min_size=D,
            max_size=D,
        )),
        draw(st.lists(
            elements=st.floats(min_value=-2, max_value=2),
            min_size=D,
            max_size=D,
        )),
    )

class TestYeoJohnson(serial.SerializedTestCase):
    @given(
        inputs=_inputs(),
        **hu.gcs_cpu_only
    )
    @settings(deadline=10000)
    def test_batch_box_cox1(self, inputs, gc, dc):
        self.batch_box_cox(inputs, gc, dc)

    @given(**hu.gcs_cpu_only)
    @settings(deadline=10000)
    def test_lambda1_is_all_zero1(self, gc, dc):
        inputs = (1, 1, [[2]], [0], [0])
        self.batch_box_cox(inputs, gc, dc)
        inputs = (2, 1, [[2], [4]], [0], [0])
        self.batch_box_cox(inputs, gc, dc)
        inputs = (1, 3, [[1, 2, 3]], [0, 0, 0], [0, 0, 0])
        self.batch_box_cox(inputs, gc, dc)
        inputs = (2, 3, [[1, 2, 3], [4, 5, 6]], [0, 0, 0], [0, 0, 0])
        self.batch_box_cox(inputs, gc, dc)

    @given(**hu.gcs_cpu_only)
    @settings(deadline=10000)
    def test_lambda1_is_partially_zero1(self, gc, dc):
        inputs = (1, 5, [[1, 2, 3, 4, 5]],
                  [2, -.5, 0, .5, 0], [0.1, 0.2, 0.3, 0.4, 0.5])
        self.batch_box_cox(inputs, gc, dc)
        inputs = (3, 5, [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [1, 2, 3, 4, 5]],
                  [2, -.5, 0, .5, 0], [0.1, 0.2, 0.3, 0.4, 0.5])
        self.batch_box_cox(inputs, gc, dc)
        inputs = (2, 6, [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
                  [2, -.5, 0, .5, 0, 1], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        self.batch_box_cox(inputs, gc, dc)
        inputs = (2, 7, [[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]],
                  [2, -.5, 0, .5, 0, 1, 0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        self.batch_box_cox(inputs, gc, dc)

    @given(**hu.gcs_cpu_only)
    @settings(deadline=10000)
    def test_bound_base_away_from_zero1(self, gc, dc):
        inputs = (2, 3, [[1e-5, 1e-6, 1e-7], [1e-7, -1e-6, 1e-5]],
                  [0, 0, 0], [0, 0, 1e-6])
        self.batch_box_cox(inputs, gc, dc)

    def batch_box_cox(self, inputs, gc, dc):
        N, D, data, lambda1, lambda2 = inputs

        data = np.array(data, dtype=np.float32).reshape(N, D)
        lambda1 = np.array(lambda1, dtype=np.float32)
        lambda2 = np.array(lambda2, dtype=np.float32)

        # Bound data away from one. See comment in _inputs() above.
        base = data + lambda2
        data[(base > 1 - TOLERANCE) & (base < 1 + TOLERANCE)] += 2 * TOLERANCE

        def ref(data, lambda1, lambda2):
            dim_1 = data.shape[1]
            output = np.copy(data)
            if data.size <= 0:
                return [output]

            for i in range(dim_1):
                output[:, i] = data[:, i] + lambda2[i]
                x1 = np.log(output[:, i] + 1)
                x2 = (np.power(output[:, i] + 1, lambda1[i]) - 1) / lambda1[i]
                x3 = -np.log(-output[:, i] + 1)
                x4 = (np.power(1 - output[:, i], 2 - lambda1[i]) - 1) / (lambda1[i] - 2)
                x = np.where(output[:, i] >= 0)
                y = np.where(output[:, i] < 0)
                if lambda1[i] == 0:
                    output[:, i][x] = x1[x]
                    output[:, i][y] = x4[y]
                elif lambda1[i] == 2:
                    output[:, i][x] = x2[x]
                    output[:, i][y] = x3[y]
                else:
                    output[:, i][x] = x2[x]
                    output[:, i][y] = x4[y]
            return [output]

        for naive in [False, True]:
            op = core.CreateOperator(
                'YeoJohnson',
                ['data', 'lambda1', 'lambda2'],
                ['output'],
                naive=naive,
                # Note examples above with D=5, 6, 7.
                # A zero value falls back to the naive implementation.
                min_block_size=0 if naive else 6
            )
            self.assertReferenceChecks(gc, op, [data, lambda1, lambda2], ref)


if __name__ == "__main__":
    import unittest
    unittest.main()
