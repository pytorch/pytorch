from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np


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
            elements=st.floats(min_value=-10, max_value=10),
        )),
        draw(st.lists(
            elements=st.floats(min_value=-2, max_value=2),
            min_size=D,
            max_size=D,
        )),
        draw(st.lists(
            elements=st.floats(min_value=-2, max_value=2),
            min_size=D,
            max_size=D,
        )),
    )


class TestBatchBoxCox(hu.HypothesisTestCase):
    @given(
        inputs=_inputs(),
        **hu.gcs_cpu_only
    )
    def test_batch_box_cox(self, inputs, gc, dc):
        N, D, data, lambda1, lambda2 = inputs

        data = np.array(data, dtype=np.float32).reshape(N, D)
        lambda1 = np.array(lambda1, dtype=np.float32)
        lambda2 = np.array(lambda2, dtype=np.float32)

        def ref(data, lambda1, lambda2):
            dim_1 = data.shape[1]
            output = np.copy(data)
            if data.size <= 0:
                return [output]

            for i in range(dim_1):
                output[:, i] = data[:, i] + lambda2[i]
                output[:, i] = np.maximum(output[:, i], 1e-6)
                if lambda1[i] == 0:
                    output[:, i] = np.log(output[:, i])
                else:
                    output[:, i] =\
                        (np.power(output[:, i], lambda1[i]) - 1) / lambda1[i]
            return [output]

        op = core.CreateOperator(
            'BatchBoxCox',
            ['data', 'lambda1', 'lambda2'],
            ['output']
        )

        self.assertReferenceChecks(gc, op, [data, lambda1, lambda2], ref)


if __name__ == "__main__":
    import unittest
    unittest.main()
