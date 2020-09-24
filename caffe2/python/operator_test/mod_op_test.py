




import numpy

from caffe2.python import core
from hypothesis import given

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np


@st.composite
def _data(draw):
    return draw(
        hu.tensor(dtype=np.int64,
            elements=st.integers(
                min_value=np.iinfo(np.int64).min, max_value=np.iinfo(np.int64).max
            )
        )
    )


class TestMod(hu.HypothesisTestCase):
    @given(
        data=_data(),
        divisor=st.integers(
            min_value=np.iinfo(np.int64).min, max_value=np.iinfo(np.int64).max
        ),
        inplace=st.booleans(),
        sign_follow_divisor=st.booleans(),
        **hu.gcs_cpu_only
    )
    def test_mod(
        self, data, divisor, inplace, sign_follow_divisor, gc, dc
    ):
        if divisor == 0:
            # invalid test case
            return None

        def ref(data):
            if sign_follow_divisor:
                output = data % divisor
            else:
                output = numpy.fmod(data, divisor)
            return [output]

        op = core.CreateOperator(
            'Mod',
            ['data'],
            ['data' if inplace else 'output'],
            divisor=divisor,
            sign_follow_divisor=sign_follow_divisor
        )

        self.assertReferenceChecks(gc, op, [data], ref)


if __name__ == "__main__":
    import unittest
    unittest.main()
