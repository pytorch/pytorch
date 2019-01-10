from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from functools import partial
from hypothesis import given
from hypothesis import strategies as st

import caffe2.python.hypothesis_test_util as hu
import math
import numpy as np


def _data_and_scale(
        data_min_size=4, data_max_size=10,
        examples_min_number=1, examples_max_number=4,
        dtype=np.float32, elements=None):
    params_ = st.tuples(
        st.integers(min_value=examples_min_number,
                    max_value=examples_max_number),
        st.integers(min_value=data_min_size,
                    max_value=data_max_size),
        st.sampled_from([np.float32, np.int32, np.int64])
    )
    return params_.flatmap(
        lambda param_: st.tuples(
            hu.arrays([param_[0], param_[1]], dtype=dtype),
            hu.arrays(
                [param_[0]], dtype=param_[2],
                elements=(st.floats(0.0, 10000.0) if param_[2] in [np.float32]
                          else st.integers(0, 10000)),
            ),
        )
    )


def divide_by_square_root(data, scale):
    output = np.copy(data)
    num_examples = len(scale)

    assert num_examples == data.shape[0]
    assert len(data.shape) == 2

    for i in range(0, num_examples):
        if scale[i] > 0:
            output[i] = np.multiply(data[i], 1 / math.sqrt(scale[i]))

    return (output, )


def grad(output_grad, ref_outputs, inputs):
    return (divide_by_square_root(output_grad, inputs[1])[0],
            None)


class TestSquareRootDivide(hu.HypothesisTestCase):
    @given(data_and_scale=_data_and_scale(),
           **hu.gcs_cpu_only)
    def test_square_root_divide(self, data_and_scale, gc, dc):
        self.assertReferenceChecks(
            device_option=gc,
            op=core.CreateOperator("SquareRootDivide",
                                   ["data", "scale"],
                                   ["output"]),
            inputs=list(data_and_scale),
            reference=partial(divide_by_square_root),
            output_to_grad="output",
            grad_reference=grad,
        )


if __name__ == "__main__":
    import unittest
    unittest.main()
