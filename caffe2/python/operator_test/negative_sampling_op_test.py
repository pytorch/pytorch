from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np


class TestNegativeSampling(hu.HypothesisTestCase):
    @given(
        input=hu.tensor(min_dim=2, max_dim=2, max_value=50),
        data_strategy=st.data(),
        categorical_limit=st.integers(min_value=100000, max_value=200000),
        num_negatives=st.integers(min_value=1, max_value=10),
        **hu.gcs_cpu_only
    )
    def test_uniform_negative_sampling_ops(self, input, data_strategy, categorical_limit,
                                   num_negatives, gc, dc):
        m = input.shape[0]
        lengths = data_strategy.draw(
            hu.tensor(max_dim=1, max_value=input.shape[0], dtype=np.int32,
                      elements=st.integers(min_value=0, max_value=27)))
        lengths_sum = np.sum(lengths)

        indices = data_strategy.draw(
            hu.arrays([lengths_sum], dtype=np.int64,
                      elements=st.sampled_from(np.arange(m))))
        op = core.CreateOperator("UniformNegativeSamplingOp",
                                 ["indices", "lengths"],
                                 ["output_indices", "output_lengths", "output_labels"],
                                categorical_limit=categorical_limit,
                                num_negatives=num_negatives)

        self.assertDeviceChecks(dc, op, [indices, lengths], [0, 1, 2])
