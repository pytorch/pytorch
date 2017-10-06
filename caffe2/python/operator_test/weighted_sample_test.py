from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from hypothesis import given
import hypothesis.strategies as st

from caffe2.python import core
from caffe2.python import workspace
import caffe2.python.hypothesis_test_util as hu


class TestWeightedSample(hu.HypothesisTestCase):
    @given(
        batch=st.integers(min_value=0, max_value=128),
        weights_len=st.integers(min_value=0, max_value=128)
    )
    def test_weighted_sample(self, batch, weights_len):
        op = core.CreateOperator("WeightedSample", ["weights"], ["indices"])

        weights = np.zeros((batch, weights_len))
        rand_indices = []
        if batch > 0 and weights_len > 0:
            for i in range(batch):
                rand_tmp = np.random.randint(0, weights_len)
                rand_indices.append(rand_tmp)
                weights[i, rand_tmp] = 1.0

        rand_indices = np.array(rand_indices, dtype=np.float32)
        workspace.FeedBlob("weights", weights.astype(np.float32))
        workspace.RunOperatorOnce(op)
        result = workspace.FetchBlob("indices")

        if batch > 0 and weights_len > 0:
            for i in range(batch):
                np.testing.assert_allclose(rand_indices[i], result[i])
        else:
            np.testing.assert_allclose(rand_indices, result)


if __name__ == "__main__":
    import unittest
    unittest.main()
