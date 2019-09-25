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
        weights_len=st.integers(min_value=0, max_value=128),
        **hu.gcs
    )
    def test_weighted_sample(self, batch, weights_len, gc, dc):

        weights = np.zeros((batch, weights_len))
        values = np.zeros((batch, weights_len))
        rand_indices = []
        rand_values = []
        if batch > 0 and weights_len > 0:
            for i in range(batch):
                rand_tmp = np.random.randint(0, weights_len)
                rand_val = np.random.rand()
                rand_indices.append(rand_tmp)
                rand_values.append(rand_val)
                weights[i, rand_tmp] = 1.0
                values[i, rand_tmp] = rand_val

        rand_indices = np.array(rand_indices, dtype=np.float32)
        rand_values = np.array(rand_values, dtype=np.float32)
        workspace.FeedBlob("weights", weights.astype(np.float32))
        workspace.FeedBlob("values", values.astype(np.float32))

        # output both indices and values
        op = core.CreateOperator(
            "WeightedSample", ["weights", "values"],
            ["sample_indices", "sample_values"]
        )
        workspace.RunOperatorOnce(op)
        result_indices = workspace.FetchBlob("sample_indices")
        result_values = workspace.FetchBlob("sample_values")
        if batch > 0 and weights_len > 0:
            for i in range(batch):
                np.testing.assert_allclose(rand_indices[i], result_indices[i])
                np.testing.assert_allclose(rand_values[i], result_values[i])
        else:
            np.testing.assert_allclose(rand_indices, result_indices)
            np.testing.assert_allclose(rand_values, result_values)
        self.assertDeviceChecks(
            dc,
            op,
            [weights.astype(np.float32), values.astype(np.float32)],
            [0, 1]
        )

        # output indices only
        op2 = core.CreateOperator(
            "WeightedSample", ["weights"], ["sample_indices_2"]
        )
        workspace.RunOperatorOnce(op2)
        result = workspace.FetchBlob("sample_indices_2")
        if batch > 0 and weights_len > 0:
            for i in range(batch):
                np.testing.assert_allclose(rand_indices[i], result[i])
        else:
            np.testing.assert_allclose(rand_indices, result)
        self.assertDeviceChecks(dc, op2, [weights.astype(np.float32)], [0])


if __name__ == "__main__":
    import unittest
    unittest.main()
