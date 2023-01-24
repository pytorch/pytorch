




import numpy as np

from hypothesis import given
import hypothesis.strategies as st

from caffe2.python import core
from caffe2.python import workspace
import caffe2.python.hypothesis_test_util as hu


class TestWeightedMultiSample(hu.HypothesisTestCase):
    @given(
        num_samples=st.integers(min_value=0, max_value=128),
        data_len=st.integers(min_value=0, max_value=10000),
        **hu.gcs_cpu_only
    )
    def test_weighted_multi_sample(self, num_samples, data_len, gc, dc):
        weights = np.zeros((data_len))
        expected_indices = []
        if data_len > 0:
            weights[-1] = 1.5
            expected_indices = np.repeat(data_len - 1, num_samples)

        workspace.FeedBlob("weights", weights.astype(np.float32))

        op = core.CreateOperator(
            "WeightedMultiSampling",
            ["weights"],
            ["sample_indices"],
            num_samples=num_samples,
        )
        workspace.RunOperatorOnce(op)
        result_indices = workspace.FetchBlob("sample_indices")
        np.testing.assert_allclose(expected_indices, result_indices)
        self.assertDeviceChecks(
            dc,
            op,
            [weights.astype(np.float32)],
            [0]
        )

        # test shape input
        shape = np.zeros((num_samples))
        workspace.FeedBlob("shape", shape)
        op2 = core.CreateOperator(
            "WeightedMultiSampling",
            ["weights", "shape"],
            ["sample_indices_2"]
        )
        workspace.RunOperatorOnce(op2)
        result_indices_2 = workspace.FetchBlob("sample_indices_2")
        if data_len > 0:
            assert len(result_indices_2) == num_samples
            for i in range(num_samples):
                assert 0 <= result_indices_2[i] < data_len
        else:
            assert len(result_indices_2) == 0

        self.assertDeviceChecks(dc, op2, [weights.astype(np.float32), shape], [0])


if __name__ == "__main__":
    import unittest
    unittest.main()
