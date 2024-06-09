from typing import List

import hypothesis.strategies as st

from caffe2.python import core, workspace
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu

import bisect
import numpy as np


class TestBisectPercentileOp(hu.HypothesisTestCase):
    def compare_reference(
            self,
            raw_data,
            pct_raw_data,
            pct_mapping,
            pct_upper,
            pct_lower,
            lengths,
    ):
        def bisect_percentile_op_ref(
            raw_data,
            pct_raw_data,
            pct_mapping,
            pct_lower,
            pct_upper,
            lengths
        ):
            results = np.zeros_like(raw_data)
            indices = [0]
            for j in range(len(lengths)):
                indices.append(indices[j] + lengths[j])
            for i in range(len(raw_data)):
                for j in range(len(raw_data[0])):
                    start = indices[j]
                    end = indices[j + 1]
                    val = raw_data[i][j]
                    pct_raw_data_i = pct_raw_data[start:end]
                    pct_lower_i = pct_lower[start:end]
                    pct_upper_i = pct_upper[start:end]
                    pct_mapping_i = pct_mapping[start:end]

                    # Corner cases
                    if val < pct_raw_data_i[0]:
                        results[i][j] = 0
                        continue
                    if val > pct_raw_data_i[-1]:
                        results[i][j] = 1.
                        continue

                    # interpolation
                    k = bisect.bisect_left(pct_raw_data_i, val)
                    if pct_raw_data_i[k] == val:
                        results[i][j] = pct_mapping_i[k]
                    else:
                        k = k - 1
                        slope = ((pct_lower_i[k + 1] - pct_upper_i[k])
                            / (pct_raw_data_i[k + 1] - pct_raw_data_i[k]))
                        results[i][j] = pct_upper_i[k] + \
                            slope * (val - pct_raw_data_i[k])

            return results

        workspace.ResetWorkspace()
        workspace.FeedBlob("raw_data", raw_data)

        op = core.CreateOperator(
            "BisectPercentile",
            ["raw_data"],
            ["pct_output"],
            percentile_raw=pct_raw_data,
            percentile_mapping=pct_mapping,
            percentile_lower=pct_lower,
            percentile_upper=pct_upper,
            lengths=lengths
        )
        workspace.RunOperatorOnce(op)

        expected_output = bisect_percentile_op_ref(
            raw_data,
            pct_raw_data,
            pct_mapping,
            pct_lower,
            pct_upper,
            lengths
        )
        output = workspace.blobs['pct_output']
        np.testing.assert_array_almost_equal(output, expected_output)

    def test_bisect_percentil_op_simple(self):
        raw_data = np.array([
            [1, 1],
            [2, 2],
            [3, 3],
            [3, 1],
            [9, 10],
            [1.5, 5],
            [1.32, 2.4],
            [2.9, 5.7],
            [-1, -1],
            [3, 7]
        ], dtype=np.float32)
        pct_raw_data = np.array([1, 2, 3, 2, 7], dtype=np.float32)
        pct_lower = np.array([0.1, 0.2, 0.9, 0.1, 0.5], dtype=np.float32)
        pct_upper = np.array([0.1, 0.8, 1.0, 0.4, 1.0], dtype=np.float32)
        pct_mapping = np.array([0.1, 0.5, 0.95, 0.25, 0.75], dtype=np.float32)
        lengths = np.array([3, 2], dtype=np.int32)
        self.compare_reference(
            raw_data, pct_raw_data, pct_mapping, pct_lower, pct_upper, lengths)

    @given(
        N=st.integers(min_value=20, max_value=100),
        lengths_in=st.lists(
            elements=st.integers(min_value=2, max_value=10),
            min_size=2,
            max_size=5,
        ),
        max_value=st.integers(min_value=100, max_value=1000),
        discrete=st.booleans(),
        p=st.floats(min_value=0, max_value=0.9),
        **hu.gcs_cpu_only
    )
    def test_bisect_percentil_op_large(
        self, N: int, lengths_in: List[int], max_value: int, discrete: bool, p: float, gc, dc
    ):
        lengths = np.array(lengths_in, dtype=np.int32)
        D = len(lengths)

        if discrete:
            raw_data = np.random.randint(0, max_value, size=(N, D))
        else:
            raw_data = np.random.randn(N, D)

        # To generate valid pct_lower and pct_upper
        pct_lower = []
        pct_upper = []
        pct_raw_data = []
        for i in range(D):
            pct_lower_val = 0.
            pct_upper_val = 0.
            pct_lower_cur = []
            pct_upper_cur = []
            # There is no duplicated values in pct_raw_data
            if discrete:
                pct_raw_data_cur = np.random.choice(
                    np.arange(max_value), size=lengths[i], replace=False)
            else:
                pct_raw_data_cur = np.random.randn(lengths[i])
                while len(set(pct_raw_data_cur)) < lengths[i]:
                    pct_raw_data_cur = np.random.randn(lengths[i])
            pct_raw_data_cur = np.sort(pct_raw_data_cur)
            for _ in range(lengths[i]):
                pct_lower_val = pct_upper_val + 0.01
                pct_lower_cur.append(pct_lower_val)
                pct_upper_val = pct_lower_val + \
                    0.01 * np.random.randint(1, 20) * (np.random.uniform() < p)
                pct_upper_cur.append(pct_upper_val)
            # normalization
            pct_lower_cur = np.array(pct_lower_cur, np.float32) / pct_upper_val
            pct_upper_cur = np.array(pct_upper_cur, np.float32) / pct_upper_val
            pct_lower.extend(pct_lower_cur)
            pct_upper.extend(pct_upper_cur)
            pct_raw_data.extend(pct_raw_data_cur)

        pct_lower = np.array(pct_lower, dtype=np.float32)
        pct_upper = np.array(pct_upper, dtype=np.float32)
        pct_mapping = (pct_lower + pct_upper) / 2.
        raw_data = np.array(raw_data, dtype=np.float32)
        pct_raw_data = np.array(pct_raw_data, dtype=np.float32)

        self.compare_reference(
            raw_data, pct_raw_data, pct_mapping, pct_lower, pct_upper, lengths)


if __name__ == "__main__":
    import unittest
    unittest.main()
