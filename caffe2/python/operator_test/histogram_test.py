import unittest

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, workspace
from hypothesis import given, settings


class TestHistogram(hu.HypothesisTestCase):
    @given(rows=st.integers(1, 1000), cols=st.integers(1, 1000), **hu.gcs_cpu_only)
    @settings(deadline=10000)
    def test_histogram__device_consistency(self, rows, cols, gc, dc):
        X = np.random.rand(rows, cols)
        bin_edges = list(np.linspace(-2, 10, num=10000))
        op = core.CreateOperator("Histogram", ["X"], ["histogram"], bin_edges=bin_edges)
        self.assertDeviceChecks(dc, op, [X], [0])

    def test_histogram__valid_inputs_0(self):
        workspace.FeedBlob(
            "X", np.array([-2.0, -2.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0])
        )
        bin_edges = [-2.0, -1.0, 0.0, 2.0, 5.0, 9.0]

        net = core.Net("test_net")
        net.Histogram(["X"], ["histogram"], bin_edges=bin_edges)

        workspace.RunNetOnce(net)
        histogram_blob = workspace.FetchBlob("histogram")

        assert list(histogram_blob) == [2, 0, 4, 3, 1]

    @given(num_tensors=st.integers(1, 5), num_bin_edges=st.integers(2, 10000))
    @settings(deadline=10000)
    def test_histogram__valid_inputs_1(self, num_tensors, num_bin_edges):
        self._test_histogram(
            [
                np.random.rand(np.random.randint(1, 1000), np.random.randint(1, 1000))
                for __ in range(num_tensors)
            ],
            list(np.logspace(-12, 5, num=num_bin_edges)),
        )

    def test_histogram__empty_input_tensor(self):
        self._test_histogram([np.array([])], list(np.linspace(-2, 2, num=10)))

    def test_histogram__non_increasing_bin_edges(self):
        with self.assertRaisesRegex(
            RuntimeError, "bin_edges must be a strictly increasing sequence of values"
        ):
            self._test_histogram(
                [np.random.rand(100), np.random.rand(98)], [0.0, 0.2, 0.1, 0.1]
            )

    def test_histogram__insufficient_bin_edges(self):
        with self.assertRaisesRegex(
            RuntimeError, "Number of bin edges must be greater than or equal to 2"
        ):
            self._test_histogram([np.random.rand(111)], [1.0])

    def _test_histogram(self, tensors, bin_edges):
        total_size = 0
        input_blob_names = []

        for idx, tensor in enumerate(tensors):
            total_size += np.size(tensor)
            tensor_blob_name = f"X{idx}"
            workspace.FeedBlob(tensor_blob_name, tensor)
            input_blob_names.append(tensor_blob_name)

        output_name = "histogram"
        net = core.Net("test_net")
        net.Histogram(input_blob_names, [output_name], bin_edges=bin_edges)

        workspace.RunNetOnce(net)
        histogram_blob = workspace.FetchBlob(output_name)

        assert np.size(histogram_blob) == len(bin_edges) - 1
        assert np.sum(histogram_blob) == total_size


if __name__ == "__main__":
    global_options = ["caffe2"]
    core.GlobalInit(global_options)
    unittest.main()
