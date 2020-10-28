import unittest

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, workspace
from hypothesis import given, settings


class TestSelfBinningHistogramBase(object):
    def __init__(self, bin_spacing, dtype):
        self.bin_spacing = bin_spacing
        self.dtype = dtype

    def _check_histogram(self, arrays, num_bins, expected_values=None, expected_counts=None):
        # Check that sizes match and counts add up.
        values = workspace.FetchBlob("histogram_values")
        counts = workspace.FetchBlob("histogram_counts")
        self.assertTrue(np.size(values) == num_bins)
        self.assertTrue(np.size(counts) == num_bins)
        self.assertTrue(np.sum(counts) == sum([np.size(array) for array in arrays]))


        if expected_counts is None:
            # Check that counts are correct for the returned values if expected_counts is not given.
            expected_counts = np.zeros(num_bins, dtype='i')
            for array in arrays:
                for i in array:
                    found = False
                    for pos in range(np.size(values)):
                        if values[pos] > i:
                            found = True
                            break
                    self.assertTrue(found, "input array must fit inside values array")
                    if self.bin_spacing == "linear":
                        self.assertTrue(pos > 0, "first value should be the smallest")
                    if pos == 0:
                        self.assertEqual(self.bin_spacing, "logarithmic")
                        expected_counts[pos] += 1
                    else:
                        expected_counts[pos - 1] += 1
        self.assertTrue(np.array_equal(expected_counts, counts), f"expected:{expected_counts}\ncounts:{counts}")
        if expected_values is not None:
            self.assertTrue(np.array_equal(expected_values, values), f"expected:{expected_values}\ncounts:{values}")


    def _run_single_op_net(self, arrays, num_bins, logspacing_start=None):
        for i in range(len(arrays)):
            workspace.FeedBlob(
                "X{}".format(i), arrays[i]
            )
        net = core.Net("test_net")
        if logspacing_start is not None:
            net.SelfBinningHistogram(
                ["X{}".format(i) for i in range(len(arrays))],
                ["histogram_values", "histogram_counts"],
                num_bins=num_bins,
                bin_spacing=self.bin_spacing,
                logspacing_start=logspacing_start,
            )
        else:
            net.SelfBinningHistogram(
                ["X{}".format(i) for i in range(len(arrays))],
                ["histogram_values", "histogram_counts"],
                num_bins=num_bins,
                bin_spacing=self.bin_spacing,
            )
        workspace.RunNetOnce(net)

    @given(rows=st.integers(1, 1000), cols=st.integers(1, 1000), **hu.gcs_cpu_only)
    @settings(deadline=10000)
    def test_histogram_device_consistency(self, rows, cols, gc, dc):
        X = np.random.rand(rows, cols)
        op = core.CreateOperator(
            "SelfBinningHistogram",
            ["X"],
            ["histogram_values", "histogram_counts"],
            num_bins=1000,
            bin_spacing=self.bin_spacing,
        )
        self.assertDeviceChecks(dc, op, [X], [0])

    def test_histogram_bin_to_fewer(self):
        X = np.array([-2.0, -2.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0], dtype=self.dtype)
        self._run_single_op_net([X], 5)
        self._check_histogram(
            [X],
            6,
        )

    def test_histogram_bin_to_more(self):
        X = np.array([-2.0, -2.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0], dtype=self.dtype)
        self._run_single_op_net([X], 100)
        self._check_histogram(
            [X],
            101,
        )

    def test_histogram_bin_to_two(self):
        """This test roughly tests [min,max+EPSILON] and [N,0]"""
        X = np.array([-2.0, -2.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0], dtype=self.dtype)
        self._run_single_op_net([X], 1)
        self._check_histogram(
            [X],
            2,
        )

    def test_histogram_min_max_equal(self):
        """This test uses exact value match, so is only relevant for float type."""
        X = np.array([0., 0., 0., 0., 0.], dtype='f')
        logspacing_start = np.float(1e-24)
        self._run_single_op_net([X], 3, logspacing_start)
        if self.bin_spacing == "linear":
            self._check_histogram(
                [X],
                4,
                expected_values=np.array([0., 0., 0., 0.], dtype='f'),
                expected_counts=[5, 0, 0, 0]
            )
        else:
            self.assertEqual(self.bin_spacing, "logarithmic")
            self._check_histogram(
                [X],
                4,
                expected_values=np.array([logspacing_start] * 4, dtype='f'),
                expected_counts=[5, 0, 0, 0],
            )

    def test_histogram_min_max_equal_nonzero(self):
        X = np.array([1., 1., 1., 1., 1.], dtype=self.dtype)
        logspacing_start = 1e-24
        self._run_single_op_net([X], 3, 1e-24)
        self._check_histogram(
            [X],
            4,
            expected_values=[1., 1., 1., 1.],
            expected_counts=[5, 0, 0, 0]
        )

    def test_histogram_empty_input_tensor(self):
        X = np.array([], dtype=self.dtype)
        self._run_single_op_net([X], 1)
        self._check_histogram(
            [X],
            2,
        )
        self._run_single_op_net([X], 10)
        self._check_histogram(
            [X],
            11,
        )

    def test_histogram_multi_input(self):
        X1 = np.array([-2.0, -2.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0], dtype=self.dtype)
        X2 = np.array([-5.0, -3.0, 7, 7, 0.0, 1.0, 2.0, -3.0, 4.0, 6.0, 9.0], dtype=self.dtype)
        self._run_single_op_net([X1, X2], 5)
        self._check_histogram(
            [X1, X2],
            6,
        )

    def test_histogram_very_small_range_for_stride_underflow(self):
        """Tests a large number of bins for a very small range of values.

        This test uses float type. 1-e38 is very small, and with 1M bins, it
        causes numeric underflow. This test is to show that this is handled.
        """
        X = np.array([0, 1e-38], dtype='f')
        self._run_single_op_net([X], 1000000)
        self._check_histogram(
            [X],
            1000001,
        )


    def test_histogram_insufficient_bins(self):
        with self.assertRaisesRegex(
            RuntimeError, "Number of bins must be greater than or equal to 1."
        ):
            self._run_single_op_net([np.random.rand(111)], 0)


class TestSelfBinningHistogramLinear(TestSelfBinningHistogramBase, hu.HypothesisTestCase):
    def __init__(self, *args, **kwargs):
        TestSelfBinningHistogramBase.__init__(self, bin_spacing="linear", dtype='d')
        hu.HypothesisTestCase.__init__(self, *args, **kwargs)

class TestSelfBinningHistogramLogarithmic(TestSelfBinningHistogramBase, hu.HypothesisTestCase):
    def __init__(self, *args, **kwargs):
        TestSelfBinningHistogramBase.__init__(self, bin_spacing="logarithmic", dtype='d')
        hu.HypothesisTestCase.__init__(self, *args, **kwargs)

class TestSelfBinningHistogramLinearFloat(TestSelfBinningHistogramBase, hu.HypothesisTestCase):
    def __init__(self, *args, **kwargs):
        TestSelfBinningHistogramBase.__init__(self, bin_spacing="linear", dtype='f')
        hu.HypothesisTestCase.__init__(self, *args, **kwargs)

class TestSelfBinningHistogramLogarithmicFloat(TestSelfBinningHistogramBase, hu.HypothesisTestCase):
    def __init__(self, *args, **kwargs):
        TestSelfBinningHistogramBase.__init__(self, bin_spacing="logarithmic", dtype='f')
        hu.HypothesisTestCase.__init__(self, *args, **kwargs)


if __name__ == "__main__":
    global_options = ["caffe2"]
    core.GlobalInit(global_options)
    unittest.main()
