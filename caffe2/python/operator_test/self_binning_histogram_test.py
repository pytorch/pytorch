import unittest

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, workspace
from hypothesis import given, settings


class TestSelfBinningHistogramBase(object):
    def __init__(self, bin_spacing, dtype, abs=False):
        self.bin_spacing = bin_spacing
        self.dtype = dtype
        self.abs = abs

    def _check_histogram(self, arrays, num_bins, expected_values=None, expected_counts=None):
        # Check that sizes match and counts add up.
        values = workspace.FetchBlob("histogram_values")
        counts = workspace.FetchBlob("histogram_counts")
        self.assertTrue(np.size(values) == num_bins)
        self.assertTrue(np.size(counts) == num_bins)
        self.assertTrue(np.sum(counts) == sum([np.size(array) for array in arrays]))

        # Check counts
        if expected_counts is None:
            # Check that counts are correct for the returned values if expected_counts is not given.
            expected_counts = np.zeros(num_bins, dtype='i')
            for array in arrays:
                for input_val in array:
                    input_val = abs(input_val) if self.abs else input_val
                    found = False
                    for pos in range(np.size(values)):
                        if values[pos] > input_val:
                            found = True
                            break
                    self.assertTrue(found, f"input value must fit inside values array: "
                                           f"input={input_val}, last_value={values[-1]}")
                    if self.bin_spacing == "linear":
                        self.assertTrue(pos > 0,
                                        f"input should not be smaller than the first bin value: "
                                        f"input={input_val}, 1st bin value={values[pos]}")
                    if pos == 0:
                        self.assertEqual(self.bin_spacing, "logarithmic")
                        expected_counts[pos] += 1
                    else:
                        expected_counts[pos - 1] += 1
        self.assertTrue(np.array_equal(expected_counts, counts), f"expected:{expected_counts}\ncounts:{counts}")
        # Check values
        if expected_values is not None:
            self.assertTrue(np.allclose(expected_values, values, rtol=1e-02, atol=1e-05),
                            f"expected:{expected_values}\nvalues:{values}")
        # Ideally, the output values are sorted in a non-decreasing order.
        for idx in range(len(values) - 1):
            self.assertTrue(values[idx] <= values[idx + 1])
        if self.abs:
            self.assertTrue(values[0] >= 0)


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
                abs=self.abs
            )
        else:
            net.SelfBinningHistogram(
                ["X{}".format(i) for i in range(len(arrays))],
                ["histogram_values", "histogram_counts"],
                num_bins=num_bins,
                bin_spacing=self.bin_spacing,
                abs=self.abs
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
        if self.bin_spacing == 'linear':
            if not self.abs:
                expected_values = [-2., 0.2, 2.4, 4.6, 6.8, 9.]
                expected_counts = [5, 2, 2, 1, 1, 0]
            else:
                expected_values = [0., 1.8, 3.6, 5.4, 7.2, 9.]
                expected_counts = [4, 4, 1, 1, 1, 0]
        else:
            expected_values = [1.e-24, 9.8e-20, 9.6e-15, 9.4e-10, 9.2e-05, 9.]
            if not self.abs:
                expected_counts = [5, 0, 0, 0, 6, 0]
            else:
                expected_counts = [3, 0, 0, 0, 8, 0]
        self._run_single_op_net([X], 5)
        self._check_histogram(
            [X],
            6,
            expected_values=expected_values,
            expected_counts=expected_counts
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
        if self.bin_spacing == 'linear':
            if not self.abs:
                expected_values = [-2., 9.]
            else:
                expected_values = [0., 9.]
        else:
            expected_values = [1.e-24, 9.]
        expected_counts = [11, 0]
        self._run_single_op_net([X], 1)
        self._check_histogram(
            [X],
            2,
            expected_values=expected_values,
            expected_counts=expected_counts
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
        self._run_single_op_net([X], 3, logspacing_start)
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
            expected_values=[0., 0.],
            expected_counts=[0, 0]
        )
        self._run_single_op_net([X], 10)
        self._check_histogram(
            [X],
            11,
            expected_values=[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            expected_counts=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )

    def test_histogram_multi_input(self):
        X1 = np.array([-2.0, -2.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0], dtype=self.dtype)
        X2 = np.array([-5.0, -3.0, 7, 7, 0.0, 1.0, 2.0, -3.0, 4.0, 6.0, 9.0], dtype=self.dtype)
        if self.bin_spacing == 'linear':
            if not self.abs:
                expected_values = [-5., -2.2, 0.6, 3.4, 6.2, 9.]
                expected_counts = [3, 6, 5, 4, 4, 0]
            else:
                expected_values = [0., 1.8, 3.6, 5.4, 7.2, 9.]
                expected_counts = [6, 7, 3, 4, 2, 0]
        else:
            expected_values = [1.e-24, 9.8e-20, 9.6e-15, 9.4e-10, 9.2e-05, 9.]
            if not self.abs:
                expected_counts = [9, 0, 0, 0, 13, 0]
            else:
                expected_counts = [4, 0, 0, 0, 18, 0]
        self._run_single_op_net([X1, X2], 5)
        self._check_histogram(
            [X1, X2],
            6,
            expected_values=expected_values,
            expected_counts=expected_counts
        )

    def test_histogram_very_small_range_for_stride_underflow(self):
        """Tests a large number of bins for a very small range of values.

        This test uses float type. 1-e302 is very small, and with 1M bins, it
        causes numeric underflow. This test is to show that this is handled.

        Note: this test was flaky due to how compiler and OS handls floats.
        Previously, 1-e38 does not induce overflow and cuases test error for some
        combinations of compiler and OS. Now 1-e302 should be small enough.
        """
        X = np.array([0, 1e-302], dtype='f')
        large_bin_number = 1000000
        self._run_single_op_net([X], large_bin_number)
        self._check_histogram(
            [X],
            large_bin_number + 1,
            expected_counts=[2] + [0] * large_bin_number  # [2, 0, 0, ..., 0]
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

class TestSelfBinningHistogramLinearWithAbs(TestSelfBinningHistogramBase, hu.HypothesisTestCase):
    def __init__(self, *args, **kwargs):
        TestSelfBinningHistogramBase.__init__(self, bin_spacing="linear", dtype='d', abs=True)
        hu.HypothesisTestCase.__init__(self, *args, **kwargs)

class TestSelfBinningHistogramLogarithmicWithAbs(TestSelfBinningHistogramBase, hu.HypothesisTestCase):
    def __init__(self, *args, **kwargs):
        TestSelfBinningHistogramBase.__init__(self, bin_spacing="logarithmic", dtype='d', abs=True)
        hu.HypothesisTestCase.__init__(self, *args, **kwargs)

class TestSelfBinningHistogramLinearFloatWithAbs(TestSelfBinningHistogramBase, hu.HypothesisTestCase):
    def __init__(self, *args, **kwargs):
        TestSelfBinningHistogramBase.__init__(self, bin_spacing="linear", dtype='f', abs=True)
        hu.HypothesisTestCase.__init__(self, *args, **kwargs)

class TestSelfBinningHistogramLogarithmicFloatWithAbs(TestSelfBinningHistogramBase, hu.HypothesisTestCase):
    def __init__(self, *args, **kwargs):
        TestSelfBinningHistogramBase.__init__(self, bin_spacing="logarithmic", dtype='f', abs=True)
        hu.HypothesisTestCase.__init__(self, *args, **kwargs)

class TestSelfBinningHistogramLinearWithNoneAbs(TestSelfBinningHistogramBase, hu.HypothesisTestCase):
    def __init__(self, *args, **kwargs):
        TestSelfBinningHistogramBase.__init__(self, bin_spacing="linear", dtype='d', abs=None)
        hu.HypothesisTestCase.__init__(self, *args, **kwargs)

class TestSelfBinningHistogramLinearFloatWithNoneAbs(TestSelfBinningHistogramBase, hu.HypothesisTestCase):
    def __init__(self, *args, **kwargs):
        TestSelfBinningHistogramBase.__init__(self, bin_spacing="linear", dtype='f', abs=None)
        hu.HypothesisTestCase.__init__(self, *args, **kwargs)

if __name__ == "__main__":
    global_options = ["caffe2"]
    core.GlobalInit(global_options)
    unittest.main()
