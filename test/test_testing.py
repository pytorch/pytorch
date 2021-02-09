import torch

import math
from pathlib import PurePosixPath

from torch.testing._internal.common_utils import \
    (TestCase, make_tensor, run_tests, slowTest)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, onlyCUDA, onlyOnCPUAndCUDA, dtypes)
from torch.testing._internal import mypy_wrapper
from torch.testing._internal import print_test_stats

# For testing TestCase methods and torch.testing functions
class TestTesting(TestCase):
    # Ensure that assertEqual handles numpy arrays properly
    @dtypes(*(torch.testing.get_all_dtypes(include_half=True, include_bfloat16=False,
                                           include_bool=True, include_complex=True)))
    def test_assertEqual_numpy(self, device, dtype):
        S = 10
        test_sizes = [
            (),
            (0,),
            (S,),
            (S, S),
            (0, S),
            (S, 0)]
        for test_size in test_sizes:
            a = make_tensor(test_size, device, dtype, low=-5, high=5)
            a_n = a.cpu().numpy()
            msg = f'size: {test_size}'
            self.assertEqual(a_n, a, rtol=0, atol=0, msg=msg)
            self.assertEqual(a, a_n, rtol=0, atol=0, msg=msg)
            self.assertEqual(a_n, a_n, rtol=0, atol=0, msg=msg)

    # Tests that when rtol or atol (including self.precision) is set, then
    # the other is zeroed.
    # TODO: this is legacy behavior and should be updated after test
    # precisions are reviewed to be consistent with torch.isclose.
    @onlyOnCPUAndCUDA
    def test__comparetensors_legacy(self, device):
        a = torch.tensor((10000000.,))
        b = torch.tensor((10000002.,))

        x = torch.tensor((1.,))
        y = torch.tensor((1. + 1e-5,))

        # Helper for reusing the tensor values as scalars
        def _scalar_helper(a, b, rtol=None, atol=None):
            return self._compareScalars(a.item(), b.item(), rtol=rtol, atol=atol)

        for op in (self._compareTensors, _scalar_helper):
            # Tests default
            result, debug_msg = op(a, b)
            self.assertTrue(result)

            # Tests setting atol
            result, debug_msg = op(a, b, atol=2, rtol=0)
            self.assertTrue(result)

            # Tests setting atol too small
            result, debug_msg = op(a, b, atol=1, rtol=0)
            self.assertFalse(result)

            # Tests setting rtol too small
            result, debug_msg = op(x, y, atol=0, rtol=1.05e-5)
            self.assertTrue(result)

            # Tests setting rtol too small
            result, debug_msg = op(x, y, atol=0, rtol=1e-5)
            self.assertFalse(result)

    @onlyOnCPUAndCUDA
    def test__comparescalars_debug_msg(self, device):
        # float x float
        result, debug_msg = self._compareScalars(4., 7.)
        expected_msg = ("Comparing 4.0 and 7.0 gives a difference of 3.0, "
                        "but the allowed difference with rtol=1.3e-06 and "
                        "atol=1e-05 is only 1.9100000000000003e-05!")
        self.assertEqual(debug_msg, expected_msg)

        # complex x complex, real difference
        result, debug_msg = self._compareScalars(complex(1, 3), complex(3, 1))
        expected_msg = ("Comparing the real part 1.0 and 3.0 gives a difference "
                        "of 2.0, but the allowed difference with rtol=1.3e-06 "
                        "and atol=1e-05 is only 1.39e-05!")
        self.assertEqual(debug_msg, expected_msg)

        # complex x complex, imaginary difference
        result, debug_msg = self._compareScalars(complex(1, 3), complex(1, 5.5))
        expected_msg = ("Comparing the imaginary part 3.0 and 5.5 gives a "
                        "difference of 2.5, but the allowed difference with "
                        "rtol=1.3e-06 and atol=1e-05 is only 1.715e-05!")
        self.assertEqual(debug_msg, expected_msg)

        # complex x int
        result, debug_msg = self._compareScalars(complex(1, -2), 1)
        expected_msg = ("Comparing the imaginary part -2.0 and 0.0 gives a "
                        "difference of 2.0, but the allowed difference with "
                        "rtol=1.3e-06 and atol=1e-05 is only 1e-05!")
        self.assertEqual(debug_msg, expected_msg)

        # NaN x NaN, equal_nan=False
        result, debug_msg = self._compareScalars(float('nan'), float('nan'), equal_nan=False)
        expected_msg = ("Found nan and nan while comparing and either one is "
                        "nan and the other isn't, or both are nan and equal_nan "
                        "is False")
        self.assertEqual(debug_msg, expected_msg)

    # Checks that compareTensors provides the correct debug info
    @onlyOnCPUAndCUDA
    def test__comparetensors_debug_msg(self, device):
        # Acquires atol that will be used
        atol = max(1e-05, self.precision)

        # Checks float tensor comparisons (2D tensor)
        a = torch.tensor(((0, 6), (7, 9)), device=device, dtype=torch.float32)
        b = torch.tensor(((0, 7), (7, 22)), device=device, dtype=torch.float32)
        result, debug_msg = self._compareTensors(a, b)
        expected_msg = ("With rtol=1.3e-06 and atol={0}, found 2 element(s) (out of 4) "
                        "whose difference(s) exceeded the margin of error (including 0 nan comparisons). "
                        "The greatest difference was 13.0 (9.0 vs. 22.0), "
                        "which occurred at index (1, 1).").format(atol)
        self.assertEqual(debug_msg, expected_msg)

        # Checks float tensor comparisons (with extremal values)
        a = torch.tensor((float('inf'), 5, float('inf')), device=device, dtype=torch.float32)
        b = torch.tensor((float('inf'), float('nan'), float('-inf')), device=device, dtype=torch.float32)
        result, debug_msg = self._compareTensors(a, b)
        expected_msg = ("With rtol=1.3e-06 and atol={0}, found 2 element(s) (out of 3) "
                        "whose difference(s) exceeded the margin of error (including 1 nan comparisons). "
                        "The greatest difference was nan (5.0 vs. nan), "
                        "which occurred at index 1.").format(atol)
        self.assertEqual(debug_msg, expected_msg)

        # Checks float tensor comparisons (with finite vs nan differences)
        a = torch.tensor((20, -6), device=device, dtype=torch.float32)
        b = torch.tensor((-1, float('nan')), device=device, dtype=torch.float32)
        result, debug_msg = self._compareTensors(a, b)
        expected_msg = ("With rtol=1.3e-06 and atol={0}, found 2 element(s) (out of 2) "
                        "whose difference(s) exceeded the margin of error (including 1 nan comparisons). "
                        "The greatest difference was nan (-6.0 vs. nan), "
                        "which occurred at index 1.").format(atol)
        self.assertEqual(debug_msg, expected_msg)

        # Checks int tensor comparisons (1D tensor)
        a = torch.tensor((1, 2, 3, 4), device=device)
        b = torch.tensor((2, 5, 3, 4), device=device)
        result, debug_msg = self._compareTensors(a, b)
        expected_msg = ("Found 2 different element(s) (out of 4), "
                        "with the greatest difference of 3 (2 vs. 5) "
                        "occuring at index 1.")
        self.assertEqual(debug_msg, expected_msg)

        # Checks bool tensor comparisons (0D tensor)
        a = torch.tensor((True), device=device)
        b = torch.tensor((False), device=device)
        result, debug_msg = self._compareTensors(a, b)
        expected_msg = ("Found 1 different element(s) (out of 1), "
                        "with the greatest difference of 1 (1 vs. 0) "
                        "occuring at index 0.")
        self.assertEqual(debug_msg, expected_msg)

        # Checks complex tensor comparisons (real part)
        a = torch.tensor((1 - 1j, 4 + 3j), device=device)
        b = torch.tensor((1 - 1j, 1 + 3j), device=device)
        result, debug_msg = self._compareTensors(a, b)
        expected_msg = ("Real parts failed to compare as equal! "
                        "With rtol=1.3e-06 and atol={0}, "
                        "found 1 element(s) (out of 2) whose difference(s) exceeded the "
                        "margin of error (including 0 nan comparisons). The greatest difference was "
                        "3.0 (4.0 vs. 1.0), which occurred at index 1.").format(atol)
        self.assertEqual(debug_msg, expected_msg)

        # Checks complex tensor comparisons (imaginary part)
        a = torch.tensor((1 - 1j, 4 + 3j), device=device)
        b = torch.tensor((1 - 1j, 4 - 21j), device=device)
        result, debug_msg = self._compareTensors(a, b)
        expected_msg = ("Imaginary parts failed to compare as equal! "
                        "With rtol=1.3e-06 and atol={0}, "
                        "found 1 element(s) (out of 2) whose difference(s) exceeded the "
                        "margin of error (including 0 nan comparisons). The greatest difference was "
                        "24.0 (3.0 vs. -21.0), which occurred at index 1.").format(atol)
        self.assertEqual(debug_msg, expected_msg)

        # Checks size mismatch
        a = torch.tensor((1, 2), device=device)
        b = torch.tensor((3), device=device)
        result, debug_msg = self._compareTensors(a, b)
        expected_msg = ("Attempted to compare equality of tensors "
                        "with different sizes. Got sizes torch.Size([2]) and torch.Size([]).")
        self.assertEqual(debug_msg, expected_msg)

        # Checks dtype mismatch
        a = torch.tensor((1, 2), device=device, dtype=torch.long)
        b = torch.tensor((1, 2), device=device, dtype=torch.float32)
        result, debug_msg = self._compareTensors(a, b, exact_dtype=True)
        expected_msg = ("Attempted to compare equality of tensors "
                        "with different dtypes. Got dtypes torch.int64 and torch.float32.")
        self.assertEqual(debug_msg, expected_msg)

        # Checks device mismatch
        if self.device_type == 'cuda':
            a = torch.tensor((5), device='cpu')
            b = torch.tensor((5), device=device)
            result, debug_msg = self._compareTensors(a, b, exact_device=True)
            expected_msg = ("Attempted to compare equality of tensors "
                            "on different devices! Got devices cpu and cuda:0.")
            self.assertEqual(debug_msg, expected_msg)

    # Helper for testing _compareTensors and _compareScalars
    # Works on single element tensors
    def _comparetensors_helper(self, tests, device, dtype, equal_nan, exact_dtype=True, atol=1e-08, rtol=1e-05):
        for test in tests:
            a = torch.tensor((test[0],), device=device, dtype=dtype)
            b = torch.tensor((test[1],), device=device, dtype=dtype)

            # Tensor x Tensor comparison
            compare_result, debug_msg = self._compareTensors(a, b, rtol=rtol, atol=atol,
                                                             equal_nan=equal_nan,
                                                             exact_dtype=exact_dtype)
            self.assertEqual(compare_result, test[2])

            # Scalar x Scalar comparison
            compare_result, debug_msg = self._compareScalars(a.item(), b.item(),
                                                             rtol=rtol, atol=atol,
                                                             equal_nan=equal_nan)
            self.assertEqual(compare_result, test[2])

    def _isclose_helper(self, tests, device, dtype, equal_nan, atol=1e-08, rtol=1e-05):
        for test in tests:
            a = torch.tensor((test[0],), device=device, dtype=dtype)
            b = torch.tensor((test[1],), device=device, dtype=dtype)

            actual = torch.isclose(a, b, equal_nan=equal_nan, atol=atol, rtol=rtol)
            expected = test[2]
            self.assertEqual(actual.item(), expected)

    # torch.close is not implemented for bool tensors
    # see https://github.com/pytorch/pytorch/issues/33048
    def test_isclose_comparetensors_bool(self, device):
        tests = (
            (True, True, True),
            (False, False, True),
            (True, False, False),
            (False, True, False),
        )

        with self.assertRaises(RuntimeError):
            self._isclose_helper(tests, device, torch.bool, False)

        self._comparetensors_helper(tests, device, torch.bool, False)

    @dtypes(torch.uint8,
            torch.int8, torch.int16, torch.int32, torch.int64)
    def test_isclose_comparetensors_integer(self, device, dtype):
        tests = (
            (0, 0, True),
            (0, 1, False),
            (1, 0, False),
        )

        self._isclose_helper(tests, device, dtype, False)

        # atol and rtol tests
        tests = [
            (0, 1, True),
            (1, 0, False),
            (1, 3, True),
        ]

        self._isclose_helper(tests, device, dtype, False, atol=.5, rtol=.5)
        self._comparetensors_helper(tests, device, dtype, False, atol=.5, rtol=.5)

        if dtype is torch.uint8:
            tests = [
                (-1, 1, False),
                (1, -1, False)
            ]
        else:
            tests = [
                (-1, 1, True),
                (1, -1, True)
            ]

        self._isclose_helper(tests, device, dtype, False, atol=1.5, rtol=.5)
        self._comparetensors_helper(tests, device, dtype, False, atol=1.5, rtol=.5)

    @onlyOnCPUAndCUDA
    @dtypes(torch.float16, torch.float32, torch.float64)
    def test_isclose_comparetensors_float(self, device, dtype):
        tests = (
            (0, 0, True),
            (0, -1, False),
            (float('inf'), float('inf'), True),
            (-float('inf'), float('inf'), False),
            (float('inf'), float('nan'), False),
            (float('nan'), float('nan'), False),
            (0, float('nan'), False),
            (1, 1, True),
        )

        self._isclose_helper(tests, device, dtype, False)
        self._comparetensors_helper(tests, device, dtype, False)

        # atol and rtol tests
        eps = 1e-2 if dtype is torch.half else 1e-6
        tests = (
            (0, 1, True),
            (0, 1 + eps, False),
            (1, 0, False),
            (1, 3, True),
            (1 - eps, 3, False),
            (-.25, .5, True),
            (-.25 - eps, .5, False),
            (.25, -.5, True),
            (.25 + eps, -.5, False),
        )

        self._isclose_helper(tests, device, dtype, False, atol=.5, rtol=.5)
        self._comparetensors_helper(tests, device, dtype, False, atol=.5, rtol=.5)

        # equal_nan = True tests
        tests = (
            (0, float('nan'), False),
            (float('inf'), float('nan'), False),
            (float('nan'), float('nan'), True),
        )

        self._isclose_helper(tests, device, dtype, True)

        self._comparetensors_helper(tests, device, dtype, True)

    # torch.close with equal_nan=True is not implemented for complex inputs
    # see https://github.com/numpy/numpy/issues/15959
    # Note: compareTensor will compare the real and imaginary parts of a
    # complex tensors separately, unlike isclose.
    @dtypes(torch.complex64, torch.complex128)
    def test_isclose_comparetensors_complex(self, device, dtype):
        tests = (
            (complex(1, 1), complex(1, 1 + 1e-8), True),
            (complex(0, 1), complex(1, 1), False),
            (complex(1, 1), complex(1, 0), False),
            (complex(1, 1), complex(1, float('nan')), False),
            (complex(1, float('nan')), complex(1, float('nan')), False),
            (complex(1, 1), complex(1, float('inf')), False),
            (complex(float('inf'), 1), complex(1, float('inf')), False),
            (complex(-float('inf'), 1), complex(1, float('inf')), False),
            (complex(-float('inf'), 1), complex(float('inf'), 1), False),
            (complex(float('inf'), 1), complex(float('inf'), 1), True),
            (complex(float('inf'), 1), complex(float('inf'), 1 + 1e-4), False),
        )

        self._isclose_helper(tests, device, dtype, False)
        self._comparetensors_helper(tests, device, dtype, False)

        # atol and rtol tests

        # atol and rtol tests
        eps = 1e-6
        tests = (
            # Complex versions of float tests (real part)
            (complex(0, 0), complex(1, 0), True),
            (complex(0, 0), complex(1 + eps, 0), False),
            (complex(1, 0), complex(0, 0), False),
            (complex(1, 0), complex(3, 0), True),
            (complex(1 - eps, 0), complex(3, 0), False),
            (complex(-.25, 0), complex(.5, 0), True),
            (complex(-.25 - eps, 0), complex(.5, 0), False),
            (complex(.25, 0), complex(-.5, 0), True),
            (complex(.25 + eps, 0), complex(-.5, 0), False),
            # Complex versions of float tests (imaginary part)
            (complex(0, 0), complex(0, 1), True),
            (complex(0, 0), complex(0, 1 + eps), False),
            (complex(0, 1), complex(0, 0), False),
            (complex(0, 1), complex(0, 3), True),
            (complex(0, 1 - eps), complex(0, 3), False),
            (complex(0, -.25), complex(0, .5), True),
            (complex(0, -.25 - eps), complex(0, .5), False),
            (complex(0, .25), complex(0, -.5), True),
            (complex(0, .25 + eps), complex(0, -.5), False),
        )

        self._isclose_helper(tests, device, dtype, False, atol=.5, rtol=.5)
        self._comparetensors_helper(tests, device, dtype, False, atol=.5, rtol=.5)

        # atol and rtol tests for isclose
        tests = (
            # Complex-specific tests
            (complex(1, -1), complex(-1, 1), False),
            (complex(1, -1), complex(2, -2), True),
            (complex(-math.sqrt(2), math.sqrt(2)),
             complex(-math.sqrt(.5), math.sqrt(.5)), True),
            (complex(-math.sqrt(2), math.sqrt(2)),
             complex(-math.sqrt(.501), math.sqrt(.499)), False),
            (complex(2, 4), complex(1., 8.8523607), True),
            (complex(2, 4), complex(1., 8.8523607 + eps), False),
            (complex(1, 99), complex(4, 100), True),
        )

        self._isclose_helper(tests, device, dtype, False, atol=.5, rtol=.5)

        # atol and rtol tests for compareTensors
        tests = (
            (complex(1, -1), complex(-1, 1), False),
            (complex(1, -1), complex(2, -2), True),
            (complex(1, 99), complex(4, 100), False),
        )

        self._comparetensors_helper(tests, device, dtype, False, atol=.5, rtol=.5)

        # equal_nan = True tests
        tests = (
            (complex(1, 1), complex(1, float('nan')), False),
            (complex(float('nan'), 1), complex(1, float('nan')), False),
            (complex(float('nan'), 1), complex(float('nan'), 1), True),
        )

        with self.assertRaises(RuntimeError):
            self._isclose_helper(tests, device, dtype, True)

        self._comparetensors_helper(tests, device, dtype, True)

    # Tests that isclose with rtol or atol values less than zero throws a
    #   RuntimeError
    @dtypes(torch.bool, torch.uint8,
            torch.int8, torch.int16, torch.int32, torch.int64,
            torch.float16, torch.float32, torch.float64)
    def test_isclose_atol_rtol_greater_than_zero(self, device, dtype):
        t = torch.tensor((1,), device=device, dtype=dtype)

        with self.assertRaises(RuntimeError):
            torch.isclose(t, t, atol=-1, rtol=1)
        with self.assertRaises(RuntimeError):
            torch.isclose(t, t, atol=1, rtol=-1)
        with self.assertRaises(RuntimeError):
            torch.isclose(t, t, atol=-1, rtol=-1)

    def test_assert_messages(self, device):
        self.assertIsNone(self._get_assert_msg(msg=None))
        self.assertEqual("\nno_debug_msg", self._get_assert_msg("no_debug_msg"))
        self.assertEqual("no_user_msg", self._get_assert_msg(msg=None, debug_msg="no_user_msg"))
        self.assertEqual("debug_msg\nuser_msg", self._get_assert_msg(msg="user_msg", debug_msg="debug_msg"))

    @onlyCUDA
    @slowTest
    def test_cuda_assert_should_stop_test_suite(self, device):
        # This test is slow because it spawn another process to run another test suite.

        # Test running of cuda assert test suite should early terminate.
        stderr = TestCase.runWithPytorchAPIUsageStderr("""\
#!/usr/bin/env python

import torch

from torch.testing._internal.common_utils import (TestCase, run_tests, slowTest)
from torch.testing._internal.common_device_type import instantiate_device_type_tests

# This test is added to ensure that test suite terminates early when
# CUDA assert was thrown since all subsequent test will fail.
# See: https://github.com/pytorch/pytorch/issues/49019
# This test file should be invoked from test_testing.py
class TestThatContainsCUDAAssertFailure(TestCase):

    @slowTest
    def test_throw_unrecoverable_cuda_exception(self, device):
        x = torch.rand(10, device=device)
        # cause unrecoverable CUDA exception, recoverable on CPU
        y = x[torch.tensor([25])].cpu()

    @slowTest
    def test_trivial_passing_test_case_on_cpu_cuda(self, device):
        x1 = torch.tensor([0., 1.], device=device)
        x2 = torch.tensor([0., 1.], device='cpu')
        self.assertEqual(x1, x2)

instantiate_device_type_tests(
    TestThatContainsCUDAAssertFailure,
    globals(),
    only_for='cuda'
)

if __name__ == '__main__':
    run_tests()
""")
        # should capture CUDA error
        self.assertIn('CUDA error: device-side assert triggered', stderr)
        # should run only 1 test because it throws unrecoverable error.
        self.assertIn('Ran 1 test', stderr)


instantiate_device_type_tests(TestTesting, globals())


class TestMypyWrapper(TestCase):
    def test_glob(self):
        # can match individual files
        self.assertTrue(mypy_wrapper.glob(
            pattern='test/test_torch.py',
            filename=PurePosixPath('test/test_torch.py'),
        ))
        self.assertFalse(mypy_wrapper.glob(
            pattern='test/test_torch.py',
            filename=PurePosixPath('test/test_testing.py'),
        ))

        # dir matters
        self.assertFalse(mypy_wrapper.glob(
            pattern='tools/codegen/utils.py',
            filename=PurePosixPath('torch/nn/modules.py'),
        ))
        self.assertTrue(mypy_wrapper.glob(
            pattern='setup.py',
            filename=PurePosixPath('setup.py'),
        ))
        self.assertFalse(mypy_wrapper.glob(
            pattern='setup.py',
            filename=PurePosixPath('foo/setup.py'),
        ))
        self.assertTrue(mypy_wrapper.glob(
            pattern='foo/setup.py',
            filename=PurePosixPath('foo/setup.py'),
        ))

        # can match dirs
        self.assertTrue(mypy_wrapper.glob(
            pattern='torch',
            filename=PurePosixPath('torch/random.py'),
        ))
        self.assertTrue(mypy_wrapper.glob(
            pattern='torch',
            filename=PurePosixPath('torch/nn/cpp.py'),
        ))
        self.assertFalse(mypy_wrapper.glob(
            pattern='torch',
            filename=PurePosixPath('tools/fast_nvcc/fast_nvcc.py'),
        ))

        # can match wildcards
        self.assertTrue(mypy_wrapper.glob(
            pattern='tools/autograd/*.py',
            filename=PurePosixPath('tools/autograd/gen_autograd.py'),
        ))
        self.assertFalse(mypy_wrapper.glob(
            pattern='tools/autograd/*.py',
            filename=PurePosixPath('tools/autograd/deprecated.yaml'),
        ))


def fakehash(char):
    return char * 40


def makecase(name, seconds, *, errored=False, failed=False, skipped=False):
    return {
        'name': name,
        'seconds': seconds,
        'errored': errored,
        'failed': failed,
        'skipped': skipped,
    }


def makereport(tests):
    suites = {
        suite_name: {
            'total_seconds': sum(case['seconds'] for case in cases),
            'cases': cases,
        }
        for suite_name, cases in tests.items()
    }
    return {
        'total_seconds': sum(s['total_seconds'] for s in suites.values()),
        'suites': suites,
    }


class TestPrintTestStats(TestCase):
    maxDiff = None

    def test_analysis(self):
        head_report = makereport({
            # input ordering of the suites is ignored
            'Grault': [
                # not printed: status same and time similar
                makecase('test_grault0', 4.78, failed=True),
                # status same, but time increased a lot
                makecase('test_grault2', 1.473, errored=True),
            ],
            # individual tests times changed, not overall suite
            'Qux': [
                # input ordering of the test cases is ignored
                makecase('test_qux1', 0.001, skipped=True),
                makecase('test_qux6', 0.002, skipped=True),
                # time in bounds, but status changed
                makecase('test_qux4', 7.158, failed=True),
                # not printed because it's the same as before
                makecase('test_qux7', 0.003, skipped=True),
                makecase('test_qux5', 11.968),
                makecase('test_qux3', 23.496),
            ],
            # new test suite
            'Bar': [
                makecase('test_bar2', 3.742, failed=True),
                makecase('test_bar1', 50.447),
            ],
            # overall suite time changed but no individual tests
            'Norf': [
                makecase('test_norf1', 3),
                makecase('test_norf2', 3),
                makecase('test_norf3', 3),
                makecase('test_norf4', 3),
            ],
            # suite doesn't show up if it doesn't change enough
            'Foo': [
                makecase('test_foo1', 42),
                makecase('test_foo2', 56),
            ],
        })

        base_reports = {
            # bbbb has no reports, so base is cccc instead
            fakehash('b'): [],
            fakehash('c'): [
                makereport({
                    'Baz': [
                        makecase('test_baz2', 13.605),
                        # no recent suites have & skip this test
                        makecase('test_baz1', 0.004, skipped=True),
                    ],
                    'Foo': [
                        makecase('test_foo1', 43),
                        # test added since dddd
                        makecase('test_foo2', 57),
                    ],
                    'Grault': [
                        makecase('test_grault0', 4.88, failed=True),
                        makecase('test_grault1', 11.967, failed=True),
                        makecase('test_grault2', 0.395, errored=True),
                        makecase('test_grault3', 30.460),
                    ],
                    'Norf': [
                        makecase('test_norf1', 2),
                        makecase('test_norf2', 2),
                        makecase('test_norf3', 2),
                        makecase('test_norf4', 2),
                    ],
                    'Qux': [
                        makecase('test_qux3', 4.978, errored=True),
                        makecase('test_qux7', 0.002, skipped=True),
                        makecase('test_qux2', 5.618),
                        makecase('test_qux4', 7.766, errored=True),
                        makecase('test_qux6', 23.589, failed=True),
                    ],
                }),
            ],
            fakehash('d'): [
                makereport({
                    'Foo': [
                        makecase('test_foo1', 40),
                        # removed in cccc
                        makecase('test_foo3', 17),
                    ],
                    'Baz': [
                        # not skipped, so not included in stdev
                        makecase('test_baz1', 3.14),
                    ],
                    'Qux': [
                        makecase('test_qux7', 0.004, skipped=True),
                        makecase('test_qux2', 6.02),
                        makecase('test_qux4', 20.932),
                    ],
                    'Norf': [
                        makecase('test_norf1', 3),
                        makecase('test_norf2', 3),
                        makecase('test_norf3', 3),
                        makecase('test_norf4', 3),
                    ],
                    'Grault': [
                        makecase('test_grault0', 5, failed=True),
                        makecase('test_grault1', 14.325, failed=True),
                        makecase('test_grault2', 0.31, errored=True),
                    ],
                }),
            ],
            fakehash('e'): [],
            fakehash('f'): [
                makereport({
                    'Foo': [
                        makecase('test_foo3', 24),
                        makecase('test_foo1', 43),
                    ],
                    'Baz': [
                        makecase('test_baz2', 16.857),
                    ],
                    'Qux': [
                        makecase('test_qux2', 6.422),
                        makecase('test_qux4', 6.382, errored=True),
                    ],
                    'Norf': [
                        makecase('test_norf1', 0.9),
                        makecase('test_norf3', 0.9),
                        makecase('test_norf2', 0.9),
                        makecase('test_norf4', 0.9),
                    ],
                    'Grault': [
                        makecase('test_grault0', 4.7, failed=True),
                        makecase('test_grault1', 13.146, failed=True),
                        makecase('test_grault2', 0.48, errored=True),
                    ],
                }),
            ],
        }

        simpler_head = print_test_stats.simplify(head_report)
        simpler_base = {}
        for commit, reports in base_reports.items():
            simpler_base[commit] = [print_test_stats.simplify(r) for r in reports]
        analysis = print_test_stats.analyze(
            head_report=simpler_head,
            base_reports=simpler_base,
        )

        self.assertEqual(
            '''\

- class Baz:
-     # was   15.23s ±   2.30s
-
-     def test_baz1: ...
-         # was   0.004s           (skipped)
-
-     def test_baz2: ...
-         # was  15.231s ±  2.300s


  class Grault:
      # was   48.86s ±   1.19s
      # now    6.25s

    - def test_grault1: ...
    -     # was  13.146s ±  1.179s (failed)

    - def test_grault3: ...
    -     # was  30.460s


  class Qux:
      # was   41.66s ±   1.06s
      # now   42.63s

    - def test_qux2: ...
    -     # was   6.020s ±  0.402s

    ! def test_qux3: ...
    !     # was   4.978s           (errored)
    !     # now  23.496s

    ! def test_qux4: ...
    !     # was   7.074s ±  0.979s (errored)
    !     # now   7.158s           (failed)

    ! def test_qux6: ...
    !     # was  23.589s           (failed)
    !     # now   0.002s           (skipped)

    + def test_qux1: ...
    +     # now   0.001s           (skipped)

    + def test_qux5: ...
    +     # now  11.968s


+ class Bar:
+     # now   54.19s
+
+     def test_bar1: ...
+         # now  50.447s
+
+     def test_bar2: ...
+         # now   3.742s           (failed)

''',
            print_test_stats.anomalies(analysis),
        )

    def test_graph(self):
        # HEAD is on master
        self.assertEqual(
            '''\
Commit graph (base is most recent master ancestor with at least one S3 report):

    : (master)
    |
    * aaaaaaaaaa (HEAD)              total time   502.99s
    * bbbbbbbbbb (base)   1 report,  total time    47.84s
    * cccccccccc          1 report,  total time   332.50s
    * dddddddddd          0 reports
    |
    :
''',
            print_test_stats.graph(
                head_sha=fakehash('a'),
                head_seconds=502.99,
                base_seconds={
                    fakehash('b'): [47.84],
                    fakehash('c'): [332.50],
                    fakehash('d'): [],
                },
                on_master=True,
            )
        )

        self.assertEqual(
            '''\
Commit graph (base is most recent master ancestor with at least one S3 report):

    : (master)
    |
    | * aaaaaaaaaa (HEAD)            total time  9988.77s
    |/
    * bbbbbbbbbb (base) 121 reports, total time  7654.32s ±   55.55s
    * cccccccccc         20 reports, total time  5555.55s ±  253.19s
    * dddddddddd          1 report,  total time  1234.56s
    |
    :
''',
            print_test_stats.graph(
                head_sha=fakehash('a'),
                head_seconds=9988.77,
                base_seconds={
                    fakehash('b'): [7598.77] * 60 + [7654.32] + [7709.87] * 60,
                    fakehash('c'): [5308.77] * 10 + [5802.33] * 10,
                    fakehash('d'): [1234.56],
                },
                on_master=False,
            )
        )

        self.assertEqual(
            '''\
Commit graph (base is most recent master ancestor with at least one S3 report):

    : (master)
    |
    | * aaaaaaaaaa (HEAD)            total time    25.52s
    | |
    | : (5 commits)
    |/
    * bbbbbbbbbb          0 reports
    * cccccccccc          0 reports
    * dddddddddd (base)  15 reports, total time    58.92s ±   25.82s
    |
    :
''',
            print_test_stats.graph(
                head_sha=fakehash('a'),
                head_seconds=25.52,
                base_seconds={
                    fakehash('b'): [],
                    fakehash('c'): [],
                    fakehash('d'): [52.25] * 14 + [152.26],
                },
                on_master=False,
                ancestry_path=5,
            )
        )

        self.assertEqual(
            '''\
Commit graph (base is most recent master ancestor with at least one S3 report):

    : (master)
    |
    | * aaaaaaaaaa (HEAD)            total time     0.08s
    |/|
    | : (1 commit)
    |
    * bbbbbbbbbb          0 reports
    * cccccccccc (base)   1 report,  total time     0.09s
    * dddddddddd          3 reports, total time     0.10s ±    0.05s
    |
    :
''',
            print_test_stats.graph(
                head_sha=fakehash('a'),
                head_seconds=0.08,
                base_seconds={
                    fakehash('b'): [],
                    fakehash('c'): [0.09],
                    fakehash('d'): [0.05, 0.10, 0.15],
                },
                on_master=False,
                other_ancestors=1,
            )
        )

        self.assertEqual(
            '''\
Commit graph (base is most recent master ancestor with at least one S3 report):

    : (master)
    |
    | * aaaaaaaaaa (HEAD)            total time     5.98s
    | |
    | : (1 commit)
    |/|
    | : (7 commits)
    |
    * bbbbbbbbbb (base)   2 reports, total time     6.02s ±    1.71s
    * cccccccccc          0 reports
    * dddddddddd         10 reports, total time     5.84s ±    0.92s
    |
    :
''',
            print_test_stats.graph(
                head_sha=fakehash('a'),
                head_seconds=5.98,
                base_seconds={
                    fakehash('b'): [4.81, 7.23],
                    fakehash('c'): [],
                    fakehash('d'): [4.97] * 5 + [6.71] * 5,
                },
                on_master=False,
                ancestry_path=1,
                other_ancestors=7,
            )
        )

    def test_regression_info(self):
        self.assertEqual(
            '''\
----- Historic stats comparison result ------

    job: foo_job
    commit: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa


  class Foo:
      # was   42.50s ±   2.12s
      # now    3.02s

    - def test_bar: ...
    -     # was   1.000s

    ! def test_foo: ...
    !     # was  41.500s ±  2.121s
    !     # now   0.020s           (skipped)

    + def test_baz: ...
    +     # now   3.000s


Commit graph (base is most recent master ancestor with at least one S3 report):

    : (master)
    |
    | * aaaaaaaaaa (HEAD)            total time     3.02s
    |/
    * bbbbbbbbbb (base)   1 report,  total time    41.00s
    * cccccccccc          1 report,  total time    43.00s
    |
    :

Removed  (across    1 suite)      1 test,  totaling -   1.00s
Modified (across    1 suite)      1 test,  totaling -  41.48s ±   2.12s
Added    (across    1 suite)      1 test,  totaling +   3.00s
''',
            print_test_stats.regression_info(
                head_sha=fakehash('a'),
                head_report=makereport({
                    'Foo': [
                        makecase('test_foo', 0.02, skipped=True),
                        makecase('test_baz', 3),
                    ]}),
                base_reports={
                    fakehash('b'): [
                        makereport({
                            'Foo': [
                                makecase('test_foo', 40),
                                makecase('test_bar', 1),
                            ],
                        }),
                    ],
                    fakehash('c'): [
                        makereport({
                            'Foo': [
                                makecase('test_foo', 43),
                            ],
                        }),
                    ],
                },
                job_name='foo_job',
                on_master=False,
                ancestry_path=0,
                other_ancestors=0,
            )
        )


if __name__ == '__main__':
    run_tests()
