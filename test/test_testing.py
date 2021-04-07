import torch

import math
import random

from torch.testing._internal.common_utils import \
    (TestCase, make_tensor, run_tests, slowTest)
from torch.testing._internal.framework_utils import calculate_shards
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, onlyCUDA, onlyOnCPUAndCUDA, dtypes)

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

    @dtypes(torch.bool, torch.long, torch.float, torch.cfloat)
    def test_make_tensor(self, device, dtype):
        def check(size, low, high, requires_grad, discontiguous):
            t = make_tensor(size, device, dtype, low=low, high=high,
                            requires_grad=requires_grad, discontiguous=discontiguous)

            self.assertEqual(t.shape, size)
            self.assertEqual(t.device, torch.device(device))
            self.assertEqual(t.dtype, dtype)

            low = -9 if low is None else low
            high = 9 if high is None else high

            if t.numel() > 0 and dtype in [torch.long, torch.float]:
                self.assertTrue(t.le(high).logical_and(t.ge(low)).all().item())

            if dtype in [torch.float, torch.cfloat]:
                self.assertEqual(t.requires_grad, requires_grad)
            else:
                self.assertFalse(t.requires_grad)

            if t.numel() > 1:
                self.assertEqual(t.is_contiguous(), not discontiguous)
            else:
                self.assertTrue(t.is_contiguous())

        for size in (tuple(), (0,), (1,), (1, 1), (2,), (2, 3), (8, 16, 32)):
            check(size, None, None, False, False)
            check(size, 2, 4, True, True)

    def test_assert_messages(self, device):
        self.assertIsNone(self._get_assert_msg(msg=None))
        self.assertEqual("\nno_debug_msg", self._get_assert_msg("no_debug_msg"))
        self.assertEqual("no_user_msg", self._get_assert_msg(msg=None, debug_msg="no_user_msg"))
        self.assertEqual("debug_msg\nuser_msg", self._get_assert_msg(msg="user_msg", debug_msg="debug_msg"))

    # The following tests (test_cuda_assert_*) are added to ensure test suite terminates early
    # when CUDA assert was thrown. Because all subsequent test will fail if that happens.
    # These tests are slow because it spawn another process to run test suite.
    # See: https://github.com/pytorch/pytorch/issues/49019
    @onlyCUDA
    @slowTest
    def test_cuda_assert_should_stop_common_utils_test_suite(self, device):
        # test to ensure common_utils.py override has early termination for CUDA.
        stderr = TestCase.runWithPytorchAPIUsageStderr("""\
#!/usr/bin/env python

import torch
from torch.testing._internal.common_utils import (TestCase, run_tests, slowTest)

class TestThatContainsCUDAAssertFailure(TestCase):

    @slowTest
    def test_throw_unrecoverable_cuda_exception(self):
        x = torch.rand(10, device='cuda')
        # cause unrecoverable CUDA exception, recoverable on CPU
        y = x[torch.tensor([25])].cpu()

    @slowTest
    def test_trivial_passing_test_case_on_cpu_cuda(self):
        x1 = torch.tensor([0., 1.], device='cuda')
        x2 = torch.tensor([0., 1.], device='cpu')
        self.assertEqual(x1, x2)

if __name__ == '__main__':
    run_tests()
""")
        # should capture CUDA error
        self.assertIn('CUDA error: device-side assert triggered', stderr)
        # should run only 1 test because it throws unrecoverable error.
        self.assertIn('Ran 1 test', stderr)


    @onlyCUDA
    @slowTest
    def test_cuda_assert_should_stop_common_device_type_test_suite(self, device):
        # test to ensure common_device_type.py override has early termination for CUDA.
        stderr = TestCase.runWithPytorchAPIUsageStderr("""\
#!/usr/bin/env python

import torch
from torch.testing._internal.common_utils import (TestCase, run_tests, slowTest)
from torch.testing._internal.common_device_type import instantiate_device_type_tests

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


    @onlyCUDA
    @slowTest
    def test_cuda_assert_should_not_stop_common_distributed_test_suite(self, device):
        # test to ensure common_distributed.py override should not early terminate CUDA.
        stderr = TestCase.runWithPytorchAPIUsageStderr("""\
#!/usr/bin/env python

import torch
from torch.testing._internal.common_utils import (run_tests, slowTest)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import MultiProcessTestCase

class TestThatContainsCUDAAssertFailure(MultiProcessTestCase):

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
        # we are currently disabling CUDA early termination for distributed tests.
        self.assertIn('Ran 2 test', stderr)


instantiate_device_type_tests(TestTesting, globals())


class TestFrameworkUtils(TestCase):
    tests = [
        'super_long_test',
        'long_test1',
        'long_test2',
        'normal_test1',
        'normal_test2',
        'normal_test3',
        'short_test1',
        'short_test2',
        'short_test3',
        'short_test4',
        'short_test5',
    ]

    test_times = {
        'super_long_test': 55,
        'long_test1': 22,
        'long_test2': 18,
        'normal_test1': 9,
        'normal_test2': 7,
        'normal_test3': 5,
        'short_test1': 1,
        'short_test2': 0.6,
        'short_test3': 0.4,
        'short_test4': 0.3,
        'short_test5': 0.01,
    }

    def test_calculate_2_shards_with_complete_test_times(self):
        expected_shards = [
            (60, ['super_long_test', 'normal_test3']),
            (58.31, ['long_test1', 'long_test2', 'normal_test1', 'normal_test2', 'short_test1', 'short_test2',
                     'short_test3', 'short_test4', 'short_test5'])
        ]
        self.assertEqual(expected_shards, calculate_shards(2, self.tests, self.test_times))


    def test_calculate_5_shards_with_complete_test_times(self):
        expected_shards = [
            (55, ['super_long_test']),
            (22, ['long_test1', ]),
            (18, ['long_test2', ]),
            (11.31, ['normal_test1', 'short_test1', 'short_test2', 'short_test3', 'short_test4', 'short_test5']),
            (12, ['normal_test2', 'normal_test3']),
        ]
        self.assertEqual(expected_shards, calculate_shards(5, self.tests, self.test_times))


    def test_calculate_2_shards_with_incomplete_test_times(self):
        incomplete_test_times = {k: v for k, v in self.test_times.items() if 'test1' in k}
        expected_shards = [
            (22, ['long_test1', 'long_test2', 'normal_test3', 'short_test3', 'short_test5']),
            (10, ['normal_test1', 'short_test1', 'super_long_test', 'normal_test2', 'short_test2', 'short_test4']),
        ]
        self.assertEqual(expected_shards, calculate_shards(2, self.tests, incomplete_test_times))


    def test_calculate_5_shards_with_incomplete_test_times(self):
        incomplete_test_times = {k: v for k, v in self.test_times.items() if 'test1' in k}
        expected_shards = [
            (22, ['long_test1', 'normal_test2', 'short_test5']),
            (9, ['normal_test1', 'normal_test3']),
            (1, ['short_test1', 'short_test2']),
            (0, ['super_long_test', 'short_test3']),
            (0, ['long_test2', 'short_test4']),
        ]
        self.assertEqual(expected_shards, calculate_shards(5, self.tests, incomplete_test_times))

    def test_calculate_2_shards_against_optimal_shards(self):
        for _ in range(100):
            random.seed(120)
            random_times = {k: random.random() * 10 for k in self.tests}
            # all test times except first two
            rest_of_tests = [i for k, i in random_times.items() if k != 'super_long_test' and k != 'long_test1']
            sum_of_rest = sum(rest_of_tests)
            random_times['super_long_test'] = max(sum_of_rest / 2, max(rest_of_tests))
            random_times['long_test1'] = sum_of_rest - random_times['super_long_test']
            # An optimal sharding would look like the below, but we don't need to compute this for the test:
            # optimal_shards = [
            #     (sum_of_rest, ['super_long_test', 'long_test1']),
            #     (sum_of_rest, [i for i in self.tests if i != 'super_long_test' and i != 'long_test1']),
            # ]
            calculated_shards = calculate_shards(2, self.tests, random_times)
            max_shard_time = max(calculated_shards[0][0], calculated_shards[1][0])
            if sum_of_rest != 0:
                # The calculated shard should not have a ratio worse than 7/6 for num_shards = 2
                self.assertGreaterEqual(7.0 / 6.0, max_shard_time / sum_of_rest)
                sorted_tests = sorted(self.tests)
                sorted_shard_tests = sorted(calculated_shards[0][1] + calculated_shards[1][1])
                # All the tests should be represented by some shard
                self.assertEqual(sorted_tests, sorted_shard_tests)


if __name__ == '__main__':
    run_tests()
