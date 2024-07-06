# Owner(s): ["module: tests"]

import collections
import doctest
import functools
import importlib
import inspect
import itertools
import math
import os
import re
import subprocess
import sys
import unittest.mock
from typing import Any, Callable, Iterator, List, Tuple

import torch

from torch.testing import make_tensor
from torch.testing._internal.common_utils import \
    (IS_FBCODE, IS_JETSON, IS_MACOS, IS_SANDCASTLE, IS_WINDOWS, TestCase, run_tests, slowTest,
     parametrize, subtest, instantiate_parametrized_tests, dtype_name, TEST_WITH_ROCM, decorateIf)
from torch.testing._internal.common_device_type import \
    (PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY, PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY, dtypes,
     get_device_type_test_bases, instantiate_device_type_tests, onlyCPU, onlyCUDA, onlyNativeDeviceTypes,
     deviceCountAtLeast, ops, expectedFailureMeta, OpDTypes)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal import opinfo
from torch.testing._internal.common_dtype import all_types_and_complex_and, floating_types
from torch.testing._internal.common_modules import modules, module_db, ModuleInfo
from torch.testing._internal.opinfo.core import SampleInput, DecorateInfo, OpInfo
import operator

# For testing TestCase methods and torch.testing functions
class TestTesting(TestCase):
    # Ensure that assertEqual handles numpy arrays properly
    @dtypes(*all_types_and_complex_and(torch.bool, torch.half))
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
            a = make_tensor(test_size, dtype=dtype, device=device, low=-5, high=5)
            a_n = a.cpu().numpy()
            msg = f'size: {test_size}'
            self.assertEqual(a_n, a, rtol=0, atol=0, msg=msg)
            self.assertEqual(a, a_n, rtol=0, atol=0, msg=msg)
            self.assertEqual(a_n, a_n, rtol=0, atol=0, msg=msg)

    def test_assertEqual_longMessage(self):
        actual = "actual"
        expected = "expected"

        long_message = self.longMessage
        try:
            # Capture the default error message by forcing TestCase.longMessage = False
            self.longMessage = False
            try:
                self.assertEqual(actual, expected)
            except AssertionError as error:
                default_msg = str(error)
            else:
                raise AssertionError("AssertionError not raised")

            self.longMessage = True
            extra_msg = "sentinel"
            with self.assertRaisesRegex(AssertionError, re.escape(f"{default_msg}\n{extra_msg}")):
                self.assertEqual(actual, expected, msg=extra_msg)
        finally:
            self.longMessage = long_message

    def _isclose_helper(self, tests, device, dtype, equal_nan, atol=1e-08, rtol=1e-05):
        for test in tests:
            a = torch.tensor((test[0],), device=device, dtype=dtype)
            b = torch.tensor((test[1],), device=device, dtype=dtype)

            actual = torch.isclose(a, b, equal_nan=equal_nan, atol=atol, rtol=rtol)
            expected = test[2]
            self.assertEqual(actual.item(), expected)

    def test_isclose_bool(self, device):
        tests = (
            (True, True, True),
            (False, False, True),
            (True, False, False),
            (False, True, False),
        )

        self._isclose_helper(tests, device, torch.bool, False)

    @dtypes(torch.uint8,
            torch.int8, torch.int16, torch.int32, torch.int64)
    def test_isclose_integer(self, device, dtype):
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

    @onlyNativeDeviceTypes
    @dtypes(torch.float16, torch.float32, torch.float64)
    def test_isclose_float(self, device, dtype):
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

        # equal_nan = True tests
        tests = (
            (0, float('nan'), False),
            (float('inf'), float('nan'), False),
            (float('nan'), float('nan'), True),
        )

        self._isclose_helper(tests, device, dtype, True)

    @unittest.skipIf(IS_SANDCASTLE, "Skipping because doesn't work on sandcastle")
    @dtypes(torch.complex64, torch.complex128)
    def test_isclose_complex(self, device, dtype):
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

        # equal_nan = True tests
        tests = (
            (complex(1, 1), complex(1, float('nan')), False),
            (complex(1, 1), complex(float('nan'), 1), False),
            (complex(float('nan'), 1), complex(float('nan'), 1), True),
            (complex(float('nan'), 1), complex(1, float('nan')), True),
            (complex(float('nan'), float('nan')), complex(float('nan'), float('nan')), True),
        )
        self._isclose_helper(tests, device, dtype, True)

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

    def test_isclose_equality_shortcut(self):
        # For values >= 2**53, integers differing by 1 can no longer differentiated by torch.float64 or lower precision
        # floating point dtypes. Thus, even with rtol == 0 and atol == 0, these tensors would be considered close if
        # they were not compared as integers.
        a = torch.tensor(2 ** 53, dtype=torch.int64)
        b = a + 1

        self.assertFalse(torch.isclose(a, b, rtol=0, atol=0))

    @dtypes(torch.float16, torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_isclose_nan_equality_shortcut(self, device, dtype):
        if dtype.is_floating_point:
            a = b = torch.nan
        else:
            a = complex(torch.nan, 0)
            b = complex(0, torch.nan)

        expected = True
        tests = [(a, b, expected)]

        self._isclose_helper(tests, device, dtype, equal_nan=True, rtol=0, atol=0)

    # The following tests (test_cuda_assert_*) are added to ensure test suite terminates early
    # when CUDA assert was thrown. Because all subsequent test will fail if that happens.
    # These tests are slow because it spawn another process to run test suite.
    # See: https://github.com/pytorch/pytorch/issues/49019
    @unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support device side asserts")
    @onlyCUDA
    @slowTest
    def test_cuda_assert_should_stop_common_utils_test_suite(self, device):
        # test to ensure common_utils.py override has early termination for CUDA.
        stderr = TestCase.runWithPytorchAPIUsageStderr("""\
#!/usr/bin/env python3

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
        self.assertIn('errors=1', stderr)


    @unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support device side asserts")
    @onlyCUDA
    @slowTest
    def test_cuda_assert_should_stop_common_device_type_test_suite(self, device):
        # test to ensure common_device_type.py override has early termination for CUDA.
        stderr = TestCase.runWithPytorchAPIUsageStderr("""\
#!/usr/bin/env python3

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
        self.assertIn('errors=1', stderr)


    @unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support device side asserts")
    @onlyCUDA
    @slowTest
    def test_cuda_assert_should_not_stop_common_distributed_test_suite(self, device):
        # test to ensure common_distributed.py override should not early terminate CUDA.
        stderr = TestCase.runWithPytorchAPIUsageStderr("""\
#!/usr/bin/env python3

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
        self.assertIn('errors=2', stderr)

    @expectedFailureMeta  # This is only supported for CPU and CUDA
    @onlyNativeDeviceTypes
    def test_get_supported_dtypes(self, device):
        # Test the `get_supported_dtypes` helper function.
        # We acquire the dtypes for few Ops dynamically and verify them against
        # the correct statically described values.
        ops_to_test = list(filter(lambda op: op.name in ['atan2', 'topk', 'xlogy'], op_db))

        for op in ops_to_test:
            dynamic_dtypes = opinfo.utils.get_supported_dtypes(op, op.sample_inputs_func, self.device_type)
            dynamic_dispatch = opinfo.utils.dtypes_dispatch_hint(dynamic_dtypes)
            if self.device_type == 'cpu':
                dtypes = op.dtypes
            else:  # device_type ='cuda'
                dtypes = op.dtypesIfCUDA

            self.assertTrue(set(dtypes) == set(dynamic_dtypes))
            self.assertTrue(set(dtypes) == set(dynamic_dispatch.dispatch_fn()))

    @onlyCPU
    @ops(
        [
            op
            for op in op_db
            if len(
                op.supported_dtypes("cpu").symmetric_difference(
                    op.supported_dtypes("cuda")
                )
            )
            > 0
        ][:1],
        dtypes=OpDTypes.none,
    )
    def test_supported_dtypes(self, device, op):
        self.assertNotEqual(op.supported_dtypes("cpu"), op.supported_dtypes("cuda"))
        self.assertEqual(op.supported_dtypes("cuda"), op.supported_dtypes("cuda:0"))
        self.assertEqual(
            op.supported_dtypes(torch.device("cuda")),
            op.supported_dtypes(torch.device("cuda", index=1)),
        )

instantiate_device_type_tests(TestTesting, globals())


class TestFrameworkUtils(TestCase):

    @unittest.skipIf(IS_WINDOWS, "Skipping because doesn't work for windows")
    @unittest.skipIf(IS_SANDCASTLE, "Skipping because doesn't work on sandcastle")
    def test_filtering_env_var(self):
        # Test environment variable selected device type test generator.
        test_filter_file_template = """\
#!/usr/bin/env python3

import torch
from torch.testing._internal.common_utils import (TestCase, run_tests)
from torch.testing._internal.common_device_type import instantiate_device_type_tests

class TestEnvironmentVariable(TestCase):

    def test_trivial_passing_test(self, device):
        x1 = torch.tensor([0., 1.], device=device)
        x2 = torch.tensor([0., 1.], device='cpu')
        self.assertEqual(x1, x2)

instantiate_device_type_tests(
    TestEnvironmentVariable,
    globals(),
)

if __name__ == '__main__':
    run_tests()
"""
        test_bases_count = len(get_device_type_test_bases())
        # Test without setting env var should run everything.
        env = dict(os.environ)
        for k in ['CI', PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY, PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY]:
            if k in env.keys():
                del env[k]
        _, stderr = TestCase.run_process_no_exception(test_filter_file_template, env=env)
        self.assertIn(f'Ran {test_bases_count} test', stderr.decode('ascii'))

        # Test with setting only_for should only run 1 test.
        env[PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY] = 'cpu'
        _, stderr = TestCase.run_process_no_exception(test_filter_file_template, env=env)
        self.assertIn('Ran 1 test', stderr.decode('ascii'))

        # Test with setting except_for should run 1 less device type from default.
        del env[PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY]
        env[PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY] = 'cpu'
        _, stderr = TestCase.run_process_no_exception(test_filter_file_template, env=env)
        self.assertIn(f'Ran {test_bases_count-1} test', stderr.decode('ascii'))

        # Test with setting both should throw exception
        env[PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY] = 'cpu'
        _, stderr = TestCase.run_process_no_exception(test_filter_file_template, env=env)
        self.assertNotIn('OK', stderr.decode('ascii'))


def make_assert_close_inputs(actual: Any, expected: Any) -> List[Tuple[Any, Any]]:
    """Makes inputs for :func:`torch.testing.assert_close` functions based on two examples.

    Args:
        actual (Any): Actual input.
        expected (Any): Expected input.

    Returns:
        List[Tuple[Any, Any]]: Pair of example inputs, as well as the example inputs wrapped in sequences
        (:class:`tuple`, :class:`list`), and mappings (:class:`dict`, :class:`~collections.OrderedDict`).
    """
    return [
        (actual, expected),
        # tuple vs. tuple
        ((actual,), (expected,)),
        # list vs. list
        ([actual], [expected]),
        # tuple vs. list
        ((actual,), [expected]),
        # dict vs. dict
        ({"t": actual}, {"t": expected}),
        # OrderedDict vs. OrderedDict
        (collections.OrderedDict([("t", actual)]), collections.OrderedDict([("t", expected)])),
        # dict vs. OrderedDict
        ({"t": actual}, collections.OrderedDict([("t", expected)])),
        # list of tuples vs. tuple of lists
        ([(actual,)], ([expected],)),
        # list of dicts vs. tuple of OrderedDicts
        ([{"t": actual}], (collections.OrderedDict([("t", expected)]),)),
        # dict of lists vs. OrderedDict of tuples
        ({"t": [actual]}, collections.OrderedDict([("t", (expected,))])),
    ]


def assert_close_with_inputs(actual: Any, expected: Any) -> Iterator[Callable]:
    """Yields :func:`torch.testing.assert_close` with predefined positional inputs based on two examples.

    .. note::

        Every test that does not test for a specific input should iterate over this to maximize the coverage.

    Args:
        actual (Any): Actual input.
        expected (Any): Expected input.

    Yields:
        Callable: :func:`torch.testing.assert_close` with predefined positional inputs.
    """
    for inputs in make_assert_close_inputs(actual, expected):
        yield functools.partial(torch.testing.assert_close, *inputs)


class TestAssertClose(TestCase):
    def test_mismatching_types_subclasses(self):
        actual = torch.rand(())
        expected = torch.nn.Parameter(actual)

        for fn in assert_close_with_inputs(actual, expected):
            fn()

    def test_mismatching_types_type_equality(self):
        actual = torch.empty(())
        expected = torch.nn.Parameter(actual)

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(TypeError, str(type(expected))):
                fn(allow_subclasses=False)

    def test_mismatching_types(self):
        actual = torch.empty(2)
        expected = actual.numpy()

        for fn, allow_subclasses in itertools.product(assert_close_with_inputs(actual, expected), (True, False)):
            with self.assertRaisesRegex(TypeError, str(type(expected))):
                fn(allow_subclasses=allow_subclasses)

    def test_unknown_type(self):
        actual = "0"
        expected = "0"

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(TypeError, str(type(actual))):
                fn()

    def test_mismatching_shape(self):
        actual = torch.empty(())
        expected = actual.clone().reshape((1,))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, "shape"):
                fn()

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), reason="MKLDNN is not available.")
    def test_unknown_layout(self):
        actual = torch.empty((2, 2))
        expected = actual.to_mkldnn()

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(ValueError, "layout"):
                fn()

    def test_meta(self):
        actual = torch.empty((2, 2), device="meta")
        expected = torch.empty((2, 2), device="meta")

        for fn in assert_close_with_inputs(actual, expected):
            fn()

    def test_mismatching_layout(self):
        strided = torch.empty((2, 2))
        sparse_coo = strided.to_sparse()
        sparse_csr = strided.to_sparse_csr()

        for actual, expected in itertools.combinations((strided, sparse_coo, sparse_csr), 2):
            for fn in assert_close_with_inputs(actual, expected):
                with self.assertRaisesRegex(AssertionError, "layout"):
                    fn()

    def test_mismatching_layout_no_check(self):
        strided = torch.randn((2, 2))
        sparse_coo = strided.to_sparse()
        sparse_csr = strided.to_sparse_csr()

        for actual, expected in itertools.combinations((strided, sparse_coo, sparse_csr), 2):
            for fn in assert_close_with_inputs(actual, expected):
                fn(check_layout=False)

    def test_mismatching_dtype(self):
        actual = torch.empty((), dtype=torch.float)
        expected = actual.clone().to(torch.int)

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, "dtype"):
                fn()

    def test_mismatching_dtype_no_check(self):
        actual = torch.ones((), dtype=torch.float)
        expected = actual.clone().to(torch.int)

        for fn in assert_close_with_inputs(actual, expected):
            fn(check_dtype=False)

    def test_mismatching_stride(self):
        actual = torch.empty((2, 2))
        expected = torch.as_strided(actual.clone().t().contiguous(), actual.shape, actual.stride()[::-1])

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, "stride"):
                fn(check_stride=True)

    def test_mismatching_stride_no_check(self):
        actual = torch.rand((2, 2))
        expected = torch.as_strided(actual.clone().t().contiguous(), actual.shape, actual.stride()[::-1])
        for fn in assert_close_with_inputs(actual, expected):
            fn()

    def test_only_rtol(self):
        actual = torch.empty(())
        expected = actual.clone()

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaises(ValueError):
                fn(rtol=0.0)

    def test_only_atol(self):
        actual = torch.empty(())
        expected = actual.clone()

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaises(ValueError):
                fn(atol=0.0)

    def test_mismatching_values(self):
        actual = torch.tensor(1)
        expected = torch.tensor(2)

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaises(AssertionError):
                fn()

    def test_mismatching_values_rtol(self):
        eps = 1e-3
        actual = torch.tensor(1.0)
        expected = torch.tensor(1.0 + eps)

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaises(AssertionError):
                fn(rtol=eps / 2, atol=0.0)

    def test_mismatching_values_atol(self):
        eps = 1e-3
        actual = torch.tensor(0.0)
        expected = torch.tensor(eps)

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaises(AssertionError):
                fn(rtol=0.0, atol=eps / 2)

    def test_matching(self):
        actual = torch.tensor(1.0)
        expected = actual.clone()

        torch.testing.assert_close(actual, expected)

    def test_matching_rtol(self):
        eps = 1e-3
        actual = torch.tensor(1.0)
        expected = torch.tensor(1.0 + eps)

        for fn in assert_close_with_inputs(actual, expected):
            fn(rtol=eps * 2, atol=0.0)

    def test_matching_atol(self):
        eps = 1e-3
        actual = torch.tensor(0.0)
        expected = torch.tensor(eps)

        for fn in assert_close_with_inputs(actual, expected):
            fn(rtol=0.0, atol=eps * 2)

    # TODO: the code that this test was designed for was removed in https://github.com/pytorch/pytorch/pull/56058
    #  We need to check if this test is still needed or if this behavior is now enabled by default.
    def test_matching_conjugate_bit(self):
        actual = torch.tensor(complex(1, 1)).conj()
        expected = torch.tensor(complex(1, -1))

        for fn in assert_close_with_inputs(actual, expected):
            fn()

    def test_matching_nan(self):
        nan = float("NaN")

        tests = (
            (nan, nan),
            (complex(nan, 0), complex(0, nan)),
            (complex(nan, nan), complex(nan, 0)),
            (complex(nan, nan), complex(nan, nan)),
        )

        for actual, expected in tests:
            for fn in assert_close_with_inputs(actual, expected):
                with self.assertRaises(AssertionError):
                    fn()

    def test_matching_nan_with_equal_nan(self):
        nan = float("NaN")

        tests = (
            (nan, nan),
            (complex(nan, 0), complex(0, nan)),
            (complex(nan, nan), complex(nan, 0)),
            (complex(nan, nan), complex(nan, nan)),
        )

        for actual, expected in tests:
            for fn in assert_close_with_inputs(actual, expected):
                fn(equal_nan=True)

    def test_numpy(self):
        tensor = torch.rand(2, 2, dtype=torch.float32)
        actual = tensor.numpy()
        expected = actual.copy()

        for fn in assert_close_with_inputs(actual, expected):
            fn()

    def test_scalar(self):
        number = torch.randint(10, size=()).item()
        for actual, expected in itertools.product((int(number), float(number), complex(number)), repeat=2):
            check_dtype = type(actual) is type(expected)

            for fn in assert_close_with_inputs(actual, expected):
                fn(check_dtype=check_dtype)

    def test_bool(self):
        actual = torch.tensor([True, False])
        expected = actual.clone()

        for fn in assert_close_with_inputs(actual, expected):
            fn()

    def test_none(self):
        actual = expected = None

        for fn in assert_close_with_inputs(actual, expected):
            fn()

    def test_none_mismatch(self):
        expected = None

        for actual in (False, 0, torch.nan, torch.tensor(torch.nan)):
            for fn in assert_close_with_inputs(actual, expected):
                with self.assertRaises(AssertionError):
                    fn()


    def test_docstring_examples(self):
        finder = doctest.DocTestFinder(verbose=False)
        runner = doctest.DocTestRunner(verbose=False, optionflags=doctest.NORMALIZE_WHITESPACE)
        globs = dict(torch=torch)
        doctests = finder.find(torch.testing.assert_close, globs=globs)[0]
        failures = []
        runner.run(doctests, out=lambda report: failures.append(report))
        if failures:
            raise AssertionError(f"Doctest found {len(failures)} failures:\n\n" + "\n".join(failures))

    def test_default_tolerance_selection_mismatching_dtypes(self):
        # If the default tolerances where selected based on the promoted dtype, i.e. float64,
        # these tensors wouldn't be considered close.
        actual = torch.tensor(0.99, dtype=torch.bfloat16)
        expected = torch.tensor(1.0, dtype=torch.float64)

        for fn in assert_close_with_inputs(actual, expected):
            fn(check_dtype=False)

    class UnexpectedException(Exception):
        """The only purpose of this exception is to test ``assert_close``'s handling of unexpected exceptions. Thus,
        the test should mock a component to raise this instead of the regular behavior. We avoid using a builtin
        exception here to avoid triggering possible handling of them.
        """
        pass

    @unittest.mock.patch("torch.testing._comparison.TensorLikePair.__init__", side_effect=UnexpectedException)
    def test_unexpected_error_originate(self, _):
        actual = torch.tensor(1.0)
        expected = actual.clone()

        with self.assertRaisesRegex(RuntimeError, "unexpected exception"):
            torch.testing.assert_close(actual, expected)

    @unittest.mock.patch("torch.testing._comparison.TensorLikePair.compare", side_effect=UnexpectedException)
    def test_unexpected_error_compare(self, _):
        actual = torch.tensor(1.0)
        expected = actual.clone()

        with self.assertRaisesRegex(RuntimeError, "unexpected exception"):
            torch.testing.assert_close(actual, expected)




class TestAssertCloseMultiDevice(TestCase):
    @deviceCountAtLeast(1)
    def test_mismatching_device(self, devices):
        for actual_device, expected_device in itertools.permutations(("cpu", *devices), 2):
            actual = torch.empty((), device=actual_device)
            expected = actual.clone().to(expected_device)
            for fn in assert_close_with_inputs(actual, expected):
                with self.assertRaisesRegex(AssertionError, "device"):
                    fn()

    @deviceCountAtLeast(1)
    def test_mismatching_device_no_check(self, devices):
        for actual_device, expected_device in itertools.permutations(("cpu", *devices), 2):
            actual = torch.rand((), device=actual_device)
            expected = actual.clone().to(expected_device)
            for fn in assert_close_with_inputs(actual, expected):
                fn(check_device=False)


instantiate_device_type_tests(TestAssertCloseMultiDevice, globals(), only_for="cuda")


class TestAssertCloseErrorMessage(TestCase):
    def test_identifier_tensor_likes(self):
        actual = torch.tensor([1, 2, 3, 4])
        expected = torch.tensor([1, 2, 5, 6])

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Tensor-likes")):
                fn()

    def test_identifier_scalars(self):
        actual = 3
        expected = 5
        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Scalars")):
                fn()

    def test_not_equal(self):
        actual = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
        expected = torch.tensor([1, 2, 5, 6], dtype=torch.float32)

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("not equal")):
                fn(rtol=0.0, atol=0.0)

    def test_not_close(self):
        actual = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
        expected = torch.tensor([1, 2, 5, 6], dtype=torch.float32)

        for fn, (rtol, atol) in itertools.product(
            assert_close_with_inputs(actual, expected), ((1.3e-6, 0.0), (0.0, 1e-5), (1.3e-6, 1e-5))
        ):
            with self.assertRaisesRegex(AssertionError, re.escape("not close")):
                fn(rtol=rtol, atol=atol)

    def test_mismatched_elements(self):
        actual = torch.tensor([1, 2, 3, 4])
        expected = torch.tensor([1, 2, 5, 6])

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Mismatched elements: 2 / 4 (50.0%)")):
                fn()

    def test_abs_diff(self):
        actual = torch.tensor([[1, 2], [3, 4]])
        expected = torch.tensor([[1, 2], [5, 4]])

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Greatest absolute difference: 2 at index (1, 0)")):
                fn()

    def test_abs_diff_scalar(self):
        actual = 3
        expected = 5

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Absolute difference: 2")):
                fn()

    def test_rel_diff(self):
        actual = torch.tensor([[1, 2], [3, 4]])
        expected = torch.tensor([[1, 4], [3, 4]])

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Greatest relative difference: 0.5 at index (0, 1)")):
                fn()

    def test_rel_diff_scalar(self):
        actual = 2
        expected = 4

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Relative difference: 0.5")):
                fn()

    def test_zero_div_zero(self):
        actual = torch.tensor([1.0, 0.0])
        expected = torch.tensor([2.0, 0.0])

        for fn in assert_close_with_inputs(actual, expected):
            # Although it looks complicated, this regex just makes sure that the word 'nan' is not part of the error
            # message. That would happen if the 0 / 0 is used for the mismatch computation although it matches.
            with self.assertRaisesRegex(AssertionError, "((?!nan).)*"):
                fn()

    def test_rtol(self):
        rtol = 1e-3

        actual = torch.tensor((1, 2))
        expected = torch.tensor((2, 2))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape(f"(up to {rtol} allowed)")):
                fn(rtol=rtol, atol=0.0)

    def test_atol(self):
        atol = 1e-3

        actual = torch.tensor((1, 2))
        expected = torch.tensor((2, 2))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape(f"(up to {atol} allowed)")):
                fn(rtol=0.0, atol=atol)

    def test_msg_str(self):
        msg = "Custom error message!"

        actual = torch.tensor(1)
        expected = torch.tensor(2)

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, msg):
                fn(msg=msg)

    def test_msg_callable(self):
        msg = "Custom error message"

        actual = torch.tensor(1)
        expected = torch.tensor(2)

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, msg):
                fn(msg=lambda _: msg)


class TestAssertCloseContainer(TestCase):
    def test_sequence_mismatching_len(self):
        actual = (torch.empty(()),)
        expected = ()

        with self.assertRaises(AssertionError):
            torch.testing.assert_close(actual, expected)

    def test_sequence_mismatching_values_msg(self):
        t1 = torch.tensor(1)
        t2 = torch.tensor(2)

        actual = (t1, t1)
        expected = (t1, t2)

        with self.assertRaisesRegex(AssertionError, re.escape("item [1]")):
            torch.testing.assert_close(actual, expected)

    def test_mapping_mismatching_keys(self):
        actual = {"a": torch.empty(())}
        expected = {}

        with self.assertRaises(AssertionError):
            torch.testing.assert_close(actual, expected)

    def test_mapping_mismatching_values_msg(self):
        t1 = torch.tensor(1)
        t2 = torch.tensor(2)

        actual = {"a": t1, "b": t1}
        expected = {"a": t1, "b": t2}

        with self.assertRaisesRegex(AssertionError, re.escape("item ['b']")):
            torch.testing.assert_close(actual, expected)


class TestAssertCloseSparseCOO(TestCase):
    def test_matching_coalesced(self):
        indices = (
            (0, 1),
            (1, 0),
        )
        values = (1, 2)
        actual = torch.sparse_coo_tensor(indices, values, size=(2, 2)).coalesce()
        expected = actual.clone()

        for fn in assert_close_with_inputs(actual, expected):
            fn()

    def test_matching_uncoalesced(self):
        indices = (
            (0, 1),
            (1, 0),
        )
        values = (1, 2)
        actual = torch.sparse_coo_tensor(indices, values, size=(2, 2))
        expected = actual.clone()

        for fn in assert_close_with_inputs(actual, expected):
            fn()

    def test_mismatching_sparse_dims(self):
        t = torch.randn(2, 3, 4)
        actual = t.to_sparse()
        expected = t.to_sparse(2)

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("number of sparse dimensions in sparse COO tensors")):
                fn()

    def test_mismatching_nnz(self):
        actual_indices = (
            (0, 1),
            (1, 0),
        )
        actual_values = (1, 2)
        actual = torch.sparse_coo_tensor(actual_indices, actual_values, size=(2, 2))

        expected_indices = (
            (0, 1, 1,),
            (1, 0, 0,),
        )
        expected_values = (1, 1, 1)
        expected = torch.sparse_coo_tensor(expected_indices, expected_values, size=(2, 2))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("number of specified values in sparse COO tensors")):
                fn()

    def test_mismatching_indices_msg(self):
        actual_indices = (
            (0, 1),
            (1, 0),
        )
        actual_values = (1, 2)
        actual = torch.sparse_coo_tensor(actual_indices, actual_values, size=(2, 2))

        expected_indices = (
            (0, 1),
            (1, 1),
        )
        expected_values = (1, 2)
        expected = torch.sparse_coo_tensor(expected_indices, expected_values, size=(2, 2))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse COO indices")):
                fn()

    def test_mismatching_values_msg(self):
        actual_indices = (
            (0, 1),
            (1, 0),
        )
        actual_values = (1, 2)
        actual = torch.sparse_coo_tensor(actual_indices, actual_values, size=(2, 2))

        expected_indices = (
            (0, 1),
            (1, 0),
        )
        expected_values = (1, 3)
        expected = torch.sparse_coo_tensor(expected_indices, expected_values, size=(2, 2))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse COO values")):
                fn()


@unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Not all sandcastle jobs support CSR testing")
class TestAssertCloseSparseCSR(TestCase):
    def test_matching(self):
        crow_indices = (0, 1, 2)
        col_indices = (1, 0)
        values = (1, 2)
        actual = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=(2, 2))
        expected = actual.clone()

        for fn in assert_close_with_inputs(actual, expected):
            fn()

    def test_mismatching_crow_indices_msg(self):
        actual_crow_indices = (0, 1, 2)
        actual_col_indices = (0, 1)
        actual_values = (1, 2)
        actual = torch.sparse_csr_tensor(actual_crow_indices, actual_col_indices, actual_values, size=(2, 2))

        expected_crow_indices = (0, 2, 2)
        expected_col_indices = actual_col_indices
        expected_values = actual_values
        expected = torch.sparse_csr_tensor(expected_crow_indices, expected_col_indices, expected_values, size=(2, 2))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse CSR crow_indices")):
                fn()

    def test_mismatching_col_indices_msg(self):
        actual_crow_indices = (0, 1, 2)
        actual_col_indices = (1, 0)
        actual_values = (1, 2)
        actual = torch.sparse_csr_tensor(actual_crow_indices, actual_col_indices, actual_values, size=(2, 2))

        expected_crow_indices = actual_crow_indices
        expected_col_indices = (1, 1)
        expected_values = actual_values
        expected = torch.sparse_csr_tensor(expected_crow_indices, expected_col_indices, expected_values, size=(2, 2))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse CSR col_indices")):
                fn()

    def test_mismatching_values_msg(self):
        actual_crow_indices = (0, 1, 2)
        actual_col_indices = (1, 0)
        actual_values = (1, 2)
        actual = torch.sparse_csr_tensor(actual_crow_indices, actual_col_indices, actual_values, size=(2, 2))

        expected_crow_indices = actual_crow_indices
        expected_col_indices = actual_col_indices
        expected_values = (1, 3)
        expected = torch.sparse_csr_tensor(expected_crow_indices, expected_col_indices, expected_values, size=(2, 2))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse CSR values")):
                fn()


@unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Not all sandcastle jobs support CSC testing")
class TestAssertCloseSparseCSC(TestCase):
    def test_matching(self):
        ccol_indices = (0, 1, 2)
        row_indices = (1, 0)
        values = (1, 2)
        actual = torch.sparse_csc_tensor(ccol_indices, row_indices, values, size=(2, 2))
        expected = actual.clone()

        for fn in assert_close_with_inputs(actual, expected):
            fn()

    def test_mismatching_ccol_indices_msg(self):
        actual_ccol_indices = (0, 1, 2)
        actual_row_indices = (0, 1)
        actual_values = (1, 2)
        actual = torch.sparse_csc_tensor(actual_ccol_indices, actual_row_indices, actual_values, size=(2, 2))

        expected_ccol_indices = (0, 2, 2)
        expected_row_indices = actual_row_indices
        expected_values = actual_values
        expected = torch.sparse_csc_tensor(expected_ccol_indices, expected_row_indices, expected_values, size=(2, 2))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse CSC ccol_indices")):
                fn()

    def test_mismatching_row_indices_msg(self):
        actual_ccol_indices = (0, 1, 2)
        actual_row_indices = (1, 0)
        actual_values = (1, 2)
        actual = torch.sparse_csc_tensor(actual_ccol_indices, actual_row_indices, actual_values, size=(2, 2))

        expected_ccol_indices = actual_ccol_indices
        expected_row_indices = (1, 1)
        expected_values = actual_values
        expected = torch.sparse_csc_tensor(expected_ccol_indices, expected_row_indices, expected_values, size=(2, 2))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse CSC row_indices")):
                fn()

    def test_mismatching_values_msg(self):
        actual_ccol_indices = (0, 1, 2)
        actual_row_indices = (1, 0)
        actual_values = (1, 2)
        actual = torch.sparse_csc_tensor(actual_ccol_indices, actual_row_indices, actual_values, size=(2, 2))

        expected_ccol_indices = actual_ccol_indices
        expected_row_indices = actual_row_indices
        expected_values = (1, 3)
        expected = torch.sparse_csc_tensor(expected_ccol_indices, expected_row_indices, expected_values, size=(2, 2))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse CSC values")):
                fn()


@unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Not all sandcastle jobs support BSR testing")
class TestAssertCloseSparseBSR(TestCase):
    def test_matching(self):
        crow_indices = (0, 1, 2)
        col_indices = (1, 0)
        values = ([[1]], [[2]])
        actual = torch.sparse_bsr_tensor(crow_indices, col_indices, values, size=(2, 2))
        expected = actual.clone()

        for fn in assert_close_with_inputs(actual, expected):
            fn()

    def test_mismatching_crow_indices_msg(self):
        actual_crow_indices = (0, 1, 2)
        actual_col_indices = (0, 1)
        actual_values = ([[1]], [[2]])
        actual = torch.sparse_bsr_tensor(actual_crow_indices, actual_col_indices, actual_values, size=(2, 2))

        expected_crow_indices = (0, 2, 2)
        expected_col_indices = actual_col_indices
        expected_values = actual_values
        expected = torch.sparse_bsr_tensor(expected_crow_indices, expected_col_indices, expected_values, size=(2, 2))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse BSR crow_indices")):
                fn()

    def test_mismatching_col_indices_msg(self):
        actual_crow_indices = (0, 1, 2)
        actual_col_indices = (1, 0)
        actual_values = ([[1]], [[2]])
        actual = torch.sparse_bsr_tensor(actual_crow_indices, actual_col_indices, actual_values, size=(2, 2))

        expected_crow_indices = actual_crow_indices
        expected_col_indices = (1, 1)
        expected_values = actual_values
        expected = torch.sparse_bsr_tensor(expected_crow_indices, expected_col_indices, expected_values, size=(2, 2))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse BSR col_indices")):
                fn()

    def test_mismatching_values_msg(self):
        actual_crow_indices = (0, 1, 2)
        actual_col_indices = (1, 0)
        actual_values = ([[1]], [[2]])
        actual = torch.sparse_bsr_tensor(actual_crow_indices, actual_col_indices, actual_values, size=(2, 2))

        expected_crow_indices = actual_crow_indices
        expected_col_indices = actual_col_indices
        expected_values = ([[1]], [[3]])
        expected = torch.sparse_bsr_tensor(expected_crow_indices, expected_col_indices, expected_values, size=(2, 2))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse BSR values")):
                fn()


@unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Not all sandcastle jobs support BSC testing")
class TestAssertCloseSparseBSC(TestCase):
    def test_matching(self):
        ccol_indices = (0, 1, 2)
        row_indices = (1, 0)
        values = ([[1]], [[2]])
        actual = torch.sparse_bsc_tensor(ccol_indices, row_indices, values, size=(2, 2))
        expected = actual.clone()

        for fn in assert_close_with_inputs(actual, expected):
            fn()

    def test_mismatching_ccol_indices_msg(self):
        actual_ccol_indices = (0, 1, 2)
        actual_row_indices = (0, 1)
        actual_values = ([[1]], [[2]])
        actual = torch.sparse_bsc_tensor(actual_ccol_indices, actual_row_indices, actual_values, size=(2, 2))

        expected_ccol_indices = (0, 2, 2)
        expected_row_indices = actual_row_indices
        expected_values = actual_values
        expected = torch.sparse_bsc_tensor(expected_ccol_indices, expected_row_indices, expected_values, size=(2, 2))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse BSC ccol_indices")):
                fn()

    def test_mismatching_row_indices_msg(self):
        actual_ccol_indices = (0, 1, 2)
        actual_row_indices = (1, 0)
        actual_values = ([[1]], [[2]])
        actual = torch.sparse_bsc_tensor(actual_ccol_indices, actual_row_indices, actual_values, size=(2, 2))

        expected_ccol_indices = actual_ccol_indices
        expected_row_indices = (1, 1)
        expected_values = actual_values
        expected = torch.sparse_bsc_tensor(expected_ccol_indices, expected_row_indices, expected_values, size=(2, 2))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse BSC row_indices")):
                fn()

    def test_mismatching_values_msg(self):
        actual_ccol_indices = (0, 1, 2)
        actual_row_indices = (1, 0)
        actual_values = ([[1]], [[2]])
        actual = torch.sparse_bsc_tensor(actual_ccol_indices, actual_row_indices, actual_values, size=(2, 2))

        expected_ccol_indices = actual_ccol_indices
        expected_row_indices = actual_row_indices
        expected_values = ([[1]], [[3]])
        expected = torch.sparse_bsc_tensor(expected_ccol_indices, expected_row_indices, expected_values, size=(2, 2))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse BSC values")):
                fn()


class TestAssertCloseQuantized(TestCase):
    def test_mismatching_is_quantized(self):
        actual = torch.tensor(1.0)
        expected = torch.quantize_per_tensor(actual, scale=1.0, zero_point=0, dtype=torch.qint32)

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, "is_quantized"):
                fn()

    def test_mismatching_qscheme(self):
        t = torch.tensor((1.0,))
        actual = torch.quantize_per_tensor(t, scale=1.0, zero_point=0, dtype=torch.qint32)
        expected = torch.quantize_per_channel(
            t,
            scales=torch.tensor((1.0,)),
            zero_points=torch.tensor((0,)),
            axis=0,
            dtype=torch.qint32,
        )

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, "qscheme"):
                fn()

    def test_matching_per_tensor(self):
        actual = torch.quantize_per_tensor(torch.tensor(1.0), scale=1.0, zero_point=0, dtype=torch.qint32)
        expected = actual.clone()

        for fn in assert_close_with_inputs(actual, expected):
            fn()

    def test_matching_per_channel(self):
        actual = torch.quantize_per_channel(
            torch.tensor((1.0,)),
            scales=torch.tensor((1.0,)),
            zero_points=torch.tensor((0,)),
            axis=0,
            dtype=torch.qint32,
        )
        expected = actual.clone()

        for fn in assert_close_with_inputs(actual, expected):
            fn()


class TestMakeTensor(TestCase):
    supported_dtypes = dtypes(
        torch.bool,
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
        torch.float16, torch.bfloat16, torch.float32, torch.float64,
        torch.complex32, torch.complex64, torch.complex128,
    )

    @supported_dtypes
    @parametrize("shape", [tuple(), (0,), (1,), (1, 1), (2,), (2, 3), (8, 16, 32)])
    @parametrize("splat_shape", [False, True])
    def test_smoke(self, dtype, device, shape, splat_shape):
        t = torch.testing.make_tensor(*shape if splat_shape else shape, dtype=dtype, device=device)

        self.assertIsInstance(t, torch.Tensor)
        self.assertEqual(t.shape, shape)
        self.assertEqual(t.dtype, dtype)
        self.assertEqual(t.device, torch.device(device))

    @supported_dtypes
    @parametrize("requires_grad", [False, True])
    def test_requires_grad(self, dtype, device, requires_grad):
        make_tensor = functools.partial(
            torch.testing.make_tensor,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

        if not requires_grad or dtype.is_floating_point or dtype.is_complex:
            t = make_tensor()
            self.assertEqual(t.requires_grad, requires_grad)
        else:
            with self.assertRaisesRegex(
                    ValueError, "`requires_grad=True` is not supported for boolean and integral dtypes"
            ):
                make_tensor()

    @supported_dtypes
    @parametrize("noncontiguous", [False, True])
    @parametrize("shape", [tuple(), (0,), (1,), (1, 1), (2,), (2, 3), (8, 16, 32)])
    def test_noncontiguous(self, dtype, device, noncontiguous, shape):
        numel = functools.reduce(operator.mul, shape, 1)

        t = torch.testing.make_tensor(shape, dtype=dtype, device=device, noncontiguous=noncontiguous)
        self.assertEqual(t.is_contiguous(), not noncontiguous or numel < 2)

    @supported_dtypes
    @parametrize(
        "memory_format_and_shape",
        [
            (None, (2, 3, 4)),
            (torch.contiguous_format, (2, 3, 4)),
            (torch.channels_last, (2, 3, 4, 5)),
            (torch.channels_last_3d, (2, 3, 4, 5, 6)),
            (torch.preserve_format, (2, 3, 4)),
        ],
    )
    def test_memory_format(self, dtype, device, memory_format_and_shape):
        memory_format, shape = memory_format_and_shape

        t = torch.testing.make_tensor(shape, dtype=dtype, device=device, memory_format=memory_format)

        self.assertTrue(
            t.is_contiguous(memory_format=torch.contiguous_format if memory_format is None else memory_format)
        )

    @supported_dtypes
    def test_noncontiguous_memory_format(self, dtype, device):
        with self.assertRaisesRegex(ValueError, "`noncontiguous` and `memory_format` are mutually exclusive"):
            torch.testing.make_tensor(
                (2, 3, 4, 5),
                dtype=dtype,
                device=device,
                noncontiguous=True,
                memory_format=torch.channels_last,
            )

    @supported_dtypes
    def test_exclude_zero(self, dtype, device):
        t = torch.testing.make_tensor(10_000, dtype=dtype, device=device, exclude_zero=True, low=-1, high=2)

        self.assertTrue((t != 0).all())

    @supported_dtypes
    def test_low_high_smoke(self, dtype, device):
        low_inclusive, high_exclusive = 0, 2

        t = torch.testing.make_tensor(10_000, dtype=dtype, device=device, low=low_inclusive, high=high_exclusive)
        if dtype.is_complex:
            t = torch.view_as_real(t)

        self.assertTrue(((t >= low_inclusive) & (t < high_exclusive)).all())

    @supported_dtypes
    def test_low_high_default_smoke(self, dtype, device):
        low_inclusive, high_exclusive = {
            torch.bool: (0, 2),
            torch.uint8: (0, 10),
            **dict.fromkeys([torch.int8, torch.int16, torch.int32, torch.int64], (-9, 10)),
        }.get(dtype, (-9, 9))

        t = torch.testing.make_tensor(10_000, dtype=dtype, device=device, low=low_inclusive, high=high_exclusive)
        if dtype.is_complex:
            t = torch.view_as_real(t)

        self.assertTrue(((t >= low_inclusive) & (t < high_exclusive)).all())

    @parametrize("low_high", [(0, 0), (1, 0), (0, -1)])
    @parametrize("value_types", list(itertools.product([int, float], repeat=2)))
    @supported_dtypes
    def test_low_ge_high(self, dtype, device, low_high, value_types):
        low, high = (value_type(value) for value, value_type in zip(low_high, value_types))

        if low == high and (dtype.is_floating_point or dtype.is_complex):
            with self.assertWarnsRegex(
                    FutureWarning,
                    "Passing `low==high` to `torch.testing.make_tensor` for floating or complex types is deprecated",
            ):
                t = torch.testing.make_tensor(10_000, dtype=dtype, device=device, low=low, high=high)
            self.assertEqual(t, torch.full_like(t, complex(low, low) if dtype.is_complex else low))
        else:
            with self.assertRaisesRegex(ValueError, "`low` must be less than `high`"):
                torch.testing.make_tensor(dtype=dtype, device=device, low=low, high=high)

    @supported_dtypes
    @parametrize("low_high", [(None, torch.nan), (torch.nan, None), (torch.nan, torch.nan)])
    def test_low_high_nan(self, dtype, device, low_high):
        low, high = low_high

        with self.assertRaisesRegex(ValueError, "`low` and `high` cannot be NaN"):
            torch.testing.make_tensor(dtype=dtype, device=device, low=low, high=high)

    @supported_dtypes
    def test_low_high_outside_valid_range(self, dtype, device):
        make_tensor = functools.partial(torch.testing.make_tensor, dtype=dtype, device=device)

        def get_dtype_limits(dtype):
            if dtype is torch.bool:
                return 0, 1

            info = (torch.finfo if dtype.is_floating_point or dtype.is_complex else torch.iinfo)(dtype)
            # We are using integer bounds here, because otherwise it would be impossible to pass `low` and `high`
            # outside their valid range. Python uses 64bit floating point numbers and thus trying to do something like
            # `torch.ffinfo(torch.float64)max * 2` will always result in `inf`. On the flipside, Pythons `int` is
            # unbounded.
            return int(info.min), int(info.max)

        lowest_inclusive, highest_inclusive = get_dtype_limits(dtype)

        with self.assertRaisesRegex(ValueError, ""):
            low, high = (-2, -1) if lowest_inclusive == 0 else (lowest_inclusive * 4, lowest_inclusive * 2)
            make_tensor(low=low, high=high)

        with self.assertRaisesRegex(ValueError, ""):
            make_tensor(low=highest_inclusive * 2, high=highest_inclusive * 4)

    @dtypes(torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    def test_low_high_boolean_integral1(self, dtype, device):
        shape = (10_000,)
        eps = 1e-4

        actual = torch.testing.make_tensor(shape, dtype=dtype, device=device, low=-(1 - eps), high=1 - eps)
        expected = torch.zeros(shape, dtype=dtype, device=device)

        torch.testing.assert_close(actual, expected)

    @dtypes(torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    def test_low_high_boolean_integral2(self, dtype, device):
        shape = (10_000,)
        if dtype is torch.bool:
            low = 1
        elif dtype is torch.int64:
            # Due to its internals, `make_tensor` is not able to sample `torch.iinfo(torch.int64).max`
            low = torch.iinfo(dtype).max - 1
        else:
            low = torch.iinfo(dtype).max
        high = low + 1

        actual = torch.testing.make_tensor(shape, dtype=dtype, device=device, low=low, high=high)
        expected = torch.full(shape, low, dtype=dtype, device=device)

        torch.testing.assert_close(actual, expected)


instantiate_device_type_tests(TestMakeTensor, globals())


def _get_test_names_for_test_class(test_cls):
    """ Convenience function to get all test names for a given test class. """
    test_names = [f'{test_cls.__name__}.{key}' for key in test_cls.__dict__
                  if key.startswith('test_')]
    return sorted(test_names)


def _get_test_funcs_for_test_class(test_cls):
    """ Convenience function to get all (test function, parametrized_name) pairs for a given test class. """
    test_funcs = [(getattr(test_cls, key), key) for key in test_cls.__dict__ if key.startswith('test_')]
    return test_funcs


class TestTestParametrization(TestCase):
    def test_default_names(self):

        class TestParametrized(TestCase):
            @parametrize("x", range(5))
            def test_default_names(self, x):
                pass

            @parametrize("x,y", [(1, 2), (2, 3), (3, 4)])
            def test_two_things_default_names(self, x, y):
                pass

        instantiate_parametrized_tests(TestParametrized)

        expected_test_names = [
            'TestParametrized.test_default_names_x_0',
            'TestParametrized.test_default_names_x_1',
            'TestParametrized.test_default_names_x_2',
            'TestParametrized.test_default_names_x_3',
            'TestParametrized.test_default_names_x_4',
            'TestParametrized.test_two_things_default_names_x_1_y_2',
            'TestParametrized.test_two_things_default_names_x_2_y_3',
            'TestParametrized.test_two_things_default_names_x_3_y_4',
        ]
        test_names = _get_test_names_for_test_class(TestParametrized)
        self.assertEqual(expected_test_names, test_names)

    def test_name_fn(self):

        class TestParametrized(TestCase):
            @parametrize("bias", [False, True], name_fn=lambda b: 'bias' if b else 'no_bias')
            def test_custom_names(self, bias):
                pass

            @parametrize("x", [1, 2], name_fn=str)
            @parametrize("y", [3, 4], name_fn=str)
            @parametrize("z", [5, 6], name_fn=str)
            def test_three_things_composition_custom_names(self, x, y, z):
                pass

            @parametrize("x,y", [(1, 2), (1, 3), (1, 4)], name_fn=lambda x, y: f'{x}__{y}')
            def test_two_things_custom_names_alternate(self, x, y):
                pass

        instantiate_parametrized_tests(TestParametrized)

        expected_test_names = [
            'TestParametrized.test_custom_names_bias',
            'TestParametrized.test_custom_names_no_bias',
            'TestParametrized.test_three_things_composition_custom_names_1_3_5',
            'TestParametrized.test_three_things_composition_custom_names_1_3_6',
            'TestParametrized.test_three_things_composition_custom_names_1_4_5',
            'TestParametrized.test_three_things_composition_custom_names_1_4_6',
            'TestParametrized.test_three_things_composition_custom_names_2_3_5',
            'TestParametrized.test_three_things_composition_custom_names_2_3_6',
            'TestParametrized.test_three_things_composition_custom_names_2_4_5',
            'TestParametrized.test_three_things_composition_custom_names_2_4_6',
            'TestParametrized.test_two_things_custom_names_alternate_1__2',
            'TestParametrized.test_two_things_custom_names_alternate_1__3',
            'TestParametrized.test_two_things_custom_names_alternate_1__4',
        ]
        test_names = _get_test_names_for_test_class(TestParametrized)
        self.assertEqual(expected_test_names, test_names)

    def test_subtest_names(self):

        class TestParametrized(TestCase):
            @parametrize("bias", [subtest(True, name='bias'),
                                  subtest(False, name='no_bias')])
            def test_custom_names(self, bias):
                pass

            @parametrize("x,y", [subtest((1, 2), name='double'),
                                 subtest((1, 3), name='triple'),
                                 subtest((1, 4), name='quadruple')])
            def test_two_things_custom_names(self, x, y):
                pass

        instantiate_parametrized_tests(TestParametrized)

        expected_test_names = [
            'TestParametrized.test_custom_names_bias',
            'TestParametrized.test_custom_names_no_bias',
            'TestParametrized.test_two_things_custom_names_double',
            'TestParametrized.test_two_things_custom_names_quadruple',
            'TestParametrized.test_two_things_custom_names_triple',
        ]
        test_names = _get_test_names_for_test_class(TestParametrized)
        self.assertEqual(expected_test_names, test_names)

    def test_apply_param_specific_decorators(self):
        # Test that decorators can be applied on a per-param basis.

        def test_dec(func):
            func._decorator_applied = True
            return func

        class TestParametrized(TestCase):
            @parametrize("x", [subtest(1, name='one'),
                               subtest(2, name='two', decorators=[test_dec]),
                               subtest(3, name='three')])
            def test_param(self, x):
                pass

        instantiate_parametrized_tests(TestParametrized)

        for test_func, name in _get_test_funcs_for_test_class(TestParametrized):
            self.assertEqual(hasattr(test_func, '_decorator_applied'), name == 'test_param_two')

    def test_compose_param_specific_decorators(self):
        # Test that multiple per-param decorators compose correctly.

        def test_dec(func):
            func._decorator_applied = True
            return func

        class TestParametrized(TestCase):
            @parametrize("x", [subtest(1),
                               subtest(2, decorators=[test_dec]),
                               subtest(3)])
            @parametrize("y", [subtest(False, decorators=[test_dec]),
                               subtest(True)])
            def test_param(self, x, y):
                pass

        instantiate_parametrized_tests(TestParametrized)

        for test_func, name in _get_test_funcs_for_test_class(TestParametrized):
            # Decorator should be applied whenever either x == 2 or y == False.
            should_apply = ('x_2' in name) or ('y_False' in name)
            self.assertEqual(hasattr(test_func, '_decorator_applied'), should_apply)

    def test_modules_decorator_misuse_error(self):
        # Test that @modules errors out when used with instantiate_parametrized_tests().

        class TestParametrized(TestCase):
            @modules(module_db)
            def test_modules(self, module_info):
                pass

        with self.assertRaisesRegex(RuntimeError, 'intended to be used in a device-specific context'):
            instantiate_parametrized_tests(TestParametrized)

    def test_ops_decorator_misuse_error(self):
        # Test that @ops errors out when used with instantiate_parametrized_tests().

        class TestParametrized(TestCase):
            @ops(op_db)
            def test_ops(self, module_info):
                pass

        with self.assertRaisesRegex(RuntimeError, 'intended to be used in a device-specific context'):
            instantiate_parametrized_tests(TestParametrized)

    def test_multiple_handling_of_same_param_error(self):
        # Test that multiple decorators handling the same param errors out.

        class TestParametrized(TestCase):
            @parametrize("x", range(3))
            @parametrize("x", range(5))
            def test_param(self, x):
                pass

        with self.assertRaisesRegex(RuntimeError, 'multiple parametrization decorators'):
            instantiate_parametrized_tests(TestParametrized)

    @parametrize("x", [1, subtest(2, decorators=[unittest.expectedFailure]), 3])
    def test_subtest_expected_failure(self, x):
        if x == 2:
            raise RuntimeError('Boom')

    @parametrize("x", [subtest(1, decorators=[unittest.expectedFailure]), 2, 3])
    @parametrize("y", [4, 5, subtest(6, decorators=[unittest.expectedFailure])])
    def test_two_things_subtest_expected_failure(self, x, y):
        if x == 1 or y == 6:
            raise RuntimeError('Boom')


class TestTestParametrizationDeviceType(TestCase):
    def test_unparametrized_names(self, device):
        # This test exists to protect against regressions in device / dtype test naming
        # due to parametrization logic.

        device = self.device_type

        class TestParametrized(TestCase):
            def test_device_specific(self, device):
                pass

            @dtypes(torch.float32, torch.float64)
            def test_device_dtype_specific(self, device, dtype):
                pass

        instantiate_device_type_tests(TestParametrized, locals(), only_for=device)

        device_cls = locals()[f'TestParametrized{device.upper()}']
        expected_test_names = [name.format(device_cls.__name__, device) for name in (
            '{}.test_device_dtype_specific_{}_float32',
            '{}.test_device_dtype_specific_{}_float64',
            '{}.test_device_specific_{}')
        ]
        test_names = _get_test_names_for_test_class(device_cls)
        self.assertEqual(expected_test_names, test_names)

    def test_empty_param_names(self, device):
        # If no param names are passed, ensure things still work without parametrization.
        device = self.device_type

        class TestParametrized(TestCase):
            @parametrize("", [])
            def test_foo(self, device):
                pass

            @parametrize("", range(5))
            def test_bar(self, device):
                pass

        instantiate_device_type_tests(TestParametrized, locals(), only_for=device)

        device_cls = locals()[f'TestParametrized{device.upper()}']
        expected_test_names = [name.format(device_cls.__name__, device) for name in (
            '{}.test_bar_{}',
            '{}.test_foo_{}')
        ]
        test_names = _get_test_names_for_test_class(device_cls)
        self.assertEqual(expected_test_names, test_names)

    def test_empty_param_list(self, device):
        # If no param values are passed, ensure a helpful error message is thrown.
        # In the wild, this could indicate reuse of an exhausted generator.
        device = self.device_type

        generator = (a for a in range(5))

        class TestParametrized(TestCase):
            @parametrize("x", generator)
            def test_foo(self, device, x):
                pass

            # Reuse generator from first test function.
            @parametrize("y", generator)
            def test_bar(self, device, y):
                pass

        with self.assertRaisesRegex(ValueError, 'An empty arg_values was passed'):
            instantiate_device_type_tests(TestParametrized, locals(), only_for=device)

    def test_default_names(self, device):
        device = self.device_type

        class TestParametrized(TestCase):
            @parametrize("x", range(5))
            def test_default_names(self, device, x):
                pass

            @parametrize("x,y", [(1, 2), (2, 3), (3, 4)])
            def test_two_things_default_names(self, device, x, y):
                pass


        instantiate_device_type_tests(TestParametrized, locals(), only_for=device)

        device_cls = locals()[f'TestParametrized{device.upper()}']
        expected_test_names = [name.format(device_cls.__name__, device) for name in (
            '{}.test_default_names_x_0_{}',
            '{}.test_default_names_x_1_{}',
            '{}.test_default_names_x_2_{}',
            '{}.test_default_names_x_3_{}',
            '{}.test_default_names_x_4_{}',
            '{}.test_two_things_default_names_x_1_y_2_{}',
            '{}.test_two_things_default_names_x_2_y_3_{}',
            '{}.test_two_things_default_names_x_3_y_4_{}')
        ]
        test_names = _get_test_names_for_test_class(device_cls)
        self.assertEqual(expected_test_names, test_names)

    def test_default_name_non_primitive(self, device):
        device = self.device_type

        class TestParametrized(TestCase):
            @parametrize("x", [1, .5, "foo", object()])
            def test_default_names(self, device, x):
                pass

            @parametrize("x,y", [(1, object()), (object(), .5), (object(), object())])
            def test_two_things_default_names(self, device, x, y):
                pass

        instantiate_device_type_tests(TestParametrized, locals(), only_for=device)

        device_cls = locals()[f'TestParametrized{device.upper()}']
        expected_test_names = sorted(name.format(device_cls.__name__, device) for name in (
            '{}.test_default_names_x_1_{}',
            '{}.test_default_names_x_0_5_{}',
            '{}.test_default_names_x_foo_{}',
            '{}.test_default_names_x3_{}',
            '{}.test_two_things_default_names_x_1_y0_{}',
            '{}.test_two_things_default_names_x1_y_0_5_{}',
            '{}.test_two_things_default_names_x2_y2_{}')
        )
        test_names = _get_test_names_for_test_class(device_cls)
        self.assertEqual(expected_test_names, test_names)

    def test_name_fn(self, device):
        device = self.device_type

        class TestParametrized(TestCase):
            @parametrize("bias", [False, True], name_fn=lambda b: 'bias' if b else 'no_bias')
            def test_custom_names(self, device, bias):
                pass

            @parametrize("x", [1, 2], name_fn=str)
            @parametrize("y", [3, 4], name_fn=str)
            @parametrize("z", [5, 6], name_fn=str)
            def test_three_things_composition_custom_names(self, device, x, y, z):
                pass

            @parametrize("x,y", [(1, 2), (1, 3), (1, 4)], name_fn=lambda x, y: f'{x}__{y}')
            def test_two_things_custom_names_alternate(self, device, x, y):
                pass

        instantiate_device_type_tests(TestParametrized, locals(), only_for=device)

        device_cls = locals()[f'TestParametrized{device.upper()}']
        expected_test_names = [name.format(device_cls.__name__, device) for name in (
            '{}.test_custom_names_bias_{}',
            '{}.test_custom_names_no_bias_{}',
            '{}.test_three_things_composition_custom_names_1_3_5_{}',
            '{}.test_three_things_composition_custom_names_1_3_6_{}',
            '{}.test_three_things_composition_custom_names_1_4_5_{}',
            '{}.test_three_things_composition_custom_names_1_4_6_{}',
            '{}.test_three_things_composition_custom_names_2_3_5_{}',
            '{}.test_three_things_composition_custom_names_2_3_6_{}',
            '{}.test_three_things_composition_custom_names_2_4_5_{}',
            '{}.test_three_things_composition_custom_names_2_4_6_{}',
            '{}.test_two_things_custom_names_alternate_1__2_{}',
            '{}.test_two_things_custom_names_alternate_1__3_{}',
            '{}.test_two_things_custom_names_alternate_1__4_{}')
        ]
        test_names = _get_test_names_for_test_class(device_cls)
        self.assertEqual(expected_test_names, test_names)

    def test_subtest_names(self, device):
        device = self.device_type

        class TestParametrized(TestCase):
            @parametrize("bias", [subtest(True, name='bias'),
                                  subtest(False, name='no_bias')])
            def test_custom_names(self, device, bias):
                pass

            @parametrize("x,y", [subtest((1, 2), name='double'),
                                 subtest((1, 3), name='triple'),
                                 subtest((1, 4), name='quadruple')])
            def test_two_things_custom_names(self, device, x, y):
                pass

        instantiate_device_type_tests(TestParametrized, locals(), only_for=device)

        device_cls = locals()[f'TestParametrized{device.upper()}']
        expected_test_names = [name.format(device_cls.__name__, device) for name in (
            '{}.test_custom_names_bias_{}',
            '{}.test_custom_names_no_bias_{}',
            '{}.test_two_things_custom_names_double_{}',
            '{}.test_two_things_custom_names_quadruple_{}',
            '{}.test_two_things_custom_names_triple_{}')
        ]
        test_names = _get_test_names_for_test_class(device_cls)
        self.assertEqual(expected_test_names, test_names)

    def test_ops_composition_names(self, device):
        device = self.device_type

        class TestParametrized(TestCase):
            @ops(op_db)
            @parametrize("flag", [False, True], lambda f: 'flag_enabled' if f else 'flag_disabled')
            def test_op_parametrized(self, device, dtype, op, flag):
                pass

        instantiate_device_type_tests(TestParametrized, locals(), only_for=device)

        device_cls = locals()[f'TestParametrized{device.upper()}']
        expected_test_names = []
        for op in op_db:
            for dtype in op.supported_dtypes(torch.device(device).type):
                for flag_part in ('flag_disabled', 'flag_enabled'):
                    expected_name = f'{device_cls.__name__}.test_op_parametrized_{op.formatted_name}_{flag_part}_{device}_{dtype_name(dtype)}'  # noqa: B950
                    expected_test_names.append(expected_name)

        test_names = _get_test_names_for_test_class(device_cls)
        self.assertEqual(sorted(expected_test_names), sorted(test_names))

    def test_modules_composition_names(self, device):
        device = self.device_type

        class TestParametrized(TestCase):
            @modules(module_db)
            @parametrize("flag", [False, True], lambda f: 'flag_enabled' if f else 'flag_disabled')
            def test_module_parametrized(self, device, dtype, module_info, training, flag):
                pass

        instantiate_device_type_tests(TestParametrized, locals(), only_for=device)

        device_cls = locals()[f'TestParametrized{device.upper()}']
        expected_test_names = []
        for module_info in module_db:
            for dtype in module_info.dtypes:
                for flag_part in ('flag_disabled', 'flag_enabled'):
                    expected_train_modes = (
                        ['train_mode', 'eval_mode'] if module_info.train_and_eval_differ else [''])
                    for training_part in expected_train_modes:
                        expected_name = '{}.test_module_parametrized_{}{}_{}_{}_{}'.format(
                            device_cls.__name__, module_info.formatted_name,
                            '_' + training_part if len(training_part) > 0 else '',
                            flag_part, device, dtype_name(dtype))
                        expected_test_names.append(expected_name)

        test_names = _get_test_names_for_test_class(device_cls)
        self.assertEqual(sorted(expected_test_names), sorted(test_names))

    def test_ops_decorator_applies_op_and_param_specific_decorators(self, device):
        # Test that decorators can be applied on a per-op / per-param basis.

        # Create a test op, OpInfo entry, and decorator to apply.
        def test_op(x):
            return -x

        def test_dec(func):
            func._decorator_applied = True
            return func

        test_op_info = OpInfo(
            'test_op',
            op=test_op,
            dtypes=floating_types(),
            sample_inputs_func=lambda _: [],
            decorators=[
                DecorateInfo(test_dec, 'TestParametrized', 'test_op_param',
                             device_type='cpu', dtypes=[torch.float64],
                             active_if=lambda p: p['x'] == 2)
            ])

        class TestParametrized(TestCase):
            @ops(op_db + [test_op_info])
            @parametrize("x", [2, 3])
            def test_op_param(self, device, dtype, op, x):
                pass

            @ops(op_db + [test_op_info])
            @parametrize("y", [
                subtest(4),
                subtest(5, decorators=[test_dec])])
            def test_other(self, device, dtype, op, y):
                pass

            @decorateIf(test_dec, lambda p: p['dtype'] == torch.int16)
            @ops(op_db)
            def test_three(self, device, dtype, op):
                pass

        device = self.device_type
        instantiate_device_type_tests(TestParametrized, locals(), only_for=device)
        device_cls = locals()[f'TestParametrized{device.upper()}']

        for test_func, name in _get_test_funcs_for_test_class(device_cls):
            should_apply = (name == 'test_op_param_test_op_x_2_cpu_float64' or
                            ('test_other' in name and 'y_5' in name) or
                            ('test_three' in name and name.endswith('_int16')))
            self.assertEqual(hasattr(test_func, '_decorator_applied'), should_apply)

    def test_modules_decorator_applies_module_and_param_specific_decorators(self, device):
        # Test that decorators can be applied on a per-module / per-param basis.

        # Create a test module, ModuleInfo entry, and decorator to apply.
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.x = torch.nn.Parameter(torch.randn(3))

            def forward(self, y):
                return self.x + y

        def test_dec(func):
            func._decorator_applied = True
            return func

        test_module_info = ModuleInfo(
            TestModule,
            module_inputs_func=lambda _: [],
            decorators=[
                DecorateInfo(test_dec, 'TestParametrized', 'test_module_param',
                             device_type='cpu', dtypes=[torch.float64],
                             active_if=lambda p: p['x'] == 2)
            ])

        class TestParametrized(TestCase):
            @modules(module_db + [test_module_info])
            @parametrize("x", [2, 3])
            def test_module_param(self, device, dtype, module_info, training, x):
                pass

            @modules(module_db + [test_module_info])
            @parametrize("y", [
                subtest(4),
                subtest(5, decorators=[test_dec])])
            def test_other(self, device, dtype, module_info, training, y):
                pass

            @decorateIf(test_dec, lambda p: p['dtype'] == torch.float64)
            @modules(module_db)
            def test_three(self, device, dtype, module_info):
                pass

        device = self.device_type
        instantiate_device_type_tests(TestParametrized, locals(), only_for=device)
        device_cls = locals()[f'TestParametrized{device.upper()}']

        for test_func, name in _get_test_funcs_for_test_class(device_cls):
            should_apply = (name == 'test_module_param_TestModule_x_2_cpu_float64' or
                            ('test_other' in name and 'y_5' in name) or
                            ('test_three' in name and name.endswith('float64')))
            self.assertEqual(hasattr(test_func, '_decorator_applied'), should_apply)

    def test_param_specific_decoration(self, device):

        def test_dec(func):
            func._decorator_applied = True
            return func

        class TestParametrized(TestCase):
            @decorateIf(test_dec, lambda params: params["x"] == 1 and params["y"])
            @parametrize("x", range(5))
            @parametrize("y", [False, True])
            def test_param(self, x, y):
                pass

        device = self.device_type
        instantiate_device_type_tests(TestParametrized, locals(), only_for=device)
        device_cls = locals()[f'TestParametrized{device.upper()}']

        for test_func, name in _get_test_funcs_for_test_class(device_cls):
            should_apply = ('test_param_x_1_y_True' in name)
            self.assertEqual(hasattr(test_func, '_decorator_applied'), should_apply)

    def test_dtypes_composition_valid(self, device):
        # Test checks that @parametrize and @dtypes compose as expected when @parametrize
        # doesn't set dtype.

        device = self.device_type

        class TestParametrized(TestCase):
            @dtypes(torch.float32, torch.float64)
            @parametrize("x", range(3))
            def test_parametrized(self, x, dtype):
                pass

        instantiate_device_type_tests(TestParametrized, locals(), only_for=device)

        device_cls = locals()[f'TestParametrized{device.upper()}']
        expected_test_names = [name.format(device_cls.__name__, device) for name in (
            '{}.test_parametrized_x_0_{}_float32',
            '{}.test_parametrized_x_0_{}_float64',
            '{}.test_parametrized_x_1_{}_float32',
            '{}.test_parametrized_x_1_{}_float64',
            '{}.test_parametrized_x_2_{}_float32',
            '{}.test_parametrized_x_2_{}_float64')
        ]
        test_names = _get_test_names_for_test_class(device_cls)
        self.assertEqual(sorted(expected_test_names), sorted(test_names))

    def test_dtypes_composition_invalid(self, device):
        # Test checks that @dtypes cannot be composed with parametrization decorators when they
        # also try to set dtype.

        device = self.device_type

        class TestParametrized(TestCase):
            @dtypes(torch.float32, torch.float64)
            @parametrize("dtype", [torch.int32, torch.int64])
            def test_parametrized(self, dtype):
                pass

        with self.assertRaisesRegex(RuntimeError, "handled multiple times"):
            instantiate_device_type_tests(TestParametrized, locals(), only_for=device)

        # Verify proper error behavior with @ops + @dtypes, as both try to set dtype.

        class TestParametrized(TestCase):
            @dtypes(torch.float32, torch.float64)
            @ops(op_db)
            def test_parametrized(self, op, dtype):
                pass

        with self.assertRaisesRegex(RuntimeError, "handled multiple times"):
            instantiate_device_type_tests(TestParametrized, locals(), only_for=device)

    def test_multiple_handling_of_same_param_error(self, device):
        # Test that multiple decorators handling the same param errors out.
        # Both @modules and @ops handle the dtype param.

        class TestParametrized(TestCase):
            @ops(op_db)
            @modules(module_db)
            def test_param(self, device, dtype, op, module_info, training):
                pass

        with self.assertRaisesRegex(RuntimeError, "handled multiple times"):
            instantiate_device_type_tests(TestParametrized, locals(), only_for=device)

    @parametrize("x", [1, subtest(2, decorators=[unittest.expectedFailure]), 3])
    def test_subtest_expected_failure(self, device, x):
        if x == 2:
            raise RuntimeError('Boom')

    @parametrize("x", [subtest(1, decorators=[unittest.expectedFailure]), 2, 3])
    @parametrize("y", [4, 5, subtest(6, decorators=[unittest.expectedFailure])])
    def test_two_things_subtest_expected_failure(self, device, x, y):
        if x == 1 or y == 6:
            raise RuntimeError('Boom')


instantiate_parametrized_tests(TestTestParametrization)
instantiate_device_type_tests(TestTestParametrizationDeviceType, globals())


class TestImports(TestCase):
    @classmethod
    def _check_python_output(cls, program) -> str:
        return subprocess.check_output(
            [sys.executable, "-W", "always", "-c", program],
            stderr=subprocess.STDOUT,
            # On Windows, opening the subprocess with the default CWD makes `import torch`
            # fail, so just set CWD to this script's directory
            cwd=os.path.dirname(os.path.realpath(__file__)),).decode("utf-8")

    def test_circular_dependencies(self) -> None:
        """ Checks that all modules inside torch can be imported
        Prevents regression reported in https://github.com/pytorch/pytorch/issues/77441 """
        ignored_modules = ["torch.utils.tensorboard",  # deps on tensorboard
                           "torch.distributed.elastic.rendezvous",  # depps on etcd
                           "torch.backends._coreml",  # depends on pycoreml
                           "torch.contrib.",  # something weird
                           "torch.testing._internal.distributed.",  # just fails
                           "torch.ao.pruning._experimental.",  # depends on pytorch_lightning, not user-facing
                           "torch.onnx._internal.fx",  # depends on onnx-script
                           "torch._inductor.runtime.triton_helpers",  # depends on triton
                           "torch._inductor.codegen.cuda",  # depends on cutlass
                           ]
        # See https://github.com/pytorch/pytorch/issues/77801
        if not sys.version_info >= (3, 9):
            ignored_modules.append("torch.utils.benchmark")
        if IS_WINDOWS or IS_MACOS or IS_JETSON:
            # Distributed should be importable on Windows(except nn.api.), but not on Mac
            if IS_MACOS or IS_JETSON:
                ignored_modules.append("torch.distributed.")
            else:
                ignored_modules.append("torch.distributed.nn.api.")
                ignored_modules.append("torch.distributed.optim.")
                ignored_modules.append("torch.distributed.rpc.")
            ignored_modules.append("torch.testing._internal.dist_utils")
            # And these both end up with transitive dependencies on distributed
            ignored_modules.append("torch.nn.parallel._replicated_tensor_ddp_interop")
            ignored_modules.append("torch.testing._internal.common_fsdp")
            ignored_modules.append("torch.testing._internal.common_distributed")

        torch_dir = os.path.dirname(torch.__file__)
        for base, folders, files in os.walk(torch_dir):
            prefix = os.path.relpath(base, os.path.dirname(torch_dir)).replace(os.path.sep, ".")
            for f in files:
                if not f.endswith(".py"):
                    continue
                mod_name = f"{prefix}.{f[:-3]}" if f != "__init__.py" else prefix
                # Do not attempt to import executable modules
                if f == "__main__.py":
                    continue
                if any(mod_name.startswith(x) for x in ignored_modules):
                    continue
                try:
                    mod = importlib.import_module(mod_name)
                except Exception as e:
                    raise RuntimeError(f"Failed to import {mod_name}: {e}") from e
                self.assertTrue(inspect.ismodule(mod))

    @unittest.skipIf(IS_WINDOWS, "TODO enable on Windows")
    def test_lazy_imports_are_lazy(self) -> None:
        out = self._check_python_output("import sys;import torch;print(all(x not in sys.modules for x in torch._lazy_modules))")
        self.assertEqual(out.strip(), "True")

    @unittest.skipIf(IS_WINDOWS, "importing torch+CUDA on CPU results in warning")
    def test_no_warning_on_import(self) -> None:
        out = self._check_python_output("import torch")
        self.assertEqual(out, "")

    def test_not_import_sympy(self) -> None:
        out = self._check_python_output("import torch;import sys;print('sympy' not in sys.modules)")
        self.assertEqual(out.strip(), "True",
                         "PyTorch should not depend on SymPy at import time as importing SymPy is *very* slow.\n"
                         "See the beginning of the following blog post for how to profile and find which file is importing sympy:\n"
                         "https://dev-discuss.pytorch.org/t/delving-into-what-happens-when-you-import-torch/1589\n\n"
                         "If you hit this error, you may want to:\n"
                         "  - Refactor your code to avoid depending on sympy files you may not need to depend\n"
                         "  - Use TYPE_CHECKING if you are using sympy + strings if you are using sympy on type annotations\n"
                         "  - Import things that depend on SymPy locally")

    @unittest.skipIf(IS_WINDOWS, "importing torch+CUDA on CPU results in warning")
    @parametrize('path', ['torch', 'functorch'])
    def test_no_mutate_global_logging_on_import(self, path) -> None:
        # Calling logging.basicConfig, among other things, modifies the global
        # logging state. It is not OK to modify the global logging state on
        # `import torch` (or other submodules we own) because users do not expect it.
        expected = 'abcdefghijklmnopqrstuvwxyz'
        commands = [
            'import logging',
            f'import {path}',
            '_logger = logging.getLogger("torch_test_testing")',
            'logging.root.addHandler(logging.StreamHandler())',
            'logging.root.setLevel(logging.INFO)',
            f'_logger.info("{expected}")'
        ]
        out = self._check_python_output("; ".join(commands))
        self.assertEqual(out.strip(), expected)

class TestOpInfos(TestCase):
    def test_sample_input(self) -> None:
        a, b, c, d, e = (object() for _ in range(5))

        # Construction with natural syntax
        s = SampleInput(a, b, c, d=d, e=e)
        assert s.input is a
        assert s.args == (b, c)
        assert s.kwargs == dict(d=d, e=e)

        # Construction with explicit args and kwargs
        s = SampleInput(a, args=(b,), kwargs=dict(c=c, d=d, e=e))
        assert s.input is a
        assert s.args == (b,)
        assert s.kwargs == dict(c=c, d=d, e=e)

        # Construction with a mixed form will error
        with self.assertRaises(AssertionError):
            s = SampleInput(a, b, c, args=(d, e))

        with self.assertRaises(AssertionError):
            s = SampleInput(a, b, c, kwargs=dict(d=d, e=e))

        with self.assertRaises(AssertionError):
            s = SampleInput(a, args=(b, c), d=d, e=e)

        with self.assertRaises(AssertionError):
            s = SampleInput(a, b, c=c, kwargs=dict(d=d, e=e))

        # Mixing metadata into "natural" construction will error
        with self.assertRaises(AssertionError):
            s = SampleInput(a, b, name="foo")

        with self.assertRaises(AssertionError):
            s = SampleInput(a, b, output_process_fn_grad=lambda x: x)

        with self.assertRaises(AssertionError):
            s = SampleInput(a, b, broadcasts_input=True)

        # But when only input is given, metadata is allowed for backward
        # compatibility
        s = SampleInput(a, broadcasts_input=True)
        assert s.input is a
        assert s.broadcasts_input

    def test_sample_input_metadata(self) -> None:
        a, b = (object() for _ in range(2))
        s1 = SampleInput(a, b=b)
        self.assertIs(s1.output_process_fn_grad(None), None)
        self.assertFalse(s1.broadcasts_input)
        self.assertEqual(s1.name, "")

        s2 = s1.with_metadata(
            output_process_fn_grad=lambda x: a,
            broadcasts_input=True,
            name="foo",
        )
        self.assertIs(s1, s2)
        self.assertIs(s2.output_process_fn_grad(None), a)
        self.assertTrue(s2.broadcasts_input)
        self.assertEqual(s2.name, "foo")


# Tests that validate the various sample generating functions on each OpInfo.
class TestOpInfoSampleFunctions(TestCase):

    @ops(op_db, dtypes=OpDTypes.any_one)
    def test_opinfo_sample_generators(self, device, dtype, op):
        # Test op.sample_inputs doesn't generate multiple samples when called
        samples = op.sample_inputs(device, dtype)
        self.assertIsInstance(samples, Iterator)

    @ops([op for op in op_db if op.reference_inputs_func is not None], dtypes=OpDTypes.any_one)
    def test_opinfo_reference_generators(self, device, dtype, op):
        # Test op.reference_inputs doesn't generate multiple samples when called
        samples = op.reference_inputs(device, dtype)
        self.assertIsInstance(samples, Iterator)

    @ops([op for op in op_db if op.error_inputs_func is not None], dtypes=OpDTypes.none)
    def test_opinfo_error_generators(self, device, op):
        # Test op.error_inputs doesn't generate multiple inputs when called
        samples = op.error_inputs(device)
        self.assertIsInstance(samples, Iterator)


instantiate_device_type_tests(TestOpInfoSampleFunctions, globals())
instantiate_parametrized_tests(TestImports)


if __name__ == '__main__':
    run_tests()
