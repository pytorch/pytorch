# Owner(s): ["module: tests"]

import collections
import dataclasses
import doctest
import functools
import importlib
import importlib.metadata
import inspect
import io
import itertools
import json
import math
import os
import platform
import re
import subprocess
import sys
import sysconfig
import tempfile
import textwrap
import time
import unittest.mock
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from collections.abc import Callable
from collections.abc import Iterator

import torch

from torch.testing import make_tensor
from torch.testing._internal.common_utils import (
    IS_CI, IS_FBCODE, IS_JETSON, IS_MACOS, IS_SANDCASTLE, IS_WINDOWS, TestCase, run_tests, slowTest,
    parametrize, reparametrize, subtest, instantiate_parametrized_tests, dtype_name,
    TEST_CUDA, TEST_WITH_CROSSREF, TEST_WITH_PERIODIC, TEST_WITH_ROCM, decorateIf, periodic, skipIfTorchDynamo, skipIfXpu,
    getRocmVersion, TemporaryFileName, sanitize_pytest_xml, TestEnvironment,
)
from torch.testing._internal.common_cuda import _get_torch_rocm_version, has_device_side_assert
from torch.testing._internal.common_device_type import \
    (PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY, PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY, dtypes,
     get_device_type_test_bases, instantiate_device_type_tests, onlyCPU, onlyCUDA, onlyNativeDeviceTypes,
     deviceCountAtLeast, ops, expectedFailureMeta, OpDTypes)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal import common_cuda, opinfo
from torch.testing._internal.common_dtype import all_types_and_complex_and, floating_types
from torch.testing._internal.common_modules import modules, module_db, ModuleInfo
from torch.testing._internal.opinfo.core import SampleInput, DecorateInfo, OpInfo
from torch.testing._internal.torchci import report as torchci_report
import operator
import string

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

    def test_callable_msg(self):
        # A callable msg is invoked only on failure, across all assert* methods.
        invoked = []

        def lazy(standard_msg):
            invoked.append(standard_msg)
            return f"{standard_msg}\nsentinel"

        # Passing: callable not invoked.
        self.assertEqual(1, 1, msg=lazy)
        self.assertTrue(True, msg=lazy)
        self.assertIn(1, [1, 2], msg=lazy)
        self.assertGreater(2, 1, msg=lazy)
        self.assertIsNone(None, msg=lazy)
        self.assertEqual(invoked, [])

        # Failing: callable invoked once with the standard message.
        failing = [
            lambda: self.assertTrue(False, msg=lazy),
            lambda: self.assertIn(9, [1, 2], msg=lazy),
            lambda: self.assertGreater(1, 2, msg=lazy),
            lambda: self.assertEqual(1, 2, msg=lazy),
        ]
        for fail in failing:
            invoked.clear()
            with self.assertRaises(AssertionError) as cm:
                fail()
            self.assertEqual(len(invoked), 1)
            self.assertEqual(str(cm.exception), f"{invoked[0]}\nsentinel")

        # A plain string msg is unchanged.
        with self.assertRaisesRegex(AssertionError, re.escape("True is not false : plain")):
            self.assertFalse(True, msg="plain")

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

    @onlyNativeDeviceTypes
    @dtypes(torch.float16, torch.float32)
    def test_isclose_equal_nan_broadcast(self, device, dtype):
        # Regression test: isclose with equal_nan=True should handle broadcasting
        # See https://github.com/pytorch/pytorch/issues/174985
        nan = torch.nan

        # One-sided broadcast: different sizes
        a = torch.tensor([nan], device=device, dtype=dtype)
        b = torch.tensor([nan, nan], device=device, dtype=dtype)
        result = torch.isclose(a, b, equal_nan=True)
        self.assertEqual(result, torch.tensor([True, True], device=device))

        # One-sided broadcast: scalar-like vs vector
        a = torch.tensor([nan], device=device, dtype=dtype)
        b = torch.tensor([nan, 1.0, nan], device=device, dtype=dtype)
        result = torch.isclose(a, b, equal_nan=True)
        self.assertEqual(result, torch.tensor([True, False, True], device=device))

        # Mutual broadcast
        a = torch.tensor([[nan], [1.0]], device=device, dtype=dtype)  # [2, 1]
        b = torch.tensor([[nan, 1.0]], device=device, dtype=dtype)    # [1, 2]
        result = torch.isclose(a, b, equal_nan=True)
        expected = torch.tensor([[True, False], [False, True]], device=device)
        self.assertEqual(result, expected)

        # Same shape (fast path) still works
        a = torch.tensor([nan, 1.0], device=device, dtype=dtype)
        b = torch.tensor([nan, 1.0], device=device, dtype=dtype)
        result = torch.isclose(a, b, equal_nan=True)
        self.assertEqual(result, torch.tensor([True, True], device=device))

    # The following tests (test_cuda_assert_*) are added to ensure test suite terminates early
    # when CUDA assert was thrown. Because all subsequent test will fail if that happens.
    # These tests are slow because it spawn another process to run test suite.
    # See: https://github.com/pytorch/pytorch/issues/49019
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
        self.assertTrue(
            has_device_side_assert(stderr),
            lambda msg: f"{msg}\nExpected device assert error in stderr, got: {stderr}",
        )
        if torch.version.cuda:
            # should run only 1 test because it throws unrecoverable error.
            self.assertIn('errors=1', stderr)


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
        self.assertTrue(
            has_device_side_assert(stderr),
            lambda msg: f"{msg}\nExpected device assert error in stderr, got: {stderr}",
        )
        if torch.version.cuda:
            # should run only 1 test because it throws unrecoverable error.
            self.assertIn('errors=1', stderr)


    @unittest.skip("https://github.com/pytorch/pytorch/issues/106308")
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
        ops_to_test = list(filter(lambda op: op.formatted_name in ['atan2', 'topk', 'xlogy'], op_db))

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

    def test_setup_and_teardown_run_for_device_specific_tests(self, device):
        # TODO: Move this (and other similar text blocks) to some fixtures/ subdir
        stderr = TestCase.runWithPytorchAPIUsageStderr(f"""\
#!/usr/bin/env python3

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import TestCase, run_tests

class TestFoo(TestCase):
    @classmethod
    def setUpClass(cls):
        # store something on the test class to query during teardown
        cls.stored_thing = "called with " + cls.__name__

    @classmethod
    def tearDownClass(cls):
        # throw here so we know teardown was run
        raise RuntimeError(cls.stored_thing)

    def test_bar(self, device):
        # make sure the test can access the stored thing
        print(self.stored_thing)

instantiate_device_type_tests(TestFoo, globals(), only_for='{self.device_type}')

if __name__ == '__main__':
    run_tests()
""")
        expected_device_class_name = f"TestFoo{self.device_type.upper()}"
        expected_error_text = f"RuntimeError: called with {expected_device_class_name}"
        self.assertIn(expected_error_text, stderr)


instantiate_device_type_tests(TestTesting, globals())


class TestFrameworkUtils(TestCase):

    def test_rocm_version_uses_sdk_version(self):
        with (
            unittest.mock.patch(
                "torch.testing._internal.common_cuda.TEST_WITH_ROCM", True
            ),
            unittest.mock.patch.object(torch.version, "rocm", "10.1.0"),
            unittest.mock.patch.object(torch.version, "hip", "7.15.26306"),
        ):
            self.assertEqual(_get_torch_rocm_version(), (10, 1, 0))
            self.assertEqual(getRocmVersion(), (10, 1, 0))

    def test_rocm_version_falls_back_to_hip_version(self):
        with (
            unittest.mock.patch(
                "torch.testing._internal.common_cuda.TEST_WITH_ROCM", True
            ),
            unittest.mock.patch.object(torch.version, "rocm", None),
            unittest.mock.patch.object(torch.version, "hip", "7.15.26306"),
        ):
            self.assertEqual(_get_torch_rocm_version(), (7, 15, 26306))
            self.assertEqual(getRocmVersion(), (7, 15, 26306))

    def test_windows_sm89_xfail_excludes_rocm(self):
        def test_fn():
            pass

        with unittest.mock.patch.multiple(
            common_cuda, IS_WINDOWS=True, TEST_WITH_ROCM=True, SM89OrLater=True
        ):
            self.assertIs(
                common_cuda.xfailCUDAIfSM89OrLaterOnWindows(test_fn), test_fn
            )

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
            if k in env:
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
        self.assertIn(f'Ran {test_bases_count - 1} test', stderr.decode('ascii'))

        # Test with setting both should throw exception
        env[PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY] = 'cpu'
        _, stderr = TestCase.run_process_no_exception(test_filter_file_template, env=env)
        self.assertNotIn('OK', stderr.decode('ascii'))


# Golden-file tests for the junit XML that CI uploads (tools/stats/upload_test_stats.py).
# junit_xml_testdata/pytest_suite.py runs for real; its XML is normalized to drop
# run-to-run noise and compared to expected/. After an intentional change:
#     REGENERATE_JUNIT_GOLDENS=1 python test/test_testing.py -k TestJunitXml
_JUNIT_TESTDATA = Path(__file__).resolve().parent / "junit_xml_testdata"
_REGENERATE_JUNIT_GOLDENS = os.environ.get("REGENERATE_JUNIT_GOLDENS") == "1"
# Run the fixture as the default config: drop the job's CI / PYTORCH_TEST_* /
# PYTEST_ADDOPTS settings and a PYTHONPATH whose repo root would shadow the
# installed torch, and don't write bytecode into test/.
_JUNIT_CHILD_ENV = {
    k: v
    for k, v in os.environ.items()
    if k not in ("PYTHONPATH", "CI", "PYTEST_ADDOPTS") and not k.startswith("PYTORCH_TEST_")
} | {"PYTHONDONTWRITEBYTECODE": "1"}


def _junit_provenance() -> str:
    versions = []
    for dist in ("pytest", "pytest-rerunfailures"):
        try:
            versions.append(f"{dist} {importlib.metadata.version(dist)}")
        except importlib.metadata.PackageNotFoundError:
            versions.append(f"{dist} missing")
    return ", ".join(versions)


def _redact_junit(text: str) -> str:
    text = re.sub(r"(?:/[\w.+-]+)+/([\w.+-]+\.py)", r"PATH/\1", text)
    return re.sub(r"(PATH/[\w.+-]+):\d+", r"\1:LINE", text)


def _normalize_junit_xml(raw: str) -> str:
    """Keep structure, attributes and the first line of text; drop run-specific noise."""
    root = ET.fromstring(raw)
    for el in root.iter():
        if "time" in el.attrib:
            el.set("time", "0.000")
        for attr in ("timestamp", "hostname"):
            el.attrib.pop(attr, None)
        for attr in ("classname", "file", "name", "message", "value", "type"):
            if attr in el.attrib:
                el.set(attr, _redact_junit(el.attrib[attr]))
        lines = [ln.strip() for ln in (el.text or "").splitlines() if ln.strip()]
        if lines:
            el.text = _redact_junit(lines[0])
            if len(lines) > 1:
                el.text += "\nELIDED"
        el.tail = None
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode").strip() + "\n"


def _count_junit_tags(normalized: str) -> collections.Counter[str]:
    return collections.Counter(el.tag for el in ET.fromstring(normalized).iter())


# The fixture runs as the default config, so its XML doesn't depend on the test
# config or device; skip the configs and GPU builds that would only repeat it.
@unittest.skipIf(IS_WINDOWS, "Skipping because doesn't work for windows")
@unittest.skipIf(IS_SANDCASTLE, "Skipping because doesn't work on sandcastle")
@skipIfTorchDynamo("subprocess test does not need Dynamo coverage")
@unittest.skipIf(TEST_WITH_CROSSREF, "subprocess test does not need crossref coverage")
@unittest.skipIf(TEST_CUDA or TEST_WITH_ROCM, "junit XML shape doesn't depend on the device")
class TestJunitXml(TestCase):
    provenance: str
    raw_xml: str

    @classmethod
    def setUpClass(cls) -> None:
        # The goldens depend on the pytest and pytest-rerunfailures versions they record.
        cls.provenance = _junit_provenance()
        header = f"<!-- provenance: {cls.provenance} -->"
        golden = _JUNIT_TESTDATA / "expected" / "pytest.xml"
        if not _REGENERATE_JUNIT_GOLDENS and golden.exists() and header not in golden.read_text().splitlines():
            raise unittest.SkipTest(
                f"goldens were generated with other versions than {cls.provenance}; "
                "rerun with REGENERATE_JUNIT_GOLDENS=1"
            )
        with tempfile.TemporaryDirectory() as tmp:
            xml_path = Path(tmp) / "report.xml"
            # Run in place under test/ so test/conftest.py and pytest.ini apply, as in CI.
            proc = subprocess.run(
                [
                    sys.executable, "-m", "pytest", "junit_xml_testdata/pytest_suite.py",
                    f"--junit-xml-reruns={xml_path}", "-p", "no:cacheprovider", "-q",
                ],
                cwd=_JUNIT_TESTDATA.parent,
                env=_JUNIT_CHILD_ENV,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if not xml_path.exists():
                raise RuntimeError(
                    f"pytest produced no XML (exit {proc.returncode})\n"
                    f"stdout:\n{proc.stdout[-3000:]}\nstderr:\n{proc.stderr[-3000:]}"
                )
            cls.raw_xml = xml_path.read_text()
        super().setUpClass()

    def _assert_matches_golden(self, name: str, doc: str) -> None:
        golden = _JUNIT_TESTDATA / "expected" / f"{name}.xml"
        if _REGENERATE_JUNIT_GOLDENS:
            golden.write_text(
                "<!-- generated by test/test_testing.py -->\n"
                f"<!-- provenance: {self.provenance} -->\n{doc}"
            )
            return
        self.assertTrue(golden.exists(), f"{golden} is missing; generate it with REGENERATE_JUNIT_GOLDENS=1")
        expected = "".join(
            ln for ln in golden.read_text().splitlines(keepends=True) if not ln.startswith("<!--")
        )
        self.assertMultiLineEqual(expected, doc, f"{golden} is stale; rerun with REGENERATE_JUNIT_GOLDENS=1")

    def test_pytest_outcome_shapes(self) -> None:
        normalized = _normalize_junit_xml(self.raw_xml)
        self._assert_matches_golden("pytest", normalized)

        counts = _count_junit_tags(normalized)
        # <testsuite tests="14"> counts the teardown error separately, but it merges
        # into its <testcase>
        self.assertEqual(counts["testcase"], 13)
        # assert_failure, raises_non_assertion, xpass_strict, rerun_then_fail
        self.assertEqual(counts["failure"], 4)
        # setup, teardown
        self.assertEqual(counts["error"], 2)
        # skip, skipif, xfail; non-strict xpass emits nothing
        self.assertEqual(counts["skipped"], 3)
        # 2 each for rerun_then_pass and rerun_then_fail
        self.assertEqual(counts["rerun"], 4)

    def test_sanitize_pytest_xml(self) -> None:
        with TemporaryFileName() as path:
            Path(path).write_text(self.raw_xml)
            sanitize_pytest_xml(path)
            normalized = _normalize_junit_xml(Path(path).read_text())
        self._assert_matches_golden("pytest_sanitized", normalized)


# Generated test files go below test/ so pytest loads test/conftest.py, as in CI.
_TEST_DIR = Path(__file__).resolve().parent
# TestJunitXml's fixture suite, which covers every outcome the pytest path reports.
_PYTEST_SUITE = _JUNIT_TESTDATA / "pytest_suite.py"

# The fixture's run context, which the report records, without the job's device
# filter (XPU and CUDA jobs set PYTORCH_TESTING_DEVICE_ONLY_FOR).
_REPORT_CHILD_ENV = {
    k: v for k, v in _JUNIT_CHILD_ENV.items() if not k.startswith("PYTORCH_TESTING_DEVICE_")
} | {
    "GITHUB_REPOSITORY": "pytorch/pytorch",
    "JOB_ID": "123456789",
    "BUILD_ENVIRONMENT": "report-build",
    "TEST_CONFIG": "report-config",
    "RUNNER_NAME": "report-runner",
}


# Each run line's keys, in order, and their types.
_RUN_TYPES = {
    "type": str, "schema_version": str, "file": str, "suite": str, "case_name": str, "language": str,
    "declared_case_name": str, "rerun_number": int, "outcome": str, "outcome_summary": str,
    "started_at": int, "ended_at": int, "properties": dict,
}
_RUN_OUTCOMES = {
    "passed", "failed", "error", "skipped", "xfailed", "xpassed",
    "crashed", "timed_out",
}


def _assert_run_line(test: TestCase, run: dict[str, Any], t0_ms: int, t1_ms: int) -> None:
    test.assertEqual([(k, type(v)) for k, v in run.items()], list(_RUN_TYPES.items()))
    test.assertEqual((run["type"], run["schema_version"], run["properties"]), ("run", "0.1", {}))
    test.assertIn(run["language"], ("python", "cpp"))
    test.assertIn(run["outcome"], _RUN_OUTCOMES)
    test.assertTrue(t0_ms <= run["started_at"] <= run["ended_at"] <= t1_ms, run)


def _report_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.jsonl"))


def _run_plugin(
    cwd: str, args: list[str], report_dir: Path, env: dict[str, str] | None = None
) -> tuple[subprocess.CompletedProcess, Path]:
    """Runs pytest with the report plugin; returns the process and its one report."""
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest", *args,
            "-p", "torch.testing._internal.torchci.plugin", f"--torchci-report-dir={report_dir}",
            "-p", "no:cacheprovider", "-q",
        ],
        cwd=cwd,
        env=_REPORT_CHILD_ENV | (env or {}),
        capture_output=True,
        text=True,
        timeout=300,
    )
    reports = _report_files(report_dir)
    if len(reports) != 1:
        raise RuntimeError(f"pytest produced {len(reports)} reports\n{proc.stdout}\n{proc.stderr}")
    return proc, reports[0]


def _runs(report: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in report.read_text().splitlines()[1:]]


# torchci test run reports, on TestJunitXml's fixture suite.
@unittest.skipIf(IS_WINDOWS, "Skipping because doesn't work for windows")
@unittest.skipIf(IS_SANDCASTLE, "Skipping because doesn't work on sandcastle")
@skipIfTorchDynamo("subprocess test does not need Dynamo coverage")
@unittest.skipIf(TEST_WITH_CROSSREF, "subprocess test does not need crossref coverage")
@unittest.skipIf(TEST_CUDA or TEST_WITH_ROCM, "report shape doesn't depend on the device")
class TestReportJsonl(TestCase):
    raw: str
    report_name: str
    t0_ms: int
    t1_ms: int

    @classmethod
    def setUpClass(cls) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cls.t0_ms = int(time.time() * 1000)
            # A non-default setting, to show up in flags.
            _, report = _run_plugin(str(_TEST_DIR), [str(_PYTEST_SUITE)], Path(tmp) / "suite", {"PYTORCH_TEST_WITH_SLOW": "1"})
            cls.t1_ms = int(time.time() * 1000)
            cls.raw = report.read_text()
            cls.report_name = report.name
        super().setUpClass()

    def test_runs(self) -> None:
        _, *runs = (json.loads(line) for line in self.raw.splitlines())
        rendered = "\n".join(
            f"{run['case_name']} {run['rerun_number']} {run['outcome']}"
            + (f": {run['outcome_summary']}" if run["outcome_summary"] else "")
            for run in runs
        )
        self.assertExpectedInline(rendered, """\
test_pass 0 passed
test_assert_failure 0 failed: AssertionError: values differ
test_raises_non_assertion 0 failed: RuntimeError: runtime error!
test_error_in_setup 0 error: RuntimeError: setup error
test_error_in_teardown 0 error: RuntimeError: teardown error
test_skipped 0 skipped: skipped unconditionally
test_skipif 0 skipped: skipped conditionally
test_xfail 0 xfailed: known bad
test_xpass_non_strict 0 xpassed
test_xpass_strict 0 failed: [XPASS(strict)] strictly expected to fail
test_rerun_then_pass 0 failed: AssertionError: attempt 1 fails
test_rerun_then_pass 1 failed: AssertionError: attempt 2 fails
test_rerun_then_pass 2 passed
test_rerun_then_fail 0 failed: AssertionError: attempt 1 fails
test_rerun_then_fail 1 failed: AssertionError: attempt 2 fails
test_rerun_then_fail 2 failed: AssertionError: attempt 3 fails
test_no_rerun_needed 0 passed""")

    def test_schema(self) -> None:
        report, *runs = (json.loads(line) for line in self.raw.splitlines())
        self.assertEqual(
            list(report),
            ["type", "schema_version", "repo", "github_workflow_job_id", "report_uuid", "environment", "flags",
             "properties"],
        )
        self.assertEqual(report["type"], "report")
        self.assertEqual(report["schema_version"], "0.1")
        self.assertEqual(report["repo"], "pytorch/pytorch")
        self.assertEqual(report["github_workflow_job_id"], 123456789)
        self.assertEqual(str(uuid.UUID(report["report_uuid"])), report["report_uuid"])
        self.assertEqual(self.report_name, f"suite-{report['report_uuid']}.jsonl")
        env = report["environment"]
        self.assertEqual(
            list(env),
            ["os", "os_version", "cpu_architecture", "cpu_capability", "python_version", "cc_compiler",
             "cc_compiler_version", "accelerator", "accelerator_version", "device_count", "device_name"],
        )
        for name, value in env.items():
            self.assertIsInstance(value, int if name == "device_count" else str)
        self.assertEqual(env["os"], {"Linux": "linux", "Darwin": "macos"}[platform.system()])
        free_threaded = "t" if sysconfig.get_config_var("Py_GIL_DISABLED") else ""
        self.assertEqual(env["python_version"], f"{sys.version_info.major}.{sys.version_info.minor}{free_threaded}")
        self.assertIn(env["cc_compiler"], ("", "gcc", "clang", "msvc"))
        self.assertIn(env["accelerator"], ("cpu", "cuda", "rocm", "xpu", "mps"))
        if env["device_count"] == 0:
            self.assertEqual(env["device_name"], "")
        flags = report["flags"]
        self.assertEqual(list(flags), sorted(TestEnvironment.env_var_values))
        self.assertTrue(all(isinstance(value, str) for value in flags.values()))
        self.assertEqual(flags["PYTORCH_TEST_WITH_SLOW"], "1")
        self.assertEqual(flags["PYTORCH_TEST_WITH_INDUCTOR"], "0")
        property_names = {
            "torch_version", "os_release", "device_memory_mib", "driver_version",
            "host_memory_mib", "build_environment", "test_config", "runner_name",
        }
        self.assertTrue(set(report["properties"]) <= property_names)
        self.assertTrue(all(isinstance(value, str) and value for value in report["properties"].values()))
        self.assertEqual(report["properties"]["build_environment"], "report-build")
        self.assertEqual(report["properties"]["test_config"], "report-config")
        self.assertEqual(report["properties"]["runner_name"], "report-runner")
        for run in runs:
            _assert_run_line(self, run, self.t0_ms, self.t1_ms)
            self.assertEqual(run["language"], "python")

    # One xdist worker, so the names travel from a worker to the controller.
    XDIST_SOURCE = """
import pytest

from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, TestCase


class TestDevice(TestCase):
    def test_device(self, device):
        pass


instantiate_device_type_tests(TestDevice, globals(), only_for="cpu")


@instantiate_parametrized_tests
class TestParametrize(TestCase):
    @parametrize("value", [1])
    def test_parametrize(self, value):
        pass


class TestPytest:
    @pytest.mark.parametrize("value", [1], ids=["one"])
    def test_pytest(self, value):
        pass


class TestSetattr:
    pass


setattr(TestSetattr, "test_added", lambda self: None)


def create_test_func():
    def test(self, device):
        pass

    return test


class TestFactory(TestCase):
    test_made = create_test_func()


instantiate_device_type_tests(TestFactory, globals(), only_for="cpu")
"""

    def test_declared_case_names(self) -> None:
        with tempfile.TemporaryDirectory(dir=_TEST_DIR) as tmp:
            (Path(tmp) / "xdist_report.py").write_text(textwrap.dedent(self.XDIST_SOURCE))
            t0_ms = int(time.time() * 1000)
            proc, report = _run_plugin(tmp, ["xdist_report.py", "-n", "1"], Path(tmp) / "xdist")
            t1_ms = int(time.time() * 1000)
            runs = _runs(report)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        for run in runs:
            _assert_run_line(self, run, t0_ms, t1_ms)
        declared = {run["case_name"]: run["declared_case_name"] for run in runs}
        self.assertEqual(declared["test_device_cpu"], "test_device")
        self.assertEqual(declared["test_parametrize_value_1"], "test_parametrize")
        self.assertEqual(declared["test_pytest[one]"], "test_pytest")
        # Added with setattr, so the function is named <lambda>.
        self.assertEqual(declared["test_added"], "test_added")
        # Made by a factory, so the function is named test.
        self.assertEqual(declared["test_made_cpu"], "test_made_cpu")

    SUBTESTS_SOURCE = """
import unittest


class TestSubtests(unittest.TestCase):
    def test_failing_subtest(self):
        for i in range(2):
            with self.subTest(i=i):
                self.assertEqual(i, 0)
"""

    def test_subtests(self) -> None:
        with tempfile.TemporaryDirectory(dir=_TEST_DIR) as tmp:
            (Path(tmp) / "subtests_report.py").write_text(textwrap.dedent(self.SUBTESTS_SOURCE))
            _, report = _run_plugin(tmp, ["subtests_report.py"], Path(tmp) / "subtests")
            runs = _runs(report)
        # One run per test, however many subtests it has.
        self.assertEqual([(run["case_name"], run["outcome"]) for run in runs], [("test_failing_subtest", "failed")])
        self.assertIn("1 != 0", runs[0]["outcome_summary"])

    SKIP_AFTER_FAILURE_SOURCE = """
import pytest


@pytest.fixture
def skips_in_teardown():
    yield
    pytest.skip("skip in teardown")


def test_fails_then_skips(skips_in_teardown):
    raise AssertionError("the real failure")
"""

    def test_skip_after_failure(self) -> None:
        with tempfile.TemporaryDirectory(dir=_TEST_DIR) as tmp:
            (Path(tmp) / "skip_report.py").write_text(textwrap.dedent(self.SKIP_AFTER_FAILURE_SOURCE))
            _, report = _run_plugin(tmp, ["skip_report.py"], Path(tmp) / "skip")
            runs = _runs(report)
        # The skip in teardown doesn't replace the failure's message.
        self.assertEqual([(run["outcome"], run["outcome_summary"]) for run in runs], [("failed", "AssertionError: the real failure")])


@unittest.skipIf(IS_WINDOWS, "Skipping because doesn't work for windows")
@unittest.skipIf(IS_SANDCASTLE, "Skipping because doesn't work on sandcastle")
@skipIfTorchDynamo("subprocess test does not need Dynamo coverage")
@unittest.skipIf(TEST_WITH_CROSSREF, "subprocess test does not need crossref coverage")
@unittest.skipIf(TEST_CUDA or TEST_WITH_ROCM, "report failures don't need GPU coverage")
class TestReportFailureIsolation(TestCase):
    """A failing writer must not change outcomes or the exit code, and warns once."""

    SOURCE = """
def test_pass():
    pass


def test_fail():
    raise RuntimeError("expected failure")
"""
    # Breaks the writer at the point FAIL_MODE names.
    CONFTEST = """
import os
import pytest

from torch.testing._internal.torchci import environment, plugin


def boom(*args, **kwargs):
    raise RuntimeError(os.environ["FAIL_MODE"] + " failed")


class FailingFile:
    def write(self, value):
        raise OSError("write failed")

    def flush(self):
        pass

    def close(self):
        pass


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    mode = os.environ.get("FAIL_MODE")
    if mode == "capture":
        environment.capture = boom
    elif mode == "write":
        plugin.open = lambda *args, **kwargs: FailingFile()
    elif mode == "name":
        plugin._item_declared_case_name = boom
    elif mode == "finish":
        plugin.ReportWriter._finish = boom
    elif mode == "worker" and hasattr(config, "workerinput"):
        plugin._item_declared_case_name = boom
    elif mode == "cache":
        plugin.ReportWriter._publish = boom
"""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.tmp = tempfile.TemporaryDirectory(dir=_TEST_DIR)
        cls.dir = Path(cls.tmp.name)
        (cls.dir / "failure_isolation.py").write_text(textwrap.dedent(cls.SOURCE))
        (cls.dir / "conftest.py").write_text(textwrap.dedent(cls.CONFTEST))
        cls.baseline = cls._run("baseline", ["-p", "no:cacheprovider"])

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()
        super().tearDownClass()

    @classmethod
    def _run(cls, name: str, args: list[str], env: dict[str, str] | None = None):
        xml = cls.dir / f"{name}.xml"
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "failure_isolation.py", "-q", f"--junitxml={xml}", *args],
            cwd=cls.dir,
            env=_JUNIT_CHILD_ENV | (env or {}),
            capture_output=True,
            text=True,
            timeout=300,
        )
        outcomes = []
        for case in ET.parse(xml).iter("testcase"):
            child = next(iter(case), None)
            outcomes.append((case.attrib["name"], "passed" if child is None else child.tag))
        return proc, outcomes

    @parametrize(
        "mode",
        [subtest(mode, name=mode) for mode in ("capture", "write", "name", "finish", "worker", "cache")],
    )
    def test_writer_error_does_not_change_tests(self, mode) -> None:
        args = ["-p", "torch.testing._internal.torchci.plugin", f"--torchci-report-dir={self.dir / mode}"]
        if mode == "cache":
            # The in-flight run is only published through the stepcurrent cache.
            args += ["--sc=report-failure", "-o", f"cache_dir={self.dir / 'cache'}"]
        else:
            args += ["-p", "no:cacheprovider"]
        if mode == "worker":
            # The writer runs on the controller; break the worker's makereport.
            args += ["-n", "1"]
        proc, outcomes = self._run(mode, args, {"FAIL_MODE": mode})
        baseline_proc, baseline_outcomes = self.baseline
        self.assertEqual(proc.returncode, baseline_proc.returncode, proc.stdout + proc.stderr)
        self.assertEqual(outcomes, baseline_outcomes)
        self.assertEqual(proc.stderr.count("torchci: report disabled after error:"), 1, proc.stderr)


instantiate_parametrized_tests(TestReportFailureIsolation)


@unittest.skipIf(IS_WINDOWS, "Skipping because doesn't work for windows")
@unittest.skipIf(IS_SANDCASTLE, "Skipping because doesn't work on sandcastle")
@skipIfTorchDynamo("subprocess test does not need Dynamo coverage")
@unittest.skipIf(TEST_WITH_CROSSREF, "subprocess test does not need crossref coverage")
@unittest.skipIf(TEST_CUDA or TEST_WITH_ROCM, "report enablement doesn't need GPU coverage")
class TestReportEnablement(TestCase):
    SOURCE = """
from torch.testing._internal.common_utils import run_tests, TestCase


class TestOne(TestCase):
    def test_pass(self):
        pass


if __name__ == "__main__":
    run_tests()
"""

    def _run_one(self, tmp: str, args: list[str]) -> subprocess.CompletedProcess:
        (Path(tmp) / "one.py").write_text(textwrap.dedent(self.SOURCE))
        return subprocess.run(
            [sys.executable, "one.py", "--use-pytest", "-p", "no:cacheprovider", *args],
            cwd=tmp,
            env=_REPORT_CHILD_ENV,
            capture_output=True,
            text=True,
            timeout=300,
        )

    def test_basic(self) -> None:
        with tempfile.TemporaryDirectory(dir=_TEST_DIR) as tmp:
            report_dir = Path(tmp) / "reports"
            t0_ms = int(time.time() * 1000)
            proc = self._run_one(tmp, [f"--save-torchci-reports={report_dir}"])
            t1_ms = int(time.time() * 1000)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            reports = _report_files(report_dir / "one")
            self.assertEqual(len(reports), 1)
            self.assertRegex(reports[0].name, r"^one-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}\.jsonl$")
            records = [json.loads(line) for line in reports[0].read_text().splitlines()]
        self.assertEqual([record["type"] for record in records], ["report", "run"])
        _assert_run_line(self, records[1], t0_ms, t1_ms)
        self.assertEqual(records[1]["outcome"], "passed")

    def test_off_is_noop(self) -> None:
        with tempfile.TemporaryDirectory(dir=_TEST_DIR) as tmp:
            proc = self._run_one(tmp, [])
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertEqual([path.name for path in Path(tmp).iterdir()], ["one.py"])

    def test_common_utils_parsing(self) -> None:
        source = """
import json
import sys

from torch.testing._internal import common_utils


result = []
for in_ci, args in [
    (False, []),
    (True, []),
    (False, ["--save-torchci-reports"]),
    (False, ["--save-torchci-reports=custom"]),
    (True, ["--no-save-torchci-reports"]),
]:
    common_utils.IS_CI = in_ci
    sys.argv = ["case.py", *args]
    common_utils.parse_cmd_line_args()
    result.append(common_utils.TEST_SAVE_TORCHCI_REPORTS)
print("RESULT=" + json.dumps(result))
"""
        proc = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(source)],
            cwd=_TEST_DIR,
            env=_REPORT_CHILD_ENV,
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        result_line = next(line for line in proc.stdout.splitlines() if line.startswith("RESULT="))
        self.assertEqual(json.loads(result_line[7:]), [None, None, "torchci-reports", "custom", None])

    def test_run_test_parsing_and_forwarding(self) -> None:
        run_test_module = importlib.import_module("run_test")
        default_dir = str(run_test_module.REPO_ROOT / "test/torchci-reports")
        absolute_dir = str(Path(tempfile.gettempdir()) / "absolute-reports")
        cases = [
            (False, [], None),
            (True, [], None),
            (False, ["--save-torchci-reports"], default_dir),
            (False, ["--save-torchci-reports=custom"], str(Path(default_dir).parent / "custom")),
            (False, [f"--save-torchci-reports={absolute_dir}"], absolute_dir),
            (True, ["--no-save-torchci-reports"], None),
        ]
        for in_ci, args, expected in cases:
            with self.subTest(in_ci=in_ci, args=args):
                with unittest.mock.patch.object(run_test_module, "IS_CI", in_ci):
                    with unittest.mock.patch.object(sys, "argv", ["run_test.py", *args]):
                        actual = run_test_module.parse_args().save_torchci_reports
                self.assertEqual(actual, expected)

        with unittest.mock.patch.object(run_test_module, "HAS_TORCHCI_REPORTS", True):
            self.assertEqual(run_test_module._torchci_report_args(absolute_dir), [f"--save-torchci-reports={absolute_dir}"])
            self.assertEqual(run_test_module._torchci_report_args(None), [])
        with unittest.mock.patch.object(run_test_module, "HAS_TORCHCI_REPORTS", False):
            self.assertEqual(run_test_module._torchci_report_args(absolute_dir), [])


@unittest.skipIf(IS_WINDOWS, "Skipping because doesn't work for windows")
@unittest.skipIf(IS_SANDCASTLE, "Skipping because doesn't work on sandcastle")
@skipIfTorchDynamo("subprocess test does not need Dynamo coverage")
@unittest.skipIf(TEST_WITH_CROSSREF, "subprocess test does not need crossref coverage")
@unittest.skipIf(TEST_CUDA or TEST_WITH_ROCM, "crash recording doesn't need GPU coverage")
class TestReportCrashes(TestCase):
    XDIST_SOURCE = """
import os


def test_before():
    pass


def test_crash():
    os._exit(1)


def test_after():
    pass


def test_fail_last():
    raise AssertionError("expected failure")
"""

    def test_xdist_worker_crash(self) -> None:
        with tempfile.TemporaryDirectory(dir=_TEST_DIR) as tmp:
            (Path(tmp) / "crash_report.py").write_text(textwrap.dedent(self.XDIST_SOURCE))
            # -x, as run_test.py passes for C++ tests, stops xdist after the last
            # test's final failure, before its teardown.
            args = ["crash_report.py", "-n", "1", "--reruns", "1", "-x"]
            proc, report = _run_plugin(tmp, args, Path(tmp) / "crash")
            runs = _runs(report)
        # 2: xdist reports a session stopped by -x as interrupted.
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        # pytest-rerunfailures reschedules the crashed test once.
        self.assertEqual(
            [(run["case_name"], run["outcome"], run["rerun_number"]) for run in runs],
            [
                ("test_before", "passed", 0),
                ("test_crash", "crashed", 0),
                ("test_crash", "crashed", 1),
                ("test_after", "passed", 0),
                ("test_fail_last", "failed", 0),
                ("test_fail_last", "failed", 1),
            ],
        )
        self.assertIn("crashed while running", runs[1]["outcome_summary"])

    SUBPROCESS_SOURCE = """
import os
import signal

from torch.testing._internal.common_utils import run_tests, TestCase


class TestCrash(TestCase):
    def test_crash(self):
        os.kill(os.getpid(), signal.SIGKILL)


if __name__ == "__main__":
    run_tests()
"""

    def test_subprocess_crash(self) -> None:
        with tempfile.TemporaryDirectory(dir=_TEST_DIR) as tmp:
            (Path(tmp) / "crash.py").write_text(textwrap.dedent(self.SUBPROCESS_SOURCE))
            report_dir = Path(tmp) / "reports"
            t0_ms = int(time.time() * 1000)
            proc = subprocess.run(
                [
                    sys.executable, "crash.py", "--use-pytest", "--subprocess",
                    f"--save-torchci-reports={report_dir}", "-p", "no:cacheprovider",
                ],
                cwd=tmp,
                env=_REPORT_CHILD_ENV,
                capture_output=True,
                text=True,
                timeout=300,
            )
            t1_ms = int(time.time() * 1000)
            reports = _report_files(report_dir / "crash")
            runs = [run for report in reports for run in _runs(report)]
            # The node id the child gets, relative to the repo root.
            expected_file = (Path(tmp) / "crash.py").resolve().relative_to(_TEST_DIR.parent).as_posix()
        self.assertNotEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        # Both child attempts (retry_shell retries once) wrote a report line before
        # dying; the parent wrote the third report, with the run.
        self.assertEqual(len(reports), 3)
        self.assertEqual(len(runs), 1)
        _assert_run_line(self, runs[0], t0_ms, t1_ms)
        self.assertEqual(
            (runs[0]["file"], runs[0]["suite"], runs[0]["case_name"], runs[0]["outcome"]),
            (expected_file, "TestCrash", "test_crash", "crashed"),
        )
        self.assertEqual(runs[0]["outcome_summary"], "the test process exited with code -9 (SIGKILL)")

    @staticmethod
    def _inflight(case_name: str, rerun_number: int) -> dict[str, Any]:
        return {
            "nodeid": f"test/test_x.py::TestX::{case_name}[param]",
            "declared_case_name": case_name,
            "rerun_number": rerun_number,
            "started_at": time.time(),
        }

    @parametrize(
        "exit_code, outcome",
        [subtest((-11, "crashed"), name="crashed"), subtest((124, "timed_out"), name="timed_out")],
    )
    def test_finish(self, exit_code, outcome) -> None:
        recovery = importlib.import_module("torch.testing._internal.torchci.recovery")
        inflight = self._inflight("test_hangs", 1)
        first = torchci_report.run_record(inflight["nodeid"], 0, "failed", inflight["started_at"], time.time(), "failed")
        torn = '{"type":"run","file":"test/te'
        t0_ms = int(inflight["started_at"] * 1000)
        with TemporaryFileName() as path:
            # A process that died while writing its last line.
            Path(path).write_text('{"type":"report"}\n' + torchci_report.line(first) + torn, encoding="utf-8")
            recovery.finish(path, inflight, exit_code)
            # With nothing in flight there is nothing to record.
            recovery.finish(path, None, -11)
            lines = Path(path).read_text(encoding="utf-8").splitlines()
        # The torn line stays a line of its own.
        self.assertEqual(lines[2], torn)
        runs = [json.loads(line) for line in (lines[1], *lines[3:])]
        for run in runs:
            _assert_run_line(self, run, t0_ms, int(time.time() * 1000))
        self.assertEqual(
            [(run["case_name"], run["rerun_number"], run["outcome"]) for run in runs],
            [("test_hangs[param]", 0, "failed"), ("test_hangs[param]", 1, outcome)],
        )
        self.assertEqual(runs[1]["declared_case_name"], "test_hangs")
        self.assertEqual(runs[1]["outcome_summary"], torchci_report.exit_summary(exit_code))

    def test_finish_skips_a_written_run(self) -> None:
        recovery = importlib.import_module("torch.testing._internal.torchci.recovery")
        inflight = self._inflight("test_complete", 0)
        run = torchci_report.run_record(inflight["nodeid"], 0, "passed", inflight["started_at"], time.time())
        with TemporaryFileName() as path:
            Path(path).write_text('{"type":"report"}\n' + torchci_report.line(run), encoding="utf-8")
            recovery.finish(path, inflight, -11)
            lines = Path(path).read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)

    @parametrize(
        "returned, elapsed, timeout, expected",
        [
            subtest((2, 10.0, 5.0, 124), name="pytest_exits_in_grace"),
            subtest((-2, 10.0, 5.0, 124), name="sigint_kills"),
            subtest((1, 10.0, 5.0, 1), name="failure_after_timeout"),
            subtest((2, 1.0, 5.0, 2), name="before_timeout"),
            subtest((2, 10.0, None, 2), name="no_timeout"),
        ],
    )
    def test_effective_exit_code(self, returned, elapsed, timeout, expected) -> None:
        recovery = importlib.import_module("torch.testing._internal.torchci.recovery")
        self.assertEqual(recovery.effective_exit_code(returned, elapsed, timeout), expected)

    def _run_test_retries(self, tmp: str, ret_code: int, cache: dict[str, str], timeout: float | None = None) -> str:
        run_test_module = importlib.import_module("run_test")
        cache_dir = Path(tmp) / ".pytest_cache/v/cache/stepcurrent/key"
        cache_dir.mkdir(parents=True)
        for name, value in cache.items():
            (cache_dir / name).write_text(value, encoding="utf-8")
        output = io.StringIO()
        with unittest.mock.patch.object(run_test_module, "REPO_ROOT", Path(tmp)):
            with unittest.mock.patch.object(run_test_module, "retry_shell", return_value=(ret_code, False)):
                run_test_module.run_test_retries([], tmp, {}, timeout, "key", output, False, "test_file", object())
        return output.getvalue()

    def test_run_test_finishes_the_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.jsonl"
            report.write_text('{"type":"report"}\n', encoding="utf-8")
            cache = {"report_path": json.dumps(str(report)), "report_inflight": json.dumps(self._inflight("test_t", 0))}
            # pytest exited on the SIGINT that followed a timeout.
            self._run_test_retries(tmp, 2, cache, timeout=0)
            run = json.loads(report.read_text(encoding="utf-8").splitlines()[1])
            left = [path.name for path in (Path(tmp) / ".pytest_cache/v/cache/stepcurrent/key").iterdir()]
        self.assertEqual(run["outcome"], "timed_out")
        # Removed, so the next retry process doesn't finish this report again.
        self.assertEqual(left, [])

    def test_run_test_survives_a_corrupt_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = self._run_test_retries(tmp, -11, {"report_path": "{"})
        self.assertIn("Could not finish the test run report:", output)


instantiate_parametrized_tests(TestReportCrashes)


class TestReportHelpers(TestCase):
    @parametrize(
        "nodeid, expected",
        [
            subtest(
                ("test/test_torch.py::TestTorch::test_add", ("test/test_torch.py", "TestTorch", "test_add", "python")),
                name="python_method",
            ),
            subtest(
                ("test/test_x.py::Outer::Inner::test_n", ("test/test_x.py", "Inner", "test_n", "python")),
                name="nested_class",
            ),
            subtest(
                ("test/test_x.py::test_fn", ("test/test_x.py", "", "test_fn", "python")),
                name="python_function",
            ),
            subtest(
                ("test/test_x.py::TestP::test_p[a::b-1]", ("test/test_x.py", "TestP", "test_p[a::b-1]", "python")),
                name="pytest_parameter",
            ),
        ],
    )
    def test_identity(self, nodeid, expected):
        self.assertEqual(torchci_report.identity(nodeid), expected)

    def test_declared_case_name_fallback(self):
        record = torchci_report.run_record("test/test_x.py::TestX::test_case[param]", 0, "passed", 1.0, 2.0)
        self.assertEqual(record["declared_case_name"], "test_case")

    def test_declared_case_name_raising_item(self):
        plugin = importlib.import_module("torch.testing._internal.torchci.plugin")

        class RaisingItem:
            nodeid = "test/test_x.py::TestX::test_case[param]"

            @property
            def obj(self):
                raise RuntimeError("no object")

        self.assertEqual(plugin._item_declared_case_name(RaisingItem()), "test_case")

    def test_lone_surrogate_is_escaped(self) -> None:
        now = time.time()
        line = torchci_report.line(
            torchci_report.run_record("test/test_x.py::test_surrogate", 0, "failed", now, now, outcome_summary="\ud800")
        )
        line.encode("utf-8")
        run = json.loads(line)
        _assert_run_line(self, run, int(now * 1000), int(now * 1000))
        self.assertEqual(run["outcome_summary"], r"\ud800")

    @skipIfTorchDynamo("Dynamo calls the patched torch._C function while tracing")
    def test_capture_skips_torch_accelerator(self) -> None:
        environment = importlib.import_module("torch.testing._internal.torchci.environment")
        with unittest.mock.patch.object(torch._C, "_accelerator_getAccelerator") as probe:
            environment.capture()
        probe.assert_not_called()

    @parametrize("rocm, expected", [subtest(("10.1.0", "10.1"), name="release"), subtest((None, "7.16"), name="hip")])
    def test_rocm_version(self, rocm, expected):
        environment = importlib.import_module("torch.testing._internal.torchci.environment")
        with unittest.mock.patch.multiple(torch.version, cuda=None, hip="7.16.26385", rocm=rocm):
            self.assertEqual(environment._accelerator(), ("rocm", expected))


instantiate_parametrized_tests(TestReportHelpers)


class TestPeriodicDecorator(TestCase):
    @parametrize("in_ci", [False, True])
    @parametrize("periodic_enabled", [False, True])
    def test_periodic_gates_on_periodic_mode(self, in_ci, periodic_enabled):
        calls = []
        with unittest.mock.patch.multiple(
            "torch.testing._internal.common_utils",
            IS_CI=in_ci,
            IS_SANDCASTLE=False,
            TEST_WITH_PERIODIC=periodic_enabled,
        ):
            class TestP(unittest.TestCase):
                def setUp(self):
                    calls.append("setUp")

                @periodic
                def test_p(self):
                    calls.append("test")

        result = unittest.TestResult()
        unittest.defaultTestLoader.loadTestsFromTestCase(TestP).run(result)

        skipped = in_ci and not periodic_enabled
        marks = {mark.name for mark in getattr(TestP.test_p, "pytestmark", ())}
        self.assertIn("periodic", marks)
        self.assertEqual(calls, [] if skipped else ["setUp", "test"])
        self.assertEqual(len(result.skipped), 1 if skipped else 0)
        self.assertEqual(result.failures, [])
        self.assertEqual(result.errors, [])

    @parametrize("on_sandcastle", [False, True])
    @parametrize("periodic_enabled", [False, True])
    def test_periodic_class_gates_setup(self, on_sandcastle, periodic_enabled):
        calls = []
        with unittest.mock.patch.multiple(
            "torch.testing._internal.common_utils",
            IS_CI=False,
            IS_SANDCASTLE=on_sandcastle,
            TEST_WITH_PERIODIC=periodic_enabled,
        ):
            @periodic
            class TestP(unittest.TestCase):
                @classmethod
                def setUpClass(cls):
                    calls.append("setUpClass")

                def test_p(self):
                    calls.append("test")

                @classmethod
                def tearDownClass(cls):
                    calls.append("tearDownClass")

        result = unittest.TestResult()
        unittest.defaultTestLoader.loadTestsFromTestCase(TestP).run(result)

        skipped = on_sandcastle and not periodic_enabled
        marks = {mark.name for mark in getattr(TestP, "pytestmark", ())}
        self.assertIn("periodic", marks)
        self.assertEqual(calls, [] if skipped else ["setUpClass", "test", "tearDownClass"])
        self.assertEqual(len(result.skipped), 1 if skipped else 0)
        self.assertEqual(result.failures, [])
        self.assertEqual(result.errors, [])

    @unittest.skipIf(
        IS_FBCODE, "run_test.py periodic filtering is only exercised by OSS CI"
    )
    @skipIfTorchDynamo("subprocess test does not need Dynamo coverage")
    def test_periodic_config_selects_only_periodic_tests(self):
        source = """\
from torch.testing._internal.common_utils import periodic, run_tests, serialTest

def test_plain_pytest():
    print("PLAIN_PYTEST_RAN")

@periodic
def test_periodic_pytest():
    print("PERIODIC_PYTEST_RAN")

@serialTest()
@periodic
def test_periodic_serial_pytest():
    print("PERIODIC_SERIAL_PYTEST_RAN")

if __name__ == "__main__":
    run_tests()
"""
        test_dir = os.path.dirname(os.path.realpath(__file__))
        with TemporaryFileName(
            prefix="test_periodic_filter_", suffix=".py", dir=test_dir
        ) as test_file:
            with open(test_file, "w") as f:
                f.write(source)

            env = os.environ.copy()
            env.pop("CI", None)
            env.pop("TEST_SHOWLOCALS", None)
            test_mode_prefixes = ("PYTORCH_TEST_WITH_", "PYTORCH_TEST_SKIP_")
            for flag in [k for k in env if k.startswith(test_mode_prefixes)]:
                del env[flag]
            for flag in (
                "PYTORCH_TEST_CUDA_MEM_LEAK_CHECK",
                "PYTORCH_TEST_DO_NOT_USE_PYTEST",
                "PYTORCH_TEST_RERUN_DISABLED_TESTS",
                "PYTORCH_TEST_RUN_EVERYTHING_IN_SERIAL",
                "TESTS_TO_INCLUDE",
            ):
                env.pop(flag, None)
            test_name = os.path.splitext(os.path.basename(test_file))[0]

            def run_test(periodic_mode):
                test_env = env.copy()
                if periodic_mode:
                    test_env["TEST_CONFIG"] = "periodic"
                    test_env["PYTORCH_TEST_WITH_SLOW"] = "1"
                else:
                    test_env.pop("TEST_CONFIG", None)
                result = subprocess.run(
                    [sys.executable, "run_test.py", "--include", test_name, "-s"],
                    cwd=test_dir,
                    env=test_env,
                    capture_output=True,
                    text=True,
                )
                output = result.stdout + result.stderr
                self.assertEqual(result.returncode, 0, msg=output)
                return output

            periodic_output = run_test(periodic_mode=True)
            default_output = run_test(periodic_mode=False)

        self.assertIn("PERIODIC_PYTEST_RAN", periodic_output)
        self.assertIn("PERIODIC_SERIAL_PYTEST_RAN", periodic_output)
        self.assertNotIn("PLAIN_PYTEST_RAN", periodic_output)
        self.assertIn("PLAIN_PYTEST_RAN", default_output)
        self.assertIn("PERIODIC_PYTEST_RAN", default_output)
        self.assertIn("PERIODIC_SERIAL_PYTEST_RAN", default_output)

    def test_periodic_does_not_leak_across_parametrized_tests(self):
        with unittest.mock.patch(
            "torch.testing._internal.common_utils.TEST_WITH_PERIODIC", True
        ):
            class TestP(unittest.TestCase):
                @parametrize("x", [1, 2])
                @decorateIf(periodic, lambda params: params["x"] == 1)
                def test_p(self, x):
                    pass

            instantiate_parametrized_tests(TestP)

        marked = TestP.test_p_x_1
        plain = TestP.test_p_x_2
        self.assertIn("periodic", {mark.name for mark in marked.pytestmark})
        plain_marks = {mark.name for mark in getattr(plain, "pytestmark", ())}
        self.assertNotIn("periodic", plain_marks)

    @periodic
    def test_periodic_smoke(self):
        self.assertTrue(TEST_WITH_PERIODIC or not (IS_CI or IS_SANDCASTLE))


instantiate_parametrized_tests(TestPeriodicDecorator)


class TestEnvironmentDefFlag(TestCase):
    """Verify env-var-vs-implication precedence in TestEnvironment.def_flag."""

    def setUp(self):
        super().setUp()
        import torch.testing._internal.common_utils as _cu
        self._cu = _cu
        self._defined: list[str] = []

    def tearDown(self):
        for name in self._defined:
            if hasattr(self._cu, name):
                delattr(self._cu, name)
            # Each test names its flags after their env vars.
            self._cu.TestEnvironment.env_var_values.pop(name, None)
            self._cu.TestEnvironment.repro_env_vars.pop(name, None)

    def _def_flag(self, name, **kwargs):
        from torch.testing._internal.common_utils import TestEnvironment
        self._defined.append(name)
        kwargs.setdefault("include_in_repro", False)
        return TestEnvironment.def_flag(name, **kwargs)

    def _def_setting(self, name, **kwargs):
        from torch.testing._internal.common_utils import TestEnvironment
        self._defined.append(name)
        return TestEnvironment.def_setting(name, **kwargs)

    def test_explicit_zero_overrides_implication(self):
        # Regression: PYTORCH_TEST_WITH_ROCM=0 must override
        # implied_by_fn=lambda: torch.version.hip is not None.
        with unittest.mock.patch.dict(os.environ, {"FOO_DF_1": "0"}):
            self.assertFalse(self._def_flag(
                "FOO_DF_1", env_var="FOO_DF_1",
                implied_by_fn=lambda: True))

    def test_explicit_one_with_no_implication(self):
        with unittest.mock.patch.dict(os.environ, {"FOO_DF_2": "1"}):
            self.assertTrue(self._def_flag(
                "FOO_DF_2", env_var="FOO_DF_2",
                implied_by_fn=lambda: False))

    def test_unset_with_implication_true(self):
        env = {k: v for k, v in os.environ.items() if k != "FOO_DF_3"}
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            self.assertTrue(self._def_flag(
                "FOO_DF_3", env_var="FOO_DF_3",
                implied_by_fn=lambda: True))

    def test_unset_with_implication_false(self):
        env = {k: v for k, v in os.environ.items() if k != "FOO_DF_4"}
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            self.assertFalse(self._def_flag(
                "FOO_DF_4", env_var="FOO_DF_4",
                implied_by_fn=lambda: False))

    def test_default_true_explicit_zero_overrides_implication(self):
        with unittest.mock.patch.dict(os.environ, {"FOO_DF_5": "0"}):
            self.assertFalse(self._def_flag(
                "FOO_DF_5", env_var="FOO_DF_5",
                default=True, implied_by_fn=lambda: True))

    def test_env_var_values(self):
        # What test run reports record: implied flags and unset settings
        # included, include_in_repro=False ones left out.
        env = {k: v for k, v in os.environ.items() if not k.startswith("FOO_EV_")}
        with unittest.mock.patch.dict(os.environ, env | {"FOO_EV_SET": "1", "FOO_EV_STR": "triton"}, clear=True):
            self._def_flag("FOO_EV_SET", env_var="FOO_EV_SET", include_in_repro=True)
            self._def_flag("FOO_EV_IMPLIED", env_var="FOO_EV_IMPLIED", include_in_repro=True, implied_by_fn=lambda: True)
            self._def_flag("FOO_EV_OFF", env_var="FOO_EV_OFF", include_in_repro=True)
            self._def_flag("FOO_EV_EXCLUDED", env_var="FOO_EV_EXCLUDED", implied_by_fn=lambda: True)
            self._def_setting("FOO_EV_STR", env_var="FOO_EV_STR")
            self._def_setting("FOO_EV_UNSET", env_var="FOO_EV_UNSET")
        values = {k: v for k, v in self._cu.TestEnvironment.env_var_values.items() if k.startswith("FOO_EV_")}
        self.assertEqual(values, {"FOO_EV_SET": "1", "FOO_EV_IMPLIED": "1", "FOO_EV_OFF": "0", "FOO_EV_STR": "triton", "FOO_EV_UNSET": ""})
        # The repro command only needs what was set explicitly.
        repro = {k: v for k, v in self._cu.TestEnvironment.repro_env_vars.items() if k.startswith("FOO_EV_")}
        self.assertEqual(repro, {"FOO_EV_SET": "1", "FOO_EV_STR": "triton"})


def make_assert_close_inputs(actual: Any, expected: Any) -> list[tuple[Any, Any]]:
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

    def test_small_float_dtype(self):
        for dtype in [
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2,
            torch.float8_e5m2fnuz,
            torch.float8_e8m0fnu,
        ]:
            w_vector = torch.tensor([3.14, 1.0], dtype=dtype)
            x_vector = torch.tensor([1.0, 3.14], dtype=dtype)
            y_vector = torch.tensor([3.14, 3.14], dtype=dtype)
            z_vector = torch.tensor([1.0, 3.14], dtype=dtype)

            for additional_dims in range(4):
                new_shape = list(w_vector.shape) + ([1] * additional_dims)
                w_tensor = w_vector.reshape(new_shape)
                x_tensor = x_vector.reshape(new_shape)
                y_tensor = y_vector.reshape(new_shape)
                z_tensor = z_vector.reshape(new_shape)

                for fn in assert_close_with_inputs(x_tensor, y_tensor):
                    expected_shape = (0,) + (0,) * (additional_dims)
                    with self.assertRaisesRegex(
                        AssertionError, re.escape(f"The first mismatched element is at index {expected_shape}")
                    ):
                        fn()

                for fn in assert_close_with_inputs(w_tensor, y_tensor):
                    expected_shape = (1,) + (0,) * (additional_dims)
                    with self.assertRaisesRegex(
                        AssertionError, re.escape(f"The first mismatched element is at index {expected_shape}")
                    ):
                        fn()
                for fn in assert_close_with_inputs(x_tensor, z_tensor):
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

    def test_dataclass_with_tensor_fields(self):
        # Python 3.13+ removed the same-object shortcut in dataclass __eq__, so
        # Foo(t, 1) == Foo(t, 1) raises when t is a multi-element tensor. assertEqual
        # / assert_close should still succeed by comparing fields directly.
        @dataclasses.dataclass
        class Foo:
            t: torch.Tensor
            i: int

        t = torch.zeros(2)
        actual = Foo(t, 1)
        expected = Foo(t, 1)

        self.assertEqual(actual, expected)
        for fn in assert_close_with_inputs(actual, expected):
            fn()

        # Distinct equal tensor values should also compare equal.
        self.assertEqual(Foo(torch.zeros(2), 1), Foo(torch.zeros(2), 1))

    def test_dataclass_compare_false_fields_ignored(self):
        @dataclasses.dataclass
        class Foo:
            t: torch.Tensor
            ignored: int = dataclasses.field(compare=False, default=0)

        self.assertEqual(Foo(torch.zeros(2), 1), Foo(torch.zeros(2), 2))

    def test_dataclass_mismatching_tensor_field_msg(self):
        @dataclasses.dataclass
        class Foo:
            t: torch.Tensor
            i: int

        # Multi-element tensors force the field-recursion path (scalar tensors
        # can make dataclass __eq__ return a bool and fall through to ObjectPair).
        actual = Foo(torch.zeros(2), 0)
        expected = Foo(torch.ones(2), 0)

        with self.assertRaisesRegex(AssertionError, re.escape("item ['t']")):
            torch.testing.assert_close(actual, expected)

    def test_nested_dataclass_with_tensor_fields(self):
        @dataclasses.dataclass
        class Inner:
            t: torch.Tensor

        @dataclasses.dataclass
        class Outer:
            inner: Inner
            i: int

        t = torch.ones(3)
        self.assertEqual(Outer(Inner(t), 1), Outer(Inner(t), 1))

    def test_dataclass_custom_eq_respected(self):
        # eq=False dataclasses with custom __eq__ must not be forced through
        # field-by-field comparison (which would ignore their semantics).
        @dataclasses.dataclass(eq=False)
        class ByShape:
            t: torch.Tensor
            tag: str

            def __eq__(self, other: object) -> bool:
                if not isinstance(other, ByShape):
                    return NotImplemented
                return self.t.shape == other.t.shape and self.tag == other.tag

        # Values differ but shape/tag match: custom __eq__ says equal.
        self.assertEqual(
            ByShape(torch.zeros(2), "a"),
            ByShape(torch.ones(2), "a"),
        )
        with self.assertRaises(AssertionError):
            self.assertEqual(
                ByShape(torch.zeros(2), "a"),
                ByShape(torch.zeros(3), "a"),
            )

    def test_dataclass_union_like_getattr_raises(self):
        # Mimics torch._export.serde.union._Union: unset fields raise on access,
        # and equality is defined by an active variant only.
        @dataclasses.dataclass(eq=False, repr=False)
        class UnionLike:
            as_int: int | None = None
            as_tensor: torch.Tensor | None = None
            _type: str = dataclasses.field(default="", repr=False, compare=False)

            def __post_init__(self) -> None:
                if self.as_int is not None:
                    self._type = "as_int"
                elif self.as_tensor is not None:
                    self._type = "as_tensor"

            def __getattribute__(self, name: str) -> object:
                attr = super().__getattribute__(name)
                field_names = {"as_int", "as_tensor"}
                if attr is None and name in field_names and name != self._type:
                    raise AttributeError(f"Field {name} is not set.")
                return attr

            def __eq__(self, other: object) -> bool:
                if not isinstance(other, UnionLike):
                    return False
                return self._type == other._type and getattr(self, self._type) == getattr(
                    other, other._type
                )

            def __repr__(self) -> str:
                return f"UnionLike({self._type}={getattr(self, self._type)})"

        actual = UnionLike(as_int=1)
        expected = UnionLike(as_int=1)
        self.assertEqual(actual, expected)
        with self.assertRaises(AssertionError):
            self.assertEqual(UnionLike(as_int=1), UnionLike(as_int=2))


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
    @parametrize("shape", [(), (0,), (1,), (1, 1), (2,), (2, 3), (8, 16, 32)])
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
    @parametrize("shape", [(), (0,), (1,), (1, 1), (2,), (2, 3), (8, 16, 32)])
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

    def test_name_fn_with_dot_raises(self):
        # Dots in test names break unittest.TestLoader.loadTestsFromName, so
        # instantiation should fail loudly rather than produce an unloadable test.
        class TestParametrized(TestCase):
            @parametrize("dtype", [torch.bfloat16], name_fn=str)
            def test_bad_name(self, dtype):
                pass

        with self.assertRaisesRegex(RuntimeError, 'contains a "." character'):
            instantiate_parametrized_tests(TestParametrized)

    def test_subtest_name_with_dot_raises(self):
        class TestParametrized(TestCase):
            @parametrize("x", [subtest(1, name="a.b")])
            def test_bad_name(self, x):
                pass

        with self.assertRaisesRegex(RuntimeError, 'contains a "." character'):
            instantiate_parametrized_tests(TestParametrized)

    def test_reparametrize(self):

        def include_is_even_arg(test_name, param_kwargs):
            x = param_kwargs["x"]
            is_even = x % 2 == 0
            new_param_kwargs = dict(param_kwargs)
            new_param_kwargs["is_even"] = is_even
            is_even_suffix = "_even" if is_even else "_odd"
            new_test_name = f"{test_name}{is_even_suffix}"
            yield (new_test_name, new_param_kwargs)

        def exclude_odds(test_name, param_kwargs):
            x = param_kwargs["x"]
            is_even = x % 2 == 0
            yield None if not is_even else (test_name, param_kwargs)

        class TestParametrized(TestCase):
            @reparametrize(parametrize("x", range(5)), include_is_even_arg)
            def test_foo(self, x, is_even):
                pass

            @reparametrize(parametrize("x", range(5)), exclude_odds)
            def test_bar(self, x):
                pass

        instantiate_parametrized_tests(TestParametrized)

        expected_test_names = [
            'TestParametrized.test_bar_x_0',
            'TestParametrized.test_bar_x_2',
            'TestParametrized.test_bar_x_4',
            'TestParametrized.test_foo_x_0_even',
            'TestParametrized.test_foo_x_1_odd',
            'TestParametrized.test_foo_x_2_even',
            'TestParametrized.test_foo_x_3_odd',
            'TestParametrized.test_foo_x_4_even',
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


class TestOmitSkippedTests(TestCase):
    def test_skipped_tests_are_omitted(self):
        with unittest.mock.patch("torch.testing._internal.common_utils.OMIT_SKIPPED_TESTS", True):

            class TestOmitted(TestCase):
                def test_runs(self):
                    pass

                @unittest.skip("never runs here")
                def test_skipped(self):
                    pass

                @unittest.skip("never runs here")
                @parametrize("x", [1, 2])
                def test_skipped_parametrized(self, x):
                    pass

                @parametrize("x", [subtest(1, decorators=[unittest.skip("never runs here")]), 2])
                def test_partly_skipped(self, x):
                    pass

            instantiate_parametrized_tests(TestOmitted)

        self.assertEqual(
            _get_test_names_for_test_class(TestOmitted),
            ['TestOmitted.test_partly_skipped_x_2', 'TestOmitted.test_runs'],
        )

    def test_skipped_tests_are_kept_by_default(self):
        with unittest.mock.patch("torch.testing._internal.common_utils.OMIT_SKIPPED_TESTS", False):

            class TestKept(TestCase):
                @unittest.skip("never runs here")
                def test_skipped(self):
                    pass

        self.assertEqual(_get_test_names_for_test_class(TestKept), ['TestKept.test_skipped'])


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

        locals_dict = dict(locals())
        instantiate_device_type_tests(TestParametrized, locals_dict, only_for=device)

        device_cls = locals_dict[f'TestParametrized{device.upper()}']
        expected_test_names = [name.format(device_cls.__name__, device) for name in (
            '{}.test_device_dtype_specific_{}_float32',
            '{}.test_device_dtype_specific_{}_float64',
            '{}.test_device_specific_{}')
        ]
        test_names = _get_test_names_for_test_class(device_cls)
        self.assertEqual(expected_test_names, test_names)

    def test_name_fn_with_dot_raises(self, device):
        device = self.device_type

        class TestParametrized(TestCase):
            @parametrize("dtype", [torch.bfloat16], name_fn=str)
            def test_bad_name(self, device, dtype):
                pass

        locals_dict = dict(locals())
        with self.assertRaisesRegex(RuntimeError, 'contains a "." character'):
            instantiate_device_type_tests(TestParametrized, locals_dict, only_for=device)

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

        locals_dict = dict(locals())
        instantiate_device_type_tests(TestParametrized, locals_dict, only_for=device)

        device_cls = locals_dict[f'TestParametrized{device.upper()}']
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
            locals_dict = dict(locals())
            instantiate_device_type_tests(TestParametrized, locals_dict, only_for=device)

    def test_default_names(self, device):
        device = self.device_type

        class TestParametrized(TestCase):
            @parametrize("x", range(5))
            def test_default_names(self, device, x):
                pass

            @parametrize("x,y", [(1, 2), (2, 3), (3, 4)])
            def test_two_things_default_names(self, device, x, y):
                pass


        locals_dict = dict(locals())
        instantiate_device_type_tests(TestParametrized, locals_dict, only_for=device)

        device_cls = locals_dict[f'TestParametrized{device.upper()}']
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

        locals_dict = dict(locals())
        instantiate_device_type_tests(TestParametrized, locals_dict, only_for=device)

        device_cls = locals_dict[f'TestParametrized{device.upper()}']
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

        locals_dict = dict(locals())
        instantiate_device_type_tests(TestParametrized, locals_dict, only_for=device)

        device_cls = locals_dict[f'TestParametrized{device.upper()}']
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

        locals_dict = dict(locals())
        instantiate_device_type_tests(TestParametrized, locals_dict, only_for=device)

        device_cls = locals_dict[f'TestParametrized{device.upper()}']
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

        locals_dict = dict(locals())
        instantiate_device_type_tests(TestParametrized, locals_dict, only_for=device)

        device_cls = locals_dict[f'TestParametrized{device.upper()}']
        expected_test_names = []
        for op in op_db:
            for dtype in op.supported_dtypes(torch.device(device).type):
                for flag_part in ('flag_disabled', 'flag_enabled'):
                    expected_name = f'{device_cls.__name__}.test_op_parametrized_{op.formatted_name}_{flag_part}_{device}_{dtype_name(dtype)}'
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

        locals_dict = dict(locals())
        instantiate_device_type_tests(TestParametrized, locals_dict, only_for=device)

        device_cls = locals_dict[f'TestParametrized{device.upper()}']
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
        locals_dict = dict(locals())
        instantiate_device_type_tests(TestParametrized, locals_dict, only_for=device)
        device_cls = locals_dict[f'TestParametrized{device.upper()}']

        for test_func, name in _get_test_funcs_for_test_class(device_cls):
            should_apply = (name == 'test_op_param_test_op_x_2_cpu_float64' or
                            ('test_other' in name and 'y_5' in name) or
                            ('test_three' in name and name.endswith('_int16')))
            self.assertEqual(hasattr(test_func, '_decorator_applied'), should_apply)

    def test_modules_decorator_applies_module_and_param_specific_decorators(self, device):
        # Test that decorators can be applied on a per-module / per-param basis.

        # Create a test module, ModuleInfo entry, and decorator to apply.
        class TestModule(torch.nn.Module):
            def __init__(self) -> None:
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
        locals_dict = dict(locals())
        instantiate_device_type_tests(TestParametrized, locals_dict, only_for=device)
        device_cls = locals_dict[f'TestParametrized{device.upper()}']

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
        locals_dict = dict(locals())
        instantiate_device_type_tests(TestParametrized, locals_dict, only_for=device)
        device_cls = locals_dict[f'TestParametrized{device.upper()}']

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

        locals_dict = dict(locals())
        instantiate_device_type_tests(TestParametrized, locals_dict, only_for=device)

        device_cls = locals_dict[f'TestParametrized{device.upper()}']
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
            locals_dict = dict(locals())
            instantiate_device_type_tests(TestParametrized, locals_dict, only_for=device)

        # Verify proper error behavior with @ops + @dtypes, as both try to set dtype.

        class TestParametrized(TestCase):
            @dtypes(torch.float32, torch.float64)
            @ops(op_db)
            def test_parametrized(self, op, dtype):
                pass

        with self.assertRaisesRegex(RuntimeError, "handled multiple times"):
            locals_dict = dict(locals())
            instantiate_device_type_tests(TestParametrized, locals_dict, only_for=device)

    def test_multiple_handling_of_same_param_error(self, device):
        # Test that multiple decorators handling the same param errors out.
        # Both @modules and @ops handle the dtype param.

        class TestParametrized(TestCase):
            @ops(op_db)
            @modules(module_db)
            def test_param(self, device, dtype, op, module_info, training):
                pass

        with self.assertRaisesRegex(RuntimeError, "handled multiple times"):
            locals_dict = dict(locals())
            instantiate_device_type_tests(TestParametrized, locals_dict, only_for=device)

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

    @skipIfXpu(msg="The test is flaky on XPU, see https://github.com/pytorch/pytorch/issues/110040")
    # The test is flaky on ROCm/XPU and has been open and close multiple times
    # https://github.com/pytorch/pytorch/issues/110040
    def test_circular_dependencies(self) -> None:
        """ Checks that all modules inside torch can be imported
        Prevents regression reported in https://github.com/pytorch/pytorch/issues/77441 """
        ignored_modules = ["torch.utils.tensorboard",  # deps on tensorboard
                           "torch.distributed.elastic.rendezvous",  # depps on etcd
                           "torch.backends._coreml",  # depends on pycoreml
                           "torch.contrib.",  # something weird
                           "torch.testing._internal.distributed.",  # just fails
                           "torch.ao.pruning._experimental.",  # depends on pytorch_lightning, not user-facing
                           "torch.onnx._internal",  # depends on onnx-script
                           "torch._inductor.runtime.triton_helpers",  # depends on triton
                           "torch._native.flydsl.intrinsics",  # depends on flydsl
                           "torch._native.cutedsl",  # depends on cutlass
                           "torch._native.ops.reductions.traits",  # depends on cutlass
                           "torch._native.ops.bmm_outer_product.triton_kernels",  # depends on triton
                           "torch._native.ops.foreach_mm",  # depends on nvmath-python, cuda-python
                           "torch._native.ops.linear_cross_entropy.fused_grad_logits_kernel",  # depends on cutlass
                           "torch._native.ops.norm.flydsl_rmsnorm_fwd",  # depends on flydsl
                           "torch._native.ops.polar.nvmath_impl",  # depends on nvmath-python, cuda-python
                           "torch._native.ops.reductions.inner_tree_kernel",  # depends on cutlass
                           "torch._native.ops.reductions.kernel_general",  # depends on cutlass
                           "torch._native.ops.reductions.kernel_rowtile",  # depends on cutlass
                           "torch._native.ops.reductions.tile",  # depends on cutlass
                           "torch._native.ops.reductions.kernel_xcta",  # depends on cutlass
                           "torch._native.ops.reductions.kernel_coltile",  # depends on cutlass
                           "torch._native.ops.scatter_add",  # depends on cutlass
                           "torch._native.ops.topk",  # depends on cutlass
                           "torch._inductor.codegen.cuda",  # depends on cutlass
                           "torch._inductor.codegen.cutedsl",  # depends on cutlass
                           "torch._inductor.kernel.flex_gemm.compile_pool",  # depends on cutlass
                           "torch._inductor.kernel.flex_gemm.output_layout_cutedsl",  # depends on cutlass
                           "torch.distributed.benchmarks",  # depends on RPC and DDP Optim
                           "torch.distributed.debug._frontend",  # depends on tabulate
                           "torch.distributed.examples",  # requires CUDA and torchvision
                           "torch.distributed.tensor.examples",  # example scripts
                           "torch.distributed._tools.sac_ilp",  # depends on pulp
                           "torch.csrc",  # files here are devtools, not part of torch
                           "torch.include",  # torch include files after install
                           "torch._inductor.kernel.vendored_templates.cutedsl",  # depends on cutlass
                           "torch._inductor.kernel.vendored_templates.flydsl",  # depends on flydsl
                           "torch._vendor.quack",  # depends on cutlass / cuda-python
                           "torch._inductor.kernel.flex_gemm.quack_ops",  # depends on cutlass
                           "torch.profiler._cuspy",  # depends on cupti-python
                           ]
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

        if sys.version_info < (3, 12):
            # depends on Python 3.12+ syntax
            ignored_modules.append("torch.testing._internal.py312_intrinsics")

        torch_dir = os.path.dirname(torch.__file__)
        for base, _, files in os.walk(torch_dir):
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

    def test_lazy_imports_are_lazy(self) -> None:
        out = self._check_python_output("import sys;import torch;print(all(x not in sys.modules for x in torch._lazy_modules))")
        self.assertEqual(out.strip(), "True")

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

    def test_not_import_triton(self) -> None:
        out = self._check_python_output("import torch;import sys;print('triton' not in sys.modules)")
        self.assertEqual(out.strip(), "True")

    @parametrize('path', ['torch', 'functorch'])
    def test_no_mutate_global_logging_on_import(self, path) -> None:
        # Calling logging.basicConfig, among other things, modifies the global
        # logging state. It is not OK to modify the global logging state on
        # `import torch` (or other submodules we own) because users do not expect it.
        expected = string.ascii_lowercase
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

    def test_reimport_after_failed_import(self) -> None:
        # A failed `import torch` leaves its submodules and C++ global state behind,
        # so retrying used to re-run one-time initialization and segfault at
        # interpreter shutdown. See https://github.com/pytorch/pytorch/issues/194172
        program = textwrap.dedent(
            '''\
            import importlib.metadata

            class _FailingEntryPoint:
                name = "injected_backend"

                def load(self):
                    raise ImportError("injected backend failure")

            _real_entry_points = importlib.metadata.entry_points

            def _entry_points(**kwargs):
                if kwargs.get("group") == "torch.backends":
                    return [_FailingEntryPoint()]
                return _real_entry_points(**kwargs)

            importlib.metadata.entry_points = _entry_points

            for _ in range(3):
                try:
                    import torch
                except Exception as e:
                    print(f"{type(e).__name__}: {e}")
            '''
        )
        proc = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            # On Windows, opening the subprocess with the default CWD makes `import torch`
            # fail, so just set CWD to this script's directory
            cwd=os.path.dirname(os.path.realpath(__file__)),
            # The test relies on the autoload running, so don't inherit a disabling value
            env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "1"},
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        lines = proc.stdout.splitlines()
        self.assertEqual(len(lines), 3, msg=proc.stdout)
        self.assertTrue(lines[0].startswith("RuntimeError: "), msg=lines[0])
        self.assertIn("Failed to load the backend extension: injected_backend", lines[0])
        for line in lines[1:]:
            self.assertTrue(line.startswith("ImportError: "), msg=line)
            self.assertIn("can only be initialized once per process", line)

    def test_reload_is_rejected(self) -> None:
        # Reloading re-executes the module body against live C++ state just like a
        # retry does, so it is refused; torch must stay usable afterwards.
        with self.assertRaisesRegex(ImportError, "can only be initialized once"):
            importlib.reload(torch)
        self.assertEqual(torch.tensor([1, 2]).sum().item(), 3)

class TestOpInfos(TestCase):
    def test_sample_input(self) -> None:
        a, b, c, d, e = (object() for _ in range(5))

        # Construction with natural syntax
        s = SampleInput(a, b, c, d=d, e=e)
        if s.input is not a:
            raise AssertionError("s.input should be a")
        if s.args != (b, c):
            raise AssertionError(f"s.args should be (b, c), got {s.args}")
        if s.kwargs != dict(d=d, e=e):
            raise AssertionError(f"s.kwargs mismatch: got {s.kwargs}")

        # Construction with explicit args and kwargs
        s = SampleInput(a, args=(b,), kwargs=dict(c=c, d=d, e=e))
        if s.input is not a:
            raise AssertionError("s.input should be a")
        if s.args != (b,):
            raise AssertionError(f"s.args should be (b,), got {s.args}")
        if s.kwargs != dict(c=c, d=d, e=e):
            raise AssertionError(f"s.kwargs mismatch: got {s.kwargs}")

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
        if s.input is not a:
            raise AssertionError("s.input should be a")
        if not s.broadcasts_input:
            raise AssertionError("s.broadcasts_input should be True")

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

# Tests test classification.
class TestHardwareClassifications(TestCase):
    def setUp(self):
        super().setUp()
        import torch.testing._internal.common_utils as _cu

        self._cu = _cu

    def _suite_test_names(self, suite) -> set[str]:
        return {test_case.id().split(".")[-1] for test_case in self._cu.HardwareClassificationTestLoader.iter_test_cases_recursively(suite)}

    def test_filter_suite(self):
        requirement = self._cu.HardwareClassification

        class GenericTest(TestCase):
            hw_classification = requirement.GENERIC

            def test_generic(self):
                pass

        class CudaTest(TestCase):
            hw_classification = requirement.CUDA

            def test_cuda(self):
                pass

        class MissingClassificationTest(TestCase):
            def test_missing_classification(self):
                pass

        suite = unittest.TestSuite(
            [
                unittest.defaultTestLoader.loadTestsFromTestCase(GenericTest),
                unittest.defaultTestLoader.loadTestsFromTestCase(CudaTest),
                unittest.defaultTestLoader.loadTestsFromTestCase(MissingClassificationTest),
            ]
        )

        loader = self._cu.HardwareClassificationTestLoader({self._cu.HardwareClassification.GENERIC})
        filtered_suite = loader.get_filtered_suite(suite)

        self.assertEqual(
            self._suite_test_names(filtered_suite),
            {"test_generic"},
        )

    def test_filter_suite_uses_inherited_metadata(self):
        requirement = self._cu.HardwareClassification

        class AcceleratorBase(TestCase):
            hw_classification = requirement.ACCELERATOR

        class AcceleratorChild(AcceleratorBase):
            def test_inherited_classification(self):
                pass

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(AcceleratorChild)
        loader = self._cu.HardwareClassificationTestLoader({self._cu.HardwareClassification.ACCELERATOR})
        filtered_suite = loader.get_filtered_suite(suite)

        self.assertEqual(
            self._suite_test_names(filtered_suite),
            {"test_inherited_classification"},
        )

    def test_hw_classification_test_loader(self):
        requirement = self._cu.HardwareClassification
        import types

        # Build a mock module with test classes
        class GenericA(TestCase):
            hw_classification = requirement.GENERIC

            def test_a1(self):
                pass

            def test_a2(self):
                pass

        class GenericB(TestCase):
            hw_classification = requirement.GENERIC

            def test_b1(self):
                pass

        class Cuda(TestCase):
            hw_classification = requirement.CUDA

            def test_c1(self):
                pass

        mod = types.ModuleType("_test_hc_module")
        mod.GenericA = GenericA
        mod.GenericB = GenericB
        mod.Cuda = Cuda

        loader = self._cu.HardwareClassificationTestLoader({requirement.GENERIC})

        with self.subTest(method="loadTestsFromModule"):
            suite = loader.loadTestsFromModule(mod)
            self.assertEqual(
                self._suite_test_names(suite),
                {"test_a1", "test_a2", "test_b1"},
            )

        with self.subTest(method="loadTestsFromNames"):
            suite = loader.loadTestsFromNames(
                ["GenericA.test_a1", "Cuda.test_c1"],
                module=mod,
            )
            self.assertEqual(self._suite_test_names(suite), {"test_a1"})

        with self.subTest(method="loadTestsFromName"):
            suite = loader.loadTestsFromName(
                "Cuda.test_c1",
                module=mod,
            )
            self.assertEqual(self._suite_test_names(suite), set())

    def test_hw_classification_test_loader_no_filter(self):
        import types

        class GenericA(TestCase):
            hw_classification = self._cu.HardwareClassification.GENERIC

            def test_a(self):
                pass

        class Cuda(TestCase):
            hw_classification = self._cu.HardwareClassification.CUDA

            def test_b(self):
                pass

        mod = types.ModuleType("_test_hc_module_none")
        mod.GenericA = GenericA
        mod.Cuda = Cuda

        loader = self._cu.HardwareClassificationTestLoader(None)
        suite = loader.loadTestsFromModule(mod)

        self.assertEqual(self._suite_test_names(suite), {"test_a", "test_b"})


instantiate_device_type_tests(TestOpInfoSampleFunctions, globals())
instantiate_parametrized_tests(TestImports)


if __name__ == '__main__':
    run_tests()
