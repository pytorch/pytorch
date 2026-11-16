# Owner(s): ["module: tests"]

import collections
import contextlib
import dataclasses
import doctest
import functools
import importlib
import inspect
import itertools
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest.mock
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
    getRocmVersion, TemporaryFileName, sanitize_pytest_xml, set_cwd, add_skip_condition, check_if_enable, lazy_skip_if, skip_reason_at_collection,
    HardwareClassification,
)
from torch.testing._internal.common_cuda import _get_torch_rocm_version, has_device_side_assert
from torch.testing._internal.common_device_type import \
    (PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY, PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY, dtypes,
     get_device_type_test_bases, instantiate_device_type_tests, onlyCPU, onlyCUDA, onlyNativeDeviceTypes,
     deviceCountAtLeast, ops, expectedFailureMeta, OpDTypes, skipCPUIf, skipCUDAIf, skipCUDAIfNoMagma, skipIf,
     DeviceTypeTestBase, PrivateUse1TestBase, preserves_device_selection, MPSTestBase, CPUTestBase, onlyAccelerator)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal import common_cuda, opinfo
from torch.testing._internal.common_dtype import all_types_and_complex_and, floating_types
from torch.testing._internal.common_modules import modules, module_db, ModuleInfo
from torch.testing._internal.opinfo.core import SampleInput, DecorateInfo, OpInfo
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


    @onlyCUDA
    @slowTest
    def test_cuda_assert_should_not_stop_common_distributed_test_suite(self, device):
        # test to ensure common_distributed.py override should not early terminate CUDA.
        # Run from a real file. python -c cannot be the spawn main module, so
        # the worker would fail to unpickle this test class.
        script = """\
#!/usr/bin/env python3

import torch
from torch.testing._internal.common_utils import (run_tests, slowTest)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import MultiProcessTestCase

class TestThatContainsCUDAAssertFailure(MultiProcessTestCase):

    @property
    def world_size(self):
        # One worker is enough to fault the device. The default world size
        # would run this index once per rank.
        return 1

    def setUp(self):
        super().setUp()
        # The parent only joins children. Spawn one so the body runs.
        self._spawn_processes()

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
"""
        # Run this suite from a real .py file.
        fd, path = tempfile.mkstemp(suffix=".py")
        os.close(fd)
        try:
            with open(path, "w") as f:
                f.write(script)
            env = os.environ.copy()
            env["PYTORCH_API_USAGE_STDERR"] = "1"
            env.pop("CI", None)
            env.pop("TEST_SHOWLOCALS", None)
            proc = subprocess.run(
                [sys.executable, path],
                env=env,
                capture_output=True,
            )
        finally:
            os.remove(path)
        stderr = proc.stderr.decode("utf-8", errors="replace")
        # The spawned worker's device-side assert is reported in stderr.
        # On CUDA, unittest records that failure as this test's error.
        self.assertTrue(
            has_device_side_assert(stderr),
            lambda msg: f"{msg}\nExpected device assert error in stderr, got: {stderr}",
        )
        if torch.version.cuda:
            self.assertIn('errors=1', stderr)

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
            elif self.device_type == 'xpu':
                dtypes = op.dtypesIfXPU
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

    @unittest.skipIf(IS_WINDOWS, "Skipping because doesn't work for windows")
    @unittest.skipIf(IS_SANDCASTLE, "Skipping because doesn't work on sandcastle")
    def test_inductor_run_cpu_env_var(self):
        # PYTORCH_TEST_INDUCTOR_RUN_CPU=1 brings back inductor's RUN_CPU test classes
        # when ONLY_FOR leaves CPU out (CI's nogpu_NO_AVX2 job), but not when EXCEPT_FOR
        # excludes CPU. If this broke, those classes would silently disappear.
        flag = "PYTORCH_TEST_INDUCTOR_RUN_CPU"
        unset = ("CI", PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY, PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY, flag)
        base_env = {k: v for k, v in os.environ.items() if k not in unset}

        def query(name, env):
            code = f"from torch.testing._internal.inductor_utils import {name}; print(bool({name}))"
            stdout, stderr = TestCase.run_process_no_exception(code, env={**base_env, **env})
            lines = stdout.decode().strip().splitlines()
            self.assertIn(lines[-1:], (["True"], ["False"]), msg=stderr.decode())
            return lines[-1]

        if query("HAS_CPU", {}) != "True":
            self.skipTest("RUN_CPU also needs a working C++ compiler")
        for env, expected in [
            ({}, "True"),
            ({PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY: "cuda"}, "False"),
            ({PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY: "cuda", flag: "1"}, "True"),
            ({PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY: "cpu", flag: "1"}, "False"),
        ]:
            with self.subTest(env=env):
                self.assertEqual(query("RUN_CPU", env), expected)


# Golden-file tests for the junit XML that CI uploads (tools/stats/upload_test_stats.py).
# junit_xml_testdata/pytest_suite.py runs for real; its XML is normalized to drop
# run-to-run noise and compared to expected/. After an intentional change:
#     REGENERATE_JUNIT_GOLDENS=1 python test/test_testing.py -k TestJunitXml
_JUNIT_TESTDATA = Path(__file__).resolve().parent / "junit_xml_testdata"
_REGENERATE_JUNIT_GOLDENS = os.environ.get("REGENERATE_JUNIT_GOLDENS") == "1"
_HW_CLASSIFICATION_TESTDATA = Path(__file__).resolve().parent / "hw_classification_testdata"
# Run fixture suites (TestJunitXml, TestHardwareClassifications,
# TestPruneSkipsAtCollection) as the default config: drop the job's CI,
# PYTORCH_TEST_*, PYTORCH_TESTING_DEVICE_*, and pytest settings and a
# PYTHONPATH whose repo root would shadow the installed torch, and don't write
# bytecode into test/.
_FIXTURE_CHILD_ENV = {
    k: v
    for k, v in os.environ.items()
    if k not in ("PYTHONPATH", "CI", "PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_DISABLE_PLUGIN_AUTOLOAD")
    and not k.startswith(("PYTORCH_TEST_", "PYTORCH_TESTING_DEVICE_"))
} | {"PYTHONDONTWRITEBYTECODE": "1"}


def _fixture_child_versions() -> dict[str, str | None]:
    """pytest and pytest-rerunfailures versions as the fixture suite's interpreter and environment see them."""
    code = (
        "import importlib.metadata as m, json\n"
        "def version(dist):\n"
        "    try:\n"
        "        return m.version(dist)\n"
        "    except m.PackageNotFoundError:\n"
        "        return None\n"
        "print(json.dumps({d: version(d) for d in ('pytest', 'pytest-rerunfailures')}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], env=_FIXTURE_CHILD_ENV, capture_output=True, text=True, check=True
    )
    return json.loads(proc.stdout)


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
    golden_skip: str | None
    has_rerunfailures: bool
    raw_xml: str
    output: str

    @classmethod
    def setUpClass(cls) -> None:
        versions = _fixture_child_versions()
        if versions["pytest"] is None:
            raise unittest.SkipTest("the fixture suite needs pytest")
        cls.has_rerunfailures = versions["pytest-rerunfailures"] is not None
        # The goldens depend on the pytest and pytest-rerunfailures versions they record.
        cls.provenance = ", ".join(f"{dist} {v or 'missing'}" for dist, v in versions.items())
        header = f"<!-- provenance: {cls.provenance} -->"
        golden = _JUNIT_TESTDATA / "expected" / "pytest.xml"
        cls.golden_skip = None
        if not _REGENERATE_JUNIT_GOLDENS and golden.exists() and header not in golden.read_text().splitlines():
            cls.golden_skip = (
                f"goldens were generated with other versions than {cls.provenance}; "
                "rerun with REGENERATE_JUNIT_GOLDENS=1"
            )
        with tempfile.TemporaryDirectory() as tmp:
            xml_path = Path(tmp) / "report.xml"
            # Run in place under test/ so test/conftest.py and pytest.ini apply, as in CI.
            # The goldens pin how skipped tests are reported, so keep the collection
            # pruner from deselecting the mark-skipped fixtures.
            proc = subprocess.run(
                [
                    sys.executable, "-m", "pytest", "junit_xml_testdata/pytest_suite.py",
                    f"--junit-xml-reruns={xml_path}", "-p", "no:cacheprovider", "-p", "no:prune_skips_plugin", "-q",
                ],
                cwd=_JUNIT_TESTDATA.parent,
                env=_FIXTURE_CHILD_ENV,
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
            cls.output = proc.stdout + proc.stderr
        super().setUpClass()

    def _assert_matches_golden(self, name: str, doc: str) -> None:
        golden = _JUNIT_TESTDATA / "expected" / f"{name}.xml"
        if _REGENERATE_JUNIT_GOLDENS:
            golden.write_text(
                "<!-- generated by test/test_testing.py -->\n"
                f"<!-- provenance: {self.provenance} -->\n{doc}"
            )
            return
        if self.golden_skip:
            self.skipTest(self.golden_skip)
        self.assertTrue(golden.exists(), f"{golden} is missing; generate it with REGENERATE_JUNIT_GOLDENS=1")
        expected = "".join(
            ln for ln in golden.read_text().splitlines(keepends=True) if not ln.startswith("<!--")
        )
        self.assertMultiLineEqual(expected, doc, f"{golden} is stale; rerun with REGENERATE_JUNIT_GOLDENS=1")

    def test_pytest_outcome_counts(self) -> None:
        # Unlike the goldens, these hold on every pytest version.
        if not self.has_rerunfailures:
            self.skipTest("the fixture suite needs pytest-rerunfailures")
        self.assertNotIn("INTERNALERROR", self.output)
        counts = _count_junit_tags(_normalize_junit_xml(self.raw_xml))
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

    def test_pytest_outcome_shapes(self) -> None:
        self._assert_matches_golden("pytest", _normalize_junit_xml(self.raw_xml))

    def test_sanitize_pytest_xml(self) -> None:
        with TemporaryFileName() as path:
            Path(path).write_text(self.raw_xml)
            sanitize_pytest_xml(path)
            normalized = _normalize_junit_xml(Path(path).read_text())
        self._assert_matches_golden("pytest_sanitized", normalized)


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

    def _def_flag(self, name, **kwargs):
        from torch.testing._internal.common_utils import TestEnvironment
        self._defined.append(name)
        kwargs.setdefault("include_in_repro", False)
        return TestEnvironment.def_flag(name, **kwargs)

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


instantiate_device_type_tests(TestAssertCloseMultiDevice, globals(), only_for=("cuda", "xpu"), allow_xpu=True)


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


class TestSkipReasonAtCollection(TestCase):
    """skip_reason_at_collection is the decision behind the collection pruner in
    test/conftest.py: which skips can be made before a test runs."""

    _CU = "torch.testing._internal.common_utils"
    _CDT = "torch.testing._internal.common_device_type"

    def _lists(self, slow=None, with_slow=True, skip_fast=False, disabled=None):
        # Pin the slow/disabled lists and mode flags so the expectations hold in
        # the slow, rerun-disabled and fbcode CI configs too.
        stack = contextlib.ExitStack()
        for name, value in (("slow_tests_dict", slow or {}), ("TEST_WITH_SLOW", with_slow), ("TEST_SKIP_FAST", skip_fast),
                            ("disabled_tests_dict", disabled or {}), ("IS_SANDCASTLE", False), ("RERUN_DISABLED_TESTS", False)):
            stack.enter_context(unittest.mock.patch(f"{self._CU}.{name}", value))
        for name in ("IS_SANDCASTLE", "IS_FBCODE"):
            stack.enter_context(unittest.mock.patch(f"{self._CDT}.{name}", False))
        return stack

    def _reason(self, cls, name):
        return skip_reason_at_collection(cls, getattr(cls, name), name)

    def test_pre_test_hooks_exist(self):
        from torch.testing._internal.common_utils import _PRE_TEST_HOOKS

        self.assertEqual([name for name in _PRE_TEST_HOOKS if not hasattr(TestCase, name)], [])

    def test_static_marks(self):
        class Static(TestCase):
            @unittest.skip("static method mark")
            def test_marked(self):
                pass

            def test_plain(self):
                pass

        @unittest.skip("static class mark")
        class StaticClass(TestCase):
            def test_in_skipped_class(self):
                pass

        with self._lists():
            self.assertEqual(self._reason(Static, "test_marked"), "static method mark")
            self.assertIsNone(self._reason(Static, "test_plain"))
            self.assertEqual(self._reason(StaticClass, "test_in_skipped_class"), "static class mark")

    def test_recorded_conditions(self):
        flag = [True]

        class Recorded(TestCase):
            def test_recorded(self):
                pass

            @skipIfTorchDynamo("policy")
            def test_dynamo(self):
                pass

            def test_raises(self):
                pass

        add_skip_condition(Recorded.test_recorded, lambda cls: flag[0], "recorded condition")
        add_skip_condition(Recorded.test_raises, lambda cls: 1 / 0, "raising condition")
        with self._lists():
            self.assertEqual(self._reason(Recorded, "test_recorded"), "recorded condition")
            flag[0] = False
            self.assertIsNone(self._reason(Recorded, "test_recorded"))
            with unittest.mock.patch(f"{self._CU}.TEST_WITH_TORCHDYNAMO", True):
                self.assertEqual(self._reason(Recorded, "test_dynamo"), "policy")
            with unittest.mock.patch(f"{self._CU}.TEST_WITH_TORCHDYNAMO", False):
                self.assertIsNone(self._reason(Recorded, "test_dynamo"))
            # A predicate that raises keeps the test.
            self.assertIsNone(self._reason(Recorded, "test_raises"))

    @parametrize("own_constructor", [False, True])
    @parametrize("mutation", [None, "parent_constructor", "parent_binder", "binder", "copied_owner", "cooperative_constructor", "audited_cooperative_constructor", "changed_bases", "standard_constructor", "standard_runner"])
    def test_class_owned_skip_mark_contract(self, own_constructor, mutation):
        from torch.testing._internal.common_utils import preserves_unittest_skip_marks

        @preserves_unittest_skip_marks("bind")
        class Parent(TestCase):
            def __init__(self, methodName="runTest"):
                super().__init__(methodName)
                setattr(self, methodName, self.bind(getattr(self, methodName)))

            def bind(self, fn):
                return fn

            @unittest.skip("existing skip")
            def test_marked(self):
                self.fail("the original mark must skip")

            def test_runtime(self):
                pass

        original_parent = Parent.__dict__["_unittest_skip_mark_contract"]

        @preserves_unittest_skip_marks()
        class Child(Parent):
            if own_constructor:
                def __init__(self, methodName="runTest"):
                    unittest.TestCase.__init__(self, methodName)

            def bind(self, fn):
                @functools.wraps(fn)
                def wrapper():
                    return fn()
                return wrapper

        cls = Child
        if mutation in ("standard_constructor", "standard_runner"):
            hook = "__init__" if mutation == "standard_constructor" else "run"
            with unittest.mock.patch.object(TestCase, hook, lambda *args: None), self._lists():
                self.assertIsNone(self._reason(cls, "test_marked"))
            return
        if mutation == "parent_constructor":
            Parent.__init__ = lambda self, name: unittest.TestCase.__init__(self, name)
        elif mutation == "parent_binder":
            Parent.bind = lambda self, fn: lambda: None
        elif mutation == "binder":
            Child.bind = functools.wraps(Child.bind)(lambda self, fn: lambda: None)
        elif mutation == "copied_owner":
            class Copy(Parent):
                _unittest_skip_mark_contract = Child.__dict__["_unittest_skip_mark_contract"]
            cls = Copy
        elif mutation == "cooperative_constructor":
            class Other(TestCase):
                def __init__(self, methodName="runTest"):
                    super().__init__(methodName)
                    self.test_marked = lambda: None

            class Cooperative(Child, Other):
                pass
            cls = Cooperative
        elif mutation in ("audited_cooperative_constructor", "changed_bases"):
            @preserves_unittest_skip_marks("bind_other")
            class Other(TestCase):
                def __init__(self, methodName="runTest"):
                    super().__init__(methodName)
                    setattr(self, methodName, self.bind_other(getattr(self, methodName)))

                def bind_other(self, fn):
                    return fn

            class Cooperative(Child, Other):
                def bind_other(self, fn):
                    return lambda: None

            if mutation == "changed_bases":
                Child.__bases__ = (Parent, Other)
                Child.bind_other = Cooperative.bind_other
            else:
                cls = Cooperative

        with self._lists():
            self.assertEqual(self._reason(cls, "test_marked"), None if mutation else "existing skip")
            self.assertIs(Parent.__dict__["_unittest_skip_mark_contract"], original_parent)
            if mutation is None:
                result = unittest.TestResult()
                unittest.TestCase.run(cls("test_marked"), result)
                self.assertEqual([reason for _, reason in result.skipped], ["existing skip"])
                self.assertEqual(result.errors, [])
                predicate = unittest.mock.Mock(return_value=True)
                add_skip_condition(cls.test_runtime, predicate, "runtime condition")
                self.assertIsNone(self._reason(cls, "test_runtime"))
                predicate.assert_not_called()

    @parametrize("shadow_contract", [False, True])
    def test_audited_constructor_aliasing_live_default(self, shadow_contract):
        from torch.testing._internal.common_utils import preserves_unittest_skip_marks

        @preserves_unittest_skip_marks()
        class Audited(TestCase):
            def __init__(self, methodName="runTest"):
                unittest.TestCase.__init__(self, methodName)
                fn = getattr(self, methodName)

                @functools.wraps(fn)
                def wrapper():
                    try:
                        fn()
                    except unittest.SkipTest:
                        pass

                setattr(self, methodName, wrapper)

            def test_runtime(self):
                self.skipTest("existing runtime skip")

        add_skip_condition(Audited.test_runtime, lambda cls: True, "existing runtime skip")
        cls = Audited
        if shadow_contract:
            class Child(Audited):
                _unittest_skip_mark_contract = None
            cls = Child
        case = cls("test_runtime")
        with unittest.mock.patch.object(TestCase, "__init__", Audited.__init__), self._lists():
            self.assertIsNone(self._reason(cls, "test_runtime"))
            result = unittest.TestResult()
            unittest.TestCase.run(case, result)
            self.assertEqual(result.testsRun, 1)
            self.assertEqual((result.skipped, result.errors, result.failures), ([], [], []))

    def test_lazy_condition_is_not_evaluated_at_collection(self):
        condition = unittest.mock.Mock(return_value=True)

        class Lazy(TestCase):
            @lazy_skip_if(condition, "runtime condition")
            def test_lazy(self):
                pass

        with self._lists():
            self.assertIsNone(self._reason(Lazy, "test_lazy"))
            condition.assert_not_called()
            with self.assertRaisesRegex(unittest.SkipTest, "runtime condition"):
                Lazy("test_lazy").test_lazy()
            condition.assert_called_once_with()

    def test_inductor_device_skip_uses_captured_condition(self):
        from torch.testing._internal.inductor_utils import skipDeviceIf

        enabled = [True]
        checks = []
        calls = []

        class Condition:
            def __bool__(self):
                checks.append("checked")
                return enabled[0]

        class Fixed(TestCase):
            device = "cpu"
            _device_at_collection = "cpu"

            @skipDeviceIf(Condition(), "existing CPU skip", device="cpu")
            def test_runs(self):
                calls.append(self.device)

        class Other(Fixed):
            device = "cuda"
            _device_at_collection = "cuda"

        class Unrecorded(Fixed):
            device = "cuda"

        enabled[0] = False
        with self._lists():
            self.assertEqual(self._reason(Fixed, "test_runs"), "existing CPU skip")
            with self.assertRaisesRegex(unittest.SkipTest, "existing CPU skip"):
                Fixed("test_runs").test_runs()
            for cls in (Other, Unrecorded):
                self.assertIsNone(self._reason(cls, "test_runs"))
                cls("test_runs").test_runs()
        self.assertEqual(checks, ["checked"])
        self.assertEqual(calls, ["cuda", "cuda"])

    def test_python_version_predicate_stays_lazy(self):
        from torch.testing._internal.common_utils import skipIfPythonVersionMismatch

        predicate = unittest.mock.Mock(return_value=False)

        @skipIfPythonVersionMismatch(predicate)
        def test_runs():
            return "ran"

        predicate.assert_not_called()
        self.assertIsNone(skip_reason_at_collection(TestCase, test_runs, "test_runs"))
        predicate.return_value = True
        self.assertEqual(test_runs(), "ran")
        predicate.assert_called_once_with(*sys.version_info[:3])

    def test_expected_failure_consumes_runtime_skip(self):
        from torch.testing._internal.common_device_type import expectedFailure

        class Expected(TestCase):
            device_type = "cpu"

            @expectedFailure("cpu")
            @skipIf(True, "inner skip")
            def test_runs(self):
                self.fail("must raise SkipTest first")

        with self._lists():
            self.assertIsNone(self._reason(Expected, "test_runs"))
            Expected("test_runs").test_runs()

    @parametrize("wrapper_kind", ["device", "count", "sandcastle", "sandcastle_if"])
    @parametrize("internal_mode", ["IS_FBCODE", "IS_SANDCASTLE"])
    def test_internal_pass_takes_precedence_over_inner_skip(self, wrapper_kind, internal_mode):
        from torch.testing._internal.common_utils import skip_but_pass_in_sandcastle, skip_but_pass_in_sandcastle_if

        with self._lists(), unittest.mock.patch(f"{self._CDT}.{internal_mode}", True), \
                unittest.mock.patch(f"{self._CU}.IS_SANDCASTLE", True):
            wrappers = {
                "device": onlyCUDA,
                "count": deviceCountAtLeast(2),
                "sandcastle": skip_but_pass_in_sandcastle("pass internally"),
                "sandcastle_if": skip_but_pass_in_sandcastle_if(True, "pass internally"),
            }

            class Passing(TestCase):
                device_type = "cpu"

                @wrappers[wrapper_kind]
                @skipIf(True, "inner skip")
                def test_runs(self, devices=None):
                    self.fail("the outer wrapper must return first")

            self.assertIsNone(self._reason(Passing, "test_runs"))
            Passing("test_runs").test_runs([])

    @parametrize("entrypoint", ["init", "run"])
    @parametrize("static", [False, True])
    def test_custom_execution_can_override_a_runtime_skip(self, entrypoint, static):
        class Original(TestCase):
            @(unittest.skip("inner skip") if static else skipIf(True, "inner skip"))
            def test_runs(self):
                self.fail("must raise SkipTest first")

        class CustomInit(Original):
            def __init__(self, methodName="runTest"):
                super().__init__(methodName)
                self.test_runs = lambda: None

        class CustomRun(Original):
            def run(self, result):
                result.startTest(self)
                with contextlib.suppress(unittest.SkipTest):
                    self.test_runs()
                result.addSuccess(self)
                result.stopTest(self)

        cls = CustomInit if entrypoint == "init" else CustomRun
        with self._lists():
            self.assertIsNone(self._reason(cls, "test_runs"))
            result = unittest.TestResult()
            case = cls("test_runs")
            if entrypoint == "init":
                unittest.TestCase.run(case, result)
            else:
                case.run(result)
            self.assertEqual(result.testsRun, 1)
            self.assertEqual(result.skipped, [])
            self.assertEqual(result.errors, [])
            self.assertEqual(result.failures, [])

    def test_xfail_takes_precedence_over_runtime_skip(self):
        import pytest
        from torch.testing._internal.common_utils import xfailIfNoAcceleratorTriton

        with unittest.mock.patch("torch.utils._triton.has_triton", return_value=False):
            class Expected(TestCase):
                device_type = "cuda"

                @xfailIfNoAcceleratorTriton
                @skipIf(True, "inner skip")
                def test_runs(self):
                    self.fail("must xfail first")

        with self._lists():
            self.assertIsNone(self._reason(Expected, "test_runs"))
            with self.assertRaises(pytest.xfail.Exception):
                Expected("test_runs").test_runs()

    @parametrize("decorator_kind", ["only", "skip", "native"])
    def test_mutable_device_selection_stays_at_runtime(self, decorator_kind):
        from torch.testing._internal.common_device_type import onlyOn, onlyNativeDeviceTypesAnd

        devices = ["custom"] if decorator_kind == "skip" else []
        decorators = {
            "only": onlyOn(devices),
            "skip": skipIf(True, "inner skip", device_type=devices),
            "native": onlyNativeDeviceTypesAnd(devices),
        }

        class Dynamic(DeviceTypeTestBase):
            device_type = "custom"
            _device_type_at_collection = "custom"

            @decorators[decorator_kind]
            def test_runs(self):
                pass

        with self._lists():
            self.assertIsNone(self._reason(Dynamic, "test_runs"))
            devices[:] = [] if decorator_kind == "skip" else ["custom"]
            Dynamic("test_runs").test_runs()

    @parametrize("class_target", [False, True])
    @parametrize("enabled", [False, True])
    def test_stable_skip_decorator(self, class_target, enabled):
        calls = []

        class Stable(TestCase):
            def test_runs(self):
                calls.append("test")

        with self._lists(), unittest.mock.patch(f"{self._CU}.TEST_XPU", enabled):
            if class_target:
                self.assertIs(skipIfXpu(Stable, msg="policy"), Stable)
            else:
                Stable.test_runs = skipIfXpu(Stable.test_runs, msg="policy")
            self.assertEqual(self._reason(Stable, "test_runs"), "skipIfXpu: policy" if enabled else None)
            result = unittest.TestResult()
            # Avoid nesting TestCase.run's Dynamo wrapper inside this test's wrapper.
            unittest.TestCase.run(Stable("test_runs"), result)
            self.assertEqual(calls, [] if enabled else ["test"])
            self.assertEqual(len(result.skipped), int(enabled))
            self.assertEqual(result.errors, [])
            self.assertEqual(result.failures, [])

    @parametrize("enabled", [False, True])
    def test_rocm_only_condition(self, enabled):
        from torch.testing._internal.common_utils import runOnRocm

        calls = []

        class RocmOnly(TestCase):
            @runOnRocm
            def test_runs(self):
                calls.append("test")

        with self._lists(), unittest.mock.patch(f"{self._CU}.TEST_WITH_ROCM", enabled):
            reason = None if enabled else "test currently only works on the ROCm stack"
            self.assertEqual(self._reason(RocmOnly, "test_runs"), reason)
            result = unittest.TestResult()
            unittest.TestCase.run(RocmOnly("test_runs"), result)
            self.assertEqual(calls, ["test"] if enabled else [])
            self.assertEqual(len(result.skipped), int(not enabled))
            self.assertEqual(result.errors, [])
            self.assertEqual(result.failures, [])

    @parametrize("custom_setup", [None, "setUp", "setUpClass"])
    def test_device_decorators_through_instantiation(self, custom_setup):
        class TestGeneric(TestCase):
            if custom_setup == "setUp":
                @preserves_device_selection
                def setUp(self):
                    super().setUp()
                    self.ready = True
            elif custom_setup == "setUpClass":
                @classmethod
                @preserves_device_selection
                def setUpClass(cls):
                    super().setUpClass()
                    cls.ready = True

            @onlyCUDA
            def test_cuda_only(self, device):
                pass

            @skipCPUIf(True, "cpu dep")
            def test_cpu_dep(self, device):
                pass

            @skipCUDAIfNoMagma
            def test_magma(self, device):
                pass

            @deviceCountAtLeast(2)
            def test_two_devices(self, devices):
                pass

            def test_plain(self, device):
                pass

        if custom_setup:
            with self._lists(skip_fast=True):
                self.assertIsNone(self._reason(TestGeneric, "test_plain"))

        scope = {"TestGeneric": TestGeneric}
        # CUDA CI jobs set PYTORCH_TESTING_DEVICE_ONLY_FOR=cuda, which overrides only_for.
        env = {PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY: "", PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY: ""}
        with unittest.mock.patch.dict(os.environ, env):
            instantiate_device_type_tests(TestGeneric, scope, only_for="cpu")
        cpu = scope["TestGenericCPU"]
        with self._lists():
            self.assertEqual(self._reason(cpu, "test_cuda_only_cpu"), "Only runs on cuda")
            self.assertEqual(self._reason(cpu, "test_cpu_dep_cpu"), "cpu dep")
            self.assertIsNone(self._reason(cpu, "test_magma_cpu"))
            self.assertIsNone(self._reason(cpu, "test_plain_cpu"))
            expected_count = None if custom_setup else "fewer than 2 devices detected"
            self.assertEqual(self._reason(cpu, "test_two_devices_cpu"), expected_count)
            with self.assertRaisesRegex(unittest.SkipTest, "Only runs on cuda"):
                cpu("test_cuda_only_cpu").test_cuda_only_cpu()

    def test_device_template_without_setup_hooks(self):
        class NoSetup:
            pass

        class TestGeneric(NoSetup):
            @onlyCUDA
            def test_cuda_only(self, device):
                self.fail("CPU must retain the existing runtime skip")

        scope = {"TestGeneric": TestGeneric}
        with unittest.mock.patch(f"{self._CDT}.get_desired_device_type_test_bases", return_value=[CPUTestBase]):
            instantiate_device_type_tests(TestGeneric, scope)
        cpu = scope["TestGenericCPU"]
        with self._lists():
            self.assertIsNone(self._reason(cpu, "test_cuda_only_cpu"))
            with self.assertRaisesRegex(unittest.SkipTest, "Only runs on cuda"):
                cpu("test_cuda_only_cpu").test_cuda_only_cpu()

    @parametrize("setup_kind", ["device", "bypass", "wrapped", "inherited"])
    def test_setup_device_selection_opt_in_is_not_inherited(self, setup_kind):
        class Audited(TestCase):
            @preserves_device_selection
            def setUp(self):
                super().setUp()

        calls = []

        class Dynamic(Audited):
            @onlyCUDA
            def test_runs(self, device):
                calls.append("test")

        if setup_kind != "inherited":
            def setUp(self):
                TestCase.setUp(self)
                if setup_kind == "bypass":
                    self.bypass_device_restrictions = True
                else:
                    self.device_type = "cuda"

            Dynamic.setUp = functools.wraps(Audited.setUp)(setUp) if setup_kind == "wrapped" else setUp

        scope = {"Dynamic": Dynamic}
        env = {PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY: "", PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY: ""}
        with unittest.mock.patch.dict(os.environ, env):
            instantiate_device_type_tests(Dynamic, scope, only_for="cpu")
        cpu = scope["DynamicCPU"]
        with self._lists():
            self.assertIsNone(self._reason(cpu, "test_runs_cpu"))
            result = unittest.TestResult()
            unittest.TestCase.run(cpu("test_runs_cpu"), result)
            self.assertEqual(calls, [] if setup_kind == "inherited" else ["test"])
            self.assertEqual(len(result.skipped), int(setup_kind == "inherited"))
            self.assertEqual(result.errors, [])
            self.assertEqual(result.failures, [])

    @parametrize("required_devices", [1, 2])
    @parametrize("decorator", [skipCPUIf(False, "unused"), skipCUDAIf(True, "unused"), onlyNativeDeviceTypes, common_cuda.with_tf32_off])
    def test_device_count_uses_captured_arguments(self, required_devices, decorator):
        calls = []
        test_op = OpInfo('test_op', op=lambda x: x, dtypes=floating_types(), sample_inputs_func=lambda _: [], decorators=(decorator,))

        class Generic(TestCase):
            @onlyAccelerator
            @deviceCountAtLeast(required_devices)
            @ops([test_op], dtypes=OpDTypes.none, skip_if_dynamo=False)
            def test_runs(self, devices, op):
                calls.append(len(devices))

        scope = {"Generic": Generic}
        with unittest.mock.patch(f"{self._CDT}.get_desired_device_type_test_bases", return_value=[MPSTestBase]):
            instantiate_device_type_tests(Generic, scope)
        cls = scope["GenericMPS"]
        name = "test_runs_test_op_mps"
        with self._lists(), unittest.mock.patch.object(MPSTestBase, "get_all_devices", side_effect=AssertionError("queried again")):
            self.assertEqual(self._reason(cls, name), "fewer than 2 devices detected" if required_devices == 2 else None)
            result = unittest.TestResult()
            unittest.TestCase.run(cls(name), result)
            self.assertEqual(len(result.skipped), int(required_devices == 2))
            self.assertEqual(calls, [] if required_devices == 2 else [1])
            self.assertEqual(result.errors, [])
            self.assertEqual(result.failures, [])
            with unittest.mock.patch(f"{self._CDT}.IS_FBCODE", True):
                self.assertIsNone(self._reason(cls, name))

    @parametrize("mutation", ["getter", "precision_hook", "precision_value", "tolerance_value", "precision", "rel_tol"])
    def test_device_count_defers_for_argument_mutation(self, mutation):
        devices = ["mps:0"]
        calls = []

        class Generic(TestCase):
            @deviceCountAtLeast(2)
            @dtypes(torch.float32)
            def test_runs(self, devices, dtype):
                calls.append(len(devices))

        if mutation == "precision_hook":
            def apply_precision(self, test, kwargs):
                kwargs["devices"].append("mps:1")

            Generic._apply_precision_override_for_test = apply_precision
        elif mutation in ("precision_value", "tolerance_value"):
            def override(self, test, dtype):
                test._device_count_at_collection[1].num_required_devices = 1
                return 0.0 if mutation == "precision_value" else (0.0, 0.0)

            setattr(Generic, "_get_precision_override" if mutation == "precision_value" else "_get_tolerance_override", override)
        elif mutation in ("precision", "rel_tol"):
            def override(self):
                test = getattr(type(self), self._testMethodName)
                test._device_count_at_collection[1].num_required_devices = 1
                return 0.0

            setattr(Generic, mutation, property(override, lambda self, value: None))
        getter_patch = unittest.mock.patch.object(MPSTestBase, "get_all_devices", return_value=devices)
        scope = {"Generic": Generic}
        with unittest.mock.patch(f"{self._CDT}.get_desired_device_type_test_bases", return_value=[MPSTestBase]), \
                getter_patch if mutation == "getter" else contextlib.nullcontext():
            instantiate_device_type_tests(Generic, scope)
        cls = scope["GenericMPS"]
        with self._lists():
            self.assertIsNone(self._reason(cls, "test_runs_mps_float32"))
            if mutation == "getter":
                devices.append("mps:1")
            result = unittest.TestResult()
            unittest.TestCase.run(cls("test_runs_mps_float32"), result)
            self.assertEqual(calls, [2] if mutation in ("getter", "precision_hook") else [1])
            self.assertEqual(result.skipped, [])
            self.assertEqual(result.errors, [])
            self.assertEqual(result.failures, [])

    @parametrize("wrapped", [False, True])
    def test_class_setup_device_selection_requires_exact_owner(self, wrapped):
        class Audited(TestCase):
            @classmethod
            @preserves_device_selection
            def setUpClass(cls):
                super().setUpClass()

        calls = []

        class Generic(Audited):
            @onlyCUDA
            def test_runs(self, device):
                calls.append("test")

        if wrapped:
            @functools.wraps(Audited.setUpClass.__func__)
            def setup(cls):
                Audited.setUpClass.__func__(cls)
                cls.device_type = "cuda"
            Generic.setUpClass = classmethod(setup)
        scope = {"Generic": Generic}
        with unittest.mock.patch(f"{self._CDT}.get_desired_device_type_test_bases", return_value=[CPUTestBase]):
            instantiate_device_type_tests(Generic, scope)
        cls = scope["GenericCPU"]
        with self._lists():
            self.assertIsNone(self._reason(cls, "test_runs_cpu"))
            cls.setUpClass()
            self.addCleanup(cls.tearDownClass)
            result = unittest.TestResult()
            unittest.TestCase.run(cls("test_runs_cpu"), result)
            self.assertEqual(calls, ["test"] if wrapped else [])
            self.assertEqual(len(result.skipped), int(not wrapped))
            self.assertEqual(result.errors, [])
            self.assertEqual(result.failures, [])

    @parametrize("rewrites_arguments", [False, True])
    def test_device_count_defers_through_unknown_wrappers(self, rewrites_arguments):
        def decorate(fn):
            @functools.wraps(fn)
            def wrapped(self, devices):
                if rewrites_arguments:
                    devices.append("mps:1")
                with contextlib.suppress(unittest.SkipTest):
                    return fn(self, devices)
            return wrapped

        class Generic(TestCase):
            @decorate
            @onlyAccelerator
            @deviceCountAtLeast(2)
            def test_runs(self, devices):
                self.assertEqual(len(devices), 2)

        scope = {"Generic": Generic}
        with unittest.mock.patch(f"{self._CDT}.get_desired_device_type_test_bases", return_value=[MPSTestBase]):
            instantiate_device_type_tests(Generic, scope)
        cls = scope["GenericMPS"]
        with self._lists():
            self.assertIsNone(self._reason(cls, "test_runs_mps"))
            result = unittest.TestResult()
            unittest.TestCase.run(cls("test_runs_mps"), result)
            self.assertEqual(result.skipped, [])
            self.assertEqual(result.errors, [])
            self.assertEqual(result.failures, [])

    def test_privateuse1_bypass_defers_to_run_time(self):
        # Out-of-tree backends opt into @onlyOn tests per instance in setUp
        # (bypass_device_restrictions), so nothing is decided for them here.
        class FakeOpenReg(PrivateUse1TestBase):
            device_type = "openreg"

        @onlyCUDA
        def test_cuda_only(self, device):
            pass

        with self._lists():
            self.assertIsNone(skip_reason_at_collection(FakeOpenReg, test_cuda_only, "test_cuda_only"))

    def test_unknown_conditions_keep_the_test(self):
        class FakeCUDA(DeviceTypeTestBase):
            device_type = "cuda"
            _device_type_at_collection = "cuda"

        @skipCUDAIfNoMagma
        def test_magma(self, device):
            pass

        @skipCUDAIf(True, "cuda dep")
        def test_dep(self, device):
            pass

        # no_magma is a class attribute that setUpClass sets, so it is unknown here.
        self.assertIsNone(skip_reason_at_collection(FakeCUDA, test_magma, "test_magma"))
        self.assertEqual(skip_reason_at_collection(FakeCUDA, test_dep, "test_dep"), "cuda dep")
        # Without a class the device type is unknown.
        self.assertIsNone(skip_reason_at_collection(None, test_dep, "test_dep"))

    def test_plain_test_cases(self):
        # Device-agnostic skipIf also decorates plain TestCases (dynamo/test_exc.py),
        # which have no device_type, and helpers build a TestCase with the default
        # methodName and call setUp() directly.
        class Plain(TestCase):
            @skipIf(True, "plain dep")
            def test_dep(self):
                pass

            @skipIf(False, "plain dep")
            def test_no_dep(self):
                pass

        with self._lists():
            self.assertEqual(self._reason(Plain, "test_dep"), "plain dep")
            self.assertIsNone(self._reason(Plain, "test_no_dep"))
            Plain("test_no_dep").test_no_dep()
            check_if_enable(Plain())

    @parametrize("descriptor", [False, True])
    @parametrize("device_base", [False, True])
    def test_instance_device_type_keeps_runtime_decision(self, descriptor, device_base):
        base = DeviceTypeTestBase if device_base else TestCase

        class Dynamic(base):
            device_type = property(lambda self: "cuda") if descriptor else "cpu"

            def setUp(self):
                super().setUp()
                if not descriptor:
                    self.device_type = "cuda"

            @onlyCUDA
            def test_runs(self):
                self.assertEqual(self.device_type, "cuda")

        with self._lists():
            self.assertIsNone(self._reason(Dynamic, "test_runs"))
            result = unittest.TestResult()
            unittest.TestCase.run(Dynamic("test_runs"), result)
            self.assertEqual(result.skipped, [])
            self.assertEqual(result.errors, [])
            self.assertEqual(result.failures, [])

    def test_slow_list_and_skip_fast(self):
        class TestLists(TestCase):
            def test_listed_slow(self):
                pass

            def test_fast(self):
                pass

            @slowTest
            def test_decorated_slow(self):
                pass

        slow = {"test_listed_slow (__main__.TestLists)": 10.0}
        slow_reason = "test is slow; run with PYTORCH_TEST_WITH_SLOW to enable test"
        with self._lists(slow=slow, with_slow=False):
            self.assertEqual(self._reason(TestLists, "test_listed_slow"), slow_reason)
            self.assertEqual(self._reason(TestLists, "test_decorated_slow"), slow_reason)
            self.assertIsNone(self._reason(TestLists, "test_fast"))
        with self._lists(slow=slow, with_slow=True, skip_fast=True):
            self.assertIsNone(self._reason(TestLists, "test_listed_slow"))
            self.assertIsNone(self._reason(TestLists, "test_decorated_slow"))
            self.assertEqual(self._reason(TestLists, "test_fast"), "test is fast; we disabled it with PYTORCH_TEST_SKIP_FAST")

    @parametrize("changed_hook", [None, "generic", "native", "last_native", "generated", "inherited"])
    def test_generated_setup_preserves_slow_policy(self, changed_hook):
        class Generic(TestCase):
            def test_listed(self, device):
                pass

            def test_fast(self, device):
                pass

        scope = {"Generic": Generic}
        bases = [CPUTestBase, MPSTestBase] if changed_hook == "last_native" else [CPUTestBase]
        with unittest.mock.patch(f"{self._CDT}.get_desired_device_type_test_bases", return_value=bases):
            instantiate_device_type_tests(Generic, scope)
        cls = scope["GenericCPU"]
        if changed_hook == "inherited":
            cls = type("Inherited", (cls,), {})
        with contextlib.ExitStack() as stack:
            if changed_hook in ("generic", "native", "last_native", "generated"):
                target = {"generic": Generic, "native": CPUTestBase, "last_native": MPSTestBase, "generated": cls}[changed_hook]
                stack.enter_context(unittest.mock.patch.object(target, "setUpClass", classmethod(lambda cls: None)))
            slow = {f"test_listed_cpu (__main__.{cls.__name__})": 10.0}
            with self._lists(slow=slow, with_slow=False):
                reason = self._reason(cls, "test_listed_cpu")
                expected = "test is slow; run with PYTORCH_TEST_WITH_SLOW to enable test" if changed_hook is None else None
                self.assertEqual(reason, expected)
                if changed_hook is None:
                    with self.assertRaises(unittest.SkipTest):
                        cls("test_listed_cpu").setUp()
            with self._lists(skip_fast=True):
                expected = "test is fast; we disabled it with PYTORCH_TEST_SKIP_FAST" if changed_hook is None else None
                self.assertEqual(self._reason(cls, "test_fast_cpu"), expected)

    @parametrize("dispatch", ["_run_custom", "_callSetUp", "__getattribute__", "__setattr__",
                              "_reset_dynamo_if_imported", "addTypeEqualityFunc", "id", "wrap_with_policy",
                              "TEST_WITH_TORCHDYNAMO", "TEST_WITH_AOT_EAGER", "TEST_WITH_TORCHINDUCTOR"])
    def test_generated_setup_requires_default_dispatch(self, dispatch):
        calls = []

        class Generic(TestCase):
            def test_listed(self, device):
                calls.append("body")

            def test_control(self, device):
                calls.append("body")

        def record_success(self, result):
            result.startTest(self)
            calls.append("body")
            result.addSuccess(self)
            result.stopTest(self)

        compiled = dispatch.startswith("TEST_WITH_")
        if compiled:
            def compile_fn(self, fn, backend, nopython):
                return lambda result: record_success(self, result)

            Generic.compile_fn = compile_fn
        elif dispatch == "_run_custom":
            Generic._run_custom = record_success
        elif dispatch == "_callSetUp":
            Generic._callSetUp = lambda self: None
        elif dispatch == "__getattribute__":
            def getattribute(self, name):
                return (lambda: None) if name == "_callSetUp" else object.__getattribute__(self, name)

            Generic.__getattribute__ = getattribute
        elif dispatch in ("_reset_dynamo_if_imported", "addTypeEqualityFunc", "id", "wrap_with_policy"):
            def bypass_setup(self, *args):
                self._callSetUp = lambda: None
                return "test"

            setattr(Generic, dispatch, bypass_setup)
            Generic._do_cuda_non_default_stream = dispatch == "id"
            Generic._ignore_not_implemented_error = dispatch == "wrap_with_policy"
        else:
            def setattr_(self, name, value):
                if name == "_testMethodName":
                    value = "test_control_cpu"
                object.__setattr__(self, name, value)

            Generic.__setattr__ = setattr_

        if dispatch != "__setattr__":
            # Only the renamed control test runs TestCase.setUp.
            Generic.tearDown = unittest.TestCase.tearDown

        scope = {"Generic": Generic}
        with unittest.mock.patch(f"{self._CDT}.get_desired_device_type_test_bases", return_value=[CPUTestBase]):
            instantiate_device_type_tests(Generic, scope)
        cls = scope["GenericCPU"]
        slow = {"test_listed_cpu (__main__.GenericCPU)": 10.0}
        with self._lists(slow=slow, with_slow=False), contextlib.ExitStack() as stack:
            for flag in ("TEST_WITH_TORCHDYNAMO", "TEST_WITH_AOT_EAGER", "TEST_WITH_TORCHINDUCTOR"):
                stack.enter_context(unittest.mock.patch(f"{self._CU}.{flag}", flag == dispatch))
            if dispatch == "id":
                stack.enter_context(unittest.mock.patch(f"{self._CU}.IS_WINDOWS", False))
            self.assertIsNone(self._reason(cls, "test_listed_cpu"))
            result = unittest.TestResult()
            # Nested runners reset compiler stance outside the outer compiled region.
            torch.compiler.disable(cls("test_listed_cpu").run)(result)
            self.assertEqual(calls, ["body"])
            self.assertEqual(result.testsRun, 1)
            self.assertEqual(result.skipped, [])
            self.assertEqual(result.errors, [])
            self.assertEqual(result.failures, [])

    def test_device_count_rechecks_generated_dispatch(self):
        calls = []

        class Generic(TestCase):
            @deviceCountAtLeast(2)
            def test_runs(self, devices):
                raise AssertionError("the custom dispatcher must bypass this method")

        scope = {"Generic": Generic}
        with unittest.mock.patch(f"{self._CDT}.get_desired_device_type_test_bases", return_value=[CPUTestBase]):
            instantiate_device_type_tests(Generic, scope)
        cls = scope["GenericCPU"]
        with self._lists():
            self.assertEqual(self._reason(cls, "test_runs_cpu"), "fewer than 2 devices detected")
            cls._callTestMethod = lambda self, fn: calls.append("custom dispatcher")
            self.assertIsNone(self._reason(cls, "test_runs_cpu"))
            result = unittest.TestResult()
            unittest.TestCase.run(cls("test_runs_cpu"), result)
            self.assertEqual(calls, ["custom dispatcher"])
            self.assertEqual(result.skipped, [])
            self.assertEqual(result.errors, [])
            self.assertEqual(result.failures, [])

    @parametrize("hook", ["tearDown", "_callTearDown"])
    def test_device_count_preserves_teardown_failures(self, hook):
        class Generic(TestCase):
            @deviceCountAtLeast(2)
            def test_runs(self, devices):
                raise AssertionError("the existing device count check must skip the body")

        def fail_teardown(self):
            raise RuntimeError("existing teardown failure")

        setattr(Generic, hook, fail_teardown)
        scope = {"Generic": Generic}
        with unittest.mock.patch(f"{self._CDT}.get_desired_device_type_test_bases", return_value=[CPUTestBase]):
            instantiate_device_type_tests(Generic, scope)
        cls = scope["GenericCPU"]
        with self._lists():
            self.assertIsNone(self._reason(cls, "test_runs_cpu"))
            result = unittest.TestResult()
            unittest.TestCase.run(cls("test_runs_cpu"), result)
            self.assertEqual([reason for _, reason in result.skipped], ["fewer than 2 devices detected"])
            self.assertEqual(len(result.errors), 1)
            self.assertIn("existing teardown failure", result.errors[0][1])
            self.assertEqual(result.failures, [])

    @parametrize("listed_slow", [False, True])
    @parametrize("chain_setup", [False, True])
    def test_overridden_setup_defers_slow_and_fast_checks(self, listed_slow, chain_setup):
        class Overridden(TestCase):
            def setUp(self):
                if chain_setup:
                    super().setUp()

            def test_runs(self):
                pass

        slow = {"test_runs (__main__.Overridden)": 10.0} if listed_slow else {}
        with self._lists(slow=slow, with_slow=not listed_slow, skip_fast=not listed_slow):
            self.assertIsNone(self._reason(Overridden, "test_runs"))
            if chain_setup:
                with self.assertRaises(unittest.SkipTest):
                    Overridden("test_runs").setUp()
            else:
                Overridden("test_runs").setUp()

    def test_setup_class_can_change_slow_metadata(self):
        class Dynamic(TestCase):
            @classmethod
            def setUpClass(cls):
                super().setUpClass()
                cls.test_runs.slow_test = True

            def test_runs(self):
                pass

        with self._lists(skip_fast=True):
            self.assertIsNone(self._reason(Dynamic, "test_runs"))
            Dynamic.setUpClass()
            self.addCleanup(Dynamic.tearDownClass)
            Dynamic("test_runs").setUp()

    def test_disabled_tests_stay_at_run_time(self):
        class TestDisabled(TestCase):
            def test_disabled(self):
                pass

        disabled = {"test_disabled (__main__.TestDisabled)": ("https://github.com/pytorch/pytorch/issues/0", [])}  # @lint-ignore
        with self._lists(disabled=disabled):
            self.assertIsNone(self._reason(TestDisabled, "test_disabled"))
            with self.assertRaisesRegex(unittest.SkipTest, "disabled"):
                check_if_enable(TestDisabled("test_disabled"))


instantiate_parametrized_tests(TestSkipReasonAtCollection)


@skipIfTorchDynamo("subprocess test does not need Dynamo coverage")
@unittest.skipIf(TEST_WITH_CROSSREF, "subprocess test does not need crossref coverage")
@unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "onlyOn passes instead of skipping in fbcode, so the fixture counts differ")
class TestPruneSkipsAtCollection(TestCase):
    """The PruneSkipsPlugin hook in test/conftest.py, run in place under test/ as in CI."""

    @classmethod
    def setUpClass(cls) -> None:
        if importlib.util.find_spec("pytest") is None:
            raise unittest.SkipTest("the fixture suite needs pytest")
        super().setUpClass()

    def _run_fixture(self, *extra_args: str, env=None, expected_returncode=0) -> str:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "prune_skips_testdata/pytest_suite.py", "-p", "no:cacheprovider", "-q", *extra_args],
            cwd=_JUNIT_TESTDATA.parent, env=_FIXTURE_CHILD_ENV | (env or {}), capture_output=True, text=True, timeout=300,
        )
        output = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, expected_returncode, output)
        return output

    def test_prunes_only_skips_decided_at_collection(self):
        output = self._run_fixture()
        self.assertIn("5 deselected", output)
        self.assertIn("2 skipped", output)
        self.assertIn("8 passed", output)
        self.assertIn("2 xfailed", output)

    def test_plugin_opt_out_reports_skips(self):
        output = self._run_fixture("-p", "no:prune_skips_plugin")
        self.assertIn("7 skipped", output)
        self.assertIn("8 passed", output)
        self.assertIn("2 xfailed", output)
        self.assertNotIn("deselected", output)

    def test_unresolvable_callable_is_kept(self):
        from conftest import PruneSkipsPlugin

        @unittest.skip("static class skip")
        class Skipped(TestCase):
            pass

        class UnresolvedItem:
            cls = Skipped
            name = "test_unresolved"

            @property
            def obj(self):
                raise RuntimeError("cannot resolve test callable")

            def iter_markers(self, name):
                return iter(())

            def get_closest_marker(self, name):
                return None

        item = UnresolvedItem()
        items = [item]
        config = unittest.mock.Mock()
        plugin = PruneSkipsPlugin()
        plugin.pytest_collection_modifyitems(config, items)
        self.assertEqual(items, [item])
        self.assertFalse(plugin.reasons)
        config.hook.pytest_deselected.assert_not_called()

    def test_prunes_before_shard_assignment(self):
        snapshots = []
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "shard.json"
            for shard_id in (1, 2):
                self._run_fixture(
                    "-p", "prune_skips_testdata.shard_probe", "--shard-probe-log", str(log),
                    "--num-shards=2", f"--shard-id={shard_id}",
                )
                snapshots.append(json.loads(log.read_text()))

        first, second = snapshots
        self.assertEqual(first["before"], second["before"])
        self.assertEqual(len(first["before"]), 11)
        excluded = {
            "TestPruneFixture::test_runs", "TestPruneFixture::test_static_mark",
            "TestCustomDispatch::test_static_mark",
            "TestPruneFixtureDeviceCPU::test_cuda_only_cpu", "test_pytest_mark", "test_pytest_static_condition",
        }
        shard_inputs = {nodeid.split("::", 1)[1] for nodeid in first["before"]}
        self.assertFalse(shard_inputs & excluded)
        self.assertIn("TestPruneFixture::test_body_skip", shard_inputs)
        self.assertIn("TestPruneFixture::test_lazy_skip", shard_inputs)
        self.assertIn("TestCustomDispatch::test_recorded_skip", shard_inputs)
        self.assertIn("test_pytest_xfail_before_unittest_skip", shard_inputs)
        self.assertFalse(set(first["after"]) & set(second["after"]))
        self.assertEqual(set(first["after"]) | set(second["after"]), set(first["before"]))
        self.assertEqual(first["after"], first["final"])
        self.assertEqual(second["after"], second["final"])

        # A checkpoint from another shard must not select that shard's test here.
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "resume.json"
            self._run_fixture(
                "-p", "prune_skips_testdata.shard_probe", "--shard-probe-log", str(log),
                "--num-shards=2", "--shard-id=1", "-p", "cacheprovider", "--rs=probe",
                "-o", f"cache_dir={directory}/cache", "--shard-probe-resume-from", second["after"][0],
            )
            self.assertEqual(json.loads(log.read_text())["final"], first["after"])

    def test_collection_failure_does_not_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "shard.json"
            self._run_fixture(
                "-p", "prune_skips_testdata.shard_probe", "--shard-probe-log", str(log),
                "-k", "(", expected_returncode=4,
            )
            self.assertFalse(log.exists())

    def test_overridden_setup_keeps_fast_tests_in_slow_config(self):
        output = self._run_fixture(
            "-k", "TestOverriddenSetup", env={"PYTORCH_TEST_WITH_SLOW": "1", "PYTORCH_TEST_SKIP_FAST": "1"},
        )
        self.assertIn("1 passed", output)
        self.assertIn("16 deselected", output)


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

    def test_ops_allowlist_empty_intersection(self, device):
        test_op = OpInfo('test_op', op=lambda x: x, dtypes=floating_types(), sample_inputs_func=lambda _: [])

        class TestParametrized(TestCase):
            @ops([test_op], dtypes=OpDTypes.none)
            @parametrize('flag', [False, True])
            def test_op_parametrized(self, device, op, flag):
                pass

        test_fn = TestParametrized.test_op_parametrized
        with unittest.mock.patch.object(type(self), 'op_allowlist', ['other_op']):
            cases = list(test_fn.parametrize_fn(test_fn, TestParametrized, type(self)))
            self.assertEqual(cases, [])
        with unittest.mock.patch.object(type(self), 'op_allowlist', ['test_op']):
            cases = list(test_fn.parametrize_fn(test_fn, TestParametrized, type(self)))
            self.assertEqual([kwargs['op'].name for _, _, kwargs, _ in cases], ['test_op', 'test_op'])
            self.assertEqual([kwargs['flag'] for _, _, kwargs, _ in cases], [False, True])

    @unittest.mock.patch.dict(os.environ, {'OPINFO_RESTRICT_TO_DSL': ''})
    def test_ops_empty_input_still_errors_with_allowlist(self, device):
        class TestParametrized(TestCase):
            @ops(iter(()))
            def test_op_parametrized(self, device, dtype, op):
                pass

        test_fn = TestParametrized.test_op_parametrized
        with unittest.mock.patch.object(type(self), 'op_allowlist', ['other_op']):
            with self.assertRaisesRegex(ValueError, 'An empty op_list was passed to @ops'):
                list(test_fn.parametrize_fn(test_fn, TestParametrized, type(self)))

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
                           "torch._native.ops.norm.rmsnorm_kernels",  # depends on cutlass
                           "torch._native.ops.polar.nvmath_impl",  # depends on nvmath-python, cuda-python
                           "torch._native.ops.reductions.kernel_general",  # depends on cutlass
                           "torch._native.ops.reductions.kernel_rowtile",  # depends on cutlass
                           "torch._native.ops.reductions.tile",  # depends on cutlass
                           "torch._native.ops.reductions.kernel_xcta",  # depends on cutlass
                           "torch._native.ops.reductions.kernel_coltile",  # depends on cutlass
                           "torch._native.ops.reductions.ordered",  # depends on cutlass
                           "torch._native.ops.scatter_add",  # depends on cutlass
                           "torch._native.ops.topk",  # depends on cutlass
                           "torch._native.ops.quantize_tensor.blockscaled_tma",  # depends on cutlass
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

    def test_import_aot_runtime_wrappers_before_dynamo(self):
        self._check_python_output(textwrap.dedent("""
            import sys
            import torch

            if "torch._dynamo" in sys.modules:
                raise AssertionError("Dynamo was loaded before runtime_wrappers")
            import torch._functorch._aot_autograd.runtime_wrappers
        """))

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

    def test_not_import_dynamo_from_user_kernel_hops(self) -> None:
        # torch._dynamo and torch._inductor are imported lazily: importing
        # triton_kernel_wrap, as an eager wrap_triton launch does, must import neither.
        out = self._check_python_output(
            "import sys;import torch._higher_order_ops.triton_kernel_wrap;"
            "print([m for m in ('torch._dynamo', 'torch._inductor') if m in sys.modules])"
        )
        # The last line: fbcode's interpreter can print warnings first.
        self.assertEqual(out.strip().splitlines()[-1], "[]")

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
    hw_classification = HardwareClassification.GENERIC

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

    def test_filter_suite_exclude(self):
        requirement = self._cu.HardwareClassification

        class GenericTest(TestCase):
            hw_classification = requirement.GENERIC

            def test_generic(self):
                pass

        class AcceleratorTest(TestCase):
            hw_classification = requirement.ACCELERATOR

            def test_accelerator(self):
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
                unittest.defaultTestLoader.loadTestsFromTestCase(cls)
                for cls in (GenericTest, AcceleratorTest, CudaTest, MissingClassificationTest)
            ]
        )
        exclude = frozenset({requirement.GENERIC})

        with self.subTest(requirement=None):
            loader = self._cu.HardwareClassificationTestLoader(None, exclude=exclude)
            self.assertEqual(
                self._suite_test_names(loader.get_filtered_suite(suite)),
                {"test_accelerator", "test_cuda", "test_missing_classification"},
            )

        with self.subTest(requirement="GENERIC CUDA"):
            loader = self._cu.HardwareClassificationTestLoader({requirement.GENERIC, requirement.CUDA}, exclude=exclude)
            self.assertEqual(self._suite_test_names(loader.get_filtered_suite(suite)), {"test_cuda"})

        with self.subTest(item="classless"):
            kept = []
            self._cu.filter_by_hw_classification([object()], None, get_class=lambda item: None, on_match=kept.append, exclude=exclude)
            self.assertEqual(len(kept), 1)

    # The fixture runs as the default config, so other test configs would only repeat it.
    @unittest.skipIf(IS_WINDOWS, "Skipping because doesn't work for windows")
    @unittest.skipIf(IS_SANDCASTLE, "Skipping because doesn't work on sandcastle")
    @skipIfTorchDynamo("subprocess test does not need Dynamo coverage")
    @unittest.skipIf(TEST_WITH_CROSSREF, "subprocess test does not need crossref coverage")
    def test_hw_classification_exclude_env_var(self):
        # Runs the fixture directly (unittest) and with --use-pytest, as run_test.py
        # runs test files, in place under test/ so test/conftest.py and pytest.ini apply.
        env = _FIXTURE_CHILD_ENV | {"PYTORCH_TEST_EXCLUDE_HW_CLASSIFICATION": "GENERIC,CPU"}
        fixture = _HW_CLASSIFICATION_TESTDATA / "hw_classification_suite.py"
        runners = [("unittest", [], [r"\bRan 2 tests\b"])]
        if _fixture_child_versions()["pytest"] is not None:
            runners.append(("pytest", ["--use-pytest", "-p", "no:cacheprovider"], [r"\b2 passed\b", r"\b2 deselected\b"]))
        for runner, extra_args, summaries in runners:
            with self.subTest(runner=runner):
                result = subprocess.run(
                    [sys.executable, str(fixture.relative_to(fixture.parents[1])), "-v", *extra_args],
                    cwd=fixture.parents[1],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                output = result.stdout + result.stderr
                self.assertEqual(result.returncode, 0, output)
                for summary in summaries:
                    self.assertRegex(output, summary)
                # -v lists each test that ran; the excluded labels must not be among them.
                ran = set(re.findall(r"\b(test_generic|test_cpu|test_cuda|test_unclassified)\b", output))
                self.assertEqual(ran, {"test_cuda", "test_unclassified"}, output)


instantiate_device_type_tests(TestOpInfoSampleFunctions, globals())
instantiate_parametrized_tests(TestImports)


@unittest.skipIf(IS_WINDOWS, "dynamo test failures are not looked up on Windows")
class TestDynamoTestFailuresDir(TestCase):
    def test_found_from_working_directory(self):
        # pytest-xdist workers of an installed build have neither a local
        # torch/../test nor a test file as __main__.
        from torch.testing._internal import dynamo_test_failures

        test_dir = os.path.dirname(os.path.realpath(__file__))
        with set_cwd(test_dir), tempfile.TemporaryDirectory() as tmp, unittest.mock.patch.object(
            dynamo_test_failures, "__file__", os.path.join(tmp, "a", "b", "c", "x.py")
        ), unittest.mock.patch.dict(
            sys.modules, {"__main__": unittest.mock.Mock(spec=[])}
        ):
            found = dynamo_test_failures.find_test_dir()
        self.assertIsNotNone(found)
        self.assertEqual(os.path.realpath(found), test_dir)


if __name__ == '__main__':
    run_tests()
