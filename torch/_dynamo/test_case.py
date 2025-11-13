"""Testing utilities for Dynamo, providing a specialized TestCase class and test running functionality.

This module extends PyTorch's testing framework with Dynamo-specific testing capabilities.
It includes:
- A custom TestCase class that handles Dynamo-specific setup/teardown
- Test running utilities with dependency checking
- Automatic reset of Dynamo state between tests
- Proper handling of gradient mode state
"""

import contextlib
import importlib
import inspect
import logging
import os
import re
import sys
import unittest
from collections.abc import Callable
from typing import Any, Union

import torch
import torch.testing
from torch._dynamo import polyfills
from torch._logging._internal import trace_log
from torch.testing._internal.common_utils import (  # type: ignore[attr-defined]
    IS_WINDOWS,
    TEST_WITH_CROSSREF,
    TEST_WITH_TORCHDYNAMO,
    TestCase as TorchTestCase,
)

from . import config, reset, utils


log = logging.getLogger(__name__)


def run_tests(needs: Union[str, tuple[str, ...]] = ()) -> None:
    from torch.testing._internal.common_utils import run_tests

    if TEST_WITH_TORCHDYNAMO or TEST_WITH_CROSSREF:
        return  # skip testing

    if (
        not torch.xpu.is_available()
        and IS_WINDOWS
        and os.environ.get("TORCHINDUCTOR_WINDOWS_TESTS", "0") == "0"
    ):
        return

    if isinstance(needs, str):
        needs = (needs,)
    for need in needs:
        if need == "cuda":
            if not torch.cuda.is_available():
                return
        else:
            try:
                importlib.import_module(need)
            except ImportError:
                return

    with torch._dynamo.config.patch(nested_graph_breaks=True):
        run_tests()


class TestCase(TorchTestCase):
    _exit_stack: contextlib.ExitStack

    @classmethod
    def tearDownClass(cls) -> None:
        cls._exit_stack.close()
        super().tearDownClass()

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._exit_stack = contextlib.ExitStack()  # type: ignore[attr-defined]
        cls._exit_stack.enter_context(  # type: ignore[attr-defined]
            config.patch(
                raise_on_ctx_manager_usage=True,
                suppress_errors=False,
                log_compilation_metrics=False,
            ),
        )

    def setUp(self) -> None:
        self._prior_is_grad_enabled = torch.is_grad_enabled()
        super().setUp()
        reset()
        utils.counters.clear()
        self.handler = logging.NullHandler()
        trace_log.addHandler(self.handler)

    def tearDown(self) -> None:
        trace_log.removeHandler(self.handler)
        for k, v in utils.counters.items():
            print(k, v.most_common())
        reset()
        utils.counters.clear()
        super().tearDown()
        if self._prior_is_grad_enabled is not torch.is_grad_enabled():
            log.warning("Running test changed grad mode")
            torch.set_grad_enabled(self._prior_is_grad_enabled)

    def assertEqual(self, x: Any, y: Any, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        if (
            config.debug_disable_compile_counter
            and isinstance(x, utils.CompileCounterInt)
            or isinstance(y, utils.CompileCounterInt)
        ):
            return
        return super().assertEqual(x, y, *args, **kwargs)

    # assertExpectedInline might also need to be disabled for wrapped nested
    # graph break tests


class CPythonTestCase(TestCase):
    """
    Test class for CPython tests located in "test/dynamo/CPython/Py_version/*".

    This class enables specific features that are disabled by default, such as
    tracing through unittest methods.
    """

    _stack: contextlib.ExitStack
    dynamo_strict_nopython = True

    # Restore original unittest methods to simplify tracing CPython test cases.
    assertEqual = unittest.TestCase.assertEqual  # type: ignore[assignment]
    assertNotEqual = unittest.TestCase.assertNotEqual  # type: ignore[assignment]
    assertTrue = unittest.TestCase.assertTrue
    assertFalse = unittest.TestCase.assertFalse
    assertIs = unittest.TestCase.assertIs
    assertIsNot = unittest.TestCase.assertIsNot
    assertIsNone = unittest.TestCase.assertIsNone
    assertIsNotNone = unittest.TestCase.assertIsNotNone
    assertIn = unittest.TestCase.assertIn
    assertNotIn = unittest.TestCase.assertNotIn
    assertIsInstance = unittest.TestCase.assertIsInstance
    assertNotIsInstance = unittest.TestCase.assertNotIsInstance
    assertAlmostEqual = unittest.TestCase.assertAlmostEqual
    assertNotAlmostEqual = unittest.TestCase.assertNotAlmostEqual
    assertGreater = unittest.TestCase.assertGreater
    assertGreaterEqual = unittest.TestCase.assertGreaterEqual
    assertLess = unittest.TestCase.assertLess
    assertLessEqual = unittest.TestCase.assertLessEqual
    assertRegex = unittest.TestCase.assertRegex
    assertNotRegex = unittest.TestCase.assertNotRegex
    assertCountEqual = unittest.TestCase.assertCountEqual
    assertMultiLineEqual = polyfills.assert_multi_line_equal
    assertSequenceEqual = polyfills.assert_sequence_equal
    assertListEqual = unittest.TestCase.assertListEqual
    assertTupleEqual = unittest.TestCase.assertTupleEqual
    assertSetEqual = unittest.TestCase.assertSetEqual
    assertDictEqual = polyfills.assert_dict_equal
    # pyrefly: ignore [bad-override]
    assertRaises = unittest.TestCase.assertRaises
    # pyrefly: ignore [bad-override]
    assertRaisesRegex = unittest.TestCase.assertRaisesRegex
    assertWarns = unittest.TestCase.assertWarns
    assertWarnsRegex = unittest.TestCase.assertWarnsRegex
    assertLogs = unittest.TestCase.assertLogs
    fail = unittest.TestCase.fail
    failureException = unittest.TestCase.failureException

    def compile_fn(
        self,
        fn: Callable[..., Any],
        backend: Union[str, Callable[..., Any]],
        nopython: bool,
    ) -> Callable[..., Any]:
        # We want to compile only the test function, excluding any setup code
        # from unittest

        method = getattr(self, self._testMethodName)
        method = torch._dynamo.optimize(backend, error_on_graph_break=nopython)(method)

        setattr(self, self._testMethodName, method)
        return fn

    def _dynamo_test_key(self) -> str:
        suffix = super()._dynamo_test_key()
        test_cls = self.__class__
        test_file = inspect.getfile(test_cls).split(os.sep)[-1].split(".")[0]
        py_ver = re.search(r"/([\d_]+)/", inspect.getfile(test_cls))
        if py_ver:
            py_ver = py_ver.group().strip(os.sep).replace("_", "")  # type: ignore[assignment]
        else:
            return suffix
        return f"CPython{py_ver}-{test_file}-{suffix}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls._stack.close()
        super().tearDownClass()

    @classmethod
    def setUpClass(cls) -> None:
        # Skip test if python versions doesn't match
        prefix = os.path.join("dynamo", "cpython") + os.path.sep
        regex = re.escape(prefix) + r"\d_\d{2}"
        search_path = inspect.getfile(cls)
        m = re.search(regex, search_path)
        if m:
            test_py_ver = tuple(map(int, m.group().removeprefix(prefix).split("_")))
            py_ver = sys.version_info[:2]
            if py_ver != test_py_ver:
                expected = ".".join(map(str, test_py_ver))
                got = ".".join(map(str, py_ver))
                raise unittest.SkipTest(
                    f"Test requires Python {expected} but got Python {got}"
                )
        else:
            raise unittest.SkipTest(
                f"Test requires a specific Python version but not found in path {inspect.getfile(cls)}"
            )

        super().setUpClass()
        cls._stack = contextlib.ExitStack()  # type: ignore[attr-defined]
        cls._stack.enter_context(  # type: ignore[attr-defined]
            config.patch(
                enable_trace_unittest=True,
            ),
        )
