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
from functools import partial
from typing import Any

import torch
import torch.testing
from torch._dynamo import polyfills
from torch._logging._internal import trace_log
from torch.testing._internal.common_utils import (  # type: ignore[attr-defined]
    HardwareClassification,
    IS_WINDOWS,
    TEST_WITH_CROSSREF,
    TEST_WITH_TORCHDYNAMO,
    TestCase as TorchTestCase,
)

from . import config, utils


log = logging.getLogger(__name__)


_AutocastStateSpec = tuple[str, Callable[[], Any], Callable[[Any], None]]
_AutocastState = tuple[Any, ...]


def _autocast_nesting() -> int:
    # There is no direct getter for the autocast nesting counter, only
    # autocast_increment_nesting()/autocast_decrement_nesting(), so read it
    # via a no-net-effect increment+decrement pair.
    n = torch.autocast_increment_nesting()
    torch.autocast_decrement_nesting()
    return n - 1


def _restore_autocast_nesting(target: int) -> None:
    delta = _autocast_nesting() - target
    for _ in range(delta):
        if torch.autocast_decrement_nesting() == 0:
            torch.clear_autocast_cache()
    for _ in range(-delta):
        torch.autocast_increment_nesting()


def _autocast_state_specs() -> tuple[_AutocastStateSpec, ...]:
    # Enabled state and dtype are per-device; cache and nesting are shared.
    device_specs = tuple(
        spec
        for device in ("cpu", "cuda")
        for spec in (
            (
                f"{device} autocast enabled state",
                partial(torch.is_autocast_enabled, device),
                partial(torch.set_autocast_enabled, device),
            ),
            (
                f"{device} autocast dtype",
                partial(torch.get_autocast_dtype, device),
                partial(torch.set_autocast_dtype, device),
            ),
        )
    )
    return device_specs + (
        (
            "autocast cache enabled state",
            torch.is_autocast_cache_enabled,
            torch.set_autocast_cache_enabled,
        ),
        ("autocast nesting depth", _autocast_nesting, _restore_autocast_nesting),
    )


def _snapshot_autocast_state() -> _AutocastState:
    return tuple(get() for _, get, _ in _autocast_state_specs())


def _restore_autocast_state(snapshot: _AutocastState) -> None:
    for (_, _, set_), value in zip(_autocast_state_specs(), snapshot):
        set_(value)


def run_tests(needs: str | tuple[str, ...] = ()) -> None:
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
                canonicalize_output_graph_node_order=True,
            ),
        )

    def setUp(self) -> None:
        self._prior_is_grad_enabled = torch.is_grad_enabled()
        self._prior_autocast_state = _snapshot_autocast_state()
        self._prior_nested_graph_breaks = config.nested_graph_breaks
        config.nested_graph_breaks = True
        super().setUp()
        utils.counters.clear()
        self.handler = logging.NullHandler()
        trace_log.addHandler(self.handler)

    def _restore_prior_autocast_state(self) -> None:
        current_autocast_state = _snapshot_autocast_state()
        if current_autocast_state != self._prior_autocast_state:
            specs = _autocast_state_specs()
            changed = ", ".join(
                label
                for (label, _, _), prior, current in zip(
                    specs, self._prior_autocast_state, current_autocast_state
                )
                if prior != current
            )
            log.warning("Running test %s changed %s", self.id(), changed)
            _restore_autocast_state(self._prior_autocast_state)

    def _restore_prior_test_state(self) -> None:
        errors: list[tuple[str, Exception]] = []
        if self._prior_is_grad_enabled is not torch.is_grad_enabled():
            log.warning("Running test %s changed grad mode", self.id())
            try:
                torch.set_grad_enabled(self._prior_is_grad_enabled)
            except Exception as error:
                errors.append(("grad mode", error))
        try:
            self._restore_prior_autocast_state()
        except Exception as error:
            errors.append(("autocast state", error))
        try:
            config.nested_graph_breaks = self._prior_nested_graph_breaks
        except Exception as error:
            errors.append(("nested_graph_breaks", error))
        if errors:
            # Raise the first failure as primary, but preserve every later failure in logs.
            for label, error in errors[1:]:
                log.error(
                    "Additional failure restoring %s",
                    label,
                    exc_info=(type(error), error, error.__traceback__),
                )
            raise errors[0][1]

    def tearDown(self) -> None:
        trace_log.removeHandler(self.handler)
        for k, v in utils.counters.items():
            log.debug("%s %s", k, v.most_common())
        utils.counters.clear()
        torch._C._autograd._saved_tensors_hooks_enable()
        teardown_error: BaseException | None = None
        try:
            super().tearDown()
        except BaseException as error:
            teardown_error = error
            raise
        finally:
            try:
                self._restore_prior_test_state()
            except Exception:
                if teardown_error is None:
                    raise
                log.exception("Failed to restore test state during teardown")

    def before_cuda_memory_leak_check(self) -> None:
        super().before_cuda_memory_leak_check()
        utils.counters.clear()

    def assertEqual(self, x: Any, y: Any, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        if config.debug_disable_compile_counter:
            if isinstance(x, utils.CompileCounterInt) or isinstance(
                y, utils.CompileCounterInt
            ):
                return
            # skip checks like self.assertEqual(len(counters["graph_break"]), 1)
            if (
                (cur_frame := inspect.currentframe())
                and (upper_frame := cur_frame.f_back)
                and (upper_code := inspect.getframeinfo(upper_frame).code_context)
                and "counters" in upper_code[0]
            ):
                return
        return super().assertEqual(x, y, *args, **kwargs)

    def assertExpectedInline(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        if config.debug_disable_compile_counter:
            return
        kwargs["skip"] = kwargs.get("skip", 0) + 1
        return super().assertExpectedInline(*args, **kwargs)


class CPythonTestCase(TestCase):
    """
    Test class for CPython tests located in "test/cpython/v{Py_version}/*".

    This class enables specific features that are disabled by default, such as
    tracing through unittest methods.
    """

    hw_classification = HardwareClassification.GENERIC
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
    # pyrefly: ignore [bad-override]
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
        backend: str | Callable[..., Any],
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
        py_ver = re.search(r"/v([\d_]+)/", inspect.getfile(test_cls))
        if py_ver:
            py_ver = py_ver.group().strip(os.sep).replace("_", "").lstrip("v")  # type: ignore[assignment]
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
        search_path = inspect.getfile(cls)

        cpython_test_regex = (
            re.escape(os.path.join("cpython") + os.path.sep) + r"v(\d)_(\d{2})"
        )

        m = re.search(cpython_test_regex, search_path)
        if m:
            test_py_ver = tuple(map(int, m.groups()))
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
                enable_trace_load_build_class=True,
            ),
        )

    @contextlib.contextmanager
    def subTest(self, *args, **kwargs):
        # pytest 9.x addSubTest uses typing._GenericAlias calls that
        # Dynamo cannot trace. Use a no-op subTest instead.
        yield

    # pyrefly: ignore [implicit-any]
    def wrap_with_policy(self, method_name: str, policy: Callable) -> None:
        pass
