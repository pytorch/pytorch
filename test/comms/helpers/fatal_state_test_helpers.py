#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import signal
import subprocess
import sys


class FatalStateTestMixin:
    """Mixin for tests that verify process-fatal scenarios via subprocess."""

    def run_subprocess(
        self, sentinel_var: str, timeout: int = 120
    ) -> subprocess.CompletedProcess:
        """Re-invoke current test binary with sentinel_var set.

        Under buck/PAR, sys.argv[0] is the test binary and re-invoking it
        runs the module-level sentinel check. Under `python -m pytest`,
        sys.argv[0] is the pytest entrypoint (e.g. .../pytest/__main__.py),
        so re-invoking it would just start pytest with no args. In that
        case, run the test class's source file directly so the
        sentinel-driven early-exit at the top of the test module fires.
        Fails the test on TimeoutExpired.
        """
        env = os.environ.copy()
        env[sentinel_var] = "1"
        entry = sys.argv[0]
        entry_name = os.path.basename(entry).lower()
        if "pytest" in entry_name or entry_name == "__main__.py":
            module_file = sys.modules[self.__class__.__module__].__file__
            if module_file:
                entry = module_file
        try:
            return subprocess.run(
                [sys.executable, entry],
                env=env,
                timeout=timeout,
                capture_output=True,
            )
        except subprocess.TimeoutExpired as e:
            raise AssertionError(
                f"Subprocess timed out after {timeout}s.\n"
                f"stdout: {(e.stdout or b'').decode(errors='replace')}\n"
                f"stderr: {(e.stderr or b'').decode(errors='replace')}"
            ) from e

    def assert_subprocess_aborted(
        self,
        result: subprocess.CompletedProcess,
        expected_stderr: str | None = None,
    ) -> None:
        """Assert subprocess was killed by SIGABRT, optionally check stderr."""
        # pyre-ignore[16]: unittest.TestCase.assertEqual
        self.assertEqual(
            result.returncode,
            -signal.SIGABRT,
            f"Expected SIGABRT (-{signal.SIGABRT}), "
            f"got {result.returncode}.\n"
            f"stderr: {result.stderr.decode(errors='replace')}",
        )
        if expected_stderr:
            # pyre-ignore[16]: unittest.TestCase.assertIn
            self.assertIn(
                expected_stderr,
                result.stderr.decode(errors="replace"),
                "Expected message not found in subprocess stderr.",
            )

    def assert_subprocess_failed(self, result: subprocess.CompletedProcess) -> None:
        """Assert subprocess exited with non-zero (for asymmetric rank scenarios)."""
        # pyre-ignore[16]: unittest.TestCase.assertNotEqual
        self.assertNotEqual(
            result.returncode,
            0,
            "Expected subprocess to fail (peers are dead).\n"
            f"stderr: {result.stderr.decode(errors='replace')}",
        )
