# Owner(s): ["module: inductor"]
"""Regression tests for CPU Inductor capability-probe diagnostics.

Requested CPU tests must stay discoverable when the toolchain probe fails, and
running them must surface the original InvalidCxxCompiler / CppCompileError
instead of "Ran 0 tests" or a missing-class AttributeError.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

from torch._inductor.exc import InvalidCxxCompiler
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal import inductor_utils
from torch.testing._internal.common_utils import HardwareClassification


class TestCPUCapabilityProbe(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_test_cpu_preserves_probe_exception(self):
        probe_exc = InvalidCxxCompiler(
            "No working C++ compiler found in torch._inductor.config.cpp.cxx: "
            "(None, 'definitely-not-a-compiler')"
        )
        with mock.patch(
            "torch._inductor.codecache.CppCodeCache.load",
            side_effect=probe_exc,
        ):
            previous = inductor_utils.CPU_PROBE_EXCEPTION
            try:
                self.assertFalse(inductor_utils.test_cpu())
                self.assertIs(inductor_utils.CPU_PROBE_EXCEPTION, probe_exc)
            finally:
                inductor_utils.CPU_PROBE_EXCEPTION = previous

    def test_require_cpu_inductor_raises_preserved_exception(self):
        probe_exc = InvalidCxxCompiler("preserved probe failure")
        with (
            mock.patch.object(inductor_utils, "HAS_CPU", False),
            mock.patch.object(inductor_utils, "CPU_PROBE_EXCEPTION", probe_exc),
        ):
            with self.assertRaises(InvalidCxxCompiler) as ctx:
                inductor_utils.require_cpu_inductor()
            self.assertIs(ctx.exception, probe_exc)

    def test_require_cpu_inductor_noop_when_has_cpu(self):
        with mock.patch.object(inductor_utils, "HAS_CPU", True):
            inductor_utils.require_cpu_inductor()

    def test_requires_cpu_inductor_mixin_raises_in_setup(self):
        probe_exc = InvalidCxxCompiler("mixin probe failure")

        class ProbeFailTest(inductor_utils.RequiresCPUInductor, unittest.TestCase):
            def test_placeholder(self):
                pass

        with (
            mock.patch.object(inductor_utils, "HAS_CPU", False),
            mock.patch.object(inductor_utils, "CPU_PROBE_EXCEPTION", probe_exc),
        ):
            result = unittest.TextTestRunner(verbosity=0).run(
                ProbeFailTest("test_placeholder")
            )
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("InvalidCxxCompiler", result.errors[0][1])

    def test_cpu_tests_requested_is_independent_of_has_cpu(self):
        # Selection intent must not collapse into the capability probe result.
        self.assertIsInstance(inductor_utils.CPU_TESTS_REQUESTED, bool)
        with mock.patch.object(inductor_utils, "HAS_CPU", False):
            # Patching HAS_CPU must not change the already-computed selection flag.
            requested = inductor_utils.CPU_TESTS_REQUESTED
        self.assertEqual(requested, inductor_utils.CPU_TESTS_REQUESTED)


def _run_probe_script(
    script: str, *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    full_env = os.environ.copy()
    # Avoid inheriting device-selection overrides from the parent test process.
    full_env.pop("PYTORCH_TESTING_DEVICE_ONLY_FOR", None)
    full_env.pop("PYTORCH_TESTING_DEVICE_EXCEPT_FOR", None)
    full_env.pop("PYTORCH_TEST_INDUCTOR_RUN_CPU", None)
    if env:
        full_env.update(env)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        # Run outside the repo so the local incomplete torch/ package cannot shadow
        # the installed torch used by these diagnostics checks.
        return subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            env=full_env,
            cwd=tempfile.gettempdir(),
            check=False,
        )
    finally:
        os.unlink(path)


_REQUESTED_CPU_SCRIPT = textwrap.dedent(
    """
    import sys
    import unittest
    from torch.testing._internal.inductor_utils import (
        CPU_TESTS_REQUESTED,
        RequiresCPUInductor,
    )

    if not CPU_TESTS_REQUESTED:
        raise SystemExit("CPU_TESTS_REQUESTED unexpectedly false")

    class CpuTests(RequiresCPUInductor, unittest.TestCase):
        def test_randint_cpu(self):
            self.assertTrue(True)

    suite = unittest.defaultTestLoader.loadTestsFromName(
        "CpuTests.test_randint_cpu",
        module=sys.modules[__name__],
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
    """
)


class TestCPUCapabilityProbeSubprocess(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_invalid_cxx_exact_name_raises_invalid_cxx_compiler(self):
        proc = _run_probe_script(
            _REQUESTED_CPU_SCRIPT, env={"CXX": "definitely-not-a-compiler"}
        )
        combined = proc.stdout + proc.stderr
        self.assertNotEqual(proc.returncode, 0, msg=combined)
        self.assertIn("InvalidCxxCompiler", combined)
        self.assertNotIn("has no attribute 'CpuTests'", combined)
        self.assertNotIn("Ran 0 tests", combined)

    def test_invalid_cxx_k_selection_exits_nonzero_with_probe_error(self):
        script = textwrap.dedent(
            """
            import sys
            import unittest
            from torch.testing._internal.inductor_utils import (
                CPU_TESTS_REQUESTED,
                RequiresCPUInductor,
            )

            if not CPU_TESTS_REQUESTED:
                raise SystemExit("CPU_TESTS_REQUESTED unexpectedly false")

            class CpuTests(RequiresCPUInductor, unittest.TestCase):
                def test_randint_cpu(self):
                    self.assertTrue(True)

            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(sys.modules[__name__])

            def flatten(tests):
                for test in tests:
                    if isinstance(test, unittest.TestSuite):
                        yield from flatten(test)
                    else:
                        yield test

            filtered = unittest.TestSuite(
                t for t in flatten(suite) if "randint_cpu" in t.id()
            )
            result = unittest.TextTestRunner(verbosity=2).run(filtered)
            if result.testsRun == 0:
                print("Ran 0 tests")
            raise SystemExit(0 if result.wasSuccessful() else 1)
            """
        )
        proc = _run_probe_script(script, env={"CXX": "definitely-not-a-compiler"})
        combined = proc.stdout + proc.stderr
        self.assertNotEqual(proc.returncode, 0, msg=combined)
        self.assertIn("InvalidCxxCompiler", combined)
        self.assertNotIn("Ran 0 tests", combined)

    def test_gpu_only_selection_unaffected_by_broken_cpu_toolchain(self):
        script = textwrap.dedent(
            """
            import sys
            import unittest
            from torch.testing._internal.inductor_utils import CPU_TESTS_REQUESTED

            if CPU_TESTS_REQUESTED:
                raise SystemExit("CPU_TESTS_REQUESTED unexpectedly true")

            class GpuOnlyTests(unittest.TestCase):
                def test_gpu_smoke(self):
                    self.assertTrue(True)

            if "CpuTests" in globals():
                raise SystemExit("CpuTests should not be defined")

            suite = unittest.defaultTestLoader.loadTestsFromName(
                "GpuOnlyTests.test_gpu_smoke",
                module=sys.modules[__name__],
            )
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            raise SystemExit(0 if result.wasSuccessful() else 1)
            """
        )
        proc = _run_probe_script(
            script,
            env={
                "CXX": "definitely-not-a-compiler",
                "PYTORCH_TESTING_DEVICE_ONLY_FOR": "cuda",
            },
        )
        combined = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, msg=combined)


if __name__ == "__main__":
    run_tests()
