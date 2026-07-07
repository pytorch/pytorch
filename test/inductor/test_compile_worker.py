# Owner(s): ["module: inductor"]
import operator
import os
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from threading import Event
from unittest.mock import patch

import torch._inductor.config as config
from torch._inductor.compile_worker.subproc_pool import (
    raise_testexc,
    SubprocException,
    SubprocPool,
)
from torch._inductor.compile_worker.timer import Timer
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import IS_FBCODE, IS_LINUX, skipIfWindows
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_TRITON


class TestCompileWorker(TestCase):
    def make_pool(self, size):
        return SubprocPool(size)

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_basic_jobs(self):
        pool = self.make_pool(2)
        try:
            a = pool.submit(operator.add, 100, 1)
            b = pool.submit(operator.sub, 100, 1)
            self.assertEqual(a.result(), 101)
            self.assertEqual(b.result(), 99)
        finally:
            pool.shutdown()

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_exception(self):
        pool = self.make_pool(2)
        try:
            a = pool.submit(raise_testexc)
            with self.assertRaisesRegex(
                SubprocException,
                "torch._inductor.compile_worker.subproc_pool.TestException",
            ):
                a.result()
        finally:
            pool.shutdown()

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_crash(self):
        pool = self.make_pool(2)
        try:
            with self.assertRaises(Exception):
                a = pool.submit(os._exit, 1)
                a.result()

            # Pool should still be usable after a crash
            b = pool.submit(operator.add, 100, 1)
            c = pool.submit(operator.sub, 100, 1)
            self.assertEqual(b.result(), 101)
            self.assertEqual(c.result(), 99)
        finally:
            pool.shutdown()

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_quiesce(self):
        pool = self.make_pool(2)
        try:
            a = pool.submit(operator.add, 100, 1)
            pool.quiesce()
            pool.wakeup()
            b = pool.submit(operator.sub, 100, 1)
            self.assertEqual(a.result(), 101)
            self.assertEqual(b.result(), 99)
        finally:
            pool.shutdown()

    @unittest.skipIf(IS_LINUX, "https://github.com/pytorch/pytorch/issues/176968")
    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_quiesce_repeatedly(self):
        pool = SubprocPool(2)
        try:
            a = pool.submit(operator.add, 100, 1)
            pool.quiesce()
            pool.wakeup()
            b = pool.submit(operator.sub, 100, 1)
            pool.quiesce()
            pool.quiesce()
            pool.wakeup()
            b = pool.submit(operator.sub, 100, 1)
            self.assertEqual(a.result(), 101)
            self.assertEqual(b.result(), 99)
        finally:
            pool.shutdown()

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_logging(self):
        os.environ["MAST_HPC_JOB_NAME"] = "test_job"
        os.environ["ROLE_RANK"] = "0"
        with tempfile.NamedTemporaryFile(delete=True) as temp_log:
            os.environ["TORCHINDUCTOR_WORKER_LOGPATH"] = temp_log.name
            pool = self.make_pool(2)
            try:
                pool.submit(operator.add, 100, 1)
                self.assertEqual(os.path.exists(temp_log.name), True)
            finally:
                pool.shutdown()

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_shutdown_terminates_sidecar_worker_pool(self):
        code = textwrap.dedent(
            """
            import operator
            import os
            import subprocess
            import time

            from torch._inductor.compile_worker.subproc_pool import (
                _test_signal_then_sleep,
                SubprocPool,
            )

            # Warm the pool so the sidecar and its worker pool are fully up.
            pool = SubprocPool(2)
            assert pool.submit(operator.add, 1, 2).result() == 3

            # Submit a job that signals (via a sentinel file) once a worker is
            # actually executing it, then blocks far longer than the shutdown
            # timeout below.
            signal_path = os.environ["WORKER_SIGNAL_PATH"]
            pool.submit(_test_signal_then_sleep, signal_path, 120)

            # Readiness barrier: wait until a worker is provably running the job
            # before timing the shutdown, so process/worker startup cost stays
            # out of the shutdown budget.
            deadline = time.time() + 60
            while not os.path.exists(signal_path):
                if time.time() >= deadline:
                    raise RuntimeError("worker never started the long-running job")
                time.sleep(0.05)

            # Bound the shutdown well below the running job's sleep so a
            # regression that waits for the in-flight job (instead of
            # terminating it) is detected quickly.
            wait = pool.process.wait

            def short_wait(timeout=None):
                return wait(timeout=30)

            pool.process.wait = short_wait

            try:
                pool.shutdown()
            except subprocess.TimeoutExpired:
                pool.process.kill()
                pool.process.wait()
                raise

            print("shutdown returned")
            """
        )
        with tempfile.TemporaryDirectory() as cwd:
            signal_path = os.path.join(cwd, "worker_started")
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=cwd,
                capture_output=True,
                text=True,
                env={**os.environ, "WORKER_SIGNAL_PATH": signal_path},
                # Generous backstop: the child pays multiple heavyweight process
                # cold-starts (esp. in fbcode). A real regression still fails
                # fast via the bounded shutdown wait above.
                timeout=120,
            )
        self.assertEqual(
            result.returncode,
            0,
            lambda msg: f"{msg}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn("shutdown returned", result.stdout)


@config.patch("quiesce_async_compile_time", 0.1)
class TestCompileWorkerWithTimer(TestCompileWorker):
    def make_pool(self, size):
        return SubprocPool(size, quiesce=True)


class TestTimer(TestCase):
    def test_basics(self):
        done = Event()

        def doit():
            done.set()

        t = Timer(0.1, doit)
        t.sleep_time = 0.1
        t.record_call()
        self.assertTrue(done.wait(4))
        t.quit()

    def test_repeated_calls(self):
        done = Event()

        def doit():
            done.set()

        t = Timer(0.1, doit)
        t.sleep_time = 0.1
        for _ in range(10):
            t.record_call()
            self.assertTrue(done.wait(4))
            done.clear()
        t.quit()

    def test_never_fires(self):
        done = Event()

        def doit():
            done.set()

        t = Timer(999, doit)
        t.sleep_time = 0.1
        t.record_call()
        self.assertFalse(done.wait(4))
        t.quit()

    def test_spammy_calls(self):
        done = Event()

        def doit():
            done.set()

        t = Timer(1, doit)
        t.sleep_time = 0.1
        for _ in range(400):
            t.record_call()
        self.assertTrue(done.wait(4))
        t.quit()


class TestSetTritonLibdevicePath(TestCase):
    @unittest.skipIf(
        IS_FBCODE,
        "knobs.nvidia.libdevice_path mismatch in fbcode CI environment; "
        "matches sibling test_libdevice_path_* disables",
    )
    @config.patch({"compile_threads": 1, "emulate_precision_casts": True})
    def test_emulate_precision_casts_sets_libdevice_path(self):
        """Test eager numerics mode sets libdevice path for CUDA libdevice calls."""
        self._test_libdevice_path_with_compilation()

    @config.patch({"compile_threads": 1, "eager_numerics.use_pytorch_libdevice": True})
    def test_libdevice_path_no_subprocess(self):
        """Test libdevice path is set with compile_threads=1 (no subprocess)."""
        self._test_libdevice_path_with_compilation()

    @config.patch("eager_numerics.use_pytorch_libdevice", True)
    def test_libdevice_path_default_threads(self):
        """Test libdevice path is set with default compile_threads (subprocess)."""
        self._test_libdevice_path_with_compilation()

    @config.patch(
        {
            "eager_numerics.use_pytorch_libdevice": True,
            "eager_numerics.division_rounding": True,
            "emulate_precision_casts": True,
            "compile_threads": 1,
        }
    )
    def test_pow_bitwise_precision(self):
        """Test that compiled pow matches eager bitwise with system libdevice."""
        import torch
        from torch.utils.cpp_extension import CUDA_HOME

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        if CUDA_HOME is None:
            self.skipTest("CUDA_HOME not set")
        expected = os.path.join(CUDA_HOME, "nvvm", "libdevice", "libdevice.10.bc")
        if not os.path.isfile(expected):
            self.skipTest(f"libdevice not found at {expected}")

        torch._dynamo.reset()
        torch.manual_seed(42)
        base = torch.randn(1000, device="cuda", dtype=torch.float32).abs() + 1e-6
        exp = torch.randn(1000, device="cuda", dtype=torch.float32)

        eager_result = torch.pow(base, exp)
        compiled_result = torch.compile(torch.pow)(base, exp)
        self.assertEqual(eager_result, compiled_result, atol=0, rtol=0)

    @config.patch({"compile_threads": 1, "emulate_precision_casts": True})
    def test_erf_bitwise_precision_with_emulate_precision_casts(self):
        """Test that erf matches eager bitwise when eager numerics mode is active."""
        import torch
        from torch.utils.cpp_extension import CUDA_HOME

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        if CUDA_HOME is None:
            self.skipTest("CUDA_HOME not set")
        expected = os.path.join(CUDA_HOME, "nvvm", "libdevice", "libdevice.10.bc")
        if not os.path.isfile(expected):
            self.skipTest(f"libdevice not found at {expected}")

        torch._dynamo.reset()
        values = torch.tensor(
            [
                -3.9194295406341553,
                -3.9188895225524902,
                0.0,
                1.0,
                3.9194295406341553,
            ],
            device="cuda",
            dtype=torch.float32,
        )

        def fn(x):
            return torch.erf(x)

        eager_result = fn(values)
        compiled_result = torch.compile(fn)(values)
        self.assertEqual(eager_result, compiled_result, atol=0, rtol=0)

    def _test_libdevice_path_with_compilation(self):
        import torch
        from torch.utils.cpp_extension import CUDA_HOME

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        if CUDA_HOME is None:
            self.skipTest("CUDA_HOME not set")

        expected = os.path.join(CUDA_HOME, "nvvm", "libdevice", "libdevice.10.bc")
        if not os.path.isfile(expected):
            self.skipTest(f"libdevice not found at {expected}")

        # Compile a simple function that uses pow (which uses libdevice)
        @torch.compile
        def fn(x):
            return torch.pow(x, 2.0)

        x = torch.randn(10, device="cuda", dtype=torch.float32)
        fn(x)

        # Verify libdevice path was set
        from triton import knobs

        self.assertEqual(knobs.nvidia.libdevice_path, expected)


class TestTritonCompileWorker(TestCase):
    @unittest.skipIf(not HAS_TRITON, "requires triton")
    def test_worker_compile_triton_warm_cache_skips_gpu_driver_setup(self):
        from torch._inductor.runtime import triton_helpers
        from torch._inductor.runtime.compile_tasks import _worker_compile_triton

        class RaisingDriver:
            @staticmethod
            def is_active():
                raise RuntimeError("0 active drivers ([]). There should only be one.")

        class FakeKernel:
            def __init__(self):
                self.precompile_calls = []
                self.prepared = False

            def precompile(self, warm_cache_only=False):
                self.precompile_calls.append(warm_cache_only)
                triton_helpers.set_driver_to_gpu()

            def prepare_for_pickle(self):
                self.prepared = True

        kernel = FakeKernel()

        def load_kernel():
            triton_helpers.set_driver_to_gpu()
            return kernel

        fake_backends = {"nvidia": types.SimpleNamespace(driver=RaisingDriver)}
        with patch.object(triton_helpers.triton.backends, "backends", fake_backends):
            result, _elapsed_us = _worker_compile_triton(load_kernel, {}, {})
            self.assertIs(result, kernel)
            self.assertEqual(kernel.precompile_calls, [True])
            self.assertTrue(kernel.prepared)

            # The skip is scoped to the worker compile call and restored afterward.
            with self.assertRaisesRegex(RuntimeError, "0 active drivers"):
                triton_helpers.set_driver_to_gpu()

    @unittest.skipIf(not HAS_TRITON, "requires triton")
    def test_worker_compile_triton_restores_gpu_driver_setup_after_error(self):
        from torch._inductor.runtime import triton_helpers
        from torch._inductor.runtime.compile_tasks import _worker_compile_triton

        class RaisingDriver:
            @staticmethod
            def is_active():
                raise RuntimeError("0 active drivers ([]). There should only be one.")

        def load_kernel():
            triton_helpers.set_driver_to_gpu()
            raise ValueError("compile failed")

        fake_backends = {"nvidia": types.SimpleNamespace(driver=RaisingDriver)}
        with patch.object(triton_helpers.triton.backends, "backends", fake_backends):
            with self.assertRaisesRegex(ValueError, "compile failed"):
                _worker_compile_triton(load_kernel, {}, {})

            # The skip is restored even if worker compilation raises.
            with self.assertRaisesRegex(RuntimeError, "0 active drivers"):
                triton_helpers.set_driver_to_gpu()


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU:
        run_tests()
