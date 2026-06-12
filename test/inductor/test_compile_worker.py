# Owner(s): ["module: inductor"]
import multiprocessing
import operator
import os
import queue
import subprocess
import sys
import tempfile
import textwrap
import traceback
import unittest
import warnings
from threading import Event

import torch
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
            import subprocess
            import time

            from torch._inductor.compile_worker.subproc_pool import SubprocPool

            pool = SubprocPool(2)
            assert pool.submit(operator.add, 1, 2).result() == 3
            pool.submit(time.sleep, 5)
            time.sleep(0.5)

            wait = pool.process.wait

            def short_wait(timeout=None):
                return wait(timeout=2)

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
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=20,
            )
        self.assertEqual(
            result.returncode,
            0,
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
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


def _pin_driver_bad_fork_worker(q):
    # Bad fork: force is_active() False (triton#9578) and check the pin makes
    # driver.active resolve instead of raising "0 active drivers" (pytorch#184643).
    try:
        import triton

        from torch._inductor.compile_worker.utils import _async_compile_initializer

        nvidia = triton.backends.backends["nvidia"]
        nvidia.driver.is_active = staticmethod(lambda: False)
        triton.runtime.driver._active = None

        _async_compile_initializer(os.getppid())

        active = triton.runtime.driver.active
        is_nvidia = isinstance(active, nvidia.driver) or (
            hasattr(active, "_obj") and isinstance(active._obj, nvidia.driver)
        )
        q.put(("ok", (type(active).__name__, is_nvidia)))
    except BaseException:
        q.put(("err", traceback.format_exc()))


class TestPinTritonWorkerDriver(TestCase):
    @unittest.skipIf(
        not HAS_TRITON or not torch.cuda.is_available(), "requires triton + cuda"
    )
    def test_compile_worker_pins_driver_in_bad_fork(self):
        # #184643: pin must resolve driver.active in a CUDA-forked worker.
        if torch.version.hip is not None:
            self.skipTest("bug and fix are nvidia-only")
        torch.cuda.get_device_properties(0)  # CUDA init before the fork
        ctx = multiprocessing.get_context("fork")
        q = ctx.Queue()
        p = ctx.Process(target=_pin_driver_bad_fork_worker, args=(q,))
        p.daemon = True
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r"os\.fork\(\) was called.*", category=RuntimeWarning
            )
            warnings.filterwarnings(
                "ignore",
                message=(
                    r"This process .* is multi-threaded, use of fork\(\) "
                    r"may lead to deadlocks in the child\."
                ),
                category=DeprecationWarning,
            )
            p.start()
        p.join(60)
        if p.is_alive():
            p.terminate()
            p.join()
            self.fail("bad-fork driver worker timed out")
        try:
            kind, payload = q.get(timeout=5)
        except queue.Empty:
            self.fail(f"bad-fork driver worker exited without a result: {p.exitcode}")
        self.assertEqual(kind, "ok", payload)
        _name, is_nvidia = payload
        self.assertTrue(
            is_nvidia, f"driver.active was not the nvidia driver: {payload}"
        )

    @unittest.skipIf(not HAS_TRITON, "requires triton")
    def test_pinned_triton_driver_api_exists(self):
        # Fail CI if triton renames an internal the pin depends on (drift tripwire).
        import triton

        driver = triton.runtime.driver
        self.assertTrue(hasattr(driver, "_active"))
        self.assertTrue(callable(driver.set_active))
        self.assertIsInstance(triton.backends.backends, dict)
        if torch.cuda.is_available() and torch.version.hip is None:
            self.assertIn("nvidia", triton.backends.backends)
            self.assertTrue(callable(triton.backends.backends["nvidia"].driver))


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU:
        run_tests()
