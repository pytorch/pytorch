# Owner(s): ["module: inductor"]
import json
import operator
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import types
import unittest
from concurrent.futures import TimeoutError as FuturesTimeoutError
from threading import Event
from unittest.mock import patch

import torch._inductor.config as config
from torch._inductor.compile_worker.subproc_pool import (
    raise_testexc,
    SubprocException,
    SubprocKind,
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
    def test_sidecar_death_fails_pending_futures(self):
        # If the sidecar (SubprocMain) process dies, its forked compile workers
        # keep the write pipe open, so the parent never sees EOF and would
        # otherwise block forever in _recv_msg. The liveness watchdog must
        # detect the dead sidecar and fail pending futures instead of hanging.
        pool = self.make_pool(2)
        try:
            # Warm the pool so workers are forked and holding the pipe fd.
            self.assertEqual(pool.submit(operator.add, 1, 2).result(), 3)
            fut = pool.submit(time.sleep, 600)
            time.sleep(1.0)
            pool.process.kill()
            try:
                fut.result(timeout=60)
            except FuturesTimeoutError:
                self.fail("pending future did not resolve after sidecar death")
            except Exception:
                pass  # expected: the watchdog fails the future
            else:
                self.fail("expected an exception after sidecar death")
        finally:
            pool.shutdown()

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_sidecar_death_eofs_without_watchdog(self):
        # With the liveness watchdog disabled, sidecar death must still fail
        # pending futures via a clean pipe EOF -- which only happens if neither
        # the parent nor the compile workers keep the result pipe's write end
        # open. Proves the fd-close fix stands on its own.
        with patch.object(SubprocPool, "_health_monitor", lambda self: None):
            pool = self.make_pool(2)
            try:
                self.assertEqual(pool.submit(operator.add, 1, 2).result(), 3)
                fut = pool.submit(time.sleep, 600)
                time.sleep(1.0)
                pool.process.kill()
                try:
                    fut.result(timeout=60)
                except FuturesTimeoutError:
                    self.fail("future did not resolve via EOF after sidecar death")
                except Exception:
                    pass  # expected: EOF path fails the future
                else:
                    self.fail("expected an exception after sidecar death")
            finally:
                pool.shutdown()

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_spawn_pool_basic_jobs(self):
        # compile_fx_subproc.py runs the pool with kind=SPAWN. The worker
        # fd-close must be gated to fork; otherwise a spawned worker closes an
        # unrelated (reused) fd and wedges the pool via a BrokenProcessPool loop.
        pool = SubprocPool(2, kind=SubprocKind.SPAWN)
        try:
            a = pool.submit(operator.add, 100, 1)
            b = pool.submit(operator.sub, 100, 1)
            self.assertEqual(a.result(), 101)
            self.assertEqual(b.result(), 99)
        finally:
            pool.shutdown()

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_watchdog_fails_futures_when_no_eof(self):
        # Isolate the liveness watchdog from the fd/EOF fix. Disabling _read_thread
        # kills the result-pipe path entirely, standing in for the "sidecar dies
        # but EOF never arrives" case: with that path dead, only _health_monitor
        # -> _on_sidecar_death can resolve pending futures. (The complementary
        # test_sidecar_death_eofs_without_watchdog covers the EOF path in
        # isolation by disabling the watchdog instead.)
        with patch.object(SubprocPool, "_read_thread", lambda self: None):
            pool = self.make_pool(2)
            try:
                fut = pool.submit(time.sleep, 600)
                time.sleep(1.0)  # let the sidecar fork workers and start the job
                pool.process.kill()
                try:
                    fut.result(timeout=60)
                except FuturesTimeoutError:
                    self.fail("watchdog did not fail the future without an EOF")
                except Exception:
                    pass  # expected: the watchdog fails the future
                else:
                    self.fail("expected an exception after sidecar death")
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
    def test_shutdown_kills_wedged_worker(self):
        # A compile worker that ignores SIGTERM must not stall pool teardown:
        # the sidecar has to escalate to SIGKILL rather than block indefinitely
        # in pool.shutdown(wait=True) (which the parent only bounds at 300s).
        code = textwrap.dedent(
            """
            import operator
            import time

            from torch._inductor.compile_worker.subproc_pool import (
                _ignore_sigterm_and_sleep_for_test,
                SubprocPool,
            )

            pool = SubprocPool(2)
            assert pool.submit(operator.add, 1, 2).result() == 3
            pool.submit(_ignore_sigterm_and_sleep_for_test)
            time.sleep(1.0)

            start = time.time()
            pool.shutdown()
            elapsed = time.time() - start
            assert elapsed < 120, f"shutdown stalled for {elapsed}s"
            print(f"shutdown returned in {elapsed:.1f}s")
            """
        )
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=180,
            )
        self.assertEqual(
            result.returncode,
            0,
            lambda msg: f"{msg}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn("shutdown returned", result.stdout)

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


class TestCompileWorkerWatchdog(TestCase):
    # The sidecar runs a watchdog that, every interval (shortened to 1s here via
    # env), reports jobs still running past that interval to the parent, which
    # turns them into a "compile_worker_status" structured-trace artifact -- so a
    # stuck/slow worker leaves a breadcrumb in tlparse instead of silently
    # wedging. See subproc_pool.SubprocMain._watchdog_loop.
    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_watchdog_reports_slow_jobs(self):
        reports = []
        got_report = Event()

        def fake_trace_structured(name, *args, **kwargs):
            if name != "artifact":
                return
            metadata = kwargs.get("metadata_fn", dict)()
            if metadata.get("name") != "compile_worker_status":
                return
            reports.append(json.loads(kwargs["payload_fn"]()))
            got_report.set()

        with patch.dict(
            os.environ, {"TORCHINDUCTOR_COMPILE_WORKER_WATCHDOG_INTERVAL": "1"}
        ):
            with patch(
                "torch._inductor.compile_worker.subproc_pool.trace_structured",
                fake_trace_structured,
            ):
                pool = SubprocPool(2)
                try:
                    slow = pool.submit(time.sleep, 8)
                    self.assertTrue(
                        got_report.wait(30), "watchdog did not report the slow job"
                    )
                    slow.result()
                finally:
                    pool.shutdown()

        self.assertTrue(reports)
        self.assertGreaterEqual(reports[-1]["elapsed_s"], 1.0)

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_watchdog_silent_for_fast_jobs(self):
        reports = []

        def fake_trace_structured(name, *args, **kwargs):
            if name != "artifact":
                return
            metadata = kwargs.get("metadata_fn", dict)()
            if metadata.get("name") == "compile_worker_status":
                reports.append(metadata)

        with patch.dict(
            os.environ, {"TORCHINDUCTOR_COMPILE_WORKER_WATCHDOG_INTERVAL": "1"}
        ):
            pool = SubprocPool(2)
            try:
                # Warm the pool before collecting reports. The first job pays cold
                # pool creation and the worker forks, which can exceed the
                # (test-shortened) interval and be legitimately reported; that
                # cost must not be attributed to the "fast" job below. The
                # callback drops a job from _inflight before its result is sent,
                # so once this result returns the warm-up job can no longer be
                # reported.
                self.assertEqual(pool.submit(operator.add, 1, 2).result(), 3)
                with patch(
                    "torch._inductor.compile_worker.subproc_pool.trace_structured",
                    fake_trace_structured,
                ):
                    self.assertEqual(pool.submit(operator.add, 1, 2).result(), 3)
                    time.sleep(2.5)  # a couple of watchdog ticks with no slow job
            finally:
                pool.shutdown()

        self.assertEqual(reports, [])

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_watchdog_reports_worker_phase(self):
        # A worker that reports a heartbeat phase should have that phase surface
        # in the STATUS report (Phase 2: shared-memory phase heartbeat).
        from torch._inductor.compile_worker.subproc_pool import (
            _report_phase_and_sleep_for_test,
        )
        from torch._inductor.compile_worker.watchdog import Phase

        reports = []
        got_phase = Event()

        def fake_trace_structured(name, *args, **kwargs):
            if name != "artifact":
                return
            metadata = kwargs.get("metadata_fn", dict)()
            if metadata.get("name") != "compile_worker_status":
                return
            record = json.loads(kwargs["payload_fn"]())
            reports.append(record)
            if record.get("phase") == "querying_cache":
                got_phase.set()

        with patch.dict(
            os.environ, {"TORCHINDUCTOR_COMPILE_WORKER_WATCHDOG_INTERVAL": "1"}
        ):
            with patch(
                "torch._inductor.compile_worker.subproc_pool.trace_structured",
                fake_trace_structured,
            ):
                pool = SubprocPool(2)
                try:
                    fut = pool.submit(
                        _report_phase_and_sleep_for_test, int(Phase.QUERYING_CACHE), 8
                    )
                    self.assertTrue(
                        got_phase.wait(30),
                        f"watchdog did not report the phase; got {reports}",
                    )
                    fut.result()
                finally:
                    pool.shutdown()

        phased = [r for r in reports if r.get("phase") == "querying_cache"]
        self.assertTrue(phased)
        self.assertIn("phase_elapsed_s", phased[-1])
        self.assertIn("worker_pid", phased[-1])

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_watchdog_reports_queued_job(self):
        # A job submitted while every worker is busy sits in the pool queue with
        # no heartbeat slot, and must be reported with phase="queued".
        reports = []
        got_queued = Event()

        def fake_trace_structured(name, *args, **kwargs):
            if name != "artifact":
                return
            metadata = kwargs.get("metadata_fn", dict)()
            if metadata.get("name") != "compile_worker_status":
                return
            record = json.loads(kwargs["payload_fn"]())
            reports.append(record)
            if record.get("phase") == "queued":
                got_queued.set()

        with patch.dict(
            os.environ, {"TORCHINDUCTOR_COMPILE_WORKER_WATCHDOG_INTERVAL": "1"}
        ):
            with patch(
                "torch._inductor.compile_worker.subproc_pool.trace_structured",
                fake_trace_structured,
            ):
                pool = SubprocPool(1)  # single worker so the 2nd job must queue
                try:
                    pool.submit(time.sleep, 30)  # occupies the sole worker
                    pool.submit(time.sleep, 30)  # no free worker -> queued
                    self.assertTrue(
                        got_queued.wait(30),
                        f"no queued-phase report; got {reports}",
                    )
                finally:
                    pool.shutdown()

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_watchdog_duration_only_for_spawn_pool(self):
        # Spawn pools don't get the heartbeat buffer, so the watchdog reports
        # duration only -- no phase (it can't tell queued from running).
        reports = []
        got_report = Event()

        def fake_trace_structured(name, *args, **kwargs):
            if name != "artifact":
                return
            metadata = kwargs.get("metadata_fn", dict)()
            if metadata.get("name") != "compile_worker_status":
                return
            reports.append(json.loads(kwargs["payload_fn"]()))
            got_report.set()

        with patch.dict(
            os.environ, {"TORCHINDUCTOR_COMPILE_WORKER_WATCHDOG_INTERVAL": "1"}
        ):
            with patch(
                "torch._inductor.compile_worker.subproc_pool.trace_structured",
                fake_trace_structured,
            ):
                pool = SubprocPool(2, kind=SubprocKind.SPAWN)
                try:
                    pool.submit(time.sleep, 30)
                    self.assertTrue(
                        got_report.wait(30), f"no watchdog report; got {reports}"
                    )
                finally:
                    pool.shutdown()

        self.assertTrue(reports)
        self.assertNotIn("phase", reports[-1])
        self.assertIn("elapsed_s", reports[-1])


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
