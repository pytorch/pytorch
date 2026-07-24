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
from torch.testing._internal.common_utils import IS_LINUX, skipIfWindows
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


class _FakeTritonKernel:
    def __init__(self):
        self.precompiled = False
        self.prepared_for_pickle = False

    def precompile(self, *, warm_cache_only):
        self.precompiled = warm_cache_only

    def prepare_for_pickle(self):
        self.prepared_for_pickle = True


class TestSubprocessEnv(TestCase):
    def assert_path_in_dir(self, path, expected_dir):
        expected_dir = os.path.abspath(expected_dir)
        self.assertEqual(
            os.path.commonpath([os.path.abspath(path), expected_dir]),
            expected_dir,
        )

    def test_pycodecache_kernel_compile_sends_full_cache_env(self):
        from torch._inductor.async_compile import AsyncCompile
        from torch._inductor.runtime.compile_tasks import (
            _worker_compile_pycodecache_kernel,
        )

        class FakeFuture:
            def result(self):
                return "key", "/tmp/kernel.py", 0

        class FakePool:
            def __init__(self):
                self.calls = []

            def submit(self, fn, *args):
                self.calls.append((fn, args))
                return FakeFuture()

        env_keys = [
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
            "TORCHINDUCTOR_CUTLASS_DIR",
        ]
        old_env = {key: os.environ.get(key) for key in env_keys}
        pool = FakePool()

        try:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/current-inductor-cache"
            os.environ.pop("TRITON_CACHE_DIR", None)
            os.environ.pop("TORCHINDUCTOR_CUTLASS_DIR", None)

            with (
                patch.object(AsyncCompile, "use_process_pool", return_value=True),
                patch.object(AsyncCompile, "process_pool", return_value=pool),
            ):
                AsyncCompile().cutedsl("kernel", "def kernel_main():\n    pass\n")
                AsyncCompile().nv_universal_gemm(
                    "kernel", "def kernel_main():\n    pass\n"
                )

            self.assertEqual(len(pool.calls), 2)
            for fn, args in pool.calls:
                self.assertIs(fn, _worker_compile_pycodecache_kernel)
                self.assertEqual(
                    args[3],
                    {
                        "TORCHINDUCTOR_CACHE_DIR": "/tmp/current-inductor-cache",
                        "TRITON_CACHE_DIR": None,
                        "TORCHINDUCTOR_CUTLASS_DIR": None,
                    },
                )
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_worker_compile_pycodecache_kernel_clears_cache_env(self):
        import torch._inductor.runtime.compile_tasks as compile_tasks
        from torch._inductor.runtime.cache_dir_utils import default_cache_dir

        old_env = {
            "TORCHINDUCTOR_CACHE_DIR": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR"),
            "TORCHINDUCTOR_CUTLASS_DIR": os.environ.get("TORCHINDUCTOR_CUTLASS_DIR"),
        }
        old_last_applied_cache_env = compile_tasks._last_applied_cache_env
        source_code = "def kernel_main():\n    pass\n"

        try:
            compile_tasks._last_applied_cache_env = None
            with tempfile.TemporaryDirectory() as cache_dir:
                triton_cache_dir = os.path.join(cache_dir, "triton")
                _, path_1, _ = compile_tasks._worker_compile_pycodecache_kernel(
                    "kernel",
                    source_code,
                    "main",
                    {
                        "TORCHINDUCTOR_CACHE_DIR": cache_dir,
                        "TRITON_CACHE_DIR": triton_cache_dir,
                        "TORCHINDUCTOR_CUTLASS_DIR": None,
                    },
                )
                self.assert_path_in_dir(path_1, cache_dir)
                self.assertEqual(os.environ["TORCHINDUCTOR_CACHE_DIR"], cache_dir)
                self.assertEqual(os.environ["TRITON_CACHE_DIR"], triton_cache_dir)
                self.assertNotIn("TORCHINDUCTOR_CUTLASS_DIR", os.environ)

                _, path_2, _ = compile_tasks._worker_compile_pycodecache_kernel(
                    "kernel",
                    source_code,
                    "main",
                    {
                        "TORCHINDUCTOR_CACHE_DIR": None,
                        "TRITON_CACHE_DIR": None,
                        "TORCHINDUCTOR_CUTLASS_DIR": None,
                    },
                )
                self.assert_path_in_dir(path_2, default_cache_dir())
                self.assertNotEqual(path_1, path_2)
                self.assertNotIn("TRITON_CACHE_DIR", os.environ)
                self.assertNotIn("TORCHINDUCTOR_CUTLASS_DIR", os.environ)
        finally:
            compile_tasks._last_applied_cache_env = old_last_applied_cache_env
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_nvgemm_precompile_sends_full_cache_env(self):
        from torch._inductor.async_compile import AsyncCompile
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_kernel import (
            _worker_nvgemm_autotuning_precompile,
        )

        class FakeFuture:
            def result(self):
                return None, 0

        class FakePool:
            def __init__(self):
                self.calls = []

            def submit(self, fn, *args):
                self.calls.append((fn, args))
                return FakeFuture()

        env_keys = [
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
            "TORCHINDUCTOR_CUTLASS_DIR",
        ]
        old_env = {key: os.environ.get(key) for key in env_keys}
        pool = FakePool()

        try:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/current-inductor-cache"
            os.environ.pop("TRITON_CACHE_DIR", None)
            os.environ.pop("TORCHINDUCTOR_CUTLASS_DIR", None)

            with patch.object(AsyncCompile, "process_pool", return_value=pool):
                AsyncCompile().nvgemm_precompile(
                    "kernel",
                    "GEMM",
                    "accumulator",
                    (),
                    None,
                    types.SimpleNamespace(
                        max_active_clusters=None, device_capability=(9, 0)
                    ),
                )

            self.assertEqual(len(pool.calls), 1)
            fn, args = pool.calls[0]
            self.assertIs(fn, _worker_nvgemm_autotuning_precompile)
            self.assertEqual(
                args[5],
                {
                    "TORCHINDUCTOR_CACHE_DIR": "/tmp/current-inductor-cache",
                    "TRITON_CACHE_DIR": None,
                    "TORCHINDUCTOR_CUTLASS_DIR": None,
                },
            )
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_worker_nvgemm_precompile_clears_cache_env(self):
        import torch
        import torch._inductor.runtime.compile_tasks as compile_tasks
        from torch._inductor.codegen.nv_universal_gemm import (
            nv_universal_gemm_kernel as nvgemm_kernel,
        )

        old_env = {
            "TORCHINDUCTOR_CACHE_DIR": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR"),
            "TORCHINDUCTOR_CUTLASS_DIR": os.environ.get("TORCHINDUCTOR_CUTLASS_DIR"),
        }
        old_last_applied_cache_env = compile_tasks._last_applied_cache_env
        meta = types.SimpleNamespace(
            sizes=(1, 1), strides=(1, 1), device="cpu", dtype=torch.float32
        )
        cuda_ctx = types.SimpleNamespace(
            max_active_clusters=None, device_capability=(9, 0)
        )

        try:
            compile_tasks._last_applied_cache_env = None
            with (
                tempfile.TemporaryDirectory() as cache_dir,
                patch.object(
                    nvgemm_kernel,
                    "_compile_nvgemm",
                    return_value=(object(), None, None, False),
                ),
                patch("torch._inductor.utils._ensure_fp4_dtype_registered"),
                patch.object(
                    nvgemm_kernel, "_patch_max_active_clusters", return_value=[]
                ),
            ):
                triton_cache_dir = os.path.join(cache_dir, "triton")
                nvgemm_kernel._worker_nvgemm_autotuning_precompile(
                    "kernel",
                    "GEMM",
                    "accumulator",
                    (meta, meta),
                    meta,
                    {
                        "TORCHINDUCTOR_CACHE_DIR": cache_dir,
                        "TRITON_CACHE_DIR": triton_cache_dir,
                        "TORCHINDUCTOR_CUTLASS_DIR": None,
                    },
                    cuda_ctx,
                )
                self.assertEqual(os.environ["TORCHINDUCTOR_CACHE_DIR"], cache_dir)
                self.assertEqual(os.environ["TRITON_CACHE_DIR"], triton_cache_dir)
                self.assertNotIn("TORCHINDUCTOR_CUTLASS_DIR", os.environ)

                nvgemm_kernel._worker_nvgemm_autotuning_precompile(
                    "kernel",
                    "GEMM",
                    "accumulator",
                    (meta, meta),
                    meta,
                    {
                        "TORCHINDUCTOR_CACHE_DIR": None,
                        "TRITON_CACHE_DIR": None,
                        "TORCHINDUCTOR_CUTLASS_DIR": None,
                    },
                    cuda_ctx,
                )
                self.assertNotIn("TORCHINDUCTOR_CACHE_DIR", os.environ)
                self.assertNotIn("TRITON_CACHE_DIR", os.environ)
                self.assertNotIn("TORCHINDUCTOR_CUTLASS_DIR", os.environ)
        finally:
            compile_tasks._last_applied_cache_env = old_last_applied_cache_env
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_worker_compile_triton_clears_libdevice_path(self):
        try:
            from triton import knobs
        except ImportError:
            self.skipTest("triton not available")

        from torch._inductor.runtime.compile_tasks import _worker_compile_triton

        old_env = os.environ.get("TRITON_LIBDEVICE_PATH")
        old_knob = knobs.nvidia.libdevice_path
        stale_libdevice_path = "/tmp/stale-libdevice.bc"

        try:
            kernel, _ = _worker_compile_triton(
                _FakeTritonKernel,
                {"TRITON_LIBDEVICE_PATH": stale_libdevice_path},
                {},
            )
            self.assertTrue(kernel.precompiled)
            self.assertTrue(kernel.prepared_for_pickle)
            self.assertEqual(os.environ["TRITON_LIBDEVICE_PATH"], stale_libdevice_path)
            self.assertEqual(knobs.nvidia.libdevice_path, stale_libdevice_path)

            _worker_compile_triton(
                _FakeTritonKernel,
                {"TRITON_LIBDEVICE_PATH": None},
                {},
            )
            self.assertNotIn("TRITON_LIBDEVICE_PATH", os.environ)
            self.assertIsNone(knobs.nvidia.libdevice_path)
        finally:
            if old_env is None:
                os.environ.pop("TRITON_LIBDEVICE_PATH", None)
            else:
                os.environ["TRITON_LIBDEVICE_PATH"] = old_env
            knobs.nvidia.libdevice_path = old_knob

    def test_serialized_fx_compile_restores_subprocess_env(self):
        from torch._inductor.compile_fx_ext import _SerializedFxCompile

        key = "TEST_INDUCTOR_SUBPROCESS_ENV"
        mutated_key = "TEST_INDUCTOR_SUBPROCESS_ENV_MUTATED"
        old_env = os.environ.get(key)
        old_mutated_env = os.environ.get(mutated_key)
        os.environ[key] = "parent-value"
        os.environ.pop(mutated_key, None)

        class StopAfterEnvCheck(Exception):
            pass

        testcase = self

        class Input:
            def deserialize(self):
                testcase.assertNotIn(key, os.environ)
                os.environ[mutated_key] = "child-value"
                raise StopAfterEnvCheck

        try:
            with self.assertRaises(StopAfterEnvCheck):
                _SerializedFxCompile._run_in_child(Input(), {key: None})
            self.assertEqual(os.environ[key], "parent-value")
            self.assertNotIn(mutated_key, os.environ)
        finally:
            if old_env is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_env
            if old_mutated_env is None:
                os.environ.pop(mutated_key, None)
            else:
                os.environ[mutated_key] = old_mutated_env


class TestSetTritonLibdevicePath(TestCase):
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

        # Verify libdevice path was set. The exact path is environment-specific
        # (OSS uses the CUDA_HOME toolkit copy; fbcode uses Triton's bundled
        # copy), so verify a libdevice bitcode path was set rather than pinning
        # it to the CUDA_HOME location.
        from triton import knobs

        actual = knobs.nvidia.libdevice_path
        self.assertTrue(actual, "libdevice path was not set")
        self.assertTrue(
            actual.endswith("libdevice.10.bc") and os.path.isfile(actual),
            f"expected a valid libdevice path, got {actual!r}",
        )


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
