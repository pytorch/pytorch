# Owner(s): ["module: inductor"]

import os
import unittest
from unittest import mock

import torch
from torch._inductor import tunableop
from torch._inductor.autotune_process import _TestBenchmarkRequest
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    skipIfWindows,
    TestCase,
)


class _FakeProcess:
    """Stands in for TuningProcess: records how it was constructed."""

    def __init__(self, device=None):
        self.device = device
        self.sent = []
        self.result = 1.0
        self.raises = None
        self.shutdown_calls = 0
        self.wait_calls = 0
        self.kill_calls = 0
        self.close_calls = 0
        # Mirrors TuningProcess.alive(): False short-circuits the bounded wait,
        # which is what a fake with no real child should report.
        self._alive = False
        self.process = None

    def put(self, job):
        self.sent.append(job)

    def get(self, timeout=None):
        if self.raises is not None:
            raise self.raises
        return self.result

    def shutdown(self, wait=True):
        self.shutdown_calls += 1

    # The teardown surface _shutdown_locked drives. Without these the fake
    # raised AttributeError, which the except Exception there swallowed, so the
    # restart tests passed whether or not the old worker was ever shut down.
    def alive(self):
        return self._alive

    def wait(self):
        self.wait_calls += 1

    def kill(self):
        self.kill_calls += 1

    def close(self):
        self.close_calls += 1


class TunableOpWorkerTest(TestCase):
    """The worker is a subprocess holding process-global TunableOp state, so the
    things worth pinning are which GPU it lands on and when it gets replaced.
    Neither needs a GPU to check, and a GPU-only test would not run in CI."""

    hw_classification = HardwareClassification.GENERIC

    def setUp(self):
        super().setUp()
        tunableop._worker = None
        tunableop._worker_key = None
        self.created = []

        def factory(device=None):
            proc = _FakeProcess(device=device)
            self.created.append(proc)
            return proc

        patcher = mock.patch(
            "torch._inductor.autotune_process.TuningProcess", side_effect=factory
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(setattr, tunableop, "_worker", None)
        self.addCleanup(setattr, tunableop, "_worker_key", None)

    def _submit(self, device=0, filename="t0.csv", job=None):
        with (
            mock.patch("torch.cuda.current_device", return_value=device),
            mock.patch("torch.cuda.tunable.get_filename", return_value=filename),
        ):
            return tunableop.submit(job or tunableop.BenchmarkJob(None, tuning=False))

    def test_worker_does_not_inherit_a_rewritten_visibility_mask(self):
        # TuningProcess.start() REPLACES CUDA_VISIBLE_DEVICES with str(device),
        # so passing a logical ordinal would point the worker at a different
        # physical GPU: with CUDA_VISIBLE_DEVICES=4,7 and logical 1, the parent
        # is on physical 7 and the worker would take physical 1.
        self._submit(device=1)
        self.assertEqual(len(self.created), 1)
        self.assertIsNone(self.created[0].device)

    def test_job_carries_the_submitting_logical_device(self):
        # The mask is left alone, so the ordinal means the same GPU in the
        # worker -- but only if the job actually carries it.
        job = tunableop.BenchmarkJob(None, tuning=False)
        self._submit(device=3, job=job)
        self.assertEqual(job.device, 3)

    def test_worker_restarts_when_the_device_changes(self):
        self._submit(device=0)
        first = self.created[0]
        self._submit(device=1)
        self.assertEqual(len(self.created), 2)
        # Graceful on a key change: sentinel, then reap. Never SIGKILL.
        self.assertEqual(first.shutdown_calls, 1)
        self.assertEqual(first.close_calls, 1)
        self.assertEqual(first.kill_calls, 0)

    def test_worker_restarts_when_the_results_file_changes(self):
        # The worker reads the results file once at startup, so a second compile
        # pointing somewhere else must not keep appending to the first file.
        self._submit(filename="a.csv")
        first = self.created[0]
        self._submit(filename="b.csv")
        self.assertEqual(len(self.created), 2)
        self.assertEqual(first.shutdown_calls, 1)
        self.assertEqual(first.close_calls, 1)
        self.assertEqual(first.kill_calls, 0)

    def test_worker_is_reused_when_device_and_file_are_unchanged(self):
        self._submit(device=2, filename="a.csv")
        self._submit(device=2, filename="a.csv")
        self.assertEqual(len(self.created), 1)

    def test_faults_are_reported_rather_than_raised(self):
        # A compile must continue when a GEMM cannot be benchmarked.
        for exc, status in (
            (TimeoutError(), "timeout"),
            (EOFError(), "crash"),
            (RuntimeError("boom"), "error"),
        ):
            tunableop._worker = None
            tunableop._worker_key = None
            self.created.clear()
            job = tunableop.BenchmarkJob(None, tuning=False)
            with (
                mock.patch("torch.cuda.current_device", return_value=0),
                mock.patch("torch.cuda.tunable.get_filename", return_value="t.csv"),
            ):
                tunableop.get_worker((0, "t.csv")).raises = exc
                outcome = tunableop.submit(job)
            self.assertEqual(outcome.status, status)
            self.assertIsNone(outcome.time)

    def test_a_failure_before_the_worker_is_also_reported(self):
        # The caller is a compile, not a benchmark harness: anything submit()
        # touches -- including resolving the current device -- has to come back
        # as an outcome rather than unwind the compile.
        job = tunableop.BenchmarkJob(None, tuning=False)
        with mock.patch(
            "torch.cuda.current_device", side_effect=RuntimeError("no context")
        ):
            outcome = tunableop.submit(job)
        self.assertEqual(outcome.status, "error")
        self.assertIsNone(outcome.time)

    def test_error_restarts_the_worker(self):
        # A catchable exception can leave a sticky GPU fault behind; the worker
        # must not survive it, or every later GEMM fails the same way.
        job = tunableop.BenchmarkJob(None, tuning=False)
        with (
            mock.patch("torch.cuda.current_device", return_value=0),
            mock.patch("torch.cuda.tunable.get_filename", return_value="t.csv"),
        ):
            worker = tunableop.get_worker((0, "t.csv"))
            worker.raises = RuntimeError("boom")
            outcome = tunableop.submit(job)
            self.assertEqual(outcome.status, "error")
            self.assertIsNone(tunableop._worker)
            # Killed, not asked: a child wedged in device teardown would never
            # act on the sentinel, and the handle is gone after this.
            self.assertEqual(worker.kill_calls, 1)
            self.assertEqual(worker.shutdown_calls, 0)

    def test_setup_failure_does_not_restart_worker(self):
        # A failure before the worker is involved (no device) must not kill a
        # healthy worker another thread may be using.
        job = tunableop.BenchmarkJob(None, tuning=False)
        with (
            mock.patch("torch.cuda.current_device", return_value=0),
            mock.patch("torch.cuda.tunable.get_filename", return_value="t.csv"),
        ):
            tunableop.get_worker((0, "t.csv"))
            first = tunableop._worker
        with mock.patch(
            "torch.cuda.current_device", side_effect=RuntimeError("no context")
        ):
            outcome = tunableop.submit(job)
            self.assertEqual(outcome.status, "error")
            self.assertIs(tunableop._worker, first)

    def test_submit_stamps_parent_settings(self):
        tunable = mock.Mock()
        tunable.get_filename.return_value = "s.csv"
        tunable.get_max_tuning_duration.return_value = 11
        tunable.get_max_tuning_iterations.return_value = 12
        # The getter reports bytes; the setter takes MB. 13 MiB in bytes.
        tunable.get_rotating_buffer_size.return_value = 13 * 1024 * 1024
        tunable.get_cublaslt_requested_algo_count.return_value = 14
        tunable.wildcard_fallback_is_enabled.return_value = True
        job = tunableop.BenchmarkJob(None, tuning=False)
        with (
            mock.patch("torch.cuda.current_device", return_value=0),
            mock.patch("torch.cuda.tunable", tunable),
        ):
            tunableop.submit(job)
        self.assertEqual(
            job.settings,
            {
                "set_max_tuning_duration": 11,
                "set_max_tuning_iterations": 12,
                "set_rotating_buffer_size": 13,  # MB, converted from bytes
                "set_cublaslt_requested_algo_count": 14,
                "wildcard_fallback_enable": True,
            },
        )

    def test_snapshot_skips_unreadable_knobs(self):
        tunable = mock.Mock()
        tunable.get_filename.return_value = "s.csv"
        tunable.get_max_tuning_duration.side_effect = RuntimeError("gone")
        job = tunableop.BenchmarkJob(None, tuning=False)
        with (
            mock.patch("torch.cuda.current_device", return_value=0),
            mock.patch("torch.cuda.tunable", tunable),
        ):
            tunableop.submit(job)
        self.assertNotIn("set_max_tuning_duration", job.settings)
        # The failure is contained: the submission still went through.
        self.assertEqual(job.filename, "s.csv")

    def test_submit_stamps_resolved_filename(self):
        # get_filename() is "" until the manager initializes; submit() must
        # resolve the C++ default rule instead of keying the worker on "".
        job = tunableop.BenchmarkJob(None, tuning=False)
        env = {k: v for k, v in os.environ.items() if k != "PYTORCH_TUNABLEOP_FILENAME"}
        with (
            mock.patch("torch.cuda.current_device", return_value=0),
            mock.patch("torch.cuda.tunable.get_filename", return_value=""),
            mock.patch.dict("os.environ", env, clear=True),
        ):
            # No PYTORCH_TUNABLEOP_FILENAME: default rule applies, ordinal included.
            tunableop.submit(job)
            self.assertEqual(job.filename, "tunableop_results0.csv")
            self.assertEqual(tunableop._worker_key, (0, "tunableop_results0.csv"))

    def test_submit_prefers_env_filename_when_unset(self):
        # The C++ manager inserts the device ordinal into the env name; the
        # worker must use the same file or the runtime never sees its results.
        job = tunableop.BenchmarkJob(None, tuning=False)
        with (
            mock.patch("torch.cuda.current_device", return_value=1),
            mock.patch("torch.cuda.tunable.get_filename", return_value=""),
            mock.patch.dict("os.environ", {"PYTORCH_TUNABLEOP_FILENAME": "env.csv"}),
        ):
            tunableop.submit(job)
            self.assertEqual(job.filename, "env1.csv")

    def test_insert_device_ordinal_matches_cpp_rule(self):
        self.assertEqual(tunableop._insert_device_ordinal("r%d.csv", 3), "r3.csv")
        self.assertEqual(tunableop._insert_device_ordinal("a.b.csv", 2), "a.b2.csv")
        self.assertEqual(tunableop._insert_device_ordinal("results", 0), "results0")

    def test_submit_uses_parent_filename_when_set(self):
        job = tunableop.BenchmarkJob(None, tuning=False)
        with (
            mock.patch("torch.cuda.current_device", return_value=0),
            mock.patch("torch.cuda.tunable.get_filename", return_value="parent.csv"),
        ):
            tunableop.submit(job)
            self.assertEqual(job.filename, "parent.csv")


class BenchmarkJobCallTest(TestCase):
    """BenchmarkJob.__call__ runs in the worker: set device first, then flags."""

    hw_classification = HardwareClassification.GENERIC

    def test_call_sets_device_flags_and_filename(self):
        calls = []

        class FakeTunable:
            def set_filename(self, filename, insert_device_ordinal=False):
                calls.append(("set_filename", filename, insert_device_ordinal))

            def enable(self, val):
                calls.append(("enable", val))

            def tuning_enable(self, val):
                calls.append(("tuning_enable", val))

            def _nearest_match_enable(self, val):
                calls.append(("_nearest_match_enable", val))

            def _record_borrowed_enable(self, val):
                calls.append(("_record_borrowed_enable", val))

        class FakeBmreq:
            benchmark_with_cudagraphs = True

            def benchmark(self):
                calls.append(("benchmark",))
                return 1.5

        job = tunableop.BenchmarkJob(FakeBmreq(), tuning=True, nearest_match=True)
        job.device = 3
        job.filename = "w.csv"
        with (
            mock.patch(
                "torch.cuda.set_device",
                side_effect=lambda d: calls.append(("set_device", d)),
            ),
            mock.patch("torch.cuda.tunable", FakeTunable()),
        ):
            self.assertEqual(job(), 1.5)
        # Device first, then filename, then flags; warm benchmark, tuning off,
        # measured benchmark; cudagraphs cleared.
        kinds = [c[0] for c in calls]
        self.assertEqual(
            kinds,
            [
                "set_device",
                "set_filename",
                "enable",
                "tuning_enable",
                "_nearest_match_enable",
                "_record_borrowed_enable",
                "benchmark",
                "tuning_enable",
                "benchmark",
            ],
        )
        self.assertEqual(calls[0], ("set_device", 3))
        self.assertEqual(calls[1], ("set_filename", "w.csv", False))
        self.assertFalse(job.bmreq.benchmark_with_cudagraphs)

    def test_call_skips_set_filename_when_empty(self):
        seen = []

        class FakeTunable:
            def set_filename(self, *a, **k):
                seen.append((a, k))

            def enable(self, val):
                pass

            def tuning_enable(self, val):
                pass

            def _record_borrowed_enable(self, val):
                pass

            def _nearest_match_enable(self, val):
                pass

        class FakeBmreq:
            benchmark_with_cudagraphs = True

            def benchmark(self):
                return 1.0

        job = tunableop.BenchmarkJob(FakeBmreq(), tuning=False)
        job.filename = ""
        with (
            mock.patch("torch.cuda.set_device"),
            mock.patch("torch.cuda.tunable", FakeTunable()),
        ):
            job()
        self.assertEqual(seen, [])


class TunableopModeValidationTest(TestCase):
    """Invalid TORCHINDUCTOR_ATEN_TUNABLEOP_MODE values warn instead of silently
    disabling."""

    hw_classification = HardwareClassification.GENERIC

    def test_invalid_mode_warns_and_disables(self):
        tunableop._mode_warned = False
        try:
            with mock.patch.object(tunableop.config, "aten_tunableop_mode", "Tune"):
                with self.assertLogs(tunableop.log, level="WARNING") as logs:
                    self.assertFalse(tunableop.is_tune_mode())
                    self.assertFalse(tunableop.is_reuse_mode())
                    self.assertFalse(tunableop.is_enabled())
            self.assertTrue(
                any("TORCHINDUCTOR_ATEN_TUNABLEOP_MODE" in m for m in logs.output)
            )
            # Warns only once: the second invalid value logs nothing.
            with mock.patch.object(tunableop.config, "aten_tunableop_mode", "bogus"):
                with self.assertNoLogs(tunableop.log, level="WARNING"):
                    tunableop.is_enabled()
        finally:
            tunableop._mode_warned = False

    def test_valid_modes_do_not_warn(self):
        tunableop._mode_warned = False
        try:
            for mode in ("off", "tune", "reuse"):
                with mock.patch.object(tunableop.config, "aten_tunableop_mode", mode):
                    with self.assertNoLogs(tunableop.log, level="WARNING"):
                        tunableop.is_enabled()
        finally:
            tunableop._mode_warned = False


class TunableOpRealWorkerTest(TestCase):
    """Runs jobs in a real TuningProcess.

    _FakeProcess above cannot model get() killing or closing the process, or
    put() restarting it, so the central claim -- a hang or crash leaves the
    worker restartable and the next submission succeeds -- has no coverage
    without this. No GPU needed: the jobs never touch CUDA, and only
    current_device/get_filename are patched in the parent.

    Jobs are _TestBenchmarkRequest instances passed directly: the request is
    callable (see its __call__ in autotune_process.py) and unpickles in the
    child, which only has torch on its path, so test-file classes cannot be
    used. submit() stamps job.device before sending.
    """

    hw_classification = HardwareClassification.GENERIC

    def setUp(self):
        super().setUp()
        tunableop._worker = None
        tunableop._worker_key = None
        self.addCleanup(tunableop.shutdown_worker)
        self.addCleanup(setattr, tunableop, "_worker", None)
        self.addCleanup(setattr, tunableop, "_worker_key", None)

    def _job(self, **kwargs):
        return _TestBenchmarkRequest(1.0, **kwargs)

    def _submit(self, job, timeout):
        # current_device is None (not 0): submit() stamps job.device with it,
        # and _TestBenchmarkRequest.benchmark only asserts the visibility mask
        # when device is not None. The stamping path itself is covered by the
        # fake tests; here the jobs must run assertion-free in the child.
        with (
            mock.patch("torch.cuda.current_device", return_value=None),
            mock.patch("torch.cuda.tunable.get_filename", return_value="t.csv"),
        ):
            return tunableop.submit(job, timeout=timeout)

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_hang_then_next_submission_succeeds(self):
        outcome = self._submit(self._job(sleep=600.0), timeout=5.0)
        self.assertEqual(outcome.status, "timeout")
        self.assertIsNone(outcome.time)
        # get() kills the worker on timeout; put() must restart it.
        outcome = self._submit(self._job(), timeout=120.0)
        self.assertEqual(outcome.status, "ok")
        self.assertEqual(outcome.time, 1.0)

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_crash_then_next_submission_succeeds(self):
        outcome = self._submit(self._job(crash=True), timeout=120.0)
        self.assertEqual(outcome.status, "crash")
        self.assertIsNone(outcome.time)
        # get() closes the dead worker on EOF; put() must restart it.
        outcome = self._submit(self._job(), timeout=120.0)
        self.assertEqual(outcome.status, "ok")
        self.assertEqual(outcome.time, 1.0)


class TunableOpRocmWorkerTest(TestCase):
    """Real BenchmarkJob through a real worker on ROCm hardware.

    Covers what the fake and _TestBenchmarkRequest tests cannot: the job and
    its bmreq surviving the pickle round trip, set_device/set_filename/flag
    ordering against lazy results-manager init in the child, and the worker
    appending to the file the parent's runtime reads. Needs ROCm CI (a GPU and
    a subprocess torch import that works); skipped elsewhere. Only
    current_device is patched -- the filename flows through the real
    set_filename so resolve_filename takes the parent's file.
    """

    hw_classification = HardwareClassification.ACCELERATOR

    @unittest.skipIf(not torch.cuda.is_available(), "needs a device")
    def test_rotating_buffer_size_round_trips_through_snapshot(self):
        # The mocked settings test cannot catch a unit mismatch, because it
        # chooses both sides of the comparison. This drives the real getter and
        # setter: get reports bytes, set takes MB, so stamping the raw value
        # would re-scale by 2**20 and overflow the setter's signed int.
        tunable = torch.cuda.tunable
        previous = tunable.get_rotating_buffer_size()
        self.addCleanup(tunable.set_rotating_buffer_size, previous // (1024 * 1024))

        tunable.set_rotating_buffer_size(4)  # MB
        parent_bytes = tunable.get_rotating_buffer_size()
        self.assertEqual(parent_bytes, 4 * 1024 * 1024)

        stamped = tunableop.snapshot_settings()["set_rotating_buffer_size"]
        tunable.set_rotating_buffer_size(stamped)  # what the worker does
        self.assertEqual(tunable.get_rotating_buffer_size(), parent_bytes)

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_real_job_tunes_and_appends_to_shared_file(self):
        import tempfile

        from torch._inductor.autotune_process import _TestTunableMatmulRequest

        tunableop._worker = None
        tunableop._worker_key = None
        self.addCleanup(tunableop.shutdown_worker)
        self.addCleanup(setattr, tunableop, "_worker", None)
        self.addCleanup(setattr, tunableop, "_worker_key", None)

        with tempfile.TemporaryDirectory(prefix="tunableop_worker_roc_") as d:
            results_file = os.path.join(d, "results.csv")
            torch.cuda.tunable.set_filename(results_file, False)
            try:
                torch.cuda.tunable.enable(False)
                job = tunableop.BenchmarkJob(_TestTunableMatmulRequest(), tuning=True)
                with mock.patch(
                    "torch.cuda.current_device",
                    return_value=torch.cuda.current_device(),
                ):
                    outcome = tunableop.submit(job, timeout=600.0)
            finally:
                torch.cuda.tunable.enable(False)
                torch.cuda.tunable.set_filename("", False)
            self.assertEqual(outcome.status, "ok")
            self.assertGreater(outcome.time, 0)
            with open(results_file) as f:
                content = f.read()
            self.assertIn("GemmTunableOp", content)


def _extern_caller(choice_name="mm", cuda_out=True):
    """An ExternKernelCaller stand-in for routing tests (no GPU, no bmreq build)."""
    from types import SimpleNamespace

    from torch._inductor import select_algorithm as sa

    caller = mock.Mock(spec=sa.ExternKernelCaller)
    caller.name = choice_name
    caller.choice = SimpleNamespace(name=choice_name)
    caller.bmreq = mock.Mock()
    caller.bmreq.benchmark_with_cudagraphs = False
    caller.bmreq.config_cudagraph_benchmarking = False
    caller.bmreq.benchmark.return_value = 2.0
    caller.tunable_dyn_dims_mask = None
    caller._benchmark_with_cudagraphs = False
    # spec autospecs the new _benchmark_inline helper as a Mock; make it run
    # the real inline body so routing tests observe real return values.
    caller._benchmark_inline = (
        lambda args, out: sa.ExternKernelCaller._benchmark_inline(caller, args, out)
    )
    out = mock.Mock()
    out.is_cuda = cuda_out
    return caller, out


class ExternRoutingTest(TestCase):
    """ExternKernelCaller.benchmark routes only tunable GEMMs to the worker."""

    hw_classification = HardwareClassification.GENERIC

    def _benchmark(self, caller, out):
        from torch._inductor import select_algorithm as sa

        return sa.ExternKernelCaller.benchmark(caller, out=out)

    def test_gemm_candidate_is_submitted_with_flags(self):
        from torch._inductor import tunableop as _top

        caller, out = _extern_caller("mm")
        caller.tunable_dyn_dims_mask = (True, False, False, False)
        seen = {}

        def _fake_submit(job, timeout=None):
            seen["job"] = job
            return _top.Outcome(1.5, "ok")

        with (
            mock.patch("torch._inductor.tunableop.is_enabled", return_value=True),
            mock.patch("torch._inductor.tunableop.is_tune_mode", return_value=True),
            mock.patch("torch._inductor.tunableop.is_reuse_mode", return_value=False),
            mock.patch("torch._inductor.tunableop.submit", side_effect=_fake_submit),
        ):
            self.assertEqual(self._benchmark(caller, out), 1.5)
        job = seen["job"]
        self.assertTrue(job.tuning)
        self.assertFalse(job.nearest_match)
        self.assertFalse(job.record_borrowed)
        self.assertEqual(job.dyn_dims_mask, (True, False, False, False))

    def test_non_gemm_candidate_stays_inline(self):
        caller, out = _extern_caller("convolution")
        with (
            mock.patch("torch._inductor.tunableop.is_enabled", return_value=True),
            mock.patch(
                "torch._inductor.tunableop.submit",
                side_effect=AssertionError("must not submit"),
            ),
        ):
            self.assertEqual(self._benchmark(caller, out), 2.0)
        caller.bmreq.benchmark.assert_called_once()

    def test_cpu_out_never_submitted(self):
        caller, out = _extern_caller("mm", cuda_out=False)
        with (
            mock.patch("torch._inductor.tunableop.is_enabled", return_value=True),
            mock.patch(
                "torch._inductor.tunableop.submit",
                side_effect=AssertionError("must not submit"),
            ),
        ):
            self.assertEqual(self._benchmark(caller, out), 2.0)

    def test_non_ok_falls_through_with_tunableop_off(self):
        from torch._inductor import tunableop as _top

        caller, out = _extern_caller("mm")
        tunable = mock.Mock()
        tunable.is_enabled.return_value = True
        tunable.tuning_is_enabled.return_value = True
        with (
            mock.patch("torch._inductor.tunableop.is_enabled", return_value=True),
            mock.patch(
                "torch._inductor.tunableop.submit",
                return_value=_top.Outcome(None, "timeout"),
            ),
            mock.patch("torch.cuda.tunable", tunable),
        ):
            self.assertEqual(self._benchmark(caller, out), 2.0)
        # TunableOp disabled around the inline fallback, restored after.
        self.assertEqual(
            [c.args for c in tunable.enable.call_args_list], [(False,), (True,)]
        )
        self.assertEqual(
            [c.args for c in tunable.tuning_enable.call_args_list],
            [(False,), (True,)],
        )

    def test_nonfinite_worker_time_is_error(self):
        for bad in (float("inf"), float("nan"), 0.0, -1.0):
            tunableop._worker = None
            tunableop._worker_key = None
            proc = _FakeProcess(device=None)
            proc.result = bad
            with (
                mock.patch("torch.cuda.current_device", return_value=0),
                mock.patch("torch.cuda.tunable.get_filename", return_value="t.csv"),
                mock.patch(
                    "torch._inductor.autotune_process.TuningProcess",
                    return_value=proc,
                ),
            ):
                outcome = tunableop.submit(tunableop.BenchmarkJob(None, tuning=False))
            self.assertEqual(outcome.status, "error", f"value={bad!r}")
            self.assertIsNone(outcome.time)
        tunableop._worker = None
        tunableop._worker_key = None


class WarmMeasureTest(TestCase):
    """_warm_then_measure pays tuning work outside the measured average."""

    hw_classification = HardwareClassification.GENERIC

    def _job(self, **kwargs):
        bmreq = mock.Mock()
        bmreq.benchmark_with_cudagraphs = True
        bmreq.config_cudagraph_benchmarking = True
        bmreq.benchmark.return_value = 3.0
        job = tunableop.BenchmarkJob(bmreq, tuning=True, **kwargs)
        job.device = 0
        job.filename = ""
        return job

    def test_measure_runs_warm_then_measured(self):
        job = self._job()
        with (
            mock.patch("torch.cuda.set_device"),
            mock.patch("torch.cuda.tunable", mock.Mock()),
        ):
            self.assertEqual(job(), 3.0)
        # Warm + measured: two full benchmarks.
        self.assertEqual(job.bmreq.benchmark.call_count, 2)
        self.assertFalse(job.bmreq.benchmark_with_cudagraphs)
        self.assertFalse(job.bmreq.config_cudagraph_benchmarking)

    def test_failed_warm_still_measures(self):
        job = self._job()
        job.bmreq.benchmark.side_effect = [RuntimeError("boom"), 4.0]
        with (
            mock.patch("torch.cuda.set_device"),
            mock.patch("torch.cuda.tunable", mock.Mock()),
        ):
            self.assertEqual(job(), 4.0)

    def test_tune_only_skips_second_benchmark(self):
        job = self._job(measure=False)
        with (
            mock.patch("torch.cuda.set_device"),
            mock.patch("torch.cuda.tunable", mock.Mock()),
        ):
            self.assertEqual(job(), 3.0)
        self.assertEqual(job.bmreq.benchmark.call_count, 1)

    def test_tune_only_failed_warm_reraises(self):
        # A failed tune-only warm re-raises (rather than reporting inf) so the
        # exception crosses the worker pipe and submit() restarts a possibly
        # poisoned worker.
        job = self._job(measure=False)
        job.bmreq.benchmark.side_effect = RuntimeError("boom")
        with (
            mock.patch("torch.cuda.set_device"),
            mock.patch("torch.cuda.tunable", mock.Mock()),
        ):
            with self.assertRaisesRegex(RuntimeError, "tune-only warm-up failed"):
                job()


if __name__ == "__main__":
    run_tests()
