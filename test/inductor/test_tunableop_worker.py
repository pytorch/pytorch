# Owner(s): ["module: inductor"]

import contextlib
import json
import os
import subprocess
import tempfile
import unittest
from unittest import mock

import torch
from torch._inductor import config as inductor_config, tunableop
from torch._inductor.autotune_process import _TestBenchmarkRequest, _TestTunableOpJob
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
        # Tests override these to drive _shutdown_locked's bounded-wait branches.
        self._alive = False
        self.process = None
        # TuningProcess.running: cleared only when the pipe itself failed.
        self.running = False

    def put(self, job):
        self.sent.append(job)

    def get(self, timeout=None):
        if self.raises is not None:
            raise self.raises
        return self.result

    def shutdown(self, wait=True):
        self.shutdown_calls += 1

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
    Neither needs a real device, so these run wherever the suite does."""

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

    def _stamped(self):
        """The envelope submit() put on the pipe."""
        return self.created[-1].sent[-1]

    def _submit(self, device=0, filename="t0.csv", job=None):
        with (
            mock.patch("torch.cuda.current_device", return_value=device),
            mock.patch("torch.cuda.tunable.get_filename", return_value=filename),
        ):
            return tunableop.submit(job or tunableop.BenchmarkJob(None, tuning=False))

    def test_worker_does_not_inherit_a_rewritten_visibility_mask(self):
        # start() REPLACES CUDA_VISIBLE_DEVICES with str(device), so a logical
        # ordinal would point the worker at a different physical GPU.
        self._submit(device=1)
        self.assertEqual(len(self.created), 1)
        self.assertIsNone(self.created[0].device)

    def test_job_carries_the_submitting_logical_device(self):
        # The mask is left alone, so the ordinal means the same GPU in the
        # worker -- but only if the job actually carries it.
        job = tunableop.BenchmarkJob(None, tuning=False)
        self._submit(device=3, job=job)
        self.assertEqual(self._stamped().device, 3)

    def test_submit_does_not_mutate_the_job(self):
        # The state submit() resolves travels in the envelope, so a job carries
        # only what its caller gave it.
        job = tunableop.BenchmarkJob(None, tuning=False)
        before = dict(vars(job))
        self._submit(device=3, job=job)
        self.assertEqual(vars(job), before)
        self.assertIsNot(self._stamped(), job)

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

    def _live_worker(self):
        self._submit(device=0)
        first = self.created[0]
        first._alive = True
        first.process = mock.Mock()
        return first

    def test_a_wedged_child_is_killed_and_reaped(self):
        # A child stuck in device teardown never acts on the sentinel; without
        # the kill it holds the GPU and stalls every later submit.
        first = self._live_worker()
        first.process.wait.side_effect = subprocess.TimeoutExpired("worker", 10.0)
        self._submit(device=1)
        self.assertEqual(first.kill_calls, 1)
        self.assertEqual(first.close_calls, 0)
        # Once for the bounded wait, once reaping the killed child.
        self.assertEqual(first.process.wait.call_count, 2)

    def test_a_broken_pipe_during_shutdown_still_kills_the_child(self):
        # shutdown() writes the exit sentinel; a child that already closed its
        # read end makes that raise, and it would otherwise be orphaned.
        first = self._live_worker()
        first.shutdown = mock.Mock(side_effect=BrokenPipeError())
        self._submit(device=1)
        self.assertEqual(first.kill_calls, 1)

    def test_a_child_that_exits_within_the_bound_is_not_killed(self):
        first = self._live_worker()
        self._submit(device=1)
        self.assertEqual(first.kill_calls, 0)
        self.assertEqual(first.close_calls, 1)
        self.assertEqual(first.process.wait.call_count, 1)

    def test_a_crashed_but_live_child_is_still_killed(self):
        # The child can close its pipes and then wedge in device teardown. The
        # parent sees EOF, so running is already False and TuningProcess.kill()
        # skips the signal -- orphaning a process that still holds the GPU.
        process = mock.Mock()
        process.poll.return_value = None  # closed its pipes, still up
        _, outcome = self._submit_with_fault(EOFError(), process=process)
        self.assertEqual(outcome.status, "crash")
        self.assertEqual(process.kill.call_count, 1)

    def test_unset_rotating_buffer_uses_the_current_devices_l2(self):
        # GetRotatingBufferSize reports the *current* device's L2, while the job
        # can target another. Comparing against the job's device would miss the
        # "unset" case on a host whose GPUs have different L2 sizes.
        current_l2 = 4 * 1024 * 1024
        job_l2 = 6 * 1024 * 1024
        bmreq = mock.Mock()
        bmreq.output_tensor_meta = mock.Mock(device=torch.device("cuda", 1))
        bmreq.input_tensor_meta = []
        tunable = mock.Mock()
        tunable.get_filename.return_value = "s.csv"
        tunable.get_rotating_buffer_size.return_value = current_l2
        job = tunableop.BenchmarkJob(bmreq, tuning=False)
        with (
            mock.patch("torch.cuda.current_device", return_value=0),
            mock.patch("torch.cuda.tunable", tunable),
            mock.patch(
                "torch.cuda.get_device_properties",
                side_effect=lambda i: mock.Mock(
                    L2_cache_size=job_l2 if i == 1 else current_l2
                ),
            ),
        ):
            tunableop.submit(job)
        self.assertEqual(self._stamped().device, 1)
        self.assertEqual(self._stamped().settings["set_rotating_buffer_size"], -1)

    def _cuda1_job(self, parent_filename):
        bmreq = mock.Mock()
        bmreq.output_tensor_meta = mock.Mock(device=torch.device("cuda", 1))
        bmreq.input_tensor_meta = []
        tunable = mock.Mock()
        tunable.get_filename.return_value = parent_filename
        return tunableop.BenchmarkJob(bmreq, tuning=False), tunable

    def test_worker_device_comes_from_the_job_not_the_process(self):
        # Inductor does not set the CUDA current device while lowering, so a
        # graph on cuda:1 can be autotuned with current_device() == 0.
        job, tunable = self._cuda1_job("")
        env = {k: v for k, v in os.environ.items() if k != "PYTORCH_TUNABLEOP_FILENAME"}
        with (
            mock.patch("torch.cuda.current_device", return_value=0),
            mock.patch("torch.cuda.tunable", tunable),
            mock.patch.dict(os.environ, env, clear=True),
        ):
            tunableop.submit(job)
        self.assertEqual(self._stamped().device, 1)
        # Nothing has set a filename, so submit() derives one from the job's
        # device and pins it; the runtime then opens what the worker wrote.
        self.assertEqual(self._stamped().filename, "tunableop_results1.csv")

    def test_pinned_filename_is_reused_for_a_job_on_another_device(self):
        # One manager per process picks one filename, so leaving the parent
        # unpinned lets a second device's results land where nothing reads.
        pinned = {}

        def get_filename():
            return pinned.get("name", "")

        def set_filename(name, insert_device_ordinal=False):
            pinned["name"] = name

        tunable = mock.Mock()
        tunable.get_filename.side_effect = get_filename
        tunable.set_filename.side_effect = set_filename
        env = {k: v for k, v in os.environ.items() if k != "PYTORCH_TUNABLEOP_FILENAME"}
        with (
            mock.patch("torch.cuda.tunable", tunable),
            mock.patch.dict(os.environ, env, clear=True),
        ):
            for device in (0, 1):
                bmreq = mock.Mock()
                bmreq.output_tensor_meta = mock.Mock(
                    device=torch.device("cuda", device)
                )
                bmreq.input_tensor_meta = []
                with mock.patch("torch.cuda.current_device", return_value=device):
                    tunableop.submit(tunableop.BenchmarkJob(bmreq, tuning=False))
                # Both devices write the file the runtime will open.
                self.assertEqual(self._stamped().filename, "tunableop_results0.csv")
        self.assertEqual(tunable.set_filename.call_count, 1)

    def test_a_set_parent_filename_wins_over_the_job_device(self):
        # The branch the empty-filename test cannot reach: SetFilename has
        # already baked an ordinal in, so re-deriving one here would disagree
        # with the file the runtime opens.
        job, tunable = self._cuda1_job("parent0.csv")
        with (
            mock.patch("torch.cuda.current_device", return_value=0),
            mock.patch("torch.cuda.tunable", tunable),
        ):
            tunableop.submit(job)
        self.assertEqual(self._stamped().device, 1)
        self.assertEqual(self._stamped().filename, "parent0.csv")

    def test_worker_is_reused_when_device_and_file_are_unchanged(self):
        self._submit(device=2, filename="a.csv")
        self._submit(device=2, filename="a.csv")
        self.assertEqual(len(self.created), 1)

    def _submit_with_fault(self, exc, running=False, process=None):
        """Arm a fresh worker's get() with exc, then submit through it."""
        tunableop._worker = None
        tunableop._worker_key = None
        self.created.clear()
        with (
            mock.patch("torch.cuda.current_device", return_value=0),
            mock.patch("torch.cuda.tunable.get_filename", return_value="t.csv"),
        ):
            with tunableop._worker_lock:
                worker = tunableop._get_worker_locked((0, "t.csv"))
            worker.raises = exc
            worker.running = running  # True means the pipe is fine, the job failed
            if process is not None:
                worker.process = process
            outcome = tunableop.submit(tunableop.BenchmarkJob(None, tuning=False))
        return worker, outcome

    def test_faults_are_reported_rather_than_raised(self):
        # A compile must continue when a GEMM cannot be benchmarked, but a
        # fault repeating for every GEMM must still be visible by default.
        tunableop._warned_faults.clear()
        self.addCleanup(tunableop._warned_faults.clear)
        for exc, status in (
            (TimeoutError(), "timeout"),
            (EOFError(), "crash"),
            (RuntimeError("boom"), "error"),
        ):
            with self.subTest(status=status):
                with self.assertLogs(tunableop.log, level="WARNING") as logs:
                    _, outcome = self._submit_with_fault(exc)
                self.assertEqual(outcome.status, status)
                self.assertIsNone(outcome.time)
                self.assertIn(status, "".join(logs.output))
                # Warned once per kind; a repeat drops to debug.
                with self.assertNoLogs(tunableop.log, level="WARNING"):
                    self._submit_with_fault(exc)

    def test_job_raised_pipe_exception_is_an_error_not_a_timeout(self):
        # The discriminator is running, which a real pipe fault clears and a
        # child-raised exception does not.
        for exc in (TimeoutError("from the job"), EOFError("from the job")):
            with self.subTest(exc=type(exc).__name__):
                worker, outcome = self._submit_with_fault(exc, running=True)
                self.assertEqual(outcome.status, "error")
                self.assertIsNone(outcome.time)
                self.assertIsNone(tunableop._worker)
                self.assertEqual(worker.kill_calls, 1)

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

    def test_an_interrupt_tears_the_worker_down(self):
        # BaseException is not Exception: an interrupt leaves the child running
        # with its result still unread in the pipe, so a reused worker would
        # hand that result to the next GEMM, and every one after it.
        with self.assertRaises(KeyboardInterrupt):
            self._submit_with_fault(KeyboardInterrupt(), running=True)
        self.assertIsNone(tunableop._worker)
        self.assertEqual(self.created[-1].kill_calls, 1)

    def test_error_restarts_the_worker(self):
        # A catchable exception can leave a sticky GPU fault behind; the worker
        # must not survive it, or every later GEMM fails the same way.
        job = tunableop.BenchmarkJob(None, tuning=False)
        with (
            mock.patch("torch.cuda.current_device", return_value=0),
            mock.patch("torch.cuda.tunable.get_filename", return_value="t.csv"),
        ):
            with tunableop._worker_lock:
                worker = tunableop._get_worker_locked((0, "t.csv"))
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
            with tunableop._worker_lock:
                tunableop._get_worker_locked((0, "t.csv"))
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
            mock.patch(
                "torch.cuda.get_device_properties",
                return_value=mock.Mock(L2_cache_size=4 * 1024 * 1024),
            ),
        ):
            tunableop.submit(job)
        self.assertEqual(
            self._stamped().settings,
            {
                "set_max_tuning_duration": 11,
                "set_max_tuning_iterations": 12,
                "set_rotating_buffer_size": 13,  # MB, converted from bytes
                "set_cublaslt_requested_algo_count": 14,
                "wildcard_fallback_enable": True,
            },
        )

    def test_unset_rotating_buffer_is_stamped_as_unset(self):
        # Unset, the getter reports the device's L2 size. Converting that to MB
        # would hand the worker a rounded-down constant instead of the parent's
        # "use L2" policy, or 0 -- meaning no rotation at all -- below 1 MiB.
        l2 = 4 * 1024 * 1024 + 512 * 1024  # deliberately not a whole MiB
        tunable = mock.Mock()
        tunable.get_filename.return_value = "s.csv"
        tunable.get_rotating_buffer_size.return_value = l2
        job = tunableop.BenchmarkJob(None, tuning=False)
        with (
            mock.patch("torch.cuda.current_device", return_value=0),
            mock.patch("torch.cuda.tunable", tunable),
            mock.patch(
                "torch.cuda.get_device_properties",
                return_value=mock.Mock(L2_cache_size=l2),
            ),
        ):
            tunableop.submit(job)
        self.assertEqual(self._stamped().settings["set_rotating_buffer_size"], -1)

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
        self.assertNotIn("set_max_tuning_duration", self._stamped().settings)
        # The failure is contained: the submission still went through.
        self.assertEqual(self._stamped().filename, "s.csv")

    def test_submit_resolves_and_pins_the_results_file(self):
        # One rule: the file the runtime's manager uses, or the one it would
        # pick, pinned so the parent and the worker cannot diverge.
        cases = (
            # name, parent filename, env var, device, expected, pinned
            ("default rule", "", None, 0, "tunableop_results0.csv", True),
            ("env name keeps the ordinal", "", "env.csv", 1, "env1.csv", True),
            ("empty env disables persistence", "", "", 0, "", False),
            ("parent's choice wins", "parent.csv", None, 0, "parent.csv", False),
        )
        key = "PYTORCH_TUNABLEOP_FILENAME"
        base = {k: v for k, v in os.environ.items() if k != key}
        for name, parent, env, device, expected, pinned in cases:
            with self.subTest(case=name):
                tunableop._worker = None
                tunableop._worker_key = None
                tunableop._env_warned.clear()
                self.created.clear()
                tunable = mock.Mock()
                tunable.get_filename.return_value = parent
                environ = dict(base) if env is None else {**base, key: env}
                expect = self.assertNoLogs if expected else self.assertLogs
                with (
                    mock.patch("torch.cuda.current_device", return_value=device),
                    mock.patch("torch.cuda.tunable", tunable),
                    mock.patch.dict(os.environ, environ, clear=True),
                    expect(tunableop.log, level="WARNING") as logs,
                ):
                    tunableop.submit(tunableop.BenchmarkJob(None, tuning=False))
                if not expected:
                    self.assertIn(key, logs.output[0])
                self.assertEqual(self._stamped().filename, expected)
                self.assertEqual(tunableop._worker_key, (device, expected))
                if pinned:
                    tunable.set_filename.assert_called_once_with(
                        expected, insert_device_ordinal=False
                    )
                else:
                    tunable.set_filename.assert_not_called()

    def test_insert_device_ordinal_rule(self):
        self.assertEqual(tunableop._insert_device_ordinal("r%d.csv", 3), "r3.csv")
        self.assertEqual(tunableop._insert_device_ordinal("a.b.csv", 2), "a.b2.csv")
        self.assertEqual(tunableop._insert_device_ordinal("results", 0), "results0")
        # An empty name disables persistence; SetFilename returns early on it.
        self.assertEqual(tunableop._insert_device_ordinal("", 0), "")


class _RecordingTunable:
    """Records every call BenchmarkJob.__call__ makes, in order."""

    def __init__(self, calls):
        self.calls = calls

    @contextlib.contextmanager
    def dynamic_dims_mask(self, **kwargs):
        self.calls.append(("enter", tuple(kwargs.values())))
        try:
            yield
        finally:
            self.calls.append(("exit",))

    def __getattr__(self, name):
        if name == "calls":
            raise AttributeError(name)
        return lambda *a, **kw: self.calls.append((name, *a, *kw.values()))


class _CallFakeBmreq:
    """Records its benchmark and returns a fixed time."""

    benchmark_with_cudagraphs = True
    config_cudagraph_benchmarking = True

    def __init__(self, calls, result=1.0):
        self.calls = calls
        self.result = result

    def benchmark(self):
        self.calls.append(("benchmark",))
        return self.result


class BenchmarkJobCallTest(TestCase):
    """BenchmarkJob.__call__ runs in the worker: set device first, then flags."""

    hw_classification = HardwareClassification.GENERIC

    def _run(self, job, calls, device=0, filename="", settings=None):
        with (
            mock.patch(
                "torch.cuda.set_device",
                side_effect=lambda d: calls.append(("set_device", d)),
            ),
            mock.patch("torch.cuda.tunable", _RecordingTunable(calls)),
        ):
            return job(device, filename, settings or {})

    def test_call_sets_device_flags_and_filename(self):
        calls = []
        job = tunableop.BenchmarkJob(
            _CallFakeBmreq(calls, 1.5), tuning=True, nearest_match=True
        )
        self.assertEqual(self._run(job, calls, device=3, filename="w.csv"), 1.5)
        # Device first, then filename, then flags; cudagraphs cleared.
        self.assertEqual(
            " ".join(c[0] for c in calls),
            "set_device set_filename enable tuning_enable "
            "_nearest_match_enable benchmark",
        )
        self.assertEqual(calls[0], ("set_device", 3))
        self.assertEqual(calls[1], ("set_filename", "w.csv", False))
        # benchmark_run_fn captures if either is set, so both must be cleared.
        self.assertFalse(job.bmreq.benchmark_with_cudagraphs)
        self.assertFalse(job.bmreq.config_cudagraph_benchmarking)

    def test_call_skips_set_filename_when_empty(self):
        calls = []
        job = tunableop.BenchmarkJob(_CallFakeBmreq(calls), tuning=False)
        self.assertEqual(self._run(job, calls), 1.0)
        self.assertNotIn("set_filename", [c[0] for c in calls])

    def test_call_applies_the_stamped_settings_after_the_flags(self):
        """Without the re-entry loop the worker silently tunes with defaults."""
        calls = []
        job = tunableop.BenchmarkJob(_CallFakeBmreq(calls, 3.0), tuning=True)
        settings = {"set_max_tuning_duration": 11, "set_rotating_buffer_size": 13}
        self.assertEqual(self._run(job, calls, settings=settings), 3.0)
        self.assertIn(("set_max_tuning_duration", 11), calls)
        self.assertIn(("set_rotating_buffer_size", 13), calls)
        # After the flags and before the benchmark: a setter applied afterwards
        # would not affect the run it was snapshotted for.
        self.assertLess(
            calls.index(("tuning_enable", True)),
            calls.index(("set_max_tuning_duration", 11)),
        )
        self.assertLess(
            calls.index(("set_rotating_buffer_size", 13)), calls.index(("benchmark",))
        )

    def test_call_enters_the_dynamic_dims_mask_around_the_benchmark(self):
        calls = []
        job = tunableop.BenchmarkJob(
            _CallFakeBmreq(calls, 2.5),
            tuning=True,
            dyn_dims_mask=(True, False, True, False),
        )
        self.assertEqual(self._run(job, calls), 2.5)
        kinds = [c[0] for c in calls]
        entered, exited = kinds.index("enter"), kinds.index("exit")
        self.assertEqual(calls[entered], ("enter", (True, False, True, False)))
        # Every benchmark must fall inside the scope, however many there are.
        self.assertTrue(entered < kinds.index("benchmark") < exited)

    def test_call_skips_the_mask_when_it_would_select_nothing(self):
        for name, mask in (("unset", None), ("all false", (False,) * 4)):
            with self.subTest(mask=name):
                calls = []
                job = tunableop.BenchmarkJob(
                    _CallFakeBmreq(calls), tuning=True, dyn_dims_mask=mask
                )
                self.assertEqual(self._run(job, calls), 1.0)
                self.assertNotIn("enter", [c[0] for c in calls])


class TunableopModeValidationTest(TestCase):
    """Invalid TORCHINDUCTOR_AUTOTUNE_TUNABLEOP_MODE values warn instead of silently
    disabling."""

    hw_classification = HardwareClassification.GENERIC

    def test_invalid_mode_warns_and_disables(self):
        tunableop._mode_warned = False
        try:
            with inductor_config.patch({"cuda.autotune_tunableop_mode": "ON"}):
                with self.assertLogs(tunableop.log, level="WARNING") as logs:
                    self.assertFalse(tunableop.is_tune_mode())
                    self.assertFalse(tunableop.is_reuse_mode())
                    self.assertFalse(tunableop.is_enabled())
            self.assertTrue(
                any("TORCHINDUCTOR_AUTOTUNE_TUNABLEOP_MODE" in m for m in logs.output)
            )
            # Echoes what was set, not the normalized form.
            self.assertTrue(any("'ON'" in m for m in logs.output))
            # Warns only once: the second invalid value logs nothing.
            with inductor_config.patch({"cuda.autotune_tunableop_mode": "bogus"}):
                with self.assertNoLogs(tunableop.log, level="WARNING"):
                    tunableop.is_enabled()
        finally:
            tunableop._mode_warned = False

    def test_mode_env_warnings(self):
        # Each kill switch leaves the mode with nothing to record or borrow;
        # TUNING=0 is harmless in reuse, which never tunes. Deliberately no
        # _env_warned reset between cases: one mode's warning must not silence
        # the other's, and a repeat must stay quiet rather than warn per GEMM.
        self.addCleanup(tunableop._env_warned.clear)
        tunableop._env_warned.clear()
        prefix = "PYTORCH_TUNABLEOP_"
        env = {k: v for k, v in os.environ.items() if not k.startswith(prefix)}
        for mode, var, value, warns in (
            ("tune", "PYTORCH_TUNABLEOP_TUNING", "0", True),
            ("tune", "PYTORCH_TUNABLEOP_RECORD_UNTUNED", "1", True),
            ("reuse", "PYTORCH_TUNABLEOP_RECORD_UNTUNED", "1", True),
            ("reuse", "PYTORCH_TUNABLEOP_NEAREST_MATCH_ENABLED", "0", True),
            ("reuse", "PYTORCH_TUNABLEOP_RECORD_BORROWED_ENABLED", "0", True),
            ("reuse", "PYTORCH_TUNABLEOP_TUNING", "0", False),
        ):
            with self.subTest(mode=mode, var=var):
                is_mode = getattr(tunableop, f"is_{mode}_mode")
                # Second pass must never warn: once per mode and variable.
                for first_pass in (True, False):
                    warn_now = warns and first_pass
                    expect = self.assertLogs if warn_now else self.assertNoLogs
                    with (
                        inductor_config.patch({"cuda.autotune_tunableop_mode": mode}),
                        mock.patch("torch.cuda.is_available", return_value=True),
                        mock.patch.dict(os.environ, {**env, var: value}, clear=True),
                        expect(tunableop.log, level="WARNING") as logs,
                    ):
                        self.assertTrue(is_mode())
                    if warn_now:
                        self.assertTrue(any(var in m for m in logs.output))

    def test_mode_is_case_normalized(self):
        # Without this "Tune" turns both passes off behind a one-time warning.
        tunableop._mode_warned = False
        self.addCleanup(setattr, tunableop, "_mode_warned", False)
        with (
            inductor_config.patch({"cuda.autotune_tunableop_mode": "Tune"}),
            mock.patch("torch.cuda.is_available", return_value=True),
            self.assertNoLogs(tunableop.log, level="WARNING"),
        ):
            self.assertTrue(tunableop.is_tune_mode())
            self.assertFalse(tunableop.is_reuse_mode())

    def test_valid_modes_do_not_warn(self):
        tunableop._mode_warned = False
        self.addCleanup(setattr, tunableop, "_mode_warned", False)
        for mode in ("off", "tune", "reuse"):
            with self.subTest(mode=mode):
                with inductor_config.patch({"cuda.autotune_tunableop_mode": mode}):
                    with self.assertNoLogs(tunableop.log, level="WARNING"):
                        tunableop.is_enabled()


class TunableOpRealWorkerTest(TestCase):
    """Runs jobs in a real TuningProcess.

    _FakeProcess cannot model get() killing the process or put() restarting it,
    so the restart claim has no coverage without this. The job and the stubs it
    needs live in autotune_process.py: the child has only torch on its path.
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
        return _TestTunableOpJob(
            tunableop.BenchmarkJob(_TestBenchmarkRequest(1.0, **kwargs), tuning=True)
        )

    def _submit(self, job, timeout):
        with (
            mock.patch("torch.cuda.current_device", return_value=0),
            mock.patch("torch.cuda.tunable.get_filename", return_value="t.csv"),
            # snapshot_settings reads L2 to tell "unset" from a real MB value.
            mock.patch(
                "torch.cuda.get_device_properties",
                return_value=mock.Mock(L2_cache_size=4 * 1024 * 1024),
            ),
        ):
            return tunableop.submit(job, timeout=timeout)

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_hang_then_next_submission_succeeds(self):
        outcome = self._submit(self._job(sleep=600.0), timeout=5.0)
        self.assertEqual(outcome.status, "timeout")
        self.assertIsNone(outcome.time)
        # submit() reaps the killed child rather than leaving it a zombie, so
        # the module-global worker is cleared and the next call builds a fresh
        # one. Asserted here because reaping has no other visible effect.
        self.assertIsNone(tunableop._worker)
        outcome = self._submit(self._job(), timeout=120.0)
        self.assertEqual(outcome.status, "ok")
        self.assertEqual(outcome.time, 1.0)

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_crash_then_next_submission_succeeds(self):
        outcome = self._submit(self._job(crash=True), timeout=120.0)
        self.assertEqual(outcome.status, "crash")
        self.assertIsNone(outcome.time)
        # Same teardown on the EOF path as on the timeout path.
        self.assertIsNone(tunableop._worker)
        outcome = self._submit(self._job(), timeout=120.0)
        self.assertEqual(outcome.status, "ok")
        self.assertEqual(outcome.time, 1.0)

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_job_raised_exception_through_the_real_pipe_is_an_error(self):
        # The fake test sets running by hand. Here TuningProcess.get decides:
        # it raises the child's exception after its own except clauses, leaving
        # running set, so the job's TimeoutError must not be filed as a pipe
        # timeout.
        outcome = self._submit(
            self._job(exc=TimeoutError("from the job")), timeout=120.0
        )
        self.assertEqual(outcome.status, "error")
        self.assertIsNone(outcome.time)
        outcome = self._submit(self._job(), timeout=120.0)
        self.assertEqual(outcome.status, "ok")
        self.assertEqual(outcome.time, 1.0)

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_job_with_settings_and_mask_survives_the_pipe(self):
        """A fully populated BenchmarkJob pickles and runs in the child.

        The fake-process tests stop at what submit() stamps; this is the only
        one recording what the child did with it -- the stamped setter calls and
        a populated dyn_dims_mask, which nothing else sends through a worker.
        """
        with tempfile.TemporaryDirectory(prefix="tunableop_pipe_") as d:
            record = os.path.join(d, "calls.json")
            wrapped = _TestTunableOpJob(
                tunableop.BenchmarkJob(
                    _TestBenchmarkRequest(1.0),
                    tuning=True,
                    nearest_match=True,
                    dyn_dims_mask=(True, False, True, False),
                ),
                record_path=record,
            )
            with mock.patch.multiple(
                "torch.cuda.tunable",
                get_max_tuning_duration=mock.Mock(return_value=11),
                get_max_tuning_iterations=mock.Mock(return_value=12),
                get_rotating_buffer_size=mock.Mock(return_value=13 * 1024 * 1024),
                get_cublaslt_requested_algo_count=mock.Mock(return_value=14),
            ):
                outcome = self._submit(wrapped, timeout=120.0)
            self.assertEqual(outcome.status, "ok")
            self.assertEqual(outcome.time, 1.0)
            with open(record) as f:
                calls = json.load(f)
        # Recorded in the child, so these assert what crossed the pipe.
        self.assertIn(
            ["set_filename", ["t.csv"], {"insert_device_ordinal": False}], calls
        )
        self.assertIn(["set_max_tuning_duration", [11], {}], calls)
        self.assertIn(["set_rotating_buffer_size", [13], {}], calls)
        mask = {"M": True, "N": False, "K": True, "BATCH": False}
        self.assertIn(["dynamic_dims_mask", [], mask], calls)
        self.assertIn("dynamic_dims_mask().__enter__", [c[0] for c in calls])


class TunableOpSettingsRoundTripTest(TestCase):
    """snapshot_settings against the real getters and setters, on any GPU."""

    hw_classification = HardwareClassification.CUDA

    @unittest.skipIf(not torch.cuda.is_available(), "needs a device")
    def test_rotating_buffer_size_round_trips_through_snapshot(self):
        # Drives the real getter and setter, which the mocked test cannot:
        # it chooses both sides of the comparison.
        tunable = torch.cuda.tunable
        previous = tunable.get_rotating_buffer_size()
        # Unset reads back as the L2 size, and any negative size restores it.
        l2 = torch.cuda.get_device_properties(torch.cuda.current_device()).L2_cache_size
        self.addCleanup(
            tunable.set_rotating_buffer_size,
            -1 if previous == l2 else previous // (1024 * 1024),
        )

        tunable.set_rotating_buffer_size(4)  # MB
        parent_bytes = tunable.get_rotating_buffer_size()
        self.assertEqual(parent_bytes, 4 * 1024 * 1024)

        stamped = tunableop.snapshot_settings()["set_rotating_buffer_size"]
        tunable.set_rotating_buffer_size(stamped)  # what the worker does
        self.assertEqual(tunable.get_rotating_buffer_size(), parent_bytes)


class TunableOpRealGpuWorkerTest(TestCase):
    """Real BenchmarkJob through a real worker on a GPU.

    Covers what no stubbed test can: real tuning on a device, and the worker
    appending to the file the parent's runtime reads. Not ROCm-gated -- the
    TunableOp GEMM dispatch and its signature are the same on cuBLASLt.
    """

    hw_classification = HardwareClassification.CUDA

    @unittest.skipIf(not torch.cuda.is_available(), "needs a device")
    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_real_job_tunes_and_appends_to_shared_file(self):
        from torch._inductor.autotune_process import _TestTunableMatmulRequest

        tunableop._worker = None
        tunableop._worker_key = None
        self.addCleanup(tunableop.shutdown_worker)
        self.addCleanup(setattr, tunableop, "_worker", None)
        self.addCleanup(setattr, tunableop, "_worker_key", None)

        was_enabled = torch.cuda.tunable.is_enabled()
        previous_filename = torch.cuda.tunable.get_filename()
        with tempfile.TemporaryDirectory(prefix="tunableop_worker_gpu_") as d:
            results_file = os.path.join(d, "results.csv")
            torch.cuda.tunable.set_filename(results_file, False)
            try:
                torch.cuda.tunable.enable(False)
                job = tunableop.BenchmarkJob(_TestTunableMatmulRequest(), tuning=True)
                outcome = tunableop.submit(job, timeout=600.0)
            finally:
                torch.cuda.tunable.enable(was_enabled)
                torch.cuda.tunable.set_filename(previous_filename, False)
            self.assertEqual(outcome.status, "ok")
            self.assertGreater(outcome.time, 0)
            with open(results_file) as f:
                content = f.read()
            self.assertIn("GemmTunableOp", content)


if __name__ == "__main__":
    run_tests()
