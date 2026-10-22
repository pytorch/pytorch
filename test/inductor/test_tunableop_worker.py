# Owner(s): ["module: inductor"]

from unittest import mock

from torch._inductor import tunableop
from torch.testing._internal.common_utils import run_tests, TestCase


class _FakeProcess:
    """Stands in for TuningProcess: records how it was constructed."""

    def __init__(self, device=None):
        self.device = device
        self.sent = []
        self.result = 1.0
        self.raises = None
        self.shutdown_calls = 0

    def put(self, job):
        self.sent.append(job)

    def get(self, timeout=None):
        if self.raises is not None:
            raise self.raises
        return self.result

    def shutdown(self, wait=True):
        self.shutdown_calls += 1


class TunableOpWorkerTest(TestCase):
    """The worker is a subprocess holding process-global TunableOp state, so the
    things worth pinning are which GPU it lands on and when it gets replaced.
    Neither needs a GPU to check, and a GPU-only test would not run in CI."""

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
        self.assertEqual(first.shutdown_calls, 1)

    def test_worker_restarts_when_the_results_file_changes(self):
        # The worker reads the results file once at startup, so a second compile
        # pointing somewhere else must not keep appending to the first file.
        self._submit(filename="a.csv")
        first = self.created[0]
        self._submit(filename="b.csv")
        self.assertEqual(len(self.created), 2)
        self.assertEqual(first.shutdown_calls, 1)

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


if __name__ == "__main__":
    run_tests()
