# Owner(s): ["oncall: profiler"]
"""Tests for CUPTI PM sampling (``torch.profiler._cupti.pm_sampling``).

Covers the PmSampler sizing (decode-image = process-wide look-back / interval, capped), metric
selection + non-empty enforcement, and discovery -- exercised through real CUPTI PM sampling
(needs a perfmon-capable GPU; the tests skip otherwise).
"""

import time
import unittest

import torch
from torch.profiler._cupti.pm_sampling import (
    _DEFAULT_WINDOW_NS,
    _MAX_SAMPLES_CAP,
    _SAMPLING_INTERVAL_NS,
    PmSampler,
    supported_metrics,
)
from torch.testing._internal.common_utils import run_tests, TEST_WITH_ROCM, TestCase
from torch.utils._import_utils import _check_module_exists


# Metrics are per-consumer (PmSampler.add_consumer), with no built-in default, so tests pass an
# explicit set (single-pass on the default 4-counter chips).
_TEST_METRICS = (
    "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "nvlrx__bytes.avg.pct_of_peak_sustained_elapsed",
    "nvltx__bytes.avg.pct_of_peak_sustained_elapsed",
)


TEST_CUDA = torch.cuda.is_available()
# cupti-python is pip-installable on ROCm hosts too, but CUPTI itself is a no-op there.
TEST_CUPTI_PYTHON = _check_module_exists("cupti") and not TEST_WITH_ROCM


def _cupti_version() -> int:
    if not TEST_CUPTI_PYTHON:
        return 0
    try:
        from torch.profiler._cupti.cupti_python import pylibcupti

        return pylibcupti().get_version()
    except Exception:
        return 0


# PM sampling goes through the CUPTI profiler API; gate it at the same >= 13.3 as the monitor.
TEST_CUPTI_V13_3 = TEST_CUPTI_PYTHON and _cupti_version() >= 130300


def _pm_sampling_available() -> bool:
    # Needs CUDA, libcupti >= 13.3, and the cupti.pm_sampling module. Whether the HW can actually
    # engage (perfmon access) is only known at start(), so tests still skip at runtime when no
    # session comes up.
    if not (TEST_CUDA and TEST_CUPTI_V13_3):
        return False
    from torch.profiler._cupti.pm_sampling import is_available

    return is_available()


TEST_CUPTI_PM_SAMPLING = _pm_sampling_available()


@unittest.skipIf(
    not TEST_CUPTI_PM_SAMPLING, "requires cupti pm_sampling + a capable CUDA GPU"
)
class TestPmSamplingWindowSizing(TestCase):
    """PM-sampling window sizing exercised through real CUPTI PM sampling: a PmSampler configured
    with a window + interval samples live GPU work, and the decoded frames confirm the sizing --
    max_samples = window // interval flows through configure()/decode(), each frame carries a column
    per metric, HW timestamps are monotonic, and the decoded span stays within the requested window.
    Skips at runtime when PM sampling cannot engage on the host (needs perfmon-capable HW)."""

    def _run_gpu_work(self, seconds: float = 0.2) -> None:
        a = torch.randn(512, 512, device="cuda")
        deadline = time.time() + seconds
        while time.time() < deadline:
            a = torch.relu(a @ a)
        torch.cuda.synchronize()

    def _collect(self, metrics=_TEST_METRICS) -> tuple[PmSampler, list]:
        # Real session: the first add_consumer enables PM sampling on the current device via CUPTI,
        # we drive some GPU work, and poll() decodes the ring into `frames` (raw HW-ns timestamps).
        frames: list = []
        sampler = PmSampler.for_device(torch.cuda.current_device())
        handle = sampler.add_consumer(list(metrics), frames.append)
        self.addCleanup(sampler.remove_consumer, handle)
        if sampler._col is None:
            self.skipTest("PM sampling could not start on this GPU")
        self._run_gpu_work()
        sampler.poll()
        return sampler, frames

    def test_max_samples_from_process_config(self):
        # max_samples = look-back / interval (process-wide env), clamped to the cap; accepted by a
        # real configure()/decode() (start sizes the ring from it via get_counter_data_size).
        sampler, _ = self._collect()
        expected = min(_DEFAULT_WINDOW_NS // _SAMPLING_INTERVAL_NS, _MAX_SAMPLES_CAP)
        self.assertEqual(sampler._max_samples, expected)

    def test_empty_metrics_rejected(self):
        # A consumer must bring a non-empty metric set; add_consumer rejects an empty one.
        sampler = PmSampler.for_device(torch.cuda.current_device())
        with self.assertRaises(ValueError):
            sampler.add_consumer((), lambda frame: None)
        self.assertIsNone(sampler._col)

    def test_multipass_metrics_rejected(self):
        # PM sampling is single-pass only; a metric that needs multiple passes (sm__throughput.*
        # needs ~8) is rejected by add_consumer -> ValueError, no session, running state untouched.
        sampler = PmSampler.for_device(torch.cuda.current_device())
        with self.assertRaises(ValueError) as cm:
            sampler.add_consumer(
                ["sm__throughput.avg.pct_of_peak_sustained_elapsed"],
                lambda frame: None,
            )
        self.assertIn("pass", str(cm.exception).lower())
        self.assertIsNone(sampler._col)

    def test_suggested_poll_interval_under_ring_span(self):
        # The recommended poll cadence drains a little before the ring fills (one window span minus a
        # buffer), so periodic polls are productive without hitting the decode overflow cap.
        sampler = PmSampler.for_device(torch.cuda.current_device())
        span = sampler._max_samples * sampler._sampling_interval_ns
        self.assertGreater(sampler.suggested_poll_interval_ns, 0)
        self.assertLess(sampler.suggested_poll_interval_ns, span)

    def test_decoded_frames_have_per_metric_columns(self):
        # Each sampled metric is a value column keyed by its metric name (self-describing frame).
        _, frames = self._collect()
        if not frames:
            self.skipTest("no PM samples produced on this GPU")
        for f in frames:
            self.assertIn("start_ns", f)
            self.assertIn("device_id", f)
            for name in _TEST_METRICS:
                self.assertIn(name, f)

    def test_decoded_span_within_lookback(self):
        import numpy as np

        _, frames = self._collect()
        ts = (
            np.concatenate([f["start_ns"] for f in frames])
            if frames
            else np.empty(0, dtype=np.int64)
        )
        if ts.size == 0:
            self.skipTest("no PM samples produced on this GPU")
        self.assertTrue((np.diff(np.sort(ts)) >= 0).all())  # monotonic HW timestamps
        self.assertLessEqual(int(ts.max() - ts.min()), _DEFAULT_WINDOW_NS)

    def test_select_metrics_by_name(self):
        # A caller passes metric name strings; each selected metric becomes its own frame column.
        chosen = list(_TEST_METRICS[:2])
        sampler, frames = self._collect(metrics=chosen)
        self.assertEqual(sampler._metric_names, chosen)
        if not frames:
            self.skipTest("no PM samples produced on this GPU")
        for f in frames:
            for name in chosen:
                self.assertIn(name, f)
            self.assertNotIn(_TEST_METRICS[2], f)  # unselected

    def test_supported_metrics_lists_chip_counters(self):
        metrics = supported_metrics()
        if not metrics:
            self.skipTest("profiler host could not enumerate metrics on this chip")
        self.assertTrue(all(isinstance(m, str) for m in metrics))
        base = {m.split(".", 1)[0] for m in metrics}
        self.assertIn(
            "sm__cycles_active", base
        )  # our default SM counter is in the menu

    def test_unknown_metric_warns(self):
        # A metric the chip does not report is warned about (not silently accepted).
        supported = supported_metrics()
        if not supported:
            self.skipTest("profiler host could not enumerate metrics on this chip")
        sampler = PmSampler.for_device(torch.cuda.current_device())
        with self.assertLogs(
            "torch.profiler._cupti.pm_sampling", level="WARNING"
        ) as cm:
            handle = sampler.add_consumer(
                ["not__a_real_metric.avg"], lambda frame: None
            )
            self.addCleanup(sampler.remove_consumer, handle)
        self.assertTrue(any("not reported by this chip" in m for m in cm.output))

    def test_multiple_consumers_share_session(self):
        # Two consumers on one device share the single session: it samples the union of their
        # metrics, and each consumer's frames carry only its own metric columns.
        a_metrics, b_metrics = [_TEST_METRICS[0]], list(_TEST_METRICS[1:3])
        a_frames: list = []
        b_frames: list = []
        sampler = PmSampler.for_device(torch.cuda.current_device())
        a = sampler.add_consumer(a_metrics, a_frames.append)
        self.addCleanup(sampler.remove_consumer, a)
        b = sampler.add_consumer(b_metrics, b_frames.append)
        self.addCleanup(sampler.remove_consumer, b)
        if sampler._col is None:
            self.skipTest("PM sampling could not start on this GPU")
        # The session samples the union (dedup, order-preserving).
        self.assertEqual(sampler._metric_names, a_metrics + b_metrics)
        self._run_gpu_work()
        sampler.poll()
        if not a_frames or not b_frames:
            self.skipTest("no PM samples produced on this GPU")
        for f in a_frames:  # only consumer A's metric
            self.assertIn(a_metrics[0], f)
            self.assertNotIn(b_metrics[0], f)
        for f in b_frames:  # only consumer B's metrics
            for name in b_metrics:
                self.assertIn(name, f)
            self.assertNotIn(a_metrics[0], f)


if __name__ == "__main__":
    run_tests()
