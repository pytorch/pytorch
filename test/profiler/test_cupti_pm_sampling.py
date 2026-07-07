# Owner(s): ["oncall: profiler"]
"""Tests for CUPTI PM sampling (``torch.profiler._cupti.pm_sampling``).

Covers the PmSampler sizing (decode-image = process-wide look-back / interval, capped), metric
selection + non-empty enforcement, and discovery -- exercised through real CUPTI PM sampling
(needs a perfmon-capable GPU; the tests skip otherwise).
"""

from __future__ import annotations

import time
import unittest

import torch
from torch.testing._internal.common_cuda import (
    TEST_CUDA,
    TEST_CUPTI as TEST_CUPTI_PYTHON,
    TEST_CUPTI_V13_3,
)
from torch.testing._internal.common_utils import run_tests, TestCase


# pm_sampling imports the cupti module at load, so only import it when cupti-python is present;
# every test that uses these is gated on TEST_CUPTI_PM_SAMPLING (which implies TEST_CUPTI_PYTHON).
if TEST_CUPTI_PYTHON:
    from torch.profiler._cupti.pm_sampling import (
        _DEFAULT_LOOKBACK_WINDOW_NS,
        _DEFAULT_SAMPLING_INTERVAL_NS,
        PmSampler,
        supported_metrics,
    )


# Metrics are per-consumer (PmSampler.add_consumer), with no built-in default, so tests pass an
# explicit set (single-pass on the default 4-counter chips).
_TEST_METRICS = (
    "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "nvlrx__bytes.avg.pct_of_peak_sustained_elapsed",
    "nvltx__bytes.avg.pct_of_peak_sustained_elapsed",
)


# Needs CUDA + libcupti >= 13.3 (cupti-python ships with PyTorch) AND the nvperf profiler-host
# libraries (libnvperf_host/target): supported_metrics() returns empty (and logs) when they can't
# load, so it doubles as the availability probe (@cache, so it runs once). Whether the HW can
# actually engage a live session (perfmon access) is only known at start(), so the session tests
# still skip at runtime when none comes up.
TEST_CUPTI_PM_SAMPLING = TEST_CUDA and TEST_CUPTI_V13_3 and bool(supported_metrics())


@unittest.skipIf(
    not TEST_CUPTI_PM_SAMPLING,
    "requires cupti pm_sampling + the nvperf libraries + a capable CUDA GPU",
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
        # Real session: add_consumer enables PM sampling on the current device via CUPTI, we drive
        # some GPU work, and poll(handle) returns a frame of the ring's samples (raw HW-ns).
        sampler = PmSampler(torch.cuda.current_device())
        handle = sampler.add_consumer(list(metrics))
        self.addCleanup(handle.detach)
        if sampler._col is None:
            self.skipTest("PM sampling could not start on this GPU")
        self._run_gpu_work()
        frame = handle.poll()
        return sampler, ([frame] if frame is not None else [])

    def test_max_samples_from_process_config(self):
        # max_samples = look-back / interval (the process-wide default here, since no
        # configure() call); accepted by a real start()/decode() (which sizes the ring
        # from it via get_counter_data_size).
        sampler, _ = self._collect()
        expected = max(1, _DEFAULT_LOOKBACK_WINDOW_NS // _DEFAULT_SAMPLING_INTERVAL_NS)
        self.assertEqual(sampler._max_samples, expected)

    def test_configure_and_get_config_first_come_first_serve(self):
        # Interval/look-back are a process-wide config (no env var): configure() sets them,
        # get_config() reports them, and it's first-come-first-serve -- a second call is ignored.
        # No CUDA needed; save/restore the class state so other tests are unperturbed.
        saved = PmSampler.get_config()

        def _restore():
            PmSampler._sampling_interval_ns = saved["sampling_interval_ns"]
            PmSampler._lookback_window_ns = saved["lookback_window_ns"]
            PmSampler._configured = saved["configured"]

        self.addCleanup(_restore)
        PmSampler._sampling_interval_ns = _DEFAULT_SAMPLING_INTERVAL_NS
        PmSampler._lookback_window_ns = _DEFAULT_LOOKBACK_WINDOW_NS
        PmSampler._configured = False
        self.assertEqual(
            PmSampler.get_config(),
            {
                "sampling_interval_ns": _DEFAULT_SAMPLING_INTERVAL_NS,
                "lookback_window_ns": _DEFAULT_LOOKBACK_WINDOW_NS,
                "configured": False,
            },
        )
        PmSampler.configure(sampling_interval_ms=0.5, lookback_window_ms=2000)
        cfg = PmSampler.get_config()
        self.assertEqual(cfg["sampling_interval_ns"], 500_000)
        self.assertEqual(cfg["lookback_window_ns"], 2_000_000_000)
        self.assertTrue(cfg["configured"])
        # first-come-first-serve: a second configure() is ignored (and warns).
        with self.assertLogs("torch.profiler._cupti.pm_sampling", level="WARNING"):
            PmSampler.configure(sampling_interval_ms=0.1)
        self.assertEqual(PmSampler.get_config()["sampling_interval_ns"], 500_000)

    def test_empty_metrics_rejected(self):
        # A consumer must bring a non-empty metric set; add_consumer rejects an empty one.
        sampler = PmSampler(torch.cuda.current_device())
        with self.assertRaises(ValueError):
            sampler.add_consumer(())
        self.assertIsNone(sampler._col)

    def test_multipass_metrics_rejected(self):
        # PM sampling is single-pass only; a metric that needs multiple passes (sm__throughput.*
        # needs ~17) is rejected by add_consumer -> ValueError, no session, running state untouched.
        # The class gate guarantees the nvperf profiler host is loadable, so the pass count is known.
        sampler = PmSampler(torch.cuda.current_device())
        with self.assertRaises(ValueError) as cm:
            sampler.add_consumer(["sm__throughput.avg.pct_of_peak_sustained_elapsed"])
        self.assertIn("pass", str(cm.exception).lower())
        self.assertIsNone(sampler._col)

    def test_returns_per_device_singleton(self):
        # PmSampler(device) returns the one instance for that device -- repeated construction and the
        # no-arg (current device) form resolve to the same object, so all consumers share one session.
        dev = torch.cuda.current_device()
        self.assertIs(PmSampler(dev), PmSampler(dev))
        self.assertIs(PmSampler(dev), PmSampler())

    def test_profiler_refcount_balanced(self):
        # The process-global profiler deinit is refcounted (not by CUPTI): a full add/remove cycle
        # returns the live-collector count to where it started, so the profiler is torn down exactly
        # once and can be re-initialized on the next session.
        before = PmSampler._active
        sampler = PmSampler(torch.cuda.current_device())
        handle = sampler.add_consumer(list(_TEST_METRICS))
        if sampler._col is None:
            handle.detach()
            self.skipTest("PM sampling could not start on this GPU")
        self.assertEqual(PmSampler._active, before + 1)
        handle.detach()
        self.assertEqual(PmSampler._active, before)

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
        self.assertLessEqual(int(ts.max() - ts.min()), _DEFAULT_LOOKBACK_WINDOW_NS)

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
        sampler = PmSampler(torch.cuda.current_device())
        with self.assertLogs(
            "torch.profiler._cupti.pm_sampling", level="WARNING"
        ) as cm:
            handle = sampler.add_consumer(["not__a_real_metric.avg"])
            self.addCleanup(handle.detach)
        self.assertTrue(any("not reported by this chip" in m for m in cm.output))

    def test_multiple_consumers_share_session(self):
        # Two consumers on one device share the single session: it samples the union of their
        # metrics, and each consumer's polled frame carries only its own metric columns.
        a_metrics, b_metrics = [_TEST_METRICS[0]], list(_TEST_METRICS[1:3])
        sampler = PmSampler(torch.cuda.current_device())
        a = sampler.add_consumer(a_metrics)
        self.addCleanup(a.detach)
        b = sampler.add_consumer(b_metrics)
        self.addCleanup(b.detach)
        if sampler._col is None:
            self.skipTest("PM sampling could not start on this GPU")
        # The session samples the union (dedup, order-preserving).
        self.assertEqual(sampler._metric_names, a_metrics + b_metrics)
        self._run_gpu_work()
        fa = a.poll()
        fb = b.poll()
        if fa is None or fb is None:
            self.skipTest("no PM samples produced on this GPU")
        self.assertIn(a_metrics[0], fa)  # only consumer A's metric
        self.assertNotIn(b_metrics[0], fa)
        for name in b_metrics:  # only consumer B's metrics
            self.assertIn(name, fb)
        self.assertNotIn(a_metrics[0], fb)

    def test_metric_union_reconfigure(self):
        # Adding/removing a consumer with a distinct metric changes the union and rebuilds the
        # session (a live in-place reconfigure segfaults); sampling must survive each grow/shrink.
        m0, m1 = [_TEST_METRICS[0]], [_TEST_METRICS[1]]
        sampler = PmSampler(torch.cuda.current_device())
        a = sampler.add_consumer(m0)
        self.addCleanup(a.detach)
        if sampler._col is None:
            self.skipTest("PM sampling could not start on this GPU")
        self.assertEqual(sampler._metric_names, m0)
        b = sampler.add_consumer(m1)  # union grows -> rebuild
        self.addCleanup(b.detach)
        self.assertEqual(sampler._metric_names, m0 + m1)
        self._run_gpu_work()
        fa = a.poll()
        if fa is None:
            self.skipTest("no PM samples produced on this GPU")
        self.assertIn(m0[0], fa)  # sampling works after the grow-rebuild
        b.detach()  # union shrinks -> rebuild
        self.assertEqual(sampler._metric_names, m0)
        self._run_gpu_work()
        fa = a.poll()
        self.assertTrue(
            fa is not None and m0[0] in fa
        )  # works after the shrink-rebuild

    def test_dropped_handle_detaches(self):
        # Safety net: dropping a handle without detach() still unregisters it -- the sampler holds
        # only the _Consumer record, not the handle, so the handle is collectable and __del__ fires.
        import gc

        sampler = PmSampler(torch.cuda.current_device())
        handle = sampler.add_consumer(list(_TEST_METRICS))
        if sampler._col is None:
            handle.detach()
            self.skipTest("PM sampling could not start on this GPU")
        self.assertEqual(len(sampler._consumers), 1)
        handle = None  # drop the only reference
        gc.collect()
        self.assertEqual(len(sampler._consumers), 0)  # __del__ detached it

    def test_retention_only_for_lagging_consumer(self):
        # decode drains, so the sampler retains samples for a consumer that hasn't polled yet, and
        # keeps nothing for a lone, caught-up consumer (GC'd by the min-cursor watermark).
        sampler = PmSampler(torch.cuda.current_device())
        a = sampler.add_consumer(list(_TEST_METRICS[:1]))
        self.addCleanup(a.detach)
        if sampler._col is None:
            self.skipTest("PM sampling could not start on this GPU")
        # Lone consumer: after it polls, nothing is retained (it consumed everything it drained).
        self._run_gpu_work()
        if a.poll() is None:
            self.skipTest("no PM samples produced on this GPU")
        self.assertEqual(sampler._retained_ts.size, 0)
        # Add a second consumer (same metric -> no reconfigure); a poll by A must now retain A's
        # drained samples for B, which lags -- and B gets them from the buffer (A already drained
        # the HW ring).
        b = sampler.add_consumer(list(_TEST_METRICS[:1]))
        self.addCleanup(b.detach)
        self._run_gpu_work()
        a.poll()
        self.assertGreater(sampler._retained_ts.size, 0)
        fb = b.poll()
        self.assertTrue(fb is not None and fb["start_ns"].size > 0)


if __name__ == "__main__":
    run_tests()
