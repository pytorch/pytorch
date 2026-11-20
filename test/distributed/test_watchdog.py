# Owner(s): ["oncall: distributed"]

import asyncio
import concurrent.futures
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch.distributed._watchdog import (
    _CancelHandle,
    _get_watchdog,
    _StreamMonitor,
    _Watchdog,
    cpu_timeout,
    op_timeout,
    shutdown,
)
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, TestCase


def _on_loop(wd: _Watchdog, fn):
    fut: concurrent.futures.Future = concurrent.futures.Future()
    wd._loop.call_soon_threadsafe(lambda: fut.set_result(fn()))
    return fut.result(timeout=5.0)


class _QueryProbe:
    """Wraps a CUDA event; reports incomplete until released, signals each query."""

    def __init__(self, event: torch.cuda.Event) -> None:
        self.event = event
        self.queried = threading.Event()
        self.released = False

    def query(self) -> bool:
        self.event.query()
        self.queried.set()
        return self.released


class TestCancelHandle(TestCase):
    def test_cancel_idempotent(self) -> None:
        handle = _CancelHandle()
        handle.cancel()
        handle.cancel()
        self.assertTrue(handle.is_cancelled)

    def test_cancel_before_set_timer(self) -> None:
        handle = _CancelHandle()
        handle.cancel()
        mock_timer = MagicMock(spec=asyncio.TimerHandle)
        handle._set_timer_handle(mock_timer)
        mock_timer.cancel.assert_called_once()

    def test_cancel_after_set_timer(self) -> None:
        handle = _CancelHandle()
        mock_timer = MagicMock(spec=asyncio.TimerHandle)
        handle._set_timer_handle(mock_timer)
        handle.cancel()
        mock_timer.cancel.assert_called_once()
        self.assertTrue(handle.is_cancelled)

    def test_cancel_before_set_stream_monitor(self) -> None:
        handle = _CancelHandle()
        handle.cancel()
        monitor = _StreamMonitor(event=MagicMock(), deadline=0.0, callback=lambda: None)
        handle._set_stream_monitor(monitor)
        self.assertTrue(monitor.cancelled)

    def test_cancel_after_set_stream_monitor(self) -> None:
        handle = _CancelHandle()
        monitor = _StreamMonitor(event=MagicMock(), deadline=0.0, callback=lambda: None)
        handle._set_stream_monitor(monitor)
        handle.cancel()
        self.assertTrue(monitor.cancelled)
        self.assertTrue(handle.is_cancelled)

    def test_not_cancelled_initially(self) -> None:
        handle = _CancelHandle()
        self.assertFalse(handle.is_cancelled)


class TestWatchdog(TestCase):
    def setUp(self) -> None:
        super().setUp()
        shutdown()

    def tearDown(self) -> None:
        shutdown()
        super().tearDown()

    def test_cpu_timeout_fires(self) -> None:
        fired = threading.Event()
        cpu_timeout(0.1, fired.set)
        self.assertTrue(fired.wait(timeout=5.0), "callback did not fire")

    def test_cpu_timeout_cancelled(self) -> None:
        fired = threading.Event()
        handle = cpu_timeout(0.2, fired.set)
        handle.cancel()
        self.assertFalse(fired.wait(timeout=0.5), "callback fired despite cancel")

    def test_multiple_cpu_timeouts(self) -> None:
        results: list[int] = []
        lock = threading.Lock()
        events = [threading.Event() for _ in range(3)]

        for i, ev in enumerate(events):

            def cb(idx: int = i, e: threading.Event = ev) -> None:
                with lock:
                    results.append(idx)
                e.set()

            cpu_timeout(0.05 * (i + 1), cb)

        for ev in events:
            self.assertTrue(ev.wait(timeout=5.0))
        self.assertEqual(sorted(results), [0, 1, 2])

    def test_callback_exception_no_crash(self) -> None:
        def bad_callback() -> None:
            raise ValueError("intentional")

        second_fired = threading.Event()
        cpu_timeout(0.05, bad_callback)
        cpu_timeout(0.15, second_fired.set)
        self.assertTrue(
            second_fired.wait(timeout=5.0),
            "second callback should fire after first raised",
        )

    @patch("torch.distributed._watchdog.torch.cuda.Event")
    def test_stream_timeout_fires(self, MockEvent: MagicMock) -> None:
        mock_event = MagicMock()
        mock_event.query.return_value = False
        MockEvent.return_value = mock_event

        fired = threading.Event()

        with patch.dict("os.environ", {"TORCH_WATCHDOG_POLL_INTERVAL_SECS": "0.05"}):
            wd = _Watchdog()
            try:
                wd.stream_timeout(0.15, fired.set)
                self.assertTrue(fired.wait(timeout=5.0), "callback did not fire")
            finally:
                wd.shutdown()

    @patch("torch.distributed._watchdog.torch.cuda.Event")
    def test_stream_timeout_no_fire_when_complete(self, MockEvent: MagicMock) -> None:
        mock_event = MagicMock()
        mock_event.query.return_value = True
        MockEvent.return_value = mock_event

        fired = threading.Event()

        with patch.dict("os.environ", {"TORCH_WATCHDOG_POLL_INTERVAL_SECS": "0.05"}):
            wd = _Watchdog()
            try:
                wd.stream_timeout(0.5, fired.set)
                time.sleep(0.3)
                self.assertFalse(fired.is_set(), "callback fired despite completion")
            finally:
                wd.shutdown()

    @patch("torch.distributed._watchdog.torch.cuda.Event")
    def test_stream_query_error_fires_callback(self, MockEvent: MagicMock) -> None:
        bad_event = MagicMock()
        bad_event.query.side_effect = RuntimeError("CUDA error")
        MockEvent.return_value = bad_event

        completed = threading.Event()
        fired = threading.Event()
        wd = _Watchdog(poll_interval=0.01)
        try:
            wd.stream_complete(completed.set, event=bad_event)
            wd.stream_timeout(60.0, fired.set)
            self.assertTrue(fired.wait(timeout=5.0), "callback did not fire on error")
            self.assertEqual(_on_loop(wd, lambda: len(wd._stream_monitors)), 0)
            self.assertFalse(completed.is_set())
        finally:
            wd.shutdown()

    @unittest.skipIf(not TEST_CUDA, "requires CUDA")
    def test_stream_monitors_skipped_during_capture(self) -> None:
        fired = threading.Event()
        wd = _Watchdog(poll_interval=0.01)
        try:
            x = torch.zeros(1, device="cuda")
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                x.add_(1)
                timeout_handle = wd.stream_timeout(0.01, fired.set)
                complete_handle = wd.stream_complete(fired.set)
            self.assertTrue(timeout_handle.is_cancelled)
            self.assertTrue(complete_handle.is_cancelled)
            self.assertEqual(_on_loop(wd, lambda: len(wd._stream_monitors)), 0)

            for _ in range(3):
                g.replay()
            torch.cuda.synchronize()
            self.assertEqual(x, torch.full((1,), 3.0, device="cuda"))
            self.assertFalse(fired.is_set())
        finally:
            wd.shutdown()

    @unittest.skipIf(not TEST_CUDA, "requires CUDA")
    def test_poll_during_capture_keeps_capture_valid(self) -> None:
        wd = _Watchdog(poll_interval=0.01)
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            torch.cuda._sleep(1_000_000)
            event = torch.cuda.Event()
            event.record()
        probe = _QueryProbe(event)
        try:
            wd.stream_complete(lambda: None, event=probe)
            self.assertTrue(probe.queried.wait(timeout=5.0))

            x = torch.zeros(1, device="cuda")
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                x.add_(1)
                probe.queried.clear()
                self.assertTrue(probe.queried.wait(timeout=5.0), "no poll")

            g.replay()
            torch.cuda.synchronize()
            self.assertEqual(x, torch.ones(1, device="cuda"))
        finally:
            probe.released = True
            wd.shutdown()

    @patch("torch.distributed._watchdog.torch.cuda.Event")
    def test_op_timeout_cpu_fires_during_block(self, MockEvent: MagicMock) -> None:
        mock_event = MagicMock()
        MockEvent.return_value = mock_event

        fired = threading.Event()

        with patch.dict("os.environ", {"TORCH_WATCHDOG_POLL_INTERVAL_SECS": "0.05"}):
            shutdown()
            with op_timeout(0.1, fired.set):
                fired.wait(timeout=5.0)

            self.assertTrue(fired.is_set(), "cpu timeout should fire during block")

    @patch("torch.distributed._watchdog.torch.cuda.Event")
    def test_op_timeout_stream_fires_after_block(self, MockEvent: MagicMock) -> None:
        mock_event = MagicMock()
        mock_event.query.return_value = False
        MockEvent.return_value = mock_event

        fired = threading.Event()

        with patch.dict("os.environ", {"TORCH_WATCHDOG_POLL_INTERVAL_SECS": "0.05"}):
            shutdown()
            with op_timeout(0.15, fired.set):
                pass

            self.assertTrue(
                fired.wait(timeout=5.0), "stream timeout should fire after block"
            )

    def test_health_watchdog_detects_stuck(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "TORCH_WATCHDOG_HEALTH_INTERVAL_SECS": "0.1",
                "TORCH_WATCHDOG_STUCK_ACTION": "log",
            },
        ):
            wd = _Watchdog()
            released = threading.Event()
            try:
                blocked = threading.Event()

                def block_loop() -> None:
                    blocked.set()
                    released.wait(timeout=5.0)

                wd._loop.call_soon_threadsafe(block_loop)
                blocked.wait(timeout=2.0)

                with self.assertLogs(
                    "torch.distributed._watchdog", level="WARNING"
                ) as cm:
                    time.sleep(0.4)

                self.assertTrue(
                    any("stuck" in msg.lower() for msg in cm.output),
                    f"Expected 'stuck' warning, got: {cm.output}",
                )
            finally:
                released.set()
                wd.shutdown()

    def test_singleton_lifecycle(self) -> None:
        wd1 = _get_watchdog()
        wd2 = _get_watchdog()
        self.assertIs(wd1, wd2)

        shutdown()

        wd3 = _get_watchdog()
        self.assertIsNot(wd1, wd3)

    @patch("torch.distributed._watchdog.torch.cuda.Event")
    def test_del_queue_populated(self, MockEvent: MagicMock) -> None:
        mock_event = MagicMock()
        mock_event.query.return_value = True
        MockEvent.return_value = mock_event

        with patch.dict("os.environ", {"TORCH_WATCHDOG_POLL_INTERVAL_SECS": "0.05"}):
            wd = _Watchdog()
            try:
                wd.stream_timeout(10.0, lambda: None)
                time.sleep(0.2)
                count = wd._drain_del_queue()
                self.assertGreaterEqual(count, 1, "event should be on del queue")
            finally:
                wd.shutdown()

    @patch("torch.distributed._watchdog.torch.cuda.Event")
    def test_stream_timeout_cancelled(self, MockEvent: MagicMock) -> None:
        mock_event = MagicMock()
        mock_event.query.return_value = False
        MockEvent.return_value = mock_event

        fired = threading.Event()

        with patch.dict("os.environ", {"TORCH_WATCHDOG_POLL_INTERVAL_SECS": "0.05"}):
            wd = _Watchdog()
            try:
                handle = wd.stream_timeout(0.15, fired.set)
                handle.cancel()
                time.sleep(0.4)
                self.assertFalse(fired.is_set(), "callback fired despite cancel")
            finally:
                wd.shutdown()


if __name__ == "__main__":
    run_tests()
