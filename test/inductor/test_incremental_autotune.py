# Owner(s): ["module: inductor"]

import gc
import queue
import threading
import time
from unittest.mock import MagicMock, patch

from torch._inductor.runtime.incremental._launcher import Launcher
from torch._inductor.runtime.incremental.config import TimingAggregation
from torch._inductor.test_case import run_tests, TestCase


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _make_launcher(name: str = "launcher") -> Launcher:
    fn = MagicMock()
    fn.return_value = f"result_{name}"
    launcher = Launcher(fn=fn, metadata={"config": f"config:{name}"})
    # Production keeps fn alive via CachingAutotuner.launchers; tests stash
    # it on the launcher so the weakref doesn't die immediately.
    launcher._fn_keepalive = fn  # pyrefly: ignore
    return launcher


class LauncherTest(TestCase):
    def test_timing_empty(self):
        launcher = _make_launcher()
        self.assertEqual(launcher.timing, float("inf"))

    def test_call_delegates_to_fn(self):
        launcher = _make_launcher("test")
        result = launcher(1, 2, stream=0)
        self.assertEqual(result, "result_test")
        launcher._fn_keepalive.assert_called_once_with(1, 2, stream=0)

    def test_call_raises_when_fn_garbage_collected(self):
        """fn is held weakly; once the only strong ref drops, __call__ asserts."""

        class _Callable:
            def __call__(self) -> str:
                return "ok"

        fn = _Callable()
        launcher = Launcher(fn=fn)
        del fn
        gc.collect()
        with self.assertRaises(AssertionError):
            launcher(stream=0)

    def test_set_fn_swaps_and_resets_warm(self):
        original = MagicMock(return_value="orig")
        launcher = Launcher(fn=original, metadata={"config": "c"})
        launcher._fn_keepalive = original  # pyrefly: ignore
        launcher(stream=0)
        self.assertTrue(launcher.is_warm)

        replacement = MagicMock(return_value="new")
        launcher.set_fn(replacement)
        launcher._fn_keepalive = replacement  # pyrefly: ignore
        self.assertFalse(launcher.is_warm)
        self.assertEqual(launcher(stream=0), "new")
        self.assertTrue(launcher.is_warm)

    def test_set_fn_preserves_timings(self):
        """set_fn assumes fn is interchangeable, so timings must not be cleared."""
        launcher = _make_launcher()
        launcher.add_timing(1.0)
        launcher.add_timing(2.0)
        new_fn = MagicMock(return_value="new")
        launcher.set_fn(new_fn)
        launcher._fn_keepalive = new_fn  # pyrefly: ignore
        self.assertEqual(launcher.num_available_timings, 2)
        self.assertAlmostEqual(launcher.timing, 1.5)

    def test_first_call_sets_warm(self):
        launcher = _make_launcher()
        self.assertFalse(launcher.is_warm)
        launcher(stream=0)
        self.assertTrue(launcher.is_warm)

    def test_metadata(self):
        launcher = Launcher(fn=lambda: None, metadata={"key": "value"})
        self.assertEqual(launcher.metadata["key"], "value")

    def test_metadata_default_is_empty(self):
        launcher = Launcher(fn=lambda: None)
        self.assertEqual(launcher.metadata, {})

    def test_untimed_call_does_not_track_timing(self):
        launcher = _make_launcher()
        launcher(stream=0)
        launcher(stream=0)
        self.assertEqual(launcher.num_in_flight_timings, 0)
        self.assertEqual(launcher.num_available_timings, 0)
        self.assertEqual(launcher.num_total_timings, 0)

    def test_timed_call_increments_in_flight_and_submits_event(self):
        launcher = _make_launcher()
        start_event = MagicMock()
        end_event = MagicMock()
        events = iter([start_event, end_event])
        with (
            patch("torch.cuda.Event", side_effect=lambda **_: next(events)),
            patch(
                "torch._inductor.runtime.incremental._launcher.submit_event"
            ) as mock_submit,
        ):
            launcher(stream=0, timed=True)
            self.assertEqual(launcher.num_in_flight_timings, 1)
            mock_submit.assert_called_once_with(
                launcher.add_timing, start_event, end_event
            )

    def test_add_timing_appends_to_available(self):
        launcher = _make_launcher()
        with launcher._lock:
            launcher._num_timings_in_flight = 2
        launcher.add_timing(1.0)
        self.assertEqual(launcher.num_in_flight_timings, 1)
        self.assertEqual(launcher.num_available_timings, 1)
        launcher.add_timing(2.0)
        self.assertEqual(launcher.num_in_flight_timings, 0)
        self.assertEqual(launcher.num_available_timings, 2)

    def test_num_total_timings_sums_in_flight_and_available(self):
        launcher = _make_launcher()
        with launcher._lock:
            launcher._num_timings_in_flight = 3
        self.assertEqual(launcher.num_total_timings, 3)
        launcher.add_timing(1.0)
        self.assertEqual(launcher.num_total_timings, 3)

    def test_concurrent_add_timing_is_thread_safe(self):
        """add_timing is called from the resolver thread; the lock must
        protect _timings and _num_timings_in_flight against the main
        thread's concurrent reads/writes.
        """
        launcher = _make_launcher()
        n_threads = 4
        per_thread = 250
        total = n_threads * per_thread
        with launcher._lock:
            launcher._num_timings_in_flight = total

        def worker() -> None:
            for _ in range(per_thread):
                launcher.add_timing(1.0)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(launcher.num_available_timings, total)
        self.assertEqual(launcher.num_in_flight_timings, 0)

    def test_timing_median_odd(self):
        launcher = _make_launcher()
        for v in [5.0, 1.0, 3.0]:
            launcher.add_timing(v)
        self.assertAlmostEqual(launcher.timing, 3.0)

    def test_timing_median_even(self):
        launcher = _make_launcher()
        for v in [4.0, 2.0]:
            launcher.add_timing(v)
        self.assertAlmostEqual(launcher.timing, 3.0)

    def test_timing_median_ignores_outliers(self):
        launcher = _make_launcher()
        for v in [1.0, 2.0, 3.0, 4.0, 100.0]:
            launcher.add_timing(v)
        self.assertAlmostEqual(launcher.timing, 3.0)

    def test_timing_uses_mean_when_configured(self):
        launcher = _make_launcher()
        for v in [1.0, 2.0, 3.0, 4.0, 100.0]:
            launcher.add_timing(v)
        with patch(
            "torch._inductor.runtime.incremental._launcher.timing_aggregation",
            TimingAggregation.MEAN,
        ):
            self.assertAlmostEqual(launcher.timing, 22.0)


class ResolverTest(TestCase):
    def setUp(self):
        from torch._inductor.runtime.incremental import _resolver

        self._resolver = _resolver
        # Short idle timeout so the daemon exits quickly between tests.
        self._timeout_patcher = patch.object(_resolver, "_IDLE_TIMEOUT_S", 0.05)
        self._timeout_patcher.start()
        _wait_until(self._daemon_stopped, timeout=10.0)

    def tearDown(self):
        while not self._resolver._global_event_queue.empty():
            try:
                self._resolver._global_event_queue.get_nowait()
            except queue.Empty:
                break
        _wait_until(self._daemon_stopped, timeout=10.0)
        self._timeout_patcher.stop()

    def _daemon_stopped(self) -> bool:
        with self._resolver._global_resolver_lock:
            return self._resolver._global_resolver_thread is None

    def test_submit_event_puts_then_ensures_daemon(self):
        """Put-then-ensure ordering: item is visible when daemon checks empty()."""
        with (
            patch.object(self._resolver, "_ensure_daemon") as mock_ensure,
            patch.object(self._resolver._global_event_queue, "put") as mock_put,
        ):
            call_order = []
            mock_put.side_effect = lambda *a, **k: call_order.append("put")
            mock_ensure.side_effect = lambda: call_order.append("ensure")

            self._resolver.submit_event(MagicMock(), MagicMock(), MagicMock())

        self.assertEqual(call_order, ["put", "ensure"])

    def test_daemon_processes_event_and_invokes_callback(self):
        """Submitted events are synchronized and the callback receives elapsed_ms."""
        done = threading.Event()
        received: list[float] = []

        def callback(elapsed_ms: float) -> None:
            received.append(elapsed_ms)
            done.set()

        start_event = MagicMock()
        end_event = MagicMock()
        start_event.elapsed_time.return_value = 42.0

        self._resolver.submit_event(callback, start_event, end_event)

        self.assertTrue(done.wait(timeout=5.0))
        self.assertEqual(received, [42.0])
        end_event.synchronize.assert_called_once()
        start_event.elapsed_time.assert_called_once_with(end_event)

    def test_daemon_idles_out_after_processing(self):
        """After processing all events, the daemon exits and clears the global slot."""
        done = threading.Event()
        start_event = MagicMock()
        start_event.elapsed_time.return_value = 1.0
        self._resolver.submit_event(lambda _: done.set(), start_event, MagicMock())
        self.assertTrue(done.wait(timeout=5.0))

        self.assertTrue(_wait_until(self._daemon_stopped, timeout=5.0))

    def test_daemon_restarts_after_idle_exit(self):
        """A second submit_event after idle-out is processed by a fresh daemon."""
        done1 = threading.Event()
        ev_a = MagicMock()
        ev_a.elapsed_time.return_value = 1.0
        self._resolver.submit_event(lambda _: done1.set(), ev_a, MagicMock())
        self.assertTrue(done1.wait(timeout=5.0))
        self.assertTrue(_wait_until(self._daemon_stopped, timeout=5.0))

        done2 = threading.Event()
        ev_b = MagicMock()
        ev_b.elapsed_time.return_value = 2.0
        self._resolver.submit_event(lambda _: done2.set(), ev_b, MagicMock())
        self.assertTrue(done2.wait(timeout=5.0))


if __name__ == "__main__":
    run_tests()
