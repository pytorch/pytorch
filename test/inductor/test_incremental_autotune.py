# Owner(s): ["module: inductor"]

import gc
import queue
import threading
import time
from unittest.mock import MagicMock, patch

from torch._inductor.runtime.incremental._launcher import Launcher
from torch._inductor.runtime.incremental._state import IncrementalAutotuneState
from torch._inductor.runtime.incremental.config import (
    max_samples_per_launcher,
    min_samples_before_filter,
    TimingAggregation,
)
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


def _make_state(
    launchers: list[Launcher], **kwargs: object
) -> IncrementalAutotuneState:
    """Create an IncrementalAutotuneState seeded with launchers."""
    state = IncrementalAutotuneState(**kwargs)
    for launcher in launchers:
        state.add_launcher(launcher)
    return state


def _force_total_timings(launcher: Launcher, n: int) -> None:
    """Force ``num_total_timings`` to ``n`` by stuffing the in-flight counter."""
    with launcher._lock:
        launcher._num_timings_in_flight = n - len(launcher._timings)


def _set_in_flight(launcher: Launcher, n: int) -> None:
    with launcher._lock:
        launcher._num_timings_in_flight = n


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
        _set_in_flight(launcher, 2)
        launcher.add_timing(1.0)
        self.assertEqual(launcher.num_in_flight_timings, 1)
        self.assertEqual(launcher.num_available_timings, 1)
        launcher.add_timing(2.0)
        self.assertEqual(launcher.num_in_flight_timings, 0)
        self.assertEqual(launcher.num_available_timings, 2)

    def test_num_total_timings_sums_in_flight_and_available(self):
        launcher = _make_launcher()
        _set_in_flight(launcher, 3)
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


class IncrementalAutotuneStateTest(TestCase):
    def test_add_launcher_queue(self):
        a = _make_launcher("a")
        b = _make_launcher("b")
        state = _make_state([a, b])
        self.assertIs(state._queue[0], a)
        self.assertIs(state._queue[1], b)
        self.assertEqual(len(state._queue), 2)

    def test_next_launcher_skips_when_in_flight_exhausts_budget(self):
        a = _make_launcher("a")
        b = _make_launcher("b")
        state = _make_state([a, b])
        _force_total_timings(a, 999)
        self.assertIs(state._next_launcher(), b)
        # Skipped (not filtered), so a remains in the active set.
        self.assertIn(a, state._queue)

    def test_next_launcher_returns_none_when_all_skipped(self):
        a = _make_launcher("a")
        state = _make_state([a])
        _force_total_timings(a, 999)
        self.assertIsNone(state._next_launcher())
        self.assertIn(a, state._queue)

    def test_next_launcher_promotes_first_done_to_best(self):
        a = _make_launcher("a")
        state = _make_state([a])
        for _ in range(max_samples_per_launcher):
            a.add_timing(1.0)
        self.assertIsNone(state._next_launcher())
        self.assertIs(state._best_launcher, a)

    def test_next_launcher_replaces_best_with_faster_done(self):
        dropped: list[Launcher] = []
        old_best = _make_launcher("old")
        new_best = _make_launcher("new")
        for _ in range(max_samples_per_launcher):
            old_best.add_timing(5.0)
            new_best.add_timing(1.0)
        state = IncrementalAutotuneState(on_discard_fn=dropped.append)
        state._best_launcher = old_best
        state._queue.append(new_best)
        self.assertIsNone(state._next_launcher())
        self.assertIs(state._best_launcher, new_best)
        self.assertIn(old_best, dropped)

    def test_next_launcher_discards_slower_done(self):
        dropped: list[Launcher] = []
        cached = _make_launcher("cached")
        loser = _make_launcher("loser")
        for _ in range(max_samples_per_launcher):
            cached.add_timing(1.0)
            loser.add_timing(5.0)
        state = IncrementalAutotuneState(on_discard_fn=dropped.append)
        state._best_launcher = cached
        state._queue.append(loser)
        self.assertIsNone(state._next_launcher())
        self.assertIs(state._best_launcher, cached)
        self.assertIn(loser, dropped)

    def test_next_launcher_threshold_discards_slow_candidate(self):
        dropped: list[Launcher] = []
        best = _make_launcher("best")
        slow = _make_launcher("slow")
        for _ in range(max_samples_per_launcher):
            best.add_timing(1.0)
        state = IncrementalAutotuneState(on_discard_fn=dropped.append)
        state._best_launcher = best
        state._queue.append(slow)
        for _ in range(min_samples_before_filter):
            slow.add_timing(10.0)
        self.assertIsNone(state._next_launcher())
        self.assertIn(slow, dropped)

    def test_next_launcher_threshold_silent_when_launcher_is_temp_best(self):
        a = _make_launcher("a")
        state = _make_state([a])
        for _ in range(min_samples_before_filter):
            a.add_timing(100.0)
        # ``a`` is its own temp_best → threshold check skipped, dispatch ``a``.
        self.assertIs(state._next_launcher(), a)

    def test_next_launcher_threshold_uses_temp_best_from_queue(self):
        """temp_best can come from the queue, not just the cached _best_launcher."""
        dropped: list[Launcher] = []
        fast = _make_launcher("fast")
        slow1 = _make_launcher("slow1")
        slow2 = _make_launcher("slow2")
        state = IncrementalAutotuneState(on_discard_fn=dropped.append)
        # Order matters: temp_best (fast) is at the back so the loop hits the
        # slow launchers first and discards them against it.
        state.add_launcher(slow1)
        state.add_launcher(slow2)
        state.add_launcher(fast)
        for _ in range(min_samples_before_filter):
            fast.add_timing(1.0)
            slow1.add_timing(10.0)
            slow2.add_timing(10.0)
        self.assertIs(state._next_launcher(), fast)
        self.assertIn(slow1, dropped)
        self.assertIn(slow2, dropped)

    def test_next_launcher_threshold_relaxed_when_best_has_few_samples(self):
        dropped: list[Launcher] = []
        best = _make_launcher("best")
        candidate = _make_launcher("candidate")
        # Best has only one sample → noisy, so we shouldn't discard a
        # candidate that's only modestly slower despite the threshold.
        best.add_timing(1.0)
        state = IncrementalAutotuneState(on_discard_fn=dropped.append)
        state._best_launcher = best
        state._queue.append(candidate)
        for _ in range(min_samples_before_filter):
            candidate.add_timing(2.0)
        self.assertIs(state._next_launcher(), candidate)
        self.assertNotIn(candidate, dropped)

    def test_converged_reflects_queue_state(self):
        state = IncrementalAutotuneState()
        self.assertTrue(state.converged)
        state._queue.append(_make_launcher())
        self.assertFalse(state.converged)

    def test_best_launcher_returns_cached_when_converged(self):
        state = IncrementalAutotuneState()
        # Empty queue → converged.
        self.assertIsNone(state.best_launcher)
        a = _make_launcher("a")
        state._best_launcher = a
        self.assertIs(state.best_launcher, a)

    def test_best_launcher_asserts_when_not_converged(self):
        state = _make_state([_make_launcher()])
        with self.assertRaises(AssertionError):
            _ = state.best_launcher
        with self.assertRaises(AssertionError):
            _ = state.best_timing

    def test_temp_best_launcher_picks_min_across_queue_and_cached(self):
        state = IncrementalAutotuneState()
        cached = _make_launcher("cached")
        a = _make_launcher("a")
        b = _make_launcher("b")
        cached.add_timing(2.0)
        a.add_timing(5.0)
        b.add_timing(1.0)
        state._queue.append(a)
        state._queue.append(b)
        state._best_launcher = cached
        self.assertIs(state._temp_best_launcher, b)

    def test_add_launcher_multiple(self):
        launchers = [_make_launcher(f"l{i}") for i in range(3)]
        state = _make_state(launchers)
        self.assertEqual(len(state._queue), 3)

    def test_dispatch_round_robin_and_convergence(self):
        """__call__ iterates launchers round-robin and calls on_convergence when done."""
        converged = threading.Event()
        converged_launcher: list[Launcher | None] = [None]

        def on_convergence(state: IncrementalAutotuneState) -> None:
            converged_launcher[0] = state.best_launcher
            converged.set()

        a = _make_launcher("a")
        b = _make_launcher("b")
        # Make b reliably slower so it gets filtered.
        a._fn_keepalive.return_value = "result_a"
        state = _make_state([a, b], on_convergence_fn=on_convergence)

        timing_for: dict[int, float] = {id(a): 1.0, id(b): 10.0}

        def fake_submit(
            callback: object,
            _start: object,
            _end: object,
        ) -> None:
            launcher = callback.__self__  # bound method → owning launcher
            callback(timing_for[id(launcher)])

        with (
            patch("torch._inductor.runtime.incremental._state.sampling_rate", 1),
            patch("torch.cuda.Event", return_value=MagicMock()),
            patch(
                "torch._inductor.runtime.incremental._launcher.submit_event"
            ) as mock_submit,
        ):
            mock_submit.side_effect = fake_submit
            for _ in range(100):
                state(stream=0)
                if converged.is_set():
                    break

        self.assertTrue(converged.is_set())
        self.assertIs(converged_launcher[0], a)

    def test_first_dispatch_per_launcher_is_untimed_warmup(self):
        """A cold launcher gets one untimed warmup dispatch before timing."""
        a = _make_launcher("a")
        b = _make_launcher("b")
        state = _make_state([a, b])

        with (
            patch("torch.cuda.Event", return_value=MagicMock()),
            patch(
                "torch._inductor.runtime.incremental._launcher.submit_event"
            ) as mock_submit,
        ):
            # First dispatch per launcher: untimed warmup; is_warm becomes True.
            state(stream=0)
            state(stream=0)
            self.assertEqual(mock_submit.call_count, 0)
            self.assertTrue(a.is_warm)
            self.assertTrue(b.is_warm)
            # Subsequent dispatches: timed.
            state(stream=0)
            state(stream=0)
            self.assertEqual(mock_submit.call_count, 2)

    def test_dispatch_drops_invalid_config_and_retries(self):
        """RuntimeError("invalid configuration ...") prunes the launcher and retries."""
        bad = _make_launcher("bad")
        good = _make_launcher("good")
        bad._fn_keepalive.side_effect = RuntimeError(
            "invalid configuration argument from CUDA launch"
        )
        state = _make_state([bad, good])

        with (
            patch("torch.cuda.Event", return_value=MagicMock()),
            patch("torch._inductor.runtime.incremental._launcher.submit_event"),
        ):
            result = state(stream=0)

        self.assertEqual(result, "result_good")
        self.assertNotIn(bad, state._queue)

    def test_on_discard_fn_invoked_on_discard_and_invalid_config(self):
        dropped: list[Launcher] = []
        bad = _make_launcher("bad")
        good = _make_launcher("good")
        bad._fn_keepalive.side_effect = RuntimeError(
            "invalid configuration argument from CUDA launch"
        )
        state = IncrementalAutotuneState(on_discard_fn=dropped.append)
        state.add_launcher(bad)
        state.add_launcher(good)

        with (
            patch("torch.cuda.Event", return_value=MagicMock()),
            patch("torch._inductor.runtime.incremental._launcher.submit_event"),
        ):
            state(stream=0)
        self.assertIn(bad, dropped)


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
