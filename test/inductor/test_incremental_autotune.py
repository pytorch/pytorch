# Owner(s): ["module: inductor"]

import gc
import queue
import threading
import time
from unittest.mock import MagicMock, patch

from torch._inductor.runtime.incremental._launcher import Launcher
from torch._inductor.runtime.incremental._state import IncrementalAutotuneState
from torch._inductor.runtime.incremental.config import (
    LauncherTimingAggregation,
    max_timings_per_launcher,
    min_timings_before_filter,
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
    fn.config = f"config:{name}"
    fn.cache_hash = f"hash:{name}"
    launcher = Launcher(raw_launcher=fn)
    # Production keeps fn alive via CachingAutotuner.launchers; tests stash
    # it on the launcher so the weakref doesn't die immediately.
    launcher._raw_launcher_keepalive = fn  # pyrefly: ignore
    return launcher


def _make_state(
    launchers: list[Launcher], **kwargs: object
) -> IncrementalAutotuneState:
    """Create an IncrementalAutotuneState seeded with launchers."""
    state = IncrementalAutotuneState(**kwargs)
    for launcher in launchers:
        # Tests use the launcher's own raw as the autotuner-side raw.
        raw = launcher.get_raw_launcher()
        state.add_launcher(launcher, raw)
    return state


def _force_total_timings(launcher: Launcher, n: int) -> None:
    """Force ``num_total_timings`` to ``n`` by stuffing the in-flight counter."""
    with launcher._lock:
        launcher._num_timings_in_flight = n - len(launcher._timings)


def _set_in_flight(launcher: Launcher, n: int) -> None:
    with launcher._lock:
        launcher._num_timings_in_flight = n


def _discard(state: IncrementalAutotuneState, launcher: Launcher) -> list:
    """Test helper: invoke ``state._discard`` under the state lock and
    return the deferred discard list. Mirrors how production callers
    (``add_launcher`` / ``_next_launcher``) wrap the call."""
    pending: list = []
    with state._lock:
        state._discard(launcher, pending)
    if state._on_discard_fn is not None:
        for entry in pending:
            # _discard appends (launcher, raw) tuples.
            state._on_discard_fn(*entry)
    return pending


class _PicklableRaw:
    """Module-level raw launcher used in pickle round-trip tests."""

    def __init__(self, config: object = "cfg", cache_hash: str = "h") -> None:
        self.config = config
        self.cache_hash = cache_hash

    def __call__(self, *_a: object, **_kw: object) -> str:
        return "ok"


class LauncherTest(TestCase):
    def test_timing_empty(self):
        launcher = _make_launcher()
        self.assertEqual(launcher.timing, float("inf"))

    def test_call_delegates_to_raw_launcher(self):
        launcher = _make_launcher("test")
        result = launcher(1, 2, stream=0)
        self.assertEqual(result, "result_test")
        launcher._raw_launcher_keepalive.assert_called_once_with(1, 2, stream=0)

    def test_call_raises_when_raw_launcher_garbage_collected(self):
        """raw_launcher is held weakly; once the only strong ref drops,
        __call__ asserts."""

        class _Callable:
            def __call__(self) -> str:
                return "ok"

        fn = _Callable()
        launcher = Launcher(raw_launcher=fn)
        del fn
        gc.collect()
        with self.assertRaises(AssertionError):
            launcher(stream=0)

    def test_set_raw_launcher_replaces_and_resets_warm(self):
        """set_raw_launcher attaches a fresh raw to a previously-dormant
        Launcher (or replaces a dead one) and resets the warm flag."""
        original = MagicMock(return_value="orig")
        launcher = Launcher(raw_launcher=original)
        launcher._raw_launcher_keepalive_orig = original  # pyrefly: ignore
        launcher(stream=0)
        self.assertTrue(launcher._warm)

        replacement = MagicMock(return_value="new")
        launcher.set_raw_launcher(replacement)
        launcher._raw_launcher_keepalive_repl = replacement  # pyrefly: ignore
        self.assertFalse(launcher._warm)
        self.assertEqual(launcher(stream=0), "new")

    def test_set_raw_launcher_preserves_timings(self):
        """set_raw_launcher assumes the new raw is interchangeable, so
        accumulated timings must not be cleared."""
        launcher = _make_launcher()
        launcher.add_timing(1.0)
        launcher.add_timing(2.0)
        new_fn = MagicMock(return_value="new")
        launcher.set_raw_launcher(new_fn)
        launcher._raw_launcher_keepalive = new_fn  # pyrefly: ignore
        self.assertEqual(launcher.num_available_timings, 2)
        self.assertAlmostEqual(launcher.timing, 1.5)

    def test_first_call_sets_warm(self):
        launcher = _make_launcher()
        self.assertFalse(launcher._warm)
        launcher(stream=0)
        self.assertTrue(launcher._warm)

    def test_untimed_call_does_not_track_timing(self):
        launcher = _make_launcher()
        launcher(stream=0)
        launcher(stream=0)
        self.assertEqual(launcher.num_in_flight_timings, 0)
        self.assertEqual(launcher.num_available_timings, 0)
        self.assertEqual(launcher.num_total_timings, 0)

    def test_timed_call_increments_in_flight_and_submits_event(self):
        launcher = _make_launcher()
        # Warm the launcher so the next time_if_warm=True call actually times.
        launcher(stream=0)
        start_event = MagicMock()
        end_event = MagicMock()
        events = iter([start_event, end_event])
        with (
            patch("torch.cuda.Event", side_effect=lambda **_: next(events)),
            patch(
                "torch._inductor.runtime.incremental._launcher.submit_event"
            ) as mock_submit,
        ):
            launcher(stream=0, time_if_warm=True)
            self.assertEqual(launcher.num_in_flight_timings, 1)
            mock_submit.assert_called_once_with(
                launcher.add_timing, start_event, end_event
            )

    def test_time_if_warm_false_when_not_yet_warm(self):
        """time_if_warm=True on a cold launcher does NOT time — the
        launcher's internal _warm gate prevents measuring a cold dispatch.
        """
        launcher = _make_launcher()
        with patch(
            "torch._inductor.runtime.incremental._launcher.submit_event"
        ) as mock_submit:
            launcher(stream=0, time_if_warm=True)
            self.assertEqual(launcher.num_in_flight_timings, 0)
            mock_submit.assert_not_called()
        # Now warm; subsequent time_if_warm=True does time.
        with patch("torch.cuda.Event", return_value=MagicMock()), patch(
            "torch._inductor.runtime.incremental._launcher.submit_event"
        ) as mock_submit:
            launcher(stream=0, time_if_warm=True)
            self.assertEqual(launcher.num_in_flight_timings, 1)
            mock_submit.assert_called_once()

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
            "torch._inductor.runtime.incremental._launcher.launcher_timing_aggregation",
            LauncherTimingAggregation.MEAN,
        ):
            self.assertAlmostEqual(launcher.timing, 22.0)

    def test_on_timing_update_callback_receives_self_old_and_new(self):
        # WeakMethod requires a bound method whose owning instance supports
        # weakref, so we use a small recorder class instead of list.append.
        class _Recorder:
            def __init__(self) -> None:
                self.calls: list[tuple[Launcher, float, float]] = []

            def record(
                self, launcher: Launcher, old_timing: float, new_timing: float
            ) -> None:
                self.calls.append((launcher, old_timing, new_timing))

        launcher = _make_launcher()
        rec = _Recorder()
        launcher.add_on_timing_update_fn(rec.record)
        launcher.add_timing(1.0)  # median: inf -> 1.0
        launcher.add_timing(2.0)  # median: 1.0 -> 1.5
        self.assertEqual(len(rec.calls), 2)
        self.assertIs(rec.calls[0][0], launcher)
        self.assertEqual(rec.calls[0][1], float("inf"))
        self.assertEqual(rec.calls[0][2], 1.0)
        self.assertIs(rec.calls[1][0], launcher)
        self.assertEqual(rec.calls[1][1], 1.0)
        self.assertEqual(rec.calls[1][2], 1.5)

    def test_on_timing_update_subscription_cancel_stops_future_fires(self):
        class _Counter:
            def __init__(self) -> None:
                self.n = 0

            def bump(self, _l: Launcher, _o: float, _n: float) -> None:
                self.n += 1

        launcher = _make_launcher()
        counter = _Counter()
        sub = launcher.add_on_timing_update_fn(counter.bump)
        launcher.add_timing(1.0)
        sub.cancel()
        launcher.add_timing(2.0)
        self.assertEqual(counter.n, 1)

    def test_on_timing_update_subscription_cancel_removes_from_registry(self):
        """``cancel()`` must physically remove the sub from the parent
        Launcher's ``_on_timing_update_subs`` so subsequent ``add_timing``
        iterations don't pay for dead subs (and so the deterministic-
        cleanup contract from the docstring holds without relying on a
        future ``add_timing`` call to prune)."""

        class _Counter:
            def __init__(self) -> None:
                self.n = 0

            def bump(self, _l: Launcher, _o: float, _n: float) -> None:
                self.n += 1

        launcher = _make_launcher()
        counter = _Counter()
        sub = launcher.add_on_timing_update_fn(counter.bump)
        self.assertEqual(len(launcher._on_timing_update_subs), 1)
        sub.cancel()
        self.assertEqual(len(launcher._on_timing_update_subs), 0)
        # Idempotent.
        sub.cancel()
        self.assertEqual(len(launcher._on_timing_update_subs), 0)

    def test_on_timing_update_subscription_cancel_is_idempotent(self):
        class _Counter:
            def __init__(self) -> None:
                self.n = 0

            def bump(self, _l: Launcher, _o: float, _n: float) -> None:
                self.n += 1

        launcher = _make_launcher()
        sub = launcher.add_on_timing_update_fn(_Counter().bump)
        sub.cancel()
        sub.cancel()  # second cancel should no-op, not raise

    def test_on_timing_update_subscription_cancel_waits_for_in_flight(self):
        """``cancel()`` must block until any in-flight call completes,
        so the caller can rely on the callback being retired by the
        time ``cancel`` returns.
        """

        class _SlowCallback:
            def __init__(self) -> None:
                self.entered = threading.Event()
                self.may_return = threading.Event()
                self.calls = 0

            def slow(self, _l: Launcher, _o: float, _n: float) -> None:
                self.calls += 1
                self.entered.set()
                # Block until the test releases us.
                self.may_return.wait(timeout=5.0)

        launcher = _make_launcher()
        slow = _SlowCallback()
        sub = launcher.add_on_timing_update_fn(slow.slow)

        # Fire add_timing on a background thread so the callback runs
        # there; we can race ``cancel`` against the in-flight callback.
        fire_thread = threading.Thread(target=launcher.add_timing, args=(1.0,))
        fire_thread.start()
        self.assertTrue(slow.entered.wait(timeout=5.0))

        # Now the callback is mid-execution; ``cancel`` must wait.
        cancel_thread = threading.Thread(target=sub.cancel)
        cancel_thread.start()
        # Cancel shouldn't have returned yet because the callback is blocked.
        cancel_thread.join(timeout=0.05)
        self.assertTrue(cancel_thread.is_alive())

        # Release the callback; both threads should now finish, and any
        # subsequent add_timing must NOT call the callback.
        slow.may_return.set()
        cancel_thread.join(timeout=5.0)
        fire_thread.join(timeout=5.0)
        self.assertFalse(cancel_thread.is_alive())
        self.assertFalse(fire_thread.is_alive())

        launcher.add_timing(2.0)
        self.assertEqual(slow.calls, 1)

    def test_on_timing_update_callbacks_multiple_owners(self):
        """A Launcher shared across owners fires every registered callback."""

        class _Counter:
            def __init__(self) -> None:
                self.n = 0

            def bump(self, _l: Launcher, _o: float, _n: float) -> None:
                self.n += 1

        launcher = _make_launcher()
        a, b = _Counter(), _Counter()
        launcher.add_on_timing_update_fn(a.bump)
        launcher.add_on_timing_update_fn(b.bump)
        launcher.add_timing(1.0)
        self.assertEqual((a.n, b.n), (1, 1))

    def test_on_timing_update_callback_pruned_when_owner_gced(self):
        """Owners that go away should auto-prune their callbacks."""

        class _Counter:
            def __init__(self) -> None:
                self.n = 0

            def bump(self, _l: Launcher, _o: float, _n: float) -> None:
                self.n += 1

        launcher = _make_launcher()
        survivor = _Counter()
        ephemeral = _Counter()
        launcher.add_on_timing_update_fn(survivor.bump)
        launcher.add_on_timing_update_fn(ephemeral.bump)
        del ephemeral
        gc.collect()
        launcher.add_timing(1.0)
        self.assertEqual(survivor.n, 1)
        # The ephemeral callback's WeakMethod was pruned in place.
        self.assertEqual(len(launcher._on_timing_update_subs), 1)


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
        for _ in range(max_timings_per_launcher):
            a.add_timing(1.0)
        self.assertIsNone(state._next_launcher())
        self.assertIs(state._certified_launcher, a)

    def test_next_launcher_replaces_best_with_faster_done(self):
        dropped: list[Launcher] = []
        old_certified = _make_launcher("old")
        new_certified = _make_launcher("new")
        for _ in range(max_timings_per_launcher):
            old_certified.add_timing(5.0)
            new_certified.add_timing(1.0)
        state = IncrementalAutotuneState(on_discard_fn=lambda launcher, _raw: dropped.append(launcher))
        state._certified_launcher = old_certified
        state._queue.append(new_certified)
        self.assertIsNone(state._next_launcher())
        self.assertIs(state._certified_launcher, new_certified)
        self.assertIn(old_certified, dropped)

    def test_next_launcher_discards_slower_done(self):
        dropped: list[Launcher] = []
        cached = _make_launcher("cached")
        loser = _make_launcher("loser")
        for _ in range(max_timings_per_launcher):
            cached.add_timing(1.0)
            loser.add_timing(5.0)
        state = IncrementalAutotuneState(on_discard_fn=lambda launcher, _raw: dropped.append(launcher))
        state._certified_launcher = cached
        state._queue.append(loser)
        self.assertIsNone(state._next_launcher())
        self.assertIs(state._certified_launcher, cached)
        self.assertIn(loser, dropped)

    def test_next_launcher_threshold_discards_slow_candidate(self):
        dropped: list[Launcher] = []
        best = _make_launcher("best")
        slow = _make_launcher("slow")
        for _ in range(max_timings_per_launcher):
            best.add_timing(1.0)
        state = IncrementalAutotuneState(on_discard_fn=lambda launcher, _raw: dropped.append(launcher))
        state._certified_launcher = best
        state._queue.append(slow)
        for _ in range(min_timings_before_filter):
            slow.add_timing(10.0)
        # Direct setup bypasses add_launcher, so seed the temp-best cache.
        state._recompute_provisional()
        self.assertIsNone(state._next_launcher())
        self.assertIn(slow, dropped)

    def test_next_launcher_threshold_silent_when_launcher_is_provisional(self):
        a = _make_launcher("a")
        state = _make_state([a])
        for _ in range(min_timings_before_filter):
            a.add_timing(100.0)
        # ``a`` is its own provisional → threshold check skipped, dispatch ``a``.
        self.assertIs(state._next_launcher(), a)

    def test_next_launcher_threshold_uses_provisional_from_queue(self):
        """provisional can come from the queue, not just the cached
        _certified_launcher.

        Order matters: provisional (fast) is at the back so the loop
        hits slow1 and slow2 first and discards them against fast's
        timing. After both losers are discarded, the queue is down to
        just fast — ``_maybe_converge_early`` (triggered from
        ``_discard``) then promotes fast to certified, so
        ``_next_launcher`` returns None (the caller's fallback path
        uses the certified launcher) rather than dispatching fast for
        more tuning.
        """
        dropped: list[Launcher] = []
        fast = _make_launcher("fast")
        slow1 = _make_launcher("slow1")
        slow2 = _make_launcher("slow2")
        state = IncrementalAutotuneState(on_discard_fn=lambda launcher, _raw: dropped.append(launcher))
        state.add_launcher(slow1, slow1.get_raw_launcher())
        state.add_launcher(slow2, slow2.get_raw_launcher())
        state.add_launcher(fast, fast.get_raw_launcher())
        for _ in range(min_timings_before_filter):
            fast.add_timing(1.0)
            slow1.add_timing(10.0)
            slow2.add_timing(10.0)
        self.assertIsNone(state._next_launcher())
        self.assertIs(state._certified_launcher, fast)
        self.assertIn(slow1, dropped)
        self.assertIn(slow2, dropped)

    def test_next_launcher_threshold_relaxed_when_best_has_few_samples(self):
        dropped: list[Launcher] = []
        best = _make_launcher("best")
        candidate = _make_launcher("candidate")
        # Best has only one sample → noisy, so we shouldn't discard a
        # candidate that's only modestly slower despite the threshold.
        best.add_timing(1.0)
        state = IncrementalAutotuneState(on_discard_fn=lambda launcher, _raw: dropped.append(launcher))
        state._certified_launcher = best
        state._queue.append(candidate)
        for _ in range(min_timings_before_filter):
            candidate.add_timing(2.0)
        # Direct setup bypasses add_launcher, so seed the temp-best cache.
        state._recompute_provisional()
        self.assertIs(state._next_launcher(), candidate)
        self.assertNotIn(candidate, dropped)

    def test_converged_reflects_queue_state(self):
        state = IncrementalAutotuneState()
        self.assertTrue(state.converged)
        state._queue.append(_make_launcher())
        self.assertFalse(state.converged)

    def test_discard_promotes_last_survivor_without_max_samples(self):
        """When threshold-discards leave a single queued launcher and
        no certified launcher yet, the survivor wins by elimination
        and is promoted to certified immediately — no need to keep
        sampling against nobody. Mirrors how ``_next_launcher`` drives
        the flow: popleft the launcher first, then ``_discard`` it.
        """
        a = _make_launcher("a")
        survivor = _make_launcher("survivor")
        state = _make_state([a, survivor])
        self.assertFalse(state.converged)
        self.assertIsNone(state._certified_launcher)
        state._queue.popleft()  # ``_next_launcher`` pops before discarding
        _discard(state, a)
        self.assertTrue(state.converged)
        self.assertIs(state._certified_launcher, survivor)
        self.assertEqual(len(state._queue), 0)
        # Survivor's timing budget was never touched.
        self.assertEqual(survivor.num_available_timings, 0)

    def test_discard_does_not_early_promote_when_certified_exists(self):
        """If a certified launcher already exists, a discard that
        leaves one queued launcher must NOT short-circuit — the queued
        one needs samples to fairly compete with the certified one."""
        certified = _make_launcher("certified")
        a = _make_launcher("a")
        b = _make_launcher("b")
        state = _make_state([a, b])
        state._certified_launcher = certified
        state._queue.popleft()
        _discard(state, a)
        self.assertFalse(state.converged)
        self.assertIs(state._certified_launcher, certified)
        self.assertIs(state._queue[0], b)

    def test_discard_does_not_early_promote_when_queue_emptied(self):
        """If a discard empties the queue entirely with no certified,
        the assertion in the converged dispatch path will fire later
        — early-promote must not silently certify nothing."""
        a = _make_launcher("a")
        state = _make_state([a])
        state._queue.popleft()
        _discard(state, a)
        self.assertTrue(state.converged)
        self.assertIsNone(state._certified_launcher)

    def test_certified_launcher_returns_cached_when_converged(self):
        state = IncrementalAutotuneState()
        # Empty queue → converged.
        self.assertIsNone(state.certified_launcher)
        a = _make_launcher("a")
        state._certified_launcher = a
        self.assertIs(state.certified_launcher, a)

    def test_certified_launcher_asserts_when_not_converged(self):
        state = _make_state([_make_launcher()])
        with self.assertRaises(AssertionError):
            _ = state.certified_launcher
        with self.assertRaises(AssertionError):
            _ = state.certified_timing

    def test_recompute_provisional_picks_min_across_queue_and_cached(self):
        state = IncrementalAutotuneState()
        cached = _make_launcher("cached")
        a = _make_launcher("a")
        b = _make_launcher("b")
        cached.add_timing(2.0)
        a.add_timing(5.0)
        b.add_timing(1.0)
        state._queue.append(a)
        state._queue.append(b)
        state._certified_launcher = cached
        state._recompute_provisional()
        self.assertIs(state._provisional_launcher, b)

    def test_add_launcher_seeds_provisional(self):
        state = IncrementalAutotuneState()
        a = _make_launcher("a")
        state.add_launcher(a, a.get_raw_launcher())
        self.assertIs(state._provisional_launcher, a)

    def test_provisional_updates_on_timing_callback(self):
        state = IncrementalAutotuneState()
        a = _make_launcher("a")
        b = _make_launcher("b")
        state.add_launcher(a, a.get_raw_launcher())
        state.add_launcher(b, b.get_raw_launcher())
        # Both launchers register the timing-update callback, so the first
        # finite timing on ``b`` should immediately make it the temp best.
        b.add_timing(1.0)
        self.assertIs(state._provisional_launcher, b)
        # A faster timing on ``a`` should now displace ``b``.
        a.add_timing(0.5)
        self.assertIs(state._provisional_launcher, a)
        # A slower timing on ``b`` doesn't change anything.
        b.add_timing(10.0)
        self.assertIs(state._provisional_launcher, a)

    def test_discard_provisional_triggers_recompute(self):
        dropped: list[Launcher] = []
        state = IncrementalAutotuneState(on_discard_fn=lambda launcher, _raw: dropped.append(launcher))
        a = _make_launcher("a")
        b = _make_launcher("b")
        state.add_launcher(a, a.get_raw_launcher())
        state.add_launcher(b, b.get_raw_launcher())
        a.add_timing(1.0)
        b.add_timing(5.0)
        self.assertIs(state._provisional_launcher, a)
        # Discard contract: caller has already popped the launcher off the queue.
        state._queue.remove(a)
        _discard(state, a)
        self.assertIs(state._provisional_launcher, b)

    def test_discard_clears_launcher_callback(self):
        state = IncrementalAutotuneState()
        a = _make_launcher("a")
        state.add_launcher(a, a.get_raw_launcher())
        self.assertEqual(len(a._on_timing_update_subs), 1)
        _discard(state, a)
        self.assertEqual(len(a._on_timing_update_subs), 0)

    def test_convergence_clears_callback_on_certified_launcher(self):
        state = IncrementalAutotuneState(
            on_convergence_fn=lambda _state: None,
        )
        a = _make_launcher("a")
        state.add_launcher(a, a.get_raw_launcher())
        for _ in range(max_timings_per_launcher):
            a.add_timing(1.0)
        # Drive a to converged: _next_launcher promotes it to _certified_launcher.
        state._next_launcher()
        self.assertTrue(state.converged)
        self.assertEqual(len(a._on_timing_update_subs), 1)
        # The converged branch of __call__ clears the callback before the
        # convergence hook fires; trigger that path with a no-op fn ref.
        a._raw_launcher_keepalive = MagicMock(return_value=None)  # pyrefly: ignore
        a.set_raw_launcher(a._raw_launcher_keepalive)
        state(stream=0)
        self.assertEqual(len(a._on_timing_update_subs), 0)

    def test_add_launcher_multiple(self):
        launchers = [_make_launcher(f"l{i}") for i in range(3)]
        state = _make_state(launchers)
        self.assertEqual(len(state._queue), 3)

    def test_unsaturated_then_saturate_loser_no_orphan_sub(self):
        """A launcher arrives unsaturated (sub registered via
        ``add_launcher``), accumulates timings, becomes saturated, is
        popped via ``_next_launcher`` and loses ``_maybe_promote``
        (cert is faster) → ``_register_provisional_sub`` is no-op
        (idempotent) and ``_discard``'s ``cancel()`` removes the
        original sub. No leak."""
        cert = _make_launcher("cert")
        loser = _make_launcher("loser")
        state = IncrementalAutotuneState()
        # Cert: saturated and fast. Pre-populate timings + add.
        for _ in range(max_timings_per_launcher):
            cert.add_timing(1.0)
        state.add_launcher(cert, cert.get_raw_launcher())
        # Loser: unsaturated, will saturate later.
        state.add_launcher(loser, loser.get_raw_launcher())
        self.assertEqual(len(loser._on_timing_update_subs), 1)
        # Saturate loser with slower timings.
        for _ in range(max_timings_per_launcher):
            loser.add_timing(10.0)
        # Pop loser via _next_launcher → _maybe_promote → loser loses
        # → _discard. The cert had auto-promoted on add_launcher, so
        # cert is _certified_launcher already.
        self.assertIs(state._certified_launcher, cert)
        # _next_launcher walks queue; finds loser saturated → promote
        # against cert → discard loser.
        state._next_launcher()
        # loser's sub was cancelled and removed from its registry.
        self.assertEqual(len(loser._on_timing_update_subs), 0)
        # Loser is not in state's tracking anymore.
        self.assertNotIn(loser, state._provisional_subs)
        self.assertNotIn(loser, state._per_launcher_raw)

    def test_dispatch_round_robin_and_convergence(self):
        """__call__ iterates launchers round-robin and calls on_convergence when done."""
        converged = threading.Event()
        converged_launcher: list[Launcher | None] = [None]

        def on_convergence(state: IncrementalAutotuneState) -> None:
            converged_launcher[0] = state.certified_launcher
            converged.set()

        a = _make_launcher("a")
        b = _make_launcher("b")
        # Make b reliably slower so it gets filtered.
        a._raw_launcher_keepalive.return_value = "result_a"
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
            patch("torch._inductor.runtime.incremental._state.timed_sampling_rate", 1),
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
            # First dispatch per launcher: untimed warmup; _warm becomes True.
            state(stream=0)
            state(stream=0)
            self.assertEqual(mock_submit.call_count, 0)
            self.assertTrue(a._warm)
            self.assertTrue(b._warm)
            # Subsequent dispatches: timed.
            state(stream=0)
            state(stream=0)
            self.assertEqual(mock_submit.call_count, 2)

    def test_dispatch_drops_invalid_config_and_retries(self):
        """RuntimeError("invalid configuration ...") prunes the launcher and retries."""
        bad = _make_launcher("bad")
        good = _make_launcher("good")
        bad._raw_launcher_keepalive.side_effect = RuntimeError(
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
        bad._raw_launcher_keepalive.side_effect = RuntimeError(
            "invalid configuration argument from CUDA launch"
        )
        state = IncrementalAutotuneState(on_discard_fn=lambda launcher, _raw: dropped.append(launcher))
        state.add_launcher(bad, bad.get_raw_launcher())
        state.add_launcher(good, good.get_raw_launcher())

        with (
            patch("torch.cuda.Event", return_value=MagicMock()),
            patch("torch._inductor.runtime.incremental._launcher.submit_event"),
        ):
            state(stream=0)
        self.assertIn(bad, dropped)


class StreamingCompileTest(TestCase):
    """Tests for the streaming-compile primitives on
    ``IncrementalAutotuneState``: pending-compile counter, blocking
    first dispatch, and the all-initial-failed failure mode."""

    def test_pending_blocks_convergence_with_empty_queue(self):
        state = IncrementalAutotuneState()
        self.assertTrue(state.converged)
        state.register_pending_compilations(2)
        self.assertFalse(state.converged)

    def test_register_pending_asserts_on_non_positive(self):
        state = IncrementalAutotuneState()
        with self.assertRaises(AssertionError):
            state.register_pending_compilations(0)
        with self.assertRaises(AssertionError):
            state.register_pending_compilations(-3)

    def test_note_compilation_failed_decrements(self):
        state = IncrementalAutotuneState()
        state.register_pending_compilations(2)
        state.note_compilation_failed()
        self.assertEqual(state._pending_compilations, 1)
        self.assertFalse(state.converged)
        state.note_compilation_failed()
        self.assertEqual(state._pending_compilations, 0)
        self.assertTrue(state.converged)

    def test_add_launcher_decrements_pending(self):
        state = IncrementalAutotuneState()
        state.register_pending_compilations(1)
        a = _make_launcher("a")
        state.add_launcher(a, a.get_raw_launcher())
        self.assertEqual(state._pending_compilations, 0)

    def test_add_launcher_without_pending_is_fine(self):
        """Direct adds (no prior register_pending) still work — counter
        stays at zero."""
        state = IncrementalAutotuneState()
        a = _make_launcher("a")
        state.add_launcher(a, a.get_raw_launcher())
        self.assertEqual(state._pending_compilations, 0)
        self.assertIs(state._queue[0], a)

    def test_maybe_converge_early_does_not_fire_with_pending(self):
        """A discard that leaves one queued launcher must NOT
        early-promote if compiles are still in flight (the in-flight
        one might be faster)."""
        a = _make_launcher("a")
        survivor = _make_launcher("survivor")
        state = _make_state([a, survivor])
        state.register_pending_compilations(1)
        state._queue.popleft()
        _discard(state, a)
        self.assertFalse(state.converged)
        self.assertIsNone(state._certified_launcher)
        self.assertIs(state._queue[0], survivor)

    def test_call_raises_when_all_initial_compiles_fail(self):
        """If every pending compile resolves via
        ``note_compilation_failed`` with no launcher ever added, the
        first dispatch surfaces ``NoTritonConfigsError`` — matching the
        normal-path failure mode."""
        from torch._inductor.runtime.triton_heuristics import (
            NoTritonConfigsError,
        )

        state = IncrementalAutotuneState()
        state.register_pending_compilations(2)

        def fail_all() -> None:
            state.note_compilation_failed()
            state.note_compilation_failed()

        threading.Thread(target=fail_all, daemon=True).start()
        with self.assertRaises(NoTritonConfigsError):
            state(stream=0)

    def test_call_blocks_until_first_launcher_arrives(self):
        """First dispatch waits for a streaming-compile worker to add a
        launcher, then dispatches it."""
        state = IncrementalAutotuneState()
        state.register_pending_compilations(1)
        launcher = _make_launcher("a")
        dispatched_event = threading.Event()
        result_holder: list[object] = []

        def caller() -> None:
            result_holder.append(state(stream=0))
            dispatched_event.set()

        caller_thread = threading.Thread(target=caller, daemon=True)
        caller_thread.start()

        # Caller must be blocked: no launcher has arrived yet.
        self.assertFalse(dispatched_event.wait(timeout=0.05))

        state.add_launcher(launcher, launcher.get_raw_launcher())
        self.assertTrue(dispatched_event.wait(timeout=5.0))
        caller_thread.join(timeout=5.0)
        self.assertEqual(result_holder, ["result_a"])

    def test_convergence_callback_fires_per_generation(self):
        """The convergence callback re-arms when new work arrives
        (add_launcher or register_pending_compilations) so a
        multi-generation flow can chain through it."""
        fire_count = [0]

        def on_conv(_state: IncrementalAutotuneState) -> None:
            fire_count[0] += 1

        a = _make_launcher("a")
        for _ in range(max_timings_per_launcher):
            a.add_timing(1.0)
        state = IncrementalAutotuneState(on_convergence_fn=on_conv)
        state.add_launcher(a, a.get_raw_launcher())
        self.assertTrue(state.converged)

        # First dispatch fires the callback once.
        a._raw_launcher_keepalive = MagicMock(return_value="r")
        a.set_raw_launcher(a._raw_launcher_keepalive)
        state(stream=0)
        self.assertEqual(fire_count[0], 1)

        # Subsequent dispatches in the same converged generation do
        # not re-fire.
        state(stream=0)
        self.assertEqual(fire_count[0], 1)

        # New work arriving re-arms the callback. Once that work
        # converges (here: register-and-fail to keep the test simple),
        # the callback fires again.
        state.register_pending_compilations(1)
        self.assertFalse(state.converged)
        state.note_compilation_failed()
        self.assertTrue(state.converged)
        state(stream=0)
        self.assertEqual(fire_count[0], 2)


class ResolverTest(TestCase):
    def setUp(self):
        from torch._inductor.runtime.incremental import _resolver

        self._resolver = _resolver
        # Short idle timeout so the daemon exits quickly between tests.
        self._timeout_patcher = patch.object(_resolver, "_RESOLVER_IDLE_TIMEOUT_S", 0.05)
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


class CacheTest(TestCase):
    def setUp(self):
        from torch._inductor.runtime.incremental import _launcher

        self._cache = _launcher
        # Snapshot and clear so each test starts with an empty registry,
        # regardless of bleed-through from earlier modules.
        self._original_registry = _launcher._caching_autotuner_launcher_registry.copy()
        _launcher._caching_autotuner_launcher_registry.clear()

    def tearDown(self):
        self._cache._caching_autotuner_launcher_registry.clear()
        self._cache._caching_autotuner_launcher_registry.update(self._original_registry)

    @staticmethod
    def _make_autotuner_mock(name="triton_kernel", src="def triton_kernel():..."):
        """Build a minimal CachingAutotuner-shaped mock for cache key tests."""
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner

        autotuner = MagicMock(spec=CachingAutotuner)
        autotuner.fn = MagicMock()
        autotuner.fn.__name__ = name
        autotuner.fn.src = src
        autotuner.inductor_meta = {"a": 1}
        autotuner.triton_meta = {"b": 2}
        # Attributes required by ``_caching_autotuner_instance_key``
        # that aren't reflected in inductor_meta / triton_meta.
        autotuner.size_hints = {"x": 16}
        autotuner.heuristic_type = "POINTWISE"
        autotuner.custom_kernel = False
        autotuner.mutated_arg_names = []
        autotuner.reset_to_zero_arg_names = []
        autotuner.optimize_mem = True
        return autotuner

    # Share a single config object per x_block across raw launchers so
    # raws produced for the same kernel/config compare equal — mirroring
    # how Triton's Config implements __eq__ over its fields, and what
    # the cache-hit assert in _get_or_create_launchers expects.
    _shared_configs: dict[int, MagicMock] = {}

    @classmethod
    def _make_raw_launcher(cls, x_block: int, cache_hash: str | None = None):
        if x_block not in cls._shared_configs:
            cls._shared_configs[x_block] = MagicMock(
                kwargs={"X": x_block}, num_warps=4, num_stages=1
            )
        raw = MagicMock()
        raw.config = cls._shared_configs[x_block]
        raw.cache_hash = cache_hash if cache_hash is not None else f"hash:{x_block}"
        return raw

    def test_kernel_key_raises_when_fn_has_no_src(self):
        autotuner = self._make_autotuner_mock()
        # Build a fn mock without ``src`` from the start so the absence is
        # structural, not the result of a deferred ``del`` on a MagicMock.
        autotuner.fn = MagicMock(spec=["__name__"])
        autotuner.fn.__name__ = "triton_kernel"
        with self.assertRaises(ValueError):
            self._cache._caching_autotuner_instance_key(autotuner)

    def test_kernel_key_normalizes_kernel_name(self):
        """Two autotuners with different kernel names but identical structure share a key."""
        a = self._make_autotuner_mock(name="triton_a", src="def triton_a():\n    pass")
        b = self._make_autotuner_mock(name="triton_b", src="def triton_b():\n    pass")
        self.assertEqual(
            self._cache._caching_autotuner_instance_key(a),
            self._cache._caching_autotuner_instance_key(b),
        )

    def test_kernel_key_distinguishes_distinct_kernels(self):
        """Distinct (src, inductor, triton) triples must produce distinct keys
        even when their concatenated bytes happen to align — guards against
        the without-delimiters hash-collision class.
        """
        a = self._make_autotuner_mock(src="def triton_kernel(): a")
        a.inductor_meta = {}
        a.triton_meta = {}

        b = self._make_autotuner_mock(src="def triton_kernel():")
        b.inductor_meta = {"a": ""}
        b.triton_meta = {}

        self.assertNotEqual(
            self._cache._caching_autotuner_instance_key(a),
            self._cache._caching_autotuner_instance_key(b),
        )

    def test_get_or_create_launcher_reuses_existing_for_same_autotuner(self):
        autotuner = self._make_autotuner_mock()
        raw = self._make_raw_launcher(x_block=16)
        first, _ = self._cache.get_or_create_launcher(autotuner, raw)
        second, _ = self._cache.get_or_create_launcher(autotuner, raw)
        self.assertIs(first, second)

    def test_get_or_create_launcher_shared_across_identical_autotuners(self):
        """Two autotuners with the same kernel content share Launchers per config."""
        a = self._make_autotuner_mock()
        b = self._make_autotuner_mock()
        raw = self._make_raw_launcher(x_block=16)
        launcher_a, _ = self._cache.get_or_create_launcher(a, raw)
        launcher_b, _ = self._cache.get_or_create_launcher(b, raw)
        self.assertIs(launcher_a, launcher_b)
        self.assertEqual(len(self._cache._caching_autotuner_launcher_registry), 1)

    def test_get_or_create_launcher_distinct_for_distinct_config(self):
        autotuner = self._make_autotuner_mock()
        a, _ = self._cache.get_or_create_launcher(
            autotuner, self._make_raw_launcher(x_block=16)
        )
        b, _ = self._cache.get_or_create_launcher(
            autotuner, self._make_raw_launcher(x_block=32)
        )
        self.assertIsNot(a, b)

    def test_get_or_create_launcher_case_1_share_live_raw(self):
        """Case 1: pool already has a live raw → second autotuner gets a
        view onto that raw (not the new raw it just created)."""
        a = self._make_autotuner_mock()
        b = self._make_autotuner_mock()
        raw_a = self._make_raw_launcher(x_block=16, cache_hash="hash_a")
        raw_b = self._make_raw_launcher(x_block=16, cache_hash="hash_b")

        # Hold strong ref to raw_a so its weakref doesn't die.
        keepalive: list[object] = [raw_a]

        launcher_a, chosen_a = self._cache.get_or_create_launcher(a, raw_a)
        launcher_b, chosen_b = self._cache.get_or_create_launcher(b, raw_b)
        self.assertIs(launcher_a, launcher_b)
        # Case 1: second autotuner's chosen is a view, not the new raw.
        self.assertIsInstance(chosen_b, self._cache.RawLauncherView)
        # The view delegates dispatch to raw_a but exposes b's cache_hash.
        self.assertEqual(chosen_b.cache_hash, "hash_b")
        self.assertEqual(chosen_b.config, raw_a.config)
        # Sanity: keepalive prevents GC.
        del keepalive

    def test_get_or_create_launcher_case_3_set_when_raw_dead(self):
        """Case 3: pool has a Launcher with timing data but its raw is
        dead → attach the new raw (not a view)."""
        a = self._make_autotuner_mock()
        raw_a = self._make_raw_launcher(x_block=16, cache_hash="hash_a")
        launcher_a, chosen_a = self._cache.get_or_create_launcher(a, raw_a)
        # First go-round attaches raw_a directly (pool miss).
        self.assertIs(chosen_a, raw_a)

        # Drop raw_a; the Launcher's weakref dies.
        del raw_a
        del chosen_a
        gc.collect()
        self.assertIsNone(launcher_a.get_raw_launcher())

        # Second autotuner shows up; pool finds the dormant Launcher and
        # case 3 attaches the new raw.
        b = self._make_autotuner_mock()
        raw_b = self._make_raw_launcher(x_block=16, cache_hash="hash_b")
        launcher_b, chosen_b = self._cache.get_or_create_launcher(b, raw_b)
        self.assertIs(launcher_a, launcher_b)
        # Case 3: chosen is the new raw itself, not a view.
        self.assertIs(chosen_b, raw_b)

    def test_get_or_create_launcher_raises_when_obj_not_autotuner(self):
        """For unrecognized obj, get_or_create_launcher raises ValueError."""
        raw = self._make_raw_launcher(x_block=16)
        with self.assertRaises(ValueError):
            self._cache.get_or_create_launcher("not an autotuner", raw)
        self.assertEqual(len(self._cache._caching_autotuner_launcher_registry), 0)

    def test_get_or_create_launcher_view_exposes_per_autotuner_cache_hash(self):
        """The view wrapper overrides cache_hash so each sharing
        autotuner sees its own value, while delegating other attrs to
        the underlying real raw."""
        a = self._make_autotuner_mock()
        b = self._make_autotuner_mock()
        raw_a = self._make_raw_launcher(x_block=16, cache_hash="hash_a")
        raw_b = self._make_raw_launcher(x_block=16, cache_hash="hash_b")
        keepalive: list[object] = [raw_a]

        _, chosen_a = self._cache.get_or_create_launcher(a, raw_a)
        _, chosen_b = self._cache.get_or_create_launcher(b, raw_b)

        # First autotuner gets the real raw.
        self.assertIs(chosen_a, raw_a)
        self.assertEqual(chosen_a.cache_hash, "hash_a")
        # Second gets a view with its own cache_hash but real_raw's config.
        self.assertIsInstance(chosen_b, self._cache.RawLauncherView)
        self.assertEqual(chosen_b.cache_hash, "hash_b")
        self.assertEqual(chosen_b.config, raw_a.config)
        del keepalive

    def test_raw_launcher_view_pickle_round_trip(self):
        """``RawLauncherView`` must survive pickle round-trip with its
        per-autotuner ``cache_hash`` and ``found_by_coordesc`` overrides
        intact. Without ``__reduce__``/``__setstate__``, pickle would
        route through ``__getattr__`` to ``_real_raw`` and silently
        drop the overrides."""
        import pickle
        from torch._inductor.runtime.incremental._launcher import (
            RawLauncherView,
        )

        view = RawLauncherView(
            _PicklableRaw(config="cfg", cache_hash="underlying_hash"),
            cache_hash="my_hash",
        )
        view.found_by_coordesc = True

        restored = pickle.loads(pickle.dumps(view))
        self.assertIsInstance(restored, RawLauncherView)
        self.assertEqual(restored.cache_hash, "my_hash")
        self.assertTrue(restored.found_by_coordesc)
        self.assertEqual(restored.config, "cfg")
        # Dispatch still works.
        self.assertEqual(restored(), "ok")

    def test_get_or_create_launcher_thread_safe(self):
        """Concurrent get_or_create_launcher calls must produce exactly one
        Launcher per (kernel, config), not one per thread.
        """
        autotuner = self._make_autotuner_mock()
        raw = self._make_raw_launcher(x_block=16)

        n_threads = 8
        results: list[object] = []
        results_lock = threading.Lock()
        barrier = threading.Barrier(n_threads)

        def worker() -> None:
            barrier.wait()
            launcher, _ = self._cache.get_or_create_launcher(autotuner, raw)
            with results_lock:
                results.append(launcher)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), n_threads)
        self.assertTrue(all(r is results[0] for r in results))

    def test_lookup_launcher_miss_returns_none(self):
        autotuner = self._make_autotuner_mock()
        raw = self._make_raw_launcher(x_block=16)
        self.assertIsNone(self._cache.lookup_launcher(autotuner, raw.config))

    def test_lookup_launcher_hit_returns_existing(self):
        autotuner = self._make_autotuner_mock()
        raw = self._make_raw_launcher(x_block=16)
        launcher, _ = self._cache.get_or_create_launcher(autotuner, raw)
        # Sharing autotuner sees the same Launcher via lookup.
        other = self._make_autotuner_mock()
        looked_up = self._cache.lookup_launcher(other, raw.config)
        self.assertIs(looked_up, launcher)

    def test_lookup_launcher_returns_none_for_unrecognized_obj(self):
        raw = self._make_raw_launcher(x_block=16)
        self.assertIsNone(
            self._cache.lookup_launcher("not an autotuner", raw.config)
        )

    def test_lookup_launcher_does_not_mutate_pool(self):
        autotuner = self._make_autotuner_mock()
        raw = self._make_raw_launcher(x_block=16)
        self._cache.lookup_launcher(autotuner, raw.config)
        # Pool entry was created lazily for the autotuner key but no
        # Launcher was registered. The inner per-config dict is empty.
        from torch._inductor.runtime.incremental._launcher import (
            _caching_autotuner_instance_key,
        )
        key = _caching_autotuner_instance_key(autotuner)
        self.assertEqual(
            self._cache._caching_autotuner_launcher_registry.get(key, {}), {}
        )


class _CachingAutotunerInstanceKeyTest(TestCase):
    """Test stability of the per-kernel pool key across pickle and across
    attribute permutations. The key MUST be deterministic for the
    parent / worker / post-pickle instances of the same kernel to find
    the same pool entry; otherwise per-kernel ``Launcher`` sharing
    misses across autotuner reincarnations."""

    @staticmethod
    def _make_autotuner_mock(**overrides):
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner

        autotuner = MagicMock(spec=CachingAutotuner)
        autotuner.fn = MagicMock()
        autotuner.fn.__name__ = "triton_kernel"
        autotuner.fn.src = "def triton_kernel():..."
        autotuner.inductor_meta = {"a": 1}
        autotuner.triton_meta = {"b": 2}
        autotuner.size_hints = {"x": 16}
        autotuner.heuristic_type = "POINTWISE"
        autotuner.custom_kernel = False
        autotuner.mutated_arg_names = []
        autotuner.reset_to_zero_arg_names = []
        autotuner.optimize_mem = True
        for k, v in overrides.items():
            setattr(autotuner, k, v)
        return autotuner

    def test_key_distinguishes_size_hints(self):
        from torch._inductor.runtime.incremental._launcher import (
            _caching_autotuner_instance_key,
        )

        a = self._make_autotuner_mock(size_hints={"x": 16})
        b = self._make_autotuner_mock(size_hints={"x": 32})
        self.assertNotEqual(
            _caching_autotuner_instance_key(a),
            _caching_autotuner_instance_key(b),
        )

    def test_key_distinguishes_heuristic_type(self):
        from torch._inductor.runtime.incremental._launcher import (
            _caching_autotuner_instance_key,
        )

        a = self._make_autotuner_mock(heuristic_type="POINTWISE")
        b = self._make_autotuner_mock(heuristic_type="REDUCTION")
        self.assertNotEqual(
            _caching_autotuner_instance_key(a),
            _caching_autotuner_instance_key(b),
        )

    def test_key_distinguishes_custom_kernel(self):
        from torch._inductor.runtime.incremental._launcher import (
            _caching_autotuner_instance_key,
        )

        a = self._make_autotuner_mock(custom_kernel=False)
        b = self._make_autotuner_mock(custom_kernel=True)
        self.assertNotEqual(
            _caching_autotuner_instance_key(a),
            _caching_autotuner_instance_key(b),
        )

    def test_key_distinguishes_mutated_arg_names(self):
        from torch._inductor.runtime.incremental._launcher import (
            _caching_autotuner_instance_key,
        )

        a = self._make_autotuner_mock(mutated_arg_names=[])
        b = self._make_autotuner_mock(mutated_arg_names=["x"])
        self.assertNotEqual(
            _caching_autotuner_instance_key(a),
            _caching_autotuner_instance_key(b),
        )

    def test_key_distinguishes_reset_to_zero_arg_names(self):
        from torch._inductor.runtime.incremental._launcher import (
            _caching_autotuner_instance_key,
        )

        a = self._make_autotuner_mock(reset_to_zero_arg_names=[])
        b = self._make_autotuner_mock(reset_to_zero_arg_names=["x"])
        self.assertNotEqual(
            _caching_autotuner_instance_key(a),
            _caching_autotuner_instance_key(b),
        )

    def test_key_distinguishes_optimize_mem(self):
        from torch._inductor.runtime.incremental._launcher import (
            _caching_autotuner_instance_key,
        )

        a = self._make_autotuner_mock(optimize_mem=True)
        b = self._make_autotuner_mock(optimize_mem=False)
        self.assertNotEqual(
            _caching_autotuner_instance_key(a),
            _caching_autotuner_instance_key(b),
        )

    def test_key_stable_across_two_equivalent_instances(self):
        """Two distinct autotuner objects with identical content
        produce the same key — required for stage-2 stash drain to
        work across the parent → worker → post-pickle plugin instance
        boundary."""
        from torch._inductor.runtime.incremental._launcher import (
            _caching_autotuner_instance_key,
        )

        a = self._make_autotuner_mock()
        b = self._make_autotuner_mock()
        self.assertEqual(
            _caching_autotuner_instance_key(a),
            _caching_autotuner_instance_key(b),
        )


class _StateLifecycleTest(TestCase):
    """Cover hooks (pre_launch, post_launch) and resolver-thread
    exception handling that aren't exercised by the unit tests."""

    def test_pre_launch_and_post_launch_hooks_fire_in_order(self):
        order: list[str] = []

        def pre(launcher, *args, stream, **kwargs):
            order.append("pre")

        def post():
            order.append("post")

        launcher = _make_launcher("a")
        state = IncrementalAutotuneState(pre_launch_fn=pre, post_launch_fn=post)
        state.add_launcher(launcher, launcher.get_raw_launcher())
        # Drive a dispatch via the converged-fallback path: set cert
        # and call once.
        state._certified_launcher = launcher
        # Drain the queue so converged is True.
        state._queue.clear()
        state._provisional_subs.pop(launcher, None)
        state._convergence_callback_fired = True  # skip cb fire path
        state(stream=0)
        self.assertEqual(order, ["pre", "post"])

    def test_post_launch_fires_even_when_launcher_raises(self):
        order: list[str] = []

        def pre(launcher, *args, stream, **kwargs):
            order.append("pre")

        def post():
            order.append("post")

        # Launcher whose underlying raw raises.
        fn = MagicMock(side_effect=RuntimeError("boom"))
        launcher = Launcher(raw_launcher=fn)
        launcher._raw_launcher_keepalive = fn  # pyrefly: ignore
        state = IncrementalAutotuneState(pre_launch_fn=pre, post_launch_fn=post)
        state.add_launcher(launcher, launcher.get_raw_launcher())
        state._certified_launcher = launcher
        state._queue.clear()
        state._provisional_subs.pop(launcher, None)
        state._convergence_callback_fired = True
        with self.assertRaises(RuntimeError):
            state(stream=0)
        # post still fired despite the exception.
        self.assertEqual(order, ["pre", "post"])

    def test_invalid_configuration_loops_not_recurses(self):
        """Many launchers all raising 'invalid configuration' should
        loop (not recurse) so the stack doesn't blow up."""
        n = 200
        launchers = [_make_launcher(f"l{i}") for i in range(n)]
        for l in launchers:
            l._raw_launcher_keepalive.side_effect = RuntimeError(
                "invalid configuration"
            )
        # Use one good launcher at the end so we eventually return.
        good = _make_launcher("good")
        good._raw_launcher_keepalive.return_value = "ok"
        all_launchers = launchers + [good]
        state = IncrementalAutotuneState()
        for l in all_launchers:
            state.add_launcher(l, l.get_raw_launcher())
        # Set provisional manually so case-2 doesn't trigger
        # threshold-discards on every launcher pre-dispatch.
        # (n+1 launchers, all unsaturated, no threshold triggers.)
        result = state(stream=0)
        self.assertEqual(result, "ok")


if __name__ == "__main__":
    run_tests()
