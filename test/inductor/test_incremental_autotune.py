# Owner(s): ["module: inductor"]

import threading
from unittest.mock import MagicMock, patch

from torch._inductor.runtime.incremental._launcher import Launcher
from torch._inductor.runtime.incremental._state import IncrementalAutotuneState
from torch._inductor.test_case import run_tests, TestCase


def _make_launcher(name: str = "launcher") -> Launcher:
    fn = MagicMock()
    fn.return_value = f"result_{name}"
    return Launcher(fn=fn, config=f"config:{name}")


def _make_state(launchers: list[Launcher], **kwargs: object) -> IncrementalAutotuneState:
    """Create an IncrementalAutotuneState with fresh stats for testing."""
    state = IncrementalAutotuneState(**kwargs)
    state.init_fresh(launchers)
    return state


class LauncherTest(TestCase):
    def test_timing_empty(self):
        launcher = _make_launcher()
        self.assertEqual(launcher.timing, float("inf"))

    def test_sample_count(self):
        launcher = _make_launcher()
        launcher._add_timing(1.0)
        launcher._add_timing(2.0)
        self.assertEqual(launcher.sample_count, 2)

    def test_add_timing_sorted(self):
        launcher = _make_launcher()
        launcher._add_timing(3.0)
        launcher._add_timing(1.0)
        launcher._add_timing(2.0)
        self.assertEqual(launcher._timings, [1.0, 2.0, 3.0])

    def test_timing_median_odd(self):
        launcher = _make_launcher()
        for v in [5.0, 1.0, 3.0]:
            launcher._add_timing(v)
        self.assertAlmostEqual(launcher.timing, 3.0)

    def test_timing_median_even(self):
        launcher = _make_launcher()
        for v in [4.0, 2.0]:
            launcher._add_timing(v)
        self.assertAlmostEqual(launcher.timing, 3.0)

    def test_timing_median_ignores_outliers(self):
        launcher = _make_launcher()
        for v in [1.0, 2.0, 3.0, 4.0, 100.0]:
            launcher._add_timing(v)
        self.assertAlmostEqual(launcher.timing, 3.0)

    def test_dispatch_count_increments_on_call(self):
        launcher = _make_launcher()
        self.assertEqual(launcher.dispatch_count, 0)
        launcher(stream=0)
        self.assertEqual(launcher.dispatch_count, 1)
        launcher(stream=0)
        self.assertEqual(launcher.dispatch_count, 2)

    def test_call_delegates_to_fn(self):
        launcher = _make_launcher("test")
        result = launcher(1, 2, stream=0)
        self.assertEqual(result, "result_test")
        launcher._fn.assert_called_once_with(1, 2, stream=0)

    def test_metadata(self):
        launcher = Launcher(fn=lambda: None, key="value")
        self.assertEqual(launcher.metadata["key"], "value")


class IncrementalAutotuneStateTest(TestCase):
    def test_init_fresh_queue(self):
        a = _make_launcher("a")
        b = _make_launcher("b")
        state = _make_state([a, b])
        self.assertIs(state._queue[0], a)
        self.assertIs(state._queue[1], b)
        self.assertEqual(len(state._queue), 2)
        state.shutdown()

    def test_next_launcher_skips_max_dispatched(self):
        a = _make_launcher("a")
        b = _make_launcher("b")
        state = _make_state([a, b])
        with a._lock:
            a.dispatch_count = 999
        self.assertIs(state._next_launcher(), b)
        state.shutdown()

    def test_next_launcher_returns_none_when_exhausted(self):
        a = _make_launcher("a")
        state = _make_state([a])
        with a._lock:
            a.dispatch_count = 999
        self.assertIsNone(state._next_launcher())
        state.shutdown()

    def test_should_skip_threshold(self):
        best = _make_launcher("best")
        slow = _make_launcher("slow")
        state = _make_state([best, slow])
        state.best_launcher = best
        best._add_timing(1.0)
        for _ in range(5):
            slow._add_timing(2.0)
        self.assertTrue(state._should_skip(slow))
        state.shutdown()

    def test_should_skip_below_threshold(self):
        best = _make_launcher("best")
        candidate = _make_launcher("candidate")
        state = _make_state([best, candidate])
        state.best_launcher = best
        best._add_timing(1.0)
        for _ in range(5):
            candidate._add_timing(1.1)
        self.assertFalse(state._should_skip(candidate))
        state.shutdown()

    def test_should_skip_not_enough_samples(self):
        best = _make_launcher("best")
        slow = _make_launcher("slow")
        state = _make_state([best, slow])
        state.best_launcher = best
        best._add_timing(1.0)
        slow._add_timing(10.0)
        self.assertFalse(state._should_skip(slow))
        state.shutdown()

    def test_converged_true_single_launcher_pending_events(self):
        launcher = _make_launcher()
        state = _make_state([launcher])
        state.best_launcher = launcher
        state._queue.clear()
        state._queue.clear()
        state._pending_events = 1
        self.assertTrue(state.converged)
        state.shutdown()

    def test_converged_false_pending_events_multiple_launchers(self):
        a = _make_launcher("a")
        b = _make_launcher("b")
        state = _make_state([a, b])
        state.best_launcher = a
        state._queue.clear()
        state._queue.clear()
        state._pending_events = 1
        self.assertFalse(state.converged)
        state.shutdown()

    def test_converged_false_nonempty_deque(self):
        a = _make_launcher("a")
        b = _make_launcher("b")
        state = _make_state([a, b])
        state.best_launcher = a
        self.assertFalse(state.converged)
        state.shutdown()

    def test_converged_true(self):
        launcher = _make_launcher()
        state = _make_state([launcher])
        state.best_launcher = launcher
        state._queue.clear()
        state._queue.clear()
        self.assertTrue(state.converged)
        state.shutdown()

    def test_resolve_timing_updates_best(self):
        a = _make_launcher("a")
        b = _make_launcher("b")
        state = _make_state([a, b])
        state._pending_events = 2
        a._add_timing(5.0)
        state.best_launcher = a

        b.resolve_timing(2.0)
        state.decrement_pending()

        self.assertIs(state.best_launcher, b)
        self.assertAlmostEqual(state.best_timing, 2.0)
        self.assertEqual(state._pending_events, 1)
        state.shutdown()

    def test_resolve_timing_slower_does_not_replace_best(self):
        a = _make_launcher("a")
        b = _make_launcher("b")
        state = _make_state([a, b])
        state._pending_events = 1
        a._add_timing(1.0)
        state.best_launcher = a

        b.resolve_timing(5.0)
        state.decrement_pending()

        self.assertIs(state.best_launcher, a)
        state.shutdown()

    def test_init_fresh_multiple_launchers(self):
        launchers = [_make_launcher(f"l{i}") for i in range(3)]
        state = _make_state(launchers)
        self.assertEqual(len(state._queue), 3)
        state.shutdown()

    def test_dispatch_round_robin_and_convergence(self):
        """dispatch() iterates launchers round-robin and calls on_convergence when done."""
        converged = threading.Event()
        converged_launcher: list[Launcher | None] = [None]

        def on_convergence(state: IncrementalAutotuneState) -> None:
            converged_launcher[0] = state.best_launcher
            converged.set()

        a = _make_launcher("a")
        b = _make_launcher("b")

        state = _make_state([a, b], on_convergence_fn=on_convergence)

        mock_event = MagicMock()

        def fake_put(
            s: IncrementalAutotuneState,
            launcher: Launcher,
            _start: object,
            _end: object,
        ) -> None:
            launcher.resolve_timing(1.0)
            s.decrement_pending()

        with patch(
            "torch._inductor.runtime.incremental._state._MAX_SAMPLES_PER_LAUNCHER", 2
        ), patch(
            "torch._inductor.runtime.incremental._state._FORCED_TIMING_ROUNDS", 1
        ), patch(
            "torch._inductor.runtime.incremental._state._SAMPLING_RATE", 1
        ), patch(
            "torch._inductor.runtime.incremental._state.torch.cuda.Event",
            return_value=mock_event,
        ), patch(
            "torch._inductor.runtime.incremental._state.submit_event"
        ) as mock_submit:
            mock_submit.side_effect = fake_put
            for _ in range(4):
                state.dispatch(stream=0)
            state.dispatch(stream=0)

        self.assertTrue(converged.is_set())
        self.assertIsNotNone(converged_launcher[0])

    def test_on_cleanup_called_on_del(self):
        cleanup_called = [False]

        def on_cleanup(state: IncrementalAutotuneState) -> None:
            cleanup_called[0] = True

        state = IncrementalAutotuneState(on_cleanup_fn=on_cleanup)
        state.__del__()
        self.assertTrue(cleanup_called[0])

    def test_init_fresh_seeds_best_from_existing_timings(self):
        """A reused launcher with prior samples is immediately ``best_launcher``."""
        fast = _make_launcher("fast")
        slow = _make_launcher("slow")
        fast._add_timing(1.0)
        slow._add_timing(5.0)

        state = _make_state([slow, fast])
        self.assertIs(state.best_launcher, fast)
        state.shutdown()

    def test_should_skip_max_dispatches(self):
        """Launcher with max dispatches is skipped."""
        launcher = _make_launcher()
        state = _make_state([launcher])
        self.assertFalse(state._should_skip(launcher))
        with launcher._lock:
            launcher.dispatch_count = 999
        self.assertTrue(state._should_skip(launcher))
        state.shutdown()

    def test_dispatch_drops_invalid_config_and_retries(self):
        """RuntimeError("invalid configuration ...") prunes the launcher and retries."""
        bad = _make_launcher("bad")
        good = _make_launcher("good")
        bad._fn.side_effect = RuntimeError(
            "invalid configuration argument from CUDA launch"
        )
        state = _make_state([bad, good])

        with patch(
            "torch._inductor.runtime.incremental._state.torch.cuda.Event",
            return_value=MagicMock(),
        ), patch("torch._inductor.runtime.incremental._state.submit_event"):
            result = state.dispatch(stream=0)

        self.assertEqual(result, "result_good")
        self.assertNotIn(bad, state._launchers)
        state.shutdown()

    def test_dispatch_propagates_background_error(self):
        """Resolver-side errors stamp the state and surface on the next dispatch."""
        launcher = _make_launcher()
        state = _make_state([launcher])
        sentinel = RuntimeError("resolver failed")
        state.set_background_error(sentinel)
        with self.assertRaises(RuntimeError) as ctx:
            state.dispatch(stream=0)
        self.assertIs(ctx.exception, sentinel)
        state.shutdown()


class ResolverTest(TestCase):
    def test_submit_event_puts_then_ensures_daemon(self):
        """Put-then-ensure ordering: item is visible when daemon checks empty()."""
        from torch._inductor.runtime.incremental import _resolver

        with patch.object(_resolver, "_ensure_daemon") as mock_ensure, patch.object(
            _resolver._global_event_queue, "put"
        ) as mock_put:
            call_order = []
            mock_put.side_effect = lambda *a, **k: call_order.append("put")
            mock_ensure.side_effect = lambda: call_order.append("ensure")

            mock_state = MagicMock()
            mock_launcher = MagicMock()
            _resolver.submit_event(
                mock_state, mock_launcher, MagicMock(), MagicMock()
            )

        self.assertEqual(call_order, ["put", "ensure"])


class CacheTest(TestCase):
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
        return autotuner

    @staticmethod
    def _make_raw_launcher(x_block: int):
        raw = MagicMock()
        raw.config = MagicMock(kwargs={"X": x_block}, num_warps=4, num_stages=1)
        raw.cache_hash = f"hash:{x_block}"
        return raw

    def test_kernel_key_none_when_fn_has_no_src(self):
        from torch._inductor.runtime.incremental import _cache

        autotuner = self._make_autotuner_mock()
        del autotuner.fn.src
        self.assertIsNone(_cache._caching_autotuner_kernel_key(autotuner))

    def test_kernel_key_normalizes_kernel_name(self):
        """Two autotuners with different kernel names but identical structure share a key."""
        from torch._inductor.runtime.incremental import _cache

        a = self._make_autotuner_mock(name="triton_a", src="def triton_a():\n    pass")
        b = self._make_autotuner_mock(name="triton_b", src="def triton_b():\n    pass")
        self.assertEqual(
            _cache._caching_autotuner_kernel_key(a),
            _cache._caching_autotuner_kernel_key(b),
        )

    def test_get_launcher_pool_shared_across_identical_autotuners(self):
        from torch._inductor.runtime.incremental import _cache

        a = self._make_autotuner_mock()
        b = self._make_autotuner_mock()
        with patch.object(_cache, "_caching_autotuner_registry", {}):
            self.assertIs(_cache.get_launcher_pool(a), _cache.get_launcher_pool(b))

    def test_get_launcher_pool_none_for_non_autotuner(self):
        from torch._inductor.runtime.incremental import _cache

        with patch.object(_cache, "_caching_autotuner_registry", {}):
            self.assertIsNone(_cache.get_launcher_pool("not an autotuner"))

    def test_get_or_create_launcher_reuses_existing(self):
        from torch._inductor.runtime.incremental import _cache
        from torch._inductor.runtime.incremental._launcher import (
            Launcher as RealLauncher,
        )

        pool: dict[object, RealLauncher] = {}
        raw = self._make_raw_launcher(x_block=16)
        first = _cache.get_or_create_launcher(pool, raw, RealLauncher)
        second = _cache.get_or_create_launcher(pool, raw, RealLauncher)
        self.assertIs(first, second)
        self.assertEqual(len(pool), 1)

    def test_get_or_create_launcher_creates_new_for_distinct_config(self):
        from torch._inductor.runtime.incremental import _cache
        from torch._inductor.runtime.incremental._launcher import (
            Launcher as RealLauncher,
        )

        pool: dict[object, RealLauncher] = {}
        a = _cache.get_or_create_launcher(
            pool, self._make_raw_launcher(x_block=16), RealLauncher
        )
        b = _cache.get_or_create_launcher(
            pool, self._make_raw_launcher(x_block=32), RealLauncher
        )
        self.assertIsNot(a, b)
        self.assertEqual(len(pool), 2)


if __name__ == "__main__":
    run_tests()
