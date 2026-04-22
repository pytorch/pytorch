# Owner(s): ["module: inductor"]

import queue
import threading
import time
from unittest.mock import MagicMock, patch

from torch._inductor.test_case import run_tests, TestCase


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


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
