# Owner(s): ["oncall: distributed"]

import threading
import time
import unittest
from datetime import timedelta

import torch
from torch.testing._internal.common_utils import run_tests, skipIfRocm, TestCase


try:
    import torch.distributed._watchdog as watchdog
    from torch._C import _distributed_c10d_watchdog as _watchdog

    _HAS_WATCHDOG = True
except ImportError:
    _HAS_WATCHDOG = False

# The process-wide watchdog polls every 1s; tests that depend on the poll use a
# local instance with a short interval so they stay fast.
_FAST_POLL = timedelta(milliseconds=5)


@unittest.skipUnless(_HAS_WATCHDOG, "watchdog requires the libuv backend")
class WatchdogTest(TestCase):
    def _fast_watchdog(self):
        return _watchdog._Watchdog(poll_interval=_FAST_POLL)

    def test_singleton_is_stable(self):
        self.assertIs(
            _watchdog._Watchdog._singleton(), _watchdog._Watchdog._singleton()
        )

    def test_cpu_timeout_fires_on_overrun(self):
        fired = threading.Event()
        with watchdog.cpu_timeout(fired.set, timedelta(milliseconds=50)):
            # time.sleep releases the GIL so the callback thread can run.
            time.sleep(0.5)
        self.assertTrue(fired.is_set())

    def test_cpu_timeout_cancelled_on_clean_exit(self):
        fired = threading.Event()
        with watchdog.cpu_timeout(fired.set, timedelta(seconds=10)):
            time.sleep(0.05)
        # The timer was cancelled on exit, so the callback must not fire.
        self.assertFalse(fired.wait(timeout=0.3))

    def test_cpu_timeout_callback_exception_is_swallowed(self):
        def boom():
            raise RuntimeError("callback error")

        # A raising callback must not crash the watchdog loop thread, and the
        # watchdog must keep working afterwards.
        with watchdog.cpu_timeout(boom, timedelta(milliseconds=50)):
            time.sleep(0.3)

        fired = threading.Event()
        with watchdog.cpu_timeout(fired.set, timedelta(milliseconds=50)):
            time.sleep(0.5)
        self.assertTrue(fired.is_set())

    def test_local_instance_handle_cancel(self):
        wd = _watchdog._Watchdog()
        fired = threading.Event()
        handle = wd.cpu_timeout(timedelta(seconds=10), fired.set)
        handle.cancel()
        self.assertFalse(fired.wait(timeout=0.3))

        # A fresh timeout on the same instance still fires.
        fired2 = threading.Event()
        wd.cpu_timeout(timedelta(milliseconds=50), fired2.set)
        self.assertTrue(fired2.wait(timeout=2.0))

    def test_no_active_stream_timeouts(self):
        wd = _watchdog._Watchdog()
        self.assertEqual(wd.num_active_stream_timeouts(), 0)

    @skipIfRocm
    def test_stream_timeout_fires_when_busy(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        wd = self._fast_watchdog()
        fired = threading.Event()
        # Keep the stream busy well past the timeout.
        torch.cuda._sleep(500_000_000)
        wd.stream_timeout(timedelta(milliseconds=50), fired.set)
        self.assertTrue(fired.wait(timeout=10.0))
        torch.cuda.synchronize()

    @skipIfRocm
    def test_stream_timeout_completes_without_firing(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        wd = self._fast_watchdog()
        fired = threading.Event()
        torch.cuda._sleep(1_000_000)
        wd.stream_timeout(timedelta(seconds=30), fired.set)
        torch.cuda.synchronize()
        self.assertFalse(fired.wait(timeout=0.5))

    @skipIfRocm
    def test_op_timeout_fires_once_when_busy(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        wd = self._fast_watchdog()
        fired = threading.Event()
        count = [0]

        def on_timeout():
            count[0] += 1
            fired.set()

        # op_timeout returns the launch handle; cancel it once the work is
        # enqueued (as the guard does), leaving the completion check to fire.
        launch = wd.op_timeout(timedelta(milliseconds=50), on_timeout)
        torch.cuda._sleep(500_000_000)
        launch.cancel()
        self.assertTrue(fired.wait(timeout=10.0))
        torch.cuda.synchronize()
        # The fire-once wrapper means the combined checks fire the callback once.
        time.sleep(0.2)
        self.assertEqual(count[0], 1)

    @skipIfRocm
    def test_op_timeout_completes_without_firing(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        wd = self._fast_watchdog()
        fired = threading.Event()
        launch = wd.op_timeout(timedelta(seconds=30), fired.set)
        torch.cuda._sleep(1_000_000)
        launch.cancel()
        torch.cuda.synchronize()
        self.assertFalse(fired.wait(timeout=0.5))

    @skipIfRocm
    def test_stream_timeout_recycles_events(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        wd = self._fast_watchdog()
        # Many short stream timeouts in a row exercise the event cache's
        # acquire/release/reuse path; they should all complete and be reaped.
        for _ in range(50):
            torch.cuda._sleep(1_000_000)
            wd.stream_timeout(timedelta(seconds=30), lambda: None)
        torch.cuda.synchronize()
        deadline = time.time() + 5.0
        while wd.num_active_stream_timeouts() > 0 and time.time() < deadline:
            time.sleep(0.02)
        self.assertEqual(wd.num_active_stream_timeouts(), 0)

    @skipIfRocm
    def test_global_stream_and_op_timeout(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        # Exercise the process-wide helpers (which use the 1s-poll singleton); a
        # long timeout means no callback fires. stream_timeout is a plain call
        # after enqueuing; op_timeout is a guard.
        torch.cuda._sleep(1_000_000)
        watchdog.stream_timeout(lambda: None, timedelta(seconds=30))
        with watchdog.op_timeout(lambda: None, timedelta(seconds=30)):
            torch.cuda._sleep(1_000_000)
        torch.cuda.synchronize()


if __name__ == "__main__":
    run_tests()
