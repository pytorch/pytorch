# Owner(s): ["module: dynamo"]

import time

from torch._dynamo.metrics_context import MetricsContext
from torch._dynamo.test_case import run_tests, TestCase


class TestMetricsContext(TestCase):
    def setUp(self):
        super().setUp()
        self.metrics = {}

    def _on_exit(self, metrics, exc_type, exc_value):
        # Save away the metrics to be validated in the test.
        self.metrics = metrics.copy()

    def test_context_exists(self):
        """
        Setting a value with entering the context should raise.
        """
        context = MetricsContext(self._on_exit)
        with self.assertRaisesRegex(RuntimeError, "outside of a MetricsContext"):
            context.increment("metric", 1)

        with self.assertRaisesRegex(RuntimeError, "outside of a MetricsContext"):
            context.set("metric", 1)

        with self.assertRaisesRegex(RuntimeError, "outside of a MetricsContext"):
            context.set_once("metric", 1)

        with self.assertRaisesRegex(RuntimeError, "outside of a MetricsContext"):
            context.update({"metric", 1})

    def test_nested_context(self):
        """
        Entering the context twice should raise.
        """
        with self.assertRaisesRegex(RuntimeError, "Cannot re-enter"):
            context = MetricsContext(self._on_exit)
            with context:
                with context:
                    pass

    def test_set(self):
        """
        Validate various ways to set metrics.
        """
        with MetricsContext(self._on_exit) as context:
            context.set("m1", 1)
            context.set_once("m2", 2)
            context.update({"m3": 3, "m4": 4})

        self.assertEqual(self.metrics, {"m1": 1, "m2": 2, "m3": 3, "m4": 4})

    def test_timed(self):
        """
        Validate the timed contextmanager.
        """
        context = MetricsContext(self._on_exit)

        # Make sure we count recursive calls correctly. We should account
        # for the full time of execution, but not double count.
        def fn(n=10):
            with context.timed("runtime_ms"):
                if n == 3:
                    time.sleep(0.7)
                elif n == 2:
                    time.sleep(0.3)
                if n > 0:
                    fn(n - 1)

        with context:
            fn()

        runtime_ms = self.metrics.get("runtime_ms", None)
        self.assertTrue(runtime_ms is not None)
        self.assertTrue(runtime_ms > 900)
        self.assertTrue(runtime_ms < 1100)

    def test_timed_decorator(self):
        """
        Validate the timed contextmanager as a decorator.
        """
        context = MetricsContext(self._on_exit)

        # Make sure we count recursive calls correctly. We should account
        # for the full time of execution, but not double count.
        @context.timed("runtime_us")
        def fn(n=10):
            if n > 0:
                fn(n - 1)
            if n == 3:
                time.sleep(0.7)
            elif n == 2:
                time.sleep(0.3)

        with context:
            fn()

        runtime_us = self.metrics.get("runtime_us", None)
        self.assertTrue(runtime_us is not None)
        self.assertTrue(runtime_us > 900000)
        self.assertTrue(runtime_us < 1100000)


if __name__ == "__main__":
    run_tests()
