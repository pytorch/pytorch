# Owner(s): ["module: dynamo"]

from torch._dynamo.metrics_context import MetricsContext
from torch._dynamo.test_case import run_tests, TestCase


class TestMetricsContext(TestCase):
    def setUp(self):
        super().setUp()
        self.metrics = {}

    def _on_exit(self, metrics):
        # Save away the metrics to be validated in the test.
        self.metrics = metrics.copy()

    def test_context_exists(self):
        """
        Setting a value without entering the context should raise.
        """
        context = MetricsContext(self._on_exit)
        with self.assertRaisesRegex(RuntimeError, "outside of a MetricsContext"):
            context.increment("m", 1)

        with self.assertRaisesRegex(RuntimeError, "outside of a MetricsContext"):
            context.set("m", 1)

        with self.assertRaisesRegex(RuntimeError, "outside of a MetricsContext"):
            context.update({"m", 1})

    def test_nested_context(self):
        """
        Only the outermost context should get an on_exit call, and it should
        include everything.
        """
        context = MetricsContext(self._on_exit)
        with context:
            with context:
                context.set("m1", 1)
            self.assertEqual(self.metrics, {})
            context.set("m2", 2)
        self.assertEqual(self.metrics, {"m1": 1, "m2": 2})

    def test_set(self):
        """
        Validate various ways to set metrics.
        """
        with MetricsContext(self._on_exit) as context:
            context.set("m1", 1)
            context.set("m2", 2)
            context.update({"m3": 3, "m4": 4})

        self.assertEqual(self.metrics, {"m1": 1, "m2": 2, "m3": 3, "m4": 4})

    def test_set_overwrite(self):
        """
        Validate the set overwrite flag.
        """
        with MetricsContext(self._on_exit) as context:
            context.set("m1", 1)
            with self.assertRaisesRegex(RuntimeError, "already been set"):
                context.set("m1", 2)

        self.assertEqual(self.metrics, {"m1": 1})

        with MetricsContext(self._on_exit) as context:
            context.set("m1", 1, overwrite=False)
            context.set("m1", 2, overwrite=True)

        self.assertEqual(self.metrics, {"m1": 2})

    def test_update_overwrite(self):
        """
        Validate the update overwrite flag.
        """
        with MetricsContext(self._on_exit) as context:
            context.update({"m1": 1, "m2": 2})
            with self.assertRaisesRegex(RuntimeError, "already been set"):
                context.update({"m1": 7, "m3": 3})

        self.assertEqual(self.metrics, {"m1": 1, "m2": 2})

        with MetricsContext(self._on_exit) as context:
            context.update({"m1": 1, "m2": 2}, overwrite=False)
            context.update({"m1": 7, "m3": 3}, overwrite=True)

        self.assertEqual(self.metrics, {"m1": 7, "m2": 2, "m3": 3})


if __name__ == "__main__":
    run_tests()
