# Owner(s): ["module: dynamo"]

from torch._dynamo.metrics_context import MetricsContext
from torch._dynamo.test_case import run_tests, TestCase


class TestMetricsContext(TestCase):
    def setUp(self):
        super().setUp()
        self.metrics = {}

    def _on_exit(self, start_ns, end_ns, metrics, exc_type, exc_value):
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

    def test_set_disallow_overwrite(self):
        """
        Validate set won't overwrite.
        """
        with MetricsContext(self._on_exit) as context:
            context.set("m1", 1)
            with self.assertRaisesRegex(RuntimeError, "already been set"):
                context.set("m1", 2)

        self.assertEqual(self.metrics, {"m1": 1})

    def test_update_disallow_overwrite(self):
        """
        Validate update won't overwite.
        """
        with MetricsContext(self._on_exit) as context:
            context.update({"m1": 1, "m2": 2})
            with self.assertRaisesRegex(RuntimeError, "already been set"):
                context.update({"m1": 7, "m3": 3})

        self.assertEqual(self.metrics, {"m1": 1, "m2": 2})

    def test_add_to_set(self):
        """
        Validate add_to_set.
        """
        with MetricsContext(self._on_exit) as context:
            context.add_to_set("m1", 1)
            context.add_to_set("m1", 2)
            context.add_to_set("m2", 3)
            context.add_to_set("m2", 4)

        self.assertEqual(self.metrics, {"m1": {1, 2}, "m2": {3, 4}})
        self.assertTrue(isinstance(self.metrics["m1"], set))
        self.assertTrue(isinstance(self.metrics["m2"], set))

    def test_set_key_value(self):
        with MetricsContext(self._on_exit) as context:
            context.set_key_value("feature_usage", "k", True)
            # Overrides allowed
            context.set_key_value("feature_usage", "k2", True)
            context.set_key_value("feature_usage", "k2", False)

        self.assertEqual(self.metrics, {"feature_usage": {"k": True, "k2": False}})


if __name__ == "__main__":
    run_tests()
