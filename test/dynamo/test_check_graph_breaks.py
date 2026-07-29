# Owner(s): ["module: dynamo"]
import importlib.util
import os
import unittest

from torch.testing._internal.common_utils import run_tests, TestCase


try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def _load_check_graph_breaks():
    # check_graph_breaks.py lives in benchmarks/dynamo, which is not a package,
    # so load it by path relative to the repo root (two dirs up from test/dynamo).
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "benchmarks",
        "dynamo",
        "check_graph_breaks.py",
    )
    spec = importlib.util.spec_from_file_location("check_graph_breaks", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipIf(not HAS_PANDAS, "check_graph_breaks requires pandas")
class CheckGraphBreaksTest(TestCase):
    def setUp(self):
        super().setUp()
        self.cgb = _load_check_graph_breaks()

    def _failed(self, actual, expected):
        failed, _msg = self.cgb.check_graph_breaks(
            pd.DataFrame(actual), pd.DataFrame(expected), "inductor_x.csv"
        )
        return set(failed)

    def test_classify_direction(self):
        classify = self.cgb._classify
        # lower_is_better (graph breaks, eager fallbacks)
        self.assertEqual(classify(3, 3, lower_is_better=True), "PASS")
        self.assertEqual(classify(6, 3, lower_is_better=True), "FAIL")
        self.assertEqual(classify(1, 3, lower_is_better=True), "IMPROVED")
        # higher_is_better (ops/graphs captured): capturing less is the regression
        self.assertEqual(classify(80, 80, lower_is_better=False), "PASS")
        self.assertEqual(classify(40, 80, lower_is_better=False), "FAIL")
        self.assertEqual(classify(90, 80, lower_is_better=False), "IMPROVED")

    def test_legacy_graph_breaks_only_baseline(self):
        # Baselines predating the coverage metrics only have graph_breaks; the
        # extra columns in the actual csv must be ignored (behavior unchanged).
        actual = [
            {
                "name": "pass",
                "unique_graphs": 2,
                "graph_breaks": 3,
                "calls_captured": 50,
            },
            {
                "name": "more_breaks",
                "unique_graphs": 2,
                "graph_breaks": 6,
                "calls_captured": 50,
            },
            {
                "name": "fewer_breaks",
                "unique_graphs": 2,
                "graph_breaks": 1,
                "calls_captured": 50,
            },
            {
                "name": "eager_failed",
                "unique_graphs": 0,
                "graph_breaks": 0,
                "calls_captured": 0,
            },
        ]
        expected = [
            {"name": "pass", "graph_breaks": 3},
            {"name": "more_breaks", "graph_breaks": 3},
            {"name": "fewer_breaks", "graph_breaks": 3},
            {"name": "eager_failed", "graph_breaks": 3},
        ]
        # only the extra-break model regresses; eager_failed (0 graphs) is skipped
        self.assertEqual(self._failed(actual, expected), {"more_breaks"})

    def test_coverage_metrics_directions(self):
        base = {
            "graph_breaks": 3,
            "calls_captured": 80,
            "unique_graphs": 5,
            "fallbacks_to_eager": 2,
        }
        actual = [
            {"name": "ops_drop", **base, "calls_captured": 40},
            {"name": "ops_gain", **base, "calls_captured": 90},
            {"name": "graphs_drop", **base, "unique_graphs": 3},
            {"name": "fallback_up", **base, "fallbacks_to_eager": 9},
            {"name": "all_good", **base},
        ]
        expected = [{"name": r["name"], **base} for r in actual]
        # regressions: fewer ops, fewer graphs, more eager fallbacks.
        # ops_gain (more ops captured) is an improvement, not a failure.
        self.assertEqual(
            self._failed(actual, expected),
            {"ops_drop", "graphs_drop", "fallback_up"},
        )

    def test_missing_baseline_entry(self):
        # A model absent from the baseline is reported (and counts as improved,
        # i.e. a non-clean check), not silently ignored.
        actual = [{"name": "newmodel", "unique_graphs": 2, "graph_breaks": 4}]
        expected = [{"name": "othermodel", "graph_breaks": 3}]
        failed, msg = self.cgb.check_graph_breaks(
            pd.DataFrame(actual), pd.DataFrame(expected), "inductor_x.csv"
        )
        self.assertTrue(failed)  # truthy: non-clean (improved list non-empty)
        self.assertIn("update_expected.py", msg)


if __name__ == "__main__":
    run_tests()
