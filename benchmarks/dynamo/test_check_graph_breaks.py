"""Tests for check_graph_breaks.py.

These tests intentionally do not import torch so they can run without a full
PyTorch build.  All they need is pandas.
"""
import importlib.util
import unittest
from pathlib import Path

import pandas as pd

# Import check_graph_breaks as a plain module so we don't pull in torch via
# the benchmarks.dynamo package __init__.
_module_path = Path(__file__).parent / "check_graph_breaks.py"
_spec_obj = importlib.util.spec_from_file_location("_check_graph_breaks", _module_path)
_mod = importlib.util.module_from_spec(_spec_obj)
_spec_obj.loader.exec_module(_mod)
check_graph_breaks = _mod.check_graph_breaks


def _actual(rows):
    """Build an 'actual results' DataFrame with all tracked columns."""
    return pd.DataFrame(
        rows,
        columns=("name", "accuracy", "graph_breaks", "recompiles", "fallback_to_eager", "unique_graphs"),
    )


def _expected(rows, include_recompiles=True, include_fallback=True):
    """Build an 'expected' DataFrame, optionally omitting new columns."""
    cols = ["name", "accuracy", "graph_breaks"]
    if include_recompiles:
        cols.append("recompiles")
    if include_fallback:
        cols.append("fallback_to_eager")
    return pd.DataFrame(rows, columns=cols)


class TestCheckGraphBreaks(unittest.TestCase):
    def test_pass_when_all_metrics_unchanged(self):
        actual = _actual([("modelA", "pass", 2, 1, 0, 3)])
        expected = _expected([("modelA", "pass", 2, 1, 0)])
        failed, _ = check_graph_breaks(actual, expected, "inductor_test.csv")
        self.assertFalse(failed)

    def test_fail_on_graph_break_regression(self):
        actual = _actual([("modelA", "pass", 5, 0, 0, 3)])
        expected = _expected([("modelA", "pass", 2, 0, 0)])
        failed, msg = check_graph_breaks(actual, expected, "inductor_test.csv")
        self.assertTrue(failed)
        self.assertIn("modelA", msg)

    def test_fail_on_recompile_regression_graph_breaks_unchanged(self):
        # Regression test for https://github.com/pytorch/pytorch/issues/113040
        # Example 2: graph_breaks stay at 0 but recompiles increase.
        # The old code returned PASS because graph_breaks == expected_graph_breaks
        # and never checked recompiles.
        actual = _actual([("modelB", "pass", 0, 3, 0, 2)])
        expected = _expected([("modelB", "pass", 0, 0, 0)])
        failed, msg = check_graph_breaks(actual, expected, "inductor_test.csv")
        self.assertTrue(failed)
        self.assertIn("modelB", msg)

    def test_fail_on_fallback_regression_other_metrics_unchanged(self):
        actual = _actual([("modelC", "pass", 0, 0, 2, 2)])
        expected = _expected([("modelC", "pass", 0, 0, 0)])
        failed, msg = check_graph_breaks(actual, expected, "inductor_test.csv")
        self.assertTrue(failed)
        self.assertIn("modelC", msg)

    def test_improved_when_recompiles_drop(self):
        actual = _actual([("modelD", "pass", 2, 0, 0, 3)])
        expected = _expected([("modelD", "pass", 2, 5, 0)])
        failed, msg = check_graph_breaks(actual, expected, "inductor_test.csv")
        self.assertTrue(failed)  # non-empty improved list is truthy
        self.assertIn("modelD", msg)
        self.assertIn("Improvement", msg)

    def test_improved_when_fallback_drops(self):
        actual = _actual([("modelE", "pass", 0, 0, 1, 2)])
        expected = _expected([("modelE", "pass", 0, 0, 4)])
        _, msg = check_graph_breaks(actual, expected, "inductor_test.csv")
        self.assertIn("Improvement", msg)

    def test_backward_compat_expected_without_recompiles(self):
        # Expected CSV from before recompile/fallback tracking; only graph_breaks.
        actual = _actual([("modelF", "pass", 2, 5, 3, 3)])
        expected = _expected([("modelF", "pass", 2)], include_recompiles=False, include_fallback=False)
        failed, _ = check_graph_breaks(actual, expected, "inductor_test.csv")
        self.assertFalse(failed)

    def test_flaky_model_regression_not_counted_as_failure(self):
        actual = _actual([("yolov3", "pass", 10, 0, 0, 3)])
        expected = _expected([("yolov3", "pass", 0, 0, 0)])
        failed, _ = check_graph_breaks(actual, expected, "inductor_test.csv")
        self.assertFalse(failed)


if __name__ == "__main__":
    unittest.main()
