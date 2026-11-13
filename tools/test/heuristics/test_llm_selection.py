from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))

from tools.testing.target_determination import determinator
from tools.testing.target_determination.heuristics import HEURISTICS, llm_selection
from tools.testing.target_determination.heuristics.llm_selection import LLMSelection
from tools.testing.test_run import TestRun


sys.path.remove(str(REPO_ROOT))

TESTS = ["test_a", "test_b", "dir/test_c"]


def selection(scores: Any, version: Any = 1) -> dict[str, Any]:
    return {"version": version, "head_sha": "0" * 40, "scores": scores}


class TestLLMSelection(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "llm_selection.json"
        patcher = mock.patch.object(llm_selection, "LLM_SELECTION_PATH", self.path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def scores(self, content: Any) -> dict[TestRun, float]:
        self.path.write_text(json.dumps(content))
        return LLMSelection().get_prediction_confidence(TESTS)._test_scores

    def test_scores_the_listed_tests(self) -> None:
        scores = self.scores(
            selection({"test_a": 1, "dir/test_c": 0.25, "test_gone": 1.0})
        )
        expected = {
            TestRun("test_a"): 0.25,
            TestRun("test_b"): 0.0,
            TestRun("dir/test_c"): 0.0625,
        }
        self.assertEqual(scores, expected)

    def test_an_empty_selection_scores_nothing(self) -> None:
        self.assertEqual(set(self.scores(selection({})).values()), {0.0})

    def test_no_selection_raises(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "no llm_selection.json"):
            LLMSelection().get_prediction_confidence(TESTS)

    def test_a_malformed_selection_raises(self) -> None:
        bad: list[Any] = [
            [],
            {"scores": {}},
            selection({}, version=2),
            selection(["test_a"]),
            selection({"test_a": "1"}),
            selection({"test_a": True}),
            selection({"test_a": 0}),
            selection({"test_a": -0.5}),
            selection({"test_a": 1.5}),
            selection({"test_a": float("nan")}),
            selection({"test_a": float("inf")}),
        ]
        for content in bad:
            with self.assertRaises(ValueError, msg=repr(content)):
                self.scores(content)

    def test_runs_in_trial_mode(self) -> None:
        heuristic = next(h for h in HEURISTICS if isinstance(h, LLMSelection))
        self.assertTrue(heuristic.trial_mode)
        self.assertEqual(heuristic.name, "LLMSelection")

    def run_td(self) -> tuple[str, dict[TestRun, float]]:
        """The TD job's log and aggregated scores with only this heuristic."""
        log = io.StringIO()
        heuristics = [LLMSelection(trial_mode=True)]
        with mock.patch.object(determinator, "HEURISTICS", heuristics):
            with mock.patch("sys.stdout", log):
                aggregated = determinator.get_test_prioritizations(TESTS, file=log)
        return log.getvalue(), aggregated.get_aggregated_priorities()._test_scores

    # test-infra's TD scorer reads these log lines (tools/torchci/td/scorer/logs.py).
    def test_td_logs_the_scores_and_they_reorder_nothing(self) -> None:
        self.path.write_text(json.dumps(selection({"test_b": 0.75})))
        log, aggregated = self.run_td()
        self.assertIn("Results from LLMSelection\n  test_b (0.1875)\n", log)
        self.assertEqual(set(aggregated.values()), {0.0})

    def test_td_logs_a_missing_selection_as_an_error(self) -> None:
        log, aggregated = self.run_td()
        self.assertIn("Error in LLMSelection: no llm_selection.json", log)
        self.assertNotIn("Results from LLMSelection", log)
        self.assertEqual(set(aggregated.values()), {0.0})


if __name__ == "__main__":
    unittest.main()
