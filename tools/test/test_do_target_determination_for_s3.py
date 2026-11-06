from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from tools.testing import do_target_determination_for_s3
from tools.testing.do_target_determination_for_s3 import (
    import_results,
    PREVIOUS_FAILURES_KEY,
    td_results_json,
)
from tools.testing.target_determination.heuristics import previously_failed_in_pr
from tools.testing.target_determination.heuristics.interface import TestPrioritizations
from tools.testing.test_run import TestRun


class TestTDResults(unittest.TestCase):
    def test_td_results_round_trip(self) -> None:
        tp = TestPrioritizations(["test_nn", "test_ops"], {TestRun("test_nn"): 1})
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "td_results.json"
            for failed in ({"test_ops", "elsewhere", "test_nn"}, set(), None):
                td_results = td_results_json(tp, failed)
                if failed is not None:
                    self.assertEqual(td_results[PREVIOUS_FAILURES_KEY], sorted(failed))
                p.write_text(json.dumps(td_results))
                with mock.patch("sys.stderr"):
                    loaded, loaded_failed = import_results(p)
                self.assertEqual(loaded_failed, failed)
                scores = [(str(x), loaded.get_score(x)) for x in loaded.get_all_tests()]
                self.assertEqual(scores, [("test_nn", 1), ("test_ops", 0)])

    def test_td_job_writes_the_previous_failures(self) -> None:
        scores = {TestRun("test_nn"): 0.5, TestRun("test_ops"): 1}
        tp = TestPrioritizations(["test_ops", "test_nn", "test_fx"], scores)
        heuristics = mock.Mock(**{"get_aggregated_priorities.return_value": tp})
        fetches: dict[str, Any] = dict.fromkeys(
            [
                "get_test_times",
                "get_test_class_times",
                "get_test_file_ratings",
                "get_test_class_ratings",
                "get_td_heuristic_historial_edited_files_json",
                "get_td_heuristic_profiling_json",
                "copy_additional_previous_failures",
            ],
            mock.DEFAULT,
        )
        lastfailed = '{"test/test_ops.py::TestOps::test_a": true}'
        # A failed or skipped download leaves no cache, as does a PR without
        # one yet, so only the step's outcome makes the failures known.
        for outcome, cache, failed in (
            ("success", lastfailed, {"test_ops"}),
            ("success", None, set()),
            ("success", "{", None),
            ("failure", None, None),
            ("failure", lastfailed, None),
            (None, None, None),
        ):
            with tempfile.TemporaryDirectory() as d:
                folder = Path(d) / ".additional_ci_files"
                folder.mkdir()

                def copy_pytest_cache(cache: str | None = cache) -> None:
                    if cache is not None:
                        (folder / "previous_failures.json").write_text(cache)

                env = {"CI": "false"}
                if outcome is not None:
                    env["PYTEST_CACHE_DOWNLOAD_OUTCOME"] = outcome
                with (
                    mock.patch.multiple(
                        do_target_determination_for_s3,
                        REPO_ROOT=Path(d),
                        get_test_prioritizations=mock.Mock(return_value=heuristics),
                        copy_pytest_cache=copy_pytest_cache,
                        **fetches,
                    ),
                    mock.patch.object(previously_failed_in_pr, "REPO_ROOT", Path(d)),
                    mock.patch.dict(os.environ, env),
                    mock.patch("sys.stdout"),
                    mock.patch("sys.stderr"),
                ):
                    if outcome is None:
                        os.environ.pop("PYTEST_CACHE_DOWNLOAD_OUTCOME", None)
                    do_target_determination_for_s3.main()
                    ranking, loaded = import_results(Path(d) / "td_results.json")
                self.assertEqual(loaded, failed, (outcome, cache))
                ranked = [(t, ranking.get_score(t)) for t in ranking.get_all_tests()]
                expected = [(t, tp.get_score(t)) for t in tp.get_all_tests()]
                self.assertEqual(ranked, expected, (outcome, cache))

    def test_import_results(self) -> None:
        with tempfile.TemporaryDirectory() as d, mock.patch("sys.stdout"):
            p = Path(d) / "td_results.json"
            stderr = io.StringIO()
            with mock.patch("sys.stderr", stderr):
                tp, failed = import_results(p)
            self.assertEqual(
                (tp.get_all_tests(), failed, stderr.getvalue()), ([], None, "")
            )
            tp = TestPrioritizations(["test_nn", "test_ops"], {TestRun("test_nn"): 1})
            # An older writer stores only TD's ranking.
            for value in ("missing", "test_nn", [1], None):
                td_results = tp.to_json()
                if value != "missing":
                    td_results[PREVIOUS_FAILURES_KEY] = value
                p.write_text(json.dumps(td_results))
                stderr = io.StringIO()
                with mock.patch("sys.stderr", stderr):
                    self.assertIsNone(import_results(p)[1], value)
                message = (
                    "Previous failures not recorded in"
                    if value == "missing"
                    else "Ignoring the previous failures in"
                )
                self.assertIn(message, stderr.getvalue())
            td_results = {
                **tp.to_json(),
                PREVIOUS_FAILURES_KEY: ["test_nn", "elsewhere"],
            }
            p.write_text(json.dumps(td_results))
            loaded, failed = import_results(p)
            self.assertEqual(failed, {"test_nn", "elsewhere"})
            scores = [(str(x), loaded.get_score(x)) for x in loaded.get_all_tests()]
            self.assertEqual(scores, [("test_nn", 1), ("test_ops", 0)])


if __name__ == "__main__":
    unittest.main()
