from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Any
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))

from tools.testing.target_determination import determinator, fetch_llm_selection
from tools.testing.target_determination.heuristics import HEURISTICS, llm_selection
from tools.testing.target_determination.heuristics.llm_selection import LLMSelection
from tools.testing.test_run import TestRun


sys.path.remove(str(REPO_ROOT))

TESTS = ["test_a", "test_b", "dir/test_c"]
SHA = "a" * 40
REPO_ID = 65600975
# The modules' own paths, before a test patches them.
FETCH_PATH = fetch_llm_selection.LLM_SELECTION_PATH
HEURISTIC_PATH = llm_selection.LLM_SELECTION_PATH


def selection(scores: Any, version: Any = 1, sha: str = "0" * 40) -> dict[str, Any]:
    return {"version": version, "head_sha": sha, "scores": scores}


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
            TestRun("test_a"): 1.0,
            TestRun("test_b"): 0.0,
            TestRun("dir/test_c"): 0.25,
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
        self.assertIn("Results from LLMSelection\n  test_b (0.75)\n", log)
        self.assertEqual(set(aggregated.values()), {0.0})

    def test_td_logs_a_missing_selection_as_an_error(self) -> None:
        log, aggregated = self.run_td()
        self.assertIn("Error in LLMSelection: no llm_selection.json", log)
        self.assertNotIn("Results from LLMSelection", log)
        self.assertEqual(set(aggregated.values()), {0.0})


def run(run_id: int, **fields: Any) -> dict[str, Any]:
    """A workflow run of llm_td_selection.yml for SHA, as the REST API lists it."""
    return {
        "id": run_id,
        "path": ".github/workflows/llm_td_selection.yml",
        "event": "pull_request_target",
        "head_sha": SHA,
        "head_repository": {"id": REPO_ID},
        "repository": {"id": REPO_ID},
        "status": "completed",
        "conclusion": "success",
        "created_at": f"2026-11-13T00:00:{run_id:02d}Z",
        "html_url": f"https://github.com/pytorch/pytorch/actions/runs/{run_id}",
        **fields,
    }


def archive(content: bytes, name: str = "llm_selection.json") -> bytes:
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as z:
        z.writestr(name, content)
    return data.getvalue()


class TestFetchLLMSelection(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "ci_files" / "llm_selection.json"
        self.content = json.dumps(selection({"test_a": 1.0}, sha=SHA)).encode()
        # Successive lists of runs the API returns, and each run's artifacts.
        self.listings: list[list[dict[str, Any]]] = [[run(1)]]
        self.artifacts: dict[int, list[dict[str, Any]]] = {1: [self.artifact()]}
        self.archives = {"zip/1": archive(self.content)}
        self.requests: list[str] = []
        self.sleeps: list[float] = []
        self.now = 0.0
        self.stdout = io.StringIO()
        env = {"HEAD_SHA": SHA, "PR_NUMBER": "7", "GITHUB_REF": "refs/pull/7/merge"}
        for patcher in (
            mock.patch.object(fetch_llm_selection, "LLM_SELECTION_PATH", self.path),
            mock.patch.object(fetch_llm_selection, "get", self.get),
            mock.patch.object(fetch_llm_selection, "fetch", self.archives.__getitem__),
            mock.patch.object(fetch_llm_selection.time, "sleep", self.sleep),
            mock.patch.object(fetch_llm_selection.time, "monotonic", lambda: self.now),
            mock.patch.dict(os.environ, env),
            mock.patch("sys.stdout", self.stdout),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def artifact(self, run_id: int = 1, **fields: Any) -> dict[str, Any]:
        url = f"zip/{run_id}"
        name = f"llm_selection_7_{SHA}"
        return {"name": name, "expired": False, "archive_download_url": url, **fields}

    def get(self, path: str) -> Any:
        self.requests.append(path)
        if path.startswith("actions/workflows/"):
            later = self.listings[1:]
            return {
                "workflow_runs": self.listings.pop(0) if later else self.listings[0]
            }
        run_id = int(path.split("/")[2])
        return {"artifacts": self.artifacts.get(run_id, [])}

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def fetched(self) -> bytes | None:
        fetch_llm_selection.main()
        return self.path.read_bytes() if self.path.exists() else None

    def test_writes_where_the_heuristic_reads(self) -> None:
        self.assertEqual(FETCH_PATH, HEURISTIC_PATH)

    def test_the_workflow_it_reads_exists(self) -> None:
        name = fetch_llm_selection.WORKFLOW_FILE
        text = (REPO_ROOT / ".github" / "workflows" / name).read_text()
        self.assertIn("\n  pull_request_target:\n", text)
        pr, sha = (
            "github.event.pull_request.number",
            "github.event.pull_request.head.sha",
        )
        self.assertIn(f"name: llm_selection_${{{{ {pr} }}}}_${{{{ {sha} }}}}\n", text)

    def test_fetches_the_selection_of_the_head(self) -> None:
        self.assertEqual(self.fetched(), self.content)
        listing = (
            "actions/workflows/llm_td_selection.yml/runs"
            f"?event=pull_request_target&head_sha={SHA}&per_page=100"
        )
        artifacts = f"actions/runs/1/artifacts?name=llm_selection_7_{SHA}"
        self.assertEqual(self.requests, [listing, artifacts])
        self.assertEqual(self.sleeps, [])

    def test_the_pr_of_a_ciflow_tag(self) -> None:
        del os.environ["PR_NUMBER"]
        os.environ["GITHUB_REF"] = "refs/tags/ciflow/trunk/7"
        self.assertEqual(self.fetched(), self.content)

    def test_without_a_pr_or_a_head_nothing_is_fetched(self) -> None:
        self.path.parent.mkdir()
        for name, ref in (("PR_NUMBER", "refs/heads/main"), ("HEAD_SHA", "x")):
            self.path.write_text("stale")
            with mock.patch.dict(os.environ, {"GITHUB_REF": ref}):
                del os.environ[name]
                self.assertIsNone(self.fetched())
        self.assertEqual(self.requests, [])

    def test_only_this_workflow_run_from_main_for_this_head_counts(self) -> None:
        for fields in (
            {"path": ".github/workflows/pull.yml"},
            {"event": "pull_request"},
            {"head_sha": "b" * 40},
            {"head_repository": {"id": 1}},
            {"head_repository": None},
            {"conclusion": "skipped"},
            {"conclusion": "failure"},
            {"conclusion": "cancelled"},
        ):
            self.listings = [[run(1, **fields)]]
            self.assertIsNone(self.fetched(), fields)

    def test_the_newest_successful_run_is_used(self) -> None:
        self.listings = [[run(1), run(3, conclusion="failure"), run(2)]]
        newer = json.dumps(selection({"test_b": 1.0}, sha=SHA)).encode()
        self.artifacts[2] = [self.artifact(2)]
        self.archives["zip/2"] = archive(newer)
        self.assertEqual(self.fetched(), newer)

    def test_waits_for_a_pending_run(self) -> None:
        pending = run(2, status="in_progress", conclusion=None)
        self.listings = [[pending, run(1)], [pending, run(1)], [run(2), run(1)]]
        self.artifacts[2] = [self.artifact(2)]
        self.archives["zip/2"] = archive(self.content)
        self.assertEqual(self.fetched(), self.content)
        self.assertEqual(self.sleeps, [15, 30])
        self.assertIn("actions/runs/2/artifacts", self.requests[-1])

    def test_gives_up_on_a_run_pending_too_long(self) -> None:
        pending = run(2, status="queued", conclusion=None)
        self.listings = [[pending, run(1)]]
        # The older run's selection is for the same head, so it still serves.
        self.assertEqual(self.fetched(), self.content)
        self.assertEqual(sum(self.sleeps), fetch_llm_selection.WAIT_SECONDS)
        self.assertEqual(max(self.sleeps), 60)
        self.listings = [[pending]]
        self.assertIsNone(self.fetched())

    def test_an_api_error_gives_nothing(self) -> None:
        error = OSError("HTTP Error 404: Not Found")
        with mock.patch.object(fetch_llm_selection, "get", side_effect=error):
            self.assertIsNone(self.fetched())
        self.assertIn("No LLM test selection: OSError", self.stdout.getvalue())
        self.archives["zip/1"] = b"not a zip"
        self.assertIsNone(self.fetched())

    def test_a_run_without_a_usable_artifact_gives_nothing(self) -> None:
        other = json.dumps(selection({}, sha="b" * 40)).encode()
        big = b" " * fetch_llm_selection.MAX_SELECTION_BYTES + self.content
        for artifacts, archives in (
            ([], {}),
            ([self.artifact(expired=True)], {}),
            ([self.artifact(name="llm_selection_8_" + SHA)], {}),
            ([self.artifact()], {"zip/1": archive(other)}),
            ([self.artifact()], {"zip/1": archive(big)}),
        ):
            self.artifacts[1] = artifacts
            self.archives.update(archives)
            self.assertIsNone(self.fetched())


if __name__ == "__main__":
    unittest.main()
