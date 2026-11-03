from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock


if TYPE_CHECKING:
    from collections.abc import Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from tools.testing.llm_td.eval import build_dataset as bd, score
from tools.testing.llm_td.eval.build_dataset import flaky_flags, strip_trailers
from tools.testing.llm_td.eval.run_eval import run_one
from tools.testing.llm_td.eval.score import (
    Choice,
    ci_jobs,
    config_of,
    cost_fraction,
    policy,
    truth_pairs,
)
from tools.testing.llm_td.eval.td_replay import top25_per_config
from tools.testing.llm_td.selection import Selection
from tools.testing.target_determination.heuristics.interface import TestPrioritizations
from tools.testing.test_run import TestRun


sys.path.remove(str(REPO_ROOT))

# config -> file -> [cpu, gpu] seconds
SECONDS = {
    "default": {"test_a": [100.0, 0.0], "test_b": [50.0, 50.0], "test_c": [10.0, 0.0], "test_d": [40.0, 0.0]},
    "slow": {"test_a": [0.0, 200.0], "test_c": [0.0, 0.0]},
}  # fmt: skip
SAMPLE = {
    "kind": "pr",
    "candidates": ["test_a", "test_b", "test_c", "test_d", "test_e"],
    "changed_files": ["torch/foo.py"],
    "truth": ["test_a", "test_b", "test_gone"],
    "truth_strong": ["test_a"],
    "truth_jobs": {"test_a": ["linux / test (slow, 1, 2, linux.2xlarge)"]},
}
NOTHING = {"out": {"tests": []}}


def pick(name: str, r: dict | None, td: dict | None) -> Choice:
    choice = policy(name, SAMPLE, r, td, SECONDS)
    if choice is None:
        raise AssertionError(f"{name} chose nothing")
    return choice


@contextlib.contextmanager
def patch_ci_jobs(jobs: dict[bool, set[tuple[str, str]]]) -> Iterator[None]:
    with mock.patch.object(score, "ci_jobs", lambda: jobs):
        score.job_runs_td.cache_clear()
        score.runs_td.cache_clear()
        try:
            yield
        finally:
            score.job_runs_td.cache_clear()
            score.runs_td.cache_clear()


class TestRunEval(unittest.TestCase):
    def run_one(self, replies: list[dict]) -> tuple[dict, mock.Mock, mock.Mock]:
        sample = {"id": "x", "title": "t", "body": None, "diffstat": "", "diff": "", "candidates": ["test_a"]}  # fmt: skip
        args = argparse.Namespace(model="m", effort="medium")
        with (
            mock.patch("tools.testing.llm_td.eval.run_eval.query_model", side_effect=replies) as query,
            mock.patch("tools.testing.llm_td.eval.run_eval.time.sleep") as sleep,
        ):  # fmt: skip
            return run_one(sample, {}, args, None), query, sleep

    def test_retry_then_success(self) -> None:
        out = {"tests": []}
        r, _, sleep = self.run_one([
            {"is_error": True, "result": "API Error: 502 Bad Gateway", "total_cost_usd": 0.1},
            {"structured_output": out, "total_cost_usd": 0.2},
        ])  # fmt: skip
        self.assertEqual(
            (r["out"], r["error"], r["transient"], r["attempts"]), (out, None, False, 2)
        )
        self.assertAlmostEqual(r["cost_usd"], 0.3)
        self.assertEqual(sleep.call_count, 1)

    def test_not_transient(self) -> None:
        # Only an error's text is matched, and an error drops the structured output as CI does.
        for reply in (
            {
                "subtype": "error_max_structured_output_retries",
                "result": "the 500 tests timed out",
            },
            {
                "is_error": True,
                "result": "bad schema",
                "structured_output": {"tests": []},
            },
        ):
            r, query, _ = self.run_one([reply])
            self.assertEqual(
                (r["out"], r["transient"], query.call_count), (None, False, 1)
            )
            self.assertIsNotNone(r["error"])

    def test_gives_up(self) -> None:
        r, query, sleep = self.run_one(
            [{"is_error": True, "result": "529 overloaded"}] * 4
        )
        self.assertTrue(r["transient"])
        self.assertEqual((query.call_count, sleep.call_count), (4, 3))


class TestScore(unittest.TestCase):
    def test_config_of(self) -> None:
        self.assertEqual(
            config_of(
                "linux-jammy-py3.10-clang21 / test (crossref, 1, 2, lf.linux.2xlarge)"
            ),
            "crossref",
        )
        self.assertEqual(config_of("win / test (default)"), "default")
        self.assertEqual(config_of("(default)"), "default")
        self.assertEqual(config_of(" weird job "), "weird job")

    def test_truth_pairs(self) -> None:
        with patch_ci_jobs({True: {("linux", "slow")}, False: set()}):
            self.assertEqual(truth_pairs(SAMPLE, strong=True), [("test_a", "slow")])
            # test_b has no jobs and falls back to the default config; test_gone is not a candidate.
            self.assertEqual(
                truth_pairs(SAMPLE, strong=False),
                [("test_a", "slow"), ("test_b", "default")],
            )
            self.assertEqual(
                truth_pairs({**SAMPLE, "kind": "revert"}, strong=True),
                [("test_a", "slow"), ("test_b", "default")],
            )

    def test_ci_jobs(self) -> None:
        workflows = {
            "pull.yml": """
                name: pull
                on: pull_request
                jobs:
                  build:
                    name: linux-jammy-py3.10-gcc11
                    uses: ./.github/workflows/_linux-build.yml
                    with:
                      test-matrix: |
                        { include: [{ config: "default", shard: 1 }] }
                  test:
                    name: linux-jammy-py3.10-gcc11
                    uses: ./.github/workflows/_linux-test.yml
                    with:
                      test-matrix: ${{ needs.build.outputs.test-matrix }}
                  xpu-build:
                    name: linux-noble-xpu-n-py3.10
                    uses: ./.github/workflows/_linux-build.yml
                    with:
                      build-environment: linux-noble-xpu-n-py3.10
                      test-matrix: |
                        { include: [{ config: "default", shard: 1 }] }
                  xpu-test:
                    name: linux-noble-xpu-n-py3.10
                    uses: ./.github/workflows/_xpu-test.yml
                    with:
                      build-environment: ${{ needs.xpu-build.outputs.build-environment }}
                      test-matrix: ${{ needs.xpu-build.outputs.test-matrix }}
                  onnx-test:
                    name: linux-jammy-py3-clang12-onnx
                    uses: ./.github/workflows/_linux-test.yml
                    with:
                      build-environment: linux-jammy-py3-clang12-onnx
                      test-matrix: |
                        { include: [{ config: "default", shard: 1 }] }
            """,
            "inductor.yml": """
                name: inductor
                on: pull_request
                jobs:
                  unit-test:
                    name: unit-test
                    uses: ./.github/workflows/inductor-unittest.yml
            """,
            "inductor-unittest.yml": """
                name: inductor-unittest
                on:
                  workflow_call:
                jobs:
                  inductor-test:
                    uses: ./.github/workflows/_linux-test.yml
                    with:
                      test-matrix: |
                        { include: [{ config: "inductor_core", shard: 1 }] }
            """,
            "trunk.yml": """
                name: trunk
                on: push
                jobs:
                  dynamo-unittest:
                    name: dynamo-unittest
                    uses: ./.github/workflows/dynamo-unittest.yml
                  macos-test:
                    name: macos-py3-arm64
                    uses: ./.github/workflows/_mac-test.yml
                    with:
                      build-environment: macos-py3-arm64
                      test-matrix: |
                        { include: [{ config: "mps", shard: 1 }] }
                  sm100-test:
                    name: linux-jammy-aarch64-cuda13.2-py3.12-gcc13-sm100
                    uses: ./.github/workflows/_linux-test.yml
                    with:
                      build-environment: linux-jammy-aarch64-cuda13.2-py3.12-gcc13
                      force-no-td: true
                      test-matrix: |
                        { include: [{ config: "smoke_b200", shard: 1 }] }
            """,
            "dynamo-unittest.yml": """
                name: dynamo-unittest
                on:
                  workflow_call:
                  pull_request:
                jobs:
                  dynamo-test:
                    uses: ./.github/workflows/_linux-test.yml
                    with:
                      test-matrix: |
                        { include: [{ config: "dynamo_core", shard: 1 }] }
                  dynamo-no-td:
                    uses: ./.github/workflows/_linux-test.yml
                    with:
                      force-no-td: true
                      test-matrix: |
                        { include: [{ config: "dynamo_wrapped", shard: 1 }] }
            """,
            "periodic.yml": """
                name: periodic
                on: schedule
                jobs:
                  test:
                    name: linux-jammy-cuda13.2-py3-gcc11-slow-gradcheck
                    uses: ./.github/workflows/_linux-test.yml
                    with:
                      test-matrix: |
                        { include: [{ config: "default", shard: 1 }] }
            """,
        }  # fmt: skip
        with tempfile.TemporaryDirectory() as d:
            for name, text in workflows.items():
                (Path(d) / name).write_text(textwrap.dedent(text))
            jobs = ci_jobs(Path(d))
        self.assertEqual(
            jobs[True],
            {
                ("linux-jammy-py3.10-gcc11", "default"),
                ("dynamo-unittest / dynamo-test", "dynamo_core"),
            },
        )
        self.assertEqual(
            jobs[False],
            {
                ("unit-test / inductor-test", "inductor_core"),
                ("dynamo-test", "dynamo_core"),
                ("linux-jammy-cuda13.2-py3-gcc11-slow-gradcheck", "default"),
                # Not on macOS, in xpu or onnx builds, or under force-no-td, also inside a called workflow.
                ("macos-py3-arm64", "mps"),
                ("linux-noble-xpu-n-py3.10", "default"),
                ("linux-jammy-py3-clang12-onnx", "default"),
                ("linux-jammy-aarch64-cuda13.2-py3.12-gcc13-sm100", "smoke_b200"),
                ("dynamo-unittest / dynamo-no-td", "dynamo_wrapped"),
                ("dynamo-no-td", "dynamo_wrapped"),
            },
        )
        truth_jobs = {
            "test_a": ["unit-test / inductor-test / test (inductor_core, 1, 2, linux.2xlarge)"],
            # The periodic job at an older CUDA version, and a pull job: only the pull failure counts.
            "test_b": [
                "linux-jammy-cuda12.8-py3-gcc11-slow-gradcheck / test (default, 1, 2, linux.2xlarge)",
                "linux-jammy-py3.10-gcc11 / test (default, 1, 2, linux.2xlarge)",
            ],
            # dynamo-unittest under trunk, with a matrix suffix, and under its own pull_request trigger.
            "test_c": [
                "dynamo-unittest / dynamo-test (3.11) / test (dynamo_core, 1, 2, linux.2xlarge)",
                "dynamo-test (3.12) / test (dynamo_core, 1, 2, linux.2xlarge)",
            ],
            # Unknown jobs fall back to their config.
            "test_d": ["linux-old / test (default, 1, 1, linux.2xlarge)", "linux-old / test (slow, 1, 1, linux.2xlarge)"],
        }  # fmt: skip
        sample = {**SAMPLE, "kind": "revert", "truth": list(truth_jobs), "candidates": list(truth_jobs), "truth_jobs": truth_jobs}  # fmt: skip
        with patch_ci_jobs(jobs):
            got = truth_pairs(sample, strong=True)
            self.assertFalse(score.job_runs_td(truth_jobs["test_b"][0]))
            self.assertFalse(score.job_runs_td(truth_jobs["test_c"][1]))
            self.assertFalse(score.runs_td("mps"))
            self.assertFalse(score.runs_td("smoke_b200"))
        self.assertEqual(
            got,
            [("test_b", "default"), ("test_c", "dynamo_core"), ("test_d", "default")],
        )

    def test_hedge(self) -> None:
        # test_e has no recorded time and test_d no positive score, so neither is ever added.
        td = {"scored": ["test_e", "test_a", "test_b", "test_c"]}
        # default: 5% of 250s fits only test_c; slow: test_a (200s) does not fit 5% of 200s.
        choice = pick("final+td5", NOTHING, td)
        self.assertEqual(choice.per_config, {"default": {"test_c"}, "slow": set()})
        self.assertEqual(pick("final+td", NOTHING, td).per_config, pick("final+td20", NOTHING, td).per_config)  # fmt: skip
        # 45% of 250s: test_a fits, test_b then does not, test_c after it still does.
        choice = pick("final+td45", NOTHING, td)
        self.assertEqual(
            choice.per_config, {"default": {"test_a", "test_c"}, "slow": set()}
        )
        self.assertTrue(choice.covers("test_a", "default"))
        self.assertFalse(choice.covers("test_a", "slow"))
        # Unknown configs hedge over every config's time: 45% of 450s skips test_a (300s).
        self.assertEqual(choice.other, {"test_b", "test_c"})
        self.assertTrue(choice.covers("test_b", "unknown"))
        # Files the selection already covers cost no budget.
        r = {"out": {"tests": [{"file": "test_a"}]}}
        self.assertEqual(
            pick("final+td45", r, td).per_config["default"],
            {"test_b", "test_c"},
        )
        self.assertIsNone(policy("final+td5", SAMPLE, NOTHING, None, SECONDS))
        for bad in ("final+tdx", "final+td5x", "nonsense"):
            with self.assertRaisesRegex(ValueError, "unknown policy"):
                policy(bad, SAMPLE, NOTHING, td, SECONDS)

    def test_hedge_failed_call_keeps_td(self) -> None:
        td = {"scored": ["test_a"], "top25": ["test_e"], "top25_cfg": {"default": ["test_a"], "slow": []}}  # fmt: skip
        for r in ({"out": None}, None, {"out": {"tests": "test_a"}}):
            choice = pick("final+td45", r, td)
            self.assertFalse(choice.sel.run_all)
            self.assertEqual(choice.per_config, {"default": {"test_a"}, "slow": set()})
            self.assertEqual(choice.other, {"test_e"})
        # A deliberate run-all still runs everything.
        self.assertTrue(pick("final+td45", {"out": {"tests": [], "run_all": True}}, td).sel.run_all)  # fmt: skip
        infra = {**SAMPLE, "changed_files": ["test/run_test.py"]}
        self.assertTrue(policy("final+td45", infra, {"out": None}, td, SECONDS).sel.run_all)  # fmt: skip

    def test_final_hm_malformed(self) -> None:
        self.assertEqual(
            pick("final-hm", {"out": {"tests": None}}, None).sel.files, set()
        )
        tests = ["test_a", {"file": "test_c", "confidence": "low"}, {"file": "test_b", "confidence": "high"}]  # fmt: skip
        self.assertEqual(
            pick("final-hm", {"out": {"tests": tests}}, None).sel.files, {"test_b"}
        )

    def test_score_same_ids(self) -> None:
        td = {"scored": [], "top25": ["test_a"], "top25_cfg": {"default": ["test_a"]}}
        result = {
            "out": {"tests": [{"file": "test_a"}]},
            "error": None,
            "cost_usd": 0.1,
        }
        files = {
            "test-times.json": {"linux": {"default": {"test_a": 10.0, "test_b": 5.0}}},
            "samples/a.json": SAMPLE, "samples/b.json": SAMPLE, "samples/c.json": SAMPLE,
            "td/a.json": td, "td/b.json": td,
            "results/m1/a.json": result, "results/m1/b.json": result, "results/m1/c.json": result,
            "results/m2/a.json": result, "results/m2/c.json": result,
        }  # fmt: skip
        with (
            tempfile.TemporaryDirectory() as d,
            patch_ci_jobs({True: {("linux", "default")}, False: set()}),
        ):
            data = Path(d)
            for name, obj in files.items():
                (data / name).parent.mkdir(parents=True, exist_ok=True)
                (data / name).write_text(json.dumps(obj))
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rows = score.score(data, ["a", "b", "c"], ["m1", "m2", "td"], ["model"], True, data / "samples", data / "td")  # fmt: skip
        self.assertEqual({(x["tag"], x["policy"], x["id"]) for x in rows}, {("m1", "model", "a"), ("m2", "model", "a"), ("td", "td25", "a")})  # fmt: skip
        self.assertEqual(err.getvalue(), "dropped 1 of 3 ids with no m2 result\ndropped 1 of 3 ids with no td replay\n")  # fmt: skip

    def test_td25_per_config(self) -> None:
        cands = ["test_a", "test_b", "test_c", "test_d", "test_e"]
        scores = {
            TestRun("test_e"): 3.0,
            TestRun("test_a", included=["TestX"]): 2.0,
            TestRun("test_c"): 1.0,
        }
        tp = TestPrioritizations(cands, scores)
        self.assertEqual(
            {t.test_file for t in tp.get_top_per_tests(25)[0]}, {"test_e", "test_a"}
        )
        # The cut is 25% + 1 of each config's own runs; in "slow" test_a's partial run comes first and counts.
        cut = top25_per_config(tp, cands, SECONDS)
        self.assertEqual(cut, {"default": ["test_a", "test_c"], "slow": ["test_a"]})
        td = {"top25": ["test_a", "test_e"], "top25_cfg": cut}
        choice = pick("td25", None, td)
        self.assertTrue(choice.covers("test_a", "slow"))
        self.assertFalse(choice.covers("test_c", "slow"))
        self.assertTrue(choice.covers("test_e", "not_in_test_times"))
        self.assertEqual(choice.files(), {"test_a", "test_c"})

    def test_cost_fraction(self) -> None:
        cands = {"test_a", "test_b", "test_c"}
        choice = Choice(Selection({"test_b"}), {"slow": {"test_a"}})
        self.assertEqual(cost_fraction(choice, cands, SECONDS), (50 / 160, 250 / 250))
        self.assertEqual(
            cost_fraction(choice, cands, SECONDS, lambda c: c == "default"),
            (50 / 160, 1.0),
        )
        self.assertEqual(
            cost_fraction(Choice(Selection(run_all=True)), cands, SECONDS), (1.0, 1.0)
        )


class TestBuildDataset(unittest.TestCase):
    def test_strip_trailers(self) -> None:
        msg = "Body line\n\nCo-authored-by: x\nPull Request resolved: https://x\nApproved by: https://y\nghstack-source-id: 1\nKeep: me\n"  # @lint-ignore
        self.assertEqual(strip_trailers(msg), "Body line\n\nKeep: me")

    def test_flaky_flags(self) -> None:
        def fail(test: str, t: str, **kw: object) -> dict:
            return {"tf": "test_a", "classname": "TestA", "test": test, "t": t, "job": "pull / test (default, 1, 1)",
                    "job_green": False, **kw}  # fmt: skip

        others = [
            fail("test_main", "2026-10-01 00:00:00", sha="m", br="main"),
            fail("test_other", "2026-10-03 00:00:00", sha="o", br="gh/x"),
            fail("test_mine", "2026-10-03 00:00:00", sha="s", br="gh/me"),
            fail("test_old", "2026-08-01 00:00:00", sha="m", br="main"),
            fail("test_landed", "2026-10-06 00:00:00", sha="m", br="main"),
            fail("test_unmapped", "2026-10-02 00:00:00", sha="u", br="gh/y"),
            fail("test_edge_in", "2026-09-21 00:00:01", sha="o", br="gh/x"),
            fail("test_edge_out", "2026-09-21 00:00:00", sha="o", br="gh/x"),
        ]
        disabled = [{"d": "2026-09-30", "name": "test_disabled (__main__.TestA)"}]
        t = "2026-10-05 00:00:00"
        rows = [
            fail(name, t, pr=1, **kw)
            for name, kw in [("test_real", {}), ("test_green", {"job_green": True}), ("test_main", {}),
                             ("test_other", {}), ("test_mine", {}), ("test_old", {}), ("test_disabled", {}), ("test_landed", {}),
                             ("test_unmapped", {}), ("test_edge_in", {}), ("test_edge_out", {})]
        ]  # fmt: skip
        rows.append(fail("test_unstable", t, pr=1, job="pull / test (default, 1, 1, unstable)"))  # fmt: skip
        landed = dt.datetime(2026, 10, 5, 12, tzinfo=dt.timezone.utc)
        with (
            mock.patch.object(bd, "pull_failures", return_value=others),
            mock.patch.object(bd, "pr_of_shas", return_value={"o": 2, "s": 1}),
            mock.patch.object(bd, "pr_authors", return_value={1: "me", 2: "x"}),
            mock.patch.object(bd, "query", return_value=disabled),
            mock.patch.object(bd, "landed_times", return_value={1: landed}),
        ):
            rows = flaky_flags(rows, dt.date(2026, 10, 1), dt.date(2026, 10, 26))
        flags = {r["test"]: r["flaky"] for r in rows}
        self.assertEqual(
            flags,
            {"test_real": False, "test_green": True, "test_main": True, "test_other": True, "test_mine": False,
             "test_old": False, "test_disabled": True, "test_landed": False, "test_unmapped": True, "test_unstable": True,
             "test_edge_in": True, "test_edge_out": False},
        )  # fmt: skip

    def test_pr_caused_failures(self) -> None:
        def fail(test: str, t: str, sha: str, br: str = "gh/x", cls: str = "TestA") -> dict:  # fmt: skip
            return {"tf": "test_a", "classname": cls, "test": test, "t": t, "sha": sha, "br": br, "job": "pull / test (default, 1, 1)"}  # fmt: skip

        rows = [
            fail("test_real", "2026-10-05 00:00:00", "a"),
            fail("test_main", "2026-10-05 00:00:00", "a"),
            fail("test_main", "2026-10-08 00:00:00", "m", br="main"),
            fail("test_cls", "2026-10-05 00:00:00", "a"),
            fail("test_cls", "2026-10-05 00:00:00", "m", br="main", cls="TestB"),
            fail("test_shared", "2026-10-05 00:00:00", "a"),
            fail("test_shared", "2026-10-06 00:00:00", "b"),
            fail("test_late", "2026-10-26 00:00:00", "a"),
            fail("test_early", "2026-10-01 00:00:00", "a"),
            fail("test_early", "2026-09-28 00:00:00", "m", br="main"),
        ]
        with (
            mock.patch.object(bd, "pr_of_shas", return_value={"a": 1, "b": 2}),
            mock.patch.object(bd, "pr_authors", return_value={1: "me", 2: "x"}),
        ):
            keep, main = bd.pr_caused_failures(rows, dt.date(2026, 10, 1), dt.date(2026, 10, 26))  # fmt: skip
        self.assertEqual(sorted(r["test"] for r in keep), ["test_cls", "test_real"])
        self.assertEqual(len(main[("test_a", "TestA", "test_main")]), 1)


if __name__ == "__main__":
    unittest.main()
