from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import json
import os
import subprocess
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
from tools.testing.llm_td import select_tests
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
    "truth_jobs": {"test_a": [["pull", "linux / test (slow, 1, 2, linux.2xlarge)"]]},
}
NOTHING = {"out": {"tests": []}}


def pick(name: str, r: dict | None, td: dict | None, s: dict = SAMPLE) -> Choice:
    choice = policy(name, s, r, td, SECONDS)
    if choice is None:
        raise AssertionError(f"{name} chose nothing")
    return choice


@contextlib.contextmanager
def patch_ci_jobs(jobs: dict[bool, set[tuple[str, str, str]]]) -> Iterator[None]:
    with mock.patch.object(score, "ci_jobs", lambda: jobs):
        score.job_runs_td.cache_clear()
        score.runs_td.cache_clear()
        try:
            yield
        finally:
            score.job_runs_td.cache_clear()
            score.runs_td.cache_clear()


class TestRunEval(unittest.TestCase):
    def run_one(self, replies: list[object]) -> tuple[dict, mock.Mock, mock.Mock]:
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
            {"type": "result", "is_error": True, "result": "API Error: 502 Bad Gateway", "total_cost_usd": 0.1},
            {"type": "result", "structured_output": out, "total_cost_usd": 0.2},
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
                "type": "result",
                "subtype": "error_max_structured_output_retries",
                "result": "the 500 tests timed out",
            },
            {
                "type": "result",
                "is_error": True,
                "result": "bad schema",
                "structured_output": {"tests": []},
            },
            # query_model's fallback for output that is not JSON: a crash, whatever its trace says.
            {"is_error": True, "result": "TypeError at cli.js:503:12"},
        ):
            r, query, _ = self.run_one([reply])
            self.assertEqual(
                (r["out"], r["transient"], query.call_count), (None, False, 1)
            )
            self.assertIsNotNone(r["error"])

    def test_timeout_is_final(self) -> None:
        from subprocess import TimeoutExpired

        r, query, _ = self.run_one([TimeoutExpired("claude", 1)])
        self.assertEqual((r["out"], r["transient"], query.call_count), (None, False, 1))  # fmt: skip

    def test_gives_up(self) -> None:
        r, query, sleep = self.run_one(
            [{"type": "result", "is_error": True, "result": "529 overloaded"}] * 4
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
        with patch_ci_jobs({True: {("pull", "linux", "slow")}, False: set()}):
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
                      tests-to-include: ${{ github.event.inputs.tests-to-include || '' }}
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
            "rocm-rx7900.yml": """
                name: rocm-rx7900
                on: pull_request
                jobs:
                  test:
                    name: linux-noble-rocm-py3.12-rx7900
                    uses: ./.github/workflows/_rocm-test.yml
                    with:
                      tests-to-include: >-
                        test_nn
                        test_torch
                      test-matrix: |
                        { include: [{ config: "default", shard: 1 }] }
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
                  shared-test:
                    name: linux-jammy-py3.10-gcc11
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
                ("pull", "linux-jammy-py3.10-gcc11", "default"),
                ("trunk", "dynamo-unittest / dynamo-test", "dynamo_core"),
            },
        )
        self.assertEqual(
            jobs[False],
            {
                ("inductor", "unit-test / inductor-test", "inductor_core"),
                ("dynamo-unittest", "dynamo-test", "dynamo_core"),
                (
                    "periodic",
                    "linux-jammy-cuda13.2-py3-gcc11-slow-gradcheck",
                    "default",
                ),
                ("periodic", "linux-jammy-py3.10-gcc11", "default"),
                # Not on macOS, in xpu or onnx builds, or under force-no-td, also inside a called workflow.
                ("trunk", "macos-py3-arm64", "mps"),
                ("pull", "linux-noble-xpu-n-py3.10", "default"),
                ("pull", "linux-jammy-py3-clang12-onnx", "default"),
                (
                    "trunk",
                    "linux-jammy-aarch64-cuda13.2-py3.12-gcc13-sm100",
                    "smoke_b200",
                ),
                ("trunk", "dynamo-unittest / dynamo-no-td", "dynamo_wrapped"),
                ("dynamo-unittest", "dynamo-no-td", "dynamo_wrapped"),
                # TESTS_TO_INCLUDE turns TD off; pull's dispatch-input expression does not.
                ("rocm-rx7900", "linux-noble-rocm-py3.12-rx7900", "default"),
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
            # A name periodic shares with pull counts by its recorded workflow, and as TD without one.
            "test_e": [["periodic", "linux-jammy-py3.10-gcc11 / test (default, 1, 1, linux.2xlarge)"]],
            "test_f": [["pull", "linux-jammy-py3.10-gcc11 / test (default, 1, 1, linux.2xlarge)"]],
            "test_g": ["linux-jammy-py3.10-gcc11 / test (default, 1, 1, linux.2xlarge)"],
            # A recorded TD workflow with an unknown job falls back to its config; a non-TD one never counts.
            "test_h": [["trunk", "linux-old / test (default, 1, 1, linux.2xlarge)"], ["inductor", "linux-old / test (slow, 1, 1, linux.2xlarge)"]],
            "test_i": [["rocm-rx7900", "linux-noble-rocm-py3.12-rx7900 / test (default, 1, 1, linux.rocm.gpu)"]],
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
            [
                ("test_b", "default"),
                ("test_c", "dynamo_core"),
                ("test_d", "default"),
                ("test_f", "default"),
                ("test_g", "default"),
                ("test_h", "default"),
            ],
        )

    def test_hedge(self) -> None:
        # test_e has no recorded time and test_d no positive score, so neither is ever added.
        td = {"scored": ["test_e", "test_a", "test_b", "test_c"]}
        # default: 5% of 250s fits only test_c; slow: test_a (200s) does not fit 5% of 200s.
        choice = pick("final+td5", NOTHING, td)
        self.assertEqual(choice.per_config, {"default": {"test_c"}, "slow": set()})
        # The default share is 20%: 50s of default fits test_d (40s) and test_c, not test_b after them.
        ranked = {"scored": ["test_d", "test_c", "test_b"]}
        self.assertEqual(pick("final+td", NOTHING, ranked).per_config, {"default": {"test_d", "test_c"}, "slow": set()})  # fmt: skip
        with mock.patch.object(score, "LLM_TD_HEDGE", 0.45):
            self.assertEqual(pick("final+td", NOTHING, td).per_config, pick("final+td45", NOTHING, td).per_config)  # fmt: skip
        self.assertEqual(pick("final+td5", NOTHING, ranked).per_config["default"], {"test_c"})  # fmt: skip
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
        self.assertTrue(pick("final+td45", {"out": None}, td, infra).sel.run_all)

    def test_final_hm_malformed(self) -> None:
        # Malformed answers run everything, as under final.
        tests = ["test_a", {"file": "test_c", "confidence": "low"}, {"file": "test_b", "confidence": "high"}]  # fmt: skip
        for out in ({"tests": None}, {"tests": tests}, ["test_a"], "test_a", None):
            self.assertTrue(pick("final-hm", {"out": out}, None).sel.run_all)
            self.assertTrue(pick("final", {"out": out}, None).sel.run_all)
        tests = [{"file": "test_c", "confidence": "low"}, {"file": "test_b", "confidence": "high"}, {"file": "test_a", "confidence": ["high"]}]  # fmt: skip
        self.assertEqual(
            pick("final-hm", {"out": {"tests": tests}}, None).sel.files, {"test_b"}
        )
        # Filtering never makes a well-formed answer malformed: only unmatched names are left at high confidence.
        tests = [{"file": "test_a", "confidence": "low"}, {"file": "bogus", "confidence": "high"}]  # fmt: skip
        self.assertEqual(pick("final", {"out": {"tests": tests}}, None).sel.files, {"test_a"})  # fmt: skip
        empty = pick("final-hm", {"out": {"tests": tests}}, None).sel
        self.assertEqual((empty.run_all, empty.files), (False, set()))
        hm = pick("final-hm", {"out": {"tests": tests, "dirs": ["sub"]}}, None, {**SAMPLE, "candidates": [*SAMPLE["candidates"], "sub/test_x"]})  # fmt: skip
        self.assertEqual((hm.sel.run_all, hm.sel.files), (False, {"sub/test_x"}))

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
            patch_ci_jobs({True: {("pull", "linux", "default")}, False: set()}),
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

    def test_score_grep(self) -> None:
        sample = {**SAMPLE, "sha": "abc", "changed_files": ["test/test_c.py"]}
        files = {
            "test-times.json": {"linux": {"default": {"test_a": 10.0}}},
            "samples/a.json": sample,
            "td/a.json": {"scored": [], "top25": [], "top25_cfg": {}},
            "results/m/a.json": {"out": {"tests": []}, "error": None, "cost_usd": 0.1},
        }  # fmt: skip
        for grep, files_a, msg in (
            (mock.Mock(return_value=["test/test_a.py"]), {"test_a", "test_c"}, ""),
            (mock.Mock(side_effect=RuntimeError("bad revision")), {"test_c"}, "scored 1 of 1 ids without importers: git grep failed at their head (not in this clone?)\n"),
        ):  # fmt: skip
            with (
                tempfile.TemporaryDirectory() as d,
                patch_ci_jobs({True: {("pull", "linux", "default")}, False: set()}),
                mock.patch.object(score, "grep_at", grep),
            ):
                data = Path(d)
                for name, obj in files.items():
                    (data / name).parent.mkdir(parents=True, exist_ok=True)
                    (data / name).write_text(json.dumps(obj))
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    rows = score.score(data, ["a"], ["m"], ["final", "final-hm"], True, data / "samples", data / "td")  # fmt: skip
            self.assertEqual([x["files"] for x in rows], [len(files_a)] * 2)
            self.assertEqual(err.getvalue(), msg)
            self.assertEqual(grep.call_args.args[1], "abc")

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
            fail("test_landed_pr", "2026-10-06 00:00:00", sha="o", br="gh/x"),
            fail("test_unmapped", "2026-10-02 00:00:00", sha="u", br="gh/y"),
            fail("test_edge_in", "2026-09-21 00:00:01", sha="o", br="gh/x"),
            fail("test_edge_out", "2026-09-21 00:00:00", sha="o", br="gh/x"),
        ]
        disabled = [{"d": "2026-09-30", "name": "test_disabled (__main__.TestA)"}]
        t = "2026-10-05 00:00:00"
        rows = [
            fail(name, t, pr=1, **kw)
            for name, kw in [("test_real", {}), ("test_green", {"job_green": True}), ("test_main", {}),
                             ("test_other", {}), ("test_mine", {}), ("test_old", {}), ("test_disabled", {}), ("test_landed", {}), ("test_landed_pr", {}),
                             ("test_unmapped", {}), ("test_edge_in", {}), ("test_edge_out", {})]
        ]  # fmt: skip
        rows.append(fail("test_unstable", t, pr=1, job="pull / test (default, 1, 1, unstable)"))  # fmt: skip
        landed = dt.datetime(2026, 10, 5, 12, tzinfo=dt.timezone.utc)
        with (
            mock.patch.object(bd, "pull_failures", return_value=others) as pull,
            mock.patch.object(bd, "pr_of_shas", return_value={"o": 2, "s": 1}),
            mock.patch.object(bd, "pr_authors", return_value={1: "me", 2: "x"}),
            mock.patch.object(bd, "query", return_value=disabled),
            mock.patch.object(bd, "landed_times", return_value={1: landed}),
        ):
            rows = flaky_flags(rows, dt.date(2026, 10, 1), dt.date(2026, 10, 26))
        pull.assert_called_once_with(dt.date(2026, 9, 3), dt.date(2026, 11, 9))
        flags = {r["test"]: r["flaky"] for r in rows}
        self.assertEqual(
            flags,
            {"test_real": False, "test_green": True, "test_main": True, "test_other": True, "test_mine": False,
             "test_old": False, "test_disabled": True, "test_landed": False, "test_landed_pr": False, "test_unmapped": True, "test_unstable": True,
             "test_edge_in": True, "test_edge_out": False},
        )  # fmt: skip

    def test_pr_caused_failures(self) -> None:
        def fail(test: str, t: str, sha: str, br: str = "gh/x", cls: str = "TestA", wf: str = "pull") -> dict:  # fmt: skip
            return {"tf": "test_a", "classname": cls, "test": test, "t": t, "sha": sha, "br": br, "wf": wf, "job": "pull / test (default, 1, 1)"}  # fmt: skip

        rows = [
            fail("test_real", "2026-10-05 00:00:00", "a"),
            fail("test_main", "2026-10-05 00:00:00", "a"),
            fail("test_main", "2026-10-07 00:00:00", "m", br="main"),
            fail("test_cls", "2026-10-05 00:00:00", "a"),
            fail("test_cls", "2026-10-05 00:00:00", "m", br="main", cls="TestB"),
            fail("test_shared", "2026-10-05 00:00:00", "a"),
            fail("test_shared", "2026-10-06 00:00:00", "b"),
            fail("test_late", "2026-10-26 00:00:00", "a"),
            fail("test_early", "2026-10-01 00:00:00", "a"),
            fail("test_early", "2026-09-28 00:00:00", "m", br="main"),
            fail("test_holdout", "2026-10-05 00:00:00", "a", wf="td-holdout-unstable"),
            fail("test_periodic", "2026-10-05 00:00:00", "a", wf="periodic"),
            # PR 1 lands at 10-07 12:00; its own breakage on main or on PRs based on it afterwards is not held against it.
            fail("test_broke", "2026-10-05 00:00:00", "a"),
            fail("test_broke", "2026-10-08 00:00:00", "m", br="main"),
            fail("test_broke_pr", "2026-10-05 00:00:00", "a"),
            fail("test_broke_pr", "2026-10-08 00:00:00", "b"),
            fail("test_unmapped", "2026-10-05 00:00:00", "a"),
            fail("test_unmapped", "2026-10-06 00:00:00", "u"),
            fail("test_rx", "2026-10-05 00:00:00", "a", wf="rocm-rx7900"),
        ]
        landed = {1: dt.datetime(2026, 10, 7, 12, tzinfo=dt.timezone.utc)}
        with (
            patch_ci_jobs({True: {("pull", "pull", "default")}, False: {("rocm-rx7900", "pull", "default")}}),
            mock.patch.object(bd, "pr_of_shas", return_value={"a": 1, "b": 2}),
            mock.patch.object(bd, "pr_authors", return_value={1: "me", 2: "x"}),
        ):  # fmt: skip
            keep, main = bd.pr_caused_failures(rows, dt.date(2026, 10, 1), dt.date(2026, 10, 26), landed)  # fmt: skip
        self.assertEqual(sorted((r["test"], r["pr"]) for r in keep), [("test_broke", 1), ("test_broke_pr", 1), ("test_cls", 1), ("test_real", 1)])  # fmt: skip
        self.assertEqual(len(main[("test_a", "TestA", "test_main")]), 1)
        self.assertNotIn(("test_a", "TestA", "test_unmapped"), main)

    def test_head_failures_skip_holdout_workflow(self) -> None:
        name = "linux / test (default, 1, 1)"
        jobs = [
            {"id": 1, "head_sha": "s", "workflow_name": "pull", "name": name, "run_attempt": 1, "conclusion": "failure"},
            {"id": 2, "head_sha": "s", "workflow_name": "td-holdout-unstable", "name": name, "run_attempt": 1, "conclusion": "success"},
        ]  # fmt: skip

        def query(sql: str) -> list[dict]:
            if "pr_by_sha" in sql:
                return [{"number": 1, "head_sha": "s"}]
            if "workflow_job" in sql:
                return jobs
            ids = sql.split("job_id IN (")[1].split(")")[0].split(",")
            return [{"job_id": i, "tf": "test_a", "classname": "TestA", "test": "test_x", "t": "2026-10-05 00:00:00"} for i in ids]  # fmt: skip

        with (
            patch_ci_jobs({True: {("pull", "linux", "default")}, False: set()}),
            mock.patch.object(bd, "query", query),
        ):
            rows = bd.head_failures([1], dt.date(2026, 10, 1), dt.date(2026, 10, 26))
        self.assertEqual([(r["job_id"], r["job_green"], r["wf"]) for r in rows], [("1", False, "pull")])  # fmt: skip

    def test_pr_text_retries_then_raises(self) -> None:
        ok = subprocess.CompletedProcess([], 0, '{"title": "t", "body": "b"}', "")
        bad = subprocess.CompletedProcess([], 1, "", "API rate limit exceeded")
        with mock.patch.object(bd.time, "sleep") as sleep:
            with mock.patch.object(bd.subprocess, "run", side_effect=[bad, bad, ok]):
                self.assertEqual(bd.pr_text(1), ("t", "b"))
            self.assertEqual(sleep.call_count, 2)
            with mock.patch.object(bd.subprocess, "run", return_value=bad):
                with self.assertRaises(subprocess.CalledProcessError):
                    bd.pr_text(1)

    def test_pr_text_failure_skips_sample(self) -> None:
        keep = [{"pr": 1, "tf": "test_a", "classname": "TestA", "test": "test_x", "sha": "s", "wf": "pull", "job": "linux / test (default, 1, 1)", "t": "2026-10-05 00:00:00"}]  # fmt: skip
        error = subprocess.CalledProcessError(1, "gh")
        with (
            tempfile.TemporaryDirectory() as out,
            mock.patch.object(bd, "fetch"),
            mock.patch.object(bd, "has_commit", return_value=True),
            mock.patch.object(bd, "git", return_value="base\n"),
            mock.patch.object(bd, "commit_time", return_value=dt.datetime(2026, 10, 4, tzinfo=dt.timezone.utc)),
            mock.patch.object(bd, "pr_text", side_effect=error),
            mock.patch.object(bd, "write_sample") as write,
        ):  # fmt: skip
            bd.pr_samples(keep, {}, Path(out), 1, dt.date(2026, 10, 1), {})
        write.assert_not_called()

    def test_pr_samples_main_after_landing(self) -> None:
        def fail(tf: str) -> dict:
            return {"pr": 1, "tf": tf, "classname": "TestA", "test": "test_x", "sha": "s", "wf": "pull", "job": "linux / test (default, 1, 1)", "t": "2026-10-05 00:00:00"}  # fmt: skip

        def day(d: int, h: int = 0) -> dt.datetime:
            return dt.datetime(2026, 10, d, h, tzinfo=dt.timezone.utc)

        # Base 10-04, landed 10-06: test_b broke on main only after the PR landed, test_a before,
        # and test_c's main failure is another test of the file.
        main = {("test_a", "TestA", "test_x"): [day(5)], ("test_b", "TestA", "test_x"): [day(7)], ("test_c", "TestA", "test_y"): [day(5)]}  # fmt: skip
        with (
            tempfile.TemporaryDirectory() as out,
            mock.patch.object(bd, "fetch"),
            mock.patch.object(bd, "has_commit", return_value=True),
            mock.patch.object(bd, "git", return_value="base\n"),
            mock.patch.object(bd, "commit_time", return_value=day(4)),
            mock.patch.object(bd, "pr_text", return_value=("t", "b")),
            mock.patch.object(bd, "write_sample") as write,
        ):  # fmt: skip
            bd.pr_samples([fail("test_a"), fail("test_b"), fail("test_c")], main, Path(out), 1, dt.date(2026, 10, 1), {1: day(6)})  # fmt: skip
        kw = write.call_args.kwargs
        job = [("pull", "linux / test (default, 1, 1)")]
        self.assertEqual((kw["truth"], kw["truth_jobs"]), (["test_b", "test_c"], {"test_b": job, "test_c": job}))  # fmt: skip

    def test_revert_truth_is_truth_jobs(self) -> None:
        rows = [{"reverted_at": t, "commit_sha": "c" * 40, "source_signal_keys": ["test_a.py::x", "test_b.py::y"]} for t in ("1", "2")]  # fmt: skip
        jobs = {"test_a": [("trunk", "linux / test (default, 1, 1)")]}
        with (
            tempfile.TemporaryDirectory() as out,
            mock.patch.object(bd, "query", return_value=rows),
            mock.patch.object(bd, "fetch"),
            mock.patch.object(bd, "has_commit", return_value=True),
            mock.patch.object(bd, "commit_time"),
            mock.patch.object(bd, "revert_truth_jobs", return_value=jobs) as truth_jobs,
            mock.patch.object(bd, "git", return_value="t\nb"),
            mock.patch.object(bd, "write_sample") as write,
        ):  # fmt: skip
            bd.revert_samples(Path(out), 1, dt.date(2026, 10, 1), dt.date(2026, 10, 26))
        truth_jobs.assert_called_once_with({"test_a", "test_b"}, mock.ANY, "1")
        kw = write.call_args.kwargs
        self.assertEqual((kw["truth"], kw["truth_jobs"]), (["test_a"], jobs))

    def test_candidates_at(self) -> None:
        ls = "test/test_a.py\ntest/ao/test_b.py\ntest/fx/test_c.py\ntest/test_skip.py\ntest/helper.py\n"
        added = "test/test_new.py\ntest/ao/test_d.py\ntest/fx/test_e.py\ntest/helper_new.py\n"
        named = 'BLOCKLISTED_PATTERNS = ["ao"]\nTESTS = discover_tests(blocklisted_patterns=BLOCKLISTED_PATTERNS, blocklisted_tests=["test_skip"], extra_tests=["extra"])\n'  # fmt: skip
        inline = 'TESTS = discover_tests(blocklisted_patterns=["fx"], blocklisted_tests=["test_skip"])\n'  # fmt: skip
        for src, want, new in ((named, ["extra", "fx/test_c", "test_a"], ["fx/test_e", "test_new"]), (inline, ["ao/test_b", "test_a"], ["ao/test_d", "test_new"])):  # fmt: skip
            with mock.patch.object(bd, "git", lambda *a, src=src: {"show": src, "ls-tree": ls, "diff": added}[a[0]]):  # fmt: skip
                self.assertEqual(bd.candidates_at("b"), want)
                self.assertEqual(bd.candidates_at("b", "s"), sorted(want + new))

    def test_write_sample_matches_select_tests(self) -> None:
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        env |= {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}  # fmt: skip

        def git(*args: str) -> str:
            return subprocess.check_output(["git", *args], cwd=repo, env=env, text=True)

        with tempfile.TemporaryDirectory() as repo:
            git("init", "-q")
            git("config", "user.email", "a@b.c")
            git("config", "user.name", "a")
            (Path(repo) / "test").mkdir()
            (Path(repo) / "test" / "run_test.py").write_text("x = 1\n" * 50)
            git("add", "-A")
            git("commit", "-qm", "base")
            git("mv", "test/run_test.py", "test/my file.py")
            git("commit", "-qm", "move")
            with (
                mock.patch.object(select_tests, "git", git),
                mock.patch.object(bd, "candidates_at", return_value=[]),
            ):
                bd.write_sample(Path(repo), "s", "pr", "HEAD", "HEAD~1", "t", "b")
            s = json.loads((Path(repo) / "s.json").read_text())
        self.assertEqual(s["changed_files"], ["test/my file.py", "test/run_test.py"])
        self.assertIn("test/run_test.py", s["diffstat"])
        self.assertIn("rename from test/run_test.py", s["diff"])


if __name__ == "__main__":
    unittest.main()
