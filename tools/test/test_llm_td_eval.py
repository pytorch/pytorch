from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import json
import os
import re
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
from tools.testing.llm_td.eval import build_dataset as bd, chart, dollars, score
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

    def test_main_refuses_other_settings(self) -> None:
        from tools.testing.llm_td.eval import run_eval

        with tempfile.TemporaryDirectory() as d:
            data = Path(d)
            (data / "results/t").mkdir(parents=True)
            (data / "results/t/a.json").write_text(json.dumps({"model": "m", "effort": "high", "agentic": False}))  # fmt: skip
            (data / "test-times.json").write_text("{}")
            (data / "ids.txt").write_text("a b")
            argv = ["run_eval", "--data", d, "--tag", "t", "--model", "m", "--ids", str(data / "ids.txt")]  # fmt: skip
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch("sys.stderr", io.StringIO()) as err,
                self.assertRaises(SystemExit),
            ):
                run_eval.main()
        self.assertIn("1 results in", err.getvalue())


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
                  dynamo-cpython-test:
                    name: dynamo-cpython-test
                    uses: ./.github/workflows/_linux-test.yml
                    with:
                      test-matrix: |
                        { include: [{ config: "dynamo_cpython", shard: 1 }] }
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
                # test.sh exports NO_TD=1 for this config.
                ("pull", "dynamo-cpython-test", "dynamo_cpython"),
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
            # An unknown macOS, xpu or onnx job does not fall back to its config.
            "test_j": [["trunk", "macos-old / test (default, 1, 1, macos-m1)"], "linux-xpu-old / test (default, 1, 1, linux.idc.xpu)"],
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

    def test_no_td_configs_match_test_sh(self) -> None:
        text = (REPO_ROOT / ".ci/pytorch/test.sh").read_text()
        funcs = re.findall(r"^(\w+)\(\) \{\n(?:(?!^\}).*\n)*?\s*export NO_TD=1", text, re.MULTILINE)  # fmt: skip
        dispatch = dict(re.findall(r'"\$\{TEST_CONFIG\}" == \*(\w+)\* \]\]; then\n\s*(\w+)', text))  # fmt: skip
        self.assertEqual(funcs, ["test_dynamo_cpython"])
        self.assertEqual({c for c, f in dispatch.items() if f in funcs}, score.NO_TD_CONFIGS)  # fmt: skip

    def test_unasked(self) -> None:
        td = {"scored": ["test_a"], "top25": ["test_e"], "top25_cfg": {"default": ["test_a"]}}  # fmt: skip
        r = {"out": {"tests": [{"file": "test_b"}]}}
        huge = {**SAMPLE, "changed_files": [f"torch/f{i}.py" for i in range(select_tests.MAX_CHANGED_FILES + 1)]}  # fmt: skip
        empty = {**SAMPLE, "changed_files": []}
        for name in ("model", "final", "final-hm", "final+td45"):
            self.assertTrue(pick(name, r, td, huge).sel.run_all, name)
            choice = pick(name, r, td, empty)
            self.assertEqual(choice.sel.run_all, name != "final+td45", name)
        # With no changed files select_tests records an error, so TD's cut is kept.
        self.assertEqual(pick("final+td45", r, td, empty).per_config, {"default": {"test_a"}})  # fmt: skip
        files = {
            "test-times.json": {"linux": {"default": {"test_a": 10.0}}},
            "samples/a.json": huge,
            "td/a.json": td,
        }  # fmt: skip
        with (
            tempfile.TemporaryDirectory() as d,
            patch_ci_jobs({True: {("pull", "linux", "default")}, False: set()}),
        ):
            data = Path(d)
            for name, obj in files.items():
                (data / name).parent.mkdir(parents=True, exist_ok=True)
                (data / name).write_text(json.dumps(obj))
            rows = score.score(data, ["a"], ["m"], ["model"], True, data / "samples", data / "td")  # fmt: skip
        self.assertEqual([(x["run_all"], x["cost_usd"], x["error"]) for x in rows], [(True, 0.0, False)])  # fmt: skip

    def test_hedge(self) -> None:
        # test_e has no recorded time and test_d no positive score, so neither is ever added.
        td = {"scored": ["test_e", "test_a", "test_b", "test_c"], "top25": [], "top25_cfg": {}}  # fmt: skip
        # default: 5% of 250s fits only test_c; slow: test_a (200s) does not fit 5% of 200s.
        choice = pick("final+td5", NOTHING, td)
        self.assertEqual(choice.per_config, {"default": {"test_c"}, "slow": set()})
        # A config in which nothing runs keeps TD's cut, as Selection.split does.
        cut = {**td, "top25_cfg": {"slow": ["test_a"]}}
        self.assertEqual(pick("final+td5", NOTHING, cut).per_config["slow"], {"test_a"})
        r = {"out": {"tests": [{"file": "test_c"}]}}
        self.assertEqual(pick("final+td5", r, cut).per_config["slow"], set())
        # The default share is 20%: 50s of default fits test_d (40s) and test_c, not test_b after them.
        ranked = {"scored": ["test_d", "test_c", "test_b"], "top25": [], "top25_cfg": {}}  # fmt: skip
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
        # An unmatched name makes the answer malformed under both.
        tests = [{"file": "test_a", "confidence": "low"}, {"file": "bogus", "confidence": "high"}]  # fmt: skip
        for name in ("final", "final-hm"):
            self.assertTrue(pick(name, {"out": {"tests": tests}}, None).sel.run_all)
        # Filtering out every name leaves a well-formed empty answer.
        tests = [{"file": "test_a", "confidence": "low"}]
        self.assertEqual(pick("final", {"out": {"tests": tests}}, None).sel.files, {"test_a"})  # fmt: skip
        empty = pick("final-hm", {"out": {"tests": tests}}, None).sel
        self.assertEqual((empty.run_all, empty.files), (False, set()))
        hm = pick("final-hm", {"out": {"tests": tests, "dirs": ["sub"]}}, None, {**SAMPLE, "candidates": [*SAMPLE["candidates"], "sub/test_x"]})  # fmt: skip
        self.assertEqual((hm.sel.run_all, hm.sel.files), (False, {"sub/test_x"}))

    def test_score_same_ids(self) -> None:
        sample = {**SAMPLE, "truth_jobs": {"test_a": [["pull", "linux / test (default, 1, 2, linux.2xlarge)"]]}}  # fmt: skip
        td = {"scored": [], "top25": ["test_a"], "top25_cfg": {"default": ["test_a"]}}
        result = {
            "out": {"tests": [{"file": "test_a"}]},
            "error": None,
            "cost_usd": 0.1,
        }
        failed = {"out": None, "error": "boom", "cost_usd": 0.0}
        # The top-level "default" is run_test.py's fallback for unknown jobs, not a job to price.
        times = {
            "linux": {"default": {"test_a": 10.0, "test_b": 5.0}},
            "linux-cuda": {"default": {"test_a": 20.0}},
            "default": {"default": {"test_a": 1000.0, "test_b": 1000.0}},
        }
        files = {
            "test-times.json": times,
            "samples/a.json": sample, "samples/b.json": sample, "samples/c.json": sample,
            "td/a.json": td, "td/b.json": td,
            "results/m1/a.json": result, "results/m1/b.json": result, "results/m1/c.json": result,
            "results/m2/a.json": failed, "results/m2/c.json": failed,
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
        fields = ("pairs", "hit", "missed", "cpu", "gpu", "run_all", "files", "error")
        by_tag = {x["tag"]: tuple(x[k] for k in fields) for x in rows}
        self.assertEqual(by_tag, {
            "m1": (1, 1, [], 10 / 15, 1.0, False, 1, False),
            "m2": (1, 1, [], 1.0, 1.0, True, 0, True),
            "td": (1, 1, [], 10 / 15, 1.0, False, 1, False),
        })  # fmt: skip

    def test_summarize(self) -> None:
        row = {"tag": "t", "policy": "p", "kind": "pr", "cost_usd": 0.1, "error": False, "cpu": 0.5, "gpu": 0.0}  # fmt: skip
        run_all = {"files": 0, "run_all": True, "cpu": 1.0, "gpu": 1.0, "error": True}
        rows = [
            {**row, "pairs": 2, "hit": 1, "files": 3, "run_all": False},
            {**row, "pairs": 1, "hit": 1, "files": 5, "run_all": False},
            {**row, "pairs": 0, "hit": 0, "files": 9, "run_all": False},
            {**row, "pairs": 1, "hit": 1, **run_all},
        ]
        (s,) = score.summarize(rows)
        self.assertEqual(s, {
            "tag": "t", "policy": "p", "kind": "pr", "n": 4, "n_failing": 3,
            "caught_any": 1.0, "caught_all": 2 / 3, "pair_recall": 3 / 4,
            "cpu": 0.625, "gpu": 0.25, "run_all": 1, "median_files": 5,
            "usd_per_call": 0.1, "errors": 1,
        })  # fmt: skip

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
        # An edited failure list whose classes git grep cannot look up forces a run-all or a sweep.
        for entry, why in (("inductor_skips", "run everything"), ("dynamo_skips", "add the dynamo_wrapped sweep")):  # fmt: skip
            sample = {**SAMPLE, "sha": "abc", "changed_files": [f"test/{entry}/TestA.test_x"]}  # fmt: skip
            with (
                tempfile.TemporaryDirectory() as d,
                patch_ci_jobs({True: {("pull", "linux", "default")}, False: set()}),
                mock.patch.object(score, "grep_at", mock.Mock(side_effect=RuntimeError)),
            ):  # fmt: skip
                data = Path(d)
                for name, obj in {**files, "samples/a.json": sample}.items():
                    (data / name).parent.mkdir(parents=True, exist_ok=True)
                    (data / name).write_text(json.dumps(obj))
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    score.score(data, ["a"], ["m"], ["final"], True, data / "samples", data / "td")  # fmt: skip
            self.assertIn(f"\n  1 of them {why} under the final policies only because an edited failure list could not be looked up\n", err.getvalue())  # fmt: skip

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
        # The cut is 25% + 1 of each config's own runs plus the files with no
        # recorded time (test_e, as if newly added); test_a's partial run counts.
        cut = top25_per_config(tp, cands, SECONDS)
        self.assertEqual(cut, {"default": ["test_a", "test_e"], "slow": ["test_a", "test_e"]})  # fmt: skip
        timed = top25_per_config(tp, ["test_a", "test_b", "test_c", "test_d"], SECONDS)
        self.assertEqual(timed, {"default": ["test_a", "test_c"], "slow": ["test_a"]})
        td = {"top25": ["test_a", "test_e"], "top25_cfg": cut}
        choice = pick("td25", None, td)
        self.assertTrue(choice.covers("test_a", "slow"))
        self.assertTrue(choice.covers("test_e", "slow"))
        self.assertFalse(choice.covers("test_c", "slow"))
        self.assertTrue(choice.covers("test_e", "not_in_test_times"))
        self.assertEqual(choice.files(), {"test_a", "test_e"})

    def test_cost_fraction(self) -> None:
        cands = {"test_a", "test_b", "test_c"}
        choice = Choice(Selection({"test_b"}), {"slow": {"test_a"}})
        self.assertEqual(cost_fraction(choice, cands, SECONDS), (50 / 160, 250 / 250))
        only_b = Choice(Selection({"test_b"}))
        self.assertEqual(cost_fraction(only_b, cands, SECONDS), (50 / 160, 50 / 250))
        self.assertEqual(
            cost_fraction(only_b, cands, SECONDS, lambda c: c == "default"),
            (50 / 160, 50 / 50),
        )
        self.assertEqual(
            cost_fraction(Choice(Selection(run_all=True)), cands, SECONDS), (1.0, 1.0)
        )


class TestChart(unittest.TestCase):
    def test_frontier(self) -> None:
        points: list[tuple[float, float]] = [(1, 50), (3, 70), (1, 60), (2, 55), (0.5, 40), (3, 70)]  # fmt: skip
        self.assertEqual(chart.frontier(points), [(0.5, 40), (1, 60), (3, 70)])

    def test_caught(self) -> None:
        row = {"tag": "t", "policy": "p", "kind": "k", "files": 1, "run_all": False, "cpu": 0.1, "gpu": 0.1, "cost_usd": 0.1, "error": False}  # fmt: skip
        rows = [{**row, "pairs": p, "hit": h} for p, h in [(2, 1), (1, 1), (0, 0)]]
        (s,) = score.summarize(rows)
        # A revert is caught by any selected failure, a PR push only by all of them.
        self.assertEqual(chart.caught(s, "revert"), (100.0, 2))
        self.assertEqual(chart.caught(s, "head"), (50.0, 2))
        (empty,) = score.summarize(rows[2:])
        with self.assertRaises(SystemExit):
            chart.caught(empty, "head")


LABELER = """\
"ciflow/inductor":
- torch/_inductor/**

- c10/core/Sym*
"ciflow/other":
- torch/foo.py
"""


class TestDollars(unittest.TestCase):
    def write(self, files: dict[str, object]) -> Path:
        data = Path(self.enterContext(tempfile.TemporaryDirectory[str]()))
        for name, obj in files.items():
            (data / name).parent.mkdir(parents=True, exist_ok=True)
            text = obj if isinstance(obj, str) else json.dumps(obj)
            (data / name).write_text(text)
        return data

    def test_inductor_globs(self) -> None:
        data = self.write({"labeler.yml": LABELER, "bad.yml": '"ciflow/x":\n- y\n'})
        globs = dollars.inductor_globs(data / "labeler.yml")
        for path, match in [
            ("torch/_inductor/codegen/a.py", True),
            ("c10/core/SymInt.h", True),
            ("c10/core/impl/SymInt.h", False),
            ("torch/foo.py", False),
        ]:
            self.assertEqual(bool(globs.fullmatch(path)), match, path)
        with self.assertRaises(SystemExit):
            dollars.inductor_globs(data / "bad.yml")

    def test_model(self) -> None:
        sample = {
            "kind": "neutral",
            "sha": "x",
            "candidates": ["test_a", "test_b", "test_c"],
        }
        td = {"scored": [], "top25": ["test_a"], "top25_cfg": {"default": ["test_a"]}}
        result = {"out": {"tests": [{"file": "test_b"}]}, "error": None}
        data = self.write({
            "labeler.yml": LABELER,
            "test-times.json": {
                "linux": {"default": {"test_a": 80.0, "test_b": 20.0}, "inductor_core": {"test_c": 100.0}},
                "linux-cuda": {"default": {"test_a": 50.0, "test_b": 50.0}},
            },
            "samples/a.json": {**sample, "changed_files": ["torch/_inductor/x.py"]},
            "samples/b.json": {**sample, "changed_files": ["torch/foo.py"]},
            "samples/c.json": {**sample, "changed_files": ["torch/foo.py"]},
            "td/a.json": td, "td/b.json": td, "td/c.json": td,
            "results/m/a.json": {**result, "cost_usd": 0.2},
            "results/m/b.json": {**result, "cost_usd": None},
        })  # fmt: skip
        jobs = {True: {("pull", "linux", "default")}, False: {("inductor", "unit-test / inductor", "inductor_core")}}  # fmt: skip
        workflows = [
            ("pull", "cpu", 100.0, 0.5, 0.2, True),
            ("inductor", "ind_cpu", 10.0, 1.0, 0.0, False),
        ]
        with (
            patch_ci_jobs(jobs),
            mock.patch.object(dollars, "ci_jobs", lambda: jobs),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            d = dollars.model(data, "m", "final", 0.2, ["a", "b", "c"], ["a", "b"], data / "samples", data / "td", data / "labeler.yml", workflows)  # fmt: skip
            today = dollars.model(data, "td", "td25", 0.2, ["a", "b", "c"], ["a", "b"], data / "samples", data / "td", data / "labeler.yml", workflows)  # fmt: skip
        self.assertEqual(
            (today["llm_k_per_month"], today["stack_k_per_month"]), (0.0, 0.0)
        )
        # c has no m result, so TD is averaged over a and b too.
        self.assertEqual(d["fractions"], {"cpu": 0.2, "gpu": 0.5, "ind_cpu": 0.0, "ind_gpu": 0.0, "n": 2, "n_ind": 1})  # fmt: skip
        self.assertEqual(d["td_fractions"], {"cpu": 0.8, "gpu": 0.5, "ind_cpu": 0.0, "ind_gpu": 0.0, "n": 2, "n_ind": 1})  # fmt: skip
        # pull: 100 * 0.5 PR * 0.8 same-repo, rescaled by (0.2 + 0.8 * 0.2) / (0.2 + 0.8 * 0.8).
        pull, ind = d["rows"]
        self.assertAlmostEqual(pull["save_k"], 40 * (1 - 0.36 / 0.84))
        self.assertAlmostEqual(ind["save_k"], 10 * (1 - 0.2))
        self.assertAlmostEqual(d["stack_k_per_month"], pull["save_k"])
        self.assertAlmostEqual(d["total_k_per_month"]["cpu"], pull["save_k"] + 8)
        # The call without a recorded cost is left out of the mean.
        llm = d["llm_k_per_month"]
        self.assertAlmostEqual(llm, 0.2 * dollars.PR_PUSHES_PER_MONTH_K)

    def test_fractions_find_importers(self) -> None:
        # The change edits test_b, which test_a imports, so final runs test_a too, as score.score does.
        data = self.write({
            "labeler.yml": LABELER,
            "test-times.json": {"linux": {"default": {"test_a": 80.0, "test_b": 20.0}}},
            "samples/a.json": {"kind": "neutral", "sha": "x", "candidates": ["test_a", "test_b"], "changed_files": ["test/test_b.py", "torch/_inductor/x.py"]},
            "td/a.json": {"scored": [], "top25": [], "top25_cfg": {}},
            "results/m/a.json": {"out": {"tests": [{"file": "test_b"}]}, "error": None},
        })  # fmt: skip
        jobs = {True: {("pull", "linux", "default")}, False: {("inductor", "unit-test / inductor", "inductor_core")}}  # fmt: skip
        with (
            patch_ci_jobs(jobs),
            mock.patch.object(dollars, "ci_jobs", lambda: jobs),
            mock.patch.object(
                score, "grep_at", return_value=["test/test_a.py"]
            ) as grep,
        ):
            f = dollars.fractions(data, ["a"], "m", "final", data / "samples", data / "td", data / "labeler.yml")  # fmt: skip
        self.assertEqual({c.args[1] for c in grep.call_args_list}, {"x"})
        self.assertEqual(f["cpu"], 1.0)


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
            fail("test_ahead", "2026-10-05 06:00:00", sha="o", br="gh/x"),
        ]
        disabled = [{"d": "2026-09-30", "name": "test_disabled (__main__.TestA)"}]
        t = "2026-10-05 00:00:00"
        rows = [
            fail(name, t, pr=1, **kw)
            for name, kw in [("test_real", {}), ("test_green", {"job_green": True}), ("test_main", {}),
                             ("test_other", {}), ("test_mine", {}), ("test_old", {}), ("test_disabled", {}), ("test_landed", {}), ("test_landed_pr", {}),
                             ("test_unmapped", {}), ("test_edge_in", {}), ("test_edge_out", {}), ("test_ahead", {})]
        ]  # fmt: skip
        rows.append(fail("test_unstable", t, pr=1, job="pull / test (default, 1, 1, unstable)"))  # fmt: skip
        rows.append(fail("test_after_landing", "2026-10-06 00:00:00", pr=1))
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
             "test_edge_in": True, "test_edge_out": False, "test_after_landing": True, "test_ahead": True},
        )  # fmt: skip

    def test_pr_caused_failures(self) -> None:
        def fail(test: str, t: str, sha: str, br: str = "gh/x", cls: str = "TestA", wf: str = "pull", job: str = "pull / test (default, 1, 1)") -> dict:  # fmt: skip
            return {"tf": "test_a", "classname": cls, "test": test, "t": t, "sha": sha, "br": br, "wf": wf, "job": job}  # fmt: skip

        unstable = "pull / test (default, 1, 1, unstable)"
        rows = [
            fail("test_real", "2026-10-05 00:00:00", "a"),
            fail("test_main", "2026-10-05 00:00:00", "a"),
            fail("test_main", "2026-10-07 00:00:00", "m", br="main"),
            fail("test_cls", "2026-10-05 00:00:00", "a"),
            fail("test_cls", "2026-10-05 00:00:00", "m", br="main", cls="TestB"),
            fail("test_shared", "2026-10-05 00:00:00", "a"),
            fail("test_shared", "2026-10-06 00:00:00", "b"),
            # Another author's failure outside MAIN_WINDOW does not excuse it.
            fail("test_far", "2026-10-05 00:00:00", "a"),
            fail("test_far", "2026-09-25 00:00:00", "b"),
            # MAIN_WINDOW is 4 days, exclusive, for main and other authors.
            fail("test_main_in", "2026-10-05 00:00:00", "a"),
            fail("test_main_in", "2026-10-01 00:00:01", "m", br="main"),
            fail("test_main_out", "2026-10-05 00:00:00", "a"),
            fail("test_main_out", "2026-10-01 00:00:00", "m", br="main"),
            fail("test_other_in", "2026-10-05 00:00:00", "a"),
            fail("test_other_in", "2026-10-01 00:00:01", "b"),
            fail("test_other_out", "2026-10-05 00:00:00", "a"),
            fail("test_other_out", "2026-10-01 00:00:00", "b"),
            fail("test_before", "2026-09-30 23:59:59", "a"),
            # A fork's main is a PR branch.
            fail("test_fork_main", "2026-10-05 00:00:00", "a"),
            fail("test_fork_main", "2026-10-06 00:00:00", "g", br="f/pytorch:main"),
            # The same author's other PR (a ghstack sibling) does not excuse it.
            fail("test_same_author", "2026-10-05 00:00:00", "a"),
            fail("test_same_author", "2026-10-06 00:00:00", "c"),
            fail("test_unstable", "2026-10-05 00:00:00", "a", job=unstable),
            # After PR 1 landed, main and other PRs cannot excuse its failures, so they are dropped.
            fail("test_after_landing", "2026-10-08 00:00:00", "a"),
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
            mock.patch.object(bd, "pr_of_shas", return_value={"a": 1, "b": 2, "c": 4}),
            mock.patch.object(bd, "pr_authors", return_value={1: "me", 2: "x", 4: "me"}),
        ):  # fmt: skip
            keep, main = bd.pr_caused_failures(rows, dt.date(2026, 10, 1), dt.date(2026, 10, 26), landed)  # fmt: skip
        self.assertEqual(sorted((r["test"], r["pr"]) for r in keep), [("test_broke", 1), ("test_broke_pr", 1), ("test_cls", 1), ("test_far", 1), ("test_main_out", 1), ("test_other_out", 1), ("test_other_out", 2), ("test_real", 1), ("test_same_author", 1), ("test_same_author", 4)])  # fmt: skip
        self.assertEqual(len(main[("test_a", "TestA", "test_main")]), 1)
        self.assertNotIn(("test_a", "TestA", "test_unmapped"), main)
        self.assertNotIn(("test_a", "TestA", "test_fork_main"), main)

    def test_landed_times_takes_the_last_trailer(self) -> None:
        log = "\n100 Reland\n\nPull Request resolved: https://github.com/pytorch/pytorch/pull/1\n\nPull Request resolved: https://github.com/pytorch/pytorch/pull/2\n\0\n200 x\n\0"  # fmt: skip
        with mock.patch.object(bd, "git", return_value=log):
            got = bd.landed_times(dt.date(2026, 10, 1))
        self.assertEqual(got, {2: dt.datetime.fromtimestamp(100, tz=dt.timezone.utc)})

    def test_revert_truth_jobs_commits(self) -> None:
        def run(*args: str, cwd: str, date: str = "2026-10-05T00:00:00+00:00") -> str:
            env = {**os.environ, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date, "GIT_AUTHOR_NAME": "a", "GIT_AUTHOR_EMAIL": "a@a", "GIT_COMMITTER_NAME": "a", "GIT_COMMITTER_EMAIL": "a@a"}  # fmt: skip
            return subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=True).stdout  # fmt: skip

        with tempfile.TemporaryDirectory() as d:
            run("init", "-q", cwd=d)
            commits = {}
            for name, hour in (("base", 0), ("change", 1), ("after", 2), ("revert", 3), ("late", 5)):  # fmt: skip
                msg = f"This reverts commit {commits['change']}." if name == "revert" else name  # fmt: skip
                run("commit", "-q", "--allow-empty", "-m", msg, cwd=d, date=f"2026-10-05T0{hour}:00:00+00:00")  # fmt: skip
                commits[name] = run("rev-parse", "HEAD", cwd=d).strip()
            run("update-ref", "refs/remotes/origin/main", "HEAD", cwd=d)
            queries: list[str] = []
            landed = dt.datetime(2026, 10, 5, 1, tzinfo=dt.timezone.utc)
            with (
                mock.patch.object(bd, "git", lambda *a, check=True: run(*a, cwd=d)),
                mock.patch.object(bd, "query", lambda sql: queries.append(sql) or [{"tf": "test_a", "wf": "trunk", "job": "j"}]),
            ):  # fmt: skip
                got = bd.revert_truth_jobs({"test_a"}, commits["change"], landed, "2026-10-05 04:00:00")  # fmt: skip
                # Decided before the change landed: no commit qualifies, so only the change itself.
                bd.revert_truth_jobs({"test_a"}, commits["change"], landed, "2026-10-05 00:30:00")  # fmt: skip
        self.assertEqual(got, {"test_a": [("trunk", "j")]})
        # The change and the commit after it, not the revert or what landed after the decision.
        self.assertIn(f"head_sha IN ('{commits['after']}','{commits['change']}')", queries[0])  # fmt: skip
        self.assertIn(f"head_sha IN ('{commits['change']}')", queries[1])

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
            mock.patch.object(bd, "pr_text", side_effect=error) as pr_text,
            mock.patch.object(bd, "write_sample") as write,
        ):  # fmt: skip
            bd.pr_samples(keep, {}, Path(out), 1, dt.date(2026, 10, 1), {})
        pr_text.assert_called_once_with(1)
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
        truth_jobs.assert_called_once_with({"test_a", "test_b"}, "c" * 40, mock.ANY, "1")  # fmt: skip
        kw = write.call_args.kwargs
        self.assertEqual((kw["truth"], kw["truth_jobs"]), (["test_a"], jobs))
        with (
            tempfile.TemporaryDirectory() as out,
            mock.patch.object(bd, "query", return_value=rows),
            mock.patch.object(bd, "fetch"),
            mock.patch.object(bd, "has_commit", return_value=True),
            mock.patch.object(bd, "commit_time"),
            mock.patch.object(bd, "revert_truth_jobs", return_value={}),
            mock.patch.object(bd, "git", return_value="t\nb"),
            mock.patch.object(bd, "write_sample") as write,
            contextlib.redirect_stdout(io.StringIO()),
        ):  # fmt: skip
            bd.revert_samples(Path(out), 1, dt.date(2026, 10, 1), dt.date(2026, 10, 26))
        write.assert_not_called()

    def test_neutral_samples(self) -> None:
        start, end = dt.date(2026, 10, 1), dt.date(2026, 10, 26)
        titles: dict[str, str] = {"a" * 40: "Fix foo", "b" * 40: "[ghstack] Revert bar", "c" * 40: "Reland: baz", "d" * 40: "Add RevertibleOp"}  # fmt: skip
        # Reverted by a message, by the autorevert bot, and with a rev_ sample.
        titles |= {"e" * 40: "Add e", "f" * 40: "Add f", "0" * 40: "Add 0"}
        calls = []

        def git(*args: str) -> str:
            calls.append(args)
            if args[:2] == ("log", "-1"):
                return titles[args[-1]] + "\n\nbody"
            if "--first-parent" in args:
                return "".join(f"{sha} {t}\n" for sha, t in titles.items())
            return f"Revert\n\nThis reverts commit {'e' * 40}.\n"

        with (
            tempfile.TemporaryDirectory() as out,
            mock.patch.object(bd, "git", git),
            mock.patch.object(bd, "query", return_value=[{"commit_sha": "f" * 40}]),
            mock.patch.object(bd, "write_sample") as write,
        ):
            (Path(out) / f"rev_{'0' * 12}.json").write_text("{}")
            bd.neutral_samples(Path(out), start, end, 10)
        self.assertEqual(sorted(c.args[1] for c in write.call_args_list), ["neu_" + "a" * 12, "neu_" + "d" * 12])  # fmt: skip
        log = next(c for c in calls if "--first-parent" in c)
        self.assertIn("--since=2026-10-01 00:00 +0000", log)
        self.assertIn("--until=2026-10-26 00:00 +0000", log)

    def test_neutral_draw_survives_a_later_revert(self) -> None:
        shas = [f"{i:02d}" + "a" * 38 for i in range(30)]

        def draw(revert_body: str, bot: list[dict]) -> list[str]:
            def git(*args: str) -> str:
                if args[:2] == ("log", "-1"):
                    return "Add x\n\nbody"
                if "--first-parent" in args:
                    return "".join(f"{sha} Add x\n" for sha in shas)
                return revert_body

            with (
                tempfile.TemporaryDirectory() as out,
                mock.patch.object(bd, "git", git),
                mock.patch.object(bd, "query", return_value=bot) as query,
                mock.patch.object(bd, "write_sample") as write,
            ):
                bd.neutral_samples(Path(out), dt.date(2026, 10, 1), dt.date(2026, 10, 26), 10)  # fmt: skip
            self.assertIn("dry_run = 0", query.call_args.args[0])
            return [c.args[3] for c in write.call_args_list]

        before = draw("", [])
        by_msg, by_bot = before[:2]
        later = draw(f"This reverts commit {by_msg}.", [{"commit_sha": by_bot}])
        self.assertEqual(later, before[2:])

    def test_checked_in_ids(self) -> None:
        ids = {n: set((REPO_ROOT / "tools/testing/llm_td/eval/ids" / f"{n}.txt").read_text().split()) for n in ("eval", "dev", "agentic", "neutral", "head")}  # fmt: skip
        self.assertLessEqual(ids["neutral"], ids["eval"])
        self.assertLessEqual(ids["agentic"], ids["eval"])
        self.assertEqual(ids["dev"] & ids["eval"], set())
        every = set().union(*ids.values())
        shas = {k: {i.removeprefix(f"{k}_") for i in every if i.startswith(f"{k}_")} for k in ("neu", "rev")}  # fmt: skip
        self.assertEqual(shas["neu"] & shas["rev"], set())
        prs = {i for i in ids["eval"] | ids["dev"] if i.startswith("pr_")}
        self.assertEqual({f"pr_{i.split('_')[1]}" for i in ids["head"]} - prs, set())

    def test_candidates_at(self) -> None:
        ls = "test/test_a.py\ntest/ao/test_b.py\ntest/fx/test_c.py\ntest/test_skip.py\ntest/helper.py\n"
        added = "test/test_new.py\ntest/ao/test_d.py\ntest/fx/test_e.py\ntest/helper_new.py\n"
        named = 'BLOCKLISTED_PATTERNS = ["ao"]\nTESTS = discover_tests(blocklisted_patterns=BLOCKLISTED_PATTERNS, blocklisted_tests=["test_skip"], extra_tests=["extra"])\n'  # fmt: skip
        inline = 'TESTS = discover_tests(blocklisted_patterns=["fx"], blocklisted_tests=["test_skip"])\n'  # fmt: skip

        def git(*a: str) -> str:
            if a[0] == "diff" and not {"--diff-filter=A", "--no-renames"} <= set(a):
                raise AssertionError(f"not only the added files: {a}")
            return {"show": src, "ls-tree": ls, "diff": added}[a[0]]

        for src, want, new in ((named, ["extra", "fx/test_c", "test_a"], ["fx/test_e", "test_new"]), (inline, ["ao/test_b", "test_a"], ["ao/test_d", "test_new"])):  # fmt: skip
            with mock.patch.object(bd, "git", git):
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
