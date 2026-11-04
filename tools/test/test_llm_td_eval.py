from __future__ import annotations

import argparse
import ast
import collections
import contextlib
import datetime as dt
import functools
import io
import json
import os
import random
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
    from collections.abc import Iterator, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from tools.testing.llm_td import select_tests, selection
from tools.testing.llm_td.eval import build_dataset as bd, clickhouse, score, td_replay
from tools.testing.llm_td.eval.build_dataset import strip_trailers
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
from tools.testing.llm_td.selection import apply_selection, Selection
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
    "title": "t",
    "body": None,
    "diffstat": "",
    "diff": "",
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


def write_data(d: str, files: Mapping[str, object]) -> Path:
    """Write `files` under `d` (str values as is, others as JSON), stamping each TD replay with its sample's current inputs and each result with no prompt_sha with its sample's."""
    data = Path(d)
    for name, obj in files.items():
        (data / name).parent.mkdir(parents=True, exist_ok=True)
        (data / name).write_text(obj if isinstance(obj, str) else json.dumps(obj))
    if not (data / "test-times.json").exists():
        return data
    times = (data / "test-times.json").read_bytes()
    for p in (data / "td").glob("*.json"):
        if (sample := data / "samples" / p.name).exists():
            inputs = score.replay_inputs("", times, sample.read_bytes())
            p.write_text(json.dumps({**json.loads(p.read_text()), "inputs": inputs}))
    minutes = select_tests.test_minutes(json.loads(times))
    for p in (data / "results").glob("*/*.json"):
        r = json.loads(p.read_text())
        if "prompt_sha" not in r and (sample := data / "samples" / p.name).exists():
            sent = score.prompt_sha(score.sample_prompt(json.loads(sample.read_text()), minutes))  # fmt: skip
            p.write_text(json.dumps({**r, "prompt_sha": sent}))
    return data


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


START, END = dt.date(2026, 10, 1), dt.date(2026, 10, 26)
JOB = "pull / test (default, 1, 1)"
TD_JOBS = {
    True: {("pull", "pull", "default")},
    False: {("rocm-rx7900", "pull", "default")},
}


def fail(test: str, t: str, sha: str, br: str = "gh/x", wf: str = "pull", job: str = JOB, tf: str = "test_a") -> dict:  # fmt: skip
    return {"tf": tf, "classname": "TestA", "test": test, "t": t, "sha": sha, "br": br, "wf": wf, "job": job}  # fmt: skip


def day(d: int, month: int = 10) -> dt.datetime:
    return dt.datetime(2026, month, d, tzinfo=dt.timezone.utc)


def history(rows: list[dict], sha2pr: dict[str, int], authors: dict[int, str], landed: dict[int, dt.datetime] | None = None, disabled: tuple = ()) -> bd.History:  # fmt: skip
    with (
        mock.patch.object(bd, "pull_failures", return_value=rows),
        mock.patch.object(bd, "pr_of_shas", return_value=sha2pr),
        mock.patch.object(bd, "pr_authors", return_value=authors),
        mock.patch.object(bd, "landed_times", return_value=landed or {}),
        mock.patch.object(bd, "query", return_value=[{"d": d, "name": f"{t} (__main__.TestA)"} for d, t in disabled]),
    ):  # fmt: skip
        return bd.History.load(START, END)


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

        from tools.testing.llm_td.eval.run_eval import TIMED_OUT

        r, query, _ = self.run_one([TimeoutExpired("claude", 1)])
        self.assertEqual((r["out"], r["error"], r["transient"], query.call_count), (None, TIMED_OUT, False, 1))  # fmt: skip
        # A timeout the CLI reports is as final as one query_model raises.
        r, query, _ = self.run_one([{"type": "result", "is_error": True, "result": "Request timed out"}])  # fmt: skip
        self.assertEqual((r["transient"], r["attempts"], query.call_count), (False, 1, 1))  # fmt: skip

    def test_transient(self) -> None:
        for text in ("API Error: 429 Too Many Requests", '{"type":"rate_limit_error"}', "ThrottlingException", "503"):  # fmt: skip
            r, query, _ = self.run_one([{"type": "result", "is_error": True, "result": text}, {"type": "result", "structured_output": {"tests": []}}])  # fmt: skip
            self.assertEqual((r["error"], r["attempts"]), (None, 2), text)

    def test_gives_up(self) -> None:
        r, query, sleep = self.run_one(
            [{"type": "result", "is_error": True, "result": "529 overloaded"}] * 4
        )
        self.assertEqual((r["out"], r["error"], r["transient"], r["attempts"]), (None, "529 overloaded", True, 4))  # fmt: skip
        self.assertEqual((query.call_count, sleep.call_count), (4, 3))

    def test_main_writes_exhausted_and_resolves_data(self) -> None:
        from tools.testing.llm_td.eval import run_eval

        sample = {"id": "a", "sha": "f" * 40, "title": "t", "body": None, "diffstat": "", "diff": "", "candidates": ["test_a"], "changed_files": ["torch/foo.py"]}  # fmt: skip
        with tempfile.TemporaryDirectory() as d:
            data = write_data(d, {"samples/a.json": sample, "test-times.json": {}})
            (data / "ids.txt").write_text("a")
            argv = ["run_eval", "--data", data.name, "--tag", "t", "--model", "m", "--ids", str(data / "ids.txt"), "--jobs", "1", "--agentic"]  # fmt: skip
            # --data is relative to this cwd.
            with (
                mock.patch("os.getcwd", return_value=str(data.parent)),
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(run_eval, "checkout", return_value="wt") as co,
                mock.patch.object(run_eval, "query_model", return_value={"type": "result", "is_error": True, "result": "529"}),
                mock.patch.object(run_eval.time, "sleep"),
                mock.patch("sys.stdout", io.StringIO()),
            ):  # fmt: skip
                run_eval.main()
            root = co.call_args.args[0]
            self.assertEqual(root, data.resolve() / "worktrees/t")
            r = json.loads((data / "results/t/a.json").read_text())
        self.assertEqual(
            (r["error"], r["transient"], r["agentic"]), ("529", True, True)
        )

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

    def test_main_refuses_another_prompt(self) -> None:
        from tools.testing.llm_td.eval import run_eval

        sample = {"id": "a", "sha": "f" * 40, "title": "t", "body": None, "diffstat": "", "diff": "", "candidates": ["test_a"], "changed_files": ["torch/foo.py"]}  # fmt: skip
        sent = run_eval.prompt_sha(run_eval.sample_prompt(sample, {}))
        result = {"model": "m", "effort": "medium", "agentic": False}
        for prompt_sha, schema, refused in ((sent, score.SCHEMA, False), (None, score.SCHEMA, True), (sent, {}, True)):  # fmt: skip
            with tempfile.TemporaryDirectory() as d:
                data = write_data(d, {"samples/a.json": sample, "test-times.json": {}, "results/t/a.json": {**result, "prompt_sha": prompt_sha}})  # fmt: skip
                (data / "ids.txt").write_text("a")
                argv = ["run_eval", "--data", d, "--tag", "t", "--model", "m", "--ids", str(data / "ids.txt")]  # fmt: skip
                with (
                    mock.patch.object(sys, "argv", argv),
                    mock.patch.object(score, "SCHEMA", schema),
                    mock.patch("sys.stderr", io.StringIO()) as err,
                    mock.patch("sys.stdout", io.StringIO()),
                    mock.patch.object(run_eval, "query_model") as query,
                ):
                    try:
                        run_eval.main()
                    except SystemExit:
                        self.assertTrue(refused)
                        self.assertIn("another prompt", err.getvalue())
                    else:
                        self.assertFalse(refused)
                query.assert_not_called()


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
                        { include: [{ config: "dynamo_cpython", shard: 1 }, { config: "dynamo_cpython_313", shard: 1 }] }
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
                ("pull", "dynamo-cpython-test", "dynamo_cpython_313"),
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
        self.assertEqual({c for c, f in dispatch.items() if f in funcs}, set(score.NO_TD_CONFIGS))  # fmt: skip

    def test_td_check_matches_run_test(self) -> None:
        # td_workflow, td_job and NO_TD_CONFIGS model this check; update them with it.
        tree = ast.parse((REPO_ROOT / "test/run_test.py").read_text())
        (default,) = [
            k.value
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and n.args
            and isinstance(n.args[0], ast.Constant)
            and n.args[0].value == "--enable-td"  # fmt: skip
            for k in n.keywords
            if k.arg == "default"
        ]
        self.assertEqual(
            ast.unparse(default),
            "IS_CI and get_pr_number() is not None and (not strtobool(os.environ.get('NO_TD', 'False'))) and (not IS_MACOS) and ('xpu' not in BUILD_ENVIRONMENT) and ('onnx' not in BUILD_ENVIRONMENT) and (GITHUB_WORKFLOW in ('trunk', 'pull') or GITHUB_WORKFLOW.startswith(('rocm-', 'periodic-rocm-')))",
        )
        (main,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"]  # fmt: skip
        (tti,) = [n for n in main.body if isinstance(n, ast.If) and ast.unparse(n.test) == "tests_to_include_env"]  # fmt: skip
        self.assertEqual(ast.unparse(tti.body[-1]), "options.enable_td = False")

    def test_unasked(self) -> None:
        td = {"scored": ["test_a"], "top25": ["test_e"], "top25_cfg": {"default": ["test_a"]}}  # fmt: skip
        r = {"out": {"tests": [{"file": "test_b"}]}}
        huge = {**SAMPLE, "changed_files": [f"torch/f{i}.py" for i in range(select_tests.MAX_CHANGED_FILES + 1)]}  # fmt: skip
        empty = {**SAMPLE, "changed_files": []}
        infra = {**SAMPLE, "changed_files": ["test/run_test.py"]}
        for name in ("model", "final", "final-hm", "final+td45"):
            for s in (huge, empty, infra):
                self.assertTrue(pick(name, r, td, s).sel.run_all, name)
        # As apply_selection does not enforce a job with no test times, even a run-all keeps TD's cut there.
        zero = {**SECONDS, "zero": {"test_a": [0.0, 0.0]}}
        cut = {**td, "top25_cfg": {"zero": ["test_b"]}}
        for s, out in ((empty, None), (SAMPLE, {"tests": [], "run_all": True})):
            choice = policy("final+td45", s, {"out": out}, cut, zero)
            if choice is None:
                raise AssertionError("final+td45 chose nothing")
            self.assertEqual([choice.covers(f, c) for f, c in (("test_a", "zero"), ("test_b", "zero"), ("test_c", "default"))], [False, True, True])  # fmt: skip
        files = {
            "test-times.json": {"linux": {"default": {"test_a": 10.0}}},
            "samples/a.json": huge,
            "td/a.json": td,
        }  # fmt: skip
        with (
            tempfile.TemporaryDirectory() as d,
            patch_ci_jobs({True: {("pull", "linux", "default")}, False: set()}),
        ):
            data = write_data(d, files)
            rows = score.score(data, ["a"], ["m"], ["model"], True, data / "samples", data / "td", "")  # fmt: skip
        self.assertEqual([(x["run_all"], x["cost_usd"], x["error"]) for x in rows], [(True, None, False)])  # fmt: skip

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
        # So does an untimed file, which is in every job's test list.
        r = {"out": {"tests": [{"file": "test_e"}]}}
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
        # With no fallback job times, unknown configs hedge nothing and keep TD's cut.
        self.assertEqual(choice.other, set())
        # As in run_test.py, an unknown config is hedged over the candidates in default/<config>, else default/default.
        fallback = {"default": {"test_a": 30.0, "test_b": 40.0, "test_d": 30.0, "test_x": 1.0}, "rare": {"test_b": 1.0, "test_c": 9.0}}  # fmt: skip
        choice = policy("final+td45", SAMPLE, NOTHING, td, SECONDS, None, fallback)
        if choice is None:
            raise AssertionError("final+td45 chose nothing")
        self.assertEqual((choice.other, choice.per_config["rare"]), ({"test_a"}, {"test_b"}))  # fmt: skip
        self.assertEqual(choice.per_config["default"], {"test_a", "test_c"})
        self.assertTrue(choice.covers("test_a", "unknown"))
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

    def test_hedge_matches_split(self) -> None:
        # hedge is Selection.split for any share LLM_TD_HEDGE could be.
        rng = random.Random(0)
        files = [f"test_{i}" for i in range(8)]
        for i in range(500):
            share = (0.05, 0.2, 0.45)[i % 3]
            scored = rng.sample(files, rng.randint(0, 8))
            seconds = {f: float(rng.choice([0, 1, 5, 20, 60])) for f in files if rng.random() < 0.7}  # fmt: skip
            untimed = [f for f in files if f not in seconds and rng.random() < 0.5]
            job = [*seconds, *untimed]
            sel = Selection(set(rng.sample(files, rng.randint(0, 2))), candidates=set(files))  # fmt: skip
            ranked = [TestRun(f) for f in scored if f in job] + [TestRun(f) for f in job if f not in scored]  # fmt: skip
            cut = set(rng.sample(job, min(len(job), 2)))
            prev = set(rng.sample(files, rng.randint(0, 2)))
            with mock.patch.object(selection, "LLM_TD_HEDGE", share):
                split = sel.split(ranked, lambda x: float(x.test_file in scored), seconds, "c", previously_failed=prev, td_include=cut)  # fmt: skip
            got = score.hedge(sel, scored, seconds, share, "c", cut, untimed, prev)
            want = cut if split.chose_nothing or not sum(seconds.values()) else split.previously_failed | split.hedged  # fmt: skip
            case = (share, scored, seconds, untimed, sel, cut, prev)
            self.assertEqual(got, want, case)

    def test_policy_matches_apply_selection(self) -> None:
        # What final+td runs in a job is what apply_selection runs there, including a job with
        # no test times, a run-all, a failed or malformed answer, and a job timed by a fallback job.
        rng = random.Random(0)
        files = [f"test_{i}" for i in range(8)]
        for _ in range(600):
            seconds = {f: float(rng.choice([0, 0, 5, 20, 60])) for f in files if rng.random() < 0.6}  # fmt: skip
            job = [*seconds, *(f for f in files if f not in seconds and rng.random() < 0.5)]  # fmt: skip
            rng.shuffle(job)
            n = rng.randint(0, len(job))
            include, exclude = [TestRun(f) for f in job[:n]], [TestRun(f) for f in job[n:]]  # fmt: skip
            scored = [f for f in job if rng.random() < 0.6]
            picked = rng.sample(job, min(len(job), rng.randint(0, 2)))
            prev = rng.sample(files, rng.randint(0, 2))
            changed = rng.choice([["torch/foo.py"]] * 3 + [[]])
            s = {**SAMPLE, "candidates": job, "truth": [], "previous_failures": prev, "changed_files": changed}  # fmt: skip
            out = rng.choice([{"tests": [{"file": f} for f in picked]}] * 3 + [{"tests": [], "run_all": True}, {"tests": "test_0"}, None])  # fmt: skip
            # The job's times are its own, the fallback job's for its config, or default/default's.
            config, how = rng.choice([("c", "own"), ("c", "fallback"), ("unknown", "default")])  # fmt: skip
            own = {"c": {f: [t, 0.0] for f, t in seconds.items()}} if how == "own" else {}  # fmt: skip
            fallback = {} if how == "own" else {"c" if how == "fallback" else "default": seconds}  # fmt: skip
            td = {"scored": scored, "top25": job[:n], "top25_cfg": {"c": job[:n]} if how == "own" else {}}  # fmt: skip
            choice = policy("final+td", s, {"out": out}, td, own, None, fallback)
            if choice is None:
                raise AssertionError("final+td chose nothing")
            # What select_tests writes: no model call without changed files, and an error is never enforced.
            sel = Selection(run_all=True) if not changed else Selection.from_model_output(out, set(job), changed)  # fmt: skip
            sel.enforce = sel.error is None
            with (
                mock.patch.object(selection, "get_pr_number", return_value=1),
                mock.patch("sys.stderr", io.StringIO()),
            ):
                applied = apply_selection(sel, include, exclude, lambda x: float(x.test_file in scored), seconds, test_config=config, td_enabled=True, previously_failed=prev, call_id="c")  # fmt: skip
            ran = {x.test_file for x in applied.include}
            self.assertEqual({f for f in job if choice.covers(f, config)}, ran, (seconds, job, n, scored, out, prev, changed, how))  # fmt: skip

    def test_hedge_previous_failures(self) -> None:
        # Previous failures in TD's cut run outside the budget; test_b is not in the cut.
        td = {"scored": ["test_a", "test_c"], "top25": ["test_a"], "top25_cfg": {"default": ["test_a"], "slow": []}}  # fmt: skip
        s = {**SAMPLE, "previous_failures": ["test_a", "test_b"]}
        choice = pick("final+td5", NOTHING, td, s)
        self.assertEqual(choice.per_config, {"default": {"test_a", "test_c"}, "slow": set()})  # fmt: skip
        # Without fallback times other jobs are not enforced and run TD's cut.
        self.assertEqual(choice.other, {"test_a"})
        self.assertEqual(pick("final+td5", NOTHING, td).per_config["default"], {"test_c"})  # fmt: skip

    def test_hedge_failed_call_keeps_td(self) -> None:
        td = {"scored": ["test_a"], "top25": ["test_e"], "top25_cfg": {"default": ["test_a"], "slow": []}}  # fmt: skip
        for r in ({"out": None}, None, {"out": {"tests": "test_a"}}):
            choice = pick("final+td45", r, td)
            self.assertFalse(choice.sel.run_all)
            self.assertEqual(choice.per_config, {"default": {"test_a"}, "slow": set()})
            self.assertEqual(choice.other, {"test_e"})
            self.assertFalse(choice.covers("test_torch", "inductor"))
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
            data = write_data(d, files)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rows = score.score(data, ["a", "b", "c"], ["m1", "m2", "td"], ["model"], True, data / "samples", data / "td", "")  # fmt: skip
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
            {**row, "pairs": 0, "hit": 0, **run_all, "error": False, "cost_usd": None},
        ]
        (s,) = score.summarize(rows)
        self.assertEqual(s, {
            "tag": "t", "policy": "p", "kind": "pr", "n": 5, "n_failing": 3,
            "caught_any": 1.0, "caught_all": 2 / 3, "pair_recall": 3 / 4,
            "cpu": 0.7, "gpu": 0.4, "run_all": 2, "median_files": 5,
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
        for grep, n_files, msg in (
            (mock.Mock(return_value=["test/test_a.py"]), [2, 2], ""),
            (mock.Mock(side_effect=RuntimeError("bad revision")), [], "dropped 1 of 1 ids with no working git grep at their head (not in this clone?)\n"),
        ):  # fmt: skip
            with (
                tempfile.TemporaryDirectory() as d,
                patch_ci_jobs({True: {("pull", "linux", "default")}, False: set()}),
                mock.patch.object(score, "grep_at", grep),
            ):
                data = write_data(d, files)
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    rows = score.score(data, ["a"], ["m"], ["final", "final-hm"], True, data / "samples", data / "td", "")  # fmt: skip
            self.assertEqual([x["files"] for x in rows], n_files)
            self.assertEqual(err.getvalue(), msg)
            self.assertEqual(grep.call_args.args[1], "abc")

    def test_grep_head_copies(self) -> None:
        cached = functools.cache(lambda pattern, sha: ["test/test_a.py"])
        with mock.patch.object(score, "grep_at", cached):
            grep = score.grep_head({"sha": "abc"}, "a", set())
            grep("p").append("test/test_b.py")
            self.assertEqual(grep("p"), ["test/test_a.py"])

    def test_score_errors_and_stale_replays(self) -> None:
        td = {"scored": [], "top25": [], "top25_cfg": {}}
        files = {
            "test-times.json": {"linux": {"default": {"test_a": 10.0}}},
            "samples/a.json": SAMPLE, "samples/b.json": SAMPLE,
            "td/a.json": td, "td/b.json": td,
            "results/m/a.json": {"out": {"tests": [{"file": "bogus"}]}, "error": None, "cost_usd": 0.1},
            "results/m/b.json": {"out": {"tests": []}, "error": None, "cost_usd": 0.1},
        }  # fmt: skip
        with (
            tempfile.TemporaryDirectory() as d,
            patch_ci_jobs({True: {("pull", "linux", "default")}, False: set()}),
        ):
            data = write_data(d, files)
            (data / "samples/b.json").write_text(json.dumps({**SAMPLE, "sha": "rebuilt"}))  # fmt: skip
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rows = score.score(data, ["a", "b"], ["m"], ["model", "final"], True, data / "samples", data / "td", "")  # fmt: skip
                # Replays of another STATS snapshot or TD code are stale too, even ones that record it.
                (data / "td/a.json").write_text(json.dumps({**json.loads((data / "td/a.json").read_text()), "stats": "new"}))  # fmt: skip
                stale = score.score(data, ["a", "b"], ["m"], ["model"], True, data / "samples", data / "td", "new")  # fmt: skip
        # A malformed answer runs everything and counts as an error, as select_tests records it.
        self.assertEqual([(x["id"], x["run_all"], x["error"]) for x in rows], [("a", True, True)] * 2)  # fmt: skip
        self.assertEqual(stale, [])
        why = "td replay of the current test-times.json, sample, STATS and TD code"
        self.assertEqual(err.getvalue(), f"dropped 1 of 2 ids with no {why}\ndropped 2 of 2 ids with no {why}\n")  # fmt: skip

    def test_score_stale_replay_of_test_times(self) -> None:
        files = {
            "test-times.json": {"linux": {"default": {"test_a": 10.0}}},
            "samples/a.json": SAMPLE,
            "td/a.json": {"scored": [], "top25": [], "top25_cfg": {}},
        }
        with (
            tempfile.TemporaryDirectory() as d,
            patch_ci_jobs({True: {("pull", "linux", "default")}, False: set()}),
        ):
            data = write_data(d, files)
            (data / "test-times.json").write_text(json.dumps({"linux": {"default": {"test_a": 20.0}}}))  # fmt: skip
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rows = score.score(data, ["a"], ["td"], [], True, data / "samples", data / "td", "")  # fmt: skip
        self.assertEqual(rows, [])
        self.assertEqual(err.getvalue(), "dropped 1 of 1 ids with no td replay of the current test-times.json, sample, STATS and TD code\n")  # fmt: skip

    def test_score_stale_results(self) -> None:
        td = {"scored": [], "top25": [], "top25_cfg": {}}
        result = {"out": {"tests": []}, "error": None, "cost_usd": 0.1}
        files = {
            "test-times.json": {"linux": {"default": {"test_a": 10.0}}},
            "samples/a.json": SAMPLE, "samples/b.json": SAMPLE, "samples/c.json": SAMPLE,
            "td/a.json": td, "td/b.json": td, "td/c.json": td,
            "results/m/a.json": result, "results/m/b.json": {**result, "prompt_sha": "old"}, "results/m/c.json": result,
        }  # fmt: skip
        with (
            tempfile.TemporaryDirectory() as d,
            patch_ci_jobs({True: {("pull", "linux", "default")}, False: set()}),
        ):
            data = write_data(d, files)
            # run_eval always records prompt_sha, so a result without one is stale too.
            (data / "results/m/c.json").write_text(json.dumps(result))
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rows = score.score(data, ["a", "b", "c"], ["m", "td"], ["model"], True, data / "samples", data / "td", "")  # fmt: skip
        self.assertEqual({x["id"] for x in rows}, {"a"})
        self.assertEqual(err.getvalue(), "dropped 2 of 3 ids with no m result of the current sample, test-times.json, prompt and schema\n")  # fmt: skip

    def test_test_seconds_skips_non_td_envs(self) -> None:
        times = {
            "linux": {"default": {"test_a": 10.0}},
            "macos-py3-arm64": {"default": {"test_a": 100.0}},
            "linux-jammy-xpu-n-py3.10": {"default": {"test_a": 1000.0}},
            "linux-jammy-py3-clang12-onnx": {"default": {"test_b": 5.0}},
        }
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test-times.json"
            path.write_text(json.dumps(times))
            self.assertEqual(score.test_seconds(path), {"default": {"test_a": [10.0, 0.0]}})  # fmt: skip

    def test_replay_stats(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            stats, code = Path(d) / "stats", Path(d) / "td"
            for p in [*(stats / a for a in score.STATS_ARTIFACTS), code / "heuristics/h.py"]:  # fmt: skip
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("1")
            replay = Path(d) / "td_replay.py"
            replay.write_text("1")
            with (
                mock.patch.object(score, "TD_CODE", code),
                mock.patch.object(score, "REPLAY_CODE", [replay]),
            ):
                digests = [score.replay_stats(stats)]
                (stats / score.STATS_ARTIFACTS[0]).write_text("2")
                digests.append(score.replay_stats(stats))
                (code / "heuristics/h.py").write_text("2")
                digests += [score.replay_stats(stats), score.replay_stats(stats)]
                replay.write_text("2")
                digests.append(score.replay_stats(stats))
        self.assertEqual(len(set(digests)), 4)
        self.assertEqual(digests[2], digests[3])
        self.assertIn(Path(score.__file__).with_name("td_replay.py"), score.REPLAY_CODE)

    def test_td_replay_main(self) -> None:
        seen = []

        def prioritize(cands: list[str], file: io.StringIO) -> mock.Mock:
            from tools.testing.target_determination.heuristics import utils

            seen.append(utils.query_changed_files())
            if "test_e" not in cands:
                print("Error in Boom: x", file=file)
            tp = TestPrioritizations(cands, {TestRun("test_b"): 2.0, TestRun("test_a", included=["TestX"]): 1.0})  # fmt: skip
            return mock.Mock(get_aggregated_priorities=lambda: tp)

        with tempfile.TemporaryDirectory() as d:
            stats = Path(d) / ".additional_ci_files"
            stats.mkdir()
            for a in score.STATS_ARTIFACTS:
                (stats / a).write_text("{}")
            broken = {**SAMPLE, "candidates": ["test_a", "test_b"]}
            data = write_data(d, {"test-times.json": {"linux": {"default": {"test_a": 100.0, "test_c": 10.0}}}, "samples/a.json": SAMPLE, "samples/b.json": broken, "td/b.json": {}})  # fmt: skip

            def replay(*argv: str) -> None:
                with (
                    mock.patch.object(sys, "argv", ["td_replay", "--data", d, "--stats", str(stats), *argv]),
                    mock.patch("tools.testing.target_determination.determinator.get_test_prioritizations", prioritize),
                    mock.patch("sys.stdout", io.StringIO()),
                ):  # fmt: skip
                    td_replay.main()

            with self.assertRaisesRegex(SystemExit, r"a heuristic raised on 1 samples, left for a rerun: \['b'\]"):  # fmt: skip
                replay()
            self.assertEqual(seen, [["torch/foo.py"]] * 2)
            self.assertFalse((data / "td/b.json").exists())
            td = json.loads((data / "td/a.json").read_text())
            self.assertEqual((td["scored"], td["top25"], td["top25_cfg"]), (["test_b", "test_a"], ["test_a", "test_b"], {"default": ["test_a", "test_b"]}))  # fmt: skip
            # What score checks a replay against is what td_replay records.
            with patch_ci_jobs({True: {("pull", "linux", "default")}, False: set()}):
                rows = score.score(data, ["a"], ["td"], [], True, data / "samples", data / "td", score.replay_stats(stats))  # fmt: skip
            self.assertEqual(len(rows), 1)
            # A current replay is not recomputed.
            (data / "ids.txt").write_text("a")
            replay("--ids", str(data / "ids.txt"))
            self.assertEqual(len(seen), 2)
            for bad in (["--shard", "2", "2"], ["--stats", d]):
                with self.assertRaises(SystemExit), mock.patch("sys.stderr", io.StringIO()):  # fmt: skip
                    replay(*bad)

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
        s = {"candidates": cands, "truth": []}
        cut = top25_per_config(tp, score.job_files(s, SECONDS))
        self.assertEqual(cut, {"default": ["test_a", "test_e"], "slow": ["test_a", "test_e"]})  # fmt: skip
        timed = {**s, "candidates": ["test_a", "test_b", "test_c", "test_d"]}
        self.assertEqual(top25_per_config(tp, score.job_files(timed, SECONDS)), {"default": ["test_a", "test_c"], "slow": ["test_a"]})  # fmt: skip
        # A file that failed in slow was in slow's job, though slow records no time of it.
        failed = {**timed, "truth": ["test_d"], "truth_jobs": {"test_d": [["pull", "linux / test (slow, 1, 1)"]]}}  # fmt: skip
        self.assertEqual(score.job_files(failed, SECONDS)["slow"], {"test_a", "test_c", "test_d"})  # fmt: skip
        tp_d = TestPrioritizations(timed["candidates"], {TestRun("test_d"): 1.0})
        self.assertNotIn("test_d", top25_per_config(tp_d, score.job_files(timed, SECONDS))["slow"])  # fmt: skip
        self.assertEqual(top25_per_config(tp_d, score.job_files(failed, SECONDS))["slow"], ["test_d"])  # fmt: skip
        td = {"top25": ["test_a", "test_e"], "top25_cfg": cut}
        choice = pick("td25", None, td)
        self.assertTrue(choice.covers("test_a", "slow"))
        self.assertTrue(choice.covers("test_e", "slow"))
        self.assertFalse(choice.covers("test_c", "slow"))
        self.assertTrue(choice.covers("test_e", "not_in_test_times"))
        self.assertEqual(choice.files(), {"test_a", "test_e"})
        # TD runs only its cut, even of the files a selection always covers.
        ind = pick("td25", None, {"top25": [], "top25_cfg": {"inductor": ["test_a"]}})
        self.assertEqual([ind.covers("test_a", "inductor"), ind.covers("test_torch", "inductor")], [True, False])  # fmt: skip

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


@contextlib.contextmanager
def scratch_repo() -> Iterator[None]:
    """A new git repo as bd.REPO, isolated from the caller's git environment and config."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull, GIT_AUTHOR_NAME="a", GIT_AUTHOR_EMAIL="a@a", GIT_COMMITTER_NAME="a", GIT_COMMITTER_EMAIL="a@a")  # fmt: skip
    with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, env, clear=True), mock.patch.object(bd, "REPO", Path(d)):  # fmt: skip
        bd.git("init", "-q")
        yield


class TestClickhouse(unittest.TestCase):
    def test_query(self) -> None:
        ok = subprocess.CompletedProcess([], 0, 'log\n{"columns": [{"name": "a"}], "rows": [[1]]}', "")  # fmt: skip
        busy = subprocess.CompletedProcess([], 1, "", "HTTP 503 Service Unavailable")
        bad = subprocess.CompletedProcess([], 2, "", "Code: 62. Syntax error")
        hung = subprocess.TimeoutExpired("gcx", clickhouse.TIMEOUT)
        # fmt: off
        cases = [
            ([ok], [{"a": 1}]),
            ([busy, hung, ok], [{"a": 1}]),
            ([bad, ok], "attempt 1 of 3 exited with code 2: Code: 62. Syntax error\nSQL: SELECT 1"),
            ([busy, busy, hung], "attempt 3 of 3 timed out after 600s\nSQL: SELECT 1"),
        ]
        # fmt: on
        for outcomes, want in cases:
            with mock.patch.object(subprocess, "run", side_effect=outcomes) as run, mock.patch.object(clickhouse.time, "sleep"):  # fmt: skip
                if isinstance(want, list):
                    self.assertEqual(clickhouse.query("SELECT 1"), want)
                else:
                    with self.assertRaises(RuntimeError) as e:
                        clickhouse.query("SELECT 1")
                    self.assertEqual(str(e.exception), f"ClickHouse query failed, {want}")  # fmt: skip
                self.assertEqual(run.call_count, min(len(outcomes), 1 if outcomes[0] is bad else 3))  # fmt: skip
                self.assertEqual(run.call_args.kwargs["timeout"], clickhouse.TIMEOUT)


class TestBuildDataset(unittest.TestCase):
    def test_strip_trailers(self) -> None:
        msg = "Body line\n\nCo-authored-by: x\nPull Request resolved: https://x\nApproved by: https://y\nghstack-source-id: 1\nKeep: me\n"  # @lint-ignore
        self.assertEqual(strip_trailers(msg), "Body line\n\nKeep: me")

    def test_history_load(self) -> None:
        rows = [
            fail("test_x", "2026-10-05 00:00:00", "a"),
            fail("test_x", "2026-10-05 00:00:00", "m", br="main"),
            fail("test_x", "2026-10-05 00:00:00", "t", br="trunk/t"),
            # A fork's main is a PR branch, here of no known PR.
            fail("test_x", "2026-10-05 00:00:00", "g", br="f/pytorch:main"),
            fail("test_x", "2026-10-05 00:00:00", "u"),
        ]
        landed = {1: day(7)}
        with (
            mock.patch.object(bd, "pull_failures", return_value=rows) as pull,
            mock.patch.object(bd, "pr_of_shas", return_value={"a": 1, "g": 2}) as pr_of_shas,
            mock.patch.object(bd, "pr_authors", return_value={1: "me"}),
            mock.patch.object(bd, "landed_times", return_value=landed) as landed_times,
            mock.patch.object(bd, "query", return_value=[]) as query,
        ):  # fmt: skip
            h = bd.History.load(START, END)
        pull.assert_called_once_with(dt.date(2026, 9, 1), dt.date(2026, 11, 9))
        landed_times.assert_called_once_with(dt.date(2026, 9, 1))
        pr_of_shas.assert_called_once_with(["a", "g", "u"])
        self.assertIn("day >= '2026-09-01' AND day < '2026-11-09'", query.call_args.args[0])  # fmt: skip
        self.assertEqual({sha: [r["pr"] for r in rs] for sha, rs in h.heads.items()}, {"a": [1], "g": [2]})  # fmt: skip
        k = ("test_a", "TestA", "test_x")
        self.assertEqual(len(h.main[k]), 2)
        self.assertEqual(sorted(who for _, who, _, _ in h.seen[k]), ["2", "main", "main", "main", "me"])  # fmt: skip
        self.assertEqual(h.landed, landed)

    def test_why_dropped(self) -> None:
        t = "2026-10-05 00:00:00"
        mine = ["test_real", "test_main", "test_main_after", "test_other_in", "test_other_out", "test_unmapped", "test_fork_main",
                "test_same_author", "test_own_head", "test_disabled", "test_disabled_out", "test_base_in", "test_base_out"]  # fmt: skip
        rows = [fail(name, t, "s") for name in mine] + [
            fail("test_green", t, "s", job="pull / test (default, 2, 2)"),
            fail("test_unstable", t, "s", job="pull / test (default, 1, 1, unstable)"),
            fail("test_holdout", t, "s", wf="td-holdout-unstable"),
            fail("test_periodic", t, "s", wf="periodic"),
            fail("test_rx", t, "s", wf="rocm-rx7900"),
            fail("test_late", "2026-10-26 00:00:00", "s"),
            fail("test_early", "2026-09-16 23:59:59", "s"),
            fail("test_early_in", "2026-09-17 00:00:00", "s"),
            fail("test_after_landing", "2026-10-08 00:00:00", "s"),
            fail("test_main", "2026-10-07 00:00:00", "m", br="main"),
            # PR 1 lands 10-07 12:00; its own breakage on main afterwards is not held against it.
            fail("test_main_after", "2026-10-08 00:00:00", "m", br="main"),
            # WINDOW is 14 days, exclusive.
            fail("test_other_in", "2026-09-21 00:00:01", "o"),
            fail("test_other_out", "2026-09-21 00:00:00", "o"),
            fail("test_unmapped", "2026-10-06 00:00:00", "u"),
            fail("test_fork_main", "2026-10-06 00:00:00", "g", br="f/pytorch:main"),
            # A same-author sibling is only checked as a PR below this one (see test_stacked).
            fail("test_same_author", "2026-10-06 00:00:00", "c"),
            fail("test_own_head", "2026-10-06 00:00:00", "s2"),
            fail("test_base_in", "2026-09-05 00:00:00", "m", br="main"),
            fail("test_base_out", "2026-09-01 00:00:00", "m", br="main"),
        ]  # fmt: skip
        disabled = (
            ("2026-09-22", "test_disabled"),
            ("2026-10-19", "test_disabled_out"),
        )
        h = history(rows, {"s": 1, "s2": 1, "o": 2, "c": 3}, {1: "me", 2: "x", 3: "me"}, {1: dt.datetime(2026, 10, 7, 12, tzinfo=dt.timezone.utc)}, disabled)  # fmt: skip
        with (
            patch_ci_jobs(TD_JOBS),
            mock.patch.object(bd, "green_jobs", return_value={("s", "pull", "pull / test (default, 2, 2)")}) as green,
        ):  # fmt: skip
            h.load_green({"s", "o"})
            h.load_green({"s"})
            got = {r["test"]: h.why_dropped(r) for r in h.heads["s"]}
            # The merge base with main at 09-15: main broke test_base_in 10 days before it.
            near_base = {r["test"]: h.why_dropped(r, base_t=day(15, 9)) for r in h.heads["s"] if r["test"].startswith("test_base")}  # fmt: skip
            # Main breaks test_main_after near this merge base, but only after PR 1 landed.
            after = h.why_dropped(next(r for r in h.heads["s"] if r["test"] == "test_main_after"), base_t=day(4))  # fmt: skip
        # Each head is queried once.
        self.assertEqual([c.args[0] for c in green.call_args_list], [["o", "s"], []])
        self.assertEqual(got, {
            "test_real": None, "test_main": "other_failure", "test_main_after": None, "test_other_in": "other_failure", "test_other_out": None,
            "test_unmapped": "other_failure", "test_fork_main": "other_failure", "test_same_author": None, "test_own_head": None,
            "test_disabled": "disabled", "test_disabled_out": None, "test_base_in": None, "test_base_out": None, "test_green": "job_green",
            "test_unstable": "job", "test_holdout": "job", "test_periodic": "job", "test_rx": "job", "test_late": "window",
            "test_early": "window", "test_early_in": None, "test_after_landing": "after_landing",
        })  # fmt: skip
        self.assertEqual(
            near_base, {"test_base_in": "main_near_base", "test_base_out": None}
        )
        self.assertIsNone(after)

    def test_stacked(self) -> None:
        # A ghstack stack, built as ghstack builds it: L (pr 1) below U (pr 2), both pushed twice. Each
        # PR's head has its previous head and a synthetic gh/USER/N/base commit with the tree of
        # the PR below as parents, so L's heads are neither U's CI base nor its ancestors.
        # N (pr 4) is a GitHub-native PR on top of L's second head, W (pr 3) another PR of the author's.
        with scratch_repo():

            def piped(args: list[str], stdin: str) -> str:
                return subprocess.run(["git", *args], cwd=bd.REPO, input=stdin, capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip

            def commit(content: str, *parents: str) -> str:
                """A commit of a tree holding `content`, so commits of equal content have equal trees."""
                blob = piped(["hash-object", "-w", "--stdin"], content)
                tree = piped(["mktree"], f"100644 blob {blob}\tf\n")
                return bd.git("commit-tree", tree, *(a for p in parents for a in ("-p", p)), "-m", content).strip()  # fmt: skip

            m = commit("main")
            l_base = commit("main", m)
            l1 = commit("L1", l_base)
            l2 = commit("L2", l1, l_base)
            u_base1 = commit("L1", m)
            u1 = commit("U1", u_base1)
            u_base2 = commit("L2", u_base1)
            u2 = commit("U2", u1, u_base2)
            n = commit("N", l2)
            w = commit("W", m)
            # L broke test_x in both versions and test_q only in its second; main broke test_v long ago on m.
            rows = [
                fail("test_x", "2026-10-05 00:00:00", l1),
                fail("test_x", "2026-10-06 00:00:00", l2),
                fail("test_q", "2026-10-06 00:00:00", l2),
                fail("test_z", "2026-10-05 00:00:00", w),
                fail("test_v", "2026-09-20 00:00:00", m, br="main"),
            ] + [fail(t, "2026-10-07 00:00:00", u) for u in (u1, u2, n) for t in ("test_x", "test_q", "test_z", "test_v", "test_y")]  # fmt: skip
            h = history(rows, {l1: 1, l2: 1, u1: 2, u2: 2, w: 3, n: 4}, {1: "me", 2: "me", 3: "me", 4: "me"})  # fmt: skip

            def truth(sha: str, base: str) -> tuple[list[str], dict]:
                dropped: collections.Counter = collections.Counter()
                kept = h.truth(sha, {"base": base, "merge_base": m, "main_base": m, "base_source": "ci"}, dropped)  # fmt: skip
                return sorted(r["test"] for r in kept or []), dict(dropped)

            with (
                patch_ci_jobs(TD_JOBS),
                mock.patch.object(bd, "green_jobs", return_value=set()),
                mock.patch.object(bd, "commit_time", return_value=day(20)),
            ):  # fmt: skip
                h.load_green({u1, u2, n, l2})
                # Each U head drops what the L version below it broke; its own failures, W's and main's stay.
                self.assertEqual(truth(u1, u_base1), (["test_q", "test_v", "test_y", "test_z"], {"stacked": 1}))  # fmt: skip
                self.assertEqual(truth(u2, u_base2), (["test_v", "test_y", "test_z"], {"stacked": 2}))  # fmt: skip
                # Diffed against L's head or against main, the native PR finds L below it by ancestry.
                self.assertEqual(truth(n, l2), (["test_v", "test_y", "test_z"], {"stacked": 2}))  # fmt: skip
                self.assertEqual(truth(n, m), (["test_v", "test_y", "test_z"], {"stacked": 2}))  # fmt: skip
                self.assertEqual(truth(l2, l_base), (["test_q", "test_x"], {}))
            self.assertEqual(h.other_heads({u1}), {l1, l2, w, n})

    def test_write_sample(self) -> None:
        with scratch_repo():
            bd.git("commit", "-q", "--allow-empty", "-m", "base")
            for f in ("gone.py", "moved.py"):
                (bd.REPO / f).write_text(f)
            bd.git("add", ".")
            bd.git("commit", "-q", "-m", "files")
            base = bd.git("rev-parse", "HEAD").strip()
            bd.git("rm", "-q", "gone.py")
            bd.git("mv", "moved.py", "new.py")
            bd.git("commit", "-q", "-m", "change")
            head = bd.git("rev-parse", "HEAD").strip()
            # select_tests' own git runs in this checkout, so only bd's is exercised.
            with (
                tempfile.TemporaryDirectory() as d,
                mock.patch.object(bd, "candidates_at", return_value=[]),
                mock.patch.object(bd, "changed_files", return_value=[]),
                mock.patch.object(bd, "diffstat_and_diff", return_value=("", "")),
            ):  # fmt: skip
                bd.write_sample(Path(d), "s", "neutral", head, base, "t", "b", truth=[])
                s = json.loads((Path(d) / "s.json").read_text())
        self.assertEqual(s["deleted"], ["gone.py", "moved.py"])

    def test_is_ancestor(self) -> None:
        with scratch_repo():
            for msg in ("a", "b"):
                bd.git("commit", "-q", "--allow-empty", "-m", msg)
            a, b = bd.git("rev-list", "HEAD").split()[::-1]
            self.assertTrue(bd.is_ancestor(a, b))
            self.assertFalse(bd.is_ancestor(b, a))
            self.assertFalse(bd.is_ancestor("0" * 40, b))

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
            td, other = ("trunk", "linux / test (default, 1, 1)"), ("periodic", "linux / test (default, 1, 1)")  # fmt: skip
            failed = [{"tf": "test_a", "wf": wf, "job": j} for wf, j in (td, other)] + [{"tf": "test_b", "wf": other[0], "job": other[1]}]  # fmt: skip
            with (
                patch_ci_jobs({True: {(*td, "default")}, False: {(*other, "default")}}),
                mock.patch.object(bd, "git", lambda *a, check=True: run(*a, cwd=d)),
                mock.patch.object(bd, "query", lambda sql: queries.append(sql) or failed),
            ):  # fmt: skip
                got = bd.revert_truth_jobs({"test_a", "test_b"}, commits["change"], landed, "2026-10-05 04:00:00")  # fmt: skip
                # Decided before the change landed: no commit qualifies, so only the change itself.
                bd.revert_truth_jobs({"test_a"}, commits["change"], landed, "2026-10-05 00:30:00")  # fmt: skip
                failed = failed[1:]
                # Failing only in jobs that do not run TD, so revert_samples skips it.
                none = bd.revert_truth_jobs({"test_a", "test_b"}, commits["change"], landed, "2026-10-05 04:00:00")  # fmt: skip
        self.assertEqual(got, {"test_a": [td]})
        self.assertEqual(none, {})
        # The change and the commit after it, not the revert or what landed after the decision.
        self.assertIn(f"head_sha IN ('{commits['after']}','{commits['change']}')", queries[0])  # fmt: skip
        self.assertIn(f"head_sha IN ('{commits['change']}')", queries[1])

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
        h = history([fail("test_x", "2026-10-05 00:00:00", "s")], {"s": 1}, {1: "me"})
        error = subprocess.CalledProcessError(1, "gh")
        with (
            tempfile.TemporaryDirectory() as out,
            patch_ci_jobs(TD_JOBS),
            mock.patch.object(bd, "green_jobs", return_value=set()),
            mock.patch.object(bd, "fetch"),
            mock.patch.object(bd, "has_commit", return_value=True),
            mock.patch.object(bd, "git", return_value="base\n"),
            mock.patch.object(bd, "commit_time", return_value=day(4)),
            mock.patch.object(bd, "ci_bases", return_value={}),
            mock.patch.object(bd, "pr_text", side_effect=error) as pr_text,
            mock.patch.object(bd, "write_sample") as write,
            contextlib.redirect_stdout(io.StringIO()),
        ):  # fmt: skip
            bd.pr_samples(h, Path(out), 1)
        pr_text.assert_called_once_with(1)
        write.assert_not_called()

    def test_unexpected_error_skips_only_its_sample(self) -> None:
        h = history([fail("test_x", "2026-10-05 00:00:00", "s"), fail("test_x", "2026-10-05 00:00:00", "t", tf="test_b")], {"s": 1, "t": 2}, {1: "me", 2: "you"})  # fmt: skip
        bad = UnicodeDecodeError("utf-8", b"", 0, 1, "bad")
        with (
            tempfile.TemporaryDirectory() as out,
            patch_ci_jobs(TD_JOBS),
            mock.patch.object(bd, "green_jobs", return_value=set()),
            mock.patch.object(bd, "fetch"),
            mock.patch.object(bd, "has_commit", return_value=True),
            mock.patch.object(bd, "git", return_value="base\n"),
            mock.patch.object(bd, "commit_time", return_value=day(4)),
            mock.patch.object(bd, "ci_bases", return_value={}),
            mock.patch.object(bd, "pr_text", side_effect=[bad, ("t", "b")]),
            mock.patch.object(bd, "write_sample") as write,
            contextlib.redirect_stdout(io.StringIO()) as stdout,
        ):  # fmt: skip
            bd.pr_samples(h, Path(out), 1)
            # With no pr_ sample written, the head pass reports and returns, so later kinds still run.
            bd.head_samples(h, Path(out), 1)
        self.assertEqual([c.args[1] for c in write.call_args_list], ["pr_2"])
        self.assertIn("1 of 2 PR samples raised and were skipped", stdout.getvalue())
        self.assertIn("pr truth: dropped by rule over 2 heads", stdout.getvalue())
        self.assertIn("no head samples", stdout.getvalue())

    def test_pr_samples(self) -> None:
        t = "2026-10-05 00:00:00"
        rows = [
            fail("test_x", t, "s", tf="test_a"),
            fail("test_x", t, "s", tf="test_b"),
            fail("test_x", t, "s", tf="test_b", job="win / test (default, 1, 1)"),
            fail("test_x", t, "s", tf="test_c"),
            # Main broke test_a 10 days before the merge base with main, and test_c's test_y, another test of the file.
            fail("test_x", "2026-09-05 00:00:00", "m", br="main", tf="test_a"),
            fail("test_y", "2026-09-05 00:00:00", "m", br="main", tf="test_c"),
            # The earliest failing head before --start is not the sample's.
            fail("test_x", "2026-09-30 00:00:00", "s0", tf="test_a"),
        ]
        h = history(rows, {"s": 1, "s0": 1}, {1: "me"})
        merge_bases = {("origin/main", "s"): "main_mb", ("lower", "s"): "lower"}
        with (
            tempfile.TemporaryDirectory() as out,
            patch_ci_jobs({True: {("pull", "pull", "default"), ("pull", "win", "default")}, False: set()}),
            mock.patch.object(bd, "green_jobs", return_value=set()),
            mock.patch.object(bd, "fetch"),
            mock.patch.object(bd, "has_commit", return_value=True),
            mock.patch.object(bd, "git", lambda *a, check=True: merge_bases[a[1:]] + "\n"),
            mock.patch.object(bd, "commit_time", return_value=day(15, 9)) as commit_time,
            mock.patch.object(bd, "ci_bases", return_value={"s": ["lower"]}) as ci_bases,
            mock.patch.object(bd, "pr_text", return_value=("t", "b")),
            mock.patch.object(bd, "write_sample") as write,
            contextlib.redirect_stdout(io.StringIO()) as stdout,
        ):  # fmt: skip
            bd.pr_samples(h, Path(out), 1)
        ci_bases.assert_called_once_with({"s": 1})
        # Main breakage is checked around the merge base with main, the diff is against the PR's CI base.
        commit_time.assert_called_once_with("main_mb")
        kw = write.call_args.kwargs
        win = ("pull", "win / test (default, 1, 1)")
        self.assertEqual((kw["truth"], kw["truth_strong"], kw["truth_jobs"]), (["test_b", "test_c"], ["test_b"], {"test_b": [("pull", JOB), win], "test_c": [("pull", JOB)]}))  # fmt: skip
        self.assertEqual({k: kw[k] for k in ("base", "merge_base", "main_base", "base_source")}, {"base": "lower", "merge_base": "lower", "main_base": "main_mb", "base_source": "ci"})  # fmt: skip
        self.assertIn("pr truth: dropped by rule over 1 heads: {'main_near_base': 1}", stdout.getvalue())  # fmt: skip

    def test_pr_and_head_truth_agree(self) -> None:
        green_job = "pull / test (default, 2, 2)"
        rows = [
            fail("test_real", "2026-10-05 00:00:00", "h1"),
            fail("test_green", "2026-10-05 00:00:00", "h1", job=green_job),
            fail("test_disabled", "2026-10-05 00:00:00", "h1"),
            fail("test_near_base", "2026-10-05 00:00:00", "h1"),
            fail("test_near_base", "2026-09-12 00:00:00", "m", br="main"),
            # 10 days before: once only head truth's window, now both kinds'.
            fail("test_other", "2026-10-05 00:00:00", "h1"),
            fail("test_other", "2026-09-25 00:00:00", "o"),
            fail("test_later", "2026-10-08 00:00:00", "h2"),
        ]
        h = history(rows, {"h1": 1, "h2": 1, "o": 2}, {1: "me", 2: "x"}, disabled=(("2026-10-01", "test_disabled"),))  # fmt: skip
        bases = {"base": "b", "merge_base": "b", "main_base": "m", "base_source": "ci"}
        with (
            tempfile.TemporaryDirectory() as d,
            patch_ci_jobs(TD_JOBS),
            mock.patch.object(bd, "green_jobs", return_value={("h1", "pull", green_job)}),
            mock.patch.object(bd, "fetch"),
            mock.patch.object(bd, "has_commit", return_value=True),
            mock.patch.object(bd, "ci_bases", return_value={}),
            mock.patch.object(bd, "pr_bases", return_value=bases),
            mock.patch.object(bd, "commit_time", return_value=day(20, 9)),
            mock.patch.object(bd, "pr_text", return_value=("t", "b")),
            mock.patch.object(bd, "diffstat_and_diff", return_value=("", "")),
            mock.patch.object(bd, "changed_files", return_value=[]),
            mock.patch.object(bd, "deleted_files", return_value=[]),
            mock.patch.object(bd, "candidates_at", return_value=[]),
            mock.patch.object(bd, "head_created", return_value={}),
            contextlib.redirect_stdout(io.StringIO()) as stdout,
        ):  # fmt: skip
            out = Path(d)
            bd.pr_samples(h, out, 1)
            bd.head_samples(h, out, 1)
            samples = {p.stem: json.loads(p.read_text()) for p in out.glob("*.json")}
        self.assertEqual(sorted(samples), ["head_1_h1", "head_1_h2", "pr_1"])
        pr, head = samples["pr_1"], samples["head_1_h1"]
        self.assertEqual((pr["sha"], pr["truth"]), ("h1", ["test_a"]))
        self.assertEqual((pr["truth"], pr["truth_jobs"]), (head["truth"], head["truth_jobs"]))  # fmt: skip
        dropped = (
            "{'disabled': 1, 'job_green': 1, 'main_near_base': 1, 'other_failure': 1}"
        )
        self.assertIn(f"pr truth: dropped by rule over 1 heads: {dropped}", stdout.getvalue())  # fmt: skip
        self.assertIn(f"head truth: dropped by rule over 2 heads: {dropped}", stdout.getvalue())  # fmt: skip

    def test_pr_bases(self) -> None:
        # ghstack's base at the push is in the head's history; a later base and a fork's base merge-base older.
        merge_bases = {("origin/main", "h"): "m", ("b_push", "h"): "b_push", ("b_later", "h"): "b_old", ("b_gone", "h"): ""}  # fmt: skip

        def git(*a: str, check: bool = True) -> str:
            if a[:2] == ("merge-base", "--independent"):
                self.assertEqual(set(a[2:]), {"b_push", "b_old"})
                return "b_push\n"
            return merge_bases[(a[1], a[2])] + "\n"

        with mock.patch.object(bd, "git", git):
            got = bd.pr_bases("h", ["b_gone", "b_later", "b_push"])
            self.assertEqual(got, {"base": "b_push", "merge_base": "b_push", "main_base": "m", "base_source": "ci"})  # fmt: skip
            self.assertEqual(bd.pr_bases("h", ["b_later"])["merge_base"], "b_old")  # type: ignore[index]
            # b_child is a later ghstack base on top of b_push, so it merge-bases at b_push too; b_push wins in either order.
            merge_bases[("b_child", "h")] = "b_push"
            for order in (["b_child", "b_push"], ["b_push", "b_child"]):
                self.assertEqual(bd.pr_bases("h", order)["base"], "b_push")  # type: ignore[index]
            main = {"base": "m", "merge_base": "m", "main_base": "m", "base_source": "main"}  # fmt: skip
            self.assertEqual(bd.pr_bases("h", ["b_gone"]), main)
            self.assertEqual(bd.pr_bases("h", []), main)
            merge_bases[("origin/main", "h")] = ""
            self.assertIsNone(bd.pr_bases("h", ["b_push"]))

    def test_head_samples(self) -> None:
        job_a, job_b = "linux / test (default, 1, 2)", "linux / test (default, 2, 2)"
        t = "2026-10-05 00:00:00"
        # h1 is the pr_ sample's head: job_a failed, then passed on a rerun, so only job_b's failure is truth.
        rows = [
            fail("test_x", t, "h1", job=job_a, tf="test_a"),
            fail("test_x", t, "h1", job=job_b, tf="test_c"),
            fail("test_x", t, "h2", job=job_a, tf="test_b"),
            fail("test_x", t, "h3", job=job_a, tf="test_d"),
            fail("test_x", t, "h4", job=job_a, tf="test_e"),
            # Every failure of h5 is in a job that passed, and h6 is another PR's.
            fail("test_x", t, "h5", job=job_a, tf="test_f"),
            fail("test_x", t, "h6", job=job_a, tf="test_g"),
        ]
        h = history(rows, {"h1": 1, "h2": 1, "h3": 1, "h4": 1, "h5": 1, "h6": 2}, {1: "me", 2: "x"})  # fmt: skip
        # h1 and h2 have a CI base each, h3 has no merge base with main, h4 has no CI base.
        merge_bases = {("origin/main", "h1"): "m1", ("b1", "h1"): "b1", ("origin/main", "h2"): "m2", ("b2", "h2"): "mb2", ("origin/main", "h3"): "", ("origin/main", "h4"): "m4"}  # fmt: skip
        with (
            tempfile.TemporaryDirectory() as out,
            patch_ci_jobs({True: {("pull", "linux", "default")}, False: set()}),
            mock.patch.object(bd, "green_jobs", return_value={("h1", "pull", job_a), ("h5", "pull", job_a)}),
            mock.patch.object(bd, "fetch"),
            mock.patch.object(bd, "has_commit", return_value=True),
            mock.patch.object(bd, "commit_time", return_value=day(4)),
            mock.patch.object(bd, "ci_bases", return_value={"h1": ["b1"], "h2": ["b2"], "h3": ["b3"]}) as ci_bases,
            # Every failure is at 10-05: h1's CI started before them, h2's after, h4's is unknown.
            mock.patch.object(bd, "head_created", return_value={"h1": "2026-10-04 00:00:00", "h2": "2026-10-06 00:00:00"}),
            mock.patch.object(bd, "git", lambda *a, check=True: merge_bases[a[1:]] + "\n"),
            mock.patch.object(bd, "write_sample") as write,
            contextlib.redirect_stdout(io.StringIO()),
        ):  # fmt: skip
            (Path(out) / "pr_1.json").write_text(json.dumps({"pr": 1, "sha": "h1", "base": "pr_base", "title": "t", "body": "b"}))  # fmt: skip
            bd.head_samples(h, Path(out), 1)
        ci_bases.assert_called_once_with({"h1": 1, "h2": 1, "h3": 1, "h4": 1})
        got = {c.args[1]: (c.args[2:], c.kwargs) for c in write.call_args_list}
        self.assertEqual(sorted(got), ["head_1_h1", "head_1_h2", "head_1_h4"])
        # h2's previous failures are the other heads' (flaky or not, also of h3 and h5, which have no sample).
        previous = {"head_1_h1": [], "head_1_h2": ["test_a", "test_c", "test_d", "test_e", "test_f"], "head_1_h4": []}  # fmt: skip
        want = {
            "head_1_h1": ("h1", "b1", "b1", "m1", "ci", {"test_c": [("pull", job_b)]}),
            "head_1_h2": ("h2", "b2", "mb2", "m2", "ci", {"test_b": [("pull", job_a)]}),
            "head_1_h4": (
                "h4",
                "m4",
                "m4",
                "m4",
                "main",
                {"test_e": [("pull", job_a)]},
            ),
        }
        for sid, (sha, base, merge_base, main_base, source, truth_jobs) in want.items():
            args, kw = got[sid]
            self.assertEqual(args, ("head", sha))
            bases = {"base": base, "merge_base": merge_base, "main_base": main_base, "base_source": source}  # fmt: skip
            self.assertEqual(kw, {"title": "t", "body": "b", **bases, "pr": 1, "truth": sorted(truth_jobs), "truth_jobs": truth_jobs, "previous_failures": previous[sid]})  # fmt: skip

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
            if (a[0], a[1].split(":")[0] if a[0] == "show" else a[3]) in (
                ("show", "lower"),
                ("ls-tree", "lower"),
            ):
                raise AssertionError(f"TESTS not read at main: {a}")
            if a[0] == "diff" and a[-2:] != (diff_base, "s"):
                raise AssertionError(f"not diffed against {diff_base}: {a}")
            return {"show": src, "ls-tree": ls, "diff": added}[a[0]]

        for src, want, new in ((named, ["extra", "fx/test_c", "test_a"], ["fx/test_e", "test_new"]), (inline, ["ao/test_b", "test_a"], ["ao/test_d", "test_new"])):  # fmt: skip
            with mock.patch.object(bd, "git", git):
                diff_base = "b"
                self.assertEqual(bd.candidates_at("b"), want)
                self.assertEqual(bd.candidates_at("b", "s"), sorted(want + new))
                diff_base = "lower"
                self.assertEqual(
                    bd.candidates_at("b", "s", "lower"), sorted(want + new)
                )

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
            (Path(repo) / "test" / "test_x.py").write_text("")
            git("add", "-A")
            git("commit", "-qm", "add")
            with (
                mock.patch.object(select_tests, "git", git),
                mock.patch.object(bd, "candidates_at", return_value=[]) as candidates,
            ):
                bd.write_sample(Path(repo), "s", "pr", "HEAD~1", "HEAD~2", "t", "b")
                bd.write_sample(Path(repo), "s2", "pr", "HEAD", "lower", "t", "b", merge_base="HEAD~1", main_base="HEAD~2")  # fmt: skip
            s, s2 = (
                json.loads((Path(repo) / f"{n}.json").read_text()) for n in ("s", "s2")
            )
        self.assertEqual(s["changed_files"], ["test/my file.py", "test/run_test.py"])
        self.assertIn("test/run_test.py", s["diffstat"])
        self.assertIn("rename from test/run_test.py", s["diff"])
        self.assertEqual((s["base"], s["merge_base"], s["main_base"]), ("HEAD~2", "HEAD~2", "HEAD~2"))  # fmt: skip
        self.assertEqual((s2["base"], s2["merge_base"], s2["main_base"], s2["changed_files"]), ("lower", "HEAD~1", "HEAD~2", ["test/test_x.py"]))  # fmt: skip
        self.assertEqual(candidates.call_args_list, [mock.call("HEAD~2", "HEAD~1", "HEAD~2"), mock.call("HEAD~2", "HEAD", "HEAD~1")])  # fmt: skip


if __name__ == "__main__":
    unittest.main()
