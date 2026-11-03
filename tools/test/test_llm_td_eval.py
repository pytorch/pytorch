from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import random
import re
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
from tools.testing.llm_td import select_tests
from tools.testing.llm_td.eval import score
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


def write_data(d: str, files: Mapping[str, object]) -> Path:
    """Write `files` under `d`, stamping each TD replay with its sample's current inputs."""
    data = Path(d)
    for name, obj in files.items():
        (data / name).parent.mkdir(parents=True, exist_ok=True)
        (data / name).write_text(json.dumps(obj))
    times = (data / "test-times.json").read_bytes()
    for p in (data / "td").glob("*.json"):
        if (sample := data / "samples" / p.name).exists():
            inputs = score.replay_inputs("", times, sample.read_bytes())
            p.write_text(json.dumps({**json.loads(p.read_text()), "inputs": inputs}))
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
            self.assertTrue(pick(name, r, td, huge).sel.run_all, name)
            self.assertTrue(pick(name, r, td, infra).sel.run_all, name)
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
            data = write_data(d, files)
            rows = score.score(data, ["a"], ["m"], ["model"], True, data / "samples", data / "td", "")  # fmt: skip
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
        # hedge is Selection.split for any share; at LLM_TD_HEDGE they must agree.
        rng = random.Random(0)
        files = [f"test_{i}" for i in range(8)]
        for _ in range(500):
            scored = rng.sample(files, rng.randint(0, 8))
            seconds = {f: float(rng.choice([0, 1, 5, 20, 60])) for f in files if rng.random() < 0.7}  # fmt: skip
            untimed = [f for f in files if f not in seconds and rng.random() < 0.5]
            job = [*seconds, *untimed]
            sel = Selection(set(rng.sample(files, rng.randint(0, 2))), candidates=set(files))  # fmt: skip
            ranked = [TestRun(f) for f in scored if f in job] + [TestRun(f) for f in job if f not in scored]  # fmt: skip
            split = sel.split(ranked, lambda x: float(x.test_file in scored), seconds, "c")  # fmt: skip
            got = score.hedge(sel, scored, seconds, score.LLM_TD_HEDGE, "c", {"TD"}, untimed)  # fmt: skip
            self.assertEqual(got, {"TD"} if split.kept_td else split.hedged, (scored, seconds, untimed, sel))  # fmt: skip

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
            (data / "samples/b.json").write_text(json.dumps({**SAMPLE, "title": "rebuilt"}))  # fmt: skip
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

    def test_replay_stats(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            stats, code = Path(d) / "stats", Path(d) / "td"
            for p in [*(stats / a for a in score.STATS_ARTIFACTS), code / "heuristics/h.py"]:  # fmt: skip
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("1")
            with mock.patch.object(score, "TD_CODE", code):
                digests = [score.replay_stats(stats)]
                (stats / score.STATS_ARTIFACTS[0]).write_text("2")
                digests.append(score.replay_stats(stats))
                (code / "heuristics/h.py").write_text("2")
                digests += [score.replay_stats(stats), score.replay_stats(stats)]
        self.assertEqual(len(set(digests)), 3)
        self.assertEqual(digests[2], digests[3])

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


if __name__ == "__main__":
    unittest.main()
