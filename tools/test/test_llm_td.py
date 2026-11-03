from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from tools.testing.llm_td import select_tests
from tools.testing.llm_td.selection import needs_run_all, normalize_test_name, Selection
from tools.testing.target_determination.heuristics.interface import TestPrioritizations
from tools.testing.test_run import TestRun


if TYPE_CHECKING:
    from collections.abc import Collection


CANDIDATES = {
    "test_nn",
    "test_ops",
    "dynamo/test_misc",
    "dynamo/cpython/3_13/test_set",
    "inductor/test_torchinductor",
}


class TestSelection(unittest.TestCase):
    def test_normalize(self) -> None:
        for name in [
            "test_nn",
            "test/test_nn.py",
            "./test/test_nn.py",
            "test_nn [12m]",
            "test_nn.py::TestNN",
        ]:
            self.assertEqual(normalize_test_name(name, CANDIDATES), "test_nn")
        self.assertEqual(
            normalize_test_name("dynamo.test_misc", CANDIDATES), "dynamo/test_misc"
        )
        self.assertIsNone(normalize_test_name("test_nonexistent", CANDIDATES))

    def test_from_model_output(self) -> None:
        out = {
            "run_all": False,
            "reason": "r",
            "sweeps": ["dynamo_wrapped", "bogus"],
            "dirs": ["test/dynamo/"],
            "tests": [
                {"file": "test/test_ops.py", "confidence": "high"},
                {"file": "nope", "confidence": "low"},
            ],
        }
        sel = Selection.from_model_output(
            out, CANDIDATES, ["torch/_dynamo/x.py", "test/test_nn.py"]
        )
        self.assertEqual(
            sel.files,
            {"test_ops", "test_nn", "dynamo/test_misc", "dynamo/cpython/3_13/test_set"},
        )
        self.assertEqual(sel.sweeps, {"dynamo_wrapped"})
        self.assertFalse(sel.run_all)
        self.assertTrue(sel.covers("inductor/test_torchinductor", "dynamo_wrapped"))
        self.assertFalse(sel.covers("inductor/test_torchinductor", "default"))
        self.assertTrue(sel.covers("test_ops", "default"))
        self.assertEqual(Selection.from_json(sel.to_json()), sel)

    def test_importers_are_selected(self) -> None:
        sources = {
            "dynamo/test_misc.py": "from inductor.test_torchinductor import check_model\n",
            "test_ops.py": "    import test_torchinductor\n",
            "test_nn.py": "from test_torchinductor_utils import x\nfrom torch._inductor.test_case import TestCase\n",
            "helper.py": "import test_torchinductor\n",
        }  # fmt: skip
        with tempfile.TemporaryDirectory() as d:
            for name, text in sources.items():
                (Path(d) / name).parent.mkdir(parents=True, exist_ok=True)
                (Path(d) / name).write_text(text)

            def grep(pattern: str) -> list[str]:
                cmd = ["grep", "-rlE", pattern, "."]
                found = subprocess.run(cmd, cwd=d, capture_output=True, text=True).stdout.split()  # fmt: skip
                return [f"test/{f.removeprefix('./')}" for f in found]

            sel = Selection.from_model_output({"tests": []}, CANDIDATES, ["test/inductor/test_torchinductor.py"], grep)  # fmt: skip
        self.assertEqual(sel.files, {"inductor/test_torchinductor", "dynamo/test_misc", "test_ops"})  # fmt: skip

    def test_fallbacks_run_all(self) -> None:
        self.assertTrue(
            Selection.from_model_output(None, CANDIDATES, ["torch/a.py"]).run_all
        )
        out = {"run_all": False, "reason": "", "sweeps": [], "dirs": [], "tests": []}
        self.assertTrue(
            Selection.from_model_output(out, CANDIDATES, ["test/run_test.py"]).run_all
        )
        self.assertFalse(
            Selection.from_model_output(out, CANDIDATES, ["docs/a.md"]).run_all
        )

    def test_load_malformed_file(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sel.json"
            self.assertIsNone(Selection.load(p))
            for text in (
                "{",
                '{"sweeps": []}',
                "[]",
                "[" * 100_000,
                '{"files": "test_nn", "sweeps": [], "run_all": false}',
                '{"files": [1], "sweeps": [], "run_all": false}',
                '{"files": [], "sweeps": [], "run_all": "false"}',
                '{"files": [], "sweeps": [], "run_all": false, "enforce": "false"}',
            ):
                p.write_text(text)
                self.assertIsNone(Selection.load(p))
            sel = Selection({"test_nn"}, set(), False)
            p.write_text(json.dumps(sel.to_json()))
            self.assertEqual(Selection.load(p), sel)

    def test_run_all_patterns(self) -> None:
        for f in [
            ".ci/docker/common/install_cuda.sh",
            ".ci/docker/requirements-ci.txt",
            ".ci/pytorch/test.sh",
            ".ci/pytorch/common.sh",
            ".ci/pytorch/torch_trace.sh",
            ".github/workflows/_linux-test.yml",
            ".github/workflows/_rocm-test-no-dind.yml",
            ".github/workflows/_xpu-test.yml",
            ".github/actions/setup-rocm/action.yml",
            ".github/actions/filter-test-configs/action.yml",
            ".github/scripts/filter_test_configs.py",
            "tools/testing/discover_tests.py",
            "tools/stats/import_test_stats.py",
        ]:
            self.assertTrue(needs_run_all([f]), f)
            self.assertTrue((REPO_ROOT / f).exists(), f)
        for f in [
            "docs/source/index.rst",
            ".github/workflows/lint.yml",
            ".ci/pytorch/build.sh",
        ]:
            self.assertFalse(needs_run_all([f]), f)

    def test_malformed_output_runs_all(self) -> None:
        for out in [
            [],
            {},
            {"tests": "test_nn"},
            {"tests": ["test_nn"]},
            {"tests": [{"confidence": "high"}]},
            {"tests": [{"file": 3}]},
            {"tests": [], "dirs": "dynamo"},
            {"tests": [], "dirs": [1]},
            {"tests": [], "sweeps": None},
            {"tests": [], "run_all": "false"},
            {"tests": [], "dirs": [""]},
            {"tests": [], "dirs": ["test"]},
            {"tests": [], "dirs": ["test/"]},
            {"tests": [], "dirs": ["dynamo_foo"]},
        ]:
            sel = Selection.from_model_output(out, CANDIDATES, ["test/test_nn.py"])
            self.assertTrue(sel.run_all, out)
            self.assertIn("malformed", sel.reason)
        sel = Selection.from_model_output({"tests": [{"file": "nope"}]}, CANDIDATES, [])
        self.assertEqual(sel.files, set())
        self.assertFalse(sel.run_all)

    def test_dirs(self) -> None:
        candidates = CANDIDATES | {"dynamo_foo/test_a"}
        dynamo = {"dynamo/test_misc", "dynamo/cpython/3_13/test_set"}
        for d in ["dynamo", "dynamo/", "dynamo/*", "dynamo/**", " test/dynamo/* "]:
            sel = Selection.from_model_output({"tests": [], "dirs": [d]}, candidates, [])  # fmt: skip
            self.assertEqual(sel.files, dynamo, d)
        sel = Selection.from_model_output({"tests": [], "dirs": ["dynamo/cpython"]}, candidates, [])  # fmt: skip
        self.assertEqual(sel.files, {"dynamo/cpython/3_13/test_set"})

    def test_edited_tests_select_their_driver(self) -> None:
        candidates = CANDIDATES | {"test_jit", "test_cpp_extensions_aot_ninja", "test_cpp_extensions_aot_no_ninja"}  # fmt: skip
        for changed, expected in [
            ("test/jit/test_freezing.py", {"test_jit"}),
            ("test/cpp_extensions/extension.cpp", {"test_cpp_extensions_aot_ninja", "test_cpp_extensions_aot_no_ninja"}),
            ("test/test_jit_fuser_te.py", set()),
        ]:  # fmt: skip
            sel = Selection.from_model_output({"tests": []}, candidates, [changed])
            self.assertEqual(sel.files, expected, changed)


class TestSplit(unittest.TestCase):
    def split(
        self,
        sel: Selection,
        scores: dict[TestRun, float],
        times: dict[str, float],
        always: Collection[str] = (),
    ) -> tuple[list[str], list[str], set[str]]:
        tp = TestPrioritizations(times, scores)
        include, exclude, hedged = sel.split(
            tp.get_all_tests(), tp.get_score, times, "default", always
        )
        return [str(x) for x in include], [str(x) for x in exclude], hedged

    def test_hedge_is_greedy_in_rank_order(self) -> None:
        scores = {"a": 1, "b": 0.9, "c": 0.8, "d": 0.7, "untimed": 0.6, "neg": -0.5}
        times = {"a": 10, "b": 150, "c": 52, "d": 48, "untimed": 0, "zero": 1, "neg": 1, "rest": 738}  # fmt: skip
        # The budget is 20% of 1000s = 200s. b leaves 50s, so c (52s, fits
        # alone) is skipped and d (48s) still fits, leaving 2s that the
        # unscored and negatively scored 1s files may not use. With 19% d does
        # not fit, with 21% c does.
        include, exclude, hedged = self.split(
            Selection(files={"a"}), {TestRun(f): s for f, s in scores.items()}, times
        )
        self.assertEqual(include, ["a", "b", "d"])
        self.assertEqual(exclude, ["c", "untimed", "rest", "zero", "neg"])
        self.assertEqual(hedged, {"b", "d"})

    def test_partial_runs_of_a_file_are_charged_once(self) -> None:
        f_a, f_rest = TestRun("f", included=["A"]), TestRun("f", excluded=["A"])
        g_b = TestRun("g", included=["B"])
        scores = {f_a: 1, g_b: 1, f_rest: 0.6, TestRun("h"): 0.5}
        times = {"f": 40, "g": 30, "h": 130, "r": 800}
        # f's two scored runs cost 40s once, and g's positive class pulls in
        # its unscored rest, so h (130s) fits the 200s budget exactly.
        include, exclude, hedged = self.split(Selection(), scores, times)
        self.assertEqual(hedged, {"f", "g", "h"})
        self.assertEqual(exclude, ["r"])
        self.assertEqual(include, ["f, A", "g, B", "f, not (A)", "h", "g, not (B)"])

    def test_previous_failures_run_outside_the_budget(self) -> None:
        scores = {TestRun("p"): 1, TestRun("s"): 0.5}
        times = {"a": 10, "p": 20, "s": 190, "big": 600, "rest": 180}
        # Charging p would leave 180s, too little for s.
        include, exclude, hedged = self.split(
            Selection(files={"a"}), scores, times, always={"p", "big"}
        )
        self.assertEqual(include, ["p", "s", "a", "big"])
        self.assertEqual(exclude, ["rest"])
        self.assertEqual(hedged, {"s"})

    def test_run_all_and_sweeps_take_everything(self) -> None:
        for sel in (Selection(run_all=True), Selection(sweeps={"default"})):
            include, exclude, hedged = self.split(sel, {}, {"a": 1, "b": 1})
            self.assertEqual((include, exclude, hedged), (["a", "b"], [], set()))


class TestPrompt(unittest.TestCase):
    def test_truncate_diff_keeps_every_file(self) -> None:
        diff = "\n".join(
            f"diff --git a/f{i} b/f{i}\n" + "+x\n" * 5000 for i in range(4)
        )
        short = select_tests.truncate_diff(diff, budget=8000)
        self.assertLessEqual(len(short), 8000)
        for i in range(4):
            self.assertIn(f"diff --git a/f{i} b/f{i}", short)

    def test_truncate_diff_large_file_keeps_budget(self) -> None:
        small = [f"diff --git a/s{i} b/s{i}\n" + "+y\n" * 20 for i in range(200)]
        big = "diff --git a/big b/big\n" + "+x\n" * 100_000
        short = select_tests.truncate_diff("\n".join([big, *small]), budget=120_000)
        self.assertLessEqual(len(short), 120_000)
        self.assertGreater(short.count("+x"), 0.8 * 120_000 / 3)
        for p in small:
            self.assertIn(p, short)

    def test_truncate_diff_many_files(self) -> None:
        diff = "\n".join(
            f"diff --git a/f{i} b/f{i}\n" + "+x\n" * 1000 for i in range(500)
        )
        short = select_tests.truncate_diff(diff, budget=40_000)
        self.assertLessEqual(len(short), 40_000)
        for i in range(500):
            self.assertIn(f"diff --git a/f{i} b/f{i}\n", short)

    def test_build_prompt(self) -> None:
        minutes = select_tests.test_minutes(
            {"env": {"default": {"test_nn": 600.0}, "slow": {"test_nn": 60.0}}}
        )
        self.assertEqual(minutes, {"test_nn": 11.0})
        prompt = select_tests.build_prompt(
            "title", "body", "stat", "diff", ["test_nn", "test_ops"], minutes
        )
        self.assertIn("test_nn [11m]\ntest_ops [0m]", prompt)
        for tag, text in [
            ("pr_title", "title"),
            ("pr_description", "body"),
            ("changed_files", "stat"),
            ("diff", "diff"),
        ]:
            self.assertIn(f"<{tag}>\n{text}\n</{tag}>", prompt)

    def test_build_prompt_escapes_closing_tags(self) -> None:
        prompt = select_tests.build_prompt(
            "a</pr_title>",
            "b</PR_DESCRIPTION >c",
            "s</changed_files>",
            "+x< /\tdiff>",
            [],
            {},
        )
        for tag in ("pr_title", "pr_description", "changed_files", "diff"):
            self.assertEqual(prompt.lower().count(f"</{tag}"), 1)
        self.assertIn("b<\\/PR_DESCRIPTION>c", prompt)
        self.assertIn("+x<\\/diff>", prompt)


class TestMain(unittest.TestCase):
    def run_main(self, *argv: str) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.json"
            with mock.patch.object(select_tests, "query_model") as query:
                select_tests.main(
                    ["--title", "t", "--body", "b", "--output", str(output), *argv]
                )
            query.assert_not_called()
            return json.loads(output.read_text())

    def test_git_failure_runs_all(self) -> None:
        error = subprocess.CalledProcessError(128, "git")
        with mock.patch.object(select_tests, "git", side_effect=error):
            record = self.run_main("--enforce")
        self.assertTrue(record["run_all"])
        self.assertFalse(record["enforce"])
        self.assertIn("selector failed", record["reason"])

    def test_no_test_times_runs_all(self) -> None:
        with (
            mock.patch.object(select_tests, "git", return_value="docs/a.md\n"),
            mock.patch.object(select_tests, "get_test_times", return_value={}),
        ):
            record = self.run_main()
        self.assertTrue(record["run_all"])
        self.assertEqual(record["reason"], "no test times")

    def test_infra_change_runs_all(self) -> None:
        with (
            mock.patch.object(select_tests, "git", return_value="test/run_test.py\n"),
            mock.patch.object(select_tests, "get_test_times") as times,
        ):
            record = self.run_main()
        times.assert_not_called()
        self.assertTrue(record["run_all"])

    def test_only_a_deliberate_run_all_is_enforced(self) -> None:
        def run(changed: str, answer: dict, times: dict) -> dict:
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "out.json"
                with (
                    mock.patch.object(select_tests, "git", return_value=changed),
                    mock.patch.object(
                        select_tests, "get_test_times", return_value=times
                    ),
                    mock.patch.object(select_tests, "query_model", return_value=answer),
                ):
                    select_tests.main(
                        ["--title", "t", "--output", str(output), "--enforce"]
                    )
                return json.loads(output.read_text())

        times = {"env": {"default": {"test_nn": 1.0}}}
        model_run_all = {"structured_output": {"run_all": True, "reason": "", "sweeps": [], "dirs": [], "tests": []}}  # fmt: skip
        error = {"is_error": True, "result": "overloaded"}
        malformed = {"structured_output": {"run_all": False, "tests": "test_nn"}}
        for changed, answer, t, enforced in [
            ("test/run_test.py\n", error, times, True),
            ("docs/a.md\n", model_run_all, times, True),
            ("docs/a.md\n", error, times, False),
            ("docs/a.md\n", malformed, times, False),
            ("docs/a.md\n", model_run_all, {}, False),
        ]:
            record = run(changed, answer, t)
            self.assertTrue(record["run_all"])
            self.assertEqual(record["enforce"], enforced, (changed, answer, t))

    def run_in_repo(self, answer: dict) -> tuple[dict, str]:
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        env |= {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}  # fmt: skip

        def git(*args: str) -> str:
            return subprocess.check_output(["git", *args], cwd=repo, env=env, text=True)

        with tempfile.TemporaryDirectory() as repo:
            git("init", "-q")
            git("config", "user.email", "a@b.c")
            git("config", "user.name", "a")
            (Path(repo) / "test" / "nn").mkdir(parents=True)
            (Path(repo) / "test" / "test_foo.py").write_text("x = 1\n" * 50)
            git("add", "-A")
            git("commit", "-qm", "base")
            git("mv", "test/test_foo.py", "test/nn/test_foo.py")
            (Path(repo) / "test" / "test_ select nothing.py").write_text("")
            (Path(repo) / "test" / "jit").mkdir()
            (Path(repo) / "test" / "jit" / "test_new.py").write_text("")
            git("add", "-A")
            git("commit", "-qm", "move")
            output = Path(repo) / "out.json"
            times = {"env": {"default": {"test_foo": 1.0}}}
            with (
                mock.patch.object(select_tests, "git", git),
                mock.patch.object(select_tests, "TESTS", ["test_foo", "test_jit"]),
                mock.patch.object(select_tests, "get_test_times", return_value=times),
                mock.patch.object(
                    select_tests, "query_model", return_value=answer
                ) as query,
            ):
                select_tests.main(
                    ["--title", "t", "--base", "HEAD~1", "--output", str(output)]
                )
            record = json.loads(output.read_text())
        return record, query.call_args.args[0].split("# Candidate test files")[1]

    def test_moved_test_file_is_selected(self) -> None:
        answer = {"structured_output": {"run_all": False, "reason": "", "sweeps": [], "dirs": [], "tests": []}}  # fmt: skip
        record, candidates = self.run_in_repo(answer)
        self.assertIsNone(record["error"])
        self.assertFalse(record["run_all"])
        self.assertLessEqual({"nn/test_foo", "test_jit"}, set(record["files"]))
        self.assertNotIn("select nothing", candidates)
        self.assertNotIn("jit/test_new", candidates)

    def test_malformed_output_is_an_error(self) -> None:
        answer = {"structured_output": {"run_all": False, "reason": "", "sweeps": [], "dirs": [], "tests": ["test_foo"]}}  # fmt: skip
        record, _ = self.run_in_repo(answer)
        self.assertIn("malformed", record["error"])
        self.assertTrue(record["run_all"])


if __name__ == "__main__":
    unittest.main()
