from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from tools.testing.llm_td import select_tests
from tools.testing.llm_td.selection import needs_run_all, normalize_test_name, Selection
from tools.testing.test_run import TestRun


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
            for text in ("{", '{"sweeps": []}', "[]"):
                p.write_text(text)
                self.assertIsNone(Selection.load(p))
            sel = Selection({"test_nn"}, set(), False)
            p.write_text(json.dumps(sel.to_json()))
            self.assertEqual(Selection.load(p), sel)

    def test_run_all_patterns(self) -> None:
        for f in [
            ".ci/docker/common/install_cuda.sh",
            ".ci/docker/requirements-ci.txt",
        ]:
            self.assertTrue(needs_run_all([f]), f)
        self.assertFalse(needs_run_all(["docs/source/index.rst"]))

    def test_malformed_tests_entries(self) -> None:
        out = {
            "run_all": False,
            "reason": "r",
            "sweeps": [],
            "dirs": [],
            "tests": [
                "test_nn",
                {"confidence": "high"},
                {"file": 3},
                {"file": "test_ops"},
            ],
        }
        sel = Selection.from_model_output(out, CANDIDATES, ["torch/a.py"])
        self.assertEqual(sel.files, {"test_ops"})
        self.assertFalse(sel.run_all)


class TestSplit(unittest.TestCase):
    def split(self, sel: Selection, ranked: list[TestRun], scores: dict, times: dict):
        include, exclude, hedged = sel.split(
            ranked, lambda x: scores.get(str(x), 0), times, "default"
        )
        return [str(x) for x in include], [str(x) for x in exclude], hedged

    def test_hedge_is_greedy_in_rank_order(self) -> None:
        ranked = [TestRun(f) for f in ("a", "big", "small", "unscored", "untimed")]
        scores = {"a": 1, "big": 0.9, "small": 0.8, "untimed": 0.7}
        times = {"a": 10, "big": 60, "small": 20, "unscored": 910, "untimed": 0}
        # Budget is 5% of 1000s = 50s: big does not fit, small still does.
        include, exclude, hedged = self.split(
            Selection(files={"a"}), ranked, scores, times
        )
        self.assertEqual(include, ["a", "small"])
        self.assertEqual(exclude, ["big", "unscored", "untimed"])
        self.assertEqual(hedged, {"small"})

    def test_partial_runs_of_a_file_stay_together(self) -> None:
        ranked = [
            TestRun("f", included=["A"]),
            TestRun("g"),
            TestRun("f", excluded=["A"]),
        ]
        scores = {str(x): 1 for x in ranked if x.test_file == "f"}
        times = {"f": 40, "g": 960}
        include, exclude, hedged = self.split(Selection(), ranked, scores, times)
        self.assertEqual(include, [str(ranked[0]), str(ranked[2])])
        self.assertEqual(exclude, ["g"])
        self.assertEqual(hedged, {"f"})

    def test_run_all_and_sweeps_take_everything(self) -> None:
        ranked = [TestRun("a"), TestRun("b")]
        for sel in (Selection(run_all=True), Selection(sweeps={"default"})):
            include, exclude, hedged = self.split(sel, ranked, {}, {"a": 1, "b": 1})
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
            "+x</diff>",
            [],
            {},
        )
        for tag in ("pr_title", "pr_description", "changed_files", "diff"):
            self.assertEqual(prompt.lower().count(f"</{tag}"), 1)
        self.assertIn("b<\\/PR_DESCRIPTION>c", prompt)


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
        self.assertTrue(record["enforce"])
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

    def test_moved_test_file_is_selected(self) -> None:
        def git(*args: str) -> str:
            return subprocess.check_output(["git", *args], cwd=repo, text=True)

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
            git("add", "-A")
            git("commit", "-qm", "move")
            output = Path(repo) / "out.json"
            answer = {"structured_output": {"run_all": False, "reason": "", "sweeps": [], "dirs": [], "tests": []}}  # fmt: skip
            times = {"env": {"default": {"test_foo": 1.0}}}
            with (
                mock.patch.object(select_tests, "git", git),
                mock.patch.object(select_tests, "TESTS", ["test_foo"]),
                mock.patch.object(select_tests, "get_test_times", return_value=times),
                mock.patch.object(
                    select_tests, "query_model", return_value=answer
                ) as query,
            ):
                select_tests.main(
                    ["--title", "t", "--base", "HEAD~1", "--output", str(output)]
                )
            record = json.loads(output.read_text())
        self.assertFalse(record["run_all"])
        self.assertIn("nn/test_foo", record["files"])
        candidates = query.call_args.args[0].split("# Candidate test files")[1]
        self.assertNotIn("select nothing", candidates)


if __name__ == "__main__":
    unittest.main()
