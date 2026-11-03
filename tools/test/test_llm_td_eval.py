from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from tools.testing.llm_td.eval.build_dataset import strip_trailers
from tools.testing.llm_td.eval.score import (
    Choice,
    config_of,
    cost_fraction,
    policy,
    runs_everything,
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

    def test_runs_everything(self) -> None:
        jobs = {
            "test_a": [
                "unit-test / linux-jammy-cpu-py3.12-gcc11-inductor / test (inductor, 1, 2, linux.2xlarge)",
                "linux / test (inductor, 1, 2, linux.2xlarge)",
            ],
            "test_b": ["linux / test (default, 1, 2, linux.2xlarge)"],
        }
        self.assertEqual(
            runs_everything({**SAMPLE, "truth_jobs": jobs}), {("test_a", "inductor")}
        )

    def test_hedge(self) -> None:
        # test_e has no recorded time and test_d no positive score, so neither is ever added.
        td = {"scored": ["test_e", "test_a", "test_b", "test_c"]}
        # default: 5% of 250s fits only test_c; slow: test_a (200s) does not fit 5% of 200s.
        choice = pick("final+td5", NOTHING, td)
        self.assertEqual(choice.per_config, {"default": {"test_c"}, "slow": set()})
        self.assertEqual(
            pick("final+td", NOTHING, td).per_config,
            choice.per_config,
        )
        # 45% of 250s: test_a fits, test_b then does not, test_c after it still does.
        choice = pick("final+td45", NOTHING, td)
        self.assertEqual(
            choice.per_config, {"default": {"test_a", "test_c"}, "slow": set()}
        )
        self.assertTrue(choice.covers("test_a", "default"))
        self.assertFalse(choice.covers("test_a", "slow"))
        # Files the selection already covers cost no budget.
        r = {"out": {"tests": [{"file": "test_a"}]}}
        self.assertEqual(
            pick("final+td45", r, td).per_config["default"],
            {"test_b", "test_c"},
        )
        self.assertIsNone(policy("final+td5", SAMPLE, NOTHING, None, SECONDS))
        for bad in ("final+tdx", "final+td5x", "nonsense"):
            with self.assertRaises(ValueError):
                policy(bad, SAMPLE, NOTHING, td, SECONDS)

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
        msg = "Body line\n\nCo-authored-by: x\nPull Request resolved: https://x\nApproved by: https://y\nghstack-source-id: 1\nKeep: me\n"
        self.assertEqual(strip_trailers(msg), "Body line\n\nKeep: me")


if __name__ == "__main__":
    unittest.main()
