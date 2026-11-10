from __future__ import annotations

import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))

import tools.testing.target_determination.exploration as exploration
import tools.testing.target_determination.heuristics.interface as interface
from tools.test.heuristics import scorer_fixture as fixture
from tools.testing.target_determination.scorer import select
from tools.testing.test_run import TestRun


sys.path.remove(str(REPO_ROOT))

CASE = fixture.BUDGET_CUT_CASE
TD_ORDER = ["h", "b", "d", "c", "a", "e", "f", "g"]
MODEL = "2026-11-10T00:00:00Z"


def make_tp(names_in_order: list[str]) -> interface.TestPrioritizations:
    """A TestPrioritizations whose TD order is `names_in_order`."""
    n = len(names_in_order)
    return interface.TestPrioritizations(
        names_in_order, {TestRun(t): float(n - i) for i, t in enumerate(names_in_order)}
    )


def fixture_inputs() -> tuple[
    interface.TestPrioritizations, dict[str, Any], dict[str, float]
]:
    """The budget cut case as select_tests' inputs, in TD_ORDER: file times
    are the fixture's costs minus exploration's overhead."""
    tp = make_tp(TD_ORDER)
    block = {
        "trained_at": MODEL,
        "p": dict(CASE["p"]),
        "budget_share": CASE["budget_share"],
    }
    times = {t: cost - exploration.OVERHEAD_SECONDS for t, cost in CASE["cost"].items()}
    return tp, block, times


class SelectTestCase(unittest.TestCase):
    def setUp(self) -> None:
        patch = mock.patch("sys.stderr", io.StringIO())
        patch.start()
        self.addCleanup(patch.stop)

    def assert_td_cut(
        self,
        tp: interface.TestPrioritizations,
        result: tuple[list[TestRun], list[TestRun], dict[str, Any]],
        percent: int = 25,
    ) -> str:
        include, exclude, info = result
        self.assertEqual((include, exclude), tp.get_top_per_tests(percent))
        for key in (
            "budget_share",
            "order",
            "total_seconds",
            "budget_seconds",
            "selected_seconds",
        ):
            self.assertIsNone(info[key], key)
        self.assertIsNotNone(info["fallback"])
        self.assertEqual(info["n_selected"], len(include))
        self.assertEqual(info["n_tests"], len(include) + len(exclude))
        return str(info["fallback"])


class TestBudget(SelectTestCase):
    def test_fixture_order_and_cut(self) -> None:
        names = CASE["tests"]
        costs = [CASE["cost"][t] for t in names]
        order = select.budget_order(names, [CASE["p"][t] for t in names], costs)
        total = math.fsum(costs)
        self.assertEqual(total, CASE["expected_total_seconds"])
        budget = CASE["budget_share"] * total
        self.assertAlmostEqual(budget, CASE["expected_budget_seconds"], delta=1e-11)
        n = select.budget_cut(order, costs, budget)
        self.assertEqual([names[i] for i in order[:n]], CASE["expected_selected"])

    def test_costlier_than_budget_first_is_taken(self) -> None:
        self.assertEqual(select.budget_cut([0, 1], [1000.0, 10.0], 100.0), 1)
        self.assertEqual(select.budget_cut([1, 0], [1000.0, 10.0], 100.0), 2)
        self.assertEqual(select.budget_cut([0, 1], [50.0, 50.0], 50.0), 1)
        self.assertEqual(select.budget_cut([0, 1], [10.0, 10.0], 1e9), 2)
        self.assertEqual(select.budget_cut([], [], 10.0), 0)

    def test_ties_by_name(self) -> None:
        order = select.budget_order(["b", "a", "c"], [0.1, 0.2, 0.1], [10, 20, 10])
        self.assertEqual(order, [1, 0, 2])


class TestSelectTests(SelectTestCase):
    def test_budget_cut(self) -> None:
        tp, block, times = fixture_inputs()
        include, exclude, info = select.select_tests(tp, block, 25, times, {})
        self.assertEqual([str(t) for t in include], CASE["expected_selected"])
        self.assertIsNone(info["fallback"])
        self.assertEqual(info["model"], MODEL)
        self.assertEqual(info["budget_share"], CASE["budget_share"])
        self.assertEqual(info["total_seconds"], CASE["expected_total_seconds"])
        self.assertAlmostEqual(
            info["budget_seconds"], CASE["expected_budget_seconds"], delta=1e-11
        )
        self.assertEqual(info["selected_seconds"], CASE["expected_selected_seconds"])
        self.assertEqual(info["td_cut_seconds"], 380.0 + 25.0 + 35.0)
        self.assertEqual((info["n_tests"], info["n_selected"]), (8, 5))
        self.assertEqual(sorted(str(t) for t in include + exclude), sorted(TD_ORDER))
        self.assertFalse({str(t) for t in include} & {str(t) for t in exclude})
        picked = set(CASE["expected_selected"])
        self.assertEqual(
            [str(t) for t in exclude], [t for t in TD_ORDER if t not in picked]
        )
        order = [str(t) for t in info["order"]]
        self.assertEqual(order[:5], CASE["expected_selected"])
        self.assertEqual(sorted(order), sorted(TD_ORDER))

    def test_share_scales_the_selection(self) -> None:
        tp, block, times = fixture_inputs()
        cheap = {**block, "p": {**block["p"], "a": 0.001}}
        expected = {
            0.1: ["b", "d", "e", "g", "f"],
            0.25: ["b", "d", "e", "g", "f", "c"],
            0.4: ["b", "d", "e", "g", "f", "c", "h"],
        }
        for share, names in expected.items():
            include, _, info = select.select_tests(
                tp, {**cheap, "budget_share": share}, 25, times, {}
            )
            self.assertEqual([str(t) for t in include], names, share)
            self.assertEqual(info["budget_seconds"], share * 1243.0)

    def test_percent_100_runs_everything(self) -> None:
        tp, block, times = fixture_inputs()
        include, exclude, info = select.select_tests(tp, block, 100, times, {})
        self.assertEqual(include, tp.get_all_tests())
        self.assertEqual(exclude, [])
        self.assertEqual(info["fallback"], "TD is off")

    def test_no_block(self) -> None:
        tp, _, times = fixture_inputs()
        reason = self.assert_td_cut(tp, select.select_tests(tp, None, 25, times, {}))
        self.assertEqual(reason, "no scorer block")

    def test_scorer_fell_back(self) -> None:
        tp, _, times = fixture_inputs()
        reason = self.assert_td_cut(
            tp, select.select_tests(tp, {"fallback": "missing"}, 25, times, {})
        )
        self.assertIn("missing", reason)

    def test_bad_p(self) -> None:
        tp, block, times = fixture_inputs()
        for bad in (None, math.nan, math.inf, 0.0, -0.1, True, "0.1"):
            p = dict(block["p"])
            if bad is None:
                del p["a"]
            else:
                p["a"] = bad
            self.assert_td_cut(
                tp, select.select_tests(tp, {**block, "p": p}, 25, times, {})
            )
        self.assert_td_cut(
            tp, select.select_tests(tp, {"trained_at": "x"}, 25, times, {})
        )

    def test_bad_budget_share(self) -> None:
        tp, block, times = fixture_inputs()
        for bad in (None, True, "0.2", math.nan, math.inf, 0.0999, 0.4001, -1):
            reason = self.assert_td_cut(
                tp,
                select.select_tests(tp, {**block, "budget_share": bad}, 25, times, {}),
            )
            self.assertIn("budget share", reason, bad)
        del block["budget_share"]
        reason = self.assert_td_cut(tp, select.select_tests(tp, block, 25, times, {}))
        self.assertEqual(reason, "no budget share")
        for edge in (0.1, 0.4):
            info = select.select_tests(
                tp, {**block, "budget_share": edge}, 25, times, {}
            )[2]
            self.assertIsNone(info["fallback"], edge)

    def test_no_times(self) -> None:
        tp, block, _ = fixture_inputs()
        reason = self.assert_td_cut(tp, select.select_tests(tp, block, 25, {}, {}))
        self.assertEqual(reason, "no test times")

    def test_fewer_than_half_timed(self) -> None:
        tp, block, times = fixture_inputs()
        few = {t: times[t] for t in ["a", "b", "c"]}
        reason = self.assert_td_cut(tp, select.select_tests(tp, block, 25, few, {}))
        self.assertIn("3 of 8", reason)
        half = {t: times[t] for t in ["a", "b", "c", "d"]}
        info = select.select_tests(tp, block, 25, half, {})[2]
        self.assertIsNone(info["fallback"])

    def test_exception_falls_back(self) -> None:
        tp, block, times = fixture_inputs()
        with mock.patch.object(
            select, "estimate_costs", side_effect=RuntimeError("boom")
        ):
            reason = self.assert_td_cut(
                tp, select.select_tests(tp, block, 25, times, {})
            )
        self.assertEqual(reason, "RuntimeError: boom")

    def test_late_failure_keeps_td_cut_seconds(self) -> None:
        tp, block, times = fixture_inputs()
        with mock.patch.object(
            select, "budget_order", side_effect=RuntimeError("late")
        ):
            result = select.select_tests(tp, block, 25, times, {})
        self.assertEqual(self.assert_td_cut(tp, result), "RuntimeError: late")
        self.assertEqual(result[2]["td_cut_seconds"], 440.0)
        self.assertEqual(result[2]["model"], MODEL)

    def test_amended_list(self) -> None:
        tp, block, times = fixture_inputs()
        tp.amend_tests(["a", "b", "c", "d", "new_test"])
        self.assert_td_cut(tp, select.select_tests(tp, block, 25, times, {}))
        tp, block, times = fixture_inputs()
        tp.amend_tests(["a", "b", "c", "d"])
        include, exclude, info = select.select_tests(tp, block, 25, times, {})
        self.assertIsNone(info["fallback"])
        self.assertEqual([str(t) for t in include], ["b", "d", "a"])
        self.assertEqual(len(include) + len(exclude), 4)


class TestFailureInfo(SelectTestCase):
    def test_position_in_the_model_order(self) -> None:
        tp, block, times = fixture_inputs()
        include, _, info = select.select_tests(tp, block, 25, times, {})
        order = [str(t) for t in info["order"]]
        self.assertEqual(order[: len(include)], [str(t) for t in include])
        for name in ("a", "h", "g"):
            got = select.failure_position_info(info, TestRun(name))
            self.assertEqual(
                got,
                {
                    "model": MODEL,
                    "budget_share": CASE["budget_share"],
                    "position": order.index(name),
                },
            )
        part = TestRun("a", included=["TestX"])
        got = select.failure_position_info(info, part)
        if got is None:
            self.fail("no scorer info")
        self.assertEqual(got["position"], order.index("a"))

    def test_none_when_td_cut_ran(self) -> None:
        tp, block, times = fixture_inputs()
        info = select.select_tests(tp, None, 25, times, {})[2]
        self.assertIsNone(info["order"])
        self.assertIsNone(select.failure_position_info(info, TestRun("a")))
        info = select.select_tests(tp, block, 25, {}, {})[2]
        self.assertIsNone(select.failure_position_info(info, TestRun("a")))

    def test_cut_metric(self) -> None:
        tp, block, times = fixture_inputs()
        info = select.select_tests(tp, block, 25, times, {})[2]
        self.assertEqual(
            select.cut_metric(info),
            {
                "model": MODEL,
                "budget_share": CASE["budget_share"],
                "fallback": None,
                "total_seconds": CASE["expected_total_seconds"],
                "budget_seconds": info["budget_seconds"],
                "selected_seconds": CASE["expected_selected_seconds"],
                "td_cut_seconds": 440.0,
                "n_tests": 8,
                "n_selected": 5,
            },
        )
        info = select.select_tests(tp, None, 25, times, {})[2]
        metric = select.cut_metric(info)
        self.assertEqual(metric["fallback"], "no scorer block")
        self.assertIsNone(metric["budget_share"])
        self.assertIsNone(metric["selected_seconds"])
        self.assertEqual((metric["n_tests"], metric["n_selected"]), (8, 3))
        json.dumps(metric, allow_nan=False)
        info = select.select_tests(tp, block, 25, times, {"x": {}})[2]
        json.dumps(select.cut_metric(info), allow_nan=False)

    def test_describe(self) -> None:
        tp, block, times = fixture_inputs()
        info = select.select_tests(tp, block, 25, times, {})[2]
        line = select.describe_cut(info)
        self.assertNotIn("\n", line)
        self.assertIn("budget share 0.250, 5 of 8 tests, 528s", line)
        info = select.select_tests(tp, None, 25, times, {})[2]
        self.assertIn("TD's ranking (no scorer block)", select.describe_cut(info))


SHARD_SCRIPT = """
import json, sys
sys.path.insert(0, ".")
from tools.testing.target_determination.heuristics.interface import (
    TestPrioritizations,
)
from tools.testing.target_determination.scorer import select
from tools.testing.test_run import TestRun

names = [f"t{i:03d}" for i in range(120)]
scores = {
    TestRun(f"t{i:03d}", included=["A", "B", "C"]): 1.0 + i % 3
    for i in range(0, 120, 7)
}
tp = TestPrioritizations(names, scores)
block = {
    "p": {n: 0.01 + (i % 11) / 100 for i, n in enumerate(names)},
    "budget_share": 0.2,
}
times = {n: float(5 + (i * 37) % 200) for i, n in enumerate(names)}
include, exclude, info = select.select_tests(tp, block, 25, times, {})
out = [[str(t) for t in include], [str(t) for t in exclude], dict(info)]
out[2]["order"] = [str(t) for t in info["order"]]
if len(sys.argv) > 1:
    from tools.testing.target_determination import exploration
    explored = exploration.select_exploration(
        tp, include, exclude, times, {}, td_enabled=True,
        which_shard=int(sys.argv[1]), budget_cut=info["fallback"] is None,
    )
    out.append([str(t) for t in explored.explored])
    out.append([str(t) for t in explored.skipped])
print(json.dumps(out))
"""


def run_shard(hash_seed: int, *args: str) -> str:
    """SHARD_SCRIPT's output in a fresh process: each shard of a job is its
    own process, with its own hash seed."""
    return subprocess.run(
        [sys.executable, "-c", SHARD_SCRIPT, *args],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONHASHSEED": str(hash_seed)},
        capture_output=True,
        text=True,
        check=True,
    ).stdout


class TestShards(SelectTestCase):
    def test_every_shard_cuts_alike(self) -> None:
        cuts = []
        for _ in range(2):
            tp, block, times = fixture_inputs()
            include, exclude, info = select.select_tests(tp, block, 25, times, {})
            info["order"] = [str(t) for t in info["order"]]
            cuts.append(([str(t) for t in include], [str(t) for t in exclude], info))
        self.assertEqual(cuts[0], cuts[1])
        self.assertIsNone(cuts[0][2]["fallback"])

    def test_same_cut_under_any_hash_seed(self) -> None:
        outputs = {run_shard(hash_seed) for hash_seed in (0, 1, 2, 3)}
        self.assertEqual(len(outputs), 1, outputs)
        include, exclude, info = json.loads(outputs.pop())
        self.assertIsNone(info["fallback"])
        self.assertTrue(include and exclude)


class TestShardsWithExploration(SelectTestCase):
    def test_every_shard_explores_alike(self) -> None:
        tp, block, times = fixture_inputs()
        include, exclude, info = select.select_tests(tp, block, 25, times, {})
        shards = []
        with mock.patch.dict(os.environ, {"SHA1": "a" * 40}):
            for which_shard in (1, 2):
                shards.append(
                    exploration.select_exploration(
                        tp,
                        include,
                        exclude,
                        times,
                        {},
                        td_enabled=True,
                        which_shard=which_shard,
                        budget_cut=info["fallback"] is None,
                    )
                )
        first, second = shards
        self.assertTrue(first.explored)
        self.assertEqual(first.explored, second.explored)
        self.assertEqual(first.skipped, second.skipped)
        if first.info is None:
            self.fail("no td_exploration info on shard 1")
        self.assertEqual(first.info["base"], "budget")
        self.assertIsNone(second.info)

    def test_same_draw_on_any_shard_and_hash_seed(self) -> None:
        env = {"SHA1": "b" * 40, "BUILD_ENVIRONMENT": "env", "TEST_CONFIG": "default"}
        with mock.patch.dict(os.environ, env):
            outputs = {
                run_shard(hash_seed, shard)
                for hash_seed, shard in ((0, "1"), (1, "2"), (2, "1"), (3, "2"))
            }
        self.assertEqual(len(outputs), 1, outputs)
        explored = json.loads(outputs.pop())[3]
        self.assertTrue(explored)


class TestImportScorer(SelectTestCase):
    def test_import(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "td_results.json"
            self.assertIsNone(select.import_scorer(path))
            path.write_text("{bad")
            self.assertIsNone(select.import_scorer(path))
            path.write_text(json.dumps({"_test_scores": [], "scorer": [1]}))
            self.assertIsNone(select.import_scorer(path))
            block = {"trained_at": "t", "p": {"a": 0.5}}
            path.write_text(
                json.dumps({"_test_scores": [], "_original_tests": [], "scorer": block})
            )
            self.assertEqual(select.import_scorer(path), block)
            with mock.patch.object(select, "TD_RESULTS_PATH", path):
                self.assertEqual(select.import_scorer(), block)


class TestWithExploration(SelectTestCase):
    def test_exploration_on_the_model_cut(self) -> None:
        names = [f"test_{i:03d}" for i in range(80)]
        tp = make_tp(names)
        block = {
            "p": {t: 0.01 + (i % 9) / 100 for i, t in enumerate(names)},
            "budget_share": 0.2,
        }
        times = {t: float(5 + (i * 37) % 200) for i, t in enumerate(names)}
        include, exclude, info = select.select_tests(tp, block, 25, times, {})
        self.assertIsNone(info["fallback"])
        self.assertNotEqual(include, tp.get_top_per_tests(25)[0])
        picked = {str(t) for t in include}
        td_order = [t for t in tp.get_all_tests() if str(t) not in picked]
        self.assertEqual(exclude, td_order)
        with mock.patch.dict(os.environ, {"SHA1": "a" * 40}):
            result = exploration.select_exploration(
                tp,
                include,
                exclude,
                times,
                {},
                td_enabled=True,
                which_shard=1,
                budget_cut=info["fallback"] is None,
            )
        self.assertTrue(result.explored)
        exploration.check_disjoint(result.explored, include, exclude)
        if result.info is None:
            self.fail("no td_exploration info")
        self.assertEqual(result.info["base"], "budget")
        self.assertAlmostEqual(
            result.info["budget_seconds"],
            exploration.EXPLORATION_SHARE * info["total_seconds"],
        )
        positions = {str(t): i for i, t in enumerate(tp.get_all_tests())}
        for explored in result.info["explored"]:
            self.assertEqual(explored["position"], positions[explored["test"]])
        self.assertEqual(len(result.explored) + len(result.skipped), len(exclude))


if __name__ == "__main__":
    unittest.main()
