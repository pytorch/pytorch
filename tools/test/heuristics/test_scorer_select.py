from __future__ import annotations

import io
import json
import math
import os
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

CASE = fixture.CUT_CASE


def make_tp(names_in_order: list[str]) -> interface.TestPrioritizations:
    """A TestPrioritizations whose TD order is `names_in_order`."""
    n = len(names_in_order)
    return interface.TestPrioritizations(
        names_in_order, {TestRun(t): float(n - i) for i, t in enumerate(names_in_order)}
    )


def fixture_inputs() -> tuple[
    interface.TestPrioritizations, dict[str, Any], dict[str, float]
]:
    """The cut case as select_tests' inputs: file times are the fixture's
    costs minus exploration's overhead."""
    tp = make_tp(CASE["td_order"])
    block = {"trained_at": "2026-11-10T00:00:00Z", "p": dict(CASE["p"])}
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
        self.assertIsNone(info["exponent"])
        self.assertIsNotNone(info["fallback"])
        return str(info["fallback"])


class TestMatch(SelectTestCase):
    def test_fixture_cut_case(self) -> None:
        names = CASE["td_order"]
        n = CASE["percent"] * len(names) // 100 + 1
        match = select.match_order(
            names, [CASE["p"][t] for t in names], [CASE["cost"][t] for t in names], n
        )
        self.assertEqual(match.exponent, CASE["expected_exponent"])
        self.assertEqual([names[i] for i in match.order[:n]], CASE["expected_include"])

    def test_exponent_grid(self) -> None:
        self.assertEqual(len(select.EXPONENTS), 21)
        self.assertEqual(select.EXPONENTS[0], 0.0)
        self.assertEqual(select.EXPONENTS[-1], 1.0)
        self.assertEqual(select.MAX_GAP, 0.25)

    def test_ties_by_name_and_smaller_exponent(self) -> None:
        match = select.match_order(["b", "a", "c"], [0.1] * 3, [10.0] * 3, 1)
        self.assertEqual(match.exponent, 0.0)
        self.assertEqual(match.order, [1, 0, 2])


class TestSelectTests(SelectTestCase):
    def test_model_cut(self) -> None:
        tp, block, times = fixture_inputs()
        include, exclude, info = select.select_tests(tp, block, 25, times, {})
        self.assertEqual([str(t) for t in include], CASE["expected_include"])
        self.assertEqual(info["exponent"], CASE["expected_exponent"])
        self.assertIsNone(info["fallback"])
        self.assertEqual(info["model"], "2026-11-10T00:00:00Z")
        self.assertAlmostEqual(info["td_cut_seconds"], 380.0 + 25.0 + 35.0)
        self.assertAlmostEqual(info["cut_seconds"], 25.0 + 415.0 + 35.0)
        self.assertEqual(
            sorted(str(t) for t in include + exclude), sorted(CASE["td_order"])
        )
        self.assertFalse({str(t) for t in include} & {str(t) for t in exclude})
        picked = set(CASE["expected_include"])
        self.assertEqual(
            [str(t) for t in exclude], [t for t in CASE["td_order"] if t not in picked]
        )
        self.assertEqual([str(t) for t in info["order"][:3]], CASE["expected_include"])
        self.assertEqual(
            sorted(str(t) for t in info["order"]), sorted(CASE["td_order"])
        )

    def test_percent_100_runs_everything(self) -> None:
        tp, block, times = fixture_inputs()
        include, exclude, info = select.select_tests(tp, block, 100, times, {})
        self.assertEqual(include, tp.get_all_tests())
        self.assertEqual(exclude, [])
        self.assertEqual(info["fallback"], "TD is off")

    def test_no_block(self) -> None:
        tp, _, times = fixture_inputs()
        self.assert_td_cut(tp, select.select_tests(tp, None, 25, times, {}))

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
        self.assertNotIn("have a time", str(info["fallback"]))

    def test_gap_too_large(self) -> None:
        tp = make_tp(["cheap1", "big1", "cheap2", "big2"])
        block = {"p": {"cheap1": 0.01, "cheap2": 0.01, "big1": 0.5, "big2": 0.5}}
        times = {"cheap1": 1.0, "cheap2": 1.0, "big1": 5000.0, "big2": 5000.0}
        reason = self.assert_td_cut(tp, select.select_tests(tp, block, 25, times, {}))
        self.assertIn("25%", reason)

    def test_exception_falls_back(self) -> None:
        tp, block, times = fixture_inputs()
        with mock.patch.object(
            select, "estimate_costs", side_effect=RuntimeError("boom")
        ):
            reason = self.assert_td_cut(
                tp, select.select_tests(tp, block, 25, times, {})
            )
        self.assertEqual(reason, "RuntimeError: boom")

    def test_amended_list(self) -> None:
        tp, block, times = fixture_inputs()
        tp.amend_tests(["a", "b", "c", "d", "new_test"])
        self.assert_td_cut(tp, select.select_tests(tp, block, 25, times, {}))
        tp, block, times = fixture_inputs()
        tp.amend_tests(["a", "b", "c", "d"])
        include, exclude, info = select.select_tests(tp, block, 25, times, {})
        self.assertIsNone(info["fallback"])
        self.assertEqual(len(include), 2)
        self.assertEqual(len(include) + len(exclude), 4)


class TestFailureInfo(SelectTestCase):
    def test_position_in_the_cut_order(self) -> None:
        tp, block, times = fixture_inputs()
        include, _, info = select.select_tests(tp, block, 25, times, {})
        order = [str(t) for t in info["order"]]
        self.assertEqual(order[: len(include)], [str(t) for t in include])
        for name in ("a", "h", "g"):
            got = select.failure_position_info(info, TestRun(name))
            self.assertEqual(
                got,
                {
                    "model": "2026-11-10T00:00:00Z",
                    "exponent": CASE["expected_exponent"],
                    "position": order.index(name),
                },
            )
        part = TestRun("a", included=["TestX"])
        got = select.failure_position_info(info, part)
        if got is None:
            self.fail("no scorer info")
        self.assertEqual(got["position"], order.index("a"))

    def test_none_when_td_cut_ran(self) -> None:
        tp, _, times = fixture_inputs()
        info = select.select_tests(tp, None, 25, times, {})[2]
        self.assertIsNone(info["order"])
        self.assertIsNone(select.failure_position_info(info, TestRun("a")))

    def test_cut_metric(self) -> None:
        tp, block, times = fixture_inputs()
        include, exclude, info = select.select_tests(tp, block, 25, times, {})
        self.assertEqual(
            select.cut_metric(info, len(include) + len(exclude)),
            {
                "model": "2026-11-10T00:00:00Z",
                "exponent": CASE["expected_exponent"],
                "fallback": None,
                "td_cut_seconds": info["td_cut_seconds"],
                "cut_seconds": info["cut_seconds"],
                "n_tests": 8,
            },
        )
        info = select.select_tests(tp, None, 25, times, {})[2]
        metric = select.cut_metric(info, 8)
        self.assertEqual(metric["fallback"], "no scorer block")
        self.assertIsNone(metric["exponent"])
        json.dumps(metric, allow_nan=False)

    def test_describe(self) -> None:
        tp, block, times = fixture_inputs()
        info = select.select_tests(tp, block, 25, times, {})[2]
        self.assertIn("exponent 0.20", select.describe_cut(info))
        info = select.select_tests(tp, None, 25, times, {})[2]
        self.assertIn("TD's ranking (no scorer block)", select.describe_cut(info))


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
        block = {"p": {t: 0.01 + (i % 9) / 100 for i, t in enumerate(names)}}
        times = {t: float(5 + (i * 37) % 200) for i, t in enumerate(names)}
        include, exclude, info = select.select_tests(tp, block, 25, times, {})
        self.assertIsNone(info["fallback"])
        self.assertNotEqual(include, tp.get_top_per_tests(25)[0])
        picked = {str(t) for t in include}
        td_order = [t for t in tp.get_all_tests() if str(t) not in picked]
        self.assertEqual(exclude, td_order)
        with mock.patch.dict(os.environ, {"SHA1": "a" * 40}):
            result = exploration.select_exploration(
                tp, include, exclude, times, {}, td_enabled=True, which_shard=1
            )
        self.assertTrue(result.explored)
        exploration.check_disjoint(result.explored, include, exclude)
        if result.info is None:
            self.fail("no td_exploration info")
        positions = {str(t): i for i, t in enumerate(tp.get_all_tests())}
        for explored in result.info["explored"]:
            self.assertEqual(explored["position"], positions[explored["test"]])
        self.assertEqual(len(result.explored) + len(result.skipped), len(exclude))


if __name__ == "__main__":
    unittest.main()
