from __future__ import annotations

import contextlib
import dataclasses
import io
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))

import tools.testing.target_determination.exploration as exploration
import tools.testing.target_determination.heuristics.interface as interface
from tools.testing.test_run import TestRun


sys.path.remove(str(REPO_ROOT))


def make_tp(scores: list[float]) -> interface.TestPrioritizations:
    tests = [f"test_{i:03d}" for i in range(len(scores))]
    return interface.TestPrioritizations(
        tests, {TestRun(t): s for t, s in zip(tests, scores)}
    )


def plan(
    tp: interface.TestPrioritizations,
    times: dict[str, float],
    seed: str = "sha:pull:env:default",
    **kwargs: object,
) -> exploration.ExplorationSplit | None:
    include, exclude = tp.get_top_per_tests(25)
    return exploration.plan_exploration(
        include,
        exploration.rank_candidates(exclude, tp.get_scored_tests()),
        times,
        {},
        seed,
        **kwargs,  # type: ignore[arg-type]
    )


def must_plan(
    tp: interface.TestPrioritizations, times: dict[str, float], **kwargs: object
) -> exploration.ExplorationSplit:
    split = plan(tp, times, **kwargs)  # type: ignore[arg-type]
    if split is None:
        raise AssertionError("exploration drew nothing")
    return split


def uniform_times(n: int, seconds: float) -> dict[str, float]:
    return {f"test_{i:03d}": seconds for i in range(n)}


class TestPlanExploration(unittest.TestCase):
    def test_td_top_unchanged_and_explored_subset_of_exclude(self) -> None:
        tp = make_tp([1 - i / 200 for i in range(200)])
        include, exclude = tp.get_top_per_tests(25)
        split = must_plan(tp, uniform_times(200, 30.0))
        self.assertEqual(tp.get_top_per_tests(25), (include, exclude))
        self.assertTrue(split.explored)
        names = [str(t) for t in exclude]
        self.assertTrue({str(t) for t in split.tests()} <= set(names))
        exploration.check_disjoint(split.tests(), include, exclude)
        ranked = tp.get_all_tests()
        for explored in split.explored:
            self.assertEqual(ranked[explored.position], explored.test)
            self.assertEqual(explored.score, tp._test_scores[explored.test])
        positions = [explored.position for explored in split.explored]
        self.assertEqual(positions, sorted(positions))

    def test_check_disjoint_rejects_bad_sets(self) -> None:
        a, b, c = TestRun("a"), TestRun("b"), TestRun("c")
        with self.assertRaises(ValueError):
            exploration.check_disjoint([c], [a], [b])
        with self.assertRaises(ValueError):
            exploration.check_disjoint([a], [a], [a, b])

    def test_budget_frame_and_cap(self) -> None:
        # 40 tests: TD's top is the first 11, each 185 s + 15 s overhead, so the
        # budget is 5% of 2200 s = 110 s
        times = {f"test_{i:03d}": 185.0 for i in range(11)}
        times |= {f"test_{i:03d}": 20.0 + i for i in range(11, 40)}
        times["test_020"] = 96.0  # 111 s with overhead: out of the frame
        split = must_plan(make_tp([0.0] * 40), times)
        self.assertAlmostEqual(split.top_seconds, 2200.0)
        self.assertAlmostEqual(split.budget_seconds, 110.0)
        costs = {name: t + 15.0 for name, t in times.items()}
        out = [n for n in costs if n >= "test_011" and costs[n] > 110.0]
        self.assertEqual(split.out_of_frame, len(out))
        self.assertAlmostEqual(split.out_of_frame_seconds, sum(costs[n] for n in out))
        self.assertAlmostEqual(
            split.skipped_seconds,
            sum(cost for name, cost in costs.items() if name >= "test_011"),
        )
        self.assertEqual(split.frame + split.out_of_frame, 29)
        spent = sum(explored.cost for explored in split.explored)
        self.assertLessEqual(spent, split.budget_seconds)
        left = split.budget_seconds - spent
        picked = {str(t) for t in split.tests()}
        for name, cost in costs.items():
            if name >= "test_011" and cost <= 110.0 and name not in picked:
                self.assertGreater(cost, left)

    def test_file_cap_applies_when_budget_is_large(self) -> None:
        times = uniform_times(400, 3000.0)
        times["test_399"] = exploration.MAX_FILE_SECONDS - 10.0
        times["test_398"] = exploration.MAX_FILE_SECONDS - 20.0
        split = must_plan(make_tp([0.0] * 400), times)
        self.assertGreater(split.budget_seconds, exploration.MAX_FILE_SECONDS)
        self.assertEqual(split.frame, 1)
        self.assertEqual([str(t) for t in split.tests()], ["test_398"])

    def test_missing_times_use_median_and_no_times_skip(self) -> None:
        tests = [TestRun("a"), TestRun("b"), TestRun("c")]
        self.assertEqual(
            exploration.estimate_costs(tests, {"a": 10.0, "b": 50.0}, {}),
            [25.0, 65.0, 45.0],
        )
        self.assertIsNone(exploration.estimate_costs(tests, {"other": 1.0}, {}))
        self.assertIsNone(plan(make_tp([0.0] * 40), {}))

    def test_budget_base_is_a_share_of_the_total(self) -> None:
        # 40 tests of 185 s + 15 s: the total is 8000 s, so the budget base
        # gives 5% of it, 400 s, where TD's cut (11 tests) gives 110 s
        times = uniform_times(40, 185.0)
        tp = make_tp([1 - i / 40 for i in range(40)])
        td_cut = must_plan(tp, times)
        budget = must_plan(tp, times, base=exploration.BASE_BUDGET)
        self.assertEqual(td_cut.base, exploration.BASE_TD_CUT)
        self.assertEqual(budget.base, exploration.BASE_BUDGET)
        self.assertAlmostEqual(td_cut.budget_seconds, 110.0)
        self.assertAlmostEqual(budget.budget_seconds, 400.0)
        self.assertEqual(budget.top_seconds, td_cut.top_seconds)
        self.assertEqual(len(td_cut.explored), 0)
        self.assertEqual(len(budget.explored), 2)
        self.assertEqual(budget.frame, 29)
        include, exclude = tp.get_top_per_tests(25)
        exploration.check_disjoint(budget.tests(), include, exclude)
        with self.assertRaises(ValueError):
            plan(tp, times, base="other")

    def test_budget_base_keeps_the_frame_cap_and_seed(self) -> None:
        times = uniform_times(400, 30.0)
        tp = make_tp([1 - i / 400 for i in range(400)])
        budget = must_plan(tp, times, base=exploration.BASE_BUDGET)
        self.assertAlmostEqual(budget.budget_seconds, 0.05 * 400 * 45.0)
        self.assertEqual(budget.out_of_frame, 0)
        self.assertEqual(
            budget.tests(),
            must_plan(tp, times, base=exploration.BASE_BUDGET).tests(),
        )
        big = uniform_times(400, 30.0)
        big["test_399"] = exploration.MAX_FILE_SECONDS
        capped = must_plan(tp, big, base=exploration.BASE_BUDGET)
        self.assertGreater(capped.budget_seconds, exploration.MAX_FILE_SECONDS)
        self.assertEqual(capped.out_of_frame, 1)

    def test_share_zero_changes_nothing(self) -> None:
        tp = make_tp([1 - i / 200 for i in range(200)])
        self.assertIsNone(plan(tp, uniform_times(200, 30.0), share=0))


class TestWeightsAndSeed(unittest.TestCase):
    def test_weights(self) -> None:
        self.assertEqual(exploration.weights([0.0, 0.0, -1.0]), [1.0, 1.0, 1.0])
        self.assertEqual(
            exploration.weights([3.0, 1.0, 0.0, -2.0]), [4.0, 2.0, 1.0, 1.0]
        )
        self.assertEqual(
            exploration.weights([math.nan, math.inf, 2.0, 0.0]), [1.0, 1.0, 5.0, 1.0]
        )
        self.assertEqual(exploration.weights([math.nan, -math.inf]), [1.0, 1.0])

    def test_deterministic_and_seed_and_call_dependent(self) -> None:
        tp = make_tp([1 - i / 400 for i in range(400)])
        times = uniform_times(400, 20.0)
        first = must_plan(tp, times)
        self.assertEqual(first, plan(tp, times))
        other_seed = must_plan(tp, times, seed="sha2:pull:env:default")
        self.assertNotEqual(first.tests(), other_seed.tests())
        self.assertEqual(first.call, other_seed.call)

        fewer = make_tp([1 - i / 399 for i in range(399)])
        other_call = must_plan(fewer, times)
        self.assertNotEqual(first.call, other_call.call)
        self.assertNotEqual(first.tests(), other_call.tests())

    def test_same_draw_under_any_hash_seed(self) -> None:
        script = (
            "import sys; sys.path.insert(0, '.')\n"
            "from tools.testing.target_determination import exploration as e\n"
            "from tools.testing.target_determination.heuristics.interface import "
            "TestPrioritizations\n"
            "from tools.testing.test_run import TestRun\n"
            "names = [f't{i:03d}' for i in range(300)]\n"
            "scores = {TestRun(f't{i:03d}', included=['A', 'B', 'C']): 1.0 + i % 3 "
            "for i in range(0, 300, 7)}\n"
            "tp = TestPrioritizations(names, scores)\n"
            "inc, exc = tp.get_top_per_tests(25)\n"
            "times = {n: 30.0 for n in names}\n"
            "s = e.plan_exploration(inc, e.rank_candidates(exc, tp.get_scored_tests()), "
            "times, {}, 'seed', replicates=50)\n"
            "print(s.call, [(str(x.test), x.count) for x in s.explored])\n"
        )
        outputs = {
            subprocess.run(
                [sys.executable, "-c", script],
                cwd=REPO_ROOT,
                env={**os.environ, "PYTHONHASHSEED": str(hash_seed)},
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            for hash_seed in (0, 1, 2, 3)
        }
        self.assertEqual(len(outputs), 1, outputs)


class TestPropensities(unittest.TestCase):
    def draw_counts(
        self,
        split: exploration.ExplorationSplit,
        frame_weights: list[float],
        frame_costs: list[float],
    ) -> list[int]:
        counts = [0] * len(frame_costs)
        seed = f"sha:pull:env:default:{split.call}"
        for r in range(split.replicates):
            for i in exploration.draw(
                seed, r, frame_weights, frame_costs, split.budget_seconds
            ):
                counts[i] += 1
        return counts

    def check(
        self, split: exploration.ExplorationSplit, counts: list[int], names: list[str]
    ) -> None:
        for explored in split.explored:
            self.assertEqual(explored.count, counts[names.index(str(explored.test))])
            self.assertEqual(explored.propensity, explored.count / split.replicates)
            self.assertGreaterEqual(explored.propensity, 1 / split.replicates)
            self.assertLessEqual(explored.propensity, 1.0)

    def test_equal_weights_and_costs_give_k_over_n(self) -> None:
        # Top: 3 tests of 485 s + 15 s, so the budget is 75 s; each of the 8
        # candidates costs 25 s, so exactly 3 fit
        times = uniform_times(11, 10.0)
        times |= {f"test_{i:03d}": 485.0 for i in range(3)}
        tp = make_tp([1.0, 1.0, 1.0] + [0.0] * 8)
        split = must_plan(tp, times)
        self.assertEqual((split.budget_seconds, split.frame), (75.0, 8))
        self.assertEqual(len(split.explored), 3)
        counts = self.draw_counts(split, [1.0] * 8, [25.0] * 8)
        self.check(split, counts, [str(t) for t in tp.get_top_per_tests(25)[1]])
        self.assertEqual(sum(counts), 3 * split.replicates)
        for count in counts:
            self.assertAlmostEqual(count / split.replicates, 3 / 8, delta=0.06)

    def test_one_pick_follows_the_weights(self) -> None:
        # Two candidates with scores 1 and 0 weigh 3 and 1; the budget is 20 s
        # and each costs 17 s, so exactly one fits
        times = {"test_000": 85.0, "test_001": 85.0, "test_002": 2.0, "test_003": 2.0}
        tp = make_tp([2.0, 2.0, 1.0, 0.0])
        split = must_plan(tp, times, share=0.1)
        self.assertEqual(len(split.explored), 1)
        counts = self.draw_counts(split, [3.0, 1.0], [17.0, 17.0])
        self.check(split, counts, ["test_002", "test_003"])
        self.assertEqual(sum(counts), split.replicates)
        self.assertAlmostEqual(counts[0] / split.replicates, 0.75, delta=0.05)


ENV = {
    "SHA1": "deadbeef",
    "GITHUB_WORKFLOW": "pull",
    "BUILD_ENVIRONMENT": "env",
    "TEST_CONFIG": "default",
    "PR_NUMBER": "123",
}


class TestSelectExploration(unittest.TestCase):
    def select(
        self,
        env: dict[str, str] = ENV,
        td_enabled: bool = True,
        which_shard: int = 1,
        **kwargs: bool,
    ) -> tuple[list[TestRun], list[TestRun], exploration.Exploration]:
        tp = make_tp([1 - i / 200 for i in range(200)])
        include, exclude = tp.get_top_per_tests(25)
        stderr = io.StringIO()
        with (
            mock.patch.dict(os.environ, env, clear=True),
            contextlib.redirect_stderr(stderr),
        ):
            selected = exploration.select_exploration(
                tp,
                include,
                exclude,
                uniform_times(200, 30.0),
                {},
                td_enabled=td_enabled,
                which_shard=which_shard,
                **kwargs,
            )
        self.stderr = stderr.getvalue()
        return include, exclude, selected

    def test_artifact_lists_partition_td_order(self) -> None:
        include, exclude, selected = self.select()
        self.assertTrue(selected.enabled)
        self.assertTrue(selected.explored)
        names = {str(t) for t in selected.explored}
        self.assertEqual(selected.skipped, [t for t in exclude if str(t) not in names])
        exploration.check_disjoint(selected.explored, include, exclude)
        artifact = [str(t) for t in include + selected.explored + selected.skipped]
        self.assertEqual(sorted(artifact), sorted(str(t) for t in include + exclude))
        self.assertEqual(len(set(artifact)), len(artifact))

    def test_metric_built_on_shard_one_only(self) -> None:
        _, _, first = self.select()
        _, _, second = self.select(which_shard=2)
        self.assertEqual(first.explored, second.explored)
        self.assertIsNone(second.info)
        self.assertIsNotNone(first.info)
        info = first.info or {}
        self.assertEqual(info.get("sha"), "deadbeef")
        self.assertEqual(info.get("pr_number"), 123)
        self.assertEqual(
            [e["test"] for e in info.get("explored", [])],
            [str(t) for t in first.explored],
        )

    def test_metric_records_the_base(self) -> None:
        _, _, td_cut = self.select()
        _, _, budget = self.select(budget_cut=True)
        if td_cut.info is None or budget.info is None:
            self.fail("no td_exploration info")
        self.assertEqual(td_cut.info["base"], "td_cut")
        self.assertEqual(budget.info["base"], "budget")
        self.assertAlmostEqual(budget.info["budget_seconds"], 0.05 * 200 * 45.0)
        self.assertAlmostEqual(
            td_cut.info["budget_seconds"], 0.05 * td_cut.info["top_seconds"]
        )
        self.assertIn("(budget)", self.stderr)

    def test_disabled_cases_leave_td_as_is(self) -> None:
        no_sha = {k: v for k, v in ENV.items() if k != "SHA1"}
        cases = [(no_sha, True), (ENV, False)]
        for env, td_enabled in cases:
            with self.subTest(env=env, td_enabled=td_enabled):
                _, exclude, selected = self.select(env, td_enabled)
                self.assertEqual(selected, (False, [], exclude, None))
                self.assertIs(selected.skipped, exclude)
                self.assertEqual(self.stderr, "")
        with mock.patch.object(exploration, "EXPLORATION_SHARE", 0):
            _, exclude, selected = self.select()
        self.assertEqual(selected, (False, [], exclude, None))

    def test_errors_fall_back_to_td(self) -> None:
        tp = make_tp([1 - i / 200 for i in range(200)])
        split = must_plan(tp, uniform_times(200, 30.0))
        top = tp.get_top_per_tests(25)[0][0]
        overlapping = dataclasses.replace(
            split,
            explored=[exploration.ExploredTest(top, 1.0, 0, 1.0, 1.0, 1, 0.001)],
        )
        for patched, error in (
            (mock.Mock(side_effect=RuntimeError("boom")), "RuntimeError: boom"),
            (mock.Mock(return_value=overlapping), "not all in TD's excluded tests"),
        ):
            with (
                self.subTest(error=error),
                mock.patch.object(exploration, "plan_exploration", patched),
            ):
                _, exclude, selected = self.select()
                self.assertEqual(selected, (True, [], exclude, None))
                self.assertIs(selected.skipped, exclude)
                self.assertIn("TD exploration failed", self.stderr)
                self.assertIn(error, self.stderr)


class TestRunExplored(unittest.TestCase):
    def test_gate(self) -> None:
        cases = [
            # td_failed, keep_going -> runs
            ((False, False), True),
            ((False, True), True),
            ((True, True), True),
            ((True, False), False),
        ]
        for args, expected in cases:
            with self.subTest(args=args):
                self.assertEqual(exploration.run_explored(*args), expected)


class TestExplorationInfo(unittest.TestCase):
    def test_info(self) -> None:
        split = exploration.ExplorationSplit(
            call="abc",
            top_seconds=1000.0,
            budget_seconds=50.0,
            frame=10,
            out_of_frame=2,
            out_of_frame_seconds=900.0,
            skipped_seconds=1200.0,
            replicates=1000,
            explored=[
                exploration.ExploredTest(
                    TestRun("test_x"), math.nan, 7, 1.0, 40.0, 120, 0.12
                ),
                exploration.ExploredTest(
                    TestRun("test_y", excluded=["TestA"]), 0.5, 9, 2.0, 30.0, 300, 0.3
                ),
            ],
        )
        info = exploration.exploration_info(
            split,
            [TestRun("test_y::TestA"), TestRun("test_z")],
            sha="deadbeef",
            pr_number=123,
            num_tests=40,
        )
        self.assertEqual(
            info,
            {
                "sha": "deadbeef",
                "pr_number": 123,
                "call": "abc",
                "base": "td_cut",
                "num_tests": 40,
                "top_tests": 2,
                "top_seconds": 1000.0,
                "budget_seconds": 50.0,
                "frame": 10,
                "out_of_frame": 2,
                "out_of_frame_seconds": 900.0,
                "skipped_seconds": 1200.0,
                "weight_scheme": exploration.WEIGHT_SCHEME,
                "replicates": 1000,
                "explored": [
                    {
                        "test": "test_x",
                        "test_file": "test_x",
                        "score": None,
                        "weight": 1.0,
                        "position": 7,
                        "cost": 40.0,
                        "count": 120,
                        "propensity": 0.12,
                        "top_part": False,
                    },
                    {
                        "test": "test_y, not (TestA)",
                        "test_file": "test_y",
                        "score": 0.5,
                        "weight": 2.0,
                        "position": 9,
                        "cost": 30.0,
                        "count": 300,
                        "propensity": 0.3,
                        "top_part": True,
                    },
                ],
            },
        )


if __name__ == "__main__":
    unittest.main()
