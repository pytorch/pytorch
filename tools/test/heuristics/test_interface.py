import pathlib
import sys
import unittest
from typing import Any, Dict, List

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(REPO_ROOT))
import tools.testing.target_determination.heuristics.interface as interface
from tools.testing.test_run import TestRun

sys.path.remove(str(REPO_ROOT))


class TestTD(unittest.TestCase):
    def assert_test_scores_almost_equal(
        self, d1: Dict[TestRun, float], d2: Dict[TestRun, float]
    ) -> None:
        # Check that dictionaries are the same, except for floating point errors
        self.assertEqual(set(d1.keys()), set(d2.keys()))
        for k, v in d1.items():
            self.assertAlmostEqual(v, d2[k], msg=f"{k}: {v} != {d2[k]}")

    def make_heuristic(self, classname: str) -> Any:
        # Create a dummy heuristic class
        class Heuristic(interface.HeuristicInterface):
            def get_prediction_confidence(
                self, tests: List[str]
            ) -> interface.TestPrioritizations:
                # Return junk
                return interface.TestPrioritizations([], {})

        return type(classname, (Heuristic,), {})


class TestTestPrioritizations(TestTD):
    def test_init_none(self) -> None:
        tests = ["test_a", "test_b"]
        test_prioritizations = interface.TestPrioritizations(tests, {})
        self.assertSetEqual(test_prioritizations._original_tests, set(tests))
        self.assertDictEqual(
            test_prioritizations._test_scores,
            {TestRun("test_a"): 0.0, TestRun("test_b"): 0.0},
        )

    def test_init_set_scores_full_files(self) -> None:
        tests = ["test_a", "test_b"]
        test_prioritizations = interface.TestPrioritizations(
            tests, {TestRun("test_a"): 0.5, TestRun("test_b"): 0.25}
        )
        self.assertSetEqual(test_prioritizations._original_tests, set(tests))
        self.assertDictEqual(
            test_prioritizations._test_scores,
            {TestRun("test_a"): 0.5, TestRun("test_b"): 0.25},
        )

    def test_init_set_scores_some_full_files(self) -> None:
        tests = ["test_a", "test_b"]
        test_prioritizations = interface.TestPrioritizations(
            tests, {TestRun("test_a"): 0.5}
        )
        self.assertSetEqual(test_prioritizations._original_tests, set(tests))
        self.assertDictEqual(
            test_prioritizations._test_scores,
            {TestRun("test_a"): 0.5, TestRun("test_b"): 0.0},
        )

    def test_init_set_scores_classes(self) -> None:
        tests = ["test_a", "test_b"]
        test_prioritizations = interface.TestPrioritizations(
            tests, {TestRun("test_a", included=["TestA"]): 0.5}
        )
        self.assertSetEqual(test_prioritizations._original_tests, set(tests))
        self.assertDictEqual(
            test_prioritizations._test_scores,
            {
                TestRun("test_a", included=["TestA"]): 0.5,
                TestRun("test_a", excluded=["TestA"]): 0.0,
                TestRun("test_b"): 0.0,
            },
        )

    def test_init_set_scores_other_class_naming_convention(self) -> None:
        tests = ["test_a", "test_b"]
        test_prioritizations = interface.TestPrioritizations(
            tests, {TestRun("test_a::TestA"): 0.5}
        )
        self.assertSetEqual(test_prioritizations._original_tests, set(tests))
        self.assertDictEqual(
            test_prioritizations._test_scores,
            {
                TestRun("test_a", included=["TestA"]): 0.5,
                TestRun("test_a", excluded=["TestA"]): 0.0,
                TestRun("test_b"): 0.0,
            },
        )

    def test_set_test_score_full_class(self) -> None:
        tests = ["test_a", "test_b"]
        test_prioritizations = interface.TestPrioritizations(tests, {})
        test_prioritizations.set_test_score(TestRun("test_a"), 0.5)
        self.assertSetEqual(test_prioritizations._original_tests, set(tests))
        self.assertDictEqual(
            test_prioritizations._test_scores,
            {TestRun("test_a"): 0.5, TestRun("test_b"): 0.0},
        )

    def test_set_test_score_mix(self) -> None:
        tests = ["test_a", "test_b"]
        test_prioritizations = interface.TestPrioritizations(
            tests, {TestRun("test_b"): -0.5}
        )
        test_prioritizations.set_test_score(TestRun("test_a"), 0.1)
        test_prioritizations.set_test_score(TestRun("test_a::TestA"), 0.2)
        test_prioritizations.set_test_score(TestRun("test_a::TestB"), 0.3)
        test_prioritizations.set_test_score(TestRun("test_a", included=["TestC"]), 0.4)
        self.assertSetEqual(test_prioritizations._original_tests, set(tests))
        self.assertDictEqual(
            test_prioritizations._test_scores,
            {
                TestRun("test_a", included=["TestA"]): 0.2,
                TestRun("test_a", included=["TestB"]): 0.3,
                TestRun("test_a", included=["TestC"]): 0.4,
                TestRun("test_a", excluded=["TestA", "TestB", "TestC"]): 0.1,
                TestRun("test_b"): -0.5,
            },
        )
        test_prioritizations.set_test_score(
            TestRun("test_a", included=["TestA", "TestB"]), 0.5
        )
        self.assertDictEqual(
            test_prioritizations._test_scores,
            {
                TestRun("test_a", included=["TestA", "TestB"]): 0.5,
                TestRun("test_a", included=["TestC"]): 0.4,
                TestRun("test_a", excluded=["TestA", "TestB", "TestC"]): 0.1,
                TestRun("test_b"): -0.5,
            },
        )
        test_prioritizations.set_test_score(
            TestRun("test_a", excluded=["TestA", "TestB"]), 0.6
        )
        self.assertDictEqual(
            test_prioritizations._test_scores,
            {
                TestRun("test_a", included=["TestA", "TestB"]): 0.5,
                TestRun("test_a", excluded=["TestA", "TestB"]): 0.6,
                TestRun("test_b"): -0.5,
            },
        )
        test_prioritizations.set_test_score(TestRun("test_a", included=["TestC"]), 0.7)
        self.assertDictEqual(
            test_prioritizations._test_scores,
            {
                TestRun("test_a", included=["TestA", "TestB"]): 0.5,
                TestRun("test_a", excluded=["TestA", "TestB", "TestC"]): 0.6,
                TestRun("test_a", included=["TestC"]): 0.7,
                TestRun("test_b"): -0.5,
            },
        )
        test_prioritizations.set_test_score(TestRun("test_a", excluded=["TestD"]), 0.8)
        self.assertDictEqual(
            test_prioritizations._test_scores,
            {
                TestRun("test_a", excluded=["TestD"]): 0.8,
                TestRun("test_a", included=["TestD"]): 0.6,
                TestRun("test_b"): -0.5,
            },
        )
        self.assertSetEqual(test_prioritizations._original_tests, set(tests))
        test_prioritizations.validate()

    def test_add_test_score_mix(self) -> None:
        tests = ["test_a", "test_b"]
        test_prioritizations = interface.TestPrioritizations(
            tests, {TestRun("test_b"): -0.5}
        )
        test_prioritizations.add_test_score(TestRun("test_a"), 0.1)
        test_prioritizations.add_test_score(TestRun("test_a::TestA"), 0.2)
        test_prioritizations.add_test_score(TestRun("test_a::TestB"), 0.3)
        test_prioritizations.add_test_score(TestRun("test_a", included=["TestC"]), 0.4)
        self.assertSetEqual(test_prioritizations._original_tests, set(tests))
        self.assert_test_scores_almost_equal(
            test_prioritizations._test_scores,
            {
                TestRun("test_a", included=["TestA"]): 0.3,
                TestRun("test_a", included=["TestB"]): 0.4,
                TestRun("test_a", included=["TestC"]): 0.5,
                TestRun("test_a", excluded=["TestA", "TestB", "TestC"]): 0.1,
                TestRun("test_b"): -0.5,
            },
        )
        test_prioritizations.add_test_score(
            TestRun("test_a", included=["TestA", "TestB"]), 0.5
        )
        self.assert_test_scores_almost_equal(
            test_prioritizations._test_scores,
            {
                TestRun("test_a", included=["TestA"]): 0.8,
                TestRun("test_a", included=["TestB"]): 0.9,
                TestRun("test_a", included=["TestC"]): 0.5,
                TestRun("test_a", excluded=["TestA", "TestB", "TestC"]): 0.1,
                TestRun("test_b"): -0.5,
            },
        )
        test_prioritizations.add_test_score(
            TestRun("test_a", excluded=["TestA", "TestB"]), 0.6
        )
        self.assert_test_scores_almost_equal(
            test_prioritizations._test_scores,
            {
                TestRun("test_a", included=["TestA"]): 0.8,
                TestRun("test_a", included=["TestB"]): 0.9,
                TestRun("test_a", included=["TestC"]): 1.1,
                TestRun("test_a", excluded=["TestA", "TestB", "TestC"]): 0.7,
                TestRun("test_b"): -0.5,
            },
        )
        test_prioritizations.add_test_score(TestRun("test_a", included=["TestC"]), 0.7)
        self.assert_test_scores_almost_equal(
            test_prioritizations._test_scores,
            {
                TestRun("test_a", included=["TestA"]): 0.8,
                TestRun("test_a", included=["TestB"]): 0.9,
                TestRun("test_a", included=["TestC"]): 1.8,
                TestRun("test_a", excluded=["TestA", "TestB", "TestC"]): 0.7,
                TestRun("test_b"): -0.5,
            },
        )
        test_prioritizations.add_test_score(TestRun("test_a", excluded=["TestD"]), 0.8)
        self.assert_test_scores_almost_equal(
            test_prioritizations._test_scores,
            {
                TestRun("test_a", included=["TestA"]): 1.6,
                TestRun("test_a", included=["TestB"]): 1.7,
                TestRun("test_a", included=["TestC"]): 2.6,
                TestRun("test_a", included=["TestD"]): 0.7,
                TestRun("test_a", excluded=["TestA", "TestB", "TestC", "TestD"]): 1.5,
                TestRun("test_b"): -0.5,
            },
        )
        test_prioritizations.add_test_score(
            TestRun("test_a", excluded=["TestD", "TestC"]), 0.1
        )
        self.assert_test_scores_almost_equal(
            test_prioritizations._test_scores,
            {
                TestRun("test_a", included=["TestA"]): 1.7,
                TestRun("test_a", included=["TestB"]): 1.8,
                TestRun("test_a", included=["TestC"]): 2.6,
                TestRun("test_a", included=["TestD"]): 0.7,
                TestRun("test_a", excluded=["TestA", "TestB", "TestC", "TestD"]): 1.6,
                TestRun("test_b"): -0.5,
            },
        )
        self.assertSetEqual(test_prioritizations._original_tests, set(tests))
        test_prioritizations.validate()


class TestAggregatedHeuristics(TestTD):
    def check(
        self,
        tests: List[str],
        test_prioritizations: List[Dict[TestRun, float]],
        expected: Dict[TestRun, float],
    ) -> None:
        aggregated_heuristics = interface.AggregatedHeuristics(tests)
        for i, test_prioritization in enumerate(test_prioritizations):
            heuristic = self.make_heuristic(f"H{i}")
            aggregated_heuristics.add_heuristic_results(
                heuristic(), interface.TestPrioritizations(tests, test_prioritization)
            )
        final_prioritzations = aggregated_heuristics.get_aggregated_priorities()
        self.assert_test_scores_almost_equal(
            final_prioritzations._test_scores,
            expected,
        )

    def test_get_aggregated_priorities_mix_1(self) -> None:
        tests = ["test_a", "test_b", "test_c"]
        self.check(
            tests,
            [
                {TestRun("test_a"): 0.5},
                {TestRun("test_a::TestA"): 0.25},
                {TestRun("test_c"): 0.8},
            ],
            {
                TestRun("test_a", excluded=["TestA"]): 0.5,
                TestRun("test_a", included=["TestA"]): 0.75,
                TestRun("test_b"): 0.0,
                TestRun("test_c"): 0.8,
            },
        )

    def test_get_aggregated_priorities_mix_2(self) -> None:
        tests = ["test_a", "test_b", "test_c"]
        self.check(
            tests,
            [
                {
                    TestRun("test_a", included=["TestC"]): 0.5,
                    TestRun("test_b"): 0.25,
                    TestRun("test_a", excluded=["TestA", "TestB", "TestC"]): 0.8,
                },
                {
                    TestRun("test_a::TestA"): 0.25,
                    TestRun("test_b::TestB"): 0.5,
                    TestRun("test_a::TestB"): 0.75,
                    TestRun("test_a", excluded=["TestA", "TestB"]): 0.8,
                },
                {TestRun("test_c"): 0.8},
            ],
            {
                TestRun("test_a", included=["TestA"]): 0.25,
                TestRun("test_a", included=["TestB"]): 0.75,
                TestRun("test_a", included=["TestC"]): 1.3,
                TestRun("test_a", excluded=["TestA", "TestB", "TestC"]): 1.6,
                TestRun("test_b", included=["TestB"]): 0.75,
                TestRun("test_b", excluded=["TestB"]): 0.25,
                TestRun("test_c"): 0.8,
            },
        )

    def test_get_aggregated_priorities_mix_3(self) -> None:
        tests = ["test_a"]
        self.check(
            tests,
            [
                {
                    TestRun("test_a", included=["TestA"]): 0.1,
                    TestRun("test_a", included=["TestC"]): 0.1,
                    TestRun("test_a", excluded=["TestA", "TestB", "TestC"]): 0.1,
                },
                {
                    TestRun("test_a", excluded=["TestD"]): 0.1,
                },
                {
                    TestRun("test_a", included=["TestC"]): 0.1,
                },
                {
                    TestRun("test_a", included=["TestB", "TestC"]): 0.1,
                },
                {
                    TestRun("test_a", included=["TestC"]): 0.1,
                    TestRun("test_a", included=["TestD"]): 0.1,
                },
                {
                    TestRun("test_a"): 0.1,
                },
            ],
            {
                TestRun("test_a", included=["TestA"]): 0.3,
                TestRun("test_a", included=["TestB"]): 0.3,
                TestRun("test_a", included=["TestC"]): 0.6,
                TestRun("test_a", included=["TestD"]): 0.3,
                TestRun("test_a", excluded=["TestA", "TestB", "TestC", "TestD"]): 0.3,
            },
        )


class TestAggregatedHeuristicsTestStats(TestTD):
    def test_get_test_stats_with_whole_tests(self) -> None:
        self.maxDiff = None
        tests = ["test1", "test2", "test3", "test4", "test5"]
        heuristic1 = interface.TestPrioritizations(
            tests,
            {
                TestRun("test3"): 0.3,
                TestRun("test4"): 0.1,
            },
        )
        heuristic2 = interface.TestPrioritizations(
            tests,
            {
                TestRun("test5"): 0.5,
            },
        )

        aggregator = interface.AggregatedHeuristics(tests)
        aggregator.add_heuristic_results(self.make_heuristic("H1")(), heuristic1)
        aggregator.add_heuristic_results(self.make_heuristic("H2")(), heuristic2)

        expected_test3_stats = {
            "test_name": "test3",
            "test_filters": "",
            "heuristics": [
                {
                    "position": 0,
                    "score": 0.3,
                    "heuristic_name": "H1",
                    "trial_mode": False,
                },
                {
                    "position": 3,
                    "score": 0.0,
                    "heuristic_name": "H2",
                    "trial_mode": False,
                },
            ],
            "aggregated": {"position": 1, "score": 0.3},
            "aggregated_trial": {"position": 1, "score": 0.3},
        }

        test3_stats = aggregator.get_test_stats(TestRun("test3"))

        self.assertDictEqual(test3_stats, expected_test3_stats)

    def test_get_test_stats_only_contains_allowed_types(self) -> None:
        self.maxDiff = None
        tests = ["test1", "test2", "test3", "test4", "test5"]
        heuristic1 = interface.TestPrioritizations(
            tests,
            {
                TestRun("test3"): 0.3,
                TestRun("test4"): 0.1,
            },
        )
        heuristic2 = interface.TestPrioritizations(
            tests,
            {
                TestRun("test5::classA"): 0.5,
            },
        )

        aggregator = interface.AggregatedHeuristics(tests)
        aggregator.add_heuristic_results(self.make_heuristic("H1")(), heuristic1)
        aggregator.add_heuristic_results(self.make_heuristic("H2")(), heuristic2)

        stats3 = aggregator.get_test_stats(TestRun("test3"))
        stats5 = aggregator.get_test_stats(TestRun("test5::classA"))

        def assert_valid_dict(dict_contents: Dict[str, Any]) -> None:
            for key, value in dict_contents.items():
                self.assertTrue(isinstance(key, str))
                self.assertTrue(
                    isinstance(value, (str, float, int, list, dict)),
                    f"{value} is not a str, float, or dict",
                )
                if isinstance(value, dict):
                    assert_valid_dict(value)
                elif isinstance(value, list):
                    for item in value:
                        assert_valid_dict(item)

        assert_valid_dict(stats3)
        assert_valid_dict(stats5)

    def test_get_test_stats_gets_rank_for_test_classes(self) -> None:
        self.maxDiff = None
        tests = ["test1", "test2", "test3", "test4", "test5"]
        heuristic1 = interface.TestPrioritizations(
            tests,
            {
                TestRun("test3"): 0.3,
                TestRun("test4"): 0.1,
            },
        )
        heuristic2 = interface.TestPrioritizations(
            tests,
            {
                TestRun("test5::classA"): 0.5,
            },
        )

        aggregator = interface.AggregatedHeuristics(tests)
        aggregator.add_heuristic_results(self.make_heuristic("H1")(), heuristic1)
        aggregator.add_heuristic_results(self.make_heuristic("H2")(), heuristic2)

        stats_inclusive = aggregator.get_test_stats(
            TestRun("test5", included=["classA"])
        )
        stats_exclusive = aggregator.get_test_stats(
            TestRun("test5", excluded=["classA"])
        )
        expected_inclusive = {
            "test_name": "test5",
            "test_filters": "classA",
            "heuristics": [
                {
                    "position": 4,
                    "score": 0.0,
                    "heuristic_name": "H1",
                    "trial_mode": False,
                },
                {
                    "position": 0,
                    "score": 0.5,
                    "heuristic_name": "H2",
                    "trial_mode": False,
                },
            ],
            "aggregated": {"position": 0, "score": 0.5},
            "aggregated_trial": {"position": 0, "score": 0.5},
        }
        expected_exclusive = {
            "test_name": "test5",
            "test_filters": "not (classA)",
            "heuristics": [
                {
                    "position": 4,
                    "score": 0.0,
                    "heuristic_name": "H1",
                    "trial_mode": False,
                },
                {
                    "position": 5,
                    "score": 0.0,
                    "heuristic_name": "H2",
                    "trial_mode": False,
                },
            ],
            "aggregated": {"position": 5, "score": 0.0},
            "aggregated_trial": {"position": 5, "score": 0.0},
        }

        self.assertDictEqual(stats_inclusive, expected_inclusive)
        self.assertDictEqual(stats_exclusive, expected_exclusive)

    def test_get_test_stats_works_with_class_granularity_heuristics(self) -> None:
        tests = ["test1", "test2", "test3", "test4", "test5"]
        heuristic1 = interface.TestPrioritizations(
            tests,
            {
                TestRun("test2"): 0.3,
            },
        )
        heuristic2 = interface.TestPrioritizations(
            tests,
            {
                TestRun("test2::TestFooClass"): 0.5,
            },
        )

        aggregator = interface.AggregatedHeuristics(tests)
        aggregator.add_heuristic_results(self.make_heuristic("H1")(), heuristic1)
        aggregator.add_heuristic_results(self.make_heuristic("H2")(), heuristic2)

        # These should not throw an error
        aggregator.get_test_stats(TestRun("test2::TestFooClass"))
        aggregator.get_test_stats(TestRun("test2"))


class TestJsonParsing(TestTD):
    def test_json_parsing_matches_TestPrioritizations(self) -> None:
        tests = ["test1", "test2", "test3", "test4", "test5"]
        tp = interface.TestPrioritizations(
            tests,
            {
                TestRun("test3", included=["ClassA"]): 0.8,
                TestRun("test3", excluded=["ClassA"]): 0.2,
                TestRun("test4"): 0.7,
                TestRun("test5"): 0.6,
            },
        )
        tp_json = tp.to_json()
        tp_json_to_tp = interface.TestPrioritizations.from_json(tp_json)

        self.assertSetEqual(tp._original_tests, tp_json_to_tp._original_tests)
        self.assertDictEqual(tp._test_scores, tp_json_to_tp._test_scores)

    def test_json_parsing_matches_TestRun(self) -> None:
        testrun = TestRun("test1", included=["classA", "classB"])
        testrun_json = testrun.to_json()
        testrun_json_to_test = TestRun.from_json(testrun_json)

        self.assertTrue(testrun == testrun_json_to_test)


if __name__ == "__main__":
    unittest.main()
