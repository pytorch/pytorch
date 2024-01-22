from collections import defaultdict
import pathlib
from typing import Dict, List, Union
import unittest
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(REPO_ROOT))
from tools.testing.test_run import TestRun
import tools.testing.target_determination.heuristics.interface as interface


class Helpers:
    def make_heuristic(name):
        class Heuristic(interface.HeuristicInterface):
            def get_test_priorities(self, tests):
                # Return junk
                return []

        return type(name, (Heuristic,), {})

    def add_test_prioritization(aggregated_heuristics, test_prioritizations):
        for i, test_prioritization in enumerate(test_prioritizations):
            heuristic = Helpers.make_heuristic(f"H{i}")
            aggregated_heuristics.add_heuristic_results(
                heuristic(), test_prioritization
            )

    def assert_prioritizations_equal(
        test,
        p1: interface.TestPrioritizations,
        p2: Dict[interface.Relevance, List[Union[str, TestRun]]],
    ):
        formatted_p2 = defaultdict(list)
        for k, v in p2.items():
            formatted_p2[k] = [TestRun(x) if isinstance(x, str) else x for x in v]

        def check(tests: List[TestRun], expected: List[TestRun]):
            test.assertEqual(len(tests), len(expected))
            for t, e in zip(tests, expected):
                test.assertEqual(t, e)

        check(p1.get_high_relevance_tests(), formatted_p2[interface.Relevance.HIGH])
        check(
            p1.get_probable_relevance_tests(),
            formatted_p2[interface.Relevance.PROBABLE],
        )
        check(
            p1.get_unranked_relevance_tests(),
            formatted_p2[interface.Relevance.UNRANKED],
        )
        check(
            p1.get_unlikely_relevance_tests(),
            formatted_p2[interface.Relevance.UNLIKELY],
        )
        check(p1.get_none_relevance_tests(), formatted_p2[interface.Relevance.NONE])

    def check(test, tests, test_prioritizations, expected):
        aggregated_heuristics = interface.AggregatedHeuristics(tests)
        Helpers.add_test_prioritization(aggregated_heuristics, test_prioritizations)
        final_prioritzations = aggregated_heuristics.get_aggregated_priorities()
        print(final_prioritzations.get_info_str())
        Helpers.assert_prioritizations_equal(
            test,
            final_prioritzations,
            expected,
        )


class TestAggregatedHeuristics(unittest.TestCase):
    def test_get_aggregated_priorities_1(self):
        tests = ["test_a", "test_b", "test_c"]
        Helpers.check(
            self,
            tests,
            [
                interface.TestPrioritizations(tests, high_relevance=["test_a"]),
                interface.TestPrioritizations(tests, high_relevance=["test_a::TestA"]),
                interface.TestPrioritizations(tests, high_relevance=["test_c"]),
            ],
            {
                interface.Relevance.HIGH: [
                    "test_a::TestA",
                    TestRun("test_a", excluded=["TestA"]),
                    "test_c",
                ],
                interface.Relevance.UNRANKED: ["test_b"],
            },
        )

    def test_get_aggregated_priorities_2(self):
        tests = ["test_a"]
        Helpers.check(
            self,
            tests,
            [
                interface.TestPrioritizations(tests, high_relevance=["test_a"]),
                interface.TestPrioritizations(tests, high_relevance=["test_a::TestA"]),
                interface.TestPrioritizations(tests, high_relevance=["test_a::TestB"]),
            ],
            {
                interface.Relevance.HIGH: [
                    "test_a::TestA",
                    "test_a::TestB",
                    TestRun("test_a", excluded=["TestA", "TestB"]),
                ],
            },
        )

    def test_get_aggregated_priorities_3(self):
        tests = ["test_a"]
        Helpers.check(
            self,
            tests,
            [
                interface.TestPrioritizations(tests, probable_relevance=["test_a"]),
                interface.TestPrioritizations(
                    tests, probable_relevance=["test_a::TestA"]
                ),
                interface.TestPrioritizations(
                    tests, probable_relevance=["test_a::TestA"]
                ),
                interface.TestPrioritizations(
                    tests, probable_relevance=["test_a::TestB"]
                ),
            ],
            {
                interface.Relevance.HIGH: [
                    "test_a::TestA",
                ],
                interface.Relevance.PROBABLE: [
                    "test_a::TestB",
                    TestRun("test_a", excluded=["TestA", "TestB"]),
                ],
            },
        )
    def test_get_aggregated_priorities_4(self):
        tests = ["test_a", "test_c"]
        Helpers.check(
            self,
            tests,
            [
                interface.TestPrioritizations(tests, high_relevance=["test_c"]),
                interface.TestPrioritizations(tests, probable_relevance=["test_a"]),
                interface.TestPrioritizations(
                    tests, probable_relevance=["test_a::TestA"]
                ),
                interface.TestPrioritizations(
                    tests, probable_relevance=["test_a::TestA"]
                ),
                interface.TestPrioritizations(
                    tests, probable_relevance=["test_a::TestB"]
                ),
            ],
            {
                interface.Relevance.HIGH: [
                    "test_c",
                    "test_a::TestA",
                ],
                interface.Relevance.PROBABLE: [
                    "test_a::TestB",
                    TestRun("test_a", excluded=["TestA", "TestB"]),
                ],
            },
        )


    def test_get_aggregated_priorities_5(self):
        tests = ["test_a", "test_c"]
        Helpers.check(
            self,
            tests,
            [
                interface.TestPrioritizations(tests, probable_relevance=["test_a"]),
                interface.TestPrioritizations(
                    tests, probable_relevance=["test_a::TestA"]
                ),
                interface.TestPrioritizations(
                    tests, probable_relevance=["test_a::TestA"]
                ),
                interface.TestPrioritizations(
                    tests, probable_relevance=["test_a::TestB"]
                ),
            ],
            {
                interface.Relevance.HIGH: [
                    "test_a::TestA",
                ],
                interface.Relevance.PROBABLE: [
                    "test_a::TestB",
                    TestRun("test_a", excluded=["TestA", "TestB"]),
                ],
                interface.Relevance.UNRANKED: [
                    "test_c",
                ]
            },
        )



if __name__ == "__main__":
    unittest.main()
