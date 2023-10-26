import io
import json
import pathlib
import sys
import unittest
from typing import Any, Dict, Set
from unittest import mock

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
try:
    # using tools/ to optimize test run.
    sys.path.append(str(REPO_ROOT))

    from tools.test.heuristics.heuristics_test_mixin import HeuristicsTestMixin
    from tools.testing.target_determination.determinator import (
        AggregatedHeuristics,
        get_test_prioritizations,
        TestPrioritizations,
    )
    from tools.testing.target_determination.heuristics import HEURISTICS
    from tools.testing.target_determination.heuristics.previously_failed_in_pr import (
        _get_previously_failing_tests,
    )
    from tools.testing.test_run import TestRun

except ModuleNotFoundError:
    print("Can't import required modules, exiting")
    sys.exit(1)


def mocked_file(contents: Dict[Any, Any]) -> io.IOBase:
    file_object = io.StringIO()
    json.dump(contents, file_object)
    file_object.seek(0)
    return file_object


class TestParsePrevTests(HeuristicsTestMixin):
    @mock.patch("pathlib.Path.exists", return_value=False)
    def test_cache_does_not_exist(self, mock_exists: Any) -> None:
        expected_failing_test_files: Set[str] = set()

        found_tests = _get_previously_failing_tests()

        self.assertSetEqual(expected_failing_test_files, found_tests)

    @mock.patch("pathlib.Path.exists", return_value=True)
    @mock.patch("builtins.open", return_value=mocked_file({"": True}))
    def test_empty_cache(self, mock_exists: Any, mock_open: Any) -> None:
        expected_failing_test_files: Set[str] = set()

        found_tests = _get_previously_failing_tests()

        self.assertSetEqual(expected_failing_test_files, found_tests)
        mock_open.assert_called()

    lastfailed_with_multiple_tests_per_file = {
        "test/test_car.py::TestCar::test_num[17]": True,
        "test/test_car.py::TestBar::test_num[25]": True,
        "test/test_far.py::TestFar::test_fun_copy[17]": True,
        "test/test_bar.py::TestBar::test_fun_copy[25]": True,
    }

    @mock.patch("pathlib.Path.exists", return_value=True)
    @mock.patch(
        "builtins.open",
        return_value=mocked_file(lastfailed_with_multiple_tests_per_file),
    )
    def test_dedupes_failing_test_files(self, mock_exists: Any, mock_open: Any) -> None:
        expected_failing_test_files = {"test_car", "test_bar", "test_far"}
        found_tests = _get_previously_failing_tests()

        self.assertSetEqual(expected_failing_test_files, found_tests)

    @mock.patch(
        "tools.testing.target_determination.heuristics.previously_failed_in_pr._get_previously_failing_tests",
        return_value={"test4"},
    )
    @mock.patch(
        "tools.testing.target_determination.heuristics.edited_by_pr._get_modified_tests",
        return_value={"test2", "test4"},
    )
    @mock.patch(
        "tools.testing.target_determination.heuristics.correlated_with_historical_failures._get_file_rating_tests",
        return_value=["test1"],
    )
    def test_get_reordered_tests(self, *args: Any) -> None:
        tests = ["test1", "test2", "test3", "test4", "test5"]

        expected_prioritizations = TestPrioritizations(
            tests_being_ranked=tests,
            high_relevance=["test4", "test2"],
            probable_relevance=["test1"],
            unranked_relevance=["test3", "test5"],
        )

        test_prioritizations = get_test_prioritizations(
            tests
        ).get_aggregated_priorities()

        self.assertHeuristicsMatch(
            test_prioritizations, expected_prioritizations=expected_prioritizations
        )


class TestInterface(HeuristicsTestMixin):
    def test_class_prioritization(self) -> None:
        tests = ["test1", "test2", "test3", "test4", "test5"]

        prioritizations = TestPrioritizations(
            tests_being_ranked=tests,
            probable_relevance=["test2::TestFooClass", "test3"],
        )

        expected_probable_tests = tuple(
            TestRun(test) for test in ["test2::TestFooClass", "test3"]
        )
        expected_unranked_tests = (
            TestRun("test1"),
            TestRun("test2", excluded=["TestFooClass"]),
            TestRun("test4"),
            TestRun("test5"),
        )

        self.assertHeuristicsMatch(
            prioritizations,
            expected_probable_tests=expected_probable_tests,
            expected_unranked_tests=expected_unranked_tests,
        )


class TestAggregatedHeuristics(HeuristicsTestMixin):
    def test_merging_multiple_test_class_heuristics(self) -> None:
        tests = ["test1", "test2", "test3", "test4"]

        print("-------------------")
        print("Gen Heuristics 1")
        heuristic1 = TestPrioritizations(
            tests_being_ranked=tests,
            probable_relevance=["test2::TestFooClass", "test3"],
        )

        print("-------------------")
        print("Gen Heuristics 2")
        heuristic2 = TestPrioritizations(
            tests_being_ranked=tests,
            high_relevance=["test2::TestFooClass", "test3::TestBarClass"],
        )

        expected_high_relevance = tuple(
            TestRun(test) for test in ["test2::TestFooClass", "test3::TestBarClass"]
        )
        expected_probable_relevance = (TestRun("test3", excluded=["TestBarClass"]),)
        expected_unranked_relevance = (
            TestRun("test1"),
            TestRun("test2", excluded=["TestFooClass"]),
            TestRun("test4"),
        )

        aggregator = AggregatedHeuristics(unranked_tests=tests)
        aggregator.add_heuristic_results(HEURISTICS[0], heuristic1)
        aggregator.add_heuristic_results(HEURISTICS[1], heuristic2)

        print("-------------------")
        print("Aggregated Heuristics")
        aggregated_pris = aggregator.get_aggregated_priorities()

        self.assertHeuristicsMatch(
            aggregated_pris,
            expected_high_tests=expected_high_relevance,
            expected_probable_tests=expected_probable_relevance,
            expected_unranked_tests=expected_unranked_relevance,
        )

    def test_merging_file_heuristic_after_class_heuristic_with_different_probabilities(self) -> None:
        tests = ["test1", "test2", "test3", "test4", "test5"]
        heuristic1 = TestPrioritizations(
            tests_being_ranked=tests,
            high_relevance=["test2::TestFooClass"],
        )
        heuristic2 = TestPrioritizations(
            tests_being_ranked=tests,
            probable_relevance=["test2", "test3"],
        )

        expected_aggregated_high_relevance = tuple(
            TestRun(test) for test in ["test2::TestFooClass"]
        )
        expected_aggregated_probable_relevance = (
            TestRun("test2", excluded=["TestFooClass"]),
            TestRun("test3"),
        )
        expected_aggregated_unranked_relevance = (
            TestRun("test1"),
            TestRun("test4"),
            TestRun("test5"),
        )

        aggregator = AggregatedHeuristics(unranked_tests=tests)
        aggregator.add_heuristic_results(HEURISTICS[0], heuristic1)
        aggregator.add_heuristic_results(HEURISTICS[1], heuristic2)

        aggregated_pris = aggregator.get_aggregated_priorities()

        self.assertHeuristicsMatch(
            aggregated_pris,
            expected_high_tests=expected_aggregated_high_relevance,
            expected_probable_tests=expected_aggregated_probable_relevance,
            expected_unranked_tests=expected_aggregated_unranked_relevance,
        )

    def test_merging_file_heuristic_after_class_heuristic_with_same_probability(self) -> None:
        tests = ["test1", "test2", "test3", "test4", "test5"]
        heuristic1 = TestPrioritizations(
            tests_being_ranked=tests,
            probable_relevance=["test2::TestFooClass"],
        )
        heuristic2 = TestPrioritizations(
            tests_being_ranked=tests,
            probable_relevance=["test3", "test2"],
        )

        expected_aggregated_high_relevance = tuple(
        )
        expected_aggregated_probable_relevance = (
            TestRun("test2::TestFooClass"),
            TestRun("test3"),
            TestRun("test2", excluded=["TestFooClass"]),
        )
        expected_aggregated_unranked_relevance = (
            TestRun("test1"),
            TestRun("test4"),
            TestRun("test5"),
        )

        aggregator = AggregatedHeuristics(unranked_tests=tests)
        aggregator.add_heuristic_results(HEURISTICS[0], heuristic1)
        aggregator.add_heuristic_results(HEURISTICS[1], heuristic2)

        aggregated_pris = aggregator.get_aggregated_priorities()

        self.assertHeuristicsMatch(
            aggregated_pris,
            expected_high_tests=expected_aggregated_high_relevance,
            expected_probable_tests=expected_aggregated_probable_relevance,
            expected_unranked_tests=expected_aggregated_unranked_relevance,
        )


if __name__ == "__main__":
    unittest.main()
