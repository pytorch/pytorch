# For testing specific heuristics
import io
import json
import pathlib
import sys
import unittest
from typing import Any, Dict, List, Set
from unittest import mock

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(REPO_ROOT))
from tools.test.heuristics.test_interface import TestTD
from tools.testing.target_determination.determinator import TestPrioritizations
from tools.testing.target_determination.heuristics.historical_class_failure_correlation import (
    HistoricalClassFailurCorrelation,
)
from tools.testing.target_determination.heuristics.previously_failed_in_pr import (
    get_previous_failures,
)
from tools.testing.test_run import TestRun

sys.path.remove(str(REPO_ROOT))

HEURISTIC_CLASS = "tools.testing.target_determination.heuristics.historical_class_failure_correlation."


def mocked_file(contents: Dict[Any, Any]) -> io.IOBase:
    file_object = io.StringIO()
    json.dump(contents, file_object)
    file_object.seek(0)
    return file_object


def gen_historical_class_failures() -> Dict[str, Dict[str, float]]:
    return {
        "file1": {
            "test1::classA": 0.5,
            "test2::classA": 0.2,
            "test5::classB": 0.1,
        },
        "file2": {
            "test1::classB": 0.3,
            "test3::classA": 0.2,
            "test5::classA": 1.5,
            "test7::classC": 0.1,
        },
        "file3": {
            "test1::classC": 0.4,
            "test4::classA": 0.2,
            "test7::classC": 1.5,
            "test8::classC": 0.1,
        },
    }


ALL_TESTS = [
    "test1",
    "test2",
    "test3",
    "test4",
    "test5",
    "test6",
    "test7",
    "test8",
]


class TestHistoricalClassFailureCorrelation(TestTD):
    @mock.patch(
        HEURISTIC_CLASS + "_get_historical_test_class_correlations",
        return_value=gen_historical_class_failures(),
    )
    @mock.patch(
        HEURISTIC_CLASS + "query_changed_files",
        return_value=["file1"],
    )
    def test_get_prediction_confidence(
        self,
        historical_class_failures: Dict[str, Dict[str, float]],
        changed_files: List[str],
    ) -> None:
        tests_to_prioritize = ALL_TESTS

        heuristic = HistoricalClassFailurCorrelation()
        test_prioritizations = heuristic.get_prediction_confidence(tests_to_prioritize)

        expected = TestPrioritizations(
            tests_to_prioritize,
            {
                TestRun("test1::classA"): 0.25,
                TestRun("test2::classA"): 0.1,
                TestRun("test5::classB"): 0.05,
                TestRun("test1", excluded=["classA"]): 0.0,
                TestRun("test2", excluded=["classA"]): 0.0,
                TestRun("test3"): 0.0,
                TestRun("test4"): 0.0,
                TestRun("test5", excluded=["classB"]): 0.0,
                TestRun("test6"): 0.0,
                TestRun("test7"): 0.0,
                TestRun("test8"): 0.0,
            },
        )

        self.assert_test_scores_almost_equal(
            test_prioritizations._test_scores, expected._test_scores
        )


class TestParsePrevTests(TestTD):
    @mock.patch("os.path.exists", return_value=False)
    def test_cache_does_not_exist(self, mock_exists: Any) -> None:
        expected_failing_test_files: Set[str] = set()

        found_tests = get_previous_failures()

        self.assertSetEqual(expected_failing_test_files, found_tests)

    @mock.patch("os.path.exists", return_value=True)
    @mock.patch("builtins.open", return_value=mocked_file({"": True}))
    def test_empty_cache(self, mock_exists: Any, mock_open: Any) -> None:
        expected_failing_test_files: Set[str] = set()

        found_tests = get_previous_failures()

        self.assertSetEqual(expected_failing_test_files, found_tests)
        mock_open.assert_called()

    lastfailed_with_multiple_tests_per_file = {
        "test/test_car.py::TestCar::test_num[17]": True,
        "test/test_car.py::TestBar::test_num[25]": True,
        "test/test_far.py::TestFar::test_fun_copy[17]": True,
        "test/test_bar.py::TestBar::test_fun_copy[25]": True,
    }

    @mock.patch("os.path.exists", return_value=True)
    @mock.patch(
        "builtins.open",
        return_value=mocked_file(lastfailed_with_multiple_tests_per_file),
    )
    def test_dedupes_failing_test_files(self, mock_exists: Any, mock_open: Any) -> None:
        expected_failing_test_files = {"test_car", "test_bar", "test_far"}
        found_tests = get_previous_failures()

        self.assertSetEqual(expected_failing_test_files, found_tests)


if __name__ == "__main__":
    unittest.main()
