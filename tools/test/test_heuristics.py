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

    from tools.testing.target_determination.determinator import (
        get_test_prioritizations,
        TestPrioritizations,
    )
    from tools.testing.target_determination.heuristics.previously_failed_in_pr import (
        _get_previously_failing_tests,
    )

except ModuleNotFoundError:
    print("Can't import required modules, exiting")
    sys.exit(1)


def mocked_file(contents: Dict[Any, Any]) -> io.IOBase:
    file_object = io.StringIO()
    json.dump(contents, file_object)
    file_object.seek(0)
    return file_object


class TestParsePrevTests(unittest.TestCase):
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

        self.assertTupleEqual(
            expected_prioritizations.get_high_relevance_tests(),
            test_prioritizations.get_high_relevance_tests(),
        )
        self.assertTupleEqual(
            expected_prioritizations.get_probable_relevance_tests(),
            test_prioritizations.get_probable_relevance_tests(),
        )
        self.assertTupleEqual(
            expected_prioritizations.get_unranked_relevance_tests(),
            test_prioritizations.get_unranked_relevance_tests(),
        )


if __name__ == "__main__":
    unittest.main()
