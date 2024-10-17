from __future__ import annotations

from typing import Any
from warnings import warn

from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
)
from tools.testing.target_determination.heuristics.utils import (
    python_test_file_to_test_name,
    query_changed_files,
)
from tools.testing.test_run import TestRun


class EditedByPR(HeuristicInterface):
    def __init__(self, **kwargs: dict[str, Any]) -> None:
        super().__init__(**kwargs)

    def get_prediction_confidence(self, tests: list[str]) -> TestPrioritizations:
        critical_tests = _get_modified_tests()
        return TestPrioritizations(
            tests, {TestRun(test): 1 for test in critical_tests if test in tests}
        )


def _get_modified_tests() -> set[str]:
    try:
        changed_files = query_changed_files()
    except Exception as e:
        warn(f"Can't query changed test files due to {e}")
        # If unable to get changed files from git, quit without doing any sorting
        return set()

    return python_test_file_to_test_name(set(changed_files))
