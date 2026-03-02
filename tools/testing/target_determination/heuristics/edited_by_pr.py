from __future__ import annotations

import re
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


# Some files run tests in other test files, so we map them to each other here.
# This is a map from file that runs the test to regex that matches the file that
# contains the test. Test file with path test/a/b.py should of the form a/b.
# Regexes should be based on repo root.
ADDITIONAL_MAPPINGS = {
    # Not files that are tracked by git but rather functions defined in
    # run_test.py that generate test files which run tests in test/cpp_extensions.
    "test_cpp_extensions_aot_ninja": [r"test\/cpp_extensions.*"],
    "test_cpp_extensions_aot_no_ninja": [r"test\/cpp_extensions.*"],
}


class EditedByPR(HeuristicInterface):
    def __init__(self, **kwargs: dict[str, Any]) -> None:
        # pyrefly: ignore [missing-attribute]
        super().__init__(**kwargs)

    def get_prediction_confidence(self, tests: list[str]) -> TestPrioritizations:
        critical_tests = _get_modified_tests()
        return TestPrioritizations(
            tests, {TestRun(test): 1 for test in critical_tests if test in tests}
        )


def _get_modified_tests() -> set[str]:
    try:
        changed_files = query_changed_files()
        should_run = python_test_file_to_test_name(set(changed_files))
        for test_file, regexes in ADDITIONAL_MAPPINGS.items():
            if any(
                re.search(regex, changed_file) is not None
                for regex in regexes
                for changed_file in changed_files
            ):
                should_run.add(test_file)
        return should_run
    except Exception as e:
        warn(f"Can't query changed test files due to {e}")
        # If unable to get changed files from git, quit without doing any sorting
    return set()
