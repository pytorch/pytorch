import json
from pathlib import Path
from typing import Any, Dict, List, Set
from warnings import warn

from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
)
from tools.testing.target_determination.heuristics.utils import (
    python_test_file_to_test_name,
)


class PreviouslyFailedInPR(HeuristicInterface):
    def __init__(self, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)

    def get_test_priorities(self, tests: List[str]) -> TestPrioritizations:
        # Tests must always be returned in a deterministic order.
        # Otherwise it breaks our test sharding logic
        critical_tests = sorted(_get_previously_failing_tests())
        test_rankings = TestPrioritizations(highly_relevant = critical_tests)

        return test_rankings


def _get_previously_failing_tests() -> Set[str]:
    PYTEST_FAILED_TESTS_CACHE_FILE_PATH = Path(".pytest_cache/v/cache/lastfailed")

    if not PYTEST_FAILED_TESTS_CACHE_FILE_PATH.exists():
        warn(
            f"No pytorch cache found at {PYTEST_FAILED_TESTS_CACHE_FILE_PATH.absolute()}"
        )
        return set()

    with open(PYTEST_FAILED_TESTS_CACHE_FILE_PATH) as f:
        last_failed_tests = json.load(f)

    prioritized_tests = _parse_prev_failing_test_files(last_failed_tests)

    return python_test_file_to_test_name(prioritized_tests)


def _parse_prev_failing_test_files(last_failed_tests: Dict[str, bool]) -> Set[str]:
    prioritized_tests = set()

    # The keys are formatted as "test_file.py::test_class::test_method[params]"
    # We just need the test_file part
    for test in last_failed_tests:
        parts = test.split("::")
        if len(parts) > 1:
            test_file = parts[0]
            prioritized_tests.add(test_file)

    return prioritized_tests
