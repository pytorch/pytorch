from typing import Any, Dict, List, Set
from warnings import warn

from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
)
from tools.testing.target_determination.heuristics.utils import (
    python_test_file_to_test_name,
    query_changed_files,
)


class EditedByPR(HeuristicInterface):
    def __init__(self, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)

    def get_test_priorities(self, tests: List[str]) -> TestPrioritizations:
        # Tests must always be returned in a deterministic order.
        # Otherwise it breaks our test sharding logic
        critical_tests = sorted(_get_modified_tests())
        test_rankings = TestPrioritizations(highly_relevant=critical_tests)

        return test_rankings


def _get_modified_tests() -> Set[str]:
    try:
        changed_files = query_changed_files()
    except Exception as e:
        warn(f"Can't query changed test files due to {e}")
        # If unable to get changed files from git, quit without doing any sorting
        return set()

    return python_test_file_to_test_name(set(changed_files))
