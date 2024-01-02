from typing import Any, Dict, List

from tools.stats.import_test_stats import (
    ADDITIONAL_CI_FILES_FOLDER,
    TEST_FILE_RATINGS_FILE,
)

from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
)

from tools.testing.target_determination.heuristics.utils import (
    get_correlated_tests,
    get_ratings_for_tests,
    normalize_ratings,
)


class CorrelatedWithHistoricalFailures(HeuristicInterface):
    def __init__(self, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)

    def get_test_priorities(self, tests: List[str]) -> TestPrioritizations:
        correlated_tests = get_correlated_tests(
            ADDITIONAL_CI_FILES_FOLDER / TEST_FILE_RATINGS_FILE
        )
        relevant_correlated_tests = [test for test in correlated_tests if test in tests]
        test_rankings = TestPrioritizations(
            tests_being_ranked=tests, probable_relevance=relevant_correlated_tests
        )

        return test_rankings

    def get_prediction_confidence(self, tests: List[str]) -> Dict[str, float]:
        test_ratings = get_ratings_for_tests(
            ADDITIONAL_CI_FILES_FOLDER / TEST_FILE_RATINGS_FILE
        )
        test_ratings = {k: v for (k, v) in test_ratings.items() if k in tests}
        return normalize_ratings(test_ratings, 1)
