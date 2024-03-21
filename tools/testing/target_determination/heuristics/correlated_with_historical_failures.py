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
    get_ratings_for_tests,
    normalize_ratings,
)
from tools.testing.test_run import TestRun


class CorrelatedWithHistoricalFailures(HeuristicInterface):
    def __init__(self, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)

    def get_prediction_confidence(self, tests: List[str]) -> TestPrioritizations:
        test_ratings = get_ratings_for_tests(
            ADDITIONAL_CI_FILES_FOLDER / TEST_FILE_RATINGS_FILE
        )
        test_ratings = {TestRun(k): v for (k, v) in test_ratings.items() if k in tests}
        return TestPrioritizations(tests, normalize_ratings(test_ratings, 0.25))
