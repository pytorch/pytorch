import subprocess
from typing import List, Dict, cast
from warnings import warn
import os
import json
from collections import defaultdict

from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
)

from tools.testing.target_determination.heuristics.utils import (
    query_changed_files,
    REPO_ROOT,
)
from tools.stats.import_test_stats import TEST_FILE_RATINGS_FILE


class CorrelatedWithHistoricalFailures(HeuristicInterface):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_test_priorities(self, tests: List[str]) -> TestPrioritizations:
        correlated_tests = _get_file_rating_tests()
        test_rankings = TestPrioritizations()
        test_rankings.probably_relevant = correlated_tests

        return test_rankings


def _get_file_rating_tests() -> List[str]:
    path = REPO_ROOT / "test" / TEST_FILE_RATINGS_FILE
    if not os.path.exists(path):
        print(f"could not find path {path}")
        return []
    with open(path) as f:
        test_file_ratings = cast(Dict[str, Dict[str, float]], json.load(f))
    try:
        changed_files = query_changed_files()
    except Exception as e:
        warn(f"Can't query changed test files due to {e}")
        return []
    ratings: Dict[str, float] = defaultdict(float)
    for file in changed_files:
        for test_file, score in test_file_ratings.get(file, {}).items():
            ratings[test_file] += score
    prioritize = sorted(ratings, key=lambda x: ratings[x])
    return prioritize
