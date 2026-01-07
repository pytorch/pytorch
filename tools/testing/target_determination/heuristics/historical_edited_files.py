from __future__ import annotations

from typing import Any

from tools.stats.import_test_stats import (
    ADDITIONAL_CI_FILES_FOLDER,
    TD_HEURISTIC_HISTORICAL_EDITED_FILES,
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


# This heuristic assumes that changed files in previous commits are good sources
# of information for what files are related to each other. If fileA and
# testFileA were edited in the same commit on main, that probably means that
# future commits that change fileA should probably run testFileA. Based on this,
# a correlation dict is built based on what files were edited in commits on main.
class HistorialEditedFiles(HeuristicInterface):
    def __init__(self, **kwargs: Any) -> None:
        # pyrefly: ignore [missing-attribute]
        super().__init__(**kwargs)

    def get_prediction_confidence(self, tests: list[str]) -> TestPrioritizations:
        test_ratings = get_ratings_for_tests(
            ADDITIONAL_CI_FILES_FOLDER / TD_HEURISTIC_HISTORICAL_EDITED_FILES
        )
        test_ratings = {TestRun(k): v for (k, v) in test_ratings.items() if k in tests}

        return TestPrioritizations(tests, normalize_ratings(test_ratings, 0.25))
