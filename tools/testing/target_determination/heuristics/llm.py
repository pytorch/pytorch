import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from tools.stats.import_test_stats import ADDITIONAL_CI_FILES_FOLDER
from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
)
from tools.testing.target_determination.heuristics.utils import normalize_ratings
from tools.testing.test_run import TestRun


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent


class LLM(HeuristicInterface):
    def __init__(self, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)

    def get_prediction_confidence(self, tests: List[str]) -> TestPrioritizations:
        critical_tests = self.get_mappings()
        filter_valid_tests = {
            TestRun(test): score
            for test, score in critical_tests.items()
            if test in tests
        }
        normalized_scores = normalize_ratings(filter_valid_tests, 0.25)
        return TestPrioritizations(tests, normalized_scores)

    def get_mappings(self) -> Dict[str, float]:
        path = (
            REPO_ROOT
            / ADDITIONAL_CI_FILES_FOLDER
            / "llm_results/mappings/indexer-files-gitdiff-output.json"
        )
        if not os.path.exists(path):
            print(f"could not find path {path}")
            return {}
        with open(path) as f:
            # Group by file
            r = defaultdict(list)
            for key, value in json.load(f).items():
                re_match = re.match("(.*).py", key)
                if re_match:
                    file = re_match.group(1)
                    r[file].append(value)
            # Average the scores for each file
            r = {file: sum(scores) / len(scores) for file, scores in r.items()}
            return r
