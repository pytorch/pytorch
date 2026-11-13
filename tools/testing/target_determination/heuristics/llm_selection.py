from __future__ import annotations

import json
import math
from typing import Any

from tools.stats.import_test_stats import ADDITIONAL_CI_FILES_FOLDER
from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
)
from tools.testing.target_determination.heuristics.utils import REPO_ROOT
from tools.testing.test_run import TestRun


LLM_SELECTION_FILE = "llm_selection.json"
LLM_SELECTION_PATH = REPO_ROOT / ADDITIONAL_CI_FILES_FOLDER / LLM_SELECTION_FILE
LLM_SELECTION_VERSION = 1
# At most 0.25 in TD's sum, as the other heuristics that rank. The TD scorer
# calibrates the score, so there only its order matters; changing the weight
# would mix two scales in the scores it has logged.
LLM_SELECTION_WEIGHT = 0.25


class LLMSelection(HeuristicInterface):
    """
    Scores the test files an LLM expects this change to break, from the
    selection file that pytorch/test-infra's tools/torchci/td/llm_select writes
    for the PR head (its README has the format) and fetch_llm_selection.py
    brings to the TD job.

    Raises when the run has no selection (a fork PR, a push to main, a failed or
    late selector). The TD scorer then reads the heuristic as not run; scoring
    nothing would instead tell it that the LLM expects no test to break.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def get_prediction_confidence(self, tests: list[str]) -> TestPrioritizations:
        scores = load_scores()
        listed = set(tests)
        weighted = {
            TestRun(t): s * LLM_SELECTION_WEIGHT
            for t, s in scores.items()
            if t in listed
        }
        return TestPrioritizations(tests, weighted)


def load_scores() -> dict[str, float]:
    """The selection file's `scores`: test file -> score in (0, 1]."""
    if not LLM_SELECTION_PATH.exists():
        raise FileNotFoundError(f"no {LLM_SELECTION_FILE} for this run")
    with open(LLM_SELECTION_PATH) as f:
        selection = json.load(f)
    if not isinstance(selection, dict):
        raise ValueError(f"{LLM_SELECTION_FILE} is not an object")
    if selection.get("version") != LLM_SELECTION_VERSION:
        raise ValueError(f"unknown version {selection.get('version')!r}")
    scores = selection.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("scores is not an object")
    for test, score in scores.items():
        # bool is an int, and json.load accepts NaN and Infinity.
        if type(score) not in (int, float) or not (
            math.isfinite(score) and 0 < score <= 1
        ):
            raise ValueError(f"the score of {test!r} is not a number in (0, 1]")
    return {test: float(score) for test, score in scores.items()}
