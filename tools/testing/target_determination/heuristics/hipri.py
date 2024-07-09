
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable
from warnings import warn

from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
)

from tools.testing.target_determination.heuristics.utils import (
    normalize_ratings,
    query_changed_files,
)
from tools.testing.test_run import TestRun

class HiPri(HeuristicInterface):
    # Literally just a heuristic for test_public_bindings.  Pretty much anything
    # that changes the public API can affect this test
    hi_pri = [
        "test_public_bindings",
    ]
    def __init__(self, **kwargs: dict[str, Any]) -> None:
        super().__init__(**kwargs)

    def get_prediction_confidence(self, tests: list[str]) -> TestPrioritizations:
        test_ratings = {TestRun(k): 1 for k in self.hi_pri if k in tests}
        return TestPrioritizations(
            tests, test_ratings
        )
