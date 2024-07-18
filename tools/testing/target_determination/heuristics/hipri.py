
from __future__ import annotations
from typing import Any

from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
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
        test_ratings = {TestRun(k): 1.0 for k in self.hi_pri if k in tests}
        return TestPrioritizations(
            tests, test_ratings
        )
