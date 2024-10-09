from __future__ import annotations

from typing import Any
from warnings import warn

from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
)
from tools.testing.target_determination.heuristics.utils import query_changed_files
from tools.testing.test_run import TestRun


class PublicBindings(HeuristicInterface):
    # Literally just a heuristic for test_public_bindings.  Pretty much anything
    # that changes the public API can affect this testp
    test_public_bindings = "test_public_bindings"
    additional_files = ["test/allowlist_for_publicAPI.json"]

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        super().__init__(**kwargs)

    def get_prediction_confidence(self, tests: list[str]) -> TestPrioritizations:
        test_ratings = {}
        try:
            changed_files = query_changed_files()
        except Exception as e:
            warn(f"Can't query changed test files due to {e}")
            changed_files = []

        if any(
            file.startswith("torch/") or file in self.additional_files
            for file in changed_files
        ):
            test_ratings[TestRun(self.test_public_bindings)] = 1.0
        return TestPrioritizations(tests, test_ratings)
