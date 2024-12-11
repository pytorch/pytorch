from __future__ import annotations

import sys
from typing import Any

from tools.testing.target_determination.heuristics import (
    AggregatedHeuristics as AggregatedHeuristics,
    HEURISTICS,
    TestPrioritizations as TestPrioritizations,
)


def get_test_prioritizations(
    tests: list[str], file: Any = sys.stdout
) -> AggregatedHeuristics:
    aggregated_results = AggregatedHeuristics(tests)
    print(f"Received {len(tests)} tests to prioritize", file=file)
    for test in tests:
        print(f"  {test}", file=file)

    for heuristic in HEURISTICS:
        try:
            new_rankings: TestPrioritizations = heuristic.get_prediction_confidence(
                tests
            )
            aggregated_results.add_heuristic_results(heuristic, new_rankings)

            print(f"Results from {heuristic.__class__.__name__}")
            print(new_rankings.get_info_str(verbose=False), file=file)
        except Exception as e:
            print(f"Error in {heuristic.__class__.__name__}: {e}", file=file)

    return aggregated_results
