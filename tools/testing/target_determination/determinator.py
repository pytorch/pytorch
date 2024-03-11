import sys
from typing import Any, Dict, List

from tools.testing.target_determination.heuristics import (
    AggregatedHeuristics as AggregatedHeuristics,
    HEURISTICS,
    TestPrioritizations as TestPrioritizations,
)


def get_test_prioritizations(
    tests: List[str], file: Any = sys.stdout
) -> AggregatedHeuristics:
    aggregated_results = AggregatedHeuristics(tests)
    print(f"Received {len(tests)} tests to prioritize", file=file)
    for test in tests:
        print(f"  {test}", file=file)

    for heuristic in HEURISTICS:
        new_rankings: TestPrioritizations = heuristic.get_prediction_confidence(tests)
        aggregated_results.add_heuristic_results(heuristic, new_rankings)

        print(new_rankings.get_info_str(), file=file)

    return aggregated_results


def get_prediction_confidences(tests: List[str]) -> Dict[str, TestPrioritizations]:
    # heuristic name -> test -> rating/confidence
    rankings: Dict[str, TestPrioritizations] = {}
    for heuristic in HEURISTICS:
        rankings[heuristic.name] = heuristic.get_prediction_confidence(tests)
    return rankings
