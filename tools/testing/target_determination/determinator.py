from typing import Any, Dict, List

from tools.testing.target_determination.heuristics import (
    AggregatedHeuristics as AggregatedHeuristics,
    HEURISTICS,
    TestPrioritizations as TestPrioritizations,
)


def get_test_prioritizations(
    tests: List[str], print_fn: Any = print
) -> AggregatedHeuristics:
    aggregated_results = AggregatedHeuristics(unranked_tests=tests)
    print_fn(f"Received {len(tests)} tests to prioritize")
    for test in tests:
        print_fn(f"  {test}")

    for heuristic in HEURISTICS:
        new_rankings: TestPrioritizations = heuristic.get_test_priorities(tests)
        aggregated_results.add_heuristic_results(heuristic, new_rankings)

        num_tests_found = len(new_rankings.get_prioritized_tests())
        print_fn(
            f"Heuristic {heuristic} identified {num_tests_found} tests "
            + f"to prioritize ({(num_tests_found / len(tests)):.2%}%)"
        )

        if num_tests_found:
            print_fn(new_rankings.get_info_str())

    return aggregated_results


def get_prediction_confidences(tests: List[str]) -> Dict[str, Dict[str, float]]:
    # heuristic name -> test -> rating/confidence
    rankings: Dict[str, Dict[str, float]] = {}
    for heuristic in HEURISTICS:
        rankings[heuristic.name] = heuristic.get_prediction_confidence(tests)
    return rankings
