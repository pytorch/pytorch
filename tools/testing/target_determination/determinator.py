from typing import List

from tools.testing.target_determination.heuristics import (
    AggregatedHeuristics as AggregatedHeuristics,
    HEURISTICS,
    TestPrioritizations as TestPrioritizations,
)


def get_test_prioritizations(tests: List[str]) -> AggregatedHeuristics:
    aggregated_results = AggregatedHeuristics(unranked_tests=tests)
    print(f"Received {len(tests)} tests to prioritize")
    for test in tests:
        print(f"  {test}")

    for heuristic in HEURISTICS:
        new_rankings: TestPrioritizations = heuristic.get_test_priorities(tests)
        aggregated_results.add_heuristic_results(str(heuristic), new_rankings)

        num_tests_found = len(new_rankings.get_prioritized_tests())
        print(
            f"Heuristic {heuristic} identified {num_tests_found} tests "
            + f"to prioritize ({(num_tests_found / len(tests)):.2%}%)"
        )

        if num_tests_found:
            new_rankings.print_info()

    return aggregated_results
