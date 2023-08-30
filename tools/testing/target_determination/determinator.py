from typing import List

from tools.stats.upload_metrics import emit_metric

from tools.testing.target_determination.heuristics import (
    HEURISTICS,
    TestPrioritizations as TestPrioritizations,
)


def get_test_prioritizations(tests: List[str]) -> TestPrioritizations:
    rankings = TestPrioritizations(tests_being_ranked=tests)
    print(f"Received {len(tests)} tests to prioritize")
    for test in tests:
        print(f"  {test}")

    for heuristic in HEURISTICS:
        new_rankings = heuristic.get_test_priorities(tests)
        rankings.integrate_priorities(new_rankings)

        num_tests_found = len(new_rankings.get_prioritized_tests())
        print(
            f"Heuristic {heuristic} identified {num_tests_found} tests "
            + f"to prioritize ({(num_tests_found / len(tests)):.2%}%)"
        )

        if num_tests_found:
            new_rankings.print_info()

    emit_metric(
        "test_reordering_prioritized_tests",
        {
            "high_relevance_tests": rankings.get_high_relevance_tests(),
            "probable_relevance_tests": rankings.get_probable_relevance_tests(),
            "unranked_relevance_tests": rankings.get_unranked_relevance_tests(),
        },
    )
    return rankings
