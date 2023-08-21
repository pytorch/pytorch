from typing import List

from tools.stats.upload_metrics import emit_metric

from tools.testing.target_determination.heuristics import (
    HEURISTICS,
    TestPrioritizations as TestPrioritizations,
)


def _print_tests(label: str, tests: List[str]) -> None:
    if not tests:
        return

    print(f"{label} tests ({len(tests)}):")
    for test in tests:
        if test in tests:
            print(f"  {test}")


def get_test_prioritizations(tests: List[str]) -> TestPrioritizations:
    rankings = TestPrioritizations(unranked_relevance = tests.copy())
    print(f"Received {len(tests)} tests to prioritize")
    for test in tests:
        print(f"  {test}")

    for heuristic in HEURISTICS:
        new_rankings = heuristic.get_test_priorities(tests)
        rankings.integrate_priorities(new_rankings)

        num_tests_found = len(new_rankings.highly_relevant) + len(
            new_rankings.probably_relevant
        )
        print(
            f"Heuristic {heuristic} identified {num_tests_found} tests "
            + f"to prioritize ({(num_tests_found / len(tests)):.2%}%)"
        )

        if num_tests_found:
            _print_tests("Highly relevant", new_rankings.highly_relevant)
            _print_tests("Probably relevant", new_rankings.probably_relevant)
            _print_tests("Unranked relevance", new_rankings.unranked_relevance)

        print("Aggregated results across all heuristics:")
        rankings.print_info(tests)

    num_tests_analyzed = (
        len(rankings.highly_relevant)
        + len(rankings.probably_relevant)
        + len(rankings.unranked_relevance)
    )

    assert num_tests_analyzed == len(tests), (
        f"Was given {len(tests)} tests to prioritize, but analysis returned {num_tests_analyzed} tests. "
        + "Breakdown:\n"
        + f"Highly relevant: {len(rankings.highly_relevant)}\n"
        + f"Probably relevant: {len(rankings.probably_relevant)}\n"
        + f"Unranked relevance: {len(rankings.unranked_relevance)}\n"
    )

    emit_metric(
        "test_reordering_prioritized_tests",
        {
            "highly_relevant_tests": rankings.highly_relevant,
            "probably_relevant_tests": rankings.probably_relevant,
            "unranked_tests": rankings.unranked_relevance,
        },
    )
    return rankings
