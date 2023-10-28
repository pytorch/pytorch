from typing import Any, Dict, List

from tools.testing.target_determination.determinator import get_test_prioritizations
from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TargetDeterminatorInterface,
)
from tools.testing.target_determination.heuristics.test_eliminators.dry_run_utils import (
    get_metrics_dict_for_td_strategy,
)


class RemoveUnrankedUsingHeuristics(TargetDeterminatorInterface):
    """
    remove unranked tests by using heuristics
    """

    def __init__(self, heuristics: List[HeuristicInterface], **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        self.heuristics: List[HeuristicInterface] = heuristics  # type: ignore[assignment]

    def gen_selected_tests(self, tests: List[str]) -> List[str]:
        """
        get selected tests by using heuristics and set selected_tests and ignored_tests
        """
        aggregate_results = get_test_prioritizations(tests, self.heuristics)
        test_prioritizations = aggregate_results.get_aggregated_priorities()
        selected_tests = self.selected_tests
        ignored_tests = self.ignored_tests
        selected_tests.extend(test_prioritizations.get_high_relevance_tests())
        selected_tests.extend(test_prioritizations.get_probable_relevance_tests())
        ignored_tests.extend(test_prioritizations.get_unranked_relevance_tests())
        # remove duplicates
        selected_tests = list(set(self.selected_tests))
        ignored_tests = list(set(self.ignored_tests))
        self.selected_tests = selected_tests
        self.ignored_tests = ignored_tests

        return self.selected_tests

    def create_metrics_dict_for_dry_run(
        self, ground_truth_failures: List[str]
    ) -> Dict[str, Any]:
        """
        Returns the prioritizations for the given tests.

        """
        return get_metrics_dict_for_td_strategy(
            self.name,
            self.selected_tests,
            self.ignored_tests,
            ground_truth_failures,
        )
