import random
from typing import Any, Dict, List, Tuple

from tools.stats.upload_metrics import emit_metric
from tools.testing.target_determination.determinator import get_test_prioritizations
from tools.testing.target_determination.dry_run import emit_td_dry_run_metrics
from tools.testing.target_determination.heuristics import AggregatedHeuristics
from tools.testing.target_determination.heuristics.interface import TestPrioritizations
from tools.testing.test_selections import get_test_case_configs

from torch.testing._internal.common_utils import IS_CI


# This strategy uses our baseline heuristics, and then randomly eliminates unranked tests
REMOVAL_PROBABILITY = 0.2
VERSION = "0.0.1"


class DefaultHeuristic:
    def __init__(self, tests_to_run: List[str], test_directory: str, is_cpp: bool):
        self.tests_to_run = tests_to_run
        self.test_directory = test_directory
        self.is_cpp = is_cpp
        self.aggregated_heuristics = AggregatedHeuristics(unranked_tests=tests_to_run)
        self.test_prioritizations, self.metrics_dict = self.get_baseline_heuristics(
            tests_to_run, test_directory, is_cpp
        )

    def get_baseline_heuristics(
        self, tests_to_run: List[str], test_directory: str, is_cpp: bool
    ) -> Tuple[TestPrioritizations, Dict[str, Any]]:
        aggregated_heuristics = self.aggregated_heuristics
        metrics_dict = {}
        if IS_CI:
            # downloading test cases configuration to local environment
            get_test_case_configs(dirpath=test_directory)
            aggregated_heuristics = get_test_prioritizations(tests_to_run)

        test_prioritizations = aggregated_heuristics.get_aggregated_priorities()
        test_prioritizations.print_info()

        if IS_CI:
            metrics_dict = {
                "high_relevance_tests": test_prioritizations.get_high_relevance_tests(),
                "probable_relevance_tests": test_prioritizations.get_probable_relevance_tests(),
                "unranked_relevance_tests": test_prioritizations.get_unranked_relevance_tests(),
                "cpp": is_cpp,
            }
        return test_prioritizations, metrics_dict

    def get_individual_test_stats(self, test_name: str) -> Dict[str, Any]:
        return self.aggregated_heuristics.get_test_stats(test_name)

    def get_test_prioritizations(self) -> TestPrioritizations:
        return self.test_prioritizations

    def get_metrics_dict(self) -> Dict[str, Any]:
        return self.metrics_dict

    def add_metrics(self, metrics_dict: Dict[str, Any]) -> None:
        self.metrics_dict.update(metrics_dict)

    def emit_metrics(self, name: str) -> None:
        emit_metric(name, self.metrics_dict)

    @emit_td_dry_run_metrics
    def do_dry_run_with_failures(
        self, ground_truth_failures: List[str]
    ) -> Tuple[str, List[str], List[str], List[str]]:
        """
        Perform a dry run with the given ground truth failures.

        Parameters:
            ground_truth_failures (List[str]): The ground truth failures.

        Returns:
            Tuple[str, List[str], List[str], List[str]]: The output of the dry run.
        """

        selected_tests = list(
            self.get_test_prioritizations().get_high_relevance_tests()
        )
        selected_tests.extend(
            list(self.get_test_prioritizations().get_probable_relevance_tests())
        )

        ignored_tests = []
        for test in self.test_prioritizations.get_unranked_relevance_tests():
            if random.random() > REMOVAL_PROBABILITY:
                selected_tests.append(test)
            else:
                ignored_tests.append(test)

        # Return the results
        return (
            f"default_heuristic_td_version_{VERSION}",
            selected_tests,
            ignored_tests,
            ground_truth_failures,
        )
