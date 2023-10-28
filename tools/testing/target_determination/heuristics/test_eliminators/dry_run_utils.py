import json
from typing import Any, Dict, List

import requests
from tools.stats.upload_metrics import EnvVarMetric


BUILD_ENVIRONMENT = EnvVarMetric("build_environment", "BUILD_ENVIRONMENT").value()
TEST_CONFIG = EnvVarMetric("test_config", "TEST_CONFIG", required=False).value()


def get_github_json() -> Dict[str, Any]:
    data = {}

    # Define the URL of the JSON file in the GitHub repo
    url = "https://raw.githubusercontent.com/pytorch/test-infra/generated-stats/stats/test-times.json"

    # Fetch the file
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        data = json.loads(response.content.decode("utf-8"))

    else:
        # TODO: Should we fail silently here?
        print(f"Failed to download the file. Status code: {response.status_code}")
    return data


def calculate_recall(
    selected_tests: List[str], ground_truth_failures: List[str]
) -> float:
    """
    Calculate recall and precision statistics for test selection.

    Parameters:
    - selected_tests: List of tests that were selected to be run.
    - ground_truth_failures: List of tests that are known to actually fail.

    Returns:
    - float containing 'recall' stat.
    """
    # Convert lists to sets for faster look-up
    selected_tests_set = set(selected_tests)
    ground_truth_failures_set = set(ground_truth_failures)

    # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
    TP = len(ground_truth_failures_set.intersection(selected_tests_set))
    FN = len(ground_truth_failures_set) - TP

    # Calculate recall and precision
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0

    return recall


def calculate_test_times_sum(ignored_tests: List[str]) -> float:
    """
    Calculate the sum of the test times for the selected tests.

    Parameters:
    - ignored_tests: List of tests that were ignored.

    Returns:
    - Dictionary containing 'test_times_sum' statistic.
    """
    # Convert lists to sets for faster look-up
    ignored_tests_set = set(ignored_tests)
    data = get_github_json()
    # Calculate test times sum
    test_times_sum = sum(
        data[BUILD_ENVIRONMENT][TEST_CONFIG][test] for test in ignored_tests_set
    )
    # assert total_test_times is a float
    assert isinstance(test_times_sum, float)
    return test_times_sum


def calculate_failed_tests_in_ignored_tests(
    ignored_tests: List[str], ground_truth_failures: List[str]
) -> int:
    ignored_tests_set = set(ignored_tests)
    ground_truth_failures_set = set(ground_truth_failures)
    return len(ground_truth_failures_set.intersection(ignored_tests_set))


def get_metrics_dict_for_td_strategy(
    strategy_name: str,
    selected_tests: List[str],
    ignored_tests: List[str],
    ground_truth_failures: List[str],
) -> Dict[str, Any]:
    metrics_dict: Dict[str, Any] = {}
    # recall here is percent of failed tests that were selected
    metrics_dict["recall"] = calculate_recall(selected_tests, ground_truth_failures)
    metrics_dict["contained_failure"] = len(ground_truth_failures) > 0
    metrics_dict["num_ignored_failed_tests"] = len(ignored_tests)
    metrics_dict["time_saved_seconds"] = calculate_test_times_sum(ignored_tests)
    metrics_dict["strategy_name"] = strategy_name
    metrics_dict["metric_type"] = "td_strategy"
    return metrics_dict
