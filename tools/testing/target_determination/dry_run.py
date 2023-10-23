import json
import os
from typing import Any, Callable, Dict, List, Tuple

import requests
from tools.stats.upload_metrics import emit_metric


class EnvVarMetric:
    name: str
    env_var: str
    required: bool = True
    # Used to cast the value of the env_var to the correct type (defaults to str)
    type_conversion_fn: Any = None

    def __init__(
        self,
        name: str,
        env_var: str,
        required: bool = True,
        type_conversion_fn: Any = None,
    ) -> None:
        self.name = name
        self.env_var = env_var
        self.required = required
        self.type_conversion_fn = type_conversion_fn

    def value(self) -> Any:
        value = os.environ.get(self.env_var)

        # Github CI will set some env vars to an empty string
        DEFAULT_ENVVAR_VALUES = [None, ""]
        if value in DEFAULT_ENVVAR_VALUES:
            if not self.required:
                return None

            raise ValueError(
                f"Missing {self.name}. Please set the {self.env_var} "
                "environment variable to pass in this value."
            )

        if self.type_conversion_fn:
            return self.type_conversion_fn(value)
        return value


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


# The emit_metrics_decorator function is a decorator that emits metrics for a test dependency strategy.
# The function takes a function as an argument, and returns a new function that emits metrics for the decorated function.


def emit_td_dry_run_metrics(
    func: Callable[..., Tuple[str, List[str], List[str], List[str]]]
) -> Callable[..., Tuple[str, List[str], List[str], List[str]]]:
    """
    Decorator to emit metrics based on the output of the decorated function.

    Parameters:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The decorated function.
    """

    def wrapper(
        *args: Any, **kwargs: Any
    ) -> Tuple[str, List[str], List[str], List[str]]:
        """
        Wrapper function to execute the decorated function and emit metrics.

        Parameters:
            *args (Any): Positional arguments for the decorated function.
            **kwargs (Any): Keyword arguments for the decorated function.

        Returns:
            Tuple[str, List[str], List[str], List[str]]: The original output of the decorated function.
        """
        # Run the decorated function and store its output
        output = func(*args, **kwargs)

        # Unpack the output to get the required variables
        strategy_name, selected_tests, ignored_tests, ground_truth_failures = output

        # Perform additional actions (in this example, emitting metrics)
        metrics_dict = get_metrics_dict_for_td_strategy(
            strategy_name, selected_tests, ignored_tests, ground_truth_failures
        )
        emit_metric(f"{strategy_name}_td_stats", metrics_dict)

        # Return the original output of the decorated function
        return output

    return wrapper
