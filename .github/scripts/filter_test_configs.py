#!/usr/bin/env python3

import json
import os
import requests
from typing import Any, Dict, Set, List
import yaml
import warnings

CIFLOW_PREFIX = "ciflow/"

# Same as shard names
VALID_TEST_CONFIG_LABELS = {f"{CIFLOW_PREFIX}{label}" for label in {
    "backwards_compat",
    "crossref",
    "default",
    "deploy",
    "distributed",
    "docs_tests",
    "dynamo",
    "force_on_cpu",
    "functorch",
    "jit_legacy",
    "multigpu",
    "nogpu_AVX512",
    "nogpu_NO_AVX2",
    "slow",
    "xla",
}}

def parse_args() -> Any:
    from argparse import ArgumentParser
    parser = ArgumentParser("Filter all test configurations and keep only requested ones")
    parser.add_argument("--test-matrix", type=str, required=True, help="the original test matrix")
    parser.add_argument("--pr-number", type=str, help="the pull request number")
    return parser.parse_args()


def get_labels(pr_number: int) -> Set[str]:
    """
    Dynamical get the latest list of labels from the pull request
    """
    # From https://docs.github.com/en/actions/learn-github-actions/environment-variables
    PYTORCH_REPO = os.environ.get("GITHUB_REPOSITORY", "pytorch/pytorch")
    PYTORCH_GITHUB_API = f"https://api.github.com/repos/{PYTORCH_REPO}"
    GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]

    REQUEST_HEADERS = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": "token " + GITHUB_TOKEN,
    }

    response = requests.get(
        f"{PYTORCH_GITHUB_API}/issues/{pr_number}/labels",
        headers=REQUEST_HEADERS,
    )

    if response.status_code != requests.codes.ok:
        warnings.warn(f"Failed to get the labels for #{pr_number} (status code {response.status_code})")
        return set()

    return {label.get("name") for label in response.json() if label.get("name")}


def filter(test_matrix: Dict[str, List[Any]], labels: Set[str]) -> Dict[str, List[Any]]:
    """
    Select the list of test config to run from the test matrix. The logic works
    as follows:

    If the PR has one or more labels as specified in the VALID_TEST_CONFIG_LABELS set, only
    these test configs will be selected.  Note that regular ciflow labels like ciflow/trunk
    don't count here, only test configs.

    If the PR has none of the ciflow label, all tests are run as usual.
    """

    filtered_test_matrix: Dict[str, List[Any]] = {
        "include": []
    }

    for entry in test_matrix.get("include", []):
        config_name = entry.get("config", "")
        if not config_name:
            continue

        label = f"{CIFLOW_PREFIX}{config_name.strip().lower()}"
        if label in labels:
            print(f"Select {config_name} because label {label} is presented in the pull request by the time the test starts")
            filtered_test_matrix["include"].append(entry)

    valid_test_config_labels = labels.intersection(VALID_TEST_CONFIG_LABELS)

    if filtered_test_matrix["include"] or valid_test_config_labels:
        # When the filter test matrix contain matches or if a valid test config label
        # is found in the PR, return the filtered test matrix
        return filtered_test_matrix
    else:
        # Found no valid label, return the same test matrix as before so that all
        # tests can be run normally
        return test_matrix


def main() -> None:
    args = parse_args()
    # Load the original test matrix set by the workflow. Its format, however,
    # doesn't follow the strict JSON format, so we load it using yaml here for
    # its more relaxed syntax
    test_matrix = yaml.safe_load(args.test_matrix)
    pr_number = args.pr_number

    if not pr_number:
        # This can be none or empty like when the workflow is dispatched manually
        filtered_test_matrix = test_matrix

    else:
        # First, query all the labels from the pull requests
        labels = get_labels(int(pr_number))
        # Then filter the test matrix and keep only the selected ones
        filtered_test_matrix = filter(test_matrix, labels)

    # Set the filtered test matrix as the output
    print(f"::set-output name=test-matrix::{json.dumps(filtered_test_matrix)}")

    filtered_test_matrix_len = len(filtered_test_matrix.get("include", []))
    # and also put a flag if the test matrix is empty, so subsequent jobs can
    # quickly check it without the need to parse the JSON string
    print(f"::set-output name=is-test-matrix-empty::{filtered_test_matrix_len == 0}")


if __name__ == "__main__":
    main()
