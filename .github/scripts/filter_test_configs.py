#!/usr/bin/env python3

import json
import os
import requests
from typing import Any, Dict, Set, List
import warnings

CIFLOW_PREFIX = "ciflow/"


def parse_args() -> Any:
    from argparse import ArgumentParser
    parser = ArgumentParser("Filter all test configurations and keep only requested ones")
    parser.add_argument("--test-matrix", type=str, required=True, help="the original test matrix")
    parser.add_argument("--pr-number", type=int, required=True, help="the pull request number")
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
        warnings.warn(f"Failed to get the labels for #{pr_number} with status code {response.status_code}")
        return set()

    return {label.get("name", "") for label in response.json()}


def filter(test_matrix: Dict[str, List[Any]], labels: Set[str]) -> Dict[str, List[Any]]:
    filtered_test_matrix: Dict[str, List[Any]] = {
        "include": []
    }

    for config in test_matrix.get("include", []):
        config_name = config.get("name", "")
        if not config_name:
            continue

        label = f"{CIFLOW_PREFIX}{config_name.strip().lower()}"
        if label in labels:
            print(f"Select {config_name} because label {label} is presented in the pull request by the time the test starts")
            filtered_test_matrix["include"].append(config)

    # If no matching label is found, the default is to run everything as normal
    return filtered_test_matrix if filtered_test_matrix["include"] else test_matrix


def main() -> None:
    args = parse_args()
    # The original test matrix set by the workflow
    test_matrix = json.loads(args.test_matrix)
    pr_number = args.pr_number

    # First, query all the labels from the pull requests
    labels = get_labels(pr_number)
    # Then filter the test matrix and keep only the selected ones
    filtered_test_matrix = filter(test_matrix, labels)

    print(json.dumps(filtered_test_matrix))
    print("=======")
    print(labels)

    # Set the filtered test matrix as the output
    print(f"::set-output name=test-matrix::{json.dumps(filtered_test_matrix)}")


if __name__ == "__main__":
    main()
