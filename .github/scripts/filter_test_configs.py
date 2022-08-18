#!/usr/bin/env python3

import os
import requests
from typing import Any, List
import warnings


def parse_args() -> Any:
    from argparse import ArgumentParser
    parser = ArgumentParser("Filter all test configurations and keep only requested ones")
    parser.add_argument("--test-matrix", type=str, required=True, help="the original test matrix")
    parser.add_argument("--pr-number", type=int, required=True, help="the pull request number")
    return parser.parse_args()


def get_labels(pr_number: int) -> List[str]:
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
        return []

    print(response.json())
    return []


def main() -> None:
    args = parse_args()
    # The original test matrix set by the workflow
    test_matrix = args.test_matrix
    pr_number = args.pr_number

    # First, query all the labels from the pull requests
    labels = get_labels(pr_number)

    print(test_matrix)
    print("====")
    print(labels)

    # DEBUG
    print(f"::set-output name=test-matrix::{test_matrix}")


if __name__ == "__main__":
    main()
