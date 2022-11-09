#!/usr/bin/env python3

import sys
import re
import json
import os
import requests
from typing import Any, Dict, Set, List
import yaml
import warnings
import random

PREFIX = "test-config/"

# Same as shard names
VALID_TEST_CONFIG_LABELS = {f"{PREFIX}{label}" for label in {
    "backwards_compat",
    "crossref",
    "default",
    "deploy",
    "distributed",
    "docs_tests",
    "dynamo",
    "force_on_cpu",
    "functorch",
    "inductor",
    "inductor_distributed",
    "inductor_timm",
    "jit_legacy",
    "multigpu",
    "nogpu_AVX512",
    "nogpu_NO_AVX2",
    "slow",
    "tsan",
    "xla",
}}

# Supported mode when running periodically. For simplicity, a random weight
# will be assigned to each mode so that they can be chosen at random
SUPPORTED_PERIODICAL_MODES = {
    # TODO: DEBUG TO BE RESET TO 0.5 before committing
    "mem_leak_check": 0.0,
    "rerun_disabled_tests": 1.0,
}


def parse_args() -> Any:
    from argparse import ArgumentParser
    parser = ArgumentParser("Filter all test configurations and keep only requested ones")
    parser.add_argument("--test-matrix", type=str, required=True, help="the original test matrix")
    parser.add_argument("--pr-number", type=str, help="the pull request number")
    parser.add_argument("--tag", type=str, help="the associated tag if it exists")
    parser.add_argument("--event-name", type=str, help="name of the event that triggered the job (pull, schedule, etc)")
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
    these test configs will be selected.  This also works with ciflow labels, for example,
    if a PR has both ciflow/trunk and test-config/functorch, only trunk functorch builds
    and tests will be run

    If the PR has none of the test-config label, all tests are run as usual.
    """

    filtered_test_matrix: Dict[str, List[Any]] = {
        "include": []
    }

    for entry in test_matrix.get("include", []):
        config_name = entry.get("config", "")
        if not config_name:
            continue

        label = f"{PREFIX}{config_name.strip()}"
        if label in labels:
            print(f"Select {config_name} because label {label} is presented in the pull request by the time the test starts")
            filtered_test_matrix["include"].append(entry)

    valid_test_config_labels = labels.intersection(VALID_TEST_CONFIG_LABELS)

    if not filtered_test_matrix["include"] and not valid_test_config_labels:
        # Found no valid label and the filtered test matrix is empty, return the same
        # test matrix as before so that all tests can be run normally
        return test_matrix
    else:
        # When the filter test matrix contain matches or if a valid test config label
        # is found in the PR, return the filtered test matrix
        return filtered_test_matrix


def set_output(name: str, val: Any) -> None:
    if os.getenv("GITHUB_OUTPUT"):
        with open(str(os.getenv("GITHUB_OUTPUT")), "a") as env:
            print(f"{name}={val}", file=env)
    else:
        print(f"::set-output name={name}::{val}")


def main() -> None:
    args = parse_args()
    # Load the original test matrix set by the workflow. Its format, however,
    # doesn't follow the strict JSON format, so we load it using yaml here for
    # its more relaxed syntax
    test_matrix = yaml.safe_load(args.test_matrix)

    if test_matrix is None:
        warnings.warn(f"Invalid test matrix input '{args.test_matrix}', exiting")
        # We handle invalid test matrix gracefully by marking it as empty
        set_output("is-test-matrix-empty", True)
        sys.exit(0)

    pr_number = args.pr_number
    tag = args.tag

    # If the tag matches, we can get the PR number from it, this is from ciflow
    # workflow dispatcher
    tag_regex = re.compile(r"^ciflow/\w+/(?P<pr_number>\d+)$")

    if pr_number:
        # If a PR number is set, query all the labels from that PR
        labels = get_labels(int(pr_number))
        # Then filter the test matrix and keep only the selected ones
        filtered_test_matrix = filter(test_matrix, labels)

    elif tag:
        m = tag_regex.match(tag)

        if m:
            pr_number = m.group("pr_number")

            # The PR number can also come from the tag in ciflow tag event
            labels = get_labels(int(pr_number))
            # Filter the test matrix and keep only the selected ones
            filtered_test_matrix = filter(test_matrix, labels)

        else:
            # There is a tag but it isn't ciflow, so there is nothing left to do
            filtered_test_matrix = test_matrix

    else:
        # No PR number, no tag, we can just return the test matrix as it is
        filtered_test_matrix = test_matrix


    # TODO: DEBUG, to be removed back to == "schedule"
    if args.event_name != "schedule":
        selected_mode = random.choices(
            list(SUPPORTED_PERIODICAL_MODES.keys()),
            weights=list(SUPPORTED_PERIODICAL_MODES.values()),
            k=1)[0]

        for config in filtered_test_matrix.get("include", []):
            config[selected_mode] = selected_mode

    # Set the filtered test matrix as the output
    set_output("test-matrix", json.dumps(filtered_test_matrix))

    filtered_test_matrix_len = len(filtered_test_matrix.get("include", []))
    # and also put a flag if the test matrix is empty, so subsequent jobs can
    # quickly check it without the need to parse the JSON string
    set_output("is-test-matrix-empty", filtered_test_matrix_len == 0)


if __name__ == "__main__":
    main()
