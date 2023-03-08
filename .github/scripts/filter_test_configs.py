#!/usr/bin/env python3

import json
import os
import re
import sys
import warnings
from typing import Any, Dict, List, Set
from urllib.request import Request, urlopen

import yaml

PREFIX = "test-config/"

# Same as shard names
VALID_TEST_CONFIG_LABELS = {
    f"{PREFIX}{label}"
    for label in {
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
        "inductor_huggingface",
        "inductor_timm",
        "inductor_torchbench",
        "jit_legacy",
        "multigpu",
        "nogpu_AVX512",
        "nogpu_NO_AVX2",
        "slow",
        "tsan",
        "xla",
    }
}

# Supported modes when running periodically
SUPPORTED_PERIODICAL_MODES = {
    "mem_leak_check",
    "rerun_disabled_tests",
}

# The link to the published list of disabled jobs
DISABLED_JOBS_URL = "https://ossci-metrics.s3.amazonaws.com/disabled-jobs.json"
# Some constants used to remove disabled jobs
JOB_NAME_SEP = "/"
BUILD_JOB_NAME = "build"
TEST_JOB_NAME = "test"
BUILD_AND_TEST_JOB_NAME = "build-and-test"
JOB_NAME_CFG_REGEX = re.compile(r"(?P<job>[\w-]+)\s+\((?P<cfg>[\w-]+)\)")


def parse_args() -> Any:
    from argparse import ArgumentParser

    parser = ArgumentParser(
        "Filter all test configurations and keep only requested ones"
    )
    parser.add_argument(
        "--test-matrix", type=str, required=True, help="the original test matrix"
    )
    parser.add_argument(
        "--workflow", type=str, help="the name of the current workflow, i.e. pull"
    )
    parser.add_argument(
        "--job-name",
        type=str,
        help="the name of the current job, i.e. linux-focal-py3.8-gcc7 / build",
    )
    parser.add_argument("--pr-number", type=str, help="the pull request number")
    parser.add_argument("--tag", type=str, help="the associated tag if it exists")
    parser.add_argument(
        "--event-name",
        type=str,
        help="name of the event that triggered the job (pull, schedule, etc)",
    )
    parser.add_argument(
        "--schedule", type=str, help="cron schedule that triggered the job"
    )
    return parser.parse_args()


def get_labels(pr_number: int) -> Set[str]:
    """
    Dynamical get the latest list of labels from the pull request
    """
    # From https://docs.github.com/en/actions/learn-github-actions/environment-variables
    pytorch_repo = os.environ.get("GITHUB_REPOSITORY", "pytorch/pytorch")
    pytorch_github_api = f"https://api.github.com/repos/{pytorch_repo}"
    github_token = os.environ["GITHUB_TOKEN"]

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {github_token}",
    }
    json_response = download_json(
        url=f"{pytorch_github_api}/issues/{pr_number}/labels",
        headers=headers,
    )

    if not json_response:
        warnings.warn(f"Failed to get the labels for #{pr_number}")
        return set()

    return {label.get("name") for label in json_response if label.get("name")}


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

    filtered_test_matrix: Dict[str, List[Any]] = {"include": []}

    for entry in test_matrix.get("include", []):
        config_name = entry.get("config", "")
        if not config_name:
            continue

        label = f"{PREFIX}{config_name.strip()}"
        if label in labels:
            print(
                f"Select {config_name} because label {label} is presented in the pull request by the time the test starts"
            )
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


def set_periodic_modes(test_matrix: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """
    Apply all periodic modes when running under a schedule
    """
    scheduled_test_matrix: Dict[str, List[Any]] = {
        "include": [],
    }

    for config in test_matrix.get("include", []):
        for mode in SUPPORTED_PERIODICAL_MODES:
            cfg = config.copy()
            cfg[mode] = mode
            scheduled_test_matrix["include"].append(cfg)

    return scheduled_test_matrix


def remove_disabled_jobs(
    workflow: str, job_name: str, test_matrix: Dict[str, List[Any]]
) -> Dict[str, List[Any]]:
    """
    Check the list of disabled jobs, remove the current job and all its dependents
    if it exists in the list. The list of disabled jobs is as follows:

    {
        "WORKFLOW / PLATFORM / JOB (CONFIG)": [
            AUTHOR,
            ISSUE_NUMBER,
            ISSUE_URL,
            WORKFLOW,
            PLATFORM,
            JOB (CONFIG),
        ],
        "pull / linux-bionic-py3.8-clang9 / test (dynamo)": [
            "pytorchbot",
            "94861",
            "https://github.com/pytorch/pytorch/issues/94861",
            "pull",
            "linux-bionic-py3.8-clang9",
            "test (dynamo)",
        ],
    }
    """
    try:
        # The job name from github is in the PLATFORM / JOB (CONFIG) format, so breaking
        # it into its two components first
        current_platform, _ = [n.strip() for n in job_name.split(JOB_NAME_SEP, 1) if n]
    except ValueError as error:
        warnings.warn(f"Invalid job name {job_name}, returning")
        return test_matrix

    # The result will be stored here
    filtered_test_matrix: Dict[str, List[Any]] = {"include": []}

    for _, record in download_json(url=DISABLED_JOBS_URL, headers={}).items():
        (
            author,
            _,
            disabled_url,
            disabled_workflow,
            disabled_platform,
            disabled_job_cfg,
        ) = record

        if disabled_workflow != workflow or disabled_platform != current_platform:
            # The current workflow or platform is not disabled by this record
            continue

        # The logic after this is fairly complicated:
        #
        # - If the disabled record doesn't have the optional job (config) name,
        #   i.e. pull / linux-bionic-py3.8-clang9, all build and test jobs will
        #   be skipped
        #
        # - If the disabled record has the job name and it's a build job, i.e.
        #   pull / linux-bionic-py3.8-clang9 / build, all build and test jobs
        #   will be skipped, because the latter requires the former
        #
        # - If the disabled record has the job name and it's a test job without
        #   the config part, i.e. pull / linux-bionic-py3.8-clang9 / test, all
        #   test jobs will be skipped. TODO: At the moment, the script uses the
        #   short-circuiting logic to skip the build job automatically when there
        #   is no test job assuming that it would be a waste of effort building
        #   for nothing. This might not be the desirable behavior, and could be
        #   fixed later if needed
        #
        # - If the disabled record has the job (config) name, only that test config
        #   will be skipped, i.e. pull / linux-bionic-py3.8-clang9 / test (dynamo)
        if not disabled_job_cfg:
            print(
                f"Issue {disabled_url} created by {author} has disabled all CI jobs for {workflow} / {job_name}"
            )
            return filtered_test_matrix

        if disabled_job_cfg == BUILD_JOB_NAME:
            print(
                f"Issue {disabled_url} created by {author} has disabled the build job for {workflow} / {job_name}"
            )
            return filtered_test_matrix

        if (
            disabled_job_cfg == TEST_JOB_NAME
            or disabled_job_cfg == BUILD_AND_TEST_JOB_NAME
        ):
            print(
                f"Issue {disabled_url} created by {author} has disabled all the test jobs for {workflow} / {job_name}"
            )
            return filtered_test_matrix

        m = JOB_NAME_CFG_REGEX.match(disabled_job_cfg)
        if m:
            disabled_job = m.group("job")
            # Make sure that the job name is a valid test job name first before checking the config
            if disabled_job == TEST_JOB_NAME or disabled_job == BUILD_AND_TEST_JOB_NAME:
                disabled_cfg = m.group("cfg")
                # Remove the disabled config from the test matrix
                filtered_test_matrix["include"] = [
                    r
                    for r in test_matrix["include"]
                    if r.get("config", "") != disabled_cfg
                ]
                return filtered_test_matrix

        warnings.warn(
            f"Found a matching disabled issue {disabled_url} for {workflow} / {job_name}, "
            f"but the name {disabled_job_cfg} is invalid"
        )

    # Found no matching disabled issue, return the same input test matrix
    return test_matrix


def download_json(url: str, headers: Dict[str, str], num_retries: int = 3) -> Any:
    for _ in range(num_retries):
        try:
            req = Request(url=url, headers=headers)
            content = urlopen(req, timeout=5).read().decode("utf-8")
            return json.loads(content)
        except Exception as e:
            warnings.warn(f"Could not download {url}: {e}")

    warnings.warn(f"All {num_retries} retries exhausted, downloading {url} failed")
    return {}


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

    labels = set()
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

    if args.event_name == "schedule" and args.schedule == "29 8 * * *":
        # we don't want to run the mem leack check or disabled tests on normal
        # periodically scheduled jobs, only the ones at this time
        filtered_test_matrix = set_periodic_modes(filtered_test_matrix)

    if args.workflow and args.job_name:
        # If both workflow and job name are available, we will check if the current job
        # is disabled and remove it and all its dependants from the test matrix
        filtered_test_matrix = remove_disabled_jobs(
            args.workflow, args.job_name, filtered_test_matrix
        )

    # Set the filtered test matrix as the output
    set_output("test-matrix", json.dumps(filtered_test_matrix))

    filtered_test_matrix_len = len(filtered_test_matrix.get("include", []))
    # and also put a flag if the test matrix is empty, so subsequent jobs can
    # quickly check it without the need to parse the JSON string
    set_output("is-test-matrix-empty", filtered_test_matrix_len == 0)

    set_output("keep-going", "keep-going" in labels)


if __name__ == "__main__":
    main()
