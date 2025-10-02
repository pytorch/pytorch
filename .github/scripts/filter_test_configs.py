#!/usr/bin/env python3
# ruff: noqa: LOG015

import json
import logging
import os
import re
import subprocess
import sys
import warnings
from enum import Enum
from functools import cache
from logging import info
from typing import Any, Callable, Optional
from urllib.request import Request, urlopen

import yaml


REENABLE_TEST_REGEX = "(?i)(Close(d|s)?|Resolve(d|s)?|Fix(ed|es)?) (#|https://github.com/pytorch/pytorch/issues/)([0-9]+)"
MAIN_BRANCH = "main"

PREFIX = "test-config/"

logging.basicConfig(level=logging.INFO)


def is_cuda_or_rocm_job(job_name: Optional[str]) -> bool:
    if not job_name:
        return False

    return "cuda" in job_name or "rocm" in job_name


# Supported modes when running periodically. Only applying the mode when
# its lambda condition returns true
SUPPORTED_PERIODICAL_MODES: dict[str, Callable[[Optional[str]], bool]] = {
    # Memory leak check is only needed for CUDA and ROCm jobs which utilize GPU memory
    "mem_leak_check": is_cuda_or_rocm_job,
    "rerun_disabled_tests": lambda job_name: True,
}

# The link to the published list of disabled jobs
DISABLED_JOBS_URL = "https://ossci-metrics.s3.amazonaws.com/disabled-jobs.json"
# and unstable jobs
UNSTABLE_JOBS_URL = "https://ossci-metrics.s3.amazonaws.com/unstable-jobs.json"

# Some constants used to handle disabled and unstable jobs
JOB_NAME_SEP = "/"
BUILD_JOB_NAME = "build"
TEST_JOB_NAME = "test"
BUILD_AND_TEST_JOB_NAME = "build-and-test"
JOB_NAME_CFG_REGEX = re.compile(r"(?P<job>[\w-]+)\s+\((?P<cfg>[\w-]+)\)")
EXCLUDED_BRANCHES = ["nightly"]
MEM_LEAK_LABEL = "enable-mem-leak-check"


class IssueType(Enum):
    DISABLED = "disabled"
    UNSTABLE = "unstable"


def parse_args() -> Any:
    from argparse import ArgumentParser

    parser = ArgumentParser(
        "Filter all test configurations and keep only requested ones"
    )
    parser.add_argument(
        "--test-matrix", type=str, required=True, help="the original test matrix"
    )
    parser.add_argument(
        "--selected-test-configs",
        type=str,
        default="",
        help="a comma-separated list of test configurations from the test matrix to keep",
    )
    parser.add_argument(
        "--workflow", type=str, help="the name of the current workflow, i.e. pull"
    )
    parser.add_argument(
        "--job-name",
        type=str,
        help="the name of the current job, i.e. linux-jammy-py3.8-gcc7 / build",
    )
    parser.add_argument("--pr-number", type=str, help="the pull request number")
    parser.add_argument("--tag", type=str, help="the associated tag if it exists")
    parser.add_argument(
        "--event-name",
        type=str,
        help="name of the event that triggered the job (pull, schedule, etc)",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        help="cron schedule that triggered the job",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=MAIN_BRANCH,
        help="the branch name",
    )
    return parser.parse_args()


@cache
def get_pr_info(pr_number: int) -> dict[str, Any]:
    """
    Dynamically get PR information
    """
    # From https://docs.github.com/en/actions/learn-github-actions/environment-variables
    pytorch_repo = os.environ.get("GITHUB_REPOSITORY", "pytorch/pytorch")
    pytorch_github_api = f"https://api.github.com/repos/{pytorch_repo}"
    github_token = os.environ["GITHUB_TOKEN"]

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {github_token}",
    }
    json_response: dict[str, Any] = download_json(
        url=f"{pytorch_github_api}/issues/{pr_number}",
        headers=headers,
    )

    if not json_response:
        warnings.warn(f"Failed to get the labels for #{pr_number}")
        return {}

    return json_response


def get_labels(pr_number: int) -> set[str]:
    """
    Dynamically get the latest list of labels from the pull request
    """
    pr_info = get_pr_info(pr_number)
    return {
        label.get("name") for label in pr_info.get("labels", []) if label.get("name")
    }


def filter_labels(labels: set[str], label_regex: Any) -> set[str]:
    """
    Return the list of matching labels
    """
    return {l for l in labels if re.match(label_regex, l)}


def filter(test_matrix: dict[str, list[Any]], labels: set[str]) -> dict[str, list[Any]]:
    """
    Select the list of test config to run from the test matrix. The logic works
    as follows:

    If the PR has one or more test-config labels as specified, only these test configs
    will be selected.  This also works with ciflow labels, for example, if a PR has both
    ciflow/trunk and test-config/functorch, only trunk functorch builds and tests will
    be run.

    If the PR has none of the test-config label, all tests are run as usual.
    """
    filtered_test_matrix: dict[str, list[Any]] = {"include": []}

    for entry in test_matrix.get("include", []):
        config_name = entry.get("config", "")
        if not config_name:
            continue

        label = f"{PREFIX}{config_name.strip()}"
        if label in labels:
            msg = f"Select {config_name} because label {label} is present in the pull request by the time the test starts"
            info(msg)
            filtered_test_matrix["include"].append(entry)

    test_config_labels = filter_labels(labels, re.compile(f"{PREFIX}.+"))
    if not filtered_test_matrix["include"] and not test_config_labels:
        info("Found no test-config label on the PR, so all test configs are included")
        # Found no test-config label and the filtered test matrix is empty, return the same
        # test matrix as before so that all tests can be run normally
        return test_matrix
    else:
        msg = f"Found {test_config_labels} on the PR so only these test configs are run"
        info(msg)
        # When the filter test matrix contain matches or if a valid test config label
        # is found in the PR, return the filtered test matrix
        return filtered_test_matrix


def filter_selected_test_configs(
    test_matrix: dict[str, list[Any]], selected_test_configs: set[str]
) -> dict[str, list[Any]]:
    """
    Keep only the selected configs if the list if not empty. Otherwise, keep all test configs.
    This filter is used when the workflow is dispatched manually.
    """
    if not selected_test_configs:
        return test_matrix

    filtered_test_matrix: dict[str, list[Any]] = {"include": []}
    for entry in test_matrix.get("include", []):
        config_name = entry.get("config", "")
        if not config_name:
            continue

        if config_name in selected_test_configs:
            filtered_test_matrix["include"].append(entry)

    return filtered_test_matrix


def set_periodic_modes(
    test_matrix: dict[str, list[Any]], job_name: Optional[str]
) -> dict[str, list[Any]]:
    """
    Apply all periodic modes when running under a schedule
    """
    scheduled_test_matrix: dict[str, list[Any]] = {
        "include": [],
    }

    for config in test_matrix.get("include", []):
        for mode, cond in SUPPORTED_PERIODICAL_MODES.items():
            if not cond(job_name):
                continue

            cfg = config.copy()
            cfg[mode] = mode
            scheduled_test_matrix["include"].append(cfg)

    return scheduled_test_matrix


def mark_unstable_jobs(
    workflow: str, job_name: str, test_matrix: dict[str, list[Any]]
) -> dict[str, list[Any]]:
    """
    Check the list of unstable jobs and mark them accordingly. Note that if a job
    is unstable, all its dependents will also be marked accordingly
    """
    return process_jobs(
        workflow=workflow,
        job_name=job_name,
        test_matrix=test_matrix,
        issue_type=IssueType.UNSTABLE,
        url=UNSTABLE_JOBS_URL,
    )


def remove_disabled_jobs(
    workflow: str, job_name: str, test_matrix: dict[str, list[Any]]
) -> dict[str, list[Any]]:
    """
    Check the list of disabled jobs, remove the current job and all its dependents
    if it exists in the list
    """
    return process_jobs(
        workflow=workflow,
        job_name=job_name,
        test_matrix=test_matrix,
        issue_type=IssueType.DISABLED,
        url=DISABLED_JOBS_URL,
    )


def _filter_jobs(
    test_matrix: dict[str, list[Any]],
    issue_type: IssueType,
    target_cfg: Optional[str] = None,
) -> dict[str, list[Any]]:
    """
    An utility function used to actually apply the job filter
    """
    # The result will be stored here
    filtered_test_matrix: dict[str, list[Any]] = {"include": []}

    # This is an issue to disable a CI job
    if issue_type == IssueType.DISABLED:
        # If there is a target config, disable (remove) only that
        if target_cfg:
            # Remove the target config from the test matrix
            filtered_test_matrix["include"] = [
                r for r in test_matrix["include"] if r.get("config", "") != target_cfg
            ]

        return filtered_test_matrix

    if issue_type == IssueType.UNSTABLE:
        for r in test_matrix["include"]:
            cpy = r.copy()

            if (target_cfg and r.get("config", "") == target_cfg) or not target_cfg:
                # If there is a target config, only mark that as unstable, otherwise,
                # mark everything as unstable
                cpy[IssueType.UNSTABLE.value] = IssueType.UNSTABLE.value

            filtered_test_matrix["include"].append(cpy)

        return filtered_test_matrix

    # No matching issue, return everything
    return test_matrix


def process_jobs(
    workflow: str,
    job_name: str,
    test_matrix: dict[str, list[Any]],
    issue_type: IssueType,
    url: str,
) -> dict[str, list[Any]]:
    """
    Both disabled and unstable jobs are in the following format:

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
        current_platform, _ = (n.strip() for n in job_name.split(JOB_NAME_SEP, 1) if n)
    except ValueError:
        warnings.warn(f"Invalid job name {job_name}, returning")
        return test_matrix

    for record in download_json(url=url, headers={}).values():
        (
            author,
            _,
            target_url,
            target_workflow,
            target_platform,
            target_job_cfg,
        ) = record

        if target_workflow != workflow:
            # The current workflow doesn't match this record
            continue

        cleanup_regex = rf"(-{BUILD_JOB_NAME}|-{TEST_JOB_NAME})$"
        # There is an exception here for binary build workflows in which the platform
        # names have the build and test suffix. For example, we have a build job called
        # manywheel-py3-cuda11_8-build / build and its subsequent test job called
        # manywheel-py3-cuda11_8-test / test. So they are linked, but their suffixes
        # are different
        target_platform_no_suffix = re.sub(cleanup_regex, "", target_platform)
        current_platform_no_suffix = re.sub(cleanup_regex, "", current_platform)

        if (
            target_platform != current_platform
            and target_platform_no_suffix != current_platform_no_suffix
        ):
            # The current platform doesn't match this record
            continue

        # The logic after this is fairly complicated:
        #
        # - If the target record doesn't have the optional job (config) name,
        #   i.e. pull / linux-bionic-py3.8-clang9, all build and test jobs will
        #   be skipped if it's a disabled record or marked as unstable if it's
        #   an unstable record
        #
        # - If the target record has the job name and it's a build job, i.e.
        #   pull / linux-bionic-py3.8-clang9 / build, all build and test jobs
        #   will be skipped if it's a disabled record or marked as unstable if
        #   it's an unstable record, because the latter requires the former
        #
        # - If the target record has the job name and it's a test job without
        #   the config part, i.e. pull / linux-bionic-py3.8-clang9 / test, all
        #   test jobs will be skipped if it's a disabled record or marked as
        #   unstable if it's an unstable record
        #
        # - If the target record has the job (config) name, only that test config
        #   will be skipped or marked as unstable
        if not target_job_cfg:
            msg = (
                f"Issue {target_url} created by {author} has {issue_type.value} "
                + f"all CI jobs for {workflow} / {job_name}"
            )
            info(msg)
            return _filter_jobs(
                test_matrix=test_matrix,
                issue_type=issue_type,
            )

        if target_job_cfg == BUILD_JOB_NAME:
            msg = (
                f"Issue {target_url} created by {author} has {issue_type.value} "
                + f"the build job for {workflow} / {job_name}"
            )
            info(msg)
            return _filter_jobs(
                test_matrix=test_matrix,
                issue_type=issue_type,
            )

        if target_job_cfg in (TEST_JOB_NAME, BUILD_AND_TEST_JOB_NAME):
            msg = (
                f"Issue {target_url} created by {author} has {issue_type.value} "
                + f"all the test jobs for {workflow} / {job_name}"
            )
            info(msg)
            return _filter_jobs(
                test_matrix=test_matrix,
                issue_type=issue_type,
            )

        m = JOB_NAME_CFG_REGEX.match(target_job_cfg)
        if m:
            target_job = m.group("job")
            # Make sure that the job name is a valid test job name first before checking the config
            if target_job in (TEST_JOB_NAME, BUILD_AND_TEST_JOB_NAME):
                target_cfg = m.group("cfg")

                # NB: There can be multiple unstable configurations, i.e. inductor, inductor_huggingface
                test_matrix = _filter_jobs(
                    test_matrix=test_matrix,
                    issue_type=issue_type,
                    target_cfg=target_cfg,
                )
        else:
            warnings.warn(
                f"Found a matching {issue_type.value} issue {target_url} for {workflow} / {job_name}, "
                + f"but the name {target_job_cfg} is invalid"
            )

    # Found no matching target, return the same input test matrix
    return test_matrix


def download_json(url: str, headers: dict[str, str], num_retries: int = 3) -> Any:
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
    print(f"Setting output {name}={val}")
    if os.getenv("GITHUB_OUTPUT"):
        with open(str(os.getenv("GITHUB_OUTPUT")), "a") as env:
            print(f"{name}={val}", file=env)
    else:
        print(f"::set-output name={name}::{val}")


def parse_reenabled_issues(s: Optional[str]) -> list[str]:
    # NB: When the PR body is empty, GitHub API returns a None value, which is
    # passed into this function
    if not s:
        return []

    # The regex is meant to match all *case-insensitive* keywords that
    # GitHub has delineated would link PRs to issues, more details here:
    # https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue.
    # E.g., "Close #62851", "fixES #62851" and "RESOLVED #62851" would all match, but not
    # "closes  #62851" --> extra space, "fixing #62851" --> not a keyword, nor "fix 62851" --> no #
    issue_numbers = [x[5] for x in re.findall(REENABLE_TEST_REGEX, s)]
    return issue_numbers


def get_reenabled_issues(pr_body: str = "") -> list[str]:
    default_branch = f"origin/{os.environ.get('GIT_DEFAULT_BRANCH', 'main')}"
    try:
        commit_messages = subprocess.check_output(
            f"git cherry -v {default_branch}".split(" ")
        ).decode("utf-8")
    except Exception as e:
        warnings.warn(f"failed to get commit messages: {e}")
        commit_messages = ""
    return parse_reenabled_issues(pr_body) + parse_reenabled_issues(commit_messages)


def check_for_setting(labels: set[str], body: str, setting: str) -> bool:
    return setting in labels or f"[{setting}]" in body


def perform_misc_tasks(
    labels: set[str],
    test_matrix: dict[str, list[Any]],
    job_name: str,
    pr_body: str,
    branch: Optional[str] = None,
    tag: Optional[str] = None,
) -> None:
    """
    In addition to apply the filter logic, the script also does the following
    misc tasks to set keep-going and is-unstable variables
    """
    set_output(
        "keep-going",
        branch == MAIN_BRANCH
        or bool(tag and re.match(r"^trunk/[a-f0-9]{40}$", tag))
        or check_for_setting(labels, pr_body, "keep-going"),
    )
    set_output(
        "ci-verbose-test-logs",
        check_for_setting(labels, pr_body, "ci-verbose-test-logs"),
    )
    set_output(
        "ci-test-showlocals", check_for_setting(labels, pr_body, "ci-test-showlocals")
    )
    set_output(
        "ci-no-test-timeout", check_for_setting(labels, pr_body, "ci-no-test-timeout")
    )
    set_output("ci-no-td", check_for_setting(labels, pr_body, "ci-no-td"))
    # Only relevant for the one linux distributed cuda job, delete this when TD
    # is rolled out completely
    set_output(
        "ci-td-distributed", check_for_setting(labels, pr_body, "ci-td-distributed")
    )

    # Obviously, if the job name includes unstable, then this is an unstable job
    is_unstable = job_name and IssueType.UNSTABLE.value in job_name
    if not is_unstable and test_matrix and test_matrix.get("include"):
        # Even when the job name doesn't mention unstable, we will also mark it as
        # unstable when the test matrix only includes unstable jobs. Basically, this
        # logic allows build or build-and-test jobs to be marked as unstable too.
        #
        # Basically, when a build job is unstable, all the subsequent test jobs are
        # also unstable. And when all test jobs are unstable, we will also treat the
        # build job as unstable. It's simpler this way
        is_unstable = all(IssueType.UNSTABLE.value in r for r in test_matrix["include"])

    set_output(
        "is-unstable",
        is_unstable,
    )

    set_output("reenabled-issues", ",".join(get_reenabled_issues(pr_body=pr_body)))

    if MEM_LEAK_LABEL in labels:
        # Enable mem leak check if label is added
        for config in test_matrix.get("include", []):
            if is_cuda_or_rocm_job(job_name):
                config["mem_leak_check"] = "mem_leak_check"


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
    tag_regex = re.compile(r"^ciflow/[\w\-]+/(?P<pr_number>\d+)$")

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

    if args.selected_test_configs:
        selected_test_configs = {
            v.strip().lower()
            for v in args.selected_test_configs.split(",")
            if v.strip()
        }
        filtered_test_matrix = filter_selected_test_configs(
            filtered_test_matrix, selected_test_configs
        )

    if args.event_name == "schedule" and args.schedule == "29 8 * * *":
        # we don't want to run the mem leak check or disabled tests on normal
        # periodically scheduled jobs, only the ones at this time
        filtered_test_matrix = set_periodic_modes(filtered_test_matrix, args.job_name)

    if args.workflow and args.job_name and args.branch not in EXCLUDED_BRANCHES:
        # If both workflow and job name are available, we will check if the current job
        # is disabled and remove it and all its dependants from the test matrix
        filtered_test_matrix = remove_disabled_jobs(
            args.workflow, args.job_name, filtered_test_matrix
        )

        filtered_test_matrix = mark_unstable_jobs(
            args.workflow, args.job_name, filtered_test_matrix
        )

    pr_body = get_pr_info(int(pr_number)).get("body", "") if pr_number else ""

    perform_misc_tasks(
        labels=labels,
        test_matrix=filtered_test_matrix,
        job_name=args.job_name,
        pr_body=pr_body if pr_body else "",
        branch=args.branch,
        tag=tag,
    )

    # Set the filtered test matrix as the output
    set_output("test-matrix", json.dumps(filtered_test_matrix))

    filtered_test_matrix_len = len(filtered_test_matrix.get("include", []))
    # and also put a flag if the test matrix is empty, so subsequent jobs can
    # quickly check it without the need to parse the JSON string
    set_output("is-test-matrix-empty", filtered_test_matrix_len == 0)


if __name__ == "__main__":
    main()
