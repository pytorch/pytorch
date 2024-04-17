#!/usr/bin/env python3

import argparse
import json
import os
import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Set, Tuple

import requests
from setuptools import distutils  # type: ignore[import]

ALL_SKIPPED_THRESHOLD = 100
SIMILARITY_THRESHOLD = 0.75
FAILURE_CHAIN_THRESHOLD = 2
MAX_CONCURRENT_ALERTS = 1
FAILED_JOB_PATTERN = (
    r"^- \[(.*)\]\(.*\) failed consecutively starting with commit \[.*\]\(.*\)$"
)

PENDING = "pending"
NEUTRAL = "neutral"
SKIPPED = "skipped"
SUCCESS = "success"
FAILURE = "failure"
CANCELED = "canceled"

ISSUES_WITH_LABEL_QUERY = """
query ($owner: String!, $name: String!, $labels: [String!]) {
  repository(owner: $owner, name: $name, followRenames: false) {
    issues(last: 10, labels: $labels, states: [OPEN]) {
      nodes {
        id
        title
        closed
        number
        body
        createdAt
        comments(first: 100) {
          nodes {
            bodyText
            databaseId
          }
        }
      }
    }
  }
}
"""

NUM_ISSUES_QUERY = """
query ($query: String!) {
  search(type: ISSUE, query: $query) {
    issueCount
  }
}
"""

DISABLED_ALERTS = [
    "rerun_disabled_tests",
    "unstable",
]


class JobStatus:
    job_name: str = ""
    jobs: List[Any] = []
    current_status: Any = None
    job_statuses: List[Any] = []
    filtered_statuses: List[Any] = []
    failure_chain: List[Any] = []
    flaky_jobs: List[Any] = []

    def __init__(self, job_name: str, job_statuses: List[Any]):
        self.job_name = job_name
        self.job_statuses = job_statuses

        self.filtered_statuses = list(
            filter(lambda j: not is_job_skipped(j), job_statuses)
        )
        self.current_status = self.get_current_status()
        self.failure_chain = self.get_most_recent_failure_chain()
        self.flaky_jobs = self.get_flaky_jobs()

    def get_current_status(self) -> Any:
        """
        When getting the current status, we want the latest status which is not pending,
        be it success or failure
        """
        for status in self.filtered_statuses:
            if status["conclusion"] != PENDING:
                return status
        return None

    def get_unique_failures(self, jobs: List[Any]) -> Dict[str, List[Any]]:
        """
        Returns list of jobs grouped by failureCaptures from the input list
        """
        failures = defaultdict(list)
        for job in jobs:
            if job["conclusion"] == "failure":
                found_similar_failure = False
                if "failureCaptures" not in job:
                    failures["unclassified"] = [job]
                    continue

                # This is now a list returned by HUD API, not a string
                failureCaptures = " ".join(job["failureCaptures"])

                for failure in failures:
                    seq = SequenceMatcher(None, failureCaptures, failure)
                    if seq.ratio() > SIMILARITY_THRESHOLD:
                        failures[failure].append(job)
                        found_similar_failure = True
                        break
                if not found_similar_failure:
                    failures[failureCaptures] = [job]

        return failures

    # A flaky job is if it's the only job that has that failureCapture and is not the most recent job
    def get_flaky_jobs(self) -> List[Any]:
        unique_failures = self.get_unique_failures(self.filtered_statuses)
        flaky_jobs = []
        for failure in unique_failures:
            failure_list = unique_failures[failure]
            if (
                len(failure_list) == 1
                and failure_list[0]["sha"] != self.current_status["sha"]
            ):
                flaky_jobs.append(failure_list[0])
        return flaky_jobs

    # The most recent failure chain is an array of jobs that have the same-ish failures.
    # A success in the middle of the chain will terminate the chain.
    def get_most_recent_failure_chain(self) -> List[Any]:
        failures = []
        found_most_recent_failure = False

        for job in self.filtered_statuses:
            if is_job_failed(job):
                failures.append(job)
                found_most_recent_failure = True
            if found_most_recent_failure and not is_job_failed(job):
                break

        return failures

    def should_alert(self) -> bool:
        # Group jobs by their failures. The length of the failure chain is used
        # to raise the alert, so we can do a simple tweak here to use the length
        # of the longest unique chain
        unique_failures = self.get_unique_failures(self.failure_chain)

        return (
            self.current_status is not None
            and self.current_status["conclusion"] != SUCCESS
            and any(
                len(failure_chain) >= FAILURE_CHAIN_THRESHOLD
                for failure_chain in unique_failures.values()
            )
            and all(
                disabled_alert not in self.job_name
                for disabled_alert in DISABLED_ALERTS
            )
        )

    def __repr__(self) -> str:
        return f"jobName: {self.job_name}"


def fetch_hud_data(repo: str, branch: str) -> Any:
    response = requests.get(f"https://hud.pytorch.org/api/hud/{repo}/{branch}/0")
    response.raise_for_status()
    hud_data = json.loads(response.text)
    return (hud_data["jobNames"], hud_data["shaGrid"])


# Creates a Dict of Job Name -> [JobData]. Essentially a Column in HUD
def map_job_data(jobNames: Any, shaGrid: Any) -> Dict[str, Any]:
    jobData = defaultdict(list)
    for sha in shaGrid:
        for ind, job in enumerate(sha["jobs"]):
            jobData[jobNames[ind]].append(job)
    return jobData


def is_job_failed(job: Any) -> bool:
    conclusion = job["conclusion"] if "conclusion" in job else None
    return conclusion is not None and conclusion != SUCCESS and conclusion != PENDING


def is_job_skipped(job: Any) -> bool:
    conclusion = job["conclusion"] if "conclusion" in job else None
    return conclusion in (NEUTRAL, SKIPPED) or conclusion is None


def get_failed_jobs(job_data: List[Any]) -> List[Any]:
    return [job for job in job_data if job["conclusion"] == "failure"]


def classify_jobs(
    all_job_names: List[str], sha_grid: Any, filtered_jobs_names: Set[str]
) -> Tuple[List[JobStatus], List[Any]]:
    """
    Creates Job Statuses which has the logic for if need to alert or if there's flaky jobs.
    Classifies jobs into jobs to alert on and flaky jobs.
    :param all_job_names: list of all job names as returned by the HUD
    :param sha_grid: list of all job data as returned by the HUD (parallel index to all_job_names)
    :param filtered_jobs_names: set of job names to actually consider
    :return:
    """
    job_data = map_job_data(all_job_names, sha_grid)
    job_statuses: List[JobStatus] = []
    for job in job_data:
        job_statuses.append(JobStatus(job, job_data[job]))

    jobs_to_alert_on = []
    flaky_jobs = []

    for job_status in job_statuses:
        if job_status.job_name not in filtered_jobs_names:
            continue
        if job_status.should_alert():
            jobs_to_alert_on.append(job_status)
        flaky_jobs.extend(job_status.flaky_jobs)

    return jobs_to_alert_on, flaky_jobs


# filter job names that don't match the regex
def filter_job_names(job_names: List[str], job_name_regex: str) -> List[str]:
    if job_name_regex:
        return [
            job_name for job_name in job_names if re.match(job_name_regex, job_name)
        ]
    return job_names


def get_recurrently_failing_jobs_alerts(
    repo: str, branch: str, job_name_regex: str
) -> List[Dict[str, Any]]:
    job_names, sha_grid = fetch_hud_data(repo=repo, branch=branch)

    filtered_job_names = set(filter_job_names(job_names, job_name_regex))
    if job_name_regex:
        print()
        print(f"Filtered to {len(filtered_job_names)} jobs:")
        if len(filtered_job_names) == 0:
            print("No jobs matched the regex")
        elif len(filtered_job_names) == len(job_names):
            print("All jobs matched the regex")
        else:
            print("\n".join(filtered_job_names))

    (recurrently_failing_jobs, flaky_jobs) = classify_jobs(
        job_names, sha_grid, filtered_job_names
    )

    alerts = []
    for job in recurrently_failing_jobs:
        entry = {
            "AlertType": "Recurrently Failing Job",
            "AlertObject": job.job_name,
            "OncallTeams": [],
            "OncallIndividuals": [],
            "Flags": [],
            "sha": job.failure_chain[-1]["sha"],
            "branch": branch,
        }
        alerts.append(entry)
    return alerts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        help="Repository to do checks for",
        type=str,
        default=os.getenv("REPO_TO_CHECK", "pytorch/pytorch"),
    )
    parser.add_argument(
        "--branch",
        help="Branch to do checks for",
        type=str,
        default=os.getenv("BRANCH_TO_CHECK", "main"),
    )
    parser.add_argument(
        "--job-name-regex",
        help="Consider only job names matching given regex (if omitted, all jobs are matched)",
        type=str,
        default=os.getenv("JOB_NAME_REGEX", ""),
    )
    parser.add_argument(
        "--with-flaky-test-alert",
        help="Run this script with the flaky test alerting",
        type=distutils.util.strtobool,
        default=os.getenv("WITH_FLAKY_TEST_ALERT", "YES"),
    )
    parser.add_argument(
        "--dry-run",
        help="Whether or not to actually post issues",
        type=distutils.util.strtobool,
        default=os.getenv("DRY_RUN", "YES"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = json.dumps(
        get_recurrently_failing_jobs_alerts(args.repo, args.branch, args.job_name_regex)
    )

    print(data)
