#!/usr/bin/env python3

import argparse
import json
import os
import re
import urllib.parse
from collections import defaultdict
from datetime import datetime, timedelta
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

REPO_OWNER = "pytorch"
PYTORCH_REPO_NAME = "pytorch"
TEST_INFRA_REPO_NAME = "test-infra"
PYTORCH_ALERT_LABEL = "pytorch-alert"
FLAKY_TESTS_LABEL = "module: flaky-tests"
NO_FLAKY_TESTS_LABEL = "no-flaky-tests-alert"
FLAKY_TESTS_SEARCH_PERIOD_DAYS = 14
DISABLED_ALERTS = [
    "rerun_disabled_tests",
    "unstable",
]

headers = {"Authorization": f"token {os.environ.get('GITHUB_TOKEN')}"}
CREATE_ISSUE_URL = (
    f"https://api.github.com/repos/{REPO_OWNER}/{TEST_INFRA_REPO_NAME}/issues"
)
UPDATE_ISSUE_URL = (
    f"https://api.github.com/repos/{REPO_OWNER}/{TEST_INFRA_REPO_NAME}/issues/"
)

GRAPHQL_URL = "https://api.github.com/graphql"


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


def fetch_alerts(
    repo: str, branch: str, alert_repo: str, labels: List[str]
) -> List[Any]:
    try:
        variables = {"owner": REPO_OWNER, "name": alert_repo, "labels": labels}
        r = requests.post(
            GRAPHQL_URL,
            json={"query": ISSUES_WITH_LABEL_QUERY, "variables": variables},
            headers=headers,
        )
        r.raise_for_status()

        data = json.loads(r.text)
        # Return only alert belonging to the target repo and branch
        return list(
            filter(
                lambda alert: f"Recurrently Failing Jobs on {repo} {branch}"
                in alert["title"],
                data["data"]["repository"]["issues"]["nodes"],
            )
        )
    except Exception as e:
        raise RuntimeError("Error fetching alerts", e)


def get_num_issues_with_label(owner: str, repo: str, label: str, from_date: str) -> int:
    query = f'repo:{owner}/{repo} label:"{label}" created:>={from_date} is:issue'
    try:
        r = requests.post(
            GRAPHQL_URL,
            json={"query": NUM_ISSUES_QUERY, "variables": {"query": query}},
            headers=headers,
        )
        r.raise_for_status()
        data = json.loads(r.text)
        return data["data"]["search"]["issueCount"]
    except Exception as e:
        raise RuntimeError("Error fetching issues count", e)


def generate_failed_job_hud_link(failed_job: JobStatus) -> str:
    # TODO: I don't think minihud is universal across multiple repositories
    #       would be good to just replace this with something that is
    hud_link = "https://hud.pytorch.org/minihud?name_filter=" + urllib.parse.quote(
        failed_job.job_name
    )
    return f"[{failed_job.job_name}]({hud_link})"


def generate_failed_job_issue(
    repo: str, branch: str, failed_jobs: List[JobStatus]
) -> Any:
    failed_jobs.sort(key=lambda status: status.job_name)
    issue = {}
    issue[
        "title"
    ] = f"[Pytorch] There are {len(failed_jobs)} Recurrently Failing Jobs on {repo} {branch}"
    body = "Within the last 50 commits, there are the following failures on the main branch of pytorch: \n"
    for job in failed_jobs:
        failing_sha = job.failure_chain[-1]["sha"]
        body += (
            f"- {generate_failed_job_hud_link(job)} failed consecutively starting with "
        )
        body += f"commit [{failing_sha}](https://hud.pytorch.org/commit/{repo}/{failing_sha})"
        body += "\n\n"

    body += "Please review the errors and revert if needed."
    issue["body"] = body
    issue["labels"] = [PYTORCH_ALERT_LABEL]

    print("Generating alerts for: ", failed_jobs)
    return issue


def gen_update_comment(original_body: str, jobs: List[JobStatus]) -> str:
    """
    Returns empty string if nothing signficant changed. Otherwise returns a
    short string meant for updating the issue.
    """
    original_jobs = []
    for line in original_body.splitlines():
        match = re.match(FAILED_JOB_PATTERN, line.strip())
        if match is not None:
            original_jobs.append(match.group(1))

    new_jobs = [job.job_name for job in jobs]
    stopped_failing_jobs = [job for job in original_jobs if job not in new_jobs]
    started_failing_jobs = [job for job in new_jobs if job not in original_jobs]

    # TODO: Add real HUD links to these eventually since not having clickable links is bad
    s = ""
    if len(stopped_failing_jobs) > 0:
        s += "These jobs stopped failing:\n"
        for job in stopped_failing_jobs:
            s += f"* {job}\n"
        s += "\n"
    if len(started_failing_jobs) > 0:
        s += "These jobs started failing:\n"
        for job in started_failing_jobs:
            s += f"* {job}\n"
    return s


def generate_no_flaky_tests_issue() -> Any:
    issue = {}
    issue[
        "title"
    ] = f"[Pytorch][Warning] No flaky test issues have been detected in the past {FLAKY_TESTS_SEARCH_PERIOD_DAYS} days!"
    issue["body"] = (
        f"No issues have been filed in the past {FLAKY_TESTS_SEARCH_PERIOD_DAYS} days for "
        f"the repository {REPO_OWNER}/{TEST_INFRA_REPO_NAME}.\n"
        "This can be an indication that the flaky test bot has stopped filing tests."
    )
    issue["labels"] = [NO_FLAKY_TESTS_LABEL]

    return issue


def update_issue(
    issue: Dict, old_issue: Any, update_comment: str, dry_run: bool
) -> None:
    print(f"Updating issue {issue} with content:{os.linesep}{update_comment}")
    if dry_run:
        print("NOTE: Dry run, not doing any real work")
        return
    r = requests.patch(
        UPDATE_ISSUE_URL + str(old_issue["number"]), json=issue, headers=headers
    )
    r.raise_for_status()
    r = requests.post(
        f"https://api.github.com/repos/{REPO_OWNER}/{TEST_INFRA_REPO_NAME}/issues/{old_issue['number']}/comments",
        data=json.dumps({"body": update_comment}),
        headers=headers,
    )
    r.raise_for_status()


def create_issue(issue: Dict, dry_run: bool) -> Dict:
    print(f"Creating issue with content:{os.linesep}{issue}")
    if dry_run:
        print("NOTE: Dry run activated, not doing any real work")
        return
    r = requests.post(CREATE_ISSUE_URL, json=issue, headers=headers)
    r.raise_for_status()
    return issue


def fetch_hud_data(repo: str, branch: str) -> Any:
    response = requests.get(f"https://hud.pytorch.org/api/hud/{repo}/{branch}/0")
    response.raise_for_status()
    hud_data = json.loads(response.text)
    return (hud_data["jobNames"], hud_data["shaGrid"])


# TODO: Do something about these flaky jobs, save them in rockset or something
def record_flaky_jobs(flaky_jobs: List[Any]) -> None:
    return


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
    return conclusion is None or conclusion == NEUTRAL or conclusion == SKIPPED


def get_failed_jobs(job_data: List[Any]) -> List[Any]:
    return [job for job in job_data if job["conclusion"] == "failure"]


def categorize_shas(sha_grid: Any) -> List[Tuple[Any, str]]:
    categorized_shas = []
    for sha in sha_grid:
        conclusions = defaultdict(lambda: 0)
        for job in sha["jobs"]:
            if "conclusion" in job:
                conclusions[job["conclusion"]] += 1
            else:
                conclusions[SKIPPED] += 1
        if conclusions[FAILURE] > 0 or conclusions[CANCELED]:
            categorized_shas.append((sha, FAILURE))
        elif conclusions[PENDING] > 0:
            categorized_shas.append((sha, PENDING))
        # If the SHA has 100+ skipped jobs, then that means this SHA is part of a stack and
        # everything in this commit is skipped
        elif conclusions[SKIPPED] > ALL_SKIPPED_THRESHOLD:
            categorized_shas.append((sha, SKIPPED))
        else:
            categorized_shas.append((sha, SUCCESS))
    return categorized_shas


def find_first_sha(categorized_sha: List[Tuple[str, str]], status: str):
    for ind, sha in enumerate(categorized_sha):
        if sha[1] == status:
            return ind
    return -1


def clear_alerts(alerts: List[Any], dry_run: bool) -> bool:
    if dry_run:
        print("NOTE: Dry run, not doing any real work")
        return
    cleared_alerts = 0
    for alert in alerts:
        r = requests.patch(
            UPDATE_ISSUE_URL + str(alert["number"]),
            json={"state": "closed"},
            headers=headers,
        )
        r.raise_for_status()
        cleared_alerts += 1
    print(f"Clearing {cleared_alerts} alerts.")
    return cleared_alerts > 0


# We need to clear alerts if there is a commit that's all green is before a commit that has a red
# If there's pending things after the all green commit, that's fine, as long as it's all green/pending
def trunk_is_green(sha_grid: Any):
    categorized_shas = categorize_shas(sha_grid)
    first_green_sha_ind = find_first_sha(categorized_shas, SUCCESS)
    first_red_sha_ind = find_first_sha(categorized_shas, FAILURE)
    first_green = categorized_shas[first_green_sha_ind][0]
    first_red = categorized_shas[first_red_sha_ind][0]

    print(
        f"The first green SHA was at index {first_green_sha_ind} at {first_green['sha']}"
        + f"and the first red SHA was at index {first_red_sha_ind} at {first_red['sha']}"
    )
    if first_green_sha_ind < 0:
        return False
    return first_green_sha_ind < first_red_sha_ind


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
    job_statuses: list[JobStatus] = []
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


def handle_flaky_tests_alert(existing_alerts: List[Dict]) -> Dict:
    if (
        not existing_alerts
        or datetime.fromisoformat(
            existing_alerts[0]["createdAt"].replace("Z", "+00:00")
        ).date()
        != datetime.today().date()
    ):
        from_date = (
            datetime.today() - timedelta(days=FLAKY_TESTS_SEARCH_PERIOD_DAYS)
        ).strftime("%Y-%m-%d")
        num_issues_with_flaky_tests_lables = get_num_issues_with_label(
            REPO_OWNER, PYTORCH_REPO_NAME, FLAKY_TESTS_LABEL, from_date
        )
        print(
            f"Num issues with `{FLAKY_TESTS_LABEL}` label: ",
            num_issues_with_flaky_tests_lables,
        )
        if num_issues_with_flaky_tests_lables == 0:
            return create_issue(generate_no_flaky_tests_issue(), False)

    print("No new alert for flaky tests bots.")
    return None


# filter job names that don't match the regex
def filter_job_names(job_names: List[str], job_name_regex: str) -> List[str]:
    if job_name_regex:
        return [
            job_name for job_name in job_names if re.match(job_name_regex, job_name)
        ]
    return job_names


def check_for_recurrently_failing_jobs_alert(
    repo: str, branch: str, job_name_regex: str, dry_run: bool
):
    job_names, sha_grid = fetch_hud_data(repo=repo, branch=branch)
    print(f"Found {len(job_names)} jobs for {repo} {branch} branch:")
    print("\n".join(job_names))

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

    (jobs_to_alert_on, flaky_jobs) = classify_jobs(
        job_names, sha_grid, filtered_job_names
    )

    # Fetch alerts
    existing_alerts = fetch_alerts(
        repo=repo,
        branch=branch,
        alert_repo=TEST_INFRA_REPO_NAME,
        labels=PYTORCH_ALERT_LABEL,
    )

    # Auto-clear any existing alerts if the current status is green
    if len(jobs_to_alert_on) == 0 or trunk_is_green(sha_grid):
        print(f"Didn't find anything to alert on for {repo} {branch}")
        clear_alerts(existing_alerts, dry_run=dry_run)
        return

    # In the current design, there should be at most one alert issue per repo and branch
    assert len(existing_alerts) <= 1

    if existing_alerts:
        # New update, edit the current active alert
        existing_issue = existing_alerts[0]
        update_comment = gen_update_comment(existing_issue["body"], jobs_to_alert_on)

        if update_comment:
            new_issue = generate_failed_job_issue(
                repo=repo, branch=branch, failed_jobs=jobs_to_alert_on
            )
            update_issue(new_issue, existing_issue, update_comment, dry_run=dry_run)
        else:
            print(f"No new change. Not updating any alert for {repo} {branch}")
    else:
        # No active alert exists, create a new one
        create_issue(
            generate_failed_job_issue(
                repo=repo, branch=branch, failed_jobs=jobs_to_alert_on
            ),
            dry_run=dry_run,
        )


def check_for_no_flaky_tests_alert(repo: str, branch: str):
    existing_no_flaky_tests_alerts = fetch_alerts(
        repo=repo,
        branch=branch,
        alert_repo=TEST_INFRA_REPO_NAME,
        labels=NO_FLAKY_TESTS_LABEL,
    )
    handle_flaky_tests_alert(existing_no_flaky_tests_alerts)
    

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


def main():
    args = parse_args()
    check_for_recurrently_failing_jobs_alert(
        args.repo, args.branch, args.job_name_regex, args.dry_run
    )
    # TODO: Fill out dry run for flaky test alerting, not going to do in one PR
    if args.with_flaky_test_alert:
        check_for_no_flaky_tests_alert(args.repo, args.branch)


if __name__ == "__main__":
    main()
