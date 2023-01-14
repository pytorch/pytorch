# Helper to get the id of the currently running job in a GitHub Actions
# workflow. GitHub does not provide this information to workflow runs, so we
# need to figure it out based on what they *do* provide.

import argparse
import json
import os
import re
import urllib
import urllib.parse

from typing import Any, Callable, Dict, List, Tuple, Optional
from urllib.request import Request, urlopen

def parse_json_and_links(conn: Any) -> Tuple[Any, Dict[str, Dict[str, str]]]:
    links = {}
    # Extract links which GH uses for pagination
    # see https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Link
    if "Link" in conn.headers:
        for elem in re.split(", *<", conn.headers["Link"]):
            try:
                url, params_ = elem.split(";", 1)
            except ValueError:
                continue
            url = urllib.parse.unquote(url.strip("<> "))
            qparams = urllib.parse.parse_qs(params_.strip(), separator=";")
            params = {k: v[0].strip('"') for k, v in qparams.items() if type(v) is list and len(v) > 0}
            params["url"] = url
            if "rel" in params:
                links[params["rel"]] = params

    return json.load(conn), links

def fetch_url(url: str, *,
              headers: Optional[Dict[str, str]] = None,
              reader: Callable[[Any], Any] = lambda x: x.read()) -> Any:
    if headers is None:
        headers = {}
    try:
        with urlopen(Request(url, headers=headers)) as conn:
            return reader(conn)
    except urllib.error.HTTPError as err:
        exception_message = (
            "Is github alright?",
            f"Recieved status code '{err.code}' when attempting to retrieve {url}:\n",
            f"{err.reason}\n\nheaders={err.headers}"
        )
        raise RuntimeError(exception_message) from err

def parse_args() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "workflow_run_id", help="The id of the workflow run, should be GITHUB_RUN_ID"
    )
    parser.add_argument(
        "runner_name",
        help="The name of the runner to retrieve the job id, should be RUNNER_NAME",
    )

    return parser.parse_args()


def fetch_jobs(url: str, headers: Dict[str, str]) -> List[Dict[str, str]]:
    response, links = fetch_url(url, headers=headers, reader=parse_json_and_links)
    jobs = response["jobs"]
    assert type(jobs) is list
    while "next" in links.keys():
        response, links = fetch_url(links["next"]["url"], headers=headers, reader=parse_json_and_links)
        jobs.extend(response["jobs"])

    return jobs


# Our strategy is to retrieve the parent workflow run, then filter its jobs on
# RUNNER_NAME to figure out which job we're currently running.
#
# Why RUNNER_NAME? Because it's the only thing that uniquely identifies a job within a workflow.
# GITHUB_JOB doesn't work, as it corresponds to the job yaml id
# (https://bit.ly/37e78oI), which has two problems:
# 1. It's not present in the workflow job JSON object, so we can't use it as a filter.
# 2. It isn't unique; for matrix jobs the job yaml id is the same for all jobs in the matrix.
#
# RUNNER_NAME on the other hand is unique across the pool of runners. Also,
# since only one job can be scheduled on a runner at a time, we know that
# looking for RUNNER_NAME will uniquely identify the job we're currently
# running.

def main() -> None:
    # From https://docs.github.com/en/actions/learn-github-actions/environment-variables
    PYTORCH_REPO = os.environ.get("GITHUB_REPOSITORY", "pytorch/pytorch")
    PYTORCH_GITHUB_API = f"https://api.github.com/repos/{PYTORCH_REPO}"
    GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
    REQUEST_HEADERS = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": "token " + GITHUB_TOKEN,
    }

    args = parse_args()
    url = f"{PYTORCH_GITHUB_API}/actions/runs/{args.workflow_run_id}/jobs?per_page=100"
    jobs = fetch_jobs(url, REQUEST_HEADERS)

    # Sort the jobs list by start time, in descending order. We want to get the most
    # recently scheduled job on the runner.
    jobs.sort(key=lambda job: job["started_at"], reverse=True)

    for job in jobs:
        if job["runner_name"] == args.runner_name:
            print(job["id"])
            return

    exit(1)

if __name__ == "__main__":
    main()
