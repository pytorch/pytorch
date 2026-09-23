# Helper to get the id of the currently running job in a GitHub Actions
# workflow. GitHub does not provide this information to workflow runs, so we
# need to figure it out based on what they *do* provide.
from __future__ import annotations

import argparse
import json
import operator
import os
import re
import sys
import time
import urllib
import urllib.parse
from typing import Any, TYPE_CHECKING
from urllib.request import Request, urlopen


if TYPE_CHECKING:
    from collections.abc import Callable
    from email.message import Message


def parse_json_and_links(conn: Any) -> tuple[Any, dict[str, dict[str, str]]]:
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
            params = {
                k: v[0].strip('"')
                for k, v in qparams.items()
                if type(v) is list and len(v) > 0
            }
            params["url"] = url
            if "rel" in params:
                links[params["rel"]] = params

    return json.load(conn), links


def retry_delay(
    status_code: int, response_headers: Message, backoff_timeout: float
) -> float:
    # Secondary rate limits carry Retry-After, and asking again before it elapses
    # is what escalates them, so never wait less than it asks. Only GitHub's own
    # rate-limit statuses count: proxies attach the header to 503s that the short
    # backoff can legitimately clear.
    if status_code not in (403, 429):
        return backoff_timeout
    # isdecimal rather than isdigit: headers arrive latin-1 decoded, and the
    # superscript digits in that range satisfy isdigit but raise in float.
    retry_after = response_headers.get("Retry-After", "").strip()
    if retry_after.isdecimal():
        return max(backoff_timeout, float(retry_after))
    return backoff_timeout


def fetch_url(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    reader: Callable[[Any], Any] = lambda x: x.read(),
    retries: int | None = 3,
    backoff_timeout: float = 0.5,
) -> Any:
    if headers is None:
        headers = {}
    try:
        with urlopen(Request(url, headers=headers)) as conn:
            return reader(conn)
    except urllib.error.HTTPError as err:
        # A depleted core quota resets on GitHub's schedule, up to an hour out, so
        # retrying inside the backoff cannot succeed and only multiplies the burn
        # against a limit every concurrent job in the repo shares.
        quota_depleted = err.headers.get("X-RateLimit-Remaining", "").strip() == "0"
        if not quota_depleted and isinstance(retries, (int, float)) and retries > 0:
            delay = retry_delay(err.code, err.headers, backoff_timeout)
            time.sleep(delay)
            return fetch_url(
                url,
                headers=headers,
                reader=reader,
                # Waiting out a Retry-After already gave GitHub the pause it asked
                # for; repeating it would multiply a minute-scale stall by the retry
                # count inside a step that cannot time itself out.
                retries=0 if delay > backoff_timeout else retries - 1,
                backoff_timeout=backoff_timeout,
            )
        exception_message = (
            "Is github alright?",
            f"Received status code '{err.code}' when attempting to retrieve {url}:\n",
            f"{err.reason}\n\nheaders={err.headers}",
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


def fetch_jobs(url: str, headers: dict[str, str]) -> list[dict[str, str]]:
    response, links = fetch_url(url, headers=headers, reader=parse_json_and_links)
    jobs = response["jobs"]
    if type(jobs) is not list:
        raise AssertionError(f"Expected jobs to be a list, got {type(jobs).__name__}")
    while "next" in links:
        response, links = fetch_url(
            links["next"]["url"], headers=headers, reader=parse_json_and_links
        )
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


def find_job_id_name(args: Any) -> tuple[str, str]:
    # From https://docs.github.com/en/actions/learn-github-actions/environment-variables
    PYTORCH_REPO = os.environ.get("GITHUB_REPOSITORY", "pytorch/pytorch")
    PYTORCH_GITHUB_API = f"https://api.github.com/repos/{PYTORCH_REPO}"
    GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
    REQUEST_HEADERS = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": "token " + GITHUB_TOKEN,
    }

    url = f"{PYTORCH_GITHUB_API}/actions/runs/{args.workflow_run_id}/jobs?per_page=100"
    jobs = fetch_jobs(url, REQUEST_HEADERS)

    # Sort the jobs list by start time, in descending order. We want to get the most
    # recently scheduled job on the runner.
    jobs.sort(key=operator.itemgetter("started_at"), reverse=True)

    for job in jobs:
        if job["runner_name"] == args.runner_name:
            return (job["id"], job["name"])

    raise RuntimeError(f"Can't find job id for runner {args.runner_name}")


def set_output(name: str, val: Any) -> None:
    print(f"Setting output {name}={val}")
    if os.getenv("GITHUB_OUTPUT"):
        with open(str(os.getenv("GITHUB_OUTPUT")), "a") as env:
            print(f"{name}={val}", file=env)
    else:
        print(f"::set-output name={name}::{val}")


def main() -> None:
    args = parse_args()
    try:
        # Get both the job ID and job name because we have already spent a request
        # here to get the job info
        job_id, job_name = find_job_id_name(args)
        set_output("job-id", job_id)
        set_output("job-name", job_name)
    except Exception as e:
        print(repr(e), file=sys.stderr)
        print(f"workflow-{args.workflow_run_id}")


if __name__ == "__main__":
    main()
