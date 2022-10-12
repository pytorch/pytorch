# Helper to get the id of the currently running job in a GitHub Actions
# workflow. GitHub does not provide this information to workflow runs, so we
# need to figure it out based on what they *do* provide.

import requests
import os
import argparse

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
parser = argparse.ArgumentParser()
parser.add_argument(
    "workflow_run_id", help="The id of the workflow run, should be GITHUB_RUN_ID"
)
parser.add_argument(
    "runner_name",
    help="The name of the runner to retrieve the job id, should be RUNNER_NAME",
)

args = parser.parse_args()


# From https://docs.github.com/en/actions/learn-github-actions/environment-variables
PYTORCH_REPO = os.environ.get("GITHUB_REPOSITORY", "pytorch/pytorch")
PYTORCH_GITHUB_API = f"https://api.github.com/repos/{PYTORCH_REPO}"
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
REQUEST_HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    "Authorization": "token " + GITHUB_TOKEN,
}

response = requests.get(
    f"{PYTORCH_GITHUB_API}/actions/runs/{args.workflow_run_id}/jobs?per_page=100",
    headers=REQUEST_HEADERS,
)

jobs = response.json()["jobs"]
while "next" in response.links.keys():
    response = requests.get(response.links["next"]["url"], headers=REQUEST_HEADERS)
    jobs.extend(response.json()["jobs"])

# Sort the jobs list by start time, in descending order. We want to get the most
# recently scheduled job on the runner.
jobs.sort(key=lambda job: job["started_at"], reverse=True)

for job in jobs:
    if job["runner_name"] == args.runner_name:
        print(job["id"])
        exit(0)

exit(1)
