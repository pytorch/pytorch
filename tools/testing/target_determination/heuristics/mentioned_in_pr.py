import os
import re
import subprocess
from typing import Any, List

import requests
from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
)
from tools.testing.test_run import TestRun


# This heuristic gives a test a rating of 1 if it is mentioned in the PR or a
# commit title.
class MentionedInPR(HeuristicInterface):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def get_prediction_confidence(self, tests: List[str]) -> TestPrioritizations:
        try:
            commit_messages = get_git_commit_info()
        except Exception as e:
            print(f"Can't get commit info due to {e}")
            commit_messages = ""
        try:
            pr_body = get_pr_body()
        except Exception as e:
            print(f"Can't get PR body due to {e}")
            pr_body = ""
        mentioned = []
        for test in tests:
            if test in commit_messages or test in pr_body:
                mentioned.append(test)
        return TestPrioritizations(tests, {TestRun(test): 1 for test in mentioned})


def get_git_commit_info() -> str:
    """Gets the commit info since the last commit on the default branch."""
    default_branch = f"origin/{os.environ.get('GIT_DEFAULT_BRANCH', 'main')}"

    merge_base = (
        subprocess.check_output(["git", "merge-base", default_branch, "HEAD"])
        .decode()
        .strip()
    )

    head = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()

    base_commit = merge_base
    if base_commit == head:
        # We are on the default branch, so check for changes since the last commit
        base_commit = "HEAD^"

    return (
        subprocess.check_output(
            ["git", "log", f"{base_commit}..HEAD"],
        )
        .decode()
        .strip()
    )


def get_pr_body() -> str:
    """Uses GitHub API to get the body of the PR, based on the PR_NUMBER or
    GITHUB_REF environment variables."""
    body = ""
    pr_number = os.environ.get("PR_NUMBER", "")
    if pr_number != "":
        body += requests.get(
            f"https://api.github.com/repos/pytorch/pytorch/pulls/{pr_number}"
        ).json()["body"]
    else:
        re_match = re.match(r"^refs/tags/.*/(\d+)$", os.environ.get("GITHUB_REF", ""))
        print(re_match)
        print(os.environ.get("GITHUB_REF", ""))
        if re_match is not None:
            print(re_match.group(1))
            body += requests.get(
                f"https://api.github.com/repos/pytorch/pytorch/pulls/{re_match.group(1)}"
            ).json()["body"]
    print(body)
    return body
