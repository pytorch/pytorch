import json
import os
import re
import subprocess
from typing import Any, List
from urllib.request import Request, urlopen

from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
)
from tools.testing.test_run import TestRun

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")


# This heuristic gives a test a rating of 1 if it is mentioned in the PR or a
# commit title.
class MentionedInPR(HeuristicInterface):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def _search_for_linked_issues(self, s: str) -> List[str]:
        return re.findall(r"#(\d+)", s) + re.findall(r"/pytorch/pytorch/.*/(\d+)", s)

    def get_prediction_confidence(self, tests: List[str]) -> TestPrioritizations:
        try:
            commit_messages = get_git_commit_info()
        except Exception as e:
            print(f"Can't get commit info due to {e}")
            commit_messages = ""
        try:
            pr_number = os.environ.get("PR_NUMBER", "")
            if pr_number == "":
                re_match = re.match(
                    r"^refs/tags/.*/(\d+)$", os.environ.get("GITHUB_REF", "")
                )
                if re_match is not None:
                    pr_number = re_match.group(1)
            pr_body = get_issue_or_pr_body(int(pr_number))
        except Exception as e:
            print(f"Can't get PR body due to {e}")
            pr_body = ""

        # Search for linked issues or PRs
        linked_issue_bodies: List[str] = []
        for issue in self._search_for_linked_issues(
            commit_messages
        ) + self._search_for_linked_issues(pr_body):
            try:
                linked_issue_bodies.append(get_issue_or_pr_body(int(issue)))
            except Exception as e:
                pass

        mentioned = []
        for test in tests:
            if (
                test in commit_messages
                or test in pr_body
                or any(test in body for body in linked_issue_bodies)
            ):
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


def get_issue_or_pr_body(number: int) -> str:
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {GITHUB_TOKEN}",
    }
    # Despite the 'issues' in the link, this also works for PRs
    url = f"https://api.github.com/repos/pytorch/pytorch/issues/{number}"
    with urlopen(Request(url, headers=headers)) as conn:
        body: str = json.loads(conn.read().decode())["body"]
        return body
