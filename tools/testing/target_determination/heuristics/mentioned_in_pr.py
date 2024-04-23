import os
import re
from typing import Any, List

from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
)
from tools.testing.target_determination.heuristics.utils import (
    get_git_commit_info,
    get_issue_or_pr_body,
)
from tools.testing.test_run import TestRun


# This heuristic searches the PR body and commit titles, as well as issues/PRs
# mentioned in the PR body/commit title for test names (search depth of 1) and
# gives the test a rating of 1.  For example, if I mention "test_foo" in the PR
# body, test_foo will be rated 1.  If I mention #123 in the PR body, and #123
# mentions "test_foo", test_foo will be rated 1.
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
