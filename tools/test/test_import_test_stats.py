import os
import unittest
from typing import List
from unittest.mock import patch

from tools.stats.import_test_stats import get_disabled_issues


class TestGetDisabledIssues(unittest.TestCase):
    def run_assert_disabled_issues(
        self, pr_body: str, commit_messages: str, expected: List[str]
    ) -> None:
        with patch.dict(
            os.environ, {"PR_BODY": pr_body, "COMMIT_MESSAGES": commit_messages}
        ):
            disabled_issues = get_disabled_issues()
        self.assertEqual(disabled_issues, expected)

    # test variations of close in PR_BODY
    def test_closes_pr_body(self) -> None:
        pr_body = "closes #123 Close #143 ClOsE #345 closed #10283"
        self.run_assert_disabled_issues(pr_body, "", ["123", "143", "345", "10283"])

    # test variations of fix in COMMIT_MESSAGES
    def test_fixes_commit_messages(self) -> None:
        commit_messages = "fix #123 FixEd #143 fixes #345 FiXeD #10283"
        self.run_assert_disabled_issues(
            "", commit_messages, ["123", "143", "345", "10283"]
        )

    # test variations of resolve in PR_BODY and COMMIT_MESSAGES
    def test_resolves_pr_commits(self) -> None:
        pr_body = "resolve #123 resolveS #143"
        commit_messages = "REsolved #345 RESOLVES #10283"
        self.run_assert_disabled_issues(
            pr_body, commit_messages, ["123", "143", "345", "10283"]
        )

    # test links
    def test_issue_links(self) -> None:
        pr_body = "closes https://github.com/pytorch/pytorch/issues/75198 fixes https://github.com/pytorch/pytorch/issues/75123"
        self.run_assert_disabled_issues(pr_body, "", ["75198", "75123"])

    # test strange spacing
    def test_spacing(self) -> None:
        pr_body = "resolve #123,resolveS #143Resolved #345\nRESOLVES #10283"
        commit_messages = "Fixed #2348fixes https://github.com/pytorch/pytorch/issues/75123resolveS #2134"
        self.run_assert_disabled_issues(
            pr_body,
            commit_messages,
            ["123", "143", "345", "10283", "2348", "75123", "2134"],
        )

    # test bad things
    def test_not_accepted(self) -> None:
        pr_body = (
            "fixes189 fixeshttps://github.com/pytorch/pytorch/issues/75123 "
            "closedhttps://githubcom/pytorch/pytorch/issues/75123"
        )
        commit_messages = (
            "fix 234, fixes # 45, fixing #123, close 234, closes#45, closing #123 resolve 234, "
            "resolves  #45, resolving #123"
        )
        self.run_assert_disabled_issues(pr_body, commit_messages, [])


if __name__ == "__main__":
    unittest.main()
