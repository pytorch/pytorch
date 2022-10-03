#!/usr/bin/env python3
# Tests implemented in this file are relying on GitHub GraphQL APIs
# In order to avoid test flakiness, results of the queries
# are cached in gql_mocks.json
# PyTorch Lint workflow does not have GITHUB_TOKEN defined to avoid
# flakiness, so if you are making changes to merge_rules or
# GraphQL queries in trymerge.py, please make sure to delete `gql_mocks.json`
# And re-run the test locally with ones PAT

import json
import os
from hashlib import sha256

from trymerge import (find_matching_merge_rule,
                      get_land_checkrun_conclusions,
                      validate_land_time_checks,
                      gh_graphql,
                      gh_get_team_members,
                      read_merge_rules,
                      validate_revert,
                      filter_pending_checks,
                      filter_failed_checks,
                      GitHubPR,
                      MergeRule,
                      MandatoryChecksMissingError,
                      WorkflowCheckState,
                      main as trymerge_main)
from gitutils import get_git_remote_name, get_git_repo_dir, GitRepo
from typing import Any, List, Optional
from unittest import TestCase, main, mock
from urllib.error import HTTPError

if 'GIT_REMOTE_URL' not in os.environ:
    os.environ['GIT_REMOTE_URL'] = "https://github.com/pytorch/pytorch"

def mocked_gh_graphql(query: str, **kwargs: Any) -> Any:
    gql_db_fname = os.path.join(os.path.dirname(__file__), "gql_mocks.json")

    def get_mocked_queries() -> Any:
        if not os.path.exists(gql_db_fname):
            return {}
        with open(gql_db_fname, encoding="utf-8") as f:
            return json.load(f)

    def save_mocked_queries(obj: Any) -> None:
        with open(gql_db_fname, encoding="utf-8", mode="w") as f:
            json.dump(obj, f, indent=2)
            f.write("\n")

    key = f"query_sha={sha256(query.encode('utf-8')).hexdigest()} " + " ".join([f"{k}={kwargs[k]}" for k in sorted(kwargs.keys())])
    mocked_queries = get_mocked_queries()

    if key in mocked_queries:
        return mocked_queries[key]

    try:
        rc = gh_graphql(query, **kwargs)
    except HTTPError as err:
        if err.code == 401:
            err_msg = "If you are seeing this message during workflow run, please make sure to update gql_mocks.json"
            err_msg += f" locally, by deleting it and running {os.path.basename(__file__)} with "
            err_msg += " GitHub Personal Access Token passed via GITHUB_TOKEN environment variable"
            if os.getenv("GITHUB_TOKEN") is None:
                err_msg = "Failed to update cached GraphQL queries as GITHUB_TOKEN is not defined." + err_msg
            raise RuntimeError(err_msg) from err
    mocked_queries[key] = rc

    save_mocked_queries(mocked_queries)

    return rc

def mock_parse_args(revert: bool = False,
                    force: bool = False) -> Any:
    class Object(object):
        def __init__(self) -> None:
            self.revert = revert
            self.force = force
            self.pr_num = 76123
            self.dry_run = True
            self.comment_id = 0
            self.on_mandatory = False
            self.on_green = False
            self.land_checks = False
            self.reason = 'this is for testing'

    return Object()

def mock_revert(repo: GitRepo, pr: GitHubPR, *,
                dry_run: bool = False,
                comment_id: Optional[int] = None,
                reason: Optional[str] = None) -> None:
    pass

def mock_merge(pr_num: int, repo: GitRepo,
               dry_run: bool = False,
               skip_mandatory_checks: bool = False,
               comment_id: Optional[int] = None,
               mandatory_only: bool = False,
               on_green: bool = False,
               land_checks: bool = False,
               timeout_minutes: int = 400,
               stale_pr_days: int = 3) -> None:
    pass

def mock_gh_get_info() -> Any:
    return {"closed": False, "isCrossRepository": False}


def mocked_read_merge_rules_NE(repo: Any, org: str, project: str) -> List[MergeRule]:
    return [
        MergeRule(name="mock with nonexistent check",
                  patterns=["*"],
                  approved_by=[],
                  mandatory_checks_name=["Lint",
                                         "Facebook CLA Check",
                                         "nonexistent"],
                  ),
    ]


def mocked_read_merge_rules(repo: Any, org: str, project: str) -> List[MergeRule]:
    return [
        MergeRule(name="super",
                  patterns=["*"],
                  approved_by=["pytorch/metamates"],
                  mandatory_checks_name=["Lint",
                                         "Facebook CLA Check",
                                         "pull / linux-xenial-cuda11.3-py3.7-gcc7 / build",
                                         ],
                  ),
    ]


def mocked_read_merge_rules_raise(repo: Any, org: str, project: str) -> List[MergeRule]:
    raise RuntimeError("testing")


class DummyGitRepo(GitRepo):
    def __init__(self) -> None:
        super().__init__(get_git_repo_dir(), get_git_remote_name())

    def commits_resolving_gh_pr(self, pr_num: int) -> List[str]:
        return ["FakeCommitSha"]

    def commit_message(self, ref: str) -> str:
        return "super awsome commit message"

class TestGitHubPR(TestCase):
    def test_merge_rules_valid(self) -> None:
        "Test that merge_rules.yaml can be parsed"
        repo = DummyGitRepo()
        self.assertGreater(len(read_merge_rules(repo, "pytorch", "pytorch")), 1)

    @mock.patch('trymerge.gh_graphql', side_effect=mocked_gh_graphql)
    @mock.patch('trymerge.read_merge_rules', side_effect=mocked_read_merge_rules)
    def test_match_rules(self, mocked_gql: Any, mocked_rmr: Any) -> None:
        "Tests that PR passes merge rules"
        pr = GitHubPR("pytorch", "pytorch", 77700)
        repo = DummyGitRepo()
        self.assertTrue(find_matching_merge_rule(pr, repo) is not None)

    @mock.patch('trymerge.gh_graphql', side_effect=mocked_gh_graphql)
    @mock.patch('trymerge.read_merge_rules', side_effect=mocked_read_merge_rules_raise)
    def test_read_merge_rules_fails(self, mocked_gql: Any, mocked_rmr: Any) -> None:
        "Tests that PR fails to read the merge rules"
        pr = GitHubPR("pytorch", "pytorch", 77700)
        repo = DummyGitRepo()
        self.assertRaisesRegex(RuntimeError, "testing", lambda: find_matching_merge_rule(pr, repo))

    @mock.patch('trymerge.gh_graphql', side_effect=mocked_gh_graphql)
    @mock.patch('trymerge.read_merge_rules', side_effect=mocked_read_merge_rules)
    def test_lint_fails(self, mocked_gql: Any, mocked_rmr: Any) -> None:
        "Tests that PR fails mandatory lint check"
        pr = GitHubPR("pytorch", "pytorch", 74649)
        repo = DummyGitRepo()
        self.assertRaises(RuntimeError, lambda: find_matching_merge_rule(pr, repo))

    @mock.patch('trymerge.gh_graphql', side_effect=mocked_gh_graphql)
    def test_get_last_comment(self, mocked_gql: Any) -> None:
        "Tests that last comment can be fetched"
        pr = GitHubPR("pytorch", "pytorch", 71759)
        comment = pr.get_last_comment()
        self.assertEqual(comment.author_login, "github-actions")
        self.assertIsNone(comment.editor_login)
        self.assertTrue("You've committed this PR" in comment.body_text)

    @mock.patch('trymerge.gh_graphql', side_effect=mocked_gh_graphql)
    def test_get_author_null(self, mocked_gql: Any) -> None:
        """ Tests that PR author can be computed
            If reply contains NULL
        """
        pr = GitHubPR("pytorch", "pytorch", 71759)
        author = pr.get_author()
        self.assertTrue(author is not None)
        self.assertTrue("@" in author)
        self.assertTrue(pr.get_diff_revision() is None)

        # PR with multiple contributors, but creator id is not among authors
        pr = GitHubPR("pytorch", "pytorch", 75095)
        self.assertEqual(pr.get_pr_creator_login(), "mruberry")
        author = pr.get_author()
        self.assertTrue(author is not None)

    @mock.patch('trymerge.gh_graphql', side_effect=mocked_gh_graphql)
    def test_large_diff(self, mocked_gql: Any) -> None:
        "Tests that PR with 100+ files can be fetched"
        pr = GitHubPR("pytorch", "pytorch", 73099)
        self.assertTrue(pr.get_changed_files_count() > 100)
        flist = pr.get_changed_files()
        self.assertEqual(len(flist), pr.get_changed_files_count())

    @mock.patch('trymerge.gh_graphql', side_effect=mocked_gh_graphql)
    def test_internal_changes(self, mocked_gql: Any) -> None:
        "Tests that PR with internal changes is detected"
        pr = GitHubPR("pytorch", "pytorch", 73969)
        self.assertTrue(pr.has_internal_changes())

    @mock.patch('trymerge.gh_graphql', side_effect=mocked_gh_graphql)
    def test_checksuites_pagination(self, mocked_gql: Any) -> None:
        "Tests that PR with lots of checksuits can be fetched"
        pr = GitHubPR("pytorch", "pytorch", 73811)
        self.assertEqual(len(pr.get_checkrun_conclusions()), 107)

    @mock.patch('trymerge.gh_graphql', side_effect=mocked_gh_graphql)
    def test_comments_pagination(self, mocked_gql: Any) -> None:
        "Tests that PR with 50+ comments can be fetched"
        pr = GitHubPR("pytorch", "pytorch", 31093)
        self.assertGreater(len(pr.get_comments()), 50)

    @mock.patch('trymerge.gh_graphql', side_effect=mocked_gh_graphql)
    def test_gql_complexity(self, mocked_gql: Any) -> None:
        "Fetch comments and conclusions for PR with 60 commits"
        # Previous version of GrapQL query used to cause HTTP/502 error
        # see https://gist.github.com/malfet/9b93bc7eeddeaf1d84546efc4f0c577f
        pr = GitHubPR("pytorch", "pytorch", 68111)
        self.assertGreater(len(pr.get_comments()), 20)
        self.assertGreater(len(pr.get_checkrun_conclusions()), 3)
        self.assertGreater(pr.get_commit_count(), 60)

    @mock.patch('trymerge.gh_graphql', side_effect=mocked_gh_graphql)
    def test_team_members(self, mocked_gql: Any) -> None:
        "Test fetching team members works"
        dev_infra_team = gh_get_team_members("pytorch", "pytorch-dev-infra")
        self.assertGreater(len(dev_infra_team), 2)
        with self.assertWarns(Warning):
            non_existing_team = gh_get_team_members("pytorch", "qwertyuiop")
            self.assertEqual(len(non_existing_team), 0)

    @mock.patch('trymerge.gh_graphql', side_effect=mocked_gh_graphql)
    def test_get_author_many_commits(self, mocked_gql: Any) -> None:
        """ Tests that authors for all commits can be fetched
        """
        pr = GitHubPR("pytorch", "pytorch", 76118)
        authors = pr.get_authors()
        self.assertGreater(pr.get_commit_count(), 100)
        self.assertGreater(len(authors), 50)
        self.assertTrue("@" in pr.get_author())

    @mock.patch('trymerge.read_merge_rules', side_effect=mocked_read_merge_rules_NE)
    @mock.patch('trymerge.gh_graphql', side_effect=mocked_gh_graphql)
    def test_pending_status_check(self, mocked_gql: Any, mocked_read_merge_rules: Any) -> None:
        """ Tests that PR with nonexistent/pending status checks fails with the right reason.
        """
        pr = GitHubPR("pytorch", "pytorch", 76118)
        repo = DummyGitRepo()
        self.assertRaisesRegex(MandatoryChecksMissingError,
                               ".*are pending/not yet run.*",
                               lambda: find_matching_merge_rule(pr, repo))

    @mock.patch('trymerge.gh_graphql', side_effect=mocked_gh_graphql)
    def test_get_author_many_reviews(self, mocked_gql: Any) -> None:
        """ Tests that all reviews can be fetched
        """
        pr = GitHubPR("pytorch", "pytorch", 76123)
        approved_by = pr.get_approved_by()
        self.assertGreater(len(approved_by), 0)
        assert pr._reviews is not None  # to pacify mypy
        self.assertGreater(len(pr._reviews), 100)

    @mock.patch('trymerge.gh_graphql', side_effect=mocked_gh_graphql)
    def test_get_checkruns_many_runs(self, mocked_gql: Any) -> None:
        """ Tests that all checkruns can be fetched
        """
        pr = GitHubPR("pytorch", "pytorch", 77700)
        conclusions = pr.get_checkrun_conclusions()
        self.assertEqual(len(conclusions), 83)
        self.assertTrue("pull / linux-docs / build-docs (cpp)" in conclusions.keys())

    @mock.patch('trymerge.gh_graphql', side_effect=mocked_gh_graphql)
    def test_cancelled_gets_ignored(self, mocked_gql: Any) -> None:
        """ Tests that cancelled workflow does not override existing successfull status
        """
        pr = GitHubPR("pytorch", "pytorch", 82169)
        conclusions = pr.get_checkrun_conclusions()
        self.assertTrue("Lint" in conclusions.keys())
        self.assertEqual(conclusions["Lint"][0], "SUCCESS")

    @mock.patch('trymerge.gh_graphql', side_effect=mocked_gh_graphql)
    def test_get_many_land_checks(self, mocked_gql: Any) -> None:
        """ Tests that all checkruns can be fetched for a commit
        """
        conclusions = get_land_checkrun_conclusions('pytorch', 'pytorch', '6882717f73deffb692219ccd1fd6db258d8ed684')
        self.assertEqual(len(conclusions), 101)
        self.assertTrue("pull / linux-docs / build-docs (cpp)" in conclusions.keys())

    @mock.patch('trymerge.gh_graphql', side_effect=mocked_gh_graphql)
    def test_failed_land_checks(self, mocked_gql: Any) -> None:
        """ Tests that PR with Land Checks fail with a RunTime error
        """
        self.assertRaisesRegex(RuntimeError,
                               ".*Failed to merge; some land checks failed.*",
                               lambda: validate_land_time_checks('pytorch', 'pytorch', '6882717f73deffb692219ccd1fd6db258d8ed684'))

    @mock.patch('trymerge.gh_get_pr_info', return_value=mock_gh_get_info())
    @mock.patch('trymerge.parse_args', return_value=mock_parse_args(True, False))
    @mock.patch('trymerge.try_revert', side_effect=mock_revert)
    def test_main_revert(self, mock_revert: Any, mock_parse_args: Any, gh_get_pr_info: Any) -> None:
        trymerge_main()
        mock_revert.assert_called_once()

    @mock.patch('trymerge.gh_get_pr_info', return_value=mock_gh_get_info())
    @mock.patch('trymerge.parse_args', return_value=mock_parse_args(False, True))
    @mock.patch('trymerge.merge', side_effect=mock_merge)
    def test_main_force(self, mock_merge: Any, mock_parse_args: Any, mock_gh_get_info: Any) -> None:
        trymerge_main()
        mock_merge.assert_called_once_with(mock.ANY,
                                           mock.ANY,
                                           dry_run=mock.ANY,
                                           skip_mandatory_checks=True,
                                           comment_id=mock.ANY,
                                           on_green=False,
                                           land_checks=False,
                                           mandatory_only=False)

    @mock.patch('trymerge.gh_get_pr_info', return_value=mock_gh_get_info())
    @mock.patch('trymerge.parse_args', return_value=mock_parse_args(False, False))
    @mock.patch('trymerge.merge', side_effect=mock_merge)
    def test_main_merge(self, mock_merge: Any, mock_parse_args: Any, mock_gh_get_info: Any) -> None:
        trymerge_main()
        mock_merge.assert_called_once_with(mock.ANY,
                                           mock.ANY,
                                           dry_run=mock.ANY,
                                           skip_mandatory_checks=False,
                                           comment_id=mock.ANY,
                                           on_green=False,
                                           land_checks=False,
                                           mandatory_only=False)

    @mock.patch('trymerge.gh_graphql', side_effect=mocked_gh_graphql)
    @mock.patch('trymerge.read_merge_rules', side_effect=mocked_read_merge_rules)
    def test_revert_rules(self, mock_gql: Any, mock_mr: Any) -> None:
        """ Tests that reverts from collaborators are allowed """
        pr = GitHubPR("pytorch", "pytorch", 79694)
        repo = DummyGitRepo()
        self.assertIsNotNone(validate_revert(repo, pr, comment_id=1189459845))

    def test_checks_filter(self) -> None:
        checks = [
            WorkflowCheckState(name="check0", status="SUCCESS", url="url0"),
            WorkflowCheckState(name="check1", status="FAILURE", url="url1"),
            WorkflowCheckState(name="check2", status="STARTUP_FAILURE", url="url2"),
            WorkflowCheckState(name="check3", status=None, url="url3"),
        ]

        checks_dict = {check.name : check for check in checks}

        pending_checks = filter_pending_checks(checks_dict)
        failing_checks = filter_failed_checks(checks_dict)

        self.assertListEqual(failing_checks, [checks[1], checks[2]])
        self.assertListEqual(pending_checks, [checks[3]])

if __name__ == "__main__":
    main()
