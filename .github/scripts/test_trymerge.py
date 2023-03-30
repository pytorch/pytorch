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
from typing import Any, Dict, List, Optional
from unittest import main, mock, TestCase
from urllib.error import HTTPError

from gitutils import get_git_remote_name, get_git_repo_dir, GitRepo

from trymerge import (
    categorize_checks,
    find_matching_merge_rule,
    FlakyRule,
    get_classifications,
    get_rockset_results,
    gh_get_team_members,
    gh_graphql,
    GitHubPR,
    main as trymerge_main,
    MandatoryChecksMissingError,
    MergeRule,
    PostCommentError,
    read_merge_rules,
    validate_revert,
)

if "GIT_REMOTE_URL" not in os.environ:
    os.environ["GIT_REMOTE_URL"] = "https://github.com/pytorch/pytorch"


def mock_query(
    fallback_function: Any,
    file_name: str,
    key_function: Any,
    *args: Any,
) -> Any:
    gql_db_fname = os.path.join(os.path.dirname(__file__), file_name)

    def get_mocked_queries() -> Any:
        if not os.path.exists(gql_db_fname):
            return {}
        with open(gql_db_fname, encoding="utf-8") as f:
            return json.load(f)

    def save_mocked_queries(obj: Any) -> None:
        with open(gql_db_fname, encoding="utf-8", mode="w") as f:
            json.dump(obj, f, indent=2)
            f.write("\n")

    key = key_function(*args)
    mocked_queries = get_mocked_queries()

    if key in mocked_queries:
        return mocked_queries[key]

    try:
        rc = fallback_function(*args)
    except HTTPError as err:
        if err.code == 401:
            err_msg = f"If you are seeing this message during workflow run, please make sure to update {file_name}"
            err_msg += f" locally, by deleting it and running {os.path.basename(__file__)} with "
            err_msg += " GitHub Personal Access Token passed via GITHUB_TOKEN environment variable"
            err_msg += (
                " the rockset api key passed via ROCKSET_API_KEY environment variable"
            )
            if (
                os.getenv("GITHUB_TOKEN") is None
                or os.getenv("ROCKSET_API_KEY") is None
            ):
                err_msg = (
                    "Failed to update cached GraphQL queries as GITHUB_TOKEN or ROCKSET_API_KEY is not defined."
                    + err_msg
                )
            raise RuntimeError(err_msg) from err
    mocked_queries[key] = rc

    save_mocked_queries(mocked_queries)

    return rc


def mocked_gh_graphql(query: str, **kwargs: Any) -> Any:
    def key_function(query: str, kwargs: Any) -> str:
        return f"query_sha={sha256(query.encode('utf-8')).hexdigest()} " + " ".join(
            [f"{k}={kwargs[k]}" for k in sorted(kwargs.keys())]
        )

    def gh_graphql_wrapper(query: str, kwargs: Any) -> Any:
        return gh_graphql(query, **kwargs)

    return mock_query(gh_graphql_wrapper, "gql_mocks.json", key_function, query, kwargs)


def mocked_rockset_results(head_sha: str, merge_base: str, num_retries: int = 3) -> Any:
    return mock_query(
        get_rockset_results,
        "rockset_mocks.json",
        lambda x, y: f"{x} {y}",
        head_sha,
        merge_base,
    )


def mock_parse_args(revert: bool = False, force: bool = False) -> Any:
    class Object(object):
        def __init__(self) -> None:
            self.revert = revert
            self.force = force
            self.pr_num = 76123
            self.dry_run = True
            self.comment_id = 0
            self.reason = "this is for testing"
            self.ignore_current = False

    return Object()


def mock_revert(
    repo: GitRepo,
    pr: GitHubPR,
    *,
    dry_run: bool = False,
    comment_id: Optional[int] = None,
    reason: Optional[str] = None,
) -> None:
    pass


def mock_merge(
    pr_num: int,
    repo: GitRepo,
    dry_run: bool = False,
    skip_mandatory_checks: bool = False,
    comment_id: Optional[int] = None,
    timeout_minutes: int = 400,
    stale_pr_days: int = 3,
    ignore_current: bool = False,
) -> None:
    pass


def mock_gh_get_info() -> Any:
    return {
        "closed": False,
        "isCrossRepository": False,
        "files": {"nodes": [], "pageInfo": {"hasNextPage": False}},
        "changedFiles": 0,
    }


def mocked_read_merge_rules_NE(repo: Any, org: str, project: str) -> List[MergeRule]:
    return [
        MergeRule(
            name="mock with nonexistent check",
            patterns=["*"],
            approved_by=[],
            mandatory_checks_name=["Lint", "Facebook CLA Check", "nonexistent"],
        ),
    ]


def mocked_read_merge_rules(repo: Any, org: str, project: str) -> List[MergeRule]:
    return [
        MergeRule(
            name="super",
            patterns=["*"],
            approved_by=["pytorch/metamates"],
            mandatory_checks_name=[
                "Lint",
                "Facebook CLA Check",
                "pull / linux-xenial-cuda11.3-py3.7-gcc7 / build",
            ],
        ),
    ]


def mocked_read_merge_rules_raise(repo: Any, org: str, project: str) -> List[MergeRule]:
    raise RuntimeError("testing")


def empty_flaky_rules() -> List[FlakyRule]:
    return []


def empty_rockset_results(head_sha: str, merge_base: str) -> List[Dict[str, Any]]:
    return []


def dummy_merge_base() -> str:
    return "dummy"


class DummyGitRepo(GitRepo):
    def __init__(self) -> None:
        super().__init__(get_git_repo_dir(), get_git_remote_name())

    def commits_resolving_gh_pr(self, pr_num: int) -> List[str]:
        return ["FakeCommitSha"]

    def commit_message(self, ref: str) -> str:
        return "super awsome commit message"


@mock.patch("trymerge.read_flaky_rules", side_effect=empty_flaky_rules)
@mock.patch("trymerge.get_rockset_results", side_effect=empty_rockset_results)
@mock.patch("trymerge.GitHubPR.get_merge_base", side_effect=dummy_merge_base)
@mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
class TestTryMerge(TestCase):
    def test_merge_rules_valid(self, *args: Any) -> None:
        "Test that merge_rules.yaml can be parsed"
        repo = DummyGitRepo()
        merge_rules = read_merge_rules(repo, "pytorch", "pytorch")
        self.assertGreater(len(merge_rules), 1)

    @mock.patch("trymerge.read_merge_rules", side_effect=mocked_read_merge_rules)
    def test_match_rules(self, *args: Any) -> None:
        "Tests that PR passes merge rules"
        pr = GitHubPR("pytorch", "pytorch", 77700)
        repo = DummyGitRepo()
        self.assertTrue(find_matching_merge_rule(pr, repo) is not None)

    @mock.patch("trymerge.read_merge_rules", side_effect=mocked_read_merge_rules_raise)
    def test_read_merge_rules_fails(self, *args: Any) -> None:
        "Tests that PR fails to read the merge rules"
        pr = GitHubPR("pytorch", "pytorch", 77700)
        repo = DummyGitRepo()
        self.assertRaisesRegex(
            RuntimeError, "testing", lambda: find_matching_merge_rule(pr, repo)
        )

    @mock.patch("trymerge.read_merge_rules", side_effect=mocked_read_merge_rules)
    def test_lint_fails(self, *args: Any) -> None:
        "Tests that PR fails mandatory lint check"
        pr = GitHubPR("pytorch", "pytorch", 90791)
        repo = DummyGitRepo()
        self.assertRaises(RuntimeError, lambda: find_matching_merge_rule(pr, repo))

    def test_get_last_comment(self, *args: Any) -> None:
        "Tests that last comment can be fetched"
        pr = GitHubPR("pytorch", "pytorch", 71759)
        comment = pr.get_last_comment()
        self.assertEqual(comment.author_login, "github-actions")
        self.assertIsNone(comment.editor_login)
        self.assertTrue("You've committed this PR" in comment.body_text)

    def test_get_author_null(self, *args: Any) -> None:
        """Tests that PR author can be computed
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

    def test_last_pushed_at(self, *args: Any) -> None:
        """Tests that last_pushed_at will return None on merge commits."""
        pr = GitHubPR("pytorch", "pytorch", 71759)
        self.assertIsNotNone(pr.last_pushed_at())

        # 307120d6d3f7fcc3f92cfd26be891d360ad6a92a is merge commit
        # and as such does not have a pushedDate
        # See https://github.com/pytorch/pytorch/pull/94146#issuecomment-1421647117
        pr = GitHubPR("pytorch", "pytorch", 94146)
        self.assertIsNone(pr.last_pushed_at())

    def test_large_diff(self, *args: Any) -> None:
        "Tests that PR with 100+ files can be fetched"
        pr = GitHubPR("pytorch", "pytorch", 73099)
        self.assertTrue(pr.get_changed_files_count() > 100)
        flist = pr.get_changed_files()
        self.assertEqual(len(flist), pr.get_changed_files_count())

    def test_internal_changes(self, *args: Any) -> None:
        "Tests that PR with internal changes is detected"
        pr = GitHubPR("pytorch", "pytorch", 73969)
        self.assertTrue(pr.has_internal_changes())

    def test_checksuites_pagination(self, *args: Any) -> None:
        "Tests that PR with lots of checksuits can be fetched"
        pr = GitHubPR("pytorch", "pytorch", 73811)
        self.assertEqual(len(pr.get_checkrun_conclusions()), 76)

    def test_comments_pagination(self, *args: Any) -> None:
        "Tests that PR with 50+ comments can be fetched"
        pr = GitHubPR("pytorch", "pytorch", 31093)
        self.assertGreater(len(pr.get_comments()), 50)

    def test_gql_complexity(self, *args: Any) -> None:
        "Fetch comments and conclusions for PR with 60 commits"
        # Previous version of GrapQL query used to cause HTTP/502 error
        # see https://gist.github.com/malfet/9b93bc7eeddeaf1d84546efc4f0c577f
        pr = GitHubPR("pytorch", "pytorch", 68111)
        self.assertGreater(len(pr.get_comments()), 20)
        self.assertGreater(len(pr.get_checkrun_conclusions()), 3)
        self.assertGreater(pr.get_commit_count(), 60)

    def test_gql_retrieve_checksuites(self, *args: Any) -> None:
        "Fetch comments and conclusions for PR with 60 commits"
        pr = GitHubPR("pytorch", "pytorch", 94787)
        self.assertEqual(len(pr.get_checkrun_conclusions()), 183)

    def test_team_members(self, *args: Any) -> None:
        "Test fetching team members works"
        dev_infra_team = gh_get_team_members("pytorch", "pytorch-dev-infra")
        self.assertGreater(len(dev_infra_team), 2)
        with self.assertWarns(Warning):
            non_existing_team = gh_get_team_members("pytorch", "qwertyuiop")
            self.assertEqual(len(non_existing_team), 0)

    def test_get_author_many_commits(self, *args: Any) -> None:
        """Tests that authors for all commits can be fetched"""
        pr = GitHubPR("pytorch", "pytorch", 76118)
        authors = pr.get_authors()
        self.assertGreater(pr.get_commit_count(), 100)
        self.assertGreater(len(authors), 50)
        self.assertTrue("@" in pr.get_author())

    @mock.patch("trymerge.read_merge_rules", side_effect=mocked_read_merge_rules_NE)
    def test_pending_status_check(self, *args: Any) -> None:
        """Tests that PR with nonexistent/pending status checks fails with the right reason."""
        pr = GitHubPR("pytorch", "pytorch", 76118)
        repo = DummyGitRepo()
        self.assertRaisesRegex(
            MandatoryChecksMissingError,
            ".*are pending/not yet run.*",
            lambda: find_matching_merge_rule(pr, repo),
        )

    def test_get_author_many_reviews(self, *args: Any) -> None:
        """Tests that all reviews can be fetched"""
        pr = GitHubPR("pytorch", "pytorch", 76123)
        approved_by = pr.get_approved_by()
        self.assertGreater(len(approved_by), 0)
        assert pr._reviews is not None  # to pacify mypy
        self.assertGreater(len(pr._reviews), 100)

    def test_get_checkruns_many_runs(self, *args: Any) -> None:
        """Tests that all checkruns can be fetched"""
        pr = GitHubPR("pytorch", "pytorch", 77700)
        conclusions = pr.get_checkrun_conclusions()
        self.assertEqual(len(conclusions), 79)
        self.assertTrue("pull / linux-docs / build-docs (cpp)" in conclusions.keys())

    def test_cancelled_gets_ignored(self, *args: Any) -> None:
        """Tests that cancelled workflow does not override existing successfull status"""
        pr = GitHubPR("pytorch", "pytorch", 82169)
        conclusions = pr.get_checkrun_conclusions()
        lint_checks = [name for name in conclusions.keys() if "Lint" in name]
        self.assertTrue(len(lint_checks) > 0)
        self.assertTrue(
            all([conclusions[name].status == "SUCCESS" for name in lint_checks])
        )

    @mock.patch("trymerge.gh_get_pr_info", return_value=mock_gh_get_info())
    @mock.patch("trymerge.parse_args", return_value=mock_parse_args(True, False))
    @mock.patch("trymerge.try_revert", side_effect=mock_revert)
    def test_main_revert(self, mock_revert: Any, *args: Any) -> None:
        trymerge_main()
        mock_revert.assert_called_once()

    @mock.patch("trymerge.gh_get_pr_info", return_value=mock_gh_get_info())
    @mock.patch("trymerge.parse_args", return_value=mock_parse_args(False, True))
    @mock.patch("trymerge.merge", side_effect=mock_merge)
    def test_main_force(
        self, mock_merge: Any, mock_parse_args: Any, *args: Any
    ) -> None:
        trymerge_main()
        mock_merge.assert_called_once_with(
            mock.ANY,
            mock.ANY,
            dry_run=mock.ANY,
            skip_mandatory_checks=True,
            comment_id=mock.ANY,
            ignore_current=False,
        )

    @mock.patch("trymerge.gh_get_pr_info", return_value=mock_gh_get_info())
    @mock.patch("trymerge.parse_args", return_value=mock_parse_args(False, False))
    @mock.patch("trymerge.merge", side_effect=mock_merge)
    def test_main_merge(self, mock_merge: Any, *args: Any) -> None:
        trymerge_main()
        mock_merge.assert_called_once_with(
            mock.ANY,
            mock.ANY,
            dry_run=mock.ANY,
            skip_mandatory_checks=False,
            comment_id=mock.ANY,
            ignore_current=False,
        )

    @mock.patch("trymerge.read_merge_rules", side_effect=mocked_read_merge_rules)
    def test_revert_rules(self, *args: Any) -> None:
        """Tests that reverts from collaborators are allowed"""
        pr = GitHubPR("pytorch", "pytorch", 79694)
        repo = DummyGitRepo()
        self.assertIsNotNone(validate_revert(repo, pr, comment_id=1189459845))

    def test_get_changed_files(self, *args: Any) -> None:
        """
        Tests that the list changed files in a PR doesn't include duplicates
        """
        pr = GitHubPR("pytorch", "pytorch", 95233)
        try:
            changed_files = pr.get_changed_files()
        except RuntimeError as error:
            self.fail(f"get_changed_files throws an exception: {error}")

        self.assertEqual(len(changed_files), pr.get_changed_files_count())

    def test_revert_codev_fails(self, *args: Any) -> None:
        pr = GitHubPR("pytorch", "pytorch", 91340)

        class GitRepoCoDev(GitRepo):
            def __init__(self) -> None:
                super().__init__(get_git_repo_dir(), get_git_remote_name())

            def commits_resolving_gh_pr(self, pr_num: int) -> List[str]:
                return ["FakeCommitSha"]

            def commit_message(self, ref: str) -> str:
                return pr.get_body()

        repo = GitRepoCoDev()
        self.assertRaisesRegex(
            PostCommentError,
            "landed via phabricator",
            lambda: validate_revert(repo, pr, comment_id=1372496233),
        )

    def test_pr_changed_submodule_detection(self, *args: Any) -> None:
        # Updates submodule during dev-cycle but reverts it later
        pr = GitHubPR("pytorch", "pytorch", 95045)
        self.assertEqual(pr.get_changed_submodules(), [])
        self.assertFalse(pr.has_invalid_submodule_updates())

        # PR updates ideep
        pr = GitHubPR("pytorch", "pytorch", 94939)
        self.assertEqual(pr.get_changed_submodules(), ["third_party/ideep"])
        self.assertTrue(pr.has_invalid_submodule_updates())

        # Automated submodule update
        pr = GitHubPR("pytorch", "pytorch", 91051)
        self.assertEqual(pr.get_changed_submodules(), ["third_party/kineto"])
        self.assertFalse(pr.has_invalid_submodule_updates())


@mock.patch("trymerge.get_rockset_results", side_effect=mocked_rockset_results)
@mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
class TestBypassFailures(TestCase):
    def test_get_classifications(self, *args: Any) -> None:
        flaky_rules = [
            FlakyRule("distributed", ["##[error]The operation was canceled."])
        ]
        pr = GitHubPR("pytorch", "pytorch", 92863)
        checks = pr.get_checkrun_conclusions()
        checks = get_classifications(
            checks, pr.last_commit()["oid"], pr.get_merge_base(), flaky_rules, []
        )
        self.assertTrue(
            checks[
                "pull / linux-bionic-py3_7-clang8-xla / test (xla, 1, 1, linux.4xlarge)"
            ].classification
            == "BROKEN_TRUNK"
        )
        self.assertTrue(
            checks[
                "pull / linux-focal-py3.7-gcc7 / test (distributed, 1, 2, linux.2xlarge)"
            ].classification
            == "FLAKY"
        )
        pending, failed = categorize_checks(
            checks, list(checks.keys()), ok_failed_checks_threshold=2
        )
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 0)
        pending, failed = categorize_checks(
            checks, list(checks.keys()), ok_failed_checks_threshold=1
        )
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 2)

    def test_ignore_current(self, *args: Any) -> None:
        # Test various interactions of the failure classifier, mostly that
        # ignore current checks takes precedence over classifications for flaky
        # or broken trunk

        flaky_rules = [
            FlakyRule("distributed", ["##[error]The operation was canceled."])
        ]
        flaky = (
            "pull / linux-focal-py3.7-gcc7 / test (distributed, 1, 2, linux.2xlarge)"
        )
        broken_trunk = (
            "pull / linux-bionic-py3_7-clang8-xla / test (xla, 1, 1, linux.4xlarge)"
        )

        pr = GitHubPR("pytorch", "pytorch", 92863)
        checks = pr.get_checkrun_conclusions()

        # No broken trunk or flaky rules
        checks = get_classifications(checks, pr.last_commit()["oid"], None, [], [flaky])
        self.assertTrue(checks[flaky].classification == "IGNORE_CURRENT_CHECK")
        self.assertTrue(checks[broken_trunk].classification is None)
        _, failed = categorize_checks(
            checks, list(checks.keys()), ok_failed_checks_threshold=0
        )
        self.assertTrue(len(failed) == 1)

        # No flaky rules
        checks = get_classifications(
            checks, pr.last_commit()["oid"], pr.get_merge_base(), [], [flaky]
        )
        self.assertTrue(checks[flaky].classification == "IGNORE_CURRENT_CHECK")
        self.assertTrue(checks[broken_trunk].classification == "BROKEN_TRUNK")
        _, failed = categorize_checks(
            checks, list(checks.keys()), ok_failed_checks_threshold=1
        )
        self.assertTrue(len(failed) == 0)

        # No broken_trunk
        checks = get_classifications(
            checks,
            pr.last_commit()["oid"],
            pr.get_merge_base(),
            flaky_rules,
            [broken_trunk],
        )
        self.assertTrue(checks[flaky].classification == "FLAKY")
        self.assertTrue(checks[broken_trunk].classification == "IGNORE_CURRENT_CHECK")
        _, failed = categorize_checks(
            checks, list(checks.keys()), ok_failed_checks_threshold=1
        )
        self.assertTrue(len(failed) == 0)


if __name__ == "__main__":
    main()
