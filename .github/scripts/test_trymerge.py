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
import warnings
from hashlib import sha256
from typing import Any, cast, Dict, List, Optional
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
    is_broken_trunk,
    main as trymerge_main,
    MandatoryChecksMissingError,
    MergeRule,
    PostCommentError,
    read_merge_rules,
    remove_job_name_suffix,
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
    class Object:
        def __init__(self) -> None:
            self.revert = revert
            self.force = force
            self.pr_num = 76123
            self.dry_run = True
            self.comment_id = 0
            self.reason = "this is for testing"
            self.ignore_current = False

    return Object()


def mock_remove_label(org: str, repo: str, pr_num: str, label: str) -> None:
    pass


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
    pr: GitHubPR,
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
            ignore_flaky_failures=True,
        ),
    ]


def mocked_read_merge_rules(repo: Any, org: str, project: str) -> List[MergeRule]:
    return [
        MergeRule(
            name="super",
            patterns=["*"],
            approved_by=["pytorch/metamates", "ngimel"],
            mandatory_checks_name=[
                "Lint",
                "Facebook CLA Check",
                "pull / linux-xenial-cuda11.3-py3.7-gcc7 / build",
            ],
            ignore_flaky_failures=True,
        ),
    ]


def mocked_read_merge_rules_raise(repo: Any, org: str, project: str) -> List[MergeRule]:
    raise RuntimeError("testing")


def empty_flaky_rules() -> List[FlakyRule]:
    return []


def xla_is_flaky_rules() -> List[FlakyRule]:
    return [
        FlakyRule("xla", ["FAILED: Build did NOT complete successfully"]),
    ]


def xla_merge_rules(repo: Any, org: str, project: str) -> List[MergeRule]:
    return [
        MergeRule(
            name=" OSS CI / pytorchbot / XLA",
            patterns=[".github/ci_commit_pins/xla.txt"],
            approved_by=["pytorchbot"],
            mandatory_checks_name=[
                "Lint",
                "EasyCLA",
                "pull / linux-bionic-py3_8-clang8-xla / build",
                "pull / linux-bionic-py3_8-clang8-xla / test (xla, 1, 1, linux.4xlarge)",
            ],
            ignore_flaky_failures=False,
        ),
    ]


def empty_rockset_results(head_sha: str, merge_base: str) -> List[Dict[str, Any]]:
    return []


class DummyGitRepo(GitRepo):
    def __init__(self) -> None:
        super().__init__(get_git_repo_dir(), get_git_remote_name())

    def commits_resolving_gh_pr(self, pr_num: int) -> List[str]:
        return ["FakeCommitSha"]

    def commit_message(self, ref: str) -> str:
        return "super awsome commit message"


@mock.patch("trymerge.read_flaky_rules", side_effect=empty_flaky_rules)
@mock.patch("trymerge.get_rockset_results", side_effect=empty_rockset_results)
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
            all(conclusions[name].status == "SUCCESS" for name in lint_checks)
        )

    @mock.patch("trymerge.gh_get_pr_info", return_value=mock_gh_get_info())
    @mock.patch("trymerge.parse_args", return_value=mock_parse_args(True, False))
    @mock.patch("trymerge.try_revert", side_effect=mock_revert)
    def test_main_revert(self, mock_revert: Any, *args: Any) -> None:
        trymerge_main()
        mock_revert.assert_called_once()

    @mock.patch("trymerge.gh_get_pr_info", return_value=mock_gh_get_info())
    @mock.patch("trymerge.parse_args", return_value=mock_parse_args(False, True))
    @mock.patch("trymerge.gh_remove_label", side_effect=mock_remove_label)
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
    @mock.patch("trymerge.gh_remove_label", side_effect=mock_remove_label)
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

        class GitRepoCoDev(DummyGitRepo):
            def commit_message(self, ref: str) -> str:
                return pr.get_body()

        repo = GitRepoCoDev()
        self.assertRaisesRegex(
            PostCommentError,
            "landed via phabricator",
            lambda: validate_revert(repo, pr, comment_id=1372496233),
        )

    def test_revert_codev_abandoned_diff_succeeds(self, *args: Any) -> None:
        pr = GitHubPR("pytorch", "pytorch", 100652)

        class GitRepoCoDev(DummyGitRepo):
            def commit_message(self, ref: str) -> str:
                return pr.get_body()

        repo = GitRepoCoDev()
        validate_revert(repo, pr, comment_id=1588195237)

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

    def test_remove_job_name_suffix(self, *args: Any) -> None:
        test_cases = [
            {
                "name": "linux-bionic-cuda12.1-py3.10-gcc9-sm86 / test (default, 1, 5, linux.g5.4xlarge.nvidia.gpu)",
                "expected": "linux-bionic-cuda12.1-py3.10-gcc9-sm86 / test (default)",
            },
            {
                "name": "android-emulator-build-test / build-and-test (default, 1, 1, ubuntu-20.04-16x)",
                "expected": "android-emulator-build-test / build-and-test (default)",
            },
            {
                "name": "linux-focal-rocm5.4.2-py3.8 / build",
                "expected": "linux-focal-rocm5.4.2-py3.8 / build",
            },
            {
                "name": "libtorch-cpu-shared-with-deps-release-build",
                "expected": "libtorch-cpu-shared-with-deps-release-build",
            },
            {
                "name": "manywheel-py3_8-cuda11_8-test / test",
                "expected": "manywheel-py3_8-cuda11_8-test / test",
            },
            {
                "name": "lintrunner / linux-job",
                "expected": "lintrunner / linux-job",
            },
            {
                "name": "Test `run_test.py` is usable without boto3/rockset",
                "expected": "Test `run_test.py` is usable without boto3/rockset",
            },
        ]

        for case in test_cases:
            self.assertEqual(case["expected"], remove_job_name_suffix(case["name"]))

    def test_is_broken_trunk(self, *args: Any) -> None:
        test_cases: List[Dict[str, Any]] = [
            {
                "head_job": None,
                "base_jobs": {
                    "job_a": {
                        "conclusion": "success",
                        "failure_captures": ["a", "b"],
                    },
                    "job_b": {
                        "conclusion": "failure",
                        "failure_captures": ["a", "b"],
                    },
                },
                "expected": False,
                "description": "Invalid input - head job",
            },
            {
                "head_job": {
                    "conclusion": "failure",
                    "failure_captures": ["a", "b"],
                },
                "base_jobs": None,
                "expected": False,
                "description": "Invalid input - base jobs",
            },
            {
                "head_job": {
                    "conclusion": "failure",
                    "failure_captures": ["a", "b"],
                },
                "base_jobs": {},
                "expected": False,
                "description": "Invalid input - empty base jobs",
            },
            {
                "head_job": {
                    "conclusion": "failure",
                    "failure_captures": ["x", "y"],
                },
                "base_jobs": {
                    "job_a": {
                        "conclusion": "success",
                        "failure_captures": ["a", "b"],
                    },
                    "job_b": {
                        "conclusion": "failure",
                        "failure_captures": ["x", "y"],
                    },
                },
                "expected": True,
                "description": "Found a match",
            },
            {
                "head_job": {
                    "conclusion": "success",
                    "failure_captures": ["x", "y"],
                },
                "base_jobs": {
                    "job_a": {
                        "conclusion": "success",
                        "failure_captures": ["a", "b"],
                    },
                    "job_b": {
                        "conclusion": "failure",
                        "failure_captures": ["x", "y"],
                    },
                },
                "expected": False,
                "description": "Not found - different conclusion",
            },
            {
                "head_job": {
                    "conclusion": "failure",
                    "failure_captures": ["a", "b"],
                },
                "base_jobs": {
                    "job_a": {
                        "conclusion": "success",
                        "failure_captures": ["a", "b"],
                    },
                    "job_b": {
                        "conclusion": "failure",
                        "failure_captures": ["x", "y"],
                    },
                },
                "expected": False,
                "description": "Not found - different captured failures",
            },
        ]

        for case in test_cases:
            self.assertEqual(
                case["expected"], is_broken_trunk(case["head_job"], case["base_jobs"])
            )

    def test_get_merge_base(
        self,
        mock_gh_graphql: Any,
        mock_get_rockset_results: Any,
        mock_read_flaky_rules: Any,
    ) -> None:
        pr = GitHubPR("pytorch", "pytorch", 104121)

        mock_merge_base = "mocked-sha"
        with mock.patch(
            "trymerge.gh_fetch_merge_base", return_value=mock_merge_base
        ) as mocked_gh_fetch_merge_base:
            self.assertEqual(mock_merge_base, pr.get_merge_base())

            # Make sure that consecutive calls will use the same merge base instead of
            # making another query
            self.assertEqual(mock_merge_base, pr.get_merge_base())
            mocked_gh_fetch_merge_base.assert_called_once()


@mock.patch("trymerge.get_rockset_results", side_effect=mocked_rockset_results)
@mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
@mock.patch("trymerge.gh_fetch_merge_base", return_value="")
class TestBypassFailures(TestCase):
    def test_get_classifications(self, *args: Any) -> None:
        flaky_rules = [
            # Try a regex rule
            FlakyRule("distributed", ["##\\[error\\]The operation [wW]as .+"])
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
        pending, failed, ignorable = categorize_checks(
            checks, list(checks.keys()), ok_failed_checks_threshold=2
        )
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 0)
        self.assertTrue(len(ignorable["FLAKY"]) == 1)
        self.assertTrue(len(ignorable["BROKEN_TRUNK"]) == 1)

        # Not set any threshold, defaults to -1 to ignore all flaky and broken trunk failures
        pending, failed, ignorable = categorize_checks(checks, list(checks.keys()))
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 0)
        self.assertTrue(len(ignorable["FLAKY"]) == 1)
        self.assertTrue(len(ignorable["BROKEN_TRUNK"]) == 1)

        pending, failed, ignorable = categorize_checks(
            checks, list(checks.keys()), ok_failed_checks_threshold=1
        )
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 2)
        self.assertTrue(len(ignorable["FLAKY"]) == 1)
        self.assertTrue(len(ignorable["BROKEN_TRUNK"]) == 1)

    def test_get_classifications_unstable(self, *args: Any) -> None:
        pr = GitHubPR("pytorch", "pytorch", 104312)
        checks = pr.get_checkrun_conclusions()
        checks = get_classifications(
            checks, pr.last_commit()["oid"], pr.get_merge_base(), [], []
        )
        workflow_name = "linux-bionic-cuda12.1-py3.10-gcc9-bazel-test"
        job_name = "build-and-test (default, 1, 1, linux.4xlarge.nvidia.gpu, unstable)"
        self.assertTrue(
            checks[f"pull / {workflow_name} / {job_name}"].classification == "UNSTABLE"
        )
        pending, failed, ignorable = categorize_checks(
            checks, list(checks.keys()), ok_failed_checks_threshold=1
        )
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 0)
        self.assertTrue(len(ignorable["UNSTABLE"]) == 1)

    def test_get_classifications_pending_unstable(self, *args: Any) -> None:
        pr = GitHubPR("pytorch", "pytorch", 105998)
        checks = pr.get_checkrun_conclusions()
        checks = get_classifications(
            checks, pr.last_commit()["oid"], pr.get_merge_base(), [], []
        )
        pending, failed, ignorable = categorize_checks(
            checks, list(checks.keys()), ok_failed_checks_threshold=1
        )
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 3)
        self.assertTrue(len(ignorable["UNSTABLE"]) == 3)

    def test_get_classifications_broken_trunk(self, *args: Any) -> None:
        # The mock merge base is the actual value returned by gh_fetch_merge_base
        test_cases = [
            {
                # This PR had one broken trunk failure but it was run on a different shard
                # than the one on the base commit. This should still count as broken trunk
                "pr_num": 104214,
                "mock_merge_base": "436d035dc74db9c703297a62163b0cad0c546665",
                "unrelated_failure_count": 1,
            },
            {
                # This PR had one broken trunk failure and it used ghstack
                "pr_num": 105145,
                "mock_merge_base": "194fe1d12f9860734cc28ed21bdabda2fbb06336",
                "unrelated_failure_count": 1,
            },
            {
                # The failure on the merge base was retried successfully and
                # its conclusion changed from failure to success. We want to
                # keep the failure record from the merge base so that it can
                # be used to detect broken trunk
                "pr_num": 107160,
                "mock_merge_base": "a5d841ef01e615e2a654fb12cf0cd08697d12ccf",
                "unrelated_failure_count": 4,
            },
        ]

        for case in test_cases:
            pr_num = case["pr_num"]
            mock_merge_base = case["mock_merge_base"]
            unrelated_failure_count = case["unrelated_failure_count"]

            pr = GitHubPR("pytorch", "pytorch", cast(int, pr_num))
            with mock.patch(
                "trymerge.gh_fetch_merge_base", return_value=mock_merge_base
            ) as mocked_gh_fetch_merge_base:
                checks = pr.get_checkrun_conclusions()
                checks = get_classifications(
                    checks, pr.last_commit()["oid"], pr.get_merge_base(), [], []
                )

                pending, failed, _ = categorize_checks(checks, list(checks.keys()))
                self.assertTrue(len(pending) == 0)
                self.assertTrue(len(failed) == 0)

                # When the ok_failed_checks_threshold is set to 0, the broken trunk failure
                # won't be ignored
                pending, failed, _ = categorize_checks(
                    checks, list(checks.keys()), ok_failed_checks_threshold=0
                )
                self.assertTrue(len(pending) == 0)
                self.assertTrue(len(failed) == unrelated_failure_count)

    def test_ignore_current(self, *args: Any) -> None:
        # Test various interactions of the failure classifier, mostly that
        # ignore current checks takes precedence over classifications for flaky
        # or broken trunk

        flaky_rules = [
            FlakyRule("distributed", ["##\\[error\\]The operation was canceled."])
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
        _, failed, ignorable = categorize_checks(
            checks, list(checks.keys()), ok_failed_checks_threshold=0
        )
        self.assertTrue(len(failed) == 1)
        self.assertTrue(len(ignorable["IGNORE_CURRENT_CHECK"]) == 1)

        # No flaky rules
        checks = get_classifications(
            checks, pr.last_commit()["oid"], pr.get_merge_base(), [], [flaky]
        )
        self.assertTrue(checks[flaky].classification == "IGNORE_CURRENT_CHECK")
        self.assertTrue(checks[broken_trunk].classification == "BROKEN_TRUNK")
        _, failed, ignorable = categorize_checks(
            checks, list(checks.keys()), ok_failed_checks_threshold=1
        )
        self.assertTrue(len(failed) == 0)
        self.assertTrue(len(ignorable["IGNORE_CURRENT_CHECK"]) == 1)

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
        _, failed, ignorable = categorize_checks(
            checks, list(checks.keys()), ok_failed_checks_threshold=1
        )
        self.assertTrue(len(failed) == 0)
        self.assertTrue(len(ignorable["IGNORE_CURRENT_CHECK"]) == 1)

    @mock.patch("trymerge.read_flaky_rules", side_effect=xla_is_flaky_rules)
    @mock.patch("trymerge.read_merge_rules", side_effect=xla_merge_rules)
    def test_dont_ignore_flaky_failures(self, *args: Any) -> None:
        """Regression test for https://github.com/pytorch/test-infra/issues/4126"""
        pr = GitHubPR("pytorch", "pytorch", 100369)
        repo = DummyGitRepo()
        # Check that failure is classified as flaky but still raises exception
        with warnings.catch_warnings(record=True) as w, self.assertRaises(RuntimeError):
            rule = find_matching_merge_rule(pr, repo)
        self.assertEqual(len(w), 1)
        self.assertIn(
            "1 checks failed but were likely due flakiness or broken trunk",
            str(w[0].message),
        )


@mock.patch("trymerge.get_rockset_results", side_effect=mocked_rockset_results)
@mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
@mock.patch("trymerge.gh_fetch_merge_base", return_value="")
class TestGitHubPRGhstackDependencies2(TestCase):
    def test_pr_dependencies(self, *args: Any) -> None:
        pr = GitHubPR("pytorch", "pytorch", 106068)
        msg = pr.gen_commit_message(filter_ghstack=True)
        assert msg == (
            "[FSDP] Break up `_post_backward_hook` into smaller funcs (#106068)\n\n\nDifferential Revision: ["
            "D47852461](https://our.internmc.facebook.com/intern/diff/D47852461)\nPull Request resolved: "
            "https://github.com/pytorch/pytorch/pull/106068\nApproved by: \n"
        )

    def test_pr_dependencies_ghstack(self, *args: Any) -> None:
        pr0 = GitHubPR("pytorch", "pytorch", 106032)
        pr1 = GitHubPR("pytorch", "pytorch", 106033)
        pr2 = GitHubPR("pytorch", "pytorch", 106034)
        pr = GitHubPR("pytorch", "pytorch", 106068)

        msg = pr.gen_commit_message(filter_ghstack=True, ghstack_deps=[pr0, pr1, pr2])
        assert msg == (
            "[FSDP] Break up `_post_backward_hook` into smaller funcs (#106068)\n\n\nDifferential Revision: ["
            "D47852461](https://our.internmc.facebook.com/intern/diff/D47852461)\nPull Request resolved: "
            "https://github.com/pytorch/pytorch/pull/106068\nApproved by: \n"
            "ghstack dependencies: #106032, #106033, #106034\n"
        )

    @mock.patch("trymerge.read_merge_rules")
    @mock.patch("trymerge.GitRepo")
    @mock.patch("trymerge.get_ghstack_prs")
    def test_merge_ghstack_into(
        self,
        mock_get_ghstack_prs: mock.MagicMock,
        mock_repo: mock.MagicMock,
        mock_merge_rules: mock.MagicMock,
        *args: Any,
    ) -> None:
        """
        Test that the merge_ghstack_into method works correctly
        """
        pr0 = GitHubPR("pytorch", "pytorch", 106032)
        pr1 = GitHubPR("pytorch", "pytorch", 106033)
        pr2 = GitHubPR("pytorch", "pytorch", 106034)
        pr = GitHubPR("pytorch", "pytorch", 106068)

        # note: in reverse order (e.g. self.pr is the last commit, top of the stack)
        mock_get_ghstack_prs.return_value = [
            (pr0, "rev0"),
            (pr1, "rev1"),
            (pr2, "rev2"),
            (pr, "rev123"),
        ]

        mock_merge_rules.return_value = [
            MergeRule(
                "Mock title", patterns=["*"], approved_by=[], mandatory_checks_name=None
            )
        ]

        mock_repo.cherry_pick.return_value = None
        mock_repo.amend_commit_message.return_value = None

        # Call the method under test
        res = pr.merge_ghstack_into(mock_repo, True)

        self.assertEqual(res, [pr2, pr])

        mock_repo.cherry_pick.assert_any_call("rev2")
        mock_repo.cherry_pick.assert_any_call("rev123")

        assert mock.call("rev1") not in mock_repo.cherry_pick.call_args_list

        # Verify the first call
        message = mock_repo.amend_commit_message.call_args_list[0].args[0]
        prefix = (
            "[FSDP] Optimize away intermediate `div_` for HSDP (#106034)\n\n\r\n"
            "### Background: Gradient Pre-Divide"
        )
        suffix = (
            "\nPull Request resolved: https://github.com/pytorch/pytorch/pull/106034\nApproved by: \nghstack "
            "dependencies: #106032, #106033\n"
        )

        assert message.startswith(prefix)
        assert message.endswith(suffix)

        # Verify the second call
        mock_repo.amend_commit_message.assert_any_call(
            "[FSDP] Break up `_post_backward_hook` into smaller funcs (#106068)\n\n\n"
            "Differential Revision: ["
            "D47852461](https://our.internmc.facebook.com/intern/diff/D47852461)\n"
            "Pull Request resolved: "
            "https://github.com/pytorch/pytorch/pull/106068\n"
            "Approved by: \n"
            "ghstack dependencies: #106032, #106033, #106034\n"
        )


if __name__ == "__main__":
    main()
