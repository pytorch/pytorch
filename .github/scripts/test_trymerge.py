#!/usr/bin/env python3
# Tests implemented in this file are relying on GitHub GraphQL APIs
# In order to avoid test flakiness, results of the queries
# are cached in gql_mocks.json
# PyTorch Lint workflow does not have GITHUB_TOKEN defined to avoid
# flakiness, so if you are making changes to merge_rules or
# GraphQL queries in trymerge.py, please make sure to delete `gql_mocks.json`
# And re-run the test locally with ones PAT

import gzip
import json
import os
import warnings
from hashlib import sha256
from typing import Any, Dict, List, Optional
from unittest import main, mock, skip, TestCase
from urllib.error import HTTPError

from github_utils import gh_graphql

from gitutils import get_git_remote_name, get_git_repo_dir, GitRepo

from trymerge import (
    categorize_checks,
    DRCI_CHECKRUN_NAME,
    find_matching_merge_rule,
    get_classifications,
    get_drci_classifications,
    get_rockset_results,
    gh_get_team_members,
    GitHubPR,
    JobCheckState,
    main as trymerge_main,
    MandatoryChecksMissingError,
    MergeRule,
    RE_GHSTACK_DESC,
    read_merge_rules,
    remove_job_name_suffix,
    validate_revert,
)

if "GIT_REMOTE_URL" not in os.environ:
    os.environ["GIT_REMOTE_URL"] = "https://github.com/pytorch/pytorch"

GQL_MOCKS = "gql_mocks.json.gz"
ROCKSET_MOCKS = "rockset_mocks.json.gz"
DRCI_MOCKS = "drci_mocks.json.gz"


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
        with gzip.open(gql_db_fname, encoding="utf-8", mode="rt") as f:
            return json.load(f)

    def save_mocked_queries(obj: Any) -> None:
        with gzip.open(gql_db_fname, encoding="utf-8", mode="wt") as f:
            json.dump(obj, f, indent=2)
            f.write("\n")

    key = key_function(*args)
    mocked_queries = get_mocked_queries()

    if key in mocked_queries:
        return mocked_queries[key]

    try:
        rc = fallback_function(*args)
    except HTTPError as err:
        if err.code == 401 or err.code == 403:
            err_msg = f"If you are seeing this message during workflow run, please make sure to update {file_name}"
            err_msg += f" locally, by deleting it and running {os.path.basename(__file__)} with"
            err_msg += " GitHub Personal Access Token passed via GITHUB_TOKEN,"
            err_msg += " the rockset api key passed via ROCKSET_API_KEY,"
            err_msg += " and drci api key passed via DRCI_BOT_KEY environment variables"
            if (
                os.getenv("GITHUB_TOKEN") is None
                or os.getenv("ROCKSET_API_KEY") is None
                or os.getenv("DRCI_BOT_KEY") is None
            ):
                err_msg = (
                    "Failed to update cached queries as GITHUB_TOKEN or ROCKSET_API_KEY or DRCI_BOT_KEY "
                    + "is not defined. "
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

    return mock_query(gh_graphql_wrapper, GQL_MOCKS, key_function, query, kwargs)


def mocked_rockset_results(head_sha: str, merge_base: str, num_retries: int = 3) -> Any:
    return mock_query(
        get_rockset_results,
        ROCKSET_MOCKS,
        lambda x, y: f"{x} {y}",
        head_sha,
        merge_base,
    )


def mocked_drci_classifications(pr_num: int, project: str, num_retries: int = 3) -> Any:
    return mock_query(
        get_drci_classifications,
        DRCI_MOCKS,
        lambda x, y: f"{x} {y}",
        pr_num,
        project,
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
            self.check_mergeability = False

    return Object()


def mock_remove_label(
    org: str, repo: str, pr_num: str, label: str, dry_run: bool
) -> None:
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
                "pull / linux-xenial-cuda11.3-py3.7-gcc7 / build",
            ],
            ignore_flaky_failures=True,
        ),
        MergeRule(
            name="xla",
            patterns=[".github/ci_commit_pins/xla.txt"],
            approved_by=["pytorchbot"],
            mandatory_checks_name=[
                "Lint",
                "EasyCLA",
                "pull / linux-focal-py3_8-clang9-xla / build",
                "pull / linux-focal-py3_8-clang9-xla / test (xla, 1, 1, linux.12xlarge)",
            ],
            ignore_flaky_failures=True,
        ),
    ]


def mocked_read_merge_rules_approvers(
    repo: Any, org: str, project: str
) -> List[MergeRule]:
    return [
        MergeRule(
            name="Core Reviewers",
            patterns=["*"],
            approved_by=["1", "2", "3", "4", "5", "6"],
            mandatory_checks_name=[
                "Lint",
                "pull",
            ],
        ),
        MergeRule(
            name="Core Maintainers",
            patterns=["*"],
            approved_by=["1", "2", "malfet"],
            mandatory_checks_name=[
                "Lint",
                "pull",
            ],
        ),
    ]


def mocked_read_merge_rules_raise(repo: Any, org: str, project: str) -> List[MergeRule]:
    raise RuntimeError("testing")


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
                "inductor / cuda11.8-py3.10-gcc7-sm86 / test (inductor_torchbench_dynamic, 1, 1, linux.g5.4xlarge.nvidia.gpu)",
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


@mock.patch("trymerge.get_rockset_results", side_effect=empty_rockset_results)
@mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
@mock.patch(
    "trymerge.get_drci_classifications", side_effect=mocked_drci_classifications
)
class TestTryMerge(TestCase):
    def test_merge_rules_valid(self, *args: Any) -> None:
        "Test that merge_rules.yaml can be parsed"
        repo = DummyGitRepo()
        merge_rules = read_merge_rules(repo, "pytorch", "pytorch")
        self.assertGreater(len(merge_rules), 1)

    @mock.patch("trymerge.read_merge_rules", side_effect=mocked_read_merge_rules)
    def test_match_rules(self, *args: Any) -> None:
        "Tests that PR passes merge rules"
        pr = GitHubPR("pytorch", "pytorch", 109999)
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

    @mock.patch(
        "trymerge.read_merge_rules", side_effect=mocked_read_merge_rules_approvers
    )
    def test_match_rules_approvers(self, *args: Any) -> None:
        "Tests that PR has the necessary approvers"
        repo = DummyGitRepo()

        pr = GitHubPR("pytorch", "pytorch", 115329)
        # Test that all potential approvers across all rules are listed if the
        # PR doesn't have one of them
        for mock_rule in ["Core Reviewers", "Core Maintainers"]:
            self.assertRaisesRegex(
                RuntimeError,
                mock_rule,
                lambda: find_matching_merge_rule(pr, repo),
            )

        pr = GitHubPR("pytorch", "pytorch", 115495)
        # Test that PR with the correct approvers doesn't raise any exception
        self.assertTrue(find_matching_merge_rule(pr, repo) is not None)

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

    def test_large_diff(self, *args: Any) -> None:
        "Tests that PR with 100+ files can be fetched"
        pr = GitHubPR("pytorch", "pytorch", 73099)
        self.assertTrue(pr.get_changed_files_count() > 100)
        flist = pr.get_changed_files()
        self.assertEqual(len(flist), pr.get_changed_files_count())

    def test_internal_changes(self, *args: Any) -> None:
        "Tests that PR with internal changes is detected"
        pr = GitHubPR("pytorch", "pytorch", 110140)
        self.assertTrue(pr.has_internal_changes())

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
        # NS(09/27/2023): GitHub seems to recycle older checkruns
        # https://github.com/pytorch/pytorch/pull/68111/checks shows 0 runs
        # self.assertGreater(len(pr.get_checkrun_conclusions()), 3)
        self.assertGreater(pr.get_commit_count(), 60)

    def test_gql_retrieve_checksuites(self, *args: Any) -> None:
        "Fetch comments and conclusions for PR with 60 commits"
        pr = GitHubPR("pytorch", "pytorch", 94787)
        self.assertEqual(len(pr.get_checkrun_conclusions()), 182)

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

    def get_co_authors(self, *args: Any) -> None:
        """Tests that co-authors are recognized"""
        pr = GitHubPR("pytorch", "pytorch", 118347)
        authors = pr.get_authors()
        self.assertIn("kit1980", authors)
        self.assertIn("Co-authored-by:", pr.gen_commit_message())

    def test_get_checkruns_many_runs(self, *args: Any) -> None:
        """Tests that all checkruns can be fetched"""
        pr = GitHubPR("pytorch", "pytorch", 105260)
        conclusions = pr.get_checkrun_conclusions()
        self.assertEqual(len(conclusions), 221)
        self.assertTrue(
            "pull / linux-docs / build-docs-cpp-false" in conclusions.keys()
        )

    def test_cancelled_gets_ignored(self, *args: Any) -> None:
        """Tests that cancelled workflow does not override existing successfull status"""
        pr = GitHubPR("pytorch", "pytorch", 110367)
        conclusions = pr.get_checkrun_conclusions()
        lint_checks = [name for name in conclusions.keys() if "Lint" in name]
        self.assertTrue(len(lint_checks) > 0)
        self.assertTrue(
            all(conclusions[name].status == "SUCCESS" for name in lint_checks)
        )

    def test_get_review_comment_by_id(self, *args: Any) -> None:
        """Tests that even if the comment requested was actually a review instead of a simple comment, we can still find it"""
        pr = GitHubPR("pytorch", "pytorch", 107070)
        review_comment_id = 1582767635
        comment = pr.get_comment_by_id(review_comment_id)
        self.assertIsNotNone(comment)

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

    def test_get_merge_base(self, *args: Any) -> None:
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
@mock.patch(
    "trymerge.get_drci_classifications", side_effect=mocked_drci_classifications
)
class TestBypassFailures(TestCase):
    def test_get_classifications(self, *args: Any) -> None:
        pr = GitHubPR("pytorch", "pytorch", 109584)
        checks = pr.get_checkrun_conclusions()
        checks = get_classifications(
            pr.pr_num,
            pr.project,
            checks,
            [],
        )
        self.assertTrue(
            checks[
                "pull / linux-focal-py3.11-clang10 / test (dynamo, 1, 2, linux.2xlarge)"
            ].classification
            == "BROKEN_TRUNK"
        )
        self.assertTrue(
            checks[
                "trunk / win-vs2019-cpu-py3 / test (default, 2, 3, windows.4xlarge.nonephemeral)"
            ].classification
            == "FLAKY"
        )
        self.assertTrue(
            checks[
                "pull / linux-jammy-py3.8-gcc11 / test (distributed, 1, 2, linux.2xlarge)"
            ].classification
            == "FLAKY"
        )
        self.assertTrue(
            checks[
                "pull / linux-focal-cuda11.8-py3.10-gcc9 / test (distributed, 1, 3, linux.8xlarge.nvidia.gpu)"
            ].classification
            == "FLAKY"
        )

        # Set the threshold larger or equal to the number of ok failures
        pending, failed, ignorable = categorize_checks(
            checks, list(checks.keys()), ok_failed_checks_threshold=6
        )
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 0)
        self.assertTrue(len(ignorable["FLAKY"]) == 4)
        self.assertTrue(len(ignorable["BROKEN_TRUNK"]) == 2)

        # Not set any threshold, defaults to -1 to ignore all flaky and broken trunk failures
        pending, failed, ignorable = categorize_checks(checks, list(checks.keys()))
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 0)
        self.assertTrue(len(ignorable["FLAKY"]) == 4)
        self.assertTrue(len(ignorable["BROKEN_TRUNK"]) == 2)

        # Set the threshold lower than the number of ok failures
        pending, failed, ignorable = categorize_checks(
            checks, list(checks.keys()), ok_failed_checks_threshold=1
        )
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 6)
        self.assertTrue(len(ignorable["FLAKY"]) == 4)
        self.assertTrue(len(ignorable["BROKEN_TRUNK"]) == 2)

        # Set the threshold to 0 like when ignore_flaky_failures is on
        pending, failed, ignorable = categorize_checks(
            checks, list(checks.keys()), ok_failed_checks_threshold=1
        )
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 6)
        self.assertTrue(len(ignorable["FLAKY"]) == 4)
        self.assertTrue(len(ignorable["BROKEN_TRUNK"]) == 2)

    def test_get_classifications_flaky_fullname(self, *args: Any) -> None:
        pr = GitHubPR("pytorch", "pytorch", 110362)
        checks = pr.get_checkrun_conclusions()
        checks = get_classifications(
            pr.pr_num,
            pr.project,
            checks,
            [],
        )
        pending, failed, ignorable = categorize_checks(checks, list(checks.keys()))
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 0)
        self.assertTrue(len(ignorable["FLAKY"]) == 1)

    def test_get_classifications_invalid_cancel(self, *args: Any) -> None:
        pr = GitHubPR("pytorch", "pytorch", 110367)
        checks = pr.get_checkrun_conclusions()
        checks = get_classifications(
            pr.pr_num,
            pr.project,
            checks,
            [],
        )
        pending, failed, ignorable = categorize_checks(checks, list(checks.keys()))
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 0)
        self.assertTrue(len(ignorable["FLAKY"]) == 0)
        self.assertTrue(len(ignorable["BROKEN_TRUNK"]) == 0)
        self.assertTrue(len(ignorable["UNSTABLE"]) == 3)

    def test_get_classifications_similar_failures(self, *args: Any) -> None:
        pr = GitHubPR("pytorch", "pytorch", 109750)
        checks = pr.get_checkrun_conclusions()
        checks = get_classifications(
            pr.pr_num,
            pr.project,
            checks,
            [],
        )
        pending, failed, ignorable = categorize_checks(checks, list(checks.keys()))
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 0)
        self.assertTrue(len(ignorable["FLAKY"]) == 1)

    def test_get_classifications_unstable(self, *args: Any) -> None:
        pr = GitHubPR("pytorch", "pytorch", 104312)
        checks = pr.get_checkrun_conclusions()
        checks = get_classifications(
            pr.pr_num,
            pr.project,
            checks,
            [],
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

        # Add another test case where there is no unstable keyword in the job name, but
        # the job has already been marked as unstable
        pr = GitHubPR("pytorch", "executorch", 3318)
        checks = pr.get_checkrun_conclusions()
        checks = get_classifications(
            pr.pr_num,
            pr.project,
            checks,
            [],
        )
        print(checks)
        workflow_name = "test-llama-app"
        job_name = "mobile-job (android)"
        self.assertTrue(
            checks[f"Android / {workflow_name} / {job_name}"].classification
            == "UNSTABLE"
        )
        pending, failed, ignorable = categorize_checks(
            checks, list(checks.keys()), ok_failed_checks_threshold=1
        )
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 0)
        self.assertTrue(len(ignorable["UNSTABLE"]) == 1)

    def test_get_classifications_broken_trunk(self, *args: Any) -> None:
        # The mock merge base is the actual value returned by gh_fetch_merge_base
        test_cases = [
            {
                # This PR had one broken trunk failure but it was run on a different shard
                # than the one on the base commit. This should still count as broken trunk
                "pr_num": 104214,
                "related_failure_count": 0,
                "flaky_or_broken_trunk": 1,
            },
            {
                # This PR had one broken trunk failure and it used ghstack
                "pr_num": 105145,
                "related_failure_count": 0,
                "flaky_or_broken_trunk": 1,
            },
            {
                # The failure on the merge base was retried successfully and
                # its conclusion changed from failure to success. We want to
                # keep the failure record from the merge base so that it can
                # be used to detect broken trunk
                "pr_num": 107160,
                "related_failure_count": 0,
                "flaky_or_broken_trunk": 1,
            },
            {
                # This PR used Dr.CI broken trunk classification
                "pr_num": 111253,
                "related_failure_count": 1,
                "flaky_or_broken_trunk": 1,
            },
        ]

        for case in test_cases:
            pr_num = case["pr_num"]
            related_failure_count = case["related_failure_count"]
            flaky_or_broken_trunk = case["flaky_or_broken_trunk"]

            pr = GitHubPR("pytorch", "pytorch", pr_num)
            checks = pr.get_checkrun_conclusions()
            checks = get_classifications(
                pr.pr_num,
                pr.project,
                checks,
                [],
            )

            pending, failed, _ = categorize_checks(checks, list(checks.keys()))
            self.assertTrue(len(pending) == 0)
            self.assertTrue(len(failed) == related_failure_count)

            # When the ok_failed_checks_threshold is set to 0, the broken trunk failure
            # won't be ignored
            pending, failed, _ = categorize_checks(
                checks, list(checks.keys()), ok_failed_checks_threshold=0
            )
            self.assertTrue(len(pending) == 0)
            self.assertTrue(
                len(failed) == flaky_or_broken_trunk + related_failure_count
            )

    def test_ignore_current(self, *args: Any) -> None:
        # Test various interactions of the failure classifier to ensure that ignore
        # current checks takes place after other classifications: flaky, unstable,
        # or broken trunk. Only actual new failures should be kept in the list of
        # ignore current checks to use to record force merge with actual failures
        flaky = "pull / linux-focal-cuda11.8-py3.10-gcc9 / test (distributed, 1, 3, linux.8xlarge.nvidia.gpu)"
        broken_trunk = (
            "pull / linux-focal-py3.11-clang10 / test (dynamo, 1, 2, linux.2xlarge)"
        )

        pr = GitHubPR("pytorch", "pytorch", 109584)
        checks = pr.get_checkrun_conclusions()

        # Known flaky failure takes precedence over ignore current (need to set the
        # merge base here to get the results from Rockset, and that categorize the
        # broken trunk failure too
        checks = get_classifications(
            pr.pr_num,
            pr.project,
            checks,
            [broken_trunk, flaky],
        )
        self.assertTrue(checks[flaky].classification == "FLAKY")
        self.assertTrue(checks[broken_trunk].classification == "BROKEN_TRUNK")
        _, failed, ignorable = categorize_checks(checks, list(checks.keys()))
        self.assertTrue(len(failed) == 0)
        self.assertTrue(len(ignorable["IGNORE_CURRENT_CHECK"]) == 0)
        self.assertTrue(len(ignorable["FLAKY"]) == 4)
        self.assertTrue(len(ignorable["BROKEN_TRUNK"]) == 2)

    def test_get_classifications_wrong_workflow_name(self, *args: Any) -> None:
        pr = GitHubPR("pytorch", "pytorch", 123104)
        checks = pr.get_checkrun_conclusions()

        check_name = "linux-binary-conda / conda-py3_8-cuda11_8-build / build"
        check_name_workflow_path = ".github/workflows/generated-linux-binary-conda-nightly.yml / conda-py3_8-cuda11_8-build / build"

        # Mock a check where the workflow name uses the full path
        checks[check_name_workflow_path] = JobCheckState(
            check_name_workflow_path,
            checks[check_name].url,
            checks[check_name].status,
            checks[check_name].classification,
            checks[check_name].job_id,
            checks[check_name].title,
            checks[check_name].summary,
        )
        del checks[check_name]

        checks = get_classifications(
            pr.pr_num,
            pr.project,
            checks,
            [],
        )
        pending, failed, ignorable = categorize_checks(
            checks,
            list(checks.keys()),
        )

        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 0)
        self.assertTrue(len(ignorable["FLAKY"]) == 1)
        self.assertTrue(len(ignorable["BROKEN_TRUNK"]) == 0)

    @mock.patch("trymerge.read_merge_rules", side_effect=xla_merge_rules)
    def test_dont_ignore_flaky_failures(self, *args: Any) -> None:
        """
        Regression test for https://github.com/pytorch/test-infra/issues/4126
        """
        pr = GitHubPR("pytorch", "pytorch", 105312)
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
@mock.patch("trymerge.get_drci_classifications", return_value={})
class TestBypassFailuresOnSandCastle(TestCase):
    def test_get_classifications(self, *args: Any) -> None:
        pr = GitHubPR("pytorch", "pytorch", 111467)
        checks = pr.get_checkrun_conclusions()
        checks = get_classifications(
            pr.pr_num,
            pr.project,
            checks,
            [],
        )
        pending, failed, ignorable = categorize_checks(checks, list(checks.keys()))
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 0)
        self.assertTrue(len(ignorable["FLAKY"]) == 1)
        self.assertTrue(len(ignorable["BROKEN_TRUNK"]) == 1)

    def test_get_classifications_drci_checkrun_not_found(self, *args: Any) -> None:
        pr = GitHubPR("pytorch", "pytorch", 111467)

        # No summary
        checks = pr.get_checkrun_conclusions()
        checks[DRCI_CHECKRUN_NAME] = JobCheckState(
            DRCI_CHECKRUN_NAME,
            "",
            "NEUTRAL",
            None,
            1,
            "",
            None,
        )
        checks = get_classifications(
            pr.pr_num,
            pr.project,
            checks,
            [],
        )
        pending, failed, ignorable = categorize_checks(checks, list(checks.keys()))
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 2)

        # Empty summary
        checks = pr.get_checkrun_conclusions()
        checks[DRCI_CHECKRUN_NAME] = JobCheckState(
            DRCI_CHECKRUN_NAME,
            "",
            "NEUTRAL",
            None,
            1,
            "",
            "",
        )
        checks = get_classifications(
            pr.pr_num,
            pr.project,
            checks,
            [],
        )
        pending, failed, ignorable = categorize_checks(checks, list(checks.keys()))
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 2)

        # No Dr.CI checkrun
        checks = pr.get_checkrun_conclusions()
        del checks[DRCI_CHECKRUN_NAME]
        checks = get_classifications(
            pr.pr_num,
            pr.project,
            checks,
            [],
        )
        pending, failed, ignorable = categorize_checks(checks, list(checks.keys()))
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 2)


@mock.patch("trymerge.get_rockset_results", side_effect=mocked_rockset_results)
@mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
@mock.patch("trymerge.gh_fetch_merge_base", return_value="")
@mock.patch(
    "trymerge.get_drci_classifications", side_effect=mocked_drci_classifications
)
class TestGitHubPRGhstackDependencies(TestCase):
    def test_pr_dependencies(self, *args: Any) -> None:
        pr = GitHubPR("pytorch", "pytorch", 106068)
        msg = pr.gen_commit_message(filter_ghstack=True)
        self.assertEqual(
            msg,
            f"{pr.get_title()} (#106068)\n\n{RE_GHSTACK_DESC.sub('', pr.get_body())}\n"
            "Pull Request resolved: https://github.com/pytorch/pytorch/pull/106068\n"
            "Approved by: https://github.com/ezyang, https://github.com/fegin\n",
        )

    def test_pr_dependencies_ghstack(self, *args: Any) -> None:
        pr0 = GitHubPR("pytorch", "pytorch", 106032)
        pr1 = GitHubPR("pytorch", "pytorch", 106033)
        pr2 = GitHubPR("pytorch", "pytorch", 106034)
        pr = GitHubPR("pytorch", "pytorch", 106068)
        msg = pr.gen_commit_message(filter_ghstack=True, ghstack_deps=[pr0, pr1, pr2])
        self.assertEqual(
            msg,
            f"{pr.get_title()} (#106068)\n\n{RE_GHSTACK_DESC.sub('', pr.get_body())}\n"
            "Pull Request resolved: https://github.com/pytorch/pytorch/pull/106068\n"
            "Approved by: https://github.com/ezyang, https://github.com/fegin\n"
            "ghstack dependencies: #106032, #106033, #106034\n",
        )

    @skip(
        reason="This test is run against a mutalbe PR that has changed, so it no longer works. The test should be changed"
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

        self.assertTrue(mock.call("rev1") not in mock_repo.cherry_pick.call_args_list)

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

        self.assertTrue(message.startswith(prefix))
        self.assertTrue(message.endswith(suffix))

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
