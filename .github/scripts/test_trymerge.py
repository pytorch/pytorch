#!/usr/bin/env python3

# Tests implemented in this file are relying on GitHub GraphQL APIs
# In order to avoid test flakiness, results of the queries
# are cached in gql_mocks.json
# PyTorch Lint workflow does not have GITHUB_TOKEN defined to avoid
# flakiness, so if you are making changes to merge_rules or
# GraphQL queries in trymerge.py, please make sure to delete `gql_mocks.json`
# And re-run the test locally with ones PAT

from __future__ import annotations

import gzip
import json
import os
import warnings
from dataclasses import replace
from hashlib import sha256
from types import MethodType
from typing import Any, TYPE_CHECKING
from unittest import main, mock, skip, TestCase
from urllib.error import HTTPError

from github_utils import gh_graphql, GHGraphQLError
from gitutils import get_git_remote_name, get_git_repo_dir, GitRepo
from greenlight_guard import (
    GREENLIGHT_LOGIN,
    GreenlightWaitWindow,
    GuardResult,
    GuardVerdict,
)
from greenlight_identity import normalize_login
from native_stack import (
    build_native_stack_commits,
    NativeStack,
    NativeStackError,
    PULL_REQUEST_RESOLVED,
    stack_dependencies_line,
    StackEntry,
)
from test_native_stack import author, GitTestCase, pr_url, TRUNK, with_entry
from trymerge import (
    _AUTHORIZED_WITHOUT_GREENLIGHT,
    _find_non_matching_files,
    _revlist_to_prs,
    can_skip_internal_checks,
    categorize_checks,
    check_greenlight_reviewed_head_sha,
    DRCI_CHECKRUN_NAME,
    ensure_mergeable_labels,
    find_matching_merge_rule,
    get_classifications,
    get_docker_build_checks,
    get_drci_classifications,
    get_native_stack_prs,
    get_prs_to_merge,
    get_topmost_docker_pr,
    gh_get_pr_info,
    gh_get_team_members,
    GitHubPR,
    HAS_NO_CONNECTED_DIFF_TITLE,
    IGNORABLE_FAILED_CHECKS_THESHOLD,
    IMPORT_STATUS_CHECKRUN_NAME,
    INTERNAL_CHANGES_CHECKRUN_NAME,
    is_ai_not_related,
    is_authorized_without_greenlight,
    is_bot_initiated_codev_merge,
    is_docker_affecting_files,
    iter_issue_timeline_until_comment,
    JobCheckState,
    main as trymerge_main,
    MandatoryChecksMissingError,
    manually_close_merged_pr,
    merge,
    merge_authorized_logins,
    MERGE_COMPLETE_LABEL,
    MergeRule,
    MergeRuleFailedError,
    NATIVE_STACK_PUSH_ATTEMPTS,
    post_starting_merge_comment,
    PostCommentError,
    RE_GHSTACK_DESC,
    read_merge_rules,
    remove_job_name_suffix,
    REVIEW_PAGE_LIMIT,
    REVIEWS_PER_PAGE,
    sha_from_committed_event,
    sha_from_force_push_after,
    validate_revert,
)


if TYPE_CHECKING:
    from collections.abc import Callable


if "GIT_REMOTE_URL" not in os.environ:
    os.environ["GIT_REMOTE_URL"] = "https://github.com/pytorch/pytorch"

GQL_MOCKS = "gql_mocks.json.gz"
DRCI_MOCKS = "drci_mocks.json.gz"

MALFORMED_TEAM_REF = "pytorch/pytorch-dev-infra/extra"


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

    # TODO: Remove me once https://github.com/pytorch/pytorch/issues/160489 is resolved
    raise ValueError(f"Key {key} could not be found in gql_mocks")

    try:
        rc = fallback_function(*args)
    except HTTPError as err:
        if err.code == 401 or err.code == 403:
            err_msg = f"If you are seeing this message during workflow run, please make sure to update {file_name}"
            err_msg += f" locally, by deleting it and running {os.path.basename(__file__)} with"
            err_msg += " GitHub Personal Access Token passed via GITHUB_TOKEN"
            err_msg += " and drci api key passed via DRCI_BOT_KEY environment variables"
            if os.getenv("GITHUB_TOKEN") is None or os.getenv("DRCI_BOT_KEY") is None:
                err_msg = (
                    "Failed to update cached queries as GITHUB_TOKEN or DRCI_BOT_KEY "
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
            self.comment_id = 12345  # Set to non-zero value
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
    comment_id: int | None = None,
    reason: str | None = None,
) -> None:
    pass


def mock_merge(
    pr: GitHubPR,
    repo: GitRepo,
    comment_id: int,
    dry_run: bool = False,
    skip_mandatory_checks: bool = False,
    timeout_minutes: int = 400,
    stale_pr_days: int = 3,
    ignore_current: bool = False,
    native_stack: NativeStack | None = None,
    trunk_at_start: str | None = None,
) -> None:
    pass


def mock_gh_get_info() -> Any:
    return {
        "closed": False,
        "isCrossRepository": False,
        "headRefName": "foo",
        "baseRefName": "bar",
        "baseRepository": {"defaultBranchRef": {"name": "bar"}},
        "files": {"nodes": [], "pageInfo": {"hasNextPage": False}},
        "changedFiles": 0,
    }


def mocked_read_merge_rules_NE(repo: Any, org: str, project: str) -> list[MergeRule]:
    return [
        MergeRule(
            name="mock with nonexistent check",
            patterns=["*"],
            approved_by=[],
            mandatory_checks_name=["Lint", "Facebook CLA Check", "nonexistent"],
            ignore_flaky_failures=True,
        ),
    ]


def mocked_read_merge_rules(repo: Any, org: str, project: str) -> list[MergeRule]:
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
) -> list[MergeRule]:
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


def mocked_read_merge_rules_greenlight(
    repo: Any, org: str, project: str
) -> list[MergeRule]:
    return [
        MergeRule(
            name="Greenlight Review Bot",
            patterns=["*"],
            approved_by=[GREENLIGHT_LOGIN],
            mandatory_checks_name=["Lint", "pull"],
        ),
        MergeRule(
            name="Core Maintainers",
            patterns=["*"],
            approved_by=["malfet"],
            mandatory_checks_name=["Lint", "pull"],
        ),
    ]


def mocked_read_merge_rules_malformed_team(
    repo: Any, org: str, project: str
) -> list[MergeRule]:
    return [
        MergeRule(
            name="Malformed Team",
            patterns=["*"],
            approved_by=[MALFORMED_TEAM_REF],
            mandatory_checks_name=["Lint", "pull"],
        ),
    ]


def mocked_read_merge_rules_raise(repo: Any, org: str, project: str) -> list[MergeRule]:
    raise RuntimeError("testing")


def xla_merge_rules(repo: Any, org: str, project: str) -> list[MergeRule]:
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


class DummyGitRepo(GitRepo):
    def __init__(self) -> None:
        super().__init__(get_git_repo_dir(), get_git_remote_name())

    def commits_resolving_gh_pr(self, pr_num: int) -> list[str]:
        return ["FakeCommitSha"]

    def commit_message(self, ref: str) -> str:
        return "super awesome commit message"


class TestGetPRInfoForbiddenHeadRepository(TestCase):
    """gh_get_pr_info tolerates a FORBIDDEN error on the headRepository path
    (org fork blocking classic PATs) and nothing else."""

    forbidden_error = {
        "type": "FORBIDDEN",
        "path": ["repository", "pullRequest", "headRepository"],
        "message": "`SomeOrg` forbids access via a personal access token (classic).",
    }

    def graphql_response(self, errors: list[dict[str, Any]]) -> dict[str, Any]:
        pull_request = {"headRefName": "some-branch", "headRepository": None}
        return {"data": {"repository": {"pullRequest": pull_request}}, "errors": errors}

    def test_forbidden_head_repository_is_tolerated(self) -> None:
        rc = self.graphql_response([self.forbidden_error])
        with mock.patch(
            "trymerge.gh_graphql", side_effect=GHGraphQLError("failed", rc)
        ):
            info = gh_get_pr_info("pytorch", "pytorch", 123)
        self.assertIsNone(info["headRepository"])
        self.assertEqual(info["headRefName"], "some-branch")

    def test_other_errors_still_raise(self) -> None:
        other_error = {"type": "NOT_FOUND", "path": ["repository", "pullRequest"]}
        for errors in ([other_error], [self.forbidden_error, other_error], []):
            rc = self.graphql_response(errors)
            with mock.patch(
                "trymerge.gh_graphql", side_effect=GHGraphQLError("failed", rc)
            ):
                with self.assertRaises(GHGraphQLError):
                    gh_get_pr_info("pytorch", "pytorch", 123)

    def test_missing_pull_request_still_raises(self) -> None:
        rc = {"data": {"repository": None}, "errors": [self.forbidden_error]}
        with mock.patch(
            "trymerge.gh_graphql", side_effect=GHGraphQLError("failed", rc)
        ):
            with self.assertRaises(GHGraphQLError):
                gh_get_pr_info("pytorch", "pytorch", 123)


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

    def test_merge_rules_still_grant_greenlight_merge_authority(
        self, *args: Any
    ) -> None:
        """GREENLIGHT_LOGIN is a copy of a login in merge_rules.yaml.

        Renaming it there and not here would leave the guard watching for an approval
        nobody can give, which silently switches the guard off.
        """
        merge_rules = read_merge_rules(DummyGitRepo(), "pytorch", "pytorch")
        self.assertTrue(
            any(
                normalize_login(login) == GREENLIGHT_LOGIN
                for rule in merge_rules
                for login in rule.approved_by
            ),
            f"no merge rule lists {GREENLIGHT_LOGIN} in approved_by",
        )

    def test_negative_pattern_excludes_subpath(self, *args: Any) -> None:
        "Patterns prefixed with '-' exclude matching files from a rule."
        patterns = [".ci/**", "-.ci/docker/**", ".github/**"]
        files = [
            ".ci/test.sh",
            ".ci/docker/Dockerfile",
            ".ci/docker/common/install_onnx.sh",
            ".github/workflows/lint.yml",
            "torch/foo.py",
        ]
        non_matching = _find_non_matching_files(patterns, files)
        self.assertEqual(
            sorted(non_matching),
            [
                ".ci/docker/Dockerfile",
                ".ci/docker/common/install_onnx.sh",
                "torch/foo.py",
            ],
        )

    def test_negative_pattern_no_negatives(self, *args: Any) -> None:
        "Without negative patterns, behavior matches the positive-only case."
        files = [".ci/test.sh", "torch/foo.py"]
        self.assertEqual(_find_non_matching_files([".ci/**"], files), ["torch/foo.py"])

    @staticmethod
    def _pr_with_merge_comment(
        author_login: str,
        author_url: str | None = None,
        diff_revision: str | None = "D123456",
        editor_login: str | None = None,
    ) -> Any:
        pr = mock.MagicMock()
        pr.get_diff_revision.return_value = diff_revision
        pr.get_comment_by_id.return_value = mock.MagicMock(
            author_login=author_login,
            author_url=author_url,
            editor_login=editor_login,
        )
        return pr

    def test_is_bot_initiated_codev_merge_meta_codesync(self, *args: Any) -> None:
        "meta-codesync, the current export bot, is recognized as a co-dev merge."
        pr = self._pr_with_merge_comment(
            "meta-codesync[bot]", "https://github.com/apps/meta-codesync"
        )
        self.assertTrue(is_bot_initiated_codev_merge(pr, 123))

    def test_is_bot_initiated_codev_merge_legacy_bots(self, *args: Any) -> None:
        "The bots meta-codesync superseded are still recognized."
        tools = self._pr_with_merge_comment(
            "facebook-github-tools[bot]",
            "https://github.com/apps/facebook-github-tools",
        )
        self.assertTrue(is_bot_initiated_codev_merge(tools, 123))
        legacy = self._pr_with_merge_comment("facebook-github-bot")
        self.assertTrue(is_bot_initiated_codev_merge(legacy, 123))

    def test_is_bot_initiated_codev_merge_false_without_diff(self, *args: Any) -> None:
        "A bot-initiated merge without an internal diff is not a co-dev merge."
        pr = self._pr_with_merge_comment(
            "meta-codesync[bot]",
            "https://github.com/apps/meta-codesync",
            diff_revision=None,
        )
        self.assertFalse(is_bot_initiated_codev_merge(pr, 123))

    def test_is_bot_initiated_codev_merge_false_when_human(self, *args: Any) -> None:
        "A human-initiated merge is never treated as a co-dev merge."
        pr = self._pr_with_merge_comment("some-human")
        self.assertFalse(is_bot_initiated_codev_merge(pr, 123))

    def test_is_bot_initiated_codev_merge_false_when_edited(self, *args: Any) -> None:
        "An edited merge comment can't be trusted to name its real author."
        pr = self._pr_with_merge_comment(
            "meta-codesync[bot]",
            "https://github.com/apps/meta-codesync",
            editor_login="some-human",
        )
        self.assertFalse(is_bot_initiated_codev_merge(pr, 123))

    def test_codev_bot_list_does_not_widen_internal_checks(self, *args: Any) -> None:
        "Auto-labeling meta-codesync must not also waive the Phabricator guard."
        pr = self._pr_with_merge_comment(
            "meta-codesync[bot]", "https://github.com/apps/meta-codesync"
        )
        self.assertTrue(is_bot_initiated_codev_merge(pr, 123))
        self.assertFalse(can_skip_internal_checks(pr, 123))

    def test_is_bot_initiated_codev_merge_without_author_url(self, *args: Any) -> None:
        """Recognition must not depend on author_url. Only `comments(last: 5)`
        selects it; GH_GET_PR_PREV_COMMENTS and the reviews fragment select
        `login` alone, so a merge comment older than the prefetched window comes
        back with author_url=None."""
        node = {
            "bodyText": "@pytorchbot merge",
            "createdAt": "2026-08-04T22:14:27Z",
            # No "url" — exactly what the paginated query returns.
            "author": {"login": "meta-codesync[bot]"},
            "authorAssociation": "NONE",
            "editor": None,
            "databaseId": 5185216222,
            "url": "https://github.com/pytorch/pytorch/pull/192125#issuecomment-1",
        }
        comment = GitHubPR._comment_from_node(node)
        self.assertIsNone(comment.author_url)

        pr = mock.MagicMock()
        pr.get_diff_revision.return_value = "D114427100"
        pr.get_comment_by_id.return_value = comment
        self.assertTrue(is_bot_initiated_codev_merge(pr, 5185216222))

    @mock.patch("trymerge.gh_post_pr_comment")
    @mock.patch("trymerge.gh_add_labels")
    @mock.patch("trymerge.is_bot_initiated_codev_merge", return_value=True)
    @mock.patch("trymerge.has_required_labels", return_value=False)
    def test_ensure_mergeable_labels_autolabels_codev_merge(
        self,
        mock_labels: Any,
        mock_codev: Any,
        mock_add: Any,
        mock_comment: Any,
        *args: Any,
    ) -> None:
        "An unlabeled co-dev merge is auto-labeled instead of raising."
        pr = mock.MagicMock(org="pytorch", project="pytorch", pr_num=123)
        ensure_mergeable_labels(pr, 456, dry_run=False)
        mock_add.assert_called_once_with(
            "pytorch", "pytorch", 123, ["topic: not user facing"], False
        )
        mock_comment.assert_called_once()

    @mock.patch("trymerge.gh_post_pr_comment", side_effect=RuntimeError("boom"))
    @mock.patch("trymerge.gh_add_labels")
    @mock.patch("trymerge.is_bot_initiated_codev_merge", return_value=True)
    @mock.patch("trymerge.has_required_labels", return_value=False)
    def test_ensure_mergeable_labels_does_not_label_without_audit_comment(
        self,
        mock_labels: Any,
        mock_codev: Any,
        mock_add: Any,
        mock_comment: Any,
        *args: Any,
    ) -> None:
        """A failed audit comment must not leave the label behind: the label alone
        satisfies has_required_labels, so the retry would silently skip the notice."""
        pr = mock.MagicMock(org="pytorch", project="pytorch", pr_num=123)
        with self.assertRaises(RuntimeError):
            ensure_mergeable_labels(pr, 456, dry_run=False)
        mock_add.assert_not_called()

    @mock.patch("trymerge.gh_add_labels")
    @mock.patch("trymerge.is_bot_initiated_codev_merge", return_value=False)
    @mock.patch("trymerge.has_required_labels", return_value=False)
    def test_ensure_mergeable_labels_raises_for_human_merge(
        self, mock_labels: Any, mock_codev: Any, mock_add: Any, *args: Any
    ) -> None:
        "An unlabeled human-initiated merge still fails the label check."
        pr = mock.MagicMock(org="pytorch", project="pytorch", pr_num=123)
        with self.assertRaises(RuntimeError):
            ensure_mergeable_labels(pr, 456, dry_run=False)
        mock_add.assert_not_called()

    @mock.patch("trymerge.gh_add_labels")
    @mock.patch("trymerge.has_required_labels", return_value=True)
    def test_ensure_mergeable_labels_noop_when_labeled(
        self, mock_labels: Any, mock_add: Any, *args: Any
    ) -> None:
        "A PR that already has a required label is left untouched."
        pr = mock.MagicMock(org="pytorch", project="pytorch", pr_num=123)
        ensure_mergeable_labels(pr, 456, dry_run=False)
        mock_add.assert_not_called()

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

    @mock.patch(
        "trymerge.read_merge_rules",
        side_effect=mocked_read_merge_rules_malformed_team,
    )
    def test_match_rules_malformed_team_ref(self, *args: Any) -> None:
        "Tests that a multi-slash approved_by entry aborts instead of matching any approval"
        pr = GitHubPR("pytorch", "pytorch", 115495)
        repo = DummyGitRepo()

        with mock.patch(
            "trymerge.gh_get_team_members", return_value=[]
        ) as mock_members:
            self.assertRaisesRegex(
                ValueError,
                MALFORMED_TEAM_REF,
                lambda: find_matching_merge_rule(pr, repo),
            )
        mock_members.assert_not_called()

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

    @skip("GitHub doesn't keep this data anymore")
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
        if pr._reviews is None:  # to pacify mypy
            raise AssertionError("pr._reviews is None")
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
        self.assertTrue("pull / linux-docs / build-docs-cpp-false" in conclusions)

    def test_cancelled_gets_ignored(self, *args: Any) -> None:
        """Tests that cancelled workflow does not override existing successful status"""
        pr = GitHubPR("pytorch", "pytorch", 110367)
        conclusions = pr.get_checkrun_conclusions()
        lint_checks = [name for name in conclusions if "Lint" in name]
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
            comment_id=mock.ANY,
            dry_run=mock.ANY,
            skip_mandatory_checks=True,
            ignore_current=False,
            native_stack=None,
            trunk_at_start=None,
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
            comment_id=mock.ANY,
            dry_run=mock.ANY,
            skip_mandatory_checks=False,
            ignore_current=False,
            native_stack=None,
            trunk_at_start=None,
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
                "name": "linux-bionic-cuda12.6-py3.10-gcc9-sm86 / test (default, 1, 5, linux.g5.4xlarge.nvidia.gpu)",
                "expected": "linux-bionic-cuda12.6-py3.10-gcc9-sm86 / test (default)",
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

    def test_app_can_revert(self, *args: Any) -> None:
        pr = GitHubPR("pytorch", "pytorch", 164660)
        repo = DummyGitRepo()
        app_comment_id, impostor_comment_id = 3375785595, 3377647892
        # Check that app can revert
        self.assertIsNotNone(validate_revert(repo, pr, comment_id=app_comment_id))
        # But impostor can not
        self.assertRaises(
            PostCommentError,
            lambda: validate_revert(repo, pr, comment_id=impostor_comment_id),
        )
        # Despite it's name being the name of the bot
        self.assertEqual(
            pr.get_comment_by_id(impostor_comment_id).author_login,
            "pytorch-auto-revert",
        )


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
            None,
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
            None,
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
            None,
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
            None,
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
            None,
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
            None,
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
                None,
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
        # merge base here to get the results from Dr. CI, and that categorize the
        # broken trunk failure too
        checks = get_classifications(
            pr.pr_num,
            pr.project,
            checks,
            {(pr.pr_num, broken_trunk), (pr.pr_num, flaky)},
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
            None,
        )
        pending, failed, ignorable = categorize_checks(
            checks,
            list(checks.keys()),
        )

        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 0)
        self.assertTrue(len(ignorable["FLAKY"]) == 1)
        self.assertTrue(len(ignorable["BROKEN_TRUNK"]) == 0)

    def test_ignore_failures_older_run_same_workflow(self, *args: Any) -> None:
        pr = GitHubPR("pytorch", "pytorch", 129013)
        checks = pr.get_checkrun_conclusions()
        checks = get_classifications(
            pr.pr_num,
            pr.project,
            checks,
            None,
        )
        pending, failed, ignorable = categorize_checks(
            checks,
            list(checks.keys()),
        )
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 0)
        self.assertTrue(len(ignorable["FLAKY"]) == 2)
        self.assertTrue(len(ignorable["UNSTABLE"]) == 13)

    @mock.patch("trymerge.read_merge_rules", side_effect=xla_merge_rules)
    def test_dont_ignore_flaky_failures(self, *args: Any) -> None:
        """
        Regression test for https://github.com/pytorch/test-infra/issues/4126
        """
        pr = GitHubPR("pytorch", "pytorch", 105312)
        repo = DummyGitRepo()
        # Check that failure is classified as flaky but still raises exception
        with warnings.catch_warnings(record=True) as w, self.assertRaises(RuntimeError):
            find_matching_merge_rule(pr, repo)
        self.assertEqual(len(w), 1)
        self.assertIn(
            "1 checks failed but were likely due flakiness or broken trunk",
            str(w[0].message),
        )

    def test_get_classifications_crcr_l3(self, *args: Any) -> None:
        """Test that CRCR L3 failures are classified as CRCR_L3
        and are always non-blocking regardless of the ok_failed_checks_threshold."""
        pr = GitHubPR("pytorch", "pytorch", 100652)
        checks = pr.get_checkrun_conclusions()
        checks = get_classifications(
            pr.pr_num,
            pr.project,
            checks,
            None,
        )
        oot_check = (
            "inductor / cuda11.8-py3.10-gcc7-sm86"
            " / test (inductor_timm, 2, 2, linux.g5.4xlarge.nvidia.gpu)"
        )
        self.assertEqual(checks[oot_check].classification, "CRCR_L3")

        # BROKEN_TRUNK classification still works independently
        bt_check = (
            "inductor / cuda11.8-py3.10-gcc7-sm86"
            " / test (inductor_torchbench_dynamic, 1, 1, linux.g5.4xlarge.nvidia.gpu)"
        )
        self.assertEqual(checks[bt_check].classification, "BROKEN_TRUNK")

        # CRCR_L3 is always non-blocking: ignored by default
        pending, failed, ignorable = categorize_checks(checks, list(checks.keys()))
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 0)
        self.assertTrue(len(ignorable["CRCR_L3"]) == 1)

        # CRCR_L3 stays ignored even with threshold=0, unlike flaky/broken_trunk
        # which get promoted to blocking failures when the threshold is exceeded
        pending, failed, ignorable = categorize_checks(
            checks, list(checks.keys()), ok_failed_checks_threshold=0
        )
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(ignorable["CRCR_L3"]) == 1)


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
            None,
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
            None,
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
            None,
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
            None,
        )
        pending, failed, ignorable = categorize_checks(checks, list(checks.keys()))
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 2)


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
        reason="This test is run against a mutable PR that has changed, so it no longer works. The test should be changed"
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

    @mock.patch.object(GitHubPR, "is_closed", return_value=False)
    @mock.patch("trymerge.find_matching_merge_rule")
    @mock.patch("trymerge.GitRepo")
    @mock.patch("trymerge.get_ghstack_prs")
    def test_merge_ghstack_into_wraps_parent_rule_error(
        self,
        mock_get_ghstack_prs: mock.MagicMock,
        mock_repo: mock.MagicMock,
        mock_find_matching_merge_rule: mock.MagicMock,
        _mock_is_closed: mock.MagicMock,
        *args: Any,
    ) -> None:
        """
        When a stacked dependency PR fails the merge-rule check, the error
        raised by merge_ghstack_into should identify the failing PR number
        and preserve the original exception subclass.
        """
        parent_pr = GitHubPR("pytorch", "pytorch", 106034)
        top_pr = GitHubPR("pytorch", "pytorch", 106068)

        mock_get_ghstack_prs.return_value = [
            (parent_pr, "rev_parent"),
            (top_pr, "rev_top"),
        ]

        inner_msg = "Approvers from one of the following sets are needed"
        mock_find_matching_merge_rule.side_effect = MergeRuleFailedError(inner_msg)

        with self.assertRaises(MergeRuleFailedError) as cm:
            top_pr.merge_ghstack_into(mock_repo, True)

        self.assertIn("#106034", str(cm.exception))
        self.assertIn(inner_msg, str(cm.exception))
        self.assertNotIsInstance(cm.exception, MandatoryChecksMissingError)

    @mock.patch.object(GitHubPR, "is_closed", return_value=False)
    @mock.patch("trymerge.find_matching_merge_rule")
    @mock.patch("trymerge.GitRepo")
    @mock.patch("trymerge.get_ghstack_prs")
    def test_merge_ghstack_into_preserves_mandatory_checks_subclass(
        self,
        mock_get_ghstack_prs: mock.MagicMock,
        mock_repo: mock.MagicMock,
        mock_find_matching_merge_rule: mock.MagicMock,
        _mock_is_closed: mock.MagicMock,
        *args: Any,
    ) -> None:
        """The wrapping must preserve MandatoryChecksMissingError so callers
        that catch it specifically (e.g. for retry behavior) keep working."""
        parent_pr = GitHubPR("pytorch", "pytorch", 106034)
        top_pr = GitHubPR("pytorch", "pytorch", 106068)
        mock_get_ghstack_prs.return_value = [
            (parent_pr, "rev_parent"),
            (top_pr, "rev_top"),
        ]
        mock_find_matching_merge_rule.side_effect = MandatoryChecksMissingError(
            "1 mandatory check(s) failed"
        )

        with self.assertRaises(MandatoryChecksMissingError) as cm:
            top_pr.merge_ghstack_into(mock_repo, True)

        self.assertIn("#106034", str(cm.exception))

    @mock.patch.object(GitHubPR, "is_closed", return_value=False)
    @mock.patch("trymerge.can_skip_internal_checks", return_value=False)
    @mock.patch("trymerge.find_matching_merge_rule")
    @mock.patch("trymerge.GitRepo")
    @mock.patch("trymerge.get_ghstack_prs")
    def test_merge_ghstack_into_gates_lower_prs_with_the_waivers(
        self,
        mock_get_ghstack_prs: mock.MagicMock,
        mock_repo: mock.MagicMock,
        mock_find_matching_merge_rule: mock.MagicMock,
        _mock_can_skip_internal_checks: mock.MagicMock,
        _mock_is_closed: mock.MagicMock,
        *args: Any,
    ) -> None:
        """The per-stacked-PR gate, which `merge -i` never used to reach at all."""
        parent_pr = GitHubPR("pytorch", "pytorch", 106034)
        top_pr = GitHubPR("pytorch", "pytorch", 106068)
        waivers = {(parent_pr.pr_num, "parent-red"), (top_pr.pr_num, "top-red")}
        mock_get_ghstack_prs.return_value = [
            (parent_pr, "rev_parent"),
            (top_pr, "rev_top"),
        ]
        mock_find_matching_merge_rule.return_value = (None, [], [], {})

        top_pr.merge_ghstack_into(mock_repo, True, ignore_current_checks=waivers)

        mock_find_matching_merge_rule.assert_called_once()
        self.assertEqual(
            mock_find_matching_merge_rule.call_args.kwargs["ignore_current_checks"],
            waivers,
        )


@mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
@mock.patch("trymerge.gh_fetch_merge_base", return_value="")
@mock.patch(
    "trymerge.get_drci_classifications", side_effect=mocked_drci_classifications
)
@mock.patch.object(DummyGitRepo, "commit_message")
class TestRevListToPR(TestCase):
    # Tests for _revlist_to_prs function
    def test__revlist_to_prs_zero_matches(
        self, mock_commit_message: mock.MagicMock, *args: Any
    ) -> None:
        # If zero PRs are mentioned in the commit message, it should raise an error
        pr_num = 154098
        pr = GitHubPR("pytorch", "pytorch", pr_num)
        repo = DummyGitRepo()
        mock_commit_message.return_value = "no PRs"
        self.assertRaisesRegex(
            RuntimeError,
            "PRs mentioned in commit dummy: 0.",
            lambda: _revlist_to_prs(repo, pr, ["dummy"]),
        )

    def test__revlist_to_prs_two_prs(
        self, mock_commit_message: mock.MagicMock, *args: Any
    ) -> None:
        # If two PRs are mentioned in the commit message, it should raise an error
        pr_num = 154394
        pr = GitHubPR("pytorch", "pytorch", pr_num)
        repo = DummyGitRepo()
        # https://github.com/pytorch/pytorch/commit/343c56e7650f55fd030aca0b9275d6d73501d3f4

        commit_message = """add sticky cache pgo

ghstack-source-id: 9bc6dee0b427819f978bfabccb72727ba8be2f81
Pull-Request-resolved: https://github.com/pytorch/pytorch/pull/154098

ghstack-source-id: 9bc6dee0b427819f978bfabccb72727ba8be2f81
Pull Request resolved: https://github.com/pytorch/pytorch/pull/154394"""
        mock_commit_message.return_value = commit_message
        self.assertRaisesRegex(
            RuntimeError,
            "PRs mentioned in commit dummy: 2.",
            lambda: _revlist_to_prs(repo, pr, ["dummy"]),
        )


@mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
@mock.patch("trymerge.gh_fetch_merge_base", return_value="")
@mock.patch(
    "trymerge.get_drci_classifications", side_effect=mocked_drci_classifications
)
class TestTimelineFunctions(TestCase):
    """Tests for the new timeline-related functions"""

    def test_sha_from_committed_event(self, *args: Any) -> None:
        """Test extracting SHA from committed event"""
        # Based on actual GitHub API format - committed events have "sha" at top level
        event = {
            "event": "committed",
            "sha": "fb21ce932ded6670c918804a0d9151b773770a7c",
        }
        self.assertEqual(
            sha_from_committed_event(event), "fb21ce932ded6670c918804a0d9151b773770a7c"
        )

        # Test with missing SHA
        event_no_sha = {"event": "committed"}
        self.assertIsNone(sha_from_committed_event(event_no_sha))

    def test_sha_from_force_push_after(self, *args: Any) -> None:
        """Test extracting SHA from force push event"""
        # NOTE: The current function doesn't handle the actual GitHub API format
        # Real force push events have "commit_id" at top level, but this function
        # looks for "after", "after_commit", "after_sha", or "head_sha" fields

        # Test with the legacy format the current function handles
        event_legacy = {
            "event": "head_ref_force_pushed",
            "after": {"sha": "ef22bcbc54bb0f787e1e4ffd3d83df18fc407f5e"},
        }
        self.assertEqual(
            sha_from_force_push_after(event_legacy),
            "ef22bcbc54bb0f787e1e4ffd3d83df18fc407f5e",
        )

        # Test with current GitHub API format (should return None with current implementation)
        event_real_api = {
            "event": "head_ref_force_pushed",
            "commit_id": "ef22bcbc54bb0f787e1e4ffd3d83df18fc407f5e",
        }
        self.assertEqual(
            sha_from_force_push_after(event_real_api),
            "ef22bcbc54bb0f787e1e4ffd3d83df18fc407f5e",
        )  # Current function doesn't handle commit_id

        # Test with missing SHA
        event_no_sha = {"event": "head_ref_force_pushed"}
        self.assertIsNone(sha_from_force_push_after(event_no_sha))

    @mock.patch("trymerge.gh_fetch_json_list")
    def test_iter_issue_timeline_until_comment(
        self, mock_gh_fetch_json_list: Any, *args: Any
    ) -> None:
        """Test timeline iteration until target comment"""
        # Mock timeline data based on actual GitHub API format
        timeline_data = [
            {"event": "commented", "id": 100, "body": "first comment"},
            {"event": "committed", "sha": "fb21ce932ded6670c918804a0d9151b773770a7c"},
            {"event": "commented", "id": 200, "body": "target comment"},
            {"event": "commented", "id": 300, "body": "after target"},
        ]
        mock_gh_fetch_json_list.return_value = timeline_data

        # Test iteration stops at target comment
        events = list(iter_issue_timeline_until_comment("pytorch", "pytorch", 123, 200))
        self.assertEqual(len(events), 3)  # Should stop at target comment
        self.assertEqual(events[0]["event"], "commented")
        self.assertEqual(events[0]["id"], 100)
        self.assertEqual(events[1]["event"], "committed")
        self.assertEqual(events[1]["sha"], "fb21ce932ded6670c918804a0d9151b773770a7c")
        self.assertEqual(events[2]["event"], "commented")
        self.assertEqual(events[2]["id"], 200)

    @mock.patch("trymerge.gh_fetch_json_list")
    def test_iter_issue_timeline_until_comment_not_found(
        self, mock_gh_fetch_json_list: Any, *args: Any
    ) -> None:
        """Test timeline iteration when target comment is not found"""
        # Mock empty timeline
        mock_gh_fetch_json_list.return_value = []

        events = list(iter_issue_timeline_until_comment("pytorch", "pytorch", 123, 999))
        self.assertEqual(len(events), 0)

    @mock.patch("trymerge.iter_issue_timeline_until_comment")
    def test_get_commit_sha_at_comment_commit_after_comment(
        self, mock_iter_timeline: Any, *args: Any
    ) -> None:
        """Test get_commit_sha_at_comment returns correct SHA after comment"""
        mock_iter_timeline.return_value = [
            {"event": "committed", "sha": "commit1"},
            {"event": "committed", "sha": "commit2"},
            {"event": "commented", "id": 100},
            {"event": "head_ref_force_pushed", "after": {"sha": "commit3"}},
        ]
        pr = GitHubPR("pytorch", "pytorch", 77700)
        sha = pr.get_commit_sha_at_comment(100)
        self.assertEqual(sha, "commit2")

    @mock.patch("trymerge.iter_issue_timeline_until_comment")
    def test_get_commit_sha_at_comment_force_push_before_comment(
        self, mock_iter_timeline: Any, *args: Any
    ) -> None:
        mock_iter_timeline.return_value = [
            {"event": "committed", "sha": "commit1"},
            {"event": "committed", "sha": "commit2"},
            {"event": "head_ref_force_pushed", "commit_id": "commit3"},
            {"event": "commented", "id": 100},
        ]
        pr = GitHubPR("pytorch", "pytorch", 77700)
        sha = pr.get_commit_sha_at_comment(100)
        self.assertEqual(sha, "commit3")

    @mock.patch("trymerge.iter_issue_timeline_until_comment")
    def test_get_commit_sha_at_comment_force_push_before_comment_legacy_mode(
        self, mock_iter_timeline: Any, *args: Any
    ) -> None:
        mock_iter_timeline.return_value = [
            {"event": "committed", "sha": "commit1"},
            {"event": "committed", "sha": "commit2"},
            {"event": "head_ref_force_pushed", "after": {"sha": "commit3"}},
            {"event": "commented", "id": 100},
        ]
        pr = GitHubPR("pytorch", "pytorch", 77700)
        sha = pr.get_commit_sha_at_comment(100)
        self.assertEqual(sha, "commit3")

    @mock.patch("trymerge.iter_issue_timeline_until_comment")
    def test_get_commit_sha_at_comment_multiple_comments(
        self, mock_iter_timeline: Any, *args: Any
    ) -> None:
        mock_iter_timeline.return_value = [
            {"event": "committed", "sha": "commit1"},
            {"event": "commented", "id": 100},
            {"event": "committed", "sha": "commit2"},
            {"event": "commented", "id": 200},
            {"event": "head_ref_force_pushed", "after": {"sha": "commit3"}},
            {"event": "commented", "id": 300},
        ]
        pr = GitHubPR("pytorch", "pytorch", 77700)
        sha = pr.get_commit_sha_at_comment(200)
        self.assertEqual(sha, "commit2")
        sha = pr.get_commit_sha_at_comment(300)
        self.assertEqual(sha, "commit3")

    @mock.patch("trymerge.iter_issue_timeline_until_comment")
    def test_get_commit_sha_at_comment_no_events(
        self, mock_iter_timeline: Any, *args: Any
    ) -> None:
        mock_iter_timeline.return_value = [
            {"event": "commented", "id": 100},
            {"event": "labeled", "label": {"name": "test"}},
        ]
        pr = GitHubPR("pytorch", "pytorch", 77700)
        sha = pr.get_commit_sha_at_comment(100)
        self.assertIsNone(sha)

    @mock.patch("trymerge.iter_issue_timeline_until_comment")
    def test_get_commit_sha_at_comment_exception(
        self, mock_iter_timeline: Any, *args: Any
    ) -> None:
        mock_iter_timeline.side_effect = Exception("API error")
        pr = GitHubPR("pytorch", "pytorch", 77700)
        sha = pr.get_commit_sha_at_comment(100)
        self.assertIsNone(sha)


class TestImportStatusCheck(TestCase):
    """CodeSync leaves `Import Status` queued on an unimported revision (#189303), so
    a pending one must not make the merge wait once CodeSync has cleared the commit."""

    BUILD_CHECK = "pull / linux-jammy-py3.10-gcc11 / build"

    @staticmethod
    def check(name: str, status: str | None, title: str | None = None) -> JobCheckState:
        return JobCheckState(name, "", status, None, None, title, None)

    def cleared_by_codesync(self) -> dict[str, JobCheckState]:
        return {
            INTERNAL_CHANGES_CHECKRUN_NAME: self.check(
                INTERNAL_CHANGES_CHECKRUN_NAME,
                "SUCCESS",
                HAS_NO_CONNECTED_DIFF_TITLE,
            )
        }

    def test_pending_import_status_is_not_pending(self) -> None:
        checks = self.cleared_by_codesync() | {
            IMPORT_STATUS_CHECKRUN_NAME: self.check(IMPORT_STATUS_CHECKRUN_NAME, None),
            self.BUILD_CHECK: self.check(self.BUILD_CHECK, "SUCCESS"),
        }
        pending, failed, _ = categorize_checks(checks, list(checks.keys()))
        self.assertEqual(pending, [])
        self.assertEqual(failed, [])

    def test_other_pending_checks_still_block(self) -> None:
        checks = self.cleared_by_codesync() | {
            IMPORT_STATUS_CHECKRUN_NAME: self.check(IMPORT_STATUS_CHECKRUN_NAME, None),
            self.BUILD_CHECK: self.check(self.BUILD_CHECK, None),
        }
        pending, failed, _ = categorize_checks(checks, list(checks.keys()))
        self.assertEqual([name for name, _, _ in pending], [self.BUILD_CHECK])
        self.assertEqual(failed, [])

    def test_pending_import_status_blocks_while_a_diff_is_connected(self) -> None:
        # The internal Diff has yet to land, so the import is still meaningful and
        # waiting on it is the pre-existing behaviour.
        checks = {
            INTERNAL_CHANGES_CHECKRUN_NAME: self.check(
                INTERNAL_CHANGES_CHECKRUN_NAME, "SUCCESS", "Diff is not landed yet"
            ),
            IMPORT_STATUS_CHECKRUN_NAME: self.check(IMPORT_STATUS_CHECKRUN_NAME, None),
        }
        pending, failed, _ = categorize_checks(checks, list(checks.keys()))
        self.assertEqual(
            [name for name, _, _ in pending], [IMPORT_STATUS_CHECKRUN_NAME]
        )
        self.assertEqual(failed, [])

    def test_pending_import_status_blocks_without_a_codesync_verdict(self) -> None:
        for internal_check in (
            None,
            self.check(INTERNAL_CHANGES_CHECKRUN_NAME, None, None),
            self.check(
                INTERNAL_CHANGES_CHECKRUN_NAME, None, HAS_NO_CONNECTED_DIFF_TITLE
            ),
            self.check(
                INTERNAL_CHANGES_CHECKRUN_NAME, "FAILURE", HAS_NO_CONNECTED_DIFF_TITLE
            ),
            # SKIPPED and NEUTRAL pass is_passing_status, but neither is a verdict
            self.check(
                INTERNAL_CHANGES_CHECKRUN_NAME, "SKIPPED", HAS_NO_CONNECTED_DIFF_TITLE
            ),
            self.check(
                INTERNAL_CHANGES_CHECKRUN_NAME, "NEUTRAL", HAS_NO_CONNECTED_DIFF_TITLE
            ),
        ):
            with self.subTest(internal_check=internal_check):
                checks = {
                    IMPORT_STATUS_CHECKRUN_NAME: self.check(
                        IMPORT_STATUS_CHECKRUN_NAME, None
                    )
                }
                if internal_check is not None:
                    checks[INTERNAL_CHANGES_CHECKRUN_NAME] = internal_check
                pending, _, _ = categorize_checks(checks, list(checks.keys()))
                self.assertIn(
                    IMPORT_STATUS_CHECKRUN_NAME, [name for name, _, _ in pending]
                )

    def test_failed_import_status_still_blocks(self) -> None:
        checks = self.cleared_by_codesync() | {
            IMPORT_STATUS_CHECKRUN_NAME: self.check(
                IMPORT_STATUS_CHECKRUN_NAME, "FAILURE"
            )
        }
        pending, failed, _ = categorize_checks(checks, list(checks.keys()))
        self.assertEqual(pending, [])
        self.assertEqual([name for name, _, _ in failed], [IMPORT_STATUS_CHECKRUN_NAME])

    def test_successful_import_status_is_not_a_failure(self) -> None:
        checks = self.cleared_by_codesync() | {
            IMPORT_STATUS_CHECKRUN_NAME: self.check(
                IMPORT_STATUS_CHECKRUN_NAME, "SUCCESS"
            )
        }
        pending, failed, _ = categorize_checks(checks, list(checks.keys()))
        self.assertEqual(pending, [])
        self.assertEqual(failed, [])

    def test_import_status_is_not_a_mandatory_check(self) -> None:
        # A pending `Import Status` is skipped by name, so listing it in
        # merge_rules.yaml would look like a gate while never acting as one.
        rules = read_merge_rules(DummyGitRepo(), "pytorch", "pytorch")
        self.assertGreater(len(rules), 0)
        for rule in rules:
            for mandatory_check in rule.mandatory_checks_name or []:
                # Mandatory names match check-runs by substring, not equality
                self.assertNotIn(mandatory_check, IMPORT_STATUS_CHECKRUN_NAME)


class TestDockerCiGates(TestCase):
    """Unit tests for the docker-image merge gates."""

    def test_is_docker_affecting_files(self) -> None:
        self.assertTrue(is_docker_affecting_files([".ci/docker/build.sh"]))
        self.assertTrue(
            is_docker_affecting_files(["README.md", ".ci/docker/ubuntu/Dockerfile"])
        )
        # Exact directory path also counts
        self.assertTrue(is_docker_affecting_files([".ci/docker"]))
        # Unrelated files, including a lookalike prefix, don't count
        self.assertFalse(is_docker_affecting_files(["torch/foo.py", "README.md"]))
        self.assertFalse(is_docker_affecting_files([".ci/docker-something/x"]))
        self.assertFalse(is_docker_affecting_files([]))

    def test_get_docker_build_checks(self) -> None:
        def check(name: str) -> JobCheckState:
            return JobCheckState(name, "", "SUCCESS", None, None, None, None)

        checks = {
            name: check(name)
            for name in (
                "docker-builds / docker-build (pytorch-linux-jammy)",
                "docker-builds",
                "linux-build / build",
                "docker-builds-nightly / x",
            )
        }
        self.assertEqual(
            set(get_docker_build_checks(checks)),
            {
                "docker-builds / docker-build (pytorch-linux-jammy)",
                "docker-builds",
            },
        )

    def test_get_topmost_docker_pr(self) -> None:
        lower_docker_pr = mock.MagicMock()
        lower_docker_pr.is_docker_affecting.return_value = True
        middle_pr = mock.MagicMock()
        middle_pr.is_docker_affecting.return_value = False
        top_docker_pr = mock.MagicMock()
        top_docker_pr.is_docker_affecting.return_value = True

        self.assertIs(
            get_topmost_docker_pr([lower_docker_pr, middle_pr]), lower_docker_pr
        )
        self.assertIs(
            get_topmost_docker_pr([lower_docker_pr, middle_pr, top_docker_pr]),
            top_docker_pr,
        )
        self.assertIsNone(get_topmost_docker_pr([middle_pr]))

    @mock.patch("trymerge.check_docker_builds_ready")
    @mock.patch("trymerge.get_ghstack_prs")
    @mock.patch("trymerge.find_matching_merge_rule")
    @mock.patch("trymerge.can_skip_internal_checks", return_value=False)
    def test_merge_into_gates_lower_ghstack_docker_pr(
        self,
        _mock_can_skip_internal_checks: mock.MagicMock,
        mock_find_matching_merge_rule: mock.MagicMock,
        mock_get_ghstack_prs: mock.MagicMock,
        mock_check_docker_builds_ready: mock.MagicMock,
    ) -> None:
        lower_pr = mock.MagicMock(spec=GitHubPR)
        lower_pr.pr_num = 1000
        lower_pr.is_closed.return_value = False
        lower_pr.is_docker_affecting.return_value = True
        top_pr = mock.MagicMock(spec=GitHubPR)
        top_pr.org = "pytorch"
        top_pr.project = "pytorch"
        top_pr.pr_num = 1001
        top_pr.is_ghstack_pr.return_value = True
        top_pr.is_closed.return_value = False
        top_pr.is_docker_affecting.return_value = False
        top_pr.is_dependabot_pr.return_value = False
        top_pr.merge_changes_locally.side_effect = RuntimeError("stop after gates")
        repo = mock.MagicMock(spec=GitRepo)
        ghstack_prs = [(lower_pr, "lower_rev"), (top_pr, "top_rev")]
        mock_get_ghstack_prs.return_value = ghstack_prs
        mock_find_matching_merge_rule.return_value = (None, [], [], {})

        with self.assertRaisesRegex(RuntimeError, "stop after gates"):
            GitHubPR.merge_into(top_pr, repo, comment_id=1)

        mock_get_ghstack_prs.assert_called_once_with(repo, top_pr, open_only=False)
        mock_check_docker_builds_ready.assert_called_once_with(lower_pr)
        top_pr.merge_changes_locally.assert_called_once_with(
            repo, False, 1, ghstack_prs=ghstack_prs, ignore_current_checks=None
        )


@mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
@mock.patch(
    "trymerge.get_drci_classifications", side_effect=mocked_drci_classifications
)
@mock.patch("trymerge.read_merge_rules", side_effect=mocked_read_merge_rules_greenlight)
class TestAuthorizedWithoutGreenlight(TestCase):
    def setUp(self) -> None:
        # The answer is memoized for the lifetime of a merge command's process; each
        # test is a different command.
        _AUTHORIZED_WITHOUT_GREENLIGHT.clear()

    def _pr_approved_by(self, *logins: str) -> GitHubPR:
        pr = GitHubPR("pytorch", "pytorch", 115495)
        pr._reviews = [(login, "APPROVED") for login in logins]
        return pr

    def test_greenlight_alone_is_required(self, *args: Any) -> None:
        pr = self._pr_approved_by(GREENLIGHT_LOGIN)
        self.assertFalse(is_authorized_without_greenlight(pr, DummyGitRepo()))

    def test_drive_by_approver_does_not_authorize_the_pr(self, *args: Any) -> None:
        pr = self._pr_approved_by(GREENLIGHT_LOGIN, "a-random-stranger")
        self.assertFalse(is_authorized_without_greenlight(pr, DummyGitRepo()))

    def test_a_real_rule_approver_makes_greenlight_unnecessary(
        self, *args: Any
    ) -> None:
        pr = self._pr_approved_by(GREENLIGHT_LOGIN, "malfet")
        self.assertTrue(is_authorized_without_greenlight(pr, DummyGitRepo()))

    def test_greenlight_is_stripped_case_insensitively(self, *args: Any) -> None:
        pr = self._pr_approved_by("PyTorchGreenlight")
        self.assertFalse(is_authorized_without_greenlight(pr, DummyGitRepo()))

    def test_dismissed_greenlight_approval_is_not_an_approval(self, *args: Any) -> None:
        pr = GitHubPR("pytorch", "pytorch", 115495)
        pr._reviews = [(GREENLIGHT_LOGIN, "DISMISSED"), ("malfet", "APPROVED")]
        self.assertEqual(pr.get_approved_by(), ["malfet"])
        self.assertTrue(is_authorized_without_greenlight(pr, DummyGitRepo()))

    def test_a_bot_suffixed_greenlight_approval_is_still_greenlights(
        self, *args: Any
    ) -> None:
        """REST spells the App `pytorchgreenlight[bot]`; dropping it must still work."""
        pr = self._pr_approved_by(f"{GREENLIGHT_LOGIN}[bot]")
        self.assertFalse(is_authorized_without_greenlight(pr, DummyGitRepo()))

    def test_the_reject_reason_is_logged(self, *args: Any) -> None:
        pr = self._pr_approved_by(GREENLIGHT_LOGIN, "a-random-stranger")
        with mock.patch("builtins.print") as mock_print:
            self.assertFalse(is_authorized_without_greenlight(pr, DummyGitRepo()))
        logged = " ".join(str(call.args[0]) for call in mock_print.call_args_list)
        self.assertIn("#115495", logged)
        self.assertIn("Core Maintainers", logged)

    def test_the_answer_is_computed_once_per_approver_set(self, *args: Any) -> None:
        pr = self._pr_approved_by(GREENLIGHT_LOGIN)
        repo = DummyGitRepo()
        with mock.patch(
            "trymerge.find_matching_merge_rule",
            side_effect=MergeRuleFailedError("no rule"),
        ) as mock_rule:
            self.assertFalse(is_authorized_without_greenlight(pr, repo))
            self.assertFalse(is_authorized_without_greenlight(pr, repo))
        mock_rule.assert_called_once()

    def test_an_approval_arriving_mid_merge_is_not_masked_by_the_cache(
        self, *args: Any
    ) -> None:
        repo = DummyGitRepo()
        self.assertFalse(
            is_authorized_without_greenlight(
                self._pr_approved_by(GREENLIGHT_LOGIN), repo
            )
        )
        self.assertTrue(
            is_authorized_without_greenlight(
                self._pr_approved_by(GREENLIGHT_LOGIN, "malfet"), repo
            )
        )

    def test_the_real_gates_arguments_are_used_verbatim(self, *args: Any) -> None:
        """A rule the real gate would skip must not answer this question instead."""
        pr = self._pr_approved_by(GREENLIGHT_LOGIN, "malfet")
        repo = DummyGitRepo()
        with mock.patch("trymerge.find_matching_merge_rule") as mock_rule:
            is_authorized_without_greenlight(
                pr,
                repo,
                skip_mandatory_checks=True,
                skip_internal_checks=True,
                ignore_current_checks={(115495, "some-check")},
            )
        self.assertEqual(
            mock_rule.call_args.kwargs,
            {
                "skip_mandatory_checks": True,
                "skip_internal_checks": True,
                "ignore_current_checks": {(115495, "some-check")},
                "approved_by_override": {"malfet"},
            },
        )

    def test_waived_checks_do_not_stand_in_for_a_missing_approval(
        self, *args: Any
    ) -> None:
        """`merge -i` relaxes check status; it never relaxes who has to approve."""
        pr = self._pr_approved_by(GREENLIGHT_LOGIN)
        self.assertFalse(
            is_authorized_without_greenlight(
                pr, DummyGitRepo(), ignore_current_checks={(pr.pr_num, "Lint")}
            )
        )

    def test_pending_mandatory_checks_keep_the_guard_on(self, *args: Any) -> None:
        """The real gate skips such a rule and falls through to the Greenlight rule."""
        pr = self._pr_approved_by(GREENLIGHT_LOGIN, "malfet")
        with mock.patch(
            "trymerge.find_matching_merge_rule",
            side_effect=MandatoryChecksMissingError("Lint is pending"),
        ):
            self.assertFalse(is_authorized_without_greenlight(pr, DummyGitRepo()))

    def test_an_unusable_rule_set_keeps_the_guard_on(self, *args: Any) -> None:
        pr = self._pr_approved_by(GREENLIGHT_LOGIN, "malfet")
        with mock.patch(
            "trymerge.find_matching_merge_rule",
            side_effect=RuntimeError("This PR has internal changes"),
        ):
            self.assertFalse(is_authorized_without_greenlight(pr, DummyGitRepo()))


@mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
class TestMergeAuthorizedLogins(TestCase):
    """greenlight's merge_authz.resolve_authorized_logins, mirrored for the guard."""

    @mock.patch(
        "trymerge.read_merge_rules", side_effect=mocked_read_merge_rules_greenlight
    )
    def test_every_rules_approvers_are_unioned_and_lowercased(self, *args: Any) -> None:
        self.assertEqual(
            merge_authorized_logins(DummyGitRepo(), "pytorch", "pytorch"),
            frozenset({GREENLIGHT_LOGIN, "malfet"}),
        )

    @mock.patch("trymerge.read_merge_rules")
    def test_team_refs_are_expanded_to_members(
        self, mock_rules: Any, *args: Any
    ) -> None:
        mock_rules.return_value = [
            MergeRule(
                name="Some Team",
                patterns=["*"],
                approved_by=["pytorch/some-team", "Malfet"],
                mandatory_checks_name=None,
            )
        ]
        with mock.patch(
            "trymerge.gh_get_team_members", return_value=["Alice", "bob"]
        ) as mock_members:
            self.assertEqual(
                merge_authorized_logins(DummyGitRepo(), "pytorch", "pytorch"),
                frozenset({"alice", "bob", "malfet"}),
            )
        mock_members.assert_called_once_with("pytorch", "some-team")

    @mock.patch("trymerge.read_merge_rules", side_effect=mocked_read_merge_rules_raise)
    def test_an_unreadable_rules_file_yields_an_empty_set(self, *args: Any) -> None:
        self.assertEqual(
            merge_authorized_logins(DummyGitRepo(), "pytorch", "pytorch"), frozenset()
        )

    @mock.patch("trymerge.read_merge_rules")
    def test_an_unexpandable_team_ref_yields_an_empty_set(
        self, mock_rules: Any, *args: Any
    ) -> None:
        """Expanding a team ref is a live request; a blip must not end the merge."""
        mock_rules.return_value = [
            MergeRule(
                name="Some Team",
                patterns=["*"],
                approved_by=["pytorch/some-team"],
                mandatory_checks_name=None,
            )
        ]
        with mock.patch(
            "trymerge.gh_get_team_members", side_effect=RuntimeError("testing")
        ):
            self.assertEqual(
                merge_authorized_logins(DummyGitRepo(), "pytorch", "pytorch"),
                frozenset(),
            )

    @mock.patch(
        "trymerge.get_drci_classifications", side_effect=mocked_drci_classifications
    )
    @mock.patch("trymerge.read_merge_rules")
    def test_a_malformed_team_ref_in_an_unmatched_rule_yields_an_empty_set(
        self, mock_rules: Any, *args: Any
    ) -> None:
        """Refusing a mis-typed team ref is deliberate: it must never resolve to nobody.

        find_matching_merge_rule drops a rule whose patterns miss the PR before it
        expands that rule's approvers, so the ref below never reaches it. This set
        spans every rule in the file regardless of patterns, and the empty set it
        produces instead is what the guard reads as "no human authorized this".
        """
        mock_rules.return_value = [
            MergeRule(
                name="Malformed Team",
                patterns=["some/path/that/no/pr/touches/**"],
                approved_by=[MALFORMED_TEAM_REF],
                mandatory_checks_name=None,
            )
        ]
        pr = GitHubPR("pytorch", "pytorch", 115495)

        with mock.patch(
            "trymerge.gh_get_team_members", return_value=[]
        ) as mock_members:
            self.assertRaisesRegex(
                MergeRuleFailedError,
                "No rule found to match PR",
                lambda: find_matching_merge_rule(pr, DummyGitRepo()),
            )
            self.assertEqual(
                merge_authorized_logins(DummyGitRepo(), "pytorch", "pytorch"),
                frozenset(),
            )
        mock_members.assert_not_called()


@mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
@mock.patch(
    "trymerge.get_drci_classifications", side_effect=mocked_drci_classifications
)
class TestApprovedByOverride(TestCase):
    @mock.patch(
        "trymerge.read_merge_rules", side_effect=mocked_read_merge_rules_approvers
    )
    def test_override_replaces_the_prs_real_approvers(self, *args: Any) -> None:
        pr = GitHubPR("pytorch", "pytorch", 115495)
        repo = DummyGitRepo()

        self.assertIsNotNone(find_matching_merge_rule(pr, repo))
        self.assertIsNotNone(
            find_matching_merge_rule(pr, repo, approved_by_override={"malfet"})
        )
        self.assertRaisesRegex(
            MergeRuleFailedError,
            "has not been reviewed yet",
            lambda: find_matching_merge_rule(pr, repo, approved_by_override=set()),
        )
        self.assertRaisesRegex(
            MergeRuleFailedError,
            "Core Maintainers",
            lambda: find_matching_merge_rule(
                pr, repo, approved_by_override={"a-random-stranger"}
            ),
        )


@mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
class TestReviewAccessors(TestCase):
    def test_get_changes_requested_by(self, *args: Any) -> None:
        pr = GitHubPR("pytorch", "pytorch", 115495)
        pr._reviews = [
            ("malfet", "APPROVED"),
            ("some-reviewer", "CHANGES_REQUESTED"),
            ("a-commenter", "COMMENTED"),
        ]
        self.assertEqual(pr.get_changes_requested_by(), ["some-reviewer"])
        self.assertEqual(pr.get_approved_by(), ["malfet"])

    @mock.patch("trymerge.gh_fetch_json_dict")
    def test_get_updated_at(self, mock_fetch: Any, *args: Any) -> None:
        pr = GitHubPR("pytorch", "pytorch", 115495)
        mock_fetch.return_value = {"updated_at": "2026-08-25T11:00:00Z"}
        self.assertEqual(pr.get_updated_at(), "2026-08-25T11:00:00Z")
        self.assertIn(
            "/repos/pytorch/pytorch/pulls/115495", mock_fetch.call_args.args[0]
        )

    @mock.patch("trymerge.gh_fetch_json_dict")
    def test_get_updated_at_returns_none_when_unavailable(
        self, mock_fetch: Any, *args: Any
    ) -> None:
        pr = GitHubPR("pytorch", "pytorch", 115495)
        mock_fetch.return_value = {}
        self.assertIsNone(pr.get_updated_at())
        mock_fetch.side_effect = HTTPError(
            "https://api.github.com",
            500,
            "boom",
            {},  # type: ignore[arg-type]
            None,
        )
        self.assertIsNone(pr.get_updated_at())

    @mock.patch("trymerge.gh_fetch_json_list")
    def test_get_bot_reviewers(self, mock_fetch: Any, *args: Any) -> None:
        """REST names an App `slug[bot]`, and that is the login callers get back."""
        pr = GitHubPR("pytorch", "pytorch", 115495)
        mock_fetch.return_value = [
            {"user": {"login": "some-app[bot]", "type": "Bot"}},
            {"user": {"login": "malfet", "type": "User"}},
            {"user": None},
            {},
        ]
        self.assertEqual(pr.get_bot_reviewers(), frozenset({"some-app[bot]"}))
        self.assertIn(
            "/repos/pytorch/pytorch/pulls/115495/reviews", mock_fetch.call_args.args[0]
        )

    @mock.patch("trymerge.gh_fetch_json_list")
    def test_get_bot_reviewers_walks_every_page(
        self, mock_fetch: Any, *args: Any
    ) -> None:
        """GitHub returns reviews oldest first, so the last page holds the recent ones."""
        full_page = [{"user": {"login": "malfet", "type": "User"}}] * REVIEWS_PER_PAGE
        mock_fetch.side_effect = [
            full_page,
            [{"user": {"login": "late-app[bot]", "type": "Bot"}}],
        ]
        pr = GitHubPR("pytorch", "pytorch", 115495)
        self.assertEqual(pr.get_bot_reviewers(), frozenset({"late-app[bot]"}))
        self.assertEqual(
            [call.kwargs["params"]["page"] for call in mock_fetch.call_args_list],
            [1, 2],
        )

    @mock.patch("trymerge.gh_fetch_json_list")
    def test_get_bot_reviewers_stops_at_the_page_limit(
        self, mock_fetch: Any, *args: Any
    ) -> None:
        mock_fetch.return_value = [
            {"user": {"login": "some-app[bot]", "type": "Bot"}}
        ] * REVIEWS_PER_PAGE
        pr = GitHubPR("pytorch", "pytorch", 115495)
        self.assertEqual(pr.get_bot_reviewers(), frozenset({"some-app[bot]"}))
        self.assertEqual(mock_fetch.call_count, REVIEW_PAGE_LIMIT)

    @mock.patch("trymerge.gh_fetch_json_list")
    def test_get_bot_reviewers_is_none_when_unavailable(
        self, mock_fetch: Any, *args: Any
    ) -> None:
        """None, not an empty set: an outage must not read as "this PR has no Apps"."""
        pr = GitHubPR("pytorch", "pytorch", 115495)
        mock_fetch.side_effect = HTTPError(
            "https://api.github.com",
            500,
            "boom",
            {},  # type: ignore[arg-type]
            None,
        )
        self.assertIsNone(pr.get_bot_reviewers())

    @mock.patch("trymerge.gh_fetch_json_list")
    def test_get_bot_reviewers_is_none_for_a_malformed_payload(
        self, mock_fetch: Any, *args: Any
    ) -> None:
        pr = GitHubPR("pytorch", "pytorch", 115495)
        mock_fetch.return_value = {"message": "Not Found"}
        self.assertIsNone(pr.get_bot_reviewers())

    @mock.patch("trymerge.gh_fetch_json_dict")
    def test_get_updated_at_is_none_for_a_malformed_payload(
        self, mock_fetch: Any, *args: Any
    ) -> None:
        pr = GitHubPR("pytorch", "pytorch", 115495)
        mock_fetch.return_value = ["not", "a", "dict"]
        self.assertIsNone(pr.get_updated_at())


class TestGreenlightGuardCallSite(TestCase):
    def setUp(self) -> None:
        for patcher in (
            mock.patch("trymerge.check_docker_builds_ready"),
            mock.patch("trymerge.can_skip_internal_checks", return_value=False),
            mock.patch(
                "trymerge.find_matching_merge_rule", return_value=(None, [], [], {})
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _pr(self, pr_num: int, ghstack: bool = False) -> Any:
        pr = mock.MagicMock(spec=GitHubPR)
        pr.org = "pytorch"
        pr.project = "pytorch"
        pr.pr_num = pr_num
        pr.is_ghstack_pr.return_value = ghstack
        pr.is_closed.return_value = False
        pr.is_docker_affecting.return_value = False
        pr.is_dependabot_pr.return_value = False
        pr.get_approved_by.return_value = [GREENLIGHT_LOGIN]
        pr.get_changes_requested_by.return_value = []
        pr.get_labels.return_value = []
        pr.merge_changes_locally.side_effect = RuntimeError("stop after gates")
        return pr

    @mock.patch("trymerge.evaluate_greenlight_guard")
    def test_non_ghstack_pr_is_checked_at_its_head(self, mock_evaluate: Any) -> None:
        pr = self._pr(1)
        pr.last_commit_sha.return_value = "head-sha"
        mock_evaluate.return_value = GuardResult(GuardVerdict.ALLOW)

        with self.assertRaisesRegex(RuntimeError, "stop after gates"):
            GitHubPR.merge_into(pr, mock.MagicMock(spec=GitRepo), comment_id=1)

        repo_full_name, prs = mock_evaluate.call_args.args
        self.assertEqual(repo_full_name, "pytorch/pytorch")
        self.assertEqual([(p.pr_num, p.head_sha) for p in prs], [(1, "head-sha")])

    @mock.patch("trymerge.get_ghstack_prs")
    @mock.patch("trymerge.evaluate_greenlight_guard")
    def test_ghstack_stack_is_checked_at_the_github_heads(
        self, mock_evaluate: Any, mock_get_ghstack_prs: Any
    ) -> None:
        """greenlight records `gh/USER/N/head`, never the `/orig` rev git cherry-picks."""
        lower_pr = self._pr(1)
        lower_pr.last_commit_sha.return_value = "lower-github-head"
        closed_pr = self._pr(2)
        closed_pr.is_closed.return_value = True
        top_pr = self._pr(3, ghstack=True)
        top_pr.last_commit_sha.return_value = "top-github-head"
        mock_get_ghstack_prs.return_value = [
            (lower_pr, "lower_rev"),
            (closed_pr, "closed_rev"),
            (top_pr, "top_rev"),
        ]
        mock_evaluate.return_value = GuardResult(GuardVerdict.ALLOW)

        with self.assertRaisesRegex(RuntimeError, "stop after gates"):
            GitHubPR.merge_into(top_pr, mock.MagicMock(spec=GitRepo), comment_id=1)

        _, prs = mock_evaluate.call_args.args
        self.assertEqual(
            [(p.pr_num, p.head_sha) for p in prs],
            [(1, "lower-github-head"), (3, "top-github-head")],
        )

    @mock.patch("trymerge.evaluate_greenlight_guard")
    def test_wait_raises_the_retryable_error(self, mock_evaluate: Any) -> None:
        pr = self._pr(1)
        pr.last_commit_sha.return_value = "head-sha"
        mock_evaluate.return_value = GuardResult(GuardVerdict.WAIT, "still reviewing")

        with self.assertRaisesRegex(MandatoryChecksMissingError, "still reviewing"):
            GitHubPR.merge_into(pr, mock.MagicMock(spec=GitRepo), comment_id=1)
        pr.merge_changes_locally.assert_not_called()

    @mock.patch("trymerge.evaluate_greenlight_guard")
    def test_deny_raises_a_non_retryable_error(self, mock_evaluate: Any) -> None:
        pr = self._pr(1)
        pr.last_commit_sha.return_value = "head-sha"
        mock_evaluate.return_value = GuardResult(GuardVerdict.DENY, "refusing")

        with self.assertRaisesRegex(MergeRuleFailedError, "refusing") as cm:
            GitHubPR.merge_into(pr, mock.MagicMock(spec=GitRepo), comment_id=1)
        self.assertNotIsInstance(cm.exception, MandatoryChecksMissingError)
        pr.merge_changes_locally.assert_not_called()

    @mock.patch("trymerge.gh_post_pr_comment")
    @mock.patch("trymerge.evaluate_greenlight_guard")
    def test_a_returned_comment_is_posted_to_the_pr_being_merged(
        self, mock_evaluate: Any, mock_post: Any
    ) -> None:
        pr = self._pr(1)
        pr.last_commit_sha.return_value = "head-sha"
        mock_evaluate.return_value = GuardResult(
            GuardVerdict.WAIT, "still reviewing", "hold tight"
        )

        with self.assertRaises(MandatoryChecksMissingError):
            GitHubPR.merge_into(
                pr, mock.MagicMock(spec=GitRepo), comment_id=1, dry_run=True
            )
        mock_post.assert_called_once_with("pytorch", "pytorch", 1, "hold tight", True)

    @mock.patch("trymerge.gh_post_pr_comment")
    @mock.patch("trymerge.evaluate_greenlight_guard")
    def test_no_comment_is_posted_when_the_guard_returns_none(
        self, mock_evaluate: Any, mock_post: Any
    ) -> None:
        pr = self._pr(1)
        pr.last_commit_sha.return_value = "head-sha"
        mock_evaluate.return_value = GuardResult(GuardVerdict.WAIT, "still reviewing")

        with self.assertRaises(MandatoryChecksMissingError):
            GitHubPR.merge_into(pr, mock.MagicMock(spec=GitRepo), comment_id=1)
        mock_post.assert_not_called()

    @mock.patch("trymerge.gh_post_pr_comment")
    @mock.patch("trymerge.evaluate_greenlight_guard")
    def test_a_failed_comment_does_not_end_the_merge(
        self, mock_evaluate: Any, mock_post: Any
    ) -> None:
        pr = self._pr(1)
        pr.last_commit_sha.return_value = "head-sha"
        mock_post.side_effect = RuntimeError("github is down")
        mock_evaluate.return_value = GuardResult(
            GuardVerdict.WAIT, "still reviewing", "hold tight"
        )

        with self.assertRaisesRegex(MandatoryChecksMissingError, "still reviewing"):
            GitHubPR.merge_into(pr, mock.MagicMock(spec=GitRepo), comment_id=1)

    @mock.patch("greenlight_ledger.gh_fetch_url")
    def test_the_guard_is_skipped_when_greenlight_did_not_approve(
        self, mock_fetch_url: Any
    ) -> None:
        pr = self._pr(1)
        pr.last_commit_sha.return_value = "head-sha"
        pr.get_approved_by.return_value = ["malfet"]

        with self.assertRaisesRegex(RuntimeError, "stop after gates"):
            GitHubPR.merge_into(pr, mock.MagicMock(spec=GitRepo), comment_id=1)
        mock_fetch_url.assert_not_called()


@mock.patch("trymerge.gh_add_labels")
@mock.patch("trymerge.check_for_sev")
@mock.patch("trymerge.post_starting_merge_comment")
@mock.patch("trymerge.ensure_mergeable_labels")
@mock.patch("trymerge.find_matching_merge_rule")
@mock.patch("trymerge.get_classifications", return_value={})
class TestGreenlightWaitPlumbing(TestCase):
    def _pr(self) -> Any:
        pr = mock.MagicMock(spec=GitHubPR)
        pr.org = "pytorch"
        pr.project = "pytorch"
        pr.pr_num = 1
        pr.last_commit_sha.return_value = "head-sha"
        pr.get_labels.return_value = []
        pr.is_ghstack_pr.return_value = False
        pr.get_checkrun_conclusions.return_value = {}
        return pr

    @mock.patch("trymerge.GitHubPR")
    def test_normal_merge_gets_a_wait_window(
        self, mock_pr_cls: Any, *args: Any
    ) -> None:
        pr = self._pr()
        mock_pr_cls.return_value = pr
        merge(pr, mock.MagicMock(spec=GitRepo), comment_id=1, dry_run=True)
        self.assertIsInstance(
            pr.merge_into.call_args.kwargs["greenlight_wait"], GreenlightWaitWindow
        )

    @mock.patch("trymerge.time.sleep")
    @mock.patch("trymerge.GitHubPR")
    def test_every_retry_shares_one_wait_window(
        self, mock_pr_cls: Any, _mock_sleep: Any, *args: Any
    ) -> None:
        pr = self._pr()
        mock_pr_cls.return_value = pr
        pr.merge_into.side_effect = [
            MandatoryChecksMissingError("waiting on greenlight"),
            None,
        ]
        merge(pr, mock.MagicMock(spec=GitRepo), comment_id=1, dry_run=True)
        windows = [
            call.kwargs["greenlight_wait"] for call in pr.merge_into.call_args_list
        ]
        self.assertEqual(len(windows), 2)
        self.assertIs(windows[0], windows[1])

    def test_force_merge_gets_no_window_so_it_cannot_wait(self, *args: Any) -> None:
        pr = self._pr()
        merge(
            pr,
            mock.MagicMock(spec=GitRepo),
            comment_id=1,
            dry_run=True,
            skip_mandatory_checks=True,
        )
        self.assertIsNone(pr.merge_into.call_args.kwargs["greenlight_wait"])


class TestGreenlightGuardWiring(TestCase):
    @mock.patch("trymerge.evaluate_greenlight_guard")
    def test_pr_facts_are_forwarded_to_the_guard(self, mock_evaluate: Any) -> None:
        pr = mock.MagicMock(spec=GitHubPR)
        pr.org = "pytorch"
        pr.project = "pytorch"
        pr.pr_num = 7
        pr.last_commit_sha.return_value = "head-sha"
        pr.get_approved_by.return_value = [GREENLIGHT_LOGIN]
        pr.get_changes_requested_by.return_value = ["some-reviewer"]
        pr.get_labels.return_value = ["Stale"]
        pr.get_updated_at.return_value = "2026-08-25T11:00:00Z"
        pr.get_bot_reviewers.return_value = frozenset({"some-app[bot]"})
        mock_evaluate.return_value = GuardResult(GuardVerdict.ALLOW)

        with mock.patch(
            "trymerge.merge_authorized_logins", return_value=frozenset({"malfet"})
        ) as mock_logins:
            check_greenlight_reviewed_head_sha(pr, None, [pr], wait_window=None)

        repo_full_name, prs = mock_evaluate.call_args.args
        self.assertEqual(repo_full_name, "pytorch/pytorch")
        self.assertEqual(len(prs), 1)
        forwarded = prs[0]
        self.assertEqual(forwarded.pr_num, 7)
        self.assertEqual(forwarded.head_sha, "head-sha")
        self.assertEqual(forwarded.approved_by, [GREENLIGHT_LOGIN])
        self.assertEqual(forwarded.changes_requested_by, ["some-reviewer"])
        self.assertEqual(forwarded.labels, ["Stale"])
        self.assertEqual(forwarded.get_updated_at(), "2026-08-25T11:00:00Z")
        self.assertEqual(forwarded.get_bot_reviewers(), frozenset({"some-app[bot]"}))
        self.assertEqual(forwarded.get_merge_authorized_logins(), frozenset({"malfet"}))
        mock_logins.assert_called_once_with(None, "pytorch", "pytorch")
        self.assertIsNone(mock_evaluate.call_args.kwargs["wait_window"])

    @mock.patch("trymerge.evaluate_greenlight_guard")
    def test_the_real_gates_arguments_reach_the_authorization_question(
        self, mock_evaluate: Any
    ) -> None:
        pr = mock.MagicMock(spec=GitHubPR)
        pr.org = "pytorch"
        pr.project = "pytorch"
        pr.pr_num = 7
        pr.last_commit_sha.return_value = "head-sha"
        pr.get_approved_by.return_value = [GREENLIGHT_LOGIN]
        pr.get_changes_requested_by.return_value = []
        pr.get_labels.return_value = []
        pr.get_bot_reviewers.return_value = frozenset()
        mock_evaluate.return_value = GuardResult(GuardVerdict.ALLOW)

        with mock.patch("trymerge.is_authorized_without_greenlight") as mock_authz:
            check_greenlight_reviewed_head_sha(
                pr,
                None,
                [pr],
                wait_window=None,
                skip_mandatory_checks=True,
                skip_internal_checks=True,
                ignore_current_checks={(7, "some-check")},
            )
            _, prs = mock_evaluate.call_args.args
            prs[0].is_authorized_without_greenlight()
        self.assertEqual(
            mock_authz.call_args.kwargs,
            {
                "skip_mandatory_checks": True,
                "skip_internal_checks": True,
                "ignore_current_checks": {(7, "some-check")},
            },
        )

    @mock.patch("trymerge.check_greenlight_reviewed_head_sha")
    @mock.patch("trymerge.check_docker_builds_ready")
    @mock.patch("trymerge.can_skip_internal_checks", return_value=True)
    @mock.patch("trymerge.find_matching_merge_rule", return_value=(None, [], [], {}))
    def test_merge_into_hands_the_guard_the_gate_it_just_ran(
        self, _rule: Any, _skip: Any, _docker: Any, mock_check: Any
    ) -> None:
        pr = mock.MagicMock(spec=GitHubPR)
        pr.org = "pytorch"
        pr.project = "pytorch"
        pr.pr_num = 1
        pr.is_ghstack_pr.return_value = False
        pr.is_dependabot_pr.return_value = False
        pr.is_docker_affecting.return_value = False
        pr.merge_changes_locally.side_effect = RuntimeError("stop after gates")

        with self.assertRaisesRegex(RuntimeError, "stop after gates"):
            GitHubPR.merge_into(
                pr,
                mock.MagicMock(spec=GitRepo),
                comment_id=1,
                skip_mandatory_checks=True,
                ignore_current_checks={(1, "some-check")},
            )

        self.assertEqual(mock_check.call_args.kwargs["skip_mandatory_checks"], True)
        self.assertEqual(mock_check.call_args.kwargs["skip_internal_checks"], True)
        self.assertEqual(
            mock_check.call_args.kwargs["ignore_current_checks"], {(1, "some-check")}
        )
        self.assertEqual(
            pr.merge_changes_locally.call_args.kwargs["ignore_current_checks"],
            {(1, "some-check")},
        )


@mock.patch("trymerge.get_drci_classifications", return_value={})
class TestIgnoreCurrentScope(TestCase):
    """The waiver set names (PR, check) pairs, so a gate waives only what was red on
    the PR it is judging, and only the checks that were red when the command ran."""

    def _categorize(
        self,
        pr_num: int,
        status: str | None,
        names: list[str],
        waived: set[tuple[int, str]],
    ) -> tuple[list[str], list[str]]:
        checks = {
            name: JobCheckState(name, "", status, None, None, None, None)
            for name in names
        }
        classified = get_classifications(pr_num, "pytorch", checks, waived)
        pending, failed, _ = categorize_checks(classified, names)
        return [x[0] for x in pending], [x[0] for x in failed]

    def test_a_waiver_does_not_carry_to_the_same_job_on_another_pr(
        self, *args: Any
    ) -> None:
        red = "pull / build"
        _, failed = self._categorize(1001, "FAILURE", [red], {(1000, red)})
        self.assertEqual(failed, [red])

    def test_a_failure_that_appeared_after_the_snapshot_still_blocks(
        self, *args: Any
    ) -> None:
        _, failed = self._categorize(
            1001, "FAILURE", ["snapshotted", "appeared-later"], {(1001, "snapshotted")}
        )
        self.assertEqual(failed, ["appeared-later"])

    def test_a_pending_check_is_never_waived(self, *args: Any) -> None:
        red = "pull / build"
        pending, failed = self._categorize(1001, None, [red], {(1001, red)})
        self.assertEqual((pending, failed), ([red], []))


@mock.patch("trymerge.gh_add_labels")
@mock.patch("trymerge.check_for_sev")
@mock.patch("trymerge.post_starting_merge_comment")
@mock.patch("trymerge.ensure_mergeable_labels")
@mock.patch("trymerge.find_matching_merge_rule")
@mock.patch("trymerge.get_classifications", return_value={})
@mock.patch("trymerge.GitHubPR")
@mock.patch("trymerge.get_ghstack_prs")
class TestIgnoreCurrentSnapshot(TestCase):
    def _pr(self, pr_num: int, red: str | None, ghstack: bool = False) -> Any:
        pr = mock.MagicMock(spec=GitHubPR)
        pr.org = "pytorch"
        pr.project = "pytorch"
        pr.pr_num = pr_num
        pr.last_commit_sha.return_value = f"sha-{pr_num}"
        pr.get_labels.return_value = []
        pr.is_ghstack_pr.return_value = ghstack
        pr.get_checkrun_conclusions.return_value = (
            {red: JobCheckState(red, "", "FAILURE", None, None, None, None)}
            if red
            else {}
        )
        return pr

    def _waivers_of_stack(
        self, ghstack_prs: Any, pr_cls: Any, lower_red: str | None, top_red: str | None
    ) -> Any:
        lower = self._pr(1000, lower_red)
        top = self._pr(1001, top_red, ghstack=True)
        ghstack_prs.return_value = [(lower, "lower"), (top, "top")]
        pr_cls.return_value = top
        merge(
            top,
            mock.MagicMock(spec=GitRepo),
            comment_id=1,
            dry_run=True,
            ignore_current=True,
        )
        return top.merge_into.call_args.kwargs["ignore_current_checks"]

    def _waived_names(self, post_comment: Any) -> list[str]:
        """The names the merge comment renders for what it waived."""
        info = post_comment.call_args.kwargs["ignore_current_checks_info"]
        return [name for name, _, _ in info]

    def test_every_stacked_pr_contributes_its_own_red_checks(
        self,
        mock_get_ghstack_prs: Any,
        mock_pr_cls: Any,
        mock_get_classifications: Any,
        mock_find_matching_merge_rule: Any,
        mock_ensure_mergeable_labels: Any,
        mock_post_comment: Any,
        *args: Any,
    ) -> None:
        """One job name, red on both PRs: the waivers must not collapse into one,
        and the comment must say which PR each came from."""
        self.assertEqual(
            self._waivers_of_stack(mock_get_ghstack_prs, mock_pr_cls, "red", "red"),
            {(1000, "red"), (1001, "red")},
        )
        self.assertEqual(
            self._waived_names(mock_post_comment), ["red (#1000)", "red (#1001)"]
        )

    def test_a_lone_pr_tags_nothing(
        self,
        mock_get_ghstack_prs: Any,
        mock_pr_cls: Any,
        mock_get_classifications: Any,
        mock_find_matching_merge_rule: Any,
        mock_ensure_mergeable_labels: Any,
        mock_post_comment: Any,
        *args: Any,
    ) -> None:
        pr = self._pr(1000, "red")
        mock_pr_cls.return_value = pr
        merge(
            pr,
            mock.MagicMock(spec=GitRepo),
            comment_id=1,
            dry_run=True,
            ignore_current=True,
        )
        self.assertEqual(self._waived_names(mock_post_comment), ["red"])
        mock_get_ghstack_prs.assert_not_called()

    def test_a_green_stack_waives_nothing(
        self, mock_get_ghstack_prs: Any, mock_pr_cls: Any, *args: Any
    ) -> None:
        """An empty set is falsy, which is what the s3 record's flag reads."""
        self.assertEqual(
            self._waivers_of_stack(mock_get_ghstack_prs, mock_pr_cls, None, None), set()
        )


@mock.patch("trymerge.get_drci_classifications", return_value={})
@mock.patch("trymerge.read_merge_rules", side_effect=mocked_read_merge_rules_greenlight)
class TestIgnoreCurrentAcrossAStack(TestCase):
    def setUp(self) -> None:
        _AUTHORIZED_WITHOUT_GREENLIGHT.clear()

    def _stacked_pr(self, pr_num: int, red: str) -> Any:
        pr = mock.MagicMock(spec=GitHubPR)
        pr.org = "pytorch"
        pr.project = "pytorch"
        pr.pr_num = pr_num
        pr.last_commit_sha.return_value = f"sha-{pr_num}"
        pr.get_approved_by.return_value = [GREENLIGHT_LOGIN, "malfet"]
        pr.get_changed_files.return_value = ["torch/foo.py"]
        pr.has_internal_changes.return_value = False
        pr.get_checkrun_conclusions.return_value = {
            "Lint": JobCheckState("Lint", "", "SUCCESS", None, None, None, None),
            red: JobCheckState(red, "", "FAILURE", None, None, None, None),
        }
        return pr

    @mock.patch("trymerge.evaluate_greenlight_guard")
    def test_each_stacked_pr_is_asked_about_its_own_waived_checks(
        self, mock_evaluate: Any, *args: Any
    ) -> None:
        """Same job name red on both PRs, snapshotted on only one: the guard's
        hypothetical must waive it on that PR and still see it red on the other."""
        red = "pull / linux-build"
        waived, blocked = self._stacked_pr(1000, red), self._stacked_pr(1001, red)
        mock_evaluate.return_value = GuardResult(GuardVerdict.ALLOW)

        check_greenlight_reviewed_head_sha(
            waived,
            None,
            [waived, blocked],
            wait_window=None,
            ignore_current_checks={(waived.pr_num, red)},
        )

        _, prs = mock_evaluate.call_args.args
        self.assertTrue(prs[0].is_authorized_without_greenlight())
        self.assertFalse(prs[1].is_authorized_without_greenlight())


class TestAdvisorNotRelated(TestCase):
    """The AI CI Advisor's `not_related` verdict as a non-blocking classification.

    Dr.CI owns the verdict/confidence/freshness predicates, so these cover what
    trymerge itself decides: how a check is matched to the bucket, and how many
    gates one merge may skip on that basis.
    """

    @staticmethod
    def _check(name: str, job_id: int | None) -> JobCheckState:
        return JobCheckState(
            name, "https://example.com", "FAILURE", None, job_id, "", ""
        )

    def test_matches_on_job_id(self) -> None:
        drci = {"AI_NOT_RELATED": [{"id": 42, "name": "some job"}]}
        self.assertTrue(is_ai_not_related(self._check("some job", 42), drci))

    def test_does_not_fall_back_to_the_name(self) -> None:
        """A name match with a different job id is a different execution."""
        drci = {"AI_NOT_RELATED": [{"id": 42, "name": "some job"}]}
        self.assertFalse(is_ai_not_related(self._check("some job", 43), drci))

    def test_a_check_with_no_job_id_never_matches(self) -> None:
        drci = {"AI_NOT_RELATED": [{"id": 42, "name": "Lint"}]}
        self.assertFalse(is_ai_not_related(self._check("Lint", None), drci))

    def test_absent_category_is_the_off_state(self) -> None:
        self.assertFalse(is_ai_not_related(self._check("some job", 42), {}))
        self.assertFalse(is_ai_not_related(self._check("some job", 42), None))

    def _categorize(self, count: int):  # type: ignore[no-untyped-def]
        checks = {
            f"job {i}": JobCheckState(
                f"job {i}", "", "FAILURE", "AI_NOT_RELATED", i, "", ""
            )
            for i in range(count)
        }
        return categorize_checks(checks, list(checks.keys()))

    def test_cleared_checks_do_not_block(self) -> None:
        _, failed, ignorable = self._categorize(IGNORABLE_FAILED_CHECKS_THESHOLD)
        self.assertEqual(failed, [])
        self.assertEqual(
            len(ignorable["AI_NOT_RELATED"]), IGNORABLE_FAILED_CHECKS_THESHOLD
        )

    def test_too_many_at_once_reads_as_an_outage_and_blocks(self) -> None:
        over = IGNORABLE_FAILED_CHECKS_THESHOLD + 1
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _, failed, ignorable = self._categorize(over)
        self.assertEqual(len(failed), over)
        # Still reported under its own category so the merge record keeps them.
        self.assertEqual(len(ignorable["AI_NOT_RELATED"]), over)
        self.assertIn("usually means an outage", str(w[-1].message))

    def test_the_cap_is_independent_of_the_flaky_budget(self) -> None:
        """ok_failed_checks_threshold tunes flaky noise, not this gate."""
        checks = {
            f"job {i}": JobCheckState(
                f"job {i}", "", "FAILURE", "AI_NOT_RELATED", i, "", ""
            )
            for i in range(IGNORABLE_FAILED_CHECKS_THESHOLD)
        }
        _, failed, _ = categorize_checks(
            checks, list(checks.keys()), ok_failed_checks_threshold=0
        )
        self.assertEqual(failed, [])

    def test_a_cleared_check_is_classified_through_get_classifications(self) -> None:
        """The happy path through the real entry point, not a hand-built state.

        The sibling tests either call the matcher directly or construct
        JobCheckState with the classification already set, so none of them
        notices if the `get_classifications` branch stops firing.
        """
        checks = {"job": self._check("job", 7)}
        drci = {"AI_NOT_RELATED": [{"id": 7, "name": "job"}]}
        with mock.patch("trymerge.get_drci_classifications", return_value=drci):
            classified = get_classifications(1, "pytorch", checks, None)
        self.assertEqual(classified["job"].classification, "AI_NOT_RELATED")

    def test_ignore_current_wins_over_being_cleared(self) -> None:
        """An explicit --ignore-current must not be counted against the cap.

        merge() builds the ignore list from raw conclusions with no
        classification step, so a cleared failure lands in it too. Claiming it
        for AI_NOT_RELATED would put the author's explicit decision under the
        classifier's budget.
        """
        checks = {"job": self._check("job", 7)}
        drci = {"AI_NOT_RELATED": [{"id": 7, "name": "job"}]}
        with mock.patch("trymerge.get_drci_classifications", return_value=drci):
            classified = get_classifications(1, "pytorch", checks, {(1, "job")})
        self.assertEqual(classified["job"].classification, "IGNORE_CURRENT_CHECK")

    def test_a_merge_i_over_the_cap_is_not_refused(self) -> None:
        """The regression the guard above prevents, at the boundary that bites.

        More cleared failures than the cap, all named by --ignore-current:
        without the guard every one classifies AI_NOT_RELATED, the cap trips,
        and a `merge -i` that succeeds today is refused.
        """
        over = IGNORABLE_FAILED_CHECKS_THESHOLD + 1
        names = [f"job {i}" for i in range(over)]
        # Job ids from 1: `is_ai_not_related` requires a truthy id, so a job
        # numbered 0 silently never matches and the count lands one under the
        # cap -- which passes this test with the guard removed.
        ids = list(range(1, over + 1))
        checks = {n: self._check(n, i) for n, i in zip(names, ids)}
        drci = {"AI_NOT_RELATED": [{"id": i, "name": n} for n, i in zip(names, ids)]}
        with mock.patch("trymerge.get_drci_classifications", return_value=drci):
            classified = get_classifications(
                1, "pytorch", checks, {(1, n) for n in names}
            )
        _, failed, _ = categorize_checks(classified, names)
        self.assertEqual(failed, [])

    def test_a_non_object_check_summary_is_no_classifications(self) -> None:
        """`null`/list/scalar decode fine; popping them must not end the merge."""
        for raw in ("null", "[]", "0", "false", '""'):
            with self.subTest(summary=raw):
                checks = {
                    DRCI_CHECKRUN_NAME: JobCheckState(
                        DRCI_CHECKRUN_NAME, "", "SUCCESS", None, 1, "", raw
                    ),
                    "job": self._check("job", 7),
                }
                with mock.patch("trymerge.get_drci_classifications", return_value={}):
                    classified = get_classifications(1, "pytorch", checks, None)
                self.assertIsNone(classified["job"].classification)

    def test_a_stale_check_summary_never_clears_a_gate(self) -> None:
        """The summary lags: Dr.CI skips rewriting it when the comment is unchanged."""
        summary = json.dumps(
            {
                "AI_NOT_RELATED": [{"id": 7, "name": "job"}],
                "FLAKY": [{"id": 9, "name": "flaky job"}],
            }
        )
        checks = {
            DRCI_CHECKRUN_NAME: JobCheckState(
                DRCI_CHECKRUN_NAME, "", "NEUTRAL", None, None, "", summary
            ),
            "job": self._check("job", 7),
            "flaky job": self._check("flaky job", 9),
        }
        with mock.patch("trymerge.get_drci_classifications", return_value={}):
            classified = get_classifications(1, "pytorch", checks, None)
        # The AI category is dropped from the fallback; FLAKY still applies.
        self.assertIsNone(classified["job"].classification)
        self.assertEqual(classified["flaky job"].classification, "FLAKY")


def stack_entry(
    position: int, closed: bool = False, head: str | None = None
) -> StackEntry:
    number = 999 + position
    base = f"user/{number - 1}" if position > 1 else "main"
    head = head or f"head-{number}"
    return StackEntry(position, number, closed, False, f"user/{number}", head, base)


def make_stack(*entries: StackEntry) -> NativeStack:
    return NativeStack(number=1, base_ref="main", entries=entries)


# #1000 landed and is closed: merging #1002 lands #1001 and #1002
NATIVE_STACK = make_stack(stack_entry(1, closed=True), stack_entry(2), stack_entry(3))
NATIVE_LANDING = [
    (NATIVE_STACK.entries[1], "tip-1000"),
    (NATIVE_STACK.entries[2], "head-1001"),
]


def stacked_pr(number: int) -> Any:
    pr = mock.MagicMock(spec=GitHubPR)
    pr.org = "pytorch"
    pr.project = "pytorch"
    pr.pr_num = number
    pr.last_commit_sha.return_value = f"head-{number}"
    pr.base_ref.return_value = f"user/{number - 1}"
    pr.default_branch.return_value = "main"
    pr.is_ghstack_pr.return_value = False
    pr.is_closed.return_value = False
    pr.is_docker_affecting.return_value = False
    pr.is_dependabot_pr.return_value = False
    pr.has_invalid_submodule_updates.return_value = False
    pr.get_labels.return_value = []
    pr.get_checkrun_conclusions.return_value = {}
    return pr


def stacked_pr_info(base: str = "user/1001", head: str = "user/1002") -> Any:
    return {
        **mock_gh_get_info(),
        "headRefName": head,
        "baseRefName": base,
        "baseRepository": {"defaultBranchRef": {"name": "main"}},
    }


class NoNetworkTestCase(TestCase):
    """Fails tests that reach GitHub through an unpatched helper. Also checked after
    the test, since retries_decorator and main() swallow the error."""

    def setUp(self) -> None:
        urlopen = self.patch("github_utils.urlopen", side_effect=AssertionError)
        self.addCleanup(urlopen.assert_not_called)

    def patch(self, target: str, **kwargs: Any) -> Any:
        patcher = mock.patch(target, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()


class TestGetNativeStackPrs(NoNetworkTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.repo = mock.MagicMock(spec=GitRepo)
        self.lower, self.top = stacked_pr(1001), stacked_pr(1002)
        self.landing = self.patch(
            "trymerge.get_native_stack_landing_prs", return_value=NATIVE_LANDING
        )
        self.pr_cls = self.patch("trymerge.GitHubPR", return_value=self.lower)

    def test_pairs_landing_entries_with_their_prs(self) -> None:
        self.assertEqual(
            get_native_stack_prs(self.repo, self.top, NATIVE_STACK),
            [(self.lower, *NATIVE_LANDING[0]), (self.top, *NATIVE_LANDING[1])],
        )
        self.landing.assert_called_once_with(
            self.repo, "pytorch", "pytorch", NATIVE_STACK, 1002, "main"
        )
        self.pr_cls.assert_called_once_with("pytorch", "pytorch", 1001)

    def test_refuses_pr_not_in_a_stack(self) -> None:
        with self.assertRaisesRegex(NativeStackError, "PR #1002 is not in a stack"):
            get_native_stack_prs(self.repo, self.top, None)
        self.landing.assert_not_called()

    def test_refuses_pr_updated_while_preparing(self) -> None:
        self.lower.last_commit_sha.return_value = "pushed-since"
        with self.assertRaisesRegex(NativeStackError, "PR #1001 was updated"):
            get_native_stack_prs(self.repo, self.top, NATIVE_STACK)

    def test_prs_to_merge_of_native_stack(self) -> None:
        prs = get_prs_to_merge(self.repo, self.top, NATIVE_STACK)
        self.assertEqual(prs, [self.lower, self.top])

    def test_prs_to_merge_of_regular_pr(self) -> None:
        self.assertEqual(get_prs_to_merge(self.repo, self.top), [self.top])
        self.landing.assert_not_called()

    @mock.patch("trymerge.get_ghstack_prs")
    def test_prs_to_merge_of_ghstack_pr(self, mock_get_ghstack_prs: Any) -> None:
        self.top.is_ghstack_pr.return_value = True
        mock_get_ghstack_prs.return_value = [(self.lower, "rev"), (self.top, "rev")]
        prs = get_prs_to_merge(self.repo, self.top, NATIVE_STACK)
        self.assertEqual(prs, [self.lower, self.top])
        mock_get_ghstack_prs.assert_called_once_with(self.repo, self.top)
        self.landing.assert_not_called()


class TestNativeStackMergeInto(NoNetworkTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.repo = mock.MagicMock(spec=GitRepo)
        self.repo.rev_parse.return_value = "merged-sha"
        self.lower, self.top = stacked_pr(1001), stacked_pr(1002)
        self.top.merge_native_stack_into.return_value = [self.lower, self.top]
        self.patch("trymerge.GitHubPR", return_value=self.lower)
        self.get_stack = self.patch(
            "trymerge.get_native_stack", return_value=NATIVE_STACK
        )
        self.landing = self.patch(
            "trymerge.get_native_stack_landing_prs", return_value=NATIVE_LANDING
        )
        self.rule = self.patch(
            "trymerge.find_matching_merge_rule", return_value=(None, [], [], {})
        )
        self.patch("trymerge.can_skip_internal_checks", return_value=False)
        self.greenlight = self.patch("trymerge.check_greenlight_reviewed_head_sha")
        self.docker = self.patch("trymerge.check_docker_builds_ready")
        self.record = self.patch("trymerge.save_merge_record")
        self.close = self.patch("trymerge.manually_close_merged_pr")
        self.patch("trymerge.time.sleep")

    def merge_into(self, stack: NativeStack = NATIVE_STACK, **kwargs: Any) -> None:
        GitHubPR.merge_into(
            self.top,
            self.repo,
            comment_id=1,
            native_stack=stack,
            trunk_at_start="trunk-at-start",
            **kwargs,
        )

    def assertStopsBeforeLanding(self, **kwargs: Any) -> None:
        self.top.merge_native_stack_into.side_effect = RuntimeError("stop after gates")
        with self.assertRaisesRegex(RuntimeError, "stop after gates"):
            self.merge_into(**kwargs)

    def test_gates_lower_prs_before_landing(self) -> None:
        waivers = {(1001, "red")}
        self.assertStopsBeforeLanding(ignore_current_checks=waivers)
        gated = [call.args[0] for call in self.rule.call_args_list]
        self.assertEqual(gated, [self.top, self.lower])
        self.assertEqual(
            self.rule.call_args.kwargs,
            {
                "skip_mandatory_checks": False,
                "skip_internal_checks": False,
                "ignore_current_checks": waivers,
            },
        )
        self.top.merge_native_stack_into.assert_called_once_with(
            self.repo,
            [(self.lower, *NATIVE_LANDING[0]), (self.top, *NATIVE_LANDING[1])],
            "trunk-at-start",
            None,
            False,
        )
        self.assertEqual(self.repo.mock_calls, [])

    def test_reads_the_stack_again_for_every_attempt(self) -> None:
        current = replace(NATIVE_STACK)
        self.get_stack.return_value = current
        self.assertStopsBeforeLanding()
        self.get_stack.assert_called_once_with("pytorch", "pytorch", 1002)
        self.assertIs(self.landing.call_args.args[3], current)

    def test_wraps_lower_pr_rule_failure(self) -> None:
        for error in (
            MergeRuleFailedError("approve"),
            MandatoryChecksMissingError("wait"),
        ):
            with self.subTest(error=error):
                self.rule.side_effect = [(None, [], [], {}), error]
                with self.assertRaises(MergeRuleFailedError) as cm:
                    self.merge_into()
                self.assertIs(type(cm.exception), type(error))
                self.assertIn(f"stacked PR #1001:\n\n{error}", str(cm.exception))
        self.top.merge_native_stack_into.assert_not_called()

    def test_refuses_pr_pushed_or_reopened_after_merge_started(self) -> None:
        for lower in (stack_entry(2, head="old"), stack_entry(2, closed=True)):
            with self.subTest(lower=lower):
                start = make_stack(stack_entry(1, closed=True), lower, stack_entry(3))
                with self.assertRaisesRegex(NativeStackError, "PR #1001 changed after"):
                    self.merge_into(start)
        self.top.merge_native_stack_into.assert_not_called()

    def test_checks_submodules_of_lower_prs_unless_forced(self) -> None:
        self.lower.has_invalid_submodule_updates.return_value = True
        self.lower.get_changed_submodules.return_value = ["third_party/kineto"]
        with self.assertRaisesRegex(RuntimeError, "#1001 updates submodules third_"):
            self.merge_into()
        self.top.merge_native_stack_into.assert_not_called()
        self.assertStopsBeforeLanding(skip_mandatory_checks=True)

    def test_greenlight_and_docker_gates_cover_every_landing_pr(self) -> None:
        self.lower.is_docker_affecting.return_value = True
        self.assertStopsBeforeLanding()
        prs = [self.lower, self.top]
        self.assertEqual(self.greenlight.call_args.args, (self.top, self.repo, prs))
        self.docker.assert_called_once_with(self.lower)
        self.assertIs(self.top.merge_native_stack_into.call_args.args[3], self.lower)

    def test_dependabot_pr_is_landed_by_the_bot(self) -> None:
        self.top.is_dependabot_pr.return_value = True
        self.assertStopsBeforeLanding()
        self.top.merge_via_github_api.assert_not_called()

    def test_labels_and_closes_every_landed_pr(self) -> None:
        self.merge_into(dry_run=False)
        self.assertIs(self.top.merge_native_stack_into.call_args.args[4], False)
        self.repo.push.assert_not_called()
        self.lower.add_numbered_label.assert_called_once_with(
            MERGE_COMPLETE_LABEL, False
        )
        self.top.add_numbered_label.assert_called_with(MERGE_COMPLETE_LABEL, False)
        self.close.assert_called_once_with(
            pr=self.top,
            additional_merged_prs=[self.lower, self.top],
            merge_commit_sha="merged-sha",
            dry_run=False,
            landed_heads={1001: "head-1001", 1002: "head-1002"},
        )
        self.assertEqual(self.record.call_args.kwargs["merge_commit_sha"], "merged-sha")

    def test_failed_push_labels_and_closes_nothing(self) -> None:
        self.top.merge_native_stack_into.side_effect = RuntimeError("push rejected")
        with self.assertRaisesRegex(RuntimeError, "push rejected"):
            self.merge_into(dry_run=False)
        self.lower.add_numbered_label.assert_not_called()
        self.top.add_numbered_label.assert_not_called()
        self.record.assert_not_called()
        self.close.assert_not_called()

    def test_regular_pr_never_reads_a_stack(self) -> None:
        self.top.merge_changes_locally.side_effect = RuntimeError("regular merge")
        with self.assertRaisesRegex(RuntimeError, "regular merge"):
            GitHubPR.merge_into(self.top, self.repo, comment_id=1)
        self.get_stack.assert_not_called()
        self.landing.assert_not_called()


class TestCloseLandedPrs(NoNetworkTestCase):
    def test_pr_updated_after_it_landed_is_told_what_did_not_land(self) -> None:
        prs = {num: stacked_pr(num) for num in (1001, 1002)}
        prs[1001].last_commit_sha.return_value = "pushed-since"
        self.patch("trymerge.GitHubPR", side_effect=lambda _, __, num: prs[num])
        post = self.patch("trymerge.gh_post_pr_comment")

        def close_pr(org: str, project: str, num: int, dry_run: bool) -> None:
            prs[num].is_closed.return_value = True

        close = self.patch("trymerge.gh_close_pr", side_effect=close_pr)
        heads = {1001: "head-1001", 1002: "head-1002"}
        manually_close_merged_pr(
            prs[1002], list(prs.values()), "merged-sha", False, landed_heads=heads
        )
        moved = (
            "This PR landed at head-1001, but its head is now pushed-since. Changes "
            "that are not in head-1001 did not land; please open a new PR for them."
        )
        comments = [(call.args[2], call.args[3]) for call in post.call_args_list]
        self.assertEqual([num for num, _ in comments], [1002, 1001, 1001])
        self.assertEqual(comments[1], (1001, moved))
        self.assertEqual([call.args[2] for call in close.call_args_list], [1002, 1001])


class TestMergeNativeStackInto(NoNetworkTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.repo = mock.MagicMock(spec=GitRepo)
        self.repo.remote = "origin"
        self.repo.rev_parse.return_value = "trunk-sha"
        self.lower, self.top = stacked_pr(1001), stacked_pr(1002)
        for pr in (self.lower, self.top):
            pr.get_author.return_value = f"Author <{pr.pr_num}@example.com>"
            pr.gen_commit_message.return_value = f"Message {pr.pr_num}\n"
            pr.get_pr_url.return_value = pr_url(pr.pr_num)
        self.native_prs = [
            (self.lower, *NATIVE_LANDING[0]),
            (self.top, *NATIVE_LANDING[1]),
        ]
        self.get_stack = self.patch(
            "trymerge.get_native_stack", return_value=NATIVE_STACK
        )
        self.landing = self.patch(
            "trymerge.get_native_stack_landing_prs", return_value=NATIVE_LANDING
        )
        self.build = self.patch(
            "trymerge.build_native_stack_commits", return_value="top-sha"
        )
        self.patch("trymerge.landed_since", return_value=False)
        self.skew = self.patch("trymerge.warn_on_docker_merge_skew")

    def land(self, docker_pr: Any = None, dry_run: bool = False) -> list[GitHubPR]:
        return GitHubPR.merge_native_stack_into(
            self.top, self.repo, self.native_prs, "trunk-at-start", docker_pr, dry_run
        )

    def test_refuses_stack_that_changed_while_merging(self) -> None:
        moved = [(NATIVE_STACK.entries[1], "tip-1000-moved"), NATIVE_LANDING[1]]
        for stack, landing in (
            (None, NATIVE_LANDING),
            (NATIVE_STACK, moved),
            (NATIVE_STACK, NATIVE_LANDING[1:]),
        ):
            with self.subTest(stack=stack, landing=landing):
                self.get_stack.return_value = stack
                self.landing.return_value = landing
                with self.assertRaisesRegex(NativeStackError, "PR #1002 changed while"):
                    self.land()
        self.build.assert_not_called()
        self.repo._run_git.assert_not_called()
        self.repo.push.assert_not_called()

    def test_gives_up_after_bounded_push_attempts(self) -> None:
        self.repo.push.side_effect = RuntimeError("rejected")
        with self.assertRaisesRegex(RuntimeError, "rejected"):
            self.land()
        self.assertEqual(self.build.call_count, NATIVE_STACK_PUSH_ATTEMPTS)
        self.assertEqual(self.repo.push.call_count, NATIVE_STACK_PUSH_ATTEMPTS)

    def test_passes_docker_pr_and_dry_run_through(self) -> None:
        self.land(docker_pr=self.lower, dry_run=True)
        self.skew.assert_called_once_with(self.repo, self.lower)
        self.repo.push.assert_called_once_with("main", True, retry=1)


class TestStackDependenciesLine(NoNetworkTestCase):
    def message(self, stack_deps: list[int] | None) -> str:
        pr = mock.MagicMock(spec=GitHubPR)
        pr.pr_num = 1002
        pr.get_title.return_value = "Add the thing"
        pr.get_body.return_value = "Body"
        pr.get_pr_url.return_value = "https://github.com/pytorch/pytorch/pull/1002"
        pr.get_approved_by.return_value = ["reviewer"]
        pr.get_pr_creator_login.return_value = "creator"
        pr.get_authors.return_value = {
            "creator": "Creator <creator@example.com>",
            "coauthor": "Co Author <co@example.com>",
        }
        return GitHubPR.gen_commit_message(pr, stack_deps=stack_deps)

    def test_lists_prs_below_before_coauthors(self) -> None:
        self.assertEqual(
            self.message([1000, 1001]),
            "Add the thing (#1002)\n\nBody\nPull Request resolved: https://github.com/pytorch/pytorch/pull/1002\nApproved by: https://github.com/reviewer\nStack dependencies: #1000, #1001\n\n\nCo-authored-by: Co Author <co@example.com>",
        )


class TestStartingMergeComment(NoNetworkTestCase):
    def test_comments_on_every_pr_that_lands(self) -> None:
        lower, top = stacked_pr(1001), stacked_pr(1002)
        post = self.patch("trymerge.gh_post_pr_comment")
        explainer = mock.MagicMock()
        explainer.get_merge_message.return_value = "Merge started"
        post_starting_merge_comment(top, [lower, top], explainer, True)
        self.assertEqual(
            post.call_args_list,
            [
                mock.call("pytorch", "pytorch", 1002, "Merge started", dry_run=True),
                mock.call(
                    "pytorch",
                    "pytorch",
                    1001,
                    "Starting merge as part of PR stack under #1002",
                    dry_run=True,
                ),
            ],
        )


class TestNativeStackMergePlumbing(NoNetworkTestCase):
    def setUp(self) -> None:
        super().setUp()
        for target in (
            "trymerge.gh_add_labels",
            "trymerge.check_for_sev",
            "trymerge.ensure_mergeable_labels",
            "trymerge.find_matching_merge_rule",
        ):
            self.patch(target)
        self.patch("trymerge.get_classifications", return_value={})
        self.post = self.patch("trymerge.post_starting_merge_comment")
        self.lower, self.top = stacked_pr(1001), stacked_pr(1002)
        self.patch("trymerge.GitHubPR", return_value=self.top)
        self.prs_to_merge = self.patch(
            "trymerge.get_prs_to_merge", return_value=[self.lower, self.top]
        )
        self.repo = mock.MagicMock(spec=GitRepo)

    def merge(self, **kwargs: Any) -> None:
        merge(
            self.top,
            self.repo,
            comment_id=1,
            dry_run=True,
            native_stack=NATIVE_STACK,
            trunk_at_start="trunk-at-start",
            **kwargs,
        )

    def test_passes_the_stack_down(self) -> None:
        for force in (False, True):
            with self.subTest(force=force):
                self.merge(skip_mandatory_checks=force)
                self.prs_to_merge.assert_called_once_with(
                    self.repo, self.top, NATIVE_STACK
                )
                self.prs_to_merge.reset_mock()
                stacked = self.post.call_args.args[:2]
                self.assertEqual(stacked, (self.top, [self.lower, self.top]))
                kwargs = self.top.merge_into.call_args.kwargs
                self.assertIs(kwargs["native_stack"], NATIVE_STACK)
                self.assertEqual(kwargs["trunk_at_start"], "trunk-at-start")

    def test_ignore_current_waives_red_checks_of_every_landing_pr(self) -> None:
        red = JobCheckState("red", "", "FAILURE", None, None, None, None)
        for pr in (self.lower, self.top):
            pr.get_checkrun_conclusions.return_value = {"red": red}
        self.merge(ignore_current=True)
        self.assertEqual(
            self.top.merge_into.call_args.kwargs["ignore_current_checks"],
            {(1001, "red"), (1002, "red")},
        )


TARGETS_ANOTHER_BRANCH = "PR targets user/1001 rather than main, refusing merge request"


class TestNativeStackMain(NoNetworkTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.args = mock_parse_args()
        self.args.pr_num = 1002
        self.patch("trymerge.parse_args", return_value=self.args)
        self.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
        self.info = self.patch(
            "trymerge.gh_get_pr_info", return_value=stacked_pr_info()
        )
        self.post = self.patch("trymerge.gh_post_pr_comment")
        self.merge = self.patch("trymerge.merge")
        self.get_stack = self.patch(
            "trymerge.get_native_stack", return_value=NATIVE_STACK
        )
        self.rev_parse = self.patch(
            "trymerge.GitRepo.rev_parse", return_value="trunk-at-start"
        )

    def assertRefused(self, refusal: str = TARGETS_ANOTHER_BRANCH) -> None:
        self.assertEqual(
            self.post.call_args_list,
            [mock.call("pytorch", "pytorch", 1002, refusal, dry_run=True)],
        )
        self.merge.assert_not_called()

    def test_refuses_pr_without_usable_stack(self) -> None:
        unreadable = (
            "Could not read the stack of PR #1002, refusing merge request: GraphQL "
            "errors: Timeout"
        )
        other_trunk = (
            "PR #1002 is in a stack based on viable/strict rather than main, refusing "
            "merge request"
        )
        viable_strict = replace(NATIVE_STACK, base_ref="viable/strict")
        for stack, error, refusal in (
            (None, None, TARGETS_ANOTHER_BRANCH),
            (None, NativeStackError("GraphQL errors: Timeout"), unreadable),
            (viable_strict, None, other_trunk),
        ):
            with self.subTest(refusal=refusal):
                self.get_stack.return_value = stack
                self.get_stack.side_effect = error
                self.post.reset_mock()
                trymerge_main()
                self.assertRefused(refusal)

    def test_merges_pr_in_stack_on_default_branch(self) -> None:
        reads = mock.Mock()
        reads.attach_mock(self.rev_parse, "rev_parse")
        reads.attach_mock(self.get_stack, "get_native_stack")
        trymerge_main()
        self.assertEqual(
            reads.mock_calls,
            [
                mock.call.rev_parse("refs/remotes/origin/main"),
                mock.call.get_native_stack("pytorch", "pytorch", 1002),
            ],
        )
        self.post.assert_not_called()
        self.merge.assert_called_once_with(
            mock.ANY,
            mock.ANY,
            comment_id=12345,
            dry_run=True,
            skip_mandatory_checks=False,
            ignore_current=False,
            native_stack=NATIVE_STACK,
            trunk_at_start="trunk-at-start",
        )

    def test_check_mergeability_does_not_read_the_stack(self) -> None:
        self.args.check_mergeability = True
        trymerge_main()
        self.get_stack.assert_not_called()
        self.rev_parse.assert_not_called()
        self.assertRefused()

    def test_bottom_and_ghstack_prs_do_not_read_a_stack(self) -> None:
        for info in (
            stacked_pr_info(base="main"),
            stacked_pr_info(base="gh/someone/1/base", head="gh/someone/1/head"),
        ):
            with self.subTest(base=info["baseRefName"]):
                self.info.return_value = info
                trymerge_main()
                kwargs = self.merge.call_args.kwargs
                self.assertIsNone(kwargs["native_stack"])
                self.assertIsNone(kwargs["trunk_at_start"])
        self.get_stack.assert_not_called()
        self.rev_parse.assert_not_called()
        self.post.assert_not_called()

    def test_revert_does_not_read_the_stack(self) -> None:
        self.args.revert = True
        revert = self.patch("trymerge.try_revert")
        trymerge_main()
        revert.assert_called_once()
        self.get_stack.assert_not_called()
        self.rev_parse.assert_not_called()
        self.merge.assert_not_called()


def native_stack_pr(entry: StackEntry) -> Any:
    """A PR of a stack pushed by GitTestCase, as merge_into reads it"""
    pr = stacked_pr(entry.number)
    pr.last_commit_sha.return_value = entry.head_oid
    pr.base_ref.return_value = entry.base_ref
    pr.head_ref.return_value = entry.head_ref
    pr.get_title.return_value = f"Change {entry.number}"
    pr.get_body.return_value = f"Body {entry.number}"
    pr.get_pr_url.return_value = pr_url(entry.number)
    pr.get_approved_by.return_value = ["reviewer"]
    pr.get_pr_creator_login.return_value = "author"
    pr.get_authors.return_value = {"author": author(entry.number)}
    pr.get_author.return_value = author(entry.number)
    pr.gen_commit_message = MethodType(GitHubPR.gen_commit_message, pr)
    pr.merge_native_stack_into = MethodType(GitHubPR.merge_native_stack_into, pr)
    return pr


class TestNativeStackMergeEndToEnd(NoNetworkTestCase, GitTestCase):
    """GitHubPR.merge_into lands a real stack: the bot clone builds the commits and
    pushes them to a local bare origin. Only GitHub is patched."""

    def setUp(self) -> None:
        GitTestCase.setUp(self)
        NoNetworkTestCase.setUp(self)
        self.origin = self.git("remote", "get-url", "origin")
        self.stack = self.push_stack()
        self.prs = {e.number: native_stack_pr(e) for e in self.stack.entries}
        self.patch("trymerge.GitHubPR", side_effect=lambda _, __, num: self.prs[num])
        self.get_stack = self.patch(
            "trymerge.get_native_stack", return_value=self.stack
        )
        self.patch("trymerge.find_matching_merge_rule", return_value=(None, [], [], {}))
        self.patch("trymerge.can_skip_internal_checks", return_value=False)
        self.patch("trymerge.check_greenlight_reviewed_head_sha")
        self.patch("trymerge.save_merge_record")
        self.patch("trymerge.time.sleep")
        self.close = self.patch("trymerge.manually_close_merged_pr")
        self.patch("trymerge.build_native_stack_commits", side_effect=self.build)
        self.bases: list[str] = []
        self.races: list[Callable[[], object]] = []

    def build(self, repo: GitRepo, base: str, landing: Any, commits: Any) -> str:
        """The real build, followed by the next race: a push from another clone that
        lands before the bot's push"""
        self.bases.append(base)
        top = build_native_stack_commits(repo, base, landing, commits)
        if self.races:
            self.races.pop(0)()
        return top

    def merge_stack(
        self, stack: NativeStack | None = None, start: str | None = None
    ) -> None:
        GitHubPR.merge_into(
            self.prs[103],
            self.repo,
            comment_id=1,
            dry_run=False,
            native_stack=stack or self.stack,
            trunk_at_start=start or self.git("rev-parse", TRUNK, cwd=self.origin),
        )

    def origin_log(self, since: str, fmt: str = "%H") -> list[str]:
        revs = f"{since}..{TRUNK}"
        log = self.git("log", "--reverse", f"--format={fmt}", revs, cwd=self.origin)
        return log.splitlines()

    def message(self, number: int, *deps: int) -> str:
        msg = (
            f"Change {number} (#{number})\n\nBody {number}\n"
            f"{PULL_REQUEST_RESOLVED}{pr_url(number)}\n"
            "Approved by: https://github.com/reviewer\n"
        )
        if deps:
            msg += f"{stack_dependencies_line(list(deps))}\n"
        return msg

    def test_pushes_one_commit_per_pr(self) -> None:
        trunk = self.add_commits(TRUNK)
        self.merge_stack()
        landed = self.origin_log(trunk)
        authors = [author(n) for n in (101, 102, 103)]
        self.assertEqual(self.origin_log(trunk, "%an <%ae>"), authors)
        origin = GitRepo(self.origin)
        raw = [origin._run_git("cat-file", "commit", commit) for commit in landed]
        self.assertEqual(
            [commit.partition("\n\n")[2] for commit in raw],
            [self.message(101), self.message(102, 101), self.message(103, 101, 102)],
        )
        c_head = self.stack.entries[2].head_oid
        changes = self.git("diff", self.root, c_head, cwd=self.origin)
        self.assertEqual(self.git("diff", trunk, landed[-1], cwd=self.origin), changes)
        self.close.assert_called_once_with(
            pr=self.prs[103],
            additional_merged_prs=list(self.prs.values()),
            merge_commit_sha=landed[-1],
            dry_run=False,
            landed_heads={e.number: e.head_oid for e in self.stack.entries},
        )

    def test_rebuilds_on_trunk_that_moved_before_the_push(self) -> None:
        self.races.append(lambda: self.add_commits(TRUNK))
        self.merge_stack()
        landed = self.origin_log(self.root)
        subjects = [f"Update {TRUNK}", *(f"Change {n} (#{n})" for n in (101, 102, 103))]
        self.assertEqual(self.origin_log(self.root, "%s"), subjects)
        self.assertEqual(self.bases, [self.root, landed[0]])

    def test_aborts_when_a_landed_pr_is_reverted_before_the_push(self) -> None:
        a = self.stack.entries[0]
        landed = self.land(a)
        stack = with_entry(self.stack, 0, closed=True)
        self.get_stack.return_value = stack
        self.races.append(lambda: self.revert(a, landed))
        with self.assertRaisesRegex(NativeStackError, "PR #101 is closed but not"):
            self.merge_stack(stack)
        self.assertEqual(self.origin_log(landed, "%s"), ['Revert "Change"'])

    def test_aborts_when_a_lower_pr_landed_after_the_merge_started(self) -> None:
        a = self.stack.entries[0]
        reverted = self.revert(a, self.land(a))
        message = r"PR #101 landed after this merge started"
        with self.assertRaisesRegex(NativeStackError, message):
            self.merge_stack(start=self.root)
        self.assertEqual(self.bases, [])
        self.assertEqual(self.origin_log(reverted), [])

    def test_relands_a_pr_reverted_before_the_merge_started(self) -> None:
        a = self.stack.entries[0]
        reverted = self.revert(a, self.land(a))
        self.merge_stack(start=reverted)
        subjects = [f"Change {n} (#{n})" for n in (101, 102, 103)]
        self.assertEqual(self.origin_log(reverted, "%s"), subjects)


if __name__ == "__main__":
    main()
