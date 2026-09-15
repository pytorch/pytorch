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
from hashlib import sha256
from typing import Any
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
    get_topmost_docker_pr,
    gh_get_pr_info,
    gh_get_team_members,
    GitHubPR,
    is_authorized_without_greenlight,
    is_bot_initiated_codev_merge,
    is_docker_affecting_files,
    iter_issue_timeline_until_comment,
    JobCheckState,
    main as trymerge_main,
    MandatoryChecksMissingError,
    merge,
    merge_authorized_logins,
    MergeRule,
    MergeRuleFailedError,
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
        # merge base here to get the results from Dr. CI, and that categorize the
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

    def test_ignore_failures_older_run_same_workflow(self, *args: Any) -> None:
        pr = GitHubPR("pytorch", "pytorch", 129013)
        checks = pr.get_checkrun_conclusions()
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
            [],
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
            repo, False, 1, ghstack_prs=ghstack_prs
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
                ignore_current_checks=["some-check"],
            )
        self.assertEqual(
            mock_rule.call_args.kwargs,
            {
                "skip_mandatory_checks": True,
                "skip_internal_checks": True,
                "ignore_current_checks": ["some-check"],
                "approved_by_override": {"malfet"},
            },
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
                ignore_current_checks=["some-check"],
            )
            _, prs = mock_evaluate.call_args.args
            prs[0].is_authorized_without_greenlight()
        self.assertEqual(
            mock_authz.call_args.kwargs,
            {
                "skip_mandatory_checks": True,
                "skip_internal_checks": True,
                "ignore_current_checks": ["some-check"],
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
                ignore_current_checks=["some-check"],
            )

        self.assertEqual(mock_check.call_args.kwargs["skip_mandatory_checks"], True)
        self.assertEqual(mock_check.call_args.kwargs["skip_internal_checks"], True)
        self.assertEqual(
            mock_check.call_args.kwargs["ignore_current_checks"], ["some-check"]
        )


if __name__ == "__main__":
    main()
