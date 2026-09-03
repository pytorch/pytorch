from __future__ import annotations

import argparse
import unittest
from unittest import mock

from apply_triage_decision import apply_controller_action, ControllerOutcome
from github_reviews import fetch_maintainer_activity, fetch_user_has_triage_permission
from ownership import EXTRA_OWNERSHIP_METADATA_PATH, TEAM_MEMBERS_PATH
from process_llm_output import build_analysis_result
from schemas import AnalysisResult, TriageInput


WORKFLOW_SHA = "a" * 40
HEAD_SHA = "b" * 40
REPOSITORY = "pytorch/ciforge"


def ownership() -> dict:
    def source(path: str, blob: str) -> dict:
        return {
            "repository": REPOSITORY,
            "path": path,
            "ref": WORKFLOW_SHA,
            "blob_sha": blob * 40,
        }

    return {
        "extra_ownership_metadata": {
            "source": source(EXTRA_OWNERSHIP_METADATA_PATH, "c"),
            "owners": {"team": "Owns the test subsystem."},
        },
        "team_members": {
            "source": source(TEAM_MEMBERS_PATH, "e"),
            "members": {"team": ["@owner"]},
        },
    }


def triage_input(
    *,
    trusted_author: bool,
    codepath_owners: list[str],
    has_maintainer_activity: bool = False,
) -> TriageInput:
    codepath_artifact = {
        "source": {
            "repository": REPOSITORY,
            "path": ".github/auto-pr-triage/codepath_owners.txt",
            "ref": WORKFLOW_SHA,
            "blob_sha": "d" * 40,
        },
        "owners": codepath_owners,
        "matched_path_groups": (
            [{"owners": codepath_owners, "paths": ["torch/file.py"]}]
            if codepath_owners
            else []
        ),
        "paths_without_owners": (
            []
            if codepath_owners
            else [{"path": "torch/file.py", "reason": "no_matching_rule"}]
        ),
    }
    extra_metadata = ownership()["extra_ownership_metadata"]
    return TriageInput.create(
        "policy",
        {
            "codepath_owners": codepath_artifact,
            "extra_ownership_metadata": extra_metadata,
        },
        {
            "repository": REPOSITORY,
            "number": 123,
            "url": f"https://github.com/{REPOSITORY}/pull/123",
            "title": "fixture",
            "body": "body",
            "base_ref": "main",
            "workflow_sha": WORKFLOW_SHA,
            "head_sha": HEAD_SHA,
            "files": [{"path": "torch/file.py"}],
            "is_open_non_draft_pr_against_main": True,
            "is_already_handled": False,
            "has_actionable_linked_issue": False,
            "author_has_triage_permission": trusted_author,
            "has_maintainer_activity": has_maintainer_activity,
            "diff_truncated_or_unavailable": False,
        },
    )


def recommendation_without_additional_owners() -> dict:
    return {
        "analyzed_file_indices": [0],
        "additional_owners": [],
        "uncovered_concerns": [],
        "confidence": "medium",
        "security_flags": [],
        "rationale": "No additional owner is warranted.",
    }


def codepath_provenance(owners: list[str]) -> dict[str, dict[str, object]]:
    return {
        owner: {
            "source": "codepath",
            "files": ["torch/file.py"],
            "total_file_count": 1,
            "llm_justification": None,
        }
        for owner in owners
    }


class FakePermissionGitHub:
    def __init__(self, permissions: dict[str, bool], login: str = "author") -> None:
        self.permissions = permissions
        self.login = login
        self.calls: list[str] = []

    def json(self, endpoint: str) -> dict:
        self.calls.append(endpoint)
        return {"user": {"login": self.login, "permissions": self.permissions}}


class AuthorPermissionTest(unittest.TestCase):
    def test_triage_or_higher_permission_is_trusted(self) -> None:
        for permission in ("triage", "push", "maintain", "admin"):
            with self.subTest(permission=permission):
                permissions = {
                    "triage": False,
                    "push": False,
                    "maintain": False,
                    "admin": False,
                }
                permissions[permission] = True
                github = FakePermissionGitHub(permissions)
                self.assertTrue(
                    fetch_user_has_triage_permission(github, REPOSITORY, "author")
                )
                self.assertEqual(
                    github.calls,
                    [f"repos/{REPOSITORY}/collaborators/author/permission"],
                )

    def test_read_only_permission_is_not_trusted(self) -> None:
        github = FakePermissionGitHub(
            {"triage": False, "push": False, "maintain": False, "admin": False}
        )
        self.assertFalse(fetch_user_has_triage_permission(github, REPOSITORY, "author"))

    def test_nested_triage_permission_wins_over_legacy_read_value(self) -> None:
        github = mock.Mock()
        github.json.return_value = {
            "permission": "read",
            "role_name": "triage",
            "user": {
                "login": "author",
                "permissions": {
                    "pull": True,
                    "triage": True,
                    "push": False,
                    "maintain": False,
                    "admin": False,
                },
            },
        }
        self.assertTrue(fetch_user_has_triage_permission(github, REPOSITORY, "author"))

    def test_permission_response_is_strictly_validated(self) -> None:
        valid = {
            "triage": True,
            "push": False,
            "maintain": False,
            "admin": False,
        }
        for github in (
            FakePermissionGitHub(valid, login="someone-else"),
            FakePermissionGitHub({"triage": "true"}),
        ):
            with self.subTest(response=github.permissions):
                with self.assertRaises(RuntimeError):
                    fetch_user_has_triage_permission(github, REPOSITORY, "author")

    def test_trusted_author_enables_analysis_without_overriding_owners(self) -> None:
        without_codepath_owners = build_analysis_result(
            triage_input(trusted_author=True, codepath_owners=[]),
            recommendation_without_additional_owners(),
        )
        self.assertEqual(
            without_codepath_owners,
            AnalysisResult.create(
                is_open_non_draft_pr_against_main=True,
                is_already_handled=False,
                author_has_triage_permission=True,
                has_actionable_linked_issue=False,
                has_maintainer_activity=False,
                ownership_analysis="completed",
                analyzed_head_sha=HEAD_SHA,
            ),
        )

        with_codepath_owners = build_analysis_result(
            triage_input(trusted_author=True, codepath_owners=["@owner", "team"]),
            recommendation_without_additional_owners(),
        )
        self.assertEqual(
            with_codepath_owners,
            AnalysisResult.create(
                is_open_non_draft_pr_against_main=True,
                is_already_handled=False,
                author_has_triage_permission=True,
                has_actionable_linked_issue=False,
                has_maintainer_activity=False,
                ownership_analysis="completed",
                codepath_owners=["@owner", "team"],
                analyzed_head_sha=HEAD_SHA,
                owner_provenance=codepath_provenance(["@owner", "team"]),
            ),
        )

    def test_trusted_author_state_is_hidden_from_worker(self) -> None:
        value = triage_input(trusted_author=True, codepath_owners=[])
        self.assertEqual(value.to_dict()["schema_version"], 10)
        self.assertTrue(
            value.to_dict()["trusted_context"]["analysis_metadata"][
                "author_has_triage_permission"
            ]
        )
        self.assertNotIn(
            "author_has_triage_permission",
            value.to_worker_dict()["trusted_context"]["analysis_metadata"],
        )


class MaintainerActivityTest(unittest.TestCase):
    def test_review_comment_and_self_request_are_activity(self) -> None:
        events = [
            {
                "event": "reviewed",
                "state": "dismissed",
                "author_association": "MEMBER",
                "user": {"login": "maintainer", "type": "User"},
            },
            {
                "event": "commented",
                "author_association": "MEMBER",
                "user": {"login": "maintainer", "type": "User"},
            },
            {
                "event": "review_requested",
                "actor": {"login": "maintainer", "type": "User"},
                "requested_reviewer": {
                    "login": "maintainer",
                    "type": "User",
                },
            },
        ]
        with (
            mock.patch(
                "github_reviews.fetch_pull_request_timeline",
                return_value=events,
            ),
            mock.patch(
                "github_reviews.fetch_user_has_triage_permission",
                return_value=True,
            ) as permission,
        ):
            activity = fetch_maintainer_activity(mock.Mock(), REPOSITORY, 123, "author")

        self.assertEqual(
            activity,
            ("@maintainer", ("comment", "review", "self_review_request")),
        )
        permission.assert_called_once_with(mock.ANY, REPOSITORY, "maintainer")

    def test_only_self_requested_review_counts(self) -> None:
        events = [
            {
                "event": "review_requested",
                "actor": {"login": "requester", "type": "User"},
                "requested_reviewer": {
                    "login": "maintainer",
                    "type": "User",
                },
            },
            {
                "event": "review_requested",
                "actor": {"login": "github-actions[bot]", "type": "Bot"},
                "requested_reviewer": {
                    "login": "maintainer",
                    "type": "User",
                },
            },
        ]
        with (
            mock.patch(
                "github_reviews.fetch_pull_request_timeline",
                return_value=events,
            ),
            mock.patch("github_reviews.fetch_user_has_triage_permission") as permission,
        ):
            activity = fetch_maintainer_activity(mock.Mock(), REPOSITORY, 123, "author")

        self.assertIsNone(activity)
        permission.assert_not_called()

    def test_author_and_bots_are_ignored_but_external_user_is_verified(self) -> None:
        events = [
            {
                "event": "commented",
                "author_association": "MEMBER",
                "user": {"login": "author", "type": "User"},
            },
            {
                "event": "commented",
                "author_association": "MEMBER",
                "user": {"login": "github-actions[bot]", "type": "Bot"},
            },
            {
                "event": "commented",
                "author_association": "MEMBER",
                "user": {"login": "pytorchbot", "type": "User"},
            },
            {
                "event": "commented",
                "author_association": "CONTRIBUTOR",
                "user": {"login": "contributor", "type": "User"},
            },
        ]
        with (
            mock.patch(
                "github_reviews.fetch_pull_request_timeline",
                return_value=events,
            ),
            mock.patch(
                "github_reviews.fetch_user_has_triage_permission",
                return_value=False,
            ) as permission,
        ):
            activity = fetch_maintainer_activity(mock.Mock(), REPOSITORY, 123, "author")

        self.assertIsNone(activity)
        permission.assert_called_once_with(mock.ANY, REPOSITORY, "contributor")

    def test_read_only_collaborator_does_not_count_as_activity(self) -> None:
        events = [
            {
                "event": "commented",
                "author_association": "COLLABORATOR",
                "user": {"login": "reader", "type": "User"},
            }
        ]
        with (
            mock.patch(
                "github_reviews.fetch_pull_request_timeline",
                return_value=events,
            ),
            mock.patch(
                "github_reviews.fetch_user_has_triage_permission",
                return_value=False,
            ) as permission,
        ):
            activity = fetch_maintainer_activity(mock.Mock(), REPOSITORY, 123, "author")

        self.assertIsNone(activity)
        permission.assert_called_once()

    def test_malformed_activity_fails_safe(self) -> None:
        with mock.patch(
            "github_reviews.fetch_pull_request_timeline",
            return_value=[
                {
                    "event": "commented",
                    "author_association": "MEMBER",
                    "user": {"login": "maintainer"},
                }
            ],
        ):
            with self.assertRaisesRegex(RuntimeError, "user is invalid"):
                fetch_maintainer_activity(mock.Mock(), REPOSITORY, 123, "author")

    def test_permission_failure_fails_safe(self) -> None:
        events = [
            {
                "event": "commented",
                "author_association": "MEMBER",
                "user": {"login": "maintainer", "type": "User"},
            }
        ]
        with (
            mock.patch(
                "github_reviews.fetch_pull_request_timeline",
                return_value=events,
            ),
            mock.patch(
                "github_reviews.fetch_user_has_triage_permission",
                side_effect=RuntimeError("unavailable"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                fetch_maintainer_activity(mock.Mock(), REPOSITORY, 123, "author")

    def test_candidate_limit_fails_safe(self) -> None:
        events = [
            {
                "event": "commented",
                "author_association": "CONTRIBUTOR",
                "user": {"login": f"user{index}", "type": "User"},
            }
            for index in range(101)
        ]
        with mock.patch(
            "github_reviews.fetch_pull_request_timeline",
            return_value=events,
        ):
            with self.assertRaisesRegex(RuntimeError, "candidate limit"):
                fetch_maintainer_activity(mock.Mock(), REPOSITORY, 123, "author")

    def test_activity_enables_analysis_and_preserves_codepath_owners(self) -> None:
        prepared = triage_input(
            trusted_author=False,
            codepath_owners=["@owner"],
            has_maintainer_activity=True,
        )

        self.assertEqual(
            build_analysis_result(prepared, recommendation_without_additional_owners()),
            AnalysisResult.create(
                is_open_non_draft_pr_against_main=True,
                is_already_handled=False,
                author_has_triage_permission=False,
                has_actionable_linked_issue=False,
                has_maintainer_activity=True,
                ownership_analysis="completed",
                codepath_owners=["@owner"],
                analyzed_head_sha=HEAD_SHA,
                owner_provenance=codepath_provenance(["@owner"]),
            ),
        )


class ApplyAuthorPermissionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.args = argparse.Namespace(
            pr=123,
            repository=REPOSITORY,
            workflow_sha=WORKFLOW_SHA,
            author_login="author",
            run_attempt=1,
            mode="shadow",
            analysis_result_json=AnalysisResult.create(
                is_open_non_draft_pr_against_main=True,
                is_already_handled=False,
                author_has_triage_permission=False,
                has_actionable_linked_issue=False,
                has_maintainer_activity=False,
                ownership_analysis="not_run",
                analyzed_head_sha=HEAD_SHA,
            ).to_json(),
        )
        self.github = mock.Mock()

    def test_apply_records_would_close_for_missing_actionable_issue(self) -> None:
        with (
            mock.patch("apply_triage_decision.require_repository_label"),
            mock.patch("apply_triage_decision.add_labels") as add_labels,
        ):
            outcome = apply_controller_action(self.args, self.github)

        self.assertEqual(outcome, ControllerOutcome("shadow_close"))
        add_labels.assert_called_once_with(
            self.github, REPOSITORY, 123, ("bot-shadow-close",)
        )
        self.github.json.assert_not_called()

    def test_live_mode_closes_for_missing_actionable_issue(self) -> None:
        self.args.mode = "live"
        self.github.json.return_value = {"number": 123, "state": "closed"}
        with (
            mock.patch("apply_triage_decision.require_repository_label"),
            mock.patch("apply_triage_decision.record_bot_close"),
        ):
            outcome = apply_controller_action(self.args, self.github)

        self.assertEqual(outcome, ControllerOutcome("closed"))
        self.github.json.assert_called_once_with(
            f"repos/{REPOSITORY}/pulls/123",
            method="PATCH",
            payload={"state": "closed"},
        )


if __name__ == "__main__":
    unittest.main()
