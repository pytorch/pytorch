from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from assess_intake import (
    fetch_actionable_labelers,
    fetch_actionable_linked_issues,
    fetch_issue_labels,
    fetch_maintainer_activity,
    fetch_maintainer_requested_reviewers,
    fetch_user_has_triage_permission,
    is_already_handled,
    main as intake_main,
    parse_description_claims,
    verify_description_claims,
)
from github_api import PullRequestRef
from schemas import IntakeResult


REPOSITORY = "pytorch/ciforge"


class FakePermissionGitHub:
    def __init__(self, permissions: dict[str, bool], login: str = "author") -> None:
        self.permissions = permissions
        self.login = login
        self.calls: list[str] = []

    def json(self, endpoint: str) -> dict:
        self.calls.append(endpoint)
        return {"user": {"login": self.login, "permissions": self.permissions}}


class GateFactTest(unittest.TestCase):
    def test_already_handled_uses_only_prior_outcome_and_opt_out_labels(self) -> None:
        self.assertFalse(is_already_handled([{"name": "open source"}]))
        self.assertFalse(is_already_handled([]))
        for label in (
            "triaged",
            "bot-triaged",
            "bot-triage-error",
            "no automated triage",
        ):
            with self.subTest(label=label):
                self.assertTrue(is_already_handled([{"name": label}]))
        for label in (
            "bot-closed",
            "bot-shadow-close",
            "bot-shadow-triaged",
            "bot-codeowners-shadow-match",
            "bot-codeowners-shadow-mismatch",
            "bot-codeowners-shadow-inconclusive",
        ):
            with self.subTest(label=label):
                self.assertFalse(is_already_handled([{"name": label}]))
        with self.assertRaisesRegex(RuntimeError, "labels are invalid"):
            is_already_handled([{"bad": "label"}])

    def test_linked_issue_state_ignores_foreign_closing_references(self) -> None:
        github = mock.Mock()
        github.graphql.return_value = {
            "repository": {
                "pullRequest": {
                    "closingIssuesReferences": {
                        "nodes": [
                            {
                                "number": 7,
                                "repository": {"nameWithOwner": REPOSITORY},
                                "labels": {
                                    "nodes": [],
                                    "pageInfo": {"hasNextPage": False},
                                },
                            },
                            {
                                "number": 8,
                                "repository": {"nameWithOwner": "other/repo"},
                                "labels": {
                                    "nodes": [{"name": "actionable"}],
                                    "pageInfo": {"hasNextPage": False},
                                },
                            },
                        ],
                        "pageInfo": {"hasNextPage": False},
                    },
                }
            }
        }

        self.assertEqual(
            fetch_actionable_linked_issues(
                PullRequestRef(github=github, repo=REPOSITORY, number=999)
            ),
            [],
        )
        github.graphql.assert_called_once()

    def test_issue_labels_skip_pull_requests_and_missing_issues(self) -> None:
        github = mock.Mock()
        github.graphql.side_effect = [
            {
                "resource": {
                    "number": 12,
                    "url": f"https://github.com/{REPOSITORY}/issues/12",
                    "labels": {
                        "nodes": [{"name": "Actionable"}],
                        "pageInfo": {"hasNextPage": False},
                    },
                }
            },
            {"resource": {}},
            {"resource": None},
            {
                "resource": {
                    "number": 15,
                    "url": f"https://github.com/{REPOSITORY}/issues/15",
                    "labels": {"nodes": [], "pageInfo": {"hasNextPage": True}},
                }
            },
        ]

        self.assertEqual(
            fetch_issue_labels(github=github, repo=REPOSITORY, number=12),
            frozenset({"actionable"}),
        )
        self.assertIsNone(fetch_issue_labels(github=github, repo=REPOSITORY, number=13))
        self.assertIsNone(fetch_issue_labels(github=github, repo=REPOSITORY, number=14))
        with self.assertRaisesRegex(RuntimeError, "more than 100 labels"):
            fetch_issue_labels(github=github, repo=REPOSITORY, number=15)


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
                    fetch_user_has_triage_permission(
                        github=github, repo=REPOSITORY, login="author"
                    )
                )
                self.assertEqual(
                    github.calls,
                    [f"repos/{REPOSITORY}/collaborators/author/permission"],
                )

    def test_read_only_permission_is_not_trusted(self) -> None:
        github = FakePermissionGitHub(
            {"triage": False, "push": False, "maintain": False, "admin": False}
        )
        self.assertFalse(
            fetch_user_has_triage_permission(
                github=github, repo=REPOSITORY, login="author"
            )
        )

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
        self.assertTrue(
            fetch_user_has_triage_permission(
                github=github, repo=REPOSITORY, login="author"
            )
        )

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
            with (
                self.subTest(response=github.permissions),
                self.assertRaises(RuntimeError),
            ):
                fetch_user_has_triage_permission(
                    github=github, repo=REPOSITORY, login="author"
                )


class MaintainerActivityTest(unittest.TestCase):
    def test_review_comment_and_other_reviewer_request_are_activity(self) -> None:
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
                    "login": "reviewer",
                    "type": "User",
                },
            },
        ]
        with (
            mock.patch(
                "assess_intake.fetch_timeline",
                return_value=events,
            ),
            mock.patch(
                "assess_intake.fetch_user_has_triage_permission",
                return_value=True,
            ) as permission,
        ):
            activity = fetch_maintainer_activity(
                PullRequestRef(github=mock.Mock(), repo=REPOSITORY, number=123),
                author_login="author",
            )

        self.assertEqual(
            activity,
            ("@maintainer", ("comment", "review", "review_request")),
        )
        permission.assert_called_once_with(
            github=mock.ANY, repo=REPOSITORY, login="maintainer"
        )

    def test_label_and_team_review_request_are_activity(self) -> None:
        events = [
            {
                "event": "labeled",
                "actor": {"login": "maintainer", "type": "User"},
                "label": {"name": "module: cuda"},
            },
            {
                "event": "review_requested",
                "actor": {"login": "maintainer", "type": "User"},
                "requested_team": {"slug": "cuda"},
            },
        ]
        with (
            mock.patch(
                "assess_intake.fetch_timeline",
                return_value=events,
            ),
            mock.patch(
                "assess_intake.fetch_user_has_triage_permission",
                return_value=True,
            ),
        ):
            activity = fetch_maintainer_activity(
                PullRequestRef(github=mock.Mock(), repo=REPOSITORY, number=123),
                author_login="author",
            )

        self.assertEqual(activity, ("@maintainer", ("label_change", "review_request")))

    def test_trigger_label_changes_are_not_activity(self) -> None:
        events = [
            {
                "event": event,
                "actor": {"login": "maintainer", "type": "User"},
                "label": {"name": "open source"},
            }
            for event in ("labeled", "unlabeled")
        ]
        with (
            mock.patch(
                "assess_intake.fetch_timeline",
                return_value=events,
            ),
            mock.patch("assess_intake.fetch_user_has_triage_permission") as permission,
        ):
            activity = fetch_maintainer_activity(
                PullRequestRef(github=mock.Mock(), repo=REPOSITORY, number=123),
                author_login="author",
            )

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
                "assess_intake.fetch_timeline",
                return_value=events,
            ),
            mock.patch(
                "assess_intake.fetch_user_has_triage_permission",
                return_value=False,
            ) as permission,
        ):
            activity = fetch_maintainer_activity(
                PullRequestRef(github=mock.Mock(), repo=REPOSITORY, number=123),
                author_login="author",
            )

        self.assertIsNone(activity)
        permission.assert_called_once_with(
            github=mock.ANY, repo=REPOSITORY, login="contributor"
        )

    def test_passive_events_are_ignored(self) -> None:
        events = [
            {
                "event": event,
                "actor": {"login": "maintainer", "type": "User"},
            }
            for event in (
                "cross-referenced",
                "deployed",
                "deployment_environment_changed",
                "mentioned",
                "referenced",
                "subscribed",
                "unsubscribed",
            )
        ]
        with (
            mock.patch(
                "assess_intake.fetch_timeline",
                return_value=events,
            ),
            mock.patch("assess_intake.fetch_user_has_triage_permission") as permission,
        ):
            activity = fetch_maintainer_activity(
                PullRequestRef(github=mock.Mock(), repo=REPOSITORY, number=123),
                author_login="author",
            )

        self.assertIsNone(activity)
        permission.assert_not_called()

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
                "assess_intake.fetch_timeline",
                return_value=events,
            ),
            mock.patch(
                "assess_intake.fetch_user_has_triage_permission",
                return_value=False,
            ) as permission,
        ):
            activity = fetch_maintainer_activity(
                PullRequestRef(github=mock.Mock(), repo=REPOSITORY, number=123),
                author_login="author",
            )

        self.assertIsNone(activity)
        permission.assert_called_once()

    def test_malformed_activity_fails_safe(self) -> None:
        with (
            mock.patch(
                "assess_intake.fetch_timeline",
                return_value=[
                    {
                        "event": "commented",
                        "author_association": "MEMBER",
                        "user": {"login": "maintainer"},
                    }
                ],
            ),
            self.assertRaisesRegex(RuntimeError, "user is invalid"),
        ):
            fetch_maintainer_activity(
                PullRequestRef(github=mock.Mock(), repo=REPOSITORY, number=123),
                author_login="author",
            )

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
                "assess_intake.fetch_timeline",
                return_value=events,
            ),
            mock.patch(
                "assess_intake.fetch_user_has_triage_permission",
                side_effect=RuntimeError("unavailable"),
            ),
            self.assertRaisesRegex(RuntimeError, "unavailable"),
        ):
            fetch_maintainer_activity(
                PullRequestRef(github=mock.Mock(), repo=REPOSITORY, number=123),
                author_login="author",
            )

    def test_candidate_limit_fails_safe(self) -> None:
        events = [
            {
                "event": "commented",
                "author_association": "CONTRIBUTOR",
                "user": {"login": f"user{index}", "type": "User"},
            }
            for index in range(101)
        ]
        with (
            mock.patch(
                "assess_intake.fetch_timeline",
                return_value=events,
            ),
            self.assertRaisesRegex(RuntimeError, "candidate limit"),
        ):
            fetch_maintainer_activity(
                PullRequestRef(github=mock.Mock(), repo=REPOSITORY, number=123),
                author_login="author",
            )


class HandoffReviewerTest(unittest.TestCase):
    def test_requested_reviewers_need_a_distinct_maintainer_requester(self) -> None:
        def request(*, event: str, actor: str, reviewer: str) -> dict:
            return {
                "event": event,
                "actor": {"login": actor, "type": "User"},
                "requested_reviewer": {"login": reviewer, "type": "User"},
            }

        events = [
            request(event="review_requested", actor="maintainer", reviewer="Reviewer"),
            request(event="review_requested", actor="maintainer", reviewer="removed"),
            request(
                event="review_request_removed", actor="maintainer", reviewer="removed"
            ),
            request(event="review_requested", actor="author", reviewer="codeowner"),
            request(event="review_requested", actor="self", reviewer="self"),
            request(event="review_requested", actor="outsider", reviewer="other"),
            request(event="review_requested", actor="maintainer", reviewer="untrusted"),
            {
                "event": "review_requested",
                "actor": {"login": "maintainer", "type": "User"},
                "requested_team": {"slug": "cuda"},
            },
        ]
        with (
            mock.patch(
                "assess_intake.fetch_timeline",
                return_value=events,
            ),
            mock.patch(
                "assess_intake.fetch_user_has_triage_permission",
                side_effect=lambda login, **_: login not in {"outsider", "untrusted"},
            ) as permission,
        ):
            reviewers = fetch_maintainer_requested_reviewers(
                PullRequestRef(github=mock.Mock(), repo=REPOSITORY, number=123),
                author_login="author",
                limit=3,
            )

        self.assertEqual(reviewers, ["Reviewer"])
        self.assertEqual(
            sorted(
                (call.kwargs["login"] for call in permission.call_args_list),
                key=str.casefold,
            ),
            ["maintainer", "outsider", "Reviewer", "untrusted"],
        )

    def test_latest_actionable_labeler_per_issue_is_returned(self) -> None:
        def label(*, event: str, actor: str, name: str = "actionable") -> dict:
            return {
                "event": event,
                "actor": {"login": actor, "type": "User"},
                "label": {"name": name},
            }

        timelines = {
            1: [
                label(event="labeled", actor="first"),
                label(event="labeled", actor="Latest"),
            ],
            2: [
                label(event="labeled", actor="removed"),
                label(event="unlabeled", actor="removed"),
            ],
            3: [label(event="labeled", actor="author")],
            4: [label(event="labeled", actor="other", name="triaged")],
            5: [label(event="labeled", actor="untrusted")],
        }
        with (
            mock.patch(
                "assess_intake.fetch_timeline",
                side_effect=lambda number, **_: timelines[number],
            ),
            mock.patch(
                "assess_intake.fetch_user_has_triage_permission",
                side_effect=lambda login, **_: login != "untrusted",
            ),
        ):
            labelers = fetch_actionable_labelers(
                github=mock.Mock(),
                repo=REPOSITORY,
                issues=[1, 2, 3, 4, 5],
                author_login="author",
                limit=3,
            )

        self.assertEqual(labelers, ["Latest"])


def external_pr(**updates: object) -> dict[str, object]:
    pr = {
        "number": 999,
        "user": {"login": "external-author"},
        "html_url": f"https://github.com/{REPOSITORY}/pull/999",
        "title": "fixture",
        "body": "fixture body",
        "base": {"ref": "main", "repo": {"full_name": REPOSITORY}},
        "head": {"sha": "b" * 40},
        "state": "open",
        "draft": False,
        "labels": [{"name": "open source"}],
    }
    pr.update(updates)
    return pr


def intake_argv(*, output_dir: Path, github_output: Path) -> list[str]:
    return [
        "assess_intake.py",
        "999",
        "--repository",
        REPOSITORY,
        "--workflow-sha",
        "a" * 40,
        "--output-dir",
        str(output_dir),
        "--github-output",
        str(github_output),
    ]


def read_outputs(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines())


class IntakeMainTest(unittest.TestCase):
    def test_intake_continues_after_open_source_label_removal(self) -> None:
        github = mock.Mock()
        github.graphql.return_value = {
            "repository": {
                "pullRequest": {
                    "closingIssuesReferences": {
                        "nodes": [
                            {
                                "number": 7,
                                "repository": {"nameWithOwner": REPOSITORY},
                                "labels": {
                                    "nodes": [{"name": "actionable"}],
                                    "pageInfo": {"hasNextPage": False},
                                },
                            }
                        ],
                        "pageInfo": {"hasNextPage": False},
                    },
                }
            }
        }
        github.json.return_value = external_pr(
            user={"login": "External-Author"},
            body="fixture body\n\ncc @soulitzer",
            labels=[],
        )

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "output"
            github_output = Path(directory) / "github-output"
            with (
                mock.patch.object(
                    sys,
                    "argv",
                    intake_argv(output_dir=output_dir, github_output=github_output),
                ),
                mock.patch("assess_intake.GitHubReader", return_value=github),
                mock.patch(
                    "assess_intake.fetch_user_has_triage_permission", return_value=False
                ),
                mock.patch(
                    "assess_intake.fetch_maintainer_activity", return_value=None
                ),
                mock.patch(
                    "assess_intake.fetch_actionable_labelers", return_value=["Labeler"]
                ) as fetch_labelers,
                mock.patch(
                    "assess_intake.fetch_maintainer_requested_reviewers",
                    return_value=["Requested"],
                ) as fetch_requested,
                mock.patch("builtins.print"),
            ):
                self.assertEqual(intake_main(), 0)
            intake = IntakeResult.from_json((output_dir / "intake.json").read_text())
            outputs = read_outputs(github_output)

        facts = intake.facts
        self.assertEqual(intake.identity.workflow_sha, "a" * 40)
        self.assertEqual(intake.identity.head_sha, "b" * 40)
        self.assertFalse(facts.is_already_handled)
        self.assertTrue(facts.has_actionable_linked_issue)
        self.assertFalse(facts.author_has_triage_permission)
        self.assertFalse(facts.has_maintainer_activity)
        self.assertTrue(facts.is_open_non_draft_pr_against_main)
        self.assertEqual(outputs, {"active": "true"})
        self.assertTrue(facts.passes_intake)
        self.assertEqual(intake.author_login, "External-Author")
        self.assertEqual(intake.body, "fixture body\n\ncc @soulitzer")
        github.json.assert_called_once_with(f"repos/{REPOSITORY}/pulls/999")
        self.assertEqual(github.graphql.call_count, 1)
        fetch_labelers.assert_called_once_with(
            github=github,
            repo=REPOSITORY,
            issues=[7],
            author_login="External-Author",
        )
        fetch_requested.assert_called_once_with(
            PullRequestRef(github=github, repo=REPOSITORY, number=999),
            author_login="External-Author",
        )
        self.assertEqual(facts.actionable_labelers, ("Labeler",))
        self.assertEqual(facts.maintainer_requested_reviewers, ("Requested",))

    def run_intake_for_external_pr(
        self,
        activity: tuple[str, tuple[str, ...]] | None,
        *,
        body: str = "fixture body",
        base_ref: str = "main",
        fetch_permission: mock.Mock | None = None,
        fetch_labels: mock.Mock | None = None,
        fetch_labelers: mock.Mock | None = None,
    ) -> tuple[IntakeResult, dict[str, str], mock.Mock, mock.Mock]:
        github = mock.Mock()
        github.json.return_value = external_pr(
            body=body, base={"ref": base_ref, "repo": {"full_name": REPOSITORY}}
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "output"
            github_output = Path(directory) / "github-output"
            with (
                mock.patch.object(
                    sys,
                    "argv",
                    intake_argv(output_dir=output_dir, github_output=github_output),
                ),
                mock.patch("assess_intake.GitHubReader", return_value=github),
                mock.patch(
                    "assess_intake.fetch_user_has_triage_permission",
                    fetch_permission or mock.Mock(return_value=False),
                ),
                mock.patch(
                    "assess_intake.fetch_issue_labels",
                    fetch_labels or mock.Mock(return_value=frozenset()),
                ),
                mock.patch(
                    "assess_intake.fetch_actionable_linked_issues", return_value=[]
                ),
                mock.patch(
                    "assess_intake.fetch_maintainer_activity", return_value=activity
                ) as fetch_activity,
                mock.patch(
                    "assess_intake.fetch_actionable_labelers",
                    fetch_labelers or mock.Mock(return_value=[]),
                ),
                mock.patch(
                    "assess_intake.fetch_maintainer_requested_reviewers",
                    return_value=[],
                ),
                mock.patch("builtins.print"),
            ):
                self.assertEqual(intake_main(), 0)
            intake = IntakeResult.from_json((output_dir / "intake.json").read_text())
            outputs = read_outputs(github_output)
        return intake, outputs, fetch_activity, github

    def test_intake_does_not_admit_without_admitting_facts(self) -> None:
        fetch_labelers = mock.Mock()
        intake, outputs, _, _ = self.run_intake_for_external_pr(
            None, fetch_labelers=fetch_labelers
        )

        facts = intake.facts
        self.assertFalse(facts.has_actionable_linked_issue)
        self.assertFalse(facts.has_maintainer_activity)
        self.assertFalse(facts.has_supporter)
        self.assertFalse(facts.has_related_actionable_issue)
        self.assertEqual(outputs["active"], "true")
        self.assertFalse(facts.passes_intake)
        fetch_labelers.assert_not_called()

    def test_intake_records_pr_maintainer_activity(self) -> None:
        intake, outputs, fetch_activity, github = self.run_intake_for_external_pr(
            ("@maintainer", ("comment",))
        )

        facts = intake.facts
        self.assertFalse(facts.is_already_handled)
        self.assertTrue(facts.has_maintainer_activity)
        self.assertTrue(facts.is_open_non_draft_pr_against_main)
        self.assertTrue(facts.passes_intake)
        # Only the PR timeline counts; linked issue timelines are never read.
        fetch_activity.assert_called_once_with(
            PullRequestRef(github=github, repo=REPOSITORY, number=999),
            author_login="external-author",
        )

    def test_intake_treats_ghstack_base_as_main(self) -> None:
        intake, outputs, _, _ = self.run_intake_for_external_pr(
            ("@maintainer", ("comment",)), base_ref="gh/External-Author/12/base"
        )

        self.assertTrue(intake.facts.is_open_non_draft_pr_against_main)
        self.assertTrue(intake.facts.passes_intake)
        self.assertEqual(outputs["active"], "true")

    def test_intake_records_inactive_target_without_further_reads(self) -> None:
        cases = (
            (
                "retargeted",
                {"base": {"ref": "release", "repo": {"full_name": REPOSITORY}}},
            ),
            (
                "ghstack head branch",
                {"base": {"ref": "gh/user/1/head", "repo": {"full_name": REPOSITORY}}},
            ),
            ("closed", {"state": "closed"}),
            ("draft", {"draft": True}),
        )
        for name, updates in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                github = mock.Mock()
                github.json.return_value = external_pr(**updates)
                output_dir = Path(directory) / "output"
                github_output = Path(directory) / "github-output"
                with (
                    mock.patch.object(
                        sys,
                        "argv",
                        intake_argv(output_dir=output_dir, github_output=github_output),
                    ),
                    mock.patch("assess_intake.GitHubReader", return_value=github),
                    mock.patch(
                        "assess_intake.fetch_user_has_triage_permission"
                    ) as fetch_permission,
                    mock.patch(
                        "assess_intake.fetch_actionable_linked_issues"
                    ) as fetch_issue,
                    mock.patch(
                        "assess_intake.fetch_maintainer_activity"
                    ) as fetch_activity,
                    mock.patch("builtins.print"),
                ):
                    self.assertEqual(intake_main(), 0)

                intake = IntakeResult.from_json(
                    (output_dir / "intake.json").read_text()
                )
                self.assertEqual(read_outputs(github_output)["active"], "false")
                self.assertFalse((output_dir / "error.json").exists())
                facts = intake.facts
                self.assertFalse(facts.is_open_non_draft_pr_against_main)
                self.assertFalse(facts.author_has_triage_permission)
                self.assertFalse(facts.has_actionable_linked_issue)
                self.assertFalse(facts.has_maintainer_activity)
                fetch_permission.assert_not_called()
                fetch_issue.assert_not_called()
                fetch_activity.assert_not_called()
                github.json.assert_called_once_with(f"repos/{REPOSITORY}/pulls/999")

    def test_intake_rejects_malformed_pr_response(self) -> None:
        github = mock.Mock()
        pr = external_pr()
        del pr["user"]
        github.json.return_value = pr

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "output"
            github_output = Path(directory) / "github-output"
            with (
                mock.patch.object(
                    sys,
                    "argv",
                    intake_argv(output_dir=output_dir, github_output=github_output),
                ),
                mock.patch("assess_intake.GitHubReader", return_value=github),
                mock.patch("builtins.print"),
            ):
                self.assertEqual(intake_main(), 1)

            error = json.loads((output_dir / "error.json").read_text())
            self.assertFalse((output_dir / "intake.json").exists())

        self.assertEqual(error["stage"], "intake")
        self.assertEqual(error["type"], "RuntimeError")


class DescriptionIntakeTest(unittest.TestCase):
    run_intake = IntakeMainTest.run_intake_for_external_pr

    def test_parse_description_claims(self) -> None:
        cases = [
            ("Supported by @Maintainer", ["Maintainer"], []),
            ("supported BY: @alice. PART OF: #12", ["alice"], [12]),
            ("Also supported by @alice, and part of #7.", ["alice"], [7]),
            ("> Supported by @alice", ["alice"], []),
            ("Part of pytorch/ciforge#34", [], [34]),
            ("Part of other/repo#56 and part of #0", [], []),
            ("Part of #12\nPart of #12", [], [12]),
            ("Supported by @alice\nsupported by @ALICE", ["alice"], []),
            ("Unsupported by @alice; counterpart of #12", [], []),
            ("Supported by alice; part of 12", [], []),
            ("Supported by @pytorch/team; part of #12a; part of #12/", [], []),
            ("Cc @alice. Related to #12. Fixes #13", [], []),
            ("Supported by @<!-- login -->", [], []),
            ("<!--\nSupported by @alice\n-->Part of #12", [], [12]),
            ("```\nPart of #12\n```\n~~~\nSupported by @alice", [], []),
            ("`Supported by @alice` `part of #12`", [], []),
            (
                " ".join(f"Supported by @a{i}" for i in range(5)),
                ["a0", "a1", "a2"],
                [],
            ),
            (" ".join(f"Part of #{i}" for i in range(1, 8)), [], [1, 2, 3, 4, 5]),
        ]
        for body, logins, issues in cases:
            with self.subTest(body=body):
                self.assertEqual(
                    parse_description_claims(body=body, repo=REPOSITORY),
                    (logins, issues),
                )

    def test_verified_claims_admit_analysis(self) -> None:
        fetch_permission = mock.Mock(
            side_effect=lambda login, **_: login == "Maintainer"
        )
        fetch_labels = mock.Mock(
            side_effect=lambda number, **_: frozenset(
                {"actionable"} if number == 34 else ()
            )
        )
        intake, outputs, _, github = self.run_intake(
            None,
            body="Supported by @outsider.\nSupported by @Maintainer.\nPart of #33. Part of #34.",
            fetch_permission=fetch_permission,
            fetch_labels=fetch_labels,
        )

        facts = intake.facts
        self.assertTrue(facts.has_supporter)
        self.assertEqual(facts.supporters, ("Maintainer",))
        self.assertTrue(facts.has_related_actionable_issue)
        self.assertTrue(facts.passes_intake)
        self.assertEqual(
            fetch_labels.call_args_list,
            [
                mock.call(github=github, repo=REPOSITORY, number=33),
                mock.call(github=github, repo=REPOSITORY, number=34),
            ],
        )

    def test_each_verified_claim_admits_alone(self) -> None:
        for body, fact in (
            ("Supported by @Maintainer", "has_supporter"),
            ("Part of #34", "has_related_actionable_issue"),
        ):
            with self.subTest(body=body):
                intake, outputs, _, _ = self.run_intake(
                    None,
                    body=body,
                    fetch_permission=mock.Mock(
                        side_effect=lambda login, **_: login == "Maintainer"
                    ),
                    fetch_labels=mock.Mock(return_value=frozenset({"actionable"})),
                )
                facts = intake.facts
                self.assertTrue(getattr(facts, fact))
                self.assertTrue(facts.passes_intake)

    def test_unverified_claims_do_not_admit(self) -> None:
        def permission(*, github: object, repo: str, login: str) -> bool:
            if login == "ghost":
                raise RuntimeError("gh failed: Not Found (HTTP 404)")
            return False

        fetch_permission = mock.Mock(side_effect=permission)
        intake, outputs, _, github = self.run_intake(
            None,
            body="Supported by @outsider, supported by @ghost, "
            "supported by @pytorchbot. Part of #35. Part of #36.",
            fetch_permission=fetch_permission,
            fetch_labels=mock.Mock(side_effect=[frozenset({"triaged"}), None]),
        )

        facts = intake.facts
        self.assertFalse(facts.has_supporter)
        self.assertEqual(facts.supporters, ())
        self.assertFalse(facts.has_related_actionable_issue)
        self.assertEqual(outputs["active"], "true")
        self.assertFalse(facts.passes_intake)
        # The author check comes first; bots are never checked.
        self.assertEqual(
            [call.kwargs["login"] for call in fetch_permission.call_args_list],
            ["external-author", "outsider", "ghost"],
        )

    def test_verification_failure_is_not_an_unverified_claim(self) -> None:
        with (
            mock.patch(
                "assess_intake.fetch_user_has_triage_permission",
                side_effect=RuntimeError("gh failed: Server Error (HTTP 502)"),
            ),
            self.assertRaisesRegex(RuntimeError, "HTTP 502"),
        ):
            verify_description_claims(
                github=mock.Mock(), repo=REPOSITORY, logins=["Maintainer"], issues=[]
            )

    def test_claims_are_gathered_even_when_already_admitted(self) -> None:
        fetch_permission = mock.Mock(return_value=False)
        fetch_labels = mock.Mock(return_value=frozenset({"actionable"}))
        intake, outputs, _, _ = self.run_intake(
            ("@maintainer", ("comment",)),
            body="Supported by @Maintainer. Part of #34.",
            fetch_permission=fetch_permission,
            fetch_labels=fetch_labels,
        )

        facts = intake.facts
        self.assertTrue(facts.has_maintainer_activity)
        self.assertFalse(facts.has_supporter)
        self.assertTrue(facts.has_related_actionable_issue)
        self.assertTrue(facts.passes_intake)
        self.assertEqual(fetch_permission.call_count, 2)
        fetch_labels.assert_called_once()


if __name__ == "__main__":
    unittest.main()
