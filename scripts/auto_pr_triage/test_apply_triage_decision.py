from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock
from urllib.parse import unquote

from apply_triage_decision import (
    ALLOWED_SHADOW_LABELS,
    BOT_CLOSED_COMMENT,
    BOT_SHADOW_CLOSE_LABEL,
    BOT_SHADOW_TRIAGED_LABEL,
    BOT_TRIAGE_ERROR_LABEL,
    CODEOWNERS_SHADOW_LABELS,
    ControllerOutcome,
    GitHubClient,
    apply_controller_action,
    main,
)
from github_reviews import stable_fallback_member
from schemas import AnalysisResult


WORKFLOW_SHA = "a" * 40
HEAD_SHA = "b" * 40
REPOSITORY = "pytorch/ciforge"


def ownership_config() -> dict[str, Any]:
    return {
        "extra_ownership_metadata": {
            "owners": {
                "autograd": "Owns autograd behavior.",
                "compiler": "Owns compiler behavior.",
            }
        },
        "codepath_owners": {
            "rules": [
                {"pattern": "/torch/autograd/", "owners": ["autograd"]},
                {"pattern": "/torch/compiler/", "owners": ["compiler"]},
            ]
        },
        "team_members": {
            "members": {
                "autograd": ["@soulitzer"],
                "compiler": ["@reviewer", "@extra"],
            }
        },
    }


def analysis_result_json(
    *,
    is_open_non_draft_pr_against_main: bool = True,
    is_already_handled: bool = False,
    author_has_triage_permission: bool = False,
    has_actionable_linked_issue: bool = True,
    has_maintainer_activity: bool = False,
    ownership_analysis: str = "completed",
    codepath_owners: tuple[str, ...] = ("@codepath-owner",),
    additional_owners: tuple[str, ...] = (),
    owner_provenance: dict[str, dict[str, Any]] | None = None,
    owner_provenance_truncated: bool = False,
    has_uncovered_concerns: bool = False,
) -> str:
    if owner_provenance is None:
        owner_provenance = {}
        if not owner_provenance_truncated:
            owner_provenance.update(
                {
                    owner: {
                        "source": "codepath",
                        "files": ["torch/file.py"],
                        "total_file_count": 1,
                        "llm_justification": None,
                    }
                    for owner in codepath_owners
                }
            )
            owner_provenance.update(
                {
                    owner: {
                        "source": "semantic",
                        "files": ["torch/semantic.py"],
                        "total_file_count": 1,
                        "llm_justification": {
                            "owned_concern": f"{owner} owns this changed behavior.",
                            "rationale": [
                                "The changed behavior falls within this ownership area.",
                                "The configured description names the affected contract.",
                                "A review from this team covers the semantic concern.",
                            ],
                            "evidence": [
                                {
                                    "file": "torch/semantic.py",
                                    "diff_excerpt": "+new semantic behavior",
                                    "relevance": "This line implements the owned behavior.",
                                }
                            ],
                        },
                    }
                    for owner in additional_owners
                }
            )
    return AnalysisResult.create(
        is_open_non_draft_pr_against_main=is_open_non_draft_pr_against_main,
        is_already_handled=is_already_handled,
        author_has_triage_permission=author_has_triage_permission,
        has_actionable_linked_issue=has_actionable_linked_issue,
        has_maintainer_activity=has_maintainer_activity,
        ownership_analysis=ownership_analysis,
        codepath_owners=codepath_owners,
        additional_owners=additional_owners,
        analyzed_head_sha=HEAD_SHA,
        owner_provenance=owner_provenance,
        owner_provenance_truncated=owner_provenance_truncated,
        has_uncovered_concerns=has_uncovered_concerns,
    ).to_json()


def controller_args(**overrides: Any) -> argparse.Namespace:
    values = {
        "pr": 123,
        "repository": REPOSITORY,
        "mode": "live",
        "workflow_sha": WORKFLOW_SHA,
        "author_login": "external-author",
        "run_attempt": 1,
        "analysis_result_json": analysis_result_json(),
        "github_step_summary": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def run_apply(
    github: FakeGitHub,
    *,
    codepath_owners: tuple[str, ...] | None = None,
    additional_owners: tuple[str, ...] = (),
    ownership_analysis: str = "completed",
    native_codeowners_shadow: bool = True,
    mode: str = "live",
    run_attempt: int = 1,
    team_members: dict[str, Any] | None = None,
    github_step_summary: Path | None = None,
    owner_provenance: dict[str, dict[str, Any]] | None = None,
    owner_provenance_truncated: bool = False,
    has_uncovered_concerns: bool = False,
) -> ControllerOutcome:
    if codepath_owners is None:
        codepath_owners = ("@codepath-owner",)
    args = controller_args(
        mode=mode,
        run_attempt=run_attempt,
        github_step_summary=github_step_summary,
        analysis_result_json=analysis_result_json(
            ownership_analysis=ownership_analysis,
            codepath_owners=codepath_owners,
            additional_owners=additional_owners,
            owner_provenance=owner_provenance,
            owner_provenance_truncated=owner_provenance_truncated,
            has_uncovered_concerns=has_uncovered_concerns,
        )
    )
    if native_codeowners_shadow and github.native_codeowners is None:
        github.native_codeowners = [
            owner
            for owner in codepath_owners
            if owner.startswith("@") and owner.casefold() != "@external-author"
        ]
    with mock.patch(
        "apply_triage_decision.load_team_members",
        return_value=team_members or ownership_config()["team_members"],
    ), mock.patch(
        "apply_triage_decision.NATIVE_CODEOWNERS_SHADOW",
        native_codeowners_shadow,
    ):
        return apply_controller_action(args, github)


def run_close(
    github: FakeGitHub,
    *,
    run_attempt: int = 1,
    mode: str = "live",
) -> ControllerOutcome:
    args = controller_args(
        mode=mode,
        run_attempt=run_attempt,
        analysis_result_json=analysis_result_json(
            has_actionable_linked_issue=False,
            ownership_analysis="not_run",
            codepath_owners=(),
        ),
    )
    return apply_controller_action(args, github)


def run_without_owners(
    github: FakeGitHub,
    *,
    ownership_analysis: str = "completed",
    native_codeowners_shadow: bool = True,
    mode: str = "live",
    has_uncovered_concerns: bool = False,
) -> ControllerOutcome:
    args = controller_args(
        mode=mode,
        analysis_result_json=analysis_result_json(
            ownership_analysis=ownership_analysis,
            codepath_owners=(),
            has_uncovered_concerns=has_uncovered_concerns,
        ),
    )
    with mock.patch(
        "apply_triage_decision.load_team_members",
        return_value=ownership_config()["team_members"],
    ), mock.patch(
        "apply_triage_decision.NATIVE_CODEOWNERS_SHADOW",
        native_codeowners_shadow,
    ):
        return apply_controller_action(args, github)


class FakeGitHub:
    def __init__(
        self,
        *,
        requested_users: list[str] | None = None,
        requested_teams: list[str] | None = None,
        labels: list[str] | None = None,
        submitted_users: list[str] | None = None,
        unattributed_submitted_review: bool = False,
        submitted_user_state: str = "APPROVED",
        actionable_issue: bool = False,
        actionable_actor: str = "soulitzer",
        author_has_triage_permission: bool = False,
        permission_error: Exception | None = None,
        unavailable_labels: list[str] | None = None,
        native_codeowners: list[str] | None = None,
        codeowner_error: Exception | None = None,
        round_robin_events: list[dict[str, Any]] | None = None,
        round_robin_timelines: dict[int, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.pr = {
            "number": 123,
            "base": {
                "repo": {"full_name": "pytorch/ciforge"},
                "ref": "main",
                "sha": WORKFLOW_SHA,
            },
            "state": "open",
            "draft": False,
            "head": {"sha": HEAD_SHA},
            "user": {"login": "external-author"},
            "title": "fixture",
            "body": "fixture body",
            "labels": [{"name": label} for label in labels or []],
        }
        self.requested_users = set(requested_users or [])
        self.requested_teams = set(requested_teams or [])
        self.submitted_users = set(submitted_users or [])
        self.unattributed_submitted_review = unattributed_submitted_review
        self.submitted_user_state = submitted_user_state
        self.actionable_issue = actionable_issue
        self.actionable_actor = actionable_actor
        self.author_has_triage_permission = author_has_triage_permission
        self.permission_error = permission_error
        self.unavailable_labels = set(unavailable_labels or [])
        self.native_codeowners = native_codeowners
        self.codeowner_error = codeowner_error
        self.round_robin_events = round_robin_events or []
        self.round_robin_timelines = round_robin_timelines or {}
        self.review_checks = 0
        self.actionable_checks = 0
        self.permission_checks = 0
        self.pr_fetches = 0
        self.live_reads: list[str] = []
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def json(
        self,
        endpoint: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
    ) -> Any:
        self.calls.append((method, endpoint, payload))
        if endpoint == "repos/pytorch/ciforge/pulls/123" and method == "GET":
            self.pr_fetches += 1
            self.live_reads.append("pr")
            return copy.deepcopy(self.pr)
        if endpoint == "repos/pytorch/ciforge/pulls/123" and method == "PATCH":
            if payload != {"state": "closed"}:
                raise AssertionError("unexpected close payload")
            self.pr["state"] = "closed"
            return copy.deepcopy(self.pr)
        if endpoint.startswith("repos/pytorch/ciforge/labels/") and method == "GET":
            label = unquote(endpoint.rsplit("/", 1)[-1])
            return {} if label in self.unavailable_labels else {"name": label}
        if endpoint.startswith("repos/pytorch/ciforge/issues/events?"):
            return copy.deepcopy(self.round_robin_events)
        if (
            endpoint.startswith("repos/pytorch/ciforge/issues/")
            and "/timeline?" in endpoint
        ):
            number = int(endpoint.split("/issues/", 1)[1].split("/", 1)[0])
            return copy.deepcopy(self.round_robin_timelines.get(number, []))
        if endpoint == "repos/pytorch/ciforge/pulls/123/requested_reviewers":
            if method == "POST":
                if (
                    not isinstance(payload, dict)
                    or not payload
                    or not set(payload) <= {"reviewers", "team_reviewers"}
                ):
                    raise AssertionError("unexpected reviewer payload")
                self.requested_users.update(payload.get("reviewers", []))
                self.requested_teams.update(payload.get("team_reviewers", []))
            else:
                self.live_reads.append("requested")
            return {
                "users": [
                    {"login": login} for login in sorted(self.requested_users)
                ],
                "teams": [
                    {"slug": slug} for slug in sorted(self.requested_teams)
                ],
            }
        if endpoint == "repos/pytorch/ciforge/issues/123/labels" and method == "POST":
            if not isinstance(payload, dict) or not isinstance(
                payload.get("labels"), list
            ):
                raise AssertionError("unexpected label payload")
            existing = {label["name"] for label in self.pr["labels"]}
            self.pr["labels"].extend(
                {"name": name}
                for name in payload["labels"]
                if name not in existing
            )
            return copy.deepcopy(self.pr["labels"])
        if endpoint == "repos/pytorch/ciforge/issues/123/comments" and method == "POST":
            if payload != {"body": BOT_CLOSED_COMMENT}:
                raise AssertionError("unexpected comment payload")
            return {"id": 1, "body": BOT_CLOSED_COMMENT}
        if endpoint.endswith("/collaborators/external-author/permission"):
            self.permission_checks += 1
            self.live_reads.append("permission")
            if self.permission_error is not None:
                raise self.permission_error
            return {
                "user": {
                    "login": "external-author",
                    "permissions": {
                        "triage": self.author_has_triage_permission,
                        "push": False,
                        "maintain": False,
                        "admin": False,
                    },
                }
            }
        raise AssertionError(f"unexpected request: {method} {endpoint}")

    def graphql(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        if variables["number"] != 123:
            raise AssertionError("unexpected pull request number")
        if "reviewRequests" in query:
            self.live_reads.append("codeowners_shadow")
            if self.codeowner_error is not None:
                raise self.codeowner_error
            nodes = []
            for owner in self.native_codeowners or []:
                if "/" in owner:
                    reviewer = {
                        "__typename": "Team",
                        "slug": owner.split("/", 1)[1],
                    }
                else:
                    reviewer = {
                        "__typename": "User",
                        "login": owner.removeprefix("@"),
                    }
                nodes.append(
                    {"asCodeOwner": True, "requestedReviewer": reviewer}
                )
            return {
                "repository": {
                    "pullRequest": {
                        "reviewRequests": {
                            "nodes": nodes,
                            "pageInfo": {
                                "endCursor": None,
                                "hasNextPage": False,
                            },
                        }
                    }
                }
            }
        if "closingIssuesReferences" in query:
            self.actionable_checks += 1
            self.live_reads.append("actionable")
            nodes: list[dict[str, Any]] = []
            if self.actionable_issue:
                nodes.append(
                    {
                        "repository": {"nameWithOwner": "pytorch/ciforge"},
                        "labels": {
                            "nodes": [
                                {"id": "actionable-label", "name": "actionable"}
                            ],
                            "pageInfo": {"hasNextPage": False},
                        },
                        "timelineItems": {
                            "nodes": [
                                {
                                    "__typename": "LabeledEvent",
                                    "actor": {
                                        "__typename": "User",
                                        "login": self.actionable_actor,
                                    },
                                    "label": {"id": "actionable-label"},
                                }
                            ],
                            "pageInfo": {"hasPreviousPage": False},
                        },
                    }
                )
            return {
                "repository": {
                    "pullRequest": {
                        "closingIssuesReferences": {
                            "nodes": nodes,
                            "pageInfo": {"hasNextPage": False},
                        }
                    }
                }
            }

        self.review_checks += 1
        self.live_reads.append("submitted")
        nodes = [
            {
                "author": {"login": login},
                "state": self.submitted_user_state,
            }
            for login in sorted(self.submitted_users)
        ]
        if self.unattributed_submitted_review:
            nodes.append({"author": None, "state": self.submitted_user_state})
        return {
            "repository": {
                "pullRequest": {
                    "reviews": {
                        "nodes": nodes,
                        "pageInfo": {
                            "endCursor": None,
                            "hasNextPage": False,
                        },
                    }
                }
            }
        }


def mutations(github: FakeGitHub) -> list[tuple[str, str, dict[str, Any] | None]]:
    return [call for call in github.calls if call[0] in {"POST", "PATCH"}]


def printed_record(output: mock.Mock, label: str) -> dict[str, Any]:
    prefix = f"{label}:\n"
    return next(
        json.loads(call.args[0].removeprefix(prefix))
        for call in output.call_args_list
        if call.args[0].startswith(prefix)
    )


def printed_plan(output: mock.Mock) -> dict[str, Any]:
    return printed_record(output, "Auto PR Triage plan")


def printed_reviewer_routing(output: mock.Mock) -> str:
    prefix = "Auto PR Triage reviewer routing:\n"
    return next(
        call.args[0].removeprefix(prefix)
        for call in output.call_args_list
        if call.args[0].startswith(prefix)
    )


class GitHubClientTest(unittest.TestCase):
    def test_graphql_uses_the_shared_json_request_path(self) -> None:
        github = GitHubClient("live", REPOSITORY, 123)
        variables = {"owner": "pytorch"}
        with mock.patch.object(
            github, "json", return_value={"data": {"ok": True}}
        ) as request:
            self.assertEqual(github.graphql("query", variables), {"ok": True})

        request.assert_called_once_with(
            "graphql",
            method="POST",
            payload={"query": "query", "variables": variables},
        )

    def test_shadow_client_rejects_live_mutations_before_execution(self) -> None:
        github = GitHubClient("shadow", REPOSITORY, 123)
        requests = (
            (
                "repos/pytorch/ciforge/pulls/123",
                "PATCH",
                {"state": "closed"},
            ),
            (
                "repos/pytorch/ciforge/issues/123/comments",
                "POST",
                {"body": BOT_CLOSED_COMMENT},
            ),
            (
                "repos/pytorch/ciforge/pulls/123/requested_reviewers",
                "POST",
                {"reviewers": ["soulitzer"]},
            ),
            (
                "repos/pytorch/ciforge/issues/123/labels",
                "POST",
                {"labels": ["triaged"]},
            ),
        )

        with mock.patch("apply_triage_decision.subprocess.run") as run:
            for endpoint, method, payload in requests:
                with self.subTest(endpoint=endpoint), self.assertRaisesRegex(
                    ValueError, "shadow-mode boundary"
                ):
                    github.json(endpoint, method=method, payload=payload)
        run.assert_not_called()

    def test_shadow_client_allows_fixed_diagnostic_labels(self) -> None:
        self.assertEqual(
            ALLOWED_SHADOW_LABELS,
            frozenset(
                {
                    BOT_TRIAGE_ERROR_LABEL,
                    BOT_SHADOW_CLOSE_LABEL,
                    BOT_SHADOW_TRIAGED_LABEL,
                    *CODEOWNERS_SHADOW_LABELS.values(),
                }
            ),
        )
        github = GitHubClient("shadow", REPOSITORY, 123)
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="[]", stderr=""
        )
        with mock.patch(
            "apply_triage_decision.subprocess.run", return_value=completed
        ) as run:
            self.assertEqual(
                github.json(
                    "repos/pytorch/ciforge/issues/123/labels",
                    method="POST",
                    payload={"labels": [BOT_SHADOW_CLOSE_LABEL]},
                ),
                [],
            )
        run.assert_called_once()

    def test_shadow_client_scopes_label_writes_to_repository_and_pr(self) -> None:
        github = GitHubClient("shadow", REPOSITORY, 123)
        payload = {"labels": [BOT_SHADOW_CLOSE_LABEL]}

        with mock.patch("apply_triage_decision.subprocess.run") as run:
            for endpoint in (
                "repos/pytorch/pytorch/issues/123/labels",
                "repos/pytorch/ciforge/issues/124/labels",
            ):
                with self.subTest(endpoint=endpoint), self.assertRaisesRegex(
                    ValueError, "shadow-mode boundary"
                ):
                    github.json(endpoint, method="POST", payload=payload)
        run.assert_not_called()

    def test_shadow_client_rejects_write_disguised_as_a_read(self) -> None:
        github = GitHubClient("shadow", REPOSITORY, 123)
        with mock.patch("apply_triage_decision.subprocess.run") as run:
            with self.assertRaisesRegex(ValueError, "cannot carry a payload"):
                github.json(
                    "repos/pytorch/ciforge/issues/123/comments",
                    payload={"body": "comment"},
                )
            with self.assertRaisesRegex(ValueError, "read-only query"):
                github.graphql("mutation { viewer { login } }", {})
        run.assert_not_called()

    def test_live_client_retains_the_existing_mutation_surface(self) -> None:
        github = GitHubClient("live", REPOSITORY, 123)
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout='{"state":"closed"}', stderr=""
        )
        with mock.patch(
            "apply_triage_decision.subprocess.run", return_value=completed
        ) as run:
            self.assertEqual(
                github.json(
                    "repos/pytorch/ciforge/pulls/123",
                    method="PATCH",
                    payload={"state": "closed"},
                ),
                {"state": "closed"},
            )
        run.assert_called_once()

    def test_client_rejects_unknown_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "mode"):
            GitHubClient("invalid", REPOSITORY, 123)


class ControllerInputTest(unittest.TestCase):
    def test_rejects_foreign_codepath_team_before_github_io(self) -> None:
        github = FakeGitHub()
        args = controller_args(
            analysis_result_json=analysis_result_json(
                codepath_owners=("@other/team",)
            )
        )

        with self.assertRaisesRegex(ValueError, "foreign codepath owner team"):
            apply_controller_action(args, github)

        self.assertEqual(github.calls, [])

    def test_inactive_target_result_is_a_read_free_noop(self) -> None:
        github = FakeGitHub()
        args = controller_args(
            analysis_result_json=analysis_result_json(
                is_open_non_draft_pr_against_main=False,
                has_actionable_linked_issue=False,
                ownership_analysis="not_run",
                codepath_owners=(),
            )
        )

        self.assertEqual(
            apply_controller_action(args, github),
            ControllerOutcome("kept_open"),
        )
        self.assertEqual(github.calls, [])

    def test_already_handled_result_is_a_read_free_noop(self) -> None:
        github = FakeGitHub()
        args = controller_args(
            analysis_result_json=analysis_result_json(
                is_already_handled=True,
                has_actionable_linked_issue=False,
                ownership_analysis="not_run",
                codepath_owners=(),
            )
        )

        self.assertEqual(
            apply_controller_action(args, github),
            ControllerOutcome("kept_open"),
        )
        self.assertEqual(github.calls, [])

    def test_incomplete_result_without_codepath_owners_is_labeled(self) -> None:
        github = FakeGitHub()

        self.assertEqual(
            run_without_owners(github, ownership_analysis="incomplete"),
            ControllerOutcome("incomplete"),
        )
        self.assertEqual(
            mutations(github),
            [
                (
                    "POST",
                    "repos/pytorch/ciforge/issues/123/labels",
                    {
                        "labels": [
                            BOT_TRIAGE_ERROR_LABEL,
                            CODEOWNERS_SHADOW_LABELS["match"],
                        ]
                    },
                )
            ],
        )

    def test_rejects_identity_mismatch(self) -> None:
        for args in (
            controller_args(workflow_sha="invalid"),
            controller_args(author_login="invalid login"),
            controller_args(run_attempt=0),
        ):
            github = FakeGitHub()
            with self.subTest(args=args), self.assertRaisesRegex(
                ValueError, "invalid pull request identity"
            ):
                apply_controller_action(args, github)
            self.assertEqual(github.calls, [])

    def test_malformed_analysis_result_fails_before_github_io(self) -> None:
        malformed = (
            "not-json",
            '{"schema_version":1}',
            json.dumps(
                {
                    "is_already_handled": False,
                    "author_has_triage_permission": False,
                    "has_actionable_linked_issue": False,
                    "has_maintainer_activity": False,
                    "ownership_analysis": "not_run",
                    "codepath_owners": ["@codepath-owner"],
                    "additional_owners": [],
                }
            ),
        )
        for raw in malformed:
            github = FakeGitHub()
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                apply_controller_action(
                    controller_args(analysis_result_json=raw), github
                )
            self.assertEqual(github.calls, [])

    def test_rejects_unknown_mode_before_github_io(self) -> None:
        github = FakeGitHub()

        with self.assertRaisesRegex(ValueError, "mode"):
            apply_controller_action(controller_args(mode="invalid"), github)

        self.assertEqual(github.calls, [])

    def test_bot_author_login_is_valid(self) -> None:
        github = FakeGitHub()
        args = controller_args(
            author_login="dependabot[bot]",
            analysis_result_json=analysis_result_json(
                is_open_non_draft_pr_against_main=False,
                ownership_analysis="not_run",
                codepath_owners=(),
            ),
        )

        self.assertEqual(
            apply_controller_action(args, github),
            ControllerOutcome("kept_open"),
        )
        self.assertEqual(github.calls, [])


class CodeownersShadowTest(unittest.TestCase):
    def test_matching_native_requests_are_labeled_without_being_requested(self) -> None:
        github = FakeGitHub()

        result = run_apply(github, native_codeowners_shadow=True)

        self.assertEqual(result, ControllerOutcome("triaged"))
        self.assertFalse(
            any(call[1].endswith("/requested_reviewers") for call in mutations(github))
        )
        self.assertEqual(
            mutations(github),
            [
                (
                    "POST",
                    "repos/pytorch/ciforge/issues/123/labels",
                    {
                        "labels": [
                            "triaged",
                            "bot-triaged",
                            CODEOWNERS_SHADOW_LABELS["match"],
                        ]
                    },
                )
            ],
        )

    def test_different_native_requests_get_mismatch_label(self) -> None:
        github = FakeGitHub(native_codeowners=[])

        with mock.patch("builtins.print") as output:
            result = run_apply(github, native_codeowners_shadow=True)

        self.assertEqual(result, ControllerOutcome("triaged"))
        self.assertEqual(
            printed_record(output, "Auto PR Triage CODEOWNERS shadow"),
            {
                "error": None,
                "expected": ["@codepath-owner"],
                "missing_from_github": ["@codepath-owner"],
                "observed": [],
                "oracle": "active_review_requests_as_code_owner",
                "status": "mismatch",
                "unexpected_from_github": [],
                "workflow_sha": WORKFLOW_SHA,
            },
        )
        label_post = mutations(github)[0]
        self.assertEqual(
            label_post[2],
            {
                "labels": [
                    "triaged",
                    "bot-triaged",
                    CODEOWNERS_SHADOW_LABELS["mismatch"],
                ]
            },
        )

    def test_inconclusive_shadow_does_not_block_semantic_addition(self) -> None:
        github = FakeGitHub(codeowner_error=RuntimeError("unavailable"))

        result = run_apply(
            github,
            additional_owners=("autograd",),
            native_codeowners_shadow=True,
        )

        self.assertEqual(result, ControllerOutcome("triaged", 1, 1))
        self.assertEqual(mutations(github)[0][2], {"reviewers": ["soulitzer"]})
        self.assertEqual(
            mutations(github)[1][2],
            {
                "labels": [
                    "triaged",
                    "bot-triaged",
                    CODEOWNERS_SHADOW_LABELS["inconclusive"],
                    "owner: autograd",
                ]
            },
        )

    def test_shadow_never_requests_native_user_or_team(self) -> None:
        github = FakeGitHub()

        result = run_apply(
            github,
            codepath_owners=("@codepath-owner", "@pytorch/compiler"),
            native_codeowners_shadow=True,
        )

        self.assertEqual(result, ControllerOutcome("triaged"))
        self.assertFalse(
            any(call[1].endswith("/requested_reviewers") for call in mutations(github))
        )

    def test_parser_only_owner_does_not_suppress_semantic_addition(self) -> None:
        github = FakeGitHub(native_codeowners=[])
        result = run_apply(
            github,
            codepath_owners=("@pytorch/autograd",),
            additional_owners=("autograd",),
            native_codeowners_shadow=True,
        )

        self.assertEqual(result, ControllerOutcome("triaged", 1, 1))
        self.assertEqual(mutations(github)[0][2], {"reviewers": ["soulitzer"]})

    def test_native_team_handle_does_not_suppress_semantic_addition(self) -> None:
        github = FakeGitHub(native_codeowners=["@pytorch/autograd"])

        result = run_apply(
            github,
            codepath_owners=(),
            additional_owners=("autograd",),
            native_codeowners_shadow=True,
        )

        self.assertEqual(result, ControllerOutcome("triaged", 1, 1))
        self.assertEqual(mutations(github)[0][2], {"reviewers": ["soulitzer"]})

    def test_shadow_excludes_the_author_from_expected_requests(self) -> None:
        github = FakeGitHub(native_codeowners=[])

        result = run_apply(
            github,
            codepath_owners=("@external-author",),
            native_codeowners_shadow=True,
        )

        self.assertEqual(result, ControllerOutcome("triaged"))
        self.assertIn(
            CODEOWNERS_SHADOW_LABELS["match"],
            mutations(github)[0][2]["labels"],
        )

    def test_missing_shadow_label_fails_before_mutation(self) -> None:
        github = FakeGitHub(
            unavailable_labels=[CODEOWNERS_SHADOW_LABELS["match"]]
        )

        with self.assertRaisesRegex(RuntimeError, "required repository label"):
            run_apply(github, native_codeowners_shadow=True)

        self.assertEqual(mutations(github), [])


class RolloutShadowModeTest(unittest.TestCase):
    def test_close_decision_only_records_shadow_close(self) -> None:
        github = FakeGitHub(labels=["open source"])

        result = run_close(github, mode="shadow")

        self.assertEqual(result, ControllerOutcome("shadow_close"))
        self.assertEqual(
            mutations(github),
            [
                (
                    "POST",
                    "repos/pytorch/ciforge/issues/123/labels",
                    {"labels": [BOT_SHADOW_CLOSE_LABEL]},
                )
            ],
        )
        self.assertEqual(github.pr["state"], "open")

    def test_close_decision_on_rerun_matches_live_noop(self) -> None:
        github = FakeGitHub(labels=["open source"])

        result = run_close(github, mode="shadow", run_attempt=2)

        self.assertEqual(result, ControllerOutcome("kept_open"))
        self.assertEqual(github.calls, [])

    def test_triage_decision_only_records_shadow_diagnostics(self) -> None:
        github = FakeGitHub()

        result = run_apply(github, mode="shadow")

        self.assertEqual(result, ControllerOutcome("shadow_triaged"))
        self.assertEqual(
            mutations(github),
            [
                (
                    "POST",
                    "repos/pytorch/ciforge/issues/123/labels",
                    {
                        "labels": [
                            BOT_SHADOW_TRIAGED_LABEL,
                            CODEOWNERS_SHADOW_LABELS["match"],
                        ]
                    },
                )
            ],
        )

    def test_uncovered_concerns_route_found_owners_without_triaging(self) -> None:
        for mode in ("shadow", "live"):
            github = FakeGitHub()
            with self.subTest(mode=mode), mock.patch("builtins.print") as output:
                result = run_apply(
                    github,
                    codepath_owners=(),
                    additional_owners=("autograd",),
                    has_uncovered_concerns=True,
                    mode=mode,
                )

            expected = (
                ControllerOutcome("routed_untriaged")
                if mode == "shadow"
                else ControllerOutcome("routed_untriaged", 1, 1)
            )
            self.assertEqual(result, expected)
            plan = printed_plan(output)
            self.assertEqual(plan["decision"], "routed_untriaged")
            self.assertTrue(plan["has_uncovered_concerns"])
            self.assertEqual(plan["planned_reviewer_requests"], ["@soulitzer"])
            if mode == "live":
                self.assertEqual(
                    mutations(github),
                    [
                        (
                            "POST",
                            "repos/pytorch/ciforge/pulls/123/requested_reviewers",
                            {"reviewers": ["soulitzer"]},
                        ),
                        (
                            "POST",
                            "repos/pytorch/ciforge/issues/123/labels",
                            {
                                "labels": [
                                    CODEOWNERS_SHADOW_LABELS["match"],
                                    "owner: autograd",
                                ]
                            },
                        ),
                    ],
                )
            else:
                self.assertEqual(
                    mutations(github),
                    [
                        (
                            "POST",
                            "repos/pytorch/ciforge/issues/123/labels",
                            {"labels": [CODEOWNERS_SHADOW_LABELS["match"]]},
                        )
                    ],
                )

    def test_submitted_configured_reviewer_completes_uncovered_handoff(self) -> None:
        for mode, expected in (
            ("shadow", ControllerOutcome("shadow_triaged")),
            ("live", ControllerOutcome("triaged", 1, 1)),
        ):
            github = FakeGitHub(submitted_users=["reviewer"])
            with self.subTest(mode=mode), mock.patch("builtins.print") as output:
                result = run_apply(
                    github,
                    codepath_owners=(),
                    additional_owners=("autograd",),
                    has_uncovered_concerns=True,
                    mode=mode,
                )

            self.assertEqual(result, expected)
            plan = printed_plan(output)
            self.assertEqual(plan["decision"], "triage")
            self.assertEqual(plan["submitted_handoff"], "@reviewer")

    def test_uncovered_handoff_rejects_author_and_unconfigured_reviewer(self) -> None:
        team_members = copy.deepcopy(ownership_config()["team_members"])
        team_members["members"]["autograd"].append("@external-author")
        for submitted_user in ("external-author", "outsider"):
            github = FakeGitHub(submitted_users=[submitted_user])
            with self.subTest(submitted_user=submitted_user):
                result = run_apply(
                    github,
                    codepath_owners=(),
                    additional_owners=("autograd",),
                    has_uncovered_concerns=True,
                    team_members=team_members,
                )

            self.assertEqual(
                result,
                ControllerOutcome("routed_untriaged", 1, 1),
            )

    def test_uncovered_concern_review_state_failure_stays_incomplete(self) -> None:
        class UnavailableReviewsGitHub(FakeGitHub):
            def graphql(
                self, query: str, variables: dict[str, Any]
            ) -> dict[str, Any]:
                if "reviews(first:" in query:
                    raise RuntimeError("submitted reviews unavailable")
                return super().graphql(query, variables)

        for mode, native_codeowners_shadow, expected in (
            ("shadow", True, ControllerOutcome("incomplete")),
            ("live", True, ControllerOutcome("incomplete")),
            ("shadow", False, ControllerOutcome("incomplete")),
            ("live", False, ControllerOutcome("incomplete", 1)),
        ):
            github = UnavailableReviewsGitHub()
            with self.subTest(
                mode=mode,
                native_codeowners_shadow=native_codeowners_shadow,
            ):
                result = run_apply(
                    github,
                    has_uncovered_concerns=True,
                    mode=mode,
                    native_codeowners_shadow=native_codeowners_shadow,
                )

            self.assertEqual(result, expected)
            self.assertTrue(
                any(
                    BOT_TRIAGE_ERROR_LABEL in payload["labels"]
                    for _, endpoint, payload in mutations(github)
                    if endpoint.endswith("/labels")
                )
            )

    def test_no_destination_shadow_results_follow_submitted_handoff(self) -> None:
        cases = (
            ([], "kept_open", [CODEOWNERS_SHADOW_LABELS["match"]]),
            (
                ["soulitzer"],
                "shadow_triaged",
                [BOT_SHADOW_TRIAGED_LABEL, CODEOWNERS_SHADOW_LABELS["match"]],
            ),
        )
        for submitted_users, status, labels in cases:
            github = FakeGitHub(submitted_users=submitted_users)
            with self.subTest(submitted_users=submitted_users):
                self.assertEqual(
                    run_without_owners(github, mode="shadow"),
                    ControllerOutcome(status),
                )
                self.assertEqual(
                    mutations(github),
                    [
                        (
                            "POST",
                            "repos/pytorch/ciforge/issues/123/labels",
                            {"labels": labels},
                        )
                    ],
                )

    def test_uncovered_concerns_do_not_change_no_owner_handoff(self) -> None:
        for submitted_users, status in (
            ([], "kept_open"),
            (["soulitzer"], "shadow_triaged"),
        ):
            github = FakeGitHub(submitted_users=submitted_users)
            with self.subTest(submitted_users=submitted_users):
                self.assertEqual(
                    run_without_owners(
                        github,
                        mode="shadow",
                        has_uncovered_concerns=True,
                    ),
                    ControllerOutcome(status),
                )

    def test_plan_and_step_summary_use_one_schema_in_both_modes(self) -> None:
        plan_keys = []
        for mode, status in (("shadow", "shadow_triaged"), ("live", "triaged")):
            github = FakeGitHub()
            with tempfile.TemporaryDirectory() as directory, self.subTest(mode=mode):
                summary = Path(directory) / "summary.md"
                with mock.patch("builtins.print") as output:
                    result = run_apply(
                        github,
                        additional_owners=("autograd",),
                        mode=mode,
                        github_step_summary=summary,
                    )
                summary_text = summary.read_text()
                plan = printed_plan(output)
                routing = printed_reviewer_routing(output)
                plan_keys.append(set(plan))

                self.assertEqual(result.status, status)
                self.assertEqual(plan["mode"], mode)
                self.assertEqual(plan["analyzed_head_sha"], HEAD_SHA)
                self.assertFalse(plan["owner_provenance_truncated"])
                self.assertIn(f"- Mode: `{mode}`", summary_text)
                self.assertIn("`soulitzer`", summary_text)
                self.assertNotIn("@soulitzer", summary_text)
                if mode == "shadow":
                    self.assertIn(
                        "shadow mode logged one planned request for @soulitzer; "
                        "no review request was sent.",
                        routing,
                    )
                else:
                    self.assertIn(
                        "live mode plans one deduplicated request for @soulitzer.",
                        routing,
                    )
        self.assertEqual(plan_keys[0], plan_keys[1])

    def test_step_summary_failure_does_not_block_either_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "missing" / "summary.md"
            for mode, status in (("shadow", "shadow_triaged"), ("live", "triaged")):
                github = FakeGitHub()
                with self.subTest(mode=mode), mock.patch("builtins.print") as output:
                    result = run_apply(
                        github,
                        mode=mode,
                        github_step_summary=summary,
                    )

                self.assertEqual(result.status, status)
                self.assertTrue(mutations(github))
                self.assertTrue(
                    any(
                        str(call.args[0]).startswith(
                            "Auto PR Triage step summary unavailable:"
                        )
                        for call in output.call_args_list
                    )
                )

    def test_plan_escapes_workflow_commands_in_llm_provenance(self) -> None:
        provenance = {
            "autograd": {
                "source": "semantic",
                "files": ["torch/semantic.py"],
                "total_file_count": 1,
                "llm_justification": {
                    "owned_concern": "Autograd owns this changed behavior.",
                    "rationale": [
                        "::warning:: This is provenance, not a workflow command.",
                        "##[error] This is provenance, not a workflow command.",
                        "The configured description covers this changed contract.",
                    ],
                    "evidence": [
                        {
                            "file": "torch/semantic.py",
                            "diff_excerpt": "+\x1b[31m::notice:: changed behavior",
                            "relevance": "##[warning] This explains the relevant change.",
                        }
                    ],
                },
            }
        }
        with mock.patch("builtins.print") as output:
            run_apply(
                FakeGitHub(),
                codepath_owners=(),
                additional_owners=("autograd",),
                owner_provenance=provenance,
                mode="shadow",
            )

        prefix = "Auto PR Triage plan:\n"
        raw_plan = next(
            call.args[0]
            for call in output.call_args_list
            if call.args[0].startswith(prefix)
        )
        self.assertNotIn("::warning::", raw_plan)
        self.assertNotIn("##[error]", raw_plan)
        self.assertIn(
            "::warning::",
            printed_plan(output)["owner_choices"]["autograd"]["provenance"][
                "llm_justification"
            ]["rationale"][0],
        )
        routing = printed_reviewer_routing(output)
        self.assertNotIn("::notice::", routing)
        self.assertNotIn("##[warning]", routing)
        self.assertNotIn("\x1b", routing)
        self.assertIn(r"\u001b", routing)
        self.assertIn(r"\u003a\u003anotice\u003a\u003a", routing)
        self.assertIn(r"\u0023\u0023[warning]", routing)

    def test_pending_owner_is_logged_without_mutating_owner_label(self) -> None:
        github = FakeGitHub(requested_users=["soulitzer"])

        with mock.patch("builtins.print") as output:
            result = run_apply(
                github,
                codepath_owners=(),
                additional_owners=("autograd",),
                mode="shadow",
            )

        self.assertEqual(result, ControllerOutcome("shadow_triaged"))
        self.assertFalse(
            any(call[1].endswith("/requested_reviewers") for call in mutations(github))
        )
        self.assertEqual(
            sum(
                endpoint.endswith("labels/owner%3A%20autograd")
                for method, endpoint, _ in github.calls
                if method == "GET"
            ),
            1,
        )
        self.assertFalse(
            any("owner: autograd" in str(call) for call in mutations(github))
        )
        plan = printed_plan(output)
        choice = plan["owner_choices"]["autograd"]
        self.assertEqual(choice["reviewer"], "@soulitzer")
        self.assertEqual(choice["state"], "pending")
        self.assertEqual(choice["provenance"]["source"], "semantic")
        self.assertEqual(plan["planned_reviewer_requests"], [])

    def test_missing_pending_owner_label_matches_live_incomplete_result(self) -> None:
        for mode in ("shadow", "live"):
            github = FakeGitHub(
                requested_users=["soulitzer"],
                unavailable_labels=["owner: autograd"],
            )
            with self.subTest(mode=mode):
                result = run_apply(
                    github,
                    codepath_owners=(),
                    additional_owners=("autograd",),
                    mode=mode,
                )

                self.assertEqual(result, ControllerOutcome("incomplete"))
                self.assertEqual(
                    mutations(github)[0][2]["labels"][0],
                    BOT_TRIAGE_ERROR_LABEL,
                )

    def test_unavailable_submitted_handoff_fails_in_both_modes(self) -> None:
        class UnavailableSubmittedReviewsGitHub(FakeGitHub):
            def graphql(
                self, query: str, variables: dict[str, Any]
            ) -> dict[str, Any]:
                if "reviews(first:" in query:
                    raise RuntimeError("submitted reviews unavailable")
                return super().graphql(query, variables)

        for mode in ("shadow", "live"):
            github = UnavailableSubmittedReviewsGitHub()
            with self.subTest(mode=mode), self.assertRaisesRegex(
                RuntimeError, "submitted reviews unavailable"
            ):
                run_without_owners(github, mode=mode)
            self.assertEqual(mutations(github), [])

    def test_multiple_owners_deduplicate_the_proposed_reviewer(self) -> None:
        github = FakeGitHub()
        team_members = copy.deepcopy(ownership_config()["team_members"])
        team_members["members"]["compiler"] = ["@soulitzer"]

        with mock.patch("builtins.print") as output:
            result = run_apply(
                github,
                codepath_owners=(),
                additional_owners=("autograd", "compiler"),
                mode="shadow",
                team_members=team_members,
            )

        self.assertEqual(result, ControllerOutcome("shadow_triaged"))
        plan = printed_plan(output)
        self.assertEqual(plan["planned_reviewer_requests"], ["@soulitzer"])
        self.assertEqual(set(plan["owner_choices"]), {"autograd", "compiler"})
        self.assertEqual(
            {
                choice["reviewer"]
                for choice in plan["owner_choices"].values()
            },
            {"@soulitzer"},
        )
        routing = printed_reviewer_routing(output)
        self.assertEqual(routing.count("Reviewer @soulitzer"), 1)
        self.assertIn("Semantic owner `autograd`", routing)
        self.assertIn("Semantic owner `compiler`", routing)
        self.assertEqual(routing.count("Planned effect:"), 1)
        self.assertEqual(len(mutations(github)), 1)

    def test_incomplete_decision_only_records_error_diagnostics(self) -> None:
        github = FakeGitHub()

        result = run_apply(
            github,
            ownership_analysis="incomplete",
            mode="shadow",
        )

        self.assertEqual(result, ControllerOutcome("incomplete"))
        self.assertEqual(
            mutations(github)[0][2],
            {
                "labels": [
                    BOT_TRIAGE_ERROR_LABEL,
                    CODEOWNERS_SHADOW_LABELS["match"],
                ]
            },
        )


class ApplyTriageTest(unittest.TestCase):
    def test_native_codepath_owner_is_not_requested_again(self) -> None:
        github = FakeGitHub()

        with mock.patch("builtins.print") as output:
            result = run_apply(github)

        self.assertEqual(result, ControllerOutcome("triaged"))
        self.assertEqual(
            mutations(github),
            [
                (
                    "POST",
                    "repos/pytorch/ciforge/issues/123/labels",
                    {
                        "labels": [
                            "triaged",
                            "bot-triaged",
                            CODEOWNERS_SHADOW_LABELS["match"],
                        ]
                    },
                )
            ],
        )
        self.assertEqual(
            github.live_reads,
            ["codeowners_shadow"],
        )
        choice = printed_plan(output)["owner_choices"]["@codepath-owner"]
        self.assertEqual(choice["state"], "native_codeowner")
        routing = printed_reviewer_routing(output)
        self.assertIn(
            "GitHub already has an active native CODEOWNERS request for "
            "@codepath-owner.",
            routing,
        )
        self.assertIn("Codepath owner `@codepath-owner`", routing)

    def test_direct_codepath_owner_does_not_load_roster_configuration(self) -> None:
        github = FakeGitHub(actionable_issue=True)
        args = controller_args()

        with mock.patch(
            "apply_triage_decision.load_team_members",
            side_effect=AssertionError("roster configuration should not load"),
        ), mock.patch("apply_triage_decision.NATIVE_CODEOWNERS_SHADOW", False):
            result = apply_controller_action(args, github)

        self.assertEqual(result, ControllerOutcome("triaged", 1, 0))

    def test_incomplete_ownership_analysis_preserves_codepath_owners(self) -> None:
        github = FakeGitHub()

        result = run_apply(github, ownership_analysis="incomplete")

        self.assertEqual(result, ControllerOutcome("incomplete"))
        self.assertEqual(
            mutations(github)[0][2],
            {
                "labels": [
                    BOT_TRIAGE_ERROR_LABEL,
                    CODEOWNERS_SHADOW_LABELS["match"],
                ]
            },
        )

    def test_incomplete_analysis_resolves_internal_codepath_owner(self) -> None:
        github = FakeGitHub()

        result = run_apply(
            github,
            codepath_owners=("autograd",),
            ownership_analysis="incomplete",
            native_codeowners_shadow=False,
        )

        self.assertEqual(result, ControllerOutcome("incomplete", 1, 1))
        self.assertEqual(mutations(github)[0][2], {"reviewers": ["soulitzer"]})
        self.assertEqual(
            mutations(github)[1][2],
            {
                "labels": [
                    BOT_TRIAGE_ERROR_LABEL,
                    "owner: autograd",
                ]
            },
        )

    def test_fresh_round_robin_reviewer_is_requested_and_labeled(self) -> None:
        github = FakeGitHub()

        with mock.patch("builtins.print") as output:
            result = run_apply(
                github,
                codepath_owners=(),
                additional_owners=("autograd",),
            )

        self.assertEqual(result, ControllerOutcome("triaged", 1, 1))
        self.assertEqual(
            mutations(github),
            [
                (
                    "POST",
                    "repos/pytorch/ciforge/pulls/123/requested_reviewers",
                    {"reviewers": ["soulitzer"]},
                ),
                (
                    "POST",
                    "repos/pytorch/ciforge/issues/123/labels",
                    {
                        "labels": [
                            "triaged",
                            "bot-triaged",
                            CODEOWNERS_SHADOW_LABELS["match"],
                            "owner: autograd",
                        ]
                    },
                ),
            ],
        )
        self.assertEqual(
            sum(
                endpoint.endswith("labels/owner%3A%20autograd")
                for method, endpoint, _ in github.calls
                if method == "GET"
            ),
            1,
        )
        choice = printed_plan(output)["owner_choices"]["autograd"]
        self.assertEqual(choice["reviewer"], "@soulitzer")
        self.assertEqual(choice["selection_reason"], "round_robin_initial")
        self.assertEqual(choice["provenance"]["source"], "semantic")
        self.assertEqual(choice["provenance"]["files"], ["torch/semantic.py"])
        self.assertEqual(
            choice["provenance"]["llm_justification"]["evidence"],
            [
                {
                    "file": "torch/semantic.py",
                    "diff_excerpt": "+new semantic behavior",
                    "relevance": "This line implements the owned behavior.",
                }
            ],
        )
        self.assertEqual(
            choice["provenance"]["llm_justification"]["owned_concern"],
            "autograd owns this changed behavior.",
        )

    def test_truncated_provenance_does_not_change_reviewer_request(self) -> None:
        github = FakeGitHub()

        with mock.patch("builtins.print") as output:
            result = run_apply(
                github,
                codepath_owners=(),
                additional_owners=("autograd",),
                owner_provenance_truncated=True,
            )

        self.assertEqual(result, ControllerOutcome("triaged", 1, 1))
        self.assertEqual(mutations(github)[0][2], {"reviewers": ["soulitzer"]})
        plan = printed_plan(output)
        self.assertTrue(plan["owner_provenance_truncated"])
        self.assertEqual(
            plan["owner_choices"]["autograd"]["provenance"],
            {"source": "semantic", "truncated": True},
        )

    def test_two_member_roster_bootstraps_first_member(self) -> None:
        github = FakeGitHub()
        team_members = copy.deepcopy(ownership_config()["team_members"])
        team_members["members"]["autograd"] = ["@first", "@second"]

        result = run_apply(
            github,
            codepath_owners=(),
            additional_owners=("autograd",),
            team_members=team_members,
        )

        self.assertEqual(result, ControllerOutcome("triaged", 1, 1))
        self.assertEqual(mutations(github)[0][2], {"reviewers": ["first"]})

    def test_two_member_roster_advances_from_history(self) -> None:
        github = FakeGitHub(
            round_robin_events=[
                {
                    "id": 91,
                    "event": "labeled",
                    "label": {"name": "owner: autograd"},
                    "issue": {"number": 7, "pull_request": {}},
                }
            ],
            round_robin_timelines={
                7: [
                    {
                        "id": 90,
                        "event": "review_requested",
                        "requested_reviewer": {"login": "first"},
                    },
                    {"id": 91, "event": "labeled"},
                ]
            },
        )
        team_members = copy.deepcopy(ownership_config()["team_members"])
        team_members["members"]["autograd"] = ["@first", "@second"]

        with mock.patch("builtins.print") as output:
            result = run_apply(
                github,
                codepath_owners=(),
                additional_owners=("autograd",),
                team_members=team_members,
            )

        self.assertEqual(result, ControllerOutcome("triaged", 1, 1))
        self.assertEqual(mutations(github)[0][2], {"reviewers": ["second"]})
        self.assertEqual(
            printed_plan(output)["owner_choices"]["autograd"]["selection_reason"],
            "round_robin_next",
        )
        self.assertIn(
            "@second was the next eligible member of `autograd`'s round-robin rotation.",
            printed_reviewer_routing(output),
        )

    def test_unassigned_owner_marker_uses_fallback_and_repairs_state(self) -> None:
        github = FakeGitHub(
            round_robin_events=[
                {
                    "id": 91,
                    "event": "labeled",
                    "label": {"name": "owner: autograd"},
                    "issue": {"number": 7, "pull_request": {}},
                }
            ],
            round_robin_timelines={7: [{"id": 91, "event": "labeled"}]},
        )
        team_members = copy.deepcopy(ownership_config()["team_members"])
        team_members["members"]["autograd"] = ["@first", "@second"]
        expected = stable_fallback_member(
            "pytorch/ciforge",
            123,
            "autograd",
            ("@first", "@second"),
            {"@external-author"},
        )

        with mock.patch("builtins.print") as output:
            result = run_apply(
                github,
                codepath_owners=(),
                additional_owners=("autograd",),
                team_members=team_members,
            )

        self.assertEqual(result, ControllerOutcome("triaged", 1, 1))
        self.assertEqual(mutations(github)[0][2], {"reviewers": [expected[1:]]})
        self.assertIn("owner: autograd", mutations(github)[1][2]["labels"])
        self.assertTrue(
            any("using stable fallback" in call.args[0] for call in output.call_args_list)
        )
        self.assertEqual(
            printed_plan(output)["owner_choices"]["autograd"]["selection_reason"],
            "stable_fallback",
        )

    def test_two_member_roster_wraps_after_second_member(self) -> None:
        github = FakeGitHub(
            round_robin_events=[
                {
                    "id": 91,
                    "event": "labeled",
                    "label": {"name": "owner: autograd"},
                    "issue": {"number": 7, "pull_request": {}},
                }
            ],
            round_robin_timelines={
                7: [
                    {
                        "id": 90,
                        "event": "review_requested",
                        "requested_reviewer": {"login": "second"},
                    },
                    {"id": 91, "event": "labeled"},
                ]
            },
        )
        team_members = copy.deepcopy(ownership_config()["team_members"])
        team_members["members"]["autograd"] = ["@first", "@second"]

        result = run_apply(
            github,
            codepath_owners=(),
            additional_owners=("autograd",),
            team_members=team_members,
        )

        self.assertEqual(result, ControllerOutcome("triaged", 1, 1))
        self.assertEqual(mutations(github)[0][2], {"reviewers": ["first"]})

    def test_internal_codepath_owner_uses_roster_and_round_robin(self) -> None:
        github = FakeGitHub()

        with mock.patch("builtins.print") as output:
            result = run_apply(
                github,
                codepath_owners=("autograd",),
                native_codeowners_shadow=False,
            )

        self.assertEqual(result, ControllerOutcome("triaged", 1, 1))
        self.assertEqual(
            mutations(github),
            [
                (
                    "POST",
                    "repos/pytorch/ciforge/pulls/123/requested_reviewers",
                    {"reviewers": ["soulitzer"]},
                ),
                (
                    "POST",
                    "repos/pytorch/ciforge/issues/123/labels",
                    {
                        "labels": [
                            "triaged",
                            "bot-triaged",
                            "owner: autograd",
                        ]
                    },
                ),
            ],
        )
        plan = printed_plan(output)
        self.assertEqual(plan["codepath_owners"], ["autograd"])
        choice = plan["owner_choices"]["autograd"]
        self.assertEqual(choice["reviewer"], "@soulitzer")
        self.assertEqual(choice["selection_reason"], "round_robin_initial")
        self.assertEqual(choice["provenance"]["source"], "codepath")
        self.assertEqual(choice["provenance"]["files"], ["torch/file.py"])

    def test_native_codepath_owner_and_additional_owner_share_reviewer(self) -> None:
        github = FakeGitHub(native_codeowners=[])

        with mock.patch("builtins.print") as output:
            result = run_apply(
                github,
                codepath_owners=("@soulitzer",),
                additional_owners=("autograd",),
            )

        self.assertEqual(result, ControllerOutcome("triaged", 1, 1))
        request = next(
            call for call in mutations(github) if call[1].endswith("/requested_reviewers")
        )
        self.assertEqual(request[2], {"reviewers": ["soulitzer"]})
        plan = printed_plan(output)
        self.assertEqual(plan["codepath_owners"], ["@soulitzer"])
        self.assertEqual(set(plan["owner_choices"]), {"autograd"})
        self.assertEqual(
            plan["owner_choices"]["autograd"]["selection_reason"],
            "round_robin_initial",
        )
        self.assertEqual(plan["planned_reviewer_requests"], ["@soulitzer"])

    def test_existing_codepath_owner_requests_are_preserved(self) -> None:
        github = FakeGitHub(
            requested_teams=["compiler"],
            submitted_users=["codepath-owner"],
        )

        with mock.patch("builtins.print") as output:
            result = run_apply(
                github,
                codepath_owners=("@codepath-owner", "@pytorch/compiler"),
                native_codeowners_shadow=False,
            )

        self.assertEqual(result, ControllerOutcome("triaged", 0, 0))
        self.assertFalse(
            any(call[1].endswith("/requested_reviewers") for call in mutations(github))
        )
        choices = printed_plan(output)["owner_choices"]
        self.assertEqual(choices["@codepath-owner"]["state"], "submitted")
        self.assertEqual(choices["@pytorch/compiler"]["state"], "pending")

    def test_large_codepath_owner_set_allows_no_new_requests(self) -> None:
        reviewers = tuple(f"owner{index}" for index in range(16))
        github = FakeGitHub(requested_users=list(reviewers))

        result = run_apply(
            github,
            codepath_owners=tuple(f"@{reviewer}" for reviewer in reviewers),
        )

        self.assertEqual(result, ControllerOutcome("triaged", 0, 0, 0))
        self.assertFalse(
            any(call[1].endswith("/requested_reviewers") for call in mutations(github))
        )

    def test_codepath_owner_policy_does_not_request_the_author(self) -> None:
        github = FakeGitHub()

        result = run_apply(
            github,
            codepath_owners=("@external-author",),
        )

        self.assertEqual(result, ControllerOutcome("triaged", 0, 0))
        self.assertFalse(
            any(call[1].endswith("/requested_reviewers") for call in mutations(github))
        )

    def test_native_codepath_owners_do_not_consume_the_request_limit(self) -> None:
        github = FakeGitHub()
        codepath_owners = tuple(f"@owner{index}" for index in range(15))

        result = run_apply(
            github,
            codepath_owners=codepath_owners,
            additional_owners=("compiler",),
        )

        self.assertEqual(result, ControllerOutcome("triaged", 1, 1))
        self.assertEqual(mutations(github)[0][2], {"reviewers": ["reviewer"]})

    def test_pending_owner_member_gets_assignment_label(self) -> None:
        github = FakeGitHub(requested_users=["soulitzer"])

        result = run_apply(
            github,
            codepath_owners=(),
            additional_owners=("autograd",),
        )

        self.assertEqual(result, ControllerOutcome("triaged", 0, 1))
        label_post = next(
            call for call in mutations(github) if call[1].endswith("/labels")
        )
        self.assertEqual(
            label_post[2],
            {
                "labels": [
                    "triaged",
                    "bot-triaged",
                    CODEOWNERS_SHADOW_LABELS["match"],
                    "owner: autograd",
                ]
            },
        )

    def test_pending_and_new_additional_owners_get_assignment_labels(self) -> None:
        github = FakeGitHub(requested_users=["soulitzer"])

        result = run_apply(
            github,
            codepath_owners=(),
            additional_owners=("autograd", "compiler"),
        )

        self.assertEqual(result, ControllerOutcome("triaged", 1, 2))
        label_post = next(
            call for call in mutations(github) if call[1].endswith("/labels")
        )
        self.assertEqual(
            label_post[2],
            {
                "labels": [
                    "triaged",
                    "bot-triaged",
                    CODEOWNERS_SHADOW_LABELS["match"],
                    "owner: autograd",
                    "owner: compiler",
                ]
            },
        )

    def test_two_fresh_owners_request_two_reviewers(self) -> None:
        github = FakeGitHub()

        result = run_apply(
            github,
            codepath_owners=(),
            additional_owners=("autograd", "compiler"),
        )

        self.assertEqual(result, ControllerOutcome("triaged", 2, 2))
        self.assertEqual(
            mutations(github)[0][2],
            {"reviewers": ["reviewer", "soulitzer"]},
        )
        self.assertEqual(
            mutations(github)[1][2],
            {
                "labels": [
                    "triaged",
                    "bot-triaged",
                    CODEOWNERS_SHADOW_LABELS["match"],
                    "owner: autograd",
                    "owner: compiler",
                ]
            },
        )

    def test_two_owners_deduplicate_shared_rotation_member(self) -> None:
        github = FakeGitHub(
            round_robin_events=[
                {
                    "id": 91,
                    "event": "labeled",
                    "label": {"name": "owner: autograd"},
                    "issue": {"number": 7, "pull_request": {}},
                }
            ],
            round_robin_timelines={
                7: [
                    {
                        "id": 90,
                        "event": "review_requested",
                        "requested_reviewer": {"login": "soulitzer"},
                    },
                    {"id": 91, "event": "labeled"},
                ]
            },
        )
        team_members = copy.deepcopy(ownership_config()["team_members"])
        team_members["members"]["autograd"] = ["@soulitzer", "@izaitsevfb"]
        team_members["members"]["nn"] = ["@izaitsevfb"]

        result = run_apply(
            github,
            codepath_owners=(),
            additional_owners=("autograd", "nn"),
            team_members=team_members,
        )

        self.assertEqual(result, ControllerOutcome("triaged", 1, 2))
        self.assertEqual(
            mutations(github)[0][2],
            {"reviewers": ["izaitsevfb"]},
        )
        self.assertEqual(
            mutations(github)[1][2],
            {
                "labels": [
                    "triaged",
                    "bot-triaged",
                    CODEOWNERS_SHADOW_LABELS["match"],
                    "owner: autograd",
                    "owner: nn",
                ]
            },
        )

    def test_submitted_owner_member_needs_no_request_or_routing_label(self) -> None:
        github = FakeGitHub(submitted_users=["soulitzer"])

        result = run_apply(
            github,
            codepath_owners=(),
            additional_owners=("autograd",),
        )

        self.assertEqual(result, ControllerOutcome("triaged"))
        self.assertFalse(
            any(call[1].endswith("/requested_reviewers") for call in mutations(github))
        )
        self.assertFalse(
            any("owner: autograd" in str(call[2]) for call in mutations(github))
        )

    def test_unknown_additional_owner_marks_routing_incomplete(self) -> None:
        github = FakeGitHub()

        result = run_apply(
            github,
            codepath_owners=(),
            additional_owners=("unknown",),
        )

        self.assertEqual(result, ControllerOutcome("incomplete"))
        self.assertEqual(
            mutations(github)[0][2],
            {
                "labels": [
                    BOT_TRIAGE_ERROR_LABEL,
                    CODEOWNERS_SHADOW_LABELS["match"],
                ]
            },
        )

    def test_codepath_team_handle_does_not_suppress_additional_owner(self) -> None:
        github = FakeGitHub(actionable_issue=True)
        config = ownership_config()
        args = controller_args(
            analysis_result_json=analysis_result_json(
                codepath_owners=("@pytorch/autograd",),
                additional_owners=("autograd",),
            )
        )

        with mock.patch(
            "apply_triage_decision.load_team_members",
            return_value=config["team_members"],
        ), mock.patch(
            "apply_triage_decision.NATIVE_CODEOWNERS_SHADOW", False
        ), mock.patch("builtins.print") as output:
            result = apply_controller_action(args, github)
        self.assertEqual(result, ControllerOutcome("triaged", 1, 1, 1))
        self.assertEqual(
            mutations(github)[0][2],
            {"reviewers": ["soulitzer"], "team_reviewers": ["autograd"]},
        )
        choices = printed_plan(output)["owner_choices"]
        self.assertEqual(choices["autograd"]["provenance"]["source"], "semantic")
        self.assertEqual(
            choices["@pytorch/autograd"]["selection_reason"],
            "direct_codepath_owner",
        )
        self.assertEqual(
            choices["@pytorch/autograd"]["provenance"]["source"],
            "codepath",
        )

    def test_missing_live_reviewer_is_selected_at_apply_time(self) -> None:
        github = FakeGitHub()

        result = run_apply(
            github,
            codepath_owners=(),
            additional_owners=("autograd",),
        )
        self.assertEqual(result, ControllerOutcome("triaged", 1, 1))
        self.assertEqual(mutations(github)[0][2], {"reviewers": ["soulitzer"]})

    def test_missing_status_label_prevents_writes(self) -> None:
        github = FakeGitHub(unavailable_labels=["bot-triaged"])

        with self.assertRaisesRegex(RuntimeError, "label.*unavailable"):
            run_apply(
                github,
                codepath_owners=(),
                additional_owners=("autograd",),
            )

        self.assertEqual(mutations(github), [])

    def test_missing_routing_label_degrades_to_incomplete(self) -> None:
        github = FakeGitHub(unavailable_labels=["owner: autograd"])

        result = run_apply(
            github,
            codepath_owners=(),
            additional_owners=("autograd",),
        )

        self.assertEqual(result, ControllerOutcome("incomplete"))
        self.assertEqual(
            mutations(github)[0][2],
            {
                "labels": [
                    BOT_TRIAGE_ERROR_LABEL,
                    CODEOWNERS_SHADOW_LABELS["match"],
                ]
            },
        )

    def test_missing_error_label_prevents_fallback_write(self) -> None:
        github = FakeGitHub(unavailable_labels=[BOT_TRIAGE_ERROR_LABEL])

        with self.assertRaisesRegex(RuntimeError, "label.*unavailable"):
            run_apply(github, ownership_analysis="incomplete")

        self.assertEqual(mutations(github), [])

    def test_internal_codepath_owner_requires_roster_and_label(self) -> None:
        cases = (
            ("unknown", [], "not configured"),
            ("autograd", ["owner: autograd"], "label.*unavailable"),
        )
        for owner, unavailable_labels, error in cases:
            github = FakeGitHub(unavailable_labels=unavailable_labels)
            with self.subTest(owner=owner), self.assertRaisesRegex(
                (ValueError, RuntimeError), error
            ):
                run_apply(
                    github,
                    codepath_owners=(owner,),
                    native_codeowners_shadow=False,
                )
            self.assertEqual(mutations(github), [])

    def test_unavailable_additional_owner_routing_preserves_codepath_handles(self) -> None:
        github = FakeGitHub(unavailable_labels=["owner: autograd"])

        result = run_apply(
            github,
            codepath_owners=("@codepath-owner",),
            additional_owners=("autograd",),
        )

        self.assertEqual(result, ControllerOutcome("incomplete"))
        self.assertEqual(
            mutations(github),
            [
                (
                    "POST",
                    "repos/pytorch/ciforge/issues/123/labels",
                    {
                        "labels": [
                            BOT_TRIAGE_ERROR_LABEL,
                            CODEOWNERS_SHADOW_LABELS["match"],
                        ]
                    },
                ),
            ],
        )

    def test_unavailable_reviewer_state_preserves_codepath_handles(self) -> None:
        class UnavailableReviewersGitHub(FakeGitHub):
            def json(
                self,
                endpoint: str,
                *,
                method: str = "GET",
                payload: dict[str, Any] | None = None,
            ) -> Any:
                if endpoint.endswith("/requested_reviewers") and method == "GET":
                    raise RuntimeError("reviewer state unavailable")
                return super().json(endpoint, method=method, payload=payload)

        github = UnavailableReviewersGitHub()
        result = run_apply(
            github,
            codepath_owners=("@codepath-owner",),
            additional_owners=("autograd",),
        )

        self.assertEqual(result, ControllerOutcome("incomplete"))
        self.assertEqual(
            mutations(github)[0][2],
            {
                "labels": [
                    BOT_TRIAGE_ERROR_LABEL,
                    CODEOWNERS_SHADOW_LABELS["match"],
                ]
            },
        )

    def test_unavailable_reviewer_state_does_not_drop_internal_codepath_owner(
        self,
    ) -> None:
        class UnavailableReviewersGitHub(FakeGitHub):
            def json(
                self,
                endpoint: str,
                *,
                method: str = "GET",
                payload: dict[str, Any] | None = None,
            ) -> Any:
                if endpoint.endswith("/requested_reviewers") and method == "GET":
                    raise RuntimeError("reviewer state unavailable")
                return super().json(endpoint, method=method, payload=payload)

        for codepath_owners in (("autograd",), ("@codepath-owner", "autograd")):
            github = UnavailableReviewersGitHub()
            with self.subTest(codepath_owners=codepath_owners), self.assertRaisesRegex(
                RuntimeError, "reviewer state unavailable"
            ):
                run_apply(
                    github,
                    codepath_owners=codepath_owners,
                    native_codeowners_shadow=False,
                )
            self.assertEqual(mutations(github), [])

    def test_apply_deliberately_trusts_analysis_result(self) -> None:
        github = FakeGitHub(labels=["triaged", "bot-triaged"])
        github.pr.update(state="closed", draft=True, title="changed")
        github.pr["head"]["sha"] = "c" * 40
        github.pr["base"]["ref"] = "release"

        self.assertEqual(run_apply(github).status, "triaged")
        self.assertEqual(github.pr_fetches, 0)
        self.assertEqual(github.actionable_checks, 0)
        self.assertEqual(github.permission_checks, 0)

    def test_reviewer_state_is_fetched_once_without_race_recheck(self) -> None:
        github = FakeGitHub()

        self.assertEqual(
            run_apply(
                github,
                codepath_owners=(),
                additional_owners=("autograd",),
            ).status,
            "triaged",
        )
        self.assertEqual(github.live_reads.count("requested"), 1)
        self.assertEqual(github.live_reads.count("submitted"), 1)

    def test_satisfied_result_without_owners_uses_submitted_review_handoff(self) -> None:
        github = FakeGitHub(
            submitted_users=["soulitzer"],
            submitted_user_state="COMMENTED",
            actionable_issue=True,
        )

        result = run_without_owners(github)

        self.assertEqual(result, ControllerOutcome("triaged", 0, 0))
        self.assertFalse(
            any(call[1].endswith("/requested_reviewers") for call in mutations(github))
        )

    def test_handoff_trusts_analyzed_triage_facts(self) -> None:
        github = FakeGitHub(
            submitted_users=["soulitzer"],
            submitted_user_state="COMMENTED",
        )

        self.assertEqual(run_without_owners(github), ControllerOutcome("triaged"))
        self.assertEqual(github.actionable_checks, 0)
        self.assertEqual(github.permission_checks, 0)

    def test_handoff_does_not_refetch_author_permission(self) -> None:
        github = FakeGitHub(
            submitted_users=["soulitzer"],
            author_has_triage_permission=True,
        )

        self.assertEqual(run_without_owners(github), ControllerOutcome("triaged"))
        self.assertEqual(github.permission_checks, 0)

    def test_no_owners_without_eligible_handoff_records_shadow_match(self) -> None:
        cases = (
            FakeGitHub(requested_users=["soulitzer"], actionable_issue=True),
            FakeGitHub(
                submitted_users=["soulitzer"],
                submitted_user_state="DISMISSED",
                actionable_issue=True,
            ),
            FakeGitHub(submitted_users=["outsider"], actionable_issue=True),
        )
        for github in cases:
            with self.subTest(calls=github.calls):
                self.assertEqual(
                    run_without_owners(github), ControllerOutcome("kept_open")
                )
            self.assertEqual(
                mutations(github),
                [
                    (
                        "POST",
                        "repos/pytorch/ciforge/issues/123/labels",
                        {"labels": [CODEOWNERS_SHADOW_LABELS["match"]]},
                    )
                ],
            )

    def test_no_owners_does_not_use_the_author_as_submitted_handoff(self) -> None:
        github = FakeGitHub(
            submitted_users=["external-author"], actionable_issue=True
        )
        config = ownership_config()
        config["team_members"]["members"]["autograd"].append(
            "@external-author"
        )
        args = controller_args(
            analysis_result_json=analysis_result_json(codepath_owners=())
        )
        with mock.patch(
            "apply_triage_decision.load_team_members",
            return_value=config["team_members"],
        ), mock.patch("apply_triage_decision.NATIVE_CODEOWNERS_SHADOW", False):
            self.assertEqual(
                apply_controller_action(args, github),
                ControllerOutcome("kept_open"),
            )
        self.assertEqual(mutations(github), [])

    def test_author_cannot_be_selected_by_round_robin(self) -> None:
        github = FakeGitHub()
        config = ownership_config()
        config["team_members"]["members"]["autograd"] = [
            "@external-author"
        ]
        args = controller_args(
            analysis_result_json=analysis_result_json(
                codepath_owners=(),
                additional_owners=("autograd",),
            )
        )
        with mock.patch(
            "apply_triage_decision.load_team_members",
            return_value=config["team_members"],
        ), mock.patch(
            "apply_triage_decision.NATIVE_CODEOWNERS_SHADOW", False
        ), self.assertRaisesRegex(RuntimeError, "no eligible"):
            apply_controller_action(args, github)
        self.assertEqual(mutations(github), [])

    def test_author_codepath_match_does_not_cover_additional_owner(self) -> None:
        github = FakeGitHub(actionable_issue=True)
        config = ownership_config()
        config["team_members"]["members"]["autograd"].append(
            "@external-author"
        )
        args = controller_args(
            analysis_result_json=analysis_result_json(
                codepath_owners=("@external-author",),
                additional_owners=("autograd",),
            )
        )

        with mock.patch(
            "apply_triage_decision.load_team_members",
            return_value=config["team_members"],
        ), mock.patch("apply_triage_decision.NATIVE_CODEOWNERS_SHADOW", False):
            result = apply_controller_action(args, github)

        self.assertEqual(result, ControllerOutcome("triaged", 1, 1))
        self.assertEqual(mutations(github)[0][2], {"reviewers": ["soulitzer"]})

    def test_triage_trusts_analyzed_triage_facts(self) -> None:
        github = FakeGitHub()
        args = controller_args()

        with mock.patch("apply_triage_decision.NATIVE_CODEOWNERS_SHADOW", False):
            self.assertEqual(
                apply_controller_action(args, github),
                ControllerOutcome("triaged", 1, 0),
            )
        self.assertEqual(github.actionable_checks, 0)
        self.assertEqual(github.permission_checks, 0)

    def test_codepath_roster_member_suppresses_additional_owner(self) -> None:
        github = FakeGitHub()
        with mock.patch("builtins.print") as output:
            result = run_apply(
                github,
                codepath_owners=("@soulitzer",),
                additional_owners=("autograd",),
            )
        self.assertEqual(result, ControllerOutcome("triaged"))
        self.assertFalse(
            any("owner: autograd" in str(call[2]) for call in mutations(github))
        )
        routing = printed_reviewer_routing(output)
        self.assertIn(
            "@soulitzer covers `autograd` through codepath ownership; "
            "no `autograd` round-robin choice was needed.",
            routing,
        )
        self.assertIn("Semantic owner `autograd`", routing)
        self.assertLess(
            routing.index("Codepath owner `@soulitzer`"),
            routing.index("Semantic owner `autograd`"),
        )
        self.assertIn(
            "Planned effect: no new Auto PR Triage reviewer request is needed.",
            routing,
        )

    def test_ambiguous_reviewer_request_stops_before_labels(self) -> None:
        class TimeoutGitHub(FakeGitHub):
            def json(
                self,
                endpoint: str,
                *,
                method: str = "GET",
                payload: dict[str, Any] | None = None,
            ) -> Any:
                if endpoint.endswith("/requested_reviewers") and method == "POST":
                    raise subprocess.TimeoutExpired("gh api", 60)
                return super().json(endpoint, method=method, payload=payload)

        github = TimeoutGitHub()
        with self.assertRaisesRegex(RuntimeError, "may already be present"):
            run_apply(
                github,
                codepath_owners=(),
                additional_owners=("compiler",),
            )
        self.assertFalse(any(call[1].endswith("/labels") for call in mutations(github)))

    def test_successful_user_request_is_not_refetched(self) -> None:
        class EmptyResponseGitHub(FakeGitHub):
            def json(
                self,
                endpoint: str,
                *,
                method: str = "GET",
                payload: dict[str, Any] | None = None,
            ) -> Any:
                if endpoint.endswith("/requested_reviewers") and method == "POST":
                    self.calls.append((method, endpoint, payload))
                    return {"requested_reviewers": []}
                return super().json(endpoint, method=method, payload=payload)

        github = EmptyResponseGitHub()
        result = run_apply(
            github,
            codepath_owners=(),
            additional_owners=("compiler",),
        )
        self.assertEqual(result, ControllerOutcome("triaged", 1, 1))
        self.assertEqual(github.live_reads.count("requested"), 1)
        self.assertTrue(any(call[1].endswith("/labels") for call in mutations(github)))

    def test_native_team_request_is_not_repeated(self) -> None:
        class EmptyResponseGitHub(FakeGitHub):
            def json(
                self,
                endpoint: str,
                *,
                method: str = "GET",
                payload: dict[str, Any] | None = None,
            ) -> Any:
                if endpoint.endswith("/requested_reviewers") and method == "POST":
                    self.calls.append((method, endpoint, payload))
                    return {"users": [], "teams": []}
                return super().json(endpoint, method=method, payload=payload)

        github = EmptyResponseGitHub()
        result = run_apply(
            github,
            codepath_owners=("@pytorch/compiler",),
        )
        self.assertEqual(result, ControllerOutcome("triaged"))
        self.assertEqual(github.live_reads.count("requested"), 0)
        self.assertTrue(any(call[1].endswith("/labels") for call in mutations(github)))


class CloseDecisionTest(unittest.TestCase):
    def test_clean_unprotected_policy_close_pr_closes(self) -> None:
        github = FakeGitHub(labels=["open source"])

        result = run_close(github)

        self.assertEqual(result, ControllerOutcome("closed"))
        self.assertEqual(
            mutations(github),
            [
                (
                    "PATCH",
                    "repos/pytorch/ciforge/pulls/123",
                    {"state": "closed"},
                ),
                (
                    "POST",
                    "repos/pytorch/ciforge/issues/123/labels",
                    {"labels": ["bot-closed"]},
                ),
                (
                    "POST",
                    "repos/pytorch/ciforge/issues/123/comments",
                    {"body": BOT_CLOSED_COMMENT},
                ),
            ],
        )
        self.assertEqual(github.live_reads, [])

    def test_missing_actionable_issue_on_retry_is_a_read_free_noop(self) -> None:
        github = FakeGitHub(labels=["open source"])

        self.assertEqual(
            run_close(github, run_attempt=2), ControllerOutcome("kept_open")
        )
        self.assertEqual(github.calls, [])

    def test_close_does_not_refetch_triage_facts_or_pr_state(self) -> None:
        github = FakeGitHub(
            labels=["triaged", "bot-triaged", "bot-closed"],
            actionable_issue=True,
            author_has_triage_permission=True,
            requested_users=["soulitzer"],
            submitted_users=["soulitzer"],
        )
        github.pr.update(draft=True, title="changed")
        github.pr["head"]["sha"] = "c" * 40
        github.pr["base"]["ref"] = "release"

        self.assertEqual(run_close(github), ControllerOutcome("closed"))
        self.assertEqual(github.live_reads, [])
        self.assertEqual(github.pr_fetches, 0)
        self.assertEqual(github.actionable_checks, 0)
        self.assertEqual(github.permission_checks, 0)

    def test_missing_bot_closed_label_prevents_writes(self) -> None:
        github = FakeGitHub(
            labels=["open source"], unavailable_labels=["bot-closed"]
        )
        with self.assertRaisesRegex(RuntimeError, "required repository label"):
            run_close(github)
        self.assertEqual(mutations(github), [])

    def test_close_timeout_reports_ambiguous_state(self) -> None:
        class TimeoutGitHub(FakeGitHub):
            def json(
                self,
                endpoint: str,
                *,
                method: str = "GET",
                payload: dict[str, Any] | None = None,
            ) -> Any:
                if endpoint.endswith("/pulls/123") and method == "PATCH":
                    raise subprocess.TimeoutExpired("gh api", 60)
                return super().json(endpoint, method=method, payload=payload)

        github = TimeoutGitHub(labels=["open source"])
        with self.assertRaisesRegex(RuntimeError, "may already be closed"):
            run_close(github)
        self.assertFalse(any(call[0] == "POST" for call in github.calls))

    def test_unconfirmed_close_response_reports_ambiguous_state(self) -> None:
        class UnconfirmedGitHub(FakeGitHub):
            def json(
                self,
                endpoint: str,
                *,
                method: str = "GET",
                payload: dict[str, Any] | None = None,
            ) -> Any:
                response = super().json(endpoint, method=method, payload=payload)
                if endpoint.endswith("/pulls/123") and method == "PATCH":
                    response["state"] = "open"
                return response

        github = UnconfirmedGitHub(labels=["open source"])
        with self.assertRaisesRegex(RuntimeError, "not confirmed"):
            run_close(github)
        self.assertFalse(any(call[0] == "POST" for call in github.calls))

    def test_annotation_failures_are_reported_after_confirmed_close(self) -> None:
        class AnnotationFailureGitHub(FakeGitHub):
            def json(
                self,
                endpoint: str,
                *,
                method: str = "GET",
                payload: dict[str, Any] | None = None,
            ) -> Any:
                if method == "POST" and (
                    endpoint.endswith("/labels")
                    or endpoint.endswith("/comments")
                ):
                    self.calls.append((method, endpoint, payload))
                    raise RuntimeError("annotation failure " + "x" * 500)
                return super().json(endpoint, method=method, payload=payload)

        github = AnnotationFailureGitHub(labels=["open source"])
        with self.assertRaises(RuntimeError) as context:
            run_close(github)
        detail = " ".join(str(context.exception).split())[:500]
        self.assertIn("bot-closed label", detail)
        self.assertIn("close comment", detail)
        self.assertEqual(github.pr["state"], "closed")

class ApplyMainTest(unittest.TestCase):
    def test_main_reports_each_controller_status(self) -> None:
        cases = (
            (
                "live",
                ControllerOutcome("triaged", 2, 1, 3),
                "3 teams; applied 1 owner labels",
            ),
            (
                "live",
                ControllerOutcome("incomplete", 2, 1, 3),
                "Applied incomplete Auto PR Triage",
            ),
            ("live", ControllerOutcome("closed"), "Closed pytorch/ciforge#123"),
            (
                "live",
                ControllerOutcome("kept_open"),
                "did not qualify for an apply action",
            ),
            (
                "live",
                ControllerOutcome("routed_untriaged", 2, 1, 3),
                "Applied partial Auto PR Triage",
            ),
            (
                "shadow",
                ControllerOutcome("shadow_close"),
                "Shadow result for pytorch/ciforge#123: would close",
            ),
            (
                "shadow",
                ControllerOutcome("shadow_triaged"),
                "Shadow result for pytorch/ciforge#123: would triage",
            ),
            (
                "shadow",
                ControllerOutcome("incomplete"),
                "Shadow result for pytorch/ciforge#123: analysis incomplete",
            ),
            (
                "shadow",
                ControllerOutcome("kept_open"),
                "Shadow result for pytorch/ciforge#123: would keep open",
            ),
            (
                "shadow",
                ControllerOutcome("routed_untriaged"),
                "would route known owners and leave untriaged",
            ),
        )
        for mode, outcome, message in cases:
            with self.subTest(status=outcome.status):
                args = argparse.Namespace(pr=123, repository=REPOSITORY, mode=mode)
                with (
                    mock.patch.dict(
                        "os.environ",
                        {"GH_TOKEN": "token"},
                    ),
                    mock.patch(
                        "apply_triage_decision.parse_args", return_value=args
                    ),
                    mock.patch("apply_triage_decision.GitHubClient") as client,
                    mock.patch(
                        "apply_triage_decision.apply_controller_action",
                        return_value=outcome,
                    ) as controller,
                    mock.patch("builtins.print") as output,
                ):
                    self.assertEqual(main(), 0)
                client.assert_called_once_with(mode, REPOSITORY, 123)
                controller.assert_called_once_with(args, client.return_value)
                self.assertIn(message, output.call_args.args[0])

    def test_main_reports_controller_failure(self) -> None:
        args = argparse.Namespace(pr=123, repository=REPOSITORY, mode="shadow")
        with (
            mock.patch.dict("os.environ", {"GH_TOKEN": "token"}),
            mock.patch("apply_triage_decision.parse_args", return_value=args),
            mock.patch("apply_triage_decision.GitHubClient") as client,
            mock.patch(
                "apply_triage_decision.apply_controller_action",
                side_effect=ValueError("action is not a mutation"),
            ) as controller,
            mock.patch("builtins.print"),
        ):
            self.assertEqual(main(), 1)
        client.assert_called_once_with("shadow", REPOSITORY, 123)
        controller.assert_called_once_with(args, client.return_value)

    def test_workflow_passes_current_controller_contract(self) -> None:
        action = (
            Path(__file__).parents[2] / ".github/actions/auto-pr-triage/action.yml"
        ).read_text()
        workflow = (
            Path(__file__).parents[2] / ".github/workflows/auto-pr-triage.yml"
        ).read_text()

        self.assertEqual(
            workflow.count("python3 scripts/auto_pr_triage/apply_triage_decision.py"),
            1,
        )
        self.assertIn("  repository:\n", action)
        self.assertIn("  analysis-result-json:\n", action)
        self.assertIn(
            "analysis-result-json: "
            "${{ steps.auto-pr-triage.outputs.analysis-result-json }}",
            workflow,
        )
        for argument in (
            "--analysis-result-json",
            "--author-login",
            "--mode",
            "--repository",
            "--run-attempt",
            "--workflow-sha",
        ):
            self.assertIn(argument, workflow)
        self.assertEqual(action.count('--repository "$REPOSITORY"'), 1)
        self.assertIn("repository: ${{ github.repository }}", workflow)
        self.assertIn('--repository "$REPOSITORY"', workflow)
        self.assertNotIn("pytorch/ciforge", action)
        self.assertNotIn("pytorch/ciforge", workflow)
        mode_lines = [
            line.strip()
            for line in workflow.splitlines()
            if line.strip().startswith("AUTO_PR_TRIAGE_MODE:")
        ]
        self.assertEqual(len(mode_lines), 1)
        self.assertIn(
            mode_lines[0],
            {"AUTO_PR_TRIAGE_MODE: shadow", "AUTO_PR_TRIAGE_MODE: live"},
        )
        self.assertIn('--mode "$AUTO_PR_TRIAGE_MODE"', workflow)
        self.assertIn(
            "needs.analyze.outputs.analysis-result-json != ''", workflow
        )
        self.assertNotIn("should-apply", workflow)
        self.assertNotIn("should-apply", action)
        self.assertNotIn("expected-head-sha", workflow)
        self.assertNotIn("expected-head-sha", action)
        self.assertIn("types: [labeled, ready_for_review]", workflow)
        self.assertIn(
            "  analyze:\n    if: >-\n"
            "      github.repository_owner == 'pytorch' &&",
            workflow,
        )
        self.assertIn("github.event.pull_request.state == 'open'", workflow)
        self.assertIn("github.event.pull_request.base.ref == 'main'", workflow)
        self.assertIn("!github.event.pull_request.draft", workflow)
        self.assertIn("github.event.action == 'ready_for_review'", workflow)
        self.assertIn(
            "contains(github.event.pull_request.labels.*.name, 'open source')",
            workflow,
        )
        self.assertIn("READY_FOR_REVIEW_EVENT", workflow)
        self.assertIn("pageInfo { hasPreviousPage }", workflow)
        self.assertIn('"$READY_EVENT_COUNT" == "1"', workflow)
        for removed in (
            "action-basis",
            "analysis-complete",
            "analyzed-head-sha",
            "analyzed-pr-text-sha256",
            "baseline-reviewers-json",
            "confidence",
            "extra-reviewers-json",
            "extra-teams-json",
            "additional-reviewers-json",
            "required-reviewers-json",
            "routing-assessment",
            "validation-failed",
            "has-actionable-linked-issue",
            "pulls-triage-bot-do-not-close",
            "human-review-file",
        ):
            self.assertNotIn(removed, action)
            self.assertNotIn(removed, workflow)
        self.assertNotIn(
            "value: ${{ steps.process_llm_output.outputs.workflow-sha }}", action
        )
        self.assertNotIn(
            "workflow-sha: ${{ steps.auto-pr-triage.outputs.workflow-sha }}", workflow
        )
        self.assertIn("WORKFLOW_SHA: ${{ github.sha }}", workflow)
        self.assertIn(
            "ANALYSIS_RESULT_JSON: "
            "${{ needs.analyze.outputs.analysis-result-json }}",
            workflow,
        )
        self.assertIn("  record-error:\n", workflow)
        self.assertIn("needs: [analyze, apply]", workflow)
        self.assertNotIn("  admit:\n", workflow)
        self.assertIn("needs.analyze.outputs.should-run == 'true'", workflow)
        self.assertIn("needs.analyze.result == 'failure'", workflow)
        self.assertIn("needs.apply.result == 'failure'", workflow)
        self.assertIn("needs.apply.result == 'skipped'", workflow)
        self.assertIn("pull-requests: write", workflow)
        self.assertIn("labels[]=bot-triage-error", workflow)
        self.assertIn("Unable to add bot-triage-error", workflow)
        self.assertIn(
            "group: auto-pr-triage-${{ github.event.pull_request.number }}",
            workflow,
        )
        self.assertEqual(action.count("continue-on-error: true"), 2)
        self.assertIn(
            "steps.prepare_llm_input.outputs.run-llm == 'true'",
            action,
        )
        self.assertIn("always() &&", action)
        self.assertIn("steps.prepare_llm_input.outcome == 'success'", action)

    def test_workflow_admits_only_first_eligible_entry_event(self) -> None:
        workflow = (
            Path(__file__).parents[2] / ".github/workflows/auto-pr-triage.yml"
        ).read_text()
        lines = workflow.splitlines()
        step_index = lines.index(
            "      - name: Admit the first eligible label or ready event"
        )
        run_index = lines.index("        run: |", step_index)
        script_lines = []
        for line in lines[run_index + 1 :]:
            if line.startswith("          "):
                script_lines.append(line[10:])
            elif not line:
                script_lines.append("")
            else:
                break
        script = "\n".join(script_lines)
        label = {"__typename": "LabeledEvent", "label": {"name": "open source"}}
        ready = {"__typename": "ReadyForReviewEvent"}
        cases = (
            ("labeled", [label], False, "true"),
            ("labeled", [label, label], False, "false"),
            ("ready_for_review", [ready, label, ready], False, "true"),
            ("ready_for_review", [label, ready, ready], False, "false"),
            ("ready_for_review", [ready], False, "false"),
            ("ready_for_review", [label, ready], True, "false"),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gh = root / "gh"
            history_path = root / "history.json"
            output_path = root / "github-output"
            gh.write_text('#!/bin/sh\nexec /bin/cat "$FAKE_HISTORY"\n')
            gh.chmod(0o755)
            for action, nodes, truncated, expected in cases:
                with self.subTest(action=action, nodes=nodes, truncated=truncated):
                    history_path.write_text(
                        json.dumps(
                            {
                                "data": {
                                    "repository": {
                                        "pullRequest": {
                                            "timelineItems": {
                                                "nodes": nodes,
                                                "pageInfo": {
                                                    "hasPreviousPage": truncated
                                                },
                                            }
                                        }
                                    }
                                }
                            }
                        )
                    )
                    output_path.write_text("")
                    environment = {
                        **os.environ,
                        "EVENT_ACTION": action,
                        "FAKE_HISTORY": str(history_path),
                        "GH_TOKEN": "token",
                        "GITHUB_OUTPUT": str(output_path),
                        "PATH": f"{root}:{os.environ['PATH']}",
                        "PR_NUMBER": "123",
                        "REPOSITORY": "pytorch/ciforge",
                    }
                    result = subprocess.run(
                        ["bash", "-eu", "-o", "pipefail", "-c", script],
                        capture_output=True,
                        text=True,
                        env=environment,
                        check=False,
                    )

                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(
                        output_path.read_text().splitlines(),
                        [f"should-run={expected}"],
                    )


if __name__ == "__main__":
    unittest.main()
