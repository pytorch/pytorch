"""Shared fixtures for the planning and live apply tests."""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock
from urllib.parse import unquote

from apply_actions import apply_action_plan, ApplyOutcome
from plan_actions import build_action_plan, gather_planner_input, log_plan
from schemas import (
    ActionPlan,
    AdditionalOwnerConcern,
    IntakeFacts,
    IntakeResult,
    OwnershipGroup,
    OwnershipResult,
    PullRequestIdentity,
)
from tests.stage_fixtures import discarded_bypass_concern, uncovered_concern


WORKFLOW_SHA = "a" * 40
HEAD_SHA = "b" * 40
REPOSITORY = "pytorch/ciforge"


def ownership_config() -> dict[str, Any]:
    return {
        "extra_ownership_metadata": {
            "owners": {
                "autograd": {"description": "Owns autograd behavior."},
                "compiler": {"description": "Owns compiler behavior."},
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


def bypass_intake_match(file: str) -> dict[str, Any]:
    return {
        "criteria_quote": "PRs that fix incorrect gradients.",
        "rationale": ["The change fixes a gradient formula the team asked to see."],
        "evidence": [
            {
                "file": file,
                "diff_excerpt": "+new semantic behavior",
                "relevance": "This line fixes the gradient the team asked to see.",
            }
        ],
    }


def stage_results(
    *,
    author_login: str = "external-author",
    is_open_non_draft_pr_against_main: bool = True,
    is_opted_out: bool = False,
    author_has_triage_permission: bool = False,
    has_actionable_linked_issue: bool | None = None,
    has_maintainer_activity: bool = False,
    llm_run_status: str = "succeeded",
    codepath_owners: tuple[str, ...] = ("@codepath-owner",),
    additional_owners: tuple[str, ...] = (),
    additional_owner_concerns: tuple[AdditionalOwnerConcern, ...] | None = None,
    has_uncovered_concerns: bool = False,
    has_related_actionable_issue: bool = False,
    supporters: tuple[str, ...] = (),
    actionable_labelers: tuple[str, ...] = (),
    maintainer_requested_reviewers: tuple[str, ...] = (),
    bypass_intake_matches: tuple[str, ...] = (),
    has_discarded_bypass_intake_match: bool = False,
    labels: tuple[str, ...] = (),
) -> tuple[IntakeResult, OwnershipResult]:
    if additional_owner_concerns is None:
        additional_owner_concerns = tuple(
            AdditionalOwnerConcern.from_dict(
                {
                    "concern": {
                        "description": f"{owner} owns this changed behavior.",
                        "files": ["torch/semantic.py"],
                        "evidence": [
                            {
                                "file": "torch/semantic.py",
                                "diff_excerpt": "+new semantic behavior",
                                "relevance": "This line implements the owned behavior.",
                            }
                        ],
                    },
                    "owner_id": owner,
                    "rationale": [
                        "The changed behavior falls within this ownership area.",
                        "The configured description names the affected contract.",
                        "A review from this team covers the semantic concern.",
                    ],
                    "confidence": "high",
                    "bypass_intake_match": (
                        bypass_intake_match("torch/semantic.py")
                        if owner in bypass_intake_matches
                        else None
                    ),
                }
            )
            for owner in additional_owners
        )
    if has_actionable_linked_issue is None:
        # Facts are gathered only for an active PR.
        has_actionable_linked_issue = (
            is_open_non_draft_pr_against_main and not is_opted_out
        )
    identity = PullRequestIdentity(REPOSITORY, 123, HEAD_SHA, WORKFLOW_SHA)
    facts = IntakeFacts(
        is_open_non_draft_pr_against_main=is_open_non_draft_pr_against_main,
        is_opted_out=is_opted_out,
        author_has_triage_permission=author_has_triage_permission,
        has_actionable_linked_issue=has_actionable_linked_issue,
        has_maintainer_activity=has_maintainer_activity,
        has_supporter=bool(supporters),
        has_related_actionable_issue=has_related_actionable_issue,
        supporters=supporters,
        actionable_labelers=actionable_labelers,
        maintainer_requested_reviewers=maintainer_requested_reviewers,
    )
    # Codepath owners own torch/file.py through one path_handles group.
    groups = (
        {"core": OwnershipGroup(tuple(codepath_owners), ("torch/file.py",), "path_handles")}
        if codepath_owners
        else {}
    )
    files_by_owner: dict[str, set[str]] = {}
    for group in groups.values():
        for owner in group.owners:
            files_by_owner.setdefault(owner, set()).update(group.files)
    ownership = OwnershipResult.create(
        llm_run_status=llm_run_status,
        codepath_owners=files_by_owner,
        ownership_groups=groups,
        additional_owner_concerns=additional_owner_concerns,
        discarded_additional_owner_concerns=(
            [discarded_bypass_concern()] if has_discarded_bypass_intake_match else []
        ),
        uncovered_concerns=[uncovered_concern()] if has_uncovered_concerns else [],
    )
    intake = IntakeResult(
        identity, facts, author_login, "fixture", "fixture body", labels
    )
    return intake, ownership


def run_args(**overrides: Any) -> argparse.Namespace:
    values = {
        "pr": 123,
        "repository": REPOSITORY,
        "workflow_sha": WORKFLOW_SHA,
        "run_attempt": 1,
        "stage_results": stage_results(),
        "github_step_summary": None,
        "action_plan_json": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def plan_pr(*, args: argparse.Namespace, github: FakeGitHub) -> ActionPlan:
    intake, ownership = args.stage_results
    planner_input = gather_planner_input(
        args=args, intake=intake, ownership=ownership, github=github
    )
    plan, why = build_action_plan(planner_input)
    if why is not None:
        log_plan(plan=plan, why=why, summary=args.github_step_summary)
    return plan


def plan_and_apply(*, args: argparse.Namespace, github: FakeGitHub) -> ApplyOutcome:
    args.action_plan_json = plan_pr(args=args, github=github).to_json()
    return apply_action_plan(args=args, github=github)


@contextlib.contextmanager
def configured(*, team_members: dict[str, Any] | None = None) -> Iterator[None]:
    members = team_members or ownership_config()["team_members"]
    with (
        mock.patch("plan_actions.load_team_members", return_value=members),
        mock.patch("apply_actions.load_team_members", return_value=members),
    ):
        yield


def scenario_args(
    github: FakeGitHub,
    *,
    codepath_owners: tuple[str, ...] | None = None,
    additional_owners: tuple[str, ...] = (),
    llm_run_status: str = "succeeded",
    run_attempt: int = 1,
    github_step_summary: Path | None = None,
    additional_owner_concerns: tuple[AdditionalOwnerConcern, ...] | None = None,
    has_uncovered_concerns: bool = False,
    supporters: tuple[str, ...] = (),
    has_related_actionable_issue: bool = False,
    actionable_labelers: tuple[str, ...] = (),
    maintainer_requested_reviewers: tuple[str, ...] = (),
    passes_intake: bool = True,
    bypass_intake_matches: tuple[str, ...] = (),
    has_discarded_bypass_intake_match: bool = False,
) -> argparse.Namespace:
    if codepath_owners is None:
        codepath_owners = ("@codepath-owner",)
    return run_args(
        run_attempt=run_attempt,
        github_step_summary=github_step_summary,
        stage_results=stage_results(
            has_actionable_linked_issue=passes_intake and not supporters,
            supporters=supporters,
            has_related_actionable_issue=has_related_actionable_issue,
            actionable_labelers=actionable_labelers,
            maintainer_requested_reviewers=maintainer_requested_reviewers,
            llm_run_status=llm_run_status,
            codepath_owners=codepath_owners,
            additional_owners=additional_owners,
            additional_owner_concerns=additional_owner_concerns,
            has_uncovered_concerns=has_uncovered_concerns,
            bypass_intake_matches=bypass_intake_matches,
            has_discarded_bypass_intake_match=has_discarded_bypass_intake_match,
        ),
    )


def run_apply(
    github: FakeGitHub, *, team_members: dict[str, Any] | None = None, **scenario: Any
) -> ApplyOutcome:
    args = scenario_args(github, **scenario)
    with configured(team_members=team_members):
        return plan_and_apply(args=args, github=github)


def run_plan(
    github: FakeGitHub, *, team_members: dict[str, Any] | None = None, **scenario: Any
) -> ActionPlan:
    args = scenario_args(github, **scenario)
    with configured(team_members=team_members):
        return plan_pr(args=args, github=github)


def unadmitted_args(run_attempt: int = 1) -> argparse.Namespace:
    return run_args(
        run_attempt=run_attempt,
        stage_results=stage_results(
            has_actionable_linked_issue=False, codepath_owners=()
        ),
    )


def run_unadmitted(github: FakeGitHub, *, run_attempt: int = 1) -> ApplyOutcome:
    return plan_and_apply(args=unadmitted_args(run_attempt), github=github)


def run_without_owners(
    github: FakeGitHub,
    *,
    llm_run_status: str = "succeeded",
    has_uncovered_concerns: bool = False,
) -> ApplyOutcome:
    args = run_args(
        stage_results=stage_results(
            llm_run_status=llm_run_status,
            codepath_owners=(),
            has_uncovered_concerns=has_uncovered_concerns,
        ),
    )
    with configured():
        return plan_and_apply(args=args, github=github)


def plan_facts(**overrides: Any) -> IntakeFacts:
    return stage_results(**overrides)[0].facts


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
        previously_requested: list[str] | None = None,
        without_push: list[str] | None = None,
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
        # Logins with a review_requested event in the timeline.
        self.previously_requested = list(previously_requested or [])
        # Logins other than the author who lack push access.
        self.without_push = {login.casefold() for login in without_push or []}
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
        if endpoint.startswith("repos/pytorch/ciforge/labels/") and method == "GET":
            label = unquote(endpoint.rsplit("/", 1)[-1])
            return {} if label in self.unavailable_labels else {"name": label}
        if endpoint == "repos/pytorch/ciforge/pulls/123/requested_reviewers":
            if method == "POST":
                if not isinstance(payload, dict) or set(payload) != {"reviewers"}:
                    raise AssertionError("unexpected reviewer payload")
                self.requested_users.update(payload["reviewers"])
            else:
                self.live_reads.append("requested")
            return {
                "users": [{"login": login} for login in sorted(self.requested_users)],
                "teams": [{"slug": slug} for slug in sorted(self.requested_teams)],
            }
        if endpoint == "repos/pytorch/ciforge/issues/123/labels" and method == "POST":
            if not isinstance(payload, dict) or not isinstance(
                payload.get("labels"), list
            ):
                raise AssertionError("unexpected label payload")
            existing = {label["name"] for label in self.pr["labels"]}
            self.pr["labels"].extend(
                {"name": name} for name in payload["labels"] if name not in existing
            )
            return copy.deepcopy(self.pr["labels"])
        if endpoint.startswith("repos/pytorch/ciforge/issues/123/labels/") and method == "DELETE":
            label = unquote(endpoint.rsplit("/", 1)[-1])
            self.pr["labels"] = [item for item in self.pr["labels"] if item["name"] != label]
            return copy.deepcopy(self.pr["labels"])
        if endpoint.startswith("repos/pytorch/ciforge/issues/123/timeline?"):
            self.live_reads.append("timeline")
            if not endpoint.endswith("&page=1"):
                return []
            return [
                {"event": "review_requested", "requested_reviewer": {"login": login}}
                for login in self.previously_requested
            ]
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
        if endpoint.endswith("/permission") and "/collaborators/" in endpoint:
            login = unquote(endpoint.split("/collaborators/", 1)[1].rsplit("/", 1)[0])
            self.live_reads.append("push_permission")
            push = login.casefold() not in self.without_push
            permissions = {"triage": push, "push": push, "maintain": False, "admin": False}
            return {"user": {"login": login, "permissions": permissions}}
        raise AssertionError(f"unexpected request: {method} {endpoint}")

    def graphql(self, *, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        if variables["number"] != 123:
            raise AssertionError("unexpected pull request number")
        if "closingIssuesReferences" in query:
            self.actionable_checks += 1
            self.live_reads.append("actionable")
            nodes: list[dict[str, Any]] = []
            if self.actionable_issue:
                nodes.append(
                    {
                        "repository": {"nameWithOwner": "pytorch/ciforge"},
                        "labels": {
                            "nodes": [{"id": "actionable-label", "name": "actionable"}],
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
    return [call for call in github.calls if call[0] in {"POST", "PATCH", "DELETE"}]


def printed_record(*, output: mock.Mock, label: str) -> dict[str, Any]:
    prefix = f"{label}:\n"
    return next(
        json.loads(call.args[0].removeprefix(prefix))
        for call in output.call_args_list
        if call.args[0].startswith(prefix)
    )


def printed_plan(output: mock.Mock) -> dict[str, Any]:
    return printed_record(output=output, label="Auto PR Triage plan")


def printed_reviewer_routing(output: mock.Mock) -> str:
    prefix = "Auto PR Triage reviewer routing:\n"
    return next(
        call.args[0].removeprefix(prefix)
        for call in output.call_args_list
        if call.args[0].startswith(prefix)
    )
