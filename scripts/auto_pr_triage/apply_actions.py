#!/usr/bin/env python3
"""Apply a live Auto PR Triage plan produced by the analyze job."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from github_api import PullRequestRef
from identifiers import OWNER_LABEL_PREFIX
from schemas import (
    Action,
    ActionPlan,
    AddLabels,
    BOT_CLOSED_COMMENT_TEMPLATE,
    ClosePullRequest,
    RequestReviewers,
)
from trusted_config import load_team_members


BOT_CLOSED_COMMENT = """Auto PR Triage closed this PR because, when it analyzed the PR, the author did not have triage-or-higher repository access, the PR did not fix or say it is part of an issue in this repository labeled `actionable`, the description did not name a supporting maintainer with triage-or-higher access, and no triage-or-higher maintainer had taken qualifying activity on the PR.

To provide more routing context, update the PR description to do one of the following:

- Reference an issue in this repository labeled `actionable` that this PR fixes, such as `Fixes #123`, or an umbrella issue this PR is part of, such as `Part of #123`.
- Name the maintainer supporting this change, such as `Supported by @maintainer`.

If you believe this PR was closed by mistake, or you have added the missing context, please reopen it. While `bot-closed` remains on the PR, Auto PR Triage treats reopening as a human override and will not close it again."""  # noqa: B950
COMMENT_TEMPLATES = {BOT_CLOSED_COMMENT_TEMPLATE: BOT_CLOSED_COMMENT}
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ApplyOutcome:
    """Describe the result of applying a live plan."""

    status: str
    requested_users: int = 0
    owner_labels: int = 0
    requested_teams: int = 0


class GitHubClient:
    """Perform bounded GitHub API calls through a noninteractive gh process."""

    def __init__(self) -> None:
        """Snapshot the environment and disable interactive gh prompts."""

        self.env = os.environ.copy()
        self.env["GH_PROMPT_DISABLED"] = "1"

    def json(
        self,
        endpoint: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
    ) -> Any:
        """Execute one API request, optionally sending a JSON body via stdin."""

        command = ["gh", "api", endpoint]
        input_text = None
        if method != "GET":
            command.extend(["--method", method])
        if payload is not None:
            command.extend(["--input", "-"])
            input_text = json.dumps(payload, separators=(",", ":"))
        result = subprocess.run(
            command,
            input=input_text,
            text=True,
            capture_output=True,
            env=self.env,
            timeout=60,
            check=False,
        )
        if result.returncode:
            detail = " ".join((result.stderr or result.stdout).split())
            raise RuntimeError(
                f"GitHub API {method} {endpoint} failed"
                + (f": {detail}" if detail else "")
            )
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("GitHub API returned invalid JSON") from exc


def add_labels(
    pr: PullRequestRef,
    /,
    *,
    labels: tuple[str, ...],
) -> None:
    """Add labels and require GitHub to confirm every one."""

    response = pr.github.json(
        f"repos/{pr.repo}/issues/{pr.number}/labels",
        method="POST",
        payload={"labels": list(labels)},
    )
    returned = (
        {
            label["name"].casefold()
            for label in response
            if isinstance(label, dict) and isinstance(label.get("name"), str)
        }
        if isinstance(response, list)
        else set()
    )
    if not {label.casefold() for label in labels} <= returned:
        raise RuntimeError("labels were not confirmed")


def failure_message(action: Action) -> str:
    """Describe a failed action whose effect may still have landed."""

    if isinstance(action, RequestReviewers):
        subject = f"{action.reason} reviewer request"
    elif isinstance(action, ClosePullRequest):
        subject = "close request"
    elif isinstance(action, AddLabels):
        subject = "label request"
    else:
        subject = "comment"
    return f"{subject} failed or returned ambiguously; it may already have landed"


def execute_action(pr: PullRequestRef, /, *, action: Action) -> None:
    """Perform one planned effect, raising on any failure or unconfirmed result."""

    if isinstance(action, ClosePullRequest):
        closed = pr.github.json(
            f"repos/{pr.repo}/pulls/{pr.number}",
            method="PATCH",
            payload={"state": "closed"},
        )
        if not (
            isinstance(closed, dict)
            and closed.get("number") == pr.number
            and closed.get("state") == "closed"
        ):
            raise RuntimeError("close was not confirmed")
    elif isinstance(action, RequestReviewers):
        payload: dict[str, list[str]] = {}
        if action.users:
            payload["reviewers"] = list(action.users)
        if action.teams:
            payload["team_reviewers"] = list(action.teams)
        pr.github.json(
            f"repos/{pr.repo}/pulls/{pr.number}/requested_reviewers",
            method="POST",
            payload=payload,
        )
    elif isinstance(action, AddLabels):
        add_labels(pr, labels=action.labels)
    else:
        body = COMMENT_TEMPLATES[action.template]
        comment = pr.github.json(
            f"repos/{pr.repo}/issues/{pr.number}/comments",
            method="POST",
            payload={"body": body},
        )
        if not (
            isinstance(comment, dict)
            and type(comment.get("id")) is int
            and comment.get("body") == body
        ):
            raise RuntimeError("comment was not confirmed")


def check_plan_reviewers(*, args: argparse.Namespace, plan: ActionPlan) -> None:
    """Reject owner_roster reviewers outside the rosters of the plan's owners.

    ActionPlan checks every other request reason against its own source; roster
    membership is checked here, from team_members.json at the workflow SHA.
    """

    context = plan.context
    requested = {
        login.casefold()
        for action in plan.actions
        if isinstance(action, RequestReviewers) and action.reason == "owner_roster"
        for login in action.users
    }
    if not requested:
        return
    team_owner_ids = (
        *context.additional_owners,
        *(owner for owner in context.codepath_owners if not owner.startswith("@")),
    )
    rosters = load_team_members(
        repository_root=REPOSITORY_ROOT, repo=args.repository, ref=args.workflow_sha
    )
    eligible = {
        member.removeprefix("@").casefold()
        for owner_id in team_owner_ids
        for member in rosters["members"].get(owner_id, ())
    }
    if not requested <= eligible:
        raise ValueError("action plan requests a reviewer outside owner rosters")


def apply_action_plan(
    *,
    args: argparse.Namespace,
    github: GitHubClient,
) -> ApplyOutcome:
    """Execute one validated plan's actions in order."""

    plan = ActionPlan.from_json(args.action_plan_json)
    context = plan.context
    identity = context.identity
    target = (identity.repository, identity.number, identity.workflow_sha)
    if target != (args.repository, args.pr, args.workflow_sha):
        raise ValueError("action plan belongs to another pull request or run")
    # Rerunning only this job reuses an earlier attempt's plan, which may be
    # days old; require a fresh plan from rerunning all jobs instead.
    if context.run_attempt != args.run_attempt:
        raise ValueError(
            f"action plan is from attempt {context.run_attempt}; rerun all jobs"
        )
    check_plan_reviewers(args=args, plan=plan)

    # Once the PR is closed, report every failed annotation instead of stopping.
    closed = False
    annotation_errors: list[str] = []
    for action in plan.actions:
        try:
            execute_action(
                PullRequestRef(github=github, repo=args.repository, number=args.pr),
                action=action,
            )
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            detail = " ".join(str(exc).split())
            message = f"{failure_message(action)}: {detail}"
            if not closed:
                raise RuntimeError(message) from exc
            annotation_errors.append(message)
        closed |= isinstance(action, ClosePullRequest)
    if annotation_errors:
        raise RuntimeError(
            "pull request was closed, but annotations were incomplete: "
            + "; ".join(annotation_errors)
        )

    if plan.decision == "close":
        return ApplyOutcome("closed")
    requests = [a for a in plan.actions if isinstance(a, RequestReviewers)]
    owners = [a for a in requests if a.reason in {"codepath_owner", "owner_roster"}]
    label_actions = [a for a in plan.actions if isinstance(a, AddLabels)]
    labels = [label for a in label_actions for label in a.labels]
    return ApplyOutcome(
        "triaged" if plan.decision == "triage" else plan.decision,
        requested_users=sum(len(a.users) for a in owners),
        owner_labels=sum(label.startswith(OWNER_LABEL_PREFIX) for label in labels),
        requested_teams=sum(len(a.teams) for a in owners),
    )


def parse_args() -> argparse.Namespace:
    """Parse the live plan and event identity."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pr", type=int, help="pull request number")
    parser.add_argument("--repository", required=True)
    parser.add_argument("--workflow-sha", required=True)
    parser.add_argument("--run-attempt", type=int, required=True)
    parser.add_argument("--action-plan-json", required=True)
    return parser.parse_args()


def main() -> int:
    """Run the live plan executor CLI."""

    args = parse_args()
    if not os.environ.get("GH_TOKEN"):
        print("Auto PR Triage apply failed: GH_TOKEN is unavailable.", file=sys.stderr)
        return 1
    try:
        outcome = apply_action_plan(args=args, github=GitHubClient())
    except Exception as exc:
        detail = " ".join(str(exc).split())
        print(
            f"Auto PR Triage apply failed: {type(exc).__name__}: {detail}",
            file=sys.stderr,
        )
        return 1

    target = f"{args.repository}#{args.pr}"
    counts = (
        f"requested {outcome.requested_users} users and "
        f"{outcome.requested_teams} teams; applied "
        f"{outcome.owner_labels} owner labels"
    )
    if outcome.status == "closed":
        print(f"Closed {target}: repository policy was not met.")
    elif outcome.status == "kept_open":
        print(f"{target} did not qualify for an apply action; kept open.")
    elif outcome.status == "incomplete":
        print(f"Applied incomplete Auto PR Triage to {target}: {counts}.")
    elif outcome.status == "routed_untriaged":
        print(
            f"Applied partial Auto PR Triage to {target}: {counts}; "
            "left untriaged for human routing."
        )
    else:
        print(f"Applied Auto PR Triage to {target}: {counts}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
