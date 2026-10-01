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
from schemas import Action, ActionPlan, RequestReviewers
from trusted_config import load_team_members


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ApplyOutcome:
    """Describe the result of applying a live plan."""

    status: str
    requested_users: int = 0


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

    subject = (
        f"{action.reason} reviewer request"
        if isinstance(action, RequestReviewers)
        else "label request"
    )
    return f"{subject} failed or returned ambiguously; it may already have landed"


def execute_action(pr: PullRequestRef, /, *, action: Action) -> None:
    """Perform one planned effect, raising on any failure or unconfirmed result."""

    if isinstance(action, RequestReviewers):
        pr.github.json(
            f"repos/{pr.repo}/pulls/{pr.number}/requested_reviewers",
            method="POST",
            payload={"reviewers": list(action.users)},
        )
    else:
        add_labels(pr, labels=action.labels)


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
    rosters = load_team_members(
        repository_root=REPOSITORY_ROOT, repo=args.repository, ref=args.workflow_sha
    )
    eligible = {
        member.removeprefix("@").casefold()
        for owner_id in context.additional_owners
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

    for action in plan.actions:
        try:
            execute_action(
                PullRequestRef(github=github, repo=args.repository, number=args.pr),
                action=action,
            )
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            detail = " ".join(str(exc).split())
            raise RuntimeError(f"{failure_message(action)}: {detail}") from exc

    return ApplyOutcome(
        "triaged" if plan.decision == "triage" else plan.decision,
        requested_users=sum(
            len(action.users)
            for action in plan.actions
            if isinstance(action, RequestReviewers) and action.reason == "owner_roster"
        ),
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
    counts = f"requested {outcome.requested_users} owner reviewers"
    if outcome.status == "missing_actionable_issue":
        print(f"Labeled {target} as missing an actionable issue.")
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
