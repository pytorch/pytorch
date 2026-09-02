#!/usr/bin/env python3
"""Derive and apply a live or shadow Auto PR Triage decision."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from codepath_owners import REPOSITORY_RE
from github_reviews import (
    fetch_round_robin_reviewers,
    fetch_requested_codeowner_handles,
    fetch_requested_reviewer_handles,
    fetch_submitted_review_state,
)
from ownership import (
    SHA_RE,
    USER_HANDLE_RE,
    all_team_members,
    load_team_members,
    owner_label,
)
from schemas import AnalysisResult, should_run_ownership_analysis


TRIAGED_LABEL = "triaged"
BOT_TRIAGED_LABEL = "bot-triaged"
BOT_TRIAGE_ERROR_LABEL = "bot-triage-error"
BOT_CLOSED_LABEL = "bot-closed"
BOT_SHADOW_CLOSE_LABEL = "bot-shadow-close"
BOT_SHADOW_TRIAGED_LABEL = "bot-shadow-triaged"
CODEOWNERS_SHADOW_LABELS = {
    "match": "bot-codeowners-shadow-match",
    "mismatch": "bot-codeowners-shadow-mismatch",
    "inconclusive": "bot-codeowners-shadow-inconclusive",
}
ALLOWED_SHADOW_LABELS = frozenset(
    {
        BOT_TRIAGE_ERROR_LABEL,
        BOT_SHADOW_CLOSE_LABEL,
        BOT_SHADOW_TRIAGED_LABEL,
        *CODEOWNERS_SHADOW_LABELS.values(),
    }
)
GRAPHQL_QUERY_RE = re.compile(r"\s*query(?:\s|\()")
# Keep GitHub's native CODEOWNERS requests authoritative during the rollout.
NATIVE_CODEOWNERS_SHADOW = True
MAX_REVIEW_REQUESTS = 15
BOT_CLOSED_COMMENT = """Auto PR Triage closed this PR because, when it analyzed the PR, the author did not have triage-or-higher repository access, the PR did not link to an issue in this repository labeled `actionable`, and no triage-or-higher maintainer had already reviewed, commented, or requested themselves for review.

To provide more routing context:

- Link this PR using a closing reference such as `Fixes #123` to an issue in this repository labeled `actionable`.

If you believe this PR was closed by mistake, or you have added the missing context, please reopen it. While `bot-closed` remains on the PR, Auto PR Triage treats reopening as a human override and will not close it again."""  # noqa: B950
AUTHOR_LOGIN_RE = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:\[bot\])?"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def validate_github_request(
    endpoint: str,
    method: str,
    payload: dict[str, Any] | None,
    repository: str,
    number: int,
) -> None:
    """Restrict this controller to reads and fixed shadow-label writes."""

    if method == "GET":
        if payload is not None:
            raise ValueError("GitHub GET request cannot carry a payload")
        return
    if method != "POST" or not isinstance(payload, dict):
        raise ValueError("GitHub request is outside the shadow-mode boundary")
    if endpoint == "graphql":
        if (
            set(payload) != {"query", "variables"}
            or not isinstance(payload["query"], str)
            or not GRAPHQL_QUERY_RE.match(payload["query"])
            or not isinstance(payload["variables"], dict)
        ):
            raise ValueError("GraphQL request is not a read-only query")
        return
    labels = payload.get("labels")
    if (
        endpoint != f"repos/{repository}/issues/{number}/labels"
        or set(payload) != {"labels"}
        or not isinstance(labels, list)
        or not labels
        or any(not isinstance(label, str) for label in labels)
        or not set(labels) <= ALLOWED_SHADOW_LABELS
    ):
        raise ValueError("GitHub request is outside the shadow-mode boundary")


@dataclass(frozen=True)
class CodepathOwnerTargets:
    """Immutable codepath-owner handles and internal owner IDs."""

    values: tuple[str, ...]

    @property
    def github_users(self) -> tuple[str, ...]:
        """Return direct GitHub user logins in the policy."""

        return tuple(
            owner[1:]
            for owner in self.values
            if owner.startswith("@") and "/" not in owner
        )

    @property
    def github_teams(self) -> tuple[str, ...]:
        """Return direct target-organization GitHub team slugs."""

        return tuple(
            owner.split("/", 1)[1]
            for owner in self.values
            if owner.startswith("@") and "/" in owner
        )

    @property
    def github_handles(self) -> set[str]:
        """Return canonical direct GitHub handles."""

        return {owner.casefold() for owner in self.values if owner.startswith("@")}

    @property
    def owner_ids(self) -> tuple[str, ...]:
        """Return internal owner IDs resolved through the trusted roster."""

        return tuple(owner for owner in self.values if not owner.startswith("@"))


@dataclass(frozen=True)
class ControllerOutcome:
    """Describe the result of applying a live or shadow decision."""

    status: str
    requested_users: int = 0
    owner_labels: int = 0
    requested_teams: int = 0


@dataclass(frozen=True)
class ReviewerState:
    """Reviewer state collected once from separate, potentially stale reads."""

    requested_reviewers: frozenset[str]
    submitted_reviewers: frozenset[str]

    @property
    def current_reviewers(self) -> frozenset[str]:
        """Return handles with a pending request or submitted review."""

        return self.requested_reviewers | self.submitted_reviewers


def sanitize_log_text(value: str) -> str:
    """Escape GitHub workflow-command markers in untrusted log text."""

    return value.replace("::", r"\u003a\u003a").replace(
        "##[", r"\u0023\u0023["
    )


def escape_log_fragment(value: str) -> str:
    """Escape control characters without adding JSON string delimiters."""

    return json.dumps(value, ensure_ascii=True)[1:-1]


def log_json_record(label: str, record: dict[str, Any]) -> None:
    """Print a sanitized, readable JSON log record."""

    serialized = json.dumps(record, indent=2, sort_keys=True)
    print(f"{label}:\n{sanitize_log_text(serialized)}")


def reviewer_choice_reason(owner: str, choice: dict[str, Any]) -> str:
    """Describe why one owner resolved to one reviewer."""

    reviewer = choice["reviewer"]
    state = choice["state"]
    if state == "selected":
        reasons = {
            "round_robin_initial": (
                f"No previous `{owner}` assignment was found, so its round-robin "
                f"rotation began with {reviewer}."
            ),
            "round_robin_next": (
                f"{reviewer} was the next eligible member of `{owner}`'s "
                "round-robin rotation."
            ),
            "stable_fallback": (
                f"The latest `{owner}` marker had no attributable current-roster "
                f"assignment, so the stable fallback chose {reviewer}."
            ),
            "direct_codepath_owner": (
                f"The checked-in codepath-owner rules directly name {reviewer}."
            ),
        }
        try:
            return reasons[choice["selection_reason"]]
        except KeyError as exc:
            raise ValueError("owner choice has an invalid selection reason") from exc
    reasons = {
        "native_codeowner": (
            f"GitHub already has an active native CODEOWNERS request for {reviewer}."
        ),
        "codepath_covered": (
            f"{reviewer} covers `{owner}` through codepath ownership; no `{owner}` "
            "round-robin choice was needed."
        ),
        "submitted": (
            f"{reviewer} already submitted a review covering `{owner}`; no new "
            "request is needed."
        ),
        "pending": (
            f"{reviewer} already has a pending request covering `{owner}`; no new "
            "request is needed."
        ),
    }
    try:
        return reasons[state]
    except KeyError as exc:
        raise ValueError("owner choice has an invalid state") from exc


def log_reviewer_routing(
    mode: str,
    owner_choices: dict[str, dict[str, Any]],
    planned_reviewer_requests: tuple[str, ...],
) -> None:
    """Explain owner resolution from each reviewer's perspective."""

    if not owner_choices:
        return
    grouped: dict[str, dict[str, Any]] = {}
    for owner, choice in owner_choices.items():
        reviewer = choice["reviewer"]
        group = grouped.setdefault(
            reviewer.casefold(), {"reviewer": reviewer, "choices": []}
        )
        group["choices"].append((owner, choice))
    planned = {reviewer.casefold() for reviewer in planned_reviewer_requests}
    blocks: list[str] = []
    for group in sorted(grouped.values(), key=lambda item: item["reviewer"].casefold()):
        reviewer = group["reviewer"]
        choices = sorted(
            group["choices"],
            key=lambda item: (
                item[1]["provenance"]["source"] != "codepath",
                item[0].casefold(),
            ),
        )
        reasons = [reviewer_choice_reason(owner, choice) for owner, choice in choices]
        lines = [f"Reviewer {reviewer}", "Why this reviewer: " + " ".join(reasons)]
        for owner, choice in choices:
            provenance = choice["provenance"]
            source = provenance["source"]
            if provenance.get("truncated"):
                lines.append(
                    f"{source.capitalize()} owner `{owner}`: ownership evidence was "
                    "omitted because the analysis-to-apply record exceeded its size "
                    "limit; routing was unchanged."
                )
                continue
            files = ", ".join(
                f"`{escape_log_fragment(path)}`" for path in provenance["files"]
            )
            omitted = provenance["total_file_count"] - len(provenance["files"])
            if omitted:
                files += f" (+{omitted} more)"
            if source == "codepath":
                lines.append(
                    f"Codepath owner `{owner}` matched supporting files: {files}."
                )
                continue
            justification = provenance["llm_justification"]
            lines.append(
                f"Semantic owner `{owner}`: "
                + escape_log_fragment(justification["owned_concern"])
            )
            lines.append(
                "Reasoning: "
                + " ".join(
                    escape_log_fragment(reason)
                    for reason in justification["rationale"]
                )
            )
            lines.append(f"Supporting files: {files}.")
            lines.append("Evidence:")
            for evidence in justification["evidence"]:
                lines.append(
                    f"- `{escape_log_fragment(evidence['file'])}`: "
                    + escape_log_fragment(evidence["relevance"])
                )
                lines.extend(
                    f"    {escape_log_fragment(line)}"
                    for line in evidence["diff_excerpt"].splitlines()
                )
        if reviewer.casefold() not in planned:
            effect = "no new Auto PR Triage reviewer request is needed."
        elif mode == "shadow":
            effect = (
                f"shadow mode logged one planned request for {reviewer}; "
                "no review request was sent."
            )
        else:
            effect = f"live mode plans one deduplicated request for {reviewer}."
        lines.append(f"Planned effect: {effect}")
        blocks.append("\n".join(lines))
    print(
        "Auto PR Triage reviewer routing:\n\n"
        + sanitize_log_text("\n\n".join(blocks))
    )


def validate_identity(args: argparse.Namespace) -> None:
    """Validate event-derived identity before any GitHub I/O."""

    if (
        args.pr < 1
        or not REPOSITORY_RE.fullmatch(args.repository)
        or not SHA_RE.fullmatch(args.workflow_sha)
        or not AUTHOR_LOGIN_RE.fullmatch(args.author_login)
        or args.run_attempt < 1
    ):
        raise ValueError("invalid pull request identity")


class GitHubClient:
    """Perform bounded GitHub API calls through a noninteractive gh process."""

    def __init__(self, mode: str, repository: str, number: int) -> None:
        """Snapshot the environment and disable interactive gh prompts."""

        if mode not in {"shadow", "live"}:
            raise ValueError("invalid Auto PR Triage mode")
        if not REPOSITORY_RE.fullmatch(repository) or number < 1:
            raise ValueError("invalid target repository or pull request")
        self.mode = mode
        self.repository = repository
        self.number = number
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

        if self.mode == "shadow":
            validate_github_request(
                endpoint,
                method,
                payload,
                self.repository,
                self.number,
            )
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
            detail = " ".join((result.stderr or result.stdout).split())[:500]
            raise RuntimeError(
                f"GitHub API {method} {endpoint} failed"
                + (f": {detail}" if detail else "")
            )
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("GitHub API returned invalid JSON") from exc

    def graphql(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        """Execute a parameterized GraphQL query and reject reported errors."""

        response = self.json(
            "graphql",
            method="POST",
            payload={"query": query, "variables": variables},
        )
        if not isinstance(response, dict):
            raise RuntimeError("GitHub GraphQL response is invalid")
        if response.get("errors"):
            raise RuntimeError("GitHub GraphQL reported an error")
        data = response.get("data")
        if not isinstance(data, dict):
            raise RuntimeError("GitHub GraphQL response has no data")
        return data


def require_repository_label(
    github: GitHubClient,
    repository: str,
    label_name: str,
) -> None:
    """Require one configured repository label before attempting a mutation."""

    encoded = quote(label_name, safe="")
    label = github.json(f"repos/{repository}/labels/{encoded}")
    actual_name = label.get("name") if isinstance(label, dict) else None
    if not isinstance(actual_name, str) or actual_name.casefold() != label_name.casefold():
        raise RuntimeError(f"required repository label is unavailable: {label_name}")


def add_labels(
    github: GitHubClient,
    repository: str,
    number: int,
    labels: tuple[str, ...],
) -> None:
    """Add labels and require GitHub to confirm every one."""

    response = github.json(
        f"repos/{repository}/issues/{number}/labels",
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


def record_bot_close(github: GitHubClient, repository: str, number: int) -> None:
    """Add fixed provenance after a confirmed bot close."""

    errors: list[str] = []
    try:
        add_labels(github, repository, number, (BOT_CLOSED_LABEL,))
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        detail = " ".join(str(exc).split())[:120]
        errors.append(
            "bot-closed label may already be present"
            + (f": {detail}" if detail else "")
        )

    try:
        comment = github.json(
            f"repos/{repository}/issues/{number}/comments",
            method="POST",
            payload={"body": BOT_CLOSED_COMMENT},
        )
        valid_comment = (
            isinstance(comment, dict)
            and isinstance(comment.get("id"), int)
            and not isinstance(comment.get("id"), bool)
            and comment.get("body") == BOT_CLOSED_COMMENT
        )
        if not valid_comment:
            errors.append("close comment was not confirmed")
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        detail = " ".join(str(exc).split())[:120]
        errors.append(
            "close comment may already be present"
            + (f": {detail}" if detail else "")
        )

    if errors:
        raise RuntimeError(
            "pull request was closed, but annotations were incomplete: "
            + "; ".join(errors)
        )


def normalize_requested_reviewer_handles(
    handles: set[str],
    repository: str,
) -> frozenset[str]:
    """Normalize target-repository user and team review requests."""

    target_org = repository.split("/", 1)[0]
    target_team_handle_re = re.compile(
        rf"@{re.escape(target_org)}/[A-Za-z0-9](?:[A-Za-z0-9_-]*[A-Za-z0-9])?",
        re.IGNORECASE,
    )
    return frozenset(
        handle.casefold()
        for handle in handles
        if USER_HANDLE_RE.fullmatch(handle)
        or target_team_handle_re.fullmatch(handle)
    )


def normalize_user_handles(handles: set[str]) -> frozenset[str]:
    """Normalize GitHub user handles and ignore teams."""

    return frozenset(
        handle.casefold() for handle in handles if USER_HANDLE_RE.fullmatch(handle)
    )


def find_submitted_handoff(
    team_members: dict[str, Any],
    submitted_reviewers: frozenset[str],
    author_handle: str,
) -> str | None:
    """Return one eligible non-author reviewer who already submitted a review."""

    eligible = {member.casefold() for member in all_team_members(team_members)}
    return min((submitted_reviewers & eligible) - {author_handle}, default=None)


def compare_codeowners_shadow(
    github: GitHubClient,
    args: argparse.Namespace,
    codepath: CodepathOwnerTargets,
) -> dict[str, Any]:
    """Compare custom path resolution with active native CODEOWNERS requests."""

    author = f"@{args.author_login}".casefold()
    expected = {
        owner.casefold()
        for owner in codepath.values
        if owner.startswith("@") and owner.casefold() != author
    }
    try:
        if codepath.owner_ids:
            raise RuntimeError(
                "shadow codepath-owner policy contains internal owner IDs"
            )
        observed = {
            owner.casefold()
            for owner in fetch_requested_codeowner_handles(
                github, args.repository, args.pr
            )
        }
    except (RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        comparison = {
            "status": "inconclusive",
            "oracle": "active_review_requests_as_code_owner",
            "workflow_sha": args.workflow_sha,
            "expected": sorted(expected),
            "observed": [],
            "missing_from_github": [],
            "unexpected_from_github": [],
            "error": f"{type(exc).__name__}: {str(exc)[:500]}",
        }
    else:
        comparison = {
            "status": "match" if expected == observed else "mismatch",
            "oracle": "active_review_requests_as_code_owner",
            "workflow_sha": args.workflow_sha,
            "expected": sorted(expected),
            "observed": sorted(observed),
            "missing_from_github": sorted(expected - observed),
            "unexpected_from_github": sorted(observed - expected),
            "error": None,
        }
    log_json_record("Auto PR Triage CODEOWNERS shadow", comparison)
    return comparison


def fetch_reviewer_state(
    github: GitHubClient,
    repository: str,
    number: int,
) -> ReviewerState:
    """Fetch requested and submitted reviewers once for this apply attempt.

    SECURITY POLICY: Auto PR Triage deliberately does not refresh this state.
    The separate reads are not atomic. Concurrent review activity can cause a
    stale request or label; those bounded effects are accepted and reversible.
    """

    requested = normalize_requested_reviewer_handles(
        fetch_requested_reviewer_handles(github, repository, number),
        repository,
    )
    submitted = normalize_user_handles(
        set(fetch_submitted_review_state(github, repository, number))
    )
    return ReviewerState(
        requested_reviewers=requested,
        submitted_reviewers=submitted,
    )


def select_owner_reviewers(
    github: GitHubClient,
    repository: str,
    number: int,
    team_members: dict[str, Any],
    owner_ids: tuple[str, ...],
    codepath: CodepathOwnerTargets,
    author_handle: str,
    reviewers: ReviewerState,
) -> tuple[dict[str, dict[str, Any]], tuple[str, ...], tuple[str, ...]]:
    """Select reviewers for internal owner IDs from one state snapshot."""

    rosters = team_members["members"]
    if not set(owner_ids) <= set(rosters):
        raise ValueError("owner is not configured")

    choices: dict[str, dict[str, Any]] = {}
    needs_round_robin: dict[str, dict[str, Any]] = {}
    for owner_id in owner_ids:
        roster = rosters[owner_id]
        members = {member.casefold(): member for member in roster}
        covered = (codepath.github_handles & set(members)) - {author_handle}
        if covered:
            choices[owner_id] = {
                "reviewer": min(covered),
                "state": "codepath_covered",
            }
            continue

        submitted = (set(members) & reviewers.submitted_reviewers) - {
            author_handle
        }
        if submitted:
            key = min(submitted)
            choices[owner_id] = {
                "reviewer": members[key],
                "state": "submitted",
            }
            continue

        pending = (set(members) & reviewers.requested_reviewers) - {
            author_handle
        }
        if pending:
            key = min(pending)
            choices[owner_id] = {
                "reviewer": members[key],
                "state": "pending",
            }
            continue
        needs_round_robin[owner_id] = {
            "label": owner_label(owner_id),
            "members": roster,
        }

    selected = (
        fetch_round_robin_reviewers(
            github,
            repository,
            number,
            needs_round_robin,
            {author_handle},
        )
        if needs_round_robin
        else {}
    )
    if set(selected) != set(needs_round_robin):
        raise RuntimeError("owner has no eligible round-robin reviewer")
    for owner_id, selection in selected.items():
        choices[owner_id] = {
            "reviewer": selection["reviewer"],
            "state": "selected",
            "selection_reason": selection["selection_reason"],
        }

    request_users = tuple(
        sorted(
            {
                choice["reviewer"][1:]
                for choice in choices.values()
                if choice["state"] == "selected"
            },
            key=str.casefold,
        )
    )
    team_labels = tuple(
        dict.fromkeys(
            owner_label(owner_id)
            for owner_id in owner_ids
            if choices[owner_id]["state"] in {"pending", "selected"}
        )
    )
    return choices, request_users, team_labels


def log_routing_plan(
    args: argparse.Namespace,
    decision: str,
    codepath: CodepathOwnerTargets,
    labels: tuple[str, ...],
    *,
    owner_choices: dict[str, dict[str, Any]] | None = None,
    planned_reviewer_requests: tuple[str, ...] = (),
    analyzed_head_sha: str,
    has_uncovered_concerns: bool,
    owner_provenance_truncated: bool,
    submitted_handoff: str | None = None,
    comparison: dict[str, Any] | None = None,
) -> None:
    """Log one deterministic pre-effect plan and write its step summary."""

    record = {
        "mode": args.mode,
        "decision": decision,
        "analyzed_head_sha": analyzed_head_sha,
        "has_uncovered_concerns": has_uncovered_concerns,
        "codepath_owners": list(codepath.values),
        "codeowners_comparison": comparison,
        "intended_labels": list(labels),
        "owner_choices": owner_choices or {},
        "owner_provenance_truncated": owner_provenance_truncated,
        "planned_reviewer_requests": list(planned_reviewer_requests),
        "run_attempt": args.run_attempt,
        "submitted_handoff": submitted_handoff,
    }
    log_json_record("Auto PR Triage plan", record)
    log_reviewer_routing(
        args.mode, owner_choices or {}, planned_reviewer_requests
    )
    summary = getattr(args, "github_step_summary", None)
    if summary is None:
        return
    reviewer_names = ", ".join(
        f"`{reviewer.removeprefix('@')}`"
        for reviewer in planned_reviewer_requests
    )
    try:
        with Path(summary).open("a", encoding="utf-8") as output:
            output.write("## Auto PR Triage decision plan\n\n")
            output.write(f"- Mode: `{args.mode}`\n")
            output.write(f"- Decision: `{decision}`\n")
            output.write(
                "- Has uncovered concerns: "
                f"`{str(has_uncovered_concerns).lower()}`\n"
            )
            output.write(
                f"- Planned reviewer requests: {reviewer_names or 'none'}\n"
            )
            for owner, choice in sorted((owner_choices or {}).items()):
                reviewer = choice["reviewer"].removeprefix("@")
                output.write(
                    f"- Owner `{owner}`: `{reviewer}` "
                    f"({choice['state']})\n"
                )
            if submitted_handoff is not None:
                output.write(
                    "- Existing configured-reviewer handoff: "
                    f"`{submitted_handoff.removeprefix('@')}`\n"
                )
            output.write(
                "- Intended labels: "
                + ", ".join(f"`{label}`" for label in labels)
                + "\n"
            )
    except OSError as exc:
        detail = " ".join(str(exc).split())[:200]
        print(
            "Auto PR Triage step summary unavailable: "
            f"{type(exc).__name__}: {detail}"
        )


def apply_controller_action(
    args: argparse.Namespace,
    github: GitHubClient,
) -> ControllerOutcome:
    """Derive and apply one live or shadow action from normalized facts."""

    if args.mode not in {"shadow", "live"}:
        raise ValueError("invalid Auto PR Triage mode")
    shadow_mode = args.mode == "shadow"
    validate_identity(args)
    analysis = AnalysisResult.from_json(args.analysis_result_json)
    target_org = args.repository.split("/", 1)[0]
    if any(
        owner.startswith("@")
        and "/" in owner
        and owner[1:].split("/", 1)[0].casefold() != target_org.casefold()
        for owner in analysis.codepath_owners
    ):
        raise ValueError("analysis result has a foreign codepath owner team")
    codepath = CodepathOwnerTargets(analysis.codepath_owners)
    additional_owners = analysis.additional_owners
    triage_incomplete = analysis.ownership_analysis == "incomplete"

    # SECURITY POLICY: Analysis is the sole authority for PR identity and triage
    # facts. Apply uses one-pass reads with accepted stale-result
    # risk and does not automatically reconcile concurrent changes. Resulting
    # mutations are bounded and reversible.
    if (
        not analysis.is_open_non_draft_pr_against_main
        or analysis.is_already_handled
    ):
        return ControllerOutcome("kept_open")

    if not should_run_ownership_analysis(
        is_open_non_draft_pr_against_main=analysis.is_open_non_draft_pr_against_main,
        is_already_handled=analysis.is_already_handled,
        author_has_triage_permission=analysis.author_has_triage_permission,
        has_actionable_linked_issue=analysis.has_actionable_linked_issue,
        has_maintainer_activity=analysis.has_maintainer_activity,
    ):
        if args.run_attempt != 1:
            return ControllerOutcome("kept_open")
        label = BOT_SHADOW_CLOSE_LABEL if shadow_mode else BOT_CLOSED_LABEL
        labels = (label,)
        log_routing_plan(
            args,
            "close",
            codepath,
            labels,
            analyzed_head_sha=analysis.analyzed_head_sha,
            has_uncovered_concerns=analysis.has_uncovered_concerns,
            owner_provenance_truncated=analysis.owner_provenance_truncated,
        )
        require_repository_label(github, args.repository, label)
        if shadow_mode:
            add_labels(github, args.repository, args.pr, labels)
            return ControllerOutcome("shadow_close")
        try:
            closed = github.json(
                f"repos/{args.repository}/pulls/{args.pr}",
                method="PATCH",
                payload={"state": "closed"},
            )
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                "close request failed or returned ambiguously; "
                f"the PR may already be closed: {exc}"
            ) from exc
        confirmed = (
            isinstance(closed, dict)
            and closed.get("number") == args.pr
            and closed.get("state") == "closed"
        )
        if not confirmed:
            raise RuntimeError(
                "close was not confirmed; the PR may already be closed"
            )
        record_bot_close(github, args.repository, args.pr)
        return ControllerOutcome("closed")

    shadow_comparison = None
    shadow_label = None
    semantic_coverage = codepath
    if NATIVE_CODEOWNERS_SHADOW:
        shadow_comparison = compare_codeowners_shadow(github, args, codepath)
        shadow_label = CODEOWNERS_SHADOW_LABELS[shadow_comparison["status"]]
        semantic_coverage = CodepathOwnerTargets(
            tuple(shadow_comparison["observed"])
            if shadow_comparison["status"] != "inconclusive"
            else ()
        )

    author_handle = f"@{args.author_login}".casefold()
    if not codepath.values and not additional_owners:
        if triage_incomplete:
            labels = tuple(
                label
                for label in (BOT_TRIAGE_ERROR_LABEL, shadow_label)
                if label is not None
            )
            for label_name in labels:
                require_repository_label(github, args.repository, label_name)
            log_routing_plan(
                args,
                "incomplete",
                codepath,
                labels,
                analyzed_head_sha=analysis.analyzed_head_sha,
                has_uncovered_concerns=analysis.has_uncovered_concerns,
                owner_provenance_truncated=analysis.owner_provenance_truncated,
                comparison=shadow_comparison,
            )
            try:
                add_labels(github, args.repository, args.pr, labels)
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                raise RuntimeError(
                    "triage-error label request failed or returned "
                    f"ambiguously; the label may already be present: {exc}"
                ) from exc
            return ControllerOutcome("incomplete")
        team_members = load_team_members(
            REPOSITORY_ROOT, args.repository, args.workflow_sha
        )
        submitted_reviewers = normalize_user_handles(
            set(fetch_submitted_review_state(github, args.repository, args.pr))
        )
        handoff = find_submitted_handoff(
            team_members, submitted_reviewers, author_handle
        )
        if handoff is None:
            labels = tuple(
                label for label in (shadow_label,) if label is not None
            )
            log_routing_plan(
                args,
                "keep_open",
                codepath,
                labels,
                analyzed_head_sha=analysis.analyzed_head_sha,
                has_uncovered_concerns=analysis.has_uncovered_concerns,
                owner_provenance_truncated=analysis.owner_provenance_truncated,
                comparison=shadow_comparison,
            )
            if shadow_label is not None:
                require_repository_label(github, args.repository, shadow_label)
                add_labels(github, args.repository, args.pr, (shadow_label,))
            return ControllerOutcome("kept_open")
        labels = (
            (BOT_SHADOW_TRIAGED_LABEL, shadow_label)
            if shadow_mode
            else (TRIAGED_LABEL, BOT_TRIAGED_LABEL, shadow_label)
        )
        labels = tuple(label for label in labels if label is not None)
        for label_name in labels:
            require_repository_label(github, args.repository, label_name)
        log_routing_plan(
            args,
            "triage",
            codepath,
            labels,
            analyzed_head_sha=analysis.analyzed_head_sha,
            has_uncovered_concerns=analysis.has_uncovered_concerns,
            owner_provenance_truncated=analysis.owner_provenance_truncated,
            submitted_handoff=handoff,
            comparison=shadow_comparison,
        )
        try:
            add_labels(github, args.repository, args.pr, labels)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                "triage label request failed or returned ambiguously; "
                f"the labels may already be present: {exc}"
            ) from exc
        return ControllerOutcome("shadow_triaged" if shadow_mode else "triaged")

    reviewers = ReviewerState(frozenset(), frozenset())
    reviewer_state_available = True
    if (
        additional_owners
        or analysis.has_uncovered_concerns
        or not NATIVE_CODEOWNERS_SHADOW
    ):
        try:
            reviewers = fetch_reviewer_state(github, args.repository, args.pr)
        except (RuntimeError, subprocess.TimeoutExpired):
            if not NATIVE_CODEOWNERS_SHADOW:
                if codepath.owner_ids or not codepath.github_handles:
                    raise
                if additional_owners or analysis.has_uncovered_concerns:
                    triage_incomplete = True
                reviewer_state_available = False
                print(
                    "Auto PR Triage reviewer state was unavailable; "
                    "applying direct codepath-owner handles only."
                )
            else:
                triage_incomplete = True
                reviewer_state_available = False
                print(
                    "Auto PR Triage reviewer state was unavailable; "
                    "skipping additional owners."
                )
    owner_choices: dict[str, dict[str, Any]] = {}
    selected_users: tuple[str, ...] = ()
    team_labels: tuple[str, ...] = ()
    team_members: dict[str, Any] | None = None
    owner_ids = (
        additional_owners
        if NATIVE_CODEOWNERS_SHADOW
        else tuple(sorted({*codepath.owner_ids, *additional_owners}))
    )
    if owner_ids and reviewer_state_available:
        try:
            team_members = load_team_members(
                REPOSITORY_ROOT, args.repository, args.workflow_sha
            )
            owner_choices, selected_users, team_labels = select_owner_reviewers(
                github,
                args.repository,
                args.pr,
                team_members,
                owner_ids,
                semantic_coverage,
                author_handle,
                reviewers,
            )
            # Round-robin selection already preflights labels for newly selected
            # owners. Only pending assignments still need their one label lookup.
            for owner_id, choice in owner_choices.items():
                if choice["state"] == "pending":
                    require_repository_label(
                        github, args.repository, owner_label(owner_id)
                    )
        except (RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            if not NATIVE_CODEOWNERS_SHADOW and (
                codepath.owner_ids or not codepath.github_handles
            ):
                raise
            triage_incomplete = True
            print(
                "Auto PR Triage additional owners were unavailable; "
                + (
                    "relying on native CODEOWNERS only."
                    if NATIVE_CODEOWNERS_SHADOW
                    else "applying direct codepath-owner handles only."
                )
                + f" {type(exc).__name__}: {' '.join(str(exc).split())[:500]}"
            )
            owner_choices = {}
            selected_users = ()
            team_labels = ()
    submitted_handoff = None
    if (
        analysis.has_uncovered_concerns
        and not triage_incomplete
        and reviewer_state_available
    ):
        if team_members is None:
            team_members = load_team_members(
                REPOSITORY_ROOT, args.repository, args.workflow_sha
            )
        submitted_handoff = find_submitted_handoff(
            team_members,
            reviewers.submitted_reviewers,
            author_handle,
        )
    codepath_users = ()
    codepath_teams = ()
    if not NATIVE_CODEOWNERS_SHADOW:
        codepath_users = tuple(
            login
            for login in codepath.github_users
            if f"@{login}".casefold() not in reviewers.current_reviewers
            and f"@{login}".casefold() != author_handle
        )
        codepath_teams = tuple(
            slug
            for slug in codepath.github_teams
            if f"@{target_org}/{slug}".casefold()
            not in reviewers.requested_reviewers
        )
    request_users = tuple(sorted({*selected_users, *codepath_users}, key=str.casefold))
    if len(request_users) + len(codepath_teams) > MAX_REVIEW_REQUESTS:
        raise ValueError("review request exceeds Auto PR Triage's 15-target limit")

    payload: dict[str, list[str]] = {}
    if request_users:
        payload["reviewers"] = list(request_users)
    if codepath_teams:
        payload["team_reviewers"] = list(codepath_teams)

    routed_untriaged = (
        analysis.has_uncovered_concerns
        and submitted_handoff is None
        and not triage_incomplete
    )
    if triage_incomplete:
        status_labels = (BOT_TRIAGE_ERROR_LABEL,)
    elif routed_untriaged:
        status_labels = ()
    elif shadow_mode:
        status_labels = (BOT_SHADOW_TRIAGED_LABEL,)
    else:
        status_labels = (TRIAGED_LABEL, BOT_TRIAGED_LABEL)
    if shadow_label is not None:
        status_labels += (shadow_label,)
    for label_name in status_labels:
        require_repository_label(github, args.repository, label_name)
    planned_reviewer_requests = (
        tuple(f"@{login}" for login in request_users)
        + tuple(f"@{target_org}/{slug}" for slug in codepath_teams)
    )
    direct_codepath_requests = (
        tuple(f"@{login}" for login in codepath_users)
        + tuple(f"@{target_org}/{slug}" for slug in codepath_teams)
    )

    def provenance_for(owner: str, source: str) -> dict[str, Any]:
        if analysis.owner_provenance_truncated:
            return {"source": source, "truncated": True}
        return analysis.owner_provenance[owner]

    for owner_id, choice in owner_choices.items():
        source = "codepath" if owner_id in analysis.codepath_owners else "semantic"
        choice["provenance"] = provenance_for(owner_id, source)
    # Complete the log-only choices with direct codepath reviewers. The request
    # and label inputs above remain unchanged.
    direct_by_key = {
        reviewer.casefold(): reviewer for reviewer in direct_codepath_requests
    }
    for owner in codepath.values:
        if not owner.startswith("@"):
            continue
        reviewer_key = owner.casefold()
        if reviewer_key == author_handle:
            continue
        if NATIVE_CODEOWNERS_SHADOW:
            active_native_request = (
                shadow_comparison is not None
                and reviewer_key in shadow_comparison["expected"]
                and reviewer_key in shadow_comparison["observed"]
            )
            state = "native_codeowner" if active_native_request else None
        elif reviewer_key in direct_by_key:
            state = "selected"
        elif reviewer_key in reviewers.submitted_reviewers:
            state = "submitted"
        elif reviewer_key in reviewers.requested_reviewers:
            state = "pending"
        else:
            state = None
        if state is None:
            continue
        choice = {
            "reviewer": direct_by_key.get(reviewer_key, owner),
            "state": state,
            "provenance": provenance_for(owner, "codepath"),
        }
        if state == "selected":
            choice["selection_reason"] = "direct_codepath_owner"
        owner_choices[owner] = choice
    labels = tuple(
        dict.fromkeys(
            status_labels if shadow_mode else (*status_labels, *team_labels)
        )
    )
    log_routing_plan(
        args,
        (
            "incomplete"
            if triage_incomplete
            else "routed_untriaged"
            if routed_untriaged
            else "triage"
        ),
        codepath,
        labels,
        owner_choices=owner_choices,
        planned_reviewer_requests=planned_reviewer_requests,
        analyzed_head_sha=analysis.analyzed_head_sha,
        has_uncovered_concerns=analysis.has_uncovered_concerns,
        owner_provenance_truncated=analysis.owner_provenance_truncated,
        submitted_handoff=submitted_handoff,
        comparison=shadow_comparison,
    )

    requested_users = 0
    if payload and not shadow_mode:
        try:
            github.json(
                f"repos/{args.repository}/pulls/{args.pr}/requested_reviewers",
                method="POST",
                payload=payload,
            )
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                "reviewer request failed or returned ambiguously; "
                f"the requests may already be present: {exc}"
            ) from exc
        requested_users = len(request_users)
    if labels:
        try:
            add_labels(github, args.repository, args.pr, labels)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                "triage label request failed or returned ambiguously; "
                f"the labels may already be present: {exc}"
            ) from exc
    if triage_incomplete:
        status = "incomplete"
    elif routed_untriaged:
        status = "routed_untriaged"
    elif shadow_mode:
        status = "shadow_triaged"
    else:
        status = "triaged"
    return ControllerOutcome(
        status,
        requested_users=requested_users,
        owner_labels=0 if shadow_mode else len(team_labels),
        requested_teams=0 if shadow_mode else len(codepath_teams),
    )


def parse_args() -> argparse.Namespace:
    """Parse the normalized analysis record and event identity."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pr", type=int, help="pull request number")
    parser.add_argument("--repository", required=True)
    parser.add_argument("--mode", required=True, choices=("shadow", "live"))
    parser.add_argument("--workflow-sha", required=True)
    parser.add_argument("--author-login", required=True)
    parser.add_argument("--run-attempt", type=int, required=True)
    parser.add_argument("--analysis-result-json", required=True)
    parser.add_argument("--github-step-summary", type=Path)
    return parser.parse_args()


def main() -> int:
    """Run the mode-selected controller CLI."""

    args = parse_args()
    if not os.environ.get("GH_TOKEN"):
        print("Auto PR Triage apply failed: GH_TOKEN is unavailable.", file=sys.stderr)
        return 1
    try:
        outcome = apply_controller_action(
            args,
            GitHubClient(args.mode, args.repository, args.pr),
        )
        if outcome.status not in {
            "shadow_triaged",
            "shadow_close",
            "triaged",
            "closed",
            "incomplete",
            "kept_open",
            "routed_untriaged",
        }:
            raise RuntimeError("controller returned an invalid outcome")
    except Exception as exc:
        detail = " ".join(str(exc).split())[:500]
        print(
            f"Auto PR Triage apply failed: {type(exc).__name__}: {detail}",
            file=sys.stderr,
        )
        return 1

    if outcome.status == "closed":
        print(f"Closed {args.repository}#{args.pr} for a missing actionable issue.")
    elif outcome.status == "shadow_close":
        print(f"Shadow result for {args.repository}#{args.pr}: would close.")
    elif outcome.status == "shadow_triaged":
        print(f"Shadow result for {args.repository}#{args.pr}: would triage.")
    elif outcome.status == "incomplete":
        if args.mode == "shadow":
            print(
                f"Shadow result for {args.repository}#{args.pr}: "
                "analysis incomplete."
            )
        else:
            print(
                f"Applied incomplete Auto PR Triage to {args.repository}#{args.pr}: "
                f"requested {outcome.requested_users} users and "
                f"{outcome.requested_teams} teams; applied "
                f"{outcome.owner_labels} owner labels."
            )
    elif outcome.status == "kept_open":
        if args.mode == "shadow":
            print(f"Shadow result for {args.repository}#{args.pr}: would keep open.")
        else:
            print(
                f"{args.repository}#{args.pr} did not qualify for an apply action; "
                "kept open."
            )
    elif outcome.status == "routed_untriaged":
        if args.mode == "shadow":
            print(
                f"Shadow result for {args.repository}#{args.pr}: would route known "
                "owners and leave untriaged for human routing."
            )
        else:
            print(
                f"Applied partial Auto PR Triage to {args.repository}#{args.pr}: "
                f"requested {outcome.requested_users} users and "
                f"{outcome.requested_teams} teams; applied "
                f"{outcome.owner_labels} owner labels; left untriaged for human "
                "routing."
            )
    else:
        print(
            f"Applied Auto PR Triage to {args.repository}#{args.pr}: requested "
            f"{outcome.requested_users} users and "
            f"{outcome.requested_teams} teams; applied "
            f"{outcome.owner_labels} owner labels."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
