#!/usr/bin/env python3
"""Stage 3: plan the exact GitHub effects of one intake and ownership result.

This runs in the read-only analyze job. main() does three things in order:

1. gather_planner_input makes every GitHub read planning may depend on (labels,
   reviewers, native CODEOWNERS requests, rosters, round-robin cursors) and
   bundles them with the stage results into planner_input.json.
2. build_action_plan is a pure function of that input. It returns the
   ActionPlan, the job output the live apply job executes, and a log-only
   PlanExplanation.
3. log_plan writes the explanation to the job log and the step summary.

Shadow mode stops after this stage; live mode then runs apply_actions.py.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

from github_api import GitHubClient, GitHubReader, PullRequestRef
from identifiers import owner_label, USER_HANDLE_RE
from reviewer_state import (
    choose_round_robin_member,
    fetch_requested_codeowner_handles,
    fetch_requested_reviewer_handles,
    fetch_round_robin_cursors,
    fetch_submitted_review_state,
)
from schemas import (
    Action,
    ActionPlan,
    AddLabels,
    BOT_CLOSED_LABEL,
    BOT_TRIAGE_ERROR_LABEL,
    BOT_TRIAGED_LABEL,
    check_stage_results,
    CLOSE_ACTIONS,
    IntakeResult,
    MAX_REVIEW_REQUESTS,
    OwnershipResult,
    PlanContext,
    PlannerInput,
    PullRequestIdentity,
    REQUEST_REASONS,
    RequestReviewers,
    ReviewerSnapshot,
    TRIAGED_LABEL,
)
from step_summary import summary_prose
from trusted_config import load_team_members


# Who requests the codepath owners' reviews. True (current rollout): GitHub's
# native CODEOWNERS, mirrored from codepath_owners.txt, requests them; the bot
# only logs a comparison with those requests and picks roster reviewers for the
# LLM's additional owners. False: the bot requests direct codepath users and
# teams itself and also picks roster reviewers for codepath team owner IDs.
# Unrelated to the workflow's shadow/live mode, which gates all writes.
NATIVE_CODEOWNERS_REQUESTS_CODEPATH_OWNERS = True
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MAX_LOGGED_OWNER_FILES = 16


@dataclass(frozen=True)
class CodepathOwnerTargets:
    """Immutable codepath-owner handles and team owner IDs."""

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
    def team_owner_ids(self) -> tuple[str, ...]:
        """Return team owner IDs resolved through the trusted roster."""

        return tuple(owner for owner in self.values if not owner.startswith("@"))


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

    return value.replace("::", r"\u003a\u003a").replace("##[", r"\u0023\u0023[")


def escape_log_fragment(value: str) -> str:
    """Escape control characters without adding JSON string delimiters."""

    return json.dumps(value, ensure_ascii=True)[1:-1]


def log_json_record(*, label: str, record: dict[str, Any]) -> None:
    """Print a sanitized, readable JSON log record."""

    serialized = json.dumps(record, indent=2, sort_keys=True)
    print(f"{label}:\n{sanitize_log_text(serialized)}")


@dataclass(frozen=True)
class TextStyle:
    """Render untrusted text inertly for one output: the job log or a summary."""

    prose: Callable[[str], str]
    code: Callable[[str], str]
    handle: Callable[[str], str]
    excerpts: bool


LOG_STYLE = TextStyle(
    escape_log_fragment,
    lambda value: f"`{escape_log_fragment(value)}`",
    lambda handle: handle,
    True,
)
# Summaries show `login` rather than @login, so pasting one never pings anyone.
SUMMARY_STYLE = TextStyle(
    summary_prose, summary_prose, lambda handle: f"`{handle.removeprefix('@')}`", False
)


def admission_items(
    *, intake: IntakeResult, ownership: OwnershipResult
) -> list[dict[str, Any]]:
    """List why the PR was admitted: each true intake fact and each bypass match."""

    facts = intake.facts
    signals = (
        (facts.author_has_triage_permission, "the author has triage-or-higher access"),
        (facts.has_actionable_linked_issue, "it fixes an issue labeled `actionable`"),
        (facts.has_maintainer_activity, "a maintainer has qualifying activity on it"),
        (facts.has_supporter, "its description names a verified supporter"),
        (facts.has_related_actionable_issue, "it is part of an `actionable` issue"),
    )
    items: list[dict[str, Any]] = [{"fact": text} for found, text in signals if found]
    for concern in ownership.additional_owner_concerns:
        if (match := concern.bypass_intake_match) is not None:
            items.append({"bypass_owner": concern.owner_id, **match.to_dict()})
    return items


def admission_text(*, items: list[dict[str, Any]], style: TextStyle) -> str:
    if not items:
        return "Not admitted: no intake fact or bypass intake match."
    reasons = []
    for item in items:
        if "fact" in item:
            reasons.append(item["fact"])
            continue
        quote = style.prose(item["criteria_quote"])
        why = " ".join(style.prose(reason) for reason in item["rationale"])
        owner = item["bypass_owner"]
        reasons.append(f'`{owner}`\'s bypass intake "{quote}" matched: {why}')
    return "Admitted because " + "; ".join(reasons) + "."


def reviewer_choice_reason(
    *, owner: str, choice: OwnerChoice, handle: Callable[[str], str]
) -> str:
    """Describe why one owner resolved to one reviewer."""

    reviewer = handle(choice.reviewer)
    previous = handle(choice.previous_assignee or "")
    prior = choice.previous_pull_request
    if choice.state == "selected":
        reasons = {
            "round_robin_initial": (
                f"No previous `{owner}` assignment was found, so its round-robin "
                f"rotation began with {reviewer}."
            ),
            "round_robin_next": (
                f"{reviewer} was the next eligible member of `{owner}`'s round-robin "
                f"rotation after {previous}, who was assigned on #{prior}."
            ),
            "stable_fallback": (
                f"The latest `{owner}` marker, on #{prior}, had no attributable "
                f"current-roster assignment, so the stable fallback chose {reviewer}."
            ),
            "direct_codepath_owner": (
                f"The checked-in codepath-owner rules directly name {reviewer}."
            ),
        }
        return reasons[choice.selection_reason]
    reasons = {
        "native_codeowner": (
            f"GitHub already has an active native CODEOWNERS request for {reviewer}."
        ),
        "codepath_covered": (
            f"{reviewer} covers `{owner}` through codepath ownership; no `{owner}` "
            "round-robin choice was needed."
        ),
        "submitted": (
            f"{reviewer} already submitted a review covering `{owner}`; no new request is needed."
        ),
        "pending": (
            f"{reviewer} already has a pending request covering `{owner}`; no new "
            "request is needed."
        ),
    }
    return reasons[choice.state]


def engagement_reason(*, reviewer: str, reason: EngagementReason) -> str:
    """Describe a reviewer's own engagement with the PR."""

    if reason.kind == "maintainer_requested":
        return f"A triage-or-higher maintainer already requested {reviewer}."
    if reason.kind == "supporter":
        text = f"The PR description names {reviewer} as a verified supporter."
    else:
        text = f"{reviewer} labeled a linked or related issue `actionable`."
    if reason.already_reviewing:
        text += f" {reviewer} already reviewed or has a pending request, so no new request is needed."
    return text


def evidence_lines(*, evidence: list[dict[str, str]], style: TextStyle) -> list[str]:
    """Render evidence items, with their diff excerpts when the style shows them."""

    lines = []
    for item in evidence:
        lines.append(f"- {style.code(item['file'])}: " + style.prose(item["relevance"]))
        if style.excerpts:
            lines.extend(
                f"    {style.prose(line)}" for line in item["diff_excerpt"].splitlines()
            )
    return lines


def owner_details(
    *, owner: str, provenance: dict[str, Any], style: TextStyle
) -> list[str]:
    """Explain why an owner owns this PR: its files, or its validated concern."""

    concern = None if provenance["source"] == "codepath" else provenance["concern"]
    all_files = provenance["files"] if concern is None else concern["files"]
    files = ", ".join(style.code(path) for path in all_files[:MAX_LOGGED_OWNER_FILES])
    if len(all_files) > MAX_LOGGED_OWNER_FILES:
        files += f" (+{len(all_files) - MAX_LOGGED_OWNER_FILES} more)"
    if concern is None:
        return [f"Codepath owner `{owner}` matched supporting files: {files}."]
    lines = [
        f"Semantic owner `{owner}`: " + style.prose(concern["description"]),
        "Reasoning: " + " ".join(style.prose(r) for r in provenance["rationale"]),
        f"Supporting files: {files}.",
        "Evidence:",
        *evidence_lines(evidence=concern["evidence"], style=style),
    ]
    bypass = provenance["bypass_intake_match"]
    if bypass is not None:
        quote = style.prose(bypass["criteria_quote"])
        reasons = " ".join(style.prose(r) for r in bypass["rationale"])
        lines.append(f'Matches bypass intake "{quote}": {reasons}')
        lines.append("Bypass evidence:")
        lines.extend(evidence_lines(evidence=bypass["evidence"], style=style))
    return lines


def reviewer_explanation(
    *, group: ReviewerExplanation, style: TextStyle
) -> tuple[list[str], list[str]]:
    """Return one reviewer's reason sentences and the details behind them."""

    reviewer = style.handle(group.reviewer)
    sentences, details = [], []
    for reason in group.reasons:
        if isinstance(reason, EngagementReason):
            sentences.append(engagement_reason(reviewer=reviewer, reason=reason))
            continue
        sentences.append(
            reviewer_choice_reason(
                owner=reason.owner, choice=reason.choice, handle=style.handle
            )
        )
        details.extend(
            owner_details(owner=reason.owner, provenance=reason.provenance, style=style)
        )
    return sentences, details


def log_reviewer_routing(
    *,
    admission: str,
    incomplete_reasons: list[str],
    groups: tuple[ReviewerExplanation, ...],
) -> None:
    """Explain admission, any incompleteness, and each reviewer."""

    blocks = [f"Why this PR was admitted: {admission}"]
    if incomplete_reasons:
        reasons = " ".join(escape_log_fragment(r) for r in incomplete_reasons)
        blocks.append(f"Why this run is incomplete: {reasons}")
    for group in groups:
        reviewer = group.reviewer
        sentences, details = reviewer_explanation(group=group, style=LOG_STYLE)
        effect = (
            f"one deduplicated request for {reviewer}."
            if group.requested
            else "no new Auto PR Triage reviewer request is needed."
        )
        lines = [
            f"Reviewer {reviewer}",
            "Why this reviewer: " + " ".join(sentences),
            *details,
            f"Planned effect: {effect}",
        ]
        blocks.append("\n".join(lines))
    print(
        "Auto PR Triage reviewer routing:\n\n" + sanitize_log_text("\n\n".join(blocks))
    )


def label_exists(*, github: GitHubClient, repository: str, label_name: str) -> bool:
    """Return whether one repository label exists; API failures propagate."""

    encoded = quote(label_name, safe="")
    label = github.json(f"repos/{repository}/labels/{encoded}")
    actual = label.get("name") if isinstance(label, dict) else None
    return isinstance(actual, str) and actual.casefold() == label_name.casefold()


def require_label(*, snapshot: ReviewerSnapshot, label_name: str) -> None:
    """Require a label the snapshot confirmed before planning an effect on it."""

    if label_name not in snapshot.existing_labels:
        raise RuntimeError(f"required repository label is unavailable: {label_name}")


def normalize_requested_reviewer_handles(
    *,
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
        if USER_HANDLE_RE.fullmatch(handle) or target_team_handle_re.fullmatch(handle)
    )


def normalize_user_handles(handles: set[str]) -> frozenset[str]:
    """Normalize GitHub user handles and ignore teams."""

    return frozenset(
        handle.casefold() for handle in handles if USER_HANDLE_RE.fullmatch(handle)
    )


def compare_codeowners(
    *,
    snapshot: ReviewerSnapshot,
    workflow_sha: str,
    codepath: CodepathOwnerTargets,
    author_login: str,
) -> dict[str, Any]:
    """Compare custom path resolution with active native CODEOWNERS requests."""

    author = f"@{author_login}".casefold()
    expected = {
        owner.casefold()
        for owner in codepath.values
        if owner.startswith("@") and owner.casefold() != author
    }
    requests = snapshot.native_codeowner_requests
    if codepath.team_owner_ids:
        error = "RuntimeError: native CODEOWNERS cannot request team owner IDs"
    elif requests is None:
        error = snapshot.errors.get("native_codeowner_requests", "not read")
    else:
        error = None
    observed = set() if error else set(requests or ())
    comparison = {
        "status": (
            "inconclusive" if error else "match" if expected == observed else "mismatch"
        ),
        "oracle": "active_review_requests_as_code_owner",
        "workflow_sha": workflow_sha,
        "expected": sorted(expected),
        "observed": sorted(observed),
        "missing_from_github": [] if error else sorted(expected - observed),
        "unexpected_from_github": [] if error else sorted(observed - expected),
        "error": error,
    }
    return comparison


def fetch_reviewer_state(pr: PullRequestRef, /) -> ReviewerState:
    """Fetch requested and submitted reviewers once for this apply attempt.

    SECURITY POLICY: Auto PR Triage deliberately does not refresh this state.
    The separate reads are not atomic. Concurrent review activity can cause a
    stale request or label; those bounded effects are accepted and reversible.
    """

    requested = normalize_requested_reviewer_handles(
        handles=fetch_requested_reviewer_handles(pr),
        repository=pr.repo,
    )
    submitted = normalize_user_handles(set(fetch_submitted_review_state(pr)))
    return ReviewerState(
        requested_reviewers=requested,
        submitted_reviewers=submitted,
    )


def planned_team_owner_ids(ownership: OwnershipResult) -> tuple[str, ...]:
    """Return the team owner IDs that need a reviewer from their roster."""

    if NATIVE_CODEOWNERS_REQUESTS_CODEPATH_OWNERS:
        return ownership.additional_owners
    codepath = CodepathOwnerTargets(tuple(ownership.codepath_owners))
    return tuple(sorted({*codepath.team_owner_ids, *ownership.additional_owners}))


def covering_handles(
    *, codepath: CodepathOwnerTargets, native_requests: tuple[str, ...] | None
) -> CodepathOwnerTargets:
    """Return the handles whose requests count as covering a roster member.

    In native mode that is GitHub's active CODEOWNERS requests; otherwise it is
    the codepath handles, which the bot requests itself.
    """

    if not NATIVE_CODEOWNERS_REQUESTS_CODEPATH_OWNERS:
        return codepath
    return CodepathOwnerTargets(tuple(native_requests or ()))


ChoiceState = Literal[
    "native_codeowner", "codepath_covered", "submitted", "pending", "selected"
]
SelectionReason = Literal[
    "round_robin_initial",
    "round_robin_next",
    "stable_fallback",
    "direct_codepath_owner",
]


@dataclass(frozen=True)
class OwnerChoice:
    """Record how one owner resolved to one reviewer.

    state says whether the reviewer is newly selected or already covers the
    owner: through native CODEOWNERS or codepath ownership, a submitted review,
    or a pending request. A selected reviewer also records why: a round-robin
    step, with the previous assignee and the PR it happened on, or a direct
    codepath owner.
    """

    reviewer: str
    state: ChoiceState
    selection_reason: SelectionReason | None = None
    previous_assignee: str | None = None
    previous_pull_request: int | None = None

    def to_log(self) -> dict[str, Any]:
        """Return the logged form, omitting fields that do not apply."""

        return {key: value for key, value in asdict(self).items() if value is not None}


def classify_team_owners(
    *,
    team_owner_ids: tuple[str, ...],
    rosters: dict[str, tuple[str, ...]],
    coverage: CodepathOwnerTargets,
    reviewers: ReviewerState,
    author_handle: str,
) -> tuple[dict[str, OwnerChoice], tuple[str, ...]]:
    """Return owners an existing reviewer covers, and owners needing a new pick."""

    choices: dict[str, OwnerChoice] = {}
    needs_round_robin: list[str] = []
    for owner_id in team_owner_ids:
        members = {member.casefold(): member for member in rosters[owner_id]}
        covered = (coverage.github_handles & set(members)) - {author_handle}
        submitted = (set(members) & reviewers.submitted_reviewers) - {author_handle}
        pending = (set(members) & reviewers.requested_reviewers) - {author_handle}
        if covered:
            choices[owner_id] = OwnerChoice(min(covered), "codepath_covered")
        elif submitted:
            choices[owner_id] = OwnerChoice(members[min(submitted)], "submitted")
        elif pending:
            choices[owner_id] = OwnerChoice(members[min(pending)], "pending")
        else:
            needs_round_robin.append(owner_id)
    return choices, tuple(needs_round_robin)


def select_team_owner_reviewers(
    *,
    identity: PullRequestIdentity,
    snapshot: ReviewerSnapshot,
    team_owner_ids: tuple[str, ...],
    coverage: CodepathOwnerTargets,
    author_handle: str,
    reviewers: ReviewerState,
) -> dict[str, OwnerChoice]:
    """Select reviewers for team owner IDs from the snapshot alone."""

    rosters = snapshot.rosters
    if rosters is None:
        raise RuntimeError(
            snapshot.errors.get("rosters", "owner rosters were not read")
        )
    if not set(team_owner_ids) <= set(rosters):
        raise ValueError("owner is not configured")
    choices, needs_round_robin = classify_team_owners(
        team_owner_ids=team_owner_ids,
        rosters=rosters,
        coverage=coverage,
        reviewers=reviewers,
        author_handle=author_handle,
    )
    for owner_id in needs_round_robin:
        if owner_label(owner_id) not in snapshot.existing_labels:
            raise RuntimeError("configured owner label is unavailable")
        cursor = (snapshot.round_robin or {}).get(owner_id)
        if cursor is None:
            error = snapshot.errors.get("round_robin", "round-robin state was not read")
            raise RuntimeError(error)
        pick = choose_round_robin_member(
            repo=identity.repository,
            current_number=identity.number,
            owner=owner_id,
            members=rosters[owner_id],
            cursor=cursor,
            ineligible_reviewers={author_handle},
        )
        if pick is not None:
            reviewer, reason = pick
            choices[owner_id] = OwnerChoice(
                reviewer,
                "selected",
                reason,
                cursor.last_assigned,
                cursor.prior_pull_request,
            )
    return choices


@dataclass(frozen=True)
class EngagementReason:
    """Record a reviewer's own engagement with the PR."""

    kind: Literal["supporter", "actionable_labeler", "maintainer_requested"]
    already_reviewing: bool


@dataclass(frozen=True)
class OwnerReason:
    """Record a reviewer chosen for, or already covering, one owner.

    provenance says why the owner owns the PR: the codepath files it matched,
    or the validated concern for a semantic owner.
    """

    owner: str
    choice: OwnerChoice
    provenance: dict[str, Any]


@dataclass(frozen=True)
class ReviewerExplanation:
    """Gather every reason one reviewer was requested or already reviews."""

    reviewer: str
    requested: bool
    reasons: tuple[EngagementReason | OwnerReason, ...]


@dataclass(frozen=True)
class PlanExplanation:
    """Explain one plan for the log and step summary; it never changes the plan."""

    admission: list[dict[str, Any]]
    incomplete_reasons: list[str]
    has_uncovered_concerns: bool
    reviewers: tuple[ReviewerExplanation, ...] = ()
    unresolved_owners: tuple[str, ...] = ()
    codeowners_comparison: dict[str, Any] | None = None

    @property
    def owner_choices(self) -> dict[str, dict[str, Any]]:
        """Return each owner's reviewer in its logged, owner-keyed form."""

        return {
            reason.owner: {**reason.choice.to_log(), "provenance": reason.provenance}
            for group in self.reviewers
            for reason in group.reasons
            if isinstance(reason, OwnerReason)
        }


def log_plan(*, plan: ActionPlan, why: PlanExplanation, summary: Path | None) -> None:
    """Log one deterministic pre-effect plan and write its step summary."""

    context = plan.context
    org = context.identity.repository.split("/", 1)[0]
    requests = tuple(a for a in plan.actions if isinstance(a, RequestReviewers))
    planned = planned_handles(requests=requests, target_org=org)
    labels = [
        label for a in plan.actions if isinstance(a, AddLabels) for label in a.labels
    ]
    if why.codeowners_comparison is not None:
        log_json_record(
            label="Auto PR Triage CODEOWNERS comparison",
            record=why.codeowners_comparison,
        )
    record = {
        "admission": why.admission,
        "decision": plan.decision,
        "incomplete_reasons": why.incomplete_reasons,
        "analyzed_head_sha": context.identity.head_sha,
        "has_uncovered_concerns": why.has_uncovered_concerns,
        "codepath_owners": list(context.codepath_owners),
        "codeowners_comparison": why.codeowners_comparison,
        "intended_labels": labels,
        "owner_choices": why.owner_choices,
        "planned_reviewer_requests": list(planned),
        "unresolved_owners": list(why.unresolved_owners),
        "run_attempt": context.run_attempt,
    }
    log_json_record(label="Auto PR Triage plan", record=record)
    admission = admission_text(items=why.admission, style=LOG_STYLE)
    log_reviewer_routing(
        admission=admission,
        incomplete_reasons=why.incomplete_reasons,
        groups=why.reviewers,
    )
    if summary is None:
        return
    reviewer_names = ", ".join(f"`{r.removeprefix('@')}`" for r in planned)
    unresolved_names = ", ".join(f"`{owner}`" for owner in why.unresolved_owners)
    uncovered = str(why.has_uncovered_concerns).lower()
    try:
        with summary.open("a", encoding="utf-8") as output:
            output.write("## Auto PR Triage decision plan\n\n")
            output.write(f"- Decision: `{plan.decision}`\n")
            output.write(f"- Has uncovered concerns: `{uncovered}`\n")
            output.write(f"- Planned reviewer requests: {reviewer_names or 'none'}\n")
            output.write(f"- Unresolved owners: {unresolved_names or 'none'}\n")
            for owner, choice in sorted(why.owner_choices.items()):
                reviewer = choice["reviewer"].removeprefix("@")
                output.write(f"- Owner `{owner}`: `{reviewer}` ({choice['state']})\n")
            label_names = ", ".join(f"`{label}`" for label in labels)
            output.write(f"- Intended labels: {label_names}\n")
            output.write("\n### Why this PR was admitted\n\n")
            output.write(
                admission_text(items=why.admission, style=SUMMARY_STYLE) + "\n"
            )
            if why.incomplete_reasons:
                output.write("\n### Why this run is incomplete\n\n")
                output.writelines(
                    f"- {summary_prose(r)}\n" for r in why.incomplete_reasons
                )
            if why.reviewers:
                output.write("\n### Why each reviewer was requested\n\n")
            for group in why.reviewers:
                sentences, details = reviewer_explanation(
                    group=group, style=SUMMARY_STYLE
                )
                status = "new request" if group.requested else "no new request"
                reviewer = SUMMARY_STYLE.handle(group.reviewer)
                output.write(f"- **{reviewer}** ({status}): {' '.join(sentences)}\n")
                output.writelines(
                    f"    - {line[2:]}\n" if line.startswith("- ") else f"  - {line}\n"
                    for line in details
                )
    except OSError as exc:
        detail = " ".join(str(exc).split())
        print(
            f"Auto PR Triage step summary unavailable: {type(exc).__name__}: {detail}"
        )


def engagement_candidates(intake: IntakeResult) -> list[tuple[str, str]]:
    """Return (login, reason) for each verified supporter and labeler but the author."""

    facts = intake.facts
    author = intake.author_login.casefold()
    by_reason = (
        ("supporter", facts.supporters),
        ("actionable_labeler", facts.actionable_labelers),
    )
    return [
        (login, reason)
        for reason, logins in by_reason
        for login in logins
        if login.casefold() != author
    ]


def find_engaged_maintainers(
    *, intake: IntakeResult, snapshot: ReviewerSnapshot
) -> list[tuple[str, EngagementReason]]:
    """Return each engaged maintainer's login and reason, marking who already reviews.

    Engaged maintainers are the triage-or-higher users tied to the PR by their
    own action: verified supporters, actionable labelers, and reviewers another
    maintainer already requested.

    Re-requesting someone who already reviewed would ping them again, so an
    unreadable reviewer state fails planning instead of risking it.
    """

    candidates = engagement_candidates(intake)
    existing: frozenset[str] = frozenset()
    if candidates:
        state = existing_reviewers(snapshot)
        if state is None:
            error = snapshot.errors.get("reviewers", "reviewer state was not read")
            raise RuntimeError(error)
        existing = state.current_reviewers
    engaged_maintainers = [
        (login, EngagementReason(kind, f"@{login.casefold()}" in existing))
        for login, kind in candidates
    ]
    engaged_maintainers += [
        (login, EngagementReason("maintainer_requested", True))
        for login in intake.facts.maintainer_requested_reviewers
    ]
    return engaged_maintainers


def existing_reviewers(snapshot: ReviewerSnapshot) -> ReviewerState | None:
    """Return who already has a pending request or a submitted review, if read."""

    requested = snapshot.requested_reviewers

    submitted = snapshot.submitted_reviewers
    if requested is None or submitted is None:
        return None
    return ReviewerState(frozenset(requested), frozenset(submitted))


def new_reviewer_requests(
    *,
    engaged_maintainers: list[tuple[str, EngagementReason]],
    owner_candidates: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] | None = None,
    existing: ReviewerState,
    author_handle: str,
    org: str,
) -> tuple[RequestReviewers, ...]:
    """Build one request per reason for candidates who are not reviewing yet.

    Supporters and actionable labelers come from engaged_maintainers;
    owner_candidates maps the codepath_owner and owner_roster reasons to their
    users and teams. Anyone with a pending request or a submitted review is
    dropped (a team can only be pending), as is the author, and each user is
    requested once, under the first reason in REQUEST_REASONS that names them.
    """

    engaged = {
        kind: (tuple(login for login, r in engaged_maintainers if r.kind == kind), ())
        for kind in ("supporter", "actionable_labeler")
    }
    candidates = {**engaged, **(owner_candidates or {})}
    seen: set[str] = set()
    requests = []
    for reason in REQUEST_REASONS:
        users, teams = candidates.get(reason, ((), ()))
        fresh = {
            user.casefold(): user
            for user in users
            if user.casefold() not in seen
            and f"@{user}".casefold() not in existing.current_reviewers
            and f"@{user}".casefold() != author_handle
        }
        new_teams = tuple(
            team
            for team in teams
            if f"@{org}/{team}".casefold() not in existing.requested_reviewers
        )
        seen |= set(fresh)
        if fresh or new_teams:
            users = tuple(fresh[key] for key in sorted(fresh))
            requests.append(RequestReviewers(users, new_teams, reason))
    return tuple(requests)


def planned_handles(
    *, requests: tuple[RequestReviewers, ...], target_org: str
) -> tuple[str, ...]:
    """Return every requested reviewer as a handle, users before teams."""

    users = tuple(f"@{user}" for request in requests for user in request.users)
    teams = tuple(f"@{target_org}/{t}" for request in requests for t in request.teams)
    return users + teams


def read_failure(exc: BaseException) -> str:
    """Summarize a failed read for the snapshot's errors."""

    return f"{type(exc).__name__}: {' '.join(str(exc).split())}"


def check_target(*, args: argparse.Namespace, intake: IntakeResult) -> None:
    identity = intake.identity
    target = (identity.repository, identity.number, identity.workflow_sha)
    if target != (args.repository, args.pr, args.workflow_sha):
        raise ValueError("intake result belongs to another pull request or run")


def gather_planner_input(
    *,
    args: argparse.Namespace,
    intake: IntakeResult,
    ownership: OwnershipResult,
    github: GitHubClient,
) -> PlannerInput:
    """Read, once, all live GitHub state the plan can depend on.

    Reads follow the same short circuits as planning: an inactive PR reads
    nothing, and round-robin history is read only for owners that no existing
    reviewer covers. Failed reads are recorded, not raised, except label reads.

    SECURITY POLICY: intake is the sole authority for PR identity and the gate
    facts, which are not read again here. The reads below happen once and are
    not reconciled with concurrent changes, so a plan can act on slightly stale
    state; its effects are bounded and reversible.
    """

    check_target(args=args, intake=intake)
    check_stage_results(intake=intake, ownership=ownership)
    facts = intake.facts
    errors: dict[str, str] = {}
    if not facts.is_active:
        empty = ReviewerSnapshot((), (), None, None, None, None, None, errors)
        return PlannerInput(intake, ownership, empty, args.run_attempt)
    codepath = CodepathOwnerTargets(tuple(ownership.codepath_owners))
    team_owner_ids = planned_team_owner_ids(ownership)
    labels = (
        TRIAGED_LABEL,
        BOT_TRIAGED_LABEL,
        BOT_TRIAGE_ERROR_LABEL,
        BOT_CLOSED_LABEL,
        *(owner_label(owner) for owner in team_owner_ids),
    )
    exists = {
        label: label_exists(github=github, repository=args.repository, label_name=label)
        for label in labels
    }
    existing = tuple(label for label in labels if exists[label])
    missing = tuple(label for label in labels if not exists[label])

    native = requested = submitted = None
    rosters: dict[str, tuple[str, ...]] | None = None
    cursors = None
    admitted = facts.passes_intake or bool(ownership.bypass_intake_matches)
    native_requests = NATIVE_CODEOWNERS_REQUESTS_CODEPATH_OWNERS
    if admitted and native_requests and not codepath.team_owner_ids:
        try:
            handles = fetch_requested_codeowner_handles(
                PullRequestRef(github=github, repo=args.repository, number=args.pr)
            )
            native = tuple(sorted({handle.casefold() for handle in handles}))
        except (RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            errors["native_codeowner_requests"] = read_failure(exc)
    has_owners = bool(codepath.values or ownership.additional_owners)
    needs_owner_reviewers = has_owners and (team_owner_ids or not native_requests)
    if admitted and (needs_owner_reviewers or engagement_candidates(intake)):
        try:
            state = fetch_reviewer_state(
                PullRequestRef(github=github, repo=args.repository, number=args.pr)
            )
            requested = tuple(sorted(state.requested_reviewers))
            submitted = tuple(sorted(state.submitted_reviewers))
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            errors["reviewers"] = read_failure(exc)
    if requested is not None and submitted is not None and team_owner_ids:
        try:
            members = load_team_members(
                repository_root=REPOSITORY_ROOT,
                repo=args.repository,
                ref=args.workflow_sha,
            )["members"]
            rosters = {owner: tuple(roster) for owner, roster in members.items()}
        except (RuntimeError, ValueError) as exc:
            errors["rosters"] = read_failure(exc)
    if rosters is not None and set(team_owner_ids) <= set(rosters):
        author_handle = f"@{intake.author_login}".casefold()
        reviewers = ReviewerState(
            frozenset(requested or ()), frozenset(submitted or ())
        )
        coverage = covering_handles(codepath=codepath, native_requests=native)
        _, needs = classify_team_owners(
            team_owner_ids=team_owner_ids,
            rosters=rosters,
            coverage=coverage,
            reviewers=reviewers,
            author_handle=author_handle,
        )
        # An owner whose label is missing fails regardless of its history.
        needs = tuple(owner for owner in needs if exists[owner_label(owner)])
        try:
            cursors = fetch_round_robin_cursors(
                PullRequestRef(github=github, repo=args.repository, number=args.pr),
                owners={owner: (owner_label(owner), rosters[owner]) for owner in needs},
            )
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            errors["round_robin"] = read_failure(exc)
    snapshot = ReviewerSnapshot(
        existing, missing, native, requested, submitted, rosters, cursors, errors
    )
    return PlannerInput(intake, ownership, snapshot, args.run_attempt)


def routing_actions(
    *, requests: tuple[RequestReviewers, ...], labels: tuple[str, ...]
) -> tuple[Action, ...]:
    """Return the reviewer requests, then one label action if there are labels."""

    return (*requests, *((AddLabels(labels),) if labels else ()))


def make_plan(
    *, planner_input: PlannerInput, decision: str, actions: tuple[Action, ...] = ()
) -> ActionPlan:
    """Bound one decision's actions by the stage results they came from."""

    intake = planner_input.intake

    ownership = planner_input.ownership
    context = PlanContext(
        intake.identity,
        intake.facts,
        planner_input.run_attempt,
        tuple(ownership.codepath_owners),
        ownership.additional_owners,
        ownership.bypass_intake_matches,
    )
    return ActionPlan(context, decision, actions)


def plan_unadmitted(
    *, planner_input: PlannerInput, why: PlanExplanation
) -> tuple[ActionPlan, PlanExplanation | None]:
    """Decide for an active PR that fails intake and has no bypass match.

    This is the only path that can close a PR. Doubt about a bypass, a failed
    analysis or a discarded claim, never closes it.
    """

    ownership = planner_input.ownership

    snapshot = planner_input.reviewers
    if why.incomplete_reasons:
        require_label(snapshot=snapshot, label_name=BOT_TRIAGE_ERROR_LABEL)
        actions = routing_actions(requests=(), labels=(BOT_TRIAGE_ERROR_LABEL,))
        return make_plan(
            planner_input=planner_input, decision="incomplete", actions=actions
        ), why
    # A close is justified only as the immediate response to the entry event.
    # A rerun is a later, manual action, often after a failure, so it may
    # still route or triage but never produces the one contributor-visible,
    # hard-to-undo effect.
    if ownership.has_discarded_bypass_intake_match or planner_input.run_attempt != 1:
        return make_plan(planner_input=planner_input, decision="kept_open"), None
    require_label(snapshot=snapshot, label_name=BOT_CLOSED_LABEL)
    return make_plan(
        planner_input=planner_input, decision="close", actions=CLOSE_ACTIONS
    ), why


@dataclass(frozen=True)
class TeamOwnerReviewers:
    """Hold each team owner's reviewer; requests and labels derive from it.

    team_owner_ids are the team owners the bot routes, and choices maps each
    resolved one to its reviewer.
    """

    team_owner_ids: tuple[str, ...]
    choices: dict[str, OwnerChoice]
    incomplete_reasons: tuple[str, ...]

    @property
    def roster_users(self) -> tuple[str, ...]:
        """Return the newly selected roster members, as logins to request."""

        picks = {c.reviewer[1:] for c in self.choices.values() if c.state == "selected"}
        return tuple(sorted(picks, key=str.casefold))

    @property
    def owner_labels(self) -> tuple[str, ...]:
        """Return owner: markers for new picks and already-pending reviewers."""

        marked = {"selected", "pending"}
        return tuple(
            owner_label(owner)
            for owner in self.team_owner_ids
            if owner in self.choices and self.choices[owner].state in marked
        )

    @property
    def unresolved_owners(self) -> tuple[str, ...]:
        return tuple(o for o in self.team_owner_ids if o not in self.choices)


def round_robin_team_owners(
    *, planner_input: PlannerInput, codepath: CodepathOwnerTargets
) -> TeamOwnerReviewers:
    """Pick each team owner's reviewer by round robin, unless already covered.

    An owner whose roster member already has a native CODEOWNERS request, a
    submitted review, or a pending request keeps that reviewer, and nobody new
    is asked; otherwise round robin picks the next roster member (see
    select_team_owner_reviewers).
    Direct codepath users and teams are not routed here: they are requested as
    named, through new_reviewer_requests.

    Which owners are optional depends on who requests codepath owners. With
    native CODEOWNERS (the current mode), GitHub covers the codepath owners, so
    everything the bot adds is optional: a failure skips it and records an
    incomplete reason. When the bot requests codepath owners itself, a failure
    that would leave codepath owners unrouted raises instead.
    """

    intake = planner_input.intake

    ownership = planner_input.ownership
    snapshot = planner_input.reviewers
    identity = intake.identity
    native = NATIVE_CODEOWNERS_REQUESTS_CODEPATH_OWNERS
    author_handle = f"@{intake.author_login}".casefold()
    additional_owners = ownership.additional_owners
    # Codepath routing cannot degrade gracefully when the bot requests codepath
    # owners and either some are team owner IDs (they need roster picks) or
    # none are direct handles (so there is no direct request to fall back on).
    needs_codepath_state = not native and (
        codepath.team_owner_ids or not codepath.github_handles
    )
    reasons: list[str] = []

    # team_owner_ids are the team owner IDs the bot routes: the additional
    # owners, plus codepath team owners when the bot requests codepath
    # owners. Each gets a covering existing reviewer or a round-robin pick; an
    # owner with neither is unresolved. Without the reviewer state, coverage is
    # unknown, so nothing is picked.
    team_owner_ids = planned_team_owner_ids(ownership)
    existing = existing_reviewers(snapshot)
    if existing is None and (additional_owners or not native):
        error = snapshot.errors.get("reviewers", "reviewer state was not read")
        if needs_codepath_state:
            raise RuntimeError(error)
        if additional_owners:
            reasons.append(
                f"Reviewer state was unavailable ({error}), so additional owners were skipped."
            )
        return TeamOwnerReviewers(team_owner_ids, {}, tuple(reasons))

    choices: dict[str, OwnerChoice] = {}
    if team_owner_ids:
        existing = existing or ReviewerState(frozenset(), frozenset())
        coverage = covering_handles(
            codepath=codepath, native_requests=snapshot.native_codeowner_requests
        )
        try:
            choices = select_team_owner_reviewers(
                identity=identity,
                snapshot=snapshot,
                team_owner_ids=team_owner_ids,
                coverage=coverage,
                author_handle=author_handle,
                reviewers=existing,
            )
            # The owner: label marks the rotation. New picks already required it;
            # an already-pending reviewer gets it too, to repair a past partial
            # apply that requested the reviewer but never added the label.
            for owner_id, choice in choices.items():
                if choice.state == "pending":
                    require_label(snapshot=snapshot, label_name=owner_label(owner_id))
            unresolved = tuple(o for o in team_owner_ids if o not in choices)
            if unresolved:
                reasons.append(
                    f"No eligible roster member for: {', '.join(unresolved)}."
                )
        except (RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            if needs_codepath_state:
                raise
            detail = f"{type(exc).__name__}: {' '.join(str(exc).split())}"
            reasons.append(
                f"Owner reviewers could not be selected ({detail}), so only "
                "codepath owners were routed."
            )
            choices = {}
    return TeamOwnerReviewers(team_owner_ids, choices, tuple(reasons))


def explain_plan(
    *,
    planner_input: PlannerInput,
    incomplete_reasons: list[str],
    engaged_maintainers: list[tuple[str, EngagementReason]] | None = None,
    team_owners: TeamOwnerReviewers | None = None,
    requests: tuple[RequestReviewers, ...] = (),
    comparison: dict[str, Any] | None = None,
) -> PlanExplanation:
    """Explain a decided plan: its admission, and why each reviewer is involved.

    A reviewer's reasons come from three sources: their own engagement
    (supporter, actionable labeler, maintainer request), a team owner they were
    chosen for or already cover, and a direct codepath owner they are. This is
    log-only: the plan's requests and labels are already decided.
    """

    intake = planner_input.intake

    ownership = planner_input.ownership
    org = intake.identity.repository.split("/", 1)[0]
    author_handle = f"@{intake.author_login}".casefold()
    concerns = {c.owner_id: c for c in ownership.additional_owner_concerns}
    requested = {
        handle.casefold(): handle
        for handle in planned_handles(requests=requests, target_org=org)
    }

    def provenance_for(owner: str) -> dict[str, Any]:
        if owner in ownership.codepath_owners:
            return {"source": "codepath", "files": ownership.codepath_owners[owner]}
        return {"source": "semantic", **concerns[owner].to_dict()}

    # Team owners, as chosen by round_robin_team_owners.
    team_choices = team_owners.choices if team_owners else {}
    owner_reasons = [
        OwnerReason(owner, choice, provenance_for(owner))
        for owner, choice in team_choices.items()
    ]
    # Direct codepath owners: requested by GitHub or by the bot, or already
    # reviewing.
    existing = existing_reviewers(planner_input.reviewers)
    existing = existing or ReviewerState(frozenset(), frozenset())
    for owner in ownership.codepath_owners:
        key = owner.casefold()
        if not owner.startswith("@") or key == author_handle:
            continue
        if NATIVE_CODEOWNERS_REQUESTS_CODEPATH_OWNERS:
            active = comparison is not None and key in comparison["expected"]
            active = active and key in comparison["observed"]
            state = "native_codeowner" if active else None
        elif key in requested:
            state = "selected"
        elif key in existing.submitted_reviewers:
            state = "submitted"
        elif key in existing.requested_reviewers:
            state = "pending"
        else:
            state = None
        if state is not None:
            reason = "direct_codepath_owner" if state == "selected" else None
            choice = OwnerChoice(requested.get(key, owner), state, reason)
            owner_reasons.append(OwnerReason(owner, choice, provenance_for(owner)))

    # Group every reason under its reviewer: engagement first, then codepath
    # owners before semantic ones.
    groups: dict[str, tuple[str, list[EngagementReason | OwnerReason]]] = {}
    for login, reason in engaged_maintainers or []:
        handle = f"@{login}"
        groups.setdefault(handle.casefold(), (handle, []))[1].append(reason)
    owner_reasons.sort(key=lambda r: (r.provenance["source"] != "codepath", r.owner))
    for reason in owner_reasons:
        handle = reason.choice.reviewer
        groups.setdefault(handle.casefold(), (handle, []))[1].append(reason)
    reviewers = tuple(
        ReviewerExplanation(handle, key in requested, tuple(reasons))
        for key, (handle, reasons) in sorted(groups.items())
    )
    return PlanExplanation(
        admission_items(intake=intake, ownership=ownership),
        incomplete_reasons,
        ownership.has_uncovered_concerns,
        reviewers,
        team_owners.unresolved_owners if team_owners else (),
        comparison,
    )


def build_action_plan(
    planner_input: PlannerInput,
) -> tuple[ActionPlan, PlanExplanation | None]:
    """Decide one plan from the planner input alone.

    Read it top to bottom as a funnel with four exits, each closing one case:
    1. An inactive or handled PR: kept_open, with nothing to explain.
    2. A PR that is not admitted (fails intake, no bypass match):
       plan_unadmitted decides; it is the only code that can close a PR.
    3. An admitted PR with no codepath owners and no additional owners: nobody
       is requested for ownership, so it is triaged only if a handoff reviewer
       is engaged, whatever its concerns.
    4. An admitted PR with owners: request a reviewer for each team owner ID
       (from round_robin_team_owners), any direct codepath users and teams
       the bot requests itself, and the supporters and actionable labelers.
       The decision is then the first that applies:
       - incomplete, if any routing fell short;
       - routed_untriaged, if a concern has no owner and there is no handoff
         reviewer;
       - otherwise triage.
    Between exits 2 and 3, setup gathers what both admitted cases share: the
    supporter and actionable-labeler candidates, whether there is a handoff
    reviewer, and the CODEOWNERS comparison. In both, every candidate is
    filtered for anyone already reviewing.

    incomplete_reasons is built once, right after admission, and any entry
    makes the plan incomplete. The explanation is log-only (None when there is
    nothing to explain). Nothing here reads GitHub or logs, so any plan can be replayed
    from its input.
    """

    intake = planner_input.intake

    ownership = planner_input.ownership
    snapshot = planner_input.reviewers
    facts = intake.facts
    org = intake.identity.repository.split("/", 1)[0]
    codepath = CodepathOwnerTargets(tuple(ownership.codepath_owners))

    # Exit 1: inactive or handled.
    if not facts.is_active:
        return make_plan(planner_input=planner_input, decision="kept_open"), None

    llm_failed = ["The LLM run failed; only codepath owners remain."]
    llm_reasons = llm_failed if ownership.llm_run_status == "failed" else []

    # Exit 2: not admitted. A PR is admitted if it passes intake or a team's
    # bypass intake matched. A PR that is not admitted is decided by
    # plan_unadmitted, the only code that can close a PR. Only a failed LLM run
    # can make it incomplete: routing never runs for it.
    if not facts.passes_intake and not ownership.bypass_intake_matches:
        return plan_unadmitted(
            planner_input=planner_input,
            why=explain_plan(
                planner_input=planner_input, incomplete_reasons=llm_reasons
            ),
        )

    # The PR is admitted from here on. Team owner IDs get their reviewers from
    # round_robin_team_owners (none without owners), whose failures add
    # incomplete reasons rather than raising, unless the bot itself must
    # request codepath owners. Any incomplete reason makes the plan incomplete:
    # whatever routing was safe still happens, but the PR gets bot-triage-error
    # instead of triaged. This is the complete list.
    team_owners = None
    if codepath.values or ownership.additional_owners:
        team_owners = round_robin_team_owners(
            planner_input=planner_input, codepath=codepath
        )
    routing_reasons = team_owners.incomplete_reasons if team_owners else ()
    incomplete_reasons = [*llm_reasons, *routing_reasons]

    # Setup shared by exits 3 and 4: supporters and actionable labelers are
    # candidates whether or not the PR has owners, and new_reviewer_requests
    # later drops every candidate who already reviews, whatever their reason.
    engaged_maintainers = find_engaged_maintainers(intake=intake, snapshot=snapshot)
    existing = existing_reviewers(snapshot) or ReviewerState(frozenset(), frozenset())
    author_handle = f"@{intake.author_login}".casefold()
    # A handoff reviewer is trusted to pull in reviewers for uncovered concerns.
    # A team that opted in through bypass intake is trusted like a supporter,
    # so its chosen reviewer also counts.
    chosen = team_owners.choices if team_owners else {}
    has_handoff = bool(
        facts.supporters
        or facts.actionable_labelers
        or facts.maintainer_requested_reviewers
        or any(owner in chosen for owner in ownership.bypass_intake_matches)
    )
    comparison = None
    if NATIVE_CODEOWNERS_REQUESTS_CODEPATH_OWNERS:
        comparison = compare_codeowners(
            snapshot=snapshot,
            workflow_sha=intake.identity.workflow_sha,
            codepath=codepath,
            author_login=intake.author_login,
        )

    # Exit 3: no codepath owners and no additional semantic owners, so nobody
    # is requested for ownership, though supporters and actionable labelers
    # still can be. The PR stays untriaged, unless there is a handoff reviewer.
    if team_owners is None:
        if incomplete_reasons:
            decision, labels = "incomplete", (BOT_TRIAGE_ERROR_LABEL,)
        elif has_handoff:
            decision, labels = "triage", (TRIAGED_LABEL, BOT_TRIAGED_LABEL)
        else:
            decision, labels = "kept_open", ()
        for label_name in labels:
            require_label(snapshot=snapshot, label_name=label_name)
        requests = new_reviewer_requests(
            engaged_maintainers=engaged_maintainers,
            existing=existing,
            author_handle=author_handle,
            org=org,
        )
        plan = make_plan(
            planner_input=planner_input,
            decision=decision,
            actions=routing_actions(requests=requests, labels=labels),
        )
        return plan, explain_plan(
            planner_input=planner_input,
            incomplete_reasons=incomplete_reasons,
            engaged_maintainers=engaged_maintainers,
            requests=requests,
            comparison=comparison,
        )

    # Exit 4: owners exist. Build the review requests: each team owner's
    # chosen reviewer, plus direct codepath users and teams when the bot, not
    # native CODEOWNERS, requests codepath owners. Every candidate then goes
    # through the same filter for anyone already reviewing.
    native = NATIVE_CODEOWNERS_REQUESTS_CODEPATH_OWNERS
    direct = ((), ()) if native else (codepath.github_users, codepath.github_teams)
    requests = new_reviewer_requests(
        engaged_maintainers=engaged_maintainers,
        owner_candidates={
            "codepath_owner": direct,
            "owner_roster": (team_owners.roster_users, ()),
        },
        existing=existing,
        author_handle=author_handle,
        org=org,
    )
    owner_targets = sum(
        len(request.users) + len(request.teams)
        for request in requests
        if request.reason in {"codepath_owner", "owner_roster"}
    )
    if owner_targets > MAX_REVIEW_REQUESTS:
        raise ValueError("review request exceeds Auto PR Triage's 15-target limit")
    if incomplete_reasons:
        decision, status_labels = "incomplete", (BOT_TRIAGE_ERROR_LABEL,)
    elif ownership.has_uncovered_concerns and not has_handoff:
        decision, status_labels = "routed_untriaged", ()
    else:
        decision, status_labels = "triage", (TRIAGED_LABEL, BOT_TRIAGED_LABEL)
    for label_name in status_labels:
        require_label(snapshot=snapshot, label_name=label_name)
    plan = make_plan(
        planner_input=planner_input,
        decision=decision,
        actions=routing_actions(
            requests=requests,
            labels=tuple(dict.fromkeys((*status_labels, *team_owners.owner_labels))),
        ),
    )
    return plan, explain_plan(
        planner_input=planner_input,
        incomplete_reasons=incomplete_reasons,
        engaged_maintainers=engaged_maintainers,
        team_owners=team_owners,
        requests=requests,
        comparison=comparison,
    )


def parse_args() -> argparse.Namespace:
    """Parse the event identity and the stage directory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pr", type=int, help="pull request number")
    parser.add_argument("--repository", required=True)
    parser.add_argument("--workflow-sha", required=True)
    parser.add_argument("--run-attempt", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--github-output", type=Path, required=True)
    parser.add_argument("--github-step-summary", type=Path)
    return parser.parse_args()


def main() -> int:
    """Publish one plan for the live apply job and the run artifacts."""

    args = parse_args()
    try:
        intake = IntakeResult.from_json((args.output_dir / "intake.json").read_text())
        ownership = OwnershipResult.from_json(
            (args.output_dir / "ownership.json").read_text()
        )
        github = GitHubReader(None)
        planner_input = gather_planner_input(
            args=args, intake=intake, ownership=ownership, github=github
        )
        (args.output_dir / "planner_input.json").write_text(
            json.dumps(planner_input.to_dict(), indent=2, sort_keys=True) + "\n"
        )
        plan, why = build_action_plan(planner_input)
        if why is not None:
            log_plan(plan=plan, why=why, summary=args.github_step_summary)
        value = plan.to_json()
        (args.output_dir / "plan.json").write_text(
            json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n"
        )
        with args.github_output.open("a") as output:
            output.write(f"action-plan-json={value}\n")
    except Exception as exc:
        detail = " ".join(str(exc).split())
        print(
            f"Auto PR Triage planning failed: {type(exc).__name__}: {detail}",
            file=sys.stderr,
        )
        return 1
    print(f"Planned {args.repository}#{args.pr}: {plan.decision}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
