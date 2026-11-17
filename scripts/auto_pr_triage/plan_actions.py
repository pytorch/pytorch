#!/usr/bin/env python3
"""Stage 3: plan the exact GitHub effects of one intake and ownership result.

This runs in the read-only analyze job. main() does three things in order:

1. gather_planner_input makes every GitHub read planning may depend on (labels,
   reviewers, rosters) and bundles them with the stage results into
   planner_input.json.
2. build_action_plan is a pure function of that input. It returns the
   ActionPlan, the job output the live apply job executes, and a log-only
   PlanExplanation.
3. log_plan writes the explanation to the job log and the step summary.

Shadow mode stops after this stage; live mode then runs apply_actions.py.
"""

from __future__ import annotations

import argparse
import hashlib
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
from identifiers import USER_HANDLE_RE
from reviewer_state import (
    fetch_requested_reviewer_handles,
    fetch_submitted_review_state,
)
from schemas import (
    Action,
    ActionPlan,
    AddLabels,
    BOT_TRIAGE_ERROR_LABEL,
    BOT_TRIAGED_LABEL,
    check_stage_results,
    IntakeResult,
    MAX_REVIEW_REQUESTS,
    MISSING_ACTIONABLE_ISSUE_ACTIONS,
    MISSING_ACTIONABLE_ISSUE_LABEL,
    MISSING_ACTIONABLE_ISSUE_LABELS,
    OwnershipResult,
    PlanContext,
    PlannerInput,
    PullRequestIdentity,
    REQUEST_REASONS,
    RequestReviewers,
    ReviewerSnapshot,
    TRIAGED_LABEL,
)
from step_summary import summary_code, summary_diff, summary_prose
from trusted_config import load_team_members


# GitHub requests codepath owners through CODEOWNERS; the bot never does.
# Planning reads them to tell whether the PR has owners, to skip a roster pick
# for a team whose member is already a codepath owner, and to explain who
# covers the PR.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MAX_LOGGED_OWNER_FILES = 16


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
    excerpt: Callable[[str], list[str]]


LOG_STYLE = TextStyle(
    escape_log_fragment,
    lambda value: f"`{escape_log_fragment(value)}`",
    lambda handle: handle,
    lambda excerpt: [
        f"    {escape_log_fragment(line)}" for line in excerpt.splitlines()
    ],
)
# Summaries show `login` rather than @login, so pasting one never pings anyone.
SUMMARY_STYLE = TextStyle(
    summary_prose,
    summary_code,
    lambda handle: f"`{handle.removeprefix('@')}`",
    lambda excerpt: [f"  {line}" for line in summary_diff(excerpt)],
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
    reasons = {
        "selected": (
            f"{reviewer} was picked at random from `{owner}`'s roster, seeded by "
            "this PR so that reruns pick the same member."
        ),
        "codeowners_pending": (
            f"{reviewer} is a codepath owner in CODEOWNERS and already has a pending "
            "request; no new request is needed."
        ),
        "codeowners_submitted": (
            f"{reviewer} is a codepath owner in CODEOWNERS and already submitted a "
            "review; no new request is needed."
        ),
        "codeowners_missing": (
            f"{reviewer} is a codepath owner in CODEOWNERS but has no pending request "
            "or review, so GitHub may not have requested them or someone may have "
            "removed the request."
        ),
        "codeowners_unconfirmed": (
            f"{reviewer} is a codepath owner in CODEOWNERS; GitHub requests codepath "
            "owners, but the reviewer state could not be read to confirm."
        ),
        "codepath_covered": (
            f"{reviewer} covers `{owner}` through codepath ownership; no `{owner}` "
            "roster pick was needed."
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
    """Render evidence items, each followed by its diff excerpt."""

    lines = []
    for item in evidence:
        lines.append(f"- {style.code(item['file'])}: " + style.prose(item["relevance"]))
        lines.extend(style.excerpt(item["diff_excerpt"]))
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
        return [f"Codepath owner `{owner}` matched CODEOWNERS rules for: {files}."]
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


# A team owner's reviewer is selected, codepath_covered, submitted, or pending; a
# codepath owner's state says what reviewer state showed about GitHub's request.
ChoiceState = Literal[
    "selected",
    "codepath_covered",
    "submitted",
    "pending",
    "codeowners_pending",
    "codeowners_submitted",
    "codeowners_missing",
    "codeowners_unconfirmed",
]


@dataclass(frozen=True)
class OwnerChoice:
    """Record how one owner resolved to one reviewer.

    For a team owner, state says whether the reviewer is newly picked from its
    roster (selected) or already covers it: as a codepath owner, with a
    submitted review, or with a pending request. For a codepath owner, whom
    GitHub requests through CODEOWNERS, state says whether reviewer state showed
    a pending request, a review, neither, or could not be read.
    """

    reviewer: str
    state: ChoiceState

    def to_log(self) -> dict[str, Any]:
        """Return the logged form."""

        return asdict(self)


def classify_team_owners(
    *,
    team_owner_ids: tuple[str, ...],
    rosters: dict[str, tuple[str, ...]],
    codepath_handles: frozenset[str],
    reviewers: ReviewerState,
    author_handle: str,
) -> tuple[dict[str, OwnerChoice], tuple[str, ...]]:
    """Return owners an existing reviewer covers, and owners needing a new pick.

    A roster member who is a codepath owner covers the team, because GitHub
    requests codepath owners through CODEOWNERS.
    """

    choices: dict[str, OwnerChoice] = {}
    needs_pick: list[str] = []
    for owner_id in team_owner_ids:
        members = {member.casefold(): member for member in rosters[owner_id]}
        covered = (codepath_handles & set(members)) - {author_handle}
        submitted = (set(members) & reviewers.submitted_reviewers) - {author_handle}
        pending = (set(members) & reviewers.requested_reviewers) - {author_handle}
        if covered:
            choices[owner_id] = OwnerChoice(min(covered), "codepath_covered")
        elif submitted:
            choices[owner_id] = OwnerChoice(members[min(submitted)], "submitted")
        elif pending:
            choices[owner_id] = OwnerChoice(members[min(pending)], "pending")
        else:
            needs_pick.append(owner_id)
    return choices, tuple(needs_pick)


def pick_roster_member(
    *,
    repository: str,
    number: int,
    owner: str,
    members: tuple[str, ...],
    author_handle: str,
) -> str | None:
    """Pick a roster member other than the author, seeded by the PR and owner.

    Picks spread evenly across members like a random choice, but the same PR
    and owner always get the same member, so reruns and replays agree.
    """

    eligible = [member for member in members if member.casefold() != author_handle]
    if not eligible:
        return None
    seed = hashlib.sha256(f"{repository}:{number}:{owner}".encode()).digest()
    return eligible[int.from_bytes(seed[:8], "big") % len(eligible)]


def select_team_owner_reviewers(
    *,
    identity: PullRequestIdentity,
    snapshot: ReviewerSnapshot,
    team_owner_ids: tuple[str, ...],
    codepath_handles: frozenset[str],
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
    choices, needs_pick = classify_team_owners(
        team_owner_ids=team_owner_ids,
        rosters=rosters,
        codepath_handles=codepath_handles,
        reviewers=reviewers,
        author_handle=author_handle,
    )
    for owner_id in needs_pick:
        reviewer = pick_roster_member(
            repository=identity.repository,
            number=identity.number,
            owner=owner_id,
            members=rosters[owner_id],
            author_handle=author_handle,
        )
        if reviewer is not None:
            choices[owner_id] = OwnerChoice(reviewer, "selected")
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

    @property
    def owner_choices(self) -> dict[str, dict[str, Any]]:
        """Return each owner's reviewer in its logged, owner-keyed form."""

        return {
            reason.owner: {**reason.choice.to_log(), "provenance": reason.provenance}
            for group in self.reviewers
            for reason in group.reasons
            if isinstance(reason, OwnerReason)
        }


def owner_section(*, owner: str, anchor: str, details: list[str]) -> list[str]:
    """Render one semantic owner's details under a heading the reviewer table links to."""

    lines = ["", f'<a id="{anchor}"></a>', "", f"### Why `{owner}` owns this"]
    for line in details:
        # Excerpt lines are indented under their evidence item; the rest are
        # list items or paragraphs, which need a blank line to end a list.
        if line.startswith(("  ", "- ")):
            lines.append(line)
        else:
            lines.extend(["", line])
    return lines


def render_step_summary(
    *,
    plan: ActionPlan,
    why: PlanExplanation,
    planned: tuple[str, ...],
    labels: list[str],
) -> str:
    """Render who was requested and why, then the plan record, collapsed."""

    handle = SUMMARY_STYLE.handle
    names = ", ".join(handle(r) for r in planned)
    heading = f"requests review from {names}" if planned else "no reviewer requested"
    label_names = ", ".join(f"`{label}`" for label in labels)
    lines = [
        f"## Auto PR Triage: {heading}",
        "",
        "| | |",
        "|---|---|",
        f"| Decision | `{plan.decision}` |",
        f"| Admission | {admission_text(items=why.admission, style=SUMMARY_STYLE)} |",
    ]
    if why.incomplete_reasons:
        reasons = " ".join(summary_prose(r) for r in why.incomplete_reasons)
        lines.append(f"| Why this run is incomplete | {reasons} |")
    lines.append(f"| Intended labels | {label_names or 'none'} |")

    # Codepath owners fit in the reviewer's row; each semantic owner's longer
    # provenance gets its own section below, linked from the row.
    sections = []
    if why.reviewers:
        lines += [
            "",
            "### Reviewers",
            "",
            "| Reviewer | Request | Why |",
            "|---|---|---|",
        ]
    for group in why.reviewers:
        sentences, _ = reviewer_explanation(group=group, style=SUMMARY_STYLE)
        cell = [" ".join(sentences)]
        for reason in group.reasons:
            if not isinstance(reason, OwnerReason):
                continue
            details = owner_details(
                owner=reason.owner, provenance=reason.provenance, style=SUMMARY_STYLE
            )
            if reason.provenance["source"] == "codepath":
                cell.extend(details)
                continue
            anchor = f"auto-pr-triage-why-{reason.owner}"
            cell.append(f"[See why `{reason.owner}` owns this](#{anchor})")
            sections += owner_section(
                owner=reason.owner, anchor=anchor, details=details
            )
        status = "new request" if group.requested else "no new request"
        lines.append(f"| {handle(group.reviewer)} | {status} | {'<br>'.join(cell)} |")
    lines += sections

    reviewer_names = ", ".join(f"`{r.removeprefix('@')}`" for r in planned)
    unresolved_names = ", ".join(f"`{owner}`" for owner in why.unresolved_owners)
    uncovered = str(why.has_uncovered_concerns).lower()
    lines += [
        "",
        "<details><summary>Auto PR Triage decision plan</summary>",
        "",
        f"- Decision: `{plan.decision}`",
        f"- Has uncovered concerns: `{uncovered}`",
        f"- Planned reviewer requests: {reviewer_names or 'none'}",
        f"- Unresolved owners: {unresolved_names or 'none'}",
    ]
    for owner, choice in sorted(why.owner_choices.items()):
        reviewer = choice["reviewer"].removeprefix("@")
        lines.append(f"- Owner `{owner}`: `{reviewer}` ({choice['state']})")
    lines += [f"- Intended labels: {label_names}", "", "</details>", ""]
    return "\n".join(lines)


def log_plan(*, plan: ActionPlan, why: PlanExplanation, summary: Path | None) -> None:
    """Log one deterministic pre-effect plan and write its step summary."""

    context = plan.context
    requests = tuple(a for a in plan.actions if isinstance(a, RequestReviewers))
    planned = planned_handles(requests)
    labels = [
        label for a in plan.actions if isinstance(a, AddLabels) for label in a.labels
    ]
    record = {
        "admission": why.admission,
        "decision": plan.decision,
        "incomplete_reasons": why.incomplete_reasons,
        "analyzed_head_sha": context.identity.head_sha,
        "has_uncovered_concerns": why.has_uncovered_concerns,
        "codepath_owners": list(context.codepath_owners),
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
    try:
        with summary.open("a", encoding="utf-8") as output:
            output.write(
                render_step_summary(plan=plan, why=why, planned=planned, labels=labels)
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
    roster_users: tuple[str, ...] = (),
    existing: ReviewerState,
    author_handle: str,
) -> tuple[RequestReviewers, ...]:
    """Build one request per reason for candidates who are not reviewing yet.

    Supporters and actionable labelers come from engaged_maintainers, and
    owner_roster from roster_users. Anyone with a pending request or a submitted
    review is dropped, as is the author, and each user is requested once, under
    the first reason in REQUEST_REASONS that names them.
    """

    candidates = {
        kind: tuple(login for login, r in engaged_maintainers if r.kind == kind)
        for kind in ("supporter", "actionable_labeler")
    }
    candidates["owner_roster"] = roster_users
    seen: set[str] = set()
    requests = []
    for reason in REQUEST_REASONS:
        fresh = {
            user.casefold(): user
            for user in candidates[reason]
            if user.casefold() not in seen
            and f"@{user}".casefold() not in existing.current_reviewers
            and f"@{user}".casefold() != author_handle
        }
        seen |= set(fresh)
        if fresh:
            users = tuple(fresh[key] for key in sorted(fresh))
            requests.append(RequestReviewers(users, reason))
    return tuple(requests)


def planned_handles(requests: tuple[RequestReviewers, ...]) -> tuple[str, ...]:
    """Return every requested reviewer as a handle."""

    return tuple(f"@{user}" for request in requests for user in request.users)


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

    Reads follow the same short circuits as planning, so an inactive PR reads
    nothing. Failed reads are recorded, not raised, except label reads.

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
        empty = ReviewerSnapshot((), (), None, None, None, errors)
        return PlannerInput(intake, ownership, empty, args.run_attempt)
    labels = (
        TRIAGED_LABEL,
        BOT_TRIAGED_LABEL,
        BOT_TRIAGE_ERROR_LABEL,
        MISSING_ACTIONABLE_ISSUE_LABEL,
    )
    exists = {
        label: label_exists(github=github, repository=args.repository, label_name=label)
        for label in labels
    }
    existing = tuple(label for label in labels if exists[label])
    missing = tuple(label for label in labels if not exists[label])

    requested = submitted = None
    rosters: dict[str, tuple[str, ...]] | None = None
    admitted = facts.passes_intake or bool(ownership.bypass_intake_matches)
    owners = ownership.codepath_owners or ownership.additional_owners
    if admitted and (owners or engagement_candidates(intake)):
        try:
            state = fetch_reviewer_state(
                PullRequestRef(github=github, repo=args.repository, number=args.pr)
            )
            requested = tuple(sorted(state.requested_reviewers))
            submitted = tuple(sorted(state.submitted_reviewers))
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            errors["reviewers"] = read_failure(exc)
    if requested is not None and submitted is not None and ownership.additional_owners:
        try:
            members = load_team_members(
                repository_root=REPOSITORY_ROOT,
                repo=args.repository,
                ref=args.workflow_sha,
            )["members"]
            rosters = {owner: tuple(roster) for owner, roster in members.items()}
        except (RuntimeError, ValueError) as exc:
            errors["rosters"] = read_failure(exc)
    snapshot = ReviewerSnapshot(
        existing, missing, requested, submitted, rosters, errors
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

    This is the only path that can mark a PR as missing an actionable issue.
    Doubt about a bypass, a failed analysis or a discarded claim, never marks it.
    """

    ownership = planner_input.ownership

    snapshot = planner_input.reviewers
    if why.incomplete_reasons:
        require_label(snapshot=snapshot, label_name=BOT_TRIAGE_ERROR_LABEL)
        actions = routing_actions(requests=(), labels=(BOT_TRIAGE_ERROR_LABEL,))
        return make_plan(
            planner_input=planner_input, decision="incomplete", actions=actions
        ), why
    # Marking a PR answers its entry event only. A rerun is a later, manual
    # action, often after a failure, so it may still route or triage an
    # admitted PR but leaves an unadmitted one for a human.
    if ownership.has_discarded_bypass_intake_match or planner_input.run_attempt != 1:
        return make_plan(planner_input=planner_input, decision="kept_open"), None
    for label_name in MISSING_ACTIONABLE_ISSUE_LABELS:
        require_label(snapshot=snapshot, label_name=label_name)
    return make_plan(
        planner_input=planner_input,
        decision="missing_actionable_issue",
        actions=MISSING_ACTIONABLE_ISSUE_ACTIONS,
    ), why


@dataclass(frozen=True)
class TeamOwnerReviewers:
    """Hold each team owner's reviewer; its requests derive from it.

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
    def unresolved_owners(self) -> tuple[str, ...]:
        return tuple(o for o in self.team_owner_ids if o not in self.choices)


def resolve_team_owner_reviewers(planner_input: PlannerInput) -> TeamOwnerReviewers:
    """Pick each additional owner's reviewer from its roster, unless covered.

    An owner whose roster member is a codepath owner, has submitted a review, or
    has a pending request keeps that reviewer, and nobody new is asked;
    otherwise a roster member is picked at random, seeded by the PR (see
    pick_roster_member). GitHub requests codepath owners through CODEOWNERS, so
    every reviewer picked here is optional: a failure skips it and records an
    incomplete reason instead of raising.
    """

    intake = planner_input.intake

    ownership = planner_input.ownership
    snapshot = planner_input.reviewers
    team_owner_ids = ownership.additional_owners
    if not team_owner_ids:
        return TeamOwnerReviewers((), {}, ())
    # Without the reviewer state, coverage is unknown, so nothing is picked.
    existing = existing_reviewers(snapshot)
    if existing is None:
        error = snapshot.errors.get("reviewers", "reviewer state was not read")
        reason = f"Reviewer state was unavailable ({error}), so additional owners were skipped."
        return TeamOwnerReviewers(team_owner_ids, {}, (reason,))
    try:
        choices = select_team_owner_reviewers(
            identity=intake.identity,
            snapshot=snapshot,
            team_owner_ids=team_owner_ids,
            codepath_handles=frozenset(o.casefold() for o in ownership.codepath_owners),
            author_handle=f"@{intake.author_login}".casefold(),
            reviewers=existing,
        )
    except (RuntimeError, ValueError) as exc:
        detail = f"{type(exc).__name__}: {' '.join(str(exc).split())}"
        reason = f"Owner reviewers could not be selected ({detail}), so additional owners were skipped."
        return TeamOwnerReviewers(team_owner_ids, {}, (reason,))
    unresolved = tuple(o for o in team_owner_ids if o not in choices)
    reasons = (f"No eligible roster member for: {', '.join(unresolved)}.",)
    return TeamOwnerReviewers(team_owner_ids, choices, reasons if unresolved else ())


def explain_plan(
    *,
    planner_input: PlannerInput,
    incomplete_reasons: list[str],
    engaged_maintainers: list[tuple[str, EngagementReason]] | None = None,
    team_owners: TeamOwnerReviewers | None = None,
    requests: tuple[RequestReviewers, ...] = (),
) -> PlanExplanation:
    """Explain a decided plan: its admission, and why each reviewer is involved.

    A reviewer's reasons come from three sources: their own engagement
    (supporter, actionable labeler, maintainer request), a team owner they were
    chosen for or already cover, and a codepath owner they are, whom GitHub
    requests through CODEOWNERS. This is log-only: the plan's requests and labels
    are already decided.
    """

    intake = planner_input.intake

    ownership = planner_input.ownership
    author_handle = f"@{intake.author_login}".casefold()
    concerns = {c.owner_id: c for c in ownership.additional_owner_concerns}
    requested = {handle.casefold() for handle in planned_handles(requests)}

    def provenance_for(owner: str) -> dict[str, Any]:
        if owner in ownership.codepath_owners:
            return {"source": "codepath", "files": ownership.codepath_owners[owner]}
        return {"source": "semantic", **concerns[owner].to_dict()}

    # Team owners, as chosen by resolve_team_owner_reviewers.
    team_choices = team_owners.choices if team_owners else {}
    owner_reasons = [
        OwnerReason(owner, choice, provenance_for(owner))
        for owner, choice in team_choices.items()
    ]
    # Codepath owners other than the author, whom GitHub requests through
    # CODEOWNERS; reviewer state shows whether that request is visible.
    existing = existing_reviewers(planner_input.reviewers)
    for owner in ownership.codepath_owners:
        key = owner.casefold()
        if key == author_handle:
            continue
        if existing is None:
            state = "codeowners_unconfirmed"
        elif key in existing.submitted_reviewers:
            state = "codeowners_submitted"
        elif key in existing.requested_reviewers:
            state = "codeowners_pending"
        else:
            state = "codeowners_missing"
        choice = OwnerChoice(owner, state)
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
    )


def build_action_plan(
    planner_input: PlannerInput,
) -> tuple[ActionPlan, PlanExplanation | None]:
    """Decide one plan from the planner input alone.

    Read it top to bottom as a funnel with four exits, each closing one case:
    1. An inactive or handled PR: kept_open, with nothing to explain.
    2. A PR that is not admitted (fails intake, no bypass match):
       plan_unadmitted decides; it is the only code that can mark a PR as
       missing an actionable issue.
    3. An admitted PR with no codepath owners and no additional owners: nobody
       is requested for ownership, so it is triaged only if a handoff reviewer
       is engaged, whatever its concerns.
    4. An admitted PR with owners: request a reviewer for each additional owner
       (from resolve_team_owner_reviewers) and the supporters and actionable
       labelers; GitHub requests the codepath owners through CODEOWNERS.
       The decision is then the first that applies:
       - incomplete, if any routing fell short;
       - routed_untriaged, if a concern has no owner and there is no handoff
         reviewer;
       - otherwise triage.
    Between exits 2 and 3, setup gathers what both admitted cases share: the
    supporter and actionable-labeler candidates and whether there is a handoff
    reviewer. In both, every candidate is filtered for anyone already reviewing.

    incomplete_reasons is built once, right after admission, and any entry
    makes the plan incomplete. The explanation is log-only (None when there is
    nothing to explain). Nothing here reads GitHub or logs, so any plan can be replayed
    from its input.
    """

    intake = planner_input.intake

    ownership = planner_input.ownership
    snapshot = planner_input.reviewers
    facts = intake.facts

    # Exit 1: inactive or handled.
    if not facts.is_active:
        return make_plan(planner_input=planner_input, decision="kept_open"), None

    llm_failed = ["The LLM run failed; only codepath owners remain."]
    llm_reasons = llm_failed if ownership.llm_run_status == "failed" else []

    # Exit 2: not admitted. A PR is admitted if it passes intake or a team's
    # bypass intake matched. A PR that is not admitted is decided by
    # plan_unadmitted, the only code that can mark a PR as missing an
    # actionable issue. Only a failed LLM run can make it incomplete: routing
    # never runs for it.
    if not facts.passes_intake and not ownership.bypass_intake_matches:
        return plan_unadmitted(
            planner_input=planner_input,
            why=explain_plan(
                planner_input=planner_input, incomplete_reasons=llm_reasons
            ),
        )

    # The PR is admitted from here on. Additional owners get their reviewers
    # from resolve_team_owner_reviewers, whose failures add incomplete reasons
    # rather than raising. Any incomplete reason makes the plan incomplete:
    # whatever routing was safe still happens, but the PR gets bot-triage-error
    # instead of triaged. This is the complete list.
    team_owners = None
    if ownership.codepath_owners or ownership.additional_owners:
        team_owners = resolve_team_owner_reviewers(planner_input)
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
        )

    # Exit 4: owners exist. Request each additional owner's chosen reviewer,
    # through the same filter for anyone already reviewing.
    requests = new_reviewer_requests(
        engaged_maintainers=engaged_maintainers,
        roster_users=team_owners.roster_users,
        existing=existing,
        author_handle=author_handle,
    )
    owner_targets = sum(
        len(request.users) for request in requests if request.reason == "owner_roster"
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
            labels=status_labels,
        ),
    )
    return plan, explain_plan(
        planner_input=planner_input,
        incomplete_reasons=incomplete_reasons,
        engaged_maintainers=engaged_maintainers,
        team_owners=team_owners,
        requests=requests,
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
