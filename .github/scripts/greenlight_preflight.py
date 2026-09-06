"""Detect PRs greenlight will never produce a verdict for, so a merge can refuse early.

Mirrors the upstream filters that drop a PR before it is ever fingerprinted, both in
pytorch/test-infra under `greenlight/src/greenlight/`: `review._recency_filter` (the
excluded labels and the review window, each with an escape for a PR whose recorded status
is non-terminal) and both arms of `review_gate.human_review_skip_reason` (changes
requested by a non-bot, and an approval from the merge-authorized set). Drift from those
costs a merge an hour of waiting for a verdict that was never coming, or refuses a PR
greenlight would in fact have reviewed.

What it does not mirror is the `--pr` recheck path, on which upstream passes
`skip_on_approval=False` and so reviews an approved PR anyway. pytorch/pytorch has no way
to reach that path, so an approval is terminal here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import NamedTuple, TYPE_CHECKING

from greenlight_identity import is_greenlight, is_known_bot, normalize_login
from greenlight_ledger import parse_utc_timestamp, STATUS_REVERTED, TERMINAL_STATUSES


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from datetime import datetime

    from greenlight_guard import PRUnderMerge


# greenlight's EXCLUDED_LABELS and its STALE_LABEL member, from its constants.py. The set
# is mirrored rather than the member so that a label added upstream lands here too. GitHub
# labels are case-sensitive and the pytorch stale bot applies this exact name.
STALE_LABEL = "Stale"
EXCLUDED_LABELS = frozenset({STALE_LABEL})

# greenlight's review_window_hours default. Outside it the scan drops the PR before
# fingerprinting, so no verdict will ever arrive. Upstream reads it from
# PYTORCH_GREENLIGHT_REVIEW_WINDOW_HOURS, so an override there silently skews this check.
REVIEW_WINDOW = timedelta(hours=24)


def _label_phrase(labels: Sequence[str]) -> str:
    names = ", ".join(f"`{label}`" for label in labels)
    return f"{names} label" if len(labels) == 1 else f"{names} labels"


EXCLUDED_LABELS_PHRASE = _label_phrase(sorted(EXCLUDED_LABELS))

FIX_REVERTED = (
    "Get an approval from a human reviewer with merge rights. Greenlight does not "
    "review a PR again after it has been reverted, so pushing a commit cannot help."
)
FIX_CHANGES_REQUESTED = (
    "Resolve or dismiss that review, or get an approval from a human reviewer with "
    "merge rights."
)
FIX_UNCLASSIFIED_REVIEWER = (
    "Re-issue the merge command. If that review is from a person, resolve or dismiss "
    "it, or get an approval from a human reviewer with merge rights."
)
FIX_HUMAN_APPROVED = (
    "Dismiss that approval, or get an approval from a reviewer the merge rule for "
    "these files accepts."
)
FIX_STALE_LABEL = (
    f"Remove the {EXCLUDED_LABELS_PHRASE}, or get an approval from a human reviewer "
    "with merge rights."
)
FIX_OUT_OF_WINDOW = (
    "Push a commit to bring it back into range, or get an approval from a human "
    "reviewer with merge rights."
)


@dataclass(frozen=True)
class CannotReview:
    """Why greenlight will not produce a verdict for this PR, and how to fix it.

    ``retryable`` marks a cause the guard could not confirm because a GitHub lookup
    failed. A transport failure must not harden into a permanent merge refusal, so those
    are reported as something to wait on rather than something settled.
    """

    cause: str
    fix: str
    retryable: bool = False


class ReviewerScan(NamedTuple):
    """Logins that read as people, and whether GitHub answered which ones are Apps."""

    humans: list[str]
    classified: bool


class _AppReviewers:
    """The PR's App-account reviewers, fetched at most once per evaluation."""

    def __init__(self, pr: PRUnderMerge) -> None:
        self._pr = pr
        self._loaded = False
        self._logins: frozenset[str] | None = None

    def logins(self) -> frozenset[str] | None:
        """Normalized App logins, or None when the lookup failed."""
        if not self._loaded:
            reported = self._pr.get_bot_reviewers()
            self._logins = (
                None
                if reported is None
                else frozenset(normalize_login(login) for login in reported)
            )
            self._loaded = True
        return self._logins


def _named_bots_removed(logins: Iterable[str]) -> list[str]:
    return sorted(
        {
            login
            for login in logins
            if not is_known_bot(login) and not is_greenlight(login)
        }
    )


def _apps_removed(candidates: list[str], apps: _AppReviewers) -> ReviewerScan:
    """Drop the candidates GitHub reports as App accounts.

    GraphQL gives no account type, so an App whose login is neither suffixed nor listed
    in greenlight's own bot set is indistinguishable from a person until REST answers.
    """
    if not candidates:
        return ReviewerScan([], True)
    known_apps = apps.logins()
    if known_apps is None:
        return ReviewerScan(candidates, False)
    return ReviewerScan(
        [login for login in candidates if normalize_login(login) not in known_apps],
        True,
    )


def blocking_reviewers(
    pr: PRUnderMerge, apps: _AppReviewers | None = None
) -> ReviewerScan:
    """The non-bot logins whose requested changes stop greenlight reviewing this PR."""
    return _apps_removed(
        _named_bots_removed(pr.changes_requested_by), apps or _AppReviewers(pr)
    )


def _merge_authorized_approvers(pr: PRUnderMerge, apps: _AppReviewers) -> ReviewerScan:
    """The PR's approvers that greenlight counts as a human merge authority.

    Membership is the flat lowercased union of every `approved_by` entry in
    merge_rules.yaml, ignoring file patterns, because that is what upstream's
    `merge_authz.resolve_authorized_logins` builds and what the scan compares against.
    """
    candidates = _named_bots_removed(pr.approved_by)
    if not candidates:
        return ReviewerScan([], True)
    authorized = pr.get_merge_authorized_logins()
    listed = [login for login in candidates if normalize_login(login) in authorized]
    return _apps_removed(listed, apps)


def _outside_review_window(pr: PRUnderMerge, now: datetime) -> str | None:
    updated_at = pr.get_updated_at()
    if updated_at is None:
        return None
    try:
        updated = parse_utc_timestamp(updated_at)
    except ValueError as e:
        print(
            f"Unparsable updated_at {updated_at!r} for PR #{pr.pr_num}, skipping the "
            f"greenlight review-window check: {e}",
            flush=True,
        )
        return None
    if updated >= now - REVIEW_WINDOW:
        return None
    return updated_at


def cannot_review(
    pr: PRUnderMerge, ledger_status: str | None, now: datetime
) -> CannotReview | None:
    """Why greenlight will never dispatch a review for this PR, and how to fix it."""
    if ledger_status == STATUS_REVERTED:
        return CannotReview(
            "the PR was reverted and greenlight never reviews it again", FIX_REVERTED
        )
    apps = _AppReviewers(pr)
    blocking = blocking_reviewers(pr, apps)
    if blocking.humans:
        listed = ", ".join(blocking.humans)
        if not blocking.classified:
            return CannotReview(
                f"GitHub did not say whether {listed} is a person or an app, and "
                "greenlight skips any PR whose requested changes a person left "
                "unresolved",
                FIX_UNCLASSIFIED_REVIEWER,
                retryable=True,
            )
        return CannotReview(
            f"{listed} requested changes, and greenlight skips any PR with unresolved "
            "requested changes",
            FIX_CHANGES_REQUESTED,
        )
    approved = _merge_authorized_approvers(pr, apps)
    if approved.humans:
        listed = ", ".join(approved.humans)
        if not approved.classified:
            return CannotReview(
                f"GitHub did not say whether {listed} is a person or an app, and "
                "greenlight skips any PR a person listed in merge_rules.yaml has "
                "approved",
                FIX_UNCLASSIFIED_REVIEWER,
                retryable=True,
            )
        return CannotReview(
            f"{listed} already approved it, and greenlight skips any PR approved by a "
            "login listed anywhere in merge_rules.yaml, even when that approval does "
            "not satisfy the merge rule for the files this PR changes",
            FIX_HUMAN_APPROVED,
        )
    # A non-terminal status is upstream's escape from the recency filter: its scan keeps
    # a labelled or out-of-window PR whose last review is in flight, cancelled, or
    # failed, because it can still re-dispatch that one.
    if ledger_status is not None and ledger_status not in TERMINAL_STATUSES:
        return None
    present = sorted(EXCLUDED_LABELS.intersection(pr.labels))
    if present:
        return CannotReview(
            f"it carries the {_label_phrase(present)}, and greenlight skips labelled PRs",
            FIX_STALE_LABEL,
        )
    stale_since = _outside_review_window(pr, now)
    if stale_since is not None:
        hours = int(REVIEW_WINDOW.total_seconds() // 3600)
        return CannotReview(
            f"the PR was last updated at {stale_since}, outside greenlight's {hours}h "
            "review window",
            FIX_OUT_OF_WINDOW,
        )
    return None
