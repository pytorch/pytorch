"""Render greenlight guard outcomes as the text a user reads.

Four surfaces share one set of outcomes: the refusal that ends a merge, the line the
merge command's retry loop logs while it waits, and one comment posted to the PR for each
of the two things a merge can wait on. The fix line for a PR greenlight structurally
cannot review belongs to `greenlight_preflight` instead, next to the check that raises it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from greenlight_ledger import LEDGER_TOKEN_ENV, LEDGER_URL
from greenlight_preflight import EXCLUDED_LABELS_PHRASE


if TYPE_CHECKING:
    from collections.abc import Sequence

    from greenlight_guard import Outcome
    from greenlight_ledger import LedgerState


GREENLIGHT_JOB_URL = (
    "https://github.com/pytorch/test-infra/actions/workflows/greenlight-pr-review.yml"
)

_REVIEWS_RUN_AT = f"Greenlight reviews run at {GREENLIGHT_JOB_URL}"

_ONLY_AUTHORITY = (
    "Greenlight's approval is the only one authorizing this merge, so it cannot land "
    "until greenlight records a verdict for the exact commit being landed."
)
# An unreadable ledger says nothing about whether greenlight reviewed the commit, and the
# verdict may well already be recorded. Claiming none exists sends the reader after a
# review that is not missing.
_UNREADABLE_AUTHORITY = (
    "Greenlight's approval is the only one authorizing this merge, so it cannot land "
    "until greenlight's verdict for the exact commit being landed can be read."
)

FIX_HUMAN = (
    "To land this now, get an approval from a human reviewer with merge rights for "
    "these files, or push a commit so greenlight reviews the new content."
)
FIX_RETRIGGER = (
    "Push a commit to trigger a fresh greenlight review, or get an approval from a "
    "human reviewer with merge rights."
)
FIX_UNSETTLED = (
    "To land without greenlight, get an approval from a human reviewer with merge "
    "rights for these files. To get a greenlight verdict instead, push a commit, "
    f"remove the {EXCLUDED_LABELS_PHRASE} if it is set, and resolve any requested "
    "changes."
)
FIX_TRANSPORT = (
    "Re-issue the merge command; if it keeps failing, ask Dev Infra to check the HUD "
    f"greenlight route at {LEDGER_URL}."
)
# Deliberately not FIX_TRANSPORT's advice: the merge workflow reads the credential from
# a repository secret, so every re-issued command fails in exactly the same place.
FIX_CREDENTIAL = (
    "Re-issuing the merge command cannot help. Ask Dev Infra to restore the merge "
    f"workflow's {LEDGER_TOKEN_ENV} secret."
)

# The same escape hatches, worded for a merge that is still running: `merge` aborts the
# moment a new commit appears on the PR, so pushing one here is not a free retrigger.
_WAIT_OPTIONS = (
    "To land without waiting, get an approval from a human reviewer with merge rights "
    f"for these files, or remove the {EXCLUDED_LABELS_PHRASE} and resolve any requested "
    "changes so greenlight picks the PR up. Pushing a commit gets greenlight to review "
    "the new content but ends this merge command, so re-issue the merge afterwards."
)

# Not _WAIT_OPTIONS: greenlight may already have recorded a verdict for this very commit,
# so every escape hatch that works by getting greenlight to look at the PR is beside the
# point. What is unreadable is the ledger, and only a human approval routes around it.
_TRANSPORT_OPTIONS = (
    "To land without waiting, get an approval from a human reviewer with merge rights "
    "for these files. Pushing a commit or changing labels cannot help, because it is the "
    "ledger rather than the PR that cannot be read."
)

FORCE_SUFFIX = "A force merge does not wait for greenlight."

LEDGER_UNREADABLE = "Greenlight ledger: could not be read"


def describe_ledger(state: LedgerState | None) -> str:
    if state is None:
        return "Greenlight ledger: no row for this PR"
    return (
        f"Greenlight ledger: {state.status} at {state.head_sha} "
        f"(run {state.run_id}, recorded {state.version.isoformat()})"
    )


def timeout_suffix(minutes: int) -> str:
    return f"Greenlight did not answer after {minutes} minutes of waiting."


def transport_timeout_suffix(minutes: int) -> str:
    # A credential the route rejects is well-formed enough to reach it and comes back as
    # an ordinary error status, so it is indistinguishable from an outage until here.
    return (
        f"Greenlight's ledger was still unreadable when this merge's {minutes}-minute "
        f"retry budget ran out. That is either the HUD greenlight route being down or a "
        f"{LEDGER_TOKEN_ENV} it rejects, and nothing this merge sees tells the two apart."
    )


def _sentence(text: str) -> str:
    return f"{text[:1].upper()}{text[1:]}."


def _bullet(outcome: Outcome, *, with_fix: bool) -> str:
    lines = [
        f"- PR #{outcome.pr_num} ({outcome.kind}): {outcome.detail}.",
        f"  - Head commit: `{outcome.head_sha}`.",
        f"  - {outcome.ledger}.",
    ]
    if with_fix and outcome.fix:
        lines.append(f"  - {outcome.fix}")
    return "\n".join(lines)


def _authority_paragraph(outcomes: Sequence[Outcome]) -> str:
    """Why the merge is blocked, which is a different sentence for an unreadable ledger.

    Only when every listed PR is blocked on the read: a stack that mixes causes still has
    at least one PR for which the ordinary wording holds.
    """
    if all(outcome.ledger == LEDGER_UNREADABLE for outcome in outcomes):
        return _UNREADABLE_AUTHORITY
    return _ONLY_AUTHORITY


def render_refusal(outcomes: Sequence[Outcome], suffix: str = "") -> str:
    """The refusal that ends a merge, leading with what actually went wrong.

    ``outcomes`` is ordered by the caller so that the PR the user has to act on comes
    first; its detail becomes the headline, because that is the sentence people read.
    """
    parts = [_sentence(outcomes[0].detail)]
    if suffix:
        parts.append(suffix)
    parts.append(_authority_paragraph(outcomes))
    parts.append("\n".join(_bullet(outcome, with_fix=True) for outcome in outcomes))
    parts.append(_REVIEWS_RUN_AT)
    return "\n\n".join(parts)


def render_wait(outcomes: Sequence[Outcome]) -> str:
    parts = [
        "Waiting on greenlight before merging.",
        *(
            f"PR #{o.pr_num} ({o.kind}): {o.detail}. Head commit: {o.head_sha}. "
            f"{o.ledger}."
            for o in outcomes
        ),
        _REVIEWS_RUN_AT,
    ]
    return " ".join(parts)


def _announced_list(outcomes: Sequence[Outcome]) -> str:
    return "\n".join(
        f"- PR #{outcome.pr_num}: {outcome.detail} (head commit `{outcome.head_sha}`)."
        for outcome in outcomes
    )


def render_announcement(outcomes: Sequence[Outcome], minutes: int, login: str) -> str:
    """The comment a merge posts to the PR when it starts waiting for a verdict."""
    return "\n\n".join(
        (
            f"This merge is waiting for `{login}` to review the commit it will land.",
            _ONLY_AUTHORITY,
            _announced_list(outcomes),
            f"The merge command keeps retrying for up to {minutes} minutes. "
            f"{_WAIT_OPTIONS}",
            _REVIEWS_RUN_AT,
        )
    )


def render_transport_announcement(outcomes: Sequence[Outcome], minutes: int) -> str:
    """The comment a merge posts to the PR when it starts waiting on a broken read."""
    return "\n\n".join(
        (
            "This merge cannot read greenlight's verdict ledger, so it cannot tell "
            "whether greenlight approved the commit it would land.",
            _UNREADABLE_AUTHORITY,
            _announced_list(outcomes),
            f"The merge command keeps retrying for up to {minutes} minutes. "
            f"{_TRANSPORT_OPTIONS}",
            _REVIEWS_RUN_AT,
        )
    )
