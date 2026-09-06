"""Land-time verification that greenlight reviewed the commit that is about to land.

`merge_rules.yaml`'s "Greenlight Review Bot" rule lets a single `pytorchgreenlight`
approval authorize a merge of any path, and pytorch/pytorch has no GitHub-native
required reviews, so that approval is never dismissed when the author pushes more
commits. This module compares the PR's head commit against the head SHA greenlight
recorded in its ledger and refuses the merge when they disagree, waiting instead when
greenlight simply has not answered yet.

It must not import `trymerge`: `trymerge` imports this module, and the cycle would
break both. Everything it needs about a PR arrives as plain data or callables, and it
returns a verdict rather than raising trymerge's exception types.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from greenlight_identity import GREENLIGHT_LOGIN, is_greenlight
from greenlight_ledger import (
    fetch_ledger_states,
    GreenlightLedgerError,
    IN_FLIGHT_STATUSES,
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_LAND,
    STATUS_NO_LAND,
    STATUS_REVERTED,
)
from greenlight_messages import (
    describe_ledger,
    FIX_CREDENTIAL,
    FIX_HUMAN,
    FIX_RETRIGGER,
    FIX_TRANSPORT,
    FIX_UNSETTLED,
    FORCE_SUFFIX,
    LEDGER_UNREADABLE,
    render_announcement,
    render_refusal,
    render_transport_announcement,
    render_wait,
    timeout_suffix,
    transport_timeout_suffix,
)
from greenlight_preflight import cannot_review, FIX_REVERTED


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from greenlight_ledger import LedgerState


MAX_WAIT_MINUTES = 60
MAX_WAIT = timedelta(minutes=MAX_WAIT_MINUTES)

# An unreadable ledger gets its own, much shorter budget. The client has already retried
# in-process, and a ledger that stays down is not the asynchronous reviewer the verdict
# budget exists for: the hour buys a review time to finish, not a route time to recover.
TRANSPORT_MAX_WAIT_MINUTES = 15
TRANSPORT_MAX_WAIT = timedelta(minutes=TRANSPORT_MAX_WAIT_MINUTES)

# greenlight re-dispatches a review that has outlived its own timeout and writes a fresh
# row when it does, so an older in-flight row is proof that no verdict is coming for this
# commit. The 5 minutes on top of greenlight's DEFAULT_TIMEOUT_MINUTES of 45 are slack for
# the gap between that timeout expiring and the next scan acting on it. greenlight's CLI
# takes the 45 as --timeout-minutes, so an override there silently skews this check.
DEAD_REVIEW_AGE = timedelta(minutes=50)

# Not one of greenlight's groupings: upstream splits NO_LAND (terminal) from CANCELLED
# and FAILED (retry-eligible). At land time they all mean the same thing, namely that
# greenlight did not say LAND for the commit in front of us.
NEGATIVE_STATUSES = frozenset(
    {STATUS_NO_LAND, STATUS_CANCELLED, STATUS_FAILED, STATUS_REVERTED}
)

_SHA_RE = re.compile(r"[0-9a-f]{40}")


class GuardVerdict(enum.Enum):
    ALLOW = "ALLOW"
    WAIT = "WAIT"
    DENY = "DENY"


@dataclass(frozen=True)
class GuardResult:
    verdict: GuardVerdict
    message: str = ""
    comment: str = ""


@dataclass(frozen=True)
class PRUnderMerge:
    """One PR a merge is about to land, plus the facts the guard needs about it.

    ``head_sha`` is the PR's GitHub head, which is what greenlight records even on the
    ghstack path, where the commit git actually cherry-picks is a different revision
    carrying the same content. The callables are callables because each costs a request
    and only the minority of PRs that greenlight alone authorizes ever need them.

    ``get_bot_reviewers`` returns None when GitHub could not be asked which reviewers are
    App accounts, which is different from a PR that has none.
    """

    pr_num: int
    head_sha: str
    approved_by: Sequence[str]
    changes_requested_by: Sequence[str]
    labels: Sequence[str]
    get_updated_at: Callable[[], str | None]
    get_bot_reviewers: Callable[[], frozenset[str] | None]
    get_merge_authorized_logins: Callable[[], frozenset[str]]
    is_authorized_without_greenlight: Callable[[], bool]


@dataclass(frozen=True)
class Outcome:
    verdict: GuardVerdict
    kind: str
    detail: str
    pr_num: int
    head_sha: str
    ledger: str
    fix: str = ""


@dataclass
class GreenlightWaitWindow:
    """The wait budgets for one merge command, carried across ``merge_into`` re-entries.

    A budget opens at its own first wait rather than when the merge command was issued:
    `merge_into` is reached only once CI is green, and pytorch CI routinely runs for
    hours, so a budget anchored at the command would already be spent before the guard
    ever ran once.

    An unreadable ledger runs on a separate, shorter anchor. The two never share one,
    because a merge deep into a verdict wait would otherwise get no transport budget at
    all, and an early HUD blip would otherwise still be counted against a fresh one far
    later. Each budget announces itself once per merge command, so a merge that crosses
    from one to the other tells the PR the deadline it actually moved to, and a PR never
    collects more than the two comments.
    """

    opened_at: datetime | None = None
    transport_opened_at: datetime | None = None
    announced: bool = False
    transport_announced: bool = False

    def register_wait(self, now: datetime) -> bool:
        """Record that the guard is waiting. True once the budget is spent."""
        if self.opened_at is None:
            self.opened_at = now
        return now - self.opened_at >= MAX_WAIT

    def register_transport_wait(self, now: datetime) -> bool:
        """Record that the ledger is unreadable. True once that budget is spent."""
        if self.transport_opened_at is None:
            self.transport_opened_at = now
        return now - self.transport_opened_at >= TRANSPORT_MAX_WAIT

    def clear_transport_wait(self) -> None:
        """Forget an earlier unreadable ledger, now that one has been read.

        Only the anchor resets. This runs on every successful read, so a flapping ledger
        would otherwise announce once per failure episode, and a running commentary of
        bot comments is worse for the PR than one stale heads-up beside a refusal that
        names the real cause.
        """
        self.transport_opened_at = None

    def claim_announcement(self) -> bool:
        """True for the first verdict wait of this merge command, and never again."""
        if self.announced:
            return False
        self.announced = True
        return True

    def claim_transport_announcement(self) -> bool:
        """True for the first unreadable-ledger wait of this merge command only."""
        if self.transport_announced:
            return False
        self.transport_announced = True
        return True


def _same_commit(left: str, right: str) -> bool:
    """Whether two shas name the same commit.

    Both sides have to be full 40-hex shas. The entire gate is this one equality, and a
    pair of empty strings -- an unreadable PR head, a ledger row written without one --
    would otherwise compare equal and wave the merge through.
    """
    normalized = left.strip().lower()
    if not _SHA_RE.fullmatch(normalized):
        return False
    return normalized == right.strip().lower()


def _settled_outcome(pr: PRUnderMerge, state: LedgerState, now: datetime) -> Outcome:
    ledger = describe_ledger(state)
    if state.status in IN_FLIGHT_STATUSES:
        if now - state.version >= DEAD_REVIEW_AGE:
            return Outcome(
                GuardVerdict.DENY,
                "DEAD_REVIEW",
                "greenlight's review of this commit has been running far longer than "
                "a review takes, so no verdict is coming",
                pr.pr_num,
                pr.head_sha,
                ledger,
                FIX_RETRIGGER,
            )
        return Outcome(
            GuardVerdict.WAIT,
            "IN_FLIGHT",
            "greenlight is still reviewing this commit",
            pr.pr_num,
            pr.head_sha,
            ledger,
            FIX_UNSETTLED,
        )
    if state.status == STATUS_LAND:
        return Outcome(GuardVerdict.ALLOW, "LAND", "", pr.pr_num, pr.head_sha, ledger)
    fix = FIX_REVERTED if state.status == STATUS_REVERTED else FIX_HUMAN
    if state.status in NEGATIVE_STATUSES:
        detail = f"greenlight recorded {state.status} for this exact commit"
        kind = "NEGATIVE_VERDICT"
    else:
        detail = f"greenlight recorded the unrecognized status {state.status} for this commit"
        kind = "UNKNOWN_STATUS"
    return Outcome(GuardVerdict.DENY, kind, detail, pr.pr_num, pr.head_sha, ledger, fix)


def _evaluate_pr(pr: PRUnderMerge, state: LedgerState | None, now: datetime) -> Outcome:
    if state is not None and _same_commit(state.head_sha, pr.head_sha):
        return _settled_outcome(pr, state, now)

    ledger = describe_ledger(state)
    # Nothing settled for this commit, so landing it depends on greenlight dispatching a
    # fresh review. Refuse now rather than after a full wait when it structurally cannot.
    blocked = cannot_review(pr, state.status if state is not None else None, now)
    if blocked is not None:
        if blocked.retryable:
            return Outcome(
                GuardVerdict.WAIT,
                "UNCLASSIFIED_REVIEWER",
                "greenlight may not produce a verdict for this commit because "
                f"{blocked.cause}",
                pr.pr_num,
                pr.head_sha,
                ledger,
                blocked.fix,
            )
        return Outcome(
            GuardVerdict.DENY,
            "CANNOT_REVIEW",
            "greenlight cannot produce a verdict for this commit because "
            f"{blocked.cause}",
            pr.pr_num,
            pr.head_sha,
            ledger,
            blocked.fix,
        )
    if state is None:
        return Outcome(
            GuardVerdict.WAIT,
            "NO_ROW",
            "greenlight has not reviewed this PR yet",
            pr.pr_num,
            pr.head_sha,
            ledger,
            FIX_UNSETTLED,
        )
    return Outcome(
        GuardVerdict.WAIT,
        "SHA_MISMATCH",
        "greenlight's latest verdict is for a different commit",
        pr.pr_num,
        pr.head_sha,
        ledger,
        FIX_UNSETTLED,
    )


def _transport_outcome(pr: PRUnderMerge, error: GreenlightLedgerError) -> Outcome:
    verdict, kind, fix = (
        (GuardVerdict.DENY, "LEDGER_CREDENTIAL", FIX_CREDENTIAL)
        if error.fatal
        else (GuardVerdict.WAIT, "TRANSPORT", FIX_TRANSPORT)
    )
    return Outcome(
        verdict,
        kind,
        f"greenlight's ledger could not be read: {error}",
        pr.pr_num,
        pr.head_sha,
        LEDGER_UNREADABLE,
        fix,
    )


def _needs_greenlight(pr: PRUnderMerge) -> bool:
    if not any(is_greenlight(login) for login in pr.approved_by):
        return False
    return not pr.is_authorized_without_greenlight()


def evaluate_greenlight_guard(
    repo_full_name: str,
    prs: Sequence[PRUnderMerge],
    *,
    wait_window: GreenlightWaitWindow | None,
    now: datetime | None = None,
    fetch_states: Callable[[str, Sequence[int]], dict[int, LedgerState]] | None = None,
) -> GuardResult:
    """Decide whether the PRs a single merge will land may proceed.

    ``wait_window`` holds the merge command's wait budgets, shared across every re-entry.
    None means the caller (a force merge) has no retry loop to wait in, so nothing waits.

    ``fetch_states`` is resolved here rather than bound as a default argument, so that
    patching the ledger reader in a test actually takes effect.
    """
    moment = now if now is not None else datetime.now(timezone.utc)
    fetch = fetch_states if fetch_states is not None else fetch_ledger_states
    guarded = [pr for pr in prs if _needs_greenlight(pr)]
    if not guarded:
        return GuardResult(GuardVerdict.ALLOW)

    on_transport_budget = False
    try:
        states = fetch(repo_full_name, [pr.pr_num for pr in guarded])
    except GreenlightLedgerError as e:
        # A brief HUD blip must not kill a merge that is otherwise ready to land, so a
        # retryable failure spends the transport budget first and only a sustained outage
        # ends in a refusal. A fatal cause has nothing to wait for and refuses now.
        outcomes = [_transport_outcome(pr, e) for pr in guarded]
        on_transport_budget = not e.fatal
    else:
        if wait_window is not None:
            wait_window.clear_transport_wait()
        outcomes = [_evaluate_pr(pr, states.get(pr.pr_num), moment) for pr in guarded]

    if all(o.verdict is GuardVerdict.ALLOW for o in outcomes):
        return GuardResult(GuardVerdict.ALLOW)
    denials = [o for o in outcomes if o.verdict is GuardVerdict.DENY]
    # Anything neither allowed nor refused holds the merge. Deriving the waits positively
    # would let a verdict added later match no branch and fall through to the allow above.
    waits = [
        o for o in outcomes if o.verdict not in (GuardVerdict.ALLOW, GuardVerdict.DENY)
    ]
    if denials:
        # Every unresolved PR in the stack is reported, waits included: the merge is
        # over either way, and fixing only the refusal would just hit the wait next.
        # Refusals lead, because they are the ones the reader has to act on.
        return GuardResult(GuardVerdict.DENY, render_refusal([*denials, *waits]))

    if wait_window is None:
        return GuardResult(GuardVerdict.DENY, render_refusal(waits, FORCE_SUFFIX))
    if on_transport_budget:
        spent = wait_window.register_transport_wait(moment)
        suffix = transport_timeout_suffix(TRANSPORT_MAX_WAIT_MINUTES)
        announcement = render_transport_announcement(waits, TRANSPORT_MAX_WAIT_MINUTES)
        claim_comment = wait_window.claim_transport_announcement
    else:
        spent = wait_window.register_wait(moment)
        suffix = timeout_suffix(MAX_WAIT_MINUTES)
        announcement = render_announcement(waits, MAX_WAIT_MINUTES, GREENLIGHT_LOGIN)
        claim_comment = wait_window.claim_announcement
    if spent:
        return GuardResult(GuardVerdict.DENY, render_refusal(waits, suffix))
    return GuardResult(
        GuardVerdict.WAIT, render_wait(waits), announcement if claim_comment() else ""
    )
