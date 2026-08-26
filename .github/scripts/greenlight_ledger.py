"""Client for greenlight's authoritative per-PR verdict ledger.

Greenlight records one row per review run in ClickHouse's `misc.greenlight_pr_state`,
naming the head SHA that run evaluated. This module reads the latest row per PR through
the HUD route that fronts that table, so the merge path can tell whether greenlight has
approved the exact commit about to land.

Row selection lives server-side and mirrors greenlight's own reader: writer and reader
have to agree on which row is authoritative.
"""

from __future__ import annotations

import json
import os
import time
import traceback
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from github_utils import gh_fetch_url


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


STATUS_LAND = "LAND"
STATUS_NO_LAND = "NO_LAND"
STATUS_CANCELLED = "CANCELLED"
STATUS_FAILED = "FAILED"
STATUS_AI_REVIEW_STARTED = "AI_REVIEW_STARTED"
STATUS_AI_REVIEW_DISPATCHED = "AI_REVIEW_DISPATCHED"

IN_FLIGHT_STATUSES = frozenset({STATUS_AI_REVIEW_STARTED, STATUS_AI_REVIEW_DISPATCHED})
# greenlight's own split, from greenlight/src/greenlight/constants.py. A terminal status
# is one its scan will never revisit; CANCELLED and FAILED are retry-eligible there, so a
# PR carrying one can still be picked up again.
TERMINAL_STATUSES = frozenset({STATUS_LAND, STATUS_NO_LAND})

LEDGER_URL = "https://hud.pytorch.org/api/greenlight/pr_state"

# The route rejects longer batches. ghstack stacks never come close, but a rejected
# batch fails closed and would block a merge that nobody can unblock.
_MAX_PRS_PER_REQUEST = 50
_NUM_RETRIES = 3
_RETRY_BACKOFF_SECONDS = 2.0


@dataclass(frozen=True)
class LedgerState:
    pr_number: int
    status: str
    head_sha: str
    run_id: int
    version: datetime


class GreenlightLedgerError(RuntimeError):
    pass


def parse_utc_timestamp(raw: str) -> datetime:
    text = raw.strip()
    # trymerge runs on Python 3.10, whose fromisoformat rejects the trailing Z that
    # both this route and the GitHub REST API emit.
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_states(payload: Any) -> dict[int, LedgerState]:
    states: dict[int, LedgerState] = {}
    for row in payload["states"]:
        state = LedgerState(
            pr_number=int(row["pr_number"]),
            status=str(row["status"]),
            head_sha=str(row["head_sha"]),
            run_id=int(row["run_id"]),
            version=parse_utc_timestamp(str(row["version"])),
        )
        states[state.pr_number] = state
    return states


def _fetch_batch(
    repo_full_name: str, pr_numbers: list[int], sleep: Callable[[float], None]
) -> dict[int, LedgerState]:
    query = urllib.parse.urlencode(
        {"repo": repo_full_name, "prNumbers": ",".join(str(n) for n in pr_numbers)}
    )
    last_error: Exception | None = None
    for attempt in range(_NUM_RETRIES):
        try:
            return _parse_states(
                gh_fetch_url(
                    f"{LEDGER_URL}?{query}",
                    headers={"x-hud-internal-bot": os.getenv("HUD_API_TOKEN", "")},
                    reader=json.load,
                )
            )
        except Exception as e:
            last_error = e
            print(
                f"Greenlight ledger read {attempt + 1}/{_NUM_RETRIES} for "
                f"{repo_full_name} PRs {pr_numbers} failed: {type(e).__name__}: {e}",
                flush=True,
            )
            if attempt + 1 < _NUM_RETRIES:
                sleep(_RETRY_BACKOFF_SECONDS * 2**attempt)
            else:
                # The guard turns the raised error into a one-line message for the PR,
                # so this is the only place the stack of the real failure is visible.
                traceback.print_exc()
    raise GreenlightLedgerError(
        f"could not read the greenlight ledger for {repo_full_name} PRs {pr_numbers} "
        f"after {_NUM_RETRIES} attempts: {type(last_error).__name__}: {last_error}"
    ) from last_error


def fetch_ledger_states(
    repo_full_name: str,
    pr_numbers: Sequence[int],
    *,
    sleep: Callable[[float], None] | None = None,
) -> dict[int, LedgerState]:
    """Latest ledger row per PR. PRs with no row are absent from the result."""
    wait = sleep if sleep is not None else time.sleep
    states: dict[int, LedgerState] = {}
    ordered = list(pr_numbers)
    for start in range(0, len(ordered), _MAX_PRS_PER_REQUEST):
        batch = ordered[start : start + _MAX_PRS_PER_REQUEST]
        states.update(_fetch_batch(repo_full_name, batch, wait))
    return states
