#!/usr/bin/env python3

from __future__ import annotations

import enum
import json
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest import main, mock, TestCase
from urllib.error import HTTPError, URLError

import greenlight_ledger
from greenlight_guard import (
    evaluate_greenlight_guard,
    GreenlightWaitWindow,
    GuardVerdict,
    MAX_WAIT,
    MAX_WAIT_MINUTES,
    Outcome,
    PRUnderMerge,
    TRANSPORT_MAX_WAIT,
    TRANSPORT_MAX_WAIT_MINUTES,
)
from greenlight_ledger import (
    fetch_ledger_states,
    GreenlightLedgerError,
    LEDGER_TOKEN_ENV,
    STATUS_AI_REVIEW_STARTED,
    STATUS_LAND,
    STATUS_NO_LAND,
)
from greenlight_messages import FIX_CREDENTIAL, FIX_TRANSPORT, transport_timeout_suffix
from greenlight_preflight import STALE_LABEL
from test_greenlight_guard import make_pr, MERGE_SHA, NOW, REPO


# A run_id as production emits it: an unquoted JSON number far past 32 bits.
LARGE_RUN_ID = 32747018107

LEDGER_TOKEN = "token"
# Not a typo for None: four characters the client can send and only the route can
# reject, which is what makes it a credential no local check can catch.
REJECTED_TOKEN = "None"


class _FutureVerdict(enum.Enum):
    """Stands in for a GuardVerdict member the shipped aggregation has never seen."""

    HOLD = "HOLD"


def make_row(
    status: str = STATUS_LAND,
    version: str = "2026-08-25T11:59:00.000Z",
    *,
    pr_number: int = 1,
    head_sha: str = MERGE_SHA,
    run_id: int = 7,
) -> dict[str, Any]:
    return {
        "pr_number": pr_number,
        "status": status,
        "head_sha": head_sha,
        "run_id": run_id,
        "version": version,
    }


def evaluate(
    *,
    prs: list[PRUnderMerge] | None = None,
    wait_window: GreenlightWaitWindow | None = None,
    now: datetime = NOW,
) -> Any:
    return evaluate_greenlight_guard(
        REPO,
        prs if prs is not None else [make_pr()],
        wait_window=wait_window if wait_window is not None else GreenlightWaitWindow(),
        now=now,
    )


class LedgerReadTestCase(TestCase):
    """Base for the tests that reach the client's request path.

    The client refuses before asking when the credential is missing, and a test run's
    ambient environment has no reason to carry the secret the merge workflow injects.
    """

    def setUp(self) -> None:
        patcher = mock.patch.dict("os.environ", {LEDGER_TOKEN_ENV: LEDGER_TOKEN})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _http_error(self, code: int = 500, reason: str = "boom") -> HTTPError:
        return HTTPError(greenlight_ledger.LEDGER_URL, code, reason, {}, None)  # type: ignore[arg-type]


class TestLedgerTransport(LedgerReadTestCase):
    @mock.patch("greenlight_ledger.gh_fetch_url")
    def test_reads_all_prs_in_one_request(self, fetch_url: mock.MagicMock) -> None:
        fetch_url.return_value = {
            "states": [
                {
                    "pr_number": 5,
                    "status": STATUS_LAND,
                    "head_sha": MERGE_SHA,
                    "run_id": 3,
                    "version": "2026-08-25T11:59:00.000Z",
                }
            ]
        }
        states = fetch_ledger_states(REPO, [5, 6])

        fetch_url.assert_called_once()
        url = fetch_url.call_args.args[0]
        self.assertIn("repo=pytorch%2Fpytorch", url)
        self.assertIn("prNumbers=5%2C6", url)
        self.assertEqual(
            fetch_url.call_args.kwargs["headers"], {"x-hud-internal-bot": LEDGER_TOKEN}
        )
        self.assertEqual(set(states), {5})
        self.assertEqual(states[5].run_id, 3)
        self.assertEqual(
            states[5].version, datetime(2026, 8, 25, 11, 59, tzinfo=timezone.utc)
        )

    @mock.patch("greenlight_ledger.gh_fetch_url")
    def test_unknown_response_fields_are_ignored(
        self, fetch_url: mock.MagicMock
    ) -> None:
        """The route is free to grow fields without breaking a deployed trymerge."""
        fetch_url.return_value = {
            "states": [
                {
                    "pr_number": 5,
                    "status": STATUS_LAND,
                    "head_sha": MERGE_SHA,
                    "run_id": 3,
                    "version": "2026-08-25T11:59:00.000Z",
                    "a_field_added_later": {"nested": True},
                }
            ],
            "an_envelope_field_added_later": 1,
        }
        states = fetch_ledger_states(REPO, [5])
        self.assertEqual(states[5].head_sha, MERGE_SHA)
        self.assertEqual(states[5].status, STATUS_LAND)

    @mock.patch("greenlight_ledger.gh_fetch_url")
    def test_every_route_error_response_is_transport_class(
        self, fetch_url: mock.MagicMock
    ) -> None:
        for code, reason in (
            (400, "repo must be owner/name"),
            (400, "repo and prNumbers are required"),
            (400, "prNumbers must be positive integers"),
            (400, "at most 50 prNumbers"),
            (401, "unauthorized"),
            (405, "method not allowed"),
            (500, "internal error"),
        ):
            with self.subTest(code=code, reason=reason):
                fetch_url.side_effect = self._http_error(code, reason)
                with self.assertRaises(GreenlightLedgerError) as cm:
                    fetch_ledger_states(REPO, [5], sleep=lambda _s: None)
                self.assertIn(str(code), str(cm.exception))
                self.assertIn(reason, str(cm.exception))
                self.assertFalse(cm.exception.fatal)

    def _blank_credential_error(self, raw: str) -> GreenlightLedgerError:
        with mock.patch.dict("os.environ", {LEDGER_TOKEN_ENV: raw}):
            with self.assertRaises(GreenlightLedgerError) as cm:
                fetch_ledger_states(REPO, [5], sleep=lambda _s: None)
        return cm.exception

    @mock.patch("greenlight_ledger.gh_fetch_url")
    def test_an_empty_credential_is_fatal_and_sends_no_request(
        self, fetch_url: mock.MagicMock
    ) -> None:
        error = self._blank_credential_error("")
        self.assertTrue(error.fatal)
        self.assertEqual(str(error), f"{LEDGER_TOKEN_ENV} is not set")
        fetch_url.assert_not_called()

    @mock.patch("greenlight_ledger.gh_fetch_url")
    def test_a_whitespace_credential_is_fatal_and_not_reported_as_absent(
        self, fetch_url: mock.MagicMock
    ) -> None:
        """A secret that is present but blank sends Dev Infra somewhere else entirely."""
        expected = f"{LEDGER_TOKEN_ENV} holds only whitespace"
        for raw in (" ", "\n", " \t\n", "\x0b", "\xa0", "\x85"):
            with self.subTest(raw=raw):
                error = self._blank_credential_error(raw)
                self.assertTrue(error.fatal)
                self.assertEqual(str(error), expected)
        fetch_url.assert_not_called()

    @mock.patch("greenlight_ledger.gh_fetch_url")
    def test_an_unsendable_credential_is_fatal_and_never_quoted(
        self, fetch_url: mock.MagicMock
    ) -> None:
        for separator in ("\n", "\r", "\x7f", "\u00e9"):
            with self.subTest(separator=separator):
                secret = f"s3cr3t{separator}tail"
                with mock.patch.dict("os.environ", {LEDGER_TOKEN_ENV: secret}):
                    with self.assertRaises(GreenlightLedgerError) as cm:
                        fetch_ledger_states(REPO, [5], sleep=lambda _s: None)
                self.assertTrue(cm.exception.fatal)
                self.assertIn(LEDGER_TOKEN_ENV, str(cm.exception))
                self.assertNotIn("s3cr3t", str(cm.exception))
        fetch_url.assert_not_called()

    @mock.patch("greenlight_ledger.gh_fetch_url")
    def test_a_credential_stored_with_surrounding_whitespace_still_authenticates(
        self, fetch_url: mock.MagicMock
    ) -> None:
        """`gh secret set < file` keeps whatever trailing newline the file had."""
        fetch_url.return_value = {"states": []}
        with mock.patch.dict("os.environ", {LEDGER_TOKEN_ENV: f" {LEDGER_TOKEN}\n"}):
            fetch_ledger_states(REPO, [5])
        self.assertEqual(
            fetch_url.call_args.kwargs["headers"], {"x-hud-internal-bot": LEDGER_TOKEN}
        )

    @mock.patch("greenlight_ledger.gh_fetch_url")
    def test_a_sendable_but_rejected_credential_stays_transient(
        self, fetch_url: mock.MagicMock
    ) -> None:
        """Nothing local separates a stale token from a live one; only the route can."""
        fetch_url.side_effect = self._http_error(401, "unauthorized")
        with mock.patch.dict("os.environ", {LEDGER_TOKEN_ENV: REJECTED_TOKEN}):
            with self.assertRaises(GreenlightLedgerError) as cm:
                fetch_ledger_states(REPO, [5], sleep=lambda _s: None)
        self.assertFalse(cm.exception.fatal)
        self.assertEqual(fetch_url.call_count, 3)

    @mock.patch("greenlight_ledger.gh_fetch_url")
    def test_the_raised_message_leads_with_the_cause(
        self, fetch_url: mock.MagicMock
    ) -> None:
        fetch_url.side_effect = self._http_error(401, "unauthorized")
        with self.assertRaises(GreenlightLedgerError) as cm:
            fetch_ledger_states(REPO, [5], sleep=lambda _s: None)
        self.assertTrue(str(cm.exception).startswith("HTTPError: HTTP Error 401"))

    @mock.patch("greenlight_ledger.gh_fetch_url")
    def test_batches_are_capped_at_the_route_limit(
        self, fetch_url: mock.MagicMock
    ) -> None:
        fetch_url.return_value = {"states": []}
        limit = greenlight_ledger._MAX_PRS_PER_REQUEST
        fetch_ledger_states(REPO, list(range(1, limit + 2)))
        self.assertEqual(fetch_url.call_count, 2)

    @mock.patch("greenlight_ledger.gh_fetch_url")
    def test_retries_then_raises_on_persistent_http_errors(
        self, fetch_url: mock.MagicMock
    ) -> None:
        fetch_url.side_effect = self._http_error()
        sleeps: list[float] = []
        with self.assertRaises(GreenlightLedgerError):
            fetch_ledger_states(REPO, [5], sleep=sleeps.append)
        self.assertEqual(fetch_url.call_count, 3)
        self.assertEqual(sleeps, [2.0, 4.0])

    @mock.patch("greenlight_ledger.time.sleep")
    @mock.patch("greenlight_ledger.gh_fetch_url")
    def test_the_backoff_sleep_is_patchable(
        self, fetch_url: mock.MagicMock, sleep: mock.MagicMock
    ) -> None:
        fetch_url.side_effect = self._http_error()
        with self.assertRaises(GreenlightLedgerError):
            fetch_ledger_states(REPO, [5])
        self.assertEqual(sleep.call_args_list, [mock.call(2.0), mock.call(4.0)])

    @mock.patch("greenlight_ledger.gh_fetch_url")
    def test_recovers_when_a_retry_succeeds(self, fetch_url: mock.MagicMock) -> None:
        fetch_url.side_effect = [self._http_error(), {"states": []}]
        self.assertEqual(fetch_ledger_states(REPO, [5], sleep=lambda _s: None), {})
        self.assertEqual(fetch_url.call_count, 2)

    @mock.patch("greenlight_ledger.gh_fetch_url")
    def test_timeouts_and_malformed_payloads_raise(
        self, fetch_url: mock.MagicMock
    ) -> None:
        failures = [
            URLError("timed out"),
            json.JSONDecodeError("bad", "", 0),
            KeyError("states"),
        ]
        payloads: list[Any] = [
            {"unexpected": []},
            {"states": [{"pr_number": 5}]},
            {
                "states": [
                    {
                        "pr_number": 5,
                        "status": STATUS_LAND,
                        "head_sha": MERGE_SHA,
                        "run_id": 1,
                        "version": "nope",
                    }
                ]
            },
        ]
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                fetch_url.side_effect = failure
                with self.assertRaises(GreenlightLedgerError):
                    fetch_ledger_states(REPO, [5], sleep=lambda _s: None)
        for payload in payloads:
            with self.subTest(payload=payload):
                fetch_url.side_effect = None
                fetch_url.return_value = payload
                with self.assertRaises(GreenlightLedgerError):
                    fetch_ledger_states(REPO, [5], sleep=lambda _s: None)

    def test_version_timestamps_are_normalized_to_aware_utc(self) -> None:
        cases = {
            "2026-08-25T11:59:00.048Z": datetime(
                2026, 8, 25, 11, 59, 0, 48000, tzinfo=timezone.utc
            ),
            "2026-08-25T13:59:00.000+02:00": datetime(
                2026, 8, 25, 11, 59, tzinfo=timezone.utc
            ),
            "2026-08-25 11:59:00.000": datetime(
                2026, 8, 25, 11, 59, tzinfo=timezone.utc
            ),
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                parsed = greenlight_ledger.parse_utc_timestamp(raw)
                # Everything downstream subtracts these from an aware `now`; a naive
                # result would raise TypeError inside the age-out check instead.
                self.assertIsNotNone(parsed.tzinfo)
                self.assertEqual(parsed, expected)


class TestDuplicateLedgerRows(LedgerReadTestCase):
    """Arrival order must not decide the verdict, within a payload or across batches."""

    @mock.patch("greenlight_ledger.gh_fetch_url")
    def test_the_newest_row_wins_regardless_of_position(
        self, fetch_url: mock.MagicMock
    ) -> None:
        stale = make_row(STATUS_LAND, "2026-08-25T10:00:00.000Z", run_id=8)
        current = make_row(STATUS_NO_LAND, "2026-08-25T11:59:00.000Z", run_id=9)
        for rows in ([current, stale], [stale, current]):
            with self.subTest(last=rows[-1]["status"]):
                fetch_url.return_value = {"states": rows}
                states = fetch_ledger_states(REPO, [1])
                self.assertEqual(states[1].status, STATUS_NO_LAND)
                self.assertEqual(states[1].run_id, 9)

    @mock.patch("greenlight_ledger.gh_fetch_url")
    def test_a_stale_land_arriving_last_cannot_wave_a_merge_through(
        self, fetch_url: mock.MagicMock
    ) -> None:
        fetch_url.return_value = {
            "states": [
                make_row(STATUS_NO_LAND, "2026-08-25T11:59:00.000Z"),
                make_row(STATUS_LAND, "2026-08-25T10:00:00.000Z"),
            ]
        }
        result = evaluate()
        self.assertEqual(result.verdict, GuardVerdict.DENY)
        self.assertIn("NEGATIVE_VERDICT", result.message)

    @mock.patch("greenlight_ledger.gh_fetch_url")
    def test_the_later_run_wins_when_two_rows_share_a_version(
        self, fetch_url: mock.MagicMock
    ) -> None:
        same = "2026-08-25T11:59:00.000Z"
        older = make_row(STATUS_LAND, same, run_id=100)
        newer = make_row(STATUS_NO_LAND, same, run_id=101)
        for rows in ([older, newer], [newer, older]):
            with self.subTest(last=rows[-1]["run_id"]):
                fetch_url.return_value = {"states": rows}
                self.assertEqual(fetch_ledger_states(REPO, [1])[1].run_id, 101)
                self.assertEqual(evaluate().verdict, GuardVerdict.DENY)

    @mock.patch("greenlight_ledger.gh_fetch_url")
    def test_a_tie_on_every_orderable_field_still_refuses(
        self, fetch_url: mock.MagicMock
    ) -> None:
        """Two rows can only tie here by disagreeing, and a gate obeys the worse one."""
        rows = [
            make_row(STATUS_LAND, "2026-08-25T11:59:00.000Z"),
            make_row(STATUS_NO_LAND, "2026-08-25T11:59:00.000Z"),
        ]
        for ordered in (rows, list(reversed(rows))):
            with self.subTest(last=ordered[-1]["status"]):
                fetch_url.return_value = {"states": ordered}
                picked = fetch_ledger_states(REPO, [1])[1]
                self.assertEqual(picked.status, STATUS_NO_LAND)

    @mock.patch("greenlight_ledger.gh_fetch_url")
    def test_the_same_instant_in_two_formats_is_still_a_tie(
        self, fetch_url: mock.MagicMock
    ) -> None:
        """Normalizing to UTC is what makes a version collision reachable at all."""
        rows = [
            make_row(STATUS_LAND, "2026-08-25T11:59:00.000Z"),
            make_row(STATUS_NO_LAND, "2026-08-25T12:59:00.000+01:00"),
        ]
        for ordered in (rows, list(reversed(rows))):
            with self.subTest(last=ordered[-1]["status"]):
                fetch_url.return_value = {"states": ordered}
                picked = fetch_ledger_states(REPO, [1])[1]
                self.assertEqual(picked.status, STATUS_NO_LAND)

    @mock.patch("greenlight_ledger.gh_fetch_url")
    def test_a_stale_row_in_a_later_batch_cannot_overwrite_a_newer_one(
        self, fetch_url: mock.MagicMock
    ) -> None:
        """The same selection has to hold across batches, or >50 PRs reopens the hole."""
        limit = greenlight_ledger._MAX_PRS_PER_REQUEST
        current = make_row(STATUS_NO_LAND, "2026-08-25T11:59:00.000Z", run_id=9)
        stale = make_row(STATUS_LAND, "2026-08-25T10:00:00.000Z", run_id=8)
        for batches in ([[current], [stale]], [[stale], [current]]):
            with self.subTest(last=batches[-1][0]["status"]):
                fetch_url.side_effect = [{"states": rows} for rows in batches]
                states = fetch_ledger_states(REPO, list(range(1, limit + 2)))
                self.assertEqual(fetch_url.call_count, 2)
                self.assertEqual(states[1].status, STATUS_NO_LAND)
                self.assertEqual(states[1].run_id, 9)
                fetch_url.reset_mock()


@mock.patch("greenlight_ledger.time.sleep")
@mock.patch("greenlight_ledger.gh_fetch_url")
class TestUnreadableLedger(LedgerReadTestCase):
    """A HUD blip must not kill a merge, but a sustained outage still refuses it."""

    def test_transport_failure_waits(self, fetch_url: Any, _sleep: Any) -> None:
        fetch_url.side_effect = self._http_error()
        result = evaluate()
        self.assertEqual(result.verdict, GuardVerdict.WAIT)
        self.assertIn("TRANSPORT", result.message)
        self.assertIn("#1", result.message)

    def test_a_sustained_outage_denies_once_the_budget_is_spent(
        self, fetch_url: Any, _sleep: Any
    ) -> None:
        fetch_url.side_effect = self._http_error()
        window = GreenlightWaitWindow()
        self.assertEqual(
            evaluate(wait_window=window).verdict,
            GuardVerdict.WAIT,
        )
        self.assertEqual(window.transport_opened_at, NOW)
        self.assertIsNone(window.opened_at)
        self.assertEqual(
            evaluate(
                wait_window=window, now=NOW + TRANSPORT_MAX_WAIT - timedelta(seconds=1)
            ).verdict,
            GuardVerdict.WAIT,
        )
        expired = evaluate(wait_window=window, now=NOW + TRANSPORT_MAX_WAIT)
        self.assertEqual(expired.verdict, GuardVerdict.DENY)
        self.assertIn("TRANSPORT", expired.message)
        self.assertIn(
            f"still unreadable when this merge's {TRANSPORT_MAX_WAIT_MINUTES}-minute "
            "retry budget ran out",
            expired.message,
        )
        self.assertNotIn(f"{MAX_WAIT_MINUTES}-minute", expired.message)
        self.assertIn(greenlight_ledger.LEDGER_URL, expired.message)

    def test_the_timeout_claims_a_budget_rather_than_a_measured_outage(
        self, fetch_url: Any, _sleep: Any
    ) -> None:
        """Only the anchor's age is known, and polling never proves the gaps were down."""
        fetch_url.side_effect = self._http_error()
        window = GreenlightWaitWindow()
        evaluate(wait_window=window)
        expired = evaluate(wait_window=window, now=NOW + TRANSPORT_MAX_WAIT * 3)
        self.assertEqual(expired.verdict, GuardVerdict.DENY)
        self.assertNotIn("stayed unreadable for", expired.message)
        self.assertIn("retry budget ran out", expired.message)

    def test_the_transport_timeout_names_the_credential_without_blaming_it(
        self, fetch_url: Any, _sleep: Any
    ) -> None:
        """No status code here attributes the failure, so the wording must not either."""
        suffix = transport_timeout_suffix(TRANSPORT_MAX_WAIT_MINUTES)
        self.assertIn(LEDGER_TOKEN_ENV, suffix)
        self.assertIn("route being down", suffix)
        for code in (401, 500, 503):
            with self.subTest(code=code):
                fetch_url.side_effect = self._http_error(code, "unreachable")
                window = GreenlightWaitWindow()
                evaluate(wait_window=window)
                expired = evaluate(wait_window=window, now=NOW + TRANSPORT_MAX_WAIT)
                self.assertEqual(expired.verdict, GuardVerdict.DENY)
                self.assertIn(suffix, expired.message)
                self.assertIn(FIX_TRANSPORT, expired.message)

    def test_a_wait_that_starts_on_transport_still_announces_the_verdict_budget(
        self, fetch_url: Any, _sleep: Any
    ) -> None:
        window = GreenlightWaitWindow()
        fetch_url.side_effect = self._http_error()
        opening = evaluate(wait_window=window)
        self.assertIn(f"up to {TRANSPORT_MAX_WAIT_MINUTES} minutes", opening.comment)
        self.assertNotIn(f"up to {MAX_WAIT_MINUTES} minutes", opening.comment)

        fetch_url.side_effect = None
        fetch_url.return_value = {"states": []}
        crossed = evaluate(wait_window=window, now=NOW + timedelta(minutes=1))
        self.assertEqual(crossed.verdict, GuardVerdict.WAIT)
        self.assertIn(f"up to {MAX_WAIT_MINUTES} minutes", crossed.comment)

        fetch_url.side_effect = self._http_error()
        back = evaluate(wait_window=window, now=NOW + timedelta(minutes=2))
        self.assertEqual(back.verdict, GuardVerdict.WAIT)
        self.assertEqual(back.comment, "")

    def test_a_flapping_ledger_still_posts_at_most_one_comment_per_budget(
        self, fetch_url: Any, _sleep: Any
    ) -> None:
        """A running commentary of bot comments is worse than one stale heads-up."""
        window = GreenlightWaitWindow()
        comments = []
        for minute in range(40):
            if minute % 2:
                fetch_url.side_effect = None
                fetch_url.return_value = {"states": []}
            else:
                fetch_url.side_effect = self._http_error()
            result = evaluate(wait_window=window, now=NOW + timedelta(minutes=minute))
            self.assertEqual(result.verdict, GuardVerdict.WAIT)
            if result.comment:
                comments.append(result.comment)
        self.assertEqual(len(comments), 2)
        self.assertIn(f"up to {TRANSPORT_MAX_WAIT_MINUTES} minutes", comments[0])
        self.assertIn(f"up to {MAX_WAIT_MINUTES} minutes", comments[1])

    def test_the_two_announcements_advise_different_things(
        self, fetch_url: Any, _sleep: Any
    ) -> None:
        fetch_url.side_effect = self._http_error()
        transport = evaluate(wait_window=GreenlightWaitWindow()).comment
        self.assertIn("cannot read greenlight's verdict ledger", transport)
        self.assertIn("human reviewer with merge rights", transport)
        self.assertNotIn(STALE_LABEL, transport)
        self.assertNotIn("picks the PR up", transport)
        self.assertNotIn("to review the commit it will land", transport)

        fetch_url.side_effect = None
        fetch_url.return_value = {"states": []}
        verdict = evaluate(wait_window=GreenlightWaitWindow()).comment
        self.assertIn(STALE_LABEL, verdict)
        self.assertIn("picks the PR up", verdict)
        self.assertIn("to review the commit it will land", verdict)

    def test_a_wait_that_starts_on_a_verdict_still_announces_the_transport_budget(
        self, fetch_url: Any, _sleep: Any
    ) -> None:
        window = GreenlightWaitWindow()
        fetch_url.return_value = {"states": []}
        opening = evaluate(wait_window=window)
        self.assertIn(f"up to {MAX_WAIT_MINUTES} minutes", opening.comment)

        fetch_url.side_effect = self._http_error()
        crossed = evaluate(wait_window=window, now=NOW + timedelta(minutes=1))
        self.assertEqual(crossed.verdict, GuardVerdict.WAIT)
        self.assertIn(f"up to {TRANSPORT_MAX_WAIT_MINUTES} minutes", crossed.comment)
        self.assertNotIn(f"up to {MAX_WAIT_MINUTES} minutes", crossed.comment)

        fetch_url.side_effect = None
        back = evaluate(wait_window=window, now=NOW + timedelta(minutes=2))
        self.assertEqual(back.verdict, GuardVerdict.WAIT)
        self.assertEqual(back.comment, "")

    def test_a_successful_read_resets_the_transport_anchor(
        self, fetch_url: Any, _sleep: Any
    ) -> None:
        window = GreenlightWaitWindow()
        fetch_url.side_effect = self._http_error()
        self.assertEqual(evaluate(wait_window=window).verdict, GuardVerdict.WAIT)
        self.assertEqual(window.transport_opened_at, NOW)

        recovered = NOW + TRANSPORT_MAX_WAIT - timedelta(minutes=1)
        fetch_url.side_effect = None
        fetch_url.return_value = {"states": []}
        self.assertEqual(
            evaluate(wait_window=window, now=recovered).verdict, GuardVerdict.WAIT
        )
        self.assertIsNone(window.transport_opened_at)

        relapsed = recovered + timedelta(minutes=1)
        fetch_url.side_effect = self._http_error()
        self.assertEqual(
            evaluate(wait_window=window, now=relapsed).verdict, GuardVerdict.WAIT
        )
        self.assertEqual(window.transport_opened_at, relapsed)

    def test_a_merge_deep_into_a_verdict_wait_still_gets_a_full_transport_budget(
        self, fetch_url: Any, _sleep: Any
    ) -> None:
        window = GreenlightWaitWindow(opened_at=NOW)
        fetch_url.side_effect = self._http_error()
        deep = NOW + MAX_WAIT - timedelta(minutes=1)
        self.assertEqual(
            evaluate(wait_window=window, now=deep).verdict, GuardVerdict.WAIT
        )
        almost = evaluate(
            wait_window=window, now=deep + TRANSPORT_MAX_WAIT - timedelta(seconds=1)
        )
        self.assertEqual(almost.verdict, GuardVerdict.WAIT)
        expired = evaluate(wait_window=window, now=deep + TRANSPORT_MAX_WAIT)
        self.assertEqual(expired.verdict, GuardVerdict.DENY)

    def test_a_force_merge_refuses_immediately(
        self, fetch_url: Any, _sleep: Any
    ) -> None:
        fetch_url.side_effect = self._http_error()
        result = evaluate_greenlight_guard(REPO, [make_pr()], wait_window=None, now=NOW)
        self.assertEqual(result.verdict, GuardVerdict.DENY)
        self.assertIn("TRANSPORT", result.message)


@mock.patch("greenlight_ledger.time.sleep")
@mock.patch("greenlight_ledger.gh_fetch_url")
class TestRefusalNamesTheRightBlocker(LedgerReadTestCase):
    """The refusal ends the merge, so it must not send the reader after the wrong thing."""

    def _timed_out_refusal(self, fetch_url: Any) -> str:
        fetch_url.side_effect = self._http_error()
        window = GreenlightWaitWindow()
        evaluate(wait_window=window)
        return str(evaluate(wait_window=window, now=NOW + TRANSPORT_MAX_WAIT).message)

    def _credential_refusal(self) -> str:
        with mock.patch.dict("os.environ", {LEDGER_TOKEN_ENV: ""}):
            return str(evaluate().message)

    def _no_row_refusal(self, fetch_url: Any) -> str:
        fetch_url.side_effect = None
        fetch_url.return_value = {"states": []}
        window = GreenlightWaitWindow()
        evaluate(wait_window=window)
        return str(evaluate(wait_window=window, now=NOW + MAX_WAIT).message)

    def test_an_unreadable_ledger_is_not_reported_as_a_missing_verdict(
        self, fetch_url: Any, _sleep: Any
    ) -> None:
        unreadable = (self._timed_out_refusal(fetch_url), self._credential_refusal())
        for message in unreadable:
            with self.subTest(headline=message.split("\n", 1)[0]):
                self.assertNotIn("until greenlight records a verdict", message)
                self.assertIn(
                    "until greenlight's verdict for the exact commit being landed can "
                    "be read",
                    message,
                )

    def test_a_pr_greenlight_never_reviewed_still_is(
        self, fetch_url: Any, _sleep: Any
    ) -> None:
        message = self._no_row_refusal(fetch_url)
        self.assertIn("until greenlight records a verdict", message)
        self.assertNotIn("can be read", message)


@mock.patch("greenlight_ledger.time.sleep")
@mock.patch("greenlight_ledger.gh_fetch_url")
class TestMissingLedgerCredential(TestCase):
    """A secret only Dev Infra can restore, so no wait budget is spent on it.

    Not a `LedgerReadTestCase`: every test here supplies its own credential state, and
    inheriting a valid one would only hide a test that forgot to.
    """

    def test_it_denies_and_lists_every_guarded_pr(
        self, fetch_url: Any, _sleep: Any
    ) -> None:
        prs = [
            make_pr(pr_num=1, head_sha="1" * 40),
            make_pr(pr_num=2, head_sha="2" * 40),
        ]
        window = GreenlightWaitWindow()
        with mock.patch.dict("os.environ", {LEDGER_TOKEN_ENV: ""}):
            result = evaluate(prs=prs, wait_window=window)
        self.assertEqual(result.verdict, GuardVerdict.DENY)
        self.assertIn("PR #1 (LEDGER_CREDENTIAL)", result.message)
        self.assertIn("PR #2 (LEDGER_CREDENTIAL)", result.message)
        self.assertNotIn("TRANSPORT", result.message)
        self.assertIn(FIX_CREDENTIAL, result.message)
        self.assertNotIn(FIX_TRANSPORT, result.message)
        self.assertEqual(result.comment, "")
        self.assertIsNone(window.opened_at)
        self.assertIsNone(window.transport_opened_at)
        fetch_url.assert_not_called()

    def test_its_headline_is_the_bare_cause(self, fetch_url: Any, _sleep: Any) -> None:
        with mock.patch.dict("os.environ", {LEDGER_TOKEN_ENV: ""}):
            result = evaluate()
        self.assertEqual(
            result.message.split("\n", 1)[0],
            f"Greenlight's ledger could not be read: {LEDGER_TOKEN_ENV} is not set.",
        )

    def test_the_kind_names_the_secret_rather_than_the_pr(
        self, fetch_url: Any, _sleep: Any
    ) -> None:
        """The kind heads the bullet, so it must not read as a charge against the PR."""
        with mock.patch.dict("os.environ", {LEDGER_TOKEN_ENV: ""}):
            result = evaluate()
        self.assertIn("- PR #1 (LEDGER_CREDENTIAL):", result.message)
        self.assertNotIn("MISCONFIGURED", result.message)


class TestParsedTimestampsReachTheAgeOutCheck(LedgerReadTestCase):
    """The route's timestamps must survive parsing into the dead-review arithmetic."""

    def _payload(self, version: str) -> dict[str, Any]:
        return {
            "states": [
                {
                    "pr_number": 1,
                    "status": STATUS_AI_REVIEW_STARTED,
                    "head_sha": MERGE_SHA,
                    "run_id": LARGE_RUN_ID,
                    "version": version,
                }
            ]
        }

    @mock.patch("greenlight_ledger.gh_fetch_url")
    def _verdict_for(self, version: str, fetch_url: mock.MagicMock) -> Any:
        fetch_url.return_value = self._payload(version)
        return evaluate()

    def test_a_fresh_route_timestamp_waits(self) -> None:
        result = self._verdict_for("2026-08-25T11:59:00.048Z")
        self.assertEqual(result.verdict, GuardVerdict.WAIT)

    def test_an_aged_out_route_timestamp_denies(self) -> None:
        result = self._verdict_for("2026-08-25T11:00:00.048Z")
        self.assertEqual(result.verdict, GuardVerdict.DENY)
        self.assertIn("DEAD_REVIEW", result.message)

    def test_a_large_run_id_survives_parsing_and_reaches_the_message(self) -> None:
        result = self._verdict_for("2026-08-25T11:00:00.048Z")
        self.assertIn(f"run {LARGE_RUN_ID}", result.message)


class TestLedgerReaderIsPatchable(TestCase):
    @mock.patch("greenlight_guard.fetch_ledger_states")
    def test_patching_the_reader_keeps_the_guard_off_the_network(
        self, fetch_states: mock.MagicMock
    ) -> None:
        fetch_states.return_value = {}
        with mock.patch("greenlight_ledger.gh_fetch_url") as fetch_url:
            result = evaluate()
        fetch_states.assert_called_once_with(REPO, [1])
        fetch_url.assert_not_called()
        self.assertEqual(result.verdict, GuardVerdict.WAIT)


class TestWaitBudgets(TestCase):
    def test_the_transport_budget_is_the_shorter_of_the_two(self) -> None:
        self.assertLess(TRANSPORT_MAX_WAIT, MAX_WAIT)


class TestUnrecognizedVerdict(TestCase):
    """A verdict added after this aggregation was written must never release a merge."""

    def _synthetic_outcome(self, verdict: Any, pr_num: int) -> Outcome:
        return Outcome(
            verdict,
            "FUTURE",
            "a verdict this aggregation predates",
            pr_num,
            str(pr_num) * 40,
            "Greenlight ledger: no row for this PR",
        )

    def _evaluate_with(self, *verdicts: Any) -> Any:
        numbers = range(1, 1 + len(verdicts))
        prs = [make_pr(pr_num=n, head_sha=str(n) * 40) for n in numbers]
        outcomes = [self._synthetic_outcome(v, n) for v, n in zip(verdicts, numbers)]
        with mock.patch("greenlight_guard.fetch_ledger_states", return_value={}):
            with mock.patch("greenlight_guard._evaluate_pr", side_effect=outcomes):
                return evaluate(prs=prs)

    def test_it_holds_rather_than_allows(self) -> None:
        result = self._evaluate_with(_FutureVerdict.HOLD)
        self.assertEqual(result.verdict, GuardVerdict.WAIT)
        self.assertIn("FUTURE", result.message)

    def test_an_allow_alongside_it_does_not_carry_the_stack(self) -> None:
        result = self._evaluate_with(GuardVerdict.ALLOW, _FutureVerdict.HOLD)
        self.assertEqual(result.verdict, GuardVerdict.WAIT)

    def test_a_deny_alongside_it_still_leads_the_refusal(self) -> None:
        result = self._evaluate_with(_FutureVerdict.HOLD, GuardVerdict.DENY)
        self.assertEqual(result.verdict, GuardVerdict.DENY)
        self.assertLess(result.message.index("PR #2"), result.message.index("PR #1"))


class TestStatusGroupings(TestCase):
    def test_terminal_statuses_match_greenlights_own_split(self) -> None:
        self.assertEqual(
            greenlight_ledger.TERMINAL_STATUSES,
            frozenset(
                {greenlight_ledger.STATUS_LAND, greenlight_ledger.STATUS_NO_LAND}
            ),
        )
        self.assertTrue(
            greenlight_ledger.TERMINAL_STATUSES.isdisjoint(
                greenlight_ledger.IN_FLIGHT_STATUSES
            )
        )


if __name__ == "__main__":
    main()
