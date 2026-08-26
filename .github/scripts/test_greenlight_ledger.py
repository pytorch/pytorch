#!/usr/bin/env python3

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest import main, mock, TestCase
from urllib.error import HTTPError, URLError

import greenlight_ledger
from greenlight_guard import (
    evaluate_greenlight_guard,
    GreenlightWaitWindow,
    GuardVerdict,
    MAX_WAIT,
)
from greenlight_ledger import (
    fetch_ledger_states,
    GreenlightLedgerError,
    STATUS_AI_REVIEW_STARTED,
    STATUS_LAND,
)
from test_greenlight_guard import make_pr, MERGE_SHA, NOW, REPO


# A run_id as production emits it: an unquoted JSON number far past 32 bits.
LARGE_RUN_ID = 32747018107


def evaluate(
    *, wait_window: GreenlightWaitWindow | None = None, now: datetime = NOW
) -> Any:
    return evaluate_greenlight_guard(
        REPO,
        [make_pr()],
        wait_window=wait_window if wait_window is not None else GreenlightWaitWindow(),
        now=now,
    )


class TestLedgerTransport(TestCase):
    def _http_error(self, code: int = 500, reason: str = "boom") -> HTTPError:
        return HTTPError(greenlight_ledger.LEDGER_URL, code, reason, {}, None)  # type: ignore[arg-type]

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
        with mock.patch.dict("os.environ", {"HUD_API_TOKEN": "token"}):
            states = fetch_ledger_states(REPO, [5, 6])

        fetch_url.assert_called_once()
        url = fetch_url.call_args.args[0]
        self.assertIn("repo=pytorch%2Fpytorch", url)
        self.assertIn("prNumbers=5%2C6", url)
        self.assertEqual(
            fetch_url.call_args.kwargs["headers"], {"x-hud-internal-bot": "token"}
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


@mock.patch("greenlight_ledger.time.sleep")
@mock.patch("greenlight_ledger.gh_fetch_url")
class TestUnreadableLedger(TestCase):
    """A HUD blip must not kill a merge, but a sustained outage still refuses it."""

    def _http_error(self) -> HTTPError:
        return HTTPError(greenlight_ledger.LEDGER_URL, 500, "boom", {}, None)  # type: ignore[arg-type]

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
        expired = evaluate(wait_window=window, now=NOW + MAX_WAIT)
        self.assertEqual(expired.verdict, GuardVerdict.DENY)
        self.assertIn("TRANSPORT", expired.message)
        self.assertIn("did not answer after 60 minutes of waiting", expired.message)
        self.assertIn(greenlight_ledger.LEDGER_URL, expired.message)

    def test_a_force_merge_refuses_immediately(
        self, fetch_url: Any, _sleep: Any
    ) -> None:
        fetch_url.side_effect = self._http_error()
        result = evaluate_greenlight_guard(REPO, [make_pr()], wait_window=None, now=NOW)
        self.assertEqual(result.verdict, GuardVerdict.DENY)
        self.assertIn("TRANSPORT", result.message)


class TestParsedTimestampsReachTheAgeOutCheck(TestCase):
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
