#!/usr/bin/env python3

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest import main, mock, TestCase

import greenlight_guard
from greenlight_guard import (
    DEAD_REVIEW_AGE,
    evaluate_greenlight_guard,
    GREENLIGHT_LOGIN,
    GreenlightWaitWindow,
    GuardVerdict,
    MAX_WAIT,
    PRUnderMerge,
)
from greenlight_ledger import (
    LedgerState,
    STATUS_AI_REVIEW_DISPATCHED,
    STATUS_AI_REVIEW_STARTED,
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_LAND,
    STATUS_NO_LAND,
    STATUS_REVERTED,
)
from greenlight_messages import GREENLIGHT_JOB_URL
from greenlight_preflight import REVIEW_WINDOW, STALE_LABEL


NOW = datetime(2026, 8, 25, 12, 0, 0, tzinfo=timezone.utc)
MERGE_SHA = "a" * 40
OTHER_SHA = "b" * 40
REPO = "pytorch/pytorch"


def make_pr(
    pr_num: int = 1,
    head_sha: str = MERGE_SHA,
    approved_by: tuple[str, ...] = (GREENLIGHT_LOGIN,),
    changes_requested_by: tuple[str, ...] = (),
    labels: tuple[str, ...] = (),
    updated_at: str | None = None,
    bot_reviewers: tuple[str, ...] | None = (),
    merge_authorized: tuple[str, ...] = (),
    authorized_without_greenlight: bool = False,
) -> PRUnderMerge:
    """A PR under merge. ``bot_reviewers=None`` is a failed reviewer-type lookup."""
    return PRUnderMerge(
        pr_num=pr_num,
        head_sha=head_sha,
        approved_by=list(approved_by),
        changes_requested_by=list(changes_requested_by),
        labels=list(labels),
        get_updated_at=lambda: updated_at,
        get_bot_reviewers=lambda: (
            None if bot_reviewers is None else frozenset(bot_reviewers)
        ),
        get_merge_authorized_logins=lambda: frozenset(merge_authorized),
        is_authorized_without_greenlight=lambda: authorized_without_greenlight,
    )


def make_state(
    status: str = STATUS_LAND,
    head_sha: str = MERGE_SHA,
    pr_number: int = 1,
    run_id: int = 7,
    age: timedelta = timedelta(minutes=1),
) -> LedgerState:
    return LedgerState(
        pr_number=pr_number,
        status=status,
        head_sha=head_sha,
        run_id=run_id,
        version=NOW - age,
    )


def evaluate(
    prs: list[PRUnderMerge],
    states: dict[int, LedgerState],
    *,
    wait_window: GreenlightWaitWindow | None = None,
    now: datetime = NOW,
) -> greenlight_guard.GuardResult:
    """Evaluate with a wait budget, a fresh one unless the caller brings their own."""
    return evaluate_greenlight_guard(
        REPO,
        prs,
        wait_window=wait_window if wait_window is not None else GreenlightWaitWindow(),
        now=now,
        fetch_states=lambda _repo, _nums: states,
    )


def evaluate_force(
    prs: list[PRUnderMerge],
    states: dict[int, LedgerState],
    *,
    now: datetime = NOW,
) -> greenlight_guard.GuardResult:
    """Evaluate as a force merge: no retry loop to wait in."""
    return evaluate_greenlight_guard(
        REPO,
        prs,
        wait_window=None,
        now=now,
        fetch_states=lambda _repo, _nums: states,
    )


class TestPredicate(TestCase):
    def test_skips_pr_greenlight_did_not_approve(self) -> None:
        fetch = mock.MagicMock()
        result = evaluate_greenlight_guard(
            REPO,
            [make_pr(approved_by=("malfet",))],
            wait_window=GreenlightWaitWindow(),
            now=NOW,
            fetch_states=fetch,
        )
        self.assertEqual(result.verdict, GuardVerdict.ALLOW)
        fetch.assert_not_called()

    def test_skips_when_another_approver_carries_a_merge_rule(self) -> None:
        fetch = mock.MagicMock()
        result = evaluate_greenlight_guard(
            REPO,
            [
                make_pr(
                    approved_by=(GREENLIGHT_LOGIN, "malfet"),
                    authorized_without_greenlight=True,
                )
            ],
            wait_window=GreenlightWaitWindow(),
            now=NOW,
            fetch_states=fetch,
        )
        self.assertEqual(result.verdict, GuardVerdict.ALLOW)
        fetch.assert_not_called()

    def test_drive_by_approver_does_not_disable_the_guard(self) -> None:
        pr = make_pr(
            approved_by=(GREENLIGHT_LOGIN, "a-random-stranger"),
            authorized_without_greenlight=False,
        )
        result = evaluate([pr], {1: make_state(status=STATUS_NO_LAND)})
        self.assertEqual(result.verdict, GuardVerdict.DENY)

    def test_greenlight_login_is_matched_case_insensitively(self) -> None:
        pr = make_pr(approved_by=("PyTorchGreenlight",))
        result = evaluate([pr], {1: make_state(status=STATUS_NO_LAND)})
        self.assertEqual(result.verdict, GuardVerdict.DENY)

    def test_a_bot_suffixed_greenlight_login_still_triggers_the_guard(self) -> None:
        pr = make_pr(approved_by=(f"{GREENLIGHT_LOGIN}[bot]",))
        result = evaluate([pr], {1: make_state(status=STATUS_NO_LAND)})
        self.assertEqual(result.verdict, GuardVerdict.DENY)


class TestLaziness(TestCase):
    """Every callable on PRUnderMerge costs a request, so none may fire needlessly."""

    def _instrumented(self, **kwargs: Any) -> tuple[PRUnderMerge, dict[str, Any]]:
        calls = {
            "get_updated_at": mock.MagicMock(return_value=None),
            "get_bot_reviewers": mock.MagicMock(return_value=frozenset()),
            "get_merge_authorized_logins": mock.MagicMock(return_value=frozenset()),
            "is_authorized_without_greenlight": mock.MagicMock(return_value=False),
        }
        base = make_pr(**kwargs)
        return (
            PRUnderMerge(
                pr_num=base.pr_num,
                head_sha=base.head_sha,
                approved_by=base.approved_by,
                changes_requested_by=base.changes_requested_by,
                labels=base.labels,
                get_updated_at=calls["get_updated_at"],
                get_bot_reviewers=calls["get_bot_reviewers"],
                get_merge_authorized_logins=calls["get_merge_authorized_logins"],
                is_authorized_without_greenlight=calls[
                    "is_authorized_without_greenlight"
                ],
            ),
            calls,
        )

    def test_nothing_is_queried_when_greenlight_did_not_approve(self) -> None:
        pr, calls = self._instrumented(approved_by=("malfet",))
        self.assertEqual(evaluate([pr], {}).verdict, GuardVerdict.ALLOW)
        for name, call in calls.items():
            with self.subTest(callable=name):
                call.assert_not_called()

    def test_preflight_facts_are_untouched_when_the_verdict_already_matches(
        self,
    ) -> None:
        pr, calls = self._instrumented(labels=(STALE_LABEL,))
        self.assertEqual(
            evaluate([pr], {1: make_state(status=STATUS_LAND)}).verdict,
            GuardVerdict.ALLOW,
        )
        calls["is_authorized_without_greenlight"].assert_called_once()
        calls["get_updated_at"].assert_not_called()
        calls["get_bot_reviewers"].assert_not_called()
        calls["get_merge_authorized_logins"].assert_not_called()

    def test_reviewer_types_are_only_read_for_an_unrecognized_blocker(self) -> None:
        pr, calls = self._instrumented(changes_requested_by=("pytorch-bot",))
        self.assertEqual(evaluate([pr], {}).verdict, GuardVerdict.WAIT)
        calls["get_bot_reviewers"].assert_not_called()

    def test_merge_rules_are_only_read_when_a_person_approved(self) -> None:
        pr, calls = self._instrumented(approved_by=(GREENLIGHT_LOGIN, "pytorchbot"))
        self.assertEqual(evaluate([pr], {}).verdict, GuardVerdict.WAIT)
        calls["get_merge_authorized_logins"].assert_not_called()
        calls["get_bot_reviewers"].assert_not_called()

    def test_reviewer_types_are_not_read_for_an_unlisted_approver(self) -> None:
        pr, calls = self._instrumented(
            approved_by=(GREENLIGHT_LOGIN, "a-random-stranger")
        )
        self.assertEqual(evaluate([pr], {}).verdict, GuardVerdict.WAIT)
        calls["get_merge_authorized_logins"].assert_called_once()
        calls["get_bot_reviewers"].assert_not_called()


class TestDecisionTable(TestCase):
    def test_land_at_head_sha_allows(self) -> None:
        result = evaluate([make_pr()], {1: make_state(status=STATUS_LAND)})
        self.assertEqual(result.verdict, GuardVerdict.ALLOW)
        self.assertEqual(result.message, "")

    def test_negative_verdict_at_head_sha_denies(self) -> None:
        for status in (STATUS_NO_LAND, STATUS_CANCELLED, STATUS_FAILED):
            with self.subTest(status=status):
                result = evaluate([make_pr()], {1: make_state(status=status)})
                self.assertEqual(result.verdict, GuardVerdict.DENY)
                self.assertIn("NEGATIVE_VERDICT", result.message)
                self.assertIn(status, result.message)
                self.assertIn(MERGE_SHA, result.message)

    def test_reverted_at_head_sha_denies_as_a_negative_verdict(self) -> None:
        result = evaluate([make_pr()], {1: make_state(status=STATUS_REVERTED)})
        self.assertEqual(result.verdict, GuardVerdict.DENY)
        self.assertIn("NEGATIVE_VERDICT", result.message)

    def test_reverted_at_head_sha_does_not_advise_pushing_a_commit(self) -> None:
        result = evaluate([make_pr()], {1: make_state(status=STATUS_REVERTED)})
        self.assertIn("after it has been reverted", result.message)
        self.assertNotIn("push a commit", result.message)

    def test_in_flight_at_head_sha_waits(self) -> None:
        for status in (STATUS_AI_REVIEW_STARTED, STATUS_AI_REVIEW_DISPATCHED):
            with self.subTest(status=status):
                result = evaluate([make_pr()], {1: make_state(status=status)})
                self.assertEqual(result.verdict, GuardVerdict.WAIT)
                self.assertIn("IN_FLIGHT", result.message)

    def test_different_head_sha_waits(self) -> None:
        for status in (STATUS_LAND, STATUS_NO_LAND, STATUS_AI_REVIEW_STARTED):
            with self.subTest(status=status):
                result = evaluate(
                    [make_pr()], {1: make_state(status=status, head_sha=OTHER_SHA)}
                )
                self.assertEqual(result.verdict, GuardVerdict.WAIT)
                self.assertIn("SHA_MISMATCH", result.message)
                self.assertIn(OTHER_SHA, result.message)
                self.assertIn(MERGE_SHA, result.message)

    def test_the_sha_mismatch_kind_does_not_collide_with_the_stale_label(self) -> None:
        """`Stale` is a GitHub label the refusal also talks about; the kind is not it."""
        result = evaluate([make_pr()], {1: make_state(head_sha=OTHER_SHA)})
        self.assertNotIn("(STALE)", result.message)

    def test_missing_row_waits(self) -> None:
        result = evaluate([make_pr()], {})
        self.assertEqual(result.verdict, GuardVerdict.WAIT)
        self.assertIn("NO_ROW", result.message)
        self.assertIn("no row for this PR", result.message)

    def test_dead_in_flight_review_denies(self) -> None:
        result = evaluate(
            [make_pr()],
            {1: make_state(status=STATUS_AI_REVIEW_STARTED, age=DEAD_REVIEW_AGE)},
        )
        self.assertEqual(result.verdict, GuardVerdict.DENY)
        self.assertIn("DEAD_REVIEW", result.message)

    def test_in_flight_review_just_under_the_age_limit_waits(self) -> None:
        result = evaluate(
            [make_pr()],
            {
                1: make_state(
                    status=STATUS_AI_REVIEW_STARTED,
                    age=DEAD_REVIEW_AGE - timedelta(seconds=1),
                )
            },
        )
        self.assertEqual(result.verdict, GuardVerdict.WAIT)

    def test_unrecognized_status_denies(self) -> None:
        result = evaluate([make_pr()], {1: make_state(status="WAT")})
        self.assertEqual(result.verdict, GuardVerdict.DENY)
        self.assertIn("UNKNOWN_STATUS", result.message)

    def test_message_names_the_run_id_and_ledger_sha(self) -> None:
        result = evaluate(
            [make_pr()], {1: make_state(status=STATUS_NO_LAND, run_id=4242)}
        )
        self.assertIn("run 4242", result.message)
        self.assertIn(GREENLIGHT_JOB_URL, result.message)


class TestShaComparison(TestCase):
    def test_case_differences_do_not_break_the_match(self) -> None:
        pr = make_pr(head_sha=MERGE_SHA.upper())
        result = evaluate([pr], {1: make_state(head_sha=MERGE_SHA)})
        self.assertEqual(result.verdict, GuardVerdict.ALLOW)

    def test_surrounding_whitespace_does_not_break_the_match(self) -> None:
        pr = make_pr(head_sha=f" {MERGE_SHA}\n")
        result = evaluate([pr], {1: make_state(head_sha=MERGE_SHA)})
        self.assertEqual(result.verdict, GuardVerdict.ALLOW)

    def test_two_equal_non_shas_never_allow(self) -> None:
        for sha in ("", "unknown", MERGE_SHA[:39], f"{MERGE_SHA}0", "z" * 40):
            with self.subTest(sha=sha):
                result = evaluate(
                    [make_pr(head_sha=sha)], {1: make_state(head_sha=sha)}
                )
                self.assertNotEqual(result.verdict, GuardVerdict.ALLOW)


class TestStack(TestCase):
    def test_denial_anywhere_in_the_stack_wins_over_a_wait(self) -> None:
        prs = [
            make_pr(pr_num=1, head_sha=MERGE_SHA),
            make_pr(pr_num=2, head_sha=OTHER_SHA),
        ]
        states = {2: make_state(pr_number=2, status=STATUS_NO_LAND, head_sha=OTHER_SHA)}
        result = evaluate(prs, states)
        self.assertEqual(result.verdict, GuardVerdict.DENY)
        self.assertIn("PR #2", result.message)

    def test_a_refusal_also_reports_the_prs_still_unsettled(self) -> None:
        prs = [
            make_pr(pr_num=1, head_sha="1" * 40),
            make_pr(pr_num=2, head_sha="2" * 40),
        ]
        states = {2: make_state(pr_number=2, status=STATUS_NO_LAND, head_sha="2" * 40)}
        result = evaluate(prs, states)
        self.assertEqual(result.verdict, GuardVerdict.DENY)
        self.assertIn("PR #1 (NO_ROW)", result.message)
        self.assertIn("PR #2 (NEGATIVE_VERDICT)", result.message)

    def test_every_denial_in_the_stack_is_reported(self) -> None:
        prs = [
            make_pr(pr_num=1, head_sha="1" * 40),
            make_pr(pr_num=2, head_sha="2" * 40),
        ]
        states = {
            1: make_state(pr_number=1, status=STATUS_NO_LAND, head_sha="1" * 40),
            2: make_state(pr_number=2, status="WAT", head_sha="2" * 40),
        }
        result = evaluate(prs, states)
        self.assertEqual(result.verdict, GuardVerdict.DENY)
        self.assertIn("PR #1 (NEGATIVE_VERDICT)", result.message)
        self.assertIn("PR #2 (UNKNOWN_STATUS)", result.message)

    def test_every_wait_in_the_stack_is_reported(self) -> None:
        prs = [
            make_pr(pr_num=1, head_sha="1" * 40),
            make_pr(pr_num=2, head_sha="2" * 40),
        ]
        states = {
            2: make_state(
                pr_number=2, status=STATUS_AI_REVIEW_STARTED, head_sha="2" * 40
            )
        }
        result = evaluate(prs, states)
        self.assertEqual(result.verdict, GuardVerdict.WAIT)
        self.assertIn("PR #1 (NO_ROW)", result.message)
        self.assertIn("PR #2 (IN_FLIGHT)", result.message)

    def test_mixed_stack_only_queries_the_greenlight_only_prs(self) -> None:
        human_approved = make_pr(pr_num=1, approved_by=("malfet",), head_sha="1" * 40)
        greenlight_a = make_pr(pr_num=2, head_sha="2" * 40)
        greenlight_b = make_pr(pr_num=3, head_sha="3" * 40)
        seen: list[Any] = []

        def fetch(repo: str, numbers: Any) -> dict[int, LedgerState]:
            seen.append((repo, list(numbers)))
            return {
                2: make_state(pr_number=2, head_sha="2" * 40),
                3: make_state(pr_number=3, head_sha="3" * 40),
            }

        result = evaluate_greenlight_guard(
            REPO,
            [human_approved, greenlight_a, greenlight_b],
            wait_window=GreenlightWaitWindow(),
            now=NOW,
            fetch_states=fetch,
        )
        self.assertEqual(result.verdict, GuardVerdict.ALLOW)
        self.assertEqual(seen, [(REPO, [2, 3])])

    def test_stack_is_checked_against_each_prs_own_head_sha(self) -> None:
        prs = [
            make_pr(pr_num=1, head_sha="1" * 40),
            make_pr(pr_num=2, head_sha="2" * 40),
        ]
        states = {
            1: make_state(pr_number=1, head_sha="1" * 40),
            2: make_state(pr_number=2, head_sha="1" * 40),
        }
        result = evaluate(prs, states)
        self.assertEqual(result.verdict, GuardVerdict.WAIT)
        self.assertIn("PR #2", result.message)


class TestPreflight(TestCase):
    def test_changes_requested_by_a_human_denies_instead_of_waiting(self) -> None:
        result = evaluate([make_pr(changes_requested_by=("some-reviewer",))], {})
        self.assertEqual(result.verdict, GuardVerdict.DENY)
        self.assertIn("CANNOT_REVIEW", result.message)
        self.assertIn("some-reviewer", result.message)
        self.assertIn("Resolve or dismiss", result.message)

    def test_changes_requested_by_a_bot_still_waits(self) -> None:
        for login in ("pytorch-bot", "facebook-github-tools[bot]", "PyTorchBot"):
            with self.subTest(login=login):
                result = evaluate([make_pr(changes_requested_by=(login,))], {})
                self.assertEqual(result.verdict, GuardVerdict.WAIT)

    def test_changes_requested_by_an_unlisted_app_still_waits(self) -> None:
        pr = make_pr(
            changes_requested_by=("some-new-app",), bot_reviewers=("Some-New-App",)
        )
        self.assertEqual(evaluate([pr], {}).verdict, GuardVerdict.WAIT)

    def test_stale_label_denies_instead_of_waiting(self) -> None:
        result = evaluate([make_pr(labels=(STALE_LABEL,))], {})
        self.assertEqual(result.verdict, GuardVerdict.DENY)
        self.assertIn("CANNOT_REVIEW", result.message)
        self.assertIn(f"`{STALE_LABEL}` label", result.message)

    def test_pr_outside_the_review_window_denies_instead_of_waiting(self) -> None:
        stale = (NOW - REVIEW_WINDOW - timedelta(minutes=1)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        result = evaluate([make_pr(updated_at=stale)], {})
        self.assertEqual(result.verdict, GuardVerdict.DENY)
        self.assertIn("CANNOT_REVIEW", result.message)
        self.assertIn("review window", result.message)
        self.assertIn(stale, result.message)

    def test_pr_inside_the_review_window_waits(self) -> None:
        recent = (NOW - REVIEW_WINDOW + timedelta(minutes=1)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        result = evaluate([make_pr(updated_at=recent)], {})
        self.assertEqual(result.verdict, GuardVerdict.WAIT)

    def test_unreadable_updated_at_falls_back_to_waiting(self) -> None:
        self.assertEqual(
            evaluate([make_pr(updated_at=None)], {}).verdict, GuardVerdict.WAIT
        )
        self.assertEqual(
            evaluate([make_pr(updated_at="not a timestamp")], {}).verdict,
            GuardVerdict.WAIT,
        )

    def test_a_redispatchable_review_survives_the_stale_label(self) -> None:
        for status in (STATUS_CANCELLED, STATUS_FAILED, STATUS_AI_REVIEW_STARTED):
            with self.subTest(status=status):
                result = evaluate(
                    [make_pr(labels=(STALE_LABEL,))],
                    {1: make_state(status=status, head_sha=OTHER_SHA)},
                )
                self.assertEqual(result.verdict, GuardVerdict.WAIT)

    def test_a_terminal_verdict_does_not_survive_the_stale_label(self) -> None:
        for status in (STATUS_LAND, STATUS_NO_LAND):
            with self.subTest(status=status):
                result = evaluate(
                    [make_pr(labels=(STALE_LABEL,))],
                    {1: make_state(status=status, head_sha=OTHER_SHA)},
                )
                self.assertEqual(result.verdict, GuardVerdict.DENY)
                self.assertIn("CANNOT_REVIEW", result.message)

    def test_a_redispatchable_review_survives_the_review_window(self) -> None:
        stale = (NOW - REVIEW_WINDOW - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        result = evaluate(
            [make_pr(updated_at=stale)],
            {1: make_state(status=STATUS_CANCELLED, head_sha=OTHER_SHA)},
        )
        self.assertEqual(result.verdict, GuardVerdict.WAIT)

    def test_changes_requested_denies_even_for_a_redispatchable_review(self) -> None:
        result = evaluate(
            [make_pr(changes_requested_by=("some-reviewer",))],
            {1: make_state(status=STATUS_CANCELLED, head_sha=OTHER_SHA)},
        )
        self.assertEqual(result.verdict, GuardVerdict.DENY)
        self.assertIn("CANNOT_REVIEW", result.message)

    def test_preflight_does_not_override_a_settled_verdict_at_the_head_sha(
        self,
    ) -> None:
        stale = (NOW - REVIEW_WINDOW - timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
        pr = make_pr(
            changes_requested_by=("some-reviewer",),
            labels=(STALE_LABEL,),
            updated_at=stale,
        )
        result = evaluate([pr], {1: make_state(status=STATUS_LAND)})
        self.assertEqual(result.verdict, GuardVerdict.ALLOW)

    def test_preflight_does_not_interrupt_a_review_of_the_head_sha(self) -> None:
        pr = make_pr(changes_requested_by=("some-reviewer",), labels=(STALE_LABEL,))
        result = evaluate([pr], {1: make_state(status=STATUS_AI_REVIEW_STARTED)})
        self.assertEqual(result.verdict, GuardVerdict.WAIT)

    def test_a_revert_denies_instead_of_waiting_out_the_budget(self) -> None:
        result = evaluate(
            [make_pr()], {1: make_state(status=STATUS_REVERTED, head_sha=OTHER_SHA)}
        )
        self.assertEqual(result.verdict, GuardVerdict.DENY)
        self.assertNotIn("push a commit", result.message)

    def test_preflight_applies_when_the_ledger_row_is_for_another_commit(self) -> None:
        pr = make_pr(labels=(STALE_LABEL,))
        result = evaluate([pr], {1: make_state(head_sha=OTHER_SHA)})
        self.assertEqual(result.verdict, GuardVerdict.DENY)
        self.assertIn("CANNOT_REVIEW", result.message)


class TestWaiting(TestCase):
    def test_the_budget_opens_at_the_first_wait_not_at_the_merge_command(self) -> None:
        window = GreenlightWaitWindow()
        much_later = NOW + timedelta(hours=4)

        first = evaluate([make_pr()], {}, wait_window=window, now=much_later)
        self.assertEqual(first.verdict, GuardVerdict.WAIT)
        self.assertEqual(window.opened_at, much_later)

        still_waiting = evaluate(
            [make_pr()],
            {},
            wait_window=window,
            now=much_later + MAX_WAIT - timedelta(seconds=1),
        )
        self.assertEqual(still_waiting.verdict, GuardVerdict.WAIT)

        expired = evaluate(
            [make_pr()], {}, wait_window=window, now=much_later + MAX_WAIT
        )
        self.assertEqual(expired.verdict, GuardVerdict.DENY)
        self.assertIn("did not answer after 60 minutes of waiting", expired.message)
        self.assertIn("NO_ROW", expired.message)

    def test_an_allow_never_opens_the_budget(self) -> None:
        window = GreenlightWaitWindow()
        result = evaluate([make_pr()], {1: make_state()}, wait_window=window)
        self.assertEqual(result.verdict, GuardVerdict.ALLOW)
        self.assertIsNone(window.opened_at)

    def test_a_caller_that_cannot_retry_denies_instead_of_waiting(self) -> None:
        result = evaluate_force([make_pr()], {})
        self.assertEqual(result.verdict, GuardVerdict.DENY)
        self.assertIn("force merge does not wait", result.message)

    def test_a_caller_that_cannot_retry_still_allows_a_matching_land(self) -> None:
        result = evaluate_force([make_pr()], {1: make_state()})
        self.assertEqual(result.verdict, GuardVerdict.ALLOW)

    def test_the_two_refusals_that_follow_a_wait_name_a_next_step(self) -> None:
        window = GreenlightWaitWindow()
        evaluate([make_pr()], {}, wait_window=window)
        refusals = (
            evaluate([make_pr()], {}, wait_window=window, now=NOW + MAX_WAIT).message,
            evaluate_force([make_pr()], {}).message,
        )
        for message in refusals:
            with self.subTest(message=message):
                self.assertIn("human reviewer with merge rights", message)
                self.assertIn("push a commit", message)
                self.assertIn(f"`{STALE_LABEL}` label", message)
                self.assertIn("resolve any requested changes", message)


class TestAnnouncement(TestCase):
    def test_the_first_wait_carries_one_comment(self) -> None:
        window = GreenlightWaitWindow()
        first = evaluate([make_pr()], {}, wait_window=window)
        self.assertEqual(first.verdict, GuardVerdict.WAIT)
        self.assertIn(GREENLIGHT_LOGIN, first.comment)
        self.assertIn(f"head commit `{MERGE_SHA}`", first.comment)
        self.assertIn("up to 60 minutes", first.comment)
        self.assertNotIn("more minutes", first.comment)
        self.assertIn("human reviewer with merge rights", first.comment)
        # merge() aborts on a new commit, so the comment must not sell a push as a
        # free retrigger of the merge that is currently running.
        self.assertIn("re-issue the merge", first.comment)

    def test_later_waits_carry_no_comment(self) -> None:
        window = GreenlightWaitWindow()
        evaluate([make_pr()], {}, wait_window=window)
        later = evaluate(
            [make_pr()], {}, wait_window=window, now=NOW + timedelta(minutes=5)
        )
        self.assertEqual(later.verdict, GuardVerdict.WAIT)
        self.assertEqual(later.comment, "")

    def test_an_allow_or_a_denial_carries_no_comment(self) -> None:
        self.assertEqual(evaluate([make_pr()], {1: make_state()}).comment, "")
        self.assertEqual(
            evaluate([make_pr()], {1: make_state(status=STATUS_NO_LAND)}).comment, ""
        )
        self.assertEqual(evaluate_force([make_pr()], {}).comment, "")

    def test_the_comment_lists_every_waiting_pr_in_the_stack(self) -> None:
        prs = [
            make_pr(pr_num=1, head_sha="1" * 40),
            make_pr(pr_num=2, head_sha="2" * 40),
        ]
        result = evaluate(prs, {})
        self.assertIn("PR #1", result.comment)
        self.assertIn("PR #2", result.comment)

    def test_a_timeout_refusal_carries_no_comment(self) -> None:
        """The wait was already announced; the refusal is delivered by the merge itself."""
        window = GreenlightWaitWindow()
        first = evaluate([make_pr()], {}, wait_window=window)
        self.assertNotEqual(first.comment, "")
        expired = evaluate([make_pr()], {}, wait_window=window, now=NOW + MAX_WAIT)
        self.assertEqual(expired.verdict, GuardVerdict.DENY)
        self.assertEqual(expired.comment, "")

    def test_a_timeout_on_the_very_first_evaluation_carries_no_comment(self) -> None:
        """A budget already spent when it opens still must not announce a wait."""
        window = GreenlightWaitWindow(opened_at=NOW - MAX_WAIT)
        expired = evaluate([make_pr()], {}, wait_window=window)
        self.assertEqual(expired.verdict, GuardVerdict.DENY)
        self.assertEqual(expired.comment, "")
        self.assertFalse(window.announced)


class TestRefusalLayout(TestCase):
    def test_the_refusal_opens_with_the_cause_not_the_precondition(self) -> None:
        result = evaluate([make_pr()], {1: make_state(status=STATUS_NO_LAND)})
        headline = result.message.split("\n", 1)[0]
        self.assertEqual(
            headline, f"Greenlight recorded {STATUS_NO_LAND} for this exact commit."
        )

    def test_the_refusal_is_paragraphs_and_bullets_not_one_run_on(self) -> None:
        prs = [
            make_pr(pr_num=1, head_sha="1" * 40),
            make_pr(pr_num=2, head_sha="2" * 40),
        ]
        states = {2: make_state(pr_number=2, status=STATUS_NO_LAND, head_sha="2" * 40)}
        result = evaluate(prs, states)
        self.assertIn("\n\n", result.message)
        self.assertIn("- PR #1 (NO_ROW):", result.message)
        self.assertIn("- PR #2 (NEGATIVE_VERDICT):", result.message)
        longest = max(len(line) for line in result.message.splitlines())
        self.assertLess(longest, 300)

    def test_the_pr_to_act_on_is_listed_before_the_ones_merely_waiting(self) -> None:
        prs = [
            make_pr(pr_num=1, head_sha="1" * 40),
            make_pr(pr_num=2, head_sha="2" * 40),
        ]
        states = {2: make_state(pr_number=2, status=STATUS_NO_LAND, head_sha="2" * 40)}
        message = evaluate(prs, states).message
        self.assertLess(message.index("PR #2"), message.index("PR #1 ("))

    def test_a_timeout_refusal_still_opens_with_the_cause(self) -> None:
        window = GreenlightWaitWindow()
        evaluate([make_pr()], {}, wait_window=window)
        expired = evaluate([make_pr()], {}, wait_window=window, now=NOW + MAX_WAIT)
        lines = expired.message.split("\n\n")
        self.assertEqual(lines[0], "Greenlight has not reviewed this PR yet.")
        self.assertIn("60 minutes", lines[1])


class TestMergeAuthorizedApproval(TestCase):
    """A PR greenlight's own scan drops because a merge_rules approver handled it."""

    def _pr(self, **kwargs: Any) -> PRUnderMerge:
        return make_pr(
            approved_by=(GREENLIGHT_LOGIN, "malfet"),
            merge_authorized=("malfet",),
            **kwargs,
        )

    def test_it_refuses_now_rather_than_waiting_an_hour(self) -> None:
        result = evaluate([self._pr()], {})
        self.assertEqual(result.verdict, GuardVerdict.DENY)
        self.assertIn("CANNOT_REVIEW", result.message)
        self.assertIn("malfet", result.message)
        self.assertIn("merge_rules.yaml", result.message)
        self.assertIn("Dismiss that approval", result.message)

    def test_the_advice_does_not_send_the_author_after_a_new_commit(self) -> None:
        """Pushing cannot help: the scan drops the PR again at any head."""
        result = evaluate([self._pr()], {})
        headline, *rest = result.message.split("\n\n")
        self.assertNotIn("push a commit", headline.lower())
        self.assertNotIn("@greenlight", result.message)

    def test_an_unreadable_reviewer_type_waits_instead_of_refusing(self) -> None:
        result = evaluate([self._pr(bot_reviewers=None)], {})
        self.assertEqual(result.verdict, GuardVerdict.WAIT)
        self.assertIn("UNCLASSIFIED_REVIEWER", result.message)

    def test_an_approval_at_the_head_sha_is_still_honoured(self) -> None:
        result = evaluate([self._pr()], {1: make_state(status=STATUS_LAND)})
        self.assertEqual(result.verdict, GuardVerdict.ALLOW)


class TestUnclassifiedReviewer(TestCase):
    """A failed reviewer-type lookup must stay retryable, never a terminal refusal."""

    def test_an_unrecognized_blocker_waits_when_the_lookup_failed(self) -> None:
        pr = make_pr(changes_requested_by=("some-new-app",), bot_reviewers=None)
        result = evaluate([pr], {})
        self.assertEqual(result.verdict, GuardVerdict.WAIT)
        self.assertIn("UNCLASSIFIED_REVIEWER", result.message)
        self.assertIn("some-new-app", result.message)

    def test_a_known_bot_blocker_never_needs_the_lookup(self) -> None:
        pr = make_pr(changes_requested_by=("pytorch-bot",), bot_reviewers=None)
        self.assertEqual(evaluate([pr], {}).verdict, GuardVerdict.WAIT)
        self.assertNotIn("UNCLASSIFIED_REVIEWER", evaluate([pr], {}).message)

    def test_a_sustained_lookup_failure_still_refuses_once_the_budget_is_spent(
        self,
    ) -> None:
        pr = make_pr(changes_requested_by=("some-new-app",), bot_reviewers=None)
        window = GreenlightWaitWindow()
        self.assertEqual(
            evaluate([pr], {}, wait_window=window).verdict, GuardVerdict.WAIT
        )
        expired = evaluate([pr], {}, wait_window=window, now=NOW + MAX_WAIT)
        self.assertEqual(expired.verdict, GuardVerdict.DENY)


if __name__ == "__main__":
    main()
