#!/usr/bin/env python3

from __future__ import annotations

from datetime import timedelta
from unittest import main, TestCase

from greenlight_identity import GREENLIGHT_LOGIN
from greenlight_ledger import (
    STATUS_AI_REVIEW_DISPATCHED,
    STATUS_AI_REVIEW_STARTED,
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_LAND,
    STATUS_NO_LAND,
)
from greenlight_preflight import (
    blocking_reviewers,
    cannot_review,
    EXCLUDED_LABELS,
    REVIEW_WINDOW,
    STALE_LABEL,
)
from test_greenlight_guard import make_pr, NOW


IN_WINDOW = (NOW - REVIEW_WINDOW + timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
OUT_OF_WINDOW = (NOW - REVIEW_WINDOW - timedelta(minutes=1)).strftime(
    "%Y-%m-%dT%H:%M:%SZ"
)

# greenlight's review._recency_filter keeps a PR whose recorded status is not terminal,
# because its scan can still re-dispatch that review; TERMINAL_STATUSES upstream is
# exactly {LAND, NO_LAND}.
REDISPATCHABLE = (
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_AI_REVIEW_STARTED,
    STATUS_AI_REVIEW_DISPATCHED,
)
SETTLED = (STATUS_LAND, STATUS_NO_LAND)


class TestBlockingReviewers(TestCase):
    def test_a_human_blocks(self) -> None:
        pr = make_pr(changes_requested_by=("some-reviewer",))
        self.assertEqual(blocking_reviewers(pr).humans, ["some-reviewer"])

    def test_a_listed_bot_does_not_block(self) -> None:
        pr = make_pr(changes_requested_by=("pytorchbot", "PyTorch-Bot"))
        self.assertEqual(blocking_reviewers(pr).humans, [])

    def test_a_bot_suffixed_login_does_not_block(self) -> None:
        pr = make_pr(changes_requested_by=("some-app[bot]",))
        self.assertEqual(blocking_reviewers(pr).humans, [])

    def test_greenlight_itself_does_not_block(self) -> None:
        pr = make_pr(changes_requested_by=(GREENLIGHT_LOGIN,))
        self.assertEqual(blocking_reviewers(pr).humans, [])

    def test_an_app_github_reports_as_a_bot_does_not_block(self) -> None:
        """REST names an App `slug[bot]`; GraphQL names the same App bare."""
        pr = make_pr(
            changes_requested_by=("some-new-app", "a-human"),
            bot_reviewers=("Some-New-App[bot]",),
        )
        self.assertEqual(blocking_reviewers(pr).humans, ["a-human"])

    def test_a_failed_reviewer_type_lookup_is_reported_as_unclassified(self) -> None:
        pr = make_pr(changes_requested_by=("some-new-app",), bot_reviewers=None)
        scan = blocking_reviewers(pr)
        self.assertEqual(scan.humans, ["some-new-app"])
        self.assertFalse(scan.classified)

    def test_a_failed_lookup_with_nothing_to_classify_is_still_classified(self) -> None:
        pr = make_pr(changes_requested_by=("pytorchbot",), bot_reviewers=None)
        scan = blocking_reviewers(pr)
        self.assertEqual(scan.humans, [])
        self.assertTrue(scan.classified)


class TestHumanApproval(TestCase):
    """greenlight's review_gate.human_review_skip_reason HUMAN_APPROVED arm."""

    def test_an_approver_listed_in_merge_rules_blocks(self) -> None:
        pr = make_pr(
            approved_by=(GREENLIGHT_LOGIN, "malfet"), merge_authorized=("malfet",)
        )
        blocked = cannot_review(pr, None, NOW)
        if blocked is None:
            self.fail("expected the merge-authorized approval to block a review")
        self.assertIn("malfet", blocked.cause)
        self.assertIn("merge_rules.yaml", blocked.cause)
        self.assertIn("Dismiss that approval", blocked.fix)
        self.assertFalse(blocked.retryable)

    def test_membership_ignores_case(self) -> None:
        pr = make_pr(
            approved_by=(GREENLIGHT_LOGIN, "Malfet"), merge_authorized=("malfet",)
        )
        self.assertIsNotNone(cannot_review(pr, None, NOW))

    def test_an_approver_absent_from_merge_rules_does_not_block(self) -> None:
        pr = make_pr(
            approved_by=(GREENLIGHT_LOGIN, "a-random-stranger"),
            merge_authorized=("malfet",),
        )
        self.assertIsNone(cannot_review(pr, None, NOW))

    def test_greenlight_is_listed_in_merge_rules_but_is_not_a_human(self) -> None:
        for login in (GREENLIGHT_LOGIN, f"{GREENLIGHT_LOGIN}[bot]"):
            with self.subTest(login=login):
                pr = make_pr(
                    approved_by=(login,), merge_authorized=(GREENLIGHT_LOGIN, "malfet")
                )
                self.assertIsNone(cannot_review(pr, None, NOW))

    def test_a_named_bot_in_merge_rules_is_not_a_human(self) -> None:
        pr = make_pr(
            approved_by=(GREENLIGHT_LOGIN, "pytorchbot"),
            merge_authorized=("pytorchbot",),
        )
        self.assertIsNone(cannot_review(pr, None, NOW))

    def test_an_app_github_reports_as_a_bot_is_not_a_human(self) -> None:
        pr = make_pr(
            approved_by=(GREENLIGHT_LOGIN, "some-new-app"),
            merge_authorized=("some-new-app",),
            bot_reviewers=("some-new-app[bot]",),
        )
        self.assertIsNone(cannot_review(pr, None, NOW))

    def test_a_failed_reviewer_type_lookup_is_retryable(self) -> None:
        pr = make_pr(
            approved_by=(GREENLIGHT_LOGIN, "some-new-app"),
            merge_authorized=("some-new-app",),
            bot_reviewers=None,
        )
        blocked = cannot_review(pr, None, NOW)
        if blocked is None:
            self.fail("expected the unclassified approver to block a review")
        self.assertTrue(blocked.retryable)

    def test_an_approval_blocks_regardless_of_the_recorded_status(self) -> None:
        """The approval skip is applied at fingerprint time, past the recency escape."""
        pr = make_pr(
            approved_by=(GREENLIGHT_LOGIN, "malfet"),
            merge_authorized=("malfet",),
            updated_at=IN_WINDOW,
        )
        for status in (*REDISPATCHABLE, *SETTLED, None):
            with self.subTest(status=status):
                self.assertIsNotNone(cannot_review(pr, status, NOW))

    def test_requested_changes_are_reported_ahead_of_an_approval(self) -> None:
        pr = make_pr(
            approved_by=(GREENLIGHT_LOGIN, "malfet"),
            changes_requested_by=("some-reviewer",),
            merge_authorized=("malfet",),
        )
        blocked = cannot_review(pr, None, NOW)
        if blocked is None:
            self.fail("expected the requested changes to block a review")
        self.assertIn("some-reviewer", blocked.cause)


class TestRecencyFilterMirror(TestCase):
    """Pins this module's mirror of greenlight review._recency_filter's shape.

    It checks the escape this module implements, not upstream's source: nothing here
    would notice if upstream changed, which is what the citations in the module under
    test are for.
    """

    def test_stale_label_blocks_only_a_settled_or_absent_review(self) -> None:
        pr = make_pr(labels=(STALE_LABEL,))
        for status in REDISPATCHABLE:
            with self.subTest(status=status):
                self.assertIsNone(cannot_review(pr, status, NOW))
        for status in SETTLED:
            with self.subTest(status=status):
                self.assertIsNotNone(cannot_review(pr, status, NOW))
        self.assertIsNotNone(cannot_review(pr, None, NOW))

    def test_every_excluded_label_blocks(self) -> None:
        for label in EXCLUDED_LABELS:
            with self.subTest(label=label):
                blocked = cannot_review(make_pr(labels=(label,)), None, NOW)
                if blocked is None:
                    self.fail(f"expected the {label} label to block a review")
                self.assertIn(f"`{label}`", blocked.cause)

    def test_out_of_window_blocks_only_a_settled_or_absent_review(self) -> None:
        pr = make_pr(updated_at=OUT_OF_WINDOW)
        for status in REDISPATCHABLE:
            with self.subTest(status=status):
                self.assertIsNone(cannot_review(pr, status, NOW))
        for status in SETTLED:
            with self.subTest(status=status):
                self.assertIsNotNone(cannot_review(pr, status, NOW))
        self.assertIsNotNone(cannot_review(pr, None, NOW))

    def test_a_recent_unlabelled_pr_is_never_blocked(self) -> None:
        pr = make_pr(updated_at=IN_WINDOW)
        for status in (*REDISPATCHABLE, *SETTLED, None):
            with self.subTest(status=status):
                self.assertIsNone(cannot_review(pr, status, NOW))

    def test_requested_changes_block_regardless_of_the_recorded_status(self) -> None:
        pr = make_pr(changes_requested_by=("some-reviewer",), updated_at=IN_WINDOW)
        for status in (*REDISPATCHABLE, *SETTLED, None):
            with self.subTest(status=status):
                blocked = cannot_review(pr, status, NOW)
                if blocked is None:
                    self.fail("expected the requested changes to block a review")
                self.assertIn("some-reviewer", blocked.cause)

    def test_every_block_names_a_way_out(self) -> None:
        cases = (
            make_pr(changes_requested_by=("some-reviewer",)),
            make_pr(changes_requested_by=("some-new-app",), bot_reviewers=None),
            make_pr(labels=(STALE_LABEL,)),
            make_pr(updated_at=OUT_OF_WINDOW),
            make_pr(
                approved_by=(GREENLIGHT_LOGIN, "malfet"), merge_authorized=("malfet",)
            ),
        )
        for pr in cases:
            with self.subTest(
                pr=pr.labels or pr.changes_requested_by or pr.approved_by
            ):
                blocked = cannot_review(pr, None, NOW)
                if blocked is None:
                    self.fail("expected this PR to be unreviewable")
                self.assertTrue(blocked.fix.endswith("."))
                self.assertTrue(blocked.fix[:1].isupper())
                self.assertIn("approval", blocked.fix)


if __name__ == "__main__":
    main()
