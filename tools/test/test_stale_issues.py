from __future__ import annotations

from datetime import date
from unittest import main, mock, TestCase

from tools.stale_issues import parse_older_than


def _today(value: date) -> type[date]:
    """A ``date`` subclass whose ``today()`` is pinned, so the relative-date
    math in ``parse_older_than`` is deterministic regardless of the real clock."""

    class PinnedDate(date):
        @classmethod
        def today(cls) -> date:
            return value

    return PinnedDate


class TestParseOlderThan(TestCase):
    def test_months_clamps_to_non_leap_february(self) -> None:
        # Three months before 2025-05-31 is February 2025, which is not a leap
        # year. The day must clamp to 28, not the hardcoded 29 that used to
        # raise "day is out of range for month".
        with mock.patch("tools.stale_issues.date", _today(date(2025, 5, 31))):
            self.assertEqual(parse_older_than("3 months"), date(2025, 2, 28))

    def test_months_keeps_leap_day_in_leap_year(self) -> None:
        # The same cutoff in a leap year should keep February 29.
        with mock.patch("tools.stale_issues.date", _today(date(2024, 5, 31))):
            self.assertEqual(parse_older_than("3 months"), date(2024, 2, 29))

    def test_months_wraps_across_year_boundary(self) -> None:
        # Two months before January 2025 is November 2024 (30 days).
        with mock.patch("tools.stale_issues.date", _today(date(2025, 1, 31))):
            self.assertEqual(parse_older_than("2 months"), date(2024, 11, 30))

    def test_days_and_weeks_are_unchanged(self) -> None:
        with mock.patch("tools.stale_issues.date", _today(date(2025, 3, 15))):
            self.assertEqual(parse_older_than("10 days"), date(2025, 3, 5))
            self.assertEqual(parse_older_than("2 weeks"), date(2025, 3, 1))


if __name__ == "__main__":
    main()
