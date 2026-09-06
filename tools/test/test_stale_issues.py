from __future__ import annotations

from datetime import date
from unittest import main, mock, TestCase

from tools.stale_issues import parse_older_than


def _today(value: date) -> type[date]:
    """A ``date`` subclass whose ``today()`` is pinned, so the relative-date
    math in ``parse_older_than`` is deterministic regardless of the real clock."""

    class PinnedDate(date):
        # date.today is typed to return Self; this stub pins it to a fixed
        # date so the relative-date math is deterministic.
        @classmethod
        def today(cls) -> date:  # pyrefly: ignore [bad-override]
            return value

    return PinnedDate


class TestParseOlderThan(TestCase):
    def test_days_and_weeks(self) -> None:
        with mock.patch("tools.stale_issues.date", _today(date(2025, 3, 15))):
            self.assertEqual(parse_older_than("10 days"), date(2025, 3, 5))
            self.assertEqual(parse_older_than("2 weeks"), date(2025, 3, 1))

    def test_months_clamps_to_non_leap_february(self) -> None:
        # Three months before 2025-05-31 is February 2025 (non-leap), so the day
        # must clamp to 28 rather than build an invalid 2025-02-31.
        with mock.patch("tools.stale_issues.date", _today(date(2025, 5, 31))):
            self.assertEqual(parse_older_than("3 months"), date(2025, 2, 28))

    def test_months_keep_leap_day_in_leap_year(self) -> None:
        with mock.patch("tools.stale_issues.date", _today(date(2024, 5, 31))):
            self.assertEqual(parse_older_than("3 months"), date(2024, 2, 29))

    def test_year_keeps_day_in_31_day_month(self) -> None:
        # A year before 2024-03-31 is 2023-03-31. March has 31 days, so the day
        # must be preserved, not clamped down to the 28 that only February needs.
        with mock.patch("tools.stale_issues.date", _today(date(2024, 3, 31))):
            self.assertEqual(parse_older_than("1 year"), date(2023, 3, 31))

    def test_year_keeps_day_in_30_day_month(self) -> None:
        # A year before 2024-04-30 is 2023-04-30 (April has 30 days).
        with mock.patch("tools.stale_issues.date", _today(date(2024, 4, 30))):
            self.assertEqual(parse_older_than("1 year"), date(2023, 4, 30))

    def test_year_clamps_leap_day_to_non_leap_february(self) -> None:
        # A year before the leap day 2024-02-29 lands in non-leap February 2023,
        # which only has 28 days, so the day must clamp to the 28th.
        with mock.patch("tools.stale_issues.date", _today(date(2024, 2, 29))):
            self.assertEqual(parse_older_than("1 year"), date(2023, 2, 28))


if __name__ == "__main__":
    main()
