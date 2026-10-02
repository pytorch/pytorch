from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.stats.upload_test_stats_intermediate import select_recent_attempts


sys.path.remove(str(REPO_ROOT))


class TestSelectRecentAttempts(unittest.TestCase):
    def test_single_attempt_is_kept(self) -> None:
        self.assertEqual(select_recent_attempts([1], 3), [1])

    def test_keeps_the_newest_attempts_oldest_first(self) -> None:
        self.assertEqual(select_recent_attempts([3, 1, 4, 2], 2), [3, 4])

    def test_fewer_attempts_than_the_cap(self) -> None:
        self.assertEqual(select_recent_attempts([2, 1], 5), [1, 2])


if __name__ == "__main__":
    unittest.main()
