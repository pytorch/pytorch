from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(REPO_ROOT))

import tools.testing.target_determination.heuristics.utils as utils
from tools.testing.test_run import TestRun


sys.path.remove(str(REPO_ROOT))


class TestHeuristicsUtils(unittest.TestCase):
    def assertDictAlmostEqual(
        self, first: dict[TestRun, Any], second: dict[TestRun, Any]
    ) -> None:
        self.assertEqual(first.keys(), second.keys())
        for key in first.keys():
            self.assertAlmostEqual(first[key], second[key])

    def test_normalize_ratings(self) -> None:
        ratings: dict[TestRun, float] = {
            TestRun("test1"): 1,
            TestRun("test2"): 2,
            TestRun("test3"): 4,
        }
        normalized = utils.normalize_ratings(ratings, 4)
        self.assertDictAlmostEqual(normalized, ratings)

        normalized = utils.normalize_ratings(ratings, 0.1)
        self.assertDictAlmostEqual(
            normalized,
            {
                TestRun("test1"): 0.025,
                TestRun("test2"): 0.05,
                TestRun("test3"): 0.1,
            },
        )

        normalized = utils.normalize_ratings(ratings, 0.2, min_value=0.1)
        self.assertDictAlmostEqual(
            normalized,
            {
                TestRun("test1"): 0.125,
                TestRun("test2"): 0.15,
                TestRun("test3"): 0.2,
            },
        )


if __name__ == "__main__":
    unittest.main()
