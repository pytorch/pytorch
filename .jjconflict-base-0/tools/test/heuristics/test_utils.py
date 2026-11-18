from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))

import tools.testing.target_determination.heuristics.utils as utils
from tools.testing.test_run import TestRun


sys.path.remove(str(REPO_ROOT))


class TestHeuristicsUtils(unittest.TestCase):
    def assertDictAlmostEqual(
        self, first: dict[TestRun, Any], second: dict[TestRun, Any]
    ) -> None:
        self.assertEqual(first.keys(), second.keys())
        for key in first:
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

    def test_td_enabled(self) -> None:
        for workflow in ("pull", "trunk", "rocm-mi300", "periodic-rocm-mi200"):
            self.assertTrue(utils.td_enabled_workflow(workflow), workflow)
        for workflow in (
            "periodic",
            "inductor",
            "slow",
            "pull-rocm",
            "trunk-rocm-sandbox",
            "rocm",
        ):
            self.assertFalse(utils.td_enabled_workflow(workflow), workflow)

        def job(env: str, macos: bool = False, no_td: bool = False) -> bool:
            return utils.td_enabled_job(env, macos=macos, no_td=no_td)

        self.assertTrue(job("linux-jammy-py3.10"))
        self.assertFalse(job("linux-xpu"))
        self.assertFalse(job("linux-noble-xpu-n-py3.11"))
        self.assertFalse(job("linux-onnx"))
        self.assertFalse(job("linux", macos=True))
        self.assertFalse(job("linux", no_td=True))

    def test_td_enabled_default(self) -> None:
        def default(
            env: str = "linux",
            workflow: str = "pull",
            is_ci: bool = True,
            pr: int | None = 1,
            macos: bool = False,
            no_td: bool = False,
        ) -> bool:
            return utils.td_enabled_default(
                build_environment=env,
                workflow=workflow,
                is_ci=is_ci,
                pr_number=pr,
                macos=macos,
                no_td=no_td,
            )

        self.assertTrue(default())
        self.assertFalse(default(is_ci=False))
        self.assertFalse(default(pr=None))
        self.assertFalse(default(env="linux-xpu"))
        self.assertFalse(default(macos=True))
        self.assertFalse(default(no_td=True))
        self.assertFalse(default(workflow="slow"))


if __name__ == "__main__":
    unittest.main()
