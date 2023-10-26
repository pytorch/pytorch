import io
import json
import pathlib
import sys
import unittest
from typing import Any, Dict, Optional, Set
from unittest import mock

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
try:
    # using tools/ to optimize test run.
    sys.path.append(str(REPO_ROOT))

    from tools.testing.target_determination.determinator import (
        AggregatedHeuristics,
        get_test_prioritizations,
        TestPrioritizations,
    )
    from tools.testing.target_determination.heuristics import HEURISTICS
    from tools.testing.target_determination.heuristics.previously_failed_in_pr import (
        _get_previously_failing_tests,
    )
    from tools.testing.test_run import TestRun, TestRuns

except ModuleNotFoundError:
    print("Can't import required modules, exiting")
    sys.exit(1)

class HeuristicsTestMixin(unittest.TestCase):
    def assert_heuristics_match(
        self,
        test_prioritizations: TestPrioritizations,
        expected_high_tests: Optional[TestRuns] = None,
        expected_probable_tests: Optional[TestRuns] = None,
        expected_unranked_tests: Optional[TestRuns] = None,
    ) -> None:
        if expected_unranked_tests:
            self.assertTupleEqual(
                test_prioritizations.get_unranked_relevance_tests(),
                expected_unranked_tests,
                "Unranked tests differ",
            )

        if expected_probable_tests:
            self.assertTupleEqual(
                test_prioritizations.get_probable_relevance_tests(),
                expected_probable_tests,
                "Probable relevance tests differ",
            )

        if expected_high_tests:
            self.assertTupleEqual(
                test_prioritizations.get_high_relevance_tests(),
                expected_high_tests,
                "High relevance tests differ",
            )