import pathlib
import sys
import unittest
from typing import Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
try:
    # using tools/ to optimize test run.
    sys.path.append(str(REPO_ROOT))

    from tools.testing.target_determination.determinator import TestPrioritizations
    from tools.testing.test_run import TestRuns

except ModuleNotFoundError:
    print("Can't import required modules, exiting")
    sys.exit(1)


class HeuristicsTestMixin(unittest.TestCase):
    def assertHeuristicsMatch(
        self,
        test_prioritizations: TestPrioritizations,
        expected_prioritizations: Optional[TestPrioritizations] = None,
        expected_high_tests: Optional[TestRuns] = None,
        expected_probable_tests: Optional[TestRuns] = None,
        expected_unranked_tests: Optional[TestRuns] = None,
    ) -> None:
        # if expected_prioritizations is set, none of the other expected values should be set
        if expected_prioritizations:
            assert not (
                expected_high_tests
                or expected_probable_tests
                or expected_unranked_tests
            )
            expected_high_tests = expected_prioritizations.get_high_relevance_tests()
            expected_probable_tests = (
                expected_prioritizations.get_probable_relevance_tests()
            )
            expected_unranked_tests = (
                expected_prioritizations.get_unranked_relevance_tests()
            )

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
