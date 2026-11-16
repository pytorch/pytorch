"""A test with a failing subtest (TestReportJsonl.test_subtests)."""

import unittest


class TestSubtests(unittest.TestCase):
    def test_failing_subtest(self):
        for i in range(2):
            with self.subTest(i=i):
                self.assertEqual(i, 0)
