"""A slow test whose flakefinder copies two xdist workers run at the same time
(TestReportJsonl.test_concurrent_copies)."""

import time
import unittest


class TestSleep(unittest.TestCase):
    def test_sleep(self):
        time.sleep(0.3)
