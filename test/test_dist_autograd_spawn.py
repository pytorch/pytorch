#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import process_group_dist_autograd_test
from common_distributed import MultiProcessTestCase
from common_utils import TEST_WITH_ASAN, run_tests


@unittest.skipIf(
    TEST_WITH_ASAN, "Skip ASAN as torch + multiprocessing spawn have known issues"
)
class DistAutogradTestWithSpawn(
    MultiProcessTestCase, process_group_dist_autograd_test.ProcessGroupDistAutogradTest
):
    def setUp(self):
        super(DistAutogradTestWithSpawn, self).setUp()
        self._spawn_processes()


if __name__ == "__main__":
    run_tests()
