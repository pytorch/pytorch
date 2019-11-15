#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

from common_distributed import MultiProcessTestCase
from common_utils import TEST_WITH_ASAN, run_tests
from distributed.rpc.process_group.dist_autograd_test import (
    ProcessGroupDistAutogradTest,
)

import unittest

@unittest.skipIf(TEST_WITH_ASAN, "Skip ASAN as torch + multiprocessing spawn have known issues")
class ProcessGroupDistAutogradTestWithSpawn(
    MultiProcessTestCase, ProcessGroupDistAutogradTest
):

    def setUp(self):
        super(ProcessGroupDistAutogradTestWithSpawn, self).setUp()
        self._spawn_processes()

if __name__ == '__main__':
    run_tests()
