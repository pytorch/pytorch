#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

from dist_autograd_test import DistAutogradTest
from common_distributed import MultiProcessTestCase
from common_utils import TEST_WITH_ASAN, run_tests
from process_group_rpc_agent_test_fixture import ProcessGroupRpcAgentTestFixture

import unittest

@unittest.skipIf(TEST_WITH_ASAN, "Skip ASAN as torch + multiprocessing spawn have known issues")
class DistAutogradTestWithSpawn(MultiProcessTestCase, ProcessGroupRpcAgentTestFixture, DistAutogradTest):

    def setUp(self):
        super(DistAutogradTestWithSpawn, self).setUp()
        self._spawn_processes()

if __name__ == '__main__':
    run_tests()
