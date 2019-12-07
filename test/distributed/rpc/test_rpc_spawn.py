#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

from torch.testlib.distributed.rpc.rpc_test import RpcTest
from torch.testlib.common_distributed import MultiProcessTestCase
from torch.testlib.common_utils import TEST_WITH_ASAN, run_tests

import unittest

@unittest.skipIf(TEST_WITH_ASAN, "Skip ASAN as torch + multiprocessing spawn have known issues")
class RpcTestWithSpawn(MultiProcessTestCase, RpcTest):

    def setUp(self):
        super(RpcTestWithSpawn, self).setUp()
        self._spawn_processes()

if __name__ == '__main__':
    run_tests()
