#!/usr/bin/env python3
import unittest

from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import TEST_WITH_ASAN, run_tests
from torch.testing._internal.distributed.rpc.rpc_test import FaultyAgentRpcTest


@unittest.skipIf(
    TEST_WITH_ASAN, "Skip ASAN as torch + multiprocessing spawn have known issues"
)
class FaultyAgentRpcTestWithSpawn(MultiProcessTestCase, FaultyAgentRpcTest):
    def setUp(self):
        super(FaultyAgentRpcTestWithSpawn, self).setUp()
        self._spawn_processes()


if __name__ == "__main__":
    run_tests()
