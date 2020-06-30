#!/usr/bin/env python3
import unittest

from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import TEST_WITH_ASAN, run_tests
from torch.testing._internal.distributed.rpc.dist_autograd_test import (
    FaultyAgentDistAutogradTest,
)
from torch.testing._internal.distributed.rpc.faulty_rpc_agent_test_fixture import (
    FaultyRpcAgentTestFixture,
)
from torch.testing._internal.distributed.rpc.jit.rpc_test_faulty import (
    JitFaultyAgentRpcTest,
)
from torch.testing._internal.distributed.rpc.rpc_test import FaultyAgentRpcTest


@unittest.skipIf(
    TEST_WITH_ASAN, "Skip ASAN as torch + multiprocessing spawn have known issues"
)
class SpawnHelper(MultiProcessTestCase, FaultyRpcAgentTestFixture):
    def setUp(self):
        super().setUp()
        self._spawn_processes()


class RpcTestWithSpawn(SpawnHelper, FaultyAgentRpcTest):
    pass


class DistAutogradTestWithSpawn(SpawnHelper, FaultyAgentDistAutogradTest):
    pass


class JitRpcTestWithSpawn(SpawnHelper, JitFaultyAgentRpcTest):
    pass


if __name__ == "__main__":
    run_tests()
