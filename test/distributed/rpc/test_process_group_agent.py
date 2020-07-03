#!/usr/bin/env python3
import unittest

from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import TEST_WITH_ASAN, run_tests
from torch.testing._internal.distributed.ddp_under_dist_autograd_test import (
    DdpComparisonTest,
    DdpUnderDistAutogradTest,
)
from torch.testing._internal.distributed.nn.api.remote_module_test import (
    RemoteModuleTest,
)
from torch.testing._internal.distributed.rpc.dist_autograd_test import DistAutogradTest
from torch.testing._internal.distributed.rpc.dist_optimizer_test import (
    DistOptimizerTest,
)
from torch.testing._internal.distributed.rpc.jit.dist_autograd_test import (
    JitDistAutogradTest,
)
from torch.testing._internal.distributed.rpc.jit.rpc_test import JitRpcTest
from torch.testing._internal.distributed.rpc.process_group_agent_test_fixture import (
    ProcessGroupRpcAgentTestFixture,
)
from torch.testing._internal.distributed.rpc.rpc_test import ProcessGroupAgentRpcTest


@unittest.skipIf(
    TEST_WITH_ASAN, "Skip ASAN as torch + multiprocessing spawn have known issues"
)
class SpawnHelper(ProcessGroupRpcAgentTestFixture, MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        self._spawn_processes()


class ProcessGroupRpcTestWithSpawn(ProcessGroupAgentRpcTest, SpawnHelper):
    pass


class ProcessGroupDistAutogradTestWithSpawn(DistAutogradTest, SpawnHelper):
    pass


class ProcessGroupDistOptimizerTestWithSpawn(DistOptimizerTest, SpawnHelper):
    pass


class ProcessGroupJitRpcTestWithSpawn(JitRpcTest, SpawnHelper):
    pass


class ProcessGroupJitDistAutogradTestWithSpawn(JitDistAutogradTest, SpawnHelper):
    pass


class ProcessGroupRemoteModuleTestWithSpawn(RemoteModuleTest, SpawnHelper):
    pass


class ProcessGroupDdpUnderDistAutogradTest(DdpUnderDistAutogradTest, SpawnHelper):
    pass


class ProcessGroupDdpComparisonTest(DdpComparisonTest, SpawnHelper):
    pass


if __name__ == "__main__":
    run_tests()
