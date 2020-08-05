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
from torch.testing._internal.distributed.rpc.rpc_test import TensorPipeAgentRpcTest
from torch.testing._internal.distributed.rpc.tensorpipe_rpc_agent_test_fixture import (
    TensorPipeRpcAgentTestFixture,
)


@unittest.skipIf(
    TEST_WITH_ASAN, "Skip ASAN as torch + multiprocessing spawn have known issues"
)
class SpawnHelper(TensorPipeRpcAgentTestFixture, MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        self._spawn_processes()


class TensorPipeRpcTestWithSpawn(TensorPipeAgentRpcTest, SpawnHelper):
    pass


class TensorPipeDistAutogradTestWithSpawn(DistAutogradTest, SpawnHelper):
    pass


class TensorPipeDistOptimizerTestWithSpawn(DistOptimizerTest, SpawnHelper):
    pass


class TensorPipeJitRpcTestWithSpawn(JitRpcTest, SpawnHelper):
    pass


class TensorPipeJitDistAutogradTestWithSpawn(JitDistAutogradTest, SpawnHelper):
    pass


class TensorPipeRemoteModuleTestWithSpawn(RemoteModuleTest, SpawnHelper):
    pass


class TensorPipeDdpUnderDistAutogradTest(DdpUnderDistAutogradTest, SpawnHelper):
    pass


class TensorPipeDdpComparisonTest(DdpComparisonTest, SpawnHelper):
    pass


if __name__ == "__main__":
    run_tests()
