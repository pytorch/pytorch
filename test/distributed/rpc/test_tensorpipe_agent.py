#!/usr/bin/env python3
import unittest

from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import TEST_WITH_ASAN, run_tests
from torch.testing._internal.distributed.ddp_under_dist_autograd_test import (
    DdpComparisonTest,
    DdpUnderDistAutogradTest,
)
from torch.testing._internal.distributed.rpc.dist_autograd_test import DistAutogradTest
from torch.testing._internal.distributed.rpc.dist_optimizer_test import (
    DistOptimizerTest,
)
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


class TensorPipeDdpUnderDistAutogradTest(DdpUnderDistAutogradTest, SpawnHelper):
    pass


class TensorPipeDdpComparisonTest(DdpComparisonTest, SpawnHelper):
    pass


if __name__ == "__main__":
    run_tests()
