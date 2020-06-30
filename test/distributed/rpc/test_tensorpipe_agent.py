#!/usr/bin/env python3
import unittest

from torch.testing._internal import dist_utils
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import TEST_WITH_ASAN, run_tests
from torch.testing._internal.distributed.ddp_under_dist_autograd_test import (
    TestDdpComparison,
    TestDdpUnderDistAutograd,
)
from torch.testing._internal.distributed.rpc.dist_autograd_test import DistAutogradTest
from torch.testing._internal.distributed.rpc.dist_optimizer_test import (
    DistOptimizerTest,
)
from torch.testing._internal.distributed.rpc.rpc_test import TensorPipeAgentRpcTest
from torch.testing._internal.distributed.rpc.tensorpipe_rpc_agent_test_fixture import (
    TensorPipeRpcAgentTestFixture,
)


# FIXME This is needed to make some functions in dist_init work. Those functions
# should be moved to methods of the fixture. When that is done, remove this.
dist_utils.TEST_CONFIG.rpc_backend_name = "TENSORPIPE"


@unittest.skipIf(
    TEST_WITH_ASAN, "Skip ASAN as torch + multiprocessing spawn have known issues"
)
class SpawnHelper(MultiProcessTestCase, TensorPipeRpcAgentTestFixture):
    def setUp(self):
        super().setUp()
        self._spawn_processes()


class RpcTestWithSpawn(SpawnHelper, TensorPipeAgentRpcTest):
    pass


class DistAutogradTestWithSpawn(SpawnHelper, DistAutogradTest):
    pass


class DistOptimizerTestWithSpawn(SpawnHelper, DistOptimizerTest):
    pass


class TestDdpUnderDistAutograd(SpawnHelper, TestDdpUnderDistAutograd):
    pass


class TestDdpComparison(SpawnHelper, TestDdpComparison):
    pass


if __name__ == "__main__":
    run_tests()
