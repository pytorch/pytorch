#!/usr/bin/env python3
import unittest

from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import TEST_WITH_ASAN, run_tests
from torch.testing._internal.distributed.rpc.tensorpipe_rpc_agent_test_fixture import (
    TensorPipeRpcAgentTestFixture,
)
from torch.testing._internal.dist_utils import (
    dist_init,
)
from torch.testing._internal.distributed.ddp_under_dist_autograd_test import (
    TestDdpComparison,
    TestDdpUnderDistAutograd,
)
import torch.distributed.rpc as rpc

@unittest.skipIf(
    TEST_WITH_ASAN, "Skip ASAN as torch + multiprocessing spawn have known issues"
)
class TestDdpUnderDistAutogradTensorPipe(
    TensorPipeRpcAgentTestFixture,
    TestDdpUnderDistAutograd,
    MultiProcessTestCase,
):
    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @dist_init
    def test_verify_backend_options(self):
        self.assertEqual(self.rpc_backend, rpc.backend_registry.BackendType.TENSORPIPE)

@unittest.skipIf(
    TEST_WITH_ASAN, "Skip ASAN as torch + multiprocessing spawn have known issues"
)
class TestDdpComparisonTensorPipe(
    TensorPipeRpcAgentTestFixture,
    TestDdpComparison,
    MultiProcessTestCase,
):
    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @dist_init
    def test_verify_backend_options(self):
        self.assertEqual(self.rpc_backend, rpc.backend_registry.BackendType.TENSORPIPE)

if __name__ == "__main__":
    run_tests()
