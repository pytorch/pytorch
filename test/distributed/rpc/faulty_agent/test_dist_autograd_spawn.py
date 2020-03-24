#!/usr/bin/env python3
import unittest

from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import TEST_WITH_ASAN, run_tests
from torch.testing._internal.distributed.rpc.dist_autograd_test import (
    FaultyAgentDistAutogradTest,
)


@unittest.skipIf(
    TEST_WITH_ASAN, "Skip ASAN as torch + multiprocessing spawn have known issues"
)
class FaultyAgentDistAutogradTestWithSpawn(MultiProcessTestCase, FaultyAgentDistAutogradTest):
    def setUp(self):
        super(FaultyAgentDistAutogradTestWithSpawn, self).setUp()
        self._spawn_processes()


if __name__ == "__main__":
    run_tests()
