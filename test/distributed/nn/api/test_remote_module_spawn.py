#!/usr/bin/env python3
import unittest

from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import TEST_WITH_ASAN, run_tests
from torch.testing._internal.distributed.nn.api.remote_module_test import (
    RemoteModuleTest,
)


@unittest.skipIf(
    TEST_WITH_ASAN, "Skip ASAN as torch + multiprocessing spawn have known issues"
)
class RemoteModuleTestWithSpawn(MultiProcessTestCase, RemoteModuleTest):
    def setUp(self):
        super().setUp()
        self._spawn_processes()


if __name__ == "__main__":
    run_tests()
