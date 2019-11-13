#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

from common_distributed import MultiProcessTestCase
from common_utils import run_tests
from distributed.rpc.process_group.dist_optimizer_test import (
    ProcessGroupDistOptimizerTest,
)


class DistOptimizerTestWithFork(MultiProcessTestCase, ProcessGroupDistOptimizerTest):
    def setUp(self):
        super(DistOptimizerTestWithFork, self).setUp()
        self._fork_processes()


if __name__ == "__main__":
    run_tests()
