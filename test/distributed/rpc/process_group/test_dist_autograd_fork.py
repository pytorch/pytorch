#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from common_distributed import MultiProcessTestCase
from common_utils import run_tests
from distributed.rpc.process_group.dist_autograd_test import (
    ProcessGroupDistAutogradTest,
)


# dist_autograd_fork tests use double as the default dtype
torch.set_default_dtype(torch.double)


class ProcessGroupDistAutogradTestWithFork(
    MultiProcessTestCase, ProcessGroupDistAutogradTest
):
    def setUp(self):
        super(ProcessGroupDistAutogradTestWithFork, self).setUp()
        self._fork_processes()


if __name__ == "__main__":
    run_tests()
