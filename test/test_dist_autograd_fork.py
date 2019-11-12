#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

import torch

# dist_autograd_fork tests use double as the default dtype
torch.set_default_dtype(torch.double)

import process_group_dist_autograd_test
from common_distributed import MultiProcessTestCase
from common_utils import run_tests


class ProcessGroupDistAutogradTestWithFork(
    MultiProcessTestCase, process_group_dist_autograd_test.ProcessGroupDistAutogradTest
):
    def setUp(self):
        super(ProcessGroupDistAutogradTestWithFork, self).setUp()
        self._fork_processes()

if __name__ == '__main__':
    run_tests()
