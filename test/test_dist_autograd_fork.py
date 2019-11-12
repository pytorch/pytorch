#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

import process_group_dist_autograd_test
import torch
from common_distributed import MultiProcessTestCase
from common_utils import run_tests


# dist_autograd_fork tests use double as the default dtype
torch.set_default_dtype(torch.double)


class DistAutogradTestWithFork(
    MultiProcessTestCase, process_group_dist_autograd_test.ProcessGroupDistAutogradTest
):
    def setUp(self):
        super(DistAutogradTestWithFork, self).setUp()
        self._fork_processes()


if __name__ == "__main__":
    run_tests()
