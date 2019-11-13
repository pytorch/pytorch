#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from common_distributed import MultiProcessTestCase
from common_utils import run_tests
from distributed.rpc.process_group.rpc_test import ProcessGroupRpcTest


# rpc_fork tests use double as the default dtype
torch.set_default_dtype(torch.double)


class ProcessGroupRpcTestWithFork(MultiProcessTestCase, ProcessGroupRpcTest):
    def setUp(self):
        super(ProcessGroupRpcTestWithFork, self).setUp()
        self._fork_processes()


if __name__ == "__main__":
    run_tests()
