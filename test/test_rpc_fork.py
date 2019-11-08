#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

import torch

# rpc_fork tests use double as the default dtype
torch.set_default_dtype(torch.double)

from process_group_rpc_test_fixture import ProcessGroupRpcTest
from common_distributed import MultiProcessTestCase
from common_utils import run_tests


class RpcTestWithFork(MultiProcessTestCase, ProcessGroupRpcTest):

    def setUp(self):
        super(RpcTestWithFork, self).setUp()
        self._fork_processes()

if __name__ == '__main__':
    run_tests()
