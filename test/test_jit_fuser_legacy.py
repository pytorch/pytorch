from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import torch

from torch.testing._internal.common_utils import run_tests, ProfilingMode, graph_executor_mode

import test_jit_fuser

class TestFuserLegacy(test_jit_fuser.TestFuser):
    def setUp(self):
        super(test_jit_fuser.TestFuser, self).setUp()
        self.old_prof_exec_state = torch._C._jit_set_profiling_executor(False)
    def tearDown(self):
        super(test_jit_fuser.TestFuser, self).tearDown()
        torch._C._jit_set_profiling_executor(self.old_prof_exec_state)
    def test_graph_executor_mode(self):
        assert(graph_executor_mode() == ProfilingMode.LEGACY)

if __name__ == '__main__':
    run_tests()
