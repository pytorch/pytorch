from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import FileCheck

from torch.testing._internal.common_utils import run_tests, IS_SANDCASTLE, ProfilingMode, GRAPH_EXECUTOR, \
    enable_profiling_mode, graph_executor_mode
from textwrap import dedent
from itertools import product, permutations

from test_jit import JitTestCase, enable_cpu_fuser, RUN_CUDA, RUN_CUDA_HALF, RUN_CUDA_MULTI_GPU, \
    backward_graph, all_backward_graphs, get_lstm_inputs, get_milstm_inputs, \
    LSTMCellC, LSTMCellF, LSTMCellS, MiLSTMCell, _inline_everything
import test_jit_fuser

class TestFuserProfiling(test_jit_fuser.TestFuser):
    def setUp(self):
        super(test_jit_fuser.TestFuser, self).setUp()
        self.old_prof_exec_state = torch._C._jit_set_profiling_executor(True)
        self.old_prof_mode_state = torch._C._jit_set_profiling_mode(True)

    def tearDown(self):
        super(test_jit_fuser.TestFuser, self).tearDown()
        torch._C._jit_set_profiling_executor(self.old_prof_exec_state)
        torch._C._jit_set_profiling_mode(self.old_prof_mode_state)
    def test_graph_executor_mode(self):
        assert(graph_executor_mode() == ProfilingMode.PROFILING)


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
