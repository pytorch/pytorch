from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch

from torch.testing._internal.common_utils import run_tests

import test_jit_fuser

class TestFuserLegacy(test_jit_fuser.TestFuser):
    def setUp(self):
        self.old_prof_exec_state = torch._C._jit_set_profiling_executor(False)

    def tearDown(self):
        torch._C._jit_set_profiling_executor(self.old_prof_exec_state)

if __name__ == '__main__':
    run_tests()
