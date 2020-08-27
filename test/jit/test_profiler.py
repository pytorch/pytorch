import os
import sys

import torch

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase, warmup_backward

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

class TestProfiler(JitTestCase):
    def setUp(self):
        self.prev_exec = torch._C._jit_set_profiling_executor(True)
        self.prev_profiling = torch._C._jit_set_profiling_mode(True)
        self.inline_autodiff = torch._C._debug_set_autodiff_subgraph_inlining(False)

    def tearDown(self):
        torch._C._jit_set_profiling_executor(self.prev_exec)
        torch._C._jit_set_profiling_mode(self.prev_profiling)
        torch._C._debug_set_autodiff_subgraph_inlining(self.inline_autodiff)

    def test_specialize_backward(self):
        @torch.jit.script
        def test_fuse(a, b):
            c = a * b
            d = c * b
            return d

        x = torch.ones(1, requires_grad=True)
        y = torch.ones(1, requires_grad=True)
        test_fuse(x, y)
        b = test_fuse(x, y)
        warmup_backward(b)
        g = torch.jit.last_executed_optimized_graph()
        # Backward has an if node guarding specializations,
        # within the if node true block there are no if statements
        optimized_block = next(g.findNode("prim::If").blocks())
        self.assertIsNone(optimized_block.findNode("prim::If"))
