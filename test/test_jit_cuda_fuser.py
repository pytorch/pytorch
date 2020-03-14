from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import FileCheck

from torch.testing._internal.common_utils import run_tests, ProfilingMode, GRAPH_EXECUTOR, disable_legacy_fuser

from test_jit import JitTestCase, RUN_CUDA

if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(True)

torch._C._jit_register_cuda_fuser()

class TestCudaFuser(JitTestCase):
    def _has_cuda_fusion_group(self, graph):
        has_cuda_fusion_group = False
        for n in graph.nodes():
            if n.kind() == 'prim::CudaFusionGroup':
                has_cuda_fusion_group = True
        return has_cuda_fusion_group

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING, "Requires profiling node to run cuda fuser")
    def test_const(self):
        with disable_legacy_fuser():
            def t(x, y):
                o = x + y
                o = o + 2.0
                return o
            t_jit = torch.jit.script(t)
            x = torch.randn(4, 8, dtype=torch.float, device="cuda")
            y = torch.randn(4, 8, dtype=torch.float, device="cuda")
            jit_o = t_jit(x, y)
            jit_o = t_jit(x, y)
            o = t(x, y)
            self.assertEqual(o, jit_o)
            self.assertTrue(self._has_cuda_fusion_group(t_jit.graph_for(x, y)))

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING, "Requires profiling node to run cuda fuser")
    def test_chunk(self):
        with disable_legacy_fuser():
            def t(x, y, z, q):
                o = x + q
                x0, x1 = torch.chunk(o, 2)
                o = x0 + x1
                o = o + y
                o = o * z
                o = torch.relu(o)
                return o
            t_jit = torch.jit.script(t)
            x = torch.randn(4, 8, dtype=torch.float, device="cuda")
            y = torch.randn(2, 8, dtype=torch.float, device="cuda")
            z = torch.randn(2, 8, dtype=torch.float, device="cuda")
            q = torch.randn(4, 8, dtype=torch.float, device="cuda")
            jit_o = t_jit(x, y, z, q)
            jit_o = t_jit(x, y, z, q)
            o = t(x, y, z, q)
            self.assertEqual(o, jit_o)
            self.assertTrue(self._has_cuda_fusion_group(t_jit.graph_for(x, y, z, q)))

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING, "Requires profiling node to run cuda fuser")
    def test_scalar_input(self):
        with disable_legacy_fuser():
            def t(x : torch.Tensor, y : torch.Tensor, z : float):
                o = x + y
                o = o + z
                return o
            t_jit = torch.jit.script(t)
            x = torch.randn(4, 8, dtype=torch.float, device="cuda")
            y = torch.randn(4, 8, dtype=torch.float, device="cuda")
            jit_o = t_jit(x, y, 2.0)
            jit_o = t_jit(x, y, 2.0)
            o = t(x, y, 2.0)
            self.assertEqual(o, jit_o)
            self.assertTrue(self._has_cuda_fusion_group(t_jit.graph_for(x, y, 2.0)))
    

if __name__ == '__main__':
    run_tests()
