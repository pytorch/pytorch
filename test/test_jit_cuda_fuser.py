from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import os

import torch

from torch.testing._internal.common_utils import run_tests, ProfilingMode, GRAPH_EXECUTOR, skipIfRocm
from torch.testing._internal.codegen.random_topo_test import runDefaultTestWithSeed

from test_jit import JitTestCase, RUN_CUDA


if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(True)

class TestCudaFuser(JitTestCase):

    def setUp(self):
        super(TestCudaFuser, self).setUp()
        self.old_cpu_fuse = torch._C._jit_can_fuse_on_cpu()
        self.old_gpu_fuse = torch._C._jit_can_fuse_on_gpu()
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)

        if(RUN_CUDA):
            torch._C._jit_register_cuda_fuser()

    def tearDown(self):
        if(RUN_CUDA):
            torch._C._jit_clear_cuda_fuser()
        torch._C._jit_override_can_fuse_on_cpu(self.old_cpu_fuse)
        torch._C._jit_override_can_fuse_on_gpu(self.old_gpu_fuse)
        super(TestCudaFuser, self).tearDown()

    def _run_helper(self, jit_op, op, should_fuse, *args):
        torch.cuda.manual_seed_all(123)
        jit_o = jit_op(*args)
        torch.cuda.manual_seed_all(123)
        jit_o = jit_op(*args)
        torch.cuda.manual_seed_all(123)
        o = op(*args)
        self.assertEqual(o, jit_o)
        self.assertTrue(self._has_cuda_fusion_group(jit_op.graph_for(*args)) == should_fuse)

    def _has_cuda_fusion_group(self, graph):
        has_cuda_fusion_group = False
        for n in graph.nodes():
            if n.kind() == 'prim::CudaFusionGroup':
                has_cuda_fusion_group = True
        return has_cuda_fusion_group

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING, "Requires profiling node to run cuda fuser")
    @skipIfRocm
    def test_half(self):
        def t(x : torch.Tensor, y : torch.Tensor, z : torch.Tensor, alpha : float):
            o_16 = torch.add(x, y)
            o_32_a = torch.add(y, z, alpha=alpha)
            o_32_b = torch.add(o_16, z)
            return (o_16, o_32_a, o_32_b)

        t_jit = torch.jit.script(t)
        alpha = 0.5
        # stick to integers, this avoid the numerical difference due to our
        # promotion
        x = torch.randint(0, 256, (4, 8)).to(dtype=torch.float16, device="cuda")
        y = torch.randint(0, 256, (4, 8)).to(dtype=torch.float16, device="cuda")
        z = torch.randint(0, 256, (4, 8)).to(dtype=torch.float16, device="cuda")
        jit_o = t_jit(x, y, z, alpha)
        jit_o = t_jit(x, y, z, alpha)
        o = t(x, y, z, alpha)
        for oo, jit_oo in zip(o, jit_o):
            self.assertEqual(oo.dtype, jit_oo.dtype)
            self.assertEqual(oo, jit_oo)
        self.assertTrue(self._has_cuda_fusion_group(t_jit.graph_for(x, y, z, alpha)))

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING, "Requires profiling node to run cuda fuser")
    @skipIfRocm
    def test_const(self):
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
    @skipIfRocm
    def test_chunk(self):
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
    @skipIfRocm
    def test_scalar_input(self):
        def t(x : torch.Tensor, y : torch.Tensor, z : float):
            o = x + y
            o = o + z
            return o
        t_jit = torch.jit.script(t)
        x = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        y = torch.randn(4, 8, 1, 32, dtype=torch.float, device="cuda")
        y = y.expand(4, 8, 32, 32)
        jit_o = t_jit(x, y, 2.0)
        jit_o = t_jit(x, y, 2.0)
        o = t(x, y, 2.0)
        self.assertEqual(o, jit_o)
        self.assertTrue(self._has_cuda_fusion_group(t_jit.graph_for(x, y, 2.0)))

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING, "Requires profiling node to run cuda fuser")
    @skipIfRocm
    def test_broadcasting(self):
        def t(x : torch.Tensor, y : torch.Tensor, z : float):
            o = x + y
            o = o + z
            return o
        t_jit = torch.jit.script(t)
        x = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        y = torch.randn(32, 32, dtype=torch.float, device="cuda")
        jit_o = t_jit(x, y, 2.0)
        jit_o = t_jit(x, y, 2.0)
        o = t(x, y, 2.0)
        self.assertEqual(o, jit_o)
        self.assertTrue(self._has_cuda_fusion_group(t_jit.graph_for(x, y, 2.0)))

    @unittest.skipIf(True, "real broadcast with different output not supported yet")
    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING, "Requires profiling node to run cuda fuser")
    @skipIfRocm
    def test_broadcasting_multiple_output_shape(self):
        def t(x : torch.Tensor, y : torch.Tensor, z : torch.Tensor):
            o = x + 12
            o1 = o + y
            o2 = o + z
            oo = o1.sum() + o2.sum()
            return oo
        t_jit = torch.jit.script(t)
        x = torch.randn(32, 32, dtype=torch.float, device="cuda")
        y = torch.randn(2, 32, 32, dtype=torch.float, device="cuda")
        z = torch.randn(4, 32, 32, dtype=torch.float, device="cuda")
        jit_o = t_jit(x, y, z)
        jit_o = t_jit(x, y, z)
        o = t(x, y, z)
        self.assertEqual(o, jit_o)
        # Currently cannot fuse this
        self.assertTrue(self._has_cuda_fusion_group(t_jit.graph_for(x, y, z)))

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING, "Requires profiling node to run cuda fuser")
    @skipIfRocm
    def test_broadcasting_multiple_output(self):
        def t(x : torch.Tensor, y : torch.Tensor, z : torch.Tensor):
            o = x + 12
            o1 = o + y
            o2 = o + z
            oo = o1.sum() + o2.sum()
            return oo
        t_jit = torch.jit.script(t)
        x = torch.randn(32, 32, dtype=torch.float, device="cuda")
        y = torch.randn(4, 32, 32, dtype=torch.float, device="cuda")
        z = torch.randn(4, 32, 32, dtype=torch.float, device="cuda")
        jit_o = t_jit(x, y, z)
        jit_o = t_jit(x, y, z)
        o = t(x, y, z)
        self.assertEqual(o, jit_o)
        # Currently cannot fuse this
        self.assertTrue(self._has_cuda_fusion_group(t_jit.graph_for(x, y, z)))

    def _binary_test_helper(self, operation):
        def t(x : torch.Tensor, y: torch.Tensor, z : float):
            o = x + z
            o = operation(o, y)
            return o
        t_jit = torch.jit.script(t)
        x = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        y = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        jit_o = t_jit(x, y, 2.0)
        jit_o = t_jit(x, y, 2.0)
        o = t(x, y, 2.0)
        self.assertEqual(o, jit_o)
        self.assertTrue(self._has_cuda_fusion_group(t_jit.graph_for(x, y, 2.0)))

    def _unary_test_helper(self, operation):
        def t(x : torch.Tensor, z : float):
            o = x + z
            o = operation(o)
            return o
        t_jit = torch.jit.script(t)
        x = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        jit_o = t_jit(x, 2.0)
        jit_o = t_jit(x, 2.0)
        o = t(x, 2.0)
        self.assertEqual(o, jit_o)
        self.assertTrue(self._has_cuda_fusion_group(t_jit.graph_for(x, 2.0)))

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING, "Requires profiling node to run cuda fuser")
    @skipIfRocm
    def test_unary_ops(self):
        operations = [torch.neg,
                      torch.abs,
                      torch.log,
                      torch.log10,
                      torch.log1p,
                      torch.log2,
                      torch.lgamma,
                      torch.exp,
                      torch.expm1,
                      torch.erf,
                      torch.erfc,
                      torch.cos,
                      torch.acos,
                      torch.cosh,
                      torch.sin,
                      torch.asin,
                      torch.tan,
                      torch.atan,
                      torch.sqrt,
                      torch.rsqrt,
                      torch.ceil,
                      torch.floor,
                      torch.round,
                      torch.trunc,
                      torch.frac,
                      torch.reciprocal,
                      torch.relu,
                      torch.sigmoid,
                      torch.tanh,
                      torch.nn.functional.gelu]
        for op in operations:
            self._unary_test_helper(op)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING, "Requires profiling node to run cuda fuser")
    @skipIfRocm
    def test_binary_ops(self):
        operations = [torch.div,
                      torch.mul,
                      torch.atan2,
                      torch.max,
                      torch.min,
                      torch.pow,
                      torch.remainder,
                      torch.fmod,
                      torch.eq,
                      torch.ne,
                      torch.ge,
                      torch.gt,
                      torch.le,
                      torch.lt]
        for op in operations:
            self._binary_test_helper(op)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING, "Requires profiling node to run cuda fuser")
    @skipIfRocm
    def test_ternary_ops(self):
        x = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        y = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        cond = torch.randint(0, 2, (4, 8, 32, 32)).to(dtype=torch.bool, device="cuda")

        def add(x : torch.Tensor, other : torch.Tensor, alpha : float):
            o = torch.relu(x)
            o = torch.add(o, other=other, alpha=alpha)
            return o
        add_jit = torch.jit.script(add)
        self._run_helper(add_jit, add, True, x, y, 2.0)

        def clamp0(x : torch.Tensor, f : float):
            o = torch.rand_like(x)
            o = o * torch.clamp(x, min=f)
            return o
        clamp0_jit = torch.jit.script(clamp0)
        self._run_helper(clamp0_jit, clamp0, True, x, 0.5)

        def clamp1(x : torch.Tensor, f : float, ff : float):
            o = torch.rand_like(x)
            o = o * torch.clamp(x, min=f, max=ff)
            return o
        clamp1_jit = torch.jit.script(clamp1)
        self._run_helper(clamp1_jit, clamp1, True, x, -0.2, 0.7)

        def threshold(x : torch.Tensor, th : float, val : float):
            o = torch.rand_like(x)
            o = x * torch.threshold(o, th, val)
            return o
        threshold_jit = torch.jit.script(threshold)
        self._run_helper(threshold_jit, threshold, True, x, 0.2, 0.9)

        def where(x : torch.Tensor, y : torch.Tensor, cond : torch.Tensor):
            o = torch.rand_like(x)
            o = o * torch.where(cond, x, y)
            return o
        where_jit = torch.jit.script(where)
        self._run_helper(where_jit, where, True, x, y, cond)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, "Requires profiling node to run cuda fuser")
    @skipIfRocm
    def test_dynamic_size(self):
        def t(x : torch.Tensor, y : torch.Tensor, z : float):
            o = x + y
            o = o + z
            return o
        t_jit = torch.jit.script(t)
        x = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        y = torch.randn(32, 32, dtype=torch.float, device="cuda")
        jit_o = t_jit(x, y, 2.0)
        jit_o = t_jit(x, y, 2.0)
        o = t(x, y, 2.0)
        self.assertEqual(o, jit_o)
        self.assertTrue(self._has_cuda_fusion_group(t_jit.graph_for(x, y, 2.0)))
        x = torch.randn(8, 32, 16, 8, dtype=torch.float, device="cuda")
        y = torch.randn(16, 8, dtype=torch.float, device="cuda")
        jit_o = t_jit(x, y, 2.0)
        o = t(x, y, 2.0)
        self.assertEqual(o, jit_o)
        x = torch.randn(8, 17, 8, dtype=torch.float, device="cuda")
        y = torch.randn(8, 17, 1, dtype=torch.float, device="cuda")
        jit_o = t_jit(x, y, 2.0)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @skipIfRocm
    def test_random_topo(self):
        os.environ["PYTORCH_CUDA_FUSER_DISABLE_FALLBACK"] = "1"
        self.assertTrue(runDefaultTestWithSeed(28449))

if __name__ == '__main__':
    run_tests()
