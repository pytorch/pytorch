import torch
import numpy as np

from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.jit_utils import JitTestCase

class kernel_arena_scope(object):
    def __enter__(self):
        self.scope = torch._C._te.KernelScope()

    def __exit__(self, typ, val, traceback):
        self.scope = None

class TestTensorExprPyBind(JitTestCase):
    def test_simple_sum(self):
        with kernel_arena_scope():
            dtype = torch._C._te.Dtype.Float
            N = 32
            dN = torch._C._te.ExprHandle.int(N)

            A = torch._C._te.Placeholder('A', dtype, [dN])
            B = torch._C._te.Placeholder('B', dtype, [dN])

            def compute(i):
                return A.load([i]) + B.load([i])
            C = torch._C._te.Compute('C', [torch._C._te.DimArg(dN, 'i')], compute)

            loopnest = torch._C._te.LoopNest([C])
            loopnest.prepare_for_codegen()
            stmt = torch._C._te.simplify(loopnest.root_stmt())

            cg = torch._C._te.construct_codegen('ir_eval', stmt, [torch._C._te.BufferArg(x) for x in [A, B, C]])

            tA = torch.rand(N) * 5
            tB = torch.rand(N) * 6
            tC = torch.empty(N)
            cg.call([tA, tB, tC])
            torch.testing.assert_allclose(tA + tB, tC)

    def test_external_calls(self):
        with kernel_arena_scope():
            dtype = torch._C._te.Dtype.Float

            ZERO = torch._C._te.ExprHandle.int(0)
            ONE = torch._C._te.ExprHandle.int(1)
            FOUR = torch._C._te.ExprHandle.int(4)
            A = torch._C._te.BufHandle('A', [ONE, FOUR], dtype)
            B = torch._C._te.BufHandle('B', [FOUR, ONE], dtype)
            C = torch._C._te.BufHandle('C', [ONE, ONE], dtype)

            s = torch._C._te.ExternalCall(C, "nnc_aten_matmul", [A, B], [])

            loopnest = torch._C._te.LoopNest(s, [C])
            loopnest.prepare_for_codegen()
            codegen = torch._C._te.construct_codegen('ir_eval', s, [torch._C._te.BufferArg(x) for x in [A, B, C]])

            tA = torch.ones(1, 4)
            tB = torch.ones(4, 1)
            tC = torch.empty(1, 1)
            codegen.call([tA, tB, tC])
            torch.testing.assert_allclose(torch.matmul(tA, tB), tC)


    def test_kernel(self):
        def f(a, b, c):
            return a + b + c
        device, size = 'cpu', (4, 4)
        x = torch.rand(size, device=device)
        y = torch.rand(size, device=device)
        z = torch.rand(size, device=device)

        torch._C._jit_override_can_fuse_on_cpu(True)
        scripted_f = torch.jit.script(f, (x, y, z))

        scripted_f(x, y, z)
        scripted_f(x, y, z)

        graph = torch.jit.last_executed_optimized_graph()
        node = graph.findNode("prim::TensorExprGroup", True)

        with kernel_arena_scope():
            graph = node.g('Subgraph')
            res1 = torch._C._te.TensorExprKernel.run(graph, (x, y, z))
            res2 = torch._C._te.TensorExprKernel.fallback(graph, (x, y, z))
            correct = f(x, y, z)
            np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
            np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)

if __name__ == '__main__':
    run_tests()
