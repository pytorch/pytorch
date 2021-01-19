import torch
import unittest

from torch.testing._internal.jit_utils import JitTestCase

class kernel_arena_scope(object):
    def __enter__(self):
        self.scope = torch._C.te.KernelScope()

    def __exit__(self, typ, val, traceback):
        self.scope = None

class TestTensorExprPyBind(JitTestCase):
    def test_simple_sum(self):
        with kernel_arena_scope():
            dtype = torch._C.te.Dtype.Float
            N = 32
            dN = torch._C.te.ExprHandle.int(N)

            A = torch._C.te.Placeholder('A', dtype, [dN])
            B = torch._C.te.Placeholder('B', dtype, [dN])

            def compute(i):
                return A.load([i]) + B.load([i])
            C = torch._C.te.Compute('C', [torch._C.te.DimArg(dN, 'i')], compute)

            loopnest = torch._C.te.LoopNest([C])
            loopnest.prepare_for_codegen()
            stmt = torch._C.te.simplify(loopnest.root_stmt())

            cg = torch._C.te.construct_codegen('ir_eval', stmt, [torch._C.te.BufferArg(x) for x in [A, B, C]])

            tA = torch.rand(N) * 5
            tB = torch.rand(N) * 6
            tC = torch.empty(N)
            cg.call([tA, tB, tC])
            torch.testing.assert_allclose(tA + tB, tC)

if __name__ == '__main__':
    unittest.main()
