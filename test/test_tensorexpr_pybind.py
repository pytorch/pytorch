import torch
import torch.nn.functional as F
import numpy as np

from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.jit_utils import JitTestCase
import unittest

LLVM_ENABLED = torch._C._llvm_enabled()

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

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_conv_with_compute(self):
        with kernel_arena_scope():
            N, C, H, W = 1, 8, 6, 6 # Image: batch size = N, channel num = C, height = H, width = W
            K, R, S = 8, 3, 3 # kernel: output channel num = K, kernel height = R, Kernel width = S
            Pad = 1 # Padding

            OH, OW = H - R + 1, W - S + 1

            (NN, KK, HH, WW) = [torch._C._te.ExprHandle.int(x) for x in [N, K, OH, OW]] # result tensor shape
            (CC, RR, SS) = [torch._C._te.ExprHandle.int(x) for x in [C, R, S]]
            (HI, WI) = [torch._C._te.ExprHandle.int(x) for x in [H, W]]
            dtype = torch._C._te.Dtype.Float

            def get_dim_args(dims):
                dim_args = []
                for dim in dims:
                    dim_args.append(torch._C._te.DimArg(dim, 'i' + str(len(dim_args))))
                return dim_args

            Pimage = torch._C._te.Placeholder('image', dtype, [NN, CC, HI, WI])
            Pweight = torch._C._te.Placeholder('weight', dtype, [KK, CC, RR, SS])

            def compute(dims):
                n, k, h, w, c, r, s = dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6]
                return Pimage.load([n, c, h+r, w+s]) * Pweight.load([k, c, r, s])

            OUT = torch._C._te.Reduce('conv',
                    get_dim_args([NN, KK, HH, WW]),
                    torch._C._te.Sum(),
                    compute,
                    get_dim_args([CC, RR, SS]),
                    )

            loopnest = torch._C._te.LoopNest([OUT])
            loops = loopnest.get_loops_for(OUT)
            loopnest.flatten(loops)
            loopnest.prepare_for_codegen()
            stmt = torch._C._te.simplify(loopnest.root_stmt())
            codegen = torch._C._te.construct_codegen('llvm', stmt, [torch._C._te.BufferArg(x) for x in [Pimage, Pweight, OUT]])

            image = torch.ones(N, C, H, W)
            weight = torch.ones(K, C, R, S)
            ref= F.conv2d(image, weight)
            out = torch.zeros_like(ref)
            codegen.call([image.float(), weight.float(), out.float()])
            torch.testing.assert_allclose(out, ref)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_conv_with_loops(self):
        def loop(v, bound, body):
            return torch._C._te.For.make(v, torch._C._te.ExprHandle.int(0), bound, body)
        with kernel_arena_scope():
            N, C, H, W = 1, 8, 6, 6 # Image: batch size = N, channel num = C, height = H, width = W
            K, R, S = 8, 3, 3 # kernel: output channel num = K, kernel height = R, Kernel width = S
            Pad = 1 # Padding

            OH, OW = H - R + 1, W - S + 1

            (NN, KK, HH, WW) = [torch._C._te.ExprHandle.int(x) for x in [N, K, OH, OW]] # result tensor shape
            (CC, RR, SS) = [torch._C._te.ExprHandle.int(x) for x in [C, R, S]]
            (HI, WI) = [torch._C._te.ExprHandle.int(x) for x in [H, W]]
            dtype = torch._C._te.Dtype.Float

            def get_dim_args(dims):
                dim_args = []
                for dim in dims:
                    dim_args.append(torch._C._te.DimArg(dim, 'i' + str(len(dim_args))))
                return dim_args

            Pimage = torch._C._te.Placeholder('image', dtype, [NN, CC, HI, WI])
            Pweight = torch._C._te.Placeholder('weight', dtype, [KK, CC, RR, SS])
            Pout = torch._C._te.Placeholder('conv', dtype, [NN, KK, HH, WW])

            dint = torch._C._te.Dtype.Int
            n, k, h, w, c, r, s = [torch._C._te.VarHandle(s, dint) for s in ["n", "k", "h", "w", "c", "r", "s"]]
            stmt = loop(n, NN,
                    loop(k, KK,
                        loop(h, HH,
                            loop(w, WW,
                                loop(c, CC,
                                    loop(r, RR,
                                        loop(s, SS,
                                            Pout.store(
                                                [n, k, h, w],
                                                Pout.load([n, k, h, w]) +
                                                Pimage.load([n, c, h+r, w+s]) *
                                                Pweight.load([k, c, r, s])
                                                )))))))
                    )

            loopnest = torch._C._te.LoopNest(stmt, [Pout.buf()])
            loopnest.prepare_for_codegen()
            stmt = torch._C._te.simplify(loopnest.root_stmt())
            codegen = torch._C._te.construct_codegen('llvm', stmt, [torch._C._te.BufferArg(x) for x in [Pimage, Pweight, Pout]])

            image = torch.ones(N, C, H, W)
            weight = torch.ones(K, C, R, S)
            ref= F.conv2d(image, weight)
            out = torch.zeros_like(ref)
            codegen.call([image.float(), weight.float(), out.float()])
            torch.testing.assert_allclose(out, ref)

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

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_with_tensor_inputs(self):
        def f(a, b, c):
            return a + b + c
        device, size = 'cpu', (4, 4)
        x = torch.rand(size, device=device)
        y = torch.rand(size, device=device)
        z = torch.rand(size, device=device)

        graph_str = """
graph(%a.1 : Float(4, 4, strides=[4, 1], requires_grad=0, device=cpu),
      %b.1 : Float(4, 4, strides=[4, 1], requires_grad=0, device=cpu),
      %c.1 : Float(4, 4, strides=[4, 1], requires_grad=0, device=cpu)):
  %6 : int = prim::Constant[value=1]()
  %7 : Float(4, 4, strides=[4, 1], requires_grad=0, device=cpu) = aten::add(%a.1, %b.1, %6)
  %3 : Float(4, 4, strides=[4, 1], requires_grad=0, device=cpu) = aten::add(%7, %c.1, %6)
  return (%3)
        """
        graph = torch._C.parse_ir(graph_str)

        with kernel_arena_scope():
            kernel = torch._C._te.TensorExprKernel(graph)
            res1 = kernel.run((x, y, z))
            res2 = kernel.fallback((x, y, z))
            correct = f(x, y, z)
            np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
            np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)


    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_with_scalar_inputs(self):
        def f(a, b, c):
            return a + b + c
        x = torch.tensor(0.1, dtype=torch.float, device='cpu')
        y = torch.tensor(0.6, dtype=torch.float, device='cpu')
        z = torch.tensor(0.7, dtype=torch.float, device='cpu')

        graph_str = """
graph(%a.1 : Float(requires_grad=0, device=cpu),
      %b.1 : Float(requires_grad=0, device=cpu),
      %c.1 : Float(requires_grad=0, device=cpu)):
  %3 : int = prim::Constant[value=1]()
  %6 : Float(requires_grad=0, device=cpu) = aten::add(%a.1, %b.1, %3)
  %9 : Float(requires_grad=0, device=cpu) = aten::add(%6, %c.1, %3)
  return (%9)
        """
        graph = torch._C.parse_ir(graph_str)

        with kernel_arena_scope():
            kernel = torch._C._te.TensorExprKernel(graph)
            res1 = kernel.run((x, y, z))
            res2 = kernel.fallback((x, y, z))
            correct = f(x, y, z)
            np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
            np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)

if __name__ == '__main__':
    run_tests()
