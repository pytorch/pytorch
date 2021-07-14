import torch
import numpy as np
import torch._C._te as te

from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.jit_utils import JitTestCase
import unittest

LLVM_ENABLED = torch._C._llvm_enabled()


class kernel_arena_scope(object):
    def __enter__(self):
        self.scope = torch._C._te.KernelScope()

    def __exit__(self, typ, val, traceback):
        self.scope = None


def construct_adder(n: int, dtype=te.Dtype.Float):
    dN = te.ExprHandle.int(n)
    A = te.Placeholder('A', dtype, [dN])
    B = te.Placeholder('B', dtype, [dN])

    def compute(i):
        return A.load([i]) + B.load([i])

    C = te.Compute('C', [te.DimArg(dN, 'i')], compute)

    loopnest = te.LoopNest([C])
    loopnest.prepare_for_codegen()
    stmt = te.simplify(loopnest.root_stmt())

    return te.construct_codegen('ir_eval', stmt, [A, B, C])


class TestTensorExprPyBind(JitTestCase):
    def test_simple_sum(self):
        with kernel_arena_scope():
            n = 32
            cg = construct_adder(n)

            tA = torch.randn(n)
            tB = torch.randn(n)
            tC = torch.empty(n)
            cg.call([tA, tB, tC])
            torch.testing.assert_allclose(tA + tB, tC)

    def test_call_raw(self):
        with kernel_arena_scope():
            n = 16
            cg = construct_adder(n, dtype=torch.float64)

            tA = torch.randn(n, dtype=torch.float64)
            tB = torch.randn(n, dtype=torch.float64)
            tC = torch.empty(n, dtype=torch.float64)
            cg.call_raw([tA.data_ptr(), tB.data_ptr(), tC.data_ptr()])
            torch.testing.assert_allclose(tA + tB, tC)

    def test_external_calls(self):
        with kernel_arena_scope():
            dtype = torch.float32

            ONE = te.ExprHandle.int(1)
            FOUR = te.ExprHandle.int(4)
            A = te.BufHandle('A', [ONE, FOUR], dtype)
            B = te.BufHandle('B', [FOUR, ONE], dtype)
            C = te.BufHandle('C', [ONE, ONE], dtype)

            s = te.ExternalCall(C, "nnc_aten_matmul", [A, B], [])

            loopnest = te.LoopNest(s, [C])
            loopnest.prepare_for_codegen()
            codegen = te.construct_codegen('ir_eval', s, [te.BufferArg(x) for x in [A, B, C]])

            tA = torch.ones(1, 4)
            tB = torch.ones(4, 1)
            tC = torch.empty(1, 1)
            codegen.call([tA, tB, tC])
            torch.testing.assert_allclose(torch.matmul(tA, tB), tC)

    def test_dynamic_shape(self):
        with kernel_arena_scope():
            dN = te.VarHandle(torch.int32)
            A = te.BufHandle(torch.float64)
            B = te.BufHandle(torch.float64)

            def compute(i):
                return A.load(i) - B.load(i)

            C = te.Compute('C', [dN], compute)

            loopnest = te.LoopNest([C])
            loopnest.prepare_for_codegen()

            cg = te.construct_codegen(
                'ir_eval',
                loopnest.simplify(),
                [A, B, C, dN])

            def test_with_shape(n):
                tA = torch.randn(n, dtype=torch.double)
                tB = torch.randn(n, dtype=torch.double)
                tC = torch.empty(n, dtype=torch.double)
                cg.call([tA, tB, tC, n])
                torch.testing.assert_allclose(tA - tB, tC)

            test_with_shape(8)
            test_with_shape(31)

    def test_dtype_error(self):
        with kernel_arena_scope():
            one = te.ExprHandle.int(1)
            te.Placeholder([one], torch.float32)  # ok
            te.Placeholder([one])  # ok
            self.assertRaises(TypeError,
                              lambda: te.Placeholder([one], "float55"))

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

        kernel = torch._C._te.TensorExprKernel(graph)
        res1 = kernel.run((x, y, z))
        res2 = kernel.fallback((x, y, z))
        correct = f(x, y, z)
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_shape_prop(self):
        device, size = 'cpu', (4, 4)
        x = torch.rand(size, device=device)
        y = torch.rand(size, device=device)

        graph_str = """
graph(%a : Tensor, %b : Tensor):
  %c : Float(4, 4, strides=[4, 1], requires_grad=0, device=cpu) = aten::mul(%a, %b)
  return (%c)
        """
        graph = torch._C.parse_ir(graph_str)

        exception_thrown = False
        try:
            kernel = torch._C._te.TensorExprKernel(graph)
        except RuntimeError:
            # Graph doesn't have shape info for inputs => compilation should
            # fail
            exception_thrown = True
            pass
        assert exception_thrown

        # Inject shape info and try compiling again
        example_inputs = [torch.rand(4, 4), torch.rand(4, 4)]
        torch._C._te.annotate_input_shapes(graph, example_inputs)

        # TODO: once we have shape propagation as well we should erase type
        # info for %c from the input IR and run shape propagation here - it
        # should be able to reconstruct that info

        # Now compilation should pass
        kernel = torch._C._te.TensorExprKernel(graph)

        res = kernel.run((x, y))
        correct = torch.mul(x, y)
        np.testing.assert_allclose(res.numpy(), correct.numpy(), atol=1e-5)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    @unittest.skip("Does not work until shape propagation is implemented")
    def test_kernel_shape_prop_module(self):
        class TestModule(torch.nn.Module):
            def forward(self, x, y):
                return x * x + y

        graph = torch.jit.script(TestModule()).graph

        # Try compiling the graph as-is. It should fail because it doesn't have
        # shape info.
        exception_thrown = False
        try:
            kernel = torch._C._te.TensorExprKernel(graph)
        except RuntimeError:
            exception_thrown = True
            pass
        assert exception_thrown

        # Try injecting shape info for graph inputs
        example_inputs = [torch.rand(4, 4), torch.rand(4, 4)]

        exception_thrown = False
        try:
            torch._C._te.annotate_input_shapes(graph, example_inputs)
        except RuntimeError:
            # Graph has a 'self' argument for which we can't set shapes
            exception_thrown = True
            pass
        assert exception_thrown

        # Remove 'self' argument and try annotating shapes one more time
        graph = torch._C._te.remove_unused_self_argument(graph)

        # Inject shape info and try compiling again
        torch._C._te.annotate_input_shapes(graph, example_inputs)

        # TODO: once we have shape propagation as well we should erase type
        # info for %c from the input IR and run shape propagation here - it
        # should be able to reconstruct that info

        # Now compilation should pass
        kernel = torch._C._te.TensorExprKernel(graph)

        device, size = 'cpu', (4, 4)
        x = torch.rand(size, device=device)
        y = torch.rand(size, device=device)

        res = kernel.run((x, y))
        correct = TestModule().forward(x, y)
        np.testing.assert_allclose(res.numpy(), correct.numpy(), atol=1e-5)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_with_t(self):
        def f(a):
            return a.t()

        device, size = 'cpu', (3, 4)
        x = torch.rand(size, device=device)

        graph_str = """
graph(%a.1 : Float(3, 4, strides=[4, 1], requires_grad=0, device=cpu)):
  %3 : Float(4, 3, strides=[4, 1], requires_grad=0, device=cpu) = aten::t(%a.1)
  return (%3)
        """
        graph = torch._C.parse_ir(graph_str)

        kernel = torch._C._te.TensorExprKernel(graph)
        res1 = kernel.run((x,))
        res2 = kernel.fallback((x,))
        correct = f(x)
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_with_transpose(self):
        def f(a):
            return a.transpose(-1, -2)

        device, size = 'cpu', (3, 4)
        x = torch.rand(size, device=device)

        graph_str = """
graph(%a.1 : Float(3, 4, strides=[4, 1], requires_grad=0, device=cpu)):
  %2 : int = prim::Constant[value=-1]()
  %3 : int = prim::Constant[value=-2]()
  %4 : Float(4, 3, strides=[4, 1], requires_grad=0, device=cpu) = aten::transpose(%a.1, %2, %3)
  return (%4)
        """
        graph = torch._C.parse_ir(graph_str)

        kernel = torch._C._te.TensorExprKernel(graph)
        res1 = kernel.run((x,))
        res2 = kernel.fallback((x,))
        correct = f(x)
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_with_permute(self):
        def f(a):
            return a.permute([2, 1, 0])

        device, size = 'cpu', (3, 4, 5)
        x = torch.rand(size, device=device)

        graph_str = """
graph(%a.1 : Float(3, 4, 5, strides=[20, 5, 1], requires_grad=0, device=cpu)):
  %1 : int = prim::Constant[value=2]()
  %2 : int = prim::Constant[value=1]()
  %3 : int = prim::Constant[value=0]()
  %4 : int[] = prim::ListConstruct(%1, %2, %3)
  %5 : Float(5, 4, 3, strides=[12, 3, 1], requires_grad=0, device=cpu) = aten::permute(%a.1, %4)
  return (%5)
        """
        graph = torch._C.parse_ir(graph_str)

        kernel = torch._C._te.TensorExprKernel(graph)
        res1 = kernel.run((x,))
        res2 = kernel.fallback((x,))
        correct = f(x)
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_with_custom_lowering(self):
        def f(a):
            return a.nan_to_num()

        device = 'cpu'
        x = torch.ones((2, 2), device=device)
        x[0, 0] = x[1, 1] = torch.nan
        graph_str = """
graph(%x : Float(2, 2, strides=[2, 1], requires_grad=0, device=cpu)):
    %none : NoneType = prim::Constant()
    %y : Float(2, 2, strides=[2, 1], requires_grad=0, device=cpu) = aten::nan_to_num(%x, %none, %none, %none)
    return (%y)
        """
        graph = torch._C.parse_ir(graph_str)

        def my_custom_lowering(inputs, out_shape, out_type, device):
            def get_dim_args(dims):
                dim_args = []
                for dim in dims:
                    dim_args.append(te.DimArg(dim, 'i' + str(len(dim_args))))
                return dim_args

            def compute(idxs):
                load = inputs[0].as_buf().load(idxs)
                return te.ifThenElse(te.ExprHandle.isnan(load), te.ExprHandle.float(0.), load)
            return te.Compute2("custom_nan_to_num", get_dim_args(out_shape), compute)

        kernel = torch._C._te.TensorExprKernel(graph, {'aten::nan_to_num' : my_custom_lowering})
        res1 = kernel.run((x,))
        res2 = kernel.fallback((x,))
        correct = f(x)
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_with_expand(self):
        def f(a):
            return a.expand((2, 3, 4))

        device = 'cpu'
        x = torch.rand((1, 3, 1), device=device)
        graph_str = """
graph(%a : Float(1, 3, 1, strides=[3, 1, 1], requires_grad=0, device=cpu)):
  %1 : int = prim::Constant[value=2]()
  %2 : int = prim::Constant[value=3]()
  %3 : int = prim::Constant[value=4]()
  %4 : int[] = prim::ListConstruct(%1, %2, %3)
  %5 : bool = prim::Constant[value=0]()
  %6 : Float(2, 3, 4, strides=[12, 4, 0], requires_grad=0, device=cpu) = aten::expand(%a, %4, %5)
  return (%6)
        """
        graph = torch._C.parse_ir(graph_str)

        kernel = torch._C._te.TensorExprKernel(graph)
        res1 = kernel.run((x,))
        res2 = kernel.fallback((x,))
        correct = f(x)
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)

    def test_forgot_kernel_arena(self):
        self.assertRaises(RuntimeError, lambda: torch._C._te.VarHandle("n", torch._C._te.Dtype.Int))


if __name__ == '__main__':
    run_tests()
