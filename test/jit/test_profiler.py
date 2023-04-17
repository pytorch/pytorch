# Owner(s): ["oncall: jit"]

import os
import sys

import torch
from torch.testing._internal.common_utils import skipIfTorchDynamo

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase, warmup_backward, FileCheck

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

@skipIfTorchDynamo()
class TestProfiler(JitTestCase):
    def setUp(self):
        self.prev_exec = torch._C._jit_set_profiling_executor(True)
        self.prev_profiling = torch._C._get_graph_executor_optimize(True)
        self.inline_autodiff = torch._C._debug_set_autodiff_subgraph_inlining(False)
        self.texpr_fuser_state = torch._C._jit_texpr_fuser_enabled()
        self.can_fuse_on_cpu = torch._C._jit_can_fuse_on_cpu()
        torch._C._jit_set_texpr_fuser_enabled(True)
        torch._C._jit_override_can_fuse_on_cpu(True)
        self.default_dtype = torch.get_default_dtype()
        self.old_reduction_enabled = torch._C._jit_set_texpr_reductions_enabled(True)
        torch.set_default_dtype(torch.double)
        self.old_fusion_inlining = torch._C._debug_get_fusion_group_inlining()
        torch._C._debug_set_fusion_group_inlining(False)
        self.old_te_must_use_llvm_cpu = torch._C._jit_get_te_must_use_llvm_cpu()
        torch._C._jit_set_te_must_use_llvm_cpu(False)

    def tearDown(self):
        torch._C._jit_set_profiling_executor(self.prev_exec)
        torch._C._get_graph_executor_optimize(self.prev_profiling)
        torch._C._debug_set_autodiff_subgraph_inlining(self.inline_autodiff)
        torch._C._jit_set_texpr_fuser_enabled(self.texpr_fuser_state)
        torch._C._jit_override_can_fuse_on_cpu(self.can_fuse_on_cpu)
        torch.set_default_dtype(self.default_dtype)
        torch._C._jit_set_texpr_reductions_enabled(self.old_reduction_enabled)
        torch._C._debug_set_fusion_group_inlining(self.old_fusion_inlining)
        torch._C._jit_set_te_must_use_llvm_cpu(self.old_te_must_use_llvm_cpu)

    def test_tensor_type_not_determined_by_inputs(self):
        @torch.jit.script
        def scalar_type_input(x, y, z):
            return x + y + 4 + z.item()

        x = torch.tensor([2, 2])
        scalar_type_input(x, x, torch.tensor(1))
        scalar_type_input(x, x, torch.tensor(1))
        scalar_type_input(x, x, torch.tensor(1.0))
        g = torch.jit.last_executed_optimized_graph()

        # item & add should not get pulled into the fusion group -
        # we expect to see Fusion Group (item / add) Fusion Group in ir dump
        FileCheck().check("TensorExpr").check("Scalar = aten::item").check_next("Tensor = aten::add").check("TensorExpr").run(g)


        @torch.jit.script
        def non_const_dtype(x, y, cond: bool):
            dtype = torch.int16 if cond else torch.int32
            return (x + y + 3).sum(dtype=dtype)

        non_const_dtype(x, x, True)
        non_const_dtype(x, x, True)
        g = torch.jit.last_executed_optimized_graph()
        # because dtype is non-const, sum should not get pulled into the Fusion Group
        FileCheck().check("TensorExpr").check("TensorExpr").check_not("aten::sum").run(g)

    def test_specialize_backward(self):
        def test_fuse(a, b):
            c = a * b
            d = c * b
            return d

        test_fuse.__disable_jit_function_caching__ = True

        scripted_f = torch.jit.script(test_fuse)
        x = torch.ones(1, requires_grad=True)
        y = torch.ones(1, requires_grad=True)
        scripted_f(x, y)
        b = scripted_f(x, y)
        warmup_backward(b)
        g = torch.jit.last_executed_optimized_graph()
        # Backward has an if node guarding specializations,
        # within the if node true block there is only one if node
        # that guards a tensorexpr group
        optimized_block = next(g.findNode("prim::If").blocks())
        if_nodes = list(optimized_block.findAllNodes("prim::If"))

        self.assertEqual(len(if_nodes), 1)
        FileCheck().check("Group[Subgraph").run(str(if_nodes[0]))
        # no broadcasts occurred, sum_to_size have been specialized out
        self.assertIsNone(optimized_block.findNode("aten::_grad_sum_to_size"))

        broadcast_f = torch.jit.script(test_fuse)
        x = torch.ones([2, 2], requires_grad=True)
        y = torch.ones([1], requires_grad=True)
        broadcast_f(x, y)
        b = broadcast_f(x, y)
        b.backward(torch.ones([2, 2], dtype=torch.float), retain_graph=True)
        b.backward(torch.ones([2, 2], dtype=torch.float))
        # warmup_backward(b, torch.ones([2, 2], dtype=torch.float))
        g = torch.jit.last_executed_optimized_graph()
        optimized_block = next(g.findNode("prim::If").blocks())
        # broadcasts occurred, currently expect to see aten::_grad_sum_to_size
        self.assertIsNotNone(optimized_block.findNode("aten::_grad_sum_to_size"))

    def test_specialized_types(self):
        @torch.jit.script
        def test_fuse(a, b):
            c = a * b
            d = c * b
            return d

        x = torch.tensor([.5])
        for _ in range(3):
            test_fuse(x, x)

        g = torch.jit.last_executed_optimized_graph()
        # Types should remain specialized for typecheck outputs & fusion outputs
        FileCheck().check("Double(").check_same("prim::TypeCheck").check_same("\n").check("Double").check_same("TensorExpr").run(g)

        # other outputs should not be specialized
        FileCheck().check("Tensor = prim::If").run(g)

    def test_aliasing_merge(self):
        @torch.jit.script
        def foo(a, b):
            c = a * b
            d = c * b
            d.add_(b)
            e = d * b
            return d + e

        x = torch.ones(1)
        y = torch.ones(1)
        foo(x, y)
        b = foo(x, y)
        g = torch.jit.last_executed_optimized_graph()
        self.assertEqual(len(list(g.findAllNodes("prim::TypeCheck"))), 2)
        FileCheck().check("TensorExpr").check("aten::add_").check("TensorExpr").run(g)

    def test_use_not_profiled(self):
        def foo(t1, t2, t3, t4, t: float):
            h = t1 + t2 + t3 + t4
            if t > 0.5:
                # Putting a use of t1 in a never-executed conditional prevents
                return t1 + 1
            return h

        t = torch.rand(8, dtype=torch.float)

        foo_script = torch.jit.script(foo)
        for _ in range(torch._C._jit_get_num_profiled_runs() + 1):
            foo_script(t, t, t, t, 0.1)

        self.assertEqual(foo(t, t, t, t, 0.1), foo_script(t, t, t, t, 0.1))
        g = torch.jit.last_executed_optimized_graph()
        # all adds fused
        FileCheck().check("graph").check_not("aten::add").check("prim::If").run(g)

    def test_not_fusing_scalar_ops(self):
        @torch.jit.script
        def foo(x: int, y: int):
            return x + y + 2 + 4 + 5 + 6

        foo(1, 2)
        foo(2, 3)
        g = torch.jit.last_executed_optimized_graph()
        FileCheck().check_not("TensorExpr").run(g)

    def test_not_optimizing_property(self):
        @torch.jit.script
        def foo(x, y):
            return x + y + 1 + 2 + 3, x.size()

        x = torch.ones(1)
        foo(x, x)
        foo(x, x)
        g = torch.jit.last_executed_optimized_graph()
        FileCheck().check("aten::size").run(g)
        x = torch.ones([2, 3, 5])
        self.assertEqual(foo(x, x), (x + x + 1 + 2 + 3, x.size()))

    def test_fallback_graph_not_specialized(self):
        @torch.jit.script
        def foo(a, b):
            c = a * b
            d = c * b
            e = d * b
            return d + e

        x = torch.ones(1)
        y = torch.ones(1)
        foo(x, y)
        foo(x, y)
        g = torch.jit.last_executed_optimized_graph()
        FileCheck().check("CallFunction").check_next("Tensor = prim::TupleUnpack").run(g)

    def test_autograd_fallback_graph(self):
        @torch.jit.script
        def foo(a, b):
            c = a * b
            d = c * b
            e = d * b
            return d + e

        x = torch.ones(1, requires_grad=True)
        y = torch.ones(1, requires_grad=True)
        foo(x, y)
        b = foo(x, y)
        b.backward(torch.ones([1], dtype=torch.float), retain_graph=True)
        b.backward(torch.ones([1], dtype=torch.float))

        g = torch.jit.last_executed_optimized_graph()
        FileCheck().check("fallback_function").check_next("CallFunction").run(g)

    def test_tensor_constant(self):
        def foo(a, b):
            return a + b + torch.tensor([2])

        x = torch.ones(1, requires_grad=False)
        foo_script = torch.jit.script(foo)
        foo_script(x, x)
        foo_script(x, x)

        self.assertEqual(foo_script(x, x), foo(x, x))
        g = torch.jit.last_executed_optimized_graph()
        FileCheck().check_count("aten::add", 2, exactly=True).run(g)

    def test_local_fusion_strategy(self):
        @torch.jit.script
        def foo(x):
            return x + x + x

        torch.jit.set_fusion_strategy([("STATIC", 1)])
        for _ in range(3):
            foo(torch.rand([10]))

        torch.jit.set_fusion_strategy([("STATIC", 10)])

        for i in range(10):
            foo(torch.rand([i]))
            foo(torch.rand([i]))

        g = torch.jit.last_executed_optimized_graph()
        FileCheck().check_count(":TensorExprGroup", 2, exactly=True).run(g)

    def test_iterative_fusion(self):
        @torch.jit.script
        def foo(a, b, c, d):
            a = a + b
            b.add_(3)
            c = c + b + d
            a = a + 1
            return a, c

        x = torch.ones(1, requires_grad=False)
        foo(x, x, x, x)
        foo(x, x, x, x)

        # when we iterate through the block, we will start
        # by fusing a = a + b with a = a + 1
        # if we were to continue iteration from that fusion point,
        # would miss the fusion opportunity of c = c + d + b

        g = torch.jit.last_executed_optimized_graph()
        self.assertEqual(len(list(g.findAllNodes("prim::TensorExprGroup"))), 2)
