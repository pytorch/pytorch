import torch
from torch.testing._internal.jit_utils import JitTestCase, RUN_CUDA, _inline_everything
from torch import nn
from torch.testing import FileCheck

import unittest

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

class TestPeephole(JitTestCase):
    def test_peephole_with_writes(self):
        def test_write(x):
            s = 0
            s += x
            s += x
            return s

        self.checkScript(test_write, (torch.ones(4, 4),))

    def test_peephole_with_non_output_writes(self):
        @torch.jit.ignore
        def nomnom(x):
            pass

        def test_write(x):
            t = torch.ones_like(x)
            z = x.clone()
            y = z + 0
            z.add_(t)
            # this makes sure z isn't blasted out of existence
            # because it isn't returned or used in a side-effectful
            # way
            nomnom(z)
            return y + y

        a = torch.ones(4, 4)
        j = self.checkScript(test_write, (a,))

    def test_peephole_no_output_aliasing(self):
        def test_peephole(x):
            y = x + 0
            return x, y

        a = torch.ones(4, 4)
        j = self.checkScript(test_peephole, (a,))
        r1, r2 = j(a)
        self.assertNotEqual(r1.data_ptr(), r2.data_ptr())

    def test_peephole(self):
        a = torch.tensor([0.4])
        b = torch.tensor([0.7])
        c = torch.tensor([0], dtype=torch.int32)

        def f(x, y):
            return x.type_as(y)

        tf = torch.jit.trace(f, (a, b))
        FileCheck().check("type_as").run(str(tf.graph))
        self.run_pass('peephole', tf.graph)
        FileCheck().check_not("type_as").run(str(tf.graph))
        tf2 = torch.jit.trace(f, (a, c))
        s = str(tf2.graph)
        self.run_pass('peephole', tf2.graph)
        self.assertEqual(s, str(s))

    def test_peephole_dynamic(self):
        def f(x, y):
            return x.type_as(y)

        fn = torch.jit.script(f)
        s = str(fn.graph)
        torch._C._jit_pass_peephole(fn.graph)
        self.assertEqual(s, str(fn.graph))

    def test_peephole_list_ops(self):
        @torch.jit.script
        def foo(x, y, z):
            return len([x, y, z])

        self.run_pass('peephole', foo.graph)
        FileCheck().check("value=3").check_next("return").run(foo.graph)

        @torch.jit.script
        def foo(x, y, z):
            li = [x, y, z]
            for i in range(len(x)):
                li.append(x)
            return len([x, y, z])

        self.run_pass('peephole', foo.graph)
        FileCheck().check_not("aten::len").run(foo.graph)

        @torch.jit.script
        def foo(x, y, z):
            li = [x, y, z]
            return li[1], li[-2]

        FileCheck().check("aten::__getitem__").run(foo.graph)
        self.run_pass('peephole', foo.graph)
        FileCheck().check_not("aten::__getitem__").run(foo.graph)

        @torch.jit.script
        def foo(x, y, z):
            li = [x, y, z]
            return li[-7]

        self.run_pass('peephole', foo.graph)
        FileCheck().check("aten::__getitem__").run(foo.graph)

        @torch.jit.script
        def foo(x, y, z):
            li = [x, y, z]
            for i in range(len(x)):
                li.append(x)
            return li[-2]

        self.run_pass('peephole', foo.graph)
        FileCheck().check("aten::__getitem__").run(foo.graph)

    @unittest.skipIf(not RUN_CUDA, "cpp tests require CUDA")
    def test_peephole_cuda(self):
        a = torch.tensor([0.4], device='cpu')
        b = torch.tensor([0.7], device='cuda')
        c = torch.tensor([0.7], device='cuda')

        def f(x, y):
            return x.type_as(y)

        trace = torch.jit.trace(f, (a, c))
        s = str(trace.graph)
        self.run_pass('peephole', trace.graph)
        self.assertEqual(s, str(trace.graph))
        trace = torch.jit.trace(f, (b, c))
        self.run_pass('peephole', trace.graph)
        self.run_pass('dce', trace.graph)
        FileCheck().check_not("type_as").run(str(trace.graph))

    @_inline_everything
    def test_peephole_type_refinements(self):
        def refine(x):
            # type: (Optional[Tensor]) -> Tensor
            return x if x is not None else torch.tensor(3)

        @torch.jit.script
        def test():
            return refine(torch.tensor(4))

        FileCheck().check("prim::unchecked_cast").run(test.graph)
        self.run_pass('peephole', test.graph)
        FileCheck().check_not("prim::unchecked_cast").run(test.graph)

        # refinement not optimzied out
        def is_int_tensor(x):
            scalar = x.item()
            if isinstance(scalar, int):
                return scalar + 3
            else:
                return 8

        self.checkScript(is_int_tensor, (torch.tensor(2),))
        self.checkScript(is_int_tensor, (torch.tensor(2.5),))
        graph = torch.jit.script(is_int_tensor).graph
        self.run_pass('peephole', graph)
        FileCheck().check("prim::unchecked_cast").run(graph)

    def test_short_circuit_optimization(self):
        @torch.jit.script
        def const_expressions(x):
            # type: (int) -> Tuple[bool, bool]
            return x == 1 and False, x == 1 or True
        self.run_pass('constant_propagation', const_expressions.graph)
        FileCheck().check_not("prim::If").check_not("aten::eq").run(const_expressions.graph)
        self.assertEqual(const_expressions(1), (False, True))

        @torch.jit.script
        def redundant_expressions(x):
            # type: (int) -> Tuple[bool, bool]
            return x == 1 and True, x == 1 or False

        self.run_pass('peephole', redundant_expressions.graph)
        self.assertEqual(redundant_expressions(1), (True, True))
        self.assertEqual(redundant_expressions(0), (False, False))
        # and True / or False are removed from graph
        FileCheck().check("aten::eq").check_not("prim::If").run(redundant_expressions.graph)

    def test_conv_dim_folding(self):
        modules = [nn.Conv1d, nn.Conv2d, nn.Conv3d]
        for mod in modules:
            class ConvDim(torch.nn.Module):
                def __init__(self):
                    super(ConvDim, self).__init__()
                    self.conv = mod(3, 32, kernel_size=3, stride=2, bias=False)

                def forward(self, x):
                    x = self.conv(x)
                    return x.dim()

            conv_dim = torch.jit.script(ConvDim())
            self.run_pass("inline", conv_dim.graph)
            self.run_pass("peephole", conv_dim.graph)
            FileCheck().check_not("conv").check_not("dim").run(conv_dim.graph)

            class ConvDimMutate(torch.nn.Module):
                def __init__(self):
                    super(ConvDimMutate, self).__init__()
                    self.conv = mod(3, 32, kernel_size=3, stride=2, bias=False)

                def forward(self, x):
                    x = self.conv(x)
                    x.resize_([4, 4])
                    return x.dim()

            conv_dim = torch.jit.script(ConvDimMutate())
            self.run_pass("inline", conv_dim.graph)
            self.run_pass("peephole", conv_dim.graph)
            FileCheck().check("conv").check("dim").run(conv_dim.graph)

    def test_normalized_is_op(self):
        def convertible_is_op(x: bool, y: bool):
            return x is True, False is x, x is y

        self.checkScript(convertible_is_op, (True, False))

        op_graph = torch.jit.script(convertible_is_op).graph
        FileCheck().check_count("aten::eq", 3, exactly=True).run(op_graph)
        FileCheck().check_count("aten::__is__", 0, exactly=True).run(op_graph)

    def test_normalized_isnot_op(self):
        def convertible_isnot_op(x: bool, y: bool):
            return x is not True, False is not x, x is not y

        self.checkScript(convertible_isnot_op, (True, False))

        op_graph = torch.jit.script(convertible_isnot_op).graph
        FileCheck().check_count("aten::ne", 3, exactly=True).run(op_graph)
        FileCheck().check_count("aten::__isnot__", 0, exactly=True).run(op_graph)
